"""Efficiency P1a C-2: core._validate_frame on 16-bit frames.

When astropy itself would return a complete pseudo-unsigned 16-bit primary image (Siril's converted frames), the
header decides the checks (imaging.complete_unsigned16_shape) and every data byte is still read once, unconverted;
everything else takes the pixel path.  Oracles: the pixel path of the same function (complete_unsigned16_shape forced
to None) and a verbatim copy of the former function.  Outcomes -- the value, or the exception type and message -- and
the number of warnings must be equal.
"""
from pathlib import Path
import bz2
import gzip
import lzma
import os
import warnings
import zipfile

import numpy as np
import pytest
from astropy.io import fits

from starstack import core, imaging, orientation
from test_foreground_median import SIRIL_CARDS, fits_bytes, siril_frame, stored_codes, write_fits


def former_validate_frame(path: Path, expected_shape=None, require_float=False):
    # core._validate_frame before efficiency P1a C-2 (commit 3117840), verbatim.
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdus:
        data = hdus[0].data
        is_float = data is not None and data.dtype.kind == "f" and data.dtype.itemsize == 4
        is_uint16 = data is not None and data.dtype.kind == "u" and data.dtype.itemsize == 2
        if require_float and not is_float:
            raise ValueError("Siril 产物必须是 32 位浮点 FITS")
        if not (is_float or is_uint16):
            raise ValueError("转换帧必须是 16 位无符号整数或 32 位浮点 FITS")
        if hdus[0].header.get("ROWORDER", "BOTTOM-UP") != "BOTTOM-UP":
            raise ValueError("此原型仅支持 BOTTOM-UP 的 FITS 坐标，请先统一行方向")
        if data.ndim != 3 or data.shape[0] != 3 or not np.isfinite(data).all():
            raise ValueError("Siril 产物必须是有限值的三通道图像")
        if expected_shape is not None and data.shape != expected_shape:
            raise ValueError("Siril 输出改变了参考画幅，不能继续合成")
        return data.shape


def outcome(fn, path, **kwargs):
    """((kind, value or "Type: message"), number of warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = "ok", fn(path, **kwargs)
        except Exception as exc:
            result = "err", f"{type(exc).__name__}: {exc}"
    return result, len(caught)


@pytest.fixture
def pixel_path(monkeypatch):
    """core._validate_frame with the header route disabled: today's pixel path."""
    def call(path, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(imaging, "complete_unsigned16_shape", lambda *args, **kw: None)
            return core._validate_frame(path, **kwargs)
    return call


def header_route(path, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=False) as hdus:
                return imaging.complete_unsigned16_shape(path, hdus, **kwargs)
    except Exception:
        return None


def assert_same_everywhere(path, pixel_path, modes):
    for kwargs in modes:
        new = outcome(core._validate_frame, path, **kwargs)
        assert new == outcome(pixel_path, path, **kwargs), kwargs
        assert new == outcome(former_validate_frame, path, **kwargs), kwargs


MODES = [{}, {"expected_shape": (3, 40, 30)}, {"expected_shape": (3, 41, 30)}, {"require_float": True},
         {"expected_shape": (3, 40, 30), "require_float": True}]
S = [("BZERO", "RAW:32768."), ("BSCALE", "RAW:1.")]


def codes(shape, seed=11):
    return (np.random.default_rng(seed).integers(0, 65536, size=shape) - 32768).astype(np.int16)


def _float(shape, seed=11):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


# name: (builder, header route expected)
MATRIX = {
    "siril u16 3x40x30 bottom-up": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S + [("ROWORDER", "BOTTOM-UP")]), True),
    "u16 no ROWORDER": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S), True),
    "u16 TOP-DOWN": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S + [("ROWORDER", "TOP-DOWN")]), True),
    "u16 ROWORDER lowercase": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S + [("ROWORDER", "bottom-up")]), True),
    "u16 other size 3x41x30": (lambda p: write_fits(p, codes((3, 41, 30)), 16, S), True),
    "u16 NAXIS 2": (lambda p: write_fits(p, codes((40, 30)), 16, S), True),
    "u16 1 plane": (lambda p: write_fits(p, codes((1, 40, 30)), 16, S), True),
    "u16 4 planes": (lambda p: write_fits(p, codes((4, 40, 30)), 16, S), True),
    "u16 NAXIS 4": (lambda p: write_fits(p, codes((1, 3, 40, 30)), 16, S), True),
    "u16 BZERO int card 32768": (lambda p: write_fits(p, codes((3, 40, 30)), 16, [("BZERO", 32768), ("BSCALE", 1)]), True),
    "u16 BLANK -32768 matching (ignored for unsigned)": (
        lambda p: write_fits(p, np.full((3, 40, 30), -32768, np.int16), 16, S + [("BLANK", -32768)]), True),
    "i16 signed BZERO 0": (lambda p: write_fits(p, codes((3, 40, 30)), 16, [("BZERO", "RAW:0.")]), False),
    "i16 no BZERO": (lambda p: write_fits(p, codes((3, 40, 30)), 16), False),
    "i16 BLANK no match (float)": (
        lambda p: write_fits(p, np.full((3, 40, 30), 5, np.int16), 16, [("BLANK", -32768)]), False),
    "i16 BLANK match (NaN)": (
        lambda p: write_fits(p, np.full((3, 40, 30), -32768, np.int16), 16, [("BLANK", -32768)]), False),
    "u16 BSCALE 2 (float)": (
        lambda p: write_fits(p, codes((3, 40, 30)), 16, [("BZERO", "RAW:32768."), ("BSCALE", "RAW:2.")]), False),
    "BZERO 1000 (float)": (lambda p: write_fits(p, codes((3, 40, 30)), 16, [("BZERO", "RAW:1000.")]), False),
    "BSCALE 1.0D0 Fortran": (
        lambda p: write_fits(p, codes((3, 40, 30)), 16, [("BZERO", "RAW:32768."), ("BSCALE", "RAW:1.0D0")]), True),
    "BZERO string": (lambda p: write_fits(p, codes((3, 40, 30)), 16, [("BZERO", "32768"), ("BSCALE", "RAW:1.")]), False),
    "u16 truncated 1 byte": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S, pad=False, truncate=1), False),
    "u16 truncated 3000 bytes": (
        lambda p: write_fits(p, codes((3, 40, 30)), 16, S, truncate=3000 + (-7200 % 2880)), False),
    "u16 data complete, no padding": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S, pad=False), True),
    "u16 trailing garbage": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S, trailer=b"x" * 5000), True),
    "NAXIS 0": (lambda p: write_fits(p, np.zeros(0, np.int16), 16, S, naxes=[]), False),
    "NAXIS3 = 0": (lambda p: write_fits(p, np.zeros(0, np.int16), 16, S, naxes=[30, 40, 0]), False),
    "float32 finite": (lambda p: write_fits(p, _float((3, 40, 30)), -32), False),
    "float32 NaN": (lambda p: write_fits(p, np.where(_float((3, 40, 30)) < .01, np.nan, 1).astype(np.float32), -32),
                    False),
    "float32 TOP-DOWN": (lambda p: write_fits(p, _float((3, 40, 30)), -32, [("ROWORDER", "TOP-DOWN")]), False),
    "float64": (lambda p: write_fits(p, np.random.default_rng(1).random((3, 40, 30)), -64), False),
    "BITPIX 8": (lambda p: write_fits(p, np.zeros((3, 40, 30), np.uint8), 8), False),
    "BITPIX 32 BZERO 2^31": (lambda p: write_fits(p, np.zeros((3, 40, 30), np.int32), 32, [("BZERO", 2147483648)]),
                             False),
    "SIMPLE = F": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S, simple=False), False),
    "random groups": (lambda p: write_fits(p, codes((3,)), 16, S + [("GROUPS", True), ("PCOUNT", 0), ("GCOUNT", 1)],
                                           naxes=[0, 3]), False),
    "NAXIS1 float card": (lambda p: write_fits(p, codes((3, 40, 30)), 16, S, naxes=["RAW:30.", 40, 3]), False),
    "duplicate BZERO (first 32768, then 0)": (
        lambda p: write_fits(p, codes((3, 40, 30)), 16, S + [("BZERO", "RAW:0.")]), False),
    "empty file": (lambda p: (p.write_bytes(b""), p)[1], False),
    "truncated header": (lambda p: (p.write_bytes(fits_bytes(
        [c.format(h=40, w=30, roworder="BOTTOM-UP") for c in SIRIL_CARDS], codes((3, 40, 30)))[:2000]), p)[1], False),
    "missing file": (lambda p: p, False),
}

STANDARD = {"SIMPLE": "SIMPLE  =                    T", "BITPIX": "BITPIX  =                   16",
            "NAXIS": "NAXIS   =                    3", "NAXIS1": "NAXIS1  =                   30",
            "NAXIS2": "NAXIS2  =                   40", "NAXIS3": "NAXIS3  =                    3",
            "BZERO": "BZERO   =               32768.", "BSCALE": "BSCALE  =                   1.",
            "ROWORDER": "ROWORDER= 'BOTTOM-UP'"}


def _cards(extra=(), **over):
    cards = dict(STANDARD)
    cards.update(over)
    return [c for c in cards.values() if c is not None] + list(extra)


# Cards astropy's fast header parse (which picks the data dtype) and its full header read differently.
HEADER_CASES = {
    "control Siril-like": (_cards(), True),
    "BZERO no space after =": (_cards(BZERO="BZERO   =32768."), False),
    "BZERO '= ' at col 6": (_cards(BZERO="BZERO= 32768."), True),
    "BZERO '=' at col 7": (_cards(BZERO="BZERO = 32768."), False),
    "HIERARCH BZERO": (_cards(BZERO="HIERARCH BZERO = 32768."), False),
    "HIERARCH BSCALE 2 + std 1": (_cards(["HIERARCH BSCALE = 2."]), False),
    "HIERARCH BITPIX -32 extra": (_cards(["HIERARCH BITPIX = -32"]), False),
    "BITPIX '=' at col 7 extra -32": (_cards(["BITPIX = -32"]), False),
    "bzero lowercase": (_cards(BZERO="bzero   =               32768."), True),
    "RVKC BZERO": (_cards(BZERO="BZERO   = 'AXIS.1: 32768'"), False),
    "RVKC BSCALE": (_cards(BSCALE="BSCALE  = 'AXIS.1: 1'"), False),
    "ROWORDER unquoted BOTTOM-UP": (_cards(ROWORDER="ROWORDER= BOTTOM-UP"), True),
    "ROWORDER unquoted TOP-DOWN": (_cards(ROWORDER="ROWORDER= TOP-DOWN"), True),
    "ROWORDER unquoted + comment": (_cards(ROWORDER="ROWORDER= BOTTOM-UP / order"), True),
    "lowercase exptime card": (_cards(["exptime =                  10."]), True),
    "BZERO duplicated 0 then 32768": (_cards(["BZERO   =                   0."]), False),
    "NAXIS3 '=' col 7 extra": (_cards(["NAXIS3 = 4"]), False),
    "BSCALE bool T": (_cards(BSCALE="BSCALE  =                    T"), True),
    "BZERO int 32768": (_cards(BZERO="BZERO   =                32768"), True),
}


def _compressed(path, kind):
    plain = path.with_name("plain.fit")
    siril_frame(plain, np.random.default_rng(3).integers(0, 65536, size=(40, 30, 3)).astype(np.uint16))
    raw = plain.read_bytes()
    if kind == "gzip0":
        path.write_bytes(gzip.compress(raw, compresslevel=0))
    elif kind == "gzip9":
        path.write_bytes(gzip.compress(raw, compresslevel=9))
    elif kind == "bzip2":
        path.write_bytes(bz2.compress(raw))
    elif kind == "xz":
        path.write_bytes(lzma.compress(raw))
    else:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("light.fit", raw)
    return path


CASES = {**{f"matrix: {name}": build for name, build in MATRIX.items()},
         **{f"header: {name}": (lambda p, c=cards: (p.write_bytes(fits_bytes(c, codes((3, 40, 30)))), p)[1], route)
            for name, (cards, route) in HEADER_CASES.items()},
         **{f"compressed: {kind}": (lambda p, k=kind: _compressed(p, k), False)
            for kind in ("gzip0", "gzip9", "bzip2", "xz", "zip")}}


@pytest.mark.parametrize("name", list(CASES))
def test_validate_frame_header_matrix(tmp_path, pixel_path, name):
    build, route_expected = CASES[name]
    path = build(tmp_path / "light_00001.fit")
    assert_same_everywhere(path, pixel_path, MODES)
    shape = header_route(path)
    assert (shape is not None) == route_expected
    # on readable files the read-through never changes the decision: the header checks alone (the size check included)
    # already exclude everything the pixel path would not return as a complete uint16 array
    assert header_route(path, read_through=False) == shape
    if shape is not None:     # sound: astropy's own pixel read is then a complete uint16 array of exactly that shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=False) as hdus:
                assert hdus[0].data.dtype == np.uint16 and hdus[0].data.shape == shape


def test_validate_frame_with_astropy_uint_disabled(tmp_path, pixel_path):
    path = MATRIX["siril u16 3x40x30 bottom-up"][0](tmp_path / "light_00001.fit")
    with fits.conf.set_temp("enable_uint", False):
        assert header_route(path) is None
        assert_same_everywhere(path, pixel_path, MODES)
    assert header_route(path) == (3, 40, 30)


def test_validate_frame_skips_conversion_for_siril_frames(tmp_path, monkeypatch):
    rng = np.random.default_rng(13)
    siril = siril_frame(tmp_path / "light_00001.fit", rng.integers(0, 65536, size=(40, 30, 3)).astype(np.uint16))
    floats = write_fits(tmp_path / "master.fit", rng.random((3, 40, 30)).astype(np.float32), -32)
    with fits.open(siril, memmap=False) as hdus:     # canary: astropy's private state still gives the header route
        assert imaging.complete_unsigned16_shape(siril, hdus) == (3, 40, 30)

    def forbidden(*args, **kwargs):
        raise AssertionError("pixels were read")
    monkeypatch.setattr(np, "fromfile", forbidden)
    monkeypatch.setattr(np, "isfinite", forbidden)
    assert core._validate_frame(siril) == (3, 40, 30)
    assert core._validate_frame(siril, (3, 40, 30)) == (3, 40, 30)
    with pytest.raises(ValueError, match="32 位浮点"):
        core._validate_master(siril)
    with pytest.raises(AssertionError, match="pixels were read"):     # a float frame is still read and checked
        core._validate_frame(floats)


def test_validate_frame_reads_the_data_unit_once(tmp_path, monkeypatch):
    path = siril_frame(tmp_path / "light_00001.fit",
                       np.random.default_rng(14).integers(0, 65536, size=(40, 30, 3)).astype(np.uint16))
    reads, real = [], imaging.data_unit_readable

    def spy(path, offset, nbytes):
        reads.append((offset, nbytes))
        return real(path, offset, nbytes)
    monkeypatch.setattr(imaging, "data_unit_readable", spy)
    assert core._validate_frame(path) == (3, 40, 30)
    assert reads == [(2880, 3 * 40 * 30 * 2)]
    # an unreadable data unit hands over to the pixel path, which raises (or passes) exactly as today
    pixel_reads, real_fromfile = [], np.fromfile
    monkeypatch.setattr(imaging, "data_unit_readable", lambda *args: False)
    monkeypatch.setattr(np, "fromfile", lambda *args, **kw: pixel_reads.append(1) or real_fromfile(*args, **kw))
    assert core._validate_frame(path) == (3, 40, 30)
    assert pixel_reads == [1]


def test_data_unit_readable_reads_every_byte(tmp_path, monkeypatch):
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 100)
    monkeypatch.setattr(imaging, "READ_THROUGH_CHUNK", 1000)       # several chunks and a short last one
    assert imaging.data_unit_readable(path, 100, 25500)
    assert not imaging.data_unit_readable(path, 100, 25501)       # one byte past the end
    assert not imaging.data_unit_readable(tmp_path / "missing.bin", 0, 1)


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-range lock")
def test_validate_frame_unreadable_data_unit_keeps_todays_error(tmp_path, pixel_path):
    import msvcrt
    path = siril_frame(tmp_path / "light_00003.fit",
                       np.random.default_rng(15).integers(0, 65536, size=(300, 200, 3)).astype(np.uint16))
    offset, nbytes, lock_at, lock_len = 2880, 3 * 300 * 200 * 2, 2880 + 100_000, 4096
    holder = open(path, "r+b")
    holder.seek(lock_at)
    msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, lock_len)   # reads through any other handle now fail there
    try:
        new = outcome(core._validate_frame, path)
        assert new == outcome(pixel_path, path) == outcome(former_validate_frame, path)
        assert new[0][0] == "err" and new[0][1].startswith("ValueError: cannot reshape array")
        assert not imaging.data_unit_readable(path, offset, nbytes)
        with fits.open(path, memmap=False) as hdus:
            # option A (header only, not adopted) would accept the frame; the default read-through does not
            assert imaging.complete_unsigned16_shape(path, hdus, read_through=False) == (3, 300, 200)
            assert imaging.complete_unsigned16_shape(path, hdus) is None
    finally:
        holder.seek(lock_at)
        msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, lock_len)
        holder.close()
    assert core._validate_frame(path) == (3, 300, 200)


def test_validate_frame_emits_no_new_warnings(tmp_path, pixel_path):
    path = siril_frame(tmp_path / "light_00001.fit",
                       np.random.default_rng(16).integers(0, 65536, size=(40, 30, 3)).astype(np.uint16),
                       extra=["exptime =                  10."])
    assert header_route(path) == (3, 40, 30)
    new, pixel = outcome(core._validate_frame, path), outcome(pixel_path, path)
    assert new == pixel == ((("ok", (3, 40, 30))), 0)


# ---- X1: frames rotated by orientation.rotate_fits (portrait shots) -------------------------------------------------
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("roworder", ["BOTTOM-UP", "TOP-DOWN"])
def test_x1_rotated_siril_frames_take_the_header_route(tmp_path, pixel_path, angle, roworder):
    values = np.random.default_rng(angle).integers(0, 65536, size=(40, 30, 3)).astype(np.uint16)
    source = siril_frame(tmp_path / "source.fit", values, roworder)
    path = tmp_path / "light_00001.fit"
    orientation.rotate_fits(source, path, angle)
    shape = (3, 30, 40) if angle in (90, 270) else (3, 40, 30)
    assert header_route(path) == shape
    modes = [{}, {"expected_shape": shape}, {"expected_shape": (3, 40, 30)}, {"expected_shape": (3, 30, 40)},
             {"require_float": True}]
    assert_same_everywhere(path, pixel_path, modes)
    if roworder == "BOTTOM-UP":
        assert core._validate_frame(path, shape) == shape


# ---- opt-in: a kept real job ------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("AESTER_P1A_REAL_JOB"), reason="set AESTER_P1A_REAL_JOB to a kept job")
def test_kept_job_frames_and_masters(pixel_path):
    work = Path(os.environ["AESTER_P1A_REAL_JOB"]) / "work"
    lights = sorted(work.glob("light_*.fit"))
    masters = [p for p in (work / "foreground_master.fit", work / "sky_master.fit") if p.exists()]
    assert lights
    shape = former_validate_frame(lights[0])
    for path in lights[:2] + lights[-1:]:
        assert header_route(path) == shape
        assert_same_everywhere(path, pixel_path, [{}, {"expected_shape": shape}])
    for path in masters:
        assert header_route(path) is None
        assert_same_everywhere(path, pixel_path, [{"expected_shape": shape, "require_float": True}])
