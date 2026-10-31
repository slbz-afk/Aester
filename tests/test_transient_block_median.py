"""P1a T-4 / T-4m: the vectorised block median and the in-place detection-image tail are bit-identical to the pre-P1a
code (frozen copies below) on random, adversarial and fuzzed inputs and in the floating-point environments tested, and
the fast kernel really runs in the product configuration."""
from __future__ import annotations

import math
import sys
import tracemalloc
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import synthetic_starfield as sf  # noqa: E402
from starstack import transient_trails as tt  # noqa: E402
from starstack.transient_trails import DEFAULT_DETECTION, _planes, _rows  # noqa: E402

FLT_MIN = float(np.finfo(np.float32).tiny)


# ------------------------------------------------------------------------------------------ frozen pre-P1a references
def _reference_block_median(a, ok, size):          # transient_trails._block_median at 3117840, verbatim
    h, w = a.shape
    ph, pw = (-h) % size, (-w) % size
    values = np.where(ok, a, np.nan).astype(np.float32)
    values = np.pad(values, ((0, ph), (0, pw)), constant_values=np.nan)
    blocks = values.reshape((h + ph) // size, size, (w + pw) // size, size).transpose(0, 2, 1, 3).reshape(
        (h + ph) // size, (w + pw) // size, size * size)
    with tt._quiet():
        med = np.nanmedian(blocks, axis=-1)
    fallback = float(np.nanmedian(med)) if np.isfinite(med).any() else 0.0
    med = np.where(np.isfinite(med), med, fallback)
    return np.repeat(np.repeat(med, size, axis=0), size, axis=1)[:h, :w].astype(np.float32)


def _reference_detection_image(frame_path, master_path, *, params=None, rows=512):   # detection_image at 3117840, verbatim
    p = dict(DEFAULT_DETECTION, **(params or {}))
    b = int(p["bin"])
    frames = _planes(frame_path).planes
    masters = _planes(master_path).planes
    height, width = frames[0][0].shape
    if masters[0][0].shape != (height, width):
        raise ValueError("frame and master are not on the same grid")
    hb, wb = height // b, width // b
    resid = np.zeros((3, hb, wb), dtype=np.float32)
    count = np.zeros((hb, wb), dtype=np.int32)
    rows = max(b, (rows // b) * b)
    for y0 in range(0, hb * b, rows):
        y1 = min(hb * b, y0 + rows)
        valid = None
        blocks = []
        for c in range(3):
            f = _rows(*frames[c], y0, y1)[:, :wb * b]
            m = _rows(*masters[c], y0, y1)[:, :wb * b]
            v = (f != 0) & (m != 0)
            valid = v if valid is None else (valid & v)
            blocks.append((f, m))
        nb = (y1 - y0) // b
        for c in range(3):
            f, m = blocks[c]
            r = np.where(valid, f - m, np.float32(0)).reshape(nb, b, wb, b).sum(axis=(1, 3))
            resid[c, y0 // b:y1 // b] = r
        count[y0 // b:y1 // b] = valid.reshape(nb, b, wb, b).sum(axis=(1, 3))
    ok = count >= max(1, (b * b) // 2)
    resid = np.where(ok[None], resid / np.maximum(count, 1)[None], np.float32(0))
    block = max(1, int(p["background_block_px"]) // b)
    snr = np.zeros((hb, wb), dtype=np.float32)
    sigma = []
    for c in range(3):
        flat = resid[c] - _reference_block_median(resid[c], ok, block)
        sample = flat[::3, ::3][ok[::3, ::3]]
        s = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) if sample.size else 0.0
        sigma.append(s)
        if s > 0:
            snr += np.where(ok, flat / np.float32(s), np.float32(0))
    snr /= np.float32(math.sqrt(3.0))
    return {"snr": snr, "valid": ok, "sigma": sigma, "bin": b, "shape": (height, width), "foreground": None}


# ------------------------------------------------------------------------------------------ helpers
def _seed(*key) -> int:
    """Stable across processes (str hashing is salted per process, so hash() is not)."""
    return zlib.crc32(repr(key).encode())


def _check(a, ok, size):
    with warnings.catch_warnings(record=True) as w0:
        warnings.simplefilter("always")
        ref = _reference_block_median(a, ok, size)
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        got = tt._block_median(a, ok, size)
    assert got.dtype == np.float32 and got.shape == ref.shape and got.flags.writeable
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
    assert [(x.category, str(x.message)) for x in w1] == [(x.category, str(x.message)) for x in w0]


def _spy(monkeypatch):
    """Record (rows shape, redo) of every _nanmedian_rows call."""
    calls, real = [], tt._nanmedian_rows

    def spy(rows):
        shape = rows.shape
        med, redo = real(rows)
        calls.append((shape, redo.copy()))
        return med, redo
    monkeypatch.setattr(tt, "_nanmedian_rows", spy)
    return calls


def _stack_rows(rows):
    """(k, 1024) rows -> a (32k, 32) plane whose 32 x 32 blocks are the rows, in order."""
    return rows.reshape(-1, 32, 32).reshape(-1, 32)


# ------------------------------------------------------------------------------------------ _block_median (T-4)
SHAPES = [(300, 400), (64, 64), (31, 17), (1, 1), (0, 64), (64, 0), (33, 1000), (1000, 33), (97, 161), (640, 20)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("size", [32, 25, 40, 24, 7, 1])
@pytest.mark.parametrize("frac", [1.0, 0.97, 0.5, 0.05, 0.0])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_random_planes_match_reference(shape, size, frac, dtype):
    rng = np.random.default_rng(_seed(shape, size, frac, np.dtype(dtype).str))
    _check(rng.normal(0, 1e-3, shape).astype(dtype), rng.random(shape) < frac, size)


def _adversarial(name):
    rng = np.random.default_rng(_seed("adversarial", name))
    shape = (300, 400)
    some = rng.random(shape) < 0.9
    if name == "per-block counts 0..1024":
        flat = np.zeros((100, 1024), bool)
        for i in range(100):
            flat[i, rng.permutation(1024)[:rng.integers(0, 1025)]] = True
        return rng.normal(0, 1, (320, 320)), flat.reshape(10, 10, 32, 32).transpose(0, 2, 1, 3).reshape(320, 320), 32
    a = rng.normal(0, 1, shape)
    if name in ("data nan", "data inf", "data -inf"):
        a[rng.random(shape) < 0.3] = {"data nan": np.nan, "data inf": np.inf, "data -inf": -np.inf}[name]
    elif name == "inf/-inf middles":
        a = np.zeros((64, 64)); a[:32, :32] = np.where(np.arange(1024).reshape(32, 32) % 2 == 0, np.inf, -np.inf); a[32:] = 1.0
        return a, np.ones(a.shape, bool), 32
    elif name == "near FLT_MAX":
        a = rng.uniform(0.9, 1.0, shape) * float(np.finfo(np.float32).max)
    elif name == "beyond float32 range":
        a = a * 1e39
    elif name == "odd subnormals f32":
        a = (rng.integers(1, 2000, shape) * 2 + 1).astype(np.uint32).view(np.float32)
    elif name == "odd subnormals f64":
        a = (rng.integers(1, 2000, shape) * 2 + 1).astype(np.uint32).view(np.float32).astype(np.float64)
    elif name == "tiny positive -> +0":
        a = rng.uniform(0, 1e-46, shape)
    elif name == "tiny negative -> -0":
        a = -rng.uniform(0, 1e-46, shape)
    elif name == "+0/-0 mixed":
        a[rng.random(shape) < 0.2] = -0.0
        a[rng.random(shape) < 0.2] = 0.0
    elif name == "many +0":
        a = np.where(rng.random(shape) < 0.6, 0.0, a)
    elif name == "zero-median regions":
        a = rng.normal(0, 1e-3, (320, 640)); a[:100, :200] = 0.0
        return a, np.ones(a.shape, bool), 32
    elif name == "+-FLT_MIN cancellations":
        pool = np.concatenate([FLT_MIN * rng.uniform(1, 3, 200), -FLT_MIN * rng.uniform(1, 3, 200), [0.0, np.inf, -np.inf, 1.0, -1.0]])
        a = rng.choice(pool, (320, 640)); a[rng.random(a.shape) < 0.25] = np.nan
        return a, np.ones(a.shape, bool), 32
    elif name in ("just below 2**-100", "just above 2**-100"):
        base = 2.0 ** (-101 if "below" in name else -99)
        pool = np.concatenate([base * rng.uniform(1, 1.0001, 300), -base * rng.uniform(1, 1.0001, 300), [0.0, np.inf, -np.inf, 1.0, -1.0]])
        a = rng.choice(pool, (320, 640)).astype(np.float32); a[rng.random(a.shape) < 0.25] = np.nan
        return a, np.ones(a.shape, bool), 32
    elif name == "all invalid":
        some = np.zeros(shape, bool)
    elif name == "only inf":
        return np.full((64, 64), np.inf), np.ones((64, 64), bool), 32
    return a, some, 32


ADVERSARIAL = ["per-block counts 0..1024", "data nan", "data inf", "data -inf", "inf/-inf middles", "near FLT_MAX",
               "beyond float32 range", "odd subnormals f32", "odd subnormals f64", "tiny positive -> +0",
               "tiny negative -> -0", "+0/-0 mixed", "many +0", "zero-median regions", "+-FLT_MIN cancellations",
               "just below 2**-100", "just above 2**-100", "all invalid", "only inf"]
ZERO_OR_TINY = ["odd subnormals f32", "tiny positive -> +0", "tiny negative -> -0", "+0/-0 mixed", "zero-median regions",
                "+-FLT_MIN cancellations", "just below 2**-100", "just above 2**-100"]


@pytest.mark.parametrize("name", ADVERSARIAL)
def test_adversarial_values_match_reference(name):
    _check(*_adversarial(name))


def test_unique_negative_zero_median_is_recomputed_by_numpy(monkeypatch):
    # Rows whose only zeros are -0.0 and whose middle value is -0.0.  numpy's sum starts from +0.0, so np.nanmedian
    # gives +0.0 while the sorted row holds -0.0 there: the raw kernel value differs, which is why such rows are redone.
    row = np.concatenate([-np.ones(400), np.full(223, -0.0), np.ones(400), [np.nan]]).astype(np.float32)
    rng = np.random.default_rng(_seed("negzero"))
    rows = np.stack([rng.permutation(row) for _ in range(4)])
    raw, redo = tt._nanmedian_rows(rows.copy())
    assert redo.tolist() == [0, 1, 2, 3]
    assert (raw.view(np.uint32) == 0x80000000).all()
    assert (np.nanmedian(rows, axis=1).view(np.uint32) == 0).all()
    a = np.concatenate([_stack_rows(rows), rng.normal(5.0, 1.0, (128, 32))], axis=1)     # plus 4 ordinary blocks
    calls = _spy(monkeypatch)
    _check(a, np.ones(a.shape, bool), 32)
    assert len(calls) == 1 and calls[0][1].tolist() == [0, 2, 4, 6]                      # bh = 4, bw = 2


def test_negative_zero_away_from_the_middle_keeps_the_kernel(monkeypatch):
    rng = np.random.default_rng(_seed("negzero-low"))
    a = rng.normal(5.0, 1.0, (256, 320))
    a[rng.random(a.shape) < 0.02] = -0.0                   # sorts below every middle value (about 5)
    calls = _spy(monkeypatch)
    _check(a, rng.random(a.shape) < 0.9, 32)
    assert len(calls) == 1 and calls[0][1].size == 0


def test_kernel_matches_nanmedian_rows():
    rng = np.random.default_rng(_seed("kernel"))
    for length in (600, 601, 1024, 1025):
        rows = rng.normal(0, 1, (length + 1, length)).astype(np.float32)
        for i in range(length + 1):                       # non-NaN counts 0..length
            rows[i, rng.permutation(length)[:length - i]] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # the all-NaN row
            ref = np.nanmedian(rows, axis=1)
        got, redo = tt._nanmedian_rows(rows.copy())
        keep = np.setdiff1d(np.arange(length + 1), redo)
        np.testing.assert_array_equal(np.isnan(got[keep]), np.isnan(ref[keep]))
        finite = keep[~np.isnan(ref[keep])]
        np.testing.assert_array_equal(got[finite].view(np.uint32), ref[finite].view(np.uint32))


@pytest.mark.parametrize("partial_rows", [0, 3, 25, 26, 100])
@pytest.mark.parametrize("length", [600, 601, 1024])
def test_kernel_partition_and_sort_branches(partial_rows, length):
    # 100 rows: complete rows take the partition, rows with missing pixels are sorted; with more than a quarter of the
    # rows partial (26 of 100) every row is sorted.  Ties, duplicates and an all-NaN row (not "partial") are included.
    rng = np.random.default_rng(_seed("branches", partial_rows, length))
    rows = rng.normal(3.0, 1.0, (100, length)).astype(np.float32)
    rows[1::7] = np.round(rows[1::7])                                    # heavy ties
    rows[2::9, : length // 2] = rows[2::9, length // 2: 2 * (length // 2)]    # duplicated halves
    rows[99] = np.nan                                                    # all-NaN row: count 0
    for i in range(partial_rows):
        rows[i, rng.permutation(length)[: 1 + rng.integers(length - 1)]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(rows, axis=1)
    work = rows.copy()
    got, redo = tt._nanmedian_rows(work)
    assert redo.size == 0
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
    complete = np.flatnonzero(~np.isnan(rows).any(axis=1))
    fully_sorted = np.all(np.diff(work[complete], axis=1) >= 0, axis=1)
    if 4 * partial_rows > 100:
        assert fully_sorted.all()                                        # the row-sort branch
    else:
        assert not fully_sorted.any()                                    # partitioned, not sorted
    if length == 1024:                                                   # the same rows as 32 x 32 blocks of a plane
        a = _stack_rows(rows)
        _check(np.where(np.isnan(a), 0.0, a), ~np.isnan(a), 32)


@pytest.mark.parametrize("size", [24, 7, 1])
def test_short_blocks_keep_numpy(monkeypatch, size):
    monkeypatch.setattr(tt, "_nanmedian_rows", lambda rows: pytest.fail("short blocks must take np.nanmedian"))
    rng = np.random.default_rng(_seed("short", size))
    _check(rng.normal(0, 1e-3, (97, 161)), rng.random((97, 161)) < 0.9, size)


@pytest.mark.parametrize("kind", ["float16", "int", "uint8 mask", "list mask", "numpy int size", "big-endian"])
def test_unusual_inputs_keep_numpy(monkeypatch, kind):
    monkeypatch.setattr(tt, "_nanmedian_rows", lambda rows: pytest.fail("unusual input must take the numpy code"))
    rng = np.random.default_rng(_seed("unusual", kind))
    a, ok, size = rng.normal(0, 1e-3, (97, 161)), rng.random((97, 161)) < 0.9, 32
    if kind == "float16":
        a = a.astype(np.float16)
    elif kind == "int":
        a = (a * 1e6).astype(np.int64)
    elif kind == "uint8 mask":
        ok = ok.astype(np.uint8) * 3
    elif kind == "list mask":
        ok = ok.tolist()
    elif kind == "numpy int size":
        size = np.int64(32)
    else:
        a = a.astype(">f8")
    _check(a, ok, size)


def test_fuzz_bounded():
    rng = np.random.default_rng(_seed("fuzz"))
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 3.3e38, -3.3e38, 1e-45, -1e-45, 3e-39, 1.0, -1.0,
                         FLT_MIN, -FLT_MIN, 2.0 ** -100, -(2.0 ** -100)])
    for _ in range(300):
        h, w = int(rng.integers(0, 120)), int(rng.integers(0, 120))
        a = rng.normal(0, 1, (h, w)) * 10.0 ** rng.integers(-47, 39, (h, w)) if rng.random() < 0.5 else np.round(rng.normal(0, 3, (h, w)))
        sel = rng.random((h, w)) < rng.uniform(0, 0.6)
        a[sel] = rng.choice(specials, sel.sum())
        if rng.random() < 0.5:
            with np.errstate(over="ignore"):
                a = a.astype(np.float32)
        _check(a, rng.random((h, w)) < rng.uniform(0, 1), int(rng.choice([25, 32, 33, 40, 24, 7])))


@pytest.mark.skipif(sys.platform != "win32", reason="MXCSR control through ucrtbase._controlfp")
@pytest.mark.parametrize("mode", ["ftz_daz", "round_down"])
def test_floating_point_environment_does_not_matter(mode):
    import ctypes
    ucrt = ctypes.CDLL("ucrtbase")
    ucrt._controlfp.restype = ctypes.c_uint
    ucrt._controlfp.argtypes = [ctypes.c_uint, ctypes.c_uint]
    new, mask = {"ftz_daz": (0x01000000, 0x03000000), "round_down": (0x00000100, 0x00000300)}[mode]
    cases = [_adversarial(name) for name in ZERO_OR_TINY]       # built in the default environment
    old = ucrt._controlfp(0, 0)
    try:
        ucrt._controlfp(new, mask)
        for case in cases:
            _check(*case)
    finally:
        ucrt._controlfp(old & mask, mask)


# ------------------------------------------------------------------------------------------ detection_image (T-4m)
def _synthetic_pair(root: Path, h: int = 300, w: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    master = np.stack([0.02 * (1.0 + 0.05 * yy / h) * k for k in (1.0, 1.2, 0.9)])
    frame = master + rng.normal(0, 0.002, master.shape)
    frame += 0.02 * np.exp(-((yy - 0.4 * xx - 30.0) ** 2) / 4.0)          # a trail
    for cube in (frame, master):
        cube[:, int(0.85 * h):, :] = 0.0                                   # ground: no data
        cube[:, :, :7] = 0.0                                               # left border: no data
    paths = root / f"frame_{h}x{w}.fit", root / f"master_{h}x{w}.fit"
    sf.write_fits_rgb32(paths[0], frame.astype(np.float32))
    sf.write_fits_rgb32(paths[1], master.astype(np.float32))
    return paths


def test_product_configuration_uses_the_kernel(tmp_path, monkeypatch):
    frame, master = _synthetic_pair(tmp_path)
    calls = _spy(monkeypatch)
    tt.detection_image(frame, master)      # DEFAULT_DETECTION: bin 2, background_block_px 64 -> 32 x 32 blocks
    assert [shape for shape, _ in calls] == [(5 * 7, 1024)] * 3


@pytest.mark.parametrize("bin_", [1, 2, 3])
@pytest.mark.parametrize("hw", [(300, 400), (301, 397)])
def test_detection_image_matches_reference(tmp_path, bin_, hw):
    frame, master = _synthetic_pair(tmp_path, *hw)
    ref = _reference_detection_image(frame, master, params={"bin": bin_}, rows=64)
    got = tt.detection_image(frame, master, params={"bin": bin_}, rows=64)
    assert got.keys() == ref.keys()
    np.testing.assert_array_equal(got["snr"].view(np.uint32), ref["snr"].view(np.uint32))
    np.testing.assert_array_equal(got["valid"], ref["valid"])
    assert (got["sigma"], got["bin"], got["shape"], got["foreground"]) == (ref["sigma"], ref["bin"], ref["shape"], None)


def test_detection_image_peak_memory_is_lower(tmp_path):
    frame, master = _synthetic_pair(tmp_path, 1200, 1600)
    peaks = []
    for fn in (_reference_detection_image, tt.detection_image):
        tracemalloc.start()
        try:
            fn(frame, master, rows=64)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] < 0.7 * peaks[0], peaks
