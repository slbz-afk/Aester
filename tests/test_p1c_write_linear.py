"""Efficiency P1c U-M14 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.1): ``imaging.write_linear`` streams a native
float32 master to the file instead of making two whole copies, with exactly the bytes of the former write (kept as
``imaging._write_linear_former``), the same ledger record, the same failures and error texts, and a per-call
self-test that falls back to the former write.

The real-data test rewrites the F48 reference job's linear masters from their own pixels and must reproduce each
file byte for byte (AESTER_FIXTURE_ROOT points a worktree at the main checkout).
"""
import errno
import hashlib
import io
import os
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import p1c_site_harness as h
from starstack import hashing, imaging, profiling

FORMER = imaging._write_linear_former           # captured before any monkeypatch
NOTE = "Unregistered temporal median of the converted frames, taken before the sky masking; fixed tripod only"
LONG_NOTE = "A long HISTORY note that astropy splits over several cards: " + "abcdefghij" * 20


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(hashing.ENV_LEDGER, raising=False)
    assert not hashing.snapshot()["active"]
    yield
    assert not hashing.snapshot()["active"]


@pytest.fixture
def ntfs(tmp_path):
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"probe")
    if not hashing._volume_eligible(hashing._norm(probe), os.stat(probe)):
        pytest.skip("the ledger needs a local NTFS/ReFS volume")
    return tmp_path


@pytest.fixture
def routes(monkeypatch):
    """Which write each write_linear call took: 'stream' or 'former' (the self-test's in-memory stream not counted)."""
    seen = []
    real_stream, real_former = imaging._stream_linear, imaging._write_linear_former

    def stream(*args, **kwargs):
        if not isinstance(args[0], io.BytesIO):
            seen.append("stream")
        return real_stream(*args, **kwargs)

    def former(*args, **kwargs):
        seen.append("former")
        return real_former(*args, **kwargs)
    monkeypatch.setattr(imaging, "_stream_linear", stream)
    monkeypatch.setattr(imaging, "_write_linear_former", former)
    return seen


def values(rng, shape):
    """float32 values with every special the sentinel has (NaN payloads, -0, inf, subnormals, extremes)."""
    data = (rng.standard_normal(shape) * 1e3).astype(np.float32)
    flat = data.reshape(-1)
    specials = np.ascontiguousarray(imaging._linear_sentinel()).reshape(-1)
    count = min(flat.size, specials.size)
    flat[rng.choice(flat.size, size=count, replace=False)] = specials[:count]
    return data


def layout(data, kind):
    """The same values in different memory layouts (what callers pass: flipud views, read_rgb's moved axes)."""
    if kind == "contiguous":
        return np.ascontiguousarray(data)
    if kind == "flipud":
        return np.flipud(np.ascontiguousarray(data[::-1]))
    if kind == "planes":                     # read_rgb: np.moveaxis of a (3, H, W) array
        return np.moveaxis(np.ascontiguousarray(np.moveaxis(data, -1, 0)), 0, -1)
    if kind == "strided":                    # every second column of a wider array
        wide = np.empty((data.shape[0], data.shape[1] * 2, data.shape[2]), dtype=np.float32)
        wide[:, ::2] = data
        return wide[:, ::2]
    if kind == "readonly":
        out = np.ascontiguousarray(data).copy()
        out.setflags(write=False)
        return out
    raise AssertionError(kind)


def write_both(folder, data, note=NOTE, extra=None, *, mode="on"):
    """(new bytes, former bytes, new scope report, former scope report): each written in its own run scope."""
    reports = []
    for name, writer in (("new", imaging.write_linear), ("former", FORMER)):
        target = folder / f"{name}.fit"
        with hashing.run_scope(f"um14-{name}", mode=mode):
            writer(target, data, note, extra)
            report = hashing.scope_report()
        reports.append({k: v for k, v in report.items() if k not in ("label", "ledger", "largest_misses")})
    return (folder / "new.fit").read_bytes(), (folder / "former.fit").read_bytes(), reports[0], reports[1]


SHAPES = [(1, 1, 3), (1, 5, 3), (4, 1, 3), (2, 240, 3), (7, 9, 3), (64, 33, 3), (65, 129, 3), (3, 4, 1), (5, 6, 2),
          (2, 3, 4), (300, 2000, 3), (1100, 2049, 3)]


# ================================================================== bit identity

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("kind", ["contiguous", "flipud", "planes", "strided", "readonly"])
def test_streamed_bytes_are_the_former_bytes(ntfs, routes, shape, kind):
    data = layout(values(np.random.default_rng(sum(shape)), shape), kind)
    new, former, new_report, former_report = write_both(ntfs, data, LONG_NOTE, {"STACKCNT": 7, "EXPTIME": 10.0})
    assert routes == ["stream"]
    assert new == former
    assert new_report == former_report          # one write record, the same in-memory hashing, no real read
    assert len(new) % 2880 == 0


@pytest.mark.parametrize("chunk", [1, 4, 12, 4 * 9 * 2, 4 * 9 * 3 + 4])
def test_any_block_size_gives_the_same_bytes(ntfs, routes, monkeypatch, chunk):
    monkeypatch.setattr(imaging, "LINEAR_CHUNK_BYTES", chunk)
    data = layout(values(np.random.default_rng(chunk), (13, 9, 3)), "flipud")
    new, former, _, _ = write_both(ntfs, data, NOTE, {"TRANSNT": 3})
    assert routes == ["stream"] and new == former


@pytest.mark.parametrize("extra", [None, {}, {"STACKCNT": 48, "EXPTIME": 15.0, "DATE-OBS": "2026-09-30T20:15:00"},
                                   {"TRANSNT": 0}])
def test_headers_with_the_callers_extras(ntfs, routes, extra):
    new, former, _, _ = write_both(ntfs, values(np.random.default_rng(3), (6, 7, 3)), NOTE, extra)
    assert routes == ["stream"] and new == former


def test_the_record_and_the_next_hash_are_a_write_hit(ntfs):
    data = values(np.random.default_rng(5), (31, 17, 3))
    target = ntfs / "m.fit"
    with hashing.run_scope("um14", mode="on"):
        imaging.write_linear(target, data, NOTE)
        before = hashing.snapshot()
        digest = hashing.sha256_file(target, source="test.um14")
        after = hashing.snapshot()
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert after["saved"]["write"]["files"] - before["saved"]["write"]["files"] == 1
    assert after["real"]["bytes"] == before["real"]["bytes"]


def test_shadow_mode_streams_too(ntfs, routes):
    new, former, _, _ = write_both(ntfs, values(np.random.default_rng(6), (9, 11, 3)), mode="shadow")
    assert routes == ["stream"] and new == former


def test_the_file_does_not_change_when_the_array_does_afterwards(ntfs):
    data = values(np.random.default_rng(8), (40, 30, 3))
    target = ntfs / "m.fit"
    with hashing.run_scope("um14", mode="on"):
        imaging.write_linear(target, data, NOTE)
    written = target.read_bytes()
    expected = io.BytesIO()
    fits.writeto(expected, np.moveaxis(data.astype(np.float32), -1, 0), imaging._linear_header(NOTE), overwrite=False)
    data[...] = 0.5
    assert target.read_bytes() == written == expected.getvalue()


# ================================================================== the former write where the stream does not apply

@pytest.mark.parametrize("case", ["float64", "int16", "2d", "4d", "empty-rows", "empty-planes", "big-endian", "list",
                                  "memmap-subclass"])
def test_anything_else_takes_the_former_write(ntfs, routes, case):
    rng = np.random.default_rng(9)
    base = values(rng, (5, 6, 3))
    with np.errstate(invalid="ignore"):          # the signalling NaNs signal in a float64 conversion
        data = _cases(base)[case]
    new = h.outcome(lambda delivered: _write(ntfs / "new.fit", data, imaging.write_linear))
    routes_seen = list(routes)
    former = h.outcome(lambda delivered: _write(ntfs / "former.fit", data, FORMER))
    assert routes_seen == ["former"]
    assert (new.error_type, new.message) == (former.error_type, former.message)
    for name in ("new", "former"):
        assert (ntfs / f"{name}.fit").exists() == (new.error_type is None)
    if new.error_type is None:
        assert (ntfs / "new.fit").read_bytes() == (ntfs / "former.fit").read_bytes()


def _cases(base):
    return {"float64": base.astype(np.float64), "int16": np.arange(base.size, dtype=np.int16).reshape(base.shape),
            "2d": base[..., 0], "4d": base[None], "empty-rows": base[:0], "empty-planes": base[..., :0],
            "big-endian": base.astype(">f4"), "list": base.tolist(), "memmap-subclass": base.view(np.memmap)}


def _write(path, data, writer, note=NOTE, extra=None):
    with hashing.run_scope("um14", mode="on"):
        writer(path, data, note, extra)


@pytest.mark.parametrize("suffix", [".gz", ".zip", ".bz2", ".xz", ".Z"])
def test_a_compressing_extension_takes_the_former_write(ntfs, routes, suffix):
    data = values(np.random.default_rng(10), (4, 5, 3))
    _write(ntfs / f"new.fit{suffix}", data, imaging.write_linear)
    assert routes == ["former"]
    _write(ntfs / f"former.fit{suffix}", data, FORMER)
    assert (ntfs / f"new.fit{suffix}").read_bytes() == (ntfs / f"former.fit{suffix}").read_bytes()


@pytest.mark.parametrize("suffix", [".GZ", ".fits", ".fit.gz.part"])
def test_other_extensions_stream(ntfs, routes, suffix):
    _write(ntfs / f"new{suffix}", values(np.random.default_rng(11), (4, 5, 3)), imaging.write_linear)
    assert routes == ["stream"]


@pytest.mark.parametrize("mode", [None, "off"])
@pytest.mark.parametrize("shape", [(6, 5, 3), (1, 1, 3), (65, 129, 3), (300, 2000, 3), (3, 4, 1)],
                         ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("kind", ["contiguous", "flipud", "planes"])
def test_without_a_ledger_it_streams_the_former_bytes(tmp_path, routes, mode, shape, kind):
    """Outside a run scope and with AESTER_HASH_LEDGER=off the stream writes the plain file (no digest): byte for
    byte the former write's file there (fits.writeto(path, ...)), and no ledger record either way."""
    data = layout(values(np.random.default_rng(12 + sum(shape)), shape), kind)
    with _ledger(mode):
        imaging.write_linear(tmp_path / "new.fit", data, LONG_NOTE, {"STACKCNT": 7})
        new_report = hashing.scope_report()
    assert routes == ["stream"]
    with _ledger(mode):
        FORMER(tmp_path / "former.fit", data, LONG_NOTE, {"STACKCNT": 7})
        former_report = hashing.scope_report()
    assert (tmp_path / "new.fit").read_bytes() == (tmp_path / "former.fit").read_bytes()
    assert _bare(new_report) == _bare(former_report)


def _ledger(mode):
    import contextlib
    return hashing.run_scope("um14", mode=mode) if mode else contextlib.nullcontext()


def _bare(report):
    """A scope report without its label and ledger identity (None outside a scope)."""
    if report is None:
        return None
    return {k: v for k, v in report.items() if k not in ("label", "ledger", "largest_misses")}


@pytest.mark.parametrize("mode", [None, "off"])
def test_without_a_ledger_an_existing_path_is_refused_with_astropys_text(tmp_path, routes, mode):
    data = values(np.random.default_rng(21), (4, 4, 3))
    target = tmp_path / "m.fit"
    target.write_bytes(b"x")
    with _ledger(mode):
        new = h.outcome(lambda d: imaging.write_linear(target, data, NOTE))
        former = h.outcome(lambda d: FORMER(target, data, NOTE))
    assert routes == ["former"] and new.error_type is OSError and "already exists" in new.message
    assert (new.error_type, new.message) == (former.error_type, former.message)
    assert target.read_bytes() == b"x"


def test_an_existing_path_is_refused_with_astropys_text(ntfs, routes):
    data = values(np.random.default_rng(13), (4, 4, 3))
    target = ntfs / "m.fit"
    target.write_bytes(b"x")
    new = h.outcome(lambda d: _write(target, data, imaging.write_linear))
    former = h.outcome(lambda d: _write(target, data, FORMER))
    assert routes == ["former"] and new.error_type is OSError and "already exists" in new.message
    assert (new.error_type, new.message) == (former.error_type, former.message)
    assert target.read_bytes() == b"x"


def test_a_header_error_is_the_former_error_and_leaves_no_file(ntfs):
    data = values(np.random.default_rng(14), (4, 4, 3))
    new = h.outcome(lambda d: _write(ntfs / "new.fit", data, imaging.write_linear, note="非 ASCII 的历史"))
    former = h.outcome(lambda d: _write(ntfs / "former.fit", data, FORMER, note="非 ASCII 的历史"))
    assert new.error_type is ValueError and (new.error_type, new.message) == (former.error_type, former.message)
    assert not (ntfs / "new.fit").exists() and not (ntfs / "former.fit").exists()
    bad = h.outcome(lambda d: _write(ntfs / "new2.fit", data, imaging.write_linear, extra={"BITPIX": "x"}))
    bad_former = h.outcome(lambda d: _write(ntfs / "former2.fit", data, FORMER, extra={"BITPIX": "x"}))
    assert (bad.error_type, bad.message) == (bad_former.error_type, bad_former.message)
    assert (ntfs / "new2.fit").exists() == (ntfs / "former2.fit").exists()


# ================================================================== the per-call self-test

def test_a_failing_self_test_takes_the_former_write_and_notes_it(ntfs, routes, monkeypatch):
    notes = []
    monkeypatch.setattr(profiling, "note_append", lambda key, value, **k: notes.append((key, value)))
    real = imaging._stream_linear

    def corrupt(stream, data, plan):
        if isinstance(stream, io.BytesIO):            # only the self-test's in-memory write
            stream.write(b"!")
        return real(stream, data, plan)
    monkeypatch.setattr(imaging, "_stream_linear", corrupt)
    data = values(np.random.default_rng(15), (4, 6, 3))
    _write(ntfs / "new.fit", data, imaging.write_linear)
    _write(ntfs / "former.fit", data, FORMER)
    assert routes == ["former"]
    assert (ntfs / "new.fit").read_bytes() == (ntfs / "former.fit").read_bytes()
    assert notes == [("write_linear", {"route": "former", "reason": "self-test: bytes differ"})]


@pytest.mark.parametrize("broken", ["sentinel", "writeto", "plan"])
def test_any_exception_in_the_self_test_or_the_plan_takes_the_former_write(ntfs, routes, monkeypatch, broken):
    notes = []
    data = values(np.random.default_rng(16), (4, 6, 3))
    monkeypatch.setattr(profiling, "note_append", lambda key, value, **k: notes.append((key, value)))
    if broken == "sentinel":
        monkeypatch.setattr(imaging, "_linear_sentinel", lambda: 1 / 0)
    elif broken == "writeto":
        real = fits.writeto

        def writeto(target, *args, **kwargs):
            if isinstance(target, io.BytesIO):
                raise MemoryError("self-test")
            return real(target, *args, **kwargs)
        monkeypatch.setattr(imaging.fits, "writeto", writeto)
    else:
        monkeypatch.setattr(imaging.fits, "HDUList", None)          # the plan's own steps raise TypeError
    _write(ntfs / "new.fit", data, imaging.write_linear)
    assert routes == ["former"]
    assert len(notes) == 1 and notes[0][0] == "write_linear" and notes[0][1]["route"] == "former"
    assert notes[0][1]["reason"].startswith({"sentinel": "self-test: ZeroDivisionError", "writeto":
                                             "self-test: MemoryError", "plan": "plan"}[broken])
    monkeypatch.undo()
    _write(ntfs / "former.fit", data, FORMER)
    assert (ntfs / "new.fit").read_bytes() == (ntfs / "former.fit").read_bytes()


def test_the_self_test_passes_and_covers_the_specials():
    header = imaging._linear_header(NOTE, {"STACKCNT": 3})
    assert imaging._linear_self_test(header) is None
    bits = np.ascontiguousarray(imaging._linear_sentinel()).view(np.uint32).reshape(-1)
    floats = bits.view(np.float32)
    assert np.isnan(floats).sum() >= 6 and ((bits & 0x00400000) == 0)[np.isnan(floats)].any()   # signalling NaNs
    assert {0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0x00000001} <= set(bits.tolist())
    assert imaging._linear_sentinel().strides[0] < 0


# ================================================================== failures while writing

@pytest.mark.parametrize("free", [None, 10])
def test_a_full_disk_raises_astropys_text_and_leaves_the_part_written(ntfs, monkeypatch, free):
    """A write error after the header (ENOSPC): both writes raise astropy's two free-space-check wrappings, with the
    'Not enough space' prefix when the volume reports less room than the HDU, and leave the partial file."""
    if free is not None:
        from astropy.utils import data as astropy_data
        monkeypatch.setattr(astropy_data, "get_free_space_in_dir", lambda path: free)
    real = hashing.HashingWriter.write

    def write(self, data):
        if self._end >= 2880 * 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, data)
    monkeypatch.setattr(hashing.HashingWriter, "write", write)
    data = values(np.random.default_rng(17), (60, 50, 3))
    new = h.outcome(lambda d: _write(ntfs / "new.fit", data, imaging.write_linear))
    former = h.outcome(lambda d: _write(ntfs / "former.fit", data, FORMER))
    assert new.error_type is OSError and (new.error_type, new.message) == (former.error_type, former.message)
    if free is not None:
        assert new.message.count("Not enough space on disk") == 2
    assert (ntfs / "new.fit").exists() and (ntfs / "former.fit").exists()


def test_an_interrupt_mid_write_propagates_and_leaves_the_part_written(ntfs, monkeypatch):
    real = hashing.HashingWriter.write

    def write(self, data):
        if self._end >= 2880:
            raise KeyboardInterrupt
        return real(self, data)
    monkeypatch.setattr(hashing.HashingWriter, "write", write)
    with pytest.raises(KeyboardInterrupt):
        _write(ntfs / "new.fit", values(np.random.default_rng(18), (60, 50, 3)), imaging.write_linear)
    assert (ntfs / "new.fit").stat().st_size == 2880


def _plain_write_fails(monkeypatch, at, error):
    """The plain route's writes (outside a ledger) raise ``error()`` once ``at`` bytes are in the file."""
    real = imaging._WriteAttempts.write

    def write(self, data):
        if self._raw.tell() >= at:
            self.attempted = True
            raise error()
        return real(self, data)
    monkeypatch.setattr(imaging._WriteAttempts, "write", write)


@pytest.mark.parametrize("free", [None, 10])
@pytest.mark.parametrize("mode", [None, "off"])
def test_without_a_ledger_a_full_disk_raises_astropys_text_and_leaves_the_part_written(ntfs, monkeypatch, free,
                                                                                       mode):
    """The plain route (no digest): a write error after the header raises astropy's two free-space-check wrappings,
    as the former write does at the same byte (here injected through the ledger's writer, the one place the former
    write's byte stream can be cut), with the file's own directory, and leaves the partial file."""
    if free is not None:
        from astropy.utils import data as astropy_data
        monkeypatch.setattr(astropy_data, "get_free_space_in_dir", lambda path: free)
    real = hashing.HashingWriter.write

    def write(self, data):
        if self._end >= 2880 * 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, data)
    monkeypatch.setattr(hashing.HashingWriter, "write", write)
    _plain_write_fails(monkeypatch, 2880 * 2, lambda: OSError(errno.ENOSPC, "No space left on device"))
    data = values(np.random.default_rng(22), (60, 50, 3))
    with _ledger(mode):
        new = h.outcome(lambda d: imaging.write_linear(ntfs / "new.fit", data, NOTE))
    former = h.outcome(lambda d: _write(ntfs / "former.fit", data, FORMER))
    assert new.error_type is OSError and (new.error_type, new.message) == (former.error_type, former.message)
    if free is not None:
        assert new.message.count("Not enough space on disk") == 2
    assert (ntfs / "new.fit").stat().st_size >= 2880 * 2 and (ntfs / "former.fit").exists()


@pytest.mark.parametrize("mode", [None, "off"])
def test_without_a_ledger_an_interrupt_mid_write_leaves_the_part_written(tmp_path, monkeypatch, mode):
    _plain_write_fails(monkeypatch, 2880, KeyboardInterrupt)
    with _ledger(mode), pytest.raises(KeyboardInterrupt):
        imaging.write_linear(tmp_path / "new.fit", values(np.random.default_rng(23), (60, 50, 3)), NOTE)
    assert (tmp_path / "new.fit").stat().st_size == 2880


@pytest.mark.parametrize("error", [MemoryError, KeyboardInterrupt])
@pytest.mark.parametrize("mode", [None, "off"])
def test_without_a_ledger_a_failure_before_any_write_leaves_no_file(tmp_path, monkeypatch, mode, error):
    """As astropy, which validates before it creates the file (and write_with_digest's DISCARD_UNWRITTEN inside a
    ledger): nothing attempted, no file."""
    real = imaging._stream_linear

    def stream(target, data, plan):
        if not isinstance(target, io.BytesIO):          # not the self-test
            raise error()
        return real(target, data, plan)
    monkeypatch.setattr(imaging, "_stream_linear", stream)
    with _ledger(mode), pytest.raises(error):
        imaging.write_linear(tmp_path / "new.fit", values(np.random.default_rng(24), (6, 5, 3)), NOTE)
    assert not (tmp_path / "new.fit").exists()


# ================================================================== memory

@pytest.mark.skipif(os.name != "nt", reason="private bytes are read on Windows")
@pytest.mark.parametrize("mode", ["on", None, "off"])
def test_the_stream_holds_no_copy_of_the_image(ntfs, mode):
    """In a ledger, outside one and with the ledger off alike (prereg U-M14: 2S -> 8 MiB in every case)."""
    import gc
    data = values(np.random.default_rng(19), (1000, 8000, 3))              # 96 MB
    size = data.nbytes
    rises = {}
    for name, writer in (("new", imaging.write_linear), ("former", FORMER)):
        gc.collect()
        with _ledger(mode), h.PeakSampler(trace=False) as peak:
            writer(ntfs / f"{name}.fit", data, NOTE)
        rises[name] = peak.rise
    assert rises["new"] <= imaging.LINEAR_CHUNK_BYTES + (16 << 20), rises
    assert rises["former"] >= 1.5 * size, rises
    assert (ntfs / "new.fit").read_bytes() == (ntfs / "former.fit").read_bytes()
    print(f"write_linear private-bytes rise, ledger {mode}: stream {rises['new'] / 2**20:.1f} MiB, "
          f"former {rises['former'] / 2**20:.1f} MiB (image {size / 2**20:.1f} MiB)")


# ================================================================== real data (§6.2 (b))

def _note_and_extra(path):
    """write_linear's note (the joined HISTORY cards) and extras (the cards between ROWORDER and HISTORY)."""
    header = fits.getheader(path)
    keys = list(header.keys())
    start, stop = keys.index("ROWORDER") + 1, keys.index("HISTORY")
    extra = {key: header[key] for key in keys[start:stop]}
    note = "".join(str(card.value) for card in header.cards if card.keyword == "HISTORY")
    return note, extra


def test_real_f48_masters_are_rewritten_byte_for_byte(tmp_path, routes):
    """The F48 job's foreground master, its native sky master and both composites, rewritten from their own pixels
    (read as read_rgb reads them: a moved-axis view) by the streamed write: every byte equal to the fixture."""
    job = h.f48_job()
    masters = [job / "work" / "foreground_master.fit", job / "work" / "sky_master.fit",
               *sorted(job.glob("products/*/master_linear.fit")),
               *sorted(job.glob("products/*/transients/master_linear_transients.fit"))]
    assert len(masters) == 4, masters
    for index, source in enumerate(masters):
        note, extra = _note_and_extra(source)
        with fits.open(source, memmap=False) as hdus:
            pixels = np.moveaxis(np.asarray(hdus[0].data, dtype=np.float32), 0, -1)
        target = tmp_path / f"{index}-{source.name}"
        with hashing.run_scope("um14-real", mode="on"):
            imaging.write_linear(target, pixels, note, extra or None)
        assert routes[-1] == "stream", source
        assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest(), source
        target.unlink()
        del pixels
    assert re.fullmatch(r"(stream){4}", "".join(routes))


# ================================================================== M1 (prereg §6.5): the convert stage, step by step

HX2_PYTHON_ENV = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "16"}   # the 6900HX proxy v2 (§6.1)
_STEPS_CHILD = r'''
import importlib, json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
module, _, name = sys.argv[3].partition(":")
result = getattr(importlib.import_module(module), name)(*json.loads(sys.argv[4]))
print("P1C-STEPS " + json.dumps(result), flush=True)
'''


def m1_steps(frames, folder, writer):
    """Run in a fresh interpreter (M1, the method of P1a §4): the convert stage's Python steps on ``frames``
    (unzeroed converted frames, zeroed in place) as the job runs them -- ledger on, the sites' own thread decisions --
    with the foreground master written by ``writer`` ('stream': write_linear; 'former': _write_linear_former).
    Returns [step, lifetime peak commit (PeakPagefileUsage), private bytes, seconds] after the imports, the median,
    the write, ``del ground`` and the zeroing."""
    import time
    import p1c_convert_data as data
    from starstack import core, imaging as im, sky_validity as sv
    steps, clock = [], [time.perf_counter()]

    def step(name):
        now = time.perf_counter()
        steps.append([name, h.peak_pagefile_usage(), h.private_bytes(), round(now - clock[0], 2)])
        clock[0] = now
    step("after import")
    paths = [Path(p) for p in frames]
    with hashing.run_scope("m1", mode="on"):
        ground = im.median_foreground(paths, lambda: None, workers=None)
        step("median")
        (im.write_linear if writer == "stream" else im._write_linear_former)(
            Path(folder) / "foreground_master.fit", ground, data.FOREGROUND_NOTE)
        step("write_linear")
        del ground
        step("del ground")
        _, height, width = core._validate_frame(paths[0])
        source = sv.StaticMaskSource(sv.load_mask_top_down(h.f48_job() / "sky_mask.png", width, height),
                                     dilation_px=data.validity_record()["dilation_px"])
        core._sky_zero(paths, source, lambda: None, width)
        step("zero")
    return steps


@pytest.mark.skipif(os.name != "nt", reason="PeakPagefileUsage is read on Windows")
def test_m1_convert_stage_steps_on_48_real_frames(tmp_path_factory, tmp_path):
    """M1 (prereg §6.5; U-M14's own step check, repeated with C-1 and C-3): the 48 re-converted F48 frames in a fresh
    process with the 6900HX proxy's OpenBLAS settings: writing the foreground master raises the lifetime peak by at
    most 0.1 GB over the median step (the former write's two copies: about 1 GB), and the convert stage's Python
    steps peak within 2.1 GB. The master written is the F48 job's foreground master."""
    import json
    import subprocess
    import sys
    import p1c_convert_data as data
    frames = data.copy_frames(data.reconverted(tmp_path_factory, 48), tmp_path / "work")
    environment = {**os.environ, **HX2_PYTHON_ENV, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("AESTER_FRAME_WORKERS", None)
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", _STEPS_CHILD, str(h.TESTS), str(h.SRC),
                             "test_p1c_write_linear:m1_steps", json.dumps([[str(p) for p in frames], str(tmp_path),
                                                                           "stream"])],
                            capture_output=True, text=True, timeout=1800, env=environment, encoding="utf-8",
                            errors="replace")
    lines = [line for line in result.stdout.splitlines() if line.startswith("P1C-STEPS ")]
    assert result.returncode == 0 and lines, (result.returncode, result.stdout[-2000:], result.stderr[-4000:])
    steps = {name: (peak, private, seconds) for name, peak, private, seconds in json.loads(lines[-1][10:])}
    for name, (peak, private, seconds) in steps.items():
        print(f"M1 {name:<14} peak {peak / 2**30:.3f} GiB  private {private / 2**30:.3f} GiB  {seconds:.2f} s")
    assert steps["write_linear"][0] - steps["median"][0] <= 0.1e9, steps
    assert max(peak for peak, _, _ in steps.values()) <= 2.1e9, steps
    assert steps["median"][1] - steps["del ground"][1] >= 0.5e9, steps        # del ground frees the 0.55 GB median
    assert hashlib.sha256((tmp_path / "foreground_master.fit").read_bytes()).hexdigest() == data.foreground_sha256()
