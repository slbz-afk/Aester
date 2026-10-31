"""P1 hashing in the stack job worker's modules: same bytes, same digests, same errors, fewer reads.

docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md (H1, and H2 at unit level).  Covers the
fused single pass of sky_validity.apply_validity_to_fits, the write digests of
imaging.write_linear and registered_support.save_array, the in-place writers
(orientation.rotate_fits, the fused pass) seen by the ledger, the parallel
verification loops of core / finite_stack / stack_range against their former serial
code, and a whole controlled job run with the ledger off, on and in shadow mode,
including the job worker's hash handover.  The comparisons with the former serial
loops run the new code outside a scope, with the ledger off, on and in shadow mode;
inside a scope the files are aged and hashed once first, so the checks under test can
be (and, on and shadow, are) answered from the ledger.  The references run outside
any scope.
"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import threading
import time
import uuid

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from starstack import (core, finite_stack as fs, fits_crop, hashing, imaging, orientation, profiling,
                       registered_support as rs, sky_validity as sv, stack_range as sr)

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def age(path, seconds=60.0):
    stamp = time.time_ns() - int(seconds * 1e9)
    os.utime(path, ns=(stamp, stamp))


def delta(before, after):
    out = {"real": after["real"]["bytes"] - before["real"]["bytes"],
           "shadow": after["shadow"]["checks"] - before["shadow"]["checks"],
           "mismatches": after["shadow"]["mismatches"] - before["shadow"]["mismatches"]}
    for kind in hashing.SAVED_KINDS:
        out[kind] = after["saved"][kind]["files"] - before["saved"][kind]["files"]
    return out


class Cancelled(Exception):
    pass


MODES = [None, "off", "on", "shadow"]


def in_mode(mode):
    return nullcontext() if mode is None else hashing.run_scope("test.job-slice", mode=mode)


def prime(*paths):
    """Age and hash each existing file once in the current scope, so later checks of it can be ledger answers."""
    for path in paths:
        path = Path(path)
        if path.is_file():
            age(path)
            hashing.sha256_file(path, source="test.prime")


def answered(before):
    """How many hashes since ``before`` were ledger answers (verified ones in shadow mode)."""
    moved = delta(before, hashing.snapshot())
    return moved["ledger"] + moved["write"] + moved["shadow"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (hashing.ENV_LEDGER, hashing.ENV_WORKERS, core.HANDOVER_TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)
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
def parallel(monkeypatch):
    monkeypatch.setattr(hashing, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setenv(hashing.ENV_WORKERS, "4")


# --------------------------------------------------------------------------- #
# sky_validity: the fused single pass                                           #
# --------------------------------------------------------------------------- #

def legacy_apply(path, validity_top_down, *, check=None, rows=256):
    """The former apply_validity_to_fits (memory-mapped in-place write), verbatim."""
    path = Path(path)
    header, offset = fits_crop.read_header(path)
    width, height, planes, bitpix, roworder = fits_crop._geometry(header)
    if validity_top_down.shape != (height, width):
        raise ValueError("validity mask does not match the frame")
    valid_storage = validity_top_down[::-1, :] if roworder == "BOTTOM-UP" else validity_top_down
    if bitpix == 16:
        dtype, zero = ">i2", np.int16(-int(float(header.get("BZERO", 0))))
    elif bitpix == -32:
        dtype, zero = ">f4", np.float32(0.0)
    else:
        raise ValueError(f"unsupported BITPIX {bitpix}")
    mm = np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=(planes, height, width))
    zeroed = 0
    try:
        for y in range(0, height, rows):
            if check is not None:
                check()
            block = ~valid_storage[y:y + rows, :]
            n = int(block.sum())
            if n:
                sub = mm[:, y:y + rows, :]
                sub[:, block] = zero
                zeroed += n
        mm.flush()
    finally:
        del mm
    return {"file": path.name, "zeroed_pixels": zeroed, "invalid_fraction": float(zeroed / (width * height))}


def make_frame(path, kind, shape=(37, 29), seed=0):
    rng = np.random.default_rng(seed)
    height, width = shape
    if kind.startswith("u16"):
        data = rng.integers(0, 65535, size=(3, height, width), dtype=np.uint16)
    elif kind == "f32-2d":
        data = (rng.random((height, width), dtype=np.float32) * 2 - .5)
    else:
        data = (rng.random((3, height, width), dtype=np.float32) * 2 - .5)
    primary = fits.PrimaryHDU(data)
    if kind.endswith("td"):
        primary.header["ROWORDER"] = "TOP-DOWN"
    hdus = [primary]
    if kind.endswith("ext"):   # bytes after the primary data (an extension) must stay untouched
        hdus.append(fits.ImageHDU(np.arange(12, dtype=np.int16).reshape(3, 4)))
    fits.HDUList(hdus).writeto(path)
    validity = rng.random(shape) > .15
    validity[: height // 4] = True      # some row blocks without an invalid pixel (not written back)
    return validity


KINDS = ["u16", "f32", "u16-td", "f32-2d", "u16-ext"]


@pytest.mark.parametrize("rows", [1, 7, 256])
@pytest.mark.parametrize("kind", KINDS)
def test_fused_pass_writes_the_bytes_of_the_memmap_write_and_both_digests_are_real(ntfs, kind, rows):
    a, b = ntfs / f"a-{kind}.fit", ntfs / f"b-{kind}.fit"
    validity = make_frame(a, kind)
    shutil.copyfile(a, b)
    original = sha(b)
    expected = legacy_apply(a, validity, rows=rows)
    with hashing.run_scope("t", mode="on"):
        stats, before, after = sv.apply_validity_with_digests(b, validity, rows=rows)
    assert a.read_bytes() == b.read_bytes()
    assert stats == {**expected, "file": b.name}
    assert before == original
    assert after == sha(b)


@pytest.mark.parametrize("scope", [None, "off", "on"])
@pytest.mark.parametrize("kind", KINDS)
def test_apply_validity_writes_the_bytes_of_the_memmap_write_in_and_out_of_a_scope(ntfs, kind, scope):
    a, b = ntfs / f"a-{kind}.fit", ntfs / f"b-{kind}.fit"
    validity = make_frame(a, kind)
    shutil.copyfile(a, b)
    expected = legacy_apply(a, validity, rows=4)
    if scope is None:
        stats = sv.apply_validity_to_fits(b, validity, rows=4)   # no digest wanted: only rows to mask are read
    else:
        with hashing.run_scope("t", mode=scope):
            stats = sv.apply_validity_to_fits(b, validity, rows=4)
    assert a.read_bytes() == b.read_bytes()
    assert stats == {**expected, "file": b.name}


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_fused_pass_records_the_result_as_written_and_drops_the_earlier_entry(ntfs, mode):
    frame = ntfs / "r_light_00001.fit"
    validity = make_frame(frame, "f32", shape=(64, 48))
    age(frame)
    with hashing.run_scope("t", mode=mode):
        old = hashing.sha256_file(frame, source="test")              # an entry for the unmasked file
        stats = sv.apply_validity_to_fits(frame, validity, rows=5)
        start = hashing.snapshot()
        new = rs.sha256(frame)
        moved = delta(start, hashing.snapshot())
    assert new == sha(frame) != old
    assert stats["zeroed_pixels"] == int((~validity).sum())
    if mode == "on":
        assert moved["write"] == 1 and moved["real"] == 0
    else:
        assert moved["shadow"] == 1 and moved["mismatches"] == 0


def test_fused_pass_outside_a_scope_or_off_leaves_no_digest(ntfs):
    reference, frame, unmasked = ntfs / "a.fit", ntfs / "b.fit", ntfs / "c.fit"
    validity = make_frame(reference, "u16")
    shutil.copyfile(reference, frame)
    shutil.copyfile(reference, unmasked)
    original = sha(reference)
    legacy_apply(reference, validity)
    stats, before, after = sv.apply_validity_with_digests(frame, validity)
    assert (before, after) == (original, None)
    assert frame.read_bytes() == reference.read_bytes()
    with hashing.run_scope("t", mode="off"):
        stats, before, after = sv.apply_validity_with_digests(unmasked, validity)
        assert (before, after) == (original, None)
        start = hashing.snapshot()
        assert core._sha(unmasked) == sha(reference)
        assert delta(start, hashing.snapshot())["write"] == 0


def test_fused_pass_checks_at_least_as_often_and_a_cancelled_pass_records_nothing(ntfs):
    frame = ntfs / "frame.fit"
    validity = make_frame(frame, "f32", shape=(40, 16))
    legacy_calls, calls = [], []
    copy = ntfs / "copy.fit"
    shutil.copyfile(frame, copy)
    legacy_apply(copy, validity, rows=6, check=lambda: legacy_calls.append(1))
    shutil.copyfile(frame, copy)
    sv.apply_validity_to_fits(copy, validity, rows=6, check=lambda: calls.append(1))
    assert len(calls) >= len(legacy_calls) == 7

    def cancel():
        calls.append(1)
        if len(calls) == 3:
            raise Cancelled()
    calls.clear()
    age(frame)
    with hashing.run_scope("t", mode="on"):
        with pytest.raises(Cancelled):
            sv.apply_validity_to_fits(frame, validity, rows=6, check=cancel)
        start = hashing.snapshot()
        assert hashing.sha256_file(frame, source="test") == sha(frame)
        assert delta(start, hashing.snapshot())["write"] == 0


def test_fused_pass_keeps_the_memmap_extension_of_a_short_file(ntfs):
    reference, frame = ntfs / "a.fit", ntfs / "b.fit"
    validity = make_frame(reference, "f32")
    raw = reference.read_bytes()[:-3000]          # the data region runs past the end of the file
    reference.write_bytes(raw)
    frame.write_bytes(raw)
    legacy_apply(reference, validity)
    with hashing.run_scope("t", mode="on"):
        stats, before, after = sv.apply_validity_with_digests(frame, validity)
    assert frame.read_bytes() == reference.read_bytes()
    assert before == hashlib.sha256(raw).hexdigest() and after is None


def test_fused_pass_errors_are_unchanged(ntfs):
    frame = ntfs / "frame.fit"
    validity = make_frame(frame, "u16")
    for path, mask in ((frame, validity[:-1]), (ntfs / "missing.fit", validity)):
        assert outcome(sv.apply_validity_to_fits, path, mask) == outcome(legacy_apply, path, mask)
        assert outcome(sv.apply_validity_to_fits, path, mask)[0] in (ValueError, FileNotFoundError)
        with hashing.run_scope("t", mode="on"):
            assert outcome(sv.apply_validity_to_fits, path, mask) == outcome(legacy_apply, path, mask)


def _hash_totals():
    return (profiling.hashed_totals()["hashed_bytes"], profiling.hash_saved_totals()["hash_saved_write_bytes"],
            profiling.hashed_memory_totals()["hashed_memory_bytes"], hashing.snapshot()["real"]["bytes"],
            hashing.snapshot()["memory"]["bytes"])


@pytest.mark.parametrize("mode", ["off", "on", "shadow"])
def test_fused_pass_accounting_counts_the_file_read_once(ntfs, mode):
    """Review finding: the pass reads the whole frame; that read is a real read, not in-memory hashing."""
    frame = ntfs / "light_00001.fit"
    validity = make_frame(frame, "f32", shape=(64, 48))
    age(frame)
    size = frame.stat().st_size
    with hashing.run_scope("t", mode=mode):
        start = _hash_totals()
        sv.apply_validity_to_fits(frame, validity, rows=5)       # the convert stage: no digest before
        core._sha(frame)                                          # its checkpoint
        convert = [after - before for after, before in zip(_hash_totals(), start)]
    # (real read by profiling, saved as a write hit, hashed in memory, real read by snapshot, memory by snapshot)
    expected = {"off": [size, 0, 0, size, 0],                   # the memory-mapped write, then the checkpoint read
                "on": [size, size, 0, size, 0],                 # the pass read, then the checkpoint is a write hit
                "shadow": [2 * size, size, 0, 2 * size, 0]}     # ... verified by a second real read
    assert convert == expected[mode]
    other = ntfs / "r_light_00001.fit"
    validity = make_frame(other, "f32", shape=(64, 48))
    age(other)
    with hashing.run_scope("t", mode=mode):
        start = _hash_totals()
        stats, before, after = sv.apply_validity_with_digests(other, validity, rows=5)   # registered exports
        assert rs.sha256(other) == sha(other)
        export = [after_ - before_ for after_, before_ in zip(_hash_totals(), start)]
    expected = {"off": [2 * size, 0, 0, 2 * size, 0],            # digest before, the write, the read-back
                "on": [size, size, size, size, size],            # one read; the result hashed in memory
                "shadow": [2 * size, size, size, 2 * size, size]}
    assert export == expected[mode]


@pytest.mark.parametrize("mode", [None, "off"])
def test_without_a_ledger_the_writers_take_their_former_paths(ntfs, monkeypatch, mode):
    """Review finding: AESTER_HASH_LEDGER=off (and no scope) restores levers 2 and 4 (no write digests)."""
    used = []
    for name in ("write_with_digest", "write_bytes_with_digest", "copy_with_digest", "register_written"):
        real = getattr(hashing, name)
        monkeypatch.setattr(hashing, name, lambda *a, _name=name, _real=real, **k: used.append(_name) or _real(*a, **k))
    monkeypatch.setattr(sv, "_apply_streaming", lambda *a, **k: used.append("fused pass"))
    work = ntfs / "work"
    work.mkdir()
    values = _export_fixture(work)
    array = np.random.default_rng(4).random((33, 21)) > .5
    data = np.random.default_rng(5).random((31, 17, 3), dtype=np.float32)
    from starstack import master_export as me
    with in_mode(mode):
        start = _hash_totals()
        record = rs.prepare_exports(work, _Source(values), None, retain_frame_support=True)
        descriptor = rs.save_array(ntfs / "new.npy", array)
        imaging.write_linear(ntfs / "new.fit", data, "note")
        assert me.save_image_with_digest(Image.fromarray(np.zeros((8, 9, 3), np.uint8)), ntfs / "p.jpg",
                                         source="t") is None
        moved = [after - before for after, before in zip(_hash_totals(), start)]
    assert used == [] and moved[1] == 0 and moved[4] == moved[2] == sum(
        np.asarray(values[stem]).nbytes for stem in values)       # only the former in-memory source_valid digests
    assert descriptor["sha256"] == sha(ntfs / "new.npy")
    assert all(frame["export_after_sha256"] == sha(work / f"r_{frame['stem']}.fit") for frame in record["frames"])


# --------------------------------------------------------------------------- #
# registered_support: prepare_exports, save_array                               #
# --------------------------------------------------------------------------- #

def _export_fixture(work):
    from test_registered_support_independent import _seq, _write_rgb
    identity = np.eye(3)
    _seq(work / rs.EXPORT_SEQUENCE, [identity, identity, identity], [True, False, True])
    mask1 = np.ones((28, 36), dtype=bool)
    mask1[12:, :9] = False
    data = np.full(mask1.shape, .31, dtype=np.float32)
    data[14, 15] = 0.0
    for stem in ("light_00001", "light_00003"):
        _write_rgb(work / f"r_{stem}.fit", data)
        age(work / f"r_{stem}.fit")
    age(work / rs.EXPORT_SEQUENCE)
    return {"light_00001": mask1, "light_00003": mask1}     # one static array, as a fixed-tripod mask


class _Source:
    def __init__(self, values):
        self.values = values

    def for_frame(self, stem):
        return self.values[stem]


@pytest.mark.parametrize("mode", ["off", "on", "shadow"])
def test_prepare_exports_record_is_identical_in_every_mode(ntfs, mode):
    records = {}
    for label in ("reference", mode):
        work = ntfs / label
        work.mkdir()
        values = _export_fixture(work)
        before = {stem: sha(work / f"r_{stem}.fit") for stem in values}
        if label == "reference":
            record = rs.prepare_exports(work, _Source(values), None, retain_frame_support=True)
        else:
            with hashing.run_scope("t", mode=mode):
                start = hashing.snapshot()
                record = rs.prepare_exports(work, _Source(values), None, retain_frame_support=True)
                moved = delta(start, hashing.snapshot())
            if mode == "on":
                assert moved["write"] >= 2 * len(values)       # export_after and the sidecar read-backs
            if mode == "shadow":
                assert moved["mismatches"] == 0 and moved["shadow"] >= 2 * len(values)
        for frame in record["frames"]:
            path = work / f"r_{frame['stem']}.fit"
            assert frame["export_before_sha256"] == before[frame["stem"]]
            assert frame["export_after_sha256"] == sha(path)
            assert frame["source_valid_sha256"] == hashlib.sha256(values[frame["stem"]].tobytes()).hexdigest()
            assert frame["explicit_validity"]["sha256"] == sha(work / frame["explicit_validity"]["file"])
        record.pop("filter_seconds")
        records[label] = record
    assert records["reference"] == records[mode]


def test_save_array_writes_np_save_bytes_and_its_read_back_is_a_write_hit(ntfs):
    array = np.random.default_rng(4).random((33, 21)) > .5
    legacy = ntfs / "legacy.npy"
    with legacy.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    with hashing.run_scope("t", mode="on"):
        start = hashing.snapshot()
        descriptor = rs.save_array(ntfs / "new.npy", array)
        moved = delta(start, hashing.snapshot())
    assert (ntfs / "new.npy").read_bytes() == legacy.read_bytes()
    assert descriptor == {"file": "new.npy", "sha256": sha(legacy), "shape": [33, 21], "dtype": "bool"}
    assert moved["write"] == 1 and moved["real"] == 0
    assert not (ntfs / "new.npy.tmp").exists()
    outside = rs.save_array(ntfs / "outside.npy", array)            # no scope: read back for real, same record
    assert outside["sha256"] == descriptor["sha256"]
    with pytest.raises(ValueError):                                   # allow_pickle=False, as before
        rs.save_array(ntfs / "objects.npy", np.array([object()]))


def test_sha256_helpers_keep_their_argument_errors(tmp_path):
    for helper in (core._sha, rs.sha256):
        with pytest.raises(AttributeError):
            helper(str(tmp_path / "x"))
    with pytest.raises(FileNotFoundError):
        fs.sha(str(tmp_path / "missing"))
    with pytest.raises(TypeError):
        sr.sha(None)


# --------------------------------------------------------------------------- #
# imaging.write_linear, orientation.rotate_fits                                 #
# --------------------------------------------------------------------------- #

def test_write_linear_writes_astropys_bytes_and_is_a_write_hit(ntfs):
    data = np.random.default_rng(5).random((31, 17, 3), dtype=np.float32)
    header = fits.Header()
    header["ORIGIN"] = "Starstack prototype"
    header["ROWORDER"] = "BOTTOM-UP"
    header["STACKCNT"] = 7
    header["HISTORY"] = "note"
    legacy = ntfs / "legacy.fit"
    fits.writeto(legacy, np.moveaxis(data.astype(np.float32), -1, 0), header, overwrite=False)
    new = ntfs / "new.fit"
    with hashing.run_scope("t", mode="on"):
        imaging.write_linear(new, data, "note", extra={"STACKCNT": 7})
        start = hashing.snapshot()
        assert core._sha(new) == sha(legacy)
        assert delta(start, hashing.snapshot())["write"] == 1
    assert new.read_bytes() == legacy.read_bytes()
    with pytest.raises(OSError) as refused:
        imaging.write_linear(new, data, "again")
    with pytest.raises(OSError) as astropy_refused:
        fits.writeto(new, np.moveaxis(data, -1, 0), header, overwrite=False)
    assert type(refused.value) is type(astropy_refused.value) and str(refused.value) == str(astropy_refused.value)


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_rotate_in_place_is_seen_by_the_ledger(ntfs, mode):
    frame = ntfs / "light_00001.fit"
    make_frame(frame, "u16", shape=(20, 20))
    age(frame)
    with hashing.run_scope("t", mode=mode):
        old = core._sha(frame)
        orientation.rotate_fits(frame, frame, 180)
        start = hashing.snapshot()
        assert core._sha(frame) == sha(frame) != old
        assert delta(start, hashing.snapshot())["mismatches"] == 0


# --------------------------------------------------------------------------- #
# core: checkpoint records, staging copies                                      #
# --------------------------------------------------------------------------- #

def legacy_records(directory, files):
    return [{"path": str(p.relative_to(directory)), "sha256": core._sha(p)} for p in files]


def outcome(function, *args):
    try:
        return "ok", function(*args)
    except Exception as exc:
        return type(exc), str(exc)


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_records_match_the_former_comprehension(tmp_path, parallel, mode):
    directory = tmp_path / "job"
    directory.mkdir()
    files = []
    for i in range(5):
        path = directory / f"f{i}.bin"
        path.write_bytes(bytes([i]) * (1000 + i))
        files.append(path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    missing = directory / "missing.bin"
    cases = [files, files + files[:2], [files[0], missing, outside, files[1]], [files[0], outside, missing],
             [missing, files[0]], []]
    with in_mode(mode):
        prime(*files, outside)
        start = hashing.snapshot()
        actual = [outcome(core._checkpoint_records, directory, case) for case in cases]
        hits = answered(start)
    assert actual == [outcome(legacy_records, directory, case) for case in cases]
    assert (hits > 0) == (mode in ("on", "shadow"))


def legacy_stage(job, stage, check):
    for i, source in enumerate(job["inputs"], 1):
        target = stage / f"frame{i:05d}{Path(source['path']).suffix.lower()}"
        if not target.exists() or core._sha(target) != source["sha256"]:
            check()
            shutil.copyfile(source["path"], target)
            if core._sha(target) != source["sha256"]:
                raise ValueError("素材复制时发生变化，请重新提交")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scenario", ["fresh", "kept", "changed-1", "changed-0-copy-fails-1",
                                      "changed-0-cancel-1", "cancel-1", "missing-source-2", "changed-1-of-6"])
def test_stage_inputs_fails_like_the_serial_copy_loop(tmp_path, monkeypatch, parallel, scenario, mode):
    sources = tmp_path / "sources"
    sources.mkdir()
    inputs = []
    for i in range(6 if scenario.endswith("-of-6") else 4):
        path = sources / f"in{i}.FIT"
        path.write_bytes(bytes([i + 1]) * 5000)
        age(path)
        inputs.append({"path": str(path), "sha256": sha(path)})
    if scenario == "missing-source-2":
        Path(inputs[2]["path"]).unlink()
    job = {"inputs": inputs}
    real_copy = shutil.copyfile

    def copy(source, target):
        index = int(Path(source).stem[2:])
        if scenario.startswith(f"changed-{index}"):
            Path(target).write_bytes(b"changed during the copy")
            return target
        if scenario == "changed-0-copy-fails-1" and index == 1:
            raise PermissionError("copy refused")
        return real_copy(source, target)

    results = []
    for runner in (legacy_stage, core._stage_inputs):
        stage = tmp_path / f"stage-{len(results)}"
        stage.mkdir()
        if scenario == "kept":
            real_copy(inputs[1]["path"], stage / "frame00002.fit")
        calls = []

        def check():
            calls.append(1)
            if scenario.endswith("cancel-1") and len(calls) == 2:
                raise Cancelled("stop")
        monkeypatch.setattr(core.shutil, "copyfile", copy)
        with in_mode(mode) if runner is core._stage_inputs else nullcontext():
            if runner is core._stage_inputs:
                prime(*(source["path"] for source in inputs), *stage.iterdir())
            results.append((outcome(runner, job, stage, check), len(calls), len(list(stage.iterdir()))))
        monkeypatch.setattr(core.shutil, "copyfile", real_copy)
        if scenario in ("fresh", "kept"):
            assert sorted(p.read_bytes() for p in stage.iterdir()) == sorted(Path(s["path"]).read_bytes() for s in inputs)
    assert results[0][0] == results[1][0]
    assert results[1][1] >= results[0][1]
    # A changed copy stops the staging after at most one more copy (the one read-back in flight).
    assert results[0][2] <= results[1][2] <= results[0][2] + 1


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_a_changed_copy_is_caught_even_when_its_old_stat_is_in_the_ledger(tmp_path, monkeypatch, parallel, mode):
    """The copy's read-back is always a real read (review finding: the ledger branch of _stage_inputs).

    The stage holds an old, wrong copy of input 2, hashed in this scope (so its
    stat is in the ledger).  The copy then rewrites it in place with other bytes
    of the same size and puts its modification time back: the stat the ledger
    knows, and not the source's digest.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    inputs = []
    for i in range(3):
        path = sources / f"in{i}.FIT"
        path.write_bytes(bytes([i + 1]) * 5000)
        age(path)
        inputs.append({"path": str(path), "sha256": sha(path)})
    real_copy = shutil.copyfile

    def corrupting(source, target):
        if Path(source).name != "in1.FIT":
            return real_copy(source, target)
        stamp = os.stat(target)
        with open(target, "r+b") as handle:
            handle.write(b"Z" * 5000)
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return target

    stages = []
    for runner in (legacy_stage, core._stage_inputs):
        stage = tmp_path / f"stage-{len(stages)}"
        stage.mkdir()
        (stage / "frame00002.fit").write_bytes(b"W" * 5000)      # an old wrong copy, same size
        age(stage / "frame00002.fit")
        monkeypatch.setattr(core.shutil, "copyfile", corrupting)
        with in_mode(mode) if runner is core._stage_inputs else nullcontext():
            if runner is core._stage_inputs:
                prime(*stage.iterdir())
            stages.append(outcome(runner, {"inputs": inputs}, stage, lambda: None))
        monkeypatch.setattr(core.shutil, "copyfile", real_copy)
    assert stages[1] == stages[0] == (ValueError, "素材复制时发生变化，请重新提交")


def test_a_cancel_waits_for_at_most_one_read_back(tmp_path, monkeypatch, parallel):
    """Review finding: the read-backs are not postponed to the end; a cancel settles at most one."""
    sources = tmp_path / "sources"
    sources.mkdir()
    inputs = []
    for i in range(6):
        path = sources / f"in{i}.FIT"
        path.write_bytes(bytes([i + 1]) * 5000)
        inputs.append({"path": str(path), "sha256": sha(path)})
    stage = tmp_path / "stage"
    stage.mkdir()
    events, lock = [], threading.Lock()
    real_sha = core._sha

    def slow_sha(path, **kwargs):
        with lock:
            events.append(("start", Path(path).name))
        time.sleep(.05)
        try:
            return real_sha(path, **kwargs)
        finally:
            with lock:
                events.append(("end", Path(path).name))
    monkeypatch.setattr(core, "_sha", slow_sha)
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 5:
            with lock:
                events.append(("cancel", None))
            raise Cancelled("stop")
    with hashing.run_scope("t", mode="on"), pytest.raises(Cancelled):
        core._stage_inputs({"inputs": inputs}, stage, check)
    cancel = events.index(("cancel", None))
    started = {name for kind, name in events[:cancel] if kind == "start"}
    ended = {name for kind, name in events[:cancel] if kind == "end"}
    assert len(started - ended) <= 1                                       # in flight at the cancel
    assert not [event for event in events[cancel:] if event[0] == "start"]  # none started after it
    assert len(list(stage.iterdir())) == 4 and ended | started == {f"frame0000{i}.fit" for i in range(1, 5)}


def test_colour_products_let_a_shadow_mismatch_fail_the_job(tmp_path, monkeypatch):
    """Review finding: the legacy fallback must not swallow HashLedgerMismatch (shadow mode)."""
    from starstack import color_export as ce

    def mismatch(path):
        raise hashing.HashLedgerMismatch(path, "t", "ledger", "0" * 64, "f" * 64)
    monkeypatch.setattr(ce, "job_colour_basis", mismatch)
    job = {"input_kind": "raw", "inputs": [{"path": "first.nef"}], "mask": None,
           "color_adjustments": {"warmth": 0.0, "tint": 0.0, "exposure_ev": 0.0}}
    with pytest.raises(hashing.HashLedgerMismatch):
        core._colour_products(tmp_path, job, tmp_path, None, tmp_path, tmp_path / "base.fit", lambda: None)


def test_run_job_opens_one_ledger_for_the_attempt(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(core, "_run_locked", lambda directory: seen.append(hashing.snapshot()))
    (tmp_path / "job").mkdir()
    core.run_job(str(tmp_path / "job"))
    assert seen[0]["active"] and seen[0]["label"] == "stack-job" and seen[0]["mode"] == "on"
    monkeypatch.setenv(hashing.ENV_LEDGER, "off")
    core.run_job(str(tmp_path / "job"))
    assert seen[1]["mode"] == "off"


# --------------------------------------------------------------------------- #
# finite_stack / stack_range: parallel verification, same failures              #
# --------------------------------------------------------------------------- #

def legacy_verify_named(work, files, check, message):
    for name, digest in files:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('有限值堆栈收据含无效文件名')
        check()
        if not (work / name).is_file() or fs.sha(work / name) != digest:
            raise ValueError(message)


def _named_files(work):
    files = []
    for i in range(6):
        path = work / f"n{i}.bin"
        path.write_bytes(bytes([i]) * (3000 + i))
        files.append((path.name, sha(path)))
    return files


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["good", "mismatch-1-badname-3", "badname-1-mismatch-3", "missing-2",
                                  "cancel-3", "mismatch-1-cancel-3", "notstr-0"])
def test_finite_verification_fails_like_the_serial_loop(tmp_path, parallel, case, mode):
    work = tmp_path
    files = _named_files(work)
    if "mismatch-1" in case:
        files[1] = (files[1][0], "0" * 64)
    if "mismatch-3" in case:
        files[3] = (files[3][0], "0" * 64)
    if "badname-3" in case:
        files[3] = ("sub/" + files[3][0], files[3][1])
    if "badname-1" in case:
        files[1] = ("../" + files[1][0], files[1][1])
    if case == "missing-2":
        (work / files[2][0]).unlink()
    if case == "notstr-0":
        files[0] = (7, files[0][1])
    results = []
    for runner in (legacy_verify_named, fs._verify_named):
        calls = []

        def check():
            calls.append(1)
            if "cancel-3" in case and len(calls) == 4:
                raise Cancelled("stop")
        with in_mode(mode) if runner is fs._verify_named else nullcontext():
            if runner is fs._verify_named:
                prime(*work.iterdir())
                start = hashing.snapshot()
            results.append((outcome(runner, work, files, check, "有限值堆栈源、序列或独立有效性已改变"), len(calls)))
            if runner is fs._verify_named:
                hits = answered(start)
    assert results[0][0] == results[1][0]
    assert results[1][1] >= results[0][1]
    assert hits == 0 or mode in ("on", "shadow")
    if case == "good":
        assert (hits > 0) == (mode in ("on", "shadow"))


def legacy_describe_engine(executable):
    path = Path(executable).expanduser().resolve()
    manifest = json.loads((path.parent / 'runtime_manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest.get('files', {}).items():
        if not isinstance(name, str) or Path(name).name != name or not (path.parent / name).is_file() or fs.sha(path.parent / name) != digest:
            raise ValueError('有限值堆栈运行时文件缺失或改变：' + str(name))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ["none", "tampered", "missing", "nested-name", "dotdot-name"])
def test_describe_engine_reports_the_same_runtime_file(tmp_path, monkeypatch, parallel, fault, mode):
    folder = tmp_path / "runtime"
    folder.mkdir()
    executable = folder / "siril-cli.exe"
    executable.write_bytes(b"engine")
    files = {executable.name: sha(executable)}
    for i in range(6):
        dll = folder / f"lib{i}.dll"
        dll.write_bytes(bytes([i]) * 4096)
        files[dll.name] = sha(dll)
    if fault == "tampered":
        (folder / "lib2.dll").write_bytes(b"other")
    if fault == "missing":
        (folder / "lib4.dll").unlink()
    if fault == "nested-name":
        files["sub/lib9.dll"] = "0" * 64
    if fault == "dotdot-name":
        files = {"../lib1.dll": files["lib1.dll"], **files}
    manifest = folder / "runtime_manifest.json"
    manifest.write_text(json.dumps({"executable": executable.name, "files": files}), encoding="utf8")
    monkeypatch.setattr(fs, "ENGINE_SHA256", sha(executable))
    monkeypatch.setattr(fs, "RUNTIME_MANIFEST_SHA256", sha(manifest))
    with in_mode(mode):
        prime(*folder.iterdir())
        start = hashing.snapshot()
        new = outcome(fs.describe_engine, str(executable))
        hits = answered(start)
    old = outcome(legacy_describe_engine, str(executable))
    assert (hits > 0) == (mode in ("on", "shadow"))
    if fault == "none":
        assert new[0] == "ok" and old == ("ok", None)
    else:
        assert new == old


def legacy_range_verify(work, records, check):
    for name, digest in records:
        if Path(name).name != name:
            raise ValueError('范围保护收据含无效文件名')
        check()
        if not (work / name).is_file() or sr.sha(work / name) != digest:
            raise ValueError('范围保护的母版、支持或注册源文件已改变，请新建作业')


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["good", "master-changed", "source-missing-badname-later", "badname-source",
                                  "cancel-2"])
def test_range_receipt_verification_fails_like_the_serial_loop(tmp_path, monkeypatch, parallel, case, mode):
    work = tmp_path / "job" / "work"
    work.mkdir(parents=True)
    engine = tmp_path / "siril.exe"
    engine.write_bytes(b"pinned engine")
    monkeypatch.setattr(sr, "ENGINE_SHA256", sha(engine))
    names = ["sky_master.fit", "r_light_00001.fit", "r_light_00002.fit", "r_light_00003.fit", "r_light_.seq", sr.VALID]
    for i, name in enumerate(names):
        (work / name).write_bytes(bytes([i + 1]) * (2000 + i))
    record = {"state": "complete", "contract": sr.contract(), "masters": {"sky_master.fit": sha(work / names[0])},
              "sources": [{"file": name, "sha256": sha(work / name)} for name in names[1:4]],
              "sequence_sha256": sha(work / names[4]), "valid_sha256": sha(work / names[5])}
    if case == "master-changed":
        (work / "sky_master.fit").write_bytes(b"changed")
    if case == "source-missing-badname-later":
        (work / names[2]).unlink()
        record["sources"][2]["file"] = "sub/" + names[3]
    if case == "badname-source":
        record["sources"][0]["file"] = "../" + names[1]
    (work / sr.RECORD).write_text(json.dumps(record), encoding="utf8")
    job = {"stack_range_mode": sr.MODE, "schema_version": 2, "stack_range_contract": sr.contract(),
           "stack_range_sha256": sr.fingerprint(sr.contract()), "engine": {"version": "1.4.4", "executable": str(engine)},
           "checkpoint_files": {"stack": [{"path": "work/" + sr.RECORD, "sha256": sha(work / sr.RECORD)}]}}
    records = [*record["masters"].items(), *[(f["file"], f["sha256"]) for f in record["sources"]],
               ("r_light_.seq", record["sequence_sha256"]), (sr.VALID, record["valid_sha256"])]
    results = []
    for new, runner in ((False, lambda check: legacy_range_verify(work, records, check)),
                        (True, lambda check: sr.verify_receipt(work, job, check=check))):
        calls = []

        def check():
            calls.append(1)
            if case == "cancel-2" and len(calls) == 3:
                raise Cancelled("stop")
        with in_mode(mode) if new else nullcontext():
            if new:
                prime(*work.iterdir(), engine)
                start = hashing.snapshot()
            result = outcome(runner, check)
            if new:
                hits = answered(start)
        results.append((result if result[0] != "ok" else "ok", len(calls)))
    assert results[0][0] == results[1][0]
    assert results[1][1] >= results[0][1]
    if case == "good":
        assert (hits > 0) == (mode in ("on", "shadow"))


@pytest.mark.parametrize("mode", ["on", "shadow"])
@pytest.mark.parametrize("change", ["rewrite", "replace-keeping-size-and-time"])
def test_reverification_after_an_in_scope_record_sees_a_change(tmp_path, parallel, mode, change):
    """Finite and range receipts re-verified in the same scope: a change the stat shows must still fail."""
    work = tmp_path / "work"
    work.mkdir()
    files = _named_files(work)
    for name, _ in files:
        age(work / name)
    message = "有限值堆栈源、序列或独立有效性已改变"
    target = work / files[2][0]
    with hashing.run_scope("t", mode=mode):
        fs._verify_named(work, files, lambda: None, message)                  # every file now in the ledger
        hashing.verify_entries(files, source=sr.SOURCE, error=lambda entry: ValueError("range"),
                               path=lambda entry: work / entry[0], expected=lambda entry: entry[1],
                               precheck=lambda entry, path: path.is_file())
        stamp = os.stat(target)
        if change == "rewrite":
            target.write_bytes(bytes([9]) * stamp.st_size)                    # a new modification time
        else:
            other = work / "replacement.tmp"
            other.write_bytes(bytes([9]) * stamp.st_size)                     # a new file id, same size and time
            os.utime(other, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            os.replace(other, target)
        assert outcome(fs._verify_named, work, files, lambda: None, message) == (ValueError, message)
        assert outcome(lambda: hashing.verify_entries(
            files, source=sr.SOURCE, error=lambda entry: ValueError("range"), path=lambda entry: work / entry[0],
            expected=lambda entry: entry[1], precheck=lambda entry, path: path.is_file())) == (ValueError, "range")


from test_finite_stack import prepared, runtime, run as finite_run  # noqa: E402,F401  (fixtures)


@pytest.mark.parametrize("mode", ["on", "shadow"])
def test_finite_run_and_finish_in_a_ledger_record_real_digests(ntfs, prepared, runtime, mode):
    directory, work = prepared[0], prepared[1]
    for path in work.iterdir():
        age(path)
    with hashing.run_scope("t", mode=mode):
        start = hashing.snapshot()
        record = finite_run(prepared, runtime)
        support = rs.finish_stack(work, work / "sky_master.fit", None, explicit_counts=True)
        final = fs.finalize_receipt(work)
        moved = delta(start, hashing.snapshot())
    assert record["counts"] == {fs.NVALID: sha(work / fs.NVALID), fs.NUSED: sha(work / fs.NUSED)}
    assert record["valid_sha256"] == sha(work / fs.VALID)
    assert record["decoded_sha256"] == sha(directory / record["attempt"] / "finite_master.fit")
    assert all(source["sha256"] == sha(work / source["file"]) for source in record["sources"])
    assert support["stack_sha256"] == sha(work / "sky_master.fit")
    assert support["count_files"] == record["counts"]
    assert final["masters"] == {"sky_master.fit": sha(work / "sky_master.fit")}
    if mode == "on":
        assert moved["ledger"] > 0 and moved["write"] >= 2       # re-verified sources; registered count copies
    else:
        assert moved["mismatches"] == 0 and moved["shadow"] > 0


def test_read_ahead_returns_or_raises_each_digest_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv(hashing.ENV_WORKERS, "3")
    paths = []
    for i in range(7):
        path = tmp_path / f"r{i}.fit"
        path.write_bytes(bytes([i]) * 70000)
        paths.append(path)
    paths[4].unlink()
    ahead = sr._ReadAhead(paths)
    try:
        assert ahead.pool is not None
        assert [ahead.get(i) for i in range(4)] == [sha(p) for p in paths[:4]]
        with pytest.raises(FileNotFoundError) as raised:
            ahead.get(4)
        with pytest.raises(FileNotFoundError) as serial:
            sr.sha(paths[4])
        assert str(raised.value) == str(serial.value)
    finally:
        ahead.close()
    monkeypatch.setenv(hashing.ENV_WORKERS, "1")
    serial_ahead = sr._ReadAhead(paths[:3])
    assert serial_ahead.pool is None and serial_ahead.get(2) == sha(paths[2])


# --------------------------------------------------------------------------- #
# A whole controlled job: off / on / shadow, and the hash handover              #
# --------------------------------------------------------------------------- #

def _seq_text(count, reference_position):
    lines = ["#Siril sequence file.", f"S 'light_' 1 {count} {count} 5 {reference_position} 6 0 0 0", "L 3"]
    lines += [f"I {index} 1" for index in range(1, count + 1)]
    for position in range(count):
        shift = 0.0 if position == reference_position else 1.0 * (position - reference_position)
        lines.append(f"R1 3.5 3.6 0.7 0 1500 1800 H 1 0 {shift} 0 1 {shift} 0 0 1")
    return "\n".join(lines) + "\n"


@pytest.fixture
def masked_job(tmp_path, monkeypatch):
    source = tmp_path / "sources"
    source.mkdir()
    for number in range(3):
        data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
        data[3:6, 5:8] += .2
        data[1, 2 + number] = .6
        imaging.write_linear(source / f"frame {number}.fit", data, "Test fixture")
    mask = np.full((18, 24), 255, dtype=np.uint8)
    mask[12, :] = 128
    mask[13:, :] = 0
    mask_png = tmp_path / "sky_mask.png"
    Image.fromarray(mask).save(mask_png)
    engine = {"available": True, "executable": str(tmp_path / "engine.exe"), "version": "1.4.4", "message": "controlled"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    state = {"fail": False}

    def run_script(executable, directory, name, commands, check):
        check()
        work = directory / "work"
        if name == "01_convert":
            for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif name == "02_probe":
            pass
        elif name == "02_export":   # P1b-3: L's export process (registration writes the sequence file)
            assert list(commands) == core.export_script(None), commands
            frames = sorted(work.glob("light_*.fit"))
            (work / core.SEQUENCE_FILE).write_text(_seq_text(len(frames), 1), encoding="utf-8")
        elif name == "02_stack":
            assert list(commands) == list(core.LEGACY_STACK_SCRIPT), commands
            if state["fail"]:
                raise RuntimeError("controlled stack failure")
            frames = sorted(work.glob("light_*.fit"))
            data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32)
            imaging.write_linear(work / "sky_master.fit", data, "Controlled engine, not Siril")
        else:
            raise AssertionError(name)
    monkeypatch.setattr(core, "run_script", run_script)

    def run(label, mode, token=None):
        monkeypatch.setenv(hashing.ENV_LEDGER, mode)
        if token is None:
            monkeypatch.delenv(core.HANDOVER_TOKEN_ENV, raising=False)
        else:
            monkeypatch.setenv(core.HANDOVER_TOKEN_ENV, token)
        job = core.submit_job(str(source), str(tmp_path / label), sky_mask_path=str(mask_png), transient_mode="off")
        start = hashing.snapshot()
        core.run_job(job["job_dir"])
        moved = delta(start, hashing.snapshot())
        return core.load_job(job["job_dir"]), moved
    return SimpleNamespace(run=run, state=state)


def _without_icc_date(data):
    """An embedded ICC profile carries its creation time (bytes 24-36; volatile, as in the workflow verifier)."""
    signature = data.find(b"acsp")
    if signature < 36:
        return data
    start = signature - 36
    return data[:start + 24] + bytes(12) + data[start + 36:]


def _comparable(job):
    """What must be identical between runs of the same inputs: digests, records, product bytes.

    Volatile by design: the ``applied_at`` time in sky_validity.json (so its checkpoint digest)
    and the creation time inside embedded ICC profiles.
    """
    directory = Path(job["job_dir"])

    def relative(path):   # products/<random export id>/name -> products/<export>/name
        parts = list(Path(path).relative_to(directory).parts)
        if parts[0] == "products":
            parts[1] = "<export>"
        return "/".join(parts)
    products = {relative(a["path"]): hashlib.sha256(_without_icc_date(Path(a["path"]).read_bytes())).hexdigest()
                for a in job["artifacts"]}
    validity = json.loads((directory / "work" / "sky_validity.json").read_text(encoding="utf-8"))
    validity.pop("applied_at")
    checkpoints = {stage: [r for r in records if not r["path"].endswith("sky_validity.json")]
                   for stage, records in job["checkpoint_files"].items()}
    weight = job["diagnostics"]["sky_validity"]["compose_weight"]
    return {"inputs": [(Path(i["path"]).name, i["size"], i["sha256"]) for i in job["inputs"]],
            "signature": job["signature"], "checkpoints": job["checkpoints"],
            "checkpoint_files": checkpoints, "products": products, "validity": validity,
            "weight": (weight["sha256"], weight["composite_sha256"], weight["shape"]),
            "message": job["message"], "state": job["state"]}


def test_a_controlled_masked_job_records_the_same_digests_in_every_mode(ntfs, masked_job):
    results = {mode: masked_job.run(mode, mode) for mode in ("off", "on", "shadow")}
    for mode, (job, _) in results.items():
        assert job["state"] == "succeeded", (mode, job.get("error"))
    baseline = _comparable(results["off"][0])
    assert _comparable(results["on"][0]) == baseline
    assert _comparable(results["shadow"][0]) == baseline
    on, shadow = results["on"][1], results["shadow"][1]
    assert on["write"] > 0                                   # fused pass, write_linear, save digests hit
    assert shadow["mismatches"] == 0 and shadow["shadow"] > 0
    assert results["off"][1]["write"] == 0
    for mode, (job, _) in results.items():                   # the job's own report, in its timing notes
        note = job["diagnostics"]["timings"]["notes"]["hash_ledger"]
        assert (note["label"], note["mode"]) == ("stack-job", mode)
        assert note["shadow"]["mismatches"] == 0 and (note["shadow"]["checks"] > 0) == (mode == "shadow")
        assert (note["saved"]["write"]["files"] > 0) == (mode != "off")
    for stage, records in baseline["checkpoint_files"].items():
        directory = Path(results["on"][0]["job_dir"])
        for record in records:
            assert record["sha256"] == sha(directory / record["path"]), (stage, record["path"])


def test_the_job_worker_hands_its_digests_to_its_launcher(ntfs, masked_job):
    token = uuid.uuid4().hex
    job, _ = masked_job.run("handover", "on", token=token)
    assert job["state"] == "succeeded", job.get("error")
    handover = hashing.handover_path(token)
    assert handover.is_file()
    assert json.loads(handover.read_text(encoding="utf-8"))["root"] == os.path.normcase(os.path.realpath(job["job_dir"]))
    notes = job["diagnostics"]["timings"]["notes"]["hash_handover"]
    directory = Path(job["job_dir"])
    with hashing.run_scope("launcher", mode="shadow"):
        assert hashing.import_handover(None, job["job_dir"], token=token) == notes["exported"] > 0
        start = hashing.snapshot()
        for records in job["checkpoint_files"].values():
            for record in records:
                assert core._sha(directory / record["path"]) == record["sha256"]
        moved = delta(start, hashing.snapshot())
    assert not handover.exists()
    assert moved["mismatches"] == 0 and moved["shadow"] > 0


def test_a_job_that_does_not_succeed_leaves_no_handover(ntfs, masked_job, monkeypatch):
    masked_job.state["fail"] = True
    token = uuid.uuid4().hex
    job, _ = masked_job.run("failed", "on", token=token)
    assert job["state"] == "failed"
    assert not hashing.handover_path(token).exists()
    masked_job.state["fail"] = False
    # Exported, then the terminal update fails: the handover is removed with the failure.
    real_update, seen = core._update, {}

    def update(job_dir, **changes):
        if changes.get("state") == "succeeded":
            seen["exported"] = hashing.handover_path(token).is_file()
            raise OSError("controlled publication failure")
        return real_update(job_dir, **changes)
    monkeypatch.setattr(core, "_update", update)
    job, _ = masked_job.run("late-failure", "on", token=token)
    assert job["state"] == "failed" and seen["exported"]
    assert not hashing.handover_path(token).exists()
    monkeypatch.setattr(core, "_update", real_update)
    job, _ = masked_job.run("no-token", "on")
    assert job["state"] == "succeeded" and "hash_handover" not in job["diagnostics"]["timings"].get("notes", {})
