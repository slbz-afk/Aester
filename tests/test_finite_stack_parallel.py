"""Efficiency P1c F-3a: the streamed q_ writer gives the reference body's bytes (docs/EFFICIENCY_P1C_PREREG_20261005.md
§3.3, "验证方法的改动").

q_ copies inherit the export's DATE card, so runs are never compared by raw q_ digests; instead the same r_light
frames are written by both bodies in one run and compared byte for byte (success and failure paths, header growth,
fallbacks, buffer reuse), and the real F48 frames are compared with the reference job's q_ by their FITS manifest
(cards without DATE / CHECKSUM / DATASUM, data digest). The numeric comparison runs again under
NPY_DISABLE_CPU_FEATURES=X86_V3 and the other numeric environments of §6.2 (g).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from pathlib import Path

import numpy as np
import pytest

import p1c_site_harness as harness
from finite_fixtures import (FixedPools, MeanEngine, f48_attempt, f48_sources, fake_runtime, fits_manifest,
                             fixed_checksum_time, frame_values, make_work, q_digests, siril_like, streamed)
from starstack import finite_stack as fs
from starstack.siril import Cancelled

HEIGHT, WIDTH = 150, 40          # three 64-row tiles, the last one short
REAL_FRAMES = int(os.environ.get("AESTER_P1C_FINITE_REAL_FRAMES", "2"))


@pytest.fixture(autouse=True)
def _checksum_time():
    with fixed_checksum_time():
        yield


def _valid(rng, height=HEIGHT, width=WIDTH):
    valid = rng.random((height, width)) > 0.2
    valid[:5, :] = False
    return valid


def _case(name):
    """(bytes of the source, valid) of one A/B case."""
    rng = np.random.default_rng(sum(map(ord, name)))      # not hash(): PYTHONHASHSEED would vary it
    valid = _valid(rng)
    data = frame_values(rng, HEIGHT, WIDTH, valid)
    file_rows = valid[::-1]
    if name == "plain":
        return siril_like(data), valid
    if name == "checksum_cards":
        return siril_like(data, checksum=True), valid
    if name == "header_grows":                         # 35 cards + END: the four new cards need a second block
        return siril_like(data, filler=15), valid
    if name == "tile_rows":                            # exactly one tile, then one row more
        valid = _valid(rng, 65)
        return siril_like(frame_values(rng, 65, WIDTH, valid)), valid
    if name == "signed_zeros_only":
        data[:, file_rows] = -0.0
        data[1, file_rows] = 0.0
        return siril_like(data), valid
    if name == "nonstandard_card":                     # verify('exception') refuses it; update-mode flush fixes it
        raw = bytearray(siril_like(data))
        card = "badkey  = 'not a FITS keyword'".ljust(80).encode("ascii")
        end = raw.index(b"END" + b" " * 77)
        raw[end:end + 80] = card
        raw[end + 80:end + 160] = b"END" + b" " * 77
        return bytes(raw), valid
    if name == "out_of_range":                         # a valid sample >= 8 in the second tile
        row, col = 70, int(np.flatnonzero(file_rows[70])[0])
        data[1, row, col] = 8.25
        return siril_like(data), valid
    if name == "non_finite":
        data[2, 100, 3] = np.nan
        return siril_like(data), valid
    if name == "no_valid_sample":
        valid = np.zeros_like(valid)
        return siril_like(np.zeros_like(data)), valid
    if name == "geometry":
        return siril_like(data), _valid(rng, HEIGHT, WIDTH + 1)
    if name == "top_down":
        return siril_like(data, roworder="TOP-DOWN"), valid
    if name == "no_roworder":
        return siril_like(data, roworder=None), valid
    if name == "int16":
        return siril_like((data * 100).astype(np.int16), bitpix=16), valid
    if name == "extra_hdu":
        return siril_like(data, extra_hdu=True), valid
    raise AssertionError(name)


SUCCESS = ("plain", "checksum_cards", "header_grows", "tile_rows", "signed_zeros_only", "nonstandard_card")
FAILURE = ("out_of_range", "non_finite", "no_valid_sample", "geometry", "top_down", "no_roworder", "int16",
           "extra_hdu")


def _outcome(run):
    try:
        return ("ok",) + tuple(run())
    except Exception as exc:                            # noqa: BLE001 -- the test compares how each body ended
        return (type(exc), str(exc))


def _ab(folder, payload, valid, *, check=lambda: None, buffer=None):
    """Write q_ from the same source with the reference body and the streamed body: {body: (outcome, q_ bytes)}."""
    folder = Path(folder)
    source = folder / "r_light_00001.fit"
    if payload is not None:
        source.write_bytes(payload)
    found = {}
    for body in ("reference", "streamed"):
        target = folder / body / "q_00001.fit"
        target.parent.mkdir(exist_ok=True)
        if body == "reference":
            outcome = _outcome(lambda: (fs._write_input(source, target, valid, check), None))
        else:
            outcome = _outcome(lambda: fs._write_input_streamed(source, target, valid, check,
                                                                buffer or fs._ReadBuffer()))
        found[body] = (outcome, target.read_bytes() if target.exists() else None)
    return found


@pytest.mark.parametrize("name", SUCCESS + FAILURE)
def test_both_bodies_give_the_same_bytes_details_and_errors(tmp_path, name):
    payload, valid = _case(name)
    found = _ab(tmp_path, payload, valid)
    (reference, reference_bytes), (stream, stream_bytes) = found["reference"], found["streamed"]
    if name in SUCCESS:
        assert reference[0] == "ok", reference
        harness.assert_same(reference[1], stream[1], name)              # details: key order, float bits
        assert stream[2] == ("plan:VerifyError" if name == "nonstandard_card" else None), stream
    else:
        assert reference[0] is not None and reference[0] != "ok", reference
        assert stream == reference, (stream, reference)                 # the same exception type and message
    assert reference_bytes is not None
    assert stream_bytes == reference_bytes, name
    if name == "header_grows":
        assert len(reference_bytes) == len(payload) + 2880, "the q_ header grew a block (astropy's resize path)"


def test_a_missing_source_fails_like_the_reference(tmp_path):
    valid = _valid(np.random.default_rng(3))
    found = _ab(tmp_path, None, valid)
    assert found["streamed"] == found["reference"]
    assert found["reference"][0][0] is FileNotFoundError and found["reference"][1] is None


@pytest.mark.parametrize("change", ["short", "longer", "identity"])
def test_a_source_that_does_not_match_its_plan_goes_to_the_reference_body(tmp_path, monkeypatch, change):
    payload, valid = _case("plain")
    planned = fs._stream_plan

    def wrong(source, valid_):
        size, offset, nbytes, shape, identity = planned(source, valid_)
        if change == "short":                          # the file ends before the planned size
            return size + 2880, offset, nbytes, shape, identity
        if change == "longer":                         # bytes after the planned size
            return size - 2880, offset, nbytes, shape, identity
        return size, offset, nbytes, shape, identity[:-1] + (identity[-1] + 1,)
    monkeypatch.setattr(fs, "_stream_plan", wrong)
    found = _ab(tmp_path, payload, valid)
    assert found["streamed"][1] == found["reference"][1]
    assert found["streamed"][0][1:2] == found["reference"][0][1:2]
    assert found["streamed"][0][2] == ("read:short" if change == "short" else "read:changed")


def test_one_buffer_serves_frames_of_any_size_without_leaking_bytes(tmp_path):
    """R12: a reused buffer never lets an earlier frame's bytes into a later q_ (a larger, a smaller, a larger
    frame again, and a failed one in between)."""
    rng = np.random.default_rng(7)
    buffer = fs._ReadBuffer()
    for number, (height, width, poison) in enumerate([(150, 40, False), (70, 24, False), (150, 40, True),
                                                      (150, 40, False), (90, 40, False)]):
        valid = _valid(rng, height, width)
        data = frame_values(rng, height, width, valid)
        if poison:
            data[0, 80, int(np.flatnonzero(valid[::-1][80])[0])] = 9.0
        folder = tmp_path / f"frame{number}"
        folder.mkdir()
        found = _ab(folder, siril_like(data, number=number + 1), valid, buffer=buffer)
        assert found["streamed"][0][:2] == found["reference"][0][:2], number
        assert found["streamed"][1] == found["reference"][1], number


def _held_after_failure(tmp_path, payload, valid, check):
    buffer = fs._ReadBuffer()
    source = tmp_path / "r_light_00001.fit"
    source.write_bytes(payload)
    with pytest.raises(BaseException) as info:
        fs._write_input_streamed(source, tmp_path / "q_00001.fit", valid, check, buffer)
    array = weakref.ref(buffer.array)
    buffer.release()
    return info.value, array


def test_a_failure_traceback_keeps_neither_the_buffer_nor_a_view_of_it(tmp_path):
    """§3.3: a failed frame's traceback would otherwise keep the 548 MB buffer to the end of the job."""
    payload, valid = _case("out_of_range")
    error, array = _held_after_failure(tmp_path, payload, valid, lambda: None)
    assert isinstance(error, ValueError) and error.__traceback__ is not None
    assert array() is None

    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 3:
            raise Cancelled("用户取消")
    (tmp_path / "cancel").mkdir()
    error, array = _held_after_failure(tmp_path / "cancel", payload, valid, cancel)
    assert isinstance(error, Cancelled) and array() is None


def test_a_cancel_while_planning_is_not_a_fallback(tmp_path, monkeypatch):
    payload, valid = _case("plain")

    def cancelled(*_):
        raise Cancelled("用户取消")
    monkeypatch.setattr(fs, "_stream_plan", cancelled)
    (tmp_path / "r_light_00001.fit").write_bytes(payload)
    with pytest.raises(Cancelled):
        fs._write_input_streamed(tmp_path / "r_light_00001.fit", tmp_path / "q.fit", valid, lambda: None,
                                 fs._ReadBuffer())
    assert not (tmp_path / "q.fit").exists()


def test_the_plan_maps_read_only_and_returns_integers(tmp_path):
    payload, valid = _case("plain")
    source = tmp_path / "r_light_00001.fit"
    source.write_bytes(payload)
    size, offset, nbytes, shape, identity = fs._stream_plan(source, valid)
    assert (size, offset, nbytes, shape) == (len(payload), 2880, 3 * HEIGHT * WIDTH * 4, (3, HEIGHT, WIDTH))
    assert all(type(value) is int for value in (size, offset, nbytes, *shape, *identity))
    source.unlink()                                     # no mapping or handle is left open (Windows would refuse)


# ---- the pool on whole jobs (run) ----------------------------------------------------------------------------------

def _run(root, runtime, pools=None, check=lambda: None, engine=None):
    job = make_work(root, count=7)
    engine = engine or MeanEngine(job.count, 2)
    record = fs.run(runtime, job.directory, job.work, check, engine, pools=pools)
    attempt = job.directory / record["attempt"]
    return job, record, attempt, engine


def _comparable(record):
    return {k: v for k, v in record.items() if k not in ("attempt", "seconds")}


@pytest.mark.parametrize("pools", [None, "reference", "streamed-1", "streamed-4", "streamed-16"])
def test_run_gives_the_same_record_and_q_bytes_for_any_pool_plan(tmp_path, monkeypatch, pools):
    runtime = fake_runtime(tmp_path / "runtime", monkeypatch)
    _, base, base_attempt, _ = _run(tmp_path / "base", runtime)
    plan = {None: None, "reference": FixedPools(write=fs.Plan(8, None, fs.REFERENCE)),
            "streamed-1": streamed(1), "streamed-4": streamed(4), "streamed-16": streamed(16)}[pools]
    if plan is not None:
        plan.inventory_plan = fs.Plan(plan.write_plan.workers)
    _, record, attempt, engine = _run(tmp_path / "pool", runtime, pools=plan)
    harness.assert_same(harness.json_bytes(_comparable(base)), harness.json_bytes(_comparable(record)))
    assert q_digests(attempt) == q_digests(base_attempt)
    assert engine.sections == [()]                      # the finite engine is not a stock stack launch
    if plan is not None:
        write = plan.stats("finite_write")
        assert write["workers"] == (1 if pools == "reference" else min(int(pools.split("-")[1]), 7))
        assert [site for site, _, _ in plan.notes] == ["finite_inventory", "finite_write"]
        assert plan.operations == ["finite_inventory", "finite_write"]
        assert [s[0] for s in plan.shadows] == ["py.finite_inventory", "py.finite_write"]
        assert plan.notes[-1][2]["body"] == plan.write_plan.body and plan.notes[-1][2]["fallbacks"] == {}


def test_frame_two_s_data_error_wins_over_frame_five_s_injected_one(tmp_path, monkeypatch):
    """§3.3: frame 2's data error is in the fixture before its export digest is recorded; frame 5's is injected into
    the write pool (its support map's second load fails). Every parallelism reports frame 2's, as the serial loop."""
    from starstack import registered_support as rs

    def poisoned(number, data):
        if number == 2:
            data = data.copy()
            data[0, 100, 20] = np.inf
        return siril_like(data, number=number)
    outcomes = {}
    runtime = fake_runtime(tmp_path / "runtime", monkeypatch)
    for workers in (1, 2, 4, 8, 16):
        root = tmp_path / f"w{workers}"
        job = make_work(root, count=7, frame_bytes=poisoned)
        loads = {}
        real = rs.load_frame_support

        def load(work, descriptor):
            name = descriptor["file"]
            loads[name] = loads.get(name, 0) + 1
            if name == "frame_support_light_00005.npy" and loads[name] == 2:
                raise RuntimeError("injected: frame 5's support map, second load")
            return real(work, descriptor)
        monkeypatch.setattr(rs, "load_frame_support", load)
        pools = streamed(workers)
        pools.inventory_plan = fs.Plan(workers)
        with pytest.raises(Exception) as info:
            fs.run(runtime, job.directory, job.work, lambda: None, MeanEngine(job.count, 2), pools=pools)
        monkeypatch.setattr(rs, "load_frame_support", real)
        outcomes[workers] = (type(info.value), str(info.value))
        assert loads["frame_support_light_00005.npy"] >= 1
    assert set(outcomes.values()) == {(ValueError, "配准源包含非有限值，不能推断其有效性")}, outcomes


# ---- numeric environments (§6.2 (g), §3.3 "NPY_DISABLE_CPU_FEATURES=X86_V3") ---------------------------------------

def numeric_q(folder, body):
    """q_ bytes and details of the A/B cases written by ``body`` (run in a fresh interpreter)."""
    with fixed_checksum_time(), tempfile.TemporaryDirectory(dir=folder) as scratch:
        out = []
        for name in SUCCESS:
            payload, valid = _case(name)
            source = Path(scratch) / f"{name}.fit"
            source.write_bytes(payload)
            target = Path(scratch) / f"q_{name}.fit"
            if body == "reference":
                details = fs._write_input(source, target, valid, lambda: None)
            else:
                details = fs._write_input_streamed(source, target, valid, lambda: None, fs._ReadBuffer())[0]
            out.append([target.read_bytes(), details])
        return out


@pytest.mark.parametrize("env", [{}, *harness.NUMERIC_ENVS], ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items())
                         or "default")
def test_both_bodies_agree_under_each_numeric_environment(tmp_path, env):
    digests = {body: harness.subprocess_digest("test_finite_stack_parallel:numeric_q", str(tmp_path), body, env=env)
               for body in ("reference", "streamed")}
    assert digests["reference"] == digests["streamed"], digests


def test_the_bodies_do_not_depend_on_the_hash_seed(tmp_path):
    found = {body: harness.hashseed_digests("test_finite_stack_parallel:numeric_q", str(tmp_path), body)
             for body in ("reference", "streamed")}
    assert len({d for per in found.values() for d in per.values()}) == 1, found


# ---- real data: the F48 reference job ------------------------------------------------------------------------------

def test_real_frames_both_bodies_same_bytes_and_the_reference_job_s_manifest(tmp_path):
    """Same-run A/B on real F48 r_light frames, and each q_'s FITS manifest equals the reference job's q_."""
    job = harness.f48_job()
    work, sources, _, _ = f48_sources(job, REAL_FRAMES)
    attempt = f48_attempt(job)
    from starstack import registered_support as rs
    buffer = fs._ReadBuffer()
    for number, source in enumerate(sources, 1):
        mask = rs.load_frame_support(work, source["validity"])
        folder = tmp_path / f"q{number}"
        folder.mkdir()
        reference = fs._write_input(work / source["file"], folder / "reference.fit", mask, lambda: None)
        details, fallback = fs._write_input_streamed(work / source["file"], folder / "streamed.fit", mask,
                                                     lambda: None, buffer)
        assert fallback is None
        harness.assert_same(reference, details, f"frame {number}")
        assert (folder / "reference.fit").read_bytes() == (folder / "streamed.fit").read_bytes(), number
        assert fits_manifest(folder / "streamed.fit") == fits_manifest(attempt / f"q_{number:05d}.fit"), number
        shutil.rmtree(folder)
