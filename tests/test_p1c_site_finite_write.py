"""A5 for the efficiency P1c site finite_write (F-3a, docs/EFFICIENCY_P1C_PREREG_20261005.md §3.3, §6.2 (a)-(h)).

finite_stack._write_inputs (the q_ copies of the finite stack, one frame per unit) against the retained pre-P1c loop
finite_stack._finite_write_reference: q_ bytes, the sources' details (key order, float bits) and the reference
position, for the synthetic matrix (workers x in flight x state policy x jitter), at real size on the F48 reference job
(workers 1, the site's default, 16), under two hash seeds; failures at frames {0, 1, middle, last} (the serial error,
the delivered prefix, the files a failed run leaves: the serial set plus q_ of frames after the failure, prereg §5
item 2), at job level through finite_stack.run (receipt) and resumed; cancels within 1 s (R18); the pool's memory peak
within its reservation; and core's plan (_FinitePools.write) for given memory figures.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

import p1c_site_harness as harness
from finite_fixtures import (FixedPools, MeanEngine, f48_sources, fake_runtime, fixed_checksum_time, make_work,
                             q_digests, siril_like, sources_of, streamed)
from starstack import core, finite_stack as fs, frame_pool, performance, registered_support as rs

COUNT = 7
REAL_FRAMES = int(os.environ.get("AESTER_P1C_FINITE_REAL_FRAMES", "4"))
GiB = 1 << 30


@pytest.fixture(autouse=True)
def _checksum_time():
    with fixed_checksum_time():
        yield


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    return make_work(tmp_path_factory.mktemp("finite-write") / "job", count=COUNT)


def _expected(work):
    return np.load(Path(work) / rs.COUNTS)


def _reference_position(sources, reference_file):
    return next(i for i, s in enumerate(sources, 1) if s["file"] == reference_file)


REFERENCE_FILE = "r_light_00003.fit"      # make_work(reference=2): the sequence's third image


@pytest.fixture(scope="module")
def reference(job, tmp_path_factory):
    """The pre-P1c loop's outputs: (q_ bytes by name, sources after the loop, reference position)."""
    attempt = tmp_path_factory.mktemp("reference-attempt")
    sources = sources_of(job.work)
    with fixed_checksum_time():
        ref = fs._finite_write_reference(job.work, attempt, sources, REFERENCE_FILE, _expected(job.work), lambda: None)
    return {p.name: p.read_bytes() for p in sorted(attempt.glob("q_*.fit"))}, sources, ref


def _pool_run(job, attempt, pools, check=lambda: None):
    sources = sources_of(job.work)
    ref = fs._write_inputs(job.work, attempt, sources, REFERENCE_FILE, _expected(job.work), check, pools)
    return {p.name: p.read_bytes() for p in sorted(attempt.glob("q_*.fit"))}, sources, ref


# ================================================================== (a) reference vs pool

@pytest.mark.parametrize("case", harness.params(harness.synthetic_cases()))
def test_synthetic_matrix_matches_the_reference(job, reference, tmp_path, case):
    before = harness.aester_threads()
    pools = streamed(case.workers, case.inflight)
    with harness.pool_conditions(case):
        found = _pool_run(job, tmp_path, pools)
    harness.assert_same(reference[0], found[0], "q_ bytes")
    harness.assert_same(harness.json_bytes(reference[1]), harness.json_bytes(found[1]), "sources")
    assert found[2] == reference[2]
    stats = pools.stats("finite_write")
    assert stats["workers"] == min(case.workers, COUNT) and stats["serial"] == (case.workers == 1)
    harness.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [1, 8])
def test_the_reference_body_runs_on_one_worker(job, reference, tmp_path, workers):
    pools = FixedPools(write=fs.Plan(workers, None, fs.REFERENCE))
    with harness.pool_conditions():
        found = _pool_run(job, tmp_path, pools)
    harness.assert_same(reference[0], found[0], "q_ bytes")
    assert found[1] == reference[1] and found[2] == reference[2]
    stats = pools.stats("finite_write")
    assert stats["serial"] and stats["workers"] == 1 and stats["reason"] == "workers"
    assert pools.notes[-1][2] == {"body": fs.REFERENCE, "fallbacks": {}}


def hashseed_side(folder, side):
    """The finite_write outputs of one side ('reference' or 'pool', 4 workers) on a fresh synthetic job."""
    root = Path(folder) / f"{side}-{os.getpid()}-{time.perf_counter_ns()}"
    job = make_work(root / "job", count=COUNT)
    attempt = root / "attempt"
    attempt.mkdir()
    sources = sources_of(job.work)
    with fixed_checksum_time():
        if side == "reference":
            ref = fs._finite_write_reference(job.work, attempt, sources, REFERENCE_FILE, _expected(job.work),
                                             lambda: None)
        else:
            ref = fs._write_inputs(job.work, attempt, sources, REFERENCE_FILE, _expected(job.work), lambda: None,
                                   streamed(4))
    found = [[p.name, p.read_bytes()] for p in sorted(attempt.glob("q_*.fit"))], sources, ref
    shutil.rmtree(root)
    return found


def test_both_sides_do_not_depend_on_the_hash_seed(tmp_path):
    digests = {side: harness.hashseed_digests("test_p1c_site_finite_write:hashseed_side", str(tmp_path), side)
               for side in ("reference", "pool")}
    assert len({d for per in digests.values() for d in per.values()}) == 1, digests


# ---- real size --------------------------------------------------------------------------------------------------------

_REAL = {}


def _real_reference(tmp_path_factory):
    """The F48 reference job's first REAL_FRAMES sources through the pre-P1c loop: q_ digests, sources, position."""
    if "reference" not in _REAL:
        job = harness.f48_job()
        work, sources, expected, _ = f48_sources(job, REAL_FRAMES)
        attempt = tmp_path_factory.mktemp("f48-reference")
        ref = fs._finite_write_reference(work, attempt, sources, sources[-1]["file"], expected, lambda: None)
        _REAL["reference"] = (q_digests(attempt), sources, ref)
        _REAL["inputs"] = (work, expected)
        shutil.rmtree(attempt)
    return _REAL["reference"], _REAL["inputs"]


def _real_pool(tmp_path, pools, check=lambda: None, *, keep=False, window=None):
    """The pool on the F48 sources into tmp_path/attempt: (q_ digests, sources, position). The q_ copies are removed
    afterwards unless ``keep`` (a cancel test times the pool, not the deletion of 548 MB files). ``window``, a context
    manager (a PeakSampler), is entered around the pool alone: the digests read each q_ whole (548 MB) afterwards."""
    reference, (work, expected) = _REAL["reference"], _REAL["inputs"]
    sources = [{key: s[key] for key in ("file", "sha256", "validity", "supported_pixels")} for s in reference[1]]
    attempt = tmp_path / "attempt"
    attempt.mkdir(exist_ok=True)
    try:
        with window if window is not None else contextlib.nullcontext():
            ref = fs._write_inputs(work, attempt, sources, sources[-1]["file"], expected, check, pools)
        return q_digests(attempt), sources, ref
    finally:
        if not keep:
            shutil.rmtree(attempt, ignore_errors=True)


class _Recorded(core._FinitePools):
    """core's own plan (``write_setting``: AESTER_FINITE_WRITE's value), its notes kept."""

    def __init__(self, write_setting="auto"):
        super().__init__(None, write_setting)
        self.notes = []

    def note_pool(self, site, stats, **extra):
        self.notes.append((site, dict(stats), extra))
        super().note_pool(site, stats, **extra)


@pytest.mark.parametrize("case", harness.params(harness.real_cases()))
def test_real_size_matches_the_reference(tmp_path_factory, tmp_path, monkeypatch, case):
    reference, _ = _real_reference(tmp_path_factory)
    if case.workers is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
        pools = _Recorded()                                      # the site's own default decision
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, str(case.workers))
        pools = _Recorded()
    before = harness.aester_threads()
    with harness.pool_conditions(case):
        found = _real_pool(tmp_path, pools)
    assert found[0] == reference[0]
    harness.assert_same(harness.json_bytes(reference[1]), harness.json_bytes(found[1]), "sources")
    assert found[2] == reference[2]
    _, stats, extra = pools.notes[-1]
    assert extra["body"] == fs.STREAMED and extra["fallbacks"] == {}
    decided = pools.decisions["finite_write"][0]["workers"]              # core's plan (cap, CPUs, memory, override)
    assert stats["workers"] == min(decided, REAL_FRAMES), (stats, pools.decisions)
    if case.workers is not None:                                         # the override, clamped to the frames
        assert decided == min(case.workers, frame_pool.MAX_FRAME_WORKERS, REAL_FRAMES)
    harness.assert_pool_quiet(before)


# ================================================================== (c) failures

def _tracked_buffers(monkeypatch):
    made = []

    class Tracked(fs._ReadBuffer):
        __slots__ = ()

        def __init__(self):
            super().__init__()
            made.append(self)
    monkeypatch.setattr(fs, "_ReadBuffer", Tracked)
    return made


def _failing_load(monkeypatch, failing_file):
    real = rs.load_frame_support

    def load(work, descriptor):
        if descriptor["file"] == failing_file:
            raise RuntimeError(f"injected failure loading {failing_file}")
        return real(work, descriptor)
    monkeypatch.setattr(rs, "load_frame_support", load)


def _attempt_outcome(job, attempt, pools, delivered):
    sources = sources_of(job.work)
    try:
        if pools is None:                                        # the pre-P1c loop
            fs._finite_write_reference(job.work, attempt, sources, REFERENCE_FILE, _expected(job.work), lambda: None)
        else:
            fs._write_inputs(job.work, attempt, sources, REFERENCE_FILE, _expected(job.work), lambda: None, pools)
    finally:
        delivered.extend(i for i, s in enumerate(sources) if "minimum" in s)


@pytest.mark.parametrize("k", harness.failure_positions(COUNT))
@pytest.mark.parametrize("kind", ["load", "data"])
def test_a_failure_at_frame_k_is_the_serial_one(tmp_path, monkeypatch, k, kind):
    if kind == "data":                                   # frame k+1 holds a valid sample >= 8
        def frame_bytes(number, data):
            if number == k + 1:
                data = data.copy()
                data[1, 70, int(np.argmax(data[1, 70] != 0))] = 9.5
            return siril_like(data, number=number)
        job = make_work(tmp_path / "job", count=COUNT, frame_bytes=frame_bytes)
    else:
        job = make_work(tmp_path / "job", count=COUNT)
        _failing_load(monkeypatch, f"frame_support_light_{k + 1:05d}.npy")
    outcomes, trees = {}, {}
    for workers in ("reference", 1, 2, 4, 8, 16):
        attempt = tmp_path / f"attempt-{workers}"
        attempt.mkdir()
        made = _tracked_buffers(monkeypatch)
        before = harness.aester_threads()
        pools = None if workers == "reference" else streamed(workers)
        with harness.pool_conditions(harness.Case(1 if workers == "reference" else workers, None, "reuse", k)):
            outcomes[workers] = harness.outcome(lambda delivered: _attempt_outcome(job, attempt, pools, delivered))
        assert pools is None or (made and all(b.array is None for b in made)), "finalize released every buffer"
        harness.assert_pool_quiet(before)
        trees[workers] = harness.tree_state(attempt)
    serial = outcomes[1]
    assert serial.error_type is not None and serial.delivered == list(range(k))
    harness.assert_like_serial(outcomes["reference"], serial)          # the pre-P1c loop: same error, prefix, bytes
    assert serial.delivered == outcomes["reference"].delivered
    assert {n: s[2] for n, s in trees[1].items()} == {n: s[2] for n, s in trees["reference"].items()}
    del outcomes["reference"]
    later = {f"q_{n:05d}.fit" for n in range(k + 2, COUNT + 1)}
    for workers, found in outcomes.items():
        harness.assert_like_serial(serial, found)
        assert found.delivered == serial.delivered                # every frame before the failure is delivered
        extra = set(trees[workers]) - set(trees[1])
        assert extra <= later, (workers, sorted(extra))            # prereg §5 item 2: q_ of frames after the failure
        for name in trees[1]:                                      # the serial files, the same bytes
            if kind == "load" or name != f"q_{k + 1:05d}.fit":
                assert trees[workers][name][2] == trees[1][name][2], (workers, name)
        if kind == "data":                                         # the failing frame's q_: the reference's leftover
            assert trees[workers][f"q_{k + 1:05d}.fit"][2] == trees[1][f"q_{k + 1:05d}.fit"][2]


def _replace_map(work, number, shape):
    """Frame ``number``'s support map replaced by one of ``shape``, its descriptor updated in the support record: a map
    load_frame_support accepts (it checks a map against its own descriptor) that the eligible counts do not match."""
    valid = np.random.default_rng(number).random(shape) > 0.2
    descriptor = rs.save_array(Path(work) / f"frame_support_light_{number:05d}.npy", valid)
    record = json.loads((Path(work) / rs.RECORD).read_text(encoding="utf-8"))
    record["frames"][number - 1]["explicit_validity"] = descriptor
    (Path(work) / rs.RECORD).write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("case", ["eligible-1d", "map-wider", "map-one-row"])
def test_a_map_the_serial_sum_cannot_take_fails_as_there(tmp_path, case):
    """The serial loop added each map to its sum before writing the frame (review F-3 lens): an eligible array that is
    not one frame (1-D: numpy's broadcast error at frame 1, no q_ at all; core plans it as unknown: the reference body)
    and a map wider than eligible (numpy's error at frame 2, before q_00002) fail with the serial error and leave the
    serial files (plus, on several workers, q_ of frames after the failure, prereg §5 item 2); a map that broadcasts
    (one row) goes on to the geometry check, which fails after the copy, as before."""
    job = make_work(tmp_path / "job", count=COUNT)
    expected = _expected(job.work)
    failing = 1 if case == "eligible-1d" else 2
    if case == "eligible-1d":
        expected = expected.reshape(-1)
    else:
        _replace_map(job.work, 2, (150, 41) if case == "map-wider" else (1, 40))
    outcomes, trees = {}, {}
    for side in ("reference", "base", "core", 1, 2, 4, 16):
        attempt = tmp_path / f"attempt-{side}"
        attempt.mkdir()
        pools = {"reference": None, "base": None, "core": core._FinitePools(None, "auto")}.get(side) \
            if not isinstance(side, int) else streamed(side)
        before = harness.aester_threads()

        def run(delivered, side=side, pools=pools, attempt=attempt):
            sources = sources_of(job.work)
            try:
                if side == "reference":
                    fs._finite_write_reference(job.work, attempt, sources, REFERENCE_FILE, expected, lambda: None)
                else:
                    fs._write_inputs(job.work, attempt, sources, REFERENCE_FILE, expected, lambda: None, pools)
            finally:
                delivered.extend(i for i, s in enumerate(sources) if "minimum" in s)
        with harness.pool_conditions(harness.Case(side if isinstance(side, int) else 1, None, "reuse", 3)):
            outcomes[side] = harness.outcome(run)
        harness.assert_pool_quiet(before)
        trees[side] = harness.tree_state(attempt)
        if side == "core":
            assert pools.decisions["finite_write"][1] == (fs.REFERENCE if case == "eligible-1d" else fs.STREAMED)
    serial = outcomes["reference"]
    assert serial.error_type is ValueError and serial.delivered == list(range(failing - 1))
    if case == "map-one-row":
        assert serial.message == "有限值堆栈需要明确BOTTOM-UP且与有效性相同画幅的输入"
    else:
        assert "could not be broadcast" in serial.message, serial.message
    serial_files = {n: s[2] for n, s in trees["reference"].items()}
    assert set(serial_files) == {f"q_{n:05d}.fit" for n in range(1, failing + (case == "map-one-row"))}
    later = {f"q_{n:05d}.fit" for n in range(failing + 1, COUNT + 1)}
    for side, found in outcomes.items():
        harness.assert_like_serial(serial, found)
        assert found.delivered == serial.delivered, side
        files = {n: s[2] for n, s in trees[side].items()}
        assert set(files) >= set(serial_files) and set(files) - set(serial_files) <= later, (side, sorted(files))
        assert all(files[n] == serial_files[n] for n in serial_files), side
        if side in ("base", 1) or (side == "core" and case == "eligible-1d"):   # one worker: the serial files only
            assert set(files) == set(serial_files), side


def test_a_failed_run_records_the_serial_error_and_resumes_to_the_clean_result(tmp_path, monkeypatch):
    runtime = fake_runtime(tmp_path / "runtime", monkeypatch)
    clean = make_work(tmp_path / "clean", count=COUNT)
    base = fs.run(runtime, clean.directory, clean.work, lambda: None, MeanEngine(COUNT, 2))
    receipts = {}
    for workers in (1, 4, 16):
        job = make_work(tmp_path / f"w{workers}", count=COUNT)
        _failing_load(monkeypatch, "frame_support_light_00004.npy")
        pools = streamed(workers)
        with pytest.raises(RuntimeError):
            fs.run(runtime, job.directory, job.work, lambda: None, MeanEngine(COUNT, 2), pools=pools)
        receipt = json.loads(next(job.directory.glob("finite-*/receipt.json")).read_text(encoding="utf-8"))
        receipts[workers] = (receipt["state"], receipt["error"])
        monkeypatch.undo()
        runtime = fake_runtime(tmp_path / "runtime", monkeypatch)
        resumed = fs.run(runtime, job.directory, job.work, lambda: None, MeanEngine(COUNT, 2), pools=streamed(workers))
        keep = lambda record: {k: v for k, v in record.items() if k not in ("attempt", "seconds")}
        assert harness.json_bytes(keep(resumed)) == harness.json_bytes(keep(base))
        assert (q_digests(job.directory / resumed["attempt"]) == q_digests(clean.directory / base["attempt"]))
    assert set(receipts.values()) == {("failed", "injected failure loading frame_support_light_00004.npy")}


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [None, 8, 16])
@pytest.mark.parametrize("when", ["submission", "waiting"])
def test_a_cancel_returns_within_the_limit_at_real_size(tmp_path_factory, tmp_path, monkeypatch, workers, when):
    _real_reference(tmp_path_factory)
    if workers is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, str(workers))
    switch = harness.CancelSwitch()
    if when == "submission":
        switch.trip_at_call(2)
    else:
        switch.trip_after(1.5)
    try:
        latency = harness.assert_cancels_in_time(lambda check: _real_pool(tmp_path, _Recorded(), check, keep=True),
                                                 switch)
    finally:
        shutil.rmtree(tmp_path / "attempt", ignore_errors=True)
    print(f"finite_write cancel ({when}, workers={workers}): {latency:.3f} s")


# ================================================================== (f) memory

def test_the_pool_s_memory_peak_stays_within_its_reservation(tmp_path_factory, tmp_path, monkeypatch):
    """Real size, the first unit made to finish last (every other result held): the private-bytes rise over the pool
    stays within granted x unit_bytes, plus the coordinator's own uint16 sum (allocated before the pool)."""
    _real_reference(tmp_path_factory)
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    pools = _Recorded()
    peak = harness.PeakSampler()
    with harness.pool_conditions(harness.Case(None, None, "reuse", 0), head_last=True, head_seconds=1.0):
        _real_pool(tmp_path, pools, window=peak)
    _, stats, _ = pools.notes[-1]
    limit = harness.pool_memory_limit(stats, blas=False)
    pixels = int(_REAL["inputs"][1].size)
    print(f"finite_write peak rise {peak.rise / GiB:.3f} GiB over {stats['workers']} workers "
          f"(unit_bytes {stats['unit_bytes'] / GiB:.3f} GiB, limit {limit / GiB:.3f} GiB, traced {peak.traced})")
    if peak.rise is not None:
        assert peak.rise <= limit + 2 * pixels + (64 << 20), (peak.rise, limit)


def test_the_reference_route_s_memory_peak_stays_within_its_unit_bytes(tmp_path_factory, tmp_path, monkeypatch):
    """R17 for the reference route (AESTER_FINITE_WRITE=reference; also memory too tight for one streamed unit): real
    size, core's plan (one worker, the declared reference unit_bytes); the private-bytes rise over the pool stays within
    unit_bytes plus the coordinator's own uint16 sum (allocated before the pool). The q_ are the reference's."""
    reference, _ = _real_reference(tmp_path_factory)
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    pools = _Recorded("reference")
    peak = harness.PeakSampler()
    with harness.pool_conditions(harness.Case(1, None, "reuse", None)):
        found = _real_pool(tmp_path, pools, window=peak)
    assert found[0] == reference[0] and found[2] == reference[2]
    _, stats, extra = pools.notes[-1]
    assert extra["body"] == fs.REFERENCE and stats["workers"] == 1 and stats["serial"]
    limit = harness.pool_memory_limit(stats, blas=False)
    pixels = int(_REAL["inputs"][1].size)
    print(f"finite_write reference route peak rise {peak.rise / GiB:.3f} GiB (unit_bytes "
          f"{stats['unit_bytes'] / GiB:.3f} GiB, coordinator sum {2 * pixels / GiB:.3f} GiB, traced {peak.traced})")
    if peak.rise is not None:
        assert peak.rise <= limit + 2 * pixels, (peak.rise, limit)


def _memory(monkeypatch, physical, commit=None):
    monkeypatch.setattr(performance, "_memory_status", lambda: (physical, physical if commit is None else commit))


@pytest.mark.parametrize("available, setting, expected", [
    (64 * GiB, None, (4, fs.STREAMED, None)),            # plenty: the cap
    (2.6 * GiB, None, (1, fs.STREAMED, "budget")),       # min(avail/3, avail-1G) holds one 0.68 GB unit only
    (1.8 * GiB, None, (1, fs.REFERENCE, "route")),       # not even one: the reference body, one worker
    (None, None, (1, fs.REFERENCE, "route")),            # memory unknown
    (64 * GiB, "max", (None, fs.STREAMED, None)),        # max: every usable CPU (at most 16) streamed
    (1.8 * GiB, "max", (1, fs.REFERENCE, "route")),      # max never runs the reference body on several threads
    (64 * GiB, "1", (1, fs.STREAMED, "workers")),        # --parallel serial: reason stays 'workers'
    (64 * GiB, "reference", (1, fs.REFERENCE, "route")),  # AESTER_FINITE_WRITE=reference
])
def test_core_plans_the_body_and_workers_from_memory(monkeypatch, available, setting, expected):
    pixels, width = 5520 * 8280, 8280
    file_bytes = 2880 + 12 * pixels
    _memory(monkeypatch, None if available is None else int(available))
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    write_setting = "auto"
    if setting == "reference":
        write_setting = "reference"
    elif setting is not None:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, setting)
    pools = core._FinitePools(None, write_setting)
    plan = pools.write(units=48, file_bytes=file_bytes, pixels=pixels, width=width)
    workers, body, reason = expected
    if workers is None:
        workers = min(frame_pool.MAX_FRAME_WORKERS, frame_pool.usable_cpus())
    elif workers == 4:                                   # the cap, unless this machine has fewer than 8 CPUs
        workers = min(4, max(1, frame_pool.usable_cpus() // 2))
    assert (plan.workers, plan.body) == (workers, body)
    streamed_unit = file_bytes + 2 * pixels + 6 * 3 * 64 * width * 4
    assert plan.unit_bytes == (streamed_unit if body == fs.STREAMED else 3 * pixels + 6 * 3 * 64 * width * 4)
    assert abs(streamed_unit - 0.68e9) < 0.01e9                            # prereg §3.3: about 0.68 GB at 45.7 MP
    assert plan.inflight == (workers if body == fs.STREAMED else None)
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, entry: notes.append((site, entry)))
    serial = {"serial": True, "reason": "workers", "workers": 1} if workers == 1 else {"serial": False, "workers": 4}
    pools.note_pool("finite_write", serial, body=body, fallbacks={})
    site, entry = notes[-1]
    assert site == "finite_write" and entry["body"] == body and entry["fallbacks"] == {}
    assert entry.get("reason") == (reason if workers == 1 else None)
    assert entry["bound"] in ("cap", "cpu", "memory", "units", "memory-unknown", "override")


def test_core_plans_a_frame_of_unknown_size_as_the_reference_body(monkeypatch):
    """finite_stack passes pixels and width as None for a count array that is not one frame (it fails in its own
    place): the reference body on one worker, noted as the route."""
    _memory(monkeypatch, 64 * GiB)
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    pools = core._FinitePools(None, "auto")
    assert pools.write(units=48, file_bytes=548470080, pixels=None, width=None) == fs.Plan()
    assert pools.decisions["finite_write"] == ({}, fs.REFERENCE)
    notes = []
    monkeypatch.setattr(performance, "note_pool", lambda site, entry: notes.append((site, entry)))
    pools.note_pool("finite_write", {"serial": True, "reason": "workers", "workers": 1}, body=fs.REFERENCE,
                    fallbacks={})
    assert notes[-1][1]["reason"] == "route" and notes[-1][1]["bound"] is None


def test_core_s_pools_implement_the_finite_stack_interface():
    import inspect
    for name, method in inspect.getmembers(fs.Pools, inspect.isfunction):
        if not name.startswith("_"):
            mine = getattr(core._FinitePools, name)
            assert list(inspect.signature(mine).parameters) == list(inspect.signature(method).parameters), name
