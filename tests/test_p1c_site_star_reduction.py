"""P1c T-5b: the star reduction's 16-row blocks (each read with its 2 r halo) on the frame pool
(docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5 (a)-(f)).

The kept reference is the reduction loop at the P1c base (tests/p1c_finishing_reference.py); the three written layers
compare as bytes and the ten statistics in order. Real size: the F48 fixture's reduction step, whose published TIFFs,
JPEG and record the serial code made.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _star_reduction_reference
from starstack import frame_pool, performance, photographic_job as job, star_reduction as sr

SHAPE = (200, 40)                     # 13 reduction blocks
UNITS = 13
_REFERENCE = {}
MODES = {"legacy": dict(extended=False, paired=False, radius=3),
         "legacy-paired": dict(extended=False, paired=True, radius=3),
         "extended": dict(extended=True, paired=False, radius=3),
         "extended-paired": dict(extended=True, paired=True, radius=3),
         "radius0": dict(extended=True, paired=True, radius=0),
         "radius32": dict(extended=False, paired=False, radius=32)}


def _inputs(seed, mode):
    spec = MODES[mode]
    stack = fd.layer_stack(seed, SHAPE, paired=spec["paired"])
    weight, local = fd.reduction_controls(seed, SHAPE, extended=spec["extended"])
    shape = stack["shape"]
    out, rout = np.zeros(shape, np.uint16), np.zeros(shape, np.int32)
    bout = np.zeros(shape, np.uint16) if spec["paired"] else None
    args = (stack["s"], stack["b"], stack["d"], stack["r"], stack["t"], out, bout, rout, weight, local)
    kwargs = dict(shape=shape, radius=spec["radius"], extended=spec["extended"], strength=.7, low=.1, high=.45,
                  paired=spec["paired"])
    return args, kwargs, (out, bout, rout)


def reference(seed, mode="extended-paired"):
    if (seed, mode) not in _REFERENCE:
        args, kwargs, outputs = _inputs(seed, mode)
        stats = _star_reduction_reference(*args, **kwargs)
        _REFERENCE[(seed, mode)] = (outputs, stats)
    return _REFERENCE[(seed, mode)]


def pooled(seed, mode="extended-paired", *, check=None, stack_hook=None):
    args, kwargs, outputs = _inputs(seed, mode)
    if stack_hook is not None:
        stack_hook(args)
    stats = sr._reduce_rows(*args, **kwargs, check=check)
    return outputs, stats


def reference_target(seed, mode):
    return reference(seed, mode)


def pooled_target(seed, mode):
    with fd.forced(h.Case(8, None, "reuse", 0)):
        return pooled(seed, mode)


def test_the_synthetic_layers_exercise_every_statistic():
    _, stats = reference(0, "extended-paired")
    assert all(value > 0 for value in stats[:4]) and stats[7] >= 0
    _, legacy = reference(0, "legacy")
    assert legacy[0] > 0 and legacy[1] > 0
    _, extended = reference(0, "extended")
    assert extended[4] > 0                                     # full-removal endpoints requested


# ================================================================== (a) reference vs pool, (e) audit

@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_reference(case):
    expected = reference(case.seed)
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        actual = pooled(case.seed)
    h.assert_same(expected, actual)
    entries = fd.pool_entries(seen, "star_reduction")
    assert len(entries) == 1 and entries[0]["units"] == UNITS
    assert entries[0]["workers"] == min(case.workers, UNITS)
    assert entries[0]["unit_bytes"] == sr.reduction_unit_bytes(SHAPE[1], 3)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_every_mode_matches_the_reference(mode, workers):
    with fd.forced(h.Case(workers, None, "fresh", workers)):
        actual = pooled(1, mode)
    h.assert_same(reference(1, mode), actual)


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_site_star_reduction:reference_target",
                                      "test_p1c_site_star_reduction:pooled_target", 2, "extended-paired")
    assert digest == h.canonical_digest(reference(2, "extended-paired"))


def test_a_numeric_environment_gives_the_reference():
    """(g) does not apply (no work leaves a scan child; no BLAS, R8); one variant as a smoke check."""
    assert h.subprocess_digest("test_p1c_site_star_reduction:pooled_target", 3, "legacy-paired",
                               env={"NPY_DISABLE_CPU_FEATURES": "X86_V3"}) == h.canonical_digest(reference(3, "legacy-paired"))


# ================================================================== (a)/(b) real size on the F48 fixture

def _f48_reduce(f48, out, **kwargs):
    published = f48.record(f48.reduction)
    recipe = published["recipe"]
    return sr.reduce_stars(f48.separation, out, background_strength=recipe["background_strength"],
                           milkyway_offset=recipe["milkyway_offset"], milkyway_mask=f48.region,
                           milkyway_evidence=f48.evidence(f48.reduction), **kwargs)


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_publishes_the_f48_reduction(tmp_path, case):
    f48 = fd.F48()
    published = f48.record(f48.reduction)
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        record = _f48_reduce(f48, tmp_path / "reduction")
    folder = Path(record["processing_directory"])
    for name in ("star_reduced_srgb16.tif", "stars_reduced_signed_codes.tif", "star_reduced.jpg"):
        assert (folder / name).read_bytes() == (f48.reduction / name).read_bytes(), name
    fd.assert_record_like({k: v for k, v in published.items() if k != "artifacts"},
                          {k: v for k, v in record.items() if k != "artifacts"})
    assert list(record["artifacts"]) == list(published["artifacts"])
    for name, item in published["artifacts"].items():
        assert {k: item[k] for k in ("sha256", "bytes", "kind")} == \
            {k: record["artifacts"][name][k] for k in ("sha256", "bytes", "kind")}
    entry = fd.pool_entries(seen, "star_reduction")[0]
    assert entry["units"] == 345 and entry["unit_bytes"] == sr.reduction_unit_bytes(8280, 8)
    assert entry["workers"] == (case.workers or entry["requested"])
    h.assert_pool_quiet(before)


# ================================================================== (c) failures

def _corrupt(k):
    def hook(args):
        r = args[3]
        r[k * 16 + 3, 5, 1] += 1                       # R != B - D in block k
    return hook


@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_block_ends_like_the_serial_loop(k, workers):
    before = h.aester_threads()

    def run(case):
        def body(delivered):
            with fd.forced(case, delivered=delivered):
                return pooled(4, stack_hook=_corrupt(k))
        return h.outcome(body)

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.message == "star residual differs from base minus diffuse"
    assert serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [4, 16])
def test_a_failed_reduction_leaves_the_serial_files(tmp_path, monkeypatch, workers):
    stack = fd.layer_stack(5, (600, 40), paired=True)
    result = fd.make_separation(tmp_path / "sep", stack)
    separation = json.loads(result.read_text(encoding="utf-8"))
    base = tifffile.imread(separation["processing_base"])
    diffuse = tifffile.imread(separation["artifacts"]["diffuse_srgb16.tif"]["path"])
    # the unit whose halo rows hold block 20's peak raises (identified by content: any thread, any order)
    monkeypatch.setattr(sr, "bright_control", _block_failure(sr.bright_control, base, diffuse, 20, radius=3))
    states = {}
    for label, count in (("serial", 1), ("parallel", workers)):
        with fd.forced(h.Case(count, None, "reuse", 2)), pytest.raises(ValueError) as error:
            sr.reduce_stars(result, tmp_path / label, bright_radius=3, background_strength=.6)
        assert str(error.value) == "injected reduction failure in block 20"
        stage = next((tmp_path / label).iterdir())
        assert stage.name.startswith(".partial-star-reduction-")
        recorded = json.loads((stage / "result.json").read_text(encoding="utf-8"))
        assert recorded["state"] == "failed" and recorded["error"] == "injected reduction failure in block 20"
        # §3.5: the parallel pool may have written rows past block 20 into the three staged maps; the staged pixels
        # are dropped once the maps are closed, so the failed folder is the same at any worker count
        assert sorted(p.name for p in stage.rglob("*")) == ["result.json"]
        states[label] = {k: v for k, v in recorded.items() if k not in ("seconds", "processing_directory")}
    assert states["serial"] == states["parallel"]


def _block_failure(original, b, d, block, *, radius):
    """bright_control that raises for the block whose peak rows are those of ``block`` (identified by content)."""
    lo, hi = max(0, block * 16 - 2 * radius), min(b.shape[0], block * 16 + 16 + 2 * radius)
    expected = np.maximum(b[lo:hi].astype(np.int32) - d[lo:hi].astype(np.int32), 0).max(axis=2).astype(np.uint16)

    def control(peak, r):
        if peak.shape == expected.shape and np.array_equal(peak, expected):
            raise ValueError(f"injected reduction failure in block {block}")
        return original(peak, r)
    return control


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [None, 8, 16])
def test_a_cancel_returns_within_a_second_at_real_size_and_leaves_no_staging_folder(tmp_path, monkeypatch, workers):
    """The pool returns within 1.0 s of check() reporting the cancel (R18); reduce_stars then closes its maps and
    removes the staging folder (1.1 GB of mapped TIFFs) before re-raising: timed separately, within the workflow's
    3 s cancel budget (prereg §6.4)."""
    class WorkflowCancelled(KeyboardInterrupt):
        pass

    f48 = fd.F48()
    switch = h.CancelSwitch(WorkflowCancelled, "stop")
    root = tmp_path / "reduction"
    original, ended = sr._reduce_rows, {}
    discard = job._discard_partial

    def trip_when_rows_start(*args, **kwargs):     # trip 0.5 s into the pool (inputs are verified before it)
        switch.trip_after(.5)
        try:
            return original(*args, **kwargs)
        finally:
            ended["pool"] = time.perf_counter()

    def timed_discard(stage, **kwargs):
        started = time.perf_counter()
        try:
            return discard(stage, **kwargs)
        finally:
            ended["discard_seconds"] = time.perf_counter() - started

    monkeypatch.setattr(sr, "_reduce_rows", trip_when_rows_start)
    monkeypatch.setattr(job, "_discard_partial", timed_discard)
    with fd.forced(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(lambda check: _f48_reduce(f48, root, check=check), switch, limit=3.0)
    pool_latency = ended["pool"] - switch.tripped_at
    assert pool_latency <= h.CANCEL_LIMIT_SECONDS, pool_latency
    assert root.is_dir() and not list(root.iterdir()), list(root.iterdir())
    print(f"workers {workers}: pool {pool_latency:.3f} s, staging removal {ended['discard_seconds']:.3f} s, "
          f"cancel to return {latency:.3f} s")


def test_the_serial_path_checks_once_per_block():
    calls = []
    with fd.forced(h.Case(1, None, "reuse", None)):
        pooled(0, check=lambda: calls.append(1))
    assert len(calls) == UNITS


# ================================================================== (f) memory

def test_the_pool_peak_stays_within_its_declared_bound():
    f48 = fd.F48()
    readers = [job._read_display(f48.milkyway / "milkyway_srgb16.tif"),
               job._read_display(f48.separation.parent / "diffuse_srgb16.tif"),
               job._read_display(f48.separation.parent / "stars_signed_codes.tif")]
    try:
        s, d, r = readers
        shape = s.shape
        out, rout = np.zeros(shape, np.uint16), np.zeros(shape, np.int32)
        weight = np.ones(shape[:2], np.float32)
        local = np.full(shape[:2], .85, np.float64)
        with fd.notes() as seen, fd.forced(h.Case(8, 16, "reuse", 0), head_last=True, head_seconds=.5), \
                h.PeakSampler(trace=False) as peak:
            sr._reduce_rows(s, s, d, r, None, out, None, rout, weight, local, shape=shape, radius=8, extended=True,
                            strength=.55, low=.1, high=.45, paired=False)
        entry = fd.pool_entries(seen, "star_reduction")[0]
        limit = h.pool_memory_limit(entry, blas=False)
        assert entry["workers"] == 8 and entry["unit_bytes"] == sr.REDUCTION_UNIT_BYTES
        if peak.rise is not None:
            assert peak.rise <= limit, (peak.rise, limit)
        print(f"star_reduction: rise {peak.rise} limit {limit}")
    finally:
        for array in readers:
            job._close_map(array)


def test_a_wide_halo_lowers_the_worker_count(monkeypatch):
    """(f) decision: r=32 reads 144 rows per block (3 x r=8), so the same memory admits fewer workers."""
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 32)
    monkeypatch.setattr(performance, "_memory_status", lambda: (2 << 30, 2 << 30))
    narrow = performance.frame_workers(345, sr.reduction_unit_bytes(8280, 8), sr.REDUCTION_WORKER_CAP, blas=False)
    wide = performance.frame_workers(345, sr.reduction_unit_bytes(8280, 32), sr.REDUCTION_WORKER_CAP, blas=False)
    assert sr.reduction_unit_bytes(8280, 32) == 3 * sr.reduction_unit_bytes(8280, 8)
    assert (narrow, wide) == (8, 5)
