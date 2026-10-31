"""P1c T-5a: the Milky Way render's 128-row blocks on the frame pool (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5,
§6.2 A5 (a)-(f)).

The kept reference is milkyway.render at the P1c base (tests/p1c_finishing_reference.py); outputs compare as bytes,
the statistics record with its key order (render_seconds, a timing, masked on both sides). Real size: the F48
fixture's Milky Way step, whose published TIFF and statistics the serial code made.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _milkyway_render_reference
from starstack import frame_pool, milkyway as mw, performance, photographic_job as job

# the float32 scene's subnormal deltas (R21) overflow head/foot to inf on purpose (the render clips alpha to 1)
pytestmark = pytest.mark.filterwarnings("ignore:overflow encountered in divide:RuntimeWarning")

DOMAIN = "display-sRGB"
ROWS = 8                                      # 96-row synthetic scenes: 12 blocks
DTYPES = (np.uint16, np.float32)
_REFERENCE = {}


def _prepared(seed, dtype):
    inputs = fd.milkyway_scene(seed, dtype)
    return inputs, mw.prepare(*inputs, mw.Recipe(work_long_edge=256, strength=.8), domain=DOMAIN)


def reference(seed, dtype=np.uint16):
    key = (seed, np.dtype(dtype).name)
    if key not in _REFERENCE:
        inputs, prepared = _prepared(seed, dtype)
        out = np.zeros_like(inputs[0])
        stats = _milkyway_render_reference(*inputs, prepared, out, domain=DOMAIN, row_size=ROWS)
        stats["render_seconds"] = None
        _REFERENCE[key] = (out, stats)
    return _REFERENCE[key]


def pooled(seed, dtype=np.uint16, *, check=None):
    inputs, prepared = _prepared(seed, dtype)
    out = np.zeros_like(inputs[0])
    stats = mw.render(*inputs, prepared, out, domain=DOMAIN, row_size=ROWS, check=check)
    stats["render_seconds"] = None
    return out, stats


def reference_target(seed, dtype):
    return reference(seed, np.dtype(dtype))


def pooled_target(seed, dtype):
    with fd.forced(h.Case(8, None, "reuse", 0)):
        return pooled(seed, np.dtype(dtype))


def test_the_scene_exercises_every_statistic_and_the_subnormal_weights():
    for dtype in DTYPES:
        _, stats = reference(0, dtype)
        assert stats["changed_pixels"] > 100 and stats["max_abs_change"] > 0
        assert not stats["identity"]
    inputs = fd.milkyway_scene(0)
    assert np.count_nonzero((inputs[3] > 0) & (inputs[3] < np.finfo(np.float32).tiny)) >= 400   # subnormal mask


# ================================================================== (a) reference vs pool, (e) audit

@pytest.mark.parametrize("dtype", DTYPES, ids=["u16", "f32"])
@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_reference(case, dtype):
    expected = reference(case.seed, dtype)
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        actual = pooled(case.seed, dtype)
    h.assert_same(expected, actual)
    entries = fd.pool_entries(seen, "milkyway_render")
    assert len(entries) == 1
    assert entries[0]["workers"] == min(case.workers, 12) and entries[0]["units"] == 12
    assert entries[0]["unit_bytes"] == mw.render_unit_bytes(ROWS, 72) and entries[0]["serial"] == (case.workers == 1)
    h.assert_pool_quiet(before)


def test_the_default_decision_runs_in_parallel_and_notes_it(monkeypatch):
    # forced(None) clears the worker setting only inside its block; the expected count below is computed after it
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    with fd.notes() as seen, fd.forced(h.Case(None, None, "reuse", None)):
        actual = pooled(1)
    h.assert_same(reference(1), actual)
    entry = fd.pool_entries(seen, "milkyway_render")[0]
    assert entry["requested"] == performance.frame_workers(12, mw.render_unit_bytes(ROWS, 72), mw.RENDER_WORKER_CAP,
                                                           blas=False)
    assert entry["workers"] >= 2 or frame_pool.usable_cpus() < 4


def test_subnormal_weights_are_identical_under_an_ftz_coordinator():
    """R21 (prereg §3.5): the blend weights can be subnormal; 1 and N workers agree even when the coordinator runs
    with flush-to-zero / denormals-are-zero set (the workers inherit its control word).

    The float32 scene is sensitive to the setting (unlocked pixels just above 4 tiny under subnormal weights), so
    the 1-worker render under flush-to-zero first differs from the one without it: a pool whose workers kept the
    default control word would render those pixels as without it and fail the comparison."""
    if frame_pool.fp_environment() is None:
        pytest.skip("no _controlfp on this platform")
    api = frame_pool._controlfp()
    saved = api(0, 0)
    with fd.forced(h.Case(1, None, "reuse", None)):
        kept = pooled(4, np.float32)
    try:
        api(0x01000000, frame_pool._MCW_DN)                         # _DN_FLUSH: flush-to-zero and denormals-are-zero
        results = {}
        for workers in (1, 8):
            with fd.forced(h.Case(workers, None, "reuse", 3)):
                results[workers] = pooled(4, np.float32)
    finally:
        api(saved, frame_pool._MCW_DN)
    differs = np.any(results[1][0] != kept[0], axis=-1)
    assert differs.sum() >= 20 and not differs[:24].any() and not differs[40:].any()   # rows 31-32: blocks 3 and 4
    assert results[1][1] != kept[1]
    h.assert_same(results[1], results[8])


def test_reruns_under_two_hash_seeds_give_one_digest():
    for dtype in ("uint16", "float32"):
        digest = h.assert_hashseed_stable("test_p1c_site_milkyway_render:reference_target",
                                          "test_p1c_site_milkyway_render:pooled_target", 2, dtype)
        assert digest == h.canonical_digest(reference(2, np.dtype(dtype)))


@pytest.mark.parametrize("env", h.NUMERIC_ENVS, ids=lambda e: "-".join(f"{k}={v}" for k, v in e.items()))
def test_numeric_environments_give_the_reference(env):
    """(g): element-wise IEEE arithmetic only (no BLAS, R8): equal under the numeric variants."""
    assert h.subprocess_digest("test_p1c_site_milkyway_render:pooled_target", 5, "uint16", env=env) == \
        h.canonical_digest(reference(5, np.uint16))


# ================================================================== (a)/(b) real size on the F48 fixture

@pytest.fixture(scope="module")
def f48_render():
    f48 = fd.F48()
    inputs = f48.milkyway_inputs()
    recipe = mw.Recipe(**f48.record(f48.milkyway)["recipe"])
    prepared = mw.prepare(*inputs, recipe, domain=DOMAIN)
    expected = job._read_display(f48.milkyway / "milkyway_srgb16.tif")
    try:
        yield f48, inputs, prepared, np.asarray(expected)
    finally:
        job._close_map(expected)


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_matches_the_published_f48_step(f48_render, case):
    f48, inputs, prepared, expected = f48_render
    out = np.empty_like(inputs[0])
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        stats = mw.render(*inputs, prepared, out, domain=DOMAIN)
    h.assert_same(expected, out)
    published = dict(f48.record(f48.milkyway)["statistics"])
    published["render_seconds"] = stats["render_seconds"] = None
    h.assert_same(published, stats)
    entry = fd.pool_entries(seen, "milkyway_render")[0]
    assert entry["units"] == 44 and entry["workers"] == (case.workers or entry["requested"])
    h.assert_pool_quiet(before)


# ================================================================== (c) failures

def _failing(k, rows=ROWS):
    original = mw.render_block

    def block(*args, **kwargs):
        if args[7] == k * rows:
            raise ValueError(f"injected render failure in block {k}")
        return original(*args, **kwargs)
    return block


@pytest.mark.parametrize("k", h.failure_positions(12))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_block_ends_like_the_serial_loop(monkeypatch, k, workers):
    monkeypatch.setattr(mw, "render_block", _failing(k))
    before = h.aester_threads()

    def run(case):
        def body(delivered):
            with fd.forced(case, delivered=delivered):
                return pooled(9)
        return h.outcome(body)

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [4, 16])
def test_a_failed_finish_leaves_the_serial_files(tmp_path, monkeypatch, workers):
    """finish_milkyway's failed attempt: the same record and the same staging folder as the serial loop. The parallel
    pool may have written rows past the failure into its unpublished map (§5 item 2); the staged pixels are dropped
    once the maps are closed (§3.5), so either folder holds its failed result.json only."""
    files = fd.milkyway_files(tmp_path / "in")
    monkeypatch.setattr(mw, "render_block", _failing(2, mw.Recipe().rows))      # 640 rows: 5 blocks of 128
    states = {}
    for label, count in (("serial", 1), ("parallel", workers)):
        with fd.forced(h.Case(count, None, "reuse", 1)):
            record = job.finish_milkyway(files["source"], tmp_path / label, files["sky"], files["protected"],
                                         files["milkyway"], milkyway_evidence=files["evidence"])
        assert record["state"] == "failed" and record["error"] == "injected render failure in block 2"
        stage = next((tmp_path / label).iterdir())
        assert stage.name.startswith(".partial-milkyway-") and Path(record["processing_directory"]) == stage
        assert sorted(p.name for p in stage.rglob("*")) == ["result.json"]
        recorded = json.loads((stage / "result.json").read_text(encoding="utf-8"))
        assert recorded["state"] == "failed" and recorded["error"] == record["error"]
        states[label] = {k: v for k, v in recorded.items() if k not in ("seconds", "processing_directory")}
    assert states["serial"] == states["parallel"]


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [None, 8, 16])
def test_a_cancel_returns_within_a_second_at_real_size(f48_render, workers):
    _, inputs, prepared, _ = f48_render
    out = np.empty_like(inputs[0])
    switch = h.CancelSwitch().trip_after(.6)
    with fd.forced(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(
            lambda check: mw.render(*inputs, prepared, out, domain=DOMAIN, check=check), switch)
    print(f"workers {workers}: cancel latency {latency:.3f} s")


def test_the_serial_path_checks_once_per_block_and_never_inside_one():
    calls = []
    with fd.forced(h.Case(1, None, "reuse", None)):
        pooled(0, check=lambda: calls.append(time.perf_counter()))
    assert len(calls) == 12


def test_a_cancelled_finish_leaves_no_staging_folder(tmp_path):
    class WorkflowCancelled(KeyboardInterrupt):
        pass

    files = fd.milkyway_files(tmp_path / "in")
    for workers in (1, 8):
        switch = h.CancelSwitch(WorkflowCancelled, "stop").trip_at_call(2)
        root = tmp_path / f"out-{workers}"
        with fd.forced(h.Case(workers, None, "reuse", None)), pytest.raises(WorkflowCancelled):
            job.finish_milkyway(files["source"], root, files["sky"], files["protected"], files["milkyway"],
                                milkyway_evidence=files["evidence"], check=switch)
        assert root.is_dir() and not list(root.iterdir()), list(root.iterdir())
        h.assert_pool_quiet()


# ================================================================== (f) memory

def test_the_pool_peak_stays_within_its_declared_bound(f48_render):
    _, inputs, prepared, expected = f48_render
    out = np.empty_like(inputs[0])
    out[:] = 0                                                       # committed before the measured block
    with fd.notes() as seen, fd.forced(h.Case(8, 16, "reuse", 0), head_last=True, head_seconds=.5), \
            h.PeakSampler(trace=False) as peak:
        mw.render(*inputs, prepared, out, domain=DOMAIN)
    entry = fd.pool_entries(seen, "milkyway_render")[0]
    limit = h.pool_memory_limit(entry, blas=False)
    assert entry["workers"] == 8 and entry["unit_bytes"] == mw.RENDER_UNIT_BYTES
    if peak.rise is not None:
        assert peak.rise <= limit, (peak.rise, limit)
    print(f"milkyway_render: rise {peak.rise} limit {limit}")
    h.assert_same(expected, out)


def test_unit_bytes_scale_with_rows_and_width_and_never_drop_below_the_measurement():
    assert mw.render_unit_bytes(128, 8280) == mw.RENDER_UNIT_BYTES
    assert mw.render_unit_bytes(256, 8280) == 2 * mw.RENDER_UNIT_BYTES
    assert mw.render_unit_bytes(128, 11648) > mw.RENDER_UNIT_BYTES
    assert mw.render_unit_bytes(8, 64) == mw.RENDER_UNIT_BYTES
