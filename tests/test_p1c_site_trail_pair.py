"""P1c T-5c: add_signed_trails' 128-row blocks on the frame pool (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5,
§6.2 A5 (a)-(f)), and the add-back's cancel point (only the worker's own automatic add-back takes one).

The kept reference is add_signed_trails at the P1c base (tests/p1c_finishing_reference.py): the two outputs compare
as bytes, the statistics record with its key order. Real size: the F48 fixture's trails photograph, base photograph
and reduced final (the reviewed fixed-colour render itself is not kept, so the pair is compared with the reference).
"""
import json
from pathlib import Path

import numpy as np
import pytest

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _trail_pair_reference
from starstack import frame_pool, photographic_job as job, workflow as wf, workflow_edit as edit
from test_workflow import setup  # noqa: F401  (the workflow fixture of the add-back tests)

SHAPE = (200, 40)
ROWS = 16                                  # 13 blocks
UNITS = 13
_REFERENCE = {}


def triple(seed, shape=SHAPE):
    """S (with trails), B and the finished F of B, uint16 RGB, with endpoint pixels that hold the headroom."""
    rng = np.random.default_rng(seed)
    b = rng.integers(1500, 62000, shape + (3,)).astype(np.int64)
    t = np.zeros_like(b)
    rows = rng.random(shape[0]) < .3
    t[rows, 5:12] = rng.integers(-900, 9000, (int(rows.sum()), 7, 3))
    s = np.clip(b + t, 0, 65535)
    finished = np.clip(b + rng.integers(-2500, 4000, b.shape), 0, 65535)
    s[3, 6] = (65535, 100, 100)                # held endpoints
    b[7, 8] = (0, 300, 300)
    return tuple(a.astype(np.uint16) for a in (s, b, finished))


def reference(seed):
    if seed not in _REFERENCE:
        s, b, f = triple(seed)
        out, paired = np.zeros_like(s), np.zeros_like(s)
        stats = _trail_pair_reference(s, b, f, out, paired, rows=ROWS)
        _REFERENCE[seed] = (out, paired, stats)
    return _REFERENCE[seed]


def pooled(seed, *, check=None, arrays=None):
    s, b, f = arrays or triple(seed)
    out, paired = np.zeros_like(s), np.zeros_like(s)
    stats = edit.add_signed_trails(s, b, f, out, paired, rows=ROWS, check=check)
    return out, paired, stats


def reference_target(seed):
    return reference(seed)


def pooled_target(seed):
    with fd.forced(h.Case(8, None, "reuse", 0)):
        return pooled(seed)


def test_the_synthetic_pair_exercises_both_counts():
    *_, stats = reference(0)
    assert stats["residual_pixels"] > 100 and stats["headroom_limited_pixels"] > 0


# ================================================================== (a) reference vs pool, (e) audit

@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_reference(case):
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        actual = pooled(case.seed)
    h.assert_same(reference(case.seed), actual)
    entry = fd.pool_entries(seen, "trail_pair")
    assert len(entry) == 1 and entry[0]["units"] == UNITS and entry[0]["workers"] == min(case.workers, UNITS)
    assert entry[0]["unit_bytes"] == edit.pair_unit_bytes(ROWS, SHAPE[1])
    h.assert_pool_quiet(before)


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_site_trail_pair:reference_target", "test_p1c_site_trail_pair:pooled_target", 2)
    assert digest == h.canonical_digest(reference(2))


# ================================================================== (a)/(b) real size on the F48 fixture

@pytest.fixture(scope="module")
def f48_pair():
    f48 = fd.F48()
    arrays = [job._read_display(p) for p in (f48.trails_photo, f48.base_photo, f48.reduction / "star_reduced_srgb16.tif")]
    out, paired = np.zeros_like(arrays[0]), np.zeros_like(arrays[0])
    stats = _trail_pair_reference(*arrays, out, paired)
    try:
        yield arrays, (out, paired, stats)
    finally:
        for array in arrays:
            job._close_map(array)


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_matches_the_reference(f48_pair, case):
    arrays, expected = f48_pair
    out, paired = np.zeros_like(arrays[0]), np.zeros_like(arrays[0])
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        stats = edit.add_signed_trails(*arrays, out, paired)
    h.assert_same(expected, (out, paired, stats))
    assert stats["residual_pixels"] > 0
    entry = fd.pool_entries(seen, "trail_pair")[0]
    assert entry["units"] == 44 and entry["workers"] == (case.workers or entry["requested"])
    h.assert_pool_quiet(before)


# ================================================================== (c) failures

class FailingRows(np.ndarray):
    """An input whose row slice starting at ``fail_at`` cannot be read (a data-identified failing unit)."""
    fail_at = None

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start == self.fail_at:
            raise ValueError(f"injected pair failure at row {key.start}")
        return super().__getitem__(key)


@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_block_ends_like_the_serial_loop(k, workers):
    before = h.aester_threads()

    def run(case):
        s, b, f = triple(6)
        source = s.view(FailingRows)
        source.fail_at = k * ROWS

        def body(delivered):
            with fd.forced(case, delivered=delivered):
                return pooled(6, arrays=(source, b, f))
        return h.outcome(body)

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [None, 8, 16])
def test_a_cancel_returns_within_a_second_at_real_size(f48_pair, workers):
    arrays, _ = f48_pair
    out, paired = np.zeros_like(arrays[0]), np.zeros_like(arrays[0])
    switch = h.CancelSwitch().trip_after(.4)
    with fd.forced(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(lambda check: edit.add_signed_trails(*arrays, out, paired, check=check),
                                           switch)
    print(f"workers {workers}: cancel latency {latency:.3f} s")


def test_without_a_check_there_is_no_cancel_point():
    with fd.forced(h.Case(4, None, "reuse", None)):
        h.assert_same(reference(1), pooled(1, check=None))


# ================================================================== (f) memory

def test_the_pool_peak_stays_within_its_declared_bound(f48_pair):
    arrays, expected = f48_pair
    out, paired = np.zeros_like(arrays[0]), np.zeros_like(arrays[0])
    with fd.notes() as seen, fd.forced(h.Case(8, 16, "reuse", 0), head_last=True, head_seconds=.5), \
            h.PeakSampler(trace=False) as peak:
        stats = edit.add_signed_trails(*arrays, out, paired)
    h.assert_same(expected, (out, paired, stats))
    entry = fd.pool_entries(seen, "trail_pair")[0]
    limit = h.pool_memory_limit(entry, blas=False)
    assert entry["workers"] == 8 and entry["unit_bytes"] == edit.PAIR_UNIT_BYTES
    if peak.rise is not None:
        assert peak.rise <= limit, (peak.rise, limit)
    print(f"trail_pair: rise {peak.rise} limit {limit}")


# ================================================================== the workflow's automatic add-back

@pytest.fixture
def addback(setup, monkeypatch):
    from test_trail_default_and_extras import build
    return setup, monkeypatch, build


@pytest.mark.usefixtures("pre_contract_jobs")
def test_a_stop_inside_the_automatic_pair_leaves_the_base_and_no_staging_folder(addback):
    """The worker's own automatic add-back is stopped inside the pair (its first block's check): the staging folder
    is removed after the maps close, the workflow keeps its base photograph and the request is cleared."""
    setup, monkeypatch, build = addback
    seen = {}
    original = edit.add_signed_trails

    def stop_then_pair(*args, check=None, **kwargs):
        seen["check"] = check
        wf.cancel_workflow(record["workflow_dir"])
        return original(*args, check=check, **kwargs)

    monkeypatch.setattr(edit, "add_signed_trails", stop_then_pair)
    record, _ = build(setup, monkeypatch)
    done = wf.run_workflow(record["workflow_dir"])
    directory = Path(record["workflow_dir"])
    assert seen["check"] is not None
    assert done["state"] == "succeeded" and done["delivery_dir"] == done["base_delivery_dir"], done.get("error")
    assert "已停止自动加回" in done["message"]
    assert not list(directory.glob(".partial-*")) and not list(directory.glob("delivery-trails-*"))
    assert not (directory / "cancel.request").exists()
    h.assert_pool_quiet()


@pytest.mark.usefixtures("pre_contract_jobs")
@pytest.mark.parametrize("workers", [1, 4])
def test_a_failed_automatic_pair_keeps_only_its_records(addback, workers):
    """A pair stopped by its first block's error (§3.5): the parallel pool may have written the second block into the
    staged maps, so the failed attempt's staged pixels are dropped and its records kept (failure.json): the same
    folder at 1 and N workers. The workflow keeps its base photograph (any add-back failure does)."""
    setup, monkeypatch, build = addback
    monkeypatch.setenv(frame_pool.ENV_WORKERS, str(workers))
    original = edit.add_signed_trails

    def failing(source, *args, **kwargs):
        rows = np.asarray(source).view(FailingRows)
        rows.fail_at = 0
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(edit, "add_signed_trails", failing)
    record, _ = build(setup, monkeypatch)
    done = wf.run_workflow(record["workflow_dir"])
    directory = Path(record["workflow_dir"])
    assert done["state"] == "succeeded" and done["delivery_dir"] == done["base_delivery_dir"], done.get("error")
    stages = list(directory.glob(".partial-reviewed-*"))
    assert len(stages) == 1 and not list(directory.glob("delivery-trails-*"))
    failure = json.loads((stages[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure == {"state": "failed", "error": "injected pair failure at row 0"}
    left = [p for p in stages[0].rglob("*") if p.is_file()]
    assert left and all(p.suffix.lower() == ".json" for p in left), left
    h.assert_pool_quiet()


@pytest.mark.usefixtures("pre_contract_jobs")
def test_a_manual_apply_takes_no_cancel_point(addback):
    """A GUI / MCP apply passes no check: a stale cancel.request never fails it (prereg §3.5 T-5c)."""
    setup, monkeypatch, build = addback
    seen = []
    original = edit.add_signed_trails

    def observe(*args, check=None, **kwargs):
        seen.append(check)
        return original(*args, check=check, **kwargs)

    monkeypatch.setattr(edit, "add_signed_trails", observe)
    record, state = build(setup, monkeypatch)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded" and seen and seen[0] is not None       # the automatic add-back: a check
    (Path(record["workflow_dir"]) / "cancel.request").write_text("stale")
    result = edit.apply_review(record["workflow_dir"], state["session"])
    assert seen[-1] is None and len(seen) == 2
    assert result["published"] and result["workflow"]["state"] == "succeeded"
