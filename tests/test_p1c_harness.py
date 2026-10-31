"""Self-test of tests/p1c_site_harness.py (efficiency P1c §6.2, A5) on a toy pool site.

The toy site has what the real ones have: a kept serial reference, a pooled form on frame_pool.run_ordered with a
per-worker scratch state (R12), a coordinator-side prepare and an in-order integer fold (R2/R3), unit probes (R18),
declared unit_bytes (R10/R17) and pool_stats (R19). Its functions are module level so that the PYTHONHASHSEED reruns
can import them in a fresh interpreter.
"""
import json
import os
import threading
import time

import numpy as np
import pytest

import p1c_site_harness as h
from starstack import frame_pool

ROWS, WIDTH, BLOCK = 96, 512, 8
UNIT_BYTES = 4 * BLOCK * WIDTH * 4 + 64 * 1024          # scratch + result block, rounded up


def toy_data(seed):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((ROWS, WIDTH)).astype(np.float32)
    data[3, 5] = -0.0
    data[7, 9] = np.float32(np.finfo(np.float32).tiny) / 4          # a subnormal input
    return data


def toy_reference(seed):
    """The serial loop: per 8-row block, a scratch buffer written before it is read, rows out, an integer fold."""
    data = toy_data(seed)
    out = np.empty_like(data)
    scratch = np.empty((BLOCK, WIDTH), dtype=np.float32)
    record = {"blocks": 0, "positive": 0, "first": None}
    for start in range(0, ROWS, BLOCK):
        block = data[start:start + BLOCK]
        np.multiply(block, np.float32(1.5), out=scratch[:len(block)])
        out[start:start + len(block)] = np.sqrt(np.abs(scratch[:len(block)])) - block
        record["blocks"] += 1
        record["positive"] += int(np.count_nonzero(out[start:start + len(block)] > 0))
        if record["first"] is None:
            record["first"] = float(out[start, 0])
    return out, record


def toy_pooled(seed, workers, *, inflight=None, check=None, pool_stats=None, fail_at=(), slow=0.0, delivered=None):
    data = toy_data(seed)
    out = np.empty_like(data)
    record = {"blocks": 0, "positive": 0, "first": None}
    failing = set(fail_at)
    delivered = [] if delivered is None else delivered

    def prepare(i, start, own_check):
        own_check()
        return start, data[start:start + BLOCK]

    def unit(payload, u):
        start, block = payload
        if u.index in failing:
            raise ValueError(f"block {u.index} failed")
        end = time.perf_counter() + slow
        while time.perf_counter() < end:
            u.check()
            time.sleep(0.002)
        scratch = u.state
        np.multiply(block, np.float32(1.5), out=scratch[:len(block)])
        rows = np.sqrt(np.abs(scratch[:len(block)])) - block
        out[start:start + len(block)] = rows
        return int(np.count_nonzero(rows > 0)), float(rows[0, 0])

    def fold(i, value):
        delivered.append(i)
        positive, first = value
        record["blocks"] += 1
        record["positive"] += positive
        if record["first"] is None:
            record["first"] = first

    frame_pool.run_ordered(unit, range(0, ROWS, BLOCK), workers=workers, name="toy", check=check, prepare=prepare,
                           on_result=fold, init=lambda: np.empty((BLOCK, WIDTH), dtype=np.float32),
                           inflight=inflight, stats=pool_stats, unit_bytes=UNIT_BYTES, blas=False)
    return out, record


def toy_target(side, seed, workers):
    """For the hash-seed reruns: the reference or the pooled toy, as plain values."""
    return toy_reference(seed) if side == "reference" else toy_pooled(seed, workers)


def toy_reference_target(seed):
    return toy_reference(seed)


def toy_pooled_target(seed):
    return toy_pooled(seed, 8)


UNITS = len(range(0, ROWS, BLOCK))


# ================================================================== (a)

def test_case_matrices_follow_the_prereg():
    synthetic = h.synthetic_cases()
    assert len(synthetic) == 6 * 2 * 2 * 5                       # workers x in flight {w, 2w} x policy x seed
    assert {c.workers for c in synthetic} == set(h.SYNTHETIC_WORKERS)
    assert {(c.workers, c.inflight) for c in synthetic if c.workers == 4} == {(4, 4), (4, 8)}
    assert {c.policy for c in synthetic} == {"reuse", "fresh"} and {c.seed for c in synthetic} == set(range(5))
    assert len({c.id for c in synthetic}) == len(synthetic)
    real = h.real_cases()
    assert [c.workers for c in real] == [1, None, 16] and {c.policy for c in real} == {"reuse"}


@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_reference_over_the_matrix(case):
    expected = toy_reference(7)
    before = h.aester_threads()
    stats = {}
    with h.pool_conditions(case):
        actual = toy_pooled(7, case.workers, inflight=case.inflight, pool_stats=stats)
    h.assert_same(expected, actual)
    assert stats["workers"] == min(case.workers, UNITS) and stats["unit_bytes"] == UNIT_BYTES
    h.assert_pool_quiet(before)
    assert frame_pool._STATE_POLICY == "reuse" and frame_pool._AUDIT_CONTEXT is False   # seams restored


def test_assert_same_is_bitwise_and_keeps_key_order(tmp_path):
    a = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(AssertionError, match="bytes differ"):
        h.assert_same(a, np.array([-0.0, 1.0], dtype=np.float32))
    with pytest.raises(AssertionError, match="dtype"):
        h.assert_same(a, a.astype(np.float64))
    with pytest.raises(AssertionError, match="key order"):
        h.assert_same({"a": 1, "b": 2}, {"b": 2, "a": 1})
    with pytest.raises(AssertionError, match="bits"):
        h.assert_same(0.0, -0.0)
    nan_a = np.frombuffer(np.uint32(0x7FC00000).tobytes(), dtype=np.float32)
    nan_b = np.frombuffer(np.uint32(0x7FC00001).tobytes(), dtype=np.float32)
    with pytest.raises(AssertionError):
        h.assert_same(nan_a, nan_b)                                  # NaN payloads count
    with pytest.raises(AssertionError):
        h.assert_same(1, True)
    (tmp_path / "x").write_bytes(b"abc")
    (tmp_path / "y").write_bytes(b"abd")
    with pytest.raises(AssertionError):
        h.assert_same(tmp_path / "x", tmp_path / "y")
    h.assert_same({"k": [np.float32(1), (1, "s"), None, b"z"]}, {"k": [np.float32(1), (1, "s"), None, b"z"]})
    assert h.json_bytes({"b": 1, "a": 2}) != h.json_bytes({"a": 2, "b": 1})


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_harness:toy_reference_target", "test_p1c_harness:toy_pooled_target", 3)
    assert digest == h.canonical_digest(toy_reference(3))
    seeds = h.hashseed_digests("test_p1c_harness:toy_target", "pool", 3, 16, seeds=("1", "99"))
    assert set(seeds) == {"1", "99"} and set(seeds.values()) == {digest}


def test_a_numeric_variant_reruns_in_its_own_environment():
    value = h.subprocess_digest("test_p1c_harness:toy_target", "pool", 3, 4, env={"OPENBLAS_NUM_THREADS": "1"})
    assert value == h.canonical_digest(toy_pooled(3, 4))
    assert {"NPY_DISABLE_CPU_FEATURES", "OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE"} == {
        key for env in h.NUMERIC_ENVS for key in env}


def test_canonical_digest_distinguishes_what_matters():
    d = h.canonical_digest
    assert d(np.zeros(3, np.float32)) != d(np.zeros(3, np.float64)) != d(np.zeros((3, 1), np.float32))
    assert d({"a": 1, "b": 2}) != d({"b": 2, "a": 1})
    assert d([1, 2]) != d((1, 2)) and d(0.0) != d(-0.0) and d(1) != d(True) and d("1") != d(1)
    with pytest.raises(TypeError):
        d(object())


# ================================================================== (b)

def test_fixture_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(h.FIXTURE_ENV, raising=False)
    assert h.fixture_root() == h.REPO_ROOT
    monkeypatch.setenv(h.FIXTURE_ENV, str(tmp_path))
    assert h.fixture_root() == tmp_path
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "a.fit").write_bytes(b"x")
    assert h.fixture_path("assets", "a.fit") == tmp_path / "assets" / "a.fit"
    monkeypatch.delenv(h.REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="fixture missing"):
        h.fixture_path("assets", "missing.fit")
    monkeypatch.setenv(h.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="fixture missing"):
        h.fixture_path("assets", "missing.fit")


def test_copy_subset_copies_sorted_writable_files(tmp_path):
    source = tmp_path / "fixture"
    (source / "work").mkdir(parents=True)
    for name in ("r_light_00002.fit", "r_light_00001.fit", "other.txt"):
        path = source / "work" / name
        path.write_bytes(name.encode())
        os.chmod(path, 0o444)                                          # a read-only fixture
    copies = h.copy_subset(source, tmp_path / "copy", ["work/r_light_*.fit"])
    assert [p.name for p in copies] == ["r_light_00001.fit", "r_light_00002.fit"]
    for path in copies:
        path.write_bytes(b"writable")                                  # the copy is not read-only
    assert (source / "work" / "r_light_00001.fit").read_bytes() == b"r_light_00001.fit"
    for path in (source / "work").iterdir():
        os.chmod(path, 0o666)


def test_the_f48_fixture_is_found_through_the_root():
    """Skips in a worktree unless AESTER_FIXTURE_ROOT points at the main checkout (the fixture is untracked). The
    layout later items read: the F48 bench slot, one succeeded 48-frame strict-support job with its work/ folder, and
    the 758 files the 10-05 restore checked."""
    run = h.f48_run()
    slot = json.loads((run / ".aester-bench-slot.json").read_text(encoding="utf-8"))
    assert slot["slot"] == "F48"
    job_dir = h.f48_job()
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert job["state"] == "succeeded" and job["stage"] == "complete" and len(job["inputs"]) == 48
    assert job["support_mode"] == "strict_siril_1_4_4" and (job_dir / "work").is_dir()
    assert sum(1 for path in run.rglob("*") if path.is_file()) == h.F48_FILES


def test_short_path_guard(tmp_path):
    h.assert_short_path(tmp_path)
    with pytest.raises(AssertionError, match="too long for Siril"):
        h.assert_short_path(tmp_path / ("x" * 300))


# ================================================================== (c)

def test_failure_positions():
    assert h.failure_positions(12) == [0, 1, 6, 11]
    assert h.failure_positions(2) == [0, 1] and h.failure_positions(1) == [0] and h.failure_positions(0) == []


@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_injected_failures_end_like_the_serial_loop(k, workers):
    before = h.aester_threads()
    serial = h.outcome(lambda delivered: toy_pooled(5, 1, fail_at=(k,), delivered=delivered))
    with h.pool_conditions(h.Case(workers, None, "reuse", k)):
        parallel = h.outcome(lambda delivered: toy_pooled(5, workers, fail_at=(k,), delivered=delivered))
    assert serial.error_type is ValueError and serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


def test_fail_on_wraps_a_callable_at_chosen_calls():
    calls = []
    wrapped = h.FailOn(lambda x: calls.append(x) or x, at={1, 3}, error=lambda key: KeyError(key))
    assert wrapped("a") == "a"
    with pytest.raises(KeyError):
        wrapped("b")
    assert wrapped("c") == "c" and wrapped.calls == [0, 1, 2] and calls == ["a", "c"]
    by_name = h.FailOn(lambda name: name, at={"r_light_00003"}, key=lambda name: name)
    with pytest.raises(RuntimeError, match="r_light_00003"):
        by_name("r_light_00003")


def test_tree_state_reports_created_modified_and_deleted(tmp_path):
    (tmp_path / "keep").write_bytes(b"1")
    (tmp_path / "change").write_bytes(b"1")
    (tmp_path / "gone").write_bytes(b"1")
    before = h.tree_state(tmp_path)
    (tmp_path / "change").write_bytes(b"22")
    (tmp_path / "gone").unlink()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "new").write_bytes(b"3")
    assert h.changed_files(before, h.tree_state(tmp_path)) == {"created": ["sub/new"], "modified": ["change"],
                                                               "deleted": ["gone"]}
    assert list(h.tree_bytes(tmp_path)) == ["change", "keep", "sub/new"]


# ================================================================== (d)

@pytest.mark.parametrize("workers", [1, 8, 16])
def test_cancel_returns_within_the_limit(workers):
    switch = h.CancelSwitch().trip_after(0.15)
    latency = h.assert_cancels_in_time(lambda check: toy_pooled(1, workers, check=check, slow=0.5), switch)
    assert latency <= h.CANCEL_LIMIT_SECONDS


def test_a_submission_time_cancel_and_a_keyboard_interrupt_subclass():
    class WorkflowCancelled(KeyboardInterrupt):
        pass

    switch = h.CancelSwitch(WorkflowCancelled, "stop").trip_at_call(3)
    h.assert_cancels_in_time(lambda check: toy_pooled(1, 4, check=check, slow=0.05), switch)
    assert len(switch.raised) >= 1 and switch.calls >= 4


def test_a_slow_cancel_fails_the_latency_check():
    switch = h.CancelSwitch().trip_after(0.05)

    def slow(check):
        def unit(x, u):
            time.sleep(0.6)                                           # never probes
            return x
        frame_pool.run_ordered(unit, range(4), workers=1, name="toy-slow", check=check)

    with pytest.raises(AssertionError, match="cancel took"):
        h.assert_cancels_in_time(slow, switch, limit=0.2)


# ================================================================== (f)

def test_peak_sampler_sees_a_pools_memory_and_stays_within_its_declared_limit():
    unit_bytes = 24 << 20

    def unit(x, u):
        block = np.ones(unit_bytes // 8, dtype=np.float64)           # one unit's working memory, touched
        time.sleep(0.03)
        return float(block[::4096].sum())

    stats = {}
    with h.pool_conditions(h.Case(4, 8, "reuse", 0), head_last=True, head_seconds=0.1), h.PeakSampler() as peak:
        frame_pool.run_ordered(unit, range(16), workers=4, inflight=8, name="toy-mem", unit_bytes=unit_bytes,
                               blas=False, stats=stats)
    limit = h.pool_memory_limit(stats, result_bytes=64, blas=False)
    assert limit == 4 * unit_bytes + 8 * 64
    assert peak.traced is not None and peak.traced >= unit_bytes     # tracemalloc: numpy buffers are traced
    if os.name == "nt":
        assert peak.samples > 0 and peak.rise is not None
        assert unit_bytes <= peak.rise <= limit + (64 << 20), (peak.rise, limit)
    print(f"rise {peak.rise} traced {peak.traced} limit {limit} samples {peak.samples}")


def test_process_counters():
    if os.name != "nt":
        assert h.private_bytes() is None and h.peak_pagefile_usage() is None
        return
    now, peak = h.private_bytes(), h.peak_pagefile_usage()
    assert 0 < now <= peak
