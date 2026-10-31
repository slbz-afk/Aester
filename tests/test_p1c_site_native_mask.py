"""Efficiency P1c G-4t (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.2, §6.2 A5): the native mask's row blocks on the
frame pool.

registered_support.native_mask (quick route) against the kept oracles: its own serial loop (workers=1) and the pre-P1a
body registered_support._native_mask_reference.
(a) the synthetic matrix over models that reach every route of a block (all-valid / all-invalid footprints, mixed ones
    through sample_support, a model whose output is not float64 (the reference decision)), R11's in flight = every
    block, the hash-seed reruns; (b) the F48 reference job's stack map, workers {1, default, 16}, against its recorded
    native map; (c) a failing model block at {0, 1, middle, last}; (d) cancellation, at real size and on a worst-case
    map (every footprint mixed), with the parallel path's probes inside a block (R18); (f) the pool's memory on the
    worst-case map; (g) the numeric variants; plus core's decision and its note (only on the quick route, inside
    finish_stack's "native_mask" operation).
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

import p1c_site_harness as h
from starstack import core, frame_pool, performance, profiling, registered_support as rs
from starstack.registration_distortion import RadialDistortion
from test_registered_support_independent import _write_rgb
from test_strict_support_kernel import Shift, Table, _edge_points

HEIGHT, WIDTH, ROWS = 90, 121, 7          # 13 row blocks
GB = 10 ** 9


def stack_map(seed=3, shape=(HEIGHT, WIDTH)):
    rng = np.random.default_rng(seed)
    valid = rng.random(shape) < .97
    valid[shape[0] // 2:, : shape[1] // 3] = False                     # large all-invalid and all-valid regions
    valid[5:9, 40:44] = False
    return valid


def models(shape=(HEIGHT, WIDTH)):
    height, width = shape
    return {"radial_u_to_p": RadialDistortion(.04, -.01, width / 2, height / 2, max(shape), .98, "u_to_p"),
            "radial_p_to_u": RadialDistortion(-.03, .01, width / 3, height / 2, max(shape), 1., "p_to_u"),
            "shift": Shift(.25, -.25),
            "edges": Table(_edge_points(width, height)),
            "float32": Table(_edge_points(width, height), dtype=np.float32)}       # native_block declines: reference


def oracle(model_name, seed=3):
    return rs._native_mask_reference(stack_map(seed), models()[model_name], rows=ROWS)


def pooled(model_name, workers, seed=3, pool_stats=None, check=None):
    return rs.native_mask(stack_map(seed), models()[model_name], rows=ROWS, quick=True, workers=workers,
                          pool_stats=pool_stats, check=check, unit_bytes=1 << 20)


def hashseed_reference(model_name):
    return oracle(model_name)


def hashseed_pool(model_name, probe_points=None):
    """The pool at 8 workers; ``probe_points`` shrinks support_kernel.PROBE_POINTS so every block's mixed footprints
    go through sample_support in probed chunks (the parallel path's R18 probes)."""
    from starstack import support_kernel as sk
    if probe_points is not None:
        sk.PROBE_POINTS = probe_points
    with h.pool_conditions(h.Case(8, None, "reuse", 2)):
        return pooled(model_name, 8)


@pytest.fixture(scope="module")
def expected():
    return {name: oracle(name) for name in models()}


class _Inflight:
    def __init__(self, value):
        self.value, self.real = value, frame_pool.run_ordered

    def __enter__(self):
        def run(*args, **kwargs):
            if self.value is not None:
                kwargs["inflight"] = self.value
            return self.real(*args, **kwargs)
        frame_pool.run_ordered = run

    def __exit__(self, *exc):
        frame_pool.run_ordered = self.real


BLOCKS = len(range(0, HEIGHT, ROWS))


# ================================================================== (a)

@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_serial_loop_and_the_reference(expected, case):
    before = h.aester_threads()
    for name, wanted in expected.items():
        pool_stats = {}
        with h.pool_conditions(case), _Inflight(case.inflight):
            actual = pooled(name, case.workers, pool_stats=pool_stats)
        h.assert_same(wanted, actual, name)
        assert actual.flags.c_contiguous
        assert pool_stats["workers"] == min(case.workers, BLOCKS) and pool_stats["unit_bytes"] == 1 << 20
        assert pool_stats["units"] == BLOCKS
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("policy", ["reuse", "fresh"])
def test_prepare_depends_only_on_earlier_blocks(expected, policy):
    with h.pool_conditions(h.Case(4, BLOCKS, policy, 4)), _Inflight(BLOCKS):
        for name, wanted in expected.items():
            h.assert_same(wanted, pooled(name, 4), name)


def test_the_serial_path_keeps_the_former_calls():
    """workers=1: check() before each block, the tap map once, after the first check (as before P1c)."""
    from starstack import support_kernel as sk
    events = []
    real = sk.tap_state

    def tap_state(valid, **kwargs):
        events.append("tap")
        return real(valid, **kwargs)
    sk.tap_state = tap_state
    try:
        rs.native_mask(stack_map(), models()["shift"], rows=ROWS, quick=True, check=lambda: events.append("check"))
    finally:
        sk.tap_state = real
    assert events == ["check", "tap"] + ["check"] * (BLOCKS - 1)


def test_the_parallel_path_probes_inside_a_block(expected, monkeypatch):
    """R18 on the parallel path only: a unit hands native_block its probe, which sends the mixed footprints (and the
    reference decision of an unusual model output) through sample_support in chunks of PROBE_POINTS, a probe before
    each; with chunks of 16 points every model still equals the oracle. The serial path passes no probe: one
    sample_support call per block, as before."""
    from starstack import support_kernel as sk
    monkeypatch.setattr(sk, "PROBE_POINTS", 16)
    probed, sizes = [], []
    real_block, real_sample = sk.native_block, sk.sample_support

    def native_block(*args, check=None):
        probed.append(check is not None)
        return real_block(*args, check=check)

    def sample_support(valid, points, **kwargs):
        sizes.append(len(points))
        return real_sample(valid, points, **kwargs)
    monkeypatch.setattr(sk, "native_block", native_block)
    monkeypatch.setattr(sk, "sample_support", sample_support)
    chunked = 0
    for name, wanted in expected.items():
        serial = None
        for workers in (1, 3, 16):
            probed.clear()
            sizes.clear()
            with h.pool_conditions(h.Case(workers, None, "reuse", 1)):
                h.assert_same(wanted, pooled(name, workers), f"{name} at {workers}")
            assert len(probed) == BLOCKS and set(probed) == {workers > 1}, (name, workers)
            if workers == 1:
                serial = list(sizes)
                assert len(serial) <= BLOCKS, name          # at most one call per block
                continue
            assert sum(sizes) == sum(serial) and (not sizes or max(sizes) <= 16), (name, workers)
            chunked += len(sizes) > len(serial)
    assert chunked, "no block had more than 16 points for sample_support"


def test_the_reference_route_and_no_model_run_no_pool():
    pool_stats = {}
    valid = stack_map()
    out = rs.native_mask(valid, models()["shift"], rows=ROWS, quick=False, workers=8, pool_stats=pool_stats)
    h.assert_same(rs._native_mask_reference(valid, models()["shift"], rows=ROWS), out)
    assert pool_stats == {}
    assert np.array_equal(rs.native_mask(valid, None, workers=8, pool_stats=pool_stats), valid) and pool_stats == {}


def test_reruns_under_two_hash_seeds_give_one_digest():
    for name in ("radial_u_to_p", "float32"):
        h.assert_hashseed_stable("test_p1c_site_native_mask:hashseed_reference",
                                 "test_p1c_site_native_mask:hashseed_pool", name)


def test_numeric_variants_give_the_same_result(expected):
    for name in ("radial_u_to_p", "radial_p_to_u"):
        wanted = h.canonical_digest(expected[name])
        for env in h.NUMERIC_ENVS:
            assert h.subprocess_digest("test_p1c_site_native_mask:hashseed_pool", name, env=env) == wanted, (name, env)
            assert h.subprocess_digest("test_p1c_site_native_mask:hashseed_pool", name, 16, env=env) == wanted, \
                (name, env, "chunks of 16")


# ================================================================== (c) failure

class _FailingModel:
    """A model whose apply() raises for the block starting at ``row`` (keyed by its first point's y)."""

    def __init__(self, inner, row):
        self.inner, self.row = inner, row

    def apply(self, points):
        if int(points[0, 1]) == self.row:
            raise ValueError(f"model failed at row {self.row}")
        return self.inner.apply(points)


@pytest.mark.parametrize("k", h.failure_positions(BLOCKS))
@pytest.mark.parametrize("workers", [2, 4, 16])
def test_a_failing_block_ends_like_the_serial_loop(k, workers):
    model = _FailingModel(models()["radial_u_to_p"], k * ROWS)
    valid = stack_map()
    serial = h.outcome(lambda delivered: rs.native_mask(valid, model, rows=ROWS, quick=True))
    assert serial.error_type is ValueError and serial.message == f"model failed at row {k * ROWS}"
    before = h.aester_threads()
    with h.pool_conditions(h.Case(workers, None, "reuse", k)):
        parallel = h.outcome(lambda delivered: rs.native_mask(valid, model, rows=ROWS, quick=True, workers=workers))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


# ================================================================== (d) cancellation (synthetic)

@pytest.mark.parametrize("call", [0, 1, 6])
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_a_cancel_at_a_submission(call, workers):
    switch = h.CancelSwitch().trip_at_call(call)
    with h.pool_conditions(h.Case(workers, None, "reuse", call)):
        h.assert_cancels_in_time(lambda check: rs.native_mask(stack_map(), models()["radial_u_to_p"], rows=ROWS,
                                                              quick=True, workers=workers, check=check), switch)


# ================================================================== (f) memory: every footprint mixed

def test_worst_case_pool_memory_stays_within_its_declared_bound():
    """A checkerboard map makes every footprint mixed, so every point goes through sample_support (~312 B each): the
    worst case the unit figure (NATIVE_UNIT_BYTES at width 8280, scaled by width) is set for."""
    height, width = 64 * 12, 8280
    valid = (np.indices((height, width)).sum(axis=0) % 2).astype(bool)
    model = Shift(.3, .3)
    unit_bytes = core._native_mask_workers(height, width, {})[1]
    pool_stats = {}
    with h.pool_conditions(h.Case(4, 8, "reuse", 0), head_last=True, head_seconds=1.0), _Inflight(8), \
            h.PeakSampler(0.002, trace=False) as peak:
        actual = rs.native_mask(valid, model, quick=True, workers=4, pool_stats=pool_stats, unit_bytes=unit_bytes)
    serial_stats = {}
    with h.PeakSampler(0.002, trace=False) as single:
        h.assert_same(rs.native_mask(valid, model, quick=True, pool_stats=serial_stats), actual)
    limit = h.pool_memory_limit(pool_stats, result_bytes=64 * width, blas=False)
    print(f"native_mask worst-case rise: 4 workers {peak.rise / 2**20:.0f} MiB (limit {limit / 2**20:.0f} MiB), "
          f"serial {single.rise / 2**20:.0f} MiB; unit_bytes {unit_bytes / 2**20:.0f} MiB")
    if os.name == "nt":
        assert peak.rise <= limit + 2 * height * width, (peak.rise, limit)    # + the tap map and the result
        assert single.rise <= unit_bytes + 2 * height * width


@pytest.mark.parametrize("workers", [None, 4, 8, 16])
def test_worst_case_cancel_returns_within_a_second(workers):
    """(d) on the map the unit figure is sized for (a checkerboard: every footprint mixed, 0.42 s per 64-row block at
    W=8280 serially, 0.32 s of it in sample_support): tripped as a unit starts a sample_support chunk while every
    worker is busy, the pool returns within 1 s (the parallel path probes before each chunk of PROBE_POINTS)."""
    from starstack import support_kernel as sk
    height, width = 64 * 24, 8280
    valid = (np.indices((height, width)).sum(axis=0) % 2).astype(bool)
    model = RadialDistortion(.01, -.002, width / 2, height / 2, width / 2, .99, "u_to_p")
    workers = core._native_mask_workers(height, width, {})[0] if workers is None else workers
    switch = h.CancelSwitch()
    calls = [0]
    lock = threading.Lock()
    real = sk.sample_support

    def sample_support(*args, **kwargs):
        with lock:
            calls[0] += 1
            trip = calls[0] == 2 * workers              # every worker is in a block by then
        if trip:
            switch.trip()
        return real(*args, **kwargs)
    sk.sample_support = sample_support
    try:
        with h.pool_conditions(h.Case(workers, None, "reuse", None)):
            latency = h.assert_cancels_in_time(lambda check: rs.native_mask(valid, model, quick=True, workers=workers,
                                                                            check=check), switch)
    finally:
        sk.sample_support = real
    print(f"native_mask worst-case cancel latency {latency:.3f} s at {workers} workers")


# ================================================================== core: the decision, the note, the operation

@pytest.mark.parametrize("available_gb,expected_workers", [(10, 4), (2.0, 3), (1.5, 2), (1.2, 1)])
def test_core_decision(monkeypatch, available_gb, expected_workers):
    monkeypatch.delenv("AESTER_FRAME_WORKERS", raising=False)
    monkeypatch.setattr(performance, "_memory_status", lambda: (int(available_gb * GB), None))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 64)
    decision = {}
    workers, unit_bytes, blocks = core._native_mask_workers(5520, 8280, decision)
    assert (workers, blocks) == (expected_workers, 87) and unit_bytes == rs.NATIVE_UNIT_BYTES
    assert decision["blas"] is False and decision["workers"] == expected_workers
    assert core._native_mask_workers(5520, 4140, {})[1] == rs.NATIVE_UNIT_BYTES          # never below the figure
    assert core._native_mask_workers(5520, 16560, {})[1] == 2 * rs.NATIVE_UNIT_BYTES     # scaled by the width


def _finished_support(work, label):
    """A work folder whose support record is ready for finish_stack (2 exports, 200 x 40: four 64-row blocks)."""
    from test_strict_support_kernel import Static, _work
    _work(work, np.random.default_rng(21), 200, 40, 2, False)
    valid = np.ones((200, 40), bool)
    valid[150:, :] = False
    rs.prepare_exports(work, Static(valid), None)
    if label == "reference":
        record = json.loads((work / rs.RECORD).read_text(encoding="utf-8"))
        record["implementation"] = rs.REFERENCE_IMPLEMENTATION
        (work / rs.RECORD).write_text(json.dumps(record), encoding="utf-8")
    _write_rgb(work / "sky_master.fit", np.random.default_rng(2).random((200, 40)) + .1)


@pytest.mark.parametrize("label", ["buffered", "reference"])
def test_finish_stack_runs_the_pool_inside_its_operation_on_the_quick_route_only(tmp_path, monkeypatch, label):
    monkeypatch.delenv("AESTER_STRICT_KERNEL", raising=False)
    _finished_support(tmp_path / "work", label)
    model = RadialDistortion(.02, 0, 20, 100, 60, 1, "u_to_p")
    timer = profiling.StageTimer()
    pool_stats = {}
    done = rs.finish_stack(tmp_path / "work", tmp_path / "work" / "sky_master.fit", model, native_workers=3,
                           native_unit_bytes=1 << 20, pool_stats=pool_stats, operation=timer.operation)
    report = timer.finish()
    stack = rs.load_array(tmp_path / "work", done["stack_valid"])
    assert np.array_equal(rs.load_array(tmp_path / "work", done["native_valid"]), rs._native_mask_reference(stack, model))
    assert report["operation_counts"]["native_mask"] == 1
    if label == "buffered":
        assert pool_stats["workers"] == 3 and pool_stats["serial"] is False and pool_stats["unit_bytes"] == 1 << 20
    else:
        assert pool_stats == {}
    # without an operation (the former callers): the same record, no timer needed
    again = rs.finish_stack(tmp_path / "work", tmp_path / "work" / "sky_master.fit", model)
    assert again["native_valid"] == done["native_valid"]


# ================================================================== real size: the F48 reference job's stack map

@pytest.fixture(scope="module")
def f48_native():
    work = h.f48_job() / "work"
    record = json.loads((work / rs.RECORD).read_text(encoding="utf-8"))
    assert record["implementation"] == rs.SUPPORT_IMPLEMENTATION
    return (rs.load_array(work, record["stack_valid"]), RadialDistortion(**record["model"]),
            rs.load_array(work, record["native_valid"]))


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_pool_reproduces_the_recorded_native_map(f48_native, case):
    stack, model, recorded = f48_native
    height, width = stack.shape
    workers, unit_bytes, blocks = core._native_mask_workers(height, width, {})
    workers = workers if case.workers is None else case.workers
    pool_stats = {}
    before = h.aester_threads()
    with h.pool_conditions(case):
        actual = rs.native_mask(stack, model, quick=True, workers=workers, pool_stats=pool_stats,
                                unit_bytes=unit_bytes)
    h.assert_same(recorded, actual)
    assert pool_stats["units"] == blocks == 87 and pool_stats["workers"] == min(workers, blocks)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [None, 8, 16])
def test_real_size_cancel_returns_within_a_second(f48_native, workers):
    stack, model, _ = f48_native
    height, width = stack.shape
    workers = core._native_mask_workers(height, width, {})[0] if workers is None else workers
    switch = h.CancelSwitch().trip_after(1.0)
    with h.pool_conditions(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(lambda check: rs.native_mask(stack, model, quick=True, workers=workers,
                                                                        check=check), switch)
    print(f"native_mask cancel latency {latency:.3f} s at {workers} workers")
