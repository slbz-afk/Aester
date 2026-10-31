"""A5 for the efficiency P1c site finite_inventory (F-3b, docs/EFFICIENCY_P1C_PREREG_20261005.md §3.3, §6.2 (a)-(h)).

finite_stack._source_inventory (the finite stack's input check, one frame per unit) against the retained pre-P1c
finite_stack._finite_inventory_reference: the entries (key order), the reference frame, the eligible descriptor and
counts, for the synthetic matrix, at real size on the F48 reference job (its first REAL_FRAMES sources copied into
basetemp as run() saw them; workers 1, the site's default, 16), under two hash seeds; failures at frames {0, 1,
middle, last} (the serial error, nothing written); cancels within 1 s (R18: a worker's export hash probes every
8 MiB); the memory peak within the reservation; and core's plan (_FinitePools.inventory).
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
from finite_fixtures import FixedPools, f48_inventory_work, make_work
from starstack import core, finite_stack as fs, frame_pool, performance, registered_support as rs

COUNT = 7
REAL_FRAMES = int(os.environ.get("AESTER_P1C_FINITE_INVENTORY_FRAMES", "8"))
GiB = 1 << 30


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    return make_work(tmp_path_factory.mktemp("finite-inventory") / "job", count=COUNT)


@pytest.fixture(scope="module")
def reference(job):
    return fs._finite_inventory_reference(job.work, lambda: None)


def _same(expected, found):
    items, reference_file, descriptor, eligible = found
    harness.assert_same(harness.json_bytes(expected[0]), harness.json_bytes(items), "items")
    assert reference_file == expected[1]
    harness.assert_same(harness.json_bytes(expected[2]), harness.json_bytes(descriptor), "descriptor")
    harness.assert_same(expected[3], eligible, "eligible")


def _pools(workers, inflight=None):
    return FixedPools(inventory=fs.Plan(workers, None, fs.REFERENCE, inflight))


# ================================================================== (a) reference vs pool

@pytest.mark.parametrize("case", harness.params(harness.synthetic_cases(policies=("reuse",))))
def test_synthetic_matrix_matches_the_reference(job, reference, case):
    """The pool has no init (no state): the state policy is moot, so only 'reuse' runs."""
    before = harness.aester_threads()
    pools = _pools(case.workers, case.inflight)
    with harness.pool_conditions(case):
        found = fs._source_inventory(job.work, lambda: None, pools)
    _same(reference, found)
    stats = pools.stats("finite_inventory")
    assert stats["workers"] == min(case.workers, COUNT) and stats["serial"] == (case.workers == 1)
    assert pools.operations == ["finite_inventory"]
    assert pools.shadows == [("py.finite_inventory", stats["workers"], COUNT, 150 * 40)]
    harness.assert_pool_quiet(before)


def hashseed_side(folder, side):
    root = Path(folder) / f"{side}-{os.getpid()}-{time.perf_counter_ns()}"
    job = make_work(root / "job", count=COUNT)
    if side == "reference":
        found = fs._finite_inventory_reference(job.work, lambda: None)
    else:
        found = fs._source_inventory(job.work, lambda: None, _pools(4))
    shutil.rmtree(root)
    return list(found)


def test_both_sides_do_not_depend_on_the_hash_seed(tmp_path):
    digests = {side: harness.hashseed_digests("test_p1c_site_finite_inventory:hashseed_side", str(tmp_path), side)
               for side in ("reference", "pool")}
    assert len({d for per in digests.values() for d in per.values()}) == 1, digests


# ---- real size: the F48 job's first REAL_FRAMES sources ----------------------------------------------------------

@pytest.fixture(scope="module")
def f48(tmp_path_factory):
    work = f48_inventory_work(harness.f48_job(), tmp_path_factory.mktemp("f48-inventory") / "work", REAL_FRAMES)
    yield work, fs._finite_inventory_reference(work, lambda: None)
    shutil.rmtree(work, ignore_errors=True)


class _Recorded(core._FinitePools):
    def __init__(self):
        super().__init__(None, "auto")
        self.notes = []

    def note_pool(self, site, stats, **extra):
        self.notes.append((site, dict(stats), extra))
        super().note_pool(site, stats, **extra)


@pytest.mark.parametrize("case", harness.params(harness.real_cases()))
def test_real_size_matches_the_reference(f48, monkeypatch, case):
    work, reference = f48
    if case.workers is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, str(case.workers))
    pools = _Recorded()
    before = harness.aester_threads()
    with harness.pool_conditions(case):
        found = fs._source_inventory(work, lambda: None, pools)
    _same(reference, found)
    _, stats, _ = pools.notes[-1]
    decided = pools.decisions["finite_inventory"][0]["workers"]          # core's plan (cap, CPUs, memory, override)
    assert stats["workers"] == min(decided, len(reference[0])), (stats, pools.decisions)
    if case.workers is not None:                                         # the override, clamped to the frames
        assert decided == min(case.workers, frame_pool.MAX_FRAME_WORKERS, len(reference[0]))
    harness.assert_pool_quiet(before)


# ================================================================== (c) failures

def _corrupt_record(job, k, kind):
    record = json.loads((job.work / rs.RECORD).read_text(encoding="utf-8"))
    frame = record["frames"][k]
    if kind == "digest":
        frame["export_after_sha256"] = "0" * 64
    elif kind == "count":
        frame["supported_pixels"] += 1
    elif kind == "validity":
        frame["explicit_validity"] = {**frame["explicit_validity"], "sha256": "1" * 64}
    (job.work / rs.RECORD).write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("k", harness.failure_positions(COUNT))
@pytest.mark.parametrize("kind", ["digest", "count", "validity", "injected"])
def test_a_failure_at_frame_k_is_the_serial_one(tmp_path, monkeypatch, k, kind):
    job = make_work(tmp_path / "job", count=COUNT)
    if kind == "injected":
        real = fs.sha

        def sha(path, check=None):
            if Path(path).name == f"r_light_{k + 1:05d}.fit":
                raise OSError(f"injected read error on {Path(path).name}")
            return real(path, check)
        monkeypatch.setattr(fs, "sha", sha)
    else:
        _corrupt_record(job, k, kind)
    before_tree = harness.tree_state(job.directory)
    outcomes = {}
    for workers in ("reference", 1, 2, 4, 8, 16):
        threads = harness.aester_threads()

        def run(delivered):
            if workers == "reference":
                return fs._finite_inventory_reference(job.work, lambda: None)
            return fs._source_inventory(job.work, lambda: None, _pools(workers))
        with harness.pool_conditions(harness.Case(1 if workers == "reference" else workers, None, "reuse", k)):
            outcomes[workers] = harness.outcome(run)
        harness.assert_pool_quiet(threads)
    serial = outcomes.pop("reference")
    assert serial.error_type is not None, serial
    for workers, found in outcomes.items():
        harness.assert_like_serial(serial, found)
    assert harness.tree_state(job.directory) == before_tree                 # the inventory writes nothing


# ================================================================== (d) cancellation

@pytest.mark.parametrize("workers", [None, 8, 16])
@pytest.mark.parametrize("when", ["submission", "waiting"])
def test_a_cancel_returns_within_the_limit_at_real_size(f48, monkeypatch, workers, when):
    work, _ = f48
    if workers is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, str(workers))
    switch = harness.CancelSwitch()
    held = contextlib.nullcontext()
    if when == "submission":
        switch.trip_at_call(3)
    else:
        # unit 0 starts late, so the pool is still waiting at the trip: with a warm file cache 8 or 16 workers
        # finish the 48 frames in under 0.5 s and nothing would be cancelled
        held = harness.pool_conditions(harness.Case(workers, None, "reuse", None), audit=False, head_last=True,
                                       head_seconds=0.8)
        switch.trip_after(0.5)
    with held:
        latency = harness.assert_cancels_in_time(lambda check: fs._source_inventory(work, check, _Recorded()),
                                                 switch)
    print(f"finite_inventory cancel ({when}, workers={workers}): {latency:.3f} s")


# ================================================================== (f) memory

def test_the_pool_s_memory_peak_stays_within_its_reservation(f48, monkeypatch):
    work, _ = f48
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    pools = _Recorded()
    with harness.pool_conditions(harness.Case(None, None, "reuse", 0), head_last=True, head_seconds=1.0), \
            harness.PeakSampler() as peak:
        fs._source_inventory(work, lambda: None, pools)
    _, stats, _ = pools.notes[-1]
    limit = harness.pool_memory_limit(stats, blas=False)
    # the coordinator's own reads around the pool: the support record, the eligible counts (2 bytes a pixel)
    pixels = 5520 * 8280
    print(f"finite_inventory peak rise {peak.rise / GiB:.3f} GiB over {stats['workers']} workers "
          f"(unit_bytes {stats['unit_bytes'] / GiB:.3f} GiB, limit {limit / GiB:.3f} GiB)")
    if peak.rise is not None:
        assert peak.rise <= limit + 4 * pixels + (64 << 20), (peak.rise, limit)


@pytest.mark.parametrize("available, setting, expected", [
    (64 * GiB, None, 4), (1.1 * GiB, None, 1), (1.2 * GiB, None, 2), (None, None, 1), (64 * GiB, "max", None),
    (64 * GiB, "2", 2)])
def test_core_plans_the_inventory_from_memory(monkeypatch, available, setting, expected):
    pixels = 5520 * 8280
    monkeypatch.setattr(performance, "_memory_status",
                        lambda: (None, None) if available is None else (int(available), int(available)))
    if setting is None:
        monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    else:
        monkeypatch.setenv(frame_pool.ENV_WORKERS, setting)
    plan = core._FinitePools(None, "auto").inventory(units=48, pixels=pixels)
    if expected is None:
        expected = min(frame_pool.MAX_FRAME_WORKERS, frame_pool.usable_cpus())
    elif setting is None:                                # memory decides, or the cap, or this machine's CPUs / 2
        expected = min(expected, max(1, frame_pool.usable_cpus() // 2))
    assert plan.workers == expected
    assert plan.unit_bytes == 2 * pixels + (8 << 20) and abs(plan.unit_bytes - 100e6) < 1e6   # 2P + 8 MiB, ~100 MB
    assert core._FinitePools(None, "auto").inventory(units=48, pixels=None) == fs.Plan()
