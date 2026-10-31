"""P1c T-5x (d): region_blend.inward_weight's 256-row tiles (each read with its halo) on the frame pool
(docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5 (a)-(f)). Called by the job's composition and by a trail
recompose with the job's check, which the serial path still calls once per tile.

The kept reference is inward_weight at the P1c base (tests/p1c_finishing_reference.py): the weight compares as bytes,
the receipt with its key order. Real size: the F48 job's composition weight.
"""
import time

import numpy as np
import pytest
import tifffile

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _inward_feather_reference, _inward_weight_reference
from starstack import frame_pool, milkyway as mw, performance, region_blend as rb

SHAPE = (1100, 48)                     # 5 tiles
UNITS = 5
WIDTHS = (None, 3, 64, 300)
_REFERENCE = {}


def weight(seed, shape=SHAPE):
    rng = np.random.default_rng(seed)
    w = np.ones(shape, np.float32)
    w[-90:] = 0                                                   # ground
    w[rng.random(shape) < .002] = 0                               # small unsupported holes
    w[400:410, 10:30] = 0
    mixed = rng.random(shape) < .05
    w[mixed] = rng.random(int(mixed.sum())).astype(np.float32)
    w[500, 5] = np.nextafter(np.float32(0), np.float32(1))       # a subnormal weight
    return w


def reference(seed, width_px=None):
    if (seed, width_px) not in _REFERENCE:
        _REFERENCE[(seed, width_px)] = _inward_weight_reference(weight(seed), width_px=width_px)
    return _REFERENCE[(seed, width_px)]


def pooled(seed, width_px=None, *, check=None, array=None):
    return rb.inward_weight(weight(seed) if array is None else array, width_px=width_px, check=check)


def reference_target(seed, width_px):
    return reference(seed, width_px)


def pooled_target(seed, width_px):
    with fd.forced(h.Case(8, None, "reuse", 0)):
        return pooled(seed, width_px)


def test_the_synthetic_weight_is_feathered():
    _, receipt = reference(0)
    assert receipt["changed_pixels"] > 100 and receipt["mixed_pixels"] > 100


@pytest.mark.parametrize("width", [1, 3, 24, 64, 300, 512, 4096])
def test_the_feather_with_its_probes_matches_the_base_feather(width):
    """milkyway._inward_feather now runs one whole-tile step per statement with abort probes between them (R18, 16
    workers) and builds its float32 start directly (no int64 intermediate): the same bytes as at aab6f21, with or
    without a probe. It also serves the Milky Way preparation, the reduction's weight and the finishing layers."""
    rng = np.random.default_rng(width)
    probes = []
    for shape in ((1, 1), (1, 37), (53, 1), (97, 131), (514, 64)):
        for density in (0., .003, .2, 1.):
            unsafe = rng.random(shape) < density
            expected = _inward_feather_reference(unsafe, width)
            for probe in (None, lambda: probes.append(1)):
                actual = mw._inward_feather(unsafe, width, probe=probe)
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes(), (shape, density, width)
    assert probes


@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_reference(case):
    before = h.aester_threads()
    with fd.notes() as seen, fd.forced(case):
        actual = pooled(case.seed)
    h.assert_same(reference(case.seed), actual)
    entry = fd.pool_entries(seen, "inward_weight")
    width_px = actual[1]["width_px"]
    assert len(entry) == 1 and entry[0]["units"] == UNITS and entry[0]["workers"] == min(case.workers, UNITS)
    assert entry[0]["unit_bytes"] == rb.inward_unit_bytes(SHAPE[1], width_px)
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("width_px", WIDTHS[1:])
@pytest.mark.parametrize("workers", [1, 4, 16])
def test_every_feather_width_matches_the_reference(width_px, workers):
    with fd.forced(h.Case(workers, None, "fresh", workers)):
        h.assert_same(reference(1, width_px), pooled(1, width_px))


def test_reruns_under_two_hash_seeds_give_one_digest():
    digest = h.assert_hashseed_stable("test_p1c_site_inward_weight:reference_target",
                                      "test_p1c_site_inward_weight:pooled_target", 2, 64)
    assert digest == h.canonical_digest(reference(2, 64))


def test_the_serial_path_keeps_one_check_per_tile():
    calls = []
    with fd.forced(h.Case(1, None, "reuse", None)):
        pooled(0, check=lambda: calls.append(1))
    assert len(calls) == UNITS
    expected = []
    _inward_weight_reference(weight(0), check=lambda: expected.append(1))
    assert len(expected) == len(calls)


# ================================================================== (c) failures

@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_tile_ends_like_the_serial_loop(k, workers):
    def run(case):
        array = weight(5)
        array[k * 256 + 40, 7] = np.nan                            # tile k only (the halos are 4 rows)
        def body(delivered):
            with fd.forced(case, delivered=delivered):
                return pooled(5, 3, array=array)
        return h.outcome(body)

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.message == "composition weight must contain finite values in 0..1"
    assert serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet()


# ================================================================== (a)/(b) real size, (d), (f)

@pytest.fixture(scope="module")
def f48_weight():
    f48 = fd.F48()
    array = tifffile.imread(f48.compose_weight).astype(np.float32)
    return array, _inward_weight_reference(array)


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_matches_the_reference(f48_weight, case):
    array, expected = f48_weight
    with fd.notes() as seen, fd.forced(case):
        actual = rb.inward_weight(array)
    h.assert_same(expected, actual)
    entry = fd.pool_entries(seen, "inward_weight")[0]
    assert entry["units"] == 22 and entry["workers"] == (case.workers or entry["requested"])
    h.assert_pool_quiet()


@pytest.mark.parametrize("workers", [None, 8, 16])
def test_a_cancel_returns_within_a_second_at_real_size(f48_weight, workers):
    array, _ = f48_weight
    switch = h.CancelSwitch().trip_after(.15)
    with fd.forced(h.Case(workers, None, "reuse", None)):
        latency = h.assert_cancels_in_time(lambda check: rb.inward_weight(array, width_px=512, check=check), switch)
    print(f"workers {workers}: cancel latency {latency:.3f} s")


def test_the_pool_peak_stays_within_its_declared_bound(f48_weight):
    array, _ = f48_weight
    expected = _inward_weight_reference(array, width_px=128)
    with fd.notes() as seen, fd.forced(h.Case(8, 16, "reuse", 0), head_last=True, head_seconds=.5), \
            h.PeakSampler(trace=False) as peak:
        actual = rb.inward_weight(array, width_px=128)
    h.assert_same(expected, actual)
    entry = fd.pool_entries(seen, "inward_weight")[0]
    limit = h.pool_memory_limit(entry, blas=False) + array.nbytes      # + the returned weight (allocated before)
    assert entry["workers"] == 8 and entry["unit_bytes"] == rb.INWARD_UNIT_BYTES
    if peak.rise is not None:
        assert peak.rise <= limit, (peak.rise, limit)
    print(f"inward_weight: rise {peak.rise} limit {limit}")


def test_a_wide_feather_lowers_the_worker_count(monkeypatch):
    """(f) decision: width_px=512 reads 1282 rows per tile (2.5 x the 514 of width_px=128)."""
    monkeypatch.delenv(frame_pool.ENV_WORKERS, raising=False)
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: 32)
    monkeypatch.setattr(performance, "_memory_status", lambda: (4 << 30, 4 << 30))
    narrow = performance.frame_workers(22, rb.inward_unit_bytes(8280, 128), rb.INWARD_WORKER_CAP, blas=False)
    wide = performance.frame_workers(22, rb.inward_unit_bytes(8280, 512), rb.INWARD_WORKER_CAP, blas=False)
    assert rb.inward_unit_bytes(8280, 128) == rb.INWARD_UNIT_BYTES
    assert (narrow, wide) == (8, 4)
