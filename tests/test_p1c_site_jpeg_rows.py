"""P1c T-5x (c): whole-image 16-to-8-bit JPEG codes built from 128-row blocks on the frame pool
(photographic_job._jpeg_rows; the reduction and the trail add-back each note under their own key)
(docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5 (a)-(f)). No cancel point.

The kept references are the two whole-image expressions at the P1c base (tests/p1c_finishing_reference.py); every
byte must be equal, and the whole-image float32 temporaries (1.1-1.6 GB at 45.7 MP) must be gone.
2026-10-09: the publication re-encode at a request's jpeg_quality (workflow._encode_jpeg, its own note key) converts
each block with display_transform.to_8bit; its reference is the same function on the whole image.
"""
import numpy as np
import pytest

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _jpeg_rows_reference_reduction, _jpeg_rows_reference_trail_pair
from starstack import display_transform as dt, photographic_job as job

SHAPE = (600, 20, 3)                   # 5 blocks
UNITS = 5
SITES = {"reduction": (_jpeg_rows_reference_reduction,
                       lambda rows: np.rint(rows.astype(np.float32)/np.float32(257)).astype(np.uint8)),
         "trail_pair": (_jpeg_rows_reference_trail_pair,
                        lambda rows: np.rint(rows.astype(np.float32)/257).astype(np.uint8)),
         "publish": (dt.to_8bit, dt.to_8bit)}


def codes(seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 65536, SHAPE, dtype=np.uint16)
    a[0, :3] = ((128, 385, 642), (0, 65535, 32896), (128 + 257 * 7, 1, 2))   # exact .5 ties of the rounding
    return a


def pooled(seed, site, *, array=None, convert=None):
    pool_stats = {}
    value = job._jpeg_rows(codes(seed) if array is None else array, convert or SITES[site][1], pool_stats=pool_stats)
    return value, pool_stats


def reference_target(seed, site):
    return SITES[site][0](codes(seed))


def pooled_target(seed, site):
    with fd.forced(h.Case(8, None, "reuse", 0)):
        return pooled(seed, site)[0]


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_pool_matches_the_whole_image_expression(case, site):
    before = h.aester_threads()
    with fd.forced(case):
        value, pool_stats = pooled(case.seed, site)
    h.assert_same(SITES[site][0](codes(case.seed)), value)
    assert value.flags.c_contiguous and pool_stats["units"] == UNITS
    assert pool_stats["workers"] == min(case.workers, UNITS) and pool_stats["unit_bytes"] == job.jpeg_unit_bytes(20)
    h.assert_pool_quiet(before)


def test_reruns_under_two_hash_seeds_give_one_digest():
    for site in SITES:
        digest = h.assert_hashseed_stable("test_p1c_site_jpeg_rows:reference_target",
                                          "test_p1c_site_jpeg_rows:pooled_target", 2, site)
        assert digest == h.canonical_digest(SITES[site][0](codes(2)))


@pytest.mark.parametrize("k", h.failure_positions(UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_block_ends_like_the_serial_loop(k, workers):
    def convert(rows):
        if rows.base is not None and np.shares_memory(rows, array[k * 128:k * 128 + 1]):
            raise ValueError(f"injected conversion failure in block {k}")
        return SITES["reduction"][1](rows)

    array = codes(3)

    def run(case):
        with fd.forced(case):
            return h.outcome(lambda delivered: pooled(3, "reduction", array=array, convert=convert))

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.message == f"injected conversion failure in block {k}"
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet()


@pytest.fixture(scope="module")
def f48_final():
    f48 = fd.F48()
    final = job._read_display(f48.reduction / "star_reduced_srgb16.tif")
    try:
        yield final
    finally:
        job._close_map(final)


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_codes_are_the_whole_image_codes(f48_final, case, site):
    expected = SITES[site][0](f48_final)
    with fd.forced(case):
        value, pool_stats = pooled(0, site, array=f48_final)
    h.assert_same(expected, value)
    assert pool_stats["units"] == 44 and pool_stats["workers"] == (case.workers or pool_stats["requested"])


@pytest.mark.parametrize("workers", [1, 8])
def test_the_whole_image_temporaries_are_gone(f48_final, workers):
    """(f) and prereg §1 T-5x: the whole-image conversion held two float32 copies of the photograph (1.1 GB at
    45.7 MP); the blocks hold the uint8 result plus workers x one block, within the declared bound."""
    with h.PeakSampler(trace=False) as whole:
        expected = SITES["reduction"][0](f48_final)
    with fd.forced(h.Case(workers, None, "reuse", 0), head_last=workers > 1, head_seconds=.2), \
            h.PeakSampler(trace=False) as blocks:
        value, pool_stats = pooled(0, "reduction", array=f48_final)
    h.assert_same(expected, value)
    print(f"jpeg codes: whole-image rise {whole.rise}, pooled rise {blocks.rise} ({pool_stats['workers']} workers)")
    if whole.rise is not None:
        float_copies = 2 * 4 * f48_final.size
        assert whole.rise >= float_copies, whole.rise
        limit = h.pool_memory_limit(pool_stats, blas=False) + value.nbytes
        assert blocks.rise <= limit, (blocks.rise, limit)
        # saved: the two float copies, less the uint8 result and the blocks in flight that replace them
        assert whole.rise - blocks.rise >= float_copies - value.nbytes - workers * job.JPEG_UNIT_BYTES, \
            (whole.rise, blocks.rise)
