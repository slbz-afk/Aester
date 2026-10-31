"""P1c T-5x (b): the starless scope check's 128-row blocks (photographic_job._starless_scope, called by the advice,
the reduction and a separated Milky Way finish, each noting under its own key) and the advice's 64-row residual check
(star_reduction._advice_residuals) on the frame pool (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.5, §6.2 A5 (a)-(f)).
Neither takes a cancel point (none before P1c, none added).

The kept references are the loops at the P1c base (tests/p1c_finishing_reference.py): the scope statistics record
compares with its key order; the residual check returns nothing and must raise exactly what the serial loop raised.
"""
import numpy as np
import pytest

import p1c_finishing_data as fd
import p1c_site_harness as h
from p1c_finishing_reference import _advice_residuals_reference, _starless_scope_reference
from starstack import photographic_job as job, star_reduction as sr

SHAPE = (600, 24)                     # 5 scope blocks, 10 residual blocks
SCOPE_UNITS, RESIDUAL_UNITS = 5, 10
_FILES = {}


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("scope")


def scene(folder, seed, paired):
    """(base path, diffuse path, sky, protect, stack) of a synthetic separated layer pair on disk."""
    key = (seed, paired)
    if key not in _FILES:
        stack = fd.layer_stack(seed, SHAPE, paired=paired)
        base = fd.write_display(folder / f"base-{seed}-{paired}.tif", stack["b"])
        diffuse = fd.write_display(folder / f"diffuse-{seed}-{paired}.tif", stack["d"])
        _FILES[key] = (base, diffuse, stack)
    base, diffuse, stack = _FILES[key]
    return base, diffuse, stack["sky"].copy(), stack["protect"].copy(), stack


def scope_reference(folder, seed, paired=False):
    base, diffuse, sky, protect, _ = scene(folder, seed, paired)
    return _starless_scope_reference(base, diffuse, sky, protect, paired_corridors=paired)


def scope_pooled(folder, seed, paired=False, *, hook=None):
    base, diffuse, sky, protect, stack = scene(folder, seed, paired)
    if hook is not None:
        hook(sky, protect, stack)
    pool_stats = {}
    stats = job._starless_scope(base, diffuse, sky, protect, paired_corridors=paired, pool_stats=pool_stats)
    return stats, pool_stats


def test_the_scope_scene_has_residuals_of_both_signs(folder):
    stats = scope_reference(folder, 0)
    assert stats["residual_pixels"] > 50 and stats["residual_min_code"] < 0 < stats["residual_max_code"]


# ================================================================== (a) scope: reference vs pool, (e) audit

@pytest.mark.parametrize("case", h.params(h.synthetic_cases()))
def test_scope_pool_matches_the_reference(folder, case):
    before = h.aester_threads()
    with fd.forced(case):
        stats, pool_stats = scope_pooled(folder, case.seed)
    h.assert_same(scope_reference(folder, case.seed), stats)
    assert pool_stats["units"] == SCOPE_UNITS and pool_stats["workers"] == min(case.workers, SCOPE_UNITS)
    assert pool_stats["unit_bytes"] == job.scope_unit_bytes(SHAPE[1])
    h.assert_pool_quiet(before)


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_paired_corridors_match_the_reference(folder, workers):
    with fd.forced(h.Case(workers, None, "fresh", workers)):
        stats, _ = scope_pooled(folder, 1, True)
    h.assert_same(scope_reference(folder, 1, True), stats)


def scope_reference_target(seed):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        _FILES.clear()
        return scope_reference(Path(tmp), seed)


def scope_pooled_target(seed):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp, fd.forced(h.Case(8, None, "reuse", 0)):
        _FILES.clear()
        return scope_pooled(Path(tmp), seed)[0]


def test_scope_reruns_under_two_hash_seeds_give_one_digest(folder):
    digest = h.assert_hashseed_stable("test_p1c_site_starless_scope:scope_reference_target",
                                      "test_p1c_site_starless_scope:scope_pooled_target", 2)
    assert digest == h.canonical_digest(scope_reference(folder, 2))


# ================================================================== (c) scope failures

def _forbidden(k):
    def hook(sky, protect, stack):
        rows, cols = np.nonzero(np.any(stack["r"][k * 128:(k + 1) * 128] != 0, axis=-1))
        protect[k * 128 + rows[0], cols[0]] = True               # a residual in a protected pixel of block k
    return hook


@pytest.mark.parametrize("k", h.failure_positions(SCOPE_UNITS))
@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_a_failing_scope_block_ends_like_the_serial_loop(folder, k, workers):
    before = h.aester_threads()

    def run(case):
        def body(delivered):
            with fd.forced(case, delivered=delivered):
                return scope_pooled(folder, 3, hook=_forbidden(k))
        return h.outcome(body)

    serial = run(h.Case(1, None, "reuse", None))
    parallel = run(h.Case(workers, None, "reuse", k))
    assert serial.error_type is ValueError and serial.message == "去星背景超出已验证的纯天空分层范围"
    assert serial.delivered == list(range(k))
    h.assert_like_serial(serial, parallel)
    h.assert_pool_quiet(before)


# ================================================================== advice residual check: (a), (c)

def residual_inputs(seed, paired=True):
    stack = fd.layer_stack(seed, SHAPE, paired=paired)
    return stack["r"], stack["b"], stack["d"], stack["s"], stack["t"], stack["shape"], paired


@pytest.mark.parametrize("case", h.params(h.synthetic_cases(seeds=(0, 1))))
def test_the_residual_check_passes_like_the_reference(case):
    args = residual_inputs(case.seed)
    assert _advice_residuals_reference(*args) is None
    with fd.notes() as seen, fd.forced(case):
        assert sr._advice_residuals(*args) is None
    entry = fd.pool_entries(seen, "starless_scope.advice_residuals")
    assert len(entry) == 1 and entry[0]["units"] == RESIDUAL_UNITS
    assert entry[0]["workers"] == min(case.workers, RESIDUAL_UNITS)


@pytest.mark.parametrize("which", ["star", "transient"])
@pytest.mark.parametrize("k", h.failure_positions(RESIDUAL_UNITS))
@pytest.mark.parametrize("workers", [2, 8, 16])
def test_a_failing_residual_block_raises_the_serial_error(which, k, workers):
    def run(case, function):
        r, b, d, s, t, shape, paired = residual_inputs(7)
        target = r if which == "star" else t
        target[k * 64 + 1, 2, 0] += 3
        if k + 1 < RESIDUAL_UNITS:
            r[(k + 1) * 64 + 1, 2, 2] += 5                       # a later block fails as well: the first one wins
        with fd.forced(case):
            return h.outcome(lambda delivered: function(r, b, d, s, t, shape, paired))

    expected = run(h.Case(1, None, "reuse", None), _advice_residuals_reference)
    serial = run(h.Case(1, None, "reuse", None), sr._advice_residuals)
    parallel = run(h.Case(workers, None, "reuse", k), sr._advice_residuals)
    assert expected.error_type is ValueError and expected.message.startswith(which)
    h.assert_like_serial(expected, serial)
    h.assert_like_serial(expected, parallel)
    h.assert_pool_quiet()


# ================================================================== (a)/(b) real size on the F48 fixture, (f)

@pytest.fixture(scope="module")
def f48_scope():
    f48 = fd.F48()
    shape = (5520, 8280)
    sky, protect = job._mask(f48.sky, shape), job._mask(f48.protect, shape, binary=True)
    base = f48.milkyway / "milkyway_srgb16.tif"
    diffuse = f48.separation.parent / "diffuse_srgb16.tif"
    expected = _starless_scope_reference(base, diffuse, sky, protect)
    return f48, base, diffuse, sky, protect, expected


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_scope_matches_the_reference(f48_scope, case):
    _, base, diffuse, sky, protect, expected = f48_scope
    pool_stats = {}
    with fd.forced(case):
        stats = job._starless_scope(base, diffuse, sky, protect, pool_stats=pool_stats)
    h.assert_same(expected, stats)
    assert stats["residual_pixels"] > 0 and pool_stats["units"] == 44
    assert pool_stats["workers"] == (case.workers or pool_stats["requested"])
    h.assert_pool_quiet()


@pytest.mark.parametrize("case", h.params(h.real_cases()))
def test_real_size_residual_check_passes(f48_scope, case):
    f48 = f48_scope[0]
    readers = [job._read_display(p) for p in (f48.separation.parent / "stars_signed_codes.tif",
                                              f48.milkyway / "milkyway_srgb16.tif",
                                              f48.separation.parent / "diffuse_srgb16.tif")]
    try:
        r, b, d = readers
        with fd.notes() as seen, fd.forced(case):
            sr._advice_residuals(r, b, d, b, None, r.shape, False)
        entry = fd.pool_entries(seen, "starless_scope.advice_residuals")[0]
        assert entry["units"] == 87 and entry["workers"] == (case.workers or entry["requested"])
    finally:
        for array in readers:
            job._close_map(array)


def test_the_scope_pool_peak_stays_within_its_declared_bound(f48_scope):
    _, base, diffuse, sky, protect, expected = f48_scope
    pool_stats = {}
    with fd.forced(h.Case(8, 16, "reuse", 0), head_last=True, head_seconds=.3), h.PeakSampler(trace=False) as peak:
        stats = job._starless_scope(base, diffuse, sky, protect, pool_stats=pool_stats)
    h.assert_same(expected, stats)
    limit = h.pool_memory_limit(pool_stats, blas=False)
    assert pool_stats["workers"] == 8 and pool_stats["unit_bytes"] == job.SCOPE_UNIT_BYTES
    if peak.rise is not None:
        assert peak.rise <= limit, (peak.rise, limit)
    print(f"starless_scope: rise {peak.rise} limit {limit}")


def test_each_call_site_notes_under_its_own_key(tmp_path):
    """advice, reduction (and a separated Milky Way finish) note frame_pool.starless_scope.<call site> (prereg §3.5)."""
    stack = fd.layer_stack(8, SHAPE, paired=False)
    result = fd.make_separation(tmp_path / "sep", stack)
    with fd.notes() as seen, fd.forced(h.Case(4, None, "reuse", None)):
        sr.suggest_star_reduction(result, tmp_path / "advice")
        sr.reduce_stars(result, tmp_path / "reduction", background_strength=.5, bright_radius=2)
    for key in ("starless_scope.advice", "starless_scope.reduction", "starless_scope.advice_residuals",
                "jpeg_rows.reduction", "star_reduction"):
        entries = fd.pool_entries(seen, key)
        assert len(entries) == 1 and entries[0]["workers"] == min(4, entries[0]["units"]), (key, entries)
