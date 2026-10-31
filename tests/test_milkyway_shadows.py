"""Independent causal checks for softer new darkening, not aesthetic scores."""
import numpy as np
import pytest

from starstack import milkyway as mw


DOMAIN = 'display-sRGB'


def inputs(gray, dtype=np.float32):
    rgb = np.repeat(np.asarray(gray, np.float32)[..., None], 3, axis=-1)
    if dtype == np.uint16:
        rgb = np.rint(rgb * 65535).astype(np.uint16)
    return (rgb, np.ones(rgb.shape[:2], np.float32),
            np.zeros(rgb.shape[:2], bool), np.ones(rgb.shape[:2], np.float32))


def run(arrays, recipe, rows=31):
    prepared = mw.prepare(*arrays, recipe, domain=DOMAIN)
    output = np.empty_like(arrays[0])
    mw.render(*arrays, prepared, output, domain=DOMAIN, row_size=rows)
    return prepared, output


@pytest.mark.parametrize('dtype', [np.float32, np.uint16])
def test_dark_plateaus_remain_dark_ordered_and_less_deepened(dtype):
    # Wide interiors have no gradients or compact stars: their contrast
    # response has a known sign and the majority plateau fixes the pivot.
    gray = np.full((128, 512), .25, np.float32)
    gray[:, :64] = .05
    gray[:, 64:128] = .10
    gray[:, 448:] = .35
    arrays = inputs(gray, dtype)
    original = arrays[0].copy()
    recipe = mw.Recipe(brightness=0, contrast=.2, structure=0, color=0, strength=1.,
                       work_long_edge=512)
    prepared, output = run(arrays, recipe)
    scale = 65535 if dtype == np.uint16 else 1
    centers = [32, 96, 256, 480]
    before = original[64, centers, 0].astype(np.float64) / scale
    after = output[64, centers, 0].astype(np.float64) / scale
    raw_contrast = .2 * (before - prepared['pivot'])
    change = after - before

    assert 0 < after[0] < before[0] < after[1] < before[1] < after[2] < after[3]
    assert np.all(change[:2] > raw_contrast[:2])
    assert np.all(change[:2] < 0)
    assert after[2] == before[2]  # The pivot does not receive a gray floor.
    # Positive light must keep its original analytic response.
    assert abs(change[3] - raw_contrast[3]) < (2/65535 if dtype == np.uint16 else 2e-7)
    np.testing.assert_array_equal(arrays[0], original)


def test_zero_crossing_has_no_step_and_tends_to_same_slope_both_sides():
    recipe = mw.Recipe(brightness=0, contrast=1, structure=0, color=0, strength=1.,
                       work_long_edge=512)
    ratios = []
    for eps in [.01, .001, .0001]:
        gray = np.full((64, 512), .25, np.float32)
        gray[:, :96] = .25 - eps
        gray[:, 416:] = .25 + eps
        prepared, _ = run(inputs(gray), recipe)
        negative = float(prepared['delta_rgb'][32, 48, 0])
        positive = float(prepared['delta_rgb'][32, 464, 0])
        assert -eps <= negative < 0 < positive
        assert abs(positive - eps) < 2e-7
        assert abs(float(prepared['delta_rgb'][32, 256, 0])) < 2e-7
        ratios.append(-negative / positive)
    assert 0 < ratios[0] < ratios[1] < ratios[2] <= 1
    assert ratios[-1] > .998  # No half-axis step/slope jump near zero.


@pytest.mark.parametrize('dtype', [np.float32, np.uint16])
def test_exact_black_field_stays_black_with_finite_prepared_maps(dtype):
    arrays = inputs(np.zeros((80, 112), np.float32), dtype)
    prepared, output = run(arrays, mw.Recipe(work_long_edge=256))
    assert prepared['pivot'] == 0
    assert np.isfinite(prepared['delta_rgb']).all()
    assert np.isfinite(prepared['safe']).all()
    np.testing.assert_array_equal(output, arrays[0])


@pytest.mark.parametrize('rows', [31, 128])
def test_softened_shadow_render_is_block_stable_and_keeps_paired_residual(rows):
    x = np.linspace(0, 1, 512, dtype=np.float32)
    gray = np.broadcast_to(.075 + .22*x + .012*np.sin(19*x), (257, 512)).copy()
    arrays = list(inputs(gray, np.uint16))
    arrays[1][-14:] = 0
    arrays[1][-17:-14] = .5
    arrays[3][:, :12] = 0
    background = arrays[0]
    source = background.copy()
    source[30:200, 48:52] += np.array([1700, 1200, 900], np.uint16)
    source[30:200, 72:75] -= np.array([220, 450, 120], np.uint16)
    before_source, before_background = source.copy(), background.copy()
    recipe = mw.Recipe(work_long_edge=512)
    prepared = mw.prepare(*arrays, recipe, domain=DOMAIN)
    q8, q = np.empty_like(background), np.empty_like(background)
    for row_size, output in [(8, q8), (rows, q)]:
        mw.render(*arrays, prepared, output, domain=DOMAIN, row_size=row_size)
    np.testing.assert_array_equal(q, q8)
    change = q.astype(np.int32)-background.astype(np.int32)
    assert np.any(change < 0) and np.any(change > 0)
    finished = np.empty_like(source)
    stats = mw.preserve_transient_residual(source, background, q, finished)
    np.testing.assert_array_equal(finished.astype(np.int32)-q.astype(np.int32),
                                  source.astype(np.int32)-background.astype(np.int32))
    excluded = (arrays[1] != 1) | (arrays[3] == 0)
    np.testing.assert_array_equal(finished[excluded], source[excluded])
    np.testing.assert_array_equal(q[excluded], background[excluded])
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(background, before_background)
    assert stats['new_endpoint_channels'] == stats['residual_mismatched_channels'] == 0
