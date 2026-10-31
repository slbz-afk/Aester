"""Independent analytic counterexamples for background and fixed display.

Truth is constructed from specified planes and scalar formulae; the product's
fitter, renderer or model output is not used to construct expected corrections.
Lifecycle and public job admission are tested separately by the integrator.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from starstack import background as bg


COEFFICIENTS = np.array([[.18, .027, -.014], [.25, -.018, .031], [.34, .011, .019]], np.float64)


def plane(coefficients=COEFFICIENTS, height=128, width=160):
    # Pixel-centre coordinates are fixed from image endpoints, not fit samples.
    x = np.arange(width, dtype=np.float64)*2/(width-1)-1
    y = np.arange(height, dtype=np.float64)*2/(height-1)-1
    return np.stack([a+b*x[None, :]+c*y[:, None] for a, b, c in coefficients], axis=2).astype(np.float32)


def model(data, coefficients=COEFFICIENTS):
    return {'version': bg.VERSION, 'shape_hw': list(data.shape[:2]),
            'fit': {'coefficients': np.asarray(coefficients).tolist()}}


def identical_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def scalar_correction(data, coefficients, mode, strength, weight=None):
    out = data.copy()
    h, w = data.shape[:2]
    neutral = sum(float(row[0]) for row in coefficients)/3
    for iy in range(h):
        for ix in range(w):
            x, y = -1+2*ix/(w-1), -1+2*iy/(h-1)
            for c, (a, b, d) in enumerate(coefficients):
                change = (a-neutral if mode != 'gradient' else 0)
                if mode != 'neutralize':
                    change += b*x+d*y
                change *= strength*(1 if weight is None else float(weight[iy, ix]))
                if change != 0:
                    out[iy, ix, c] = np.float32(float(data[iy, ix, c])-change)
    return out


def display_recipe(**updates):
    recipe = {'version': bg.VERSION, 'black': 0., 'scale': 1.,
              'gains_rgb': [1., 1., 1.], 'asinh_strength': 12.,
              'method': 'maxRGB_asinh_exponential_shoulder_then_sRGB_encoding',
              'color_calibrated': False}
    recipe.update(updates)
    return recipe


def scalar_display(data, recipe):
    output = np.empty(data.shape, np.uint8)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            samples = [max((float(data[y, x, c])-recipe['black'])/recipe['scale'], 0)*recipe['gains_rgb'][c]
                       for c in range(3)]
            peak = max(samples)
            value = -math.expm1(-math.asinh(recipe['asinh_strength']*peak)/math.asinh(recipe['asinh_strength']))
            for c, channel in enumerate(samples):
                linear = channel*value/peak if peak else 0.
                encoded = 12.92*linear if linear <= .0031308 else 1.055*linear**(1/2.4)-.055
                output[y, x, c] = round(min(max(encoded, 0), 1)*255)
    return output


def test_full_rectangular_cells_recover_known_three_colour_plane():
    data = plane()
    before = data.copy()
    fitted = bg.fit_background(data, np.ones(data.shape[:2], bool))
    np.testing.assert_allclose(fitted['fit']['coefficients'], COEFFICIENTS, atol=3e-8, rtol=0)
    assert len(fitted['samples_xy']) == len(fitted['samples_rgb']) == len(fitted['cells']) == 256
    assert identical_bits(data, before)


def test_constant_coloured_background_has_no_spatial_gradient():
    coefficients = np.array([[.12, 0, 0], [.24, 0, 0], [.47, 0, 0]])
    data = plane(coefficients)
    fitted = bg.fit_background(data, np.ones(data.shape[:2], bool))
    np.testing.assert_allclose(np.asarray(fitted['fit']['coefficients'])[:, 1:], 0, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(bg.apply_background(data, fitted, mode='gradient'), data)


def test_partial_grid_fragments_cannot_be_misrepresented_by_cell_centres():
    data = plane(height=128, width=128)
    mask = np.zeros((128, 128), bool)
    # Every 8x8 cell contains 32 selected pixels, but no complete rectangle.
    # Old selected-pixel-centroid fitting would silently accept all 256 cells.
    for row in range(0, 128, 8):
        mask[row:row+4] = True
    with pytest.raises(ValueError):
        bg.fit_background(data, mask)


def test_three_corner_sample_median_is_not_a_plane_value_at_median_coordinate():
    # The analytic counterexample motivating complete symmetric windows:
    # z=x+y on (0,0),(0,2),(2,0) has median(z)=2, but median(x)+median(y)=0.
    xy = np.array([[0., 0.], [0., 2.], [2., 0.]])
    assert np.median(xy.sum(axis=1)) != np.median(xy, axis=0).sum()
    data = plane(height=128, width=128)
    mask = np.zeros((128, 128), bool)
    for y in range(0, 128, 8):
        for x in range(0, 128, 8):
            yy, xx = np.indices((8, 8))
            mask[y:y+8, x:x+8] = xx+yy < 9
    with pytest.raises(ValueError):
        bg.fit_background(data, mask)


def test_one_hole_or_one_non_pure_sky_weight_drops_whole_sample_cell():
    data = plane(height=128, width=128)
    mask = np.ones((128, 128), bool)
    weight = np.ones((128, 128), np.float32)
    mask[1, 1] = False
    weight[1, 9] = np.nextafter(np.float32(1), np.float32(0))
    fitted = bg.fit_background(data, mask, weight)
    cells = {tuple(c['grid_xy']) for c in fitted['cells']}
    assert (0, 0) not in cells and (1, 0) not in cells and len(cells) == 254
    np.testing.assert_allclose(fitted['fit']['coefficients'], COEFFICIENTS, atol=3e-8, rtol=0)


def test_feathered_sky_is_not_used_to_estimate_unmixed_sky_plane():
    data = plane()
    with pytest.raises(ValueError):
        bg.fit_background(data, np.ones(data.shape[:2], bool), np.full(data.shape[:2], .999, np.float32))


def test_large_cells_use_actual_strided_pixel_centres():
    data = plane(height=1031, width=1043)
    fitted = bg.fit_background(data, np.ones(data.shape[:2], bool))
    np.testing.assert_allclose(fitted['fit']['coefficients'], COEFFICIENTS, atol=3e-8, rtol=0)
    for point, cell in zip(fitted['samples_xy'], fitted['cells']):
        ix, iy = cell['grid_xy']
        x0, x1 = ix*1043//16, (ix+1)*1043//16
        y0, y1 = iy*1031//16, (iy+1)*1031//16
        sx, sy = math.ceil((x1-x0)/64), math.ceil((y1-y0)/64)
        sample_x, sample_y = list(range(x0, x1, sx)), list(range(y0, y1, sy))
        expected = [(sample_x[0]+sample_x[-1])/1042-1,
                    (sample_y[0]+sample_y[-1])/1030-1]
        np.testing.assert_allclose(point, expected, atol=3e-16, rtol=0)
        assert cell['pixels_sampled'] == len(sample_x)*len(sample_y)


@pytest.mark.parametrize('layout', ['empty', 'eleven_cells', 'one_corner', 'one_strip', 'diagonal'])
def test_insufficient_spatial_information_fails(layout):
    data = plane(height=128, width=128)
    mask = np.zeros((128, 128), bool)
    if layout == 'eleven_cells':
        for k in range(11): mask[0:8, k*8:(k+1)*8] = True
    elif layout == 'one_corner': mask[:32, :32] = True
    elif layout == 'one_strip': mask[:8] = True
    elif layout == 'diagonal':
        for k in range(16): mask[k*8:(k+1)*8, k*8:(k+1)*8] = True
    with pytest.raises(ValueError):
        bg.fit_background(data, mask)


def test_protected_extended_nebula_is_not_fitted_as_background():
    base = plane(height=128, width=128)
    data = base.copy()
    data[32:96, 32:96] += np.array([.15, .025, .075], np.float32)
    mask = np.ones((128, 128), bool)
    mask[32:96, 32:96] = False
    fitted = bg.fit_background(data, mask)
    np.testing.assert_allclose(fitted['fit']['coefficients'], COEFFICIENTS, atol=3e-8, rtol=0)
    expected = scalar_correction(data, COEFFICIENTS, 'gradient', 1)
    np.testing.assert_allclose(bg.apply_background(data, fitted, mode='gradient'), expected, atol=6e-8, rtol=0)
    # Protected emission remains additive signal, not a target for flattening.
    np.testing.assert_allclose((expected-base)[64, 64]-(expected-base)[31, 64],
                               (data-base)[64, 64]-(data-base)[31, 64]
                               - COEFFICIENTS[:, 2]*(2*(64-31)/127), atol=6e-8, rtol=0)


@pytest.mark.parametrize('mode', ['gradient', 'neutralize', 'both'])
@pytest.mark.parametrize('strength', [0., .3, 1.])
def test_correction_matches_independent_pixelwise_algebra(mode, strength):
    rng = np.random.default_rng(731)
    data = rng.uniform(-.2, 2., (131, 17, 3)).astype(np.float32)
    weight = rng.uniform(0, 1, data.shape[:2]).astype(np.float32)
    weight[0], weight[-1] = 0, 1
    before = data.copy()
    actual = bg.apply_background(data, model(data), mode=mode, strength=strength, weight=weight)
    expected = scalar_correction(data, COEFFICIENTS, mode, strength, weight)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1.2e-7)
    assert identical_bits(data, before)
    assert not np.shares_memory(data, actual)
    assert actual.min() < 0 and actual.max() > 1


def test_modes_separate_spatial_slope_from_constant_channel_difference():
    data = plane(height=31, width=41)
    m = model(data)
    gradient = bg.apply_background(data, m, mode='gradient')
    neutral = bg.apply_background(data, m, mode='neutralize')
    both = bg.apply_background(data, m, mode='both')
    centre = COEFFICIENTS[:, 0]
    np.testing.assert_allclose(gradient, np.broadcast_to(centre, data.shape), atol=3e-8, rtol=0)
    np.testing.assert_allclose(both, centre.mean(), atol=3e-8, rtol=0)
    np.testing.assert_allclose(neutral[-1, -1]-neutral[0, 0],
                               2*(COEFFICIENTS[:, 1]+COEFFICIENTS[:, 2]), atol=6e-8, rtol=0)


@pytest.mark.parametrize('mode', ['gradient', 'neutralize', 'both'])
def test_zero_strength_and_zero_change_preserve_negative_zero(mode):
    data = np.resize(np.array([-0., 0., -.01, 1.2, 3.], np.float32), (7, 9, 3))
    assert identical_bits(bg.apply_background(data, model(data), mode=mode, strength=0), data)
    zeros = np.array([[.2, 0, 0]]*3)
    assert identical_bits(bg.apply_background(data, model(data, zeros), mode=mode), data)


def test_each_unmodified_channel_keeps_its_signed_zero():
    data = np.full((5, 7, 3), -0., np.float32)
    coefficients = np.array([[0, .1, 0], [0, 0, 0], [0, -.1, 0]], np.float64)
    actual = bg.apply_background(data, model(data, coefficients), mode='gradient')
    assert identical_bits(actual[:, :, 1], data[:, :, 1])
    assert identical_bits(actual[:, 3], data[:, 3])  # coordinate x=0


def test_weight_zero_preserves_all_ground_bits_and_top_down_landmark():
    data = np.full((131, 17, 3), -0., np.float32)
    data[0, 0] = [-.05, .7, 1.5]
    weight = np.zeros(data.shape[:2], np.float32)
    weight[-1] = 1
    actual = bg.apply_background(data, model(data), weight=weight)
    assert identical_bits(actual[:-1], data[:-1])
    assert not identical_bits(actual[-1], data[-1])


def test_readonly_input_and_model_are_not_modified():
    data = plane(height=17, width=19)
    data.flags.writeable = False
    m = model(data); before = deepcopy(m)
    result = bg.apply_background(data, m)
    assert m == before and result.flags.writeable


@pytest.mark.parametrize('change', [
    {'version': 'other'}, {'shape_hw': [6, 8]},
    {'fit': {'coefficients': [[0, 0, 0]]}},
    {'fit': {'coefficients': [[0, 0, 0], [0, float('nan'), 0], [0, 0, 0]]}},
    {'fit': {'coefficients': [[0, 0, 0], [0, float('inf'), 0], [0, 0, 0]]}},
])
def test_wrong_model_cannot_silently_change_image(change):
    data = plane(height=7, width=9)
    m = model(data); m.update(change)
    with pytest.raises(ValueError): bg.apply_background(data, m)


@pytest.mark.parametrize('value', [-.01, 1.01, True, '1', float('nan'), float('inf')])
def test_invalid_strength_fails_before_modifying_source(value):
    data = plane(height=7, width=9); before = data.copy()
    with pytest.raises(ValueError): bg.apply_background(data, model(data), strength=value)
    assert identical_bits(data, before)


@pytest.mark.parametrize('weight', [np.zeros((7, 9), bool), np.zeros((6, 9), np.float32),
                                  np.full((7, 9), -1., np.float32), np.full((7, 9), 1.01, np.float32),
                                  np.full((7, 9), np.nan, np.float32)])
def test_invalid_weight_fails(weight):
    data = plane(height=7, width=9)
    with pytest.raises(ValueError): bg.apply_background(data, model(data), weight=weight)


def test_overflow_is_explicit_and_original_not_changed():
    data = np.full((7, 9, 3), 3e38, np.float32); before = data.copy()
    coefficients = np.array([[0, 1e39, 0]]*3)
    with pytest.raises(ValueError): bg.apply_background(data, model(data, coefficients))
    assert identical_bits(data, before)


@pytest.mark.parametrize('invalid', [
    np.zeros((7, 9, 3), np.float64), np.zeros((7, 9, 3), np.uint16),
    np.zeros((7, 9), np.float32), np.zeros((7, 9, 4), np.float32),
    np.full((7, 9, 3), np.nan, np.float32), np.zeros((1, 9, 3), np.float32),
])
def test_wrong_image_layout_and_nonfinite_values_are_rejected(invalid):
    with pytest.raises(ValueError): bg.display_parameters(invalid)
    with pytest.raises(ValueError): bg.render_display(invalid, display_recipe())


def test_zero_values_are_valid_user_selected_background_not_automatic_missing_data():
    data = np.zeros((128, 128, 3), np.float32)
    fitted = bg.fit_background(data, np.ones((128, 128), bool))
    np.testing.assert_array_equal(fitted['fit']['coefficients'], np.zeros((3, 3)))
    assert len(fitted['cells']) == 256


def test_display_matches_scalar_maxrgb_shoulder_and_single_srgb_encoding():
    rng = np.random.default_rng(302)
    data = rng.uniform(-.2, 8., (11, 13, 3)).astype(np.float32)
    data[0, 0] = [-1, 0, .00001]
    recipe = display_recipe(black=.03, scale=.4, gains_rgb=[1.1, .9, .7])
    actual = bg.render_display(data, recipe, max_side=1600)
    np.testing.assert_array_equal(actual, scalar_display(data, recipe))


def test_display_uses_top_down_rows_without_extra_flip():
    data = np.full((7, 9, 3), .02, np.float32)
    data[0, 0] = [4., 2., 1.]
    actual = bg.render_display(data, display_recipe())
    assert actual[0, 0].max() > actual[-1, 0].max()
    np.testing.assert_array_equal(actual, scalar_display(data, display_recipe()))


def test_negative_linear_values_are_averaged_before_preview_clipping():
    data = np.empty((2, 4, 3), np.float32)
    data[:, 0::2] = -1
    data[:, 1::2] = 1
    actual = bg.render_display(data, display_recipe(), max_side=2)
    np.testing.assert_array_equal(actual, np.zeros((2, 2, 3), np.uint8))


def test_area_average_precedes_nonlinear_tone_curve():
    data = np.empty((2, 4, 3), np.float32)
    data[:, 0::2] = 0
    data[:, 1::2] = 8
    expected = scalar_display(np.full((2, 2, 3), 4., np.float32), display_recipe())
    np.testing.assert_array_equal(bg.render_display(data, display_recipe(), max_side=2), expected)


def test_frozen_display_does_not_reestimate_percentiles_per_candidate():
    data = plane(height=7, width=9)
    recipe = bg.display_parameters(data)
    before = deepcopy(recipe)
    brighter = data+.2
    actual = bg.render_display(brighter, recipe)
    np.testing.assert_array_equal(actual, scalar_display(brighter, recipe))
    assert recipe == before
    assert not np.array_equal(actual, bg.render_display(brighter, bg.display_parameters(brighter)))


def test_zero_and_extreme_hdr_display_is_finite_and_bounded_without_modifying_input():
    data = np.resize(np.array([-3e38, -0., 0., .01, 3e38], np.float32), (7, 9, 3))
    before = data.copy()
    actual = bg.render_display(data, display_recipe(scale=1e-30))
    assert actual.dtype == np.uint8 and np.isfinite(actual).all()
    assert identical_bits(data, before)


@pytest.mark.parametrize('field,value', [
    ('black', True), ('black', float('nan')), ('scale', 0), ('scale', -1),
    ('asinh_strength', 0), ('asinh_strength', 101), ('color_calibrated', True),
    ('version', 'other'), ('method', 'sRGB_Y'), ('gains_rgb', [1, 1]),
    ('gains_rgb', [1, True, 1]), ('gains_rgb', [.249, 1, 1]),
    ('gains_rgb', [1, 4.01, 1]), ('gains_rgb', [1, float('inf'), 1]),
])
def test_display_recipe_refuses_unsupported_parameters(field, value):
    data = plane(height=7, width=9)
    with pytest.raises(ValueError): bg.render_display(data, display_recipe(**{field: value}))


def test_plane_fitter_recovers_orthogonal_noise_free_of_slope_bias():
    from starstack.background_plane import fit_plane
    # On a symmetric Cartesian grid, xy is orthogonal to 1,x,y. Its noise
    # component must not bias intercept or either slope in least squares.
    coords = np.array([(x, y) for x in [-1., -.5, 0., .5, 1.] for y in [-1., -.5, 0., .5, 1.]])
    truth = np.column_stack([a+b*coords[:, 0]+c*coords[:, 1] for a, b, c in COEFFICIENTS])
    truth += (coords[:, 0]*coords[:, 1])[:, None]*np.array([.01, -.03, .05])
    fitted = fit_plane(coords, truth)
    np.testing.assert_allclose(fitted['coefficients'], COEFFICIENTS, atol=5e-15, rtol=0)
    # E[x^2]=E[y^2]=1/2 on this grid, hence RMS(xy)=1/2.
    np.testing.assert_allclose(fitted['residual_rms_rgb'], [.005, .015, .025], atol=5e-16, rtol=0)
    assert fitted['sample_count'] == 25
    np.testing.assert_allclose(fitted['condition_number'], math.sqrt(2), atol=1e-14, rtol=0)


def test_plane_fitter_rejects_a_collinear_exposure_of_xy_domain():
    from starstack.background_plane import fit_plane
    x = np.linspace(-1, 1, 20)
    with pytest.raises(ValueError): fit_plane(np.column_stack((x, x)), np.ones((20, 3)))


@pytest.mark.parametrize('condition,accepted', [(99., True), (101., False)])
def test_plane_fitter_condition_limit_has_analytic_counterexample(condition, accepted):
    from starstack.background_plane import fit_plane
    # Repeated corners give mutually orthogonal 1,x,y columns; singular values
    # are sqrt(N) times 1,1/condition,1, so the exact condition is known.
    coords = np.array([(x, y) for _ in range(3) for x in [-1/condition, 1/condition] for y in [-1., 1.]])
    rgb = np.column_stack([a+b*coords[:, 0]+c*coords[:, 1] for a, b, c in COEFFICIENTS])
    before_xy, before_rgb = coords.copy(), rgb.copy()
    if accepted:
        fitted = fit_plane(coords, rgb)
        np.testing.assert_allclose(fitted['coefficients'], COEFFICIENTS, atol=1e-13, rtol=0)
        # This is an SVD-derived diagnostic, not a coefficient or acceptance
        # boundary. At condition ~100 the small singular value amplifies
        # float64 roundoff; allow 1e-12 relative diagnostic error. The product's
        # hard limit stays 100 and the independent 101 case must still fail.
        np.testing.assert_allclose(fitted['condition_number'], condition, atol=0, rtol=1e-12)
    else:
        with pytest.raises(ValueError): fit_plane(coords, rgb)
    assert identical_bits(coords, before_xy) and identical_bits(rgb, before_rgb)
