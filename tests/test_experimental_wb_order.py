"""Analytic WB-order tests; backend forwarding is mocked, not live Siril."""
from copy import deepcopy

import numpy as np
import pytest

from starstack import experimental_wb_order as wb
from starstack import raw_color


PATTERNS = [[[0, 1], [3, 2]], [[2, 1], [3, 0]],
            [[1, 0], [2, 3]], [[1, 2], [0, 3]]]


@pytest.fixture
def recipe():
    mixing = np.array([[1., 0., 0.], [.2, .8, 0.], [0., .25, .75]])
    matrix = np.vstack((mixing @ np.linalg.inv(raw_color.XYZ_FROM_LINEAR_SRGB), np.zeros(3)))
    fields = {'num_colors': 3, 'color_desc': 'RGBG', 'raw_pattern': PATTERNS[0],
              'rgb_xyz_matrix': matrix.tolist(), 'camera_whitebalance': [2., 1., 1.5, 0.]}
    metadata = {'available': True, 'white_balance_applied': False, 'color_correction_applied': False,
                'fields': {key: {'value': value} for key, value in fields.items()}}
    return raw_color.build_recipe(metadata, 'a'*64, common_scale=1000.)


@pytest.mark.parametrize('pattern', PATTERNS)
def test_cfa_four_phases_signed_values_and_block_boundary(pattern):
    cfa = np.linspace(-10, 20, 131*7, dtype=np.float32).reshape(131, 7)
    cfa[0, 0] = -0.
    original = cfa.copy()
    cfa.flags.writeable = False
    result = wb.apply_cfa_wb(cfa, pattern, [2., 1., .5])
    expected = np.empty_like(cfa)
    slots = [2., 1., .5, 1.]
    for y in range(cfa.shape[0]):
        for x in range(cfa.shape[1]):
            expected[y, x] = np.float64(cfa[y, x])*slots[pattern[y % 2][x % 2]]
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(cfa.view(np.uint32), original.view(np.uint32))
    assert result.dtype == np.float32 and not np.shares_memory(result, cfa)
    assert result.min() < 0 and result.max() > 1


def test_rgb_wb_and_diagnostic_removal_are_not_in_place():
    rgb = np.array([[[-2., -0., 8.], [4., 5., 6.]]], np.float32)
    original = rgb.copy()
    balanced = wb.apply_rgb_wb(rgb, [2., 1., .5])
    np.testing.assert_array_equal(balanced, [[[-4., -0., 4.], [8., 5., 3.]]])
    restored = wb.remove_rgb_wb(balanced, [2., 1., .5])
    np.testing.assert_array_equal(restored.view(np.uint32), original.view(np.uint32))
    np.testing.assert_array_equal(rgb, original)
    assert not np.shares_memory(balanced, rgb) and not np.shares_memory(restored, balanced)


def test_known_matrix_on_already_balanced_data_applies_wb_zero_more_times(recipe):
    # Balanced camera [.1,.26,.6] maps analytically to linear [.1,.3,.7].
    balanced = np.array([[[100., 260., 600.], [250., 250., 250.]]], np.float32)
    before, recipe_before = balanced.copy(), deepcopy(recipe)
    result = wb.color_from_balanced(balanced, recipe)
    np.testing.assert_allclose(result, [[[.1, .3, .7], [.25, .25, .25]]], atol=3e-8)
    twice = raw_color.apply_color(balanced, recipe)
    assert not np.allclose(result, twice)
    np.testing.assert_array_equal(balanced, before)
    assert recipe == recipe_before


def test_color_uses_accepted_stored_coefficients_not_reconstructed_matrix(recipe):
    recipe = deepcopy(recipe)
    # Allowed tiny deviation from an analytically zero entry. Large input makes
    # stored-vs-rebuilt behavior distinguishable after float32 rounding.
    recipe['matrix']['values'][0][1] = 1e-14
    raw_color.validate_recipe(recipe)
    result = wb.color_from_balanced(np.array([[[0., 1e15, 0.]]], np.float32), recipe)
    assert result[0, 0, 0] == pytest.approx(.01, rel=1e-6)


def test_once_wb_color_agrees_with_unbalanced_color_reference_at_float32_precision(recipe):
    rng = np.random.default_rng(519)
    unbalanced = rng.uniform(-400, 8000, (137, 9, 3)).astype(np.float32)
    balanced = wb.apply_rgb_wb(unbalanced, recipe['white_balance']['gains_rgb'])
    result = wb.color_from_balanced(balanced, recipe)
    expected = raw_color.apply_color(unbalanced, recipe)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    assert result.min() < 0 and result.max() > 1


@pytest.mark.parametrize('gains', [[True, 1., 2.], [1., '1', 2.], [1., 2.], [1., 2., 3., 4.],
                                  [1., 0., 2.], [1., -1., 2.], [1., np.nan, 2.], [1., np.inf, 2.]])
def test_invalid_gains_rejected_by_all_gain_helpers(gains):
    with pytest.raises(ValueError):
        wb.apply_cfa_wb(np.zeros((3, 3), np.float32), PATTERNS[0], gains)
    for function in (wb.apply_rgb_wb, wb.remove_rgb_wb):
        with pytest.raises(ValueError):
            function(np.zeros((2, 2, 3), np.float32), gains)


@pytest.mark.parametrize('pattern', [[[0, True], [3, 2]], [[0., 1.], [3., 2.]],
                                    [[0, 1], [2, 1]], [[0, 1, 3, 2]]])
def test_pattern_does_not_hide_bool_or_guess_phase(pattern):
    with pytest.raises(ValueError):
        wb.apply_cfa_wb(np.zeros((3, 3), np.float32), pattern, [2., 1., .5])


@pytest.mark.parametrize('bad', [np.zeros((2, 2, 3), bool), np.zeros((2, 2, 3), np.float64),
                                np.zeros((2, 2), np.float32), np.zeros((2, 2, 4), np.float32),
                                np.zeros((0, 2, 3), np.float32), np.full((2, 2, 3), np.nan, np.float32)])
def test_invalid_rgb_inputs_are_not_coerced(bad, recipe):
    for function in (wb.apply_rgb_wb, wb.remove_rgb_wb):
        with pytest.raises(ValueError):
            function(bad, [2., 1., .5])
    with pytest.raises(ValueError):
        wb.color_from_balanced(bad, recipe)


def test_overflow_is_value_error_in_each_path(recipe):
    maximum = np.finfo(np.float32).max
    with pytest.raises(ValueError, match='overflow'):
        wb.apply_cfa_wb(np.full((3, 3), maximum, np.float32), PATTERNS[0], [2., 1., 1.])
    with pytest.raises(ValueError, match='overflow'):
        wb.apply_rgb_wb(np.full((1, 1, 3), maximum, np.float32), [2., 1., 1.])
    with pytest.raises(ValueError, match='overflow'):
        wb.remove_rgb_wb(np.full((1, 1, 3), maximum, np.float32), [.5, 1., 1.])
    tiny = deepcopy(recipe)
    tiny['scale']['common_denominator_DN'] = 1e-30  # valid explicit common-scale recipe
    with pytest.raises(ValueError, match='overflow'):
        wb.color_from_balanced(np.full((1, 1, 3), maximum, np.float32), tiny)


def test_invalid_color_recipe_fails_before_color_transform(recipe):
    altered = deepcopy(recipe)
    altered['matrix']['values'][0][0] += .1
    with pytest.raises(ValueError, match='matrix'):
        wb.color_from_balanced(np.ones((2, 2, 3), np.float32), altered)


def test_order_wrapper_preserves_explicit_bounds_and_distinguishes_nonlinear_order(monkeypatch, recipe, tmp_path):
    calls = []

    def fake_nonlinear(cfa, pattern, output_root, **kwargs):
        calls.append((cfa.copy(), kwargs))
        # Deliberately not RCD: nonlinear sentinel exposes incorrect WB order.
        rgb = np.repeat(np.square(cfa)[..., None], 3, axis=2)
        return rgb, {'state': 'succeeded', 'backend': 'test-only nonlinear sentinel'}

    monkeypatch.setattr(wb, 'rcd_fixed_range', fake_nonlinear)
    cfa = np.full((4, 4), 2., np.float32)
    before, before_record = wb.run_wb_order(cfa, PATTERNS[0], tmp_path, recipe,
        order='before', lower_dn=-10., upper_dn=100., engine='test-engine')
    after, after_record = wb.run_wb_order(cfa, PATTERNS[0], tmp_path, recipe,
        order='after', lower_dn=-10., upper_dn=100., engine='test-engine')
    np.testing.assert_array_equal(before[0, 0], [16., 16., 16.])
    np.testing.assert_array_equal(after[0, 0], [8., 4., 6.])
    assert calls[0][0][0, 0] == 4. and calls[1][0][0, 0] == 2.
    assert calls[0][1] == calls[1][1] == {'lower_dn': -10., 'upper_dn': 100., 'engine': 'test-engine'}
    for record in (before_record, after_record):
        assert record['white_balance_applications'] == 1
        assert record['matrix_applied'] is record['common_scale_applied'] is False
        assert record['output_units'] == 'white_balanced_camera_RGB_DN'
        assert record['record_persistence'] == 'returned_to_caller_only'
        assert record['rcd_record']['backend'] == 'test-only nonlinear sentinel'
    np.testing.assert_array_equal(cfa, np.full((4, 4), 2., np.float32))


def test_order_wrapper_rejects_invalid_order_before_backend(monkeypatch, recipe, tmp_path):
    monkeypatch.setattr(wb, 'rcd_fixed_range', lambda *a, **k: pytest.fail('Unexpected backend call'))
    with pytest.raises(ValueError, match='order'):
        wb.run_wb_order(np.zeros((2, 2), np.float32), PATTERNS[0], tmp_path, recipe,
                       order=True, lower_dn=-1., upper_dn=1.)
