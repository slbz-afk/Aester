"""Discriminating checks for the external LibRaw comparison's WB scale gate."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparator():
    pytest.importorskip('rawpy')
    path = Path(__file__).resolve().parents[1]/'scripts/validate_libraw_color_reference.py'
    spec = importlib.util.spec_from_file_location('libraw_reference_review', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('gains', [[2., 1., 1.5], [1., 1., 1.], [1., 1., 2.]])
def test_green_minimum_or_tie_records_matching_scale(comparator, gains):
    contract = comparator.matching_wb_scale(gains)
    assert contract['user_wb'] == [*gains, 1.]
    assert contract['product_normalization'] == 'G1_equals_one'
    assert contract['libraw_normalization'] == 'minimum_of_four_pre_mul_slots'
    assert contract['libraw_highlight_mode'] == 'Clip'
    assert contract['minimum_coefficient'] == contract['green_coefficient'] == 1.
    assert contract['precondition_satisfied'] is True
    assert contract['libraw_to_product_common_scale'] == 1.
    assert '/0.22.1/' in contract['source']


@pytest.mark.parametrize('gains', [[.5, 1., 2.], [2., 1., .25]])
def test_non_green_minimum_is_a_global_scale_difference(comparator, gains):
    # An independent unsaturated camera vector makes the two conventions
    # observably different before either side's color matrix or quantization.
    signal = np.array([.03, .02, .01])
    green_normalized = signal * np.array(gains)
    clip_normalized = signal * np.array(gains) / min(gains)
    np.testing.assert_allclose(clip_normalized / green_normalized, 1/min(gains))
    assert np.max(clip_normalized) < 1.
    assert not np.allclose(clip_normalized, green_normalized)
    with pytest.raises(ValueError, match='common brightness-scale difference'):
        comparator.matching_wb_scale(gains)


def test_precondition_is_exact_not_an_allclose_to_one(comparator):
    with pytest.raises(ValueError, match='scale mismatch'):
        comparator.matching_wb_scale([np.nextafter(1., 0.), 1., 2.])


def test_non_unit_green_cannot_claim_product_normalization(comparator):
    with pytest.raises(ValueError, match='G1 == 1'):
        comparator.matching_wb_scale([4., 2., 3.])


def test_compare_rejects_before_reference_decode(comparator, monkeypatch):
    fields = {
        'black_level_per_channel': [100] * 4, 'white_level': 1100,
        'num_colors': 3, 'color_desc': 'RGBG', 'raw_pattern': [[0, 1], [3, 2]],
        'camera_whitebalance': [.5, 1., 2., 1.],
        'rgb_xyz_matrix': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]],
    }
    metadata = {'available': True, 'white_balance_applied': False,
                'color_correction_applied': False, 'versions': {},
                'fields': {key: {'value': value} for key, value in fields.items()}}
    monkeypatch.setattr(comparator, 'read_raw_metadata', lambda source: metadata)
    monkeypatch.setattr(comparator.core, '_sha', lambda source: 'a' * 64)

    def forbidden_decode(*args, **kwargs):
        pytest.fail('A mismatched WB scale must fail before reference decoding')

    monkeypatch.setattr(comparator.rawpy, 'imread', forbidden_decode)
    with pytest.raises(ValueError, match='scale mismatch'):
        comparator.compare(Path('unused.dng'), 'synthetic-metadata')
