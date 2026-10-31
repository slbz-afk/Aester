"""Analytic Gaussian truth and failure cases for fixed-reference star metrics."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1]/'scripts/star_quality_metrics.py'
_SPEC = importlib.util.spec_from_file_location('star_quality_metrics', _PATH)
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


def gaussian(shape=(96, 112), center=(54.3, 47.7), sigmas=(1.4, 2.0), amplitude=100.):
    yy, xx = np.indices(shape)
    return amplitude*np.exp(-.5*((xx-center[0])**2/sigmas[0]**2+(yy-center[1])**2/sigmas[1]**2))


def select(reference, **kwargs):
    return metrics.select_star_rois(reference, reference_label='analytic fixed G',
        saturation_mask=np.zeros(reference.shape, bool), aperture_radius=9.,
        annulus_inner=11., annulus_outer=14., min_separation=30., **kwargs)


def rgb_from(signal, backgrounds=(13., 21., 7.), colors=(1.7, 1., .6)):
    return signal[..., None]*np.array(colors)+np.array(backgrounds)


def test_elliptical_gaussian_flux_centroid_moments_and_color_truth():
    signal = gaussian()
    selected = select(signal+21, origin_xy=(100, 200))
    assert len(selected['stars']) == 1
    report = metrics.measure_star_rois(rgb_from(signal), selected)
    star = report['stars'][0]
    expected_flux = 2*np.pi*100*1.4*2.
    np.testing.assert_allclose(star['flux_rgb'], expected_flux*np.array([1.7, 1., .6]), rtol=8e-5)
    moment = star['moments_G']
    np.testing.assert_allclose(moment['centroid_xy'], [154.3, 247.7], atol=1e-4)
    assert moment['sigma_major_px'] == pytest.approx(2., abs=3e-4)
    assert moment['sigma_minor_px'] == pytest.approx(1.4, abs=1e-4)
    assert moment['ellipticity'] == pytest.approx(.3, abs=1e-4)
    assert moment['is_FWHM'] is False
    assert star['color_ratios'] == pytest.approx({'R_over_G': 1.7, 'B_over_G': .6})
    json.dumps(report, allow_nan=False)


def test_same_fixed_rois_detect_broadening_and_centroid_shift_without_reselection():
    original = gaussian(sigmas=(1.2, 1.2))
    selected = select(original+10)
    before = deepcopy(selected)
    # Keep integrated flux fixed while broadening only x and shifting x by .4.
    broadened = gaussian(center=(54.7, 47.7), sigmas=(2.0, 1.2), amplitude=60.)
    a = metrics.measure_star_rois(rgb_from(original), selected)['stars'][0]
    b = metrics.measure_star_rois(rgb_from(broadened), selected)['stars'][0]
    assert b['flux_rgb'][1] == pytest.approx(a['flux_rgb'][1], rel=8e-5)
    assert b['moments_G']['sigma_major_px']-a['moments_G']['sigma_major_px'] > .79
    assert b['moments_G']['centroid_xy'][0]-a['moments_G']['centroid_xy'][0] == pytest.approx(.4, abs=1e-4)
    assert selected == before


def test_signed_flux_includes_negative_pixels_but_moments_document_positive_weights():
    reference = gaussian()
    selected = select(reference)
    candidate = np.full((*reference.shape, 3), 10.)
    star = selected['stars'][0]
    x0, y0, x1, y1 = star['bounds_xyxy']
    ap = np.array(star['aperture_mask'])
    local_yx = np.argwhere(ap)
    for (y, x), amount in zip(local_yx[:3], [10., -3., 5.]):
        candidate[y0+y, x0+x] += amount
    result = metrics.measure_star_rois(candidate, selected)['stars'][0]
    assert result['flux_rgb'] == [12., 12., 12.]
    assert result['moments_G']['negative_aperture_samples'] == 1
    assert result['moments_G']['weighting'] == 'positive_background_subtracted_G'


def test_reference_rejects_crowded_saturated_and_edge_stars():
    shape = (128, 160)
    reference = (gaussian(shape, (40, 40), (1, 1)) + gaussian(shape, (47, 40), (1, 1))
                 + gaussian(shape, (110, 80), (1, 1)) + gaussian(shape, (5, 100), (1, 1)))
    saturation = np.zeros(shape, bool)
    saturation[80, 110] = True
    selected = metrics.select_star_rois(reference, reference_label='known scene', saturation_mask=saturation)
    assert not selected['stars']
    assert selected['rejected']['crowded'] >= 2
    assert selected['rejected']['near_edge'] >= 1


@pytest.mark.parametrize('kind,reason', [('valid', 'invalid_or_nonfinite_support'),
                                      ('nan', 'invalid_or_nonfinite_support'),
                                      ('saturation', 'saturated_support')])
def test_candidate_exclusion_retains_selected_star(kind, reason):
    signal = gaussian()
    selected = select(signal)
    rgb = rgb_from(signal)
    valid, saturated = np.ones(signal.shape, bool), np.zeros(signal.shape, bool)
    if kind == 'valid':
        valid[48, 54] = False
    elif kind == 'nan':
        rgb[48, 54, 1] = np.nan
    else:
        saturated[48, 54] = True
    report = metrics.measure_star_rois(rgb, selected, valid_mask=valid, saturation_mask=saturated)
    assert len(report['stars']) == 1
    assert report['stars'][0]['status'] == 'excluded'
    assert reason in report['stars'][0]['exclusion_reasons']
    assert report['stars'][0]['color_ratios'] is None


def test_color_excludes_nonpositive_flux_and_unverified_saturation():
    signal = gaussian()
    selected = metrics.select_star_rois(signal, reference_label='no saturation evidence')
    result = metrics.measure_star_rois(rgb_from(signal), selected)['stars'][0]
    assert 'saturation_not_screened' in result['color_exclusion_reasons']
    selected = select(signal)
    result = metrics.measure_star_rois(rgb_from(signal, colors=(-1., 1., .6)), selected)['stars'][0]
    assert result['color_ratios'] is None
    assert 'nonpositive_channel_flux' in result['color_exclusion_reasons']


def test_low_background_snr_keeps_signed_flux_but_excludes_color_ratios():
    signal = gaussian()
    selected = select(signal)
    star = selected['stars'][0]
    x0, y0, x1, y1 = star['bounds_xyxy']
    ap, ann = np.array(star['aperture_mask']), np.array(star['annulus_mask'])
    candidate = np.full((*signal.shape, 3), 20.)
    tile = candidate[y0:y1, x0:x1]
    tile[ap] += .1
    # Symmetric fixed annulus noise has median zero and known nonzero MAD.
    values = np.zeros(int(ann.sum()))
    pairs = len(values)//2
    values[:pairs], values[pairs:2*pairs] = -10., 10.
    tile[ann] += values[:, None]
    result = metrics.measure_star_rois(candidate, selected)['stars'][0]
    np.testing.assert_allclose(result['flux_rgb'], ap.sum()*.1)
    assert result['status'] == 'measured'
    assert result['color_ratios'] is None
    assert result['color_exclusion_reasons'] == ['low_background_only_snr']


def test_no_stars_and_peak_resource_cap_have_explicit_outcomes():
    flat = np.full((80, 90), 10.)
    selected = select(flat)
    assert selected['stars'] == []
    assert metrics.measure_star_rois(rgb_from(flat), selected)['stars'] == []
    reference = gaussian(center=(30., 30.)) + gaussian(center=(80., 60.))
    with pytest.raises(ValueError, match='max_candidate_peaks'):
        select(reference, max_candidate_peaks=1)


def test_reference_masks_are_fixed_even_when_candidate_peak_moves_or_changes_color():
    signal = gaussian()
    reference = signal+20
    selected = select(reference)
    stored = json.dumps(selected, sort_keys=True)
    candidate = rgb_from(signal)
    original = candidate.copy()
    candidate.flags.writeable = False
    report = metrics.measure_star_rois(candidate, selected)
    np.testing.assert_array_equal(candidate, original)
    assert json.dumps(selected, sort_keys=True) == stored
    assert report['selection_id'] == selected['selection_id']
    assert selected == select(reference)


def test_selection_integrity_geometry_and_bad_parameters():
    signal = gaussian()
    selected = select(signal)
    altered = deepcopy(selected)
    altered['stars'][0]['center_xy'][0] += 1
    with pytest.raises(ValueError, match='identity'):
        metrics.measure_star_rois(rgb_from(signal), altered)
    with pytest.raises(ValueError, match='shape'):
        metrics.measure_star_rois(rgb_from(signal)[:-1], selected)
    with pytest.raises(ValueError, match='boolean mask'):
        metrics.select_star_rois(signal, reference_label='G', valid_mask=np.ones(signal.shape))
    with pytest.raises(ValueError, match='aperture_radius'):
        metrics.select_star_rois(signal, reference_label='G', aperture_radius=True)
