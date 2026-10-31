"""Read-only, fixed-aperture star diagnostics for linear image comparisons.

Select once on a declared, dense reference G image, before inspecting candidate
differences. A sparse green CFA must first be represented by the caller in the
same declared pixel coordinates; zeros at red/blue sites are not valid G data.
This module never registers, resamples, edits, ranks image beauty, or writes.

Flux is a signed aperture sum after a per-channel median annulus background.
Sizes are positive-G-weight second moments in a finite aperture, NOT FWHM.
Positive weights bias noisy moments; the background-only SNR proxy excludes
source shot noise and correlations. Isolation only excludes detected peaks.
Neither these metrics nor a visually good crop certify photographic quality.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


SCHEMA_VERSION = 1


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f'{name} must be a finite positive number')
    return float(value)


def _image(array, channels=None):
    value = np.asarray(array)
    expected = 2 if channels is None else 3
    if (value.ndim != expected or value.dtype.kind not in 'iuf'
            or min(value.shape[:2]) < 3 or (channels is not None and value.shape[2] != channels)):
        raise ValueError('Expected a numeric 2D reference or HWC RGB candidate')
    return value


def _mask(value, shape, default, name):
    if value is None:
        return np.full(shape, default, dtype=bool)
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f'{name} must be a boolean mask of the image shape')
    return value


def _background(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    sigma = float(1.4826 * np.median(np.abs(values-median)))
    return median, sigma


def _footprint(cx, cy, aperture, inner, outer, shape):
    h, w = shape
    # Include every pixel center in the complete outer circle; no edge padding.
    x0, x1 = math.floor(cx-outer), math.ceil(cx+outer)+1
    y0, y1 = math.floor(cy-outer), math.ceil(cy+outer)+1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance2 = (xx-cx)**2 + (yy-cy)**2
    ap = distance2 <= aperture**2
    ann = (distance2 >= inner**2) & (distance2 <= outer**2)
    if np.count_nonzero(ap) < 5 or np.count_nonzero(ann) < 12:
        raise ValueError('Aperture/annulus have too few pixel centers')
    return (x0, y0, x1, y1), ap, ann


def _selection_id(selection):
    content = {key: value for key, value in selection.items() if key != 'selection_id'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def select_star_rois(reference_G, *, reference_label, valid_mask=None,
                     saturation_mask=None, aperture_radius=5., annulus_inner=7.,
                     annulus_outer=10., detection_sigma=6., min_separation=22.,
                     max_stars=100, origin_xy=(0, 0), max_candidate_peaks=10000):
    """Select isolated reference peaks and freeze their aperture/annulus masks.

    Coordinates and origin are TOP-DOWN integer pixel centers. The origin maps
    an already cropped reference back to the parent frame, without resampling.
    Candidate RGB images must have the identical crop/shape. The caller supplies
    saturation from decoded source evidence, not from a rendered >1 threshold.
    Without that mask, saturation remains unverified and color ratios are not
    eligible unless the measurement call supplies its own explicit mask.

    Detection uses a global median/MAD floor followed by a local annulus check;
    uneven backgrounds, unresolved companions and cosmic rays require caller
    review. Peaks within min_separation reject one another, not just the weaker
    peak. If capped, stars retain row/column order (not a brightness ranking).
    """
    reference = _image(reference_G)
    if not isinstance(reference_label, str) or not reference_label.strip():
        raise ValueError('Declare the fixed reference source and construction')
    if (not isinstance(origin_xy, (tuple, list)) or len(origin_xy) != 2
            or any(type(v) is not int for v in origin_xy)):
        raise ValueError('origin_xy must contain two integer pixel coordinates')
    aperture = _positive(aperture_radius, 'aperture_radius')
    inner = _positive(annulus_inner, 'annulus_inner')
    outer = _positive(annulus_outer, 'annulus_outer')
    detection = _positive(detection_sigma, 'detection_sigma')
    separation = _positive(min_separation, 'min_separation')
    if not aperture < inner < outer:
        raise ValueError('Require aperture_radius < annulus_inner < annulus_outer')
    for value, name, limit in ((max_stars, 'max_stars', 10000),
                               (max_candidate_peaks, 'max_candidate_peaks', 10000)):
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f'{name} must be an integer in 1..{limit}')
    valid = _mask(valid_mask, reference.shape, True, 'valid_mask') & np.isfinite(reference)
    saturated = _mask(saturation_mask, reference.shape, False, 'saturation_mask')
    usable = valid & ~saturated
    if np.count_nonzero(usable) < 20:
        raise ValueError('Too few finite, valid, non-saturated reference samples')
    background, noise = _background(reference[usable])
    work = np.where(usable, reference, -np.inf)
    peaks = usable & (work > background+detection*noise)
    # Deterministic plateau tie break: retain the first row/column local maximum.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            y0, y1 = max(0, -dy), min(reference.shape[0], reference.shape[0]-dy)
            x0, x1 = max(0, -dx), min(reference.shape[1], reference.shape[1]-dx)
            here = work[y0:y1, x0:x1]
            other = work[y0+dy:y1+dy, x0+dx:x1+dx]
            comparison = here > other if (dy, dx) < (0, 0) else here >= other
            peaks[y0:y1, x0:x1] &= comparison
    count = int(np.count_nonzero(peaks))
    if count > max_candidate_peaks:
        raise ValueError(f'{count} reference peaks exceed max_candidate_peaks; use bounded crops')
    peak_yx = np.argwhere(peaks)
    rejected = {key: 0 for key in ('near_edge', 'invalid_support', 'saturated_support',
                                  'low_local_contrast', 'no_positive_signal', 'crowded')}
    stars = []
    for py, px in peak_yx:
        # Include all detected peaks, even those later rejected for support.
        neighbors = np.sum((peak_yx-np.array([py, px]))**2, axis=1)
        if np.count_nonzero(neighbors < separation**2) > 1:
            rejected['crowded'] += 1
            continue
        footprint = _footprint(float(px), float(py), aperture, inner, outer, reference.shape)
        if footprint is None:
            rejected['near_edge'] += 1
            continue
        bounds, ap, ann = footprint
        x0, y0, x1, y1 = bounds
        support = ap | ann
        if not valid[y0:y1, x0:x1][support].all():
            rejected['invalid_support'] += 1
            continue
        if saturated[y0:y1, x0:x1][support].any():
            rejected['saturated_support'] += 1
            continue
        tile = np.asarray(reference[y0:y1, x0:x1], dtype=np.float64)
        local_bg, local_noise = _background(tile[ann])
        if float(reference[py, px])-local_bg <= detection*max(noise, local_noise):
            rejected['low_local_contrast'] += 1
            continue
        weights = np.maximum(tile-local_bg, 0)*ap
        total = float(weights.sum())
        if total <= 0:
            rejected['no_positive_signal'] += 1
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx, cy = float((xx*weights).sum()/total), float((yy*weights).sum()/total)
        # Recenter once on the reference, then freeze this exact geometry.
        footprint = _footprint(cx, cy, aperture, inner, outer, reference.shape)
        if footprint is None:
            rejected['near_edge'] += 1
            continue
        bounds, ap, ann = footprint
        x0, y0, x1, y1 = bounds
        if not valid[y0:y1, x0:x1][ap | ann].all():
            rejected['invalid_support'] += 1
            continue
        if saturated[y0:y1, x0:x1][ap | ann].any():
            rejected['saturated_support'] += 1
            continue
        stars.append({'star_id': f'star-{len(stars)+1:04d}',
                      'center_local_xy': [cx, cy],
                      'center_xy': [cx+origin_xy[0], cy+origin_xy[1]],
                      'bounds_xyxy': list(bounds), 'peak_local_xy': [int(px), int(py)],
                      'aperture_mask': ap.tolist(), 'annulus_mask': ann.tolist()})
    retained = len(stars)
    result = {'schema_version': SCHEMA_VERSION, 'coordinate_convention': 'TOP-DOWN_pixel_centers',
              'reference_label': reference_label, 'shape_yx': list(reference.shape),
              'origin_xy': list(origin_xy), 'saturation_screened': saturation_mask is not None,
              'parameters': {'aperture_radius': aperture, 'annulus_inner': inner,
                  'annulus_outer': outer, 'detection_sigma': detection,
                  'min_separation': separation, 'max_stars': max_stars},
              'reference_background': background, 'reference_noise_MAD': noise,
              'detected_peaks': count, 'rejected': rejected,
              'truncated_by_max_stars': max(0, retained-max_stars), 'stars': stars[:max_stars],
              'limitations': ['Selection uses only the declared fixed dense reference, not candidate differences',
                  'Isolation only concerns peaks detected in that reference; companions and hot pixels may remain',
                  'Global/local MAD thresholds can omit stars on uneven backgrounds',
                  'No saturation mask means unsaturated status is unverified',
                  'Candidate algorithms must share crop, coordinates, linear units, and color convention']}
    result['selection_id'] = _selection_id(result)
    return result


def measure_star_rois(candidate_rgb, selection, *, valid_mask=None,
                      saturation_mask=None, min_color_snr=5.):
    """Measure a candidate with the frozen reference geometry; never reselect.

    Color ratios are R/G and B/G integrated flux, not peak color. They require
    positive flux in all channels, a declared saturation mask, and sufficient
    background-only SNR (or exactly zero estimated annulus noise). A caller must
    intersect eligible star IDs across candidates when comparing summaries;
    this function reports each failure and never silently drops a selected ROI.
    """
    rgb = _image(candidate_rgb, 3)
    if (not isinstance(selection, dict) or selection.get('schema_version') != SCHEMA_VERSION
            or selection.get('selection_id') != _selection_id(selection)
            or selection.get('coordinate_convention') != 'TOP-DOWN_pixel_centers'
            or selection.get('shape_yx') != list(rgb.shape[:2])):
        raise ValueError('Selection identity, shape, or coordinate convention mismatch')
    threshold = _positive(min_color_snr, 'min_color_snr')
    valid = _mask(valid_mask, rgb.shape[:2], True, 'valid_mask')
    saturated = _mask(saturation_mask, rgb.shape[:2], False, 'saturation_mask')
    screened = selection['saturation_screened'] or saturation_mask is not None
    results = []
    for star in selection['stars']:
        x0, y0, x1, y1 = star['bounds_xyxy']
        ap, ann = np.asarray(star['aperture_mask']), np.asarray(star['annulus_mask'])
        expected = (y1-y0, x1-x0)
        if (not 0 <= x0 < x1 <= rgb.shape[1] or not 0 <= y0 < y1 <= rgb.shape[0]
                or ap.dtype != np.bool_ or ann.dtype != np.bool_ or ap.shape != expected
                or ann.shape != expected or np.any(ap & ann) or not ap.any() or not ann.any()):
            raise ValueError('Invalid frozen aperture/annulus geometry')
        tile = np.asarray(rgb[y0:y1, x0:x1], dtype=np.float64)
        support = ap | ann
        record = {'star_id': star['star_id'], 'center_xy': star['center_xy'],
                  'status': 'excluded', 'exclusion_reasons': [], 'flux_rgb': None,
                  'background_rgb': None, 'background_noise_rgb': None,
                  'background_only_flux_sigma_rgb': None, 'background_only_snr_rgb': None,
                  'moments_G': None, 'color_ratios': None, 'color_exclusion_reasons': []}
        if not valid[y0:y1, x0:x1][support].all() or not np.isfinite(tile[support]).all():
            record['exclusion_reasons'].append('invalid_or_nonfinite_support')
        if saturated[y0:y1, x0:x1][support].any():
            record['exclusion_reasons'].append('saturated_support')
        if record['exclusion_reasons']:
            results.append(record)
            continue
        backgrounds, noises = zip(*(_background(tile[..., c][ann]) for c in range(3)))
        background, noise = np.array(backgrounds), np.array(noises)
        signal = tile-background
        flux = signal[ap].sum(axis=0)
        nap, nann = int(ap.sum()), int(ann.sum())
        # Gaussian independent-background approximation, including uncertainty
        # of the median background. This is explicitly not full photometry SNR.
        flux_sigma = noise*np.sqrt(nap + (np.pi/2)*nap**2/nann)
        snr = [float(f/s) if s > 0 else None for f, s in zip(flux, flux_sigma)]
        record.update(status='measured', flux_rgb=flux.tolist(),
                      background_rgb=background.tolist(), background_noise_rgb=noise.tolist(),
                      background_only_flux_sigma_rgb=flux_sigma.tolist(), background_only_snr_rgb=snr,
                      aperture_pixels=nap, annulus_pixels=nann)
        g = signal[..., 1]
        weights = np.where(ap, np.maximum(g, 0), 0)
        total = float(weights.sum())
        if total > 0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cx, cy = float((xx*weights).sum()/total), float((yy*weights).sum()/total)
            dx, dy = xx-cx, yy-cy
            covariance = np.array([[(weights*dx*dx).sum(), (weights*dx*dy).sum()],
                                   [(weights*dx*dy).sum(), (weights*dy*dy).sum()]])/total
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
            minor, major = np.sqrt(eigenvalues)
            record['moments_G'] = {'centroid_xy': [cx+selection['origin_xy'][0], cy+selection['origin_xy'][1]],
                'sigma_major_px': float(major), 'sigma_minor_px': float(minor),
                'ellipticity': float(1-minor/major) if major > 0 else None,
                'covariance_xy': covariance.tolist(), 'weighting': 'positive_background_subtracted_G',
                'negative_aperture_samples': int(np.count_nonzero(g[ap] < 0)),
                'finite_aperture': True, 'is_FWHM': False}
        if not screened:
            record['color_exclusion_reasons'].append('saturation_not_screened')
        if np.any(flux <= 0):
            record['color_exclusion_reasons'].append('nonpositive_channel_flux')
        if np.any((flux_sigma > 0) & (flux < threshold*flux_sigma)):
            record['color_exclusion_reasons'].append('low_background_only_snr')
        if not record['color_exclusion_reasons']:
            record['color_ratios'] = {'R_over_G': float(flux[0]/flux[1]),
                                      'B_over_G': float(flux[2]/flux[1])}
        results.append(record)
    report = {'schema_version': SCHEMA_VERSION, 'selection_id': selection['selection_id'],
            'reference_label': selection['reference_label'],
            'coordinate_convention': selection['coordinate_convention'],
            'saturation_screened': bool(screened), 'min_color_background_snr': threshold,
            'stars': results,
            'limitations': ['Flux is signed and finite-aperture; no aperture correction or photometric calibration',
                'Positive-weight moments are noise-biased and are not FWHM',
                'SNR proxy assumes independent Gaussian background; excludes source shot noise and correlations',
                'Compare identical selection_id and intersect eligible star IDs; report all exclusions',
                'Equal linear units and color transforms are caller obligations; arrays cannot prove them']}
    # Catch numerical overflow rather than publishing NaN/Infinity as evidence.
    json.dumps(report, allow_nan=False)
    return report
