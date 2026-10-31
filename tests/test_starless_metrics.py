"""Starless package step 1 (docs/STARLESS_PACKAGE_20260923.md §六): synthetic self-checks S1–S8.

Thresholds are the pre-registered ones; a failure is recorded, not tuned away.
"""
import json
import numpy as np
import pytest
from starstack import starless_metrics as m

SHAPE = (1024, 1024)
SIGMA_C = 0.004
SIGMA_L = float(SIGMA_C * np.sqrt((m._LUM ** 2).sum()))


def field(seed=7, *, nebula=False):
    """Background + gradient + per-channel noise (+ optional broad blob), then injected stars."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:SHAPE[0], :SHAPE[1]]
    base = 0.12 + 0.02 * x / SHAPE[1] + 0.01 * y / SHAPE[0]
    if nebula:
        base = base + 20 * SIGMA_L * np.exp(-((y - 520) ** 2 + (x - 500) ** 2) / (2 * 120.0 ** 2))
    background = np.repeat(base[..., None], 3, -1).astype(np.float32)
    noise = rng.normal(0, SIGMA_C, background.shape).astype(np.float32)
    clean = np.rint(np.clip(background + noise, 0, 1) * 65535).astype(np.uint16)
    starry, truth = m.inject_stars(clean, rng, count=60, saturated=8, amplitude_range=(10 * SIGMA_L, 200 * SIGMA_L))
    return starry, clean, truth, background, noise


def region():
    return np.ones(SHAPE, bool)


def fill(starry, truth, maker, radius=5):
    """Replace r ≤ radius around every injected star with maker(window_float, star) → float window."""
    image = starry.astype(np.float32) / 65535
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (yy ** 2 + xx ** 2) <= radius ** 2
    for star in truth['stars']:
        y0, x0 = int(round(star['y'])), int(round(star['x']))
        window = image[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1]
        window[disk] = maker(window, star)[disk]
    return np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)


def test_s1_perfect_removal_scores_near_zero():
    starry, clean, truth, *_ = field()
    r = m.evaluate(starry, clean, region(), truth=truth)
    stars = r['stars']
    assert stars['snr_kept'] >= 30
    assert abs(stars['residual_ratio']['median']) <= 0.05 and stars['residual_ratio']['p90'] <= 0.15
    assert r['nonstar']['p95'] == 0.0
    assert 0.6 <= stars['fill_noise_ratio']['median'] <= 1.5
    assert abs(stars['luminance_offset']['median']) <= 0.5
    assert stars['colour_shift']['p90'] <= 2.0
    injected = r['injected']
    assert abs(injected['residual_ratio_measured']['median']) <= 0.05 and abs(injected['residual_ratio_analytic']['median']) <= 0.05
    assert r['saturated_stars']['measured'] >= 1 and r['saturation_footprint']['pixels'] > 0


def test_s2_no_removal_is_exactly_one():
    starry, *_ = field()
    r = m.evaluate(starry, starry, region())
    assert 0.99 <= r['stars']['residual_ratio']['median'] <= 1.01
    assert r['nonstar']['p99'] == 0.0 and r['nonstar']['fraction_over_3sigma'] == 0.0
    assert all(r['bands'][b]['ratio'] == 1.0 for b in m.BAND_NAMES)
    assert r['saturation_footprint']['perturbation']['p95'] == 0.0


def test_s3_smooth_fill_is_detected():
    starry, clean, truth, *_ = field()
    yy, xx = np.mgrid[-5:6, -5:6]
    outside = (yy ** 2 + xx ** 2) > 25
    smooth = fill(starry, truth, lambda window, star: np.broadcast_to(window[outside].mean(0), window.shape))
    r = m.evaluate(starry, smooth, region())
    assert r['stars']['fill_noise_ratio']['median'] <= 0.3
    assert abs(r['stars']['residual_ratio']['median']) <= 0.1


def test_s4_dark_pit_is_negative_offset():
    starry, clean, truth, background, noise = field()
    image = clean.astype(np.float32) / 65535
    yy, xx = np.mgrid[-5:6, -5:6]
    disk = (yy ** 2 + xx ** 2) <= 25
    for star in truth['stars']:
        y0, x0 = int(round(star['y'])), int(round(star['x']))
        window = image[y0 - 5:y0 + 6, x0 - 5:x0 + 6]
        window[disk] -= np.float32(3 * SIGMA_L)
    pit = np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)
    r = m.evaluate(starry, pit, region())
    assert r['stars']['luminance_offset']['median'] <= -2


def test_s5_colour_ring_is_detected():
    starry, clean, truth, *_ = field()
    image = clean.astype(np.float32) / 65535
    yy, xx = np.mgrid[-5:6, -5:6]
    disk = (yy ** 2 + xx ** 2) <= 25
    for star in truth['stars']:
        y0, x0 = int(round(star['y'])), int(round(star['x']))
        window = image[y0 - 5:y0 + 6, x0 - 5:x0 + 6]
        window[disk, 2] += np.float32(4 * SIGMA_C)
    tinted = np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)
    r = m.evaluate(starry, tinted, region())
    assert r['stars']['colour_shift']['p90'] >= 2.5


def test_s6_nebula_eating_is_detected():
    starry, clean, truth, background, noise = field(nebula=True)
    plain = field(nebula=False)[1]          # same seed: same noise, no blob
    r = m.evaluate(starry, plain, region())
    assert r['bands']['broad']['ratio'] <= 0.5
    assert r['nonstar']['p95'] >= 3


def test_s7_detection_recall_and_precision():
    starry, clean, truth, *_ = field()
    det = m.detect_stars(starry, region())
    ys, xs = det['tables']['measure']
    injected = [s for s in truth['stars'] if not s['saturated']]
    hits = 0
    for star in injected:
        d = np.hypot(ys - star['y'], xs - star['x'])
        hits += int(d.size and d.min() <= 2)
    assert hits / len(injected) >= 0.9
    for y, x in zip(ys, xs):
        assert min(np.hypot(y - s['y'], x - s['x']) for s in truth['stars']) <= 2
    assert det['tables']['saturated'][0].size >= 1


def test_s8_determinism_and_digest():
    starry, clean, truth, *_ = field()
    a = m.evaluate(starry, clean, region())
    b = m.evaluate(starry, clean, region())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    det = m.detect_stars(starry, region())
    quiet = ~m.dilate(det['exclusion'], 20) & ~m.dilate(det['saturated_pixels'], 30)
    quiet[:40] = quiet[-40:] = quiet[:, :40] = quiet[:, -40:] = False
    y, x = np.argwhere(quiet)[0]          # a pixel outside every star footprint
    changed = clean.copy()
    changed[y, x, 1] += 400
    assert m.report_digest(m.evaluate(starry, changed, region())) != m.report_digest(a)
    assert m.array_sha256(changed) != m.array_sha256(clean)


def test_input_validation():
    a = np.zeros((32, 32, 3), np.uint16)
    with pytest.raises(ValueError):
        m.evaluate(a, a, np.ones((32, 32), bool))
    b = np.zeros((80, 80, 3), np.uint16)
    with pytest.raises(ValueError):
        m.evaluate(b, b, np.ones((80, 80), np.uint8))
    with pytest.raises(ValueError):
        m.evaluate(b, np.zeros((80, 81, 3), np.uint16), np.ones((80, 80), bool))
