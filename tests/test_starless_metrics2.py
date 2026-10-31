"""metrics-2 dense-field self-checks D1–D7 (docs/STARLESS_METRICS2_20260923.md §三). Thresholds are pre-registered."""
import json
import numpy as np
import pytest
from starstack import starless_metrics as m1
from starstack import starless_metrics2 as m2

SHAPE = (1024, 1024)
SIGMA_C = 0.004
SIGMA_L = float(SIGMA_C * np.sqrt((m1._LUM ** 2).sum()))
SPLIT = 512


def field(seed=21):
    """Left half sparse (~1 star / kpx, ≥20 px apart), right half dense (~25 / kpx, ≥4 px apart)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:SHAPE[0], :SHAPE[1]]
    base = 0.12 + 0.02 * x / SHAPE[1] + 0.01 * y / SHAPE[0]
    background = np.repeat(base[..., None], 3, -1).astype(np.float32)
    noise = rng.normal(0, SIGMA_C, background.shape).astype(np.float32)
    clean = np.rint(np.clip(background + noise, 0, 1) * 65535).astype(np.uint16)
    amp = (10 * SIGMA_L, 200 * SIGMA_L)
    left, truth_l = m1.inject_stars(np.ascontiguousarray(clean[:, :SPLIT]), rng, count=500, saturated=4,
                                    amplitude_range=amp, min_separation=20, margin=20)
    right, truth_r = m1.inject_stars(np.ascontiguousarray(clean[:, SPLIT:]), rng, count=13000, saturated=4,
                                     amplitude_range=amp, min_separation=4, margin=20)
    starry = np.concatenate([left, right], axis=1)
    for star in truth_r["stars"]:
        star["x"] += SPLIT
    truth = {"stars": truth_l["stars"] + truth_r["stars"]}
    return starry, clean, truth


def region():
    return np.ones(SHAPE, bool)


def fill(clean, truth, maker, radius=2):
    image = clean.astype(np.float32) / 65535
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (yy ** 2 + xx ** 2) <= radius ** 2
    for star in truth["stars"]:
        if star["saturated"]:
            continue
        y0, x0 = int(round(star["y"])), int(round(star["x"]))
        window = image[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1]
        window[disk] = maker(window[disk])
    return np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)


@pytest.fixture(scope="module")
def data():
    starry, clean, truth = field()
    return starry, clean, truth


def test_d1_strata_follow_density(data):
    starry, clean, truth = data
    r = m2.evaluate2(starry, clean, region())
    L = m1.luminance16(starry)
    sigma = m1.noise_sigma(L, 64)
    bg = m1.block_median(L, 64)
    excl = m1.local_maxima(L - bg) & ((L - bg) >= 4 * sigma)
    strata = m2.stratify(excl, region())
    ps = strata["pixel_stratum"]
    assert (ps[:, SPLIT:] == 2).mean() >= 0.8, "right half should be dense"
    assert (ps[:, :SPLIT] == 0).mean() >= 0.8, "left half should be sparse"
    assert r["strata"]["dense"]["region_fraction"] >= 0.4 and r["strata"]["sparse"]["region_fraction"] >= 0.4


def test_d2_perfect_removal_in_dense_stratum(data):
    starry, clean, truth = data
    r = m2.evaluate2(starry, clean, region())
    dense = r["strata"]["dense"]
    assert dense["stars"]["snr_kept"] >= 2000
    assert abs(dense["stars"]["residual_ratio"]["median"]) <= 0.10 and dense["stars"]["residual_ratio"]["p90"] <= 0.30
    assert dense["energy"]["positive_hp_ratio"]["median"] <= 0.30
    assert 0.7 <= dense["energy"]["negative_hp_over_noise"]["median"] <= 1.4
    assert abs(dense["stars"]["luminance_offset"]["median"]) <= 0.5
    assert dense["stars"]["colour_shift"]["p90"] <= 2.0


def test_d3_no_removal_is_exactly_one(data):
    starry, clean, truth = data
    r = m2.evaluate2(starry, starry, region())
    for name, s in r["strata"].items():
        if s["stars"]["snr_kept"]:
            assert abs(s["stars"]["residual_ratio"]["median"] - 1) <= 0.01, name
        for key in ("positive_hp_ratio", "negative_hp_ratio", "lowpass_ratio"):
            if s["blocks"]:                       # the synthetic field has no medium stratum
                assert s["energy"][key]["median"] == 1.0, (name, key)


def test_d4_dark_pits_in_dense_stratum(data):
    starry, clean, truth = data
    pit = fill(clean, truth, lambda v: v - np.float32(3 * SIGMA_L))
    r = m2.evaluate2(starry, pit, region())
    dense = r["strata"]["dense"]
    assert dense["stars"]["luminance_offset"]["median"] <= -2
    assert dense["energy"]["negative_hp_over_noise"]["median"] >= 2


def test_d5_colour_tint_in_dense_stratum(data):
    starry, clean, truth = data

    def tint(v):
        out = v.copy()
        out[:, 2] += np.float32(4 * SIGMA_C)
        return out
    tinted = fill(clean, truth, tint)
    r = m2.evaluate2(starry, tinted, region())
    assert r["strata"]["dense"]["stars"]["colour_shift"]["p90"] >= 2.5


def test_d6_coverage_versus_metrics1(data):
    starry, clean, truth = data
    det1 = m1.detect_stars(starry, region())
    y1, x1 = det1["tables"]["measure"]
    injected_right = [s for s in truth["stars"] if s["x"] >= SPLIT and not s["saturated"]]
    right1 = int((x1 >= SPLIT).sum())
    assert right1 < 0.05 * len(injected_right), (right1, len(injected_right))
    L = m1.luminance16(starry)
    sigma = m1.noise_sigma(L, 64)
    bg = m1.block_median(L, 64)
    ys, xs, _ = m2.dense_candidates(L, sigma, bg, region(), starry)
    right2 = xs >= SPLIT
    assert right2.sum() >= 10 * max(right1, 1)
    bright = [s for s in injected_right if s["amplitude"] >= 20 * SIGMA_L and 20 <= s["y"] < SHAPE[0] - 20]
    cy, cx = ys[right2], xs[right2]
    hits = 0
    for s in bright:
        d = np.hypot(cy - s["y"], cx - s["x"])
        hits += int(d.size and d.min() <= 2)
    assert hits / len(bright) >= 0.5, hits / len(bright)


def test_d7_determinism_and_digest(data):
    starry, clean, truth = data
    a = m2.evaluate2(starry, clean, region())
    b = m2.evaluate2(starry, clean, region())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    changed = clean.copy()
    changed[40, 40, 1] += 400          # sparse corner, far from any injected star (margin 20, ≥30 px apart) — only affects energies / lowpass
    assert m2.report_digest(m2.evaluate2(starry, changed, region())) != m2.report_digest(a)
