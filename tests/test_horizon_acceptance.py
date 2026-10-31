"""Tests for the S3 horizon metrics (synthetic arrays; no Siril, no photographs).

Run:  python -B -m unittest tests.test_horizon_acceptance   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import horizon_acceptance as ha  # noqa: E402

H, W = 700, 900
EDGE = 560                                       # first ground row (mask < 1) in every column


def _mask():
    mask = np.ones((H, W), dtype=np.float32)
    mask[EDGE, :] = 0.5
    mask[EDGE + 1:, :] = 0.0
    return mask


def _sky(rng, level=(0.26, 0.30, 0.24), noise=0.003):
    sky = np.empty((H, W, 3), dtype=np.float32)
    for c in range(3):
        sky[..., c] = rng.normal(level[c], noise, size=(H, W))
        sky[..., c] += np.linspace(0, 0.02, H)[:, None]        # the sky's own vertical gradient (same in frame and master)
    return sky


class ColourBandTest(unittest.TestCase):
    def test_contaminated_master_fails_and_clean_master_passes(self):
        rng = np.random.default_rng(3)
        mask = _mask()
        frame = _sky(rng)
        clean = _sky(rng) * 1.02                              # the night's colour/level drift: a constant factor cancels
        band = clean.copy()
        h = np.arange(H)[:, None]
        excess = np.clip((EDGE - h) / 300.0, 0, 1)             # 0 far above the edge, 1 at the edge
        tint = 1 - 0.15 * (1 - excess)                          # green deficit growing towards the edge (R/G, B/G up to 1.18)
        band[..., 1] *= np.where(h < EDGE, tint, 1.0)
        ranges = ha.column_ranges(W)
        self.assertEqual(len(ranges), 3)
        bad = ha.colour_band(band, frame, mask, ranges)
        good = ha.colour_band(clean, frame, mask, ranges)
        self.assertGreater(bad["max_abs_deviation"], 0.10)
        self.assertLess(good["max_abs_deviation"], 0.02)
        self.assertFalse(ha.judge_colour_band(bad, ha.CRITERIA["H3"])["passed"])
        self.assertTrue(ha.judge_colour_band(good, ha.CRITERIA["H3"])["passed"])
        first = next(iter(bad["ranges"].values()))
        self.assertGreater(first["RG"]["height_above_2pct_px"], 200)
        self.assertGreater(first["RG"]["at_edge"], 1.10)

    def test_range_without_ground_is_reported(self):
        mask = np.ones((H, W), dtype=np.float32)               # no ground anywhere
        rng = np.random.default_rng(1)
        out = ha.colour_band(_sky(rng), _sky(rng), mask, [(0, 300)])
        self.assertEqual(out["ranges"]["x0-300"]["status"], "no_ground_edge_in_range")
        self.assertFalse(ha.judge_colour_band(out, ha.CRITERIA["H3"])["passed"])


class ContributorColourTest(unittest.TestCase):
    """H3c: the master must have the colour of a plain mean of its normalised contributors — a drifting sky colour and
    a frame composition that changes with height are fine; a foreign (ground) tint is not."""

    def _frames(self, rng, n=12, h=120, cols=80):
        crops, stats = [], []
        for k in range(n):
            drift = 1.0 + 0.004 * (k - n / 2)                       # R/G drifts over the night
            level = 0.30 + 0.002 * k                                 # background drifts too
            crop = np.empty((h, cols, 3), dtype=np.float32)
            crop[..., 1] = rng.normal(level, 0.003, size=(h, cols))
            crop[..., 0] = crop[..., 1] * 0.85 * drift
            crop[..., 2] = crop[..., 1] * 0.80 * drift
            cut = 8 + 6 * k                                          # frame k observes only rows >= cut above the edge
            crop[:cut] = 0.0
            crops.append(crop)
            stats.append({"median": [float(np.median(crop[..., c][crop[..., 1] > 0])) for c in range(3)],
                          "mad": [float(np.median(np.abs(crop[..., c][crop[..., 1] > 0] - np.median(crop[..., c][crop[..., 1] > 0])))) for c in range(3)]})
        return crops, stats

    def test_clean_stack_matches_and_tinted_stack_fails(self):
        rng = np.random.default_rng(11)
        crops, stats = self._frames(rng)
        ref = stats[6]
        normalised = [ha.normalise_like_stack(c, st, ref) for c, st in zip(crops, stats)]
        acc = np.zeros_like(normalised[0], dtype=np.float64); cnt = np.zeros(normalised[0].shape[:2], dtype=np.int32)
        for c in normalised:
            ok = c[..., 1] > 0
            acc[ok] += c[ok]; cnt[ok] += 1
        master = np.where(cnt[..., None] > 0, acc / np.maximum(cnt, 1)[..., None], 0.0).astype(np.float32)
        edges = [4, 29, 54, 79, 104, 120]
        clean = ha.contributor_colour(master, normalised, edges, min_frame_pixels=100)
        self.assertEqual(clean["frames"], 12)
        self.assertLess(clean["max_abs_deviation"], 0.003)
        self.assertTrue(ha.judge_contributor_colour({"x": clean}, ha.CRITERIA["H3c"])["passed"])
        self.assertLess(clean["bands"][0]["contributors_mean"], clean["bands"][-1]["contributors_mean"])   # fewer frames near the edge
        # the same master with a green deficit growing towards the edge (ground tint) is caught
        tinted = master.copy()
        rows = np.arange(master.shape[0])[:, None]
        tinted[..., 1] *= (1 - 0.10 * np.clip(1 - rows / 100.0, 0, 1))
        bad = ha.contributor_colour(tinted, normalised, edges, min_frame_pixels=100)
        self.assertGreater(bad["max_abs_deviation"], 0.05)
        self.assertFalse(ha.judge_contributor_colour({"x": bad}, ha.CRITERIA["H3c"])["passed"])

    def test_normalisation_moves_a_frame_to_the_reference_level(self):
        crop = np.full((10, 10, 3), 0.5, dtype=np.float32); crop[0] = 0.0
        stats = {"median": [0.5, 0.5, 0.5], "mad": [0.01, 0.01, 0.01]}
        ref = {"median": [0.3, 0.31, 0.29], "mad": [0.02, 0.02, 0.02]}
        out = ha.normalise_like_stack(crop, stats, ref)
        np.testing.assert_allclose(out[1:, :, 0], 0.3, atol=1e-6)
        np.testing.assert_allclose(out[1:, :, 1], 0.31, atol=1e-6)
        self.assertEqual(float(out[0].max()), 0.0)
        gs = ha.global_stats([np.array([[0.0, 1.0, 2.0, 3.0]])] * 3)
        self.assertEqual(gs["median"], [2.0, 2.0, 2.0])
        self.assertEqual(gs["mad"], [1.0, 1.0, 1.0])


class ConsistencyAndSeamTest(unittest.TestCase):
    def test_consistency_counts_disagreements(self):
        step = 4
        cov = np.ones((H // step, W // step), dtype=np.int32) * 5
        cov[EDGE // step:, :] = 0
        master_g = np.full((H, W), 0.3, dtype=np.float32)
        master_g[EDGE:, :] = 0.0
        out = ha.master_coverage_consistency(master_g, cov, step)
        self.assertEqual(out["agreement"], 1.0)
        master_g[EDGE - 8:EDGE, :] = 0.0                       # two sample rows zero although covered
        out = ha.master_coverage_consistency(master_g, cov, step)
        self.assertLess(out["agreement"], 1.0)
        self.assertEqual(out["zero_master_with_coverage"], 2 * (W // step))

    def test_seam_rules(self):
        step = 4
        rng = np.random.default_rng(5)
        mask = _mask()
        sky = _sky(rng); ground = np.full((H, W, 3), 0.20, dtype=np.float32)
        cov = np.ones((H // step, W // step), dtype=np.int32) * 5
        cov[(EDGE - 6) // step:, :] = 0                        # unobserved from 6 px above the edge downwards (sampled)
        cov_full = np.repeat(np.repeat(cov, step, axis=0), step, axis=1)[:H, :W]
        weight = mask.copy(); weight[cov_full == 0] = 0.0
        sky_masked = sky.copy(); sky_masked[cov_full == 0] = 0.0
        composite = sky_masked * weight[..., None] + ground * (1 - weight[..., None])
        out = ha.seam(composite, sky_masked, ground, mask, cov, step, effective_weight=weight)
        for name in ("sky_only", "ground_only", "unobserved_sky"):
            self.assertLess(out["regions"][name]["max_abs_diff"], 1e-6)
        self.assertGreater(out["observations"]["rim_min_ratio_p05"], 0.6)
        # H5 2.0.0: a naturally dark ground is correct if the blend equation holds.
        judged = ha.judge_seam(out, ha.CRITERIA["H5"])
        self.assertEqual(judged["status"], "passed")
        self.assertFalse(out["observations"]["used_for_verdict"])
        bad = sky_masked * mask[..., None] + ground * (1 - mask[..., None])
        out_bad = ha.seam(bad, sky_masked, ground, mask, cov, step, effective_weight=weight)
        self.assertGreater(out_bad["regions"]["unobserved_sky"]["max_abs_diff"], 0.1)
        self.assertEqual(ha.judge_seam(out_bad, ha.CRITERIA["H5"])["status"], "failed")
        # The original mask has a feather row but no effective feather survives coverage.
        self.assertEqual(out["mask_feather_pixels"], W)
        self.assertEqual(out["regions"]["feather"]["samples"], 0)
        feather = next(c for c in judged["checks"] if c["name"] == "feather")
        self.assertEqual(feather["status"], "not_applicable")
        hard = (mask >= 1.0).astype(np.float32)
        hard_weight = hard * (cov_full > 0)
        hard_composite = sky_masked * hard_weight[..., None] + ground * (1 - hard_weight[..., None])
        out_hard = ha.seam(hard_composite, sky_masked, ground, hard, cov, step, effective_weight=hard_weight)
        self.assertEqual(out_hard["mask_feather_pixels"], 0)
        self.assertEqual(out_hard["regions"]["feather"]["samples"], 0)
        self.assertEqual(ha.judge_seam(out_hard, ha.CRITERIA["H5"])["status"], "passed")

    def test_feather_band_blend_rule_and_dark_seam(self):
        """H5 2.0.0: injected errors fail the same-location blend rule; ratios are diagnostic."""
        step = 4
        rng = np.random.default_rng(6)
        mask = _mask().copy()
        band = 16
        ramp = np.linspace(1.0, 0.0, band + 2)[1:-1].astype(np.float32)
        mask[EDGE:EDGE + band, :] = ramp[:, None]                   # linear feather from the old hard edge downwards
        sky = _sky(rng); ground = np.full((H, W, 3), 0.28, dtype=np.float32)
        cov = np.ones((H // step, W // step), dtype=np.int32) * 5
        weight = mask.copy()
        composite = sky * weight[..., None] + ground * (1 - weight[..., None])
        out = ha.seam(composite, sky, ground, mask, cov, step, effective_weight=weight)
        self.assertGreater(out["regions"]["feather"]["samples"], 0)
        self.assertLess(out["regions"]["feather"]["max_abs_diff"], 1e-6)
        self.assertGreater(out["observations"]["band_columns"], 0)
        self.assertGreater(out["observations"]["band_min_ratio_p05"], 0.9)          # a blend of 0.30-ish sky and 0.28 ground never undershoots
        judged = ha.judge_seam(out, ha.CRITERIA["H5"])
        self.assertFalse(any("feather" in f or "seam inside" in f for f in judged["failing"]), judged)
        seamed = composite.copy()
        seamed[EDGE + 6:EDGE + 10, :, :] *= 0.5                     # a dark seam inside the band
        out_seam = ha.seam(seamed, sky, ground, mask, cov, step, effective_weight=weight)
        self.assertGreater(out_seam["regions"]["feather"]["max_abs_diff"], 1e-3)
        self.assertLess(out_seam["observations"]["band_min_ratio_p05"], 0.85)
        failing = ha.judge_seam(out_seam, ha.CRITERIA["H5"])["failing"]
        self.assertTrue(any("feather: same-location" in f for f in failing), failing)


if __name__ == "__main__":
    unittest.main()
