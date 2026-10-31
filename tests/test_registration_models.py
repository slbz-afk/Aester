"""Tests for starstack.registration_models (pure NumPy model comparison).

Run:  python -B -m unittest tests.test_registration_models   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import registration_models as rm  # noqa: E402


def grid_points(width=1200, height=901, n=300, seed=3):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(5, width - 5, n), rng.uniform(5, height - 5, n)])


def rotation(deg, scale=1.0, tx=0.0, ty=0.0):
    t = math.radians(deg)
    return np.array([[scale * math.cos(t), -scale * math.sin(t), tx], [scale * math.sin(t), scale * math.cos(t), ty], [0, 0, 1.0]])


class Split(unittest.TestCase):
    def test_alternation_inside_every_cell(self):
        pts = grid_points(n=600)
        mask = rm.stratified_split(pts, 1200, 901, columns=4, rows=3)
        self.assertEqual(len(mask), 600)
        col = np.clip((pts[:, 0] * 4 / 1200).astype(int), 0, 3)
        row = np.clip((pts[:, 1] * 3 / 901).astype(int), 0, 2)
        for c in range(4):
            for r in range(3):
                idx = np.where((col == c) & (row == r))[0]
                if len(idx) == 0:
                    continue
                fit = mask[idx].sum()
                self.assertIn(fit - (len(idx) - fit), (0, 1))     # balanced within one point per cell
        # deterministic: same input, same mask; permuting the input permutes the mask
        np.testing.assert_array_equal(mask, rm.stratified_split(pts, 1200, 901))
        perm = np.random.default_rng(1).permutation(600)
        np.testing.assert_array_equal(rm.stratified_split(pts[perm], 1200, 901), mask[perm])

    def test_region_labels(self):
        self.assertEqual(rm.region_of(600, 450, 1200, 901), "center")
        self.assertEqual(rm.region_of(10, 10, 1200, 901), "corner")
        self.assertEqual(rm.region_of(1190, 890, 1200, 901), "corner")
        self.assertEqual(rm.region_of(600, 10, 1200, 901), "other")


class ExactRecovery(unittest.TestCase):
    def test_similarity_affine_homography_recover_known_transforms(self):
        src = grid_points()
        sim = rotation(1.7, 1.003, 12.5, -7.25)
        np.testing.assert_allclose(rm.fit_similarity(src, rm.apply_matrix(sim, src)), sim, atol=1e-9)
        aff = np.array([[1.002, 0.03, 4.0], [-0.01, 0.998, -3.0], [0, 0, 1.0]])
        np.testing.assert_allclose(rm.fit_affine(src, rm.apply_matrix(aff, src)), aff, atol=1e-9)
        hom = np.array([[1.001, 0.0005, -2.75], [0.0004, 1.0012, -5.9], [1.2e-7, 2.3e-7, 1.0]])
        fitted = rm.fit_homography(src, rm.apply_matrix(hom, src))
        np.testing.assert_allclose(fitted, hom, rtol=1e-6, atol=1e-9)
        # a similarity cannot represent a perspective term: residual must be visible on validation
        dst = rm.apply_matrix(hom, src)
        mask = rm.stratified_split(src, 1200, 901)
        e = rm.evaluate_model(rm.apply_matrix(rm.fit_similarity(src[mask], dst[mask]), src), dst, mask, 1200, 901)
        self.assertGreater(e["validation"]["overall"]["rms_px"], 0.05)

    def test_polynomial_recovers_quadratic_map(self):
        src = grid_points()
        centre = np.array([599.5, 450.0])
        q = (src - centre) / 600.0
        dst = src + np.column_stack([3.0 * q[:, 0] ** 2 - 1.5 * q[:, 0] * q[:, 1], 2.0 * q[:, 1] ** 2 + 0.7 * q[:, 0]])
        model = rm.fit_polynomial(src, dst, 2, centre=centre, scale=600.0)
        np.testing.assert_allclose(rm.apply_polynomial(model, src), dst, atol=1e-9)
        model3 = rm.fit_polynomial(src, dst, 3, centre=centre, scale=600.0)
        np.testing.assert_allclose(rm.apply_polynomial(model3, src), dst, atol=1e-8)

    def test_radial_model_recovers_common_distortion(self):
        # true geometry: undistorted frames related by a homography, both observed through r(1 + k1 r^2)
        src_u = grid_points(n=400, seed=7)
        hom = rotation(2.5, 1.0, 9.0, -4.0)
        hom[2, 0], hom[2, 1] = 3e-7, -2e-7
        dst_u = rm.apply_matrix(hom, src_u)
        centre = np.array([599.5, 450.0])
        scale = 750.0
        k1_true = -0.04

        def distort(p):
            q = (p - centre) / scale
            r2 = (q ** 2).sum(axis=1, keepdims=True)
            # solve r_d = r_u (1 + k1 r_u^2) numerically: inverse of the undistort model used by the module
            out = q.copy()
            for _ in range(50):
                r2o = (out ** 2).sum(axis=1, keepdims=True)
                out = q / (1.0 + k1_true * r2o)
            return centre + scale * out

        src_d, dst_d = distort(src_u), distort(dst_u)
        model = rm.fit_homography_radial(src_d, dst_d, centre, scale=scale)
        self.assertAlmostEqual(model["k1"], k1_true, places=4)
        self.assertAlmostEqual(model["k2"], 0.0, places=4)
        mask = rm.stratified_split(src_d, 1200, 901)
        ud = rm._undistort(dst_d, centre, scale, model["k1"], model["k2"])
        e = rm.evaluate_model(rm.apply_homography_radial(model, src_d), ud, mask, 1200, 901)
        self.assertLess(e["validation"]["overall"]["rms_px"], 1e-3)
        # a plain homography leaves the (differential) distortion behind: 0.28 px in the corners
        # for this 2.5 deg rotation, i.e. two orders of magnitude above the radial model
        plain = rm.fit_homography(src_d[mask], dst_d[mask])
        e2 = rm.evaluate_model(rm.apply_matrix(plain, src_d), dst_d, mask, 1200, 901)
        self.assertGreater(e2["validation"]["corner"]["rms_px"], 0.1)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            rm.fit_homography(np.zeros((3, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            rm.fit_affine(np.zeros((4, 2)), np.zeros((5, 2)))
        with self.assertRaises(ValueError):
            rm.fit_similarity([[0.0, float("nan")], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]])


class Comparison(unittest.TestCase):
    def test_compare_models_reports_all_families_and_siril_matrix(self):
        src = grid_points(n=500, seed=11)
        hom = rotation(0.8, 1.0, -3.0, 6.0)
        hom[2, 0] = 1.5e-7
        rng = np.random.default_rng(4)
        dst = rm.apply_matrix(hom, src) + rng.normal(0.0, 0.05, src.shape)
        biased = hom.copy()
        biased[0, 2] += 0.4                         # Siril-like matrix with a 0.4 px offset error
        result = rm.compare_models(src, dst, 1200, 901, siril_matrix=biased)
        self.assertEqual(result["count"], 500)
        self.assertEqual(result["fit_count"] + result["validation_count"], 500)
        names = list(result["models"])
        self.assertEqual(names, ["siril_H_as_is", "similarity", "affine", "homography", "homography_radial", "polynomial2", "polynomial3"])
        self.assertFalse(result["models"]["siril_H_as_is"]["independent_validation"])
        v_siril = result["models"]["siril_H_as_is"]["residuals"]["validation"]["overall"]
        v_hom = result["models"]["homography"]["residuals"]["validation"]["overall"]
        self.assertAlmostEqual(v_siril["mean_dx_px"], 0.4, delta=0.03)
        self.assertLess(v_hom["rms_px"], 0.09)
        self.assertGreater(v_siril["rms_px"], 0.35)
        self.assertEqual(len(result["fit_mask"]), 500)
        self.assertEqual(len(result["caveats"]), 3)

    def test_entry_script_runs_on_simulated_products(self):
        from starstack import siril_coordinate_proof as cp
        from starstack import synthetic_starfield as sf
        with tempfile.TemporaryDirectory() as d:
            truth = sf.generate_dataset(Path(d) / "stage")
            cp.simulate_siril_outputs(truth, Path(d) / "work", row_order="TOP-DOWN", x_offset=0.0, y_offset=0.0,
                                      matrix_direction="frame_to_reference")
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "s2_model_comparison.py"),
                                   "--seq", str(Path(d) / "work" / "light_.seq"), "--lst-dir", str(Path(d) / "work" / "cache"),
                                   "--source-frame", "3", "--destination-frame", "2", "--width", "1200", "--height", "901",
                                   "--output-root", str(Path(d) / "out")], capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            runs = list((Path(d) / "out").glob("model-comparison-f3-to-f2-*"))
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "MODEL_COMPARISON.md").exists())
            import json
            result = json.loads((runs[0] / "model_comparison.json").read_text(encoding="utf-8"))
            # simulated matrices are exact: Siril's chained matrix and the refit agree to centroid noise
            for name in ("siril_H_as_is", "homography"):
                self.assertLess(result["comparison"]["models"][name]["residuals"]["validation"]["overall"]["rms_px"], 0.1)


if __name__ == "__main__":
    unittest.main()
