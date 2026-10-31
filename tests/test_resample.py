"""Tests for starstack.resample (registration-grade warping) on synthetic stars.

Run:  python -B -m unittest tests.test_resample   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import fits_crop, resample  # noqa: E402
from starstack import synthetic_starfield as sf  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402


def _stars():
    layout = sf.default_layout()
    img16, rendered = sf.render_frame(layout, np.eye(3), seed=3)
    return img16.astype(np.float32), np.array([[s["x"], s["y"]] for s in rendered])


def _centroid_errors(image, expected):
    h, w = image.shape
    errs = []
    for x, y in expected:
        if 10 <= x < w - 10 and 10 <= y < h - 10:
            c = fits_crop._centroid(image, float(x), float(y), 4)
            if c and c[3] - c[2] > 50:
                errs.append((c[0] - x, c[1] - y))
    return np.asarray(errs)


class KernelTest(unittest.TestCase):
    def test_kernels_are_interpolating(self):
        for fn in (resample.lanczos_kernel, resample.cubic_kernel):
            self.assertAlmostEqual(float(fn(np.array([0.0]))[0]), 1.0)
            self.assertAlmostEqual(float(fn(np.array([1.0]))[0]), 0.0, places=12)
            self.assertAlmostEqual(float(fn(np.array([2.0]))[0]), 0.0, places=12)
        self.assertEqual(float(resample.lanczos_kernel(np.array([3.0]))[0]), 0.0)
        with self.assertRaises(ValueError):
            resample.warp_image(np.zeros((4, 4), dtype=np.float32), lambda p: p, kernel="nope")


class WarpTest(unittest.TestCase):
    def test_fractional_shift_is_unbiased_and_keeps_star_width(self):
        img, xy = _stars()
        sigma0 = fits_crop.measure_star_shapes(img, [tuple(p) for p in xy])["median_sigma_major_px"]
        for kernel, sigma_tol in (("cubic", 0.005), ("lanczos3", 0.01)):
            shift = np.array([0.3, 0.7])
            warped = resample.warp_image(img, lambda p: p - shift, kernel=kernel)
            errs = _centroid_errors(np.nan_to_num(warped), xy + shift)
            self.assertGreater(len(errs), 150)
            self.assertLess(np.abs(errs.mean(axis=0)).max(), 0.02, kernel)          # no phase bias
            self.assertLess(np.median(np.hypot(*errs.T)), 0.05, kernel)
            sigma1 = fits_crop.measure_star_shapes(np.nan_to_num(warped), [tuple(p) for p in xy + shift])["median_sigma_major_px"]
            self.assertLess(abs(sigma1 / sigma0 - 1.0), sigma_tol, kernel)       # cubic / lanczos do not broaden the stars
        bilinear = resample.warp_image(img, lambda p: p - np.array([0.5, 0.5]), kernel="bilinear")
        sigma_b = fits_crop.measure_star_shapes(np.nan_to_num(bilinear), [tuple(p) for p in xy + 0.5])["median_sigma_major_px"]
        self.assertGreater(sigma_b / sigma0, 1.02)                                 # bilinear does (that is why it is not the default)

    def test_identity_is_exact_and_border_is_not_invented(self):
        img, _ = _stars()
        same = resample.warp_image(img, lambda p: p, kernel="cubic")
        np.testing.assert_allclose(same, img, rtol=0, atol=1e-3)                     # including the border: taps are edge-clamped (1.0.1)
        self.assertFalse(np.isnan(same).any())
        shifted = resample.warp_image(img, lambda p: p + np.array([10.0, 0.0]), kernel="cubic")
        self.assertTrue(np.isnan(shifted[100, -5]))                                  # source position outside the image: nothing invented
        self.assertFalse(np.isnan(shifted[100, -11]))                                # position inside (x = 169 of 0..169): resampled
        self.assertFalse(np.isnan(shifted[100, 100]))
        # a half-pixel shift keeps a flat field flat up to the very edge (edge extension, normalised weights)
        flat = resample.warp_image(np.full((40, 50), 7.0, dtype=np.float32), lambda p: p + np.array([0.5, 0.5]), kernel="cubic")
        inside = np.isfinite(flat)
        self.assertTrue(inside[:-1, :-1].all())
        np.testing.assert_allclose(flat[inside], 7.0, rtol=0, atol=1e-4)

    def test_radial_mapping_and_cubes(self):
        img, xy = _stars()
        h, w = img.shape
        d = RadialDistortion(0.02, -0.005, (w - 1) / 2 - 15, (h - 1) / 2 + 11, float(np.hypot(w, h) / 2))
        warped = resample.warp_image(img, d.apply, kernel="cubic")                # output p samples the source at D(p)
        errs = _centroid_errors(np.nan_to_num(warped), d.inverse(xy))             # so a source star at u lands at D^-1(u)
        self.assertGreater(len(errs), 150)
        self.assertLess(np.abs(errs.mean(axis=0)).max(), 0.02)
        self.assertLess(np.median(np.hypot(*errs.T)), 0.05)
        cube = np.stack([img, img * 0.5, img * 0.25])
        out = resample.warp_planes(cube, d.apply, kernel="cubic")
        self.assertEqual(out.shape, cube.shape)
        np.testing.assert_allclose(np.nan_to_num(out[1]), np.nan_to_num(out[0]) * 0.5, rtol=1e-5, atol=1e-2)
        flat = resample.warp_image(np.full((120, 160), 1000.0, dtype=np.float32), lambda p: p + 0.37, kernel="lanczos3")
        self.assertAlmostEqual(float(np.nanmin(flat)), 1000.0, places=2)
        self.assertAlmostEqual(float(np.nanmax(flat)), 1000.0, places=2)


if __name__ == "__main__":
    unittest.main()
