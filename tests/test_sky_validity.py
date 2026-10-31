"""Tests for starstack.sky_validity (no Siril, no astropy; synthetic FITS only).

Run:  python -B -m unittest tests.test_sky_validity   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import sky_validity as sv  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402
from starstack.synthetic_starfield import read_fits_minimal, write_fits_rgb16  # noqa: E402


def _write_float_fits(path: Path, image_top_down: np.ndarray) -> None:
    """Minimal 32-bit float RGB FITS, bottom-up, three equal planes (what Siril's -32b products look like)."""
    from starstack.synthetic_starfield import _card
    height, width = image_top_down.shape
    cards = [_card("SIMPLE", True), _card("BITPIX", -32), _card("NAXIS", 3), _card("NAXIS1", width), _card("NAXIS2", height),
             _card("NAXIS3", 3), _card("ROWORDER", "BOTTOM-UP"), "END".ljust(80)]
    header = "".join(cards)
    header += " " * ((2880 - len(header) % 2880) % 2880)
    plane = image_top_down[::-1, :].astype(">f4")
    data = np.repeat(plane[None, :, :], 3, axis=0).tobytes()
    data += b"\0" * ((2880 - len(data) % 2880) % 2880)
    path.write_bytes(header.encode("ascii") + data)


def _read_float_fits_top_down(path: Path) -> np.ndarray:
    from starstack import fits_crop
    view, header, roworder = fits_crop.plane_memmap(path, 0)
    a = np.asarray(view, dtype=np.float32)
    return a[::-1] if roworder == "BOTTOM-UP" else a


class MaskTest(unittest.TestCase):
    def test_dilation_is_a_square_of_the_given_radius(self):
        inv = np.zeros((11, 11), dtype=bool)
        inv[5, 5] = True
        out = sv.dilate(inv, 2)
        self.assertEqual(int(out.sum()), 25)
        self.assertTrue(out[3:8, 3:8].all())
        self.assertFalse(out[2, :].any() or out[8, :].any() or out[:, 2].any() or out[:, 8].any())
        np.testing.assert_array_equal(sv.dilate(inv, 0), inv)

    def test_static_validity_excludes_partial_mask_weight_and_a_margin(self):
        mask = np.ones((20, 30), dtype=np.float32)
        mask[14:, :] = 0.0            # ground
        mask[13, :] = 0.5             # feathered edge
        valid = sv.static_validity(mask, dilation_px=3)
        self.assertTrue(valid[:10, :].all())        # rows 0..9 valid: 13 - 3 = 10 is the first invalid row
        self.assertFalse(valid[10:, :].any())
        valid0 = sv.static_validity(mask, dilation_px=0)
        self.assertTrue(valid0[:13, :].all() and not valid0[13:, :].any())


class ApplyTest(unittest.TestCase):
    def test_zeroes_invalid_pixels_in_place_16bit_with_bottom_up_storage(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light_00001.fit"
            image = np.full((20, 30), 20000, dtype=np.uint16)
            image[2, 3] = 30000                      # a marker in the sky (top-down row 2)
            write_fits_rgb16(path, image)
            valid = np.ones((20, 30), dtype=bool)
            valid[15:, :] = False                    # bottom 5 rows invalid (top-down)
            valid[0, 0] = False
            info = sv.apply_validity_to_fits(path, valid)
            self.assertEqual(info["zeroed_pixels"], 5 * 30 + 1)
            _, back = read_fits_minimal(path)
            self.assertEqual(int(back[15:, :].max()), 0)
            self.assertEqual(int(back[0, 0]), 0)
            self.assertEqual(int(back[2, 3]), 30000)
            self.assertEqual(int(back[:15, 1:].min()), 20000)
            # idempotent
            info2 = sv.apply_validity_to_fits(path, valid)
            self.assertEqual(info2["zeroed_pixels"], info["zeroed_pixels"])

    def test_zeroes_invalid_pixels_in_a_float_fits(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light_00002.fit"
            image = np.full((12, 16), 0.3, dtype=np.float32)
            image[1, 1] = 0.9
            _write_float_fits(path, image)
            valid = np.ones((12, 16), dtype=bool)
            valid[9:, :] = False
            sv.apply_validity_to_fits(path, valid)
            back = _read_float_fits_top_down(path)
            self.assertEqual(float(back[9:, :].max()), 0.0)
            self.assertAlmostEqual(float(back[1, 1]), 0.9, places=6)
            self.assertAlmostEqual(float(back[:9, :].min()), 0.3, places=6)

    def test_shape_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light_00001.fit"
            write_fits_rgb16(path, np.full((20, 30), 100, dtype=np.uint16))
            with self.assertRaises(ValueError):
                sv.apply_validity_to_fits(path, np.ones((10, 30), dtype=bool))


class CoverageTest(unittest.TestCase):
    def _validity(self):
        valid = np.ones((40, 60), dtype=bool)
        valid[30:, :] = False                       # ground in the bottom quarter, every frame
        return valid

    def test_translations_shift_the_ground_edge_in_the_reference_grid(self):
        valid = self._validity()
        # frame -> reference: frame B's content appears 10 px lower (its ground edge lower -> it observes the reference's
        # rows 30..39), frame C's content appears 10 px higher (its ground reaches up to reference row 20)
        transforms = {"A": np.eye(3), "B": np.array([[1, 0, 0], [0, 1, 10], [0, 0, 1]], dtype=float),
                      "C": np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]], dtype=float)}
        cov = sv.coverage_map(valid, transforms, None, step=1)
        self.assertEqual(cov.shape, (40, 60))
        # row 5 of the reference is frame B's row -5 (outside the frame) and frame C's row 15 (valid): A + C = 2
        self.assertEqual(int(cov[5, 10]), 2)
        self.assertEqual(int(cov[25, 10]), 2)         # A valid (25<30), B row 15 valid, C row 35 ground -> 2
        self.assertEqual(int(cov[35, 10]), 1)         # A ground, B row 25 valid, C row 45 outside -> 1
        self.assertEqual(int(cov[15, 10]), 3)         # A, B row 5, C row 25 all valid

    def test_model_round_trip_keeps_identity_coverage(self):
        valid = self._validity()
        model = RadialDistortion(-0.02, 0.005, 30.0, 20.0, 36.0, magnification=0.98, direction="u_to_p")
        cov = sv.coverage_map(valid, {"ref": np.eye(3)}, model, step=1)
        np.testing.assert_array_equal(cov, valid.astype(np.int32))

    def test_effective_weight_zeroes_only_unobserved_band_pixels(self):
        valid = self._validity()
        mask = np.ones((40, 60), dtype=np.float32)
        mask[33:, :] = 0.0
        mask[30:33, :] = 0.5                        # feathered band inside the reference's invalid zone
        transforms = {"A": np.eye(3), "B": np.array([[1, 0, 0], [0, 1, 10], [0, 0, 1]], dtype=float)}
        weight, info = sv.effective_sky_weight(mask, valid, transforms, None)
        self.assertEqual(info["band_pixels"], 3 * 60)
        self.assertEqual(info["no_coverage_pixels"], 0)   # frame B observes rows 30..32 (its rows 20..22)
        np.testing.assert_array_equal(weight, mask)
        weight2, info2 = sv.effective_sky_weight(mask, valid, {"A": np.eye(3)}, None)
        self.assertEqual(info2["no_coverage_pixels"], 3 * 60)
        self.assertEqual(float(weight2[30:33, :].max()), 0.0)
        np.testing.assert_array_equal(weight2[:30, :], mask[:30, :])

    def test_summary_reports_reduced_coverage_band(self):
        valid = self._validity()
        mask = np.ones((40, 60), dtype=np.float32)
        mask[30:, :] = 0.0
        transforms = {"A": np.eye(3), "B": np.array([[1, 0, 0], [0, 1, -8], [0, 0, 1]], dtype=float)}   # B's ground reaches up to row 22
        cov = sv.coverage_map(valid, transforms, None, step=2)
        summary = sv.coverage_summary(cov, mask, step=2, frames=2)
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["min"], 1)
        self.assertLess(summary["fraction_full"], 1.0)
        self.assertEqual(summary["fraction_zero"], 0.0)
        band = summary["reduced_coverage_band_px"]
        self.assertGreaterEqual(band["median"], 6)      # rows 22..29 have coverage 1 -> ~8 px band (2-px sampling)
        self.assertLessEqual(band["median"], 10)
        # a frame shifted the other way reduces coverage at the top of the image too; that must not count as the band
        transforms["C"] = np.array([[1, 0, 0], [0, 1, 6], [0, 0, 1]], dtype=float)   # C's content 6 px lower: top rows lack C
        cov3 = sv.coverage_map(valid, transforms, None, step=2)
        summary3 = sv.coverage_summary(cov3, mask, step=2, frames=3)
        self.assertLessEqual(summary3["reduced_coverage_band_px"]["median"], 10)

    def test_per_frame_source_only_removes_sky_and_falls_back_to_static(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            mask = np.ones((40, 60), dtype=np.float32); mask[30:, :] = 0.0
            static = sv.StaticMaskSource(mask, dilation_px=2)
            own = np.full((40, 60), 255, dtype=np.uint8); own[5:10, 20:30] = 0      # a transient occluder in the sky
            own[20:, :] = 255                                                          # (tries to add sky over the ground: ignored)
            Image.fromarray(own).save(Path(d) / "light_00002.png")
            source = sv.PerFrameMaskSource(static, {"light_00002": Path(d) / "light_00002.png"}, 60, 40)
            v1, v2 = source.for_frame("light_00001"), source.for_frame("light_00002")
            np.testing.assert_array_equal(v1, static.validity)
            self.assertFalse(v2[3:12, 18:32].any())            # occluder plus 2-px dilation
            self.assertTrue(v2[:3, :].all())
            self.assertFalse(v2[28:, :].any())                 # the ground stays invalid: a frame mask cannot add sky
            self.assertEqual(source.describe()["source"], "job_mask_per_frame")
            self.assertEqual(source.describe()["own_mask_count"], 1)
            self.assertEqual(static.describe()["source"], "job_mask_static")
            transforms = {"light_00001": np.eye(3), "light_00002": np.eye(3), "light_00003": np.eye(3)}
            cov = sv.coverage_map(source, transforms, None, step=1)
            self.assertEqual(int(cov[7, 25]), 2)               # frame 2 does not observe the occluded patch
            self.assertEqual(int(cov[7, 5]), 3)
            self.assertEqual(int(cov[35, 5]), 0)
            by = source.by_frame(transforms)
            cov2 = sv.coverage_map(by, transforms, None, step=1)
            np.testing.assert_array_equal(cov, cov2)

    def test_coverage_precomputed_forward_map_matches_per_frame_map(self):
        valid = self._validity()
        model = RadialDistortion(-0.02, 0.005, 30.0, 20.0, 36.0, magnification=0.98, direction="u_to_p")
        transforms = {"A": np.eye(3), "B": np.array([[1, 0, 1.5], [0, 1, 7], [0, 0, 1]], dtype=float)}
        pts = np.column_stack(np.mgrid[0:40:3, 0:60:3][::-1]).reshape(-1, 2).astype(np.float64)
        u = model.apply(pts)
        for h in transforms.values():
            a = sv.frame_to_reference_native(pts, h, model)
            b = sv.frame_to_reference_native(pts, h, model, undistorted_ref=u)
            np.testing.assert_allclose(a, b, atol=1e-9)

    def test_coverage_png_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            cov = np.arange(12, dtype=np.int32).reshape(3, 4) * 100
            sv.write_coverage_png(cov, Path(d) / "c.png")
            np.testing.assert_array_equal(sv.read_coverage_png(Path(d) / "c.png"), cov)


if __name__ == "__main__":
    unittest.main()
