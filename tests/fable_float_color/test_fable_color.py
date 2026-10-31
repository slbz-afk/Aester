"""Independent counterexamples for raw_color.py and color_preview.py.

Expectations are computed here from the documented conventions (LibRaw 0.22.1
cam_xyz_coeff: cam_rgb = cam_xyz @ xyz_rgb, row-normalise, invert; WB gains
G1 = 1 over one common DN scale; display: signed area average -> clip -> EV ->
luminance asinh -> sRGB encode), not by calling the functions under test.
Run with ``pytest`` or ``python -m unittest``.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Product integration: normal package import; original Fable snapshot is untouched.

from starstack import raw_color  # noqa: E402
from starstack.color_preview import preview_recipe, render_preview, save_preview  # noqa: E402

# Linear sRGB (D65) -> XYZ, the values verified against LibRaw 0.22.1
# src/tables/colorconst.cpp (xyz_rgb) during this review.
XYZ_FROM_SRGB = np.array([[.4124564, .3575761, .1804375],
                          [.2126729, .7151522, .0721750],
                          [.0193339, .1191920, .9503041]])
# A realistic XYZ -> camera matrix (Adobe-style ColorMatrix of a Nikon-like sensor).
CAM_XYZ = np.array([[1.1423, -.4564, -.1123],
                    [-.4816, 1.2895, .2119],
                    [-.0210, .1061, .7282],
                    [0., 0., 0.]])
SHA = "0123456789abcdef" * 4


def metadata(cam_xyz=CAM_XYZ, wb=(2.0, 1.0, 1.5, 0.0), pattern=((0, 1), (3, 2)),
             num_colors=3, color_desc="RGBG", versions=None):
    values = {"num_colors": num_colors, "color_desc": color_desc,
              "raw_pattern": [list(r) for r in pattern],
              "rgb_xyz_matrix": np.asarray(cam_xyz).tolist(),
              "camera_whitebalance": list(wb)}
    return {"available": True, "status": "ok", "source": "libraw_reported",
            "white_balance_applied": False, "color_correction_applied": False,
            "versions": versions or {"rawpy": "0.27.1", "libraw": [0, 22, 1]},
            "fields": {k: {"value": v} for k, v in values.items()}}


def expected_matrix(cam_xyz=CAM_XYZ):
    """Reviewer's own construction of the documented LibRaw convention."""
    cam_rgb = np.asarray(cam_xyz)[:3] @ XYZ_FROM_SRGB
    row_sums = cam_rgb.sum(axis=1)
    return np.linalg.inv(cam_rgb / row_sums[:, None]), row_sums


def forward_camera(rgb_linear, cam_xyz=CAM_XYZ):
    """Linear sRGB colour -> raw camera response (unbalanced) via XYZ."""
    return np.asarray(cam_xyz)[:3] @ (XYZ_FROM_SRGB @ np.asarray(rgb_linear, float))


class MatrixDirectionAndNormalisation(unittest.TestCase):
    def test_non_neutral_colours_round_trip_through_forward_chain(self):
        result = raw_color.cam_xyz_to_rgb_cam(CAM_XYZ)
        matrix = np.asarray(result["values"])
        _, row_sums = expected_matrix()
        for rgb in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [.2, .5, .8], [.9, .1, .3], [.02, .02, .02], [1, 1, 1]):
            camera = forward_camera(rgb)          # what the sensor would measure
            balanced = camera / row_sums          # WB that maps sRGB white to (1,1,1)
            with self.subTest(rgb=rgb):
                np.testing.assert_allclose(matrix @ balanced, rgb, atol=1e-12)
                # The matrix is for balanced input: feeding raw camera values is wrong.
                if rgb != [1, 1, 1] and len(set(rgb)) > 1:
                    self.assertFalse(np.allclose(matrix @ camera, rgb, atol=1e-3))
        np.testing.assert_allclose(result["row_sums"], row_sums, atol=1e-15)
        np.testing.assert_allclose(matrix, expected_matrix()[0], atol=1e-12)
        np.testing.assert_allclose(matrix @ np.ones(3), np.ones(3), atol=1e-12)  # neutral preserved

    def test_row_scaling_of_cam_xyz_does_not_change_the_matrix(self):
        scaled = CAM_XYZ.copy()
        scaled[:3] *= np.array([[.5], [2.], [1.25]])
        a = np.asarray(raw_color.cam_xyz_to_rgb_cam(CAM_XYZ)["values"])
        b = np.asarray(raw_color.cam_xyz_to_rgb_cam(scaled)["values"])
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_degenerate_models_rejected(self):
        zero_row = CAM_XYZ.copy()
        zero_row[1] = 0.
        fourth = CAM_XYZ.copy()
        fourth[3] = [.1, .2, .3]
        singular = CAM_XYZ.copy()
        singular[2] = singular[0] * 1.000000001
        for bad in (zero_row, fourth, singular, np.zeros((4, 3)), CAM_XYZ[:3], np.full((4, 3), np.nan)):
            with self.subTest():
                with self.assertRaises(ValueError):
                    raw_color.cam_xyz_to_rgb_cam(bad)

    def test_librAW_row_sum_threshold_boundary(self):
        # LibRaw zeroes rows whose sum is <= 0.00001; the module refuses them instead.
        cam_rgb_rows = np.linalg.inv(XYZ_FROM_SRGB)  # rows of cam_xyz that give cam_rgb = I
        tiny = np.vstack((cam_rgb_rows * np.array([[5e-6], [1.], [1.]]), np.zeros(3)))
        with self.assertRaises(ValueError):
            raw_color.cam_xyz_to_rgb_cam(tiny)
        just_above = np.vstack((cam_rgb_rows * np.array([[2e-5], [1.], [1.]]), np.zeros(3)))
        self.assertEqual(len(raw_color.cam_xyz_to_rgb_cam(just_above)["values"]), 3)


class WhiteBalanceScaleAndApplication(unittest.TestCase):
    def test_apply_matches_reviewer_reference_and_keeps_signed_and_overrange(self):
        recipe = raw_color.build_recipe(metadata(), SHA, 1023, 10.25)
        self.assertEqual(recipe["scale"]["common_denominator_DN"], 1023 - 10.25)
        self.assertEqual(recipe["white_balance"]["gains_rgb"], [2.0, 1.0, 1.5])
        rng = np.random.default_rng(2)
        camera = rng.normal(300, 700, (13, 9, 3)).astype(np.float32)
        camera[0, 0] = [-500., 60000., 3.]
        before = camera.copy()
        out = raw_color.apply_color(camera, recipe, block_rows=4)
        matrix, _ = expected_matrix()
        expected = (camera.astype(np.float64) * np.array([2., 1., 1.5]) / (1023 - 10.25)) @ matrix.T
        np.testing.assert_allclose(out, expected, rtol=3e-6, atol=3e-6)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue((out < 0).any() and (out > 1).any())
        np.testing.assert_array_equal(camera.view(np.uint32), before.view(np.uint32))

    def test_neutral_patch_at_reference_illuminant_maps_to_grey(self):
        recipe = raw_color.build_recipe(metadata(wb=(1.8, 1.0, 1.3, 1.0)), SHA, common_scale=5000)
        # A neutral object gives camera DN proportional to 1/gain per channel.
        level = 2500.
        camera = np.array([[[level / 1.8, level, level / 1.3]]], np.float32)
        out = raw_color.apply_color(camera, recipe)
        np.testing.assert_allclose(out[0, 0], [.5, .5, .5], rtol=1e-6)

    def test_wb_is_applied_before_the_matrix_not_after(self):
        recipe = raw_color.build_recipe(metadata(), SHA, common_scale=1000)
        matrix, _ = expected_matrix()
        gains = np.array([2., 1., 1.5])
        camera = np.array([[[400., 300., 200.]]], np.float32)
        out = raw_color.apply_color(camera, recipe)
        right = matrix @ (gains * camera[0, 0] / 1000)
        wrong = gains * (matrix @ (camera[0, 0] / 1000))
        np.testing.assert_allclose(out[0, 0], right, rtol=2e-6)
        self.assertFalse(np.allclose(out[0, 0], wrong, rtol=1e-3))

    def test_second_green_slot_conventions(self):
        ok_zero = raw_color.build_recipe(metadata(wb=(2., 1., 1.5, 0.)), SHA, common_scale=1000)
        self.assertEqual(ok_zero["white_balance"]["green_mapping"], "explicit_LibRaw_zero_G2_uses_G1")
        ok_equal = raw_color.build_recipe(metadata(wb=(2., 1., 1.5, 1.)), SHA, common_scale=1000)
        self.assertEqual(ok_equal["white_balance"]["green_mapping"], "reported_G1_equals_G2")
        self.assertEqual(ok_zero["white_balance"]["gains_rgb"], ok_equal["white_balance"]["gains_rgb"])
        for wb in ((2., 1., 1.5, 1.01), (2., 1., 1.5, .99), (2., 1., 1.5, -1.), (2., 0., 1.5, 0.),
                   (0., 1., 1.5, 0.), (2., 1., 1.5, float("inf")), (2., 1., 1.5)):
            with self.subTest(wb=wb):
                with self.assertRaises(ValueError):
                    raw_color.build_recipe(metadata(wb=wb), SHA, common_scale=1000)

    def test_gains_normalised_by_green_not_by_minimum(self):
        # LibRaw scale_colors divides pre_mul by its minimum; the recipe normalises
        # by G1. They agree only when green is the smallest coefficient. Record the
        # module's convention explicitly for a blue-heavy WB where B < G.
        recipe = raw_color.build_recipe(metadata(wb=(1.5, 1.0, .8, 0.)), SHA, common_scale=1000)
        self.assertEqual(recipe["white_balance"]["gains_rgb"], [1.5, 1.0, .8])
        self.assertEqual(recipe["white_balance"]["normalization"], "G1_equals_one")

    def test_unsupported_models_rejected(self):
        cases = {
            "four colours": metadata(num_colors=4),
            "RGBE": metadata(color_desc="RGBE"),
            "non-diagonal": metadata(pattern=((0, 2), (1, 3))),
            "no red": metadata(pattern=((1, 1), (3, 2))),
            "fourth row": metadata(cam_xyz=np.vstack((CAM_XYZ[:3], [[.1, .1, .1]]))),
            "wb applied": {**metadata(), "white_balance_applied": True},
            "unavailable": {**metadata(), "available": False},
            "bool count": metadata(num_colors=True),
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    raw_color.build_recipe(bad, SHA, common_scale=1000)
        for slot3 in (((0, 1), (3, 2)), ((3, 0), (2, 1)), ((1, 2), (0, 3))):
            with self.subTest(pattern=slot3):
                recipe = raw_color.build_recipe(metadata(pattern=slot3), SHA, common_scale=1000)
                self.assertEqual(recipe["camera"]["model"]["raw_pattern"], [list(r) for r in slot3])

    def test_scale_arguments(self):
        with self.assertRaises(ValueError):
            raw_color.build_recipe(metadata(), SHA, 1000, 1000)
        with self.assertRaises(ValueError):
            raw_color.build_recipe(metadata(), SHA, 1000, -1)
        with self.assertRaises(ValueError):
            raw_color.build_recipe(metadata(), SHA, 1000, 100, common_scale=899)
        same = raw_color.build_recipe(metadata(), SHA, 1000, 100, common_scale=900)
        self.assertEqual(same["scale"]["common_denominator_DN"], 900)
        with self.assertRaises(ValueError):
            raw_color.build_recipe(metadata(), SHA)


class RecipeFixednessAndValidation(unittest.TestCase):
    def setUp(self):
        self.recipe = raw_color.build_recipe(metadata(), SHA, 1023, 10.25)

    def test_frame_with_other_wb_uses_reference_gains(self):
        other = metadata(wb=(1.2, 1.0, 2.7, 0.))
        validated = raw_color.validate_recipe(self.recipe, other)
        self.assertEqual(validated, self.recipe)
        camera = np.array([[[100., 100., 100.]]], np.float32)
        out = raw_color.apply_color(camera, self.recipe)
        matrix, _ = expected_matrix()
        np.testing.assert_allclose(out[0, 0], matrix @ (np.array([2., 1., 1.5]) * 100 / (1023 - 10.25)), rtol=2e-6)

    def test_frame_with_other_matrix_pattern_or_versions_rejected(self):
        changed_matrix = CAM_XYZ.copy()
        changed_matrix[0, 0] += 1e-4
        for name, frame in {"matrix": metadata(cam_xyz=changed_matrix),
                            "pattern": metadata(pattern=((1, 0), (2, 3))),
                            "versions": metadata(versions={"rawpy": "0.27.2", "libraw": [0, 22, 1]}),
                            "colours": metadata(num_colors=4)}.items():
            with self.subTest(change=name):
                with self.assertRaises(ValueError):
                    raw_color.validate_recipe(self.recipe, frame)

    def test_tampered_recipe_fields_rejected_but_extensions_allowed(self):
        edits = {
            "gains": lambda r: r["white_balance"]["gains_rgb"].__setitem__(0, 2.5),
            "scale": lambda r: r["scale"].__setitem__("common_denominator_DN", 1000.),
            "matrix": lambda r: r["matrix"]["values"][1].__setitem__(1, 1.0),
            "direction": lambda r: r["matrix"].__setitem__("direction", "other"),
            "reference": lambda r: r["reference"].__setitem__("white_DN", 2047.),
            "conditions": lambda r: r["conditions"].__setitem__("input_black_subtracted_required", False),
            "output": lambda r: r["output"].__setitem__("clipping_applied", True),
            "fingerprint": lambda r: r["camera"].__setitem__("metadata_fingerprint", "0" * 64),
            "model": lambda r: r["camera"]["model"]["cam_xyz"][0].__setitem__(0, 1.2),
            "schema": lambda r: r.__setitem__("schema_version", 2),
        }
        for name, edit in edits.items():
            recipe = copy.deepcopy(self.recipe)
            edit(recipe)
            with self.subTest(edit=name):
                with self.assertRaises(ValueError):
                    raw_color.validate_recipe(recipe)
        extended = copy.deepcopy(self.recipe)
        extended["frame_compatibility"] = {"anything": 1}
        extended["notes"] = "extra"
        self.assertEqual(raw_color.validate_recipe(extended), extended)

    def test_one_ulp_matrix_difference_preserves_recorded_values(self):
        """F1 revised contract: roundoff is accepted, stored coefficients stay fixed."""
        recipe = copy.deepcopy(self.recipe)
        value = recipe["matrix"]["values"][0][0]
        recipe["matrix"]["values"][0][0] = float(np.nextafter(value, value + 1))
        self.assertEqual(raw_color.validate_recipe(recipe), recipe)
        recipe = copy.deepcopy(self.recipe)
        recipe["matrix"]["condition_number"] = float(np.nextafter(recipe["matrix"]["condition_number"], 10))
        self.assertEqual(raw_color.validate_recipe(recipe), recipe)

    def test_recipe_is_strict_json_and_round_trips(self):
        text = json.dumps(self.recipe, allow_nan=False)
        self.assertEqual(raw_color.validate_recipe(json.loads(text)), self.recipe)


class ApplyColorContracts(unittest.TestCase):
    def setUp(self):
        self.recipe = raw_color.build_recipe(metadata(), SHA, common_scale=1000)
        self._tmp = tempfile.TemporaryDirectory(prefix="fable_review_color_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_wrong_dtype_shape_state_or_nonfinite_rejected(self):
        good = np.ones((2, 3, 3), np.float32)
        bad_inputs = [np.ones((2, 3, 3), np.float64), np.ones((2, 3, 4), np.float32),
                      np.ones((2, 3), np.float32), np.ones((2, 3, 3), np.uint16)]
        for bad in bad_inputs:
            with self.subTest(bad=(bad.dtype, bad.shape)):
                with self.assertRaises(ValueError):
                    raw_color.apply_color(bad, self.recipe)
        nan = good.copy()
        nan[1, 2, 0] = np.nan
        with self.assertRaises(ValueError):
            raw_color.apply_color(nan, self.recipe, block_rows=1)
        with self.assertRaises(ValueError):
            raw_color.apply_color(good, self.recipe, row_order="BOTTOM-UP")
        with self.assertRaises(ValueError):
            raw_color.apply_color(good, self.recipe, input_state="camera_rgb_with_black")

    def test_float32_overflow_raises_value_error(self):
        """F4 revised contract: rejected arithmetic has a consistent ValueError type."""
        recipe = raw_color.build_recipe(metadata(), SHA, common_scale=1.0)
        camera = np.full((1, 1, 3), 3e38, np.float32)
        with self.assertRaises(ValueError):
            raw_color.apply_color(camera, recipe)

    def test_output_aliasing_and_same_file_rejected(self):
        camera = np.ones((2, 2, 3), np.float32)
        with self.assertRaises(ValueError):
            raw_color.apply_color(camera, self.recipe, out=camera)
        path = self.tmp / "same.bin"
        np.zeros(12, np.float32).tofile(path)
        a = np.memmap(path, np.float32, "r", shape=(2, 2, 3))
        b = np.memmap(path, np.float32, "r+", shape=(2, 2, 3))
        try:
            with self.assertRaises(ValueError):
                raw_color.apply_color(a, self.recipe, out=b)
        finally:
            del a, b

    def test_independent_memmap_output_and_tile_independence(self):
        rng = np.random.default_rng(9)
        camera = rng.normal(0, 1000, (11, 7, 3)).astype(np.float32)
        out = np.lib.format.open_memmap(self.tmp / "out.npy", mode="w+", dtype=np.float32, shape=camera.shape)
        try:
            raw_color.apply_color(camera, self.recipe, out=out, block_rows=3)
            out.flush()
            whole = raw_color.apply_color(camera, self.recipe, block_rows=11)
            np.testing.assert_array_equal(np.asarray(out), whole)
        finally:
            del out


class PreviewIsolation(unittest.TestCase):
    """The preview must not alter the master and must average before clipping."""

    @staticmethod
    def encode(linear_rgb, ev):
        """Reviewer implementation of the documented display transform."""
        x = np.maximum(np.asarray(linear_rgb, float), 0) * 2. ** ev
        y = x @ np.array([.2126, .7152, .0722])
        ratio = 8 / np.arcsinh(8) if y <= 0 else np.arcsinh(8 * y) / (np.arcsinh(8) * y)
        x = np.clip(x * ratio, 0, 1)
        srgb = np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
        return np.round(srgb * 255).astype(np.uint8)

    def test_negative_samples_average_before_clipping(self):
        # Four samples average to 0.25; clip-first would give 0.375.
        block = np.array([[-.5, .5], [.5, .5]], np.float32)
        rgb = np.stack([block] * 3, axis=-1)
        recipe = preview_recipe(0)
        pixel = render_preview(rgb, recipe, max_side=1)[0, 0]
        np.testing.assert_array_equal(pixel, self.encode([.25] * 3, 0))
        self.assertFalse(np.array_equal(pixel, self.encode([.375] * 3, 0)))

    def test_exposure_applies_to_linear_values_and_single_encoding(self):
        for value, ev in ((.01, 0), (.03, 2), (.2, -1), (.4, 1), (3., 0)):
            rgb = np.full((2, 2, 3), value, np.float32)
            with self.subTest(value=value, ev=ev):
                rendered = render_preview(rgb, preview_recipe(ev))
                self.assertEqual(rendered.shape, (2, 2, 3))
                np.testing.assert_array_equal(rendered, np.broadcast_to(self.encode([value] * 3, ev), (2, 2, 3)))

    def test_rgb_ratios_preserved_through_tone_curve_then_clipped(self):
        rgb = np.zeros((1, 1, 3), np.float32)
        rgb[0, 0] = [.4, .1, .05]
        pixel = render_preview(rgb, preview_recipe(0))[0, 0]
        np.testing.assert_array_equal(pixel, self.encode([.4, .1, .05], 0))
        bright = np.zeros((1, 1, 3), np.float32)
        bright[0, 0] = [40., 1., .5]
        np.testing.assert_array_equal(render_preview(bright, preview_recipe(0))[0, 0], self.encode([40., 1., .5], 0))
        self.assertEqual(render_preview(bright, preview_recipe(0))[0, 0, 0], 255)

    def test_master_untouched_even_when_read_only_and_memmapped(self):
        rgb = np.random.default_rng(1).normal(0, 2, (9, 14, 3)).astype(np.float32)
        before = rgb.view(np.uint32).copy()
        rgb.flags.writeable = False
        render_preview(rgb, preview_recipe(2), max_side=5)
        np.testing.assert_array_equal(rgb.view(np.uint32), before)
        with tempfile.TemporaryDirectory(prefix="fable_review_preview_") as tmp:
            path = Path(tmp) / "master.npy"
            np.save(path, rgb)
            digest = path.read_bytes()
            mapped = np.load(path, mmap_mode="r")
            try:
                result = save_preview(Path(tmp) / "p.jpg", mapped, preview_recipe(2), max_side=6)
            finally:
                del mapped
            self.assertEqual(path.read_bytes(), digest)
            self.assertEqual(result["dimensions"], [6, 4])
            self.assertTrue(result["display_only"])
            from PIL import Image
            with Image.open(Path(tmp) / "p.jpg") as im:
                self.assertEqual(im.size, (6, 4))
                self.assertTrue(im.info.get("icc_profile"))

    def test_recipe_and_input_validation(self):
        rgb = np.ones((2, 2, 3), np.float32)
        for ev in (12.5, -13, float("nan"), True, "2"):
            with self.subTest(ev=ev):
                with self.assertRaises(ValueError):
                    preview_recipe(ev)
        tampered = preview_recipe(0)
        tampered["asinh_strength"] = 4.0
        with self.assertRaises(ValueError):
            render_preview(rgb, tampered)
        with self.assertRaises(ValueError):
            render_preview(rgb.astype(np.float64), preview_recipe(0))
        bad = rgb.copy()
        bad[0, 0, 1] = np.inf
        with self.assertRaises(ValueError):
            render_preview(bad, preview_recipe(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
