"""Independent end-to-end and integrity counterexamples for raw_develop.py,
raw_audit.read_dng_tags / black_model / subtract_declared_black.

A tiny synthetic CFA DNG is written with tifffile (no rawpy, no photographs).
The audit record is built by hand in the documented schema; the expected
master values come from a scalar demosaic reference and the documented colour
chain written here, never from the functions under test.
Run with ``pytest`` or ``python -m unittest``.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

# Product integration: normal package import; original Fable snapshot is untouched.

from starstack import core, raw_audit, raw_develop  # noqa: E402

XYZ_FROM_SRGB = np.array([[.4124564, .3575761, .1804375],
                          [.2126729, .7151522, .0721750],
                          [.0193339, .1191920, .9503041]])
CAM_XYZ = [[1.1423, -.4564, -.1123], [-.4816, 1.2895, .2119], [-.0210, .1061, .7282], [0., 0., 0.]]
PATTERN = [[0, 1], [3, 2]]           # LibRaw slot encoding of RGGB
BLACK = [[10.25, 9.75], [10.75, 10.]]  # DNG BlackLevel 2x2, top-left origin
WHITE = 1023
H, W = 10, 12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dng_tags(*, black_rational=((41, 4), (39, 4), (43, 4), (40, 4)), repeat=(2, 2), white=WHITE,
             active=(0, 0, H, W), extra=()):
    flat_black = tuple(v for pair in black_rational for v in pair)
    cm = [11423, -4564, -1123, -4816, 12895, 2119, -210, 1061, 7282]
    tags = [(50706, "B", 4, (1, 4, 0, 0), False), (50707, "B", 4, (1, 3, 0, 0), False),
            (33421, "H", 2, (2, 2), False), (33422, "B", 4, (0, 1, 1, 2), False),
            (50710, "B", 3, (0, 1, 2), False),
            (50713, "H", 2, repeat, False), (50714, "2I", len(black_rational), flat_black, False),
            (50717, "I", 1, white, False), (50829, "I", 4, active, False),
            (50721, "2i", 9, tuple(v for c in cm for v in (c, 10000)), False),
            (50778, "H", 1, 21, False), (50728, "2I", 3, (1, 2, 1, 1, 2, 3), False)]
    return tags + list(extra)


def write_dng(path: Path, codes: np.ndarray, **kwargs) -> None:
    tifffile.imwrite(path, codes, photometric=32803, metadata=None, extratags=dng_tags(**kwargs))


def synthetic_codes(seed=20260916) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 1400, size=(H, W)).astype(np.uint16)  # some below black, some >= white
    codes[0, 0], codes[1, 1], codes[H - 1, W - 1] = 3, 1400, 0
    return codes


def libraw_metadata(wb=(2., 1., 1.5, 0.), cam_xyz=CAM_XYZ, pattern=PATTERN):
    values = {"num_colors": 3, "color_desc": "RGBG", "raw_pattern": pattern,
              "rgb_xyz_matrix": cam_xyz, "camera_whitebalance": list(wb)}
    return {"available": True, "status": "ok", "source": "libraw_reported",
            "white_balance_applied": False, "color_correction_applied": False,
            "versions": {"rawpy": "0.27.1", "libraw": [0, 22, 1]},
            "fields": {k: {"value": v} for k, v in values.items()}}


def expected_black_model():
    return {"status": "available", "source": "file_tag", "black_values": [10.25, 9.75, 10.75, 10.],
            "repeat_shape": [2, 2], "white_level": float(WHITE), "origin": "full_raw_image_top_left",
            "interpretation": "DNG_declared_black_not_independently_measured", "reasons": []}


def reference_signal(codes: np.ndarray) -> np.ndarray:
    black = np.tile(np.asarray(BLACK), (H // 2, W // 2))
    return (codes.astype(np.float64) - black).astype(np.float32)


def colour(pattern, y, x):
    c = pattern[y % 2][x % 2]
    return 1 if c == 3 else c


def reference_demosaic(cfa: np.ndarray, pattern) -> np.ndarray:
    h, w = cfa.shape
    out = np.empty((h, w, 3), np.float32)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                if colour(pattern, y, x) == ch:
                    out[y, x, ch] = cfa[y, x]
                    continue
                num, den = 0.0, 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        sy, sx = y + dy, x + dx
                        if 0 <= sy < h and 0 <= sx < w and colour(pattern, sy, sx) == ch:
                            wgt = (2 - abs(dy)) * (2 - abs(dx))
                            num += float(cfa[sy, sx]) * wgt
                            den += wgt
                out[y, x, ch] = np.float32(num / den)
    return out


def reference_colour_matrix(cam_xyz=CAM_XYZ) -> np.ndarray:
    cam_rgb = np.asarray(cam_xyz)[:3] @ XYZ_FROM_SRGB
    return np.linalg.inv(cam_rgb / cam_rgb.sum(axis=1)[:, None])


class Workspace:
    """One synthetic source DNG + one audit directory in the documented schema."""

    def __init__(self, root: Path, name="frame", codes=None, wb=(2., 1., 1.5, 0.), cam_xyz=CAM_XYZ, **dng):
        self.codes = synthetic_codes() if codes is None else codes
        self.source_dir = root / f"{name}_source"
        self.source_dir.mkdir()
        self.source = self.source_dir / f"{name}.dng"
        write_dng(self.source, self.codes, **dng)
        self.audit_dir = root / f"{name}_audit"
        self.audit_dir.mkdir()
        np.save(self.audit_dir / "raw_codes.npy", self.codes, allow_pickle=False)
        file_metadata = raw_audit.read_dng_tags(self.source)  # fixture: the record must hold this reading
        self.audit = {
            "schema_version": 1, "state": "succeeded", "audit_dir": str(self.audit_dir),
            "source": {"path": str(self.source), "sha256": sha256(self.source),
                       "size_bytes": self.source.stat().st_size},
            "geometry": {"width": W, "height": H, "row_order": "TOP-DOWN",
                         "coordinate_basis": "decoder_full_raw_mosaic", "orientation_applied": False,
                         "full_pattern": PATTERN, "visible_pattern": PATTERN,
                         "active_area_libraw": [0, 0, H, W], "pattern_source": "libraw_reported"},
            "libraw_metadata": libraw_metadata(wb=wb, cam_xyz=cam_xyz),
            "file_metadata": file_metadata,
            "black_model": raw_audit.black_model(file_metadata, self.codes.shape),
            "artifacts": {"raw_codes": {"path": str(self.audit_dir / "raw_codes.npy"),
                                        "sha256": sha256(self.audit_dir / "raw_codes.npy"),
                                        "shape": [H, W], "dtype": "uint16"}},
        }
        self.write_audit()

    def write_audit(self):
        core.atomic_json(self.audit_dir / "audit.json", self.audit)

    def snapshot(self) -> dict:
        """Bytes of everything a development must leave untouched."""
        files = {p: p.read_bytes() for p in (*self.audit_dir.iterdir(), *self.source_dir.iterdir()) if p.is_file()}
        return {str(p): hashlib.sha256(b).hexdigest() for p, b in files.items()}


class DngTagsAndBlackModel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="fable_review_dng_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rational_black_levels_and_model_match_hand_expectation(self):
        path = self.tmp / "a.dng"
        write_dng(path, synthetic_codes())
        dng = raw_audit.read_dng_tags(path)
        self.assertEqual(dng["status"], "ok")
        self.assertEqual(dng["raw_shape"], [H, W])
        self.assertEqual(dng["tags"]["BlackLevel"]["numeric"], [10.25, 9.75, 10.75, 10.])
        self.assertEqual(dng["tags"]["BlackLevel"]["value"], [41, 4, 39, 4, 43, 4, 40, 4])
        self.assertEqual(dng["tags"]["ColorMatrix1"]["numeric"][:3], [1.1423, -.4564, -.1123])
        self.assertFalse(dng["tags"]["LinearizationTable"]["present"])
        self.assertEqual(raw_audit.black_model(dng, (H, W)), expected_black_model())

    def test_mosaic_ifd_owns_black_and_white_not_a_preview_ifd(self):
        path = self.tmp / "two_ifd.dng"
        preview = np.zeros((4, 4, 3), np.uint8)
        decoy = [(50706, "B", 4, (1, 4, 0, 0), False), (50714, "I", 1, 999, False), (50713, "H", 2, (1, 1), False),
                 (50717, "I", 1, 1, False), (50829, "I", 4, (0, 0, 4, 4), False),
                 (50721, "2i", 9, tuple(v for c in (10000, 0, 0, 0, 10000, 0, 0, 0, 10000) for v in (c, 10000)), False)]
        real = [t for t in dng_tags() if t[0] not in (50706, 50707, 50721)]  # colour matrix only in IFD0
        with tifffile.TiffWriter(path) as writer:
            writer.write(preview, photometric="rgb", metadata=None, extratags=decoy)
            writer.write(synthetic_codes(), photometric=32803, metadata=None, extratags=real)
        dng = raw_audit.read_dng_tags(path)
        self.assertEqual(dng["status"], "ok", dng["errors"])
        with tifffile.TiffFile(path) as tf:
            first, second = (int(p.offset) for p in tf.pages)
        self.assertEqual(dng["raw_ifd_offset"], second)
        self.assertEqual(dng["tags"]["BlackLevel"]["ifd_offset"], second)
        self.assertEqual(dng["tags"]["BlackLevel"]["numeric"], [10.25, 9.75, 10.75, 10.])
        self.assertEqual(dng["tags"]["WhiteLevel"]["value"], WHITE)
        self.assertEqual(dng["tags"]["ActiveArea"]["value"], [0, 0, H, W])
        self.assertEqual(dng["tags"]["ColorMatrix1"]["ifd_offset"], first)
        self.assertEqual(dng["tags"]["DNGVersion"]["ifd_offset"], first)
        self.assertEqual(raw_audit.black_model(dng, (H, W)), expected_black_model())

    def test_unsupported_calibration_metadata_is_refused_not_guessed(self):
        base = synthetic_codes()
        cases = {
            "linearization": dict(extra=[(50712, "H", 4, (0, 1, 2, 3), False)]),
            "black_delta_v": dict(extra=[(50716, "2i", 2, (1, 1, 1, 1), False)]),
            "opcode_list_1": dict(extra=[(51008, "B", 4, (0, 0, 0, 1), False)]),
            "opcode_list_2": dict(extra=[(51009, "B", 4, (0, 0, 0, 1), False)]),
            "cropped_active_area": dict(active=(0, 2, H, W)),
            "black_at_white": dict(black_rational=((WHITE, 1), (1, 1), (1, 1), (1, 1))),
            "repeat_3x1": dict(repeat=(3, 1), black_rational=((1, 1), (2, 1), (3, 1))),
        }
        for name, kwargs in cases.items():
            path = self.tmp / f"{name}.dng"
            write_dng(path, base, **kwargs)
            dng = raw_audit.read_dng_tags(path)
            with self.subTest(case=name):
                model = raw_audit.black_model(dng, (H, W))
                self.assertEqual(model["status"], "unsupported", model)
                self.assertTrue(model["reasons"])
                with self.assertRaises(ValueError):
                    raw_audit.subtract_declared_black(base, model)
        # Scalar black, 1x2 repeat, and (documented boundary) an OpcodeList3 are accepted.
        for name, kwargs in {"scalar": dict(repeat=(1, 1), black_rational=((1008, 1),)),
                             "one_by_two": dict(repeat=(1, 2), black_rational=((10, 1), (11, 1))),
                             "opcode_list_3": dict(extra=[(51022, "B", 4, (0, 0, 0, 1), False)])}.items():
            path = self.tmp / f"{name}.dng"
            write_dng(path, base, **kwargs)
            with self.subTest(case=name):
                self.assertEqual(raw_audit.black_model(raw_audit.read_dng_tags(path), (H, W))["status"], "available")
        with self.subTest(case="shape_mismatch"):
            path = self.tmp / "shape.dng"
            write_dng(path, base)
            self.assertEqual(raw_audit.black_model(raw_audit.read_dng_tags(path), (H, W + 2))["status"], "unsupported")
        with self.subTest(case="not_dng_suffix"):
            path = self.tmp / "frame.arw"
            write_dng(path, base)
            self.assertEqual(raw_audit.read_dng_tags(path)["status"], "not_dng")

    def test_black_subtraction_positions_dtype_and_rounding(self):
        codes = synthetic_codes()
        signal = raw_audit.subtract_declared_black(codes, expected_black_model())
        self.assertEqual(signal.dtype, np.float32)
        np.testing.assert_array_equal(signal.view(np.uint32), reference_signal(codes).view(np.uint32))
        self.assertTrue((signal < 0).any() and (signal >= WHITE - 10.75).any())
        # 1x2 and 2x1 repeats index (row parity, column parity) with top-left origin.
        for repeat, values in (([1, 2], [5., 7.]), ([2, 1], [5., 7.])):
            model = {**expected_black_model(), "repeat_shape": repeat, "black_values": values}
            out = raw_audit.subtract_declared_black(codes, model)
            levels = np.tile(np.asarray(values).reshape(repeat), (H // repeat[0], W // repeat[1]))
            with self.subTest(repeat=repeat):
                np.testing.assert_array_equal(out, (codes.astype(np.float64) - levels).astype(np.float32))
        # A non-dyadic black is subtracted in float64 and rounded once to float32.
        model = {**expected_black_model(), "repeat_shape": [1, 1], "black_values": [2047.6]}
        out = raw_audit.subtract_declared_black(codes, model)
        np.testing.assert_array_equal(out, (codes.astype(np.float64) - 2047.6).astype(np.float32))
        for bad in (codes.astype(np.float32), codes.astype(np.uint32), codes.astype(np.int8)[None]):
            with self.assertRaises(ValueError):
                raw_audit.subtract_declared_black(bad, expected_black_model())


class DevelopEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="fable_review_develop_")
        self.tmp = Path(self._tmp.name)
        self.ws = Workspace(self.tmp)
        self.out = self.tmp / "outputs"

    def tearDown(self):
        self._tmp.cleanup()

    def develop(self, ws=None, **kwargs):
        return raw_develop.develop_raw((ws or self.ws).audit_dir, self.out, **kwargs)

    def test_masters_match_reference_chain_and_inputs_stay_untouched(self):
        before = self.ws.snapshot()
        result = self.develop()
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        self.assertEqual(self.ws.snapshot(), before)
        target = Path(result["development_dir"])
        self.assertTrue(target.is_relative_to(self.out))
        artifacts = result["artifacts"]
        for key in ("camera_rgb", "linear_srgb", "preview", "color_recipe", "audit_snapshot"):
            path = Path(artifacts[key]["path"])
            with self.subTest(artifact=key):
                self.assertEqual(path.parent, target)
                self.assertEqual(sha256(path), artifacts[key]["sha256"])
        camera = tifffile.imread(artifacts["camera_rgb"]["path"])
        expected_camera = reference_demosaic(reference_signal(self.ws.codes), PATTERN)
        np.testing.assert_array_equal(camera.view(np.uint32), expected_camera.view(np.uint32))
        self.assertEqual(artifacts["camera_rgb"]["negative_samples"], int((expected_camera < 0).sum()))
        self.assertGreater(artifacts["camera_rgb"]["negative_samples"], 0)
        recipe = json.loads(Path(artifacts["color_recipe"]["path"]).read_text(encoding="utf-8"))
        scale = WHITE - (9.75 + 10.75) / 2  # white minus the mean of the two green-site blacks
        self.assertEqual(recipe["scale"]["common_denominator_DN"], scale)
        self.assertEqual(recipe["reference"]["source_sha256"], sha256(self.ws.source))
        self.assertEqual(recipe["white_balance"]["gains_rgb"], [2., 1., 1.5])
        self.assertEqual(recipe["frame_compatibility"]["black_model"], expected_black_model())
        linear = tifffile.imread(artifacts["linear_srgb"]["path"])
        expected_linear = (expected_camera.astype(np.float64) * np.array([2., 1., 1.5]) / scale) @ reference_colour_matrix().T
        np.testing.assert_allclose(linear, expected_linear, rtol=4e-6, atol=4e-6)
        self.assertTrue((linear < 0).any() and (linear > 1).any())
        self.assertEqual(artifacts["linear_srgb"]["negative_samples"], int((linear < 0).sum()))
        self.assertEqual(artifacts["linear_srgb"]["samples_above_numeric_one"], int((linear > 1).sum()))
        self.assertIsNone(artifacts["linear_srgb"]["icc_profile"])
        with tifffile.TiffFile(artifacts["linear_srgb"]["path"]) as tf:
            description = json.loads(tf.pages[0].description)
            self.assertEqual(description["transfer"], "linear")
            self.assertEqual(tf.pages[0].dtype, np.float32)
            self.assertNotIn("InterColorProfile", tf.pages[0].tags)
        from PIL import Image
        with Image.open(artifacts["preview"]["path"]) as im:
            self.assertEqual(im.size, (W, H))
            self.assertTrue(im.info.get("icc_profile"))
        self.assertTrue(artifacts["preview"]["display_only"])
        self.assertEqual(result["processing"]["black_subtractions"], 1)
        self.assertFalse(result["processing"]["photographic_quality_accepted"])
        saved = json.loads((target / "development.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["state"], "succeeded")
        self.assertTrue(saved["source_unchanged"])

    def test_reused_recipe_fixes_wb_across_frames_and_rejects_incompatible_frames(self):
        first = self.develop()
        recipe_path = first["artifacts"]["color_recipe"]["path"]
        # Same frame again with the recipe: bitwise identical masters.
        second = self.develop(recipe_path=recipe_path, exposure_ev=0)
        self.assertEqual(second["state"], "succeeded", second.get("error"))
        self.assertEqual(second["recipe_mode"], "reused_reference")
        for key in ("camera_rgb", "linear_srgb"):
            a = tifffile.imread(first["artifacts"][key]["path"])
            b = tifffile.imread(second["artifacts"][key]["path"])
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        # A later frame with a different as-shot WB gets the reference gains, not its own.
        other = Workspace(self.tmp, name="other_wb", codes=self.ws.codes, wb=(1.1, 1., 2.9, 0.))
        third = self.develop(other, recipe_path=recipe_path)
        self.assertEqual(third["state"], "succeeded", third.get("error"))
        np.testing.assert_array_equal(tifffile.imread(third["artifacts"]["linear_srgb"]["path"]).view(np.uint32),
                                      tifffile.imread(first["artifacts"]["linear_srgb"]["path"]).view(np.uint32))
        fresh = self.develop(other)  # without the recipe the frame WB is adopted -> different colours
        self.assertFalse(np.array_equal(tifffile.imread(fresh["artifacts"]["linear_srgb"]["path"]),
                                        tifffile.imread(first["artifacts"]["linear_srgb"]["path"])))
        # Different matrix, CFA phase or declared black/white: refused before any output.
        changed_matrix = [list(r) for r in CAM_XYZ]
        changed_matrix[0][0] += 1e-3
        incompatible = {
            "matrix": Workspace(self.tmp, name="matrix", codes=self.ws.codes, cam_xyz=changed_matrix),
            "black": Workspace(self.tmp, name="black", codes=self.ws.codes, black_rational=((1008, 1),), repeat=(1, 1)),
            "white": Workspace(self.tmp, name="white", codes=self.ws.codes, white=4095),
        }
        for name, ws in incompatible.items():
            existing = set(self.out.iterdir())
            with self.subTest(frame=name):
                with self.assertRaises(ValueError):
                    self.develop(ws, recipe_path=recipe_path)
                self.assertEqual(set(self.out.iterdir()), existing)

    def test_tampering_is_detected_and_the_documented_boundary_is_explicit(self):
        before = self.ws.snapshot()
        # 1. Edited CFA evidence without its sidecar: refused, nothing written.
        codes_path = self.ws.audit_dir / "raw_codes.npy"
        original = codes_path.read_bytes()
        edited = self.ws.codes.copy()
        edited[3, 4] += 1
        np.save(codes_path, edited, allow_pickle=False)
        with self.assertRaises(ValueError):
            self.develop()
        self.assertFalse(self.out.exists())
        # 2. Edited evidence WITH a matching sidecar hash: accepted. Plain hashes are
        #    change detectors, not signatures; this pins the boundary, it is not a pass.
        self.ws.audit["artifacts"]["raw_codes"]["sha256"] = sha256(codes_path)
        self.ws.write_audit()
        result = self.develop()
        self.assertEqual(result["state"], "succeeded")
        camera = tifffile.imread(result["artifacts"]["camera_rgb"]["path"])
        self.assertEqual(camera[3, 4, colour(PATTERN, 3, 4)], reference_signal(edited)[3, 4])
        self.assertEqual(result["source"]["sha256"], sha256(self.ws.source))  # still claims the untouched DNG
        codes_path.write_bytes(original)
        self.ws.audit["artifacts"]["raw_codes"]["sha256"] = sha256(codes_path)
        self.ws.write_audit()
        # 3. Edited audit black model contradicting the file tags: refused.
        self.ws.audit["black_model"]["black_values"] = [0., 0., 0., 0.]
        self.ws.write_audit()
        with self.assertRaises(ValueError):
            self.develop()
        self.ws.audit["black_model"] = expected_black_model()
        self.ws.write_audit()
        # 4. Rewritten DNG tags (same size not required): size or hash mismatch refused.
        dng_bytes = self.ws.source.read_bytes()
        write_dng(self.ws.source, self.ws.codes, white=4095)
        with self.assertRaises(ValueError):
            self.develop()
        self.ws.source.write_bytes(dng_bytes)
        # 5. Foreign raw_codes path, wrong row order, wrong pattern basis: refused.
        for change in ("foreign", "row_order", "basis", "state"):
            audit = json.loads(json.dumps(self.ws.audit))
            if change == "foreign":
                other = self.tmp / "foreign.npy"
                other.write_bytes(original)
                audit["artifacts"]["raw_codes"]["path"] = str(other)
            elif change == "row_order":
                audit["geometry"]["row_order"] = "BOTTOM-UP"
            elif change == "basis":
                audit["geometry"]["coordinate_basis"] = "visible_crop"
            else:
                audit["state"] = "failed"
            core.atomic_json(self.ws.audit_dir / "audit.json", audit)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.develop()
        self.ws.write_audit()
        self.assertEqual(self.ws.snapshot(), before)

    def test_output_location_and_disk_guard(self):
        for bad in (self.ws.source_dir, self.ws.audit_dir / "child"):
            with self.subTest(output=bad):
                with self.assertRaises(ValueError):
                    raw_develop.develop_raw(self.ws.audit_dir, bad)
        usage = collections.namedtuple("usage", "total used free")
        with mock.patch.object(raw_develop.shutil, "disk_usage", lambda _: usage(1, 1, 0)):
            with self.assertRaises(ValueError):
                self.develop()
        self.assertEqual(list(self.out.iterdir()), [])
        with self.assertRaises(ValueError):
            self.develop(exposure_ev=13)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_output_root_inside_the_raw_folder_is_rejected(self):
        """F2 revised contract: RAW folder descendants are excluded as well."""
        with self.assertRaises(ValueError):
            raw_develop.develop_raw(self.ws.audit_dir, self.ws.source_dir / "outputs")
        self.assertFalse((self.ws.source_dir / "outputs").exists())
        # The audit folder's parent (the documented `outputs` layout) stays allowed.
        self.assertEqual(raw_develop.develop_raw(self.ws.audit_dir, self.ws.audit_dir.parent)["state"], "succeeded")

    def test_failures_are_recorded_and_retries_do_not_touch_earlier_products(self):
        def boom(*args, **kwargs):
            raise RuntimeError("injected preview failure")
        with mock.patch.object(raw_develop, "save_preview", boom):
            failed = self.develop()
        self.assertEqual(failed["state"], "failed")
        self.assertIn("injected", failed["error"])
        failed_dir = Path(failed["development_dir"])
        record = json.loads((failed_dir / "development.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "failed")
        self.assertNotIn("preview", record["artifacts"])
        self.assertNotIn("source_unchanged", record)
        failed_bytes = {p.name: p.read_bytes() for p in failed_dir.iterdir()}
        ok = self.develop()
        self.assertEqual(ok["state"], "succeeded")
        self.assertNotEqual(Path(ok["development_dir"]), failed_dir)
        self.assertEqual({p.name: p.read_bytes() for p in failed_dir.iterdir()}, failed_bytes)

    def test_source_or_audit_changed_during_the_run_is_not_reported_as_success(self):
        real = raw_develop.save_preview
        for victim in ("source", "audit"):
            def mutate_then_preview(*args, **kwargs):
                if victim == "source":
                    with self.ws.source.open("ab") as handle:
                        handle.write(b"\0")
                else:
                    self.ws.audit["note"] = "edited while running"
                    self.ws.write_audit()
                return real(*args, **kwargs)
            with mock.patch.object(raw_develop, "save_preview", mutate_then_preview):
                result = self.develop()
            with self.subTest(victim=victim):
                self.assertEqual(result["state"], "failed")
                self.assertIn("哈希", result["error"])
                self.assertNotIn("source_unchanged", result)
            # restore for the next iteration
            self.ws.source.write_bytes(self.ws.source.read_bytes()[:-1]) if victim == "source" else None
            self.ws.audit.pop("note", None)
            self.ws.write_audit()

    def test_complete_recipe_of_a_failed_run_can_be_reused(self):
        def boom(*args, **kwargs):
            raise RuntimeError("injected")
        with mock.patch.object(raw_develop, "save_preview", boom):
            failed = self.develop()
        recipe = Path(failed["development_dir"]) / "color_recipe.json"
        self.assertTrue(recipe.exists())
        # The recipe file itself is complete and valid, so reuse succeeds; the
        # development record of the failed run is what must not be trusted.
        again = self.develop(recipe_path=recipe)
        self.assertEqual(again["state"], "succeeded")
        self.assertEqual(again["recipe_source"]["sha256"], sha256(recipe))


if __name__ == "__main__":
    unittest.main(verbosity=2)
