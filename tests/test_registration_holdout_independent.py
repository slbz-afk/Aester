"""Independent fit-isolation and frozen-evaluation acceptance, with known IDs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from starstack import holdout_catalog as cat
from starstack import registration_holdout as experiment
from starstack import registration_distortion as radial


def scene():
    """Analytic source-to-reference translations, not production warping helpers."""
    train = {f"train-{index:02d}": (float(x), float(y))
             for index, (x, y) in enumerate((x, y) for y in (25, 60, 100, 135)
                                             for x in (25, 70, 120, 170))}
    holdout = {"test-corner-1": (10.0, 10.0), "test-corner-2": (190.0, 150.0),
               "test-center": (100.0, 80.0), "test-other": (100.0, 15.0)}
    points = {**train, **holdout}
    frames = []
    for name, dx, dy in (("reference", 0, 0), ("target-a", 3, -2), ("target-b", -4, 3)):
        frames.append({"id": name, "observations": [
            {"star_id": sid, "x": x-dx, "y": y-dy} for sid, (x, y) in points.items()]})
    return {"version": "1.0.0", "width": 200, "height": 160,
            "reference": "reference", "identity_basis": "synthetic_truth",
            "stars": [{"id": sid, "role": "train" if sid in train else "holdout"}
                      for sid in points], "frames": frames}


class RegistrationHoldoutIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = scene()
        cls.view = cat.training_view(cls.catalog)
        cls.candidate = experiment.fit_training(cls.view)

    def test_all_direct_and_shared_fit_inputs_are_training_only(self):
        view = cat.training_view(scene())
        frames = {f["id"]: {o["star_id"]: (o["x"], o["y"]) for o in f["observations"]}
                  for f in view["frames"]}
        dst_expected = np.array([frames["reference"][sid] for sid in sorted(frames["reference"])])
        expected = {fid: np.array([points[sid] for sid in sorted(points)])
                    for fid, points in frames.items() if fid != "reference"}
        direct_calls = []
        shared_calls = []
        original_direct = experiment.fit_homography
        original_shared = radial.fit_joint_radial

        def direct(source, target):
            self.assertTrue(any(np.array_equal(source, points) for points in expected.values()))
            np.testing.assert_array_equal(target, dst_expected)
            direct_calls.append(len(source))
            return original_direct(source, target)

        def shared(inputs, width, height, **options):
            self.assertEqual(set(inputs), set(expected))
            for fid, frame in inputs.items():
                np.testing.assert_array_equal(frame["src"], expected[fid])
                np.testing.assert_array_equal(frame["dst"], dst_expected)
                np.testing.assert_array_equal(options["masks"][fid], np.ones(len(expected[fid]), bool))
            shared_calls.append(set(inputs))
            return original_shared(inputs, width, height, **options)

        with patch.object(experiment, "fit_homography", side_effect=direct), \
             patch.object(radial, "fit_joint_radial", side_effect=shared):
            result = experiment.fit_training(view)
        self.assertEqual(direct_calls, [16, 16])
        self.assertEqual(len(shared_calls), 1)
        self.assertEqual(set(result["training_ids"]), set(view["stars"]))
        self.assertNotIn("test-corner", json.dumps(result))

    def test_every_holdout_coordinate_can_change_without_changing_candidate(self):
        changed = copy.deepcopy(self.catalog)
        for frame in changed["frames"]:
            for obs in frame["observations"]:
                if obs["star_id"].startswith("test-"):
                    obs["x"], obs["y"] = 199-obs["x"], 159-obs["y"]
        candidate = experiment.fit_training(cat.training_view(changed))
        self.assertEqual(cat.canonical_sha256(candidate), cat.canonical_sha256(self.candidate))
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary)/"a.json", Path(temporary)/"b.json"
            self.assertEqual(experiment.freeze_candidate(a, self.candidate),
                             experiment.freeze_candidate(b, candidate))
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_catalog_order_does_not_change_model(self):
        changed = copy.deepcopy(self.catalog)
        changed["stars"].reverse()
        changed["frames"].reverse()
        for frame in changed["frames"]:
            frame["observations"].reverse()
        candidate = experiment.fit_training(cat.training_view(changed))
        self.assertEqual(candidate, self.candidate)

    def test_training_and_evaluation_leave_input_evidence_unchanged(self):
        view = copy.deepcopy(self.view)
        before = copy.deepcopy(view)
        candidate = experiment.fit_training(view)
        self.assertEqual(view, before)
        model_before = copy.deepcopy(candidate)
        document = copy.deepcopy(self.catalog)
        document_before = copy.deepcopy(document)
        experiment.evaluate_frozen(document, candidate, tolerance_px=1e-5)
        self.assertEqual(candidate, model_before)
        self.assertEqual(document, document_before)

    def test_trainer_rejects_full_catalog_or_holdout_view(self):
        for wrong in (self.catalog, cat.holdout_view(self.catalog)):
            with self.assertRaises(ValueError):
                experiment.fit_training(wrong)

    def test_evaluation_does_not_fit_any_model(self):
        with patch.object(experiment, "fit_homography", side_effect=AssertionError("refit")), \
             patch.object(radial, "fit_joint_radial", side_effect=AssertionError("radial refit")), \
             patch.object(radial, "fit_homography", side_effect=AssertionError("inner refit")):
            report = experiment.evaluate_frozen(self.catalog, self.candidate, tolerance_px=1e-5)
        self.assertEqual(report["judgement"]["status"], "passed")
        for model in report["models"].values():
            for frame in model.values():
                self.assertEqual(len(frame["residuals"]), 4)
                self.assertLess(frame["regions"]["overall"]["max_px"], 1e-5)

    def test_known_identity_large_error_is_not_censored_by_matching_radius(self):
        changed = copy.deepcopy(self.catalog)
        for frame in changed["frames"]:
            if frame["id"] == "target-a":
                for obs in frame["observations"]:
                    if obs["star_id"] == "test-center":
                        obs["x"] += 20.0
        report = experiment.evaluate_frozen(changed, self.candidate, tolerance_px=1.0)
        self.assertEqual(report["judgement"]["status"], "failed")
        for model in report["models"].values():
            entry = model["target-a"]
            self.assertEqual(entry["regions"]["overall"]["measured"], 4)
            point = next(r for r in entry["residuals"] if r["star_id"] == "test-center")
            self.assertAlmostEqual(point["residual_xy"][0], 20.0, places=5)
            self.assertAlmostEqual(point["error_px"], 20.0, places=5)
            self.assertEqual(entry["missing_star_ids"], [])

    def test_missing_observation_is_insufficient_not_a_silent_pass(self):
        changed = copy.deepcopy(self.catalog)
        for frame in changed["frames"]:
            if frame["id"] == "target-a":
                frame["observations"] = [o for o in frame["observations"] if o["star_id"] != "test-center"]
        report = experiment.evaluate_frozen(changed, self.candidate, tolerance_px=1e-5)
        self.assertEqual(report["judgement"]["status"], "insufficient_data")
        for model in report["models"].values():
            self.assertEqual(model["target-a"]["missing_star_ids"], ["test-center"])
            self.assertEqual(model["target-a"]["regions"]["center"]["measured"], 0)

    def test_missing_spatial_region_is_insufficient(self):
        changed = copy.deepcopy(self.catalog)
        removed = {"test-corner-1", "test-corner-2"}
        changed["stars"] = [s for s in changed["stars"] if s["id"] not in removed]
        for frame in changed["frames"]:
            frame["observations"] = [o for o in frame["observations"] if o["star_id"] not in removed]
        report = experiment.evaluate_frozen(changed, self.candidate, tolerance_px=1e-5)
        self.assertEqual(report["judgement"]["status"], "insufficient_data")
        for model in report["models"].values():
            self.assertEqual(model["target-a"]["regions"]["corner"]["eligible"], 0)

    def test_two_corner_fixture_reports_other_corners_as_unmeasured(self):
        report = experiment.evaluate_frozen(self.catalog, self.candidate, tolerance_px=1e-5)
        # The declared pooled-region checks may pass; their verdict must not hide
        # which two physical corners this particular catalog never measured.
        self.assertEqual(report["judgement"]["status"], "passed")
        for model in report["models"].values():
            for frame in model.values():
                corners = frame["corner_coverage"]
                self.assertEqual(set(corners), {"top_left", "top_right", "bottom_left", "bottom_right"})
                for name in ("top_left", "bottom_right"):
                    self.assertEqual(corners[name]["eligible"], 1)
                    self.assertEqual(corners[name]["measured"], 1)
                    self.assertEqual(corners[name]["missing"], 0)
                for name in ("top_right", "bottom_left"):
                    self.assertEqual(corners[name]["eligible"], 0)
                    self.assertEqual(corners[name]["measured"], 0)
                    self.assertEqual(corners[name]["status"], "insufficient_data")

    def test_no_threshold_does_not_claim_precision_pass(self):
        report = experiment.evaluate_frozen(self.catalog, self.candidate)
        self.assertEqual(report["judgement"]["status"], "insufficient_data")

    def test_training_observation_changed_after_freezing_is_rejected(self):
        changed = copy.deepcopy(self.catalog)
        changed["frames"][0]["observations"][0]["x"] += 0.25
        with self.assertRaisesRegex(ValueError, "training data mismatch"):
            experiment.evaluate_frozen(changed, self.candidate, tolerance_px=1e-5)

    def test_external_identity_declaration_is_not_verification(self):
        changed = copy.deepcopy(self.catalog)
        changed["identity_basis"] = "external_verified"
        candidate = experiment.fit_training(cat.training_view(changed))
        report = experiment.evaluate_frozen(changed, candidate, tolerance_px=1e-5)
        self.assertEqual(report["judgement"]["status"], "insufficient_data")

    def test_missing_training_model_preserves_requested_frame(self):
        changed = copy.deepcopy(self.catalog)
        for frame in changed["frames"]:
            if frame["id"] == "target-a":
                frame["observations"] = [o for o in frame["observations"] if o["star_id"].startswith("test-")]
        candidate = experiment.fit_training(cat.training_view(changed))
        report = experiment.evaluate_frozen(changed, candidate, tolerance_px=1e-5)
        self.assertEqual(report["judgement"]["status"], "insufficient_data")
        for model in report["models"].values():
            self.assertEqual(model["target-a"]["regions"]["overall"]["eligible"], 4)
            self.assertEqual(model["target-a"]["regions"]["overall"]["measured"], 0)

    def test_freeze_write_once_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, catalog_path = root/"candidate.json", root/"catalog.json"
            catalog_path.write_text(json.dumps(self.catalog), encoding="utf-8")
            digest = experiment.freeze_candidate(model_path, self.candidate)
            with self.assertRaises(FileExistsError):
                experiment.freeze_candidate(model_path, self.candidate)
            report = experiment.evaluate_file(catalog_path, model_path, digest, tolerance_px=1e-5)
            self.assertEqual(report["judgement"]["status"], "passed")
            model_path.write_bytes(model_path.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError, "changed"):
                experiment.evaluate_file(catalog_path, model_path, digest, tolerance_px=1e-5)

    def test_invalid_tolerance_is_rejected(self):
        for value in (True, -1, float("nan"), float("inf"), "0.5", [], 10**5000):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(ValueError):
                    experiment.evaluate_frozen(self.catalog, self.candidate, tolerance_px=value)


if __name__ == "__main__":
    unittest.main()
