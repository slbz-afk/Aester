"""Tests for registration_diagnostics (CONTRACT.md v1).

Expected values come from hand calculation, closed-form transforms written
here with NumPy, or an independent grid-count area reference; the functions
under test are never used to produce their own expectations.

Runs with ``pytest`` or ``python -m unittest`` (unittest.TestCase classes).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from starstack import registration_diagnostics as rd

W, H = 100, 80  # reference frame: deliberately non-square
CONVENTION = {"row_order": "TOP-DOWN", "origin": "top_left_pixel_center", "units": "pixel"}
REFERENCE = {"frame_id": "ref", "width": W, "height": H, "coordinate_space_id": "ref_space"}
IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def translation(tx, ty):
    return [[1, 0, tx], [0, 1, ty], [0, 0, 1]]


def rotation_about(theta_deg, cx, cy):
    """p' = R(theta) (p - c) + c, as a 3x3 list (independent of the module)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]]


def apply_h(h, x, y):
    """Reference projection written independently: column vector [x, y, 1]."""
    m = np.asarray(h, dtype=float)
    v = m @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def match(star_id, source_xy, reference_xy, role="fit"):
    return {"star_id": star_id, "source_xy": list(source_xy), "reference_xy": list(reference_xy), "role": role}


def frame(frame_id="f", h=None, matches=(), width=W, height=H, elapsed=None, space="src_space", **extra):
    result = {"frame_id": frame_id, "width": width, "height": height, "coordinate_space_id": space,
              "source_to_reference": copy.deepcopy(IDENTITY if h is None else h), "matches": list(matches)}
    if elapsed is not None:
        result["elapsed_seconds"] = elapsed
    result.update(extra)
    return result


def document(frames, reference=REFERENCE):
    return {"schema_version": 1, "coordinate_convention": dict(CONVENTION),
            "reference": dict(reference), "frames": list(frames)}


def grid_area(h, src_w, src_h, ref_w=W, ref_h=H, n=1000):
    """Independent area reference: midpoint grid over the reference rectangle,
    counting samples whose inverse image lies inside the source rectangle."""
    inv = np.linalg.inv(np.asarray(h, dtype=float))
    xs = -0.5 + (np.arange(n) + 0.5) * ref_w / n
    ys = -0.5 + (np.arange(n) + 0.5) * ref_h / n
    gx, gy = np.meshgrid(xs, ys)
    ones = np.ones_like(gx)
    back = np.einsum("ij,jkl->ikl", inv, np.stack([gx, gy, ones]))
    sx, sy = back[0] / back[2], back[1] / back[2]
    inside = (sx >= -0.5) & (sx <= src_w - 0.5) & (sy >= -0.5) & (sy <= src_h - 0.5)
    return inside.sum() * (ref_w / n) * (ref_h / n)


class TransformsAndResidualSigns(unittest.TestCase):
    def test_identity_translation_rotation_scale_perspective_match_closed_form(self):
        cases = {
            "identity": IDENTITY,
            "integer_translation": translation(3, -2),
            "fractional_translation": translation(0.25, -0.75),
            "rotation_about_centre": rotation_about(30, 49.5, 39.5),
            "rotation_about_corner": rotation_about(-12.5, 0, 0),
            "anisotropic_scale": [[1.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
            "scale_about_centre": [[1.1, 0, -4.95], [0, 1.1, -3.95], [0, 0, 1]],
            "mild_perspective": [[1, 0.01, 2], [0.005, 1, -1], [1e-4, -5e-5, 1]],
            "scaled_by_minus_seven": (np.asarray(translation(3, -2)) * -7).tolist(),
            "h22_not_one": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        }
        # An asymmetric point set: nothing is symmetric in x/y, so swaps are caught.
        sources = [(10, 5), (45, 42), (3, 60), (70, 12), (88, 70), (20, 30)]
        for name, h in cases.items():
            matches = []
            for i, (sx, sy) in enumerate(sources):
                px, py = apply_h(h, sx, sy)
                if not (-0.5 <= px <= W - 0.5 and -0.5 <= py <= H - 0.5):
                    continue
                # measured position deliberately offset so the residual sign is testable
                matches.append(match(f"s{i}", (sx, sy), (px + 0.5, py - 1.0), "holdout"))
            self.assertGreaterEqual(len(matches), 3, name)
            result = rd.analyze_frame(REFERENCE, frame(h=h, matches=matches))
            with self.subTest(transform=name):
                self.assertEqual(result["state"], "succeeded")
                for m, r in zip(matches, result["residuals"]):
                    px, py = apply_h(h, *m["source_xy"])
                    self.assertEqual(r["star_id"], m["star_id"])
                    np.testing.assert_allclose(r["predicted_reference_xy"], [px, py], rtol=1e-13, atol=1e-11)
                    # residual = predicted - measured
                    self.assertAlmostEqual(r["dx_px"], -0.5, places=9)
                    self.assertAlmostEqual(r["dy_px"], 1.0, places=9)
                    self.assertAlmostEqual(r["distance_px"], math.hypot(0.5, 1.0), places=9)

    def test_integer_transforms_are_exact_and_positive_negative_scaling_agree(self):
        base = frame(h=translation(3, -2), matches=[match("a", (45, 42), (48, 40)), match("b", (8, 10), (11, 8), "holdout")])
        exact = rd.analyze_frame(REFERENCE, base)
        self.assertEqual(exact["residuals"][0]["predicted_reference_xy"], [48.0, 40.0])
        self.assertEqual(exact["residuals"][0]["dx_px"], 0.0)
        self.assertEqual(exact["residuals"][0]["distance_px"], 0.0)
        for k in (2.5, -7, 1e-6, 1e6):
            scaled = frame(h=(np.asarray(translation(3, -2)) * k).tolist(), matches=base["matches"])
            other = rd.analyze_frame(REFERENCE, scaled)
            with self.subTest(k=k):
                for a, b in zip(exact["residuals"], other["residuals"]):
                    np.testing.assert_allclose(b["predicted_reference_xy"], a["predicted_reference_xy"], rtol=1e-13)
                np.testing.assert_allclose(other["geometric_overlap"]["intersection_area_px2"],
                                           exact["geometric_overlap"]["intersection_area_px2"], rtol=1e-12)
                self.assertEqual(other["source_to_reference"], scaled["source_to_reference"])  # echoed as given

    def test_xy_order_and_frame_sizes_are_not_confused(self):
        # Source frame 120x90, reference 100x80; anisotropic scale must act on x with 0.5 and y with 0.8.
        h = [[0.5, 0, 0], [0, 0.8, 0], [0, 0, 1]]
        result = rd.analyze_frame(REFERENCE, frame(h=h, width=120, height=90,
                                                    matches=[match("a", (100, 20), (50, 16)), match("b", (20, 80), (10, 64))]))
        self.assertEqual(result["residuals"][0]["predicted_reference_xy"], [50.0, 16.0])
        self.assertEqual(result["residuals"][1]["predicted_reference_xy"], [10.0, 64.0])
        self.assertEqual(result["source_geometry"], {"width": 120, "height": 90, "coordinate_space_id": "src_space"})
        # region uses the measured reference position: (50,16) -> u=0.505 (centre), v=0.206 (top) -> other
        self.assertEqual(result["residuals"][0]["region"], "other")
        # A point valid in the 120-wide source but outside the 100-wide reference is rejected.
        with self.assertRaises(ValueError) as raised:
            rd.analyze_frame(REFERENCE, frame(h=h, width=120, height=90, matches=[match("c", (110, 10), (105, 8))]))
        self.assertIn("match_out_of_bounds", str(raised.exception))


class MatrixRejection(unittest.TestCase):
    def code_of(self, h, width=W, height=H):
        with self.assertRaises(ValueError) as raised:
            rd.analyze_frame(REFERENCE, frame(h=h, width=width, height=height))
        return str(raised.exception).split(":")[0]

    def test_singular_ill_conditioned_zero_nonfinite_and_shape(self):
        self.assertEqual(self.code_of([[1, 2, 3], [2, 4, 6], [0, 0, 1]]), "matrix_singular_or_ill_conditioned")
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1e-13, 0], [0, 0, 1]]), "matrix_singular_or_ill_conditioned")
        self.assertEqual(self.code_of([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), "matrix_zero")
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, float("nan")], [0, 0, 1]]), "matrix_not_finite")
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, float("inf")], [0, 0, 1]]), "matrix_not_finite")
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, 0]]), "matrix_shape_invalid")
        self.assertEqual(self.code_of([[1, 0], [0, 1], [0, 0]]), "matrix_shape_invalid")
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, "0"], [0, 0, 1]]), "matrix_not_finite")
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, True], [0, 0, 1]]), "matrix_not_finite")
        missing = frame()
        missing["source_to_reference"] = None
        with self.assertRaises(ValueError) as raised:
            rd.analyze_frame(REFERENCE, missing)
        self.assertEqual(str(raised.exception).split(":")[0], "matrix_shape_invalid")
        del missing["source_to_reference"]
        with self.assertRaises(ValueError) as raised:
            rd.analyze_frame(REFERENCE, missing)
        self.assertEqual(str(raised.exception).split(":")[0], "matrix_shape_invalid")
        # Just inside the condition limit is accepted (engineering boundary, not a quality threshold).
        ok = rd.analyze_frame(REFERENCE, frame(h=[[1, 0, 0], [0, 1e-11, 0], [0, 0, 1]]))
        self.assertEqual(ok["state"], "succeeded")

    def test_projective_infinity_inside_or_at_the_frame_edge(self):
        # den = 1 - 0.02 x changes sign between x=-0.5 and x=99.5
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, 0], [-0.02, 0, 1]]), "frame_crosses_projective_infinity")
        # den = 0 exactly at the right corners x = 99.5
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, 0], [-1 / 99.5, 0, 1]]), "frame_crosses_projective_infinity")
        # relative cancellation: den at (99.5, y) is ~1e-13 of the term magnitudes
        self.assertEqual(self.code_of([[1, 0, 0], [0, 1, 0], [-1 / 99.5 * (1 - 1e-13), 0, 1]]),
                         "frame_crosses_projective_infinity")
        # same matrix with a smaller source frame keeps every corner on one side -> valid
        ok = rd.analyze_frame(REFERENCE, frame(h=[[1, 0, 0], [0, 1, 0], [-0.02, 0, 1]], width=40, height=30))
        self.assertEqual(ok["state"], "succeeded")
        self.assertTrue(all(math.isfinite(v) for p in ok["geometric_overlap"]["projected_source_polygon_xy"] for v in p))

    def test_no_fallback_to_affine_and_json_stays_finite(self):
        report = rd.analyze_sequence(document([frame("bad", h=[[1, 0, 0], [0, 1, 0], [-0.02, 0, 1]])]))
        self.assertEqual(report["state"], "failed")
        bad = report["frames"][0]
        self.assertEqual(bad["state"], "invalid")
        self.assertIsNone(bad["source_to_reference"])
        self.assertIsNone(bad["geometric_overlap"])
        self.assertEqual(bad["error"]["code"], "frame_crosses_projective_infinity")
        json.dumps(report, allow_nan=False)  # nothing non-finite reaches the report


class RegionsAndStatistics(unittest.TestCase):
    REGION_POINTS = {
        "center": [(50, 40), (29.5, 23.5), (69.4, 55.4)],
        "top_left": [(10, 5), (-0.5, -0.5), (29.4, 23.4)],
        "top_right": [(90, 5), (69.5, 0), (99.5, 23.4)],
        "bottom_left": [(10, 70), (0, 55.5), (29.4, 79.5)],
        "bottom_right": [(95, 75), (69.5, 55.5), (99.5, 79.5)],
        "other": [(50, 5), (5, 40), (69.5, 40), (50, 79.5), (29.5, 5), (99.5, 40)],
    }

    def test_region_assignment_uses_measured_reference_position_and_contract_boundaries(self):
        matches = []
        expected = []
        for region, points in self.REGION_POINTS.items():
            for i, (x, y) in enumerate(points):
                # source position is far from the measured one on purpose: the predicted
                # position (identity -> source) must NOT decide the region.
                matches.append(match(f"{region}_{i}", (50, 40), (x, y), "unknown"))
                expected.append(region)
        result = rd.analyze_frame(REFERENCE, frame(matches=matches))
        self.assertEqual([r["region"] for r in result["residuals"]], expected)
        counts = {region: result["statistics"]["unknown"]["regions"][region]["count"] for region in rd.REGIONS}
        self.assertEqual(counts, {region: len(points) for region, points in self.REGION_POINTS.items()})
        self.assertEqual(sum(counts.values()), result["statistics"]["unknown"]["overall"]["count"])

    def test_statistics_hand_values_per_role_and_region(self):
        # identity transform; measured = source + offset  ->  residual = -offset, distance = |offset|
        def star(star_id, x, y, dx, dy, role):
            return match(star_id, (x, y), (x + dx, y + dy), role)
        matches = [
            star("f1", 50, 40, 1, 0, "fit"), star("f2", 52, 41, 0, 2, "fit"), star("f3", 48, 39, 3, 0, "fit"),
            star("f4", 55, 45, 0, 4, "fit"), star("f5", 45, 35, 6, 8, "fit"),      # centre distances 1,2,3,4,10
            star("h1", 5, 5, 0.3, 0.4, "holdout"), star("h2", 6, 6, 0.9, 1.2, "holdout"),  # top-left 0.5, 1.5
            star("h3", 94, 74, -0.6, -0.8, "holdout"),                              # bottom-right 1.0
            star("u1", 90, 5, 0, 0, "unknown"),                                     # top-right, zero residual
        ]
        s = rd.analyze_frame(REFERENCE, frame(matches=matches))["statistics"]
        fit = s["fit"]["overall"]
        self.assertEqual(fit["count"], 5)
        self.assertAlmostEqual(fit["rms_px"], math.sqrt(26), places=12)
        self.assertAlmostEqual(fit["median_px"], 3.0, places=12)
        self.assertAlmostEqual(fit["p95_px"], 8.8, places=12)   # linear interpolation: 4 + 0.8*(10-4)
        self.assertAlmostEqual(fit["max_px"], 10.0, places=12)
        self.assertEqual(s["fit"]["regions"]["center"], fit)
        for region in rd.REGIONS:
            if region != "center":
                self.assertEqual(s["fit"]["regions"][region], {"count": 0, "rms_px": None, "median_px": None, "p95_px": None, "max_px": None})
        hold = s["holdout"]["overall"]
        self.assertEqual(hold["count"], 3)
        self.assertAlmostEqual(hold["rms_px"], math.sqrt((0.25 + 2.25 + 1.0) / 3), places=12)
        self.assertAlmostEqual(hold["median_px"], 1.0, places=12)
        self.assertAlmostEqual(hold["p95_px"], 1.0 + 0.9 * 0.5, places=12)  # index 1.9 between 1.0 and 1.5
        self.assertAlmostEqual(hold["max_px"], 1.5, places=12)
        tl = s["holdout"]["regions"]["top_left"]
        self.assertEqual(tl["count"], 2)
        self.assertAlmostEqual(tl["rms_px"], math.sqrt(1.25), places=12)
        self.assertAlmostEqual(tl["median_px"], 1.0, places=12)
        self.assertAlmostEqual(tl["p95_px"], 1.45, places=12)
        self.assertAlmostEqual(tl["max_px"], 1.5, places=12)
        br = s["holdout"]["regions"]["bottom_right"]
        self.assertEqual(br["count"], 1)
        self.assertAlmostEqual(br["max_px"], 1.0, places=12)
        self.assertEqual({br["rms_px"], br["median_px"], br["p95_px"]}, {br["max_px"]})  # count=1 semantics
        unk = s["unknown"]["overall"]
        self.assertEqual(unk, {"count": 1, "rms_px": 0.0, "median_px": 0.0, "p95_px": 0.0, "max_px": 0.0})
        self.assertEqual(s["unknown"]["regions"]["top_right"]["count"], 1)

    def test_outliers_are_not_filtered_and_empty_matches_have_null_statistics(self):
        matches = [match("ok", (50, 40), (50, 40), "holdout"), match("wild", (60, 40), (10, 70), "holdout")]
        result = rd.analyze_frame(REFERENCE, frame(matches=matches))
        overall = result["statistics"]["holdout"]["overall"]
        self.assertEqual(overall["count"], 2)
        self.assertAlmostEqual(overall["max_px"], math.hypot(50, -30), places=9)
        empty = rd.analyze_frame(REFERENCE, frame(matches=[]))
        self.assertEqual(empty["state"], "succeeded")
        self.assertEqual(empty["residuals"], [])
        self.assertEqual(empty["holdout_evidence"], "unavailable")
        self.assertEqual(empty["match_counts"], {"fit": 0, "holdout": 0, "unknown": 0})
        for role in rd.ROLES:
            self.assertEqual(empty["statistics"][role]["overall"]["count"], 0)
            self.assertIsNone(empty["statistics"][role]["overall"]["rms_px"])
        self.assertEqual(empty["geometric_overlap"]["intersection_area_px2"], W * H)


class RolesHoldoutAndTime(unittest.TestCase):
    def test_holdout_evidence_and_role_separation(self):
        fit_only = rd.analyze_frame(REFERENCE, frame(matches=[match("a", (50, 40), (50.5, 40))]))
        self.assertEqual(fit_only["holdout_evidence"], "unavailable")
        self.assertEqual(fit_only["statistics"]["holdout"]["overall"]["count"], 0)
        self.assertEqual(fit_only["statistics"]["fit"]["overall"]["count"], 1)
        mixed = rd.analyze_frame(REFERENCE, frame(matches=[match("a", (50, 40), (50.5, 40)),
                                                            match("b", (10, 10), (10, 10.25), "holdout"),
                                                            match("c", (90, 70), (90, 70), "unknown")]))
        self.assertEqual(mixed["holdout_evidence"], "caller_declared")
        self.assertEqual(mixed["match_counts"], {"fit": 1, "holdout": 1, "unknown": 1})
        self.assertNotIn("independently_verified", json.dumps(mixed))

    def test_bad_roles_ids_and_out_of_bounds_make_the_frame_invalid(self):
        cases = {
            "star_id_duplicate": [match("a", (1, 1), (1, 1)), match("a", (2, 2), (2, 2))],
            "star_id_invalid": [match("", (1, 1), (1, 1))],
            "role_invalid": [match("a", (1, 1), (1, 1), "train")],
            "match_xy_invalid": [match("a", (1,), (1, 1))],
            "match_out_of_bounds": [match("a", (100, 1), (1, 1))],
            "matches_invalid": "not a list",
        }
        for code, matches in cases.items():
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as raised:
                    rd.analyze_frame(REFERENCE, frame(matches=matches))
                self.assertEqual(str(raised.exception).split(":")[0], code)
        for bad in ([match("a", (1, 1), (-0.6, 1))], [match("a", (1, 1), (1, 79.6))], [match("a", (-0.51, 1), (1, 1))]):
            with self.assertRaises(ValueError) as raised:
                rd.analyze_frame(REFERENCE, frame(matches=bad))
            self.assertIn("match_out_of_bounds", str(raised.exception))
        edge = rd.analyze_frame(REFERENCE, frame(matches=[match("a", (-0.5, -0.5), (99.5, 79.5))]))
        self.assertEqual(edge["state"], "succeeded")
        for bad in ({"star_id": "x", "source_xy": [1, 1], "reference_xy": [1, 1]},
                    {"star_id": "x", "source_xy": [1, 1], "role": "fit"},
                    {"star_id": "x", "source_xy": [1, "1"], "reference_xy": [1, 1], "role": "fit"},
                    {"star_id": "x", "source_xy": [1, True], "reference_xy": [1, 1], "role": "fit"}):
            with self.assertRaises(ValueError):
                rd.analyze_frame(REFERENCE, frame(matches=[bad]))

    def test_elapsed_seconds_semantics_and_input_order(self):
        report = rd.analyze_sequence(document([frame("late", elapsed=30), frame("early", elapsed=10.5), frame("unknown_time")]))
        self.assertEqual([f["frame_id"] for f in report["frames"]], ["late", "early", "unknown_time"])
        self.assertEqual([f["elapsed_seconds"] for f in report["frames"]], [30, 10.5, None])
        for bad in ("12", True, float("nan"), float("inf"), [1]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as raised:
                    rd.analyze_frame(REFERENCE, frame(elapsed=bad))
                self.assertIn("elapsed_seconds_invalid", str(raised.exception))
        explicit_null = rd.analyze_frame(REFERENCE, frame(elapsed_seconds=None))
        self.assertIsNone(explicit_null["elapsed_seconds"])


class GeometricOverlap(unittest.TestCase):
    def area(self, h, width=W, height=H, reference=REFERENCE):
        result = rd.analyze_frame(reference, frame(h=h, width=width, height=height))
        return result["geometric_overlap"]

    def test_hand_computed_areas(self):
        cases = [
            ("identity", IDENTITY, W, H, 8000.0),
            ("translation_3_-2", translation(3, -2), W, H, 97 * 78),
            ("fractional_translation", translation(0.25, -0.75), W, H, 99.75 * 79.25),
            ("partial", translation(60, 50), W, H, 40 * 30),
            ("edge_contact", translation(100, 0), W, H, 0.0),
            ("corner_contact", translation(100, 80), W, H, 0.0),
            ("disjoint", translation(150, 0), W, H, 0.0),
            ("negative_disjoint", translation(-100, 0), W, H, 0.0),
            ("rotate_90_about_centre", rotation_about(90, 49.5, 39.5), W, H, 80 * 80),
            ("scale_half_about_origin", [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], W, H, 50 * 40),
            ("scale_double_about_origin", [[2, 0, 0], [0, 2, 0], [0, 0, 1]], W, H, 8000.0),
            ("smaller_source", IDENTITY, 50, 40, 50 * 40),
            ("larger_source", IDENTITY, 300, 200, 8000.0),
            ("mirror_x_about_centre", [[-1, 0, 99], [0, 1, 0], [0, 0, 1]], W, H, 8000.0),
        ]
        for name, h, width, height, expected in cases:
            with self.subTest(case=name):
                overlap = self.area(h, width, height)
                self.assertAlmostEqual(overlap["intersection_area_px2"], expected, places=9)
                self.assertAlmostEqual(overlap["reference_area_fraction"], expected / 8000, places=12)
                self.assertEqual(overlap["coverage_kind"], "continuous_frame_footprint_only")
                self.assertEqual(len(overlap["projected_source_polygon_xy"]), 4)
        square = {"frame_id": "sq", "width": 100, "height": 100, "coordinate_space_id": "sq"}
        octagon = self.area(rotation_about(45, 49.5, 49.5), 100, 100, reference=square)
        self.assertAlmostEqual(octagon["intersection_area_px2"], 2 * (math.sqrt(2) - 1) * 100 ** 2, places=6)
        self.assertEqual(len(octagon["intersection_polygon_xy"]), 8)

    def test_contact_cases_keep_zero_area_and_disjoint_is_empty(self):
        edge = self.area(translation(100, 0))
        self.assertEqual(edge["intersection_area_px2"], 0.0)
        self.assertEqual(edge["reference_area_fraction"], 0.0)
        self.assertTrue(all(p[0] == 99.5 for p in edge["intersection_polygon_xy"]))
        self.assertEqual(self.area(translation(150, 0))["intersection_polygon_xy"], [])
        corner = self.area(translation(100, 80))
        self.assertEqual(corner["intersection_area_px2"], 0.0)
        self.assertLessEqual(len(corner["intersection_polygon_xy"]), 1)

    def test_polygon_vertices_lie_on_the_boundary_in_order(self):
        overlap = self.area(translation(3, -2))
        self.assertEqual(overlap["projected_source_polygon_xy"], [[2.5, -2.5], [102.5, -2.5], [102.5, 77.5], [2.5, 77.5]])
        self.assertEqual(overlap["intersection_polygon_xy"], [[2.5, -0.5], [99.5, -0.5], [99.5, 77.5], [2.5, 77.5]])

    def test_true_perspective_intersection_matches_independent_grid_count(self):
        cases = [
            [[1, 0.02, 4], [0.01, 1, -3], [2e-4, -1e-4, 1]],
            [[0.9, -0.1, 12], [0.15, 1.05, -6], [-3e-4, 2e-4, 1]],
            rotation_about(17, 20, 10),
        ]
        for h in cases:
            with self.subTest(h=h):
                overlap = self.area(h)
                expected = grid_area(h, W, H, n=1000)
                self.assertGreater(expected, 1000)
                self.assertLess(expected, 8000)
                self.assertAlmostEqual(overlap["intersection_area_px2"] / expected, 1.0, delta=5e-3)
                # a bounding box of the projected quadrilateral is NOT what is reported
                xs = [p[0] for p in overlap["projected_source_polygon_xy"]]
                ys = [p[1] for p in overlap["projected_source_polygon_xy"]]
                bbox = (min(max(xs), W - 0.5) - max(min(xs), -0.5)) * (min(max(ys), H - 0.5) - max(min(ys), -0.5))
                self.assertLess(overlap["intersection_area_px2"], bbox - 1.0)

    def test_frame_crossing_infinity_has_no_area(self):
        report = rd.analyze_sequence(document([frame("cross", h=[[1, 0, 0], [0, 1, 0], [-0.02, 0, 1]])]))
        self.assertIsNone(report["frames"][0]["geometric_overlap"])


class SequenceBehaviour(unittest.TestCase):
    def test_states_and_invalid_records(self):
        good = frame("good", matches=[match("a", (50, 40), (50, 40), "holdout")])
        bad = frame("bad", h=[[1, 2, 3], [2, 4, 6], [0, 0, 1]])
        self.assertEqual(rd.analyze_sequence(document([good]))["state"], "succeeded")
        self.assertEqual(rd.analyze_sequence(document([bad]))["state"], "failed")
        report = rd.analyze_sequence(document([good, bad, frame("ok2")]))
        self.assertEqual(report["state"], "partial")
        self.assertEqual([f["state"] for f in report["frames"]], ["succeeded", "invalid", "succeeded"])
        invalid = report["frames"][1]
        self.assertEqual(invalid["frame_id"], "bad")
        self.assertEqual(invalid["source_geometry"], {"width": W, "height": H, "coordinate_space_id": "src_space"})
        for key in ("source_to_reference", "residuals", "statistics", "geometric_overlap"):
            self.assertIsNone(invalid[key])
        self.assertEqual(invalid["holdout_evidence"], "unavailable")
        self.assertEqual(invalid["error"]["code"], "matrix_singular_or_ill_conditioned")
        self.assertIsNone(report["frames"][0]["error"])
        self.assertEqual(report["coordinate_convention"], CONVENTION)
        self.assertEqual(report["reference"], REFERENCE)
        self.assertTrue(report["limitations"])
        self.assertEqual(set(report), {"schema_version", "state", "coordinate_convention", "reference", "frames", "limitations"})

    def test_invalid_match_keeps_validated_matrix_but_no_results(self):
        report = rd.analyze_sequence(document([frame("m", h=translation(1, 1), matches=[match("a", (1, 1), (1, 1), "pilot")])]))
        invalid = report["frames"][0]
        self.assertEqual(invalid["error"]["code"], "role_invalid")
        self.assertEqual(invalid["source_to_reference"], translation(1, 1))
        self.assertIsNone(invalid["geometric_overlap"])

    def test_top_level_errors(self):
        good = frame("a")
        bad_documents = {
            "schema": {**document([good]), "schema_version": 2},
            "schema_bool": {**document([good]), "schema_version": True},
            "convention_extra_key": {**document([good]), "coordinate_convention": {**CONVENTION, "extra": 1}},
            "convention_bottom_up": {**document([good]), "coordinate_convention": {**CONVENTION, "row_order": "BOTTOM-UP"}},
            "convention_missing": {**document([good]), "coordinate_convention": {"row_order": "TOP-DOWN"}},
            "reference_missing": {k: v for k, v in document([good]).items() if k != "reference"},
            "reference_bool_size": document([good], reference={**REFERENCE, "width": True}),
            "reference_string_size": document([good], reference={**REFERENCE, "height": "80"}),
            "reference_zero_size": document([good], reference={**REFERENCE, "height": 0}),
            "frames_empty": document([]),
            "frames_not_list": {**document([good]), "frames": {"a": good}},
            "frame_not_object": document(["x"]),
            "frame_id_missing": document([{k: v for k, v in good.items() if k != "frame_id"}]),
            "frame_id_duplicate": document([good, frame("a", h=translation(1, 0))]),
            "not_object": [],
        }
        for name, doc in bad_documents.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    rd.analyze_sequence(doc)
        with self.assertRaises(ValueError):
            rd.analyze_frame({**REFERENCE, "width": 0}, good)

    def test_inputs_are_not_mutated_and_results_are_deterministic(self):
        doc = document([
            frame("a", h=rotation_about(20, 49.5, 39.5), matches=[match("s", (50, 40), (50.5, 39.5), "holdout"), match("t", (20, 20), (36.6, 19.1))], elapsed=5),
            frame("b", h=[[1, 2, 3], [2, 4, 6], [0, 0, 1]]),
            frame("c", h=[[1, 0.01, 2], [0.005, 1, -1], [1e-4, -5e-5, 1]], matches=[match("u", (10, 10), (12, 9), "unknown")]),
        ])
        snapshot = copy.deepcopy(doc)
        first = rd.analyze_sequence(doc)
        second = rd.analyze_sequence(doc)
        self.assertEqual(doc, snapshot)
        self.assertEqual(json.dumps(first, sort_keys=False), json.dumps(second, sort_keys=False))
        frame_a = rd.analyze_frame(REFERENCE, doc["frames"][0])
        self.assertEqual(doc, snapshot)
        self.assertEqual(json.dumps(frame_a), json.dumps(first["frames"][0]))


class CommandLine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="regdiag_cli_")
        self.tmp = Path(self._tmp.name)
        self.example = HERE / "example_input.json"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(Path(rd.__file__)), *args],
                              capture_output=True, text=True, cwd=self.tmp)

    def write(self, name, payload):
        path = self.tmp / name
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        return path

    def test_example_input_succeeds_with_exit_0_and_hand_checked_values(self):
        if not self.example.exists():
            self.skipTest("example_input.json is not next to the delivery folder")
        out = self.tmp / "report.json"
        run = self.run_cli("--input", str(self.example), "--output", str(out))
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(report["state"], "succeeded")
        self.assertEqual(report["provenance"]["input_sha256"], hashlib.sha256(self.example.read_bytes()).hexdigest())
        f = report["frames"][0]
        by_id = {r["star_id"]: r for r in f["residuals"]}
        self.assertEqual(by_id["holdout_center"]["predicted_reference_xy"], [51.0, 38.0])
        self.assertEqual((by_id["holdout_center"]["dx_px"], by_id["holdout_center"]["dy_px"]), (0.25, -0.5))
        self.assertEqual(by_id["holdout_center"]["region"], "center")
        self.assertEqual(by_id["holdout_corner"]["region"], "top_left")
        self.assertEqual(f["statistics"]["holdout"]["overall"]["count"], 2)
        self.assertEqual(f["geometric_overlap"]["intersection_area_px2"], 97 * 78)
        self.assertEqual(len(list(self.tmp.iterdir())), 1)  # no temporary files left behind

    def test_exit_codes_for_partial_failed_and_errors(self):
        good = frame("good", matches=[match("a", (50, 40), (50, 40))])
        bad = frame("bad", h=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        partial = self.write("partial.json", document([good, bad]))
        run = self.run_cli("--input", str(partial), "--output", str(self.tmp / "partial_report.json"))
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(json.loads((self.tmp / "partial_report.json").read_text(encoding="utf-8"))["state"], "partial")
        failed = self.write("failed.json", document([bad]))
        run = self.run_cli("--input", str(failed), "--output", str(self.tmp / "failed_report.json"))
        self.assertEqual(run.returncode, 3, run.stderr)
        self.assertEqual(json.loads((self.tmp / "failed_report.json").read_text(encoding="utf-8"))["state"], "failed")
        for name, payload in {"nan.json": '{"schema_version": NaN}', "inf.json": '{"schema_version": -Infinity}',
                              "dupkey.json": '{"schema_version": 1, "schema_version": 1}',
                              "schema2.json": json.dumps({**document([good]), "schema_version": 2}),
                              "empty_frames.json": json.dumps(document([])),
                              "broken.json": "{not json"}.items():
            path = self.write(name, payload)
            out = self.tmp / (name + ".report.json")
            run = self.run_cli("--input", str(path), "--output", str(out))
            with self.subTest(name=name):
                self.assertEqual(run.returncode, 1)
                self.assertTrue(run.stderr.startswith("error:"), run.stderr)
                self.assertFalse(out.exists())
        run = self.run_cli("--input", str(self.tmp / "missing.json"), "--output", str(self.tmp / "x.json"))
        self.assertEqual(run.returncode, 1)
        self.assertFalse((self.tmp / "x.json").exists())

    def test_existing_output_is_never_overwritten_by_default(self):
        good = self.write("good.json", document([frame("good")]))
        out = self.write("existing.json", "keep me")
        run = self.run_cli("--input", str(good), "--output", str(out))
        self.assertEqual(run.returncode, 1)
        self.assertEqual(out.read_text(encoding="utf-8"), "keep me")
        run = self.run_cli("--input", str(good), "--output", str(out), "--overwrite")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["state"], "succeeded")
        run = self.run_cli("--input", str(good), "--output", str(good), "--overwrite")
        self.assertEqual(run.returncode, 1)
        self.assertEqual(json.loads(good.read_text(encoding="utf-8"))["schema_version"], 1)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["existing.json", "good.json"])

    def test_unwritable_output_directory_leaves_no_report(self):
        good = self.write("good.json", document([frame("good")]))
        run = self.run_cli("--input", str(good), "--output", str(self.tmp / "no_such_dir" / "r.json"))
        self.assertEqual(run.returncode, 1)
        self.assertFalse((self.tmp / "no_such_dir").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
