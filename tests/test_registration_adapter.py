"""Tests for starstack.registration_adapter with hand-computed expectations.

Run:  python -B -m unittest tests.test_registration_adapter   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starstack import registration_adapter as ra  # noqa: E402
from starstack import siril_sequence as ss  # noqa: E402

TD = ra.CoordinateHypothesis(name="td", row_order="TOP-DOWN", x_offset=0.0, y_offset=0.0,
                             matrix_direction="frame_to_reference", verified=False, basis="test")
BU = ra.CoordinateHypothesis(name="bu", row_order="BOTTOM-UP", x_offset=0.0, y_offset=0.0,
                             matrix_direction="frame_to_reference", verified=False, basis="test")
BU_HALF_INV = ra.CoordinateHypothesis(name="bu_half_inv", row_order="BOTTOM-UP", x_offset=0.5, y_offset=0.5,
                                      matrix_direction="reference_to_frame", verified=False, basis="test")


def translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


class Hypothesis(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            ra.CoordinateHypothesis("x", "LEFT-RIGHT", 0.0, 0.0, "frame_to_reference", False, "")
        with self.assertRaises(ValueError):
            ra.CoordinateHypothesis("x", "TOP-DOWN", 0.0, 0.0, "sideways", False, "")
        with self.assertRaises(ValueError):
            ra.CoordinateHypothesis("x", "TOP-DOWN", float("nan"), 0.0, "frame_to_reference", False, "")
        with self.assertRaises(ValueError):
            ra.CoordinateHypothesis("x", "TOP-DOWN", True, 0.0, "frame_to_reference", False, "")

    def test_point_conversion_hand_values(self):
        # TOP-DOWN, no offset: identity
        self.assertEqual(ra.convert_point(TD, 10.0, 20.0, 100), (10.0, 20.0))
        # BOTTOM-UP, height 100: y_td = 99 - y
        self.assertEqual(ra.convert_point(BU, 10.0, 20.0, 100), (10.0, 79.0))
        # BOTTOM-UP with +0.5 offsets: x_td = x + 0.5, y_td = 99 - (y + 0.5)
        self.assertEqual(ra.convert_point(BU_HALF_INV, 10.0, 20.0, 100), (10.5, 78.5))

    def test_native_point_is_inverse(self):
        for hyp in (TD, BU, BU_HALF_INV):
            x, y = ra.convert_point(hyp, 123.25, 45.75, 901)
            nx, ny = ra.native_point(hyp, x, y, 901)
            self.assertAlmostEqual(nx, 123.25, places=9)
            self.assertAlmostEqual(ny, 45.75, places=9)

    def test_flip_conjugates_translation(self):
        # A Siril translation (tx, ty) seen through a BOTTOM-UP flip is (tx, -ty) in TOP-DOWN.
        h = ra.convert_matrix(BU, translation(7.25, -3.5), 901, 901)
        np.testing.assert_allclose(h, translation(7.25, 3.5), atol=1e-12)
        # TOP-DOWN leaves it unchanged
        np.testing.assert_allclose(ra.convert_matrix(TD, translation(7.25, -3.5), 901, 901), translation(7.25, -3.5))

    def test_reference_to_frame_inverts_before_conjugation(self):
        h = ra.convert_matrix(BU_HALF_INV, translation(2.0, 3.0), 50, 50)
        # inverse translation (-2, -3), flipped in y => (-2, +3); offsets cancel for a translation
        np.testing.assert_allclose(h, translation(-2.0, 3.0), atol=1e-12)

    def test_rotation_conjugation_preserves_point_mapping(self):
        # Whatever the hypothesis, converting points then applying the converted
        # matrix must equal applying the Siril matrix then converting.
        t = math.radians(2.0)
        h_siril = np.array([[math.cos(t), -math.sin(t), 5.0], [math.sin(t), math.cos(t), -7.0], [0.0, 0.0, 1.0]])
        for hyp in (TD, BU, BU_HALF_INV):
            h_td = ra.convert_matrix(hyp, h_siril, 901, 901)
            if hyp.matrix_direction == "reference_to_frame":
                h_siril_f2r = np.linalg.inv(h_siril)
            else:
                h_siril_f2r = h_siril
            for (x, y) in ((10.0, 20.0), (600.0, 450.0), (1199.0, 900.0)):
                v = h_siril_f2r @ np.array([x, y, 1.0])
                expected = ra.convert_point(hyp, v[0] / v[2], v[1] / v[2], 901)
                sx, sy = ra.convert_point(hyp, x, y, 901)
                w = h_td @ np.array([sx, sy, 1.0])
                self.assertAlmostEqual(w[0] / w[2], expected[0], places=9)
                self.assertAlmostEqual(w[1] / w[2], expected[1], places=9)

    def test_convert_matrix_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ra.convert_matrix(TD, [[1, 0], [0, 1]], 10, 10)
        with self.assertRaises(ValueError):
            ra.convert_matrix(TD, [[1, 0, float("inf")], [0, 1, 0], [0, 0, 1]], 10, 10)


class ToSirilMatrix(unittest.TestCase):
    def test_inverse_of_convert_matrix_for_both_directions_and_row_orders(self):
        rng = np.random.default_rng(3)
        for hyp in ra.HYPOTHESES.values():
            for _ in range(5):
                h_td = np.array([[1.001, 0.02, rng.uniform(-50, 50)], [-0.02, 0.999, rng.uniform(-50, 50)], [1e-6, -2e-6, 1.0]])
                h_siril = ra.to_siril_matrix(hyp, h_td, 901, 901)
                self.assertAlmostEqual(h_siril[2, 2], 1.0)
                back = ra.convert_matrix(hyp, h_siril, 901, 901)
                np.testing.assert_allclose(back / back[2, 2], h_td / h_td[2, 2], atol=1e-9)   # projective: equal up to scale
                # and a point mapped in top-down coordinates by h_td is mapped identically by h_siril in Siril coordinates
                x_td, y_td = 123.25, 456.75
                w = h_td @ np.array([x_td, y_td, 1.0])
                sx, sy = ra.native_point(hyp, x_td, y_td, 901)
                stored = h_siril if hyp.matrix_direction == "frame_to_reference" else np.linalg.inv(h_siril)
                v = stored @ np.array([sx, sy, 1.0])
                ex, ey = ra.convert_point(hyp, v[0] / v[2], v[1] / v[2], 901)
                self.assertAlmostEqual(ex, w[0] / w[2], places=6)
                self.assertAlmostEqual(ey, w[1] / w[2], places=6)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ra.to_siril_matrix(TD, [[1, 0], [0, 1]], 10, 10)
        with self.assertRaises(ValueError):
            ra.to_siril_matrix(TD, np.zeros((3, 3)), 10, 10)


class ProvedHypothesis(unittest.TestCase):
    def test_default_is_the_proved_siril_convention(self):
        hyp = ra.HYPOTHESES[ra.DEFAULT_HYPOTHESIS]
        self.assertTrue(hyp.verified)
        self.assertEqual((hyp.row_order, hyp.x_offset, hyp.y_offset, hyp.matrix_direction),
                         ("TOP-DOWN", -0.5, -0.5, "frame_to_reference"))
        self.assertIn("coordinate-proof-siril-20260917T092505Z-970195", hyp.basis)
        # Siril (0.5, 0.5) is the centre of the top-left pixel -> diagnostics (0, 0)
        self.assertEqual(ra.convert_point(hyp, 0.5, 0.5, 901), (0.0, 0.0))
        # a pure translation is unchanged by a constant offset; a rotation about
        # the Siril origin acquires the offset conjugation
        np.testing.assert_allclose(ra.convert_matrix(hyp, translation(7.25, -3.5), 901, 901), translation(7.25, -3.5), atol=1e-12)
        for name, other in ra.HYPOTHESES.items():
            if name != ra.DEFAULT_HYPOTHESIS:
                self.assertFalse(other.verified, name)


class Association(unittest.TestCase):
    def test_mutual_nearest_with_ambiguity_and_unmatched(self):
        ref = [{"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0}, {"x": 300.0, "y": 100.0},
               {"x": 303.0, "y": 100.0}, {"x": 500.0, "y": 500.0}]
        # frame = reference shifted by (-10, 0); star 3 lands between ref 2 and 3 (ambiguous),
        # star 4 is far from everything, ref 4 has no counterpart.
        frame = [{"x": 90.0, "y": 100.0}, {"x": 190.0, "y": 100.5}, {"x": 291.5, "y": 100.0}, {"x": 700.0, "y": 700.0}]
        a = ra.associate_stars(ref, frame, translation(10.0, 0.0), radius=4.0)
        self.assertEqual([(p["frame_index"], p["reference_index"]) for p in a["pairs"]], [(0, 0), (1, 1)])
        self.assertAlmostEqual(a["pairs"][1]["distance_px"], 0.5)
        self.assertEqual(a["ambiguous_frame"], [2])
        self.assertEqual(a["unmatched_frame"], [3])
        self.assertEqual(a["unmatched_reference"], [2, 3, 4])
        self.assertAlmostEqual(a["matched_fraction_of_frame"], 0.5)
        self.assertEqual(a["pairs"][0]["projected_xy"], [100.0, 100.0])

    def test_radius_excludes_large_residuals(self):
        ref = [{"x": 100.0, "y": 100.0}]
        frame = [{"x": 100.0, "y": 103.0}]
        self.assertEqual(len(ra.associate_stars(ref, frame, np.eye(3), radius=2.0)["pairs"]), 0)
        self.assertEqual(len(ra.associate_stars(ref, frame, np.eye(3), radius=3.5)["pairs"]), 1)

    def test_empty_inputs_and_bad_radius(self):
        a = ra.associate_stars([], [{"x": 1.0, "y": 1.0}], np.eye(3), radius=1.0)
        self.assertEqual(a["unmatched_frame"], [0])
        with self.assertRaises(ValueError):
            ra.associate_stars([], [], np.eye(3), radius=0.0)


def _write_sequence(directory: Path, matrices: dict[int, np.ndarray], stars: dict[str, list[tuple[float, float]]],
                    name: str = "light_") -> tuple[Path, dict]:
    """Write a 3-image sequence and star lists; matrices are Siril H per position."""
    lines = [f"S '{name}' 1 3 3 5 0 6 0 0 0", "L 3", "I 1 1", "I 2 1", "I 3 1"]
    for pos in range(3):
        h = matrices.get(pos, np.eye(3))
        coeff = " ".join(f"{v:g}" for v in np.asarray(h).reshape(-1))
        lines.append(f"R1 3.5 3.5 0.8 0 0.02 {len(stars.get(f'{name}{pos + 1:05d}', []))} H {coeff}")
    seq = directory / f"{name}.seq"
    seq.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cache = directory / "cache"
    cache.mkdir(exist_ok=True)
    paths = {}
    for stem, points in stars.items():
        rows = [f"# {len(points)} stars found using the following parameters:", "# sigma=1.00 max_stars=2000 layer=1",
                "# star#\tlayer\tB\tA\tbeta\tX\tY\tFWHMx [px]\tFWHMy [px]\tFWHMx [\"]\tFWHMy [\"]\tangle\tRMSE\tmag\tSat\tProfile\tRA\tDec"]
        for n, (x, y) in enumerate(points, 1):
            sat = 1 if n == 1 else 0
            rows.append(f"{n}\t1\t1\t{1000 - n}\t-1\t{x:.2f}\t{y:.2f}\t3\t3\t0\t0\t0\t1e-3\t-1\t{sat}\tGaussian\t0\t0")
        path = cache / f"{stem}.lst"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        paths[stem] = path
    return seq, paths


class DocumentBuilding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.width, self.height = 400, 300
        self.ref_points = [(50.0, 40.0), (200.0, 150.0), (350.0, 40.0), (60.0, 260.0), (340.0, 250.0), (120.0, 150.0)]
        self.h2 = translation(4.0, -3.0)              # frame2 -> reference in Siril coordinates
        self.h3 = translation(-2.0, 5.0)
        f2 = [(x - 4.0, y + 3.0) for x, y in self.ref_points]        # exact
        f3 = [(x + 2.0, y - 5.0) for x, y in self.ref_points]
        f3[1] = (f3[1][0] + 0.6, f3[1][1])                            # one 0.6 px residual at the centre star
        self.seq, self.lst = _write_sequence(d, {1: self.h2, 2: self.h3},
                                             {"light_00001": self.ref_points, "light_00002": f2, "light_00003": f3})

    def tearDown(self):
        self.tmp.cleanup()

    def test_document_roles_and_matrices(self):
        seq = ss.read_seq(self.seq)
        lists = {stem: ss.read_lst(p) for stem, p in self.lst.items()}
        doc, assoc = ra.build_document(sequence=seq, star_lists=lists, width=self.width, height=self.height,
                                       hypothesis=BU, positions=[1, 2], layer=1, radius=3.0)
        self.assertEqual(doc["coordinate_convention"], {"row_order": "TOP-DOWN", "origin": "top_left_pixel_center", "units": "pixel"})
        self.assertEqual(doc["reference"]["frame_id"], "light_00001")
        self.assertEqual([f["frame_id"] for f in doc["frames"]], ["light_00002", "light_00003"])
        f2 = doc["frames"][0]
        self.assertEqual({m["role"] for m in f2["matches"]}, {"unknown"})
        self.assertEqual(len(f2["matches"]), 6)
        # BOTTOM-UP flips the y component of the translation
        np.testing.assert_allclose(f2["source_to_reference"], translation(4.0, 3.0), atol=1e-12)
        # star coordinates are converted consistently: source maps onto reference exactly for frame 2
        for m in f2["matches"]:
            sx, sy = m["source_xy"]
            self.assertAlmostEqual(sx + 4.0, m["reference_xy"][0])
            self.assertAlmostEqual(sy + 3.0, m["reference_xy"][1])
        self.assertEqual(assoc["light_00002"]["status"], "associated")
        self.assertEqual(len(assoc["light_00003"]["pairs"]), 6)

    def test_rejects_registered_output_sequence(self):
        d = Path(self.tmp.name) / "registered"
        d.mkdir()
        seq, lst = _write_sequence(d, {}, {"light_00001": self.ref_points, "light_00002": self.ref_points})
        with self.assertRaises(ValueError):
            ra.build_document(sequence=ss.read_seq(seq), star_lists={s: ss.read_lst(p) for s, p in lst.items()},
                              width=10, height=10, hypothesis=TD, positions=[1], layer=1, radius=3.0)

    def test_rejects_reference_as_frame_and_missing_lists(self):
        seq = ss.read_seq(self.seq)
        lists = {stem: ss.read_lst(p) for stem, p in self.lst.items()}
        with self.assertRaises(ValueError):
            ra.build_document(sequence=seq, star_lists=lists, width=1, height=1, hypothesis=TD, positions=[0], layer=1, radius=3.0)
        with self.assertRaises(ValueError):
            ra.build_document(sequence=seq, star_lists={}, width=1, height=1, hypothesis=TD, positions=[1], layer=1, radius=3.0)
        doc, assoc = ra.build_document(sequence=seq, star_lists={"light_00001": lists["light_00001"]}, width=1, height=1,
                                       hypothesis=TD, positions=[1], layer=1, radius=3.0)
        self.assertEqual(assoc["light_00002"], {"status": "no_star_list"})
        self.assertEqual(doc["frames"][0]["matches"], [])

    def test_alternate_reference_chains_matrices(self):
        seq = ss.read_seq(self.seq)
        lists = {stem: ss.read_lst(p) for stem, p in self.lst.items()}
        doc, assoc = ra.build_document(sequence=seq, star_lists=lists, width=self.width, height=self.height,
                                       hypothesis=TD, positions=[0, 2], layer=1, radius=3.0, reference_position=1)
        self.assertEqual(doc["reference"]["frame_id"], "light_00002")
        by_id = {f["frame_id"]: f for f in doc["frames"]}
        # frame1 -> frame2 = inv(H2) @ I = translation(-4, +3); frame3 -> frame2 = inv(H2) @ H3 = translation(-6, +8)
        np.testing.assert_allclose(by_id["light_00001"]["source_to_reference"], translation(-4.0, 3.0), atol=1e-12)
        np.testing.assert_allclose(by_id["light_00003"]["source_to_reference"], translation(-6.0, 8.0), atol=1e-12)
        self.assertEqual(len(assoc["light_00003"]["pairs"]), 6)
        with self.assertRaises(ValueError):
            ra.build_document(sequence=seq, star_lists=lists, width=1, height=1, hypothesis=TD, positions=[1], layer=1,
                              radius=3.0, reference_position=1)

    def test_end_to_end_summaries(self):
        result = ra.run_sequence_diagnostics(seq_path=self.seq, lst_paths=self.lst, width=self.width, height=self.height,
                                             hypothesis=TD, positions=[1, 2], layer=1, radius=3.0)
        self.assertEqual(result["roles"], "all unknown (Siril registration candidates)")
        self.assertFalse(result["hypothesis"]["verified"])
        self.assertEqual(result["comparison_reference"], "light_00001")
        frames = {f["frame_id"]: f for f in result["report"]["frames"]}
        self.assertEqual(frames["light_00002"]["state"], "succeeded")
        self.assertEqual(frames["light_00002"]["match_counts"], {"fit": 0, "holdout": 0, "unknown": 6})
        s2 = result["summaries"]["light_00002"]
        self.assertEqual(s2["association"]["matched_count"], 6)
        self.assertEqual(s2["association"]["ambiguous_count"], 0)
        self.assertAlmostEqual(s2["pooled"]["all_roles_pooled"]["overall"]["rms_px"], 0.0)
        s3 = result["summaries"]["light_00003"]["pooled"]["all_roles_pooled"]
        # only the centre star carries the 0.6 px residual: projected = ref + (0.6, 0)
        self.assertEqual(s3["center"]["count"], 2)
        self.assertAlmostEqual(s3["center"]["max_px"], 0.6, places=6)
        self.assertAlmostEqual(s3["center"]["mean_dx_px"], 0.3, places=6)
        self.assertAlmostEqual(s3["corners_pooled"]["rms_px"], 0.0)
        self.assertAlmostEqual(s3["overall"]["rms_px"], math.sqrt(0.36 / 6), places=9)
        # unsaturated-only view drops the first (saturated) star of each list
        uns = result["summaries"]["light_00003"]["pooled"]["unsaturated_only"]
        self.assertEqual(sum(v["count"] for v in uns.values()), 5)
        grid = result["summaries"]["light_00003"]["grid"]
        self.assertTrue(grid["available"])
        self.assertEqual(sum(c["count"] for c in grid["cells"].values()), 6)
        self.assertIn("xcol_2/ynative_1", grid["cells"])   # (200,150) in a 400x300 image, 4x3 grid
        self.assertEqual(len(result["caveats"]), 3)


class FixedPositionScreen(unittest.TestCase):
    def test_same_position_detections_are_listed_and_pairs_need_displacement(self):
        a = {"stars": [{"x": 10.0, "y": 10.0}, {"x": 500.0, "y": 300.0}, {"x": 700.0, "y": 20.0}]}
        b = {"stars": [{"x": 10.4, "y": 9.8}, {"x": 520.0, "y": 300.0}, {"x": 700.0, "y": 21.5}]}
        screen = ra.fixed_position_screen({"f1": a, "f2": b}, {"f1|f2": 20.0}, tolerance_px=1.0)
        pair = screen["pairs"]["f1|f2"]
        self.assertTrue(pair["informative"])
        self.assertEqual(pair["candidate_count"], 1)
        self.assertEqual((pair["candidates"][0]["a_index"], pair["candidates"][0]["b_index"]), (0, 0))
        self.assertAlmostEqual(pair["candidates"][0]["distance_px"], math.hypot(0.4, 0.2))
        # a pair whose frames barely move cannot separate fixed pixels from stars
        weak = ra.fixed_position_screen({"f1": a, "f2": b}, {"f1|f2": 2.0}, tolerance_px=1.0)["pairs"]["f1|f2"]
        self.assertFalse(weak["informative"])
        self.assertEqual(weak["candidate_count"], 1)
        empty = ra.fixed_position_screen({"f1": a, "f2": {"stars": []}})["pairs"]["f1|f2"]
        self.assertFalse(empty["informative"])

    def test_end_to_end_screen_uses_smallest_displacement(self):
        with tempfile.TemporaryDirectory() as d:
            pts = [(50.0, 40.0), (200.0, 150.0), (350.0, 40.0), (60.0, 260.0)]
            seq, lst = _write_sequence(Path(d), {1: translation(10.0, 0.0)},
                                       {"light_00001": pts, "light_00002": [(x - 10.0, y) for x, y in pts] + [(350.0, 40.0)]})
            result = ra.run_sequence_diagnostics(seq_path=seq, lst_paths=lst, width=400, height=300, hypothesis=TD, positions=[1], radius=3.0)
            pair = result["fixed_position_screen"]["pairs"]["light_00001|light_00002"]
            self.assertAlmostEqual(pair["expected_shift_px"], 10.0)
            self.assertTrue(pair["informative"])
            self.assertEqual(pair["candidate_count"], 1)      # the extra detection sits at the same pixel in both lists
            self.assertEqual(pair["candidates"][0]["xy_a"], [350.0, 40.0])


class UnregisteredFrame(unittest.TestCase):
    def test_zero_matrix_frame_fails_gracefully_and_screen_pairs_are_uninformative(self):
        """Siril writes a zero H (and deselects) a frame it could not match; nothing may crash."""
        with tempfile.TemporaryDirectory() as d:
            pts = [(50.0, 40.0), (200.0, 150.0), (350.0, 40.0), (60.0, 260.0), (120.0, 200.0), (300.0, 220.0)]
            seq, lst = _write_sequence(Path(d), {1: translation(10.0, 0.0), 2: np.zeros((3, 3))},
                                       {"light_00001": pts, "light_00002": [(x - 10.0, y) for x, y in pts], "light_00003": pts})
            result = ra.run_sequence_diagnostics(seq_path=seq, lst_paths=lst, width=400, height=300, hypothesis=TD,
                                                 positions=[1, 2], radius=3.0)
            frames = {f["frame_id"]: f for f in result["report"]["frames"]}
            self.assertEqual(frames["light_00002"]["state"], "succeeded")
            self.assertNotEqual(frames["light_00003"]["state"], "succeeded")
            self.assertEqual(result["summaries"]["light_00003"]["association"]["status"], "not_registered")
            self.assertFalse(result["summaries"]["light_00003"]["pooled"]["available"])
            pairs = result["fixed_position_screen"]["pairs"]
            self.assertTrue(pairs["light_00001|light_00002"]["informative"])
            self.assertFalse(pairs["light_00001|light_00003"]["informative"])
            self.assertIn("singular", pairs["light_00001|light_00003"]["reason"])
            self.assertIsNone(pairs["light_00002|light_00003"]["expected_shift_px"])


class PooledSummary(unittest.TestCase):
    def test_failed_frame_is_reported_unavailable(self):
        self.assertEqual(ra.pooled_summary({"state": "invalid", "error": "x"}), {"available": False, "reason": "x"})
        self.assertEqual(ra.grid_summary({"state": "invalid", "error": "x"}, TD, 10, 10), {"available": False, "reason": "x"})

    def test_pooling_by_region(self):
        residuals = [
            {"star_id": "a", "region": "center", "distance_px": 1.0, "dx_px": 1.0, "dy_px": 0.0, "reference_xy": [5.0, 5.0]},
            {"star_id": "b", "region": "top_left", "distance_px": 2.0, "dx_px": 0.0, "dy_px": 2.0, "reference_xy": [0.5, 0.5]},
            {"star_id": "c", "region": "bottom_right", "distance_px": 4.0, "dx_px": 0.0, "dy_px": -4.0, "reference_xy": [9.5, 9.5]},
            {"star_id": "d", "region": "other", "distance_px": 3.0, "dx_px": -3.0, "dy_px": 0.0, "reference_xy": [9.5, 0.5]},
        ]
        s = ra.pooled_summary({"state": "succeeded", "residuals": residuals}, {"b": {"saturated": True}})
        pooled = s["all_roles_pooled"]
        self.assertEqual(pooled["center"]["count"], 1)
        self.assertAlmostEqual(pooled["corners_pooled"]["rms_px"], math.sqrt((4 + 16) / 2))
        self.assertAlmostEqual(pooled["corners_pooled"]["mean_dy_px"], -1.0)
        self.assertAlmostEqual(pooled["other"]["mean_dx_px"], -3.0)
        self.assertAlmostEqual(pooled["overall"]["rms_px"], math.sqrt((1 + 4 + 16 + 9) / 4))
        self.assertEqual(s["unsaturated_only"]["corners_pooled"]["count"], 1)
        grid = ra.grid_summary({"state": "succeeded", "residuals": residuals}, BU, 10, 10, columns=2, rows=2)
        # BOTTOM-UP: reference_xy (0.5, 0.5) in TOP-DOWN is Siril y = 9 - 0.5 = 8.5 -> ynative_1
        self.assertIn("xcol_0/ynative_1", grid["cells"])
        self.assertEqual(grid["cells"]["xcol_0/ynative_1"]["count"], 1)
        self.assertEqual(sum(c["count"] for c in grid["cells"].values()), 4)


if __name__ == "__main__":
    unittest.main()
