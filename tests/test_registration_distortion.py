"""Tests for starstack.registration_distortion on synthetic correspondences (no Siril, no photographs).

Run:  python -B -m unittest tests.test_registration_distortion   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import registration_distortion as rd  # noqa: E402
from starstack.registration_models import apply_matrix  # noqa: E402

W, H = 2400, 1600


def _reference_stars(count=700, seed=11):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(20, W - 20, count), rng.uniform(20, H - 20, count)])


def _frame_matrix(span: int) -> np.ndarray:
    """Sky motion per unit span: mostly translation with a little rotation about a far point (reference -> frame)."""
    angle = np.radians(0.02 * span)
    c, s = np.cos(angle), np.sin(angle)
    tx, ty = 4.0 * span, 7.0 * span
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def _synthetic_frames(kind: str, spans, *, k1=0.02, noise=0.03, seed=5) -> dict:
    """Correspondences frame -> reference under a known shared effect plus centroid noise."""
    rng = np.random.default_rng(seed)
    u_ref = _reference_stars()
    scale = float(np.hypot(W, H) / 2.0)
    dist = rd.RadialDistortion(k1, 0.0, (W - 1) / 2.0, (H - 1) / 2.0, scale)
    frames = {}
    for span in spans:
        m = _frame_matrix(span)                      # reference (undistorted) -> frame (undistorted)
        u_frame = apply_matrix(m, u_ref)
        if kind == "radial":
            p_ref = dist.inverse(u_ref)              # observed = distorted
            p_frame = dist.inverse(u_frame)
        elif kind == "field":
            p_ref = u_ref
            # a vertical displacement growing towards the bottom, linear in span (refraction-like)
            v = np.column_stack([0.05 * (u_ref[:, 0] / W - 0.5), 0.25 * (u_ref[:, 1] / H) ** 2])
            p_frame = apply_matrix(m, u_ref - span * v)
        else:
            raise ValueError(kind)
        keep = (p_frame[:, 0] > 5) & (p_frame[:, 0] < W - 5) & (p_frame[:, 1] > 5) & (p_frame[:, 1] < H - 5)
        src = p_frame[keep] + rng.normal(0, noise, (keep.sum(), 2))
        dst = p_ref[keep] + rng.normal(0, noise, (keep.sum(), 2))
        # "Siril" H: a plain homography fitted on all pairs (what a global fit would do)
        from starstack.registration_models import fit_homography
        frames[f"syn_{span + 20:05d}"] = {"span": span, "src": src, "dst": dst, "H_siril": fit_homography(src, dst)}
    return frames


class RadialDistortionTest(unittest.TestCase):
    def test_inverse_round_trip_and_corner_shift(self):
        d = rd.RadialDistortion(0.03, -0.005, 1200.0, 800.0, 1440.0)
        p = np.array([[0.0, 0.0], [2399.0, 0.0], [1200.0, 800.0], [300.0, 1500.0]])
        u = d.apply(p)
        np.testing.assert_allclose(d.inverse(u), p, atol=1e-6)
        self.assertGreater(d.corner_shift_px(W, H), 20.0)
        self.assertEqual(rd.RadialDistortion(0, 0, 1, 1, 10).corner_shift_px(W, H), 0.0)

    def test_span_from_ids(self):
        self.assertEqual(rd.span_of("light_00016", "light_00011"), 5)
        self.assertEqual(rd.span_of("light_00001", "light_00011"), -10)
        with self.assertRaises(ValueError):
            rd.span_of("light", "light_00011")


class JointFitsTest(unittest.TestCase):
    def test_radial_truth_is_recovered_and_predicts_held_out_frames(self):
        spans = [-9, -7, -5, -3, -1, 1, 2, 3, 4, 5]
        frames = _synthetic_frames("radial", spans, k1=0.02)
        result = rd.analyse(frames, W, H, degrees=(2,))
        base = result["models"]["homography_per_frame"]["frames"]
        radial = result["models"]["homography_plus_shared_radial"]
        far = "syn_00011"   # span -9
        self.assertGreater(base[far]["residuals"]["validation"]["corner"]["rms_px"], 0.25)
        self.assertLess(radial["frames"][far]["residuals"]["validation"]["corner"]["rms_px"], 0.08)
        self.assertAlmostEqual(radial["parameters"]["k1"], 0.02, delta=0.004)
        self.assertLess(abs(radial["parameters"]["cx"] - (W - 1) / 2.0), 60.0)
        for fid in frames:
            self.assertLess(radial["frames"][fid]["residuals"]["validation"]["overall"]["rms_px"], 0.08, fid)
        transfer = result["transfer"]["fit_negative_span_predict_positive"]
        for fid, value in transfer["radial"]["held_out_validation_corner_rms_px"].items():
            self.assertLess(value, 0.1, fid)
        # the plain per-frame homography evaluated as-is ("siril") is the worst at the corners
        self.assertGreater(result["models"]["siril_H_as_is"]["frames"][far]["residuals"]["validation"]["corner"]["rms_px"], 0.2)

    def test_field_truth_is_explained_by_the_field_not_by_radial(self):
        spans = [-9, -7, -5, -3, -1, 1, 2, 3, 4, 5]
        frames = _synthetic_frames("field", spans)
        result = rd.analyse(frames, W, H, degrees=(2,))
        far = "syn_00011"
        base = result["models"]["homography_per_frame"]["frames"][far]["residuals"]["validation"]
        field = result["models"]["homography_plus_shared_field_deg2"]["frames"][far]["residuals"]["validation"]
        radial = result["models"]["homography_plus_shared_radial"]["frames"][far]["residuals"]["validation"]
        self.assertGreater(base["corner"]["rms_px"], 0.15)
        self.assertLess(field["corner"]["rms_px"], 0.08)
        self.assertGreater(radial["corner"]["rms_px"], field["corner"]["rms_px"] * 1.5)
        probes = result["models"]["homography_plus_shared_field_deg2"]["field_per_unit_span_px"]
        self.assertGreater(abs(probes["bottom_left"][1]), abs(probes["top_left"][1]))
        transfer = result["transfer"]["fit_short_spans_predict_long"]
        held = transfer["field_deg2"]["held_out_validation_corner_rms_px"]
        self.assertTrue(held)
        for fid, value in held.items():
            self.assertLess(value, 0.1, fid)

    def test_document_loader_and_grid_means(self):
        frames = _synthetic_frames("radial", [-2, 3])
        document = {"reference": {"frame_id": "syn_00020", "width": W, "height": H},
                    "frames": [{"frame_id": fid, "source_to_reference": f["H_siril"].tolist(),
                                "matches": [{"source_xy": s.tolist(), "reference_xy": d.tolist()} for s, d in zip(f["src"], f["dst"])]}
                               for fid, f in frames.items()]}
        document["frames"].append({"frame_id": "syn_00030", "source_to_reference": None, "matches": []})   # unregistered frame
        loaded = rd.frames_from_document(document, exclude=("syn_00023",))
        self.assertEqual(list(loaded), ["syn_00018"])
        self.assertEqual(loaded["syn_00018"]["span"], -2)
        self.assertEqual(loaded["syn_00018"]["src"].shape, frames["syn_00018"]["src"].shape)
        pred = frames["syn_00018"]["dst"] + np.array([0.5, -0.25])
        cells = rd.grid_signed_means(pred, frames["syn_00018"]["dst"], np.ones(len(pred), dtype=bool), W, H)
        self.assertEqual(len(cells), 12)
        for cell in cells.values():
            if cell["count"]:
                self.assertAlmostEqual(cell["mean_dx_px"], 0.5, places=9)
                self.assertAlmostEqual(cell["mean_dy_px"], -0.25, places=9)


class ReassociationTest(unittest.TestCase):
    def test_full_lists_are_paired_under_the_model_and_a_missing_frame_is_extrapolated(self):
        rng = np.random.default_rng(21)
        u_ref = _reference_stars(count=900, seed=13)
        scale = float(np.hypot(W, H) / 2.0)
        dist = rd.RadialDistortion(0.02, -0.004, (W - 1) / 2.0 + 20.0, (H - 1) / 2.0 - 10.0, scale)
        spans = [-9, -6, -3, -1, 2, 4, 6, 9]
        lists, h_true, span_map = {}, {}, {}
        ref_list = dist.inverse(u_ref) + rng.normal(0, 0.03, u_ref.shape)
        lists["syn_00020"] = ref_list
        for span in spans:
            m = _frame_matrix(span)                                  # reference -> frame (undistorted)
            p_frame = dist.inverse(apply_matrix(m, u_ref)) + rng.normal(0, 0.03, u_ref.shape)
            keep = (p_frame[:, 0] > 5) & (p_frame[:, 0] < W - 5) & (p_frame[:, 1] > 5) & (p_frame[:, 1] < H - 5)
            extra = np.column_stack([rng.uniform(5, W - 5, 40), rng.uniform(5, H - 5, 40)])   # unmatched detections
            frame_xy = np.vstack([p_frame[keep], extra])
            rng.shuffle(frame_xy)
            fid = f"syn_{span + 20:05d}"
            lists[fid], span_map[fid] = frame_xy, span
            h_true[fid] = np.linalg.inv(m)                            # frame -> reference (undistorted)
        known = {fid: h for fid, h in h_true.items() if fid != "syn_00029"}   # span +9 has no homography (as if Siril dropped it)
        result = rd.reassociate_sequence(dist, known, span_map, lists, "syn_00020", W, H)
        self.assertEqual(sorted(result), sorted(span_map))
        for fid, e in result.items():
            self.assertEqual(e["status"], "associated", fid)
            self.assertGreater(e["matched_fraction"], 0.85, fid)
            self.assertLess(e["residuals"]["validation"]["corner"]["rms_px"], 0.08, fid)
            self.assertLess(e["residuals"]["validation"]["overall"]["max_px"], 0.3, fid)
        self.assertEqual(result["syn_00029"]["initial_homography"], "extrapolated_from_neighbours")
        self.assertEqual(result["syn_00026"]["initial_homography"], "joint_fit")
        # a badly wrong start must be reported, not silently fitted
        bad = rd.reassociate_frame(dist, np.array([[1, 0, 900.0], [0, 1, 700.0], [0, 0, 1]]), lists["syn_00020"], lists["syn_00026"], W, H)
        self.assertEqual(bad["status"], "too_few_pairs")

    def test_extrapolation_follows_a_smooth_motion(self):
        known = {s: np.linalg.inv(_frame_matrix(s)) for s in (-4, -2, 0, 2, 4, 6)}
        predicted = rd.extrapolate_homography(known, 8)
        truth = np.linalg.inv(_frame_matrix(8))
        corners = np.array([[0, 0, 1], [W - 1, 0, 1], [0, H - 1, 1], [W - 1, H - 1, 1]], dtype=np.float64).T
        a, b = predicted @ corners, truth @ corners
        err = np.hypot(*((a[:2] / a[2]) - (b[:2] / b[2])))
        self.assertLess(err.max(), 1.0)


class EntryScriptTest(unittest.TestCase):
    def test_script_writes_report_from_document(self):
        frames = _synthetic_frames("radial", [-6, -4, -2, 2, 4, 6])
        document = {"schema_version": 1, "coordinate_convention": "test", "reference": {"frame_id": "syn_00020", "width": W, "height": H},
                    "frames": [{"frame_id": fid, "source_to_reference": f["H_siril"].tolist(),
                                "matches": [{"star_id": f"{fid}#{i}", "source_xy": s.tolist(), "reference_xy": d.tolist(), "role": "unknown"}
                                            for i, (s, d) in enumerate(zip(f["src"], f["dst"]), 1)]}
                               for fid, f in frames.items()]}
        with tempfile.TemporaryDirectory() as d:
            doc = Path(d) / "diagnostics_input.json"
            doc.write_text(json.dumps(document), encoding="utf-8")
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "s2_distortion_term.py"), "--document", str(doc),
                                   "--exclude-frames", "syn_00014", "--degrees", "2", "--output-root", str(Path(d) / "out")],
                                  cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            run = next((Path(d) / "out").glob("distortion-term-*"))
            result = json.loads((run / "distortion_term.json").read_text(encoding="utf-8"))
            self.assertEqual(result["frame_count"], 5)
            self.assertNotIn("syn_00014", result["spans"])
            self.assertIn("homography_plus_shared_radial", result["models"])
            text = (run / "DISTORTION.md").read_text(encoding="utf-8")
            self.assertIn("迁移测试", text)


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a.view(np.uint32) if a.dtype == np.float32 else a


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__, str(exc))


def _old_poly_inverse(self, q, iterations=8):
    """Verbatim copy of the pre-P1a ``RadialDistortion._poly_inverse`` (DISTORTION_VERSION 1.3.1): the oracle of G-2."""
    rq = np.hypot(q[:, 0], q[:, 1])
    r = rq.copy()
    for _ in range(iterations):
        f = r * (1.0 + self.k1 * r ** 2 + self.k2 * r ** 4) - rq
        df = 1.0 + 3.0 * self.k1 * r ** 2 + 5.0 * self.k2 * r ** 4
        r = r - f / np.where(np.abs(df) < 1e-12, 1e-12, df)
    factor = np.where(rq > 0, r / np.where(rq > 0, rq, 1.0), 1.0)
    return q * factor[:, None]


class PolyInverseTest(unittest.TestCase):
    """P1a G-2: the buffered Newton computes r**2 and r**4 once per iteration and keeps every result bit."""

    MODELS = {
        "real_u_to_p": rd.RadialDistortion(-0.052419985952268246, 0.019182497910353292, 4194.53, 2728.64, 4975.66, 0.96555, "u_to_p"),
        "real_p_to_u": rd.RadialDistortion(-0.052419985952268246, 0.019182497910353292, 4194.53, 2728.64, 4975.66, 0.96555, "p_to_u"),
        "pincushion": rd.RadialDistortion(0.0537, -0.0176, 4100.0, 2700.0, 4975.0, 1.0, "u_to_p"),
        "df_near_zero": rd.RadialDistortion(-1.0 / 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, "u_to_p"),
        "zero_k": rd.RadialDistortion(0.0, 0.0, 1.0, 2.0, 3.0, 1.0, "u_to_p"),
        "huge_k": rd.RadialDistortion(1e150, 1e200, 0.0, 0.0, 1.0, 1.0, "p_to_u"),
        "tiny_k": rd.RadialDistortion(1e-300, -1e-300, 0.5, 0.5, 1.0, 1.0, "p_to_u"),
    }

    def _points(self):
        rng = np.random.default_rng(29)
        payload = np.array([0x7FF8000000000001, 0xFFF8000000000ABC, 0x7FF4000000000000], dtype=np.uint64).view(np.float64)
        rows, cols = np.mgrid[2688:2816, 4096:4224]
        return {"random": np.column_stack([rng.uniform(-50, 8330, 20000), rng.uniform(-50, 5570, 20000)]),
                "block": np.column_stack([cols.ravel(), rows.ravel()]).astype(np.float64),
                "centre": np.array([[4194.53, 2728.64], [0.0, 0.0]]),
                "extreme": np.array([[1e300, 1e300], [1e-320, -1e-320], [0.0, -0.0], [np.inf, 0.0], [np.nan, 1.0]]),
                "nan_payloads": np.column_stack([payload, np.array([1.0, payload[1], 2.0])]),
                "single": np.array([[123.25, 456.75]]), "empty": np.zeros((0, 2))}

    def test_matches_reference(self):
        points = self._points()
        with np.errstate(all="ignore"):
            for (mname, m), (pname, p) in itertools.product(self.MODELS.items(), points.items()):
                for call, fn in (("apply", lambda: m.apply(p)), ("inverse", lambda: m.inverse(p)),
                                 ("inverse3", lambda: m.inverse(p, iterations=3)),
                                 ("raw0", lambda: m._poly_inverse(np.asarray(p, np.float64), 0)),
                                 ("raw_float32", lambda: m._poly_inverse(np.asarray(p, np.float32)))):
                    with self.subTest(model=mname, points=pname, call=call):
                        new = fn()
                        with mock.patch.object(rd.RadialDistortion, "_poly_inverse", _old_poly_inverse):
                            old = fn()
                        self.assertEqual((new.dtype, new.shape), (old.dtype, old.shape))
                        self.assertTrue(np.array_equal(_bits(new), _bits(old)))
                buf = np.empty(p.size + 1)
                shifted = buf[1:].reshape(p.shape)                   # 8-byte misaligned copy
                shifted[...] = p
                with self.subTest(model=mname, points=pname, call="misaligned"):
                    self.assertTrue(np.array_equal(_bits(m.apply(shifted)), _bits(m.apply(p))))

    def test_first_floating_point_error_unchanged(self):
        cases = [(rd.RadialDistortion(1e200, 1.0, 0.0, 0.0, 1.0, 1.0, "u_to_p"), np.array([[1e80, 0.0]])),
                 (rd.RadialDistortion(0.5, 1e300, 0.0, 0.0, 1.0, 1.0, "u_to_p"), np.array([[1e100, 3.0]])),
                 (rd.RadialDistortion(-0.05, 0.02, 0.0, 0.0, 1.0, 1.0, "u_to_p"), np.array([[1e160, 1e160]])),
                 (rd.RadialDistortion(-0.05, 0.02, 0.0, 0.0, 1.0, 1.0, "u_to_p"), np.array([[np.inf, 0.0]])),
                 (rd.RadialDistortion(1e-300, 1e-300, 0.0, 0.0, 1.0, 1.0, "u_to_p"), np.array([[1e-200, 0.0]]))]
        for m, q in cases:
            with self.subTest(k=(m.k1, m.k2), q=q.tolist()):
                with np.errstate(all="raise"):
                    new = _outcome(lambda: m._poly_inverse(q))
                    old = _outcome(lambda: _old_poly_inverse(m, q))
                self.assertEqual(new[0], "err")
                self.assertEqual(new[1:], old[1:])


if __name__ == "__main__":
    unittest.main()
