"""Tests for starstack.registration_calibration on simulated Siril products (no Siril, no photographs).

Run:  python -B -m unittest tests.test_registration_calibration   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import fits_crop  # noqa: E402
from starstack import registration_calibration as rc  # noqa: E402
from starstack import registration_distortion as rd  # noqa: E402
from starstack import resample  # noqa: E402
from starstack import siril_coordinate_proof as cp  # noqa: E402
from starstack import siril_sip_proof as sp  # noqa: E402
from starstack import synthetic_starfield as sf  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402


class CalibrateTest(unittest.TestCase):
    def test_distorted_sequence_is_fitted_and_written(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            truth = sp.generate_dataset(work)                       # distorted stars, undistorted truth
            sp.simulate_siril_outputs(truth, work, applied=False)  # what a plain two-pass probe leaves behind
            for path in work.glob("r_light_*.fit"):
                path.unlink()                                       # the probe writes no r_ frames
            messages = []
            summary = rc.calibrate(work, min_pairs=50, log=messages.append)
            self.assertEqual(summary["status"], "fitted", summary)
            self.assertEqual(summary["file"], "distortion.wcs")
            self.assertTrue((work / "distortion.wcs").exists())
            self.assertTrue((work / "distortion.json").exists())
            params = summary["parameters"]
            # the truth is a p_to_u model; the u_to_p fit is its inverse, so k1 comes out with the opposite sign (barrel: k1 < 0 here)
            self.assertAlmostEqual(params["k1"], -truth["distortion"]["k1"], delta=0.006)
            self.assertLess(params["magnification"], 1.0)
            self.assertLess(abs(params["cx"] - truth["distortion"]["cx"]), 40.0)
            v = summary["validation_corner_rms_median_px"]
            self.assertLess(v["with_distortion"], 0.5 * v["homography_only"])
            self.assertTrue(summary["improved"] and summary["sane"])
            self.assertEqual(summary["frames_dropped_by_siril"], [])
            self.assertGreaterEqual(summary["reassociation"]["matched_fraction_median"], 0.8)
            self.assertTrue(messages)
            header = (work / "distortion.wcs").read_bytes().decode("ascii")
            self.assertIn("CTYPE1  = 'RA---TAN-SIP'", header)
            self.assertIn("IMAGEW  =                 1200", header)
            self.assertIn("AP_ORDER=                    5", header)            # RUN_LOG §22: Siril crashes above order 5 ...
            self.assertEqual(summary["distortion_direction"], "u_to_p")         # ... so the model is fitted in the direction Siril applies
            self.assertTrue(summary["sip_inverse"]["exact"])
            self.assertEqual(summary["sip_inverse"]["max_fit_error_px"], 0.0)
            self.assertEqual(params.get("direction"), "u_to_p")
            self.assertFalse(summary["sip_forward"]["exact"])
            self.assertEqual(json.loads((work / "distortion.json").read_text(encoding="utf-8"))["status"], "fitted")
            # RUN_LOG §22: every frame gets its own homography under the final model, in Siril's convention too
            registration = summary["registration"]
            self.assertEqual(registration["registered"], 4, registration["unregistered_frames"])
            self.assertEqual(registration["reference"], summary["reference"])
            frames = registration["frames"]
            self.assertEqual({f["status"] for f in frames.values()}, {"registered"})
            self.assertEqual(frames[summary["reference"]]["H_top_down_magnified"], np.eye(3).tolist())
            magnified = RadialDistortion(**params)
            truth_frames = {f["frame"]: f for f in truth["frames"]}
            ref_no = int(summary["reference"][-5:])
            ref_by_id = {s["star_id"]: (s["x"], s["y"]) for s in truth_frames[ref_no]["rendered_stars"]}
            for stem, f in frames.items():
                if stem == summary["reference"]:
                    continue
                self.assertGreaterEqual(f["pairs"], 100, stem)
                self.assertLess(f["validation"]["overall_rms_px"], 0.15, stem)
                # the written matrix maps this frame's *recorded* (distorted) stars onto the reference's stars in the magnified
                # undistorted grid, i.e. where Siril's r_ frames put them
                tf = truth_frames[int(stem[-5:])]
                common = [st for st in tf["rendered_stars"] if st["star_id"] in ref_by_id]
                p_frame = np.array([[st["x"], st["y"]] for st in common])
                p_ref = np.array([ref_by_id[st["star_id"]] for st in common])
                h = np.asarray(f["H_top_down_magnified"])
                v = np.hstack([magnified.apply(p_frame), np.ones((len(common), 1))]) @ h.T
                err = np.hypot(*(v[:, :2] / v[:, 2:3] - magnified.apply(p_ref)).T)
                self.assertLess(float(np.median(err)), 0.1, stem)
                # Siril's stored form reads back to the same matrix through the proved hypothesis
                from starstack import registration_adapter as adapter
                back = adapter.convert_matrix(adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS], f["H_siril"], truth["height"], truth["height"])
                np.testing.assert_allclose(back / back[2, 2], h / h[2, 2], atol=1e-6)
            # the sequence file the product hands to seqapplyreg
            written = rc.write_registration_sequence(work, summary)
            self.assertEqual(written["registered"], 4)
            from starstack import siril_sequence as ss
            seq = ss.read_seq(work / "light_.seq")
            self.assertEqual(seq["header"]["nb_selected"], 4)
            self.assertEqual(seq["header"]["reference_image"], summary["sequence_template"]["header"]["reference_image"])
            self.assertIn("D1 2 distortion.wcs", (work / "light_.seq").read_text(encoding="utf-8").splitlines())
            for image in seq["images"]:
                self.assertTrue(image["selected"])
                stored = np.asarray(seq["regdata"][1][image["position"]]["H"])
                np.testing.assert_allclose(stored, np.asarray(frames[image["stem"]]["H_siril"]), atol=1e-9)
                self.assertGreater(seq["regdata"][1][image["position"]]["number_of_stars"], 0)
            # a frame the registration rejected is deselected and written in Siril's zero form
            rejected = dict(summary, registration=json.loads(json.dumps(registration)))
            victim = next(stem for stem in frames if stem != summary["reference"])
            rejected["registration"]["frames"][victim].update(status="not_registered", reason="test", H_siril=None)
            rejected["registration"]["registered"] = 3
            rejected["registration"]["unregistered_frames"] = [victim]
            written2 = rc.write_registration_sequence(work, rejected)
            self.assertEqual((written2["registered"], written2["unregistered_frames"]), (3, [victim]))
            seq2 = ss.read_seq(work / "light_.seq")
            pos = next(im["position"] for im in seq2["images"] if im["stem"] == victim)
            self.assertFalse(seq2["images"][pos]["selected"])
            self.assertEqual(seq2["regdata"][1][pos]["H"], [[0.0] * 3] * 3)
            self.assertEqual(seq2["header"]["nb_selected"], 3)
            with self.assertRaises(ValueError):
                rc.write_registration_sequence(work, {"status": "fitted"})

    def test_cancellation_probe_is_called_and_propagates(self):
        """RUN_LOG §24: ``check`` is the job's cancel probe — called throughout the fit, and what it raises comes out
        unchanged (not swallowed into status 'unavailable'); a cancelled calibration writes nothing."""
        from starstack.siril import Cancelled
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            truth = sp.generate_dataset(work)
            sp.simulate_siril_outputs(truth, work, applied=False)
            for path in work.glob("r_light_*.fit"):
                path.unlink()
            calls = []
            summary = rc.calibrate(work, min_pairs=50, check=lambda: calls.append(1))
            self.assertEqual(summary["status"], "fitted")
            self.assertGreaterEqual(len(calls), 20, "the probe must run inside the fit iterations and per-frame loops, not just once")
            (work / "distortion.json").unlink()
            (work / "distortion.wcs").unlink()
            # cancel part-way: after the n-th probe the job is cancelled
            for cut in (1, 3, len(calls) // 2, len(calls) - 1):
                count = {"n": 0}

                def probe(cut=cut):
                    count["n"] += 1
                    if count["n"] > cut:
                        raise Cancelled("用户取消")
                with self.assertRaises(Cancelled):
                    rc.calibrate(work, min_pairs=50, check=probe)
                self.assertFalse((work / "distortion.json").exists(), f"cut {cut}: a cancelled calibration must not record a result")
                self.assertFalse((work / "distortion.wcs").exists(), f"cut {cut}: a cancelled calibration must not write the distortion file")
            # other failures are still calibration outcomes, not job failures
            summary = rc.calibrate(work, min_pairs=50, check=lambda: (_ for _ in ()).throw(ValueError("probe broke")))
            self.assertEqual(summary["status"], "unavailable")
            self.assertIn("ValueError", summary["reason"])

    def test_undistorted_sequence_reports_no_improvement_and_writes_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            truth = sf.generate_dataset(work, stem="light")
            cp.simulate_siril_outputs(truth, work, row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5, matrix_direction="frame_to_reference")
            summary = rc.calibrate(work, min_pairs=50)
            self.assertEqual(summary["status"], "no_improvement", summary)
            self.assertIsNone(summary["file"])
            self.assertFalse((work / "distortion.wcs").exists())
            self.assertTrue((work / "distortion.json").exists())

    def test_missing_products_are_unavailable_not_errors(self):
        with tempfile.TemporaryDirectory() as d:
            summary = rc.calibrate(Path(d))
            self.assertEqual(summary["status"], "unavailable")
            self.assertIn("sequence", summary["reason"])
            self.assertIsNone(rc.load_calibration(Path(d) / "nowhere"))
            self.assertEqual(rc.load_calibration(Path(d))["status"], "unavailable")


class QualityGateTest(unittest.TestCase):
    """RUN_LOG §27: frames far off the sequence's own detection quality stay out of the fit and the stack."""

    def _template(self, n=10, bad=None):
        images = [{"index": i + 1, "position": i, "stem": f"light_{i + 1:05d}"} for i in range(n)]
        quality = {str(i): [3.5, 3.6, 0.75, 0.0, 0.02, 1800] for i in range(n)}
        for pos, fields in (bad or {}).items():
            quality[str(pos)] = fields
        return {"header": {}, "layers": 3, "images": images, "quality": quality}

    def test_outlier_frame_is_excluded_by_relative_thresholds(self):
        template = self._template(bad={0: [4.5, 10.5, 0.66, 0.0, 0.02, 640]})            # frame 1 of the Z8 night, in numbers
        gate = rc.quality_gate(template, "light_00005")
        self.assertTrue(gate["applied"])
        self.assertEqual(gate["excluded_frames"], ["light_00001"])
        reasons = gate["excluded"][0]["reasons"]
        self.assertTrue(any(r.startswith("wfwhm") for r in reasons) and any(r.startswith("stars") for r in reasons))
        self.assertEqual(gate["frames"]["light_00002"]["flags"], [])
        self.assertAlmostEqual(gate["frames"]["light_00001"]["ratios"]["fwhm"], 4.5 / 3.5, places=3)

    def test_reference_is_never_excluded_and_small_sequences_only_report(self):
        template = self._template(bad={4: [6.0, 8.0, 0.6, 0.0, 0.02, 500]})
        gate = rc.quality_gate(template, "light_00005")                                   # the flagged frame is the reference
        self.assertEqual(gate["excluded_frames"], [])
        self.assertTrue(gate["reference_flagged"])
        small = self._template(n=6, bad={0: [6.0, 8.0, 0.6, 0.0, 0.02, 500]})
        gate = rc.quality_gate(small, "light_00003")
        self.assertFalse(gate["applied"])
        self.assertEqual(gate["excluded_frames"], [])
        self.assertEqual(gate["flagged_not_applied"], ["light_00001"])
        gate = rc.quality_gate(small, "light_00003", thresholds={"min_frames": 4})
        self.assertEqual(gate["excluded_frames"], ["light_00001"])
        gate = rc.quality_gate(template, "light_00002", thresholds={"enabled": False})
        self.assertFalse(gate["applied"]); self.assertEqual(gate["excluded_frames"], [])

    def test_holdout_rms_gate(self):
        template = self._template()
        rms = {f"light_{i + 1:05d}": 0.2 for i in range(10)}; rms["light_00007"] = 0.55
        gate = rc.quality_gate(template, "light_00005", holdout_rms=rms)
        self.assertEqual(gate["excluded_frames"], ["light_00007"])
        self.assertIn("conditional validation rms", gate["excluded"][0]["reasons"][0])
        self.assertEqual(gate["residual_semantics"], "conditional_validation_not_independent")

    def test_calibrate_excludes_a_flagged_frame_and_the_sequence_deselects_it(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            truth = sp.generate_dataset(work)
            sp.simulate_siril_outputs(truth, work, applied=False)
            for path in work.glob("r_light_*.fit"):
                path.unlink()
            # make frame 3 a blurry, star-poor frame in Siril's quality fields (the star list itself is untouched)
            lines = (work / "light_.seq").read_text(encoding="utf-8").splitlines()
            r_seen = 0
            for i, line in enumerate(lines):
                if line.startswith("R1 "):
                    r_seen += 1
                    if r_seen == 3:
                        tokens = line.split()
                        tokens[1:7] = ["5.5", "9.0", "0.6", "0", "0.02", "60"]
                        lines[i] = " ".join(tokens)
            probe_seq = "\n".join(lines) + "\n"
            (work / "light_.seq").write_text(probe_seq, encoding="utf-8")
            summary = rc.calibrate(work, min_pairs=50, min_frames=2, quality_gate_thresholds={"min_frames": 3})   # 4 frames: gate on, 2 left besides the reference
            self.assertEqual(summary["status"], "fitted", summary.get("reason"))
            gate = summary["quality_gate"]
            self.assertEqual(gate["excluded_frames"], ["light_00003"])
            reg = summary["registration"]
            self.assertEqual(reg["frames"]["light_00003"]["status"], "excluded_quality")
            self.assertIn("quality gate", reg["frames"]["light_00003"]["reason"])
            self.assertNotIn("light_00003", reg["registered_frames"])
            self.assertEqual(reg["quality_excluded_frames"], ["light_00003"])
            self.assertEqual(reg["registered"], len(truth["frames"]) - 1)
            self.assertNotIn("light_00003", summary["fit_frames"])
            written = rc.write_registration_sequence(work, summary)
            self.assertEqual(written["registered"], len(truth["frames"]) - 1)
            seq = (work / "light_.seq").read_text(encoding="utf-8").splitlines()
            self.assertIn("I 3 0", seq)
            r_lines = [l for l in seq if l.startswith("R1 ")]
            self.assertTrue(r_lines[2].startswith("R1 0 0 0 0 0 0 H 0 0 0"))
            # off: nothing excluded, everything else equal (the probe sequence restored: the writer above deselected frame 3)
            (work / "distortion.json").unlink(); (work / "distortion.wcs").unlink()
            (work / "light_.seq").write_text(probe_seq, encoding="utf-8")
            summary_off = rc.calibrate(work, min_pairs=50, min_frames=2, quality_gate_thresholds={"enabled": False})
            self.assertEqual(summary_off["quality_gate"]["excluded_frames"], [])
            self.assertEqual(summary_off["registration"]["registered"], len(truth["frames"]))


class WarpBackTest(unittest.TestCase):
    def test_native_from_undistorted_puts_stars_back(self):
        layout = sf.default_layout()
        width, height = layout["width"], layout["height"]
        d = sp.default_distortion(width, height)
        model = d.with_magnification(d.fitting_magnification(width, height))
        native_xy = np.array([[s["x"], s["y"]] for s in layout["stars"]], dtype=np.float64)
        # the "stack" shows those stars at their (magnified) undistorted positions
        stack_layout = {"width": width, "height": height,
                        "stars": [{**s, "x": float(ux), "y": float(uy)} for s, (ux, uy) in zip(layout["stars"], model.apply(native_xy))]}
        image, _ = sf.render_frame(stack_layout, np.eye(3), seed=8)
        stack_bottom_up = np.repeat((image.astype(np.float32) / 65535.0)[::-1][..., None], 3, axis=2)
        native = rc.native_from_undistorted(stack_bottom_up, model.as_dict())
        self.assertEqual(native.shape, stack_bottom_up.shape)
        plane_td = native[::-1, :, 1] * 65535.0
        errs = []
        for x, y in native_xy:
            if 12 <= x < width - 12 and 12 <= y < height - 12:
                c = fits_crop._centroid(plane_td, float(x), float(y), 4)
                if c and c[3] - c[2] > 50:
                    errs.append(np.hypot(c[0] - x, c[1] - y))
        self.assertGreater(len(errs), 140)
        self.assertLess(float(np.median(errs)), 0.05)
        self.assertLess(float(np.max(errs)), 0.3)
        self.assertTrue(np.isfinite(native).all())

    def test_native_from_undistorted_matches_per_plane_oracle(self):
        """P1a G-1: the fused one-pass warp gives the bits, flags and errors of the former per-plane code."""
        def bits(a):
            return a.view(np.uint32) if a.dtype == np.float32 else a

        def outcome(fn):
            try:
                return ("ok", fn())
            except Exception as exc:  # noqa: BLE001
                return ("err", type(exc).__name__, str(exc))

        h, w = 67, 131
        rng = np.random.default_rng(13)
        model = rd.RadialDistortion(-0.05, 0.02, 65.0, 33.0, 80.0, 0.96, "u_to_p")
        params = model.as_dict()
        planar_bu = np.moveaxis(np.ascontiguousarray((rng.random((3, h, w)) * 0.8).astype(np.float32)[:, ::-1]), 0, -1)
        inputs = {"planar_bottom_up": planar_bu, "interleaved": np.ascontiguousarray(planar_bu),
                  "float64": planar_bu.astype(np.float64), "big_endian": planar_bu.astype(">f4"),
                  "one_channel": planar_bu[..., :1], "four_channels": np.concatenate([planar_bu, planar_bu[..., :1]], axis=-1),
                  "two_d": planar_bu[..., 0], "four_d": planar_bu[None], "zero_channels": planar_bu[..., :0],
                  "zero_rows": planar_bu[:0]}
        for (name, x), kernel in itertools.product(inputs.items(), (None, "", "cubic", "lanczos3", "nope")):
            with self.subTest(input=name, kernel=kernel):
                a = outcome(lambda: _former_native_from_undistorted(x, params, kernel=kernel))
                b = outcome(lambda: rc.native_from_undistorted(x, params, kernel=kernel))
                self.assertEqual(a[0], b[0])
                if a[0] == "err":
                    self.assertEqual(a[1:], b[1:])
                else:
                    self.assertEqual((a[1].shape, a[1].dtype, a[1].flags.c_contiguous),
                                     (b[1].shape, b[1].dtype, b[1].flags.c_contiguous))
                    self.assertTrue(np.array_equal(bits(a[1]), bits(b[1])))
        # the product input (planar BOTTOM-UP, 3 channels) takes the fused path, once; a 0-channel image the former path
        with mock.patch.object(resample, "warp_planes_fused", wraps=resample.warp_planes_fused) as fused:
            native = rc.native_from_undistorted(planar_bu, params)
            self.assertEqual(fused.call_count, 1)
            self.assertTrue(native.flags.c_contiguous and native.flags.owndata and native.flags.writeable)
            outcome(lambda: rc.native_from_undistorted(planar_bu[..., :0], params))
            self.assertEqual(fused.call_count, 1)


def _former_native_from_undistorted(x, parameters, *, kernel=None):
    """The pre-P1a body of ``native_from_undistorted`` (per-plane ``warp_planes``), the oracle of G-1."""
    model = rd.RadialDistortion(**parameters)
    top_down = np.moveaxis(np.asarray(x, dtype=np.float32)[::-1], -1, 0)
    kwargs = {"kernel": kernel} if kernel else {}
    warped = resample.warp_planes(top_down, model.apply, fill=0.0, **kwargs)
    return np.ascontiguousarray(np.moveaxis(warped, 0, -1)[::-1])


if __name__ == "__main__":
    unittest.main()
