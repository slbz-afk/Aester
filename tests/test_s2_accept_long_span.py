"""Tests for scripts/s2_accept_long_span.py on synthetic products (no Siril, no photographs).

Run:  python -B -m unittest tests.test_s2_accept_long_span   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from starstack import core  # noqa: E402
from starstack import registration_calibration as rc  # noqa: E402
from starstack import siril_sip_proof as sp  # noqa: E402
from starstack import synthetic_starfield as sf  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402
import test_s2_accept_two_pass as twopass  # noqa: E402

BOXES = twopass.BOXES


def _module():
    spec = importlib.util.spec_from_file_location("s2_accept_long_span", ROOT / "scripts" / "s2_accept_long_span.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_run(root: Path, *, dropped: list[str], recovered: list[str], ratio: float = 0.2,
                   layout: str = "combined") -> tuple[Path, Path]:
    """A finished distortion-flow job as the product would leave it, inside a run directory with inputs.json.
    ``layout``: the stack-phase Siril files, pre-P1b "combined" or "split" (efficiency P1b-3)."""
    run = root / "run"
    job = run / "jobs" / "job-synthetic"
    work = job / "work"
    truth = sp.generate_dataset(work)
    sp.simulate_siril_outputs(truth, work, applied=True)
    model = RadialDistortion(**truth["distortion"])
    _, ref_plane = sf.read_fits_minimal(work / "r_light_00001.fit")
    stack = np.repeat((ref_plane.astype(np.float32) / 65535.0)[::-1][..., None], 3, axis=2)
    twopass._write_float_master(work / "sky_master_undistorted.fit", ref_plane, 4)
    native = rc.native_from_undistorted(stack, model.as_dict())
    twopass._write_float_master(work / "sky_master.fit", np.clip(native[::-1, :, 1] * 65535.0, 0, 65535).astype(np.uint16), 4)
    (work / rc.DISTORTION_WCS).write_bytes(b"synthetic")
    frames = {f"light_0000{k}": {"span": k - 1, "status": "associated", "initial_homography": "extrapolated_from_neighbours" if f"light_0000{k}" in dropped else "joint_fit",
                                  "matched": 165, "frame_count": 170, "matched_fraction": 0.97, "corner_rms_px": 0.06, "overall_rms_px": 0.05} for k in range(2, 5)}
    calibration = {"status": "fitted", "file": rc.DISTORTION_WCS, "parameters": model.as_dict(), "corner_shift_px": 11.9, "magnification": 1.0,
                   "validation_corner_rms_median_px": {"homography_only": 0.30, "with_distortion": 0.30 * ratio}, "frames_dropped_by_siril": dropped,
                   "fit_frames": sorted(frames), "seconds": 0.5,
                   "reassociation": {"frames": frames, "matched_fraction_median": 0.97, "corner_rms_median_px": 0.06, "recovered_dropped_frames": recovered}}
    (work / rc.DISTORTION_JSON).write_text(json.dumps(calibration), encoding="utf-8")
    inputs, records = [], []
    for k in range(1, 5):
        path = run / "inputs" / f"frame0000{k}.dng"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw" * k)
        inputs.append({"path": str(path), "size": path.stat().st_size, "sha256": f"sha{k}"})
        records.append({"frame": k, "stem": f"light_0000{k}", "source": str(root / "dng" / f"DSC_{2203 + 3 * k}.dng"), "size": 3 * k})
    (run / "inputs.json").write_text(json.dumps({"method": "hardlink", "every": 3, "start": 0, "raw_dir": str(root / "dng"), "count": 4, "frames": records}), encoding="utf-8")
    (job / "02_probe.ssf").write_text("requires 1.4.4 1.5.0\n" + "\n".join(core.PROBE_SCRIPT) + "\n", encoding="utf-8")
    twopass.write_stack_phase(job, core.OWN_REGISTRATION_STACK_SCRIPT, "log: choosing image 1 as new reference for registration\n",
                              "log: Total execution time: 20.0 s\n", layout)
    (job / "02_probe.log").write_text("log: Total execution time: 8.0 s\n", encoding="utf-8")
    (job / "01_convert.log").write_text("log: Total execution time: 5.0 s\n", encoding="utf-8")
    diagnostics = {"input_count": 4, "stacked_count": 4, "skipped_count": 0, "reference_selection": core.REFERENCE_SELECTION,
                   "reference_input": inputs[0]["path"], "reference_stem": "light_00001", "reference_position": 0,
                   "registration": core.registration_commands(core.DISTORTION_WCS, own_registration=True), "registration_source": core.REGISTRATION_SOURCE_OWN,
                   "registration_detail": {"source": core.REGISTRATION_SOURCE_OWN, "registered": 4 - len(dropped) + len(recovered), "unregistered_frames": [],
                                           "validation_overall_rms_median_px": 0.05, "validation_corner_rms_median_px": 0.06},
                   "sky_geometry": "native_via_undistorted_stack",
                   "distortion": {**{k: calibration[k] for k in ("status", "parameters", "corner_shift_px", "magnification")}, "applied": True}}
    (job / "job.json").write_text(json.dumps({"schema_version": 1, "job_dir": str(job), "state": "succeeded", "stage": "complete", "progress": 100,
                                              "message": "", "error": None, "created_at": "2026-09-17T00:00:00+00:00", "updated_at": "2026-09-17T00:00:00+00:00",
                                              "inputs": inputs, "checkpoints": ["convert", "distortion", "stack"], "checkpoint_files": {}, "artifacts": [],
                                              "limitations": [], "engine": {"executable": "fake", "version": "1.4.4"}, "diagnostics": diagnostics}), encoding="utf-8")
    return run, job


class EndToEndTest(unittest.TestCase):
    def test_main_evaluates_a_long_span_job(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run, job = _synthetic_run(root, dropped=["light_00004"], recovered=["light_00004"])
            out = root / "out"
            m.BOXES.clear()
            m.BOXES.update(BOXES)
            code = m.main(["--job-dir", str(job), "--output-root", str(out), "--frames", "2", "3", "4", "--cadence-seconds", "13"])
            new_run = next(out.glob("accept-long-span-*"))
            report = json.loads((new_run / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0, report["verdict"])
            self.assertTrue(report["T1"]["checks"]["inputs_are_the_selected_raws"])
            self.assertTrue(report["T1"]["checks"]["stack_script_matches_product"])
            self.assertTrue(report["T1"]["checks"]["registration_commands"])
            self.assertEqual(report["T1"]["registration_source"], core.REGISTRATION_SOURCE_OWN)
            self.assertTrue(report["T1"]["checks"]["registered_frames_selected_in_sequence"])
            self.assertTrue(report["T1"]["checks"]["stacked_count_equals_registered"])
            self.assertTrue(report["T1"]["checks"]["registered_fraction_at_least"])
            self.assertEqual(report["T1"]["excluded_frames"], {})
            self.assertTrue(report["T2"]["dropped_all_recovered"])
            self.assertTrue(report["T3"]["judgement"]["passed"], report["T3"]["judgement"])
            spans = {row["frame"]: row for row in report["T3"]["span_table"]}
            self.assertEqual(spans["light_00004"]["span_images"], 3)
            self.assertAlmostEqual(spans["light_00004"]["span_minutes"], 3 * 39 / 60, places=6)
            self.assertLess(spans["light_00004"]["max_rms_px"], 0.15)
            self.assertTrue(report["T3b"]["judgement"]["passed"], report["T3b"]["judgement"])
            self.assertTrue(report["T4"]["judgement"]["passed"], report["T4"]["judgement"])
            for name, entry in report["T4"]["results"].items():
                p = entry["paired_comparison"]
                self.assertGreaterEqual(p["paired"], 5, name)
                self.assertLessEqual(p["new_median_sigma_major_px"] / p["old_median_sigma_major_px"], m.CRITERIA["T4"]["max_ratio_master_over_frame"], name)
            # the T4 measurement is position-matched and unbiased: a master made of the reference frame itself gives ratio 1
            work = job / "work"
            _, light_plane = sf.read_fits_minimal(work / "light_00001.fit")
            twopass._write_float_master(work / "sky_master.fit", light_plane, 4)
            (root / "same").mkdir()
            same = m.master_vs_reference_frame(work, m.siril_sequence.read_seq(work / core.SEQUENCE_FILE), m.adapter.HYPOTHESES[m.adapter.DEFAULT_HYPOTHESIS],
                                               m.BOXES, 1, root / "same", log=lambda *_: None)
            for name, entry in same["results"].items():
                p = entry["paired_comparison"]
                self.assertLess(abs(p["new_median_sigma_major_px"] / p["old_median_sigma_major_px"] - 1.0), 0.01, name)
                self.assertLess(abs(p["median_ratio_sigma_minor"] - 1.0), 0.01, name)
            self.assertEqual(report["inputs"]["reference_source"], str(root / "dng" / "DSC_2206.dng"))
            text = (new_run / "ACCEPTANCE.md").read_text(encoding="utf-8")
            self.assertIn("通过", text)
            self.assertIn("跨度趋势", text)

    def test_dropped_frame_not_recovered_fails_t2(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run, job = _synthetic_run(root, dropped=["light_00004"], recovered=[])
            out = root / "out"
            m.BOXES.clear()
            m.BOXES.update(BOXES)
            code = m.main(["--job-dir", str(job), "--output-root", str(out), "--frames", "2", "3", "4"])
            report = json.loads((next(out.glob("accept-long-span-*")) / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 2)
            self.assertFalse(report["T2"]["passed"])
            self.assertEqual(report["T2"]["dropped_not_recovered"], ["light_00004"])
            self.assertTrue(any(f.startswith("T2") for f in report["verdict"]["failures"]))

    def test_t1_accepts_both_stack_layouts(self):
        """Efficiency P1b-3: the T1 script and reference-log checks hold for the split layout as before."""
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            for layout in ("combined", "split"):
                _, job = _synthetic_run(Path(d) / layout, dropped=[], recovered=[], layout=layout)
                record = json.loads((job / "job.json").read_text(encoding="utf-8"))
                sequence = m.siril_sequence.read_seq(job / "work" / core.SEQUENCE_FILE)
                checks = m.job_checks(record, job, sequence, None)["checks"]
                self.assertIs(checks["stack_script_matches_product"], True, layout)
                self.assertIs(checks["reference_matches_siril_log"], True, layout)


class RegisteredFractionTest(unittest.TestCase):
    def test_excluded_frame_counts_against_the_fraction_and_is_listed(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run, job = _synthetic_run(root, dropped=[], recovered=[])
            data = json.loads((job / "job.json").read_text(encoding="utf-8"))
            detail = data["diagnostics"]["registration_detail"]
            detail["registered"] = 3
            detail["unregistered_frames"] = ["light_00004"]
            detail["frames"] = {"light_00004": {"status": "not_registered", "reason": "only 3 pairs at 3 px"}}
            data["diagnostics"]["stacked_count"] = 3
            (job / "job.json").write_text(json.dumps(data), encoding="utf-8")
            # the product would have deselected the frame in the sequence and exported three r_ frames
            seq_text = (job / "work" / "light_.seq").read_text(encoding="utf-8").replace("I 4 1", "I 4 0")
            (job / "work" / "light_.seq").write_text(seq_text, encoding="utf-8")
            (job / "work" / "r_light_00004.fit").unlink()
            out = root / "out"
            m.BOXES.clear()
            m.BOXES.update(BOXES)
            m.main(["--job-dir", str(job), "--output-root", str(out), "--frames", "2", "3", "--min-registered-fraction", "0.7"])
            report = json.loads((next(out.glob("accept-long-span-*")) / "acceptance.json").read_text(encoding="utf-8"))
            t1 = report["T1"]
            self.assertTrue(t1["checks"]["registered_fraction_at_least"])
            self.assertTrue(t1["checks"]["stacked_count_equals_registered"])
            self.assertEqual(t1["excluded_frames"], {"light_00004": "only 3 pairs at 3 px"})
            out2 = root / "out2"
            m.main(["--job-dir", str(job), "--output-root", str(out2), "--frames", "2", "3", "--min-registered-fraction", "1.0"])
            report2 = json.loads((next(out2.glob("accept-long-span-*")) / "acceptance.json").read_text(encoding="utf-8"))
            self.assertFalse(report2["T1"]["checks"]["registered_fraction_at_least"])
            self.assertTrue(any(f.startswith("T1 registered_fraction") for f in report2["verdict"]["failures"]))


class QualityGateExclusionTest(unittest.TestCase):
    def test_gate_excluded_frame_is_expected_and_not_judged(self):
        """RUN_LOG §27: a frame the quality gate kept out is listed with reasons, deselected, not exported, does not count
        against the registered fraction and is not judged by T2/T3."""
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run, job = _synthetic_run(root, dropped=[], recovered=[])
            data = json.loads((job / "job.json").read_text(encoding="utf-8"))
            detail = data["diagnostics"]["registration_detail"]
            detail["registered"] = 3
            detail["unregistered_frames"] = ["light_00004"]
            detail["frames"] = {"light_00004": {"status": "excluded_quality", "reason": "quality gate: wfwhm 2.9x median; stars 0.35x median"}}
            data["diagnostics"]["stacked_count"] = 3
            data["diagnostics"]["quality_gate"] = {"applied": True, "in_effect": True, "thresholds": {"wfwhm_ratio": 1.5, "fwhm_ratio": 1.3, "nstars_ratio": 0.5, "holdout_rms_ratio": 2.0},
                                                    "medians": {"fwhm": 3.5, "wfwhm": 3.6, "nstars": 1800, "holdout_rms": 0.2},
                                                    "excluded": [{"stem": "light_00004", "reasons": ["wfwhm 2.9x median", "stars 0.35x median"]}], "excluded_frames": ["light_00004"]}
            (job / "job.json").write_text(json.dumps(data), encoding="utf-8")
            # the calibration record: frame 4 fails T2's own thresholds (it is blurry) — must not count once excluded
            cal = json.loads((job / "work" / "distortion.json").read_text(encoding="utf-8"))
            cal["reassociation"]["frames"]["light_00004"].update({"matched_fraction": 0.4, "corner_rms_px": 1.4})
            (job / "work" / "distortion.json").write_text(json.dumps(cal), encoding="utf-8")
            seq_text = (job / "work" / "light_.seq").read_text(encoding="utf-8").replace("I 4 1", "I 4 0")
            (job / "work" / "light_.seq").write_text(seq_text, encoding="utf-8")
            (job / "work" / "r_light_00004.fit").unlink()
            out = root / "out"
            m.BOXES.clear(); m.BOXES.update(BOXES)
            m.main(["--job-dir", str(job), "--output-root", str(out), "--frames", "2", "3", "4", "--min-registered-fraction", "1.0"])
            report = json.loads((next(out.glob("accept-long-span-*")) / "acceptance.json").read_text(encoding="utf-8"))
            t1 = report["T1"]
            self.assertTrue(t1["checks"]["registered_fraction_at_least"])                      # 3 of the 3 non-excluded frames
            self.assertTrue(t1["checks"]["quality_excluded_listed_with_reasons"])
            self.assertTrue(t1["checks"]["quality_excluded_deselected_and_not_exported"])
            self.assertTrue(t1["passed"], t1["checks"])
            self.assertEqual(report["T2"]["quality_excluded"], ["light_00004"])
            self.assertNotIn("light_00004", report["T2"]["failing"]); self.assertNotIn("light_00004", report["T2"]["tolerated"])
            self.assertTrue(report["T2"]["passed"], report["T2"])
            self.assertNotIn("light_00004", [v["frame"] for v in report["T3"]["results"].values()])
            self.assertTrue(report["verdict"]["passed"], report["verdict"])
            text = (next(out.glob("accept-long-span-*")) / "ACCEPTANCE.md").read_text(encoding="utf-8")
            self.assertIn("像质门槛", text); self.assertIn("light_00004", text)


class BoxRuleTest(unittest.TestCase):
    def test_boxes_lie_inside_the_common_sky_coverage(self):
        m = _module()
        width, height = 8280, 5520
        model = RadialDistortion(-0.05, 0.012, 4140.0, 2760.0, float(np.hypot(width, height) / 2.0), direction="u_to_p")
        model = model.with_magnification(model.fitting_magnification(width, height))
        frames = {}
        for k, (dx, dy) in enumerate([(0.0, 0.0), (-380.0, -560.0), (185.0, 300.0), (-120.0, 40.0)], 1):
            # frame -> reference: the frame's content sits at reference position + (dx, dy) ... i.e. H maps u_frame to u_frame + (dx, dy)
            frames[f"light_{k:05d}"] = {"status": "registered", "H_top_down_magnified": [[1, 0, dx], [0, 1, dy], [0, 0, 1]]}
        calibration = {"parameters": model.as_dict(), "registration": {"frames": frames}}
        coverage = m.common_sky_coverage(calibration, width, height, 4540.0)
        self.assertTrue(coverage["rectangular"])
        boxes = m.boxes_by_rule(coverage, margin=200)
        for name, (x0, x1, y0, y1) in boxes.items():
            self.assertGreaterEqual(x0, coverage["x0"] + 200 - 1, name)
            self.assertLessEqual(x1, coverage["x1"] - 200 + 1, name)
            self.assertGreaterEqual(y0, coverage["y0"] + 200 - 1, name)
            self.assertLessEqual(y1, coverage["y1"] - 200 + 1, name)
            self.assertEqual((x1 - x0, y1 - y0), m.BOX_SIZES[name], name)
            # every box corner is inside every frame's sky, mapped through the job's own model and matrices
            for f in frames.values():
                h_inv = np.linalg.inv(np.asarray(f["H_top_down_magnified"], dtype=np.float64))
                corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)
                p = model.inverse(m._apply(h_inv, corners))
                self.assertTrue((p[:, 0] >= 0).all() and (p[:, 0] <= width - 1).all(), name)
                self.assertTrue((p[:, 1] >= 0).all() and (p[:, 1] <= 4540.0).all(), name)
        # the rule is data-driven: a frame moved far enough shrinks the region below the box sizes and the rule refuses
        frames["light_00005"] = {"status": "registered", "H_top_down_magnified": [[1, 0, -6000.0], [0, 1, 0], [0, 0, 1]]}
        with self.assertRaises(ValueError):
            m.boxes_by_rule(m.common_sky_coverage(calibration, width, height, 4540.0), margin=200)


class JudgementTest(unittest.TestCase):
    def test_master_judgement_is_position_matched_ratio_only(self):
        m = _module()
        good = {"paired": 40, "old_median_sigma_major_px": 2.0, "new_median_sigma_major_px": 2.06, "median_ratio_sigma_minor": 1.02,
                "median_ratio_area": 1.05, "median_ratio_flux": 0.8}
        results = {"topright": {"paired_comparison": dict(good)}, "centre": {"paired_comparison": dict(good)}}
        self.assertTrue(m.judge_master(results, m.CRITERIA["T4"])["passed"])            # flux and area are not judged
        results["centre"]["paired_comparison"]["new_median_sigma_major_px"] = 2.2
        self.assertFalse(m.judge_master(results, m.CRITERIA["T4"])["passed"])
        results["centre"]["paired_comparison"].update(new_median_sigma_major_px=2.0, median_ratio_sigma_minor=1.06)
        self.assertFalse(m.judge_master(results, m.CRITERIA["T4"])["passed"])
        results["centre"]["paired_comparison"].update(median_ratio_sigma_minor=1.0, paired=4)
        self.assertFalse(m.judge_master(results, m.CRITERIA["T4"])["passed"])

    def test_calibration_ratio_threshold_is_half(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            frames = {"light_00002": {"span": 1, "status": "associated", "matched_fraction": 0.9, "corner_rms_px": 0.2}}
            record = {"status": "fitted", "validation_corner_rms_median_px": {"homography_only": 1.0, "with_distortion": 0.6},
                      "frames_dropped_by_siril": [], "reassociation": {"frames": frames, "recovered_dropped_frames": []}}
            (work / rc.DISTORTION_JSON).write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(m.calibration_checks(work, m.CRITERIA["T2"])["passed"])
            record["validation_corner_rms_median_px"]["with_distortion"] = 0.4
            (work / rc.DISTORTION_JSON).write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(m.calibration_checks(work, m.CRITERIA["T2"])["passed"])


if __name__ == "__main__":
    unittest.main()
