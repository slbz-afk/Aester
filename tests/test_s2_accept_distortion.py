"""Tests for scripts/s2_accept_distortion.py on synthetic products (no Siril, no photographs).

Run:  python -B -m unittest tests.test_s2_accept_distortion   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import importlib.util
import json
import shutil
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
    spec = importlib.util.spec_from_file_location("s2_accept_distortion", ROOT / "scripts" / "s2_accept_distortion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_native(truth: dict, model: RadialDistortion) -> np.ndarray:
    """Float RGB (H, W, 3) BOTTOM-UP image with the reference stars at their *native* positions."""
    layout = {"width": truth["width"], "height": truth["height"],
              "stars": [{**s, "amplitude": 20000.0, "sigma": 1.6} for s in truth["frames"][0]["rendered_stars"]]}
    image, _ = sf.render_frame(layout, np.eye(3), seed=31)
    return np.repeat((image.astype(np.float32) / 65535.0)[::-1][..., None], 3, axis=2)


class EndToEndTest(unittest.TestCase):
    LAYOUT = "combined"     # the pre-P1b single-process 02_stack

    def test_main_evaluates_a_distortion_flow_job(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            new_job = root / "new"
            work = new_job / "work"
            truth = sp.generate_dataset(work)
            sp.simulate_siril_outputs(truth, work, applied=True)          # r_ frames at undistorted positions, seq in undistorted space
            model = RadialDistortion(**truth["distortion"])                # magnification 1: matches the simulated products
            width, height = truth["width"], truth["height"]
            # undistorted stack = the simulated reference r_ frame; native master = warp-back of it
            _, ref_plane = sf.read_fits_minimal(work / "r_light_00001.fit")
            stack = np.repeat((ref_plane.astype(np.float32) / 65535.0)[::-1][..., None], 3, axis=2)
            twopass._write_float_master(work / "sky_master_undistorted.fit", ref_plane, 4)
            native = rc.native_from_undistorted(stack, model.as_dict())
            twopass._write_float_master(work / "sky_master.fit", np.clip(native[::-1, :, 1] * 65535.0, 0, 65535).astype(np.uint16), 4)
            (work / rc.DISTORTION_WCS).write_bytes(b"synthetic")
            frames = {f"light_0000{k}": {"span": k - 1, "status": "associated", "initial_homography": "joint_fit", "matched": 165, "frame_count": 170,
                                          "matched_fraction": 0.97, "corner_rms_px": 0.06, "overall_rms_px": 0.05} for k in range(2, 5)}
            calibration = {"status": "fitted", "file": rc.DISTORTION_WCS, "parameters": model.as_dict(), "corner_shift_px": 11.9, "magnification": 1.0,
                           "validation_corner_rms_median_px": {"homography_only": 0.12, "with_distortion": 0.06}, "frames_dropped_by_siril": [],
                           "fit_frames": sorted(frames), "seconds": 0.5,
                           "reassociation": {"frames": frames, "matched_fraction_median": 0.97, "corner_rms_median_px": 0.06, "recovered_dropped_frames": []}}
            (work / rc.DISTORTION_JSON).write_text(json.dumps(calibration), encoding="utf-8")
            inputs = []
            for k in range(1, 5):
                path = root / f"DSC_{2203 + k}.dng"
                path.write_bytes(b"raw" * k)
                inputs.append({"path": str(path), "size": path.stat().st_size, "sha256": f"sha{k}"})
            (new_job / "02_probe.ssf").write_text("requires 1.4.4 1.5.0\n" + "\n".join(core.PROBE_SCRIPT) + "\n", encoding="utf-8")
            twopass.write_stack_phase(new_job, core.stack_script(core.DISTORTION_WCS),
                                      "log: choosing image 1 as new reference for registration\n",
                                      "log: Total execution time: 20.0 s\n", self.LAYOUT)
            (new_job / "02_probe.log").write_text("log: Total execution time: 8.0 s\n", encoding="utf-8")
            (new_job / "01_convert.log").write_text("log: Total execution time: 5.0 s\n", encoding="utf-8")
            base = {"schema_version": 1, "state": "succeeded", "stage": "complete", "progress": 100, "message": "", "error": None,
                    "created_at": "2026-09-17T00:00:00+00:00", "updated_at": "2026-09-17T00:00:00+00:00", "inputs": inputs,
                    "checkpoints": ["convert", "distortion", "stack"], "checkpoint_files": {}, "artifacts": [], "limitations": [],
                    "engine": {"executable": "fake", "version": "1.4.4"}}
            diagnostics = {"input_count": 4, "stacked_count": 4, "skipped_count": 0, "reference_selection": core.REFERENCE_SELECTION,
                           "reference_input": inputs[0]["path"], "reference_stem": "light_00001", "reference_position": 0,
                           "registration": core.registration_commands(core.DISTORTION_WCS), "sky_geometry": "native_via_undistorted_stack",
                           "distortion": {**{k: calibration[k] for k in ("status", "parameters", "corner_shift_px", "magnification")}, "applied": True}}
            (new_job / "job.json").write_text(json.dumps({**base, "job_dir": str(new_job), "diagnostics": diagnostics}), encoding="utf-8")
            # the "old" one-pass job: same sequence file, master = native reference smeared by 2 px
            old_job = root / "old"
            (old_job / "work").mkdir(parents=True)
            shutil.copyfile(work / "light_.seq", old_job / "work" / "light_.seq")
            shutil.copytree(work / "cache", old_job / "work" / "cache")
            native_plane = native[::-1, :, 1] * 65535.0
            smeared = 0.5 * (native_plane + np.roll(native_plane, 2, axis=1))
            twopass._write_float_master(old_job / "work" / "sky_master.fit", np.clip(smeared, 0, 65535).astype(np.uint16), 4)
            (old_job / "job.json").write_text(json.dumps({**base, "job_dir": str(old_job)}), encoding="utf-8")
            out = root / "out"
            m.BOXES.clear()
            m.BOXES.update(BOXES)
            code = m.main(["--reference-job", str(old_job), "--job-dir", str(new_job), "--output-root", str(out), "--frames", "2", "3", "4"])
            run = next(out.glob("accept-distortion-*"))
            report = json.loads((run / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0, report["verdict"])
            self.assertTrue(report["T1"]["passed"], report["T1"]["checks"])
            self.assertIs(report["T1"]["checks"]["stack_script_matches_product"], True)
            self.assertTrue(report["T2"]["passed"], report["T2"])
            self.assertTrue(report["T3"]["judgement"]["passed"], report["T3"]["judgement"])
            self.assertTrue(report["T3b"]["judgement"]["passed"], report["T3b"]["judgement"])
            self.assertTrue(report["T4"]["judgement"]["passed"], report["T4"]["judgement"])
            for key, entry in report["T3"]["results"].items():
                self.assertLess(entry["offsets"]["rms_px"], 0.15, key)
            for name, entry in report["T3b"]["results"].items():
                self.assertLess(entry["offsets"]["median_px"], 0.1, name)
            text = (run / "ACCEPTANCE.md").read_text(encoding="utf-8")
            self.assertIn("通过", text)
            self.assertIn("T3b", text)


class SplitLayoutEndToEndTest(EndToEndTest):
    LAYOUT = "split"        # efficiency P1b-3: 02_export (registration) + 02_stack (seqclean + stack)


class JudgementTest(unittest.TestCase):
    def test_master_judgement_uses_minor_axis_area_and_flux(self):
        m = _module()
        good = {"paired": 40, "old_median_sigma_major_px": 2.0, "new_median_sigma_major_px": 2.0, "median_ratio_sigma_minor": 0.98,
                "median_ratio_area": 0.97, "median_ratio_flux": 1.05}
        results = {"topright": {"paired_comparison": dict(good)}, "centre": {"paired_comparison": {**good, "median_ratio_sigma_minor": 1.0, "median_ratio_area": 1.0}}}
        self.assertTrue(m.judge_master(results, m.CRITERIA["T4"])["passed"])
        results["centre"]["paired_comparison"]["median_ratio_flux"] = 0.9
        self.assertFalse(m.judge_master(results, m.CRITERIA["T4"])["passed"])
        results["centre"]["paired_comparison"]["median_ratio_flux"] = 1.0
        results["topright"]["paired_comparison"].update(median_ratio_sigma_minor=1.01, median_ratio_area=1.01)
        self.assertFalse(m.judge_master(results, m.CRITERIA["T4"])["passed"])   # top-right must improve on the minor axis or area

    def test_native_judgement(self):
        m = _module()
        ok = {"a": {"offsets": {"median_px": 0.1, "rms_px": 0.3}}}
        self.assertTrue(m.judge_native(ok, m.CRITERIA["T3b"])["passed"])
        bad = {"a": {"offsets": {"median_px": 0.3, "rms_px": 0.3}}}
        self.assertFalse(m.judge_native(bad, m.CRITERIA["T3b"])["passed"])


if __name__ == "__main__":
    unittest.main()
