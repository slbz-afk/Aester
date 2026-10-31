"""Tests for scripts/s2_accept_two_pass.py on synthetic data (no photographs, no Siril).

Run:  python -B -m unittest tests.test_s2_accept_two_pass   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import registration_adapter as adapter  # noqa: E402
from starstack import siril_coordinate_proof as cp  # noqa: E402
from starstack import siril_sequence  # noqa: E402
from starstack import synthetic_starfield as sf  # noqa: E402

BOXES = {"topright": (800, 1200, 0, 300), "centre": (400, 800, 300, 600), "topleft": (0, 400, 0, 300), "rightlower": (800, 1200, 600, 901)}


def _module():
    spec = importlib.util.spec_from_file_location("s2_accept_two_pass", ROOT / "scripts" / "s2_accept_two_pass.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_float_master(path: Path, image_top_down: np.ndarray, stackcnt: int) -> None:
    """Minimal 32-bit float RGB FITS (BOTTOM-UP) like Siril's ``-32b`` master, values 0..1."""
    height, width = image_top_down.shape
    cards = [sf._card("SIMPLE", True), sf._card("BITPIX", -32), sf._card("NAXIS", 3), sf._card("NAXIS1", width),
             sf._card("NAXIS2", height), sf._card("NAXIS3", 3), sf._card("ROWORDER", "BOTTOM-UP"), sf._card("STACKCNT", stackcnt), "END".ljust(80)]
    header = "".join(cards)
    header += " " * ((2880 - len(header) % 2880) % 2880)
    plane = (image_top_down[::-1, :].astype(np.float64) / 65535.0).astype(">f4")
    data = np.repeat(plane[None, :, :], 3, axis=0).tobytes()
    data += b"\0" * ((2880 - len(data) % 2880) % 2880)
    path.write_bytes(header.encode("ascii") + data)


def write_stack_phase(job: Path, combined: list, choice_line: str, seconds_line: str, layout: str = "combined") -> None:
    """The stack-phase Siril files of a synthetic L job.  "combined": the pre-P1b single 02_stack (``combined`` =
    export commands + the stack command).  "split" (efficiency P1b-3): 02_export with the registration commands
    and Siril's reference choice, then 02_stack with seqclean + the stack."""
    from starstack import core
    header = "requires 1.4.4 1.5.0\n"
    if layout == "combined":
        (job / "02_stack.ssf").write_text(header + "\n".join(combined) + "\n", encoding="utf-8")
        (job / "02_stack.log").write_text(choice_line + seconds_line, encoding="utf-8")
        return
    assert layout == "split" and combined[-1] == core.STACK_COMMAND
    (job / "02_export.ssf").write_text(header + "\n".join(combined[:-1]) + "\n", encoding="utf-8")
    (job / "02_export.log").write_text(choice_line + "log: Total execution time: 4.0 s\n", encoding="utf-8")
    (job / "02_stack.ssf").write_text(header + "\n".join(core.LEGACY_STACK_SCRIPT) + "\n", encoding="utf-8")
    (job / "02_stack.log").write_text(seconds_line, encoding="utf-8")


def _synthetic_products(root: Path) -> dict:
    """A 'new' work dir (frames, light_.seq, cache, float master = reference frame) and an 'old' job whose
    master is the reference frame smeared by a 1.5 px horizontal shift (registration smear)."""
    new_work = root / "new" / "work"
    truth = sf.generate_dataset(new_work, stem="light")
    cp.simulate_siril_outputs(truth, new_work, row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5, matrix_direction="frame_to_reference")
    _, reference = sf.read_fits_minimal(new_work / "light_00001.fit")
    ref = reference.astype(np.float64)
    _write_float_master(new_work / "sky_master.fit", np.clip(ref, 0, 65535).astype(np.uint16), 4)
    old_work = root / "old" / "work"
    old_work.mkdir(parents=True)
    shutil.copyfile(new_work / "light_.seq", old_work / "light_.seq")
    shutil.copytree(new_work / "cache", old_work / "cache")
    smeared = 0.5 * (ref + np.roll(ref, 2, axis=1))  # two-frame stack with a 2 px misregistration
    _write_float_master(old_work / "sky_master.fit", np.clip(smeared, 0, 65535).astype(np.uint16), 4)
    return {"truth": truth, "new_work": new_work, "old_job": root / "old"}


class Parsers(unittest.TestCase):
    def test_reference_choice_english_and_chinese(self):
        m = _module()
        self.assertEqual(m.parse_reference_choice("log: Trial #1: After sequence analysis, we are choosing image 11 as new reference for registration"), 11)
        self.assertEqual(m.parse_reference_choice("log: 试验#1：序列分析后，我们选择图像13作为新参考用于对齐\n"), 13)
        self.assertEqual(m.parse_reference_choice("log: 试验#1：...选择图像11作为新参考\nlog: 试验#2：...选择图像12作为新参考"), 12)
        self.assertIsNone(m.parse_reference_choice("log: Reference Image:\nlog: Found 2000 stars in reference"))

    def test_execution_time(self):
        m = _module()
        self.assertAlmostEqual(m.parse_execution_seconds("log: 总执行时间: 1.84 ms\n"), 0.00184)
        self.assertAlmostEqual(m.parse_execution_seconds("log: Total execution time: 2.50 min\n"), 150.0)
        self.assertIsNone(m.parse_execution_seconds("nothing"))


class Planning(unittest.TestCase):
    def test_check_frames_cover_ends_and_farthest_without_reference(self):
        m = _module()
        images = [{"index": i + 1, "position": i, "stem": f"light_{i + 1:05d}"} for i in range(16)]
        seq = {"header": {"reference_image": 10}, "images": images}
        self.assertEqual(m.plan_check_frames(seq, [2, 16]), [1, 16, 2])
        seq0 = {"header": {"reference_image": 0}, "images": images}
        self.assertEqual(m.plan_check_frames(seq0, [1, 99]), [16])  # 1 is the reference, 99 does not exist

    def test_link_inputs_keeps_order_and_bytes(self):
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "dng"
            src.mkdir()
            inputs = []
            for name in ("DSC_2205.dng", "DSC_2204.dng"):
                (src / name).write_bytes(name.encode() * 10)
                inputs.append({"path": str(src / name), "sha256": "x"})
            info = m.link_inputs({"inputs": inputs}, Path(d) / "inputs")
            self.assertEqual(info["count"], 2)
            self.assertIn(info["method"], ("hardlink", "copy"))
            listed = sorted(p.name for p in (Path(d) / "inputs").iterdir())
            self.assertEqual(listed, ["001_DSC_2205.dng", "002_DSC_2204.dng"])
            self.assertEqual((Path(d) / "inputs" / "001_DSC_2205.dng").read_bytes(), b"DSC_2205.dng" * 10)
            self.assertEqual((src / "DSC_2205.dng").read_bytes(), b"DSC_2205.dng" * 10)


class Judgements(unittest.TestCase):
    def test_adapter_judgement_tolerates_one_documented_frame(self):
        m = _module()
        c = m.CRITERIA["T2"]
        good = {"state": "succeeded", "matched": 1500, "corners_rms_px": 0.3}
        frames = {f"f{i}": dict(good) for i in range(15)}
        self.assertEqual(m.judge_adapter(frames, c), {"passed": True, "failing": [], "tolerated": []})
        frames["f1"] = {"state": "succeeded", "matched": 435, "corners_rms_px": 0.97}
        self.assertEqual(m.judge_adapter(frames, c)["tolerated"], ["f1"])
        self.assertTrue(m.judge_adapter(frames, c)["passed"])
        frames["f2"] = {"state": "succeeded", "matched": 1200, "corners_rms_px": 0.8}
        self.assertFalse(m.judge_adapter(frames, c)["passed"])  # two tolerated frames is a fail
        frames["f2"] = {"state": "succeeded", "matched": 1200, "corners_rms_px": 1.3}
        self.assertEqual(m.judge_adapter(frames, c)["failing"], ["f2"])
        frames["f3"] = {"state": "failed", "matched": None, "corners_rms_px": None}
        self.assertIn("f3", m.judge_adapter(frames, c)["failing"])

    def test_crop_and_master_judgements(self):
        m = _module()
        crops = {"light_00016|topright": {"offsets": {"rms_px": 0.22}}, "light_00016|centre": {"offsets": {"rms_px": 0.12}},
                 "light_00001|topright": {"offsets": {"rms_px": 0.8}}, "light_00001|centre": {"offsets": {"rms_px": 0.3}}}
        j = m.judge_crops(crops, m.CRITERIA["T3"])
        self.assertTrue(j["passed"])
        self.assertEqual(j["tolerated"], ["light_00001"])
        crops["light_00002|topright"] = {"offsets": {"rms_px": 2.3}}
        self.assertEqual(m.judge_crops(crops, m.CRITERIA["T3"])["failing"], ["light_00002"])
        masters = {"topright": {"paired_comparison": {"paired": 30, "old_median_sigma_major_px": 2.2, "new_median_sigma_major_px": 2.0}},
                   "centre": {"paired_comparison": {"paired": 200, "old_median_sigma_major_px": 1.5, "new_median_sigma_major_px": 1.52}}}
        self.assertTrue(m.judge_master(masters, m.CRITERIA["T4"])["passed"])
        masters["centre"]["paired_comparison"]["new_median_sigma_major_px"] = 1.6
        self.assertFalse(m.judge_master(masters, m.CRITERIA["T4"])["passed"])
        masters["centre"]["paired_comparison"]["new_median_sigma_major_px"] = 1.5
        masters["topright"]["paired_comparison"]["new_median_sigma_major_px"] = 2.2
        self.assertFalse(m.judge_master(masters, m.CRITERIA["T4"])["passed"])
        self.assertFalse(m.judge_master({"topright": {"paired_comparison": {"paired": 2}}}, m.CRITERIA["T4"])["passed"])

    def test_paired_shapes_uses_common_indices_only(self):
        m = _module()
        def star(index, major, minor, elong, flux=1000.0):
            return {"index": index, "sigma_major_px": major, "sigma_minor_px": minor, "elongation": elong, "flux": flux}
        old = {"stars": [star(1, 2.0, 1.5, 1.4), star(2, 2.2, 1.6, 1.5), star(9, 9.0, 9.0, 9.0)]}
        new = {"stars": [star(1, 1.0, 1.2, 1.1, 1100.0), star(2, 2.2, 1.6, 1.5, 1000.0)]}
        p = m.paired_shapes(old, new)
        self.assertEqual(p["paired"], 2)
        self.assertAlmostEqual(p["median_ratio_new_over_old"], 0.75)
        self.assertAlmostEqual(p["fraction_new_sharper"], 0.5)
        self.assertAlmostEqual(p["median_ratio_sigma_minor"], 0.9)
        self.assertAlmostEqual(p["median_ratio_flux"], 1.05)
        self.assertAlmostEqual(p["median_ratio_area"], (0.4 + 1.0) / 2)
        self.assertEqual(m.paired_shapes({"stars": []}, new), {"paired": 0})


class SyntheticEndToEnd(unittest.TestCase):
    def test_pixel_crops_and_master_comparison_on_simulated_products(self):
        m = _module()
        hypothesis = adapter.CoordinateHypothesis(name="sim", row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5,
                                                  matrix_direction="frame_to_reference", verified=True, basis="simulation")
        with tempfile.TemporaryDirectory() as d:
            products = _synthetic_products(Path(d))
            new_work, old_job = products["new_work"], products["old_job"]
            sequence = siril_sequence.read_seq(new_work / "light_.seq")
            run = Path(d) / "run"
            run.mkdir()
            frames = m.plan_check_frames(sequence, [2])
            self.assertEqual(sorted(frames), [2, 4])
            crops = m.pixel_crop_checks(new_work, sequence, hypothesis, frames, BOXES, 1, run, log=lambda *_: None)
            self.assertEqual(crops["reference"], "light_00001")
            for key, entry in crops["results"].items():
                self.assertGreater(entry["stars_in_box"], 3, key)
                self.assertIsNotNone(entry["offsets"]["rms_px"], key)
                self.assertLess(entry["offsets"]["rms_px"], 0.15, key)   # exact simulated H: sub-pixel agreement
                self.assertTrue((run / "crops" / f"{entry['frame']}_{key.split('|')[1]}.json").exists())
            self.assertTrue(m.judge_crops(crops["results"], m.CRITERIA["T3"])["passed"])
            self.assertTrue(any(p.name.startswith("montage_") for p in (run / "crops").iterdir()))
            old_seq = siril_sequence.read_seq(old_job / "work" / "light_.seq")
            masters = m.master_comparison(old_job, new_work, old_seq, sequence, hypothesis, BOXES, 1, run, log=lambda *_: None)
            for name, entry in masters["results"].items():
                p = entry["paired_comparison"]
                self.assertGreaterEqual(p["paired"], 5, name)
                self.assertLess(p["median_ratio_new_over_old"], 0.9, name)   # the smeared old master is wider everywhere
                self.assertTrue((run / "master" / f"{name}_old.png").exists())
                self.assertTrue((run / "master" / f"{name}_new.png").exists())
            self.assertTrue(m.judge_master(masters["results"], m.CRITERIA["T4"])["passed"])
            # the same comparison with the masters swapped (new = smeared) must fail T4
            run2 = Path(d) / "run2"
            run2.mkdir()
            swapped = m.master_comparison(Path(d) / "new", old_job / "work", sequence, old_seq, hypothesis, BOXES, 1, run2, log=lambda *_: None)
            self.assertFalse(m.judge_master(swapped["results"], m.CRITERIA["T4"])["passed"])

    def test_main_evaluates_an_existing_job_end_to_end(self):
        self._evaluate_existing_job("combined")

    def test_main_evaluates_a_split_layout_job_end_to_end(self):
        report = self._evaluate_existing_job("split")             # efficiency P1b-3: 02_export + 02_stack
        self.assertAlmostEqual(report["T1"]["siril_seconds"]["export"], 4.0)

    def _evaluate_existing_job(self, layout):
        """Whole evaluation path (T1-T4, JSON, Markdown) on synthetic jobs; Siril is never run."""
        m = _module()
        with tempfile.TemporaryDirectory() as d:
            products = _synthetic_products(Path(d))
            new_job, old_job = Path(d) / "new", products["old_job"]
            new_work = products["new_work"]
            for k in range(1, 5):
                shutil.copyfile(new_work / f"light_0000{k}.fit", new_work / f"r_light_0000{k}.fit")
            inputs_dir = Path(d) / "inputs"
            inputs_dir.mkdir()
            inputs = []
            for k in range(1, 5):
                path = inputs_dir / f"DSC_{2203 + k}.dng"
                path.write_bytes(b"raw" * k)
                inputs.append({"path": str(path), "size": path.stat().st_size, "sha256": f"sha{k}"})
            write_stack_phase(new_job, m.core.STACK_SCRIPT,
                              "log: Trial #1: After sequence analysis, we are choosing image 1 as new reference for registration\n",
                              "log: Total execution time: 1.50 min\n", layout)
            (new_job / "01_convert.log").write_text("log: 总执行时间: 30.0 s\n", encoding="utf-8")
            base = {"schema_version": 1, "state": "succeeded", "stage": "complete", "progress": 100, "message": "", "error": None,
                    "created_at": "2026-09-17T00:00:00+00:00", "updated_at": "2026-09-17T00:00:00+00:00", "inputs": inputs,
                    "checkpoints": [], "checkpoint_files": {}, "artifacts": [], "limitations": ["x (见 diagnostics.reference_input)"],
                    "engine": {"executable": "fake", "version": "1.4.4"}}
            new_meta = {**base, "job_dir": str(new_job), "diagnostics": {
                "input_count": 4, "stacked_count": 4, "skipped_count": 0, "reference_selection": m.core.REFERENCE_SELECTION,
                "reference_input": inputs[0]["path"], "reference_stem": "light_00001", "reference_position": 0,
                "registration": list(m.core.REGISTRATION_COMMANDS)}}
            (new_job / "job.json").write_text(json.dumps(new_meta), encoding="utf-8")
            (old_job / "job.json").write_text(json.dumps({**base, "job_dir": str(old_job)}), encoding="utf-8")
            out = Path(d) / "out"
            # the synthetic geometry is 1200x901: use boxes that fit it
            m.BOXES.clear()
            m.BOXES.update(BOXES)
            m.CRITERIA["T2"]["min_pairs"] = 100
            m.CRITERIA["T2"]["tolerated_min_pairs"] = 50
            code = m.main(["--reference-job", str(old_job), "--job-dir", str(new_job), "--output-root", str(out), "--frames", "2"])
            self.assertEqual(code, 0)
            run = next(out.glob("accept-twopass-*"))
            report = json.loads((run / "acceptance.json").read_text(encoding="utf-8"))
            self.assertTrue(report["verdict"]["passed"], report["verdict"])
            self.assertTrue(report["T1"]["passed"], report["T1"]["checks"])
            self.assertIs(report["T1"]["checks"]["stack_script_matches_product"], True)
            self.assertEqual(report["T1"]["reference"]["log_choice"], 1)
            self.assertEqual(report["T1"]["reference"]["original_input"], inputs[0]["path"])
            self.assertAlmostEqual(report["T1"]["siril_seconds"]["stack"], 90.0)
            self.assertEqual(sorted(report["T2"]["new"]["frames"]), ["light_00002", "light_00003", "light_00004"])
            self.assertTrue((run / "diagnostics-new" / "SUMMARY.md").exists())
            self.assertTrue((run / "diagnostics-baseline" / "SUMMARY.md").exists())
            self.assertIn("通过", (run / "ACCEPTANCE.md").read_text(encoding="utf-8"))
            return report

    def test_stack_scripts_match_either_layout_and_nothing_else(self):
        m = _module()
        core = m.core
        with tempfile.TemporaryDirectory() as d:
            job = Path(d)
            self.assertIsNone(m.stack_scripts_match(job, core.STACK_SCRIPT))
            for layout in ("combined", "split"):
                for name in ("02_export.ssf", "02_stack.ssf", "02_export.log", "02_stack.log"):
                    (job / name).unlink(missing_ok=True)
                write_stack_phase(job, core.STACK_SCRIPT, "log: choosing image 2 as new reference\n", "", layout)
                self.assertIs(m.stack_scripts_match(job, core.STACK_SCRIPT), True, layout)
                self.assertIs(m.stack_scripts_match(job, core.OWN_REGISTRATION_STACK_SCRIPT), False, layout)
                self.assertEqual(m.parse_reference_choice(m.registration_log(job)), 2, layout)
            # split layout with a stack script that still exports (the pre-P1b text under the new name) fails
            (job / "02_stack.ssf").write_text("requires 1.4.4 1.5.0\n" + "\n".join(core.STACK_SCRIPT) + "\n", encoding="utf-8")
            self.assertIs(m.stack_scripts_match(job, core.STACK_SCRIPT), False)
            # EFF-D4: a contract job's stack carries its settings lines before seqclean; other extra lines still fail
            contract = m.stack_settings.with_settings(core.LEGACY_STACK_SCRIPT, {"threads": 10, "memory_gib": 9})
            (job / "02_stack.ssf").write_text("requires 1.4.4 1.5.0\n" + "\n".join(contract) + "\n", encoding="utf-8")
            self.assertIs(m.stack_scripts_match(job, core.STACK_SCRIPT), True)
            (job / "02_stack.ssf").write_text("requires 1.4.4 1.5.0\nsetcpu 4\n" + "\n".join(core.LEGACY_STACK_SCRIPT)
                                              + "\n", encoding="utf-8")
            self.assertIs(m.stack_scripts_match(job, core.STACK_SCRIPT), False)

    def test_render_and_json_round_trip_on_a_minimal_report(self):
        m = _module()
        empty_judgement = {"passed": True, "failing": [], "tolerated": []}
        report = {"run": "accept-test", "reference_job": {"dir": "old"}, "job": {"dir": "new", "wall_seconds": 12.0},
                  "verdict": {"passed": True, "failures": [], "tolerated": {}},
                  "environment": {"timestamp": "now"},
                  "T1": {"passed": True, "checks": {"state_succeeded": True, "reference_matches_siril_log": None},
                         "reference": {"stem": "light_00011", "position": 10, "index": 11, "log_choice": 11, "input": "x.dng"},
                         "master": {"BITPIX": "-32"}, "r_frame": {"BITPIX": "-32"}, "r_frame_count": 16,
                         "siril_seconds": {"convert": 30.0, "stack": 120.0}},
                  "T2": {"new": {"reference": "light_00011", "frames": {"light_00001": {"matched": 435, "frame_stars": 2000, "center_rms_px": 0.3,
                                                                                         "corners_rms_px": 0.97, "corners_max_px": 2.0, "overall_rms_px": 0.5}}},
                         "baseline": {"reference": "light_00001", "frames": {}}, "judgement": empty_judgement},
                  "T3": {"reference": "light_00011", "results": {"light_00016|topright": {"frame": "light_00016", "offsets": {"count": 31, "rms_px": 0.22, "median_px": 0.2, "max_px": 0.5, "mean_dx_px": -0.1, "mean_dy_px": 0.05},
                                                                                            "shape_reference": {"median_sigma_major_px": 2.3}, "shape_stack2": {"median_sigma_major_px": 2.3}}},
                         "judgement": empty_judgement},
                  "T4": {"results": {"topright": {"paired_comparison": {"paired": 30, "old_median_sigma_major_px": 2.2, "new_median_sigma_major_px": 2.0,
                                                                        "median_ratio_new_over_old": 0.91, "ratio_p25": 0.85, "ratio_p75": 0.97, "fraction_new_sharper": 0.8,
                                                                        "old_median_elongation": 1.4, "new_median_elongation": 1.3}},
                                     "empty": {"paired_comparison": {"paired": 0}}},
                         "judgement": {"passed": True, "failing": [], "must_improve_box_improved": True}}}
        text = m.render(report)
        self.assertIn("通过", text)
        self.assertIn("light_00011", text)
        self.assertIn("| empty | 0 |", text)
        json.dumps(report)


if __name__ == "__main__":
    unittest.main()
