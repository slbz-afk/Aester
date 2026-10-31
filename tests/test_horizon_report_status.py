"""Independent report integration fixtures; no Siril, photographs or old reports."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import s3_accept_horizon as script
from starstack import imaging, sky_validity
from starstack.compose_evidence import write_weight


class HorizonReportStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="horizon-status-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job_dir = self.root / "job"
        self.work = self.job_dir / "work"
        self.work.mkdir(parents=True)
        self.output = self.root / "new"
        self.output.mkdir()
        self.mask_path = self.job_dir / "sky_mask.png"

    def _fixture(self, *, weight_record=True):
        h, w = 520, 64
        mask_bytes = np.full((h, w), 255, dtype=np.uint8)
        mask_bytes[455:466] = np.linspace(255, 0, 11, dtype=np.uint8)[:, None]
        mask_bytes[466:] = 0
        Image.fromarray(mask_bytes).save(self.mask_path)
        weight = mask_bytes.astype(np.float32) / 255
        sky = np.full((h, w, 3), 0.25, dtype=np.float32)
        ground = np.full_like(sky, 0.001)
        # Direct per-channel algebra; no production blend/effective-weight helper.
        composite = np.empty_like(sky)
        for channel in range(3):
            composite[..., channel] = weight * sky[..., channel] + (1 - weight) * ground[..., channel]
        for name, image in (("sky_master.fit", sky), ("foreground_master.fit", ground), ("light_00001.fit", sky), ("r_light_00001.fit", sky)):
            imaging.write_linear(self.work / name, image[::-1], "Synthetic report fixture")
        sky_validity.write_coverage_png(np.ones((130, 16), dtype=np.uint16), self.work / sky_validity.COVERAGE_PNG)
        (self.work / "sky_coverage.json").write_text(json.dumps({"step": 4}), encoding="utf-8")
        export = self.job_dir / "published"
        export.mkdir()
        composite_path = export / "composite.fit"
        imaging.write_linear(composite_path, composite[::-1], "Independent blend truth")
        validity = {"applied": True, "source": "job_mask_static", "dilation_px": script.core.SKY_VALIDITY_DILATION_PX, "frames_masked": 1}
        if weight_record:
            validity["compose_weight"] = write_weight(export, weight, composite_path)
        job = {"state": "succeeded", "mask": str(self.mask_path), "inputs": ["synthetic.fit"],
               "checkpoints": ["foreground", "convert"], "diagnostics": {"reference_stem": "light_00001", "sky_validity": validity},
               "artifacts": [{"label": "合成线性母版", "path": str(composite_path)}]}
        self._write_job(job)
        return job

    def _write_job(self, job):
        (self.job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")

    def _run(self):
        return script.horizon_checks(self.job_dir, self.mask_path, compare_job=None, run=self.output, log=lambda _: None)

    @staticmethod
    def _long_report(job_dir):
        out = {"job": {"dir": str(job_dir)}, "verdict": {"passed": True, "failures": []}}
        for name in ("T1", "T2", "T3", "T3b", "T4"):
            out[name] = {"passed": True} if name in ("T1", "T2") else {"judgement": {"passed": True}}
        return out

    def test_real_fits_full_report_passes_with_dark_ground(self):
        self._fixture()
        result = self._run()
        self.assertEqual(result["verdict"]["status"], "passed")
        self.assertEqual(result["H5"]["judgement"]["status"], "passed")
        self.assertIn("metrics", result["H5"])
        self.assertEqual(result["H5"]["metrics"]["sample_step"], 1)
        self.assertEqual(result["job_json_sha256"], hashlib.sha256((self.job_dir / "job.json").read_bytes()).hexdigest())
        rendered = script.render(result, None)
        self.assertIn("同位置", rendered)
        self.assertIn("天然暗地景", rendered)
        self.assertNotIn("no dark rim", rendered)
        json.dumps(result, allow_nan=False)

    def test_old_job_without_weight_is_incomplete_not_failed(self):
        self._fixture(weight_record=False)
        result = self._run()
        self.assertEqual(result["H5"]["judgement"]["status"], "insufficient_data")
        self.assertEqual(result["verdict"]["status"], "insufficient_data")
        self.assertFalse(result["verdict"]["passed"])
        self.assertEqual(result["verdict"]["failures"], [])
        self.assertTrue(result["verdict"]["incomplete"])

    def test_missing_sky_does_not_prevent_weight_missing_result(self):
        self._fixture(weight_record=False)
        (self.work / "sky_master.fit").unlink()
        result = self._run()
        self.assertEqual(result["H5"]["judgement"]["status"], "insufficient_data")
        self.assertEqual(result["H2"]["judgement"]["status"], "insufficient_data")
        self.assertIn("样本不足", script.render(result, None))

    def test_h3_diagnostic_missing_does_not_fail_other_measured_gates(self):
        self._fixture()
        (self.work / "light_00001.fit").unlink()
        result = self._run()
        self.assertEqual(result["H3"]["judgement"]["status"], "insufficient_data")
        self.assertEqual(result["verdict"]["status"], "passed")

    def test_no_contributor_frames_is_incomplete(self):
        self._fixture()
        (self.work / "r_light_00001.fit").unlink()
        result = self._run()
        self.assertEqual(result["H3c"]["judgement"]["status"], "insufficient_data")
        self.assertEqual(result["verdict"]["status"], "insufficient_data")

    def test_known_failure_takes_priority_over_missing_weight(self):
        job = self._fixture(weight_record=False)
        job["diagnostics"]["sky_validity"]["applied"] = False
        self._write_job(job)
        result = self._run()
        self.assertEqual(result["verdict"]["status"], "failed")
        self.assertTrue(result["verdict"]["failures"])
        self.assertTrue(result["verdict"]["incomplete"])

    def test_overview_retains_same_stage_failure_and_incomplete_reasons(self):
        stage = {"judgement": script.summarize_checks([
            {"name": "wrong", "status": "failed", "samples": 1, "reason": "wrong pixel"},
            {"name": "missing", "status": "insufficient_data", "samples": 0, "reason": "missing band"},
        ])}
        check = script._gate_check("H5", stage)
        self.assertIn("wrong pixel", check["reason"])
        self.assertIn("missing band", check["reason"])

    def test_malformed_comparison_is_diagnostic_only(self):
        self._fixture()
        other = self.root / "comparison"
        other.mkdir()
        for metadata in ([], {"diagnostics": "broken"}):
            with self.subTest(metadata=metadata):
                (other / "job.json").write_text(json.dumps(metadata), encoding="utf-8")
                result = script.horizon_checks(self.job_dir, self.mask_path, compare_job=other, run=self.output, log=lambda _: None)
                self.assertEqual(result["H3_before"]["judgement"]["status"], "failed")
                self.assertEqual(result["verdict"]["status"], "passed")

    def test_job_without_mask_is_not_applicable(self):
        self._write_job({"state": "succeeded", "inputs": []})
        result = self._run()
        self.assertEqual(result["verdict"]["status"], "not_applicable")
        self.assertFalse(result["verdict"]["passed"])

    def test_missing_job_is_explicit(self):
        result = self._run()
        self.assertEqual(result["verdict"]["status"], "insufficient_data")
        self.assertIn("FileNotFoundError", script.render(result, None))

    def test_invalid_job_json_is_failed(self):
        (self.job_dir / "job.json").write_text("{", encoding="utf-8")
        self.assertEqual(self._run()["verdict"]["status"], "failed")

    def test_malformed_nested_job_metadata_gets_a_report(self):
        for extras in ({"diagnostics": "broken"}, {"artifacts": ["broken"]}, {"diagnostics": {"sky_validity": "broken"}}):
            with self.subTest(extras=extras):
                self._write_job(dict(state="succeeded", mask="exists", **extras))
                result = self._run()
                self.assertEqual(result["verdict"]["status"], "failed")
                self.assertIn("ValueError", script.render(result, None))

    def test_nonfinite_job_metadata_gets_a_failed_report(self):
        self._write_job({"state": "succeeded", "mask": "exists", "unexpected": float("nan")})
        result = self._run()
        self.assertEqual(result["verdict"]["status"], "failed")
        json.dumps(result, allow_nan=False)

    def test_unknown_programming_error_is_not_swallowed(self):
        with self.assertRaises(RuntimeError):
            script._capture(lambda: (_ for _ in ()).throw(RuntimeError("bug")))

    def test_empty_contributor_results_not_passed(self):
        self.assertEqual(script._judge_contributors({}, script.ha.CRITERIA["H3c"])["status"], "insufficient_data")

    def test_contributor_failure_priority_over_missing_band(self):
        result = {"range": {"bands": [{"RG": {"ratio": 1.03}, "BG": {"ratio": 1.0}}, {}], "max_abs_deviation": .03, "bands_without_data": [(25, 50)]}}
        self.assertEqual(script._judge_contributors(result, script.ha.CRITERIA["H3c"])["status"], "failed")

    def test_contributor_missing_channel_is_incomplete(self):
        result = {"range": {"bands": [{"RG": {"ratio": 1.0}, "BG": {"ratio": None}}], "max_abs_deviation": 0, "bands_without_data": []}}
        self.assertEqual(script._judge_contributors(result, script.ha.CRITERIA["H3c"])["status"], "insufficient_data")

    def test_contributor_nonfinite_metric_is_failed(self):
        result = {"range": {"bands": [{"RG": {"ratio": float("nan")}, "BG": {"ratio": 1.0}}], "max_abs_deviation": float("nan"), "bands_without_data": []}}
        self.assertEqual(script._judge_contributors(result, script.ha.CRITERIA["H3c"])["status"], "failed")

    def test_long_span_empty_and_partial_never_pass(self):
        for report in (None, {}, {"verdict": {"passed": True}}, {"T1": {"passed": True}, "verdict": {"passed": True}}):
            with self.subTest(report=report):
                self.assertEqual(script._long_span_check(report)["status"], "insufficient_data")

    def test_long_span_keeps_existing_failure(self):
        report = self._long_report(self.job_dir)
        report["verdict"]["passed"] = False
        self.assertEqual(script._long_span_check(report)["status"], "failed")
        report = self._long_report(self.job_dir)
        report["T3"]["judgement"]["passed"] = False
        self.assertEqual(script._long_span_check(report)["status"], "failed")

    def test_long_span_complete_record_is_passed(self):
        self.assertEqual(script._long_span_check(self._long_report(self.job_dir))["status"], "passed")

    def test_run_creates_new_report_and_preserves_source_bytes(self):
        self._fixture(weight_record=False)
        source = self.root / "old-run"
        source.mkdir()
        (source / "acceptance.json").write_text(json.dumps(self._long_report(self.job_dir)), encoding="utf-8")
        for name in ("HORIZON.md", "horizon.json", "horizon_before_after.png"):
            (source / name).write_bytes(b"frozen historical report")
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        args = ["--run", str(source), "--sky-mask", str(self.mask_path), "--sky-limit-y", "400", "--output-root", str(self.output)]
        with patch.object(script.long_span, "main", side_effect=AssertionError("old run must not rerun engine")):
            self.assertEqual(script.main(args), 3)
        self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, before)
        reports = list(self.output.glob("horizon-evaluation-*/horizon.json"))
        self.assertEqual(len(reports), 1)
        result = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"]["status"], "insufficient_data")
        self.assertTrue((reports[0].parent / "HORIZON.md").is_file())
        self.assertEqual(result["source_run"], str(source.resolve()))

    def test_cli_missing_acceptance_writes_incomplete_report(self):
        source = self.root / "empty-run"
        source.mkdir()
        self.assertEqual(script.main(["--run", str(source), "--sky-mask", str(self.mask_path), "--sky-limit-y", "400", "--output-root", str(self.output)]), 3)
        self.assertEqual(len(list(self.output.glob("horizon-evaluation-*/HORIZON.md"))), 1)

    def test_cli_nonfinite_old_acceptance_writes_failed_report(self):
        source = self.root / "bad-run"
        source.mkdir()
        (source / "acceptance.json").write_text('{"verdict": {"passed": true}, "metric": NaN}', encoding="utf-8")
        self.assertEqual(script.main(["--run", str(source), "--sky-mask", str(self.mask_path), "--sky-limit-y", "400", "--output-root", str(self.output)]), 2)
        reports = list(self.output.glob("horizon-evaluation-*/horizon.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text(encoding="utf-8"))["verdict"]["status"], "failed")

    def test_exit_status_mapping_is_four_state(self):
        self.assertEqual(script._EXIT_CODES, {"passed": 0, "failed": 2, "insufficient_data": 3, "not_applicable": 4})
