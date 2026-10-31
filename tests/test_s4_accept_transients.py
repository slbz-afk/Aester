"""The S4 acceptance script on a synthetic product job (controlled engine, sky mask, one injected trail): every
R-check runs end to end, including the recompose round trip (RUN_LOG §28).  No Siril, no photographs.

Run:  python -B -m unittest tests.test_s4_accept_transients   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from starstack import core  # noqa: E402

HAVE_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "astropy", "tifffile", "PIL"))


@unittest.skipUnless(HAVE_STACK, "product imaging stack (numpy/astropy/tifffile/PIL) not installed")
class AcceptanceScriptTest(unittest.TestCase):
    HEIGHT, WIDTH = 240, 320
    GROUND_ROW = 200

    def setUp(self):
        import numpy as np
        from PIL import Image
        from starstack import imaging
        from starstack import registration_calibration as rc
        from test_core_two_pass_reference import JobDiagnosticsTest, _seq_text, admit_before_contract
        self.tmp = Path(tempfile.mkdtemp(prefix="s4-accept-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        source = self.tmp / "素材"
        source.mkdir()
        rng = np.random.default_rng(11)
        h, w = self.HEIGHT, self.WIDTH
        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        stars = np.zeros((h, w))
        for _ in range(50):
            x, y, a = rng.uniform(5, w - 5), rng.uniform(5, h - 5), rng.uniform(0.05, 0.3)
            stars += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4 ** 2))
        ux, uy = 280 - 40, 60 - 180          # from (40, 180) (in the sky, above the ground row) up to (280, 60)
        length = (ux ** 2 + uy ** 2) ** 0.5
        ux, uy = ux / length, uy / length
        along = (xx - 40) * ux + (yy - 180) * uy
        across = -(xx - 40) * uy + (yy - 180) * ux
        curve = 0.12 + 1.25 * np.exp(-((along - 0.55 * length) / (0.22 * length)) ** 2)     # a meteor light curve
        trail = np.where((along >= 0) & (along <= length), 0.12 * curve * np.exp(-across ** 2 / (2 * 1.5 ** 2)), 0.0)   # bright: R2 wants >= 1.0 contrast
        names = ["DSC_0001.fit", "DSC_0002.fit", "DSC_2269.fit", "DSC_0004.fit", "DSC_0005.fit"]
        for number, name in enumerate(names, 1):
            top_down = np.stack([0.03 + stars * g for g in (1.0, 0.9, 0.8)], axis=-1)
            if name == "DSC_2269.fit":
                top_down = top_down + trail[..., None] * np.array([0.9, 1.0, 1.1])
            top_down = top_down + rng.normal(0, 0.002, top_down.shape)
            imaging.write_linear(source / name, np.ascontiguousarray(top_down[::-1]).astype(np.float32), "Test fixture")
        mask = np.full((h, w), 255, dtype=np.uint8)
        mask[self.GROUND_ROW:, :] = 0
        self.mask_png = self.tmp / "sky_mask.png"
        Image.fromarray(mask).save(self.mask_png)
        self.source, self.output = source, self.tmp / "results"
        self.engine = {"available": True, "executable": str(self.tmp / "fake engine.exe"), "version": "1.4.4", "message": "Test engine"}
        self.saved = {name: getattr(core, name) for name in ("doctor", "_spawn_worker", "run_script")}
        admit_before_contract(self)
        self.saved_disk_usage = core.shutil.disk_usage
        self.saved_calibrate = rc.calibrate
        core.doctor = lambda *_: dict(self.engine)
        core._spawn_worker = lambda directory: None
        core.shutil.disk_usage = lambda _: SimpleNamespace(free=10**12)
        rc.calibrate = lambda work, **kwargs: JobDiagnosticsTest._fake_calibration(work, {f"light_{i:05d}" for i in range(1, 6)}, count=5)

        def run_script(executable, directory, name, commands, cancel_check):
            work = directory / "work"
            if name == "01_convert":
                for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                    shutil.copyfile(path, work / f"light_{i:05d}.fit")
            elif name == "02_probe":
                pass
            elif name == "02_export":   # P1b-3: L's export process
                assert list(commands) == core.export_script(core.DISTORTION_WCS, own_registration=True), commands
                frames = sorted(work.glob("light_*.fit"))
                for path in frames:
                    shutil.copyfile(path, work / f"r_{path.name}")      # identity export of the masked frames
                (work / core.SEQUENCE_FILE).write_text(_seq_text(len(frames), 1), encoding="utf-8")
            elif name == "02_stack":    # P1b-3: L's stack process
                assert list(commands) == list(core.LEGACY_STACK_SCRIPT), commands
                frames = sorted(work.glob("r_light_*.fit"))
                stack = np.stack([imaging.read_rgb(path) for path in frames])
                valid = stack[..., 1] != 0
                master = np.where(valid[..., None], stack, np.nan)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    master = np.nan_to_num(np.nanmedian(master, axis=0)).astype(np.float32)
                imaging.write_linear(work / "sky_master.fit", master, "Controlled engine: zero-excluded median", extra={"STACKCNT": len(frames)})
            else:
                raise AssertionError(name)

        core.run_script = run_script

    def tearDown(self):
        from starstack import registration_calibration as rc
        rc.calibrate = self.saved_calibrate
        for name, value in self.saved.items():
            setattr(core, name, value)
        core.shutil.disk_usage = self.saved_disk_usage

    def test_script_judges_a_synthetic_job(self):
        import s4_accept_transients as accept
        job = core.submit_job(str(self.source), str(self.output), sky_mask_path=str(self.mask_png))
        core.run_job(job["job_dir"])
        result = core.load_job(job["job_dir"])
        self.assertEqual(result["state"], "succeeded", result.get("error"))
        run = self.tmp / "run"
        run.mkdir()
        code = accept.main(["--job-dir", job["job_dir"], "--sky-mask", str(self.mask_png), "--run", str(run), "--known-input", "DSC_2269.fit",
                            "--known-min-length", "100", "--old-roi", "0", "0", str(self.WIDTH), str(self.HEIGHT)])
        report = json.loads((run / "transients_acceptance.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, report["verdict"])
        self.assertTrue(report["R1"]["passed"])
        self.assertTrue(report["R2"]["auto_detected"])
        self.assertEqual(report["R2"]["stem"], "light_00005")      # inputs sort by name: DSC_2269 is the fifth
        self.assertGreaterEqual(report["R2"]["length_px"], 100)
        self.assertEqual(report["R2"]["suggestion"]["class"], "meteor")
        self.assertLessEqual(report["R3"]["fraction_G"], 0.02)
        self.assertGreaterEqual(report["R4"]["grid_fraction"], 0.8)
        self.assertGreaterEqual(report["R4"]["native_fraction"], 0.75)
        self.assertFalse(report["R4"]["overridden_to_keep"])
        r5 = report["R5"]
        self.assertEqual(r5["changed_outside_footprints"], 0)
        self.assertEqual(r5["changed_in_ground"], 0)
        self.assertTrue(r5["omit_no_composite"] and r5["base_unchanged_after_omit"] and r5["restored_bit_exact"])
        self.assertTrue((run / "TRANSIENTS.md").is_file())
        self.assertTrue((run / "transients_before_after.png").is_file())
        self.assertEqual(len(report["R7"]["candidates"]), 1)
        self.assertTrue(report["R7"]["passed"] and report["R7"]["measured"] == 1 and report["R7"]["incomplete"] == [])
        self.assertIn("按修订后判据", (run / "TRANSIENTS.md").read_text(encoding="utf-8"))

    def test_listing_check_is_falsifiable(self):
        import s4_accept_transients as accept
        good = {"candidates": [{"id": "a", "measurement": {"status": "measured"}, "suggestion": {"class": "meteor", "reasons": ["x"]},
                                "evidence": {"continues": [], "persistent": False, "periodic": {"periodic": False},
                                             "light_curve": {"ends": {"start": "faded", "stop": "faded"}}}}]}
        self.assertTrue(accept.listing_check(good)["passed"])
        for broken in ({"suggestion": {"class": "comet", "reasons": ["x"]}}, {"suggestion": {"class": "meteor", "reasons": []}},
                       {"evidence": {"continues": []}}, {"evidence": {"continues": [], "persistent": False, "periodic": {},
                                                                     "light_curve": {"ends": {"start": "faded"}}}}):
            record = {"candidates": [dict(good["candidates"][0], **broken)]}
            self.assertFalse(accept.listing_check(record)["passed"], broken)
        self.assertFalse(accept.listing_check({"candidates": []})["passed"])       # nothing measured is not a pass


class StemLookupTest(unittest.TestCase):
    """R2 on a real run: the subset inputs are hard links named frame000NN.dng, the known trail is named by its source
    (DSC_2269.dng) -> the run manifest maps it (1.0.0 returned None and failed R2 on the first photographed run)."""

    def test_manifest_and_spellings(self):
        import s4_accept_transients as accept
        tmp = Path(tempfile.mkdtemp(prefix="s4-stem-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        job = {"inputs": [{"path": str(tmp / "inputs" / f"frame{i:05d}.dng")} for i in range(1, 49)]}
        frames = [{"frame": i, "stem": f"light_{i:05d}", "source": f"C:\\raw\\DSC_{2200 + 3 * i}.dng"} for i in range(1, 49)]
        (tmp / "inputs.json").write_text(json.dumps({"method": "hardlink", "frames": frames}), encoding="utf-8")
        self.assertEqual(accept._stem_of(job, "DSC_2266.dng", tmp), "light_00022")
        self.assertEqual(accept._stem_of(job, "dsc_2266.DNG", tmp), "light_00022")
        self.assertEqual(accept._stem_of(job, "frame00022.dng", tmp), "light_00022")
        self.assertEqual(accept._stem_of(job, "light_00022", tmp), "light_00022")
        self.assertIsNone(accept._stem_of(job, "DSC_9999.dng", tmp))
        self.assertIsNone(accept._stem_of(job, "DSC_2266.dng", None))
        self.assertIsNone(accept._stem_of(job, "frame00099.dng", tmp))


if __name__ == "__main__":
    unittest.main()
