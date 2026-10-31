"""Tests for the Siril zero/no-data proof (synthetic only; Siril itself is not run here).

Run:  python -B -m unittest tests.test_siril_nodata_proof   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from starstack import siril_nodata_proof as npf  # noqa: E402
from starstack import siril_sequence  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack.synthetic_starfield import read_fits_minimal, write_fits_rgb16  # noqa: E402
import s3_siril_nodata_proof as script  # noqa: E402


class DatasetTest(unittest.TestCase):
    def test_frames_hold_the_declared_regions(self):
        with tempfile.TemporaryDirectory() as d:
            truth = npf.generate_dataset(Path(d) / "work")
            self.assertEqual(len(truth["frames"]), npf.FRAMES)
            for f in truth["frames"]:
                _, plane = read_fits_minimal(Path(d) / "work" / f"light_{f['frame']:05d}.fit")
                for name, r in truth["regions"].items():
                    sub = plane[r["y0"]:r["y1"], r["x0"]:r["x1"]].astype(float) / npf.FULL
                    if f["frame"] in r["frames"]:
                        if r["kind"] == "zero":
                            self.assertEqual(sub.max(), 0.0, (name, f["frame"]))
                        else:
                            self.assertAlmostEqual(float(np.median(sub)), npf.GROUND, delta=0.002, msg=(name, f["frame"]))
                    else:
                        self.assertAlmostEqual(float(np.median(sub)), f["sky_level"], delta=0.002, msg=(name, f["frame"]))
                x0, x1, y0, y1 = truth["sky_control"]
                self.assertAlmostEqual(float(np.median(plane[y0:y1, x0:x1]) / npf.FULL), f["sky_level"], delta=0.002)


class SequenceTest(unittest.TestCase):
    def _truth(self, d):
        return npf.generate_dataset(Path(d) / "work")

    def test_hand_written_sequence_parses_with_our_translations(self):
        with tempfile.TemporaryDirectory() as d:
            truth = self._truth(d)
            target = Path(d) / "work"
            info = npf.write_translation_sequence(target, truth)
            self.assertEqual(info["how"], "written_by_hand")
            seq = siril_sequence.read_seq(target / "light_.seq")
            self.assertEqual(len(seq["images"]), npf.FRAMES)
            hyp = adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
            for pos, reg in enumerate(seq["regdata"][1]):
                h_td = adapter.convert_matrix(hyp, reg["H"], truth["height"], truth["height"])
                dx, dy = truth["frames"][pos]["shift_px"]
                np.testing.assert_allclose(h_td, [[1, 0, dx], [0, 1, dy], [0, 0, 1]], atol=1e-6)

    def test_siril_created_sequence_without_registration_lines_gets_ours(self):
        with tempfile.TemporaryDirectory() as d:
            truth = self._truth(d)
            target = Path(d) / "work"
            n = npf.FRAMES
            # exactly what Siril 1.4.4 writes after `seqheader` on unloaded frames (RUN_LOG §24): reference -1, layers -1
            text = ("#Siril sequence file. Contains list of images, selection, registration data and statistics\n"
                    "#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle\n"
                    f"S 'light_' 1 {n} {n} 5 -1 6 0 0 0\nL -1\n" + "".join(f"I {i + 1} 1\n" for i in range(n)))
            (target / "light_.seq").write_text(text, encoding="utf-8")
            info = npf.write_translation_sequence(target, truth)
            self.assertEqual(info["how"], "added_r_lines_to_siril_sequence")
            seq = siril_sequence.read_seq(target / "light_.seq")
            self.assertEqual(len(seq["regdata"][1]), n)
            self.assertEqual(seq["header"]["reference_image"], 0)
            self.assertEqual(seq["header"]["nb_selected"], n)
            self.assertEqual(seq["layers"], 3)


class DeselectTest(unittest.TestCase):
    def test_deselected_frame_written_unselected_with_zero_matrix(self):
        for siril_created in (False, True):
            with tempfile.TemporaryDirectory() as d:
                truth = npf.generate_dataset(Path(d) / "work")
                target = Path(d) / "work"
                n = npf.FRAMES
                if siril_created:
                    (target / "light_.seq").write_text(f"S 'light_' 1 {n} {n} 5 -1 6 0 0 0\nL -1\n" + "".join(f"I {i + 1} 1\n" for i in range(n)), encoding="utf-8")
                info = npf.write_translation_sequence(target, truth, deselect=npf.DESELECTED_FRAME)
                self.assertEqual(info["deselected"], npf.DESELECTED_FRAME)
                seq = siril_sequence.read_seq(target / "light_.seq")
                self.assertEqual(seq["header"]["nb_selected"], n - 1)
                flags = {im["index"]: im["selected"] for im in seq["images"]}
                self.assertFalse(flags[npf.DESELECTED_FRAME]); self.assertTrue(all(v for k, v in flags.items() if k != npf.DESELECTED_FRAME))
                self.assertEqual(seq["regdata"][1][npf.DESELECTED_FRAME - 1]["H"], [[0.0, 0.0, 0.0]] * 3)
                self.assertNotEqual(seq["regdata"][1][0]["H"], [[0.0, 0.0, 0.0]] * 3)

    def test_deselected_analysis_reads_export_and_stackcnt(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            truth = npf.generate_dataset(run / "work")
            target = run / "v_applyreg_deselected_filtered"; target.mkdir()
            ref = truth["frames"][0]["sky_level"]
            img = np.full((npf.HEIGHT, npf.WIDTH), ref)
            write_fits_rgb16(target / "stack.fit", np.round(img * npf.FULL).astype(np.uint16), extra_cards=["STACKCNT=                    7".ljust(80)])
            for k in range(1, npf.FRAMES + 1):
                if k != npf.DESELECTED_FRAME:
                    write_fits_rgb16(target / f"r_light_{k:05d}.fit", np.round(img * npf.FULL).astype(np.uint16))
            out = npf.analyze_variant(truth, target, variant="applyreg_deselected_filtered")
            self.assertTrue(out["verdicts"]["deselected_frame_skipped"])
            self.assertEqual(out["deselected"]["exported_frames"], npf.FRAMES - 1)
            self.assertEqual(out["deselected"]["stackcnt"], 7)


class AnalysisTest(unittest.TestCase):
    def test_simulated_behaviours_are_classified(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            run.mkdir()
            npf.generate_dataset(run / "work")
            script.simulate(run)
            results = script.analyze(run, siril_version="SIMULATED")
            self.assertEqual(results["variants"]["rej_addscale"]["verdicts"]["zeros"], "zeros_excluded")
            self.assertTrue(results["variants"]["rej_addscale"]["verdicts"]["single_valid_frame_survives"])
            self.assertEqual(results["variants"]["rej_addscale"]["normalisation"]["target"], "reference")
            self.assertEqual(results["variants"]["mean_nonorm"]["verdicts"]["zeros"], "zeros_included")
            self.assertEqual(results["variants"]["mean_nonorm"]["normalisation"]["target"], "mean")
            text = npf.render(results)
            self.assertIn("zeros_excluded", text)
            self.assertIn("zeros_included", text)

    def test_missing_master_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            truth = npf.generate_dataset(run / "work")
            (run / "v_rej_addscale").mkdir()
            out = npf.analyze_variant(truth, run / "v_rej_addscale", variant="rej_addscale")
            self.assertEqual(out["status"], "no_stack_output")

    def test_applyreg_regions_are_read_at_their_shifted_places(self):
        """A master whose zero-majority region kept the sky value except in the all-zero region, plus an exported frame 2
        with the interpolation rim: the analysis must read the regions at the reference position (shift 0 for frame 1)."""
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            truth = npf.generate_dataset(run / "work")
            target = run / "v_applyreg_rej_addscale"
            target.mkdir()
            ref = truth["frames"][0]["sky_level"]
            img = np.full((npf.HEIGHT, npf.WIDTH), ref)
            r = truth["regions"]["Z_all"]
            img[r["y0"]:r["y1"], r["x0"]:r["x1"]] = 0.0
            write_fits_rgb16(target / "stack.fit", np.round(img * npf.FULL).astype(np.uint16))
            # exported frame 2: sky with the Z_maj zeros shifted by (2.5, -1.5) and a 2-px soft edge
            f2 = np.full((npf.HEIGHT, npf.WIDTH), ref)
            z = truth["regions"]["Z_maj"]
            dx, dy = npf.frame_shift(2)
            x0, y0 = int(round(z["x0"] + dx)), int(round(z["y0"] + dy))
            f2[y0:y0 + (z["y1"] - z["y0"]), x0:x0 + (z["x1"] - z["x0"])] = 0.0
            f2[y0:y0 + (z["y1"] - z["y0"]), x0 - 2:x0] = ref * 0.5
            write_fits_rgb16(target / "r_light_00002.fit", np.round(f2 * npf.FULL).astype(np.uint16))
            out = npf.analyze_variant(truth, target, variant="applyreg_rej_addscale")
            self.assertEqual(out["verdicts"]["zeros"], "zeros_excluded")
            self.assertIn("interpolation_rim", out)
            self.assertLess(out["interpolation_rim"]["min_over_level"], 0.01)
            self.assertGreaterEqual(out["interpolation_rim"]["rim_width_px_below_0.98"], 1)


if __name__ == "__main__":
    unittest.main()
