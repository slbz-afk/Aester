"""Tests for starstack.registration_refit and starstack.fits_crop plus the two
entry scripts, on synthetic data only (no photographs, no Siril).

Run:  python -B -m unittest tests.test_s2_refit_and_crops   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import fits_crop  # noqa: E402
from starstack import registration_models as rm  # noqa: E402
from starstack import registration_refit as rr  # noqa: E402
from starstack import siril_coordinate_proof as cp  # noqa: E402
from starstack import siril_sequence  # noqa: E402
from starstack import synthetic_starfield as sf  # noqa: E402


def _stars(xy):
    return [{"x": float(x), "y": float(y)} for x, y in xy]


class Refit(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.width, self.height = 1200, 901
        self.ref = rng.uniform([10, 10], [1190, 891], size=(600, 2))
        # true frame->reference and intermediate->reference transforms
        self.h_frame_true = sf.rotation_scale_about(1.5, 1.004, 600.0, 450.0, -9.0, 14.0)
        self.h_mid_true = sf.translation(3.0, -2.0)
        noise = lambda n: rng.normal(0.0, 0.05, (n, 2))  # noqa: E731
        inv_f, inv_m = np.linalg.inv(self.h_frame_true), np.linalg.inv(self.h_mid_true)
        self.frame = rm.apply_matrix(inv_f, self.ref) + noise(600)
        self.mid = rm.apply_matrix(inv_m, self.ref) + noise(600)
        # a "Siril" matrix fitted only from the central 200 stars: exact there, biased in the corners
        centre = np.array([600.0, 450.0])
        r = np.hypot(*(self.ref - centre).T)
        central = np.argsort(r)[:200]
        biased = rm.fit_homography(self.frame[central], self.ref[central] + np.array([0.0, 0.0]))
        biased[0, 2] += 0.6            # plus a deliberate offset error
        self.h_siril = biased / biased[2, 2]

    def test_refits_beat_the_biased_matrix_on_held_out_pairs(self):
        cands = rr.candidate_matrices(_stars(self.ref), _stars(self.frame), self.h_siril, self.width, self.height,
                                      intermediate_stars=_stars(self.mid), h_intermediate_siril=self.h_mid_true, radius=6.0)
        self.assertEqual(set(cands), {"siril", "direct", "chained"})
        for name in ("direct", "chained"):
            np.testing.assert_allclose(np.asarray(cands[name]["H"]), self.h_frame_true, rtol=2e-3, atol=0.05)
        ev = rr.evaluate_candidates(_stars(self.ref), _stars(self.frame), cands, self.h_siril, self.width, self.height, radius=6.0)
        self.assertGreater(ev["pairs"], 550)
        siril = ev["candidates"]["siril"]["views"]["all_pairs"]["overall"]["rms_px"]
        for name in ("direct", "chained"):
            view = ev["candidates"][name]["views"]["excluding_fit_inputs"]
            self.assertGreater(view["count"], 200)                       # fit inputs were really excluded
            self.assertLess(view["overall"]["rms_px"], 0.12)
            self.assertLess(view["overall"]["rms_px"], siril / 3)
        self.assertEqual(ev["candidates"]["siril"]["fit_inputs_among_pairs"], 0)
        self.assertGreater(siril, 0.4)

    def test_too_few_pairs_is_reported_not_raised(self):
        cands = rr.candidate_matrices(_stars(self.ref[:10]), _stars(self.frame[:10]), self.h_siril, self.width, self.height)
        self.assertIsNone(cands["direct"]["H"])
        ev = rr.evaluate_candidates(_stars(self.ref[:10]), _stars(self.frame[:10]), cands, self.h_siril, self.width, self.height)
        self.assertFalse(ev["candidates"]["direct"]["available"])

    def test_refit_frame_on_simulated_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            truth = sf.generate_dataset(Path(d) / "stage")
            cp.simulate_siril_outputs(truth, Path(d) / "work", row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5,
                                      matrix_direction="frame_to_reference")
            seq = siril_sequence.read_seq(Path(d) / "work" / "light_.seq")
            lists = {img["stem"]: siril_sequence.read_lst(Path(d) / "work" / "cache" / f"{img['stem']}.lst") for img in seq["images"]}
            result = rr.refit_frame(seq, lists, 3, intermediate_position=1, width=1200, height=901)
            self.assertEqual((result["reference"], result["frame"], result["intermediate"]), ("light_00001", "light_00004", "light_00002"))
            for name in ("siril", "direct", "chained"):
                v = result["evaluation"]["candidates"][name]["views"]["all_pairs"]["overall"]
                self.assertLess(v["rms_px"], 0.08)           # simulated matrices are exact: everything agrees to noise


class Crops(unittest.TestCase):
    def test_crop_and_warp_recover_known_translation(self):
        layout = sf.default_layout()
        ref_img, _ = sf.render_frame(layout, np.eye(3), seed=3)
        shift = sf.translation(7.25, -3.5)                   # frame -> reference in TOP-DOWN coordinates
        frame_img, rendered = sf.render_frame(layout, shift, seed=4)
        with tempfile.TemporaryDirectory() as d:
            sf.write_fits_rgb16(Path(d) / "ref.fit", ref_img)
            sf.write_fits_rgb16(Path(d) / "frame.fit", frame_img)
            header, offset = fits_crop.read_header(Path(d) / "ref.fit")
            self.assertEqual((header["BITPIX"], header["ROWORDER"], offset), ("16", "BOTTOM-UP", 2880))
            box = (700, 1100, 500, 850)
            crop = fits_crop.crop_top_down(Path(d) / "ref.fit", 1, *box)
            np.testing.assert_array_equal(crop, ref_img[500:850, 700:1100].astype(np.float32))
            warped = fits_crop.warp_to_reference(Path(d) / "frame.fit", 1, shift, box)
            self.assertGreater(warped["coverage_fraction"], 0.99)
            stars = [(s["x"] - 700, s["y"] - 500) for s in layout["stars"] if 710 <= s["x"] < 1090 and 510 <= s["y"] < 840]
            self.assertGreater(len(stars), 8)
            res = fits_crop.measure_star_offsets(crop, warped["crop"], stars, radius=4)
            self.assertGreater(res["summary"]["count"], 8)
            self.assertLess(res["summary"]["rms_px"], 0.08)      # sub-pixel agreement after the exact warp
            wrong = fits_crop.warp_to_reference(Path(d) / "frame.fit", 1, sf.translation(7.25, -2.5), box)
            res_wrong = fits_crop.measure_star_offsets(crop, wrong["crop"], stars, radius=4)
            # the wrong matrix samples the frame 1 px above each star, so the star lands 1 px lower (+y)
            self.assertAlmostEqual(res_wrong["summary"]["mean_dy_px"], +1.0, delta=0.1)
            png = Path(d) / "x.png"
            fits_crop.write_png_gray(png, fits_crop.asinh_stretch(crop))
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
            stacked = fits_crop.stack_two(crop, warped["crop"])
            self.assertTrue(np.isfinite(stacked).all())

    def test_unsupported_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.fit"
            cards = [f"{'SIMPLE':<8}= {'T':>20}", f"{'BITPIX':<8}= {8:>20d}", f"{'NAXIS':<8}= {2:>20d}",
                     f"{'NAXIS1':<8}= {4:>20d}", f"{'NAXIS2':<8}= {4:>20d}", "END"]
            header = "".join(c.ljust(80) for c in cards)
            bad.write_bytes((header + " " * (2880 - len(header))).encode() + b"\0" * 2880)
            with self.assertRaises(ValueError):
                fits_crop.crop_top_down(bad, 0, 0, 2, 0, 2)


class SirilVariantsHelpers(unittest.TestCase):
    def _module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("s2variants", ROOT / "scripts" / "s2_siril_register_variants.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help_option_parsing_keeps_2pass_and_drops_numbers(self):
        m = self._module()
        text = "register sequencename [-2pass] [-noout] [-drizzle] [-prefix=] [-minpairs=] [-transf=] [-layer=] [-maxstars=] -1 x-y"
        opts = m.parse_help_options(text)
        for expected in ("-2pass", "-noout", "-minpairs", "-transf", "-maxstars"):
            self.assertIn(expected, opts)
        self.assertNotIn("-1", opts)
        self.assertNotIn("-y", opts)

    def test_log_pair_parsing(self):
        m = self._module()
        log = "log: Matching stars in image 16: done\nlog: Initial pair matches: 546\nlog: Pair matches after fitting: 512\nlog: Matching stars in image 2: done\nlog: Initial pair matches: 991\n"
        self.assertEqual(m.parse_pairs_from_log(log), {16: {"initial": 546, "after_fitting": 512}, 2: {"initial": 991}})
        zh = "log: 匹配星点于图像 1：完成\nlog: 初始星点匹配：435\nlog: 拟合后星点匹配：427\nlog: 找到 2000 个 高斯 形星于 图像，通道 #1（FWHM 4.693588）\nlog: 星点数量达到上限，重新调整阈值和筛选条件\n"
        self.assertEqual(m.parse_pairs_from_log(zh), {1: {"initial": 435, "after_fitting": 427}})
        det = m.parse_detection_from_log(zh)
        self.assertEqual(det["detection_fwhm_in_log_order"], [4.693588])
        self.assertEqual(det["threshold_readjust_messages"], 1)

    def test_evaluate_sequence_and_links_on_simulated_products(self):
        m = self._module()
        with tempfile.TemporaryDirectory() as d:
            truth = sf.generate_dataset(Path(d) / "work")
            work = Path(d) / "work"
            cp.simulate_siril_outputs(truth, work, row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5, matrix_direction="frame_to_reference")
            for k in range(1, 5):
                (work / f"syn_0000{k}.fit").rename(work / f"light_0000{k}.fit")
            ev = m.evaluate_sequence(work / "light_.seq", work / "cache", 1200, 901, 6.0)
            self.assertEqual(ev["reference"], "light_00001")
            self.assertEqual(sorted(ev["frames"]), ["light_00002", "light_00003", "light_00004"])
            for fr in ev["frames"].values():
                self.assertGreater(fr["pairs"], 150)
                self.assertLess(fr["overall_rms_px"], 0.08)
            info = m.link_frames(work, Path(d) / "v_test", [f"light_0000{k}" for k in range(1, 5)])
            self.assertEqual(info["count"], 4)
            self.assertEqual((Path(d) / "v_test" / "light_00001.fit").stat().st_size, (work / "light_00001.fit").stat().st_size)


class EntryScripts(unittest.TestCase):
    def test_refit_then_crop_check_on_simulated_products(self):
        with tempfile.TemporaryDirectory() as d:
            truth = sf.generate_dataset(Path(d) / "work")       # FITS frames written as syn_0000k.fit
            work = Path(d) / "work"
            cp.simulate_siril_outputs(truth, work, row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5,
                                      matrix_direction="frame_to_reference")
            for k in range(1, 5):                                   # the simulated sequence is named light_
                (work / f"syn_0000{k}.fit").rename(work / f"light_0000{k}.fit")
            run = lambda *a: subprocess.run([sys.executable, "-B", *a], cwd=str(ROOT), capture_output=True, text=True)  # noqa: E731
            proc = run(str(ROOT / "scripts" / "s2_refit_long_span.py"), "--seq", str(work / "light_.seq"), "--lst-dir", str(work / "cache"),
                       "--frame", "4", "--via", "2", "--width", "1200", "--height", "901", "--output-root", str(Path(d) / "out"))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            refit_json = next((Path(d) / "out").glob("refit-f4-via2-*/refit_long_span.json"))
            refit = json.loads(refit_json.read_text(encoding="utf-8"))
            self.assertIn("H_diagnostics", refit["candidates"]["chained"])
            proc = run(str(ROOT / "scripts" / "s2_pixel_crop_check.py"), "--refit", str(refit_json), "--box", "600", "1100", "400", "850",
                       "--detail", "800", "1000", "500", "700", "--output-root", str(Path(d) / "out"))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            crop_dir = next((Path(d) / "out").glob("crop-check-*"))
            check = json.loads((crop_dir / "crop_check.json").read_text(encoding="utf-8"))
            for name in ("siril", "direct", "chained"):
                self.assertTrue(check["candidates"][name]["available"])
                self.assertLess(check["candidates"][name]["offsets"]["summary"]["rms_px"], 0.15)
                self.assertTrue((crop_dir / f"{name}_stack2.png").exists())
                self.assertTrue((crop_dir / f"detail_{name}_stack2.png").exists())
            self.assertTrue((crop_dir / "reference.png").exists())


if __name__ == "__main__":
    unittest.main()
