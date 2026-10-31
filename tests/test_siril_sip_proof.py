"""Tests for starstack.siril_sip_proof and its entry script (synthetic only; Siril never runs here).

Run:  python -B -m unittest tests.test_siril_sip_proof   (from WORKSPACE, PYTHONPATH=src)
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

from starstack import siril_sip_proof as sp  # noqa: E402
from starstack import siril_sequence as sip_seq  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402
from starstack.synthetic_starfield import read_fits_minimal  # noqa: E402


class SipHeaderTest(unittest.TestCase):
    def test_forward_polynomial_matches_the_radial_model_in_fits_coordinates(self):
        width, height = 1200, 901
        d = sp.default_distortion(width, height)
        coeffs = sp.sip_forward_coefficients(d)
        self.assertEqual(coeffs["order"], 5)
        rng = np.random.default_rng(1)
        p_td = np.column_stack([rng.uniform(0, width - 1, 300), rng.uniform(0, height - 1, 300)])
        u_td = d.apply(p_td)
        crpix = (d.cx + 1.0, height - d.cy)
        p_fits = np.column_stack([p_td[:, 0] + 1.0, height - p_td[:, 1]])
        u_fits_expected = np.column_stack([u_td[:, 0] + 1.0, height - u_td[:, 1]])
        u_fits = sp.sip_apply_forward(coeffs, p_fits, crpix)
        np.testing.assert_allclose(u_fits, u_fits_expected, atol=1e-9)
        inverse = sp.sip_inverse_coefficients(d, width, height)
        self.assertLess(inverse["max_fit_error_px"], 0.05)   # order-5 inverse of a 2 % corner distortion: ~0.02 px at the extreme corners
        cards = sp.sip_cards(d, width, height)
        self.assertTrue(all(len(c) == 80 for c in cards))
        self.assertIn("A_0_0", [c[:8].strip() for c in cards])   # zero terms written explicitly, like Siril
        flipped = sp.sip_cards(d, width, height, crpix2_convention="top_down")
        crpix2 = float(next(c for c in flipped if c.startswith("CRPIX2")).split("=")[1].split("/")[0])
        self.assertAlmostEqual(crpix2, d.cy + 1.0)
        keys = [c[:8].strip() for c in cards]
        for key in ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "A_ORDER", "B_ORDER", "A_3_0", "B_0_3", "A_5_0", "AP_ORDER", "BP_1_0"):
            self.assertIn(key, keys)
        self.assertIn("'RA---TAN-SIP'", next(c for c in cards if c.startswith("CTYPE1")))
        # global magnification enters as the linear SIP terms and keeps the polynomial exact
        magnified = d.with_magnification(d.fitting_magnification(width, height))
        self.assertLess(magnified.magnification, 1.0)
        coeffs_m = sp.sip_forward_coefficients(magnified)
        self.assertAlmostEqual(coeffs_m["A"][(1, 0)], magnified.magnification - 1.0)
        u_m = sp.sip_apply_forward(coeffs_m, p_fits, crpix)
        u_td_m = magnified.apply(p_td)
        np.testing.assert_allclose(u_m, np.column_stack([u_td_m[:, 0] + 1.0, height - u_td_m[:, 1]]), atol=1e-9)
        corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
        um = magnified.apply(corners)
        self.assertTrue((um[:, 0] >= -1e-6).all() and (um[:, 0] <= width - 1 + 1e-6).all())
        self.assertTrue((um[:, 1] >= -1e-6).all() and (um[:, 1] <= height - 1 + 1e-6).all())

    def test_order_five_inverse_is_exact_only_for_the_inverse_direction_model(self):
        # RUN_LOG §22: on the real Z8 20 mm model (3.6 % at the corner) the order-5 AP/BP least-squares inverse of the p_to_u form
        # leaves ~1 px at the extreme corners of the 8280x5520 frame, and Siril crashes on higher orders; the u_to_p form makes
        # the order-5 AP/BP exact.
        width, height = 8280, 5520
        z8 = RadialDistortion(0.053731, -0.017605, 4188.34, 2686.79, float(np.hypot(width, height) / 2.0))
        z8 = z8.with_magnification(z8.fitting_magnification(width, height))
        errors = {order: sp.sip_inverse_coefficients(z8, width, height, order)["max_fit_error_px"] for order in (5, 7, 9)}
        self.assertGreater(errors[5], 0.5)
        self.assertLess(errors[9], 0.02)
        self.assertFalse(sp.sip_inverse_coefficients(z8, width, height)["exact"])
        inv = RadialDistortion(-0.0497, 0.0125, 4188.34, 2686.79, float(np.hypot(width, height) / 2.0), direction="u_to_p")
        inv = inv.with_magnification(inv.fitting_magnification(width, height))
        self.assertLess(inv.magnification, 1.0)
        terms = sp.sip_inverse_coefficients(inv, width, height)
        self.assertTrue(terms["exact"])
        self.assertEqual(terms["order"], 5)
        forward = sp.sip_forward_coefficients(inv, width, height)
        self.assertFalse(forward["exact"])
        self.assertGreater(forward["max_fit_error_px"], 0.3)      # the fitted side, unused by the product's export
        # the written inverse, evaluated as Siril does (FITS 1-based, bottom-up, relative to CRPIX), reproduces the model exactly
        cards = sp.sip_cards(inv, width, height)
        keys = [c[:8].strip() for c in cards]
        self.assertEqual(int(next(c for c in cards if c.startswith("AP_ORDER")).split("=")[1].split("/")[0]), 5)
        self.assertEqual(sum(1 for k in keys if k.startswith("AP_") and k != "AP_ORDER"), 21)
        ap = {tuple(int(v) for v in k[3:].split("_")): float(c.split("=")[1].split("/")[0]) for k, c in zip(keys, cards) if k.startswith("AP_") and k != "AP_ORDER"}
        bp = {tuple(int(v) for v in k[3:].split("_")): float(c.split("=")[1].split("/")[0]) for k, c in zip(keys, cards) if k.startswith("BP_") and k != "BP_ORDER"}
        crpix = (inv.cx + 1.0, height - inv.cy)
        grid = np.array([[x, y] for y in np.linspace(0, height - 1, 9) for x in np.linspace(0, width - 1, 13)], dtype=np.float64)
        u_fits = np.column_stack([grid[:, 0] + 1.0, height - grid[:, 1]]) - np.array(crpix)
        f = sum(c * u_fits[:, 0] ** p * u_fits[:, 1] ** q for (p, q), c in ap.items())
        g = sum(c * u_fits[:, 0] ** p * u_fits[:, 1] ** q for (p, q), c in bp.items())
        p_fits = u_fits + np.column_stack([f, g]) + np.array(crpix)
        p_td = np.column_stack([p_fits[:, 0] - 1.0, height - p_fits[:, 1]])
        self.assertLess(float(np.max(np.hypot(*(p_td - inv.inverse(grid)).T))), 1e-6)
        # round trip and containment of the u_to_p model
        rng = np.random.default_rng(2)
        pts = np.column_stack([rng.uniform(0, width - 1, 500), rng.uniform(0, height - 1, 500)])
        np.testing.assert_allclose(inv.inverse(inv.apply(pts)), pts, atol=1e-8)
        corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
        um = inv.apply(corners)
        self.assertTrue((um[:, 0] >= -1e-6).all() and (um[:, 0] <= width - 1 + 1e-6).all())
        self.assertTrue((um[:, 1] >= -1e-6).all() and (um[:, 1] <= height - 1 + 1e-6).all())

    def test_z8_like_proof_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            truth = sp.generate_dataset(Path(d) / "work", distortion=sp.z8_like_distortion())
            by_order = truth["sip"]["inverse_max_fit_error_px_by_order"]
            self.assertGreater(by_order["5"], 5 * by_order["9"])
            self.assertEqual(truth["sip"]["direction"], "p_to_u")
            self.assertGreater(truth["distortion_corner_shift_px"], 20.0)
        with tempfile.TemporaryDirectory() as d:
            truth = sp.generate_dataset(Path(d) / "work", distortion=sp.z8_like_inverse_distortion())
            self.assertEqual(truth["sip"]["direction"], "u_to_p")
            self.assertEqual(truth["sip"]["inverse_max_fit_error_px"], 0.0)
            self.assertGreater(truth["sip"]["forward_max_fit_error_px"], 0.02)
            self.assertGreater(truth["distortion_corner_shift_px"], 20.0)
            header = (Path(d) / "work" / "disto_solution_mag.wcs").read_bytes().decode("ascii")
            self.assertIn("AP_ORDER=                    5", header)
            self.assertEqual(truth["distortion"]["direction"], "u_to_p")


class DatasetTest(unittest.TestCase):
    def test_generated_truth_is_self_consistent_and_solution_file_carries_the_header(self):
        with tempfile.TemporaryDirectory() as d:
            truth = sp.generate_dataset(Path(d) / "work")
            work = Path(d) / "work"
            self.assertEqual(sorted(p.name for p in work.glob("light_*.fit")), [f"light_0000{k}.fit" for k in range(1, 5)])
            dist = RadialDistortion(**truth["distortion"])
            for frame in truth["frames"]:
                und = {s["star_id"]: (s["x"], s["y"]) for s in frame["undistorted_stars"]}
                for star in frame["rendered_stars"]:
                    u = np.array(und[star["star_id"]])
                    p = dist.inverse(u[None, :])[0]
                    self.assertLess(np.hypot(p[0] - star["x"], p[1] - star["y"]), 1e-6)
            header, plane = read_fits_minimal(work / "disto_solution.fit")
            self.assertEqual(header["CTYPE1"], "RA---TAN-SIP")
            self.assertEqual(int(header["A_ORDER"]), 5)
            self.assertAlmostEqual(float(header["CRPIX2"]), truth["sip"]["crpix_fits"][1])
            ref_header, ref_plane = read_fits_minimal(work / "light_00001.fit")
            np.testing.assert_array_equal(plane, ref_plane)
            self.assertNotIn("CTYPE1", ref_header)
            self.assertTrue((work / "disto_solution_mag.wcs").exists())
            self.assertLess(truth["solution_wcs_magnified"]["magnification"], 1.0)
            wcs = (work / "disto_solution.wcs").read_bytes()
            self.assertEqual(len(wcs) % 2880, 0)
            text = wcs.decode("ascii")
            for key in ("NAXIS   =                    0", "IMAGEW  =                 1200", "IMAGEH  =                  901", "CTYPE1  = 'RA---TAN-SIP'", "A_ORDER =                    5"):
                self.assertIn(key, text)


class AnalysisTest(unittest.TestCase):
    def _run(self, applied: bool, magnification: float = 1.0) -> dict:
        with tempfile.TemporaryDirectory() as d:
            truth = sp.generate_dataset(Path(d) / "work")
            target = Path(d) / "v_x"
            target.mkdir()
            for path in (Path(d) / "work").glob("light_*.fit"):
                (target / path.name).write_bytes(path.read_bytes())
            sp.simulate_siril_outputs(truth, target, applied=applied)
            return sp.analyze_run(truth, target, variant="x", magnification=magnification)

    def test_simulated_applied_and_ignored_are_told_apart(self):
        applied = self._run(True)
        self.assertEqual(applied["status"], "sip_applied_SIMULATED")
        for stem, e in applied["frames"].items():
            self.assertGreaterEqual(e["star_list"]["fraction_at_distorted_truth"], 0.8, stem)   # lists are raw detections
            if "homography" in e:
                self.assertLess(e["homography"]["corner_error_vs_undistorted_truth_px"], 0.15, stem)
                self.assertGreater(e["homography"]["corner_error_vs_distorted_fit_px"], 0.5, stem)
            self.assertLess(e["output"]["at_undistorted_reference"]["rms_px"], 0.3, stem)
        wrong_scale = self._run(True, magnification=0.97)     # truth scaled but Siril output not: must not pass
        self.assertNotEqual(wrong_scale["status"], "sip_applied_SIMULATED")
        ignored = self._run(False)
        self.assertEqual(ignored["status"], "sip_ignored_SIMULATED")
        for stem, e in ignored["frames"].items():
            if "homography" in e:
                self.assertLess(e["homography"]["corner_error_vs_distorted_fit_px"], 0.15, stem)
            self.assertLess(e["output"]["at_distorted_reference"]["rms_px"], 0.3, stem)


class OwnRegistrationTest(unittest.TestCase):
    def test_written_sequence_and_shifted_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            truth = sp.generate_dataset(Path(d) / "work")
            target = Path(d) / "v_own"
            target.mkdir()
            for path in (Path(d) / "work").glob("light_*.fit"):
                (target / path.name).write_bytes(path.read_bytes())
            sp.simulate_siril_outputs(truth, target, applied=False)          # a plain register: H fitted on distorted stars
            m = truth["solution_wcs_magnified"]["magnification"]
            written = sp.own_registration_sequence(target, truth, distortion_file="disto_solution_mag.wcs", magnification=m,
                                                   shift_px=(7.3, -4.1), zero_quality_frames=(4,))
            seq = sip_seq.read_seq(target / "light_.seq")
            self.assertEqual(seq["header"]["nb_selected"], 4)
            self.assertTrue(any(line.startswith("D1 2 disto_solution_mag.wcs") for line in (target / "light_.seq").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(seq["regdata"][1][3]["number_of_stars"], 0)     # frame 4's quality fields zeroed
            self.assertEqual(seq["regdata"][1][0]["H"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            # reading back through the proved hypothesis gives exactly the matrices we intended to write
            hyp = adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
            for image in seq["images"]:
                h_back = adapter.convert_matrix(hyp, seq["regdata"][1][image["position"]]["H"], truth["height"], truth["height"])
                np.testing.assert_allclose(h_back, np.asarray(written["homographies_top_down"][image["stem"]]), atol=1e-6)
            # a fake seqapplyreg that follows the written matrices -> applied; one that ignores them -> ignored
            sp.simulate_seqapplyreg(truth, target, magnification=m)
            applied = sp.analyze_run(truth, target, variant="own", magnification=m, own_registration=True, expected_shift_px=(7.3, -4.1))
            self.assertEqual(applied["status"], "own_registration_applied_SIMULATED", applied["reason"])
            for stem, e in applied["frames"].items():
                if e["frame"] != applied["reference_frame"]:
                    self.assertGreater(e["output"]["at_unshifted_undistorted_reference"]["interior"]["median_px"], 5.0, stem)
            sp.simulate_siril_outputs(truth, target, applied=True)           # rewrites seq + r_ frames as Siril's own registration would
            sp.own_registration_sequence(target, truth, distortion_file="disto_solution_mag.wcs", magnification=m, shift_px=(7.3, -4.1))
            ignored = sp.analyze_run(truth, target, variant="own", magnification=m, own_registration=True, expected_shift_px=(7.3, -4.1))
            self.assertEqual(ignored["status"], "own_registration_ignored_SIMULATED", ignored["reason"])


class EntryScriptTest(unittest.TestCase):
    def test_simulate_subcommand_writes_report(self):
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "s2_siril_sip_proof.py"), "simulate", "--applied",
                                   "--output-root", d], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            run = next(Path(d).glob("sip-proof-*"))
            results = json.loads((run / "sip_proof.json").read_text(encoding="utf-8"))
            self.assertEqual(results["variants"]["disto_file"]["status"], "sip_applied_SIMULATED")
            self.assertEqual(results["variants"]["nodisto"]["status"], "sip_ignored_SIMULATED")
            self.assertEqual(results["variants"]["ownreg_mag_shift"]["status"], "own_registration_applied_SIMULATED")
            self.assertIn("SIMULATED", (run / "SIP_PROOF.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
