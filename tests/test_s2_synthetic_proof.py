"""Tests for the synthetic star field generator, the coordinate-proof analysis and
the two S2 entry scripts (no Siril: simulated outputs only, labelled as such).

Run:  python -B -m unittest tests.test_s2_synthetic_proof   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import siril_coordinate_proof as cp  # noqa: E402
from starstack import synthetic_starfield as sf  # noqa: E402


def centroid(image: np.ndarray, x: float, y: float, background: float, half: int = 6) -> tuple[float, float]:
    r0, r1 = int(round(y)) - half, int(round(y)) + half + 1
    c0, c1 = int(round(x)) - half, int(round(x)) + half + 1
    patch = image[r0:r1, c0:c1].astype(np.float64) - background
    patch[patch < 0] = 0
    yy, xx = np.mgrid[r0:r1, c0:c1]
    total = patch.sum()
    return float((patch * xx).sum() / total), float((patch * yy).sum() / total)


class Generator(unittest.TestCase):
    def test_layout_is_asymmetric_and_separated(self):
        layout = sf.default_layout()
        self.assertEqual(len(layout["stars"]), 170)
        xy = np.array([[s["x"], s["y"]] for s in layout["stars"]])
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        self.assertGreaterEqual(d.min(), 12.0)
        # more stars in the upper-left half than in the lower-right half (row order is detectable)
        upper_left = ((xy[:, 0] < 600) & (xy[:, 1] < 450)).sum()
        lower_right = ((xy[:, 0] >= 600) & (xy[:, 1] >= 450)).sum()
        self.assertGreater(upper_left, 2 * lower_right)
        # a vertically flipped layout does not coincide with itself
        flipped = xy.copy()
        flipped[:, 1] = (layout["height"] - 1) - flipped[:, 1]
        d2 = np.hypot(flipped[:, None, 0] - xy[None, :, 0], flipped[:, None, 1] - xy[None, :, 1]).min(axis=1)
        self.assertLess((d2 < 1.0).mean(), 0.1)

    def test_rendered_star_centroids_match_truth(self):
        layout = sf.default_layout()
        image, rendered = sf.render_frame(layout, sf.translation(7.25, -3.5), seed=2)
        self.assertEqual(image.dtype, np.uint16)
        self.assertEqual(image.shape, (901, 1200))
        bright = sorted(rendered, key=lambda s: -s["amplitude"])[:20]
        errors = []
        for star in bright:
            if 8 <= star["x"] <= 1191 and 8 <= star["y"] <= 892:
                cx, cy = centroid(image, star["x"], star["y"], 800.0)
                errors.append(np.hypot(cx - star["x"], cy - star["y"]))
        self.assertGreater(len(errors), 10)
        self.assertLess(max(errors), 0.2)
        # the translation moved the stars: rendered frame coordinates are truth - (7.25, -3.5)
        by_id = {s["star_id"]: s for s in rendered}
        self.assertAlmostEqual(by_id["center"]["x"], 603.25 - 7.25)
        self.assertAlmostEqual(by_id["center"]["y"], 447.75 + 3.5)

    def test_rotation_matrix_about_centre(self):
        m = sf.rotation_scale_about(90.0, 2.0, 100.0, 100.0)
        v = m @ np.array([110.0, 100.0, 1.0])
        # (10, 0) offset rotated by +90 deg (x towards y) and scaled by 2 -> (0, 20)
        self.assertAlmostEqual(v[0], 100.0)
        self.assertAlmostEqual(v[1], 120.0)
        c = m @ np.array([100.0, 100.0, 1.0])
        self.assertAlmostEqual(c[0], 100.0)
        self.assertAlmostEqual(c[1], 100.0)

    def test_fits_round_trip_and_bottom_up_storage(self):
        image = np.zeros((5, 7), dtype=np.uint16)
        image[0, 1] = 40000    # top row in TOP-DOWN terms
        image[4, 6] = 7
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.fit"
            sf.write_fits_rgb16(path, image, comments=["unit test"])
            raw = path.read_bytes()
            self.assertEqual(len(raw) % 2880, 0)
            self.assertTrue(raw.startswith(b"SIMPLE  =                    T"))
            self.assertIn(b"ROWORDER= 'BOTTOM-UP'", raw)
            self.assertIn(b"NAXIS3  =                    3", raw)
            self.assertIn(b"BZERO   =                32768", raw)
            header, back = sf.read_fits_minimal(path)
            np.testing.assert_array_equal(back, image)
            # first stored row (after the header) is the BOTTOM row of the TOP-DOWN image
            data = np.frombuffer(raw[2880:2880 + 5 * 7 * 2], dtype=">i2").astype(np.int32) + 32768
            np.testing.assert_array_equal(data.reshape(5, 7)[0], image[4])
            self.assertEqual(data.reshape(5, 7)[4, 1], 40000)
        with self.assertRaises(ValueError):
            sf.write_fits_rgb16(Path(d) / "x.fit", image.astype(np.float32))

    def test_generate_dataset_writes_frames_and_truth(self):
        with tempfile.TemporaryDirectory() as d:
            truth = sf.generate_dataset(Path(d) / "stage")
            self.assertEqual([f["filename"] for f in truth["frames"]], [f"syn_0000{k}.fit" for k in range(1, 5)])
            self.assertEqual(truth["coordinate_convention"]["row_order"], "TOP-DOWN")
            for frame in truth["frames"]:
                path = Path(d) / "stage" / frame["filename"]
                self.assertTrue(path.exists())
                # header block + 1200*901*3*2 bytes of data padded to 2880-byte blocks
                self.assertEqual(path.stat().st_size, 2880 + 2253 * 2880)
            np.testing.assert_allclose(truth["frames"][0]["frame_to_reference"], np.eye(3))
            saved = json.loads((Path(d) / "stage" / "truth.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["frames"][1]["description"], truth["frames"][1]["description"])
            with self.assertRaises(FileExistsError):
                sf.generate_dataset(Path(d) / "stage")


class ProofAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.truth = sf.generate_dataset(Path(cls.tmp.name) / "stage")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _simulate(self, name, **convention):
        work = Path(self.tmp.name) / name
        cp.simulate_siril_outputs(self.truth, work, **convention)
        return work

    def test_all_sixteen_conventions_are_recovered(self):
        for n, (row_order, offset, direction) in enumerate(itertools.product(("TOP-DOWN", "BOTTOM-UP"), (0.0, 0.5, -0.5, 1.0),
                                                                             ("frame_to_reference", "reference_to_frame"))):
            work = self._simulate(f"sim{n}", row_order=row_order, x_offset=offset, y_offset=offset, matrix_direction=direction)
            proof = cp.analyze_run(self.truth, work / "light_.seq", work / "cache")
            self.assertEqual(proof["status"], "proved", (row_order, offset, direction, proof.get("reason")))
            self.assertEqual(proof["row_order"], row_order)
            self.assertEqual(proof["x_offset"], offset)
            self.assertEqual(proof["y_offset"], offset)
            self.assertEqual(proof["matrix_direction"], direction)
            self.assertTrue(proof["simulated"], "the SIMULATED marker must force the flag")
            wrong = "BOTTOM-UP" if row_order == "TOP-DOWN" else "TOP-DOWN"
            self.assertLessEqual(proof["evidence"]["row_order_candidates"][wrong]["matched_fraction_of_truth"], 0.2)

    def test_simulated_proof_never_verifies_a_hypothesis(self):
        work = self._simulate("sim_hyp", row_order="BOTTOM-UP", x_offset=0.0, y_offset=0.0, matrix_direction="frame_to_reference")
        proof = cp.analyze_run(self.truth, work / "light_.seq", work / "cache")
        hyp = cp.hypothesis_from_proof(proof)
        self.assertFalse(hyp.verified)
        self.assertEqual(hyp.row_order, "BOTTOM-UP")
        self.assertIn("SIMULATED", hyp.basis)
        with self.assertRaises(ValueError):
            cp.hypothesis_from_proof({"status": "inconclusive"})

    def test_registered_sequence_and_wrong_reference_are_rejected(self):
        work = self._simulate("sim_rej", row_order="TOP-DOWN", x_offset=0.0, y_offset=0.0, matrix_direction="frame_to_reference")
        seq = work / "light_.seq"
        text = seq.read_text(encoding="utf-8")
        identity = "\n".join(line if not line.startswith("R1") else "R1 3.5 3.5 0.8 0 0.02 10 H 1 0 0 0 1 0 0 0 1"
                             for line in text.splitlines()) + "\n"
        (work / "r_light_.seq").write_text(identity, encoding="utf-8")
        with self.assertRaises(ValueError):
            cp.analyze_run(self.truth, work / "r_light_.seq", work / "cache")
        (work / "ref2.seq").write_text(text.replace("S 'light_' 1 4 4 5 0 6", "S 'light_' 1 4 4 5 1 6"), encoding="utf-8")
        with self.assertRaises(ValueError):
            cp.analyze_run(self.truth, work / "ref2.seq", work / "cache")

    def test_inconsistent_direction_is_inconclusive(self):
        work = self._simulate("sim_mix", row_order="TOP-DOWN", x_offset=0.0, y_offset=0.0, matrix_direction="frame_to_reference")
        seq = work / "light_.seq"
        lines = seq.read_text(encoding="utf-8").splitlines()
        r_lines = [i for i, line in enumerate(lines) if line.startswith("R1")]
        # invert the last frame's matrix only: direction votes disagree
        tokens = lines[r_lines[-1]].split()
        h = np.array([float(t) for t in tokens[8:17]]).reshape(3, 3)
        inv = np.linalg.inv(h)
        inv = inv / inv[2, 2]
        lines[r_lines[-1]] = " ".join(tokens[:8] + [f"{v:g}" for v in inv.reshape(-1)])
        (work / "mixed.seq").write_text("\n".join(lines) + "\n", encoding="utf-8")
        proof = cp.analyze_run(self.truth, work / "mixed.seq", work / "cache")
        self.assertEqual(proof["status"], "inconclusive")
        self.assertIn("direction", proof["reason"])

    def test_swapped_star_lists_fail_binding(self):
        work = self._simulate("sim_swap", row_order="TOP-DOWN", x_offset=0.0, y_offset=0.0, matrix_direction="frame_to_reference")
        cache = work / "cache"
        a, b = (cache / "light_00002.lst"), (cache / "light_00003.lst")
        ta, tb = a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8")
        a.write_text(tb, encoding="utf-8")
        b.write_text(ta, encoding="utf-8")
        proof = cp.analyze_run(self.truth, work / "light_.seq", cache)
        self.assertEqual(proof["status"], "inconclusive")
        self.assertFalse(proof["evidence"]["frame_binding"]["light_00002"]["binding_ok"])
        self.assertEqual(proof["evidence"]["frame_binding"]["light_00002"]["best_truth_frame"], 3)


class EntryScripts(unittest.TestCase):
    """The two S2 scripts must run from a clean interpreter (as they will on Windows)."""

    def _run(self, *args):
        return subprocess.run([sys.executable, "-B", *args], cwd=str(ROOT), capture_output=True, text=True)

    def test_simulate_then_real_diagnostics_on_simulated_products(self):
        with tempfile.TemporaryDirectory() as d:
            proc = self._run(str(ROOT / "scripts" / "s2_siril_coordinate_proof.py"), "simulate", "--row-order", "BOTTOM-UP",
                             "--offset", "0.5", "--direction", "frame_to_reference", "--output-root", d)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            runs = sorted(Path(d).glob("coordinate-proof-SIMULATED-*"))
            self.assertEqual(len(runs), 1)
            proof = json.loads((runs[0] / "coordinate_proof.json").read_text(encoding="utf-8"))
            self.assertTrue(proof["simulated"])
            self.assertEqual(proof["status"], "proved")
            self.assertIn("SIMULATED", (runs[0] / "PROOF.md").read_text(encoding="utf-8"))
            # analyze without --simulated still reports simulated (marker in the seq file)
            proc = self._run(str(ROOT / "scripts" / "s2_siril_coordinate_proof.py"), "analyze", str(runs[0]))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(json.loads((runs[0] / "coordinate_proof.json").read_text(encoding="utf-8"))["simulated"])
            # a simulated proof must not be accepted as a verified hypothesis by the real-sequence script
            work = runs[0] / "job" / "work"
            proc = self._run(str(ROOT / "scripts" / "s2_real_sequence_diagnostics.py"), "--seq", str(work / "light_.seq"),
                             "--lst-dir", str(work / "cache"), "--frames", "2", "3", "--width", "1200", "--height", "901",
                             "--proof", str(runs[0] / "coordinate_proof.json"), "--output-root", str(Path(d) / "diag"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("SIMULATED", proc.stderr)
            # with an explicit unverified hypothesis it runs and reports the simulated geometry exactly
            proc = self._run(str(ROOT / "scripts" / "s2_real_sequence_diagnostics.py"), "--seq", str(work / "light_.seq"),
                             "--lst-dir", str(work / "cache"), "--frames", "2", "3", "4", "--width", "1200", "--height", "901",
                             "--hypothesis", "siril_native_as_bottomup_unverified", "--label", "t",
                             "--output-root", str(Path(d) / "diag"))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = sorted((Path(d) / "diag").glob("real-diagnostics-t-*"))
            self.assertEqual(len(out), 1)
            summary = json.loads((out[0] / "association_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["hypothesis"]["verified"])
            for stem in ("light_00002", "light_00003", "light_00004"):
                pooled = summary["summaries"][stem]["pooled"]["all_roles_pooled"]
                self.assertLess(pooled["overall"]["rms_px"], 0.1)     # only centroid noise + 0.01 rounding
                self.assertGreater(summary["summaries"][stem]["association"]["matched_fraction_of_frame"], 0.95)
            report = json.loads((out[0] / "diagnostics_report.json").read_text(encoding="utf-8"))
            self.assertEqual([f["state"] for f in report["frames"]], ["succeeded"] * 3)
            self.assertTrue((out[0] / "SUMMARY.md").exists())
            # the registered (all-identity) sequence is refused
            seq_text = (work / "light_.seq").read_text(encoding="utf-8")
            (work / "r_light_.seq").write_text("\n".join(l if not l.startswith("R1") else "R1 1 1 1 0 0 1 H 1 0 0 0 1 0 0 0 1"
                                                         for l in seq_text.splitlines()) + "\n", encoding="utf-8")
            proc = self._run(str(ROOT / "scripts" / "s2_real_sequence_diagnostics.py"), "--seq", str(work / "r_light_.seq"),
                             "--lst-dir", str(work / "cache"), "--width", "1200", "--height", "901", "--output-root", str(Path(d) / "diag"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("identity", proc.stderr)

    def test_run_directory_is_absolute_for_relative_output_root(self):
        # Siril is given the script path on its command line and requires it to be absolute.
        import importlib.util
        spec = importlib.util.spec_from_file_location("s2proof", ROOT / "scripts" / "s2_siril_coordinate_proof.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            old_cwd = os.getcwd()
            os.chdir(d)
            try:
                run = module._new_run_dir(Path("relative-root") / "x", "unit")
            finally:
                os.chdir(old_cwd)
            self.assertTrue(run.is_absolute())
            self.assertTrue(str(run).startswith(str(Path(d).resolve())))
            self.assertTrue(run.is_dir())

    def test_generate_subcommand(self):
        with tempfile.TemporaryDirectory() as d:
            proc = self._run(str(ROOT / "scripts" / "s2_siril_coordinate_proof.py"), "generate", "--output-root", d)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            runs = sorted(Path(d).glob("coordinate-proof-generate-*"))
            self.assertEqual(len(runs), 1)
            self.assertEqual(len(list((runs[0] / "job" / "stage").glob("syn_*.fit"))), 4)
            self.assertTrue((runs[0] / "generate.json").exists())


if __name__ == "__main__":
    unittest.main()
