"""Tests for starstack.siril_sequence (pure parsing, hand-written inputs).

Run:  python -B -m unittest tests.test_siril_sequence   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starstack import siril_sequence as ss  # noqa: E402

SEQ_TEXT = """#Siril sequence file. Contains list of images, selection, and registration data
#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle
S 'light_' 1 3 3 5 0 6 0 0 0
L 3
I 1 1
I 2 1
I 3 0
R0 4.6 4.6 0.68 0 0.02 2000 H 1 0 0 0 1 0 0 0 1
R0 3.6 7.2 0.75 0 0.02 2000 H 1 0 0 0 1 0 0 0 1
R0 3.6 7.2 0.75 0 0.02 2000 H 1 0 0 0 1 0 0 0 1
R1 4.67145 4.67145 0.679823 0 0.023258 2000 H 1 0 0 0 1 0 0 0 1
R1 3.62519 7.28301 0.74794 0 0.0227705 2000 H 1.00095 0.000184871 -2.75443 0.000524852 1.001 -5.95075 7.83879e-008 1.27785e-007 1
R1 3.63285 7.26932 0.751688 0 0.0227463 2000 H 1.00208 0.000405334 -6.97063 0.0011521 1.00217 -13.1966 1.64169e-007 2.74367e-007 1
M1-0 45705600 45705600 1349.72 1304 339.98 -999999 -999999 -999999 -999999 -999999 7 22746 65535 66.4324
X some future line type
"""

LST_TEXT = (
    "# 3 stars found using the following parameters:\n"
    "# sigma=1.00 roundness=0.50 radius=5 relax=0 profile=0 minbeta=1.5 max_stars=2000 layer=1 minA=0.00 maxA=0.00 maxR=1.00\n"
    "# star#\tlayer\tB\tA\tbeta\tX\tY\tFWHMx [px]\tFWHMy [px]\tFWHMx [\"]\tFWHMy [\"]\tangle\tRMSE\tmag\tSat\tProfile\tRA\tDec\n"
    "1\t1\t1744.799521\t47090.921156\t     -1.00\t   2756.69\t   2990.95\t      6.79\t      6.35\t    304.83\t    284.95\t32.50\t4.035e+002\t    -15.52\t1\tGaussian\t9990000000.000000\t9990000000.000000\n"
    "2\t1\t  0.025360\t  0.294767\t     -1.00\t   5615.94\t   2848.44\t      8.63\t      8.27\t    387.16\t    371.21\t46.36\t1.269e-002\t     -3.44\t0\tGaussian\t9990000000.000000\t9990000000.000000\n"
    "3\t1\t1.0\t2.0\t-1.00\t10.25\t20.75\t3\t3\t0\t0\t0\t1e-3\t-1\t0\tMoffat\t0\t0\n"
)


class SeqParsing(unittest.TestCase):
    def test_header_images_and_regdata(self):
        seq = ss.parse_seq_text(SEQ_TEXT)
        self.assertEqual(seq["header"]["name"], "light_")
        self.assertEqual(seq["header"]["nb_images"], 3)
        self.assertEqual(seq["header"]["reference_image"], 0)
        self.assertEqual(seq["header"]["version"], 6)
        self.assertEqual(seq["layers"], 3)
        self.assertEqual([i["stem"] for i in seq["images"]], ["light_00001", "light_00002", "light_00003"])
        self.assertEqual([i["selected"] for i in seq["images"]], [True, True, False])
        self.assertEqual([i["is_reference"] for i in seq["images"]], [True, False, False])
        self.assertEqual(sorted(seq["regdata"]), [0, 1])
        self.assertEqual(seq["all_identity_per_layer"], {0: True, 1: False})
        r = seq["regdata"][1][1]
        self.assertEqual(r["number_of_stars"], 2000)
        self.assertAlmostEqual(r["fwhm"], 3.62519)
        self.assertAlmostEqual(r["H"][0][2], -2.75443)
        self.assertAlmostEqual(r["H"][2][1], 1.27785e-007)
        self.assertEqual(r["H"][2][2], 1.0)
        self.assertEqual(seq["regdata"][1][0]["H"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(len(seq["stats"]), 1)
        self.assertEqual(seq["stats"][0]["layer"], 1)
        self.assertEqual(seq["unparsed_lines"], ["X some future line type"])

    def test_image_stem_zero_padding(self):
        self.assertEqual(ss.image_stem("light_", 16, 5), "light_00016")
        self.assertEqual(ss.image_stem("syn_", 1, 5), "syn_00001")

    def test_rejects_image_count_mismatch(self):
        text = SEQ_TEXT.replace("I 3 0\n", "")
        with self.assertRaises(ss.SequenceError):
            ss.parse_seq_text(text)

    def test_rejects_regdata_count_mismatch(self):
        text = SEQ_TEXT.replace("R1 3.63285 7.26932 0.751688 0 0.0227463 2000 H 1.00208 0.000405334 -6.97063 0.0011521 1.00217 -13.1966 1.64169e-007 2.74367e-007 1\n", "")
        with self.assertRaises(ss.SequenceError):
            ss.parse_seq_text(text)

    def test_rejects_reference_outside_list(self):
        text = SEQ_TEXT.replace("S 'light_' 1 3 3 5 0 6 0 0 0", "S 'light_' 1 3 3 5 7 6 0 0 0")
        with self.assertRaises(ss.SequenceError):
            ss.parse_seq_text(text)

    def test_rejects_missing_s_line_and_bad_r_line(self):
        with self.assertRaises(ss.SequenceError):
            ss.parse_seq_text("L 3\nI 1 1\n")
        with self.assertRaises(ss.SequenceError):
            ss.parse_seq_text(SEQ_TEXT.replace("2000 H 1 0 0 0 1 0 0 0 1\nR1 4.67", "2000 Q 1 0 0 0 1 0 0 0 1\nR1 4.67", 1))
        with self.assertRaises(ss.SequenceError):
            ss.parse_seq_text(SEQ_TEXT.replace("H 1 0 0 0 1 0 0 0 1\nR1 4.67", "H 1 0 0 0 nan 0 0 0 1\nR1 4.67", 1))

    def test_read_seq_attaches_source_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light_.seq"
            path.write_text(SEQ_TEXT, encoding="utf-8")
            seq = ss.read_seq(path)
            self.assertEqual(seq["source"]["size_bytes"], os.path.getsize(path))
            self.assertEqual(len(seq["source"]["sha256"]), 64)


class RegistrationWriter(unittest.TestCase):
    def test_rewrites_only_the_layer_homographies_selection_and_d_line(self):
        h2 = [[1.0005, 0.0001, -3.25], [-0.0002, 1.0004, 7.5], [1e-7, -2e-7, 1.0]]
        text = ss.write_registration_text(SEQ_TEXT, {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1: h2}, layer=1, distortion_file="distortion.wcs")
        parsed = ss.parse_seq_text(text)
        self.assertEqual(parsed["header"]["nb_selected"], 2)
        self.assertEqual([im["selected"] for im in parsed["images"]], [True, True, False])
        self.assertEqual(parsed["regdata"][1][1]["H"], h2)
        self.assertEqual(parsed["regdata"][1][2]["H"], [[0.0] * 3] * 3)                       # deselected: Siril's zero form
        self.assertEqual(parsed["regdata"][1][2]["number_of_stars"], 0)
        self.assertEqual(parsed["regdata"][1][1]["fwhm"], 3.62519)                            # quality fields kept
        self.assertEqual(parsed["regdata"][0], ss.parse_seq_text(SEQ_TEXT)["regdata"][0])     # other layers untouched
        lines = text.splitlines()
        self.assertIn("D1 2 distortion.wcs", lines)
        self.assertLess(lines.index("D1 2 distortion.wcs"), lines.index([l for l in lines if l.startswith("R0")][0]))
        self.assertIn("X some future line type", lines)
        self.assertIn("M1-0 45705600 45705600 1349.72 1304 339.98 -999999 -999999 -999999 -999999 -999999 7 22746 65535 66.4324", lines)
        self.assertEqual(text.count("D1 "), 1)
        # rewriting again replaces the D line instead of adding one, and can re-select frame 3 with quality values
        again = ss.write_registration_text(text, {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1: h2, 2: h2}, layer=1, distortion_file="other.wcs",
                                           quality={2: [3.1, 3.2, 0.7, 0, 0.02, 1500]}, reference_position=1)
        self.assertEqual(again.count("D1 "), 1)
        self.assertIn("D1 2 other.wcs", again.splitlines())
        parsed2 = ss.parse_seq_text(again)
        self.assertEqual(parsed2["header"]["nb_selected"], 3)
        self.assertEqual(parsed2["header"]["reference_image"], 1)
        self.assertEqual(parsed2["regdata"][1][2]["number_of_stars"], 1500)
        self.assertEqual(parsed2["regdata"][1][2]["fwhm"], 3.1)

    def test_rejects_selected_frame_without_matrix_and_bad_matrices(self):
        with self.assertRaises(ss.SequenceError):
            ss.write_registration_text(SEQ_TEXT, {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, selected={1: True})
        with self.assertRaises(ss.SequenceError):
            ss.write_registration_text(SEQ_TEXT, {0: [[1, 0], [0, 1]]})
        with self.assertRaises(ss.SequenceError):
            ss.write_registration_text(SEQ_TEXT, {5: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        with self.assertRaises(ss.SequenceError):
            ss.write_registration_text(SEQ_TEXT, {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, layer=2)   # no R2 lines

    def test_write_registration_replaces_file_and_returns_parse(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light_.seq"
            path.write_text(SEQ_TEXT, encoding="utf-8")
            parsed = ss.write_registration(path, {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1: [[1, 0, 2.5], [0, 1, -1.5], [0, 0, 1]]}, distortion_file="distortion.wcs")
            self.assertEqual(parsed["regdata"][1][1]["H"][0][2], 2.5)
            self.assertEqual(parsed["source"]["path"], str(path))
            self.assertFalse((Path(d) / "light_.seq.tmp").exists())
            self.assertIn("D1 2 distortion.wcs", path.read_text(encoding="utf-8"))


class LstParsing(unittest.TestCase):
    def test_columns_values_and_types(self):
        lst = ss.parse_lst_text(LST_TEXT)
        self.assertEqual(lst["declared_count"], 3)
        self.assertEqual(lst["parameters"]["max_stars"], "2000")
        self.assertEqual(lst["parameters"]["layer"], "1")
        self.assertEqual(len(lst["stars"]), 3)
        first = lst["stars"][0]
        self.assertEqual(first["index"], 1)
        self.assertEqual(first["layer"], 1)
        self.assertEqual(first["x"], 2756.69)
        self.assertEqual(first["y"], 2990.95)
        self.assertEqual(first["sat"], 1)
        self.assertEqual(first["profile"], "Gaussian")
        self.assertAlmostEqual(first["fwhmx_px"], 6.79)
        self.assertAlmostEqual(first["rmse"], 403.5)
        self.assertEqual(lst["stars"][2]["profile"], "Moffat")
        self.assertEqual(lst["stars"][2]["x"], 10.25)

    def test_declared_count_mismatch_is_an_error(self):
        with self.assertRaises(ss.SequenceError):
            ss.parse_lst_text(LST_TEXT.replace("# 3 stars", "# 4 stars"))

    def test_column_count_mismatch_is_an_error(self):
        broken = LST_TEXT + "4\t1\t1.0\n"
        with self.assertRaises(ss.SequenceError):
            ss.parse_lst_text(broken)

    def test_missing_header_is_an_error(self):
        with self.assertRaises(ss.SequenceError):
            ss.parse_lst_text("1\t1\t1\t1\t1\t1\t1\n")
        with self.assertRaises(ss.SequenceError):
            ss.parse_lst_text("# 1 stars found\n# a\tb\n1\t2\n")

    def test_read_lst_attaches_source_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "light_00001.lst"
            path.write_text(LST_TEXT, encoding="utf-8")
            lst = ss.read_lst(path)
            self.assertEqual(lst["source"]["path"], str(path))
            self.assertEqual(len(lst["source"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
