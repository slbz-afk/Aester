"""Independent selection-only contracts using explicit Siril sequence bytes."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starstack import sequence_selection as selection
from starstack.siril_sequence import SequenceError, read_seq


def sequence_bytes(*, selected=(1, 1, 1, 1, 1), header_count=None, ending=b"\r\n"):
    count = sum(selected) if header_count is None else header_count
    lines = [b"# sequence selection fixture", f"  S 'light_' 1 {len(selected)} {count} 5 0 6 0 0 0".encode(), b"L 3"]
    for index, flag in enumerate(selected, 1):
        lines.append(f" I\t{index}  {flag}\t1024 768  ".encode())
    for layer in (0, 1):
        for index in range(len(selected)):
            # Intentionally retain more digits and formatting than a float
            # round trip through the existing registration writer preserves.
            lines.append(f" R{layer}\t3.2 4 0.7 0 0.003 82 H 1.000000000000000222 0 {index}.1234567890123456789 -0.0 1 0 1.2e-013 0 1  ".encode())
    lines.extend([b"D1 2 distortion.wcs", b"M1-0 42 17 -999999", b"X unknown extension preserved"])
    return ending.join(lines)  # deliberately no terminal newline


class SequenceSelection(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "light_.seq"

    def write(self, **kwargs):
        original = sequence_bytes(**kwargs)
        self.path.write_bytes(original)
        return original

    def assert_rejected_unchanged(self, exclusions, *, source=None, minimum=3):
        original = self.path.read_bytes() if source is None else source
        with self.assertRaises(ValueError):
            selection.apply_exclusions(self.path, exclusions, minimum=minimum)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_only_selection_tokens_change_including_matrix_bytes_and_line_endings(self):
        original = self.write()
        result = selection.apply_exclusions(self.path, ["light_00002", "light_00004"])
        expected = original.replace(b"1 5 5 5 0 6", b"1 5 3 5 0 6").replace(b"I\t2  1\t", b"I\t2  0\t").replace(b"I\t4  1\t", b"I\t4  0\t")
        self.assertEqual(self.path.read_bytes(), expected)
        self.assertEqual(result["selected"], ["light_00001", "light_00003", "light_00005"])
        self.assertEqual(result["newly_excluded"], ["light_00002", "light_00004"])
        self.assertEqual(read_seq(self.path)["header"]["nb_selected"], 3)
        self.assertTrue(result["changed"])

    def test_never_reselects_an_already_deselected_frame(self):
        self.write(selected=(1, 0, 1, 1, 1))
        result = selection.apply_exclusions(self.path, ["light_00002", "light_00004"])
        self.assertEqual(result["selected"], ["light_00001", "light_00003", "light_00005"])
        self.assertEqual(result["previously_deselected"], ["light_00002"])
        self.assertEqual(result["newly_excluded"], ["light_00004"])

    def test_duplicate_exclusions_and_retry_are_idempotent(self):
        self.write()
        first = selection.apply_exclusions(self.path, ["light_00004", "light_00004"])
        content = self.path.read_bytes()
        with mock.patch.object(selection.os, "replace") as replace:
            again = selection.apply_exclusions(self.path, ["light_00004"])
        replace.assert_not_called()
        self.assertEqual(self.path.read_bytes(), content)
        self.assertEqual(first["sha256_after"], again["sha256_after"])
        self.assertFalse(again["changed"])
        self.assertEqual(again["newly_excluded"], [])

    def test_empty_exclusions_leave_consistent_file_identical(self):
        original = self.write(ending=b"\n")
        result = selection.apply_exclusions(self.path, [])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(result["changed"])

    def test_stale_header_count_is_repaired_and_reported(self):
        original = self.write(selected=(1, 0, 1, 1, 1), header_count=5)
        result = selection.apply_exclusions(self.path, [])
        self.assertEqual(self.path.read_bytes(), original.replace(b"1 5 5 5 0 6", b"1 5 4 5 0 6"))
        self.assertTrue(result["header_count_repaired"])
        self.assertEqual(result["header_selected_before"], 5)

    def test_unknown_name_does_not_apply_other_valid_exclusions(self):
        self.write()
        self.assert_rejected_unchanged(["light_00002", "light_99999"])

    def test_fewer_than_three_preserves_entire_file(self):
        self.write()
        self.assert_rejected_unchanged(["light_00001", "light_00002", "light_00003"])

    def test_minimum_argument_requires_positive_python_integer(self):
        self.write()
        for minimum in (True, 0, -1, 1.0, "3", None):
            with self.subTest(minimum=minimum):
                self.assert_rejected_unchanged([], minimum=minimum)

    def test_invalid_exclusion_arguments_preserve_file(self):
        self.write()
        for excluded in (None, "light_00002", [None], [False], [""]):
            with self.subTest(excluded=excluded):
                self.assert_rejected_unchanged(excluded)

    def test_excluded_reference_keeps_coordinate_reference_position(self):
        self.write()
        selection.apply_exclusions(self.path, ["light_00001"])
        parsed = read_seq(self.path)
        self.assertEqual(parsed["header"]["reference_image"], 0)
        self.assertFalse(parsed["images"][0]["selected"])

    def test_missing_registration_rejected_before_write(self):
        content = self.write()
        self.path.write_bytes(b"\r\n".join(line for line in content.split(b"\r\n") if not line.lstrip().startswith(b"R1")))
        self.assert_rejected_unchanged([])

    def test_zero_homography_must_be_excluded_before_selection_can_pass(self):
        content = self.write()
        lines = content.split(b"\r\n")
        index = next(i for i, line in enumerate(lines) if line.lstrip().startswith(b"R1"))
        lines[index] = b"R1 0 0 0 0 0 0 H 0 0 0 0 0 0 0 0 0"
        self.path.write_bytes(b"\r\n".join(lines))
        self.assert_rejected_unchanged([])
        result = selection.apply_exclusions(self.path, ["light_00001"])
        self.assertEqual(result["selected_count"], 4)

    def test_duplicate_indices_and_multiple_headers_are_rejected(self):
        for change in (lambda b: b.replace(b"I\t2  ", b"I\t1  "), lambda b: b + b"\r\nS 'light_' 1 5 5 5 0 6 0 0 0"):
            with self.subTest(change=change):
                self.path.write_bytes(change(sequence_bytes()))
                self.assert_rejected_unchanged([])

    def test_fractional_indices_and_nonbinary_selection_are_rejected(self):
        for bad in (b"I\t2.5  1\t", b"I\t2  2\t", b"I\t2  0.5\t"):
            with self.subTest(bad=bad):
                self.path.write_bytes(sequence_bytes().replace(b"I\t2  1\t", bad))
                self.assert_rejected_unchanged([])

    def test_replace_failure_leaves_original_and_cleans_temporary(self):
        original = self.write()
        with mock.patch.object(selection.os, "replace", side_effect=PermissionError("injected sharing failure")):
            with self.assertRaises(PermissionError):
                selection.apply_exclusions(self.path, ["light_00002"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
