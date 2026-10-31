"""Independent counterexamples for snapshot/src/starstack/float_demosaic.py.

Written before reading snapshot/tests. Expectations come from a scalar,
explicit-weight reference written here (weights (2-|dy|)*(2-|dx|) over the
in-bounds 3x3 window, same colour only, measured sites copied), never from the
function under test. Run with ``pytest`` or ``python -m unittest``.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Product integration: normal package import; original Fable snapshot is untouched.

from starstack.float_demosaic import demosaic_bilinear, rgb_pattern  # noqa: E402

# Four phases, written with the LibRaw/rawpy slot-3 second green where it
# matters, so that the 3 -> 1 mapping is exercised on every phase.
PATTERNS = {
    "RGGB": [[0, 1], [3, 2]],
    "GRBG": [[1, 0], [2, 3]],
    "GBRG": [[3, 2], [0, 1]],
    "BGGR": [[2, 3], [1, 0]],
}


def colour(pattern, y, x):
    c = int(pattern[y % 2][x % 2])
    return 1 if c == 3 else c


def reference(cfa, pattern):
    """Explicit-weight bilinear reference in float64, cast once to float32."""
    h, w = cfa.shape
    out = np.empty((h, w, 3), np.float32)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                if colour(pattern, y, x) == ch:
                    out[y, x, ch] = cfa[y, x]  # copy, keeps -0.0 and extremes
                    continue
                num = 0.0
                den = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        sy, sx = y + dy, x + dx
                        if 0 <= sy < h and 0 <= sx < w and colour(pattern, sy, sx) == ch:
                            wgt = (2 - abs(dy)) * (2 - abs(dx))
                            num += float(cfa[sy, sx]) * wgt
                            den += wgt
                assert den > 0, "every Bayer site has all colours in its clipped 3x3"
                out[y, x, ch] = np.float32(num / den)
    return out


def dyadic(rng, shape, lo=-40, hi=90, step=0.25):
    """Values that are exactly representable so both sides are exact."""
    return (rng.integers(lo / step, hi / step, size=shape) * step).astype(np.float32)


class InterpolationMatchesExplicitReference(unittest.TestCase):
    SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3), (5, 7), (9, 4), (12, 11)]

    def test_all_phases_shapes_and_tile_sizes_bit_exact(self):
        rng = np.random.default_rng(20260916)
        for name, pattern in PATTERNS.items():
            for shape in self.SHAPES:
                cfa = dyadic(rng, shape)
                expected = reference(cfa, pattern)
                for block_rows in (1, 2, 3, 128):
                    with self.subTest(pattern=name, shape=shape, block_rows=block_rows):
                        actual = demosaic_bilinear(cfa, pattern, block_rows=block_rows)
                        self.assertEqual(actual.dtype, np.float32)
                        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_random_floats_within_one_ulp(self):
        rng = np.random.default_rng(7)
        for name, pattern in PATTERNS.items():
            cfa = rng.normal(0, 3000, (17, 13)).astype(np.float32)
            expected = reference(cfa, pattern)
            actual = demosaic_bilinear(cfa, pattern, block_rows=5)
            with self.subTest(pattern=name):
                np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)

    def test_row_column_orientation_is_top_down_row_major(self):
        """An asymmetric phase (GRBG) distinguishes pattern[row%2][col%2] from its transpose."""
        pattern = PATTERNS["GRBG"]  # (0,1)=R, (1,0)=B
        cfa = np.zeros((4, 4), np.float32)
        cfa[0, 1] = 100.  # a red sample
        cfa[1, 0] = -7.   # a blue sample
        out = demosaic_bilinear(cfa, pattern)
        self.assertEqual(out[0, 1, 0], 100.)
        self.assertEqual(out[1, 0, 2], -7.)
        # Red site (0,1): blue neighbours in its window are (1,0)=-7 and (1,2)=0, weight 1 each.
        self.assertEqual(out[0, 1, 2], -3.5)
        # Red at the green corner (0,0): only the red at (0,1) is in its window.
        self.assertEqual(out[0, 0, 0], 100.)
        # Blue at (0,0): only the blue at (1,0) is in its window.
        self.assertEqual(out[0, 0, 2], -7.)
        # Blue at the green site (1,1): neighbours (1,0) and (1,2) weigh 2 each.
        self.assertEqual(out[1, 1, 2], -3.5)
        # A transposed reading of the pattern would have put red at (1,0).
        self.assertNotEqual(out[1, 0, 0], -7.)


class MeasuredSitesAndValueRanges(unittest.TestCase):
    def test_measured_values_are_copied_bitwise(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        big = np.finfo(np.float32).max
        values = np.array([-0., 0., -1.5, .75, big, -big, tiny, -tiny, 65535., -1008.], np.float32)
        cfa = np.resize(values, (7, 11)).astype(np.float32)
        for name, pattern in PATTERNS.items():
            out = demosaic_bilinear(cfa, pattern, block_rows=2)
            with self.subTest(pattern=name):
                self.assertTrue(np.isfinite(out).all())
                for y, x in np.ndindex(cfa.shape):
                    ch = colour(pattern, y, x)
                    self.assertEqual(out[y, x, ch].view(np.uint32), cfa[y, x].view(np.uint32))
                # sign of zero is a measured property, so it must survive
                zeros = [(y, x) for y, x in np.ndindex(cfa.shape) if cfa[y, x] == 0]
                self.assertTrue(any(np.signbit(cfa[y, x]) for y, x in zeros))
                for y, x in zeros:
                    self.assertEqual(bool(np.signbit(out[y, x, colour(pattern, y, x)])),
                                     bool(np.signbit(cfa[y, x])))

    def test_extreme_magnitudes_cannot_overflow_interpolation(self):
        big = np.finfo(np.float32).max
        cfa = np.full((6, 6), big, np.float32)
        cfa[::2, ::2] = -big
        out = demosaic_bilinear(cfa, PATTERNS["RGGB"])
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(out, reference(cfa, PATTERNS["RGGB"]))

    def test_fractional_black_residuals_and_overwhite_are_kept(self):
        # After a fractional black (e.g. 10.25) subtraction values are not integers,
        # can be negative, and can exceed the white level; none may be altered.
        codes = np.arange(6 * 8, dtype=np.float64).reshape(6, 8) * 137 % 1100
        black = np.array([[10.25, 9.75], [10.75, 10.]])
        cfa = (codes - np.tile(black, (3, 4))).astype(np.float32)
        self.assertTrue((cfa < 0).any() and (cfa > 1023).any())
        for name, pattern in PATTERNS.items():
            out = demosaic_bilinear(cfa, pattern, block_rows=4)
            with self.subTest(pattern=name):
                np.testing.assert_array_equal(out.view(np.uint32), reference(cfa, pattern).view(np.uint32))
                self.assertLess(out.min(), 0)
                self.assertGreater(out.max(), 1023)


class LocalityAndLinearity(unittest.TestCase):
    def test_single_pixel_change_affects_only_its_3x3_neighbourhood(self):
        rng = np.random.default_rng(3)
        cfa = dyadic(rng, (11, 13))
        base = demosaic_bilinear(cfa, PATTERNS["GBRG"])
        for (y, x) in ((0, 0), (5, 6), (10, 12), (3, 0)):
            changed = cfa.copy()
            changed[y, x] += 1000.
            out = demosaic_bilinear(changed, PATTERNS["GBRG"], block_rows=4)
            diff = np.any(out != base, axis=2)
            ys, xs = np.nonzero(diff)
            with self.subTest(site=(y, x)):
                self.assertTrue(diff[y, x])
                self.assertTrue((abs(ys - y) <= 1).all() and (abs(xs - x) <= 1).all())

    def test_linearity_without_global_normalisation(self):
        rng = np.random.default_rng(11)
        a, b = dyadic(rng, (9, 10)), dyadic(rng, (9, 10))
        pattern = PATTERNS["BGGR"]
        da, db = demosaic_bilinear(a, pattern), demosaic_bilinear(b, pattern)
        combo = demosaic_bilinear((3 * a - .25 * b).astype(np.float32), pattern)
        np.testing.assert_allclose(combo, 3 * da - .25 * db, rtol=2e-7, atol=1e-5)
        # Adding a constant (e.g. a different black) adds the same constant everywhere.
        shifted = demosaic_bilinear((a + 512.).astype(np.float32), pattern)
        np.testing.assert_allclose(shifted, da + 512., rtol=2e-7, atol=1e-4)


class PatternAndInputContracts(unittest.TestCase):
    def test_slot_three_maps_to_green_and_is_not_confused_with_blue(self):
        for name, pattern in PATTERNS.items():
            with self.subTest(pattern=name):
                p = rgb_pattern(pattern)
                self.assertEqual(sorted(p.ravel().tolist()), [0, 1, 1, 2])
                self.assertNotIn(3, p.ravel().tolist())

    def test_rejected_patterns(self):
        for bad in ([[0, 1], [2, 3]], [[0, 0], [1, 2]], [[1, 1], [1, 1]], [[0, 2], [3, 1]],
                    [[0, 1, 2]], [[0, 1], [1, 4]], [[0, 1], [1, -1]], [[0., 1.], [1., 2.]], "RGGB"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rgb_pattern(bad)

    def test_rejected_inputs(self):
        pattern = PATTERNS["RGGB"]
        for bad in (np.zeros((4, 4), np.float64), np.zeros((4, 4), np.uint16),
                    np.zeros((1, 4), np.float32), np.zeros((4, 1), np.float32),
                    np.zeros((4, 4, 1), np.float32), [[0., 1.], [1., 2.]]):
            with self.subTest(bad=type(bad)):
                with self.assertRaises(ValueError):
                    demosaic_bilinear(bad, pattern)

    def test_nonfinite_anywhere_is_rejected_and_input_untouched(self):
        pattern = PATTERNS["RGGB"]
        for value in (np.nan, np.inf, -np.inf):
            for site in ((0, 0), (4, 3), (8, 9)):
                cfa = np.ones((9, 10), np.float32)
                cfa[site] = value
                before = cfa.view(np.uint32).copy()
                with self.subTest(value=value, site=site):
                    with self.assertRaises(ValueError):
                        demosaic_bilinear(cfa, pattern, block_rows=2)
                    np.testing.assert_array_equal(cfa.view(np.uint32), before)


class OutputAliasingAndBackingFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="fable_review_demosaic_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_in_memory_alias_rejected_before_any_write(self):
        out = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        before = out.copy()
        with self.assertRaises(ValueError):
            demosaic_bilinear(out[:, :, 1], PATTERNS["RGGB"], out=out)
        np.testing.assert_array_equal(out, before)

    def test_two_mappings_of_one_file_rejected(self):
        path = self.tmp / "same.bin"
        np.arange(48, dtype=np.float32).tofile(path)
        before = path.read_bytes()
        source = np.memmap(path, np.float32, "r", shape=(4, 4))
        out = np.memmap(path, np.float32, "r+", shape=(4, 4, 3))
        try:
            with self.assertRaises(ValueError):
                demosaic_bilinear(source, PATTERNS["RGGB"], out=out)
            self.assertEqual(path.read_bytes(), before)
        finally:
            del source, out

    def test_hard_link_variant_of_the_input_file_rejected(self):
        path = self.tmp / "input.bin"
        np.arange(48, dtype=np.float32).tofile(path)
        link = self.tmp / "alias.bin"
        try:
            os.link(path, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"hard links unavailable here: {exc}")
        before = path.read_bytes()
        source = np.memmap(path, np.float32, "r", shape=(4, 4))
        out = np.memmap(link, np.float32, "r+", shape=(4, 4, 3))
        try:
            with self.assertRaises(ValueError):
                demosaic_bilinear(source, PATTERNS["RGGB"], out=out)
            self.assertEqual(path.read_bytes(), before)
        finally:
            del source, out

    def test_symlinked_path_variant_of_the_input_file_rejected(self):
        path = self.tmp / "input.bin"
        np.arange(48, dtype=np.float32).tofile(path)
        link = self.tmp / "link.bin"
        try:
            os.symlink(path, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable here: {exc}")
        source = np.memmap(path, np.float32, "r", shape=(4, 4))
        out = np.memmap(link, np.float32, "r+", shape=(4, 4, 3))
        try:
            with self.assertRaises(ValueError):
                demosaic_bilinear(source, PATTERNS["RGGB"], out=out)
        finally:
            del source, out

    def test_views_of_one_mapping_are_rejected_by_the_backing_file_rule(self):
        # Interleaved views do not overlap byte-wise (shares_memory is False), but
        # they live in the same file: the conservative same-file rule must refuse.
        path = self.tmp / "big.bin"
        np.zeros(4 * 4 * 4, np.float32).tofile(path)
        whole = np.memmap(path, np.float32, "r+", shape=(4, 4, 4))
        source = whole[:, :, 3]
        out = whole[:, :, :3]
        try:
            self.assertFalse(np.shares_memory(source, out))
            with self.assertRaises(ValueError):
                demosaic_bilinear(source, PATTERNS["RGGB"], out=out)
        finally:
            del source, out, whole

    def test_independent_memmap_output_receives_reference_values(self):
        source_path, out_path = self.tmp / "cfa.npy", self.tmp / "rgb.npy"
        rng = np.random.default_rng(5)
        cfa = dyadic(rng, (7, 9))
        np.save(source_path, cfa)
        source = np.load(source_path, mmap_mode="r")
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(7, 9, 3))
        try:
            returned = demosaic_bilinear(source, PATTERNS["GRBG"], out=out, block_rows=3)
            self.assertIs(returned, out)
            out.flush()
        finally:
            del returned, out, source
        np.testing.assert_array_equal(np.load(out_path).view(np.uint32),
                                      reference(cfa, PATTERNS["GRBG"]).view(np.uint32))
        np.testing.assert_array_equal(np.load(source_path).view(np.uint32), cfa.view(np.uint32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
