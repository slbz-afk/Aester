"""P1a G-1: ``resample.warp_planes_fused`` reproduces the per-plane ``warp_planes`` / ``warp_image`` bit for bit.

``warp_planes`` and ``warp_image`` are unchanged and serve as the independent oracle.  Small non-square cube (blocks of
32 and 16 leave partial edge blocks), NaN payloads, +-inf, a float32 subnormal and 3e38 in the data.
Run:  python -B -m unittest tests.test_resample_fused   (from WORKSPACE, PYTHONPATH=src)
"""
from __future__ import annotations

import itertools
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import registration_distortion as rd  # noqa: E402
from starstack import resample  # noqa: E402

FUSED = resample.warp_planes_fused


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint32) if a.dtype == np.float32 else a.view(np.uint64) if a.dtype == np.float64 else a


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("err", type(exc).__name__, str(exc))


H, W = 67, 131          # non-square; blocks of 32 and 16 leave partial blocks at the right and bottom edges
_RNG = np.random.default_rng(20260929)
CUBE = (_RNG.random((3, H, W)) * 2 - 0.3).astype(np.float32)
CUBE[1, 17, 33] = np.nan
CUBE[2, 0, 0] = np.uint32(0x7FC00123).view(np.float32)            # NaN payloads, on the border too
CUBE[0, H - 1, W - 1] = np.uint32(0xFFC00456).view(np.float32)
CUBE[0, 30, 60] = np.inf
CUBE[2, 31, 60] = -np.inf
CUBE[1, 40, 90] = np.float32(3e38)
CUBE[1, 41, 90] = np.float32(1e-45)                               # float32 subnormal
RADIAL = rd.RadialDistortion(-0.05, 0.02, 65.0, 33.0, 80.0, 0.96, "u_to_p")


def _nan_map(q):
    s = RADIAL.apply(q)
    s[::97] = np.nan
    s[5::131, 1] = np.inf
    return s


def _edge_map(q):
    s = q.astype(np.float64).copy()
    s[:, 0] = np.where(q[:, 0] % 3 == 0, -0.5, np.where(q[:, 0] % 3 == 1, W - 0.5, np.nextafter(W - 0.5, 0)))
    s[:, 1] = np.where(q[:, 1] % 2 == 0, np.nextafter(-0.5, -1), q[:, 1])
    return s


MAPS = {"radial": RADIAL.apply, "nan_inf": _nan_map, "edges": _edge_map,
        "shift_out": lambda q: q + np.array([37.3, -21.7]), "float32_mapping": lambda q: (q + 0.25).astype(np.float32)}
SHAPES = (None, (80, 150), (1, 1), (33, 3))


class FusedWarpTest(unittest.TestCase):
    def assertSameArray(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.flags.c_contiguous, b.flags.c_contiguous)
        self.assertTrue(np.array_equal(_bits(a), _bits(b)))

    def test_cubic_matches_per_plane_oracle(self):
        with np.errstate(all="ignore"):
            for (mname, m), fill, normalise, shape, block in itertools.product(
                    MAPS.items(), (np.nan, 0.0), (True, False), SHAPES, (32, 16)):
                with self.subTest(mapping=mname, fill=fill, normalise=normalise, shape=shape, block=block):
                    kw = dict(fill=fill, normalise=normalise, output_shape=shape, block=block)
                    self.assertSameArray(FUSED(CUBE, m, **kw), resample.warp_planes(CUBE, m, **kw))

    def test_other_kernels_match_per_plane_oracle(self):
        with np.errstate(all="ignore"):
            for kernel, mname, normalise, shape in itertools.product(
                    ("lanczos3", "lanczos2", "bilinear"), ("radial", "nan_inf", "edges"), (True, False), (None, (33, 3))):
                with self.subTest(kernel=kernel, mapping=mname, normalise=normalise, shape=shape):
                    kw = dict(kernel=kernel, normalise=normalise, output_shape=shape, block=32)
                    self.assertSameArray(FUSED(CUBE, MAPS[mname], **kw), resample.warp_planes(CUBE, MAPS[mname], **kw))

    def test_default_arguments(self):
        with np.errstate(all="ignore"):
            self.assertSameArray(FUSED(CUBE, RADIAL.apply), resample.warp_planes(CUBE, RADIAL.apply))

    def test_layouts_and_dtypes(self):
        base = (np.random.default_rng(3).random((3, 24, 37)) * 1000).astype(np.float32)
        layouts = {
            "c_contig": base, "row_flipped_view": np.ascontiguousarray(base[:, ::-1])[:, ::-1],
            "read_rgb_planar": np.moveaxis(np.moveaxis(np.ascontiguousarray(base[:, ::-1]), 0, -1)[::-1], -1, 0),
            "interleaved": np.moveaxis(np.ascontiguousarray(np.moveaxis(base, 0, -1)[::-1]), -1, 0)[:, ::-1],
            "float64": base.astype(np.float64) + 1e-9, "big_endian": base.astype(">f4"), "big_endian_f8": base.astype(">f8"),
            "int16": (base * 10).astype(np.int16), "fortran": np.asfortranarray(base),
            "col_flipped": np.ascontiguousarray(base[:, :, ::-1])[:, :, ::-1], "one_plane": base[:1],
            "five_planes": np.concatenate([base, base[:2] * 0.5]), "one_row": base[:, :1], "one_col": base[:, :, :1],
            "one_px": base[:, :1, :1], "zero_rows": base[:, :0],
        }

        def m(q):
            return q * 0.93 + np.array([2.2, 1.3])
        with np.errstate(all="ignore"):
            for (name, c), shape in itertools.product(layouts.items(), (None, (20, 40))):
                with self.subTest(layout=name, shape=shape):
                    self.assertSameArray(FUSED(c, m, block=16, output_shape=shape),
                                         resample.warp_planes(c, m, block=16, output_shape=shape))

    def test_errors_match_in_order(self):
        good, bad = [["1", "2"], ["3", "4"]], [["1", "2"], ["3", "abc"]]
        later_bad, first_bad = np.array([good, good, bad]), np.array([bad, good, good])
        cases = {
            "unknown_kernel": (CUBE, {"kernel": "nope"}), "2d": (CUBE[0], {}), "4d": (CUBE[None], {}), "1d": (CUBE[0, 0], {}),
            "zero_planes": (CUBE[:0], {}), "bad_output_shape": (CUBE, {"output_shape": (1, 2, 3)}),
            "negative_output": (CUBE, {"output_shape": (-1, 2)}), "block0": (CUBE, {"block": 0}),
            "strings": (np.full((3, 2, 2), "a"), {}),
            "later_bad_unknown_kernel": (later_bad, {"kernel": "nope"}), "later_bad_output_shape": (later_bad, {"output_shape": (1, 2, 3)}),
            "later_bad_fill": (later_bad, {"fill": "x"}), "later_bad_block0": (later_bad, {"block": 0}),
            "first_bad_unknown_kernel": (first_bad, {"kernel": "nope"}), "later_bad": (later_bad, {}),
        }
        with np.errstate(all="ignore"):                     # the 4-D case warps CUBE's inf per plane (0 * inf)
            for name, (c, kw) in cases.items():
                with self.subTest(case=name):
                    a = _outcome(lambda: resample.warp_planes(c, lambda q: q, **kw))
                    b = _outcome(lambda: FUSED(c, lambda q: q, **kw))
                    self.assertEqual(a[0], b[0])
                    if a[0] == "err":
                        self.assertEqual(a[1:], b[1:])
                    else:
                        self.assertTrue(np.array_equal(_bits(a[1]), _bits(b[1])))

    def test_out_parameter(self):
        with np.errstate(all="ignore"):
            ref = resample.warp_planes(CUBE, RADIAL.apply, fill=0.0)
            native = np.full((H, W, 3), 7.0, dtype=np.float32)          # garbage that must be overwritten by fill
            view = np.moveaxis(native[::-1], -1, 0)                      # native_from_undistorted's strided view
            self.assertIs(FUSED(CUBE, RADIAL.apply, fill=0.0, out=view), view)
        self.assertTrue(np.array_equal(_bits(np.ascontiguousarray(np.moveaxis(ref, 0, -1)[::-1])), _bits(native)))
        for bad_out in (np.empty((3, H, W + 1), np.float32), np.empty((3, H, W), np.float64)):
            with self.assertRaisesRegex(ValueError, "out must be a float32"):
                FUSED(CUBE, RADIAL.apply, out=bad_out)
        for cube in (CUBE[0], CUBE[:0], CUBE[None]):                    # fallback cubes: out= is refused, not ignored
            with self.assertRaisesRegex(ValueError, "out requires a non-empty"):
                FUSED(cube, RADIAL.apply, out=np.empty((1, H, W), np.float32))

    def test_mapping_called_once_per_block(self):          # documents deviation 1 (impure mappings see fewer calls)
        calls = {"fused": 0, "old": 0}

        def counting(key):
            def m(q):
                calls[key] += 1
                return q + 0.3
            return m
        with np.errstate(all="ignore"):
            FUSED(CUBE, counting("fused"), block=32)
            resample.warp_planes(CUBE, counting("old"), block=32)
        blocks = 3 * 5                                      # ceil(67/32) x ceil(131/32)
        self.assertEqual((calls["fused"], calls["old"]), (blocks, 3 * blocks))

    def test_product_inputs_raise_no_floating_point_error(self):
        cube = np.random.default_rng(5).random((3, H, W), dtype=np.float32)       # all finite, 0..1, like read_rgb
        with np.errstate(all="raise"), warnings.catch_warnings():
            warnings.simplefilter("error")
            a = resample.warp_planes(cube, RADIAL.apply, fill=0.0)
            b = FUSED(cube, RADIAL.apply, fill=0.0)
        self.assertTrue(np.array_equal(_bits(a), _bits(b)))

    def test_elementwise_math_is_position_independent(self):
        """Premise of computing once instead of per plane (G-1) and of reusing r**2 / r**4 (G-2)."""
        rng = np.random.default_rng(11)
        base = np.concatenate([rng.uniform(-3, 3, 40000), rng.uniform(-1e6, 1e6, 1000),
                               np.array([0.0, -0.0, 1e-310, 1e300, np.inf, -np.inf, np.nan])])
        other = base[::-1].copy()
        funcs = {"power4": lambda a, b: np.power(a, 4), "pow3": lambda a, b: a ** 3, "pow2": lambda a, b: a ** 2,
                 "square": lambda a, b: np.square(a), "hypot": np.hypot, "sinc": lambda a, b: np.sinc(a),
                 "floor": lambda a, b: np.floor(a), "abs": lambda a, b: np.abs(a)}
        with np.errstate(all="ignore"):
            for name, f in funcs.items():
                full = f(base, other)
                for offset, length, shift in itertools.product(range(17), (1, 7, 15, 16, 17, 255, 16384, 16385, len(base) - 17),
                                                               (0, 1, 3)):
                    seg, seg2 = base[offset:offset + length], other[offset:offset + length]
                    buf = np.empty(len(seg) + 3)
                    x = buf[shift:shift + len(seg)]
                    x[...] = seg
                    if not np.array_equal(f(x, seg2).view(np.uint64), full[offset:offset + length].view(np.uint64)):
                        self.fail(f"{name} is position dependent at offset {offset}, length {length}, shift {shift}")


if __name__ == "__main__":
    unittest.main()
