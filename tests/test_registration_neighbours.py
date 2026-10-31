"""P1a G-5: the exact cell search in mutual_nearest / associate_stars reproduces the all-pairs rules bit for bit.

The oracles are verbatim copies of the pre-G-5 code (registration_distortion 1.3.1 ``mutual_nearest``,
registration_adapter 0.2.2 ``associate_stars``), so the comparison does not rest on the production fallback.
Run:  python -B -m unittest tests.test_registration_neighbours   (from WORKSPACE, PYTHONPATH=src)
Also run once with NPY_DISABLE_CPU_FEATURES=X86_V3 (baseline SIMD kernels), see the design (final.md section 9).
"""
from __future__ import annotations

import contextlib
import io
import math
import struct
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starstack import registration_adapter as ra  # noqa: E402
from starstack import registration_distortion as rd  # noqa: E402
from starstack import registration_models as rm  # noqa: E402


# --------------------------------------------------------------------------- #
# Frozen oracles (verbatim pre-G-5 code)                                      #
# --------------------------------------------------------------------------- #

def oracle_mutual_nearest(predicted, reference, radius):
    if len(predicted) == 0 or len(reference) == 0:
        return np.zeros((0, 2), dtype=int)
    d2 = ((predicted[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    nearest_ref = d2.argmin(axis=1)
    nearest_pred = d2.argmin(axis=0)
    i = np.arange(len(predicted))
    ok = (nearest_pred[nearest_ref] == i) & (d2[i, nearest_ref] <= radius * radius)
    return np.column_stack([i[ok], nearest_ref[ok]])


def oracle_associate_stars(reference_stars, frame_stars, h_frame_to_reference, *, radius, mutual=True):
    if not (isinstance(radius, (int, float)) and math.isfinite(radius) and radius > 0):
        raise ValueError("radius must be a positive finite number")
    h = np.asarray(h_frame_to_reference, dtype=np.float64)
    ref = np.array([[s["x"], s["y"]] for s in reference_stars], dtype=np.float64).reshape(-1, 2)
    src = np.array([[s["x"], s["y"]] for s in frame_stars], dtype=np.float64).reshape(-1, 2)
    result = {"radius_px": float(radius), "mutual": bool(mutual), "reference_count": int(len(ref)),
              "frame_count": int(len(src)), "pairs": [], "unmatched_frame": [], "unmatched_reference": [],
              "ambiguous_frame": [], "matched_fraction_of_frame": 0.0}
    if len(ref) == 0 or len(src) == 0:
        result["unmatched_frame"] = list(range(len(src)))
        result["unmatched_reference"] = list(range(len(ref)))
        return result
    ones = np.ones((src.shape[0], 1))
    v = np.hstack([src, ones]) @ h.T
    projected = v[:, :2] / v[:, 2:3]
    d2 = ((projected[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    nearest_ref = d2.argmin(axis=1)
    nearest_d = np.sqrt(d2[np.arange(len(src)), nearest_ref])
    nearest_src = d2.argmin(axis=0)
    pairs = []
    ambiguous = []
    matched_ref = set()
    for i in range(len(src)):
        j = int(nearest_ref[i])
        if nearest_d[i] > radius:
            continue
        row = d2[i]
        within = int(np.count_nonzero(row <= radius * radius))
        if within > 1:
            ambiguous.append(i)
            continue
        if mutual and int(nearest_src[j]) != i:
            continue
        if j in matched_ref:
            continue
        matched_ref.add(j)
        pairs.append({"frame_index": i, "reference_index": j, "distance_px": float(nearest_d[i]),
                      "projected_xy": [float(projected[i, 0]), float(projected[i, 1])]})
    matched_src = {p["frame_index"] for p in pairs}
    result["pairs"] = pairs
    result["ambiguous_frame"] = ambiguous
    result["unmatched_frame"] = [i for i in range(len(src)) if i not in matched_src and i not in ambiguous]
    result["unmatched_reference"] = [j for j in range(len(ref)) if j not in matched_ref]
    result["matched_fraction_of_frame"] = len(pairs) / len(src)
    return result


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def same(a, b) -> bool:
    """Recursive equality on key order, Python type and float bits (NaN-aware); arrays on dtype, shape and values."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a.view(np.uint8), b.view(np.uint8)))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def stars(xy):
    return [{"x": float(x), "y": float(y)} for x, y in np.asarray(xy, dtype=np.float64).reshape(-1, 2)]


def lattice(rng, n, lo=-40, hi=40, step=1.0):
    return rng.integers(lo, hi, (n, 2)).astype(np.float64) * step


def star_field(rng, count=2000, kept=1700, noise=1.5, width=8280.0, height=5520.0):
    ref = np.column_stack([rng.uniform(0, width, count), rng.uniform(0, height, count)])
    take = rng.permutation(count)[:kept]
    extra = np.column_stack([rng.uniform(-50, width + 50, count - kept), rng.uniform(-50, height + 50, count - kept)])
    frame = np.vstack([ref[take] + rng.standard_normal((kept, 2)) * noise, extra])
    return ref, frame[rng.permutation(len(frame))]


RADII = (0, 0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0, math.sqrt(2.0), math.sqrt(5.0), 3, 4, 8.0, 20.0)
EDGE_OFFSETS = ((5.0, (3.0, 4.0)), (2.5, (1.5, 2.0)), (1.25, (0.75, 1.0)), (5.0, (5.0, 0.0)), (3.0, (0.0, 3.0)))
# radii whose nextafter(r * r) still has sqrt == r (found by search; the associate_stars sqrt test keeps such rows)
SQRT_BOUNDARY_RADII = (11.41699556812879, 15.050354396107027, 19.241959706496978, 14.329202012745514, 1.4728572069449926)


class DistanceBitsTest(unittest.TestCase):
    def test_power_two_is_the_square_ufunc(self):
        rng = np.random.default_rng(1)
        with np.errstate(all="ignore"):
            v = np.concatenate([rng.standard_normal(200_000) * 10.0 ** rng.integers(-320, 309, 200_000),
                                [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324]])
            self.assertTrue(same(v ** 2, np.square(v)))
            self.assertTrue(same(v ** 2, v * v))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(under="warn"):
                np.array([1e-170]) ** 2
        self.assertEqual([str(w.message) for w in caught], ["underflow encountered in square"])

    def test_pair_d2_has_the_bits_of_the_dense_cube_sum(self):
        rng = np.random.default_rng(2)
        for trial in range(20):
            n, m = int(rng.integers(1, 150)), int(rng.integers(1, 150))
            scale = 10.0 ** rng.integers(-150, 8, (1, 2)) if trial % 2 else 1.0
            p, r = rng.standard_normal((n, 2)) * scale * 1e3, rng.standard_normal((m, 2)) * scale * 1e3
            with np.errstate(all="ignore"):
                dense = ((p[:, None, :] - r[None, :, :]) ** 2).sum(axis=2)
                ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
                pairs = rm.pair_d2(p, r, ii.ravel(), jj.ravel()).reshape(n, m)
            self.assertTrue(same(dense, pairs), trial)


class HelperTest(unittest.TestCase):
    def test_cell_pairs_holds_every_close_pair_exactly_once(self):
        rng = np.random.default_rng(3)
        for trial in range(40):
            radius = float(rng.choice([0.0, 0.5, 1.0, 3.0, 20.0]))
            rr, cell = rm.grid_threshold(radius)
            a = rng.uniform(-60, 60, (int(rng.integers(1, 120)), 2)) * max(radius, 1.0)
            b = np.vstack([a[: len(a) // 2] + rng.uniform(-1, 1, (len(a) // 2, 2)) * radius,
                           rng.uniform(-60, 60, (30, 2)) * max(radius, 1.0)])
            ci, cj = rm.cell_pairs(a, b, cell, rm.pair_budget(len(a), len(b)))
            got = set(zip(ci.tolist(), cj.tolist()))
            self.assertEqual(len(got), len(ci), "a pair appears twice")
            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
            limit = (radius * (1 + 1e-12)) ** 2 + 1e-300
            close = set(zip(*[x.tolist() for x in np.nonzero(d2 <= limit)]))
            self.assertLessEqual(close, got, trial)

    def test_first_min_breaks_ties_by_lowest_index(self):
        owner = np.array([0, 0, 0, 2, 2, 1], dtype=np.intp)
        other = np.array([5, 3, 4, 9, 7, 1], dtype=np.intp)
        d2 = np.array([1.0, 2.0, 1.0, 0.5, 0.5, 3.0])
        best, best_d2 = rm.first_min(owner, other, d2, 4)
        self.assertEqual(best.tolist(), [4, 1, 7, -1])
        self.assertEqual(best_d2.tolist(), [1.0, 3.0, 0.5, math.inf])

    def test_guards(self):
        for radius in (0, 0.0, 3, 3.0, np.float64(3.0), 1e6):
            self.assertIsNotNone(rm.grid_threshold(radius), radius)
        class Radius(float):
            pass
        for radius in (True, np.float32(3.0), np.float16(3.0), np.int64(3), -1.0, math.nan, math.inf, 1.5e6, "3", None,
                       Radius(3.0)):
            self.assertIsNone(rm.grid_threshold(radius), repr(radius))
        for config in ({"under": "warn"}, {"under": "raise"}, {"under": "call", "call": lambda err, flag: None},
                       {"all": "raise"}, {"under": "print"}):
            with np.errstate(**config):
                self.assertIsNone(rm.grid_threshold(3.0), config)
        for config in ({"all": "ignore"}, {"over": "raise", "invalid": "raise", "divide": "raise", "under": "ignore"}):
            with np.errstate(**config):
                self.assertIsNotNone(rm.grid_threshold(3.0), config)
        ok = np.zeros((3, 2))
        self.assertTrue(rm.grid_points_ok(ok))
        for bad in (np.nan, np.inf, -np.inf, 1.0000001e8):
            p = ok.copy(); p[1, 1] = bad
            self.assertFalse(rm.grid_points_ok(p), bad)
        self.assertFalse(rm.grid_points_ok(ok.astype(np.float32)))
        self.assertFalse(rm.grid_points_ok(np.zeros((3, 3))))
        self.assertFalse(rm.grid_points_ok(np.zeros((0, 2))))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for mode in ("raise", "warn"):
                with np.errstate(all=mode):
                    for v in (np.nan, np.inf, -np.inf, 1e308, 5e-324, -0.0):
                        rm.grid_points_ok(np.array([[v, 1.0], [1.0, v]]))
            self.assertFalse(rm.grid_points_ok(np.array([[np.nan, 1.0]])))


class MutualNearestTest(unittest.TestCase):
    def check(self, p, r, radius, label=""):
        with np.errstate(all="ignore"):
            expected, got = oracle_mutual_nearest(p, r, radius), rd.mutual_nearest(p, r, radius)
        self.assertTrue(same(expected, got), f"{label} r={radius!r}: {expected.tolist()[:6]} vs {got.tolist()[:6]}")

    def test_lattices_with_ties_duplicates_and_negative_coordinates(self):
        for seed in range(120):
            rng = np.random.default_rng(100 + seed)
            step = float(rng.choice([1.0, 0.5, 0.25]))
            p, r = lattice(rng, int(rng.integers(0, 80)), step=step), lattice(rng, int(rng.integers(0, 80)), step=step)
            if len(p) > 2 and seed % 2:
                p[rng.integers(0, len(p), len(p) // 3)] = p[0]
            if len(r) > 2 and seed % 3 == 0:
                r[rng.integers(0, len(r), len(r) // 3)] = r[0]
            for radius in RADII:
                self.check(p, r, radius, f"seed {seed}")

    def test_exact_radius_edge(self):
        rng = np.random.default_rng(7)
        for radius, off in EDGE_OFFSETS:
            base = rng.integers(-100, 100, (40, 2)).astype(np.float64) * 7.0
            p = base + rng.choice([-1.0, 1.0], (40, 2)) * np.array(off)
            r = np.vstack([base, base[:8] + [[radius, 0.0]]])
            self.check(p, r, radius, "edge")
            self.assertGreater(len(rd.mutual_nearest(p, r, radius)), 0)

    def test_cell_boundaries_and_the_coordinate_limit(self):
        rng = np.random.default_rng(8)
        for radius in (0.5, 1.0, 3.0, 5.0, 20.0):
            cell = rm.grid_threshold(radius)[1]
            k = rng.integers(-30, 30, (80, 2)).astype(np.float64) * cell
            for p in (k, np.nextafter(k, np.inf), np.nextafter(k, -np.inf)):
                self.check(p, k + rng.uniform(-1, 1, k.shape) * radius, radius, "cell boundary")
        big = 1e8 - rng.integers(0, 6, (60, 2)).astype(np.float64) * 0.5
        self.check(big, big[::-1].copy(), 0.5, "near +limit")
        self.check(-big, -big[::-1].copy(), 1.0, "near -limit")

    def test_star_field(self):
        for seed in range(2):
            ref, frame = star_field(np.random.default_rng(9 + seed))
            for radius in (20.0, 10.0, 5.0, 3.0):
                self.check(frame, ref, radius, "field")

    def test_non_finite_and_out_of_range_inputs_keep_the_dense_rule(self):
        rng = np.random.default_rng(10)
        p = lattice(rng, 50, 0, 20); r = lattice(rng, 50, 0, 20) + 0.25
        for bad in (np.nan, np.inf, -np.inf, 2e8, -1e300):
            q = p.copy(); q[5, 1] = bad
            self.check(q, r, 1.0, f"pred {bad}")
            q = r.copy(); q[5, 0] = bad
            self.check(p, q, 1.0, f"ref {bad}")
        q, s = p.copy(), r.copy(); q[3] = [np.inf, 7.0]; s[4] = [np.inf, 7.0]
        self.check(q, s, 1.0, "inf - inf")
        for radius in (np.float32(1.5), np.float16(1.5), np.int64(2), True, -1.0, math.nan, math.inf, 2e6):
            self.check(p, r, radius, "radius type")
        self.check(p.astype(np.float32), r.astype(np.float32), 1.5, "float32")
        # pinned dense semantics: one NaN coordinate anywhere leaves no pair
        q = p.copy(); q[7, 0] = np.nan
        self.assertEqual(rd.mutual_nearest(q, p, 1.0).shape, (0, 2))
        self.assertEqual(rd.mutual_nearest(p, q, 1.0).shape, (0, 2))

    def test_clustered_input_over_the_pair_budget_keeps_the_dense_rule(self):
        rng = np.random.default_rng(14)
        r = rng.uniform(0, 20.0, (2000, 2))
        p = r + rng.standard_normal(r.shape) * 2.0
        self.assertIsNone(rm.cell_pairs(p, r, rm.grid_threshold(20.0)[1], rm.pair_budget(len(p), len(r))))
        self.check(p, r, 20.0, "clustered")

    def test_empty_inputs(self):
        for p, r in ((np.zeros((0, 2)), np.zeros((3, 2))), (np.zeros((3, 2)), np.zeros((0, 2)))):
            self.check(p, r, 1.0, "empty")


class AssociateStarsTest(unittest.TestCase):
    def check(self, refs, frame, h, radius, label=""):
        for mutual in (True, False):
            with np.errstate(all="ignore"):
                expected = oracle_associate_stars(refs, frame, h, radius=radius, mutual=mutual)
                got = ra.associate_stars(refs, frame, h, radius=radius, mutual=mutual)
            self.assertTrue(same(expected, got), f"{label} r={radius!r} mutual={mutual}")

    def test_lattices_with_shifts(self):
        for seed in range(120):
            rng = np.random.default_rng(300 + seed)
            step = float(rng.choice([1.0, 0.5]))
            r, p = lattice(rng, int(rng.integers(0, 70)), step=step), lattice(rng, int(rng.integers(0, 70)), step=step)
            h = np.eye(3); h[0, 2] = float(rng.choice([0.0, 0.5, -1.0])); h[1, 2] = float(rng.choice([0.0, -0.5, 1.0]))
            radius = float(rng.choice([0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0])) if seed % 4 else int(rng.choice([1, 2, 3]))
            self.check(stars(r), stars(p), h, radius, f"seed {seed}")

    def test_exact_edge(self):
        rng = np.random.default_rng(11)
        for radius, off in EDGE_OFFSETS:
            base = rng.integers(-100, 100, (40, 2)).astype(np.float64) * 7.0
            p = base + rng.choice([-1.0, 1.0], (40, 2)) * np.array(off)
            self.check(stars(np.vstack([base, base[:8] + [[radius, 0.0]]])), stars(p), np.eye(3), radius, "edge")

    def test_sqrt_boundary_rows_are_kept_like_the_dense_rule(self):
        """d2 = nextafter(r*r) > r*r while sqrt(d2) == r: the row passes the distance test, 'within' counts 0."""
        kept = 0
        for r0 in SQRT_BOUNDARY_RADII:
            rr = r0 * r0
            b = math.sqrt(float(np.nextafter(rr, np.inf)) - rr)
            for _ in range(200):
                d2 = r0 * r0 + b * b
                if d2 > rr and math.sqrt(d2) <= r0:
                    break
                b = float(np.nextafter(b, np.inf if d2 <= rr else 0.0))
            refs = [{"x": 0.0, "y": 0.0}, {"x": 400.0, "y": 0.0}]
            frame = [{"x": r0, "y": b}, {"x": 400.0 + r0 / 2, "y": 0.0}]
            self.check(refs, frame, np.eye(3), r0, "sqrt boundary")
            kept += any(p["frame_index"] == 0 for p in ra.associate_stars(refs, frame, np.eye(3), radius=r0)["pairs"])
        self.assertEqual(kept, len(SQRT_BOUNDARY_RADII))

    def test_star_field_through_a_homography(self):
        rng = np.random.default_rng(12)
        ref, frame = star_field(rng)
        ang = 0.004
        h = np.array([[math.cos(ang), -math.sin(ang), 3.0], [math.sin(ang), math.cos(ang), -2.0], [1e-9, -2e-9, 1.0]])
        self.check(stars(ref), stars(frame), h, 8.0, "field")

    def test_non_finite_projection_keeps_the_dense_rule(self):
        rng = np.random.default_rng(13)
        refs, frame = stars(lattice(rng, 40, 0, 20)), stars(lattice(rng, 40, 0, 20) + 0.25)
        bad = [dict(s) for s in frame]; bad[5]["x"] = math.nan
        self.check(refs, bad, np.eye(3), 1.0, "NaN star")
        for h in (np.array([[1, 0, 0], [0, 1, 0], [np.nan, 0, 1.0]]), np.array([[1e9, 0, 0], [0, 1e9, 0], [0, 0, 1.0]]),
                  np.array([[1, 0, 0], [0, 1, 0], [-1.0 / 3.0, 0, 1.0]])):
            self.check(refs, frame, h, 1.0, "odd homography")
        # pinned dense semantics: the NaN row claims reference 0 with a NaN distance and blocks every mutual pair
        pairs = ra.associate_stars(refs, bad, np.eye(3), radius=1.0)["pairs"]
        self.assertEqual([(p["frame_index"], p["reference_index"]) for p in pairs], [(5, 0)])
        self.assertTrue(math.isnan(pairs[0]["distance_px"]))

    def test_validation_and_empty_lists_unchanged(self):
        refs = [{"x": 1.0, "y": 1.0}]
        for radius in (0, -1.0, math.nan, math.inf, np.float32(2.0), "3", True):
            outcomes = []
            for fn in (oracle_associate_stars, ra.associate_stars):
                try:
                    outcomes.append(("ok", fn(refs, refs, np.eye(3), radius=radius)))
                except Exception as exc:  # noqa: BLE001
                    outcomes.append((type(exc).__name__, str(exc)))
            self.assertTrue(same(outcomes[0], outcomes[1]), repr(radius))
        self.check([], refs, np.eye(3), 1.0, "no reference")
        self.check(refs, [], np.eye(3), 1.0, "no frame stars")


class _Log:
    def __init__(self):
        self.lines = []

    def write(self, msg):
        self.lines.append(str(msg))


def outcome(fn, args, kwargs, config):
    """Result (or exception type and text), warnings and errcall/log records of one call under ``np.errstate``."""
    calls, log = [], _Log()
    config = dict(config)
    if config.pop("_call", False):
        config["call"] = lambda err, flag: calls.append((err, int(flag)))
    if config.pop("_log", False):
        config["call"] = log
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        with np.errstate(**config):
            try:
                result = ("ok", fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001
                result = (type(exc).__name__, str(exc))
    return result, [(w.category, str(w.message)) for w in caught], calls, log.lines


ERRSTATES = ({}, {"all": "raise"}, {"all": "warn"}, {"under": "raise"}, {"under": "warn"}, {"all": "ignore"},
             {"over": "raise", "invalid": "raise", "divide": "raise", "under": "ignore"}, {"all": "call", "_call": True},
             {"all": "log", "_log": True})


class ErrstateTest(unittest.TestCase):
    """Warnings, FloatingPointError and errcall/log records are those of the dense rules under every errstate
    (critique issue 1: the dense rule squares every pair, the cell search only candidates and ``xy / cell``)."""

    def compare(self, oracle, production, args, kwargs, label):
        for config in ERRSTATES:
            a, b = outcome(oracle, args, kwargs, config), outcome(production, args, kwargs, config)
            self.assertTrue(same(a[0], b[0]) and a[1:] == b[1:], f"{label} {config}: {a} vs {b}")

    def test_mutual_nearest(self):
        rng = np.random.default_rng(15)
        ref, frame = star_field(rng, count=300, kept=250)
        cases = {
            "underflow outside the candidates": (np.array([[0.0, 0.0], [1e-170, 500.0]]),
                                                 np.array([[0.0, 0.25], [3e-170, 900.0]]), 1.0),
            "underflow in a candidate": (np.array([[0.0, 0.0], [1e-170, 0.0], [500.0, 500.0]]),
                                         np.array([[3e-170, 0.0], [500.0, 500.0 + 1e-200]]), 1.0),
            "underflow only in xy / cell": (np.array([[1e-320, 0.0], [5.0, 5.0]]), np.array([[5.0, 5.0], [1.0, 1.0]]), 1.0),
            "overflow beyond the coordinate limit": (np.array([[1e200, 0.0], [10.0, 10.0]]),
                                                     np.array([[-1e200, 0.0], [10.0, 10.5]]), 1.0),
            "inf - inf": (np.array([[0.0, 0.0], [np.inf, 3.0]]), np.array([[0.1, 0.0], [np.inf, 10.2]]), 1.0),
            "NaN": (np.array([[0.0, 0.0], [np.nan, 3.0]]), np.array([[0.1, 0.0], [10.0, 10.2]]), 1.0),
            "star field": (frame, ref, 5.0),
        }
        for label, args in cases.items():
            self.compare(oracle_mutual_nearest, rd.mutual_nearest, args, {}, label)

    def test_associate_stars(self):
        rng = np.random.default_rng(16)
        ref, frame = star_field(rng, count=300, kept=250)
        cases = {
            "underflow outside the candidates": (stars([[0.0, 0.25], [3e-170, 900.0]]), stars([[0.0, 0.0], [1e-170, 500.0]]),
                                                 np.eye(3)),
            "underflow only in xy / cell": (stars([[5.0, 5.0], [1.0, 1.0]]), stars([[1e-320, 0.0], [5.0, 5.0]]), np.eye(3)),
            "w = 0": (stars(ref[:30]), stars(frame[:30]), np.array([[1, 0, 0], [0, 1, 0], [-1.0 / frame[3, 0], 0, 1.0]])),
            "star field": (stars(ref), stars(frame), np.eye(3)),
        }
        for label, args in cases.items():
            for mutual in (True, False):
                self.compare(oracle_associate_stars, ra.associate_stars, args, {"radius": 2.0, "mutual": mutual}, label)


class PathTakenTest(unittest.TestCase):
    """Product-shaped calls really take the cell search (the dense fallbacks are not reached)."""

    def test_star_fields_take_the_cell_search(self):
        from unittest import mock
        ref, frame = star_field(np.random.default_rng(18))
        dense = AssertionError("the dense rule ran")
        with mock.patch.object(rd, "_mutual_nearest_dense", side_effect=dense), \
                mock.patch.object(ra, "_associate_dense", side_effect=dense):
            for radius in (20.0, 10.0, 5.0, 3.0):
                self.assertTrue(same(oracle_mutual_nearest(frame, ref, radius), rd.mutual_nearest(frame, ref, radius)))
            self.assertTrue(same(oracle_associate_stars(stars(ref), stars(frame), np.eye(3), radius=8.0),
                                 ra.associate_stars(stars(ref), stars(frame), np.eye(3), radius=8.0)))


if __name__ == "__main__":
    unittest.main()
