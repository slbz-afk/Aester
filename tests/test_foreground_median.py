"""Efficiency P1a C-1: the foreground median (imaging.median_foreground).

The oracle is the former memory-mapped np.median implementation, kept verbatim as
imaging._median_foreground_memmap.  Every comparison is bit for bit (uint32 views), and each test also
asserts which route the new code took: the 16-bit selection network, the float np.median path on explicit
reads, or the former code itself.
"""
from collections import Counter
from pathlib import Path
import bz2
import gzip
import lzma
import math
import mmap
import os
import warnings
import zipfile

import numpy as np
import pytest
from astropy.io import fits

from starstack import imaging, orientation

ORACLE = imaging._median_foreground_memmap    # captured before any monkeypatch


# ---- hand-written FITS files (full control of cards, padding and truncation) -----------------------------------------
SIRIL_CARDS = (   # the cards of a real Siril 1.4.4 converted light frame (ROWORDER last of the structural ones)
    "SIMPLE  =                    T / file does conform to FITS standard",
    "BITPIX  =                   16 / number of bits per data pixel",
    "NAXIS   =                    3 / number of data axes",
    "NAXIS1  = {w:>20} / length of data axis 1",
    "NAXIS2  = {h:>20} / length of data axis 2",
    "NAXIS3  =                    3 / length of data axis 3",
    "EXTEND  =                    T / FITS dataset may contain extensions",
    "BZERO   =               32768. / Offset data range to that of unsigned short",
    "BSCALE  =                   1. / Default scaling factor",
    "PROGRAM = 'Siril 1.4.4'        / Software that created this HDU",
    "ROWORDER= '{roworder:<9}'          / Order of the rows in image array",
    "EXPTIME =                  10. / [s]  Exposure time duration",
    "XBINNING=                    1 / Camera binning mode",
    "YBINNING=                    1 / Camera binning mode",
    "XPIXSZ  =     4.34999990463257 / [um] Pixel X axis size",
    "YPIXSZ  =     4.34999990463257 / [um] Pixel Y axis size",
)
DTYPES = {16: ">i2", -32: ">f4", 8: ">u1", 32: ">i4", 64: ">i8", -64: ">f8"}


def card(key, value=None):
    if value is None:
        return f"{key:<80}"
    if isinstance(value, bool):
        text = f"{key:<8}= {'T' if value else 'F':>20}"
    elif isinstance(value, str) and not value.startswith("RAW:"):
        text = f"{key:<8}= '{value:<8}'"
    else:
        text = f"{key:<8}= {value[4:] if isinstance(value, str) else repr(value):>20}"
    return text[:80].ljust(80)


def fits_bytes(cards, stored, bitpix=16, pad=True, truncate=0, trailer=b""):
    head = "".join(c[:80].ljust(80) for c in cards) + "END".ljust(80)
    head += " " * (-len(head) % 2880)
    data = np.ascontiguousarray(stored).astype(DTYPES[bitpix]).tobytes() if np.size(stored) else b""
    if pad:
        data += b"\0" * (-len(data) % 2880)
    if truncate:
        data = data[:-truncate]
    return head.encode("ascii") + data + trailer


def write_fits(path, stored, bitpix=16, extra=(), naxes=None, simple=True, **layout):
    """``stored``: the raw codes/values in storage order (planes, rows, columns)."""
    stored = np.asarray(stored)
    naxes = list(stored.shape[::-1]) if naxes is None else naxes
    cards = [card("SIMPLE", simple), card("BITPIX", bitpix), card("NAXIS", len(naxes))]
    cards += [card(f"NAXIS{i + 1}", n) for i, n in enumerate(naxes)]
    cards += [card(key, value) for key, value in extra]
    Path(path).write_bytes(fits_bytes(cards, stored, bitpix, **layout))
    return Path(path)


def stored_codes(physical, roworder="BOTTOM-UP"):
    """(H, W, 3) uint16 values in output (bottom-up) row order -> (3, H, W) int16 codes as the file stores them."""
    rows = physical[::-1] if roworder == "TOP-DOWN" else physical
    return (np.moveaxis(rows, -1, 0).astype(np.int32) - 32768).astype(np.int16)


def siril_frame(path, physical, roworder="BOTTOM-UP", extra=()):
    """A Siril-like converted frame; ``roworder=None`` leaves the ROWORDER card out."""
    codes = stored_codes(physical, roworder)
    cards = [c.format(h=codes.shape[1], w=codes.shape[2], roworder=roworder or "") for c in SIRIL_CARDS
             if roworder is not None or not c.startswith("ROWORDER")]
    Path(path).write_bytes(fits_bytes(cards + list(extra), codes))
    return Path(path)


def float_frame(path, physical_float, roworder="BOTTOM-UP"):
    rows = physical_float[::-1] if roworder == "TOP-DOWN" else physical_float
    return write_fits(path, np.moveaxis(rows, -1, 0).astype(np.float32), -32, [("ROWORDER", roworder)])


def frames_of(rng, kind, n, h, w):
    """(n, H, W, 3) uint16 physical values."""
    if kind == "random":
        return rng.integers(0, 65536, size=(n, h, w, 3)).astype(np.uint16)
    if kind == "ties":
        return (rng.integers(0, 3, size=(n, h, w, 3)) * 32767).astype(np.uint16)
    if kind == "extremes":
        return np.where(rng.random((n, h, w, 3)) < 0.5, 0, 65535).astype(np.uint16)
    if kind == "narrow":
        return rng.integers(30000, 30003, size=(n, h, w, 3)).astype(np.uint16)
    if kind == "zero_masked":    # like frames zeroed in place by sky validity: exact 0 in a random subset of frames
        values = rng.integers(0, 65536, size=(n, h, w, 3)).astype(np.uint16)
        values[rng.random((n, h, w)) < 0.4] = 0
        return values
    raise AssertionError(kind)


def banded_frames(rng, n, h, w):
    """Every data kind in its own band of rows, so one set of frames covers them all."""
    kinds = ("random", "ties", "extremes", "narrow", "zero_masked")
    edges = np.linspace(0, h, len(kinds) + 1).astype(int)
    out = np.empty((n, h, w, 3), dtype=np.uint16)
    for kind, y0, y1 in zip(kinds, edges[:-1], edges[1:]):
        out[:, y0:y1] = frames_of(rng, kind, n, y1 - y0, w)
    return out


def roworder_for(order, i):
    return {"all-bottom": "BOTTOM-UP", "all-top": "TOP-DOWN", "none": None,
            "mixed": "TOP-DOWN" if i % 3 == 1 else "BOTTOM-UP"}[order]


# ---- comparison helpers ---------------------------------------------------------------------------------------------
def outcome(fn, paths, stop=None):
    """("ok", array, cancel checks) or ("err", "Type: message", cancel checks)."""
    calls = [0]

    def check():
        calls[0] += 1
        if stop is not None and calls[0] >= stop:
            raise RuntimeError("stop numerical fusion")
    try:
        result = fn(paths, check)
    except Exception as exc:
        return "err", f"{type(exc).__name__}: {exc}", calls[0]
    return "ok", result, calls[0]


def assert_same(old, new):
    assert (new[0], new[2]) == (old[0], old[2])
    if old[0] == "err":
        assert new[1] == old[1]
    else:
        assert new[1].dtype == old[1].dtype == np.float32 and new[1].shape == old[1].shape
        assert np.array_equal(new[1].view(np.uint32), old[1].view(np.uint32))


@pytest.fixture
def routes(monkeypatch):
    """Records the route of every median_foreground call: ("network", net), ("float", None) or ("former", None)."""
    seen = []
    real_network, real_memmap = imaging._median_network, imaging._median_foreground_memmap

    def network(frames):
        net = real_network(frames)
        seen.append(("network" if net is not None else "float", net))
        return net

    def former(paths, cancel_check):
        seen.append(("former", None))
        return real_memmap(paths, cancel_check)
    monkeypatch.setattr(imaging, "_median_network", network)
    monkeypatch.setattr(imaging, "_median_foreground_memmap", former)
    return seen


def middle(n):
    return (n // 2 - 1, n // 2) if n % 2 == 0 else (n // 2,)


# ---- 1. the selection network ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(1, 25))
def test_median_pairs_exact_on_all_zero_one_inputs(n):
    """0-1 principle, exhaustive and bit-sliced: input x = 64 * word + bit; wires 0..5 enumerate the bit, wires 6.. the
    word.  Position j of an ascending sort of a 0-1 input holds 1 exactly when the input has >= n - j ones."""
    words = np.arange(1 << max(n - 6, 0), dtype=np.uint64)
    low = min(n, 6)
    lanes = []
    for i in range(n):
        if i < 6:
            pattern = np.uint64(sum(1 << b for b in range(64) if (b >> i) & 1))
            lanes.append(np.full(words.shape, pattern, dtype=np.uint64))
        else:
            lanes.append(np.where((words >> np.uint64(i - 6)) & np.uint64(1), ~np.uint64(0), np.uint64(0)))
    for a, b in imaging.median_pairs(n):
        assert 0 <= a < b < n
        lanes[a], lanes[b] = lanes[a] & lanes[b], lanes[a] | lanes[b]     # min, max of 0-1 lanes
    high = np.bitwise_count(words).astype(np.int64)
    low_ones = [bin(b & ((1 << low) - 1)).count("1") for b in range(64)]
    masks = np.array([sum(1 << b for b in range(64) if low_ones[b] >= t) for t in range(8)], dtype=np.uint64)
    for j in middle(n):
        assert np.array_equal(lanes[j], masks[np.clip(n - j - high, 0, 7)])


def _textbook_batcher(lo, hi):
    """Batcher's odd-even merge sort in its recursive textbook form on wires lo..hi (hi - lo + 1 a power of two)."""
    def merge(lo, hi, r):
        step = r * 2
        if step < hi - lo:
            yield from merge(lo, hi, step)
            yield from merge(lo + r, hi, step)
            yield from ((i, i + r) for i in range(lo + r, hi - r, step))
        else:
            yield lo, lo + r
    if hi - lo >= 1:
        mid = lo + (hi - lo) // 2
        yield from _textbook_batcher(lo, mid)
        yield from _textbook_batcher(mid + 1, hi)
        yield from merge(lo, hi, 1)


def _asap(pairs):
    """The network as a multiset of (as-soon-as-possible depth, comparator): independent of the emission order."""
    depth, out = Counter(), Counter()
    for a, b in pairs:
        d = max(depth[a], depth[b]) + 1
        depth[a] = depth[b] = d
        out[(d, a, b)] += 1
    return out


@pytest.mark.parametrize("n", list(range(1, 131)) + [255, 256, 257, 300])
def test_median_pairs_generator_is_batcher(n):
    """The shipped network = the textbook Batcher network on the next power of two, padding comparators dropped and
    pruned backwards to the middle outputs (an independent construction)."""
    size = 1
    while size < n:
        size *= 2
    full = [(a, b) for a, b in _textbook_batcher(0, size - 1) if b < n]
    need, kept = set(middle(n)), []
    for a, b in reversed(full):
        if a in need or b in need:
            kept.append((a, b))
            need.update((a, b))
    assert _asap(imaging.median_pairs(n)) == _asap(kept[::-1])


@pytest.mark.parametrize("n", [47, 48, 49, 96, 144, 300])
def test_median_pairs_random_large_n(n):
    rng = np.random.default_rng(n)
    adversarial = [np.sort(rng.integers(0, 65536, size=(n, 500)), axis=0),
                   np.sort(rng.integers(0, 65536, size=(n, 500)), axis=0)[::-1],
                   rng.integers(0, 3, size=(n, 1000)), rng.integers(65533, 65536, size=(n, 1000))]
    for count in range(max(0, n // 2 - 3), min(n, n // 2 + 3) + 1):
        adversarial.append(np.where(np.argsort(rng.random((n, 200)), axis=0) < count, 1, 0))
    data = np.concatenate([rng.integers(0, 65536, size=(n, 20000))] + adversarial, axis=1).astype(np.uint16)
    lanes = [row.copy() for row in data]
    for a, b in imaging.median_pairs(n):
        lanes[a], lanes[b] = np.minimum(lanes[a], lanes[b]), np.maximum(lanes[a], lanes[b])
    ordered = np.sort(data, axis=0)
    for j in middle(n):
        assert np.array_equal(lanes[j], ordered[j])


# ---- 2. the conversion table -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("bscale,bzero", [(1, 32768), (1.0, 32768.0), (2.0, 32768.0), (1.0, 0.0), (0.25, 1000.5)])
def test_conversion_table_is_former_conversion_for_every_code(bscale, bzero):
    codes = np.arange(-32768, 32768, dtype=np.int32).astype(">i2")
    block = np.concatenate([codes, codes[:3 * 64 * 342 - codes.size]]).astype(">i2").reshape(3, 64, 342)
    assert block.dtype == np.dtype(">i2")        # like the former memmap block
    raw = np.array(block, dtype=np.float32)      # the former op chain, verbatim
    raw *= float(bscale)
    raw += float(bzero)
    raw /= np.float32(65535)
    table = imaging.median_conversion_table(float(bscale), float(bzero))
    assert table.dtype == np.float32 and table.shape == (65536,)
    index = block.astype(np.int32) + 32768
    assert np.array_equal(table[index].view(np.uint32), raw.view(np.uint32))
    assert np.array_equal((block.view(">u2") ^ 0x8000).astype(np.int32), index)    # the network's key is code + 32768


def test_table_usable_rejects_what_would_break_the_order_statistics():
    table = imaging.median_conversion_table
    with np.errstate(all="ignore"):
        assert imaging._table_usable(table(1.0, 32768.0))
        assert imaging._table_usable(table(1.0, 0.0))
        assert imaging._table_usable(table(-0.0, 0.0))          # every entry +0.0
        assert not imaging._table_usable(table(-1.0, 32768.0))  # decreasing
        assert not imaging._table_usable(table(1e39, 0.0))      # inf
        assert not imaging._table_usable(table(float("nan"), 0.0))
        assert not imaging._table_usable(table(-0.0, -0.0))     # +0.0 and -0.0 mixed


# ---- 3. whole-function identity on the integer path ------------------------------------------------------------------
@pytest.mark.parametrize("order", ["all-bottom", "all-top", "mixed", "none"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 20, 47, 48, 49])
def test_integer_path_bit_identical_to_former(tmp_path, routes, n, order):
    rng = np.random.default_rng(1000 * n + len(order))
    values = banded_frames(rng, n, 131, 11)     # 131 rows: two full 64-row blocks and a 3-row one
    paths = [siril_frame(tmp_path / f"light_{i:05d}.fit", values[i], roworder_for(order, i)) for i in range(n)]
    old = outcome(ORACLE, paths)
    new = outcome(imaging.median_foreground, paths)
    assert_same(old, new)
    assert [r for r, _ in routes] == ["network"]


@pytest.mark.parametrize("order", ["all-top", "mixed"])
@pytest.mark.parametrize("n", [3, 8, 48])
def test_integer_path_forced_small_tiles(tmp_path, monkeypatch, routes, n, order):
    """Several tiles per 64-row block (tile heights that do not divide 64), TOP-DOWN frames included: the mirrored
    tile slices of a TOP-DOWN frame with s > 0 are exercised.  H = 195: the last block has 3 rows."""
    width = 23
    rng = np.random.default_rng(7 * n)
    values = rng.integers(0, 65536, size=(n, 195, width, 3)).astype(np.uint16)
    values[:, 5:9] = rng.integers(0, 3, size=(n, 4, width, 3)) * 20000
    paths = [siril_frame(tmp_path / f"light_{i:05d}.fit", values[i], roworder_for(order, i)) for i in range(n)]
    old = outcome(ORACLE, paths)
    for rows in (4, 7, 10):
        monkeypatch.setattr(imaging, "MEDIAN_LANE_BUDGET", rows * (n + 1) * width * 2)
        routes.clear()
        assert_same(old, outcome(imaging.median_foreground, paths))
        assert [r for r, _ in routes] == ["network"] and routes[0][1].rows == rows


# ---- 4. float and former routes ---------------------------------------------------------------------------------------
def _scaled(tmp_path, tag, scalings, n=5, kind="random"):
    values = frames_of(np.random.default_rng(len(tag)), kind, n, 131, 7)
    paths = []
    for i in range(n):
        bscale, bzero = scalings[i % len(scalings)]
        paths.append(write_fits(tmp_path / f"{tag}{i}.fit", stored_codes(values[i]), 16,
                                [("BZERO", bzero), ("BSCALE", bscale), ("ROWORDER", "BOTTOM-UP")]))
    return paths


def _compressed(tmp_path, kind):
    rng = np.random.default_rng(3)
    paths = []
    for i in range(4):
        plain = siril_frame(tmp_path / f"plain{i}.fit", rng.integers(0, 65536, size=(70, 9, 3)).astype(np.uint16))
        raw, target = plain.read_bytes(), tmp_path / f"light_{i:05d}.fit"
        if kind == "gzip0":
            target.write_bytes(gzip.compress(raw, compresslevel=0))
        elif kind == "bzip2":
            target.write_bytes(bz2.compress(raw))
        elif kind == "xz":
            target.write_bytes(lzma.compress(raw))
        else:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
                archive.writestr("light.fit", raw)
        paths.append(target)
    return paths


def _siril_set(tmp_path, n=4, h=131, w=9, seed=5):
    values = frames_of(np.random.default_rng(seed), "random", n, h, w)
    return [siril_frame(tmp_path / f"g{i}.fit", values[i]) for i in range(n)]


def _unquoted_roworder(tmp_path):
    rng = np.random.default_rng(4)
    paths = []
    for i in range(3):
        codes = stored_codes(rng.integers(0, 65536, size=(40, 30, 3)).astype(np.uint16))
        cards = [c.format(h=40, w=30, roworder="BOTTOM-UP") for c in SIRIL_CARDS]
        cards = [("ROWORDER= BOTTOM-UP" if c.startswith("ROWORDER") else c) for c in cards]
        paths.append(tmp_path / f"u{i}.fit")
        paths[-1].write_bytes(fits_bytes(cards, codes))
    return paths


FLOAT_AND_FORMER = {
    # name: (builder, expected route or None when the call fails before any route is chosen)
    "mixed u16 and float32": (lambda t: _siril_set(t, 2) + [float_frame(t / "f.fit", np.random.default_rng(1).random(
        (131, 9, 3)).astype(np.float32))], "float"),
    "per-frame BZERO": (lambda t: _scaled(t, "pz", [("RAW:1.", "RAW:32768."), ("RAW:1.", "RAW:32767.")]), "float"),
    "negative BSCALE": (lambda t: _scaled(t, "nb", [("RAW:-1.", "RAW:32768.")]), "float"),
    "BSCALE -0.0 BZERO -0.0 (mixed signed zeros)": (
        lambda t: _scaled(t, "mz", [("RAW:-0.0", "RAW:-0.0")], kind="ties"), "float"),
    "BSCALE overflow to inf": (lambda t: _scaled(t, "ov", [("RAW:1E+39", "RAW:0.")]), "float"),
    "float32 with NaN": (lambda t: [write_fits(t / f"fn{i}.fit", np.where(
        np.random.default_rng(i).random((3, 70, 6)) < .1, np.nan, .5).astype(np.float32), -32) for i in range(5)], "float"),
    "float32 signed zeros": (lambda t: [write_fits(t / f"fz{i}.fit", np.where(
        np.random.default_rng(i).random((3, 70, 6)) < .5, -0.0, 0.0).astype(np.float32), -32) for i in range(4)], "float"),
    "u16 BSCALE 2 (network)": (lambda t: _scaled(t, "s2", [("RAW:2.", "RAW:32768.")]), "network"),
    "signed int16 BZERO 0 (network)": (lambda t: _scaled(t, "i0", [("RAW:1.", "RAW:0.")]), "network"),
    "BSCALE 0.25 BZERO 1000.5 (network)": (lambda t: _scaled(t, "qb", [("RAW:0.25", "RAW:1000.5")]), "network"),
    "BSCALE -0.0 BZERO 0 (network, all +0.0)": (lambda t: _scaled(t, "pz0", [("RAW:-0.0", "RAW:0.")], kind="ties"),
                                                "network"),
    "BSCALE 1.0D0 Fortran exponent": (lambda t: _scaled(t, "fd", [("RAW:1.0D0", "RAW:32768.")]), "network"),
    "BLANK present, ignored as before": (lambda t: [siril_frame(t / f"bl{i}.fit", frames_of(
        np.random.default_rng(i), "ties", 1, 64, 5)[0], extra=[card("BLANK", -32768)]) for i in range(3)], "network"),
    "data complete, padding missing": (lambda t: _siril_set(t, 2) + [write_fits(
        t / "nopad.fit", stored_codes(frames_of(np.random.default_rng(2), "random", 1, 131, 9)[0]), 16,
        [("BZERO", "RAW:32768.")], pad=False)], "network"),
    "duplicate BZERO card": (lambda t: _siril_set(t, 2) + [write_fits(
        t / "dup.fit", stored_codes(frames_of(np.random.default_rng(2), "random", 1, 131, 9)[0]), 16,
        [("BZERO", "RAW:32768."), ("BSCALE", "RAW:1."), ("BZERO", "RAW:0.")])], "former"),
    "truncated data unit": (lambda t: _siril_set(t, 2)[:1] + [write_fits(
        t / "trunc.fit", np.zeros((3, 131, 9), np.int16), 16, [("BZERO", "RAW:32768.")], truncate=3000)], "former"),
    "truncated data unit first": (lambda t: [write_fits(
        t / "trunc.fit", np.zeros((3, 131, 9), np.int16), 16, [("BZERO", "RAW:32768.")], truncate=3000)]
        + _siril_set(t, 2), "former"),
    "different height": (lambda t: _siril_set(t, 2) + [siril_frame(t / "other.fit", frames_of(
        np.random.default_rng(2), "random", 1, 130, 9)[0])], "former"),
    "single plane first": (lambda t: [write_fits(t / "one.fit", np.zeros((1, 131, 9), np.int16), 16,
                                                 [("BZERO", "RAW:32768.")])] + _siril_set(t, 1), "former"),
    "NAXIS 2 second": (lambda t: _siril_set(t, 1) + [write_fits(t / "two.fit", np.zeros((131, 9), np.int16), 16,
                                                                [("BZERO", "RAW:32768.")])], "former"),
    "BITPIX 8": (lambda t: _siril_set(t, 1) + [write_fits(t / "b8.fit", np.zeros((3, 131, 9), np.uint8), 8)], "former"),
    "BSCALE not a number": (lambda t: _siril_set(t, 1) + [write_fits(
        t / "bad.fit", np.zeros((3, 131, 9), np.int16), 16, [("BZERO", "RAW:32768."), ("BSCALE", "abc")])], "former"),
    "header 2-D, data 3-D": (lambda t: [write_fits(t / "n3.fit", np.zeros((3, 131, 9), np.int16), 16,
                                                   [("BZERO", "RAW:32768.")], naxes=[9, 131])] + _siril_set(t, 1),
                             "former"),
    "unquoted ROWORDER": (_unquoted_roworder, "former"),
    "gzip level 0 content": (lambda t: _compressed(t, "gzip0"), "former"),
    "bzip2 content": (lambda t: _compressed(t, "bzip2"), "former"),
    "xz content": (lambda t: _compressed(t, "xz"), "former"),
    "zip content": (lambda t: _compressed(t, "zip"), "former"),
    "no frames": (lambda t: [], "former"),
    "missing third file": (lambda t: _siril_set(t, 2) + [t / "missing.fit"] + _siril_set(t, 3)[2:], None),
}


@pytest.mark.parametrize("name", list(FLOAT_AND_FORMER))
def test_float_and_former_routes_bit_identical(tmp_path, routes, name):
    build, expected = FLOAT_AND_FORMER[name]
    paths = build(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = outcome(ORACLE, paths)
        new = outcome(imaging.median_foreground, paths)
    assert_same(old, new)
    assert [r for r, _ in routes] == ([expected] if expected else [])


@pytest.mark.filterwarnings("ignore:File may have been truncated")
def test_former_route_errors_are_the_former_errors(tmp_path):
    """Spot checks that the failures above are the real former failures, not two equal new ones."""
    good = _siril_set(tmp_path, 2)
    other = siril_frame(tmp_path / "other.fit", frames_of(np.random.default_rng(2), "random", 1, 130, 9)[0])
    assert outcome(imaging.median_foreground, good + [other])[1] == \
        "ValueError: Frame dimensions differ; use one camera and framing"
    assert outcome(imaging.median_foreground, [])[1].startswith("IndexError")
    assert outcome(imaging.median_foreground, good + [tmp_path / "missing.fit"])[1].startswith("FileNotFoundError")
    assert outcome(imaging.median_foreground, _unquoted_roworder(tmp_path))[1].startswith("VerifyError")
    trunc = write_fits(tmp_path / "trunc.fit", np.zeros((3, 131, 9), np.int16), 16, [("BZERO", "RAW:32768.")],
                       truncate=3000)
    assert outcome(imaging.median_foreground, [trunc] + good)[1].startswith("TypeError")


# ---- 5. astropy's own layout ------------------------------------------------------------------------------------------
def _cards_frame(path, cards, codes):
    Path(path).write_bytes(fits_bytes(cards, codes))
    return Path(path)


STANDARD = ["SIMPLE  =                    T", "BITPIX  =                   16", "NAXIS   =                    3",
            "NAXIS1  =                   30", "NAXIS2  =                   40", "NAXIS3  =                    3",
            "BZERO   =               32768.", "BSCALE  =                   1.", "ROWORDER= 'BOTTOM-UP'"]


def test_median_uses_astropy_layout(tmp_path, routes):
    rng = np.random.default_rng(8)
    path = siril_frame(tmp_path / "light.fit", rng.integers(0, 65536, size=(40, 30, 3)).astype(np.uint16))
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
        frame = imaging._median_frame(path, hdus)
        assert frame == imaging._MedianFrame(os.fspath(path), 2880, (3, 40, 30), 16, 1.0, 32768.0, False)
        hdus[0]._orig_bitpix = -32     # astropy would read another dtype: the explicit reader must not run
        assert imaging._median_frame(path, hdus) is None
    # Cards the fast and the full header parser read differently; the former loop took its dtype from the fast parse
    # and its scaling from the full header, and so does the explicit reader.
    for tag, bzero in (("hierarch", "HIERARCH BZERO = 32768."), ("col7", "BZERO = 32768.")):
        cards = [bzero if c.startswith("BZERO") else c for c in STANDARD]
        paths = [_cards_frame(tmp_path / f"{tag}{i}.fit", cards, stored_codes(
            rng.integers(0, 65536, size=(40, 30, 3)).astype(np.uint16))) for i in range(3)]
        routes.clear()
        assert_same(outcome(ORACLE, paths), outcome(imaging.median_foreground, paths))
        assert [r for r, _ in routes] == ["network"]


def test_median_does_not_memory_map_frames(tmp_path, monkeypatch, routes):
    rng = np.random.default_rng(9)
    siril = _siril_set(tmp_path, 5, 131, 9)
    floats = [float_frame(tmp_path / f"f{i}.fit", rng.random((131, 9, 3)).astype(np.float32)) for i in range(3)]
    expected = [outcome(ORACLE, siril), outcome(ORACLE, floats)]

    def forbidden(*args, **kwargs):
        raise AssertionError("median_foreground memory-mapped a frame")
    monkeypatch.setattr(mmap, "mmap", forbidden)
    assert_same(expected[0], outcome(imaging.median_foreground, siril))
    assert_same(expected[1], outcome(imaging.median_foreground, floats))
    assert [r for r, _ in routes] == ["network", "float"]


# ---- 6. cancellation, self-test, diagnostics, warnings ---------------------------------------------------------------
@pytest.mark.parametrize("kind", ["network", "float"])
def test_median_cancel_points_and_handles(tmp_path, routes, kind):
    rng = np.random.default_rng(10)
    height = 195
    paths = _siril_set(tmp_path, 3, height, 9)
    if kind == "float":
        paths.append(float_frame(tmp_path / "f.fit", rng.random((height, 9, 3)).astype(np.float32)))
    for stop in (1, 2, 3, None):
        routes.clear()
        old, new = outcome(ORACLE, paths, stop), outcome(imaging.median_foreground, paths, stop)
        assert_same(old, new)
        assert [r for r, _ in routes] == [kind]
        assert new[2] == (stop or math.ceil(height / 64))
    for path in paths:     # Windows refuses to delete a file whose handle is still open
        path.unlink()
    assert not any(path.exists() for path in paths)


def test_median_self_test_failure_falls_back(tmp_path, monkeypatch, routes):
    paths = _siril_set(tmp_path, 6, 131, 9)
    old = outcome(ORACLE, paths)
    monkeypatch.setattr(imaging, "_median_self_test", lambda *args: False)
    assert_same(old, outcome(imaging.median_foreground, paths))
    assert [r for r, _ in routes] == ["float"]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 20, 47, 48, 49, 144])
def test_median_self_test_passes_on_this_numpy(n):
    table = imaging.median_conversion_table(1.0, 32768.0)
    assert imaging._median_self_test(n, table, 1.0, 32768.0)


@pytest.mark.parametrize("mutant", ["no comparators", "network of n - 1", "first half only"])
@pytest.mark.parametrize("n", [5, 20, 48])
def test_median_self_test_catches_a_broken_network(tmp_path, monkeypatch, routes, n, mutant):
    """A wrong network is caught by the runtime self-test; the call then takes the float path and is still
    bit-identical.  (Dropping the last comparator of an even n is NOT wrong: it orders the middle pair, whose mean
    does not depend on the order.)"""
    real = imaging.median_pairs
    broken = {"no comparators": lambda count: (),
              "network of n - 1": lambda count: real(count - 1),
              "first half only": lambda count: real(count)[:len(real(count)) // 2]}[mutant]
    monkeypatch.setattr(imaging, "median_pairs", broken)
    assert not imaging._median_self_test(n, imaging.median_conversion_table(1.0, 32768.0), 1.0, 32768.0)
    paths = _siril_set(tmp_path, n, 70, 5)
    assert_same(outcome(ORACLE, paths), outcome(imaging.median_foreground, paths))
    assert [r for r, _ in routes] == ["float"]


def test_median_notes_its_kernel_for_the_timings(tmp_path, monkeypatch):
    notes = []
    from starstack import profiling
    monkeypatch.setattr(profiling, "note", lambda key, value: notes.append((key, value)))
    imaging.median_foreground(_siril_set(tmp_path, 3, 70, 5), lambda: None)
    assert notes == [("foreground_median", {"kernel": imaging.MEDIAN_KERNEL, "frames": 3, "tile_rows": 64,
                                            "comparators": len(imaging.median_pairs(3))})]


@pytest.mark.parametrize("bscale,bzero", [("RAW:1E+39", "RAW:0."), ("RAW:1.", "RAW:1E+39"), ("RAW:1.", "RAW:32768.")])
def test_median_warns_as_before_for_any_scaling(tmp_path, routes, bscale, bzero):
    """Scalings that overflow float32 take the float path (the table is unusable) with the former warnings, no more."""
    paths = _scaled(tmp_path, "sc", [(bscale, bzero)])
    results = []
    for fn in (ORACLE, imaging.median_foreground):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = outcome(fn, paths)
        results.append((result, [str(w.message) for w in caught]))
    assert_same(results[0][0], results[1][0])
    assert results[1][1] == results[0][1]
    assert [r for r, _ in routes] == (["network"] if bzero == "RAW:32768." else ["float"])


def test_median_emits_no_new_warnings(tmp_path):
    rng = np.random.default_rng(12)
    paths = [siril_frame(tmp_path / f"light_{i:05d}.fit", rng.integers(0, 65536, size=(40, 30, 3)).astype(np.uint16),
                         extra=["exptime =                  10."]) for i in range(3)]
    counts = []
    for fn in (ORACLE, imaging.median_foreground):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(paths, lambda: None)
        counts.append(len(caught))
    assert counts == [0, 0]


# ---- 7. X1: frames rotated by orientation.rotate_fits (portrait shots) -----------------------------------------------
@pytest.mark.parametrize("angle", [90, 180, 270])
def test_x1_rotated_siril_frames_take_the_network_route(tmp_path, routes, angle):
    rng = np.random.default_rng(angle)
    values = banded_frames(rng, 5, 131, 37)
    rotated = []
    for i in range(5):
        # one TOP-DOWN source: rotate_fits keeps the row order, so the median sees a mixed set
        source = siril_frame(tmp_path / f"src{i}.fit", values[i], "TOP-DOWN" if i == 3 else "BOTTOM-UP")
        rotated.append(tmp_path / f"light_{i:05d}.fit")
        orientation.rotate_fits(source, rotated[-1], angle)
    with fits.open(rotated[0], memmap=False) as hdus:
        assert hdus[0].data.dtype == np.uint16
        assert hdus[0].data.shape == ((3, 37, 131) if angle in (90, 270) else (3, 131, 37))
    old = outcome(ORACLE, rotated)
    assert_same(old, outcome(imaging.median_foreground, rotated))
    assert [r for r, _ in routes] == ["network"]


# ---- 8. opt-in: row bands of a kept real job's converted frames -------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("AESTER_P1A_REAL_JOB"), reason="set AESTER_P1A_REAL_JOB to a kept job")
def test_kept_job_frame_bands_bit_identical(tmp_path, routes):
    """Copies four row bands of every converted frame of a kept job (read only) into small FITS files; new vs former."""
    frames = sorted((Path(os.environ["AESTER_P1A_REAL_JOB"]) / "work").glob("light_*.fit"))
    assert frames
    with fits.open(frames[0], memmap=True, do_not_scale_image_data=True) as hdus:
        _, height, width = hdus[0].shape
    for y0, y1 in ((0, 192), (896, 1088), (height // 2 // 64 * 64, height // 2 // 64 * 64 + 192),
                   (height - 144, height)):
        band = tmp_path / f"band{y0}"
        band.mkdir()
        paths = []
        for frame in frames:
            with fits.open(frame, memmap=True, do_not_scale_image_data=True) as hdus:
                offset = hdus[0].fileinfo()["datLoc"]
            head = frame.read_bytes()[:offset].decode("ascii")
            old_card, new_card = f"NAXIS2  = {height:>20}", f"NAXIS2  = {y1 - y0:>20}"
            assert head.count(old_card) == 1
            with open(frame, "rb") as raw:
                parts = []
                for plane in range(3):
                    raw.seek(offset + (plane * height + y0) * width * 2)
                    parts.append(raw.read((y1 - y0) * width * 2))
            data = b"".join(parts)
            paths.append(band / frame.name)
            paths[-1].write_bytes(head.replace(old_card, new_card).encode("ascii") + data + b"\0" * (-len(data) % 2880))
        routes.clear()
        assert_same(outcome(ORACLE, paths), outcome(imaging.median_foreground, paths))
        assert [r for r, _ in routes] == ["network"]
