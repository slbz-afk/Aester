"""Bit-identical fast paths for the HueSatMap looks (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md A3).

``prepare`` validates a table once; ``lookup`` then performs exactly the arithmetic of
``experimental_huesat.lookup_huesat`` (same operations in the same order, so the same float64 results) without
re-validating the table and its inputs on every block.  ``rgb_to_hsv`` / ``hsv_to_rgb`` equal
``experimental_dcp.rgb_to_hsv`` / ``hsv_to_rgb`` value for value; they select the per-pixel branch with ``np.choose``
instead of one masked assignment per branch.  The reference implementations stay untouched (strict Z8 chain) and the
tests compare against them.  Callers pass finite values (the display transform checks its input and the look output).
"""
from __future__ import annotations

import numpy as np

# hsv_to_rgb branch table over the candidates (v, t, p, q): sector k → (R, G, B)
_R = np.array([0, 3, 2, 2, 1, 0], dtype=np.intp)
_G = np.array([1, 0, 0, 3, 2, 2], dtype=np.intp)
_B = np.array([2, 2, 1, 0, 0, 3], dtype=np.intp)


def prepare(table):
    """(float64 table, H, S) after the checks ``lookup_huesat`` applies to the table."""
    if type(table) is not np.ndarray or table.dtype.type not in (np.float32, np.float64):
        raise ValueError("table must be an exact float32 or float64 numpy.ndarray")
    if table.ndim != 3 or table.shape[2] != 3:
        raise ValueError("table must have shape (H, S, 3)")
    H, S = int(table.shape[0]), int(table.shape[1])
    if H < 2 or S < 2:
        raise ValueError("table must have H >= 2 and S >= 2")
    if not np.isfinite(table).all():
        raise ValueError("table entries must be finite")
    if not np.all(table[..., 1:] >= 0):
        raise ValueError("table multipliers must be nonnegative")
    t = np.ascontiguousarray(table, dtype=np.float64)
    t.setflags(write=False)
    return t, H, S


def lookup(prepared, hue, sat):
    """Bilinear hue × saturation interpolation; ``hue`` float64 degrees, ``sat`` float64 in [0, 1]."""
    t, H, S = prepared
    flat = t.reshape(-1, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        wrapped = np.remainder(hue, 360.0)
        h = (wrapped * H) / 360.0
        h = np.where(h >= H, 0.0, h)
        i0 = np.floor(h).astype(np.intp)
        i1 = (i0 + 1) % H
        fh = h - i0
        s = sat * (S - 1)
        j0 = np.minimum(np.floor(s).astype(np.intp), S - 2)
        j1 = j0 + 1
        fs = s - j0
        w00 = (1.0 - fh) * (1.0 - fs)
        w01 = (1.0 - fh) * fs
        w10 = fh * (1.0 - fs)
        w11 = fh * fs
        # ((v00·w00 + v01·w01) + v10·w10) + v11·w11, accumulated in the reference order
        out = flat[i0 * S + j0] * w00[..., None]
        out += flat[i0 * S + j1] * w01[..., None]
        out += flat[i1 * S + j0] * w10[..., None]
        out += flat[i1 * S + j1] * w11[..., None]
    return out


def rgb_to_hsv(rgb):
    """Float64 nonnegative linear RGB → (hue degrees, saturation, value), as ``experimental_dcp.rgb_to_hsv``."""
    v = rgb.max(axis=-1)
    lo = rgb.min(axis=-1)
    d = v - lo
    s = np.divide(d, v, out=np.zeros_like(v), where=v != 0)
    channel = rgb.argmax(axis=-1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    num = np.choose(channel, (g - b, b - r, r - g))
    offset = np.choose(channel, (0.0, 2.0, 4.0))
    nonzero = d != 0
    q = np.divide(num, d, out=np.zeros_like(v), where=nonzero)
    h = np.where(nonzero, (q + offset) * 60., 0.0)
    return np.remainder(h, 360), s, v


def hsv_to_rgb(h, s, v):
    """Inverse with saturation above one allowed, as ``experimental_dcp.hsv_to_rgb``."""
    sector = np.remainder(h, 360.) / 60.
    floor = np.floor(sector)
    i = floor.astype(np.intp) % 6
    f = sector - floor
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    choices = (v, t, p, q)
    out = np.empty(h.shape + (3,), dtype=np.float64)
    out[..., 0] = np.choose(_R[i], choices)
    out[..., 1] = np.choose(_G[i], choices)
    out[..., 2] = np.choose(_B[i], choices)
    return out
