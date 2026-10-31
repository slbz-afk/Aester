"""Lossy display-linear tone curve with one multiplier per RGB pixel.

DeepSeek implemented the fixed contract; Codex tightened integer boundaries.
No camera calibration, gamma encoding, masks or scientific-master edits here.
"""
import math

import numpy as np

_MAX_ROWS = 128


def _as_finite_float(name, value, lo, hi):
    # Compare integers before conversion: 10**20+1 rounds to 1e20 in float64.
    if type(value) in (int, float) and not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo!r} and {hi!r}")
    if type(value) is int:
        try:
            result = float(value)
        except OverflowError:
            raise ValueError(f"{name} cannot be represented as float") from None
    elif type(value) is float:
        result = value
    else:
        raise ValueError(f"{name} must be a builtin int or builtin float")

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < lo or result > hi:
        raise ValueError(f"{name} must be between {lo!r} and {hi!r}")
    return result


def linked_tone(rgb, *, black, white, asinh_strength, midtone):
    if type(rgb) is not np.ndarray:
        raise ValueError("rgb must be a plain numpy.ndarray")
    if rgb.dtype != np.float32:
        raise ValueError("rgb dtype must be float32")
    if rgb.ndim != 3:
        raise ValueError("rgb must have shape H x W x 3")
    height, width, channels = rgb.shape
    if channels != 3 or height < 2 or width < 2:
        raise ValueError("rgb must have shape H x W x 3 with H >= 2 and W >= 2")

    black_f = _as_finite_float("black", black, -1e20, 1e20)
    white_f = _as_finite_float("white", white, -1e20, 1e20)
    k = _as_finite_float("asinh_strength", asinh_strength, 0.0, 100.0)
    m = _as_finite_float("midtone", midtone, 1e-4, 1.0 - 1e-4)

    scale = white_f - black_f
    if not math.isfinite(scale) or scale < 1e-20:
        raise ValueError("white - black must be at least 1e-20")

    result = np.empty((height, width, channels), dtype=np.float32)

    with np.errstate(all="ignore"):
        for row0 in range(0, height, _MAX_ROWS):
            row1 = min(row0 + _MAX_ROWS, height)
            block = rgb[row0:row1]

            block64 = block.astype(np.float64, copy=True)
            if not np.isfinite(block64).all():
                raise ValueError("rgb must contain only finite values")

            u = (block64 - black_f) / scale
            np.maximum(u, 0.0, out=u)
            if not np.isfinite(u).all():
                raise ValueError("non-finite intermediate value")

            p = u.max(axis=2)
            if not np.isfinite(p).all():
                raise ValueError("non-finite intermediate value")

            q = np.minimum(p, 1.0)

            if k > 0.0:
                a = np.arcsinh(k * q) / np.arcsinh(k)
            else:
                a = q
            if not np.isfinite(a).all():
                raise ValueError("non-finite intermediate value")

            denom = (2.0 * m - 1.0) * a - m
            if not np.isfinite(denom).all():
                raise ValueError("non-finite intermediate value")

            v = ((m - 1.0) * a) / denom
            if not np.isfinite(v).all():
                raise ValueError("non-finite intermediate value")

            gain = np.zeros_like(p)
            np.divide(v, p, out=gain, where=p > 0.0)
            if not np.isfinite(gain).all():
                raise ValueError("non-finite intermediate value")

            out_block = u * gain[..., np.newaxis]
            if not np.isfinite(out_block).all():
                raise ValueError("non-finite intermediate value")

            out32 = out_block.astype(np.float32)
            if not np.isfinite(out32).all():
                raise ValueError("non-finite output value")

            result[row0:row1] = out32

    return result
