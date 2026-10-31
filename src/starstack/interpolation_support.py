"""Validity-aware interpolation footprint checks.

These are explicitly defined bilinear and cubic (Keys a=-0.5) kernels. They
are NOT a proof or claim about the kernel actually used by Siril.

Callers must propagate validity at every resampling stage; a net-coordinate
warp is insufficient because interpolation footprints can mix valid and
invalid source pixels.
"""

import numpy as np

__all__ = ["SUPPORT_VERSION", "sample_support"]

SUPPORT_VERSION = "1.0.0"

_BILINEAR_OFFSETS = (0, 1)
_CUBIC_OFFSETS = (-1, 0, 1, 2)
_KEYS_A = -0.5


def _validate_validity(validity):
    try:
        arr = np.asarray(validity)
    except (TypeError, ValueError) as exc:
        raise ValueError("validity must be a nonempty 2-D boolean array") from exc
    if arr.dtype != np.dtype(bool):
        raise ValueError("validity must have boolean dtype, not %r" % (arr.dtype,))
    if arr.ndim != 2:
        raise ValueError("validity must be 2-D")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("validity must be nonempty")
    return arr


def _validate_points(points):
    try:
        arr = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be a real numeric array of shape (N, 2)") from exc
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(
            "points must be real numeric (no bool, complex, strings, or object)"
        )
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError("points must contain only finite coordinates")
    # Preserve the input precision until after the positional test. Narrowing
    # a finite extended float first can overflow or move a border coordinate.
    return arr


def _validate_kernel(kernel):
    if not isinstance(kernel, str) or kernel not in ("bilinear", "cubic"):
        raise ValueError("kernel must be 'bilinear' or 'cubic'")
    return kernel


def _validate_boundary(boundary):
    if not isinstance(boundary, str) or boundary not in ("clamp", "strict"):
        raise ValueError("boundary must be 'clamp' or 'strict'")
    return boundary


def _bilinear_weights(t, offsets):
    off = np.asarray(offsets, dtype=np.float64)
    return np.maximum(1.0 - np.abs(t[:, None] - off[None, :]), 0.0)


def _cubic_weights(t, offsets):
    off = np.asarray(offsets, dtype=np.float64)
    z = np.abs(t[:, None] - off[None, :])
    a = _KEYS_A
    near = (a + 2.0) * z**3 - (a + 3.0) * z**2 + 1.0
    far = a * z**3 - 5.0 * a * z**2 + 8.0 * a * z - 4.0 * a
    return np.where(z < 1.0, near, np.where((z >= 1.0) & (z < 2.0), far, 0.0))


def sample_support(validity, points, *, kernel="cubic", boundary="clamp") -> np.ndarray:
    """Return per-point validity for explicitly defined interpolation kernels.

    The bilinear and cubic (Keys a=-0.5) kernels implemented here are
    explicitly defined for this helper; they are NOT a proof or claim about
    the kernel actually used by Siril.

    The caller must propagate validity at every resampling stage. A
    net-coordinate warp is insufficient because an interpolation footprint
    can mix valid and invalid source pixels.

    Parameters
    ----------
    validity : 2-D boolean numpy-compatible array
        Nonempty source-pixel validity mask, shape (height, width).
    points : real numeric numpy-compatible array, shape (N, 2)
        x, y zero-based pixel-centre coordinates, top-down.
    kernel : {"bilinear", "cubic"}
        Explicit interpolation kernel.
    boundary : {"clamp", "strict"}
        Edge treatment for interpolation taps.

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
        True only when every tap with a nonzero weight references valid
        source pixels (after clipping for ``boundary="clamp"``).
    """
    valid = _validate_validity(validity)
    pts = _validate_points(points)
    kernel = _validate_kernel(kernel)
    boundary = _validate_boundary(boundary)

    n = pts.shape[0]
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out

    h, w = valid.shape
    x = pts[:, 0]
    y = pts[:, 1]

    # Position test uses comparisons only, so a giant finite coordinate is
    # rejected here and never reaches floor/integer conversion.
    inside_pos = (
        (x >= -0.5)
        & (x < (w - 0.5))
        & (y >= -0.5)
        & (y < (h - 0.5))
    )
    if not np.any(inside_pos):
        return out

    xv = x[inside_pos].astype(np.float64, copy=False)
    yv = y[inside_pos].astype(np.float64, copy=False)
    fx = np.floor(xv)
    fy = np.floor(yv)
    ix0 = fx.astype(np.intp)
    iy0 = fy.astype(np.intp)
    tx = xv - fx
    ty = yv - fy

    if kernel == "bilinear":
        offsets = _BILINEAR_OFFSETS
        x_weights = _bilinear_weights(tx, offsets)
        y_weights = _bilinear_weights(ty, offsets)
    else:
        offsets = _CUBIC_OFFSETS
        x_weights = _cubic_weights(tx, offsets)
        y_weights = _cubic_weights(ty, offsets)

    result = np.ones(xv.shape[0], dtype=bool)
    for xi, xoff in enumerate(offsets):
        wx = x_weights[:, xi]
        ix = ix0 + xoff
        if boundary == "clamp":
            ix_safe = np.clip(ix, 0, w - 1)
        else:
            x_inside = (ix >= 0) & (ix < w)

        for yi, yoff in enumerate(offsets):
            wy = y_weights[:, yi]
            # Exact zero test: no tolerance, and negative weights contribute.
            contrib = (wx * wy) != 0.0
            iy = iy0 + yoff

            if boundary == "clamp":
                iy_safe = np.clip(iy, 0, h - 1)
                tap_ok = valid[iy_safe, ix_safe]
            else:
                inside = x_inside & (iy >= 0) & (iy < h)
                tap_ok = np.zeros_like(inside, dtype=bool)
                if np.any(inside):
                    tap_ok[inside] = valid[iy[inside], ix[inside]]

            result &= (~contrib) | tap_ok

    out[inside_pos] = result
    return out
