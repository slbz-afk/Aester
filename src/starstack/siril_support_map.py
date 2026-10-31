"""Source coordinates for the pinned Siril 1.4.4 registration export.

Contract: ordinary FITS images, equal input/output size, scale=1, current
framing; stored frame and reference homographies are frozen *before* export.
The optional header is the actual D1 2 WCS file (DISTO_MAP_D2S), not an ideal
radial-model dictionary. SIP interpolates a float32 coordinate grid and then
resamples the image once. A matrix expressed in another coordinate convention
must not be silently substituted for the stored sequence matrix.

This reproduces the source-map algorithm; it is not a bitwise promise across
OpenCV/compiler builds. The support caller must cover phase/roundoff guards.
See outputs/support-integration-20260919/KERNEL_AUDIT.md for source and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
import threading

import numpy as np

SIRIL_SUPPORT_MAP_VERSION = "1.0.0"
__all__ = ["SIRIL_SUPPORT_MAP_VERSION", "ExportParameters", "export_parameters", "make_export_mapping",
           "distortion_grid", "release_distortion_grid"]

_GRID_LOCK = threading.Lock()
_GRID_FAILED = {}                        # (width, height, spec) -> message of a failed build (P1c F-2)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _matrix(value, name):
    arr = np.asarray(value)
    if arr.shape != (3, 3) or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real 3x3 matrix")
    try:
        arr = arr.astype(np.float64, copy=True)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real 3x3 matrix") from exc
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite real 3x3 matrix")
    # Homographies are scale-equivalent. Normalize before inversion so ordinary
    # equivalent scaling does not make the determinant test overflow/underflow.
    scale = np.max(np.abs(arr))
    if scale == 0:
        raise ValueError(f"{name} is singular")
    try:
        inv = np.linalg.inv(arr / scale)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} is singular") from exc
    if not np.isfinite(inv).all():
        raise ValueError(f"{name} inverse is not finite")
    return arr


def _sip_spec(header):
    try:
        if not str(header["CTYPE1"]).endswith("-SIP") or not str(header["CTYPE2"]).endswith("-SIP"):
            raise ValueError("WCS header must declare SIP on both axes")
        cx = _number(header["CRPIX1"], "CRPIX1") - 1.
        cy = _number(header["CRPIX2"], "CRPIX2") - 1.
        axes = []
        for prefix in ("AP", "BP"):
            order = header[prefix + "_ORDER"]
            if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral) or not 0 <= order <= 5:
                raise ValueError("inverse SIP orders must be integers from 0 through 5")
            # Preserve total-degree grouping used by Siril's double evaluator.
            terms = tuple((p, total-p, _number(header.get(f"{prefix}_{p}_{total-p}", 0.), f"{prefix}_{p}_{total-p}"))
                          for total in range(int(order)+1) for p in range(total, -1, -1))
            axes.append((int(order), terms))
        return cx, cy, tuple(axes)
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError("a complete inverse-SIP WCS header is required") from exc


@lru_cache(maxsize=1)
def _distortion_grid(width, height, spec):
    """Share one read-only source-map pair; fill without full-image doubles."""
    cx, cy, axes = spec
    maps = np.empty((2, height, width), dtype=np.float32)
    u = np.arange(width, dtype=np.float64)[None, :] - cx
    upowers = [np.ones_like(u)] + [u ** i for i in range(1, 6)]
    # Working doubles are bounded to a few rows, independent of image height.
    block_rows = max(1, min(128, 524288 // width))
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, height, block_rows):
            end = min(start + block_rows, height)
            v = np.arange(start, end, dtype=np.float64)[:, None] - cy
            vpowers = [np.ones_like(v)] + [v ** i for i in range(1, 6)]
            for axis, ((order, terms), base, centre) in enumerate(zip(axes, (u, v), (cx, cy))):
                value = np.broadcast_to(base, (end-start, width)).copy()
                for total in range(order+1):
                    group = np.zeros_like(value)
                    for p, q, coefficient in terms:
                        if p+q == total and coefficient != 0.:
                            group += coefficient * upowers[p] * vpowers[q]
                    value += group
                maps[axis, start:end] = value + centre
    if not np.isfinite(maps).all():
        raise ValueError("SIP source grid is not finite in float32")
    maps.flags.writeable = False
    return maps


def distortion_grid(width, height, spec, *, check=None):
    """The shared SIP source-map pair, built at most once even with concurrent first callers.

    Efficiency P1c F-2: ``check`` (a frame-pool worker's probe) runs before the wait for the lock, which another
    worker may hold for a whole build. A build that failed because the grid is not finite is remembered for its
    (width, height, spec) until release_distortion_grid(): later calls raise the same error at once instead of each
    rebuilding it under the lock (2.26 s and 366 MB at 45.7 MP)."""
    if check is not None:
        check()
    key = (width, height, spec)
    with _GRID_LOCK:
        failed = _GRID_FAILED.get(key)
        if failed is not None:
            raise ValueError(failed)
        try:
            return _distortion_grid(width, height, spec)
        except ValueError as exc:                # _distortion_grid's only ValueError: the grid is not finite
            _GRID_FAILED[key] = str(exc)
            raise


def release_distortion_grid() -> None:
    """Drop the cached 2*H*W float32 grid (and any remembered failed build); arrays still referenced elsewhere stay
    alive until released there."""
    with _GRID_LOCK:
        _distortion_grid.cache_clear()
        _GRID_FAILED.clear()


@dataclass(frozen=True, eq=False)
class ExportParameters:
    """What make_export_mapping computes before it builds its closure (read-only inverse)."""
    inverse: np.ndarray
    spec: tuple | None
    width: int
    height: int


def export_parameters(h_frame, h_reference, width, height, wcs_header=None) -> ExportParameters:
    """make_export_mapping's validation and set-up: same checks, same order, same messages."""
    for value, name in ((width, "width"), (height, "height")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 2 <= value < 32767:
            raise ValueError(f"{name} must be an integer from 2 through 32766")
    width, height = int(width), int(height)
    frame = _matrix(h_frame, "frame homography")
    reference = _matrix(h_reference, "reference homography")
    flip = np.array([[1., 0., 0.], [0., -1., height-1.], [0., 0., 1.]])
    effective = np.linalg.inv(reference) @ frame
    # Siril matrix convention is used as stored; do not undo the +0.5 adapter.
    inverse = np.linalg.inv(flip @ effective @ flip)
    if not np.isfinite(inverse).all():
        raise ValueError("combined homography inverse is not finite")
    spec = _sip_spec(wcs_header) if wcs_header is not None else None
    inverse.flags.writeable = False
    return ExportParameters(inverse, spec, width, height)


def make_export_mapping(h_frame, h_reference, width, height, wcs_header=None):
    """Create ``mapping(points[N,2]) -> source[N,2]`` in TOP-DOWN centres.

    No disk access occurs in the callable. An unsupported projective/SIP map
    point returns ``(-1, height)``, outside the source. Its footprint must be
    rejected by the caller, never clipped into valid pixels. Input points must
    be finite; call in bounded chunks. SIP arrays are lazy and shared for one
    repeated header/shape, requiring 8*width*height bytes on first use.
    """
    params = export_parameters(h_frame, h_reference, width, height, wcs_header)
    width, height, inverse, spec = params.width, params.height, params.inverse, params.spec

    def mapping(points):
        p = np.asarray(points)
        if p.ndim != 2 or p.shape[1] != 2 or p.dtype.kind not in "iuf":
            raise ValueError("points must be a finite real Nx2 array")
        with np.errstate(over="ignore", invalid="ignore"):
            p = p.astype(np.float64, copy=False)
        if not np.isfinite(p).all():
            raise ValueError("points must be a finite real Nx2 array")
        answer = np.empty((len(p), 2), dtype=np.float64)
        answer[:] = (-1., float(height))
        if not len(p):
            return answer
        xb, yb = p[:, 0], height-1.-p[:, 1]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            den = xb*inverse[2, 0] + yb*inverse[2, 1] + inverse[2, 2]
            reciprocal = 1./den
            if spec is not None:
                reciprocal = reciprocal.astype(np.float32)
            sx = (xb*inverse[0, 0] + yb*inverse[0, 1] + inverse[0, 2])*reciprocal
            sy = (xb*inverse[1, 0] + yb*inverse[1, 1] + inverse[1, 2])*reciprocal
            if spec is not None:
                sx, sy = sx.astype(np.float32), sy.astype(np.float32)
        valid = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(den) & (den != 0.)
        if spec is None:
            answer[valid, 0] = sx[valid]
            answer[valid, 1] = height-1.-sy[valid]
            return answer
        # Guard before integer narrowing; the grid interpolation needs all four
        # integer map locations even at zero weight, as the engine does.
        valid &= (sx >= 0.) & (sx < width-1.) & (sy >= 0.) & (sy < height-1.)
        positions = np.flatnonzero(valid)
        if not len(positions):
            return answer
        sx, sy = sx[positions], sy[positions]
        ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        c1, c2 = sx-ix.astype(np.float32), sy-iy.astype(np.float32)
        one = np.float32(1.)
        w11, w12 = (one-c1)*(one-c2), c1*(one-c2)
        w21, w22 = (one-c1)*c2, c1*c2
        grid = distortion_grid(width, height, spec)
        # One coordinate component at a time avoids a strided two-component
        # advanced-indexing temporary. Keep float32 operations left-associated
        # exactly as before; changing their order can move footprint anchors.
        for axis in range(2):
            out = ((grid[axis, iy, ix]*w11 + grid[axis, iy, ix+1]*w12)
                   + grid[axis, iy+1, ix]*w21) + grid[axis, iy+1, ix+1]*w22
            if axis == 0:
                answer[positions, 0] = out
            else:
                answer[positions, 1] = height-1.-out.astype(np.float64)
        return answer

    mapping.contract = {"version": SIRIL_SUPPORT_MAP_VERSION, "engine": "Siril 1.4.4",
                        "framing": "current", "scale": 1, "width": width, "height": height,
                        "distortion": "D1 2 / DISTO_MAP_D2S" if spec is not None else "none",
                        "unsupported_sentinel": [-1., float(height)]}
    mapping.export_parameters = params
    return mapping
