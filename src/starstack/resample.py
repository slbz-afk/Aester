"""Registration-grade image warping in pure NumPy (Lanczos / cubic, block processed).

Why not Pillow: ``Image.transform(..., BICUBIC)`` shifts star centroids by up
to ~0.09 px at fractional offsets (measured on synthetic stars, RUN_LOG §20),
which is the same order as the registration accuracy this project is
built to keep; its BILINEAR is unbiased but broadens stars by a few
percent.  The separable kernels here are evaluated at the exact source
coordinate of every output pixel (a general ``mapping``, e.g. a radial
distortion), so they have no phase bias, and a Lanczos-3 kernel keeps the
star profile within ~0.5 % — comparable to Siril's own lanczos4 export.

Conventions: TOP-DOWN images (``image[y, x]``), zero-based pixel centres;
``mapping(xy)`` receives output pixel centres (N, 2) and returns source
coordinates (N, 2) in the same convention.  Source positions outside the
half-open pixel area [-0.5, width-0.5) x [-0.5, height-0.5) are filled with
``fill`` (NaN by default).  For positions inside that area, kernel taps
outside the image are clamped to the edge pixel.  This edge extension is
not proof that the complete interpolation footprint was observed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RESAMPLE_VERSION", "DEFAULT_KERNEL", "lanczos_kernel", "cubic_kernel", "warp_image", "warp_planes",
           "warp_planes_fused"]

RESAMPLE_VERSION = "1.0.1"
DEFAULT_KERNEL = "cubic"  # unbiased, no measurable star broadening, ~0.5 s per Mpx (RUN_LOG §20)


def lanczos_kernel(t: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos window ``sinc(t) sinc(t/a)`` for |t| < a, zero outside."""
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    inside = np.abs(t) < a
    x = t[inside]
    out[inside] = np.sinc(x) * np.sinc(x / a)
    out[np.abs(t) < 1e-12] = 1.0
    return out


def cubic_kernel(t: np.ndarray, b: float = -0.5) -> np.ndarray:
    """Keys cubic convolution kernel (a = -0.5), support |t| < 2."""
    t = np.abs(np.asarray(t, dtype=np.float64))
    out = np.zeros_like(t)
    near = t < 1
    far = (t >= 1) & (t < 2)
    out[near] = (b + 2) * t[near] ** 3 - (b + 3) * t[near] ** 2 + 1
    out[far] = b * t[far] ** 3 - 5 * b * t[far] ** 2 + 8 * b * t[far] - 4 * b
    return out


def _taps(kind: str) -> tuple[int, callable]:
    if kind == "lanczos3":
        return 3, lambda t: lanczos_kernel(t, 3)
    if kind == "lanczos2":
        return 2, lambda t: lanczos_kernel(t, 2)
    if kind == "cubic":
        return 2, cubic_kernel
    if kind == "bilinear":
        return 1, lambda t: np.clip(1.0 - np.abs(t), 0.0, None)
    raise ValueError(f"unknown kernel {kind!r}")


def warp_image(image: np.ndarray, mapping, *, kernel: str = DEFAULT_KERNEL, block: int = 128, fill: float = np.nan,
               output_shape: tuple[int, int] | None = None, normalise: bool = True) -> np.ndarray:
    """Resample ``image`` so that output pixel ``p`` takes the value of the source at ``mapping(p)``.

    ``normalise`` renormalises the kernel weights per pixel (Lanczos weights
    do not sum exactly to one), which keeps flat fields flat to 1e-6.
    """
    src = np.asarray(image, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError("warp_image expects a 2-D plane; use warp_planes for cubes")
    height, width = src.shape
    out_h, out_w = output_shape or (height, width)
    radius, kernel_fn = _taps(kernel)
    out = np.full((out_h, out_w), fill, dtype=np.float32)
    offsets = np.arange(-radius + 1, radius + 1)          # e.g. -2..3 for lanczos3 (2a taps)
    for y0 in range(0, out_h, block):
        y1 = min(out_h, y0 + block)
        for x0 in range(0, out_w, block):
            x1 = min(out_w, x0 + block)
            ys, xs = np.mgrid[y0:y1, x0:x1]
            pts = np.column_stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)])
            s = np.asarray(mapping(pts), dtype=np.float64)
            sx, sy = s[:, 0], s[:, 1]
            fx, fy = np.floor(sx), np.floor(sy)
            ix, iy = fx.astype(np.int64), fy.astype(np.int64)
            tx, ty = sx - fx, sy - fy
            # a sample whose *position* lies inside the image is resampled; kernel taps that reach beyond the border are
            # clamped to the edge pixel (edge extension), so the outermost rows/columns are not left empty (RUN_LOG §25:
            # the warp-back used to leave 1-px empty lines where the magnified corners touch the grid).  Positions outside
            # the image keep ``fill`` — nothing is invented beyond the data.
            valid = np.isfinite(sx) & np.isfinite(sy) & (sx >= -0.5) & (sx < width - 0.5) & (sy >= -0.5) & (sy < height - 0.5)
            if not valid.any():
                continue
            ixv, iyv, txv, tyv = ix[valid], iy[valid], tx[valid], ty[valid]
            wx = np.stack([kernel_fn(txv - o) for o in offsets], axis=1)     # (n, taps)
            wy = np.stack([kernel_fn(tyv - o) for o in offsets], axis=1)
            acc = np.zeros(len(ixv), dtype=np.float64)
            wsum = np.zeros(len(ixv), dtype=np.float64)
            for j, oy in enumerate(offsets):
                rows_index = np.clip(iyv + oy, 0, height - 1)
                for i, ox in enumerate(offsets):
                    w = wx[:, i] * wy[:, j]
                    acc += w * src[rows_index, np.clip(ixv + ox, 0, width - 1)]
                    wsum += w
            values = acc / wsum if normalise else acc
            block_out = np.full(len(sx), fill, dtype=np.float32)
            block_out[valid] = values.astype(np.float32)
            out[y0:y1, x0:x1] = block_out.reshape(y1 - y0, x1 - x0)
    return out


def warp_planes(cube: np.ndarray, mapping, **kwargs) -> np.ndarray:
    """Apply ``warp_image`` to every plane of a ``(planes, H, W)`` cube."""
    cube = np.asarray(cube)
    if cube.ndim == 2:
        return warp_image(cube, mapping, **kwargs)
    return np.stack([warp_image(plane, mapping, **kwargs) for plane in cube], axis=0)


def _flat_planes(src: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(P, H*W)`` C-contiguous float32 rows over the planes of a (P, H, W) TOP-DOWN float32 cube, and whether
    they are stored row-flipped (BOTTOM-UP).

    No copy when the cube is C-contiguous or a row-flipped view of a C-contiguous cube (``imaging.read_rgb``'s
    planar BOTTOM-UP array); otherwise one contiguous copy.  ``np.take`` needs a C-contiguous source (it copies a
    strided one on every call), hence the normalisation.  Elements keep their float32 bits.
    """
    planes, height, width = src.shape
    flipped = src[:, ::-1]
    if height > 1 and flipped.flags.c_contiguous:
        return flipped.reshape(planes, height * width), True
    return np.ascontiguousarray(src).reshape(planes, height * width), False


def _fused_block(flat, flipped, height, width, mapping, kernel_fn, offsets, y0, y1, x0, x1, fill, normalise):
    """One output block for every plane: ``(P, y1-y0, x1-x0)`` float32, or None when no position is inside.

    Per element exactly warp_image's operations: mapping, floor/fraction, validity, kernel weights, then for
    taps j (rows, outer) and i (columns, inner) ``acc += w * value`` and ``wsum += w`` in float64, ``acc / wsum``
    (or ``acc``), cast to float32.  Only what does not depend on the plane is computed once.  (Blocks are
    independent and cover disjoint output rectangles: the unit a later ``workers=`` pool would hand out.)
    """
    ys, xs = np.mgrid[y0:y1, x0:x1]
    pts = np.column_stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)])
    s = np.asarray(mapping(pts), dtype=np.float64)
    sx, sy = s[:, 0], s[:, 1]
    fx, fy = np.floor(sx), np.floor(sy)
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    tx, ty = sx - fx, sy - fy
    valid = np.isfinite(sx) & np.isfinite(sy) & (sx >= -0.5) & (sx < width - 0.5) & (sy >= -0.5) & (sy < height - 0.5)
    if not valid.any():
        return None
    ixv, iyv, txv, tyv = ix[valid], iy[valid], tx[valid], ty[valid]
    wx = np.stack([kernel_fn(txv - o) for o in offsets], axis=1)     # (n, taps), the same calls as warp_image
    wy = np.stack([kernel_fn(tyv - o) for o in offsets], axis=1)
    nplanes, n = flat.shape[0], len(ixv)
    cols = [np.clip(ixv + ox, 0, width - 1) for ox in offsets]      # the column indices warp_image clips per tap
    acc = np.zeros((nplanes, n), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    prod = np.empty((nplanes, n), dtype=np.float64)
    for j, oy in enumerate(offsets):
        rows_index = np.clip(iyv + oy, 0, height - 1)
        row_base = ((height - 1) - rows_index) * width if flipped else rows_index * width
        for i in range(len(offsets)):
            w = wx[:, i] * wy[:, j]
            vals = np.take(flat, row_base + cols[i], axis=1)          # (P, n) float32 = src[p][rows_index, cols[i]]
            np.multiply(w, vals, out=prod)                            # w * value in float64, w first as before
            acc += prod
            wsum += w
    values = acc / wsum if normalise else acc
    block_out = np.full((nplanes, len(sx)), fill, dtype=np.float32)
    block_out[:, valid] = values.astype(np.float32)
    return block_out.reshape(nplanes, y1 - y0, x1 - x0)


def warp_planes_fused(cube, mapping, *, kernel: str = DEFAULT_KERNEL, block: int = 128, fill: float = np.nan,
                      output_shape: tuple[int, int] | None = None, normalise: bool = True,
                      out: np.ndarray | None = None) -> np.ndarray:
    """``np.stack([warp_image(p, mapping, ...) for p in cube])`` for a ``(P, H, W)`` cube, bit for bit, in one pass.

    The mapping is called once per output block (not once per plane per block): it must be a pure function of
    its points (``RadialDistortion.apply`` and closures over it are).  ``out`` (optional) is a float32
    ``(P, out_h, out_w)`` array or view that does not overlap ``cube`` (e.g. a TOP-DOWN planar view of the
    caller's final BOTTOM-UP ``(H, W, P)`` array); it is set to ``fill`` first, as ``np.full`` would, and returned.

    Anything but a non-empty 3-D cube goes to ``warp_planes`` unchanged (same results and errors); ``out`` is
    refused there.  The checks run in warp_image's order as it runs them on plane 0 (conversion, output shape,
    kernel, fill, block); the other planes are converted after those.  Floating-point warnings arise op by op over
    all planes, so their count and cross-plane order (and the first exception under ``np.errstate(all="raise")``)
    can differ from the per-plane form; the values cannot.  ``warp_image`` / ``warp_planes`` stay the reference
    (P1a G-1).
    """
    cube = np.asarray(cube)
    if cube.ndim != 3 or cube.shape[0] == 0:
        if out is not None:
            raise ValueError("out requires a non-empty (planes, H, W) cube")
        return warp_planes(cube, mapping, kernel=kernel, block=block, fill=fill, output_shape=output_shape,
                           normalise=normalise)
    # warp_image's own checks, in its order, as it runs them on plane 0: convert the plane, output shape, kernel,
    # output allocation and fill, block grid.  The other planes are converted after these.
    height, width = np.asarray(cube[0], dtype=np.float32).shape
    out_h, out_w = output_shape or (height, width)
    radius, kernel_fn = _taps(kernel)
    nplanes = cube.shape[0]
    if out is None:
        out = np.full((nplanes, out_h, out_w), fill, dtype=np.float32)
    else:
        if out.shape != (nplanes, out_h, out_w) or out.dtype != np.float32:
            raise ValueError("out must be a float32 (planes, out_h, out_w) array")
        np.copyto(out, fill, casting="unsafe")                        # what np.full does with the fill value
    offsets = np.arange(-radius + 1, radius + 1)
    row_starts = range(0, out_h, block)
    src = np.asarray(cube, dtype=np.float32)
    flat, flipped = _flat_planes(src)
    for y0 in row_starts:
        y1 = min(out_h, y0 + block)
        for x0 in range(0, out_w, block):
            x1 = min(out_w, x0 + block)
            values = _fused_block(flat, flipped, height, width, mapping, kernel_fn, offsets, y0, y1, x0, x1,
                                  fill, normalise)
            if values is not None:
                out[:, y0:y1, x0:x1] = values
    return out
