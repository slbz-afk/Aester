"""Anchor footprint support for integer source anchors.

Two NumPy-only helpers are exposed:

``build_anchor_grid``
    Precompute a read-only boolean grid from the ``uint64`` summed-area table
    of invalid source pixels.  ``grid[y, x]`` is true exactly when the
    inclusive footprint ``y-before .. y+after`` by ``x-before .. x+after`` of
    anchor ``(y, x)`` lies completely inside the source and contains no
    invalid source pixel.  Construction uses only row-length unsigned
    differences, so no full-image difference temporary is created.

``anchor_support``
    Look up ``(N, 2)`` points in such a grid.  Bounds checks run on the
    original coordinates before flooring or narrowing to integer indices, so
    huge finite coordinates and ``uint64`` maxima can never wrap into a valid
    index.

No interpolation is performed, no floating point arithmetic participates in
grid construction, the summed-area table is never modified, and querying does
not depend on the table at all.
"""

import numpy as np

from .support_bounds import _validate_validity

__all__ = ["build_anchor_grid", "anchor_grid_from_validity", "anchor_support"]


def _require_plain_int(name, value, minimum, *, strict):
    """Validate an exact built-in ``int`` parameter range."""
    if type(value) is not int:
        raise ValueError("%s must be an exact built-in int" % (name,))
    if strict:
        if value <= minimum:
            raise ValueError(
                "%s must be an exact built-in int > %d" % (name, minimum)
            )
    elif value < minimum:
        raise ValueError(
            "%s must be an exact built-in int >= %d" % (name, minimum)
        )


def _validate_integral(integral):
    """Validate the summed-area table structure and zero border only."""
    if type(integral) is not np.ndarray:
        raise ValueError("integral must be an exact numpy.ndarray")
    if integral.dtype != np.uint64:
        raise ValueError("integral must have dtype uint64")
    if integral.ndim != 2:
        raise ValueError("integral must be rank 2")
    if integral.shape[0] < 2 or integral.shape[1] < 2:
        raise ValueError("integral must have at least two rows and two columns")
    if np.any(integral[0, :]):
        raise ValueError("integral first row must be all zeros")
    if np.any(integral[:, 0]):
        raise ValueError("integral first column must be all zeros")


def build_anchor_grid(integral, *, before=4, after=5, rows=128, check=None):
    """Precompute per-anchor footprint support from a summed-area table.

    Parameters
    ----------
    integral : numpy.ndarray
        Exact ``numpy.ndarray`` of dtype ``uint64`` and rank 2 holding the
        summed-area table of invalid pixels: ``integral[i, j]`` is the number
        of invalid pixels in ``source[:i, :j]``.  Its first row and first
        column must be exactly zero, and both dimensions must be at least two,
        so the source shape is ``H, W = integral.shape - 1`` with ``H, W >= 1``.
        Arbitrary strides are allowed and the table is never modified.
    before, after : int
        Non-negative exact built-in ints.  Anchor ``(y, x)`` covers the
        inclusive rectangle ``y - before .. y + after`` by
        ``x - before .. x + after``.  The defaults are the asymmetric 10x10
        window with anchor offset 4.
    rows : int
        Positive exact built-in int: the largest number of grid rows handled
        per tile.
    check : callable or None
        Optional zero-argument callback invoked immediately before every
        computed row tile.  Exceptions raised by it propagate unchanged.

    Returns
    -------
    numpy.ndarray
        Newly allocated, C-contiguous, read-only ``bool`` array of shape
        ``(H, W)``.  ``grid[y, x]`` is true exactly when the inclusive
        footprint of anchor ``(y, x)`` lies fully inside the source and
        contains no invalid source pixel.  Anchors whose footprint would leave
        the source, or every anchor when the source is smaller than the
        footprint, are false.

    Raises
    ------
    ValueError
        If a scalar parameter has the wrong type/range, or if ``integral`` has
        the wrong type, dtype, rank, shape, or a nonzero border.
    """
    _require_plain_int("before", before, 0, strict=False)
    _require_plain_int("after", after, 0, strict=False)
    _require_plain_int("rows", rows, 0, strict=True)
    _validate_integral(integral)

    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    grid = np.zeros((height, width), dtype=bool)

    # Kept as a Python int on purpose: enormous offsets (e.g. 10**10000) must
    # never be narrowed to a fixed-width integer type.
    span = before + after

    if span < height and span < width:
        # Anchor column ``x = before + k`` uses the inclusive table columns
        # ``k`` and ``k + span + 1`` for ``k`` in ``[0, width - span)``.
        low_cols = slice(0, width - span)
        high_cols = slice(span + 1, width + 1)
        x_start = before
        x_stop = width - after
        y_stop = height - after

        for y_start in range(before, y_stop, rows):
            if check is not None:
                check()
            tile_stop = y_start + rows
            if tile_stop > y_stop:
                tile_stop = y_stop
            for y in range(y_start, tile_stop):
                # Both differences are row differences of the same two column
                # boundaries, so each is non-negative and the unsigned
                # subtraction never wraps.  The fourth inclusion-exclusion
                # term cancels only when the rectangle count is exactly zero.
                after_diff = (
                    integral[y + after + 1, high_cols]
                    - integral[y + after + 1, low_cols]
                )
                before_diff = (
                    integral[y - before, high_cols]
                    - integral[y - before, low_cols]
                )
                grid[y, x_start:x_stop] = after_diff == before_diff
                del after_diff, before_diff

    grid.flags.writeable = False
    return grid


def anchor_grid_from_validity(validity, *, before=4, after=5, rows=128, check=None):
    """``build_anchor_grid(invalid_integral(validity), before=, after=, rows=, check=)`` without the uint64
    table: same errors in the same order, the same check() calls, and the same newly allocated, C-contiguous,
    read-only bool array (0/1 bytes). The rectangle count is zero exactly when every pixel of the rectangle
    is valid, and that AND is separable: columns first, then rows.
    """
    height, width = _validate_validity(validity)                  # invalid_integral's checks first
    _require_plain_int("before", before, 0, strict=False)          # then build_anchor_grid's
    _require_plain_int("after", after, 0, strict=False)
    _require_plain_int("rows", rows, 0, strict=True)
    grid = np.zeros((height, width), dtype=bool)
    span = before + after                                          # Python int, never narrowed
    if span < height and span < width:
        cols = width - span                                        # anchors x in [before, width-after)
        y_stop = height - after
        for y_start in range(before, y_stop, rows):                # the same tiles as build_anchor_grid
            if check is not None:
                check()
            tile_stop = y_start + rows
            if tile_stop > y_stop:
                tile_stop = y_stop
            count = tile_stop - y_start
            source = validity[y_start - before:tile_stop + after]  # rows y-before .. y+after of the tile
            # AND over columns x-before .. x+after; logical_and(., True) also writes 0/1 bytes when span == 0
            horizontal = np.logical_and(source[:, 0:cols], True)
            for k in range(1, span + 1):
                horizontal &= source[:, k:k + cols]
            target = grid[y_start:tile_stop, before:width - after]
            np.copyto(target, horizontal[0:count])                 # AND over rows y-before .. y+after
            for k in range(1, span + 1):
                target &= horizontal[k:k + count]
    grid.flags.writeable = False
    return grid


def anchor_support(anchors, points):
    """Look up point support in a precomputed anchor grid.

    Parameters
    ----------
    anchors : numpy.ndarray
        Exact, non-empty ``bool`` array of rank 2 with shape ``(H, W)``, such
        as the array returned by :func:`build_anchor_grid`.  Any strides are
        accepted and the array is never modified.
    points : numpy.ndarray
        Exact array of shape ``(N, 2)`` with a real signed/unsigned integer or
        floating dtype.  Column 0 holds ``x`` and column 1 holds ``y``.
        Floating input must be finite.  ``N`` may be zero, and the array is
        never modified.

    Returns
    -------
    numpy.ndarray
        Newly allocated ``bool`` array of shape ``(N,)``.  A point is in
        bounds when ``0 <= x < W`` and ``0 <= y < H``; in that case the result
        is ``anchors[floor(y), floor(x)]``.  Every out-of-bounds point is
        false, including negative coordinates, huge finite coordinates, and
        ``uint64`` maxima.  Values are never clipped into valid pixels, and
        the bounds test uses the original coordinates (widened exactly) before
        any floor or integer narrowing.

    Raises
    ------
    ValueError
        If ``anchors`` or ``points`` violates the contract, or if ``points``
        contains a non-finite value.
    """
    if type(anchors) is not np.ndarray:
        raise ValueError("anchors must be an exact numpy.ndarray")
    if anchors.dtype != np.bool_:
        raise ValueError("anchors must have dtype bool")
    if anchors.ndim != 2:
        raise ValueError("anchors must be rank 2")
    if anchors.size == 0:
        raise ValueError("anchors must be non-empty")

    if type(points) is not np.ndarray:
        raise ValueError("points must be an exact numpy.ndarray")
    if points.ndim != 2:
        raise ValueError("points must be rank 2")
    if points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    kind = points.dtype.kind
    if kind == "f":
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite coordinates")
    elif kind not in ("i", "u"):
        raise ValueError("points must have a real integer or floating dtype")

    height, width = anchors.shape
    count = points.shape[0]
    support = np.zeros(count, dtype=bool)
    if count == 0:
        return support

    # Widen to a dtype that represents every original coordinate exactly, so
    # the bounds test is performed on the original values rather than on
    # narrowed indices (and never raises on narrow integer dtypes).
    if kind == "f":
        wide = np.promote_types(points.dtype, np.float64)
    elif kind == "u":
        wide = np.dtype(np.uint64)
    else:
        wide = np.dtype(np.int64)

    xs = points[:, 0].astype(wide, copy=False)
    ys = points[:, 1].astype(wide, copy=False)
    zero = np.array(0, dtype=wide)
    x_limit = np.array(width, dtype=wide)
    y_limit = np.array(height, dtype=wide)

    inside = (xs >= zero) & (xs < x_limit) & (ys >= zero) & (ys < y_limit)

    if kind == "f":
        xi = np.floor(xs[inside]).astype(np.intp)
        yi = np.floor(ys[inside]).astype(np.intp)
    else:
        xi = xs[inside].astype(np.intp)
        yi = ys[inside].astype(np.intp)

    support[inside] = anchors[yi, xi]
    return support
