"""Bright-trail candidates and the transient layer (S4 step 1, RUN_LOG §28).

The sky stack can reject transients (meteor, satellite, aircraft) as
outliers, but may retain residual trails. The base master stays untouched. After
the stack, every registered frame Siril exported (``r_light_*.fit``, on the
grid the stack was made in) minus the master leaves transient signals,
noise and possible registration/background residuals; thin straight segments are candidates. Each is
measured at full resolution (endpoints, width, along-trail profile, flux,
colour ratios), given evidence from the other frames (continuation in an
adjacent frame, persistence at the same place), from its own brightness
profile (periodic modulation, whether the ends fade or stop abruptly) and a
*suggested* class.  A policy per class, a manual label and
a per-candidate override decide which candidates are added back through an
independent transient layer: the source frame's residual inside the trail's
footprint, normalised like the stack normalises frames (median/MAD to the
reference), warped back to native geometry with the same resampler as the
master.  Omitting a candidate adds nothing; the base master is never edited.

Conventions: arrays are TOP-DOWN (``a[y, x]``), zero-based pixel centres;
coordinates on the *stack grid* are those of the exported frames (undistorted,
magnified grid when the job registered with its own term, native otherwise).
Pure NumPy plus ``fits_crop``; no astropy dependency, so the detection and
the tests run wherever the frames are.

1.1.0 (RUN_LOG §28, after the first photographed run): frames and the master
are memmapped once per frame (``Planes``) instead of re-opened per crop; the
component statistics and the fragment attachment of ``find_segments`` are
vectorised (same rules, same results); a segment whose measured run is
shorter than ``min_length_px`` is ``too_short`` instead of a candidate; the
periodic-modulation evidence is computed on the envelope-normalised,
detrended profile with the 2-sample sampling alternation removed (the 1.0.0
test fired on any smooth profile); new light-curve evidence (do the ends
fade or stop abruptly, where they can be judged) drives the suggestion when
no cross-frame evidence exists.
"""
from __future__ import annotations

import contextlib
import json
import math
import warnings
from pathlib import Path

import numpy as np

from . import fits_crop
from .resample import warp_image, warp_planes_fused

__all__ = ["TRANSIENT_VERSION", "CLASSES", "DECISIONS", "DEFAULT_DETECTION", "DEFAULT_EVIDENCE", "DEFAULT_POLICY", "DEFAULT_LAYER",
           "CANDIDATES_FILE", "Planes", "detection_image", "label_components", "find_segments", "measure_segment", "detect_frame",
           "frame_stats", "add_evidence", "suggest", "resolve_decisions", "layer_box", "warp_box_to_native", "apply_boxes",
           "write_candidates", "read_candidates", "write_crop_png"]

TRANSIENT_VERSION = "1.2.1"
CLASSES = ("meteor", "satellite", "aircraft", "artifact", "unknown")
DECISIONS = ("keep", "omit")
CANDIDATES_FILE = "candidates.json"

# RUN_LOG §28 — fixed before any result was seen; all lengths in full-resolution pixels of the stack grid.
DEFAULT_DETECTION = {
    "bin": 2, "threshold_sigma": 4.0, "mark_threshold_sigma": 3.0, "background_block_px": 64,
    "min_length_px": 60, "min_elongation": 5.0, "max_line_rms_px": 3.0,
    "merge_angle_deg": 3.0, "merge_offset_px": 8.0, "merge_gap_px": 300, "chain_angle_deg": 6.0,
    "weak_snr": 8.0, "max_foreground_pixels": 400_000,
    "core_half_min_px": 2, "core_half_max_px": 20, "side_inner_px": 8, "side_outer_px": 28,
    "profile_step_px": 4, "endpoint_snr": 2.0, "endpoint_smooth_px": 16, "end_search_px": 50,
}
# 1.1.0 additions (RUN_LOG §28, recorded as post-hoc): ``ends_*`` judge the light curve at the two ends of the measured
# run — an end is judged only when the profile is bright enough for the fade to be visible above the endpoint threshold
# (90th percentile of the per-position S/N >= ends_min_p90_snr), has data beyond it and is not at the frame border.
DEFAULT_EVIDENCE = {
    "continue_angle_deg": 3.0, "continue_offset_px": 15.0, "continue_overlap_max": 0.2, "continue_gap_lengths": 5.0,
    "persistent_overlap_min": 0.6, "persistent_frames": 3,
    "periodic_min_lag_px": 8, "periodic_corr": 0.4, "dashed_fraction": 0.25, "dashed_level": 0.3,
    "ends_fraction": 0.05, "ends_faded_max": 0.35, "ends_abrupt_min": 0.5, "ends_min_p90_snr": 12.0, "edge_px": 20,
}
DEFAULT_POLICY = {"meteor": "keep", "satellite": "omit", "aircraft": "omit", "artifact": "omit", "unknown": "omit"}
DEFAULT_LAYER = {"footprint_margin_px": 2, "feather_px": 2, "end_extension_px": 4, "stats_step": 16, "warp_margin_px": 6}


# ----------------------------------------------------------------------------------------------------------------------
# plane access
# ----------------------------------------------------------------------------------------------------------------------

def _rows(view, header: dict, roworder: str, y0: int, y1: int) -> np.ndarray:
    """Rows ``y0:y1`` (top-down) of a storage-order plane as float32 in the [0, 1] scale of the float frames."""
    height = view.shape[0]
    if roworder == "BOTTOM-UP":
        block = np.asarray(view[height - y1:height - y0][::-1], dtype=np.float32)
    else:
        block = np.asarray(view[y0:y1], dtype=np.float32)
    if int(header["BITPIX"]) == 16:
        block = (block + np.float32(32768.0)) / np.float32(65535.0)
    return block


SAMPLED_PROBE_ROWS = 32    # P1c T-3: a pooled crop's sampled read calls its abort probe every this many sampled rows


def _sampled(view, header: dict, roworder: str, x0: int, x1: int, y0: int, y1: int, step: int,
             check=None) -> np.ndarray:
    """``_rows(view, header, roworder, y0, y1)[:, x0:x1][::step, ::step]`` for an in-image box (``0 <= x0 < x1 <= W``,
    ``0 <= y0 < y1 <= H``), reading only the sampled elements (efficiency P1c T-3m).

    Every value is the same file element through the same element-wise float32 conversion as ``_rows``; only which
    elements are read differs. The top-down rows are y0, y0 + step, ..., y0 + step (n - 1); BOTTOM-UP storage holds
    top-down row y at H - 1 - y, so they are the storage rows H - 1 - y0 - step (n - 1) .. H - 1 - y0 (step ``step``),
    reversed. That start is never negative: y0 + step (n - 1) <= y1 - 1 <= H - 1 (equal when y1 == H).

    ``check`` (P1c T-3, rule R18): called before every SAMPLED_PROBE_ROWS sampled rows, which are then converted into
    their rows of the result (the same cast, element by element). A sampled row is one page-fault read when the file
    is not cached, so a tall crop read cold (a 16 GB computer, which cannot keep 48 frames cached after the scan) still
    reaches a probe every few hundredths of a second."""
    height = view.shape[0]
    count = (y1 - y0 + step - 1) // step
    if roworder == "BOTTOM-UP":
        rows = view[height - 1 - y0 - step * (count - 1):height - y0:step][::-1, x0:x1:step]
    else:
        rows = view[y0:y1:step, x0:x1:step]
    if check is None or count <= SAMPLED_PROBE_ROWS:
        if check is not None:
            check()
        block = np.asarray(rows, dtype=np.float32)
    else:
        block = np.empty(rows.shape, dtype=np.float32)
        for start in range(0, count, SAMPLED_PROBE_ROWS):
            check()
            block[start:start + SAMPLED_PROBE_ROWS] = rows[start:start + SAMPLED_PROBE_ROWS]
    if int(header["BITPIX"]) == 16:
        block = (block + np.float32(32768.0)) / np.float32(65535.0)
    return block


class Planes:
    """The three planes of an RGB FITS file, memmapped once (a frame and the master are read many times per frame:
    detection rows, measurement crops, review crops, layer boxes).  Every function below takes a path or a ``Planes``."""

    def __init__(self, path):
        self.path = Path(path)
        self.planes = [fits_crop.plane_memmap(self.path, c) for c in range(3)]
        self.shape = tuple(int(v) for v in self.planes[0][0].shape)       # (height, width)

    def rows(self, plane: int, y0: int, y1: int) -> np.ndarray:
        return _rows(*self.planes[plane], y0, y1)

    def crop(self, plane: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        """Top-down crop in the [0, 1] scale (``fits_crop.crop_top_down`` keeps 16-bit files in code units)."""
        height, width = self.shape
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=np.float32)
        return self.rows(plane, y0, y1)[:, x0:x1]

    def __fspath__(self) -> str:
        return str(self.path)


def _planes(source) -> Planes:
    return source if isinstance(source, Planes) else Planes(source)


def _crop(source, plane: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    return _planes(source).crop(plane, x0, x1, y0, y1)


def frame_stats(source, *, step: int = 16) -> dict:
    """Per-channel median and MAD of a frame's non-zero pixels, sampled every ``step`` px (the stack's addscale statistics)."""
    planes = _planes(source)
    medians, mads = [], []
    for plane in range(3):
        view, header, roworder = planes.planes[plane]
        sample = _rows(view, header, roworder, 0, view.shape[0])[::step, ::step] if view.shape[0] <= 4096 else None
        if sample is None:
            rows = []
            for y0 in range(0, view.shape[0], 1024):
                rows.append(_rows(view, header, roworder, y0, min(view.shape[0], y0 + 1024))[::step, ::step])
            sample = np.vstack(rows)
        values = sample[sample != 0]
        if values.size == 0:
            medians.append(0.0)
            mads.append(0.0)
            continue
        median = float(np.median(values))
        medians.append(median)
        mads.append(float(np.median(np.abs(values - median))))
    return {"median": medians, "mad": mads}


def normalisation(stats: dict, ref_stats: dict) -> list[float]:
    """addscale-like per-channel scale of a frame towards the reference (1 when a MAD is missing)."""
    return [float(ref_stats["mad"][c] / stats["mad"][c]) if stats["mad"][c] > 0 and ref_stats["mad"][c] > 0 else 1.0 for c in range(3)]


# ----------------------------------------------------------------------------------------------------------------------
# detection image
# ----------------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    """nanmedian over all-NaN slices (no data) is expected here; keep the warnings out of the job logs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield


# numpy 2.5.3 (lib/_nanfunctions_impl.py:1092-1094): np.nanmedian(x, axis=-1) sends slices of >= 600 values through
# apply_along_axis(_nanmedian1d), i.e. np.median of each slice's non-NaN values.  _nanmedian_rows selects the same
# order statistics with one partition per row (a sort for rows with missing pixels).  A row whose selected middle value
# is zero or tiny (|x| < 2**-100, subnormals included) is recomputed by np.nanmedian on its original values, so the sign
# of zero, subnormal arithmetic and the floating-point environment never decide a result.  Shorter slices take np.ma.median (odd count: (x + x) / 2); they,
# and every input outside the proven domain, keep the pre-P1a code (_block_median_numpy).  (P1a T-4)
_NANMEDIAN_LONG_SLICE = 600
_FLOAT32_EXPONENT_BITS = 0x7F800000
_TINY_EXPONENT_BITS = 27 << 23           # biased float32 exponent of 2**-100


def _nanmedian_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median of the non-NaN values of each row of a private C-contiguous float32 (m, L) array, L >= 600; ``rows`` is
    reordered in place.  Returns ``(med, redo)``: ``med`` equals ``np.nanmedian(rows, axis=1)`` bit for bit in every row
    except the rows listed in ``redo`` (a selected middle value is zero or |x| < 2**-100), which the caller recomputes
    with numpy.  An all-NaN row gives NaN.  Row-local, so a later ``workers=`` may split ``rows`` into row ranges.

    Complete rows are partitioned at the lower middle (a single kth, which numpy runs as a SIMD quickselect where the
    CPU allows, and which is fast on baseline kernels too); the upper middle is the smallest value after it.  Both are
    the values of those order statistics; only the sign of a zero could differ from a sort, and zeros are redone.
    Rows with missing pixels are sorted.  If most rows miss pixels, one sort of all rows is cheaper."""
    m, length = rows.shape
    k_lo, k_hi = (length - 1) // 2, length // 2
    with_nan = np.flatnonzero(np.isnan(rows.max(axis=1)))               # max propagates NaN: exactly the rows holding one
    count = np.full(m, length, dtype=np.intp)                            # non-NaN values per row
    count[with_nan] -= np.count_nonzero(np.isnan(rows[with_nan]), axis=1)
    partial = with_nan[count[with_nan] > 0]
    if 4 * partial.size > m:
        rows.sort(axis=1)                                                # NaN sorts to the end of each row
        index = np.arange(m)
        lo = rows[index, (np.maximum(count, 1) - 1) // 2]                # lower middle (the middle for an odd count)
        hi = rows[index, count // 2]                                     # upper middle (the same element for an odd count)
    else:
        rows.partition(k_lo, axis=1)
        lo = rows[:, k_lo].copy()
        hi = rows[:, k_hi:].min(axis=1) if k_hi > k_lo else lo.copy()   # the smallest value above k_lo: order statistic k_hi
        if partial.size:
            sub = rows[partial]
            sub.sort(axis=1)
            c = count[partial]
            index = np.arange(partial.size)
            lo[partial] = sub[index, (c - 1) // 2]
            hi[partial] = sub[index, c // 2]
    redo = np.flatnonzero(((lo.view(np.uint32) & _FLOAT32_EXPONENT_BITS) < _TINY_EXPONENT_BITS)
                          | ((hi.view(np.uint32) & _FLOAT32_EXPONENT_BITS) < _TINY_EXPONENT_BITS))
    even = np.flatnonzero((count % 2 == 0) & (count > 0))
    lo[even] = (lo[even] + hi[even]) / np.float32(2)                     # numpy: float32 sum of the two, then / 2
    lo[count == 0] = np.nan
    return lo, redo


def _block_median_numpy(a: np.ndarray, ok: np.ndarray, size: int) -> np.ndarray:
    """The pre-P1a ``_block_median`` (one np.nanmedian per block), verbatim: the route for inputs outside T-4's domain."""
    h, w = a.shape
    ph, pw = (-h) % size, (-w) % size
    values = np.where(ok, a, np.nan).astype(np.float32)
    values = np.pad(values, ((0, ph), (0, pw)), constant_values=np.nan)
    blocks = values.reshape((h + ph) // size, size, (w + pw) // size, size).transpose(0, 2, 1, 3).reshape((h + ph) // size, (w + pw) // size, size * size)
    with _quiet():
        med = np.nanmedian(blocks, axis=-1)
    fallback = float(np.nanmedian(med)) if np.isfinite(med).any() else 0.0
    med = np.where(np.isfinite(med), med, fallback)
    return np.repeat(np.repeat(med, size, axis=0), size, axis=1)[:h, :w].astype(np.float32)


def _block_median(a: np.ndarray, ok: np.ndarray, size: int) -> np.ndarray:
    """Median over ``size`` x ``size`` blocks of the valid pixels, nearest-upsampled back; invalid blocks get the global median.

    Bit-identical to ``_block_median_numpy`` (P1a T-4): one row sort instead of one np.nanmedian call per block, for
    float32/float64 planes with a same-shape bool mask and blocks of >= 600 pixels; other inputs take the numpy code."""
    h, w = a.shape
    if not (type(size) is int and size > 0 and size * size >= _NANMEDIAN_LONG_SLICE
            and type(a) is np.ndarray and a.dtype in (np.float32, np.float64)
            and type(ok) is np.ndarray and ok.dtype == np.bool_ and ok.shape == a.shape):
        return _block_median_numpy(a, ok, size)
    ph, pw = (-h) % size, (-w) % size
    bh, bw = (h + ph) // size, (w + pw) // size
    values = np.full((h + ph, w + pw), np.nan, dtype=np.float32)
    np.copyto(values[:h, :w], a, casting="unsafe", where=ok)            # == np.pad(np.where(ok, a, nan).astype(float32))
    blocks = np.empty((bh, bw, size, size), dtype=np.float32)
    blocks[...] = values.reshape(bh, size, bw, size).transpose(0, 2, 1, 3)   # always a copy: values stays unsorted
    with _quiet():
        med, redo = _nanmedian_rows(blocks.reshape(bh * bw, size * size))
        del blocks
        if redo.size:                                                    # numpy's own median of the original rows
            by, bx = np.divmod(redo, bw)
            med[redo] = np.nanmedian(values.reshape(bh, size, bw, size)[by, :, bx, :].reshape(redo.size, size * size), axis=-1)
    del values
    med = med.reshape(bh, bw)
    fallback = float(np.nanmedian(med)) if np.isfinite(med).any() else 0.0
    med = np.where(np.isfinite(med), med, fallback)
    up = np.empty((bh, size, bw, size), dtype=np.float32)
    up[...] = med[:, None, :, None]                                      # == np.repeat(np.repeat(med, size, 0), size, 1)
    return up.reshape(bh * size, bw * size)[:h, :w]


def _binned_residuals(frames, masters, b: int, hb: int, wb: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Per bin of ``b`` x ``b`` pixels: the float32 sum of the valid frame-minus-master residuals of each channel
    (3, hb, wb) and the int32 number of valid pixels, reading ``rows`` full-resolution rows at a time.  A separate
    function so that the last row chunk (about 80 MB at 45.7 MP) is freed before the float64 stage (P1a T-4m)."""
    resid = np.zeros((3, hb, wb), dtype=np.float32)
    count = np.zeros((hb, wb), dtype=np.int32)
    rows = max(b, (rows // b) * b)
    for y0 in range(0, hb * b, rows):
        y1 = min(hb * b, y0 + rows)
        valid = None
        blocks = []
        for c in range(3):
            f = _rows(*frames[c], y0, y1)[:, :wb * b]
            m = _rows(*masters[c], y0, y1)[:, :wb * b]
            v = (f != 0) & (m != 0)
            valid = v if valid is None else (valid & v)
            blocks.append((f, m))
        nb = (y1 - y0) // b
        for c in range(3):
            f, m = blocks[c]
            r = np.where(valid, f - m, np.float32(0)).reshape(nb, b, wb, b).sum(axis=(1, 3))
            resid[c, y0 // b:y1 // b] = r
        count[y0 // b:y1 // b] = valid.reshape(nb, b, wb, b).sum(axis=(1, 3))
    return resid, count


def detection_image(frame_path, master_path, *, params: dict | None = None, rows: int = 512) -> dict:
    """Binned, background-flattened residual S/N of one exported frame against the master (RUN_LOG §28).

    Returns ``snr`` (sum over channels of residual/sigma, over sqrt(3)) and ``valid`` on the binned grid, the per-channel
    sigma (MAD-based, full-resolution residual scaled to the binned mean) and the full-resolution shape.
    """
    p = dict(DEFAULT_DETECTION, **(params or {}))
    b = int(p["bin"])
    frames = _planes(frame_path).planes
    masters = _planes(master_path).planes
    height, width = frames[0][0].shape
    if masters[0][0].shape != (height, width):
        raise ValueError("frame and master are not on the same grid")
    hb, wb = height // b, width // b
    resid, count = _binned_residuals(frames, masters, b, hb, wb, rows)
    ok = count >= max(1, (b * b) // 2)
    denom = np.maximum(count, 1)
    del count
    invalid = ~ok
    block = max(1, int(p["background_block_px"]) // b)
    snr = np.zeros((hb, wb), dtype=np.float32)
    sigma = []
    for c in range(3):
        # One float64 plane per channel, updated in place (P1a T-4m): the same ufunc loops and values as the former
        # (3, hb, wb) float64 stack, np.where and temporaries, at about 55% of their peak memory.
        flat = resid[c] / denom                           # float32 / int32 -> float64
        flat[invalid] = 0.0                               # was np.where(ok[None], ..., np.float32(0))
        flat -= _block_median(flat, ok, block)
        sample = flat[::3, ::3][ok[::3, ::3]]
        s = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) if sample.size else 0.0
        sigma.append(s)
        if s > 0:
            flat /= np.float32(s)
            flat[invalid] = 0.0                           # was np.where(ok, flat / np.float32(s), np.float32(0))
            snr += flat
        del flat
    snr /= np.float32(math.sqrt(3.0))
    return {"snr": snr, "valid": ok, "sigma": sigma, "bin": b, "shape": (height, width), "foreground": None}


# ----------------------------------------------------------------------------------------------------------------------
# connected components (sparse union-find, 8-connected, pure NumPy/Python)
# ----------------------------------------------------------------------------------------------------------------------

def label_components(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8-connected components of a boolean map; returns (ys, xs, labels) for the foreground pixels, labels 0..k-1."""
    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    n = ys.size
    if n == 0:
        return ys, xs, np.zeros(0, dtype=np.int64)
    lin = ys.astype(np.int64) * width + xs          # row-major, already sorted
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ny, nx = ys + dy, xs + dx
        inside = (nx >= 0) & (nx < width) & (ny < height)
        source = np.nonzero(inside)[0]
        target = ny[inside].astype(np.int64) * width + nx[inside]
        pos = np.searchsorted(lin, target)
        pos_ok = pos < n
        hit = np.zeros(source.size, dtype=bool)
        hit[pos_ok] = lin[pos[pos_ok]] == target[pos_ok]
        for i, j in zip(source[hit].tolist(), pos[hit].tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, labels = np.unique(roots, return_inverse=True)
    return ys, xs, labels.astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# segments from the detection image
# ----------------------------------------------------------------------------------------------------------------------

def _axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """PCA of (N, 2) points: centre, unit major axis, std along, std across, extent along."""
    centre = points.mean(axis=0)
    d = points - centre
    if len(points) < 2:
        return centre, np.array([1.0, 0.0]), 0.0, 0.0, 0.0
    cov = d.T @ d / len(points)
    values, vectors = np.linalg.eigh(cov)
    u = vectors[:, 1]
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u
    along = d @ u
    return centre, u, float(math.sqrt(max(values[1], 0.0))), float(math.sqrt(max(values[0], 0.0))), float(along.max() - along.min())


def _straightness(points: np.ndarray, centre: np.ndarray, u: np.ndarray, window: float = 16.0) -> float:
    """RMS of the per-window mean perpendicular offset along the axis (curvature, independent of the width)."""
    d = points - centre
    along = d @ u
    across = d @ np.array([-u[1], u[0]])
    if along.size < 2:
        return 0.0
    bins = np.floor((along - along.min()) / window).astype(int)
    sums = np.bincount(bins, weights=across)
    counts = np.bincount(bins)
    means = sums[counts > 0] / counts[counts > 0]
    return float(math.sqrt(float(np.mean(means ** 2)))) if means.size else 0.0


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _component_stats(fx: np.ndarray, fy: np.ndarray, labels: np.ndarray, b: int) -> dict:
    """Per-component PCA statistics for every labelled component at once (what ``_axis`` computes, vectorised): centre,
    unit major axis (same sign convention), std along/across (across floored at b/sqrt(12)), extent along the axis, size,
    angle; plus the label-sorted point order (``order``, ``starts``, ``ends``) to slice a component's points."""
    order = np.argsort(labels, kind="stable")
    lab = labels[order]
    x, y = fx[order].astype(np.float64), fy[order].astype(np.float64)
    count = int(lab[-1]) + 1
    starts = np.searchsorted(lab, np.arange(count))
    ends = np.append(starts[1:], lab.size)
    n = (ends - starts).astype(np.float64)
    cx = np.add.reduceat(x, starts) / n
    cy = np.add.reduceat(y, starts) / n
    dx, dy = x - cx[lab], y - cy[lab]
    cxx = np.add.reduceat(dx * dx, starts) / n
    cyy = np.add.reduceat(dy * dy, starts) / n
    cxy = np.add.reduceat(dx * dy, starts) / n
    half_trace = 0.5 * (cxx + cyy)
    disc = np.sqrt(np.maximum(0.25 * (cxx - cyy) ** 2 + cxy * cxy, 0.0))
    l_major, l_minor = half_trace + disc, half_trace - disc
    ux = np.where(cxy != 0, l_major - cyy, np.where(cxx >= cyy, 1.0, 0.0))
    uy = np.where(cxy != 0, cxy, np.where(cxx >= cyy, 0.0, 1.0))
    norm = np.hypot(ux, uy)
    norm = np.where(norm > 0, norm, 1.0)
    ux, uy = ux / norm, uy / norm
    flip = (ux < 0) | ((ux == 0) & (uy < 0))
    ux, uy = np.where(flip, -ux, ux), np.where(flip, -uy, uy)
    along = dx * ux[lab] + dy * uy[lab]
    extent = np.maximum.reduceat(along, starts) - np.minimum.reduceat(along, starts)
    s_along = np.sqrt(np.maximum(l_major, 0.0))
    s_across = np.maximum(np.sqrt(np.maximum(l_minor, 0.0)), b / math.sqrt(12.0))
    single = n < 2
    s_along = np.where(single, 0.0, s_along)
    extent = np.where(single, 0.0, extent)
    ux, uy = np.where(single, 1.0, ux), np.where(single, 0.0, uy)
    return {"order": order, "starts": starts, "ends": ends, "n": n.astype(np.int64), "centre": np.column_stack([cx, cy]),
            "u": np.column_stack([ux, uy]), "s_along": s_along, "s_across": s_across, "extent": extent,
            "angle": np.degrees(np.arctan2(uy, ux)) % 180.0}


def find_segments(detection: dict, *, params: dict | None = None, threshold: float | None = None, corridor=None) -> list[dict]:
    """Straight thin segments in a detection image (RUN_LOG §28): thresholded components, collinear chaining of dashes,
    seeds (length, elongation, straightness), fragments attached to seeds, seeds merged.  Coordinates are full-resolution
    stack-grid pixel centres.  ``corridor`` = (x0, y0, x1, y1, half_width) restricts the search to a manual mark."""
    p = dict(DEFAULT_DETECTION, **(params or {}))
    b = int(detection["bin"])
    thr = float(p["threshold_sigma"] if threshold is None else threshold)
    binary = (detection["snr"] >= thr) & detection["valid"]
    if corridor is not None:
        x0, y0, x1, y1, half = corridor
        hb, wb = binary.shape
        yy, xx = np.mgrid[:hb, :wb]
        fx, fy = xx * b + (b - 1) / 2.0, yy * b + (b - 1) / 2.0
        binary &= _corridor_mask(fx, fy, x0, y0, x1, y1, half)
    n_fg = int(binary.sum())
    if n_fg > int(p["max_foreground_pixels"]):
        return [{"status": "saturated", "reason": "detector_budget_exceeded", "clipping_evidence": "unknown", "foreground_pixels": n_fg}]
    ys, xs, labels = label_components(binary)
    if ys.size == 0:
        return []
    fx = xs * b + (b - 1) / 2.0
    fy = ys * b + (b - 1) / 2.0
    stats = _component_stats(fx, fy, labels, b)
    n_comp = stats["n"].size
    starts, ends, order = stats["starts"], stats["ends"], stats["order"]

    def points_of(i: int) -> np.ndarray:
        idx = order[starts[i]:ends[i]]
        return np.column_stack([fx[idx], fy[idx]])

    # collinear chaining of small elongated components (dashes) -> pooled sets
    parent = list(range(n_comp))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    dashy = np.nonzero((stats["n"] >= 4) & (stats["s_along"] / stats["s_across"] >= 2.0))[0]
    if dashy.size >= 2:
        centres = stats["centre"][dashy]
        angles = stats["angle"][dashy]
        gap = float(p["merge_gap_px"])
        chain = float(p["chain_angle_deg"])
        for a_i, i in enumerate(dashy.tolist()):
            dc = centres - centres[a_i]
            dist = np.hypot(dc[:, 0], dc[:, 1])
            near = np.nonzero((dist > 0) & (dist <= gap))[0]
            if near.size == 0:
                continue
            axis_diff = np.abs(angles[near] - angles[a_i]) % 180.0
            axis_diff = np.minimum(axis_diff, 180.0 - axis_diff)
            link = np.degrees(np.arctan2(dc[near, 1], dc[near, 0])) % 180.0
            link_diff = np.abs(link - angles[a_i]) % 180.0
            link_diff = np.minimum(link_diff, 180.0 - link_diff)
            for a_j in near[(axis_diff <= chain) & (link_diff <= chain)].tolist():
                union(i, int(dashy[a_j]))
    groups = {}
    for i in range(n_comp):
        groups.setdefault(find(i), []).append(i)
    # sets: pooled description only where it can matter (a possible seed by length and elongation, or a chained group);
    # every other set is a fragment that only needs its centre and its members
    min_length, min_elong = float(p["min_length_px"]), float(p["min_elongation"])
    seeds, rest = [], []
    for members in groups.values():
        if len(members) == 1:
            i = members[0]
            if stats["extent"][i] >= min_length and stats["s_along"][i] / stats["s_across"][i] >= min_elong:
                desc = _describe(points_of(i), members, b)
            else:
                rest.append({"centre": stats["centre"][i], "members": members})
                continue
        else:
            desc = _describe(np.vstack([points_of(i) for i in members]), members, b)
        if desc["length"] >= min_length and desc["elongation"] >= min_elong and desc["line_rms"] <= p["max_line_rms_px"]:
            seeds.append(desc)
        else:
            rest.append({"centre": desc["centre"], "members": members})
    if not seeds:
        return []
    # attach fragments to seeds by lateral distance and gap along the seed line (the nearest seed laterally, first on ties)
    attached = {id(s): list(s["members"]) for s in seeds}
    if rest:
        rc = np.array([f["centre"] for f in rest], dtype=np.float64)
        best_lat = np.full(len(rest), np.inf)
        best_seed = np.full(len(rest), -1, dtype=np.int64)
        for k, seed in enumerate(seeds):
            d = rc - seed["centre"]
            u = seed["u"]
            lateral = np.abs(d @ np.array([-u[1], u[0]]))
            gap_along = np.maximum(0.0, np.abs(d @ u) - seed["length"] / 2.0)
            better = (lateral <= p["merge_offset_px"]) & (gap_along <= p["merge_gap_px"]) & (lateral < best_lat)
            best_lat[better] = lateral[better]
            best_seed[better] = k
        for f_i in np.nonzero(best_seed >= 0)[0].tolist():
            attached[id(seeds[best_seed[f_i]])].extend(rest[f_i]["members"])
    merged = [_describe(np.vstack([points_of(i) for i in attached[id(s)]]), attached[id(s)], b) for s in seeds]
    # merge collinear seeds
    parent = list(range(len(merged)))
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            a, c = merged[i], merged[j]
            if _angle_diff_deg(a["angle"], c["angle"]) > p["merge_angle_deg"]:
                continue
            d1, g1 = _relative_position(a, c["centre"], extent=c["length"])
            d2, g2 = _relative_position(c, a["centre"], extent=a["length"])
            if max(d1, d2) <= p["merge_offset_px"] and min(g1, g2) <= p["merge_gap_px"]:
                union(i, j)
    final_groups = {}
    for i in range(len(merged)):
        final_groups.setdefault(find(i), []).append(i)
    out = []
    for members in final_groups.values():
        pts = np.vstack([merged[i]["points"] for i in members])
        idx = sorted(set(sum((merged[i]["members"] for i in members), [])))
        desc = _describe(pts, idx, b)
        if desc["length"] < p["min_length_px"] or desc["line_rms"] > p["max_line_rms_px"]:
            continue
        out.append({"status": "detected", "x0": desc["p0"][0], "y0": desc["p0"][1], "x1": desc["p1"][0], "y1": desc["p1"][1],
                    "length_px": desc["length"], "elongation": desc["elongation"], "line_rms_px": desc["line_rms"],
                    "pixels": desc["n"], "components": len(idx)})
    out.sort(key=lambda s: -s["length_px"])
    return out


def _describe(pts: np.ndarray, members: list, b: int) -> dict:
    centre, u, s_along, s_across, extent = _axis(pts)
    s_across = max(s_across, b / math.sqrt(12.0))
    along = (pts - centre) @ u
    p0 = centre + u * along.min()
    p1 = centre + u * along.max()
    return {"points": pts, "members": list(members), "centre": centre, "u": u, "angle": math.degrees(math.atan2(u[1], u[0])) % 180.0,
            "length": float(extent), "elongation": float(s_along / s_across), "line_rms": _straightness(pts, centre, u),
            "n": int(len(pts)), "p0": [float(p0[0]), float(p0[1])], "p1": [float(p1[0]), float(p1[1])]}


def _relative_position(seg: dict, point: np.ndarray, extent: float = 0.0) -> tuple[float, float]:
    """(lateral distance of ``point`` from the segment's line, gap along the line beyond the segment's ends minus ``extent``/2)."""
    d = np.asarray(point, dtype=np.float64) - seg["centre"]
    u = seg["u"]
    lateral = abs(float(d @ np.array([-u[1], u[0]])))
    along = float(d @ u)
    half = seg["length"] / 2.0
    gap = max(0.0, abs(along) - half - extent / 2.0)
    return lateral, gap


def _corridor_mask(fx: np.ndarray, fy: np.ndarray, x0: float, y0: float, x1: float, y1: float, half: float) -> np.ndarray:
    ux, uy = x1 - x0, y1 - y0
    length = math.hypot(ux, uy)
    if length == 0:
        return np.hypot(fx - x0, fy - y0) <= half
    ux, uy = ux / length, uy / length
    dx, dy = fx - x0, fy - y0
    along = dx * ux + dy * uy
    across = np.abs(-dx * uy + dy * ux)
    return (across <= half) & (along >= -half) & (along <= length + half)


# ----------------------------------------------------------------------------------------------------------------------
# full-resolution measurement of one segment
# ----------------------------------------------------------------------------------------------------------------------

def _residual_crops(frame_path, master_path, box: tuple[int, int, int, int]) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Per-channel residual crops (frame - master, NaN where either has no data) and the frame crops themselves (NaN where
    no data) for a top-down box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    frame_planes, master_planes = _planes(frame_path), _planes(master_path)
    crops, frames = [], []
    for c in range(3):
        f = frame_planes.crop(c, x0, x1, y0, y1)
        m = master_planes.crop(c, x0, x1, y0, y1)
        r = f - m
        nodata = (f == 0) | (m == 0)
        r[nodata] = np.nan
        f = f.copy()
        f[nodata] = np.nan
        crops.append(r)
        frames.append(f)
    return crops, frames, (max(0, x0), max(0, y0))


def _sample_profiles(crops, origin, p0, u, n_vec, t_values, d_values):
    """Residual samples on the (t, d) lattice around the line: list per channel of arrays (len(t), len(d))."""
    ox, oy = origin
    tt, dd = np.meshgrid(t_values, d_values, indexing="ij")
    x = p0[0] + tt * u[0] + dd * n_vec[0] - ox
    y = p0[1] + tt * u[1] + dd * n_vec[1] - oy
    return [fits_crop._bilinear(c, x, y) for c in crops]


def measure_segment(frame_path, master_path, x0: float, y0: float, x1: float, y1: float, *, params: dict | None = None,
                    search_beyond_px: float | None = None) -> dict:
    """Full-resolution measurement of a segment in a frame's residual against the master (RUN_LOG §28).

    Refines the line centre and the endpoints (longest run with smoothed along-trail S/N >= ``endpoint_snr``), measures
    the core half-width (FWHM/2 of the median cross-profile, clamped), the along-trail contrast, per-channel flux,
    colour ratios, the total S/N, the along-trail brightness series used for the modulation and light-curve evidence,
    and for each end whether data continues beyond it and whether it sits at the frame border.  An automatic segment
    (``search_beyond_px`` None) whose measured run is shorter than ``min_length_px`` is ``too_short`` (1.1.0): the
    detection chained things that do not form one trail at full resolution.
    """
    p = dict(DEFAULT_DETECTION, **(params or {}))
    frame_path, master_path = _planes(frame_path), _planes(master_path)
    p0 = np.array([x0, y0], dtype=np.float64)
    p1 = np.array([x1, y1], dtype=np.float64)
    length = float(np.hypot(*(p1 - p0)))
    if length < 1.0:
        return {"status": "degenerate"}
    u = (p1 - p0) / length
    n_vec = np.array([-u[1], u[0]])
    ext = float(p["end_search_px"] if search_beyond_px is None else search_beyond_px)
    ext = max(ext, 0.1 * length) if search_beyond_px is None else ext
    outer = int(p["side_outer_px"])
    inner, half_min, half_max = int(p["side_inner_px"]), int(p["core_half_min_px"]), int(p["core_half_max_px"])
    reach = half_max + outer + 2                      # the cross-profile must hold the side band of the widest core
    xs = [p0[0] - ext * u[0], p1[0] + ext * u[0]]
    ys = [p0[1] - ext * u[1], p1[1] + ext * u[1]]
    box = (int(math.floor(min(xs) - reach)), int(math.floor(min(ys) - reach)), int(math.ceil(max(xs) + reach)) + 1, int(math.ceil(max(ys) + reach)) + 1)
    crops, frame_crops, origin = _residual_crops(frame_path, master_path, box)
    if crops[0].size == 0:
        return {"status": "outside"}
    step = float(p["profile_step_px"])
    t_values = np.arange(-ext, length + ext + 1e-6, step)
    d_values = np.arange(-reach + 1, reach, 1.0)

    def stats(profiles, half):
        core_sel = np.abs(d_values) <= half
        side_sel = (np.abs(d_values) >= half + inner) & (np.abs(d_values) <= half + outer)
        left_sel = side_sel & (d_values < 0)
        right_sel = side_sel & (d_values > 0)
        core, side, left, right = [], [], [], []
        with _quiet():
            for prof in profiles:
                core.append(np.nanmedian(prof[:, core_sel], axis=1))
                side.append(np.nanmedian(prof[:, side_sel], axis=1))
                left.append(np.nanmedian(prof[:, left_sel], axis=1))
                right.append(np.nanmedian(prof[:, right_sel], axis=1))
        return np.array(core), np.array(side), np.array(left), np.array(right)

    def side_level(frame_profiles, half):
        """The frame's own background in the side band (the residual's side band is ~0 by construction)."""
        side_sel = (np.abs(d_values) >= half + inner) & (np.abs(d_values) <= half + outer)
        with _quiet():
            return np.array([np.nanmedian(prof[:, side_sel], axis=1) for prof in frame_profiles])

    def pixel_noise(profiles, half):
        """Per-channel pixel scatter of the residual in the side band (MAD about each position's side median)."""
        side_sel = (np.abs(d_values) >= half + inner) & (np.abs(d_values) <= half + outer)
        out = []
        with _quiet():
            for prof in profiles:
                block = prof[:, side_sel]
                dev = block - np.nanmedian(block, axis=1, keepdims=True)
                dev = dev[np.isfinite(dev)]
                out.append(1.4826 * float(np.median(np.abs(dev))) if dev.size >= 8 else 0.0)
        return np.array(out)

    def snr_series(core, side, left, right, half, pix):
        """Per-position S/N of (core median - side median): pixel noise scaled to the two medians' sample counts."""
        sig = core - side
        n_core = 2 * half + 1
        n_side = 2 * (outer - inner + 1)
        noise = pix * 1.2533 * math.sqrt(1.0 / n_core + 1.0 / n_side)
        parts = [np.nan_to_num(sig[c] / noise[c], nan=0.0) if noise[c] > 0 else np.zeros_like(sig[c]) for c in range(3)]
        return sig, noise, np.sum(parts, axis=0) / math.sqrt(3.0)

    half = half_min
    centre_shift = 0.0
    profiles = _sample_profiles(crops, origin, p0, u, n_vec, t_values, d_values)
    for _ in range(2):
        core, side, left, right = stats(profiles, half)
        sig, noise, snr = snr_series(core, side, left, right, half, pixel_noise(profiles, half))
        window = max(1, int(round(p["endpoint_smooth_px"] / step)))
        kernel = np.ones(window) / window
        smooth = np.convolve(np.nan_to_num(snr), kernel, mode="same")
        run = _longest_run(smooth >= p["endpoint_snr"])
        if run is None:
            return {"status": "no_signal", "noise": noise.tolist(), "centre_shift_px": centre_shift}
        a, z = run
        # median cross-profile over the run (channel-summed S/N) -> centre offset and FWHM
        with _quiet():
            prof_sum = np.zeros(d_values.size)
            for c in range(3):
                if noise[c] > 0:
                    prof_sum += np.nan_to_num(np.nanmedian(profiles[c][a:z + 1], axis=0) / noise[c])
        base = np.nanmedian(prof_sum[np.abs(d_values) >= half + inner]) if np.isfinite(prof_sum).any() else 0.0
        prof_sum = prof_sum - base
        peak_i = int(np.argmax(prof_sum))
        peak = prof_sum[peak_i]
        if peak <= 0:
            return {"status": "no_signal", "noise": noise.tolist(), "centre_shift_px": centre_shift}
        above = prof_sum >= 0.5 * peak
        lo = peak_i
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = peak_i
        while hi < d_values.size - 1 and above[hi + 1]:
            hi += 1
        fwhm = float(d_values[hi] - d_values[lo] + 1.0)
        weights = prof_sum[lo:hi + 1]
        shift = float(np.sum(d_values[lo:hi + 1] * weights) / np.sum(weights))     # sub-pixel centroid of the half-maximum region
        new_half = int(min(half_max, max(half_min, round(fwhm / 2.0))))
        if abs(shift) <= 3.0 and abs(shift) >= 0.05:
            p0 = p0 + n_vec * shift
            centre_shift += shift
            profiles = _sample_profiles(crops, origin, p0, u, n_vec, t_values, d_values)
        half = new_half
    core, side, left, right = stats(profiles, half)
    pix = pixel_noise(profiles, half)
    sig, noise, snr = snr_series(core, side, left, right, half, pix)
    window = max(1, int(round(p["endpoint_smooth_px"] / step)))
    smooth = np.convolve(np.nan_to_num(snr), np.ones(window) / window, mode="same")
    run = _longest_run(smooth >= p["endpoint_snr"])
    if run is None:
        return {"status": "no_signal", "noise": noise.tolist(), "centre_shift_px": centre_shift}
    a, z = run

    def snr_at(t_arr):
        """Channel-summed S/N at 1 px positions; NaN where the core has no data (ground, frame border)."""
        profs = _sample_profiles(crops, origin, p0, u, n_vec, t_arr, d_values)
        c_, s_, _, _ = stats(profs, half)
        parts = [np.nan_to_num((c_[c] - s_[c]) / noise[c], nan=0.0) if noise[c] > 0 else np.zeros(t_arr.size) for c in range(3)]
        out = np.sum(parts, axis=0) / math.sqrt(3.0)
        out[~np.isfinite(c_).any(axis=0)] = np.nan
        return out

    # endpoints refined at 1 px: walk outwards from the coarse end while both the raw and the 3 px smoothed S/N stay at
    # the threshold (an isolated noise hit further out does not count); positions without data stop the walk
    t_a, t_z = float(t_values[a]) - step / 2.0, float(t_values[z]) + step / 2.0
    fine_kernel = np.ones(3) / 3.0
    span = int(2 * step)
    for end in ("start", "stop"):
        centre_t = t_values[a] if end == "start" else t_values[z]
        t_fine = centre_t + np.arange(-span, span + 1, 1.0)
        raw = snr_at(t_fine)
        fine = np.convolve(np.nan_to_num(raw), fine_kernel, mode="same")
        hit = (fine >= p["endpoint_snr"]) & (raw >= p["endpoint_snr"])
        direction = -1 if end == "start" else 1
        i = hit.size - 1 if end == "start" else 0          # start from the innermost position and walk outwards
        last = None
        while 0 <= i < hit.size and hit[i]:
            last = i
            i += direction
        if last is not None:
            if end == "start":
                t_a = float(t_fine[last]) - 0.5
            else:
                t_z = float(t_fine[last]) + 0.5
    q0, q1 = p0 + u * t_a, p0 + u * t_z
    if search_beyond_px is None and (t_z - t_a) < float(p["min_length_px"]):
        return {"status": "too_short", "length_px": float(t_z - t_a), "x0": float(q0[0]), "y0": float(q0[1]), "x1": float(q1[0]),
                "y1": float(q1[1]), "noise": noise.tolist(), "centre_shift_px": float(centre_shift),
                "total_snr": float(np.nansum(snr[a:z + 1]) / math.sqrt(z - a + 1))}
    # the ends: is there data just beyond them (a trail cut by the ground or the frame border cannot be judged there)?
    beyond = snr_at(np.array([t_a - 1.5, t_z + 1.5]))
    height, width = frame_path.shape
    edge = float(p.get("edge_px", DEFAULT_EVIDENCE["edge_px"]))
    ends = {}
    for name, q, k in (("start", q0, 0), ("stop", q1, 1)):
        at_border = bool(q[0] < edge or q[0] > width - 1 - edge or q[1] < edge or q[1] > height - 1 - edge)
        ends[name] = {"data_beyond": bool(np.isfinite(beyond[k])), "at_border": at_border}
    # the run's coarse samples are trimmed to the refined endpoints (the 16 px smoothing lets the coarse run overshoot a
    # hard end by up to two samples, which would read as a fading end in the light curve)
    inside = np.nonzero((t_values[a:z + 1] >= t_a) & (t_values[a:z + 1] <= t_z))[0]
    if inside.size:
        a, z = a + int(inside[0]), a + int(inside[-1])
    level = side_level(_sample_profiles(frame_crops, origin, p0, u, n_vec, t_values, d_values), half)
    with _quiet():
        contrast = np.array([np.nanmedian((core[c][a:z + 1] - side[c][a:z + 1]) / level[c][a:z + 1]) for c in range(3)])
        level_median = [_finite(np.nanmedian(level[c][a:z + 1])) for c in range(3)]
    # flux inside |d| <= half + 2 over the run, side-band background removed per position, per channel
    foot = np.abs(d_values) <= half + 2
    flux = []
    for c in range(3):
        block = profiles[c][a:z + 1][:, foot] - side[c][a:z + 1][:, None]
        flux.append(float(np.nansum(block) * step))
    g = flux[1]
    colour = {"RG": float(flux[0] / g) if g > 0 else None, "BG": float(flux[2] / g) if g > 0 else None}
    total_snr = float(np.nansum(snr[a:z + 1]) / math.sqrt(z - a + 1))
    series = np.nan_to_num(snr[a:z + 1]).tolist()
    # Keep the search-window evidence separate from the actually selected extent.
    # The longest-run endpoint choice above does not prove that other runs belong
    # to this trail, or that the selected extent is the entire physical trail.
    visible = smooth >= p["endpoint_snr"]
    selected = (t_values >= t_a) & (t_values <= t_z)
    selected_visible = visible & selected
    visible_runs = int(np.count_nonzero(visible & ~np.r_[False, visible[:-1]]))
    selected_runs = int(np.count_nonzero(selected_visible & ~np.r_[False, selected_visible[:-1]]))
    unselected = visible & ~selected
    edges = np.diff(np.r_[False, unselected, False].astype(np.int8))
    significant_unselected = []
    # Reuse the existing minimum measurable length; this is a coarse diagnostic,
    # never an instruction to merge disconnected segments or to add pixels.
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        span_px = float((stop - start) * step)
        if span_px >= float(p["min_length_px"]):
            t0, t1 = float(t_values[start] - step / 2), float(t_values[stop - 1] + step / 2)
            q_start, q_stop = p0 + u * t0, p0 + u * t1
            significant_unselected.append({"t0_px": t0, "t1_px": t1, "span_px": span_px,
                "sample_count": int(stop - start), "x0": float(q_start[0]), "y0": float(q_start[1]),
                "x1": float(q_stop[0]), "y1": float(q_stop[1])})
    side_sel = (np.abs(d_values) >= half + inner) & (np.abs(d_values) <= half + outer)
    side_finite = [int(np.isfinite(prof[a:z + 1][:, side_sel]).sum()) for prof in profiles]
    quality = {"width_limit_reached": bool(half >= half_max),
               "profile_boundary_hit": bool(lo == 0 or hi == d_values.size - 1),
               "sideband_insufficient": any(count < 8 for count in side_finite),
               "visibility_scope_version": "2.0.0",
               "search_visible_runs": visible_runs, "selected_visible_runs": selected_runs,
               "search_extent_t_px": [float(t_values[0]), float(t_values[-1])],
               "selected_extent_t_px": [float(t_a), float(t_z)],
               "unselected_significant_runs": significant_unselected,
               "unselected_significance_min_length_px": float(p["min_length_px"]),
               "run_extent_basis": "thresholded_smoothed_samples_step_px_not_physical_endpoints",
               "disconnected_visible_runs": selected_runs > 1,
               "disconnected_visible_runs_scope": "selected_extent",
               # Compatibility: visible_runs retains the historical search count.
               "visible_runs": visible_runs, "visible_runs_scope": "full_search_extent",
               "sideband_finite_samples": side_finite,
               "clipping_evidence": "unknown", "flux_scope": "measured_core_only_not_complete_trail"}
    return {"status": "measured", "x0": float(q0[0]), "y0": float(q0[1]), "x1": float(q1[0]), "y1": float(q1[1]),
            "length_px": float(t_z - t_a), "half_width_px": int(half), "fwhm_px": float(fwhm), "centre_shift_px": float(centre_shift),
            "contrast": {"R": _finite(contrast[0]), "G": _finite(contrast[1]), "B": _finite(contrast[2])},
            "contrast_median": _finite(float(np.nanmedian(contrast))), "background": {"R": level_median[0], "G": level_median[1], "B": level_median[2]},
            "flux": {"R": flux[0], "G": flux[1], "B": flux[2]},
            "colour": colour, "noise": noise.tolist(), "total_snr": total_snr, "profile_step_px": step, "series": series, "ends": ends,
            "quality": quality, "clipping_evidence": "unknown"}


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _longest_run(mask: np.ndarray):
    best, start = None, None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    return best


# ----------------------------------------------------------------------------------------------------------------------
# per-frame detection
# ----------------------------------------------------------------------------------------------------------------------

MANUAL_ASSOCIATION_VERSION = "1.0.0"
MANUAL_ASSOCIATION_RULE = {"angle_deg": 3.0, "max_endpoint_offset_px": 3.0, "minimum_overlap_each": 0.5}


def marked_line_relation(mark: dict, segment: dict) -> dict:
    """Frozen conservative grid geometry; it describes overlap, not physical identity."""
    a = np.array([[mark["x0"], mark["y0"]], [mark["x1"], mark["y1"]]], dtype=np.float64)
    b = np.array([[segment["x0"], segment["y0"]], [segment["x1"], segment["y1"]]], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"matches": False, "reason": "nonfinite_geometry"}
    u, v = a[1] - a[0], b[1] - b[0]
    la, lb = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if min(la, lb) <= 0:
        return {"matches": False, "reason": "degenerate_geometry"}
    u, v = u / la, v / lb
    angle = math.degrees(math.acos(float(np.clip(abs(u @ v), 0, 1))))
    delta = b - a[0]
    offset = float(np.max(np.abs(delta[:, 0] * u[1] - delta[:, 1] * u[0])))
    t = delta @ u
    overlap = max(0.0, min(la, float(t.max())) - max(0.0, float(t.min())))
    fractions = [overlap / la, overlap / lb]
    limit = min(float(mark.get("half_width_px", 30)), MANUAL_ASSOCIATION_RULE["max_endpoint_offset_px"])
    matches = angle <= MANUAL_ASSOCIATION_RULE["angle_deg"] and offset <= limit and min(fractions) >= MANUAL_ASSOCIATION_RULE["minimum_overlap_each"]
    return {"matches": bool(matches), "angle_deg": angle, "max_endpoint_offset_px": offset,
            "overlap_each": fractions, "rule_version": MANUAL_ASSOCIATION_VERSION}


def measure_mark(frame, master, mark: dict, segments: list[dict], *, params: dict) -> tuple[dict, dict]:
    """Choose a unique geometric target; never substitute the longest competing trail."""
    detected = [seg for seg in segments if seg.get("status") == "detected"]
    relations = [marked_line_relation(mark, seg) for seg in detected]
    matches = [i for i, relation in enumerate(relations) if relation["matches"]]
    evidence = {"version": MANUAL_ASSOCIATION_VERSION, "rule": dict(MANUAL_ASSOCIATION_RULE),
                "detected_count": len(detected), "matching_indices": matches, "relations": relations,
                "detector_budget_exceeded": any(seg.get("status") == "saturated" for seg in segments)}
    endpoints = {key: mark[key] for key in ("x0", "y0", "x1", "y1")}
    if len(matches) == 1:
        seg = {**detected[matches[0]], "mark_association": {**evidence, "status": "unique_geometric_match"}}
        measurement = measure_segment(frame, master, seg["x0"], seg["y0"], seg["x1"], seg["y1"], params=params)
    elif detected:
        reason = "multiple_geometric_matches" if matches else "no_geometric_match"
        seg = {"status": "mark_unresolved", **endpoints, "mark_association": {**evidence, "status": reason}}
        measurement = {"status": "ambiguous_target", "reason": reason, "clipping_evidence": "unknown"}
    else:
        seg = {"status": "mark_only", **endpoints, "mark_association": {**evidence, "status": "no_detected_segment"}}
        measurement = measure_segment(frame, master, **endpoints, params=params, search_beyond_px=0.0)
    return seg, measurement


def detect_frame(frame_path, master_path, *, params: dict | None = None, marks=(), check=None) -> dict:
    """Candidates of one exported frame: automatic segments plus manual marks (corridor search at the lower threshold)."""
    p = dict(DEFAULT_DETECTION, **(params or {}))
    frame_path, master_path = _planes(frame_path), _planes(master_path)
    det = detection_image(frame_path, master_path, params=p)
    if check:
        check()
    segments = find_segments(det, params=p)
    saturated = bool(segments and segments[0].get("status") == "saturated")
    results = []
    for seg in ([] if saturated else segments):
        m = measure_segment(frame_path, master_path, seg["x0"], seg["y0"], seg["x1"], seg["y1"], params=p)
        results.append({"source": "auto", "detection": seg, "measurement": m})
        if check:
            check()
    for mark in marks:
        corridor = (mark["x0"], mark["y0"], mark["x1"], mark["y1"], float(mark.get("half_width_px", 30)))
        segs = find_segments(det, params=p, threshold=p["mark_threshold_sigma"], corridor=corridor)
        seg, m = measure_mark(frame_path, master_path, mark, segs, params=p)
        results.append({"source": "manual", "mark": dict(mark), "detection": seg, "measurement": m})
    return {"sigma": det["sigma"], "bin": det["bin"], "shape": list(det["shape"]), "saturated": saturated,
            "foreground_pixels": segments[0].get("foreground_pixels") if saturated else None, "segments": results}


# ----------------------------------------------------------------------------------------------------------------------
# evidence across frames and suggestions
# ----------------------------------------------------------------------------------------------------------------------

def _line_relation(a: dict, b: dict) -> dict:
    """Geometry between two measured segments on the same grid: angle difference, lateral offset of b's midpoint from
    a's line, along-line overlap fraction (of the shorter) and the gap between the projected intervals."""
    pa0, pa1 = np.array([a["x0"], a["y0"]]), np.array([a["x1"], a["y1"]])
    pb0, pb1 = np.array([b["x0"], b["y0"]]), np.array([b["x1"], b["y1"]])
    ua = pa1 - pa0
    la = float(np.hypot(*ua))
    ua = ua / la
    na = np.array([-ua[1], ua[0]])
    ub = pb1 - pb0
    lb = float(np.hypot(*ub))
    angle = _angle_diff_deg(math.degrees(math.atan2(ua[1], ua[0])), math.degrees(math.atan2(ub[1], ub[0])))
    mid_b = (pb0 + pb1) / 2.0
    lateral = abs(float((mid_b - pa0) @ na))
    ta = (0.0, la)
    tb = sorted([float((pb0 - pa0) @ ua), float((pb1 - pa0) @ ua)])
    overlap = max(0.0, min(ta[1], tb[1]) - max(ta[0], tb[0]))
    gap = max(0.0, max(ta[0], tb[0]) - min(ta[1], tb[1]))
    return {"angle_deg": angle, "lateral_px": lateral, "overlap_fraction": overlap / max(1e-9, min(la, lb)), "gap_px": gap,
            "gap_lengths": gap / max(1e-9, min(la, lb))}


def _running(x: np.ndarray, window: int, fn) -> np.ndarray:
    """``fn`` over a centred running window (odd width, edge-padded), e.g. a running median or percentile."""
    window = max(1, int(window))
    window += 1 - window % 2
    pad = window // 2
    view = np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="edge"), window)
    return fn(view, axis=-1)


def _periodicity(series: list, step: float, e: dict) -> dict:
    """Periodic brightness modulation with dark gaps (aircraft strobes, tumbling flashes) along the measured run.

    1.1.0: the per-position S/N series is divided by its local envelope (running 90th percentile over 64 px) so that the
    modulation depth is judged relative to the trail's own brightness, the 2-sample alternation that the 4 px lattice
    sampling imprints on a diagonal trail is removed, the running median over a quarter of the run is subtracted
    (smooth light curves are not modulation), and the autocorrelation of what remains must have a local maximum of
    at least ``periodic_corr`` at a lag of at least three samples (a two-sample period cannot be told from the sampling
    alternation) with at least three periods in the run; the dark fraction is the share of positions below
    ``dashed_level`` of the local envelope.  The 1.0.0 test (median-subtracted autocorrelation, global 90th percentile)
    fired on any smooth profile, including the photographed meteor.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < 12:
        return {"periodic": False, "reason": "too short"}
    envelope = _running(x, round(64.0 / step), lambda v, axis: np.percentile(v, 90, axis=axis))
    floor = max(1e-9, 0.05 * float(np.percentile(x, 90)))
    m = x / np.maximum(envelope, floor)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    m = m - float(np.mean(m * sign)) * sign
    dashed = float(np.mean(m < e["dashed_level"]))
    r = m - _running(m, max(12, n // 4), np.median)
    denom = float(np.dot(r, r))
    if denom <= 0:
        return {"periodic": False, "reason": "flat", "dashed_fraction": dashed}
    min_lag = max(3, int(round(e["periodic_min_lag_px"] / step)))
    max_lag = n // 3
    if max_lag < min_lag:
        return {"periodic": False, "reason": "too short for three periods", "dashed_fraction": dashed}
    corr = np.array([float(np.dot(r[:-lag], r[lag:])) / denom for lag in range(1, max_lag + 2)])
    best_lag, best = None, None
    for lag in range(min_lag, max_lag + 1):
        i = lag - 1
        if corr[i] >= corr[i - 1] and corr[i] >= corr[i + 1] and (best is None or corr[i] > best):
            best, best_lag = float(corr[i]), lag
    periodic = best is not None and best >= e["periodic_corr"] and dashed >= e["dashed_fraction"]
    return {"periodic": bool(periodic), "autocorrelation_peak": best, "period_px": (best_lag * step) if best_lag else None,
            "dashed_fraction": dashed}


def _light_curve(m: dict, e: dict) -> dict:
    """Do the ends of the measured run fade out or stop abruptly?  Levels are medians of the outer ``ends_fraction`` of
    the series relative to its 90th percentile.  An end is judged only where a fade could be seen: the profile's 90th
    percentile S/N is at least ``ends_min_p90_snr`` (otherwise the endpoint threshold itself cuts the run at about a
    third of the peak), data continues beyond the end and the end is not at the frame border. Fading is supporting
    evidence, not a necessary condition for a meteor: abrupt, cropped or unjudged ends do not exclude one. A
    shutter-limited plateau is compatible with a satellite, but this profile alone does not certify identity.
    This is a spatial brightness profile, not a measured time curve or direction of flight."""
    x = np.asarray(m.get("series") or [], dtype=np.float64)
    n = x.size
    out = {"start_level": None, "stop_level": None, "peak_position": None, "ends": {"start": "unjudged", "stop": "unjudged"},
           "p90_snr": None}
    if n < 4:
        out["reason"] = "too short"
        return out
    p90 = float(np.percentile(x, 90))
    out["p90_snr"] = p90
    if p90 <= 0:
        out["reason"] = "flat"
        return out
    k = max(2, int(round(e["ends_fraction"] * n)))
    levels = {"start": float(np.median(x[:k]) / p90), "stop": float(np.median(x[-k:]) / p90)}
    out["start_level"], out["stop_level"] = levels["start"], levels["stop"]
    smooth = np.convolve(x, np.ones(5) / 5.0, mode="same")
    out["peak_position"] = float(np.argmax(smooth) / max(1, n - 1))
    info = m.get("ends") or {}
    for name in ("start", "stop"):
        end = info.get(name) or {}
        if p90 < e["ends_min_p90_snr"] or not end.get("data_beyond", False) or end.get("at_border", False):
            out["ends"][name] = "unjudged"
        elif levels[name] <= e["ends_faded_max"]:
            out["ends"][name] = "faded"
        elif levels[name] >= e["ends_abrupt_min"]:
            out["ends"][name] = "abrupt"
        else:
            out["ends"][name] = "between"
    return out


def add_evidence(candidates: list[dict], frame_order: list[str], *, params: dict | None = None, check=None,
                 check_every: int = 8) -> None:
    """Evidence in place: continuation in an adjacent frame, persistence at the same place, periodic modulation, light curve.
    ``check`` (efficiency P1c T-3): called every ``check_every`` outer iterations of the pairwise loop (a cancel probe;
    the order of the work is unchanged)."""
    e = dict(DEFAULT_EVIDENCE, **(params or {}))
    position = {stem: i for i, stem in enumerate(frame_order)}
    measured = [c for c in candidates if c["measurement"].get("status") == "measured"]
    for index, c in enumerate(measured):
        if check is not None and index % check_every == 0:
            check()
        m = c["measurement"]
        ev = {"continues": [], "persistent_frames": [c["stem"]], "periodic": _periodicity(m.get("series", []), m.get("profile_step_px", 4.0), e),
              "light_curve": _light_curve(m, e)}
        for other in measured:
            if other is c or other["stem"] == c["stem"]:
                continue
            rel = _line_relation(m, other["measurement"])
            if rel["angle_deg"] > e["continue_angle_deg"] or rel["lateral_px"] > e["continue_offset_px"]:
                continue
            adjacent = abs(position.get(other["stem"], -99) - position.get(c["stem"], 99)) == 1
            if rel["overlap_fraction"] >= e["persistent_overlap_min"]:
                if other["stem"] not in ev["persistent_frames"]:
                    ev["persistent_frames"].append(other["stem"])
            elif adjacent and rel["overlap_fraction"] <= e["continue_overlap_max"] and rel["gap_lengths"] <= e["continue_gap_lengths"]:
                ev["continues"].append({"candidate": other["id"], "stem": other["stem"], **{k: round(v, 3) for k, v in rel.items()}})
        ev["persistent"] = len(ev["persistent_frames"]) >= e["persistent_frames"]
        ev["adjacent_frames_available"] = [s for s in frame_order if abs(position[s] - position[c["stem"]]) == 1]
        c["evidence"] = ev
    for c in candidates:
        if c["measurement"].get("status") != "measured":
            c["evidence"] = {"continues": [], "persistent_frames": [c["stem"]], "persistent": False, "periodic": {"periodic": False, "reason": "not measured"},
                             "light_curve": {"ends": {"start": "unjudged", "stop": "unjudged"}, "reason": "not measured"}}


def suggest(candidate: dict, *, weak_snr: float = DEFAULT_DETECTION["weak_snr"]) -> dict:
    """Suggested class and the evidence it rests on (RUN_LOG §28 rule, light-curve step added in 1.1.0); never more than
    a suggestion: persistent -> artifact; periodic dashes -> aircraft; continues in an adjacent frame -> satellite; weak
    -> unknown; both ends fade -> meteor; both ends abrupt -> satellite; anything else -> unknown."""
    m = candidate["measurement"]
    ev = candidate.get("evidence") or {}
    reasons = []
    if m.get("status") != "measured":
        return {"class": "unknown", "reasons": [f"not measured ({m.get('status')})"]}
    snr = float(m.get("total_snr") or 0.0)
    if ev.get("persistent"):
        reasons.append(f"same place in {len(ev['persistent_frames'])} frames: {', '.join(ev['persistent_frames'])}")
        return {"class": "artifact", "reasons": reasons}
    per = ev.get("periodic") or {}
    if per.get("periodic"):
        reasons.append(f"periodic brightness modulation along the trail (period {per.get('period_px'):.0f} px, autocorrelation {per.get('autocorrelation_peak'):.2f}, dark fraction {per.get('dashed_fraction'):.2f})")
        return {"class": "aircraft", "reasons": reasons}
    if ev.get("continues"):
        for link in ev["continues"]:
            reasons.append(f"geometrically compatible segment in adjacent frame {link['stem']} (angle {link['angle_deg']:.1f}°, lateral {link['lateral_px']:.1f} px, gap {link['gap_px']:.0f} px); same-object identity unverified")
        return {"class": "satellite", "reasons": reasons}
    if snr < weak_snr:
        reasons.append(f"total S/N {snr:.1f} below {weak_snr:.0f}")
        return {"class": "unknown", "reasons": reasons}
    adjacent = ev.get("adjacent_frames_available") or []
    reasons.append(f"single frame candidate; no compatible continuation was linked in the scanned adjacent-frame catalog ({', '.join(adjacent) if adjacent else 'none available'}); original-exposure absence is not established; no periodic modulation, total S/N {snr:.1f}")
    lc = ev.get("light_curve") or {}
    ends = lc.get("ends") or {}
    judged = {name: ends.get(name, "unjudged") for name in ("start", "stop")}
    levels = {name: lc.get(f"{name}_level") for name in ("start", "stop")}

    def describe(name):
        level = levels[name]
        return f"{name} {judged[name]}" + (f" ({level:.2f} of the peak)" if isinstance(level, (int, float)) else "")

    reasons.append("light curve: " + ", ".join(describe(name) for name in ("start", "stop")))
    if judged["start"] == "faded" and judged["stop"] == "faded":
        reasons.append("both measured ends fade inside the frame; supports a meteor suggestion, not physical identity certification")
        return {"class": "meteor", "reasons": reasons}
    if judged["start"] == "abrupt" and judged["stop"] == "abrupt":
        reasons.append("both measured ends stop abruptly near full brightness; compatible with a shutter-limited trail, not physical identity certification")
        return {"class": "satellite", "reasons": reasons}
    reasons.append("the ends do not decide identity; an abrupt or unjudged end does not exclude a meteor (unjudged: too faint, no data beyond, or at the frame border)")
    return {"class": "unknown", "reasons": reasons}


def resolve_decisions(record: dict) -> None:
    """Resolve intent separately from measurability; eligibility never asserts application."""
    from .protection_intent import resolve_candidate_intents
    result = resolve_candidate_intents(record["candidates"], record.get("policy"))
    # Preserve existing candidate objects for callers holding review references.
    for candidate, resolved in zip(record["candidates"], result["candidates"]):
        candidate.update(resolved)
    record["policy"] = result["policy"]
    record["protection_intent_summary"] = result["summary"]


# ----------------------------------------------------------------------------------------------------------------------
# transient layer
# ----------------------------------------------------------------------------------------------------------------------

def layer_box(frame_path, master_path, candidate: dict, scale: list[float], frame_median: list[float], ref_median: list[float],
              *, params: dict | None = None) -> dict:
    """The candidate's footprint on the stack grid as a box: normalised residual of the source frame inside the trail,
    feathered, zero where the frame or the master has no data.  Returns {"x0", "y0", "data": (3, h, w) top-down}."""
    lp = dict(DEFAULT_LAYER, **(params or {}))
    m = candidate["measurement"]
    p0 = np.array([m["x0"], m["y0"]], dtype=np.float64)
    p1 = np.array([m["x1"], m["y1"]], dtype=np.float64)
    length = float(np.hypot(*(p1 - p0)))
    u = (p1 - p0) / length
    n_vec = np.array([-u[1], u[0]])
    half = float(m["half_width_px"]) + lp["footprint_margin_px"]
    feather = float(lp["feather_px"])
    ext = float(lp["end_extension_px"])
    reach = half + feather + 1.0
    xs = [p0[0] - ext * u[0], p1[0] + ext * u[0]]
    ys = [p0[1] - ext * u[1], p1[1] + ext * u[1]]
    x0, y0 = int(math.floor(min(xs) - reach)), int(math.floor(min(ys) - reach))
    x1, y1 = int(math.ceil(max(xs) + reach)) + 1, int(math.ceil(max(ys) + reach)) + 1
    frame_path, master_path = _planes(frame_path), _planes(master_path)
    height, width = frame_path.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return {"x0": 0, "y0": 0, "data": np.zeros((3, 0, 0), dtype=np.float32)}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - p0[0], yy - p0[1]
    along = dx * u[0] + dy * u[1]
    across = np.abs(-dx * u[1] + dy * u[0])
    inside_t = (along >= -ext) & (along <= length + ext)
    w = np.clip((half + feather - across) / feather, 0.0, 1.0)
    w[~inside_t] = 0.0
    data = np.zeros((3, y1 - y0, x1 - x0), dtype=np.float32)
    finite_nonzero = np.ones(w.shape, dtype=bool)
    for c in range(3):
        f = _crop(frame_path, c, x0, x1, y0, y1)
        mm = _crop(master_path, c, x0, x1, y0, y1)
        resid = scale[c] * (f - frame_median[c]) + ref_median[c] - mm
        resid[(f == 0) | (mm == 0)] = 0.0
        data[c] = (resid * w).astype(np.float32)
        finite_nonzero &= np.isfinite(f) & np.isfinite(mm) & (f != 0) & (mm != 0)
    footprint = w > 0
    support = {"grid_footprint_pixels": int(footprint.sum()),
               "grid_finite_nonzero_pixels": int(np.count_nonzero(footprint & finite_nonzero)),
               "grid_zero_or_nonfinite_pixels": int(np.count_nonzero(footprint & ~finite_nonzero)),
               "grid_nonzero_residual_pixels": int(np.count_nonzero(np.any(np.isfinite(data) & (data != 0), axis=0))),
               "support_evidence": "nonzero_finite_samples_not_complete_interpolation_support"}
    from .transient_support import attach_source_support
    box = {"x0": int(x0), "y0": int(y0), "data": data, "weight": w, "footprint_pixels": int(footprint.sum()),
           "application_evidence": support, "grid_shape": (height, width)}
    supported_box = attach_source_support(box, finite_nonzero)
    supported_box["application_evidence"]["grid_nonzero_residual_pixels"] = int(np.count_nonzero(np.any(supported_box["data"] != 0, axis=0)))
    return supported_box


def warp_box_to_native(box: dict, model, native_shape: tuple[int, int], *, kernel: str | None = None, margin: int | None = None, check=None) -> dict:
    """Warp a stack-grid box to native geometry (the master's warp-back: native pixel p shows grid point model.apply(p))."""
    from .transient_support import mask_warped_box
    from .resample import DEFAULT_KERNEL
    def finish(result):
        if "validity" in box and "grid_shape" not in box:
            return {**result, "native_support": {"status": "unavailable", "reason": "registered_grid_shape_absent"}}
        return mask_warped_box(box, result, model, box.get("grid_shape"), kernel=kernel or DEFAULT_KERNEL, check=check)
    data = box["data"]
    if data.size == 0:
        return finish({"x0": 0, "y0": 0, "data": np.zeros((3, 0, 0), dtype=np.float32)})
    height, width = native_shape
    margin = DEFAULT_LAYER["warp_margin_px"] if margin is None else margin
    if model is None:
        return finish({"x0": box["x0"], "y0": box["y0"], "data": data.copy()})
    gx0, gy0 = box["x0"], box["y0"]
    gh, gw = data.shape[1:]
    corners = np.array([[gx0, gy0], [gx0 + gw - 1, gy0], [gx0, gy0 + gh - 1], [gx0 + gw - 1, gy0 + gh - 1]], dtype=np.float64)
    native = model.inverse(corners)
    nx0 = max(0, int(math.floor(native[:, 0].min())) - margin)
    ny0 = max(0, int(math.floor(native[:, 1].min())) - margin)
    nx1 = min(width, int(math.ceil(native[:, 0].max())) + margin + 1)
    ny1 = min(height, int(math.ceil(native[:, 1].max())) + margin + 1)
    if nx1 <= nx0 or ny1 <= ny0:
        return finish({"x0": 0, "y0": 0, "data": np.zeros((3, 0, 0), dtype=np.float32)})

    def mapping(points):
        src = model.apply(points + np.array([nx0, ny0], dtype=np.float64))
        return src - np.array([gx0, gy0], dtype=np.float64)

    kwargs = {"kernel": kernel} if kernel else {}
    if data.shape[0] == 3:
        out = warp_planes_fused(data, mapping, fill=0.0, output_shape=(ny1 - ny0, nx1 - nx0), **kwargs)
    else:   # a malformed box keeps the former per-plane path (and its IndexError)
        out = np.stack([warp_image(data[c], mapping, fill=0.0, output_shape=(ny1 - ny0, nx1 - nx0), **kwargs)
                        for c in range(3)])
    return finish({"x0": int(nx0), "y0": int(ny0), "data": np.nan_to_num(out).astype(np.float32)})


def apply_boxes(target_bottom_up: np.ndarray, boxes: list[dict], *, weight_top_down: np.ndarray | None = None) -> np.ndarray:
    """Add native boxes (top-down) into a BOTTOM-UP (H, W, 3) sky array in place; ``weight_top_down`` (0..1) scales them."""
    height = target_bottom_up.shape[0]
    for box in boxes:
        data = box["data"]
        if data.size == 0:
            continue
        x0, y0 = box["x0"], box["y0"]
        h, w = data.shape[1:]
        block = np.moveaxis(data, 0, -1)
        if weight_top_down is not None:
            block = block * weight_top_down[y0:y0 + h, x0:x0 + w][..., None]
        rows = slice(height - (y0 + h), height - y0)
        target_bottom_up[rows, x0:x0 + w] += block[::-1]
    return target_bottom_up


# ----------------------------------------------------------------------------------------------------------------------
# records and review crops
# ----------------------------------------------------------------------------------------------------------------------

def write_candidates(directory: Path, record: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CANDIDATES_FILE
    from .records import atomic_json
    atomic_json(path, record, default=_json_default)
    return path


def read_candidates(directory: Path) -> dict | None:
    path = Path(directory) / CANDIDATES_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serialisable: {type(value).__name__}")


def _crop_box(m: dict) -> tuple[int, int, int, int]:
    """The review crop's top-down box (x0, y0, x1, y1) around a measured candidate (may reach outside the image)."""
    x0, y0, x1, y1 = m["x0"], m["y0"], m["x1"], m["y1"]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    extent = max(64.0, abs(x1 - x0), abs(y1 - y0)) * 1.2 + 32
    half = extent / 2.0
    return (int(cx - half), int(cy - half), int(cx + half) + 1, int(cy + half) + 1)


def write_crop_png(frame_path, master_path, candidate: dict, path: Path, *, size: int = 512, check=None) -> dict:
    """Review image: the source frame (G) and its residual against the master around the candidate, side by side.

    Efficiency P1c T-3m: reads only the pixels the downsampled image shows (``_sampled``), bit-identical to
    ``_write_crop_png_reference``, which reads the box's full-width rows (up to about 1 GB for a full-height box).
    Every PNG value is the same file element through the same element-wise float32 operations (add, subtract, divide:
    correctly rounded on every SIMD path; arcsinh, rint and isfinite have baseline loops only in numpy 2.5.3, pinned by
    tests/test_p1c_site_trail_crops.py). A frame and master on different grids (shapes), which the scan refuses before
    any crop (``detection_image``), keep the reference. ``check`` (P1c T-3: a pool unit's abort probe) is called
    within the sampled reads (``_sampled``); nothing is written when it raises."""
    m = candidate["measurement"]
    if m.get("status") != "measured":
        return {"written": False}
    box = _crop_box(m)
    frame_planes, master_planes = _planes(frame_path), _planes(master_path)
    if frame_planes.shape != master_planes.shape:
        return _write_crop_png_reference(frame_planes, master_planes, candidate, path, size=size)
    height, width = frame_planes.shape
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(width, box[2]), min(height, box[3])
    if x1 <= x0 or y1 <= y0:
        return {"written": False}
    step = max(1, int(math.ceil(max(y1 - y0, x1 - x0) / size)))
    frame = _sampled(*frame_planes.planes[1], x0, x1, y0, y1, step, check=check)
    master = _sampled(*master_planes.planes[1], x0, x1, y0, y1, step, check=check)
    resid = frame - master
    resid[(frame == 0) | (master == 0)] = np.nan
    left = fits_crop.asinh_stretch(frame)
    right = fits_crop.asinh_stretch(resid)
    gap = np.zeros((left.shape[0], 4), dtype=np.uint8)
    fits_crop.write_png_gray(path, np.hstack([left, gap, right]))
    return {"written": True, "box": list(box), "downsample": step, "layout": "source G | residual G"}


def _write_crop_png_reference(frame_path, master_path, candidate: dict, path: Path, *, size: int = 512) -> dict:
    """The review image as before P1c T-3m (full-width rows of the box, then column slice and downsampling): kept for
    inputs outside the sampled reader's scope (frame and master shapes differ) and as the tests' reference."""
    m = candidate["measurement"]
    if m.get("status") != "measured":
        return {"written": False}
    box = _crop_box(m)
    frame = _crop(frame_path, 1, box[0], box[2], box[1], box[3])
    master = _crop(master_path, 1, box[0], box[2], box[1], box[3])
    if frame.size == 0:
        return {"written": False}
    resid = frame - master
    resid[(frame == 0) | (master == 0)] = np.nan
    step = max(1, int(math.ceil(max(frame.shape) / size)))
    left = fits_crop.asinh_stretch(frame[::step, ::step])
    right = fits_crop.asinh_stretch(resid[::step, ::step])
    gap = np.zeros((left.shape[0], 4), dtype=np.uint8)
    fits_crop.write_png_gray(path, np.hstack([left, gap, right]))
    return {"written": True, "box": list(box), "downsample": step, "layout": "source G | residual G"}
