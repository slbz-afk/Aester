"""Buffered strict-support kernels (efficiency P1a). Outputs are bit-identical to the reference paths,
which stay in place as the test oracles:

- ``registered_mask_kernel`` == ``registered_support.registered_mask`` with a ``make_export_mapping`` mapping;
- ``tap_state`` / ``native_block`` == the per-point result of
  ``interpolation_support.sample_support(..., kernel="cubic", boundary="clamp")``.

Nothing here writes files. One ``ExportMaskKernel`` per worker (its buffers are per instance).
"""
from __future__ import annotations

import numpy as np

from .interpolation_support import sample_support
from .siril_support_map import ExportParameters, distortion_grid

KERNEL_POINTS = 65536
TAPS_NONE, TAPS_MIXED, TAPS_ALL = 0, 1, 3
# Efficiency P1c G-4t, the frame pool's parallel path only (rule R18): mixed footprints go through sample_support in
# chunks of PROBE_POINTS with a probe before each. A 64-row block at W=8280 whose every footprint is mixed spends
# 0.32 s in one sample_support call on the development PC; a chunk takes about 80 ms. sample_support decides each
# point from that point alone (elementwise), so the chunks give the same booleans as one call; without a probe (the
# serial path) it is the former single call.
PROBE_POINTS = 1 << 17


def kernel_rows(width: int) -> int:
    """Rows per block, about 64k points: 7 rows at 8280 px, 6 at 9504 px (about 8 MB of buffers)."""
    return max(1, KERNEL_POINTS // width)


class ExportMaskKernel:
    """Centre support (before the output cross) of the Siril 1.4.4 export map, one row block at a time.

    Per element: exactly the IEEE operations, dtypes and association of make_export_mapping's closure,
    followed by support_anchor.anchor_support. Points the closure drops (non-finite, outside the SIP
    domain) are given coordinate 0 so every index stays in range, and are masked out afterwards.
    Not thread-safe: one instance per worker.
    """

    def __init__(self, width: int, height: int, spec, rows: int | None = None):
        self.width, self.height, self.spec = width, height, spec
        self.rows = kernel_rows(width) if rows is None else rows
        shape = (self.rows, width)

        def f64():
            return np.empty(shape, np.float64)

        def f32():
            return np.empty(shape, np.float32)

        def i64():
            return np.empty(shape, np.int64)

        self.xs = np.arange(width, dtype=np.float64)        # == the closure's points[:, 0] (exact integers)
        self.den, self.t64, self.px, self.py = f64(), f64(), f64(), f64()
        self.valid, self.tmp = np.empty(shape, np.bool_), np.empty(shape, np.bool_)
        self.ix, self.iy, self.flat = i64(), i64(), i64()
        self.grid_flat = None                                # SIP: fetched by the first block with a valid point
        if spec is not None:
            self.rec, self.sx, self.sy = f32(), f32(), f32()
            self.c1, self.c2, self.a, self.bb = f32(), f32(), f32(), f32()
            self.w11, self.w12, self.w21, self.w22 = f32(), f32(), f32(), f32()
            self.acc, self.g = f32(), f32()
            self.f01, self.f10, self.f11 = i64(), i64(), i64()
        else:
            self.rec64, self.sy64 = f64(), f64()

    def matches(self, params: ExportParameters) -> bool:
        return (self.width, self.height, self.spec) == (params.width, params.height, params.spec)

    def set_frame(self, inverse: np.ndarray) -> None:
        self.inv = inverse
        # xb*inverse[r, 0] does not depend on the row: the same float64 products the closure forms per point.
        self.X0, self.X1, self.X2 = self.xs * inverse[0, 0], self.xs * inverse[1, 0], self.xs * inverse[2, 0]

    def block(self, y0: int, y1: int, anchors_flat: np.ndarray, out: np.ndarray, grid_check=None) -> None:
        """``grid_check`` (efficiency P1c F-2, the frame pool's parallel path only): probed before this kernel waits
        for the shared SIP grid, which another worker may be building (2.26 s at 45.7 MP)."""
        s = slice(0, y1 - y0)
        inv, W, H = self.inv, self.width, self.height
        yb = (H - 1.) - np.arange(y0, y1, dtype=np.float64)[:, None]        # height-1.-p[:, 1]
        den, t64, valid, tmp = self.den[s], self.t64[s], self.valid[s], self.tmp[s]
        px, py, ix, iy, flat = self.px[s], self.py[s], self.ix[s], self.iy[s], self.flat[s]
        if self.spec is not None:
            rec, sx, sy = self.rec[s], self.sx[s], self.sy[s]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.add(self.X2, yb * inv[2, 1], out=den)
                np.add(den, inv[2, 2], out=den)
                np.divide(1., den, out=t64)
                np.copyto(rec, t64, casting="same_kind")                        # reciprocal.astype(float32)
                np.add(self.X0, yb * inv[0, 1], out=t64)
                np.add(t64, inv[0, 2], out=t64)
                np.multiply(t64, rec, out=t64)                                   # float64 * float32 -> float64
                np.copyto(sx, t64, casting="same_kind")                         # sx.astype(float32)
                np.add(self.X1, yb * inv[1, 1], out=t64)
                np.add(t64, inv[1, 2], out=t64)
                np.multiply(t64, rec, out=t64)
                np.copyto(sy, t64, casting="same_kind")
            np.isfinite(sx, out=valid)
            valid &= np.isfinite(sy, out=tmp)
            valid &= np.isfinite(den, out=tmp)
            valid &= np.not_equal(den, 0., out=tmp)
            valid &= np.greater_equal(sx, 0., out=tmp)                          # float32 vs weak Python float
            valid &= np.less(sx, W - 1., out=tmp)
            valid &= np.greater_equal(sy, 0., out=tmp)
            valid &= np.less(sy, H - 1., out=tmp)
            if not valid.any():                    # the closure returns early: grid untouched, all sentinel
                out[...] = False
                return
            if self.grid_flat is None:             # the closure's first grid access for this frame's kernel
                grid = distortion_grid(W, H, self.spec, check=grid_check)
                self.grid_flat = (grid[0].reshape(-1), grid[1].reshape(-1))   # views: the grid is C-contiguous
            invalid = np.logical_not(valid, out=tmp)
            np.copyto(sx, 0., where=invalid)                                    # dropped points -> index 0
            np.copyto(sy, 0., where=invalid)
            c1, c2, a, bb = self.c1[s], self.c2[s], self.a[s], self.bb[s]
            # ix = floor(sx).astype(int64); c1 = sx - ix.astype(float32) == sx - floor(sx) (0 <= floor < 2**15).
            np.floor(sx, out=a)
            np.copyto(ix, a, casting="unsafe")
            np.subtract(sx, a, out=c1)
            np.floor(sy, out=bb)
            np.copyto(iy, bb, casting="unsafe")
            np.subtract(sy, bb, out=c2)
            one = np.float32(1.)
            np.subtract(one, c1, out=a)
            np.subtract(one, c2, out=bb)
            w11, w12, w21, w22 = self.w11[s], self.w12[s], self.w21[s], self.w22[s]
            np.multiply(a, bb, out=w11)                                         # (one-c1)*(one-c2)
            np.multiply(c1, bb, out=w12)                                        # c1*(one-c2)
            np.multiply(a, c2, out=w21)                                         # (one-c1)*c2
            np.multiply(c1, c2, out=w22)                                        # c1*c2
            np.multiply(iy, W, out=flat)
            flat += ix
            f01, f10, f11 = self.f01[s], self.f10[s], self.f11[s]
            np.add(flat, 1, out=f01)
            np.add(flat, W, out=f10)
            np.add(flat, W + 1, out=f11)
            acc, g = self.acc[s], self.g[s]
            for axis in range(2):
                plane = self.grid_flat[axis]
                # ((g00*w11 + g01*w12) + g10*w21) + g11*w22 in float32, left-associated as in the closure.
                # plane[index] (1-D source, int64 index) is NumPy's fast fancy-indexing path.
                np.multiply(plane[flat], w11, out=acc)
                np.multiply(plane[f01], w12, out=g)
                np.add(acc, g, out=acc)
                np.multiply(plane[f10], w21, out=g)
                np.add(acc, g, out=acc)
                np.multiply(plane[f11], w22, out=g)
                np.add(acc, g, out=acc)
                if axis == 0:
                    np.copyto(px, acc)                                          # answer[:, 0] = out
                else:
                    np.copyto(py, acc)
                    np.subtract(H - 1., py, out=py)                             # height-1.-out.astype(float64)
        else:
            rec64, sy64 = self.rec64[s], self.sy64[s]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.add(self.X2, yb * inv[2, 1], out=den)
                np.add(den, inv[2, 2], out=den)
                np.divide(1., den, out=rec64)
                np.add(self.X0, yb * inv[0, 1], out=px)
                np.add(px, inv[0, 2], out=px)
                np.multiply(px, rec64, out=px)
                np.add(self.X1, yb * inv[1, 1], out=sy64)
                np.add(sy64, inv[1, 2], out=sy64)
                np.multiply(sy64, rec64, out=sy64)
                np.isfinite(px, out=valid)
                valid &= np.isfinite(sy64, out=tmp)
                valid &= np.isfinite(den, out=tmp)
                valid &= np.not_equal(den, 0., out=tmp)
                np.subtract(H - 1., sy64, out=py)                               # height-1.-sy
        # anchor_support on the float64 answer; the closure's sentinel (-1, H) is outside, i.e. gated by `valid`.
        inside = valid
        inside &= np.greater_equal(px, 0., out=tmp)
        inside &= np.less(px, W, out=tmp)
        inside &= np.greater_equal(py, 0., out=tmp)
        inside &= np.less(py, H, out=tmp)
        outside = np.logical_not(inside, out=tmp)
        np.copyto(px, 0., where=outside)
        np.copyto(py, 0., where=outside)
        np.floor(px, out=px)
        np.floor(py, out=py)
        np.copyto(ix, px, casting="unsafe")
        np.copyto(iy, py, casting="unsafe")
        np.multiply(iy, W, out=flat)
        flat += ix
        np.logical_and(anchors_flat[flat], inside, out=out)                    # anchors[floor(y), floor(x)]


def kernel_for(kernel: ExportMaskKernel | None, params: ExportParameters) -> ExportMaskKernel:
    """Reuse ``kernel`` when its geometry and SIP spec match, else make a new one."""
    return kernel if kernel is not None and kernel.matches(params) else ExportMaskKernel(params.width, params.height, params.spec)


def registered_mask_kernel(validity, h_stored, *, params: ExportParameters, anchors, kernel: ExportMaskKernel,
                           check=None, grid_check=None) -> np.ndarray:
    """registered_support.registered_mask(validity, h_stored, model, mapping=make_export_mapping(...),
    anchors=anchors, check=check), bit for bit, for the mapping whose parameters are ``params``.
    check() runs once per kernel block (see the P1a validation report for the cadence); ``grid_check`` (P1c F-2's
    parallel path) also before the wait for the shared SIP grid."""
    # The reference validations, same order and messages (registered_support.registered_mask).
    if type(validity) is not np.ndarray or validity.ndim != 2 or validity.dtype != np.bool_ or not all(validity.shape):
        raise ValueError("validity must be a nonempty bool image")
    h = np.asarray(h_stored, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("registration matrix must be finite and invertible")
    height, width = validity.shape
    if type(anchors) is not np.ndarray or anchors.dtype != np.bool_ or anchors.shape != validity.shape:
        raise ValueError("anchor support must be bool and match source geometry")
    if (params.width, params.height) != (width, height) or not kernel.matches(params):
        raise ValueError("support kernel geometry mismatch")   # unreachable from prepare_exports
    anchors_flat = (anchors if anchors.flags.c_contiguous else np.ascontiguousarray(anchors)).reshape(-1)
    output = np.zeros_like(validity)        # same memory layout as the reference (np.save bytes depend on it)
    kernel.set_frame(params.inverse)
    for y0 in range(0, height, kernel.rows):
        if check:
            check()
        y1 = min(y0 + kernel.rows, height)
        kernel.block(y0, y1, anchors_flat, output[y0:y1], grid_check)
    # The unchanged output cross (registered_support.registered_mask).
    centre = output.copy()
    output[0, :] = output[-1, :] = False
    output[:, 0] = output[:, -1] = False
    output[1:-1, 1:-1] &= (centre[:-2, 1:-1] & centre[2:, 1:-1]
                           & centre[1:-1, :-2] & centre[1:-1, 2:])
    return output


def tap_state(valid: np.ndarray, *, rows: int = 256) -> np.ndarray:
    """uint8 (H+1, W+1). Entry [j+1, i+1] for floor(y) = j in [-1, H-1], floor(x) = i in [-1, W-1] is
    TAPS_ALL (3) when all 16 clamped cubic taps valid[clip(j-1..j+2), clip(i-1..i+2)] are True,
    TAPS_NONE (0) when none is, TAPS_MIXED (1) otherwise. Built in row tiles (peak: output + a few MB)."""
    H, W = valid.shape
    code = np.empty((H + 1, W + 1), dtype=np.uint8)
    for a in range(0, H + 1, rows):
        b = min(a + rows, H + 1)
        source = valid[np.clip(np.arange(a - 2, b + 2), 0, H - 1)]       # rows clip(j'-2 .. j'+1)
        pad = np.pad(source, ((0, 0), (2, 2)), mode="edge")             # cols clip(c-2)
        rows_all = pad[:, 0:W + 1] & pad[:, 1:W + 2]
        rows_all &= pad[:, 2:W + 3]
        rows_all &= pad[:, 3:W + 4]
        rows_any = pad[:, 0:W + 1] | pad[:, 1:W + 2]
        rows_any |= pad[:, 2:W + 3]
        rows_any |= pad[:, 3:W + 4]
        n = b - a
        all4 = rows_all[0:n] & rows_all[1:n + 1]
        all4 &= rows_all[2:n + 2]
        all4 &= rows_all[3:n + 3]
        any4 = rows_any[0:n] | rows_any[1:n + 1]
        any4 |= rows_any[2:n + 2]
        any4 |= rows_any[3:n + 3]
        tile = code[a:b]
        np.copyto(tile, all4, casting="unsafe")                           # 0/1
        tile <<= 1                                                        # 0/2
        tile |= any4                                                      # 0 none, 1 mixed, 3 all
    return code


def sample_support_probed(stack_valid, points, check=None):
    """sample_support(stack_valid, points, kernel="cubic", boundary="clamp"); with ``check`` (P1c G-4t's parallel
    path) and more than PROBE_POINTS points, in chunks of PROBE_POINTS with check() before each (same result)."""
    if check is None or len(points) <= PROBE_POINTS:
        return sample_support(stack_valid, points, kernel="cubic", boundary="clamp")
    out = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), PROBE_POINTS):
        check()
        out[start:start + PROBE_POINTS] = sample_support(stack_valid, points[start:start + PROBE_POINTS],
                                                         kernel="cubic", boundary="clamp")
    return out


def native_block(stack_valid, mapped, code_flat, height, width, n, check=None):
    """Per-point sample_support(stack_valid, mapped, cubic, clamp) for one block of n points, deciding
    all-valid / all-invalid footprints from the tap map; None when mapped is not a float64 (n, 2) ndarray.
    ``check`` (P1c G-4t, the frame pool's parallel path only): the mixed footprints in probed chunks
    (sample_support_probed)."""
    if type(mapped) is not np.ndarray or mapped.dtype != np.float64 or mapped.shape != (n, 2):
        return None
    x, y = mapped[:, 0], mapped[:, 1]
    inside = (x >= -0.5) & (x < (width - 0.5)) & (y >= -0.5) & (y < (height - 0.5))   # sample_support's test
    accepted = np.zeros(n, dtype=bool)
    pos = np.flatnonzero(inside)
    if not len(pos):
        return accepted
    xi = np.floor(x[pos]).astype(np.intp)             # [-1, W-1]
    yi = np.floor(y[pos]).astype(np.intp)             # [-1, H-1]
    yi += 1
    yi *= (width + 1)
    yi += xi
    yi += 1                                           # flat index of [floor(y)+1, floor(x)+1]
    code = code_flat[yi]
    accepted[pos[code == TAPS_ALL]] = True
    slow = pos[code == TAPS_MIXED]
    if len(slow):
        accepted[slow] = sample_support_probed(stack_valid, mapped[slow], check)
    return accepted
