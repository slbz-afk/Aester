"""Sample-preserving float Bayer interpolation for numerical verification.

This intentionally simple bilinear baseline is not the production quality
algorithm. It has local support, no frame-dependent normalization, no WB and
no clipping. Input/output coordinates are full-sensor TOP-DOWN. At the outer
edge, only existing neighbours contribute; measured samples are copied back.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


ALGORITHM = 'float-bilinear-local-v1'


def _backing_file(array):
    while array is not None:
        filename = getattr(array, 'filename', None)
        if filename is not None:
            return Path(filename).resolve()
        array = getattr(array, 'base', None)
    return None


def rgb_pattern(pattern) -> np.ndarray:
    """Accept ordinary RGB/RGBG Bayer indices; green slot 3 maps to RGB 1."""
    p = np.asarray(pattern)
    if p.shape != (2, 2) or p.dtype.kind not in 'iu' or np.any(p < 0) or np.any(p > 3):
        raise ValueError('需要已经确认的 2×2 Bayer 阵列')
    p = np.where(p == 3, 1, p).astype(np.uint8)
    if (sorted(p.ravel().tolist()) != [0, 1, 1, 2]
            or not (p[0, 0] == p[1, 1] == 1 or p[0, 1] == p[1, 0] == 1)):
        raise ValueError('R/B 与两个绿色采样必须位于 Bayer 的对角位置')
    return p


def demosaic_bilinear(cfa: np.ndarray, pattern, *, out=None, block_rows=128) -> np.ndarray:
    """Interpolate missing colors with separable weights [1,2,1].

    Float64 accumulators avoid overflow/rounding during sums; storage remains
    float32. Green measured positions override the interpolated estimate.
    Bounded row tiles include the one-pixel halo from the original array.
    """
    p = rgb_pattern(pattern)
    if (not isinstance(cfa, np.ndarray) or cfa.ndim != 2 or cfa.dtype != np.float32
            or min(cfa.shape) < 2 or cfa.size > 120_000_000):
        raise ValueError('浮点去马赛克需要已扣黑的二维 float32 CFA（边长至少 2）')
    if type(block_rows) is not int or not 1 <= block_rows <= 4096:
        raise ValueError('无效的分块行数')
    h, w = cfa.shape
    if out is None:
        out = np.empty((h, w, 3), np.float32)
    if (not isinstance(out, np.ndarray) or out.shape != (h, w, 3)
            or out.dtype != np.float32 or not out.flags.writeable
            or np.shares_memory(out, cfa)):
        raise ValueError('输出需为独立、可写的 float32 RGB 数组')
    source_file, output_file = _backing_file(cfa), _backing_file(out)
    if source_file is not None and output_file is not None:
        if source_file == output_file or source_file.samefile(output_file):
            raise ValueError('输出不能覆盖输入的映射文件')
    columns = np.arange(w)[None, :]
    for top in range(0, h, block_rows):
        bottom = min(h, top + block_rows)
        rows = np.arange(top, bottom)[:, None]
        measured = p[rows % 2, columns % 2]
        # Reject an invalid input before it can silently become a colored dot.
        if not np.isfinite(cfa[max(0, top-1):min(h, bottom+1)]).all():
            raise ValueError('CFA 含 NaN 或无穷数值')
        for channel in range(3):
            numerator = np.zeros((bottom-top, w), np.float64)
            denominator = np.zeros_like(numerator)
            for dy, wy in ((-1, 1), (0, 2), (1, 1)):
                ys, ye = max(top, -dy), min(bottom, h-dy)
                if ye <= ys:
                    continue
                for dx, wx in ((-1, 1), (0, 2), (1, 1)):
                    xs, xe = max(0, -dx), min(w, w-dx)
                    mask = p[np.arange(ys+dy, ye+dy)[:, None] % 2,
                             np.arange(xs+dx, xe+dx)[None, :] % 2] == channel
                    weights = mask * (wy*wx)
                    target = np.s_[ys-top:ye-top, xs:xe]
                    numerator[target] += cfa[ys+dy:ye+dy, xs+dx:xe+dx] * weights
                    denominator[target] += weights
            if np.any(denominator == 0):
                raise ValueError('Bayer 边缘缺少所需颜色的相邻采样')
            estimate = numerator / denominator
            estimate[measured == channel] = cfa[top:bottom][measured == channel]
            out[top:bottom, :, channel] = estimate
    return out
