"""Colour-difference helpers for acceptance scripts: sRGB → CIELAB (D65) and CIEDE2000.

Straight implementations of the published formulae (Sharma, Wu & Dalal 2005 for ΔE00); no tuning.
"""
from __future__ import annotations

import numpy as np

from . import icc_profiles as icc

_D65 = icc.xy_to_xyz(icc.D65_XY)


def lab_from_linear_srgb(rgb: np.ndarray) -> np.ndarray:
    xyz = np.asarray(rgb, dtype=np.float64) @ icc.SRGB.rgb_to_xyz.T
    r = xyz / _D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(r > eps, np.cbrt(np.maximum(r, 0)), (kappa * r + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_from_encoded_srgb8(rgb8: np.ndarray) -> np.ndarray:
    return lab_from_linear_srgb(icc.srgb_decode(np.asarray(rgb8, dtype=np.float64) / 255.0))


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = (np.asarray(lab1, dtype=np.float64)[..., i] for i in range(3))
    L2, a2, b2 = (np.asarray(lab2, dtype=np.float64)[..., i] for i in range(3))
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0.0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lm = (L1 + L2) / 2
    Cmp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hm = np.where(C1p * C2p == 0, hsum,
                  np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                           np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    dtheta = 30 * np.exp(-((hm - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cmp ** 7 / (Cmp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2)
    Sc = 1 + 0.045 * Cmp
    Sh = 1 + 0.015 * Cmp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def block_means_linear(rgb8: np.ndarray, block: int = 8) -> np.ndarray:
    """Linear-light means over ``block``×``block`` tiles of an encoded 8-bit sRGB image (edges dropped): the JPEG
    comparison that measures rendering identity rather than the codec's quantisation of pixel noise."""
    lin = icc.srgb_decode(np.asarray(rgb8, dtype=np.float64) / 255.0)
    h, w = lin.shape[:2]
    hb, wb = h // block * block, w // block * block
    if hb == 0 or wb == 0:
        raise ValueError("image smaller than one block")
    return lin[:hb, :wb].reshape(hb // block, block, wb // block, block, 3).mean(axis=(1, 3))
