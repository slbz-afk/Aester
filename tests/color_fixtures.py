"""Synthetic inputs for the colour-contract tests (docs/COLOR_CONTRACT_PACKAGE_20260922.md).

A Z8-like reference DNG (public tag values from docs/diagnostics/z8-dng-color-metadata-20260916.json) and a physically
consistent camera master: DN = black + ratio_c × signal, sensor clipping at WhiteLevel, a no-data strip, a ground band.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

Z8_BLACK = 1008
Z8_WHITE = 15892
Z8_AS_SHOT_NEUTRAL = (0.589183, 1.0, 0.644836)
Z8_COLOR_MATRIX1 = (1.3998, -.8092, .0284, -.4723, 1.2772, .2159, -.0164, .0624, .8674)
Z8_COLOR_MATRIX2 = (1.1423, -.4564, -.1123, -.4816, 1.2895, .2119, -.021, .1061, .7282)
# Approximate unbalanced camera response of a neutral scene element (R, G, B relative to G) = the as-shot neutral
CAM_NEUTRAL = np.array(Z8_AS_SHOT_NEUTRAL)


def _rational(values, den):
    out = []
    for v in values:
        out += [int(round(v * den)), den]
    return tuple(out)


def write_reference_dng(path, *, analog_balance=(1, 1, 1), calibration_identity=True, illuminant2=21,
                        include_matrix2=True) -> Path:
    path = Path(path)
    mosaic = np.random.default_rng(0).integers(Z8_BLACK, 3000, (8, 8)).astype(np.uint16)
    cal = (1, 0, 0, 0, 1, 0, 0, 0, 1) if calibration_identity else (1.1, 0, 0, 0, 1, 0, 0, 0, 0.9)
    tags = [
        (50706, 1, 4, (1, 4, 0, 0), True), (50708, 2, None, "Nikon Z 8", True),
        (271, 2, None, "NIKON CORPORATION", True), (272, 2, None, "NIKON Z 8", True),
        (33421, 3, 2, (2, 2), True), (33422, 1, 4, (0, 1, 1, 2), True),
        (50713, 3, 2, (2, 2), True), (50714, 5, 4, _rational([Z8_BLACK] * 4, 256), True),
        (50717, 3, 1, Z8_WHITE, True), (50728, 5, 3, _rational(Z8_AS_SHOT_NEUTRAL, 1000000), True),
        (50721, 10, 9, _rational(Z8_COLOR_MATRIX1, 10000), True),
        (50778, 3, 1, 17, True), (50779, 3, 1, illuminant2, True),
        (50727, 5, 3, _rational(analog_balance, 1000000), True),
        (50723, 10, 9, _rational(cal, 10000), True), (50724, 10, 9, _rational(cal, 10000), True),
        (50730, 10, 1, _rational([0.2], 100), True),
    ]
    if include_matrix2:
        tags.append((50722, 10, 9, _rational(Z8_COLOR_MATRIX2, 10000), True))
    tifffile.imwrite(path, mosaic, photometric=32803, extratags=tags)
    return path


def synthetic_camera_master(h=240, w=320, *, seed=3, stars=((50, 50, 3000), (120, 80, 15000), (200, 60, 40000), (260, 120, 9000)),
                            red_light=(30, 150, 6000), nodata_columns=8, ground_from=180):
    """TOP-DOWN float32 camera RGB in DN/65535 units, plus (sky_mask bool, saturated_expected bool, nodata bool).

    Sky: neutral-ish gradient with noise; stars: Gaussian, some exceeding WhiteLevel (sensor clipping applied per channel);
    ``red_light``: a bright source that clips only in R; ground: darker band with a deep-shadow patch whose noise crosses the
    black level (negative signal must survive in M); no-data strip: exact zeros.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    signal = 400.0 + 200.0 * (yy / h)                       # DN above black, green channel
    cam = np.empty((h, w, 3), np.float64)
    for c in range(3):
        cam[..., c] = Z8_BLACK + CAM_NEUTRAL[c] * signal + rng.normal(0, 12, (h, w))
    for sx, sy, amp in stars:
        g = amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.5 ** 2))
        for c in range(3):
            cam[..., c] += CAM_NEUTRAL[c] * g
    if red_light is not None:
        sx, sy, amp = red_light
        g = amp * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 2.0 ** 2))
        cam[..., 0] += 2.5 * g          # strongly red: clips in R only
        cam[..., 1] += 0.15 * g
        cam[..., 2] += 0.1 * g
    cam = np.minimum(cam, Z8_WHITE)                           # sensor clipping
    ground = yy >= ground_from
    for c in range(3):
        cam[..., c][ground] = Z8_BLACK + (30, 60, 40)[c] + rng.normal(0, 4, int(ground.sum()))
    deep = ground & (xx >= w - 40)                            # deep shadow: signal below the read noise → some DN < black
    for c in range(3):
        cam[..., c][deep] = Z8_BLACK + 2.0 + rng.normal(0, 8, int(deep.sum()))
    cam[:, :nodata_columns] = 0.0
    cam = np.clip(cam, 0, 65535)
    saturated = (cam >= 0.97 * Z8_WHITE).any(axis=2)
    nodata = (cam == 0).all(axis=2)
    return (cam / 65535.0).astype(np.float32), ~ground, saturated, nodata


def libraw_like_metadata() -> dict:
    """The LibRaw-shaped metadata dict ``raw_color.build_recipe`` accepts, built from the public Z8 values."""
    n = Z8_AS_SHOT_NEUTRAL
    return {"available": True, "white_balance_applied": False, "color_correction_applied": False,
            "versions": {"source": "fixture"},
            "fields": {"num_colors": {"value": 3}, "color_desc": {"value": "RGBG"},
                       "raw_pattern": {"value": [[0, 1], [1, 2]]},
                       "rgb_xyz_matrix": {"value": [list(Z8_COLOR_MATRIX2[0:3]), list(Z8_COLOR_MATRIX2[3:6]),
                                                    list(Z8_COLOR_MATRIX2[6:9]), [0, 0, 0]]},
                       "camera_whitebalance": {"value": [1 / n[0], 1 / n[1], 1 / n[2], 1 / n[1]]}}}
