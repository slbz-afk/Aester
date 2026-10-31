"""Shared photographic intent for preview and delivery; never modifies the M master.

Warmth and tint are relative stops of camera-channel gain, not calibrated Kelvin.
Exposure multiplies the display input. The baseline tone anchor stays fixed so
automatic metering cannot cancel a user's exposure change.
"""
from dataclasses import replace
import math
import numpy as np

VERSION = "color-adjustments-1"
DEFAULTS = {"warmth": 0.0, "tint": 0.0, "exposure_ev": 0.0}


def normalize(spec=None):
    if spec is not None and not isinstance(spec, dict):
        raise ValueError("颜色调整必须为对象")
    if set(spec or {}) - DEFAULTS.keys():
        raise ValueError("未知颜色调整参数")
    out = dict(DEFAULTS)
    for key, value in (spec or {}).items():
        bound = 4 if key == "exposure_ev" else 100
        if type(value) not in (int, float) or not math.isfinite(value) or not -bound <= value <= bound:
            raise ValueError(f"{key} 必须为 [-{bound}, {bound}] 内有限数值")
        out[key] = float(value)
    return out


def apply(params, master_meta, spec=None):
    """Return display Params; zero intent returns the identical baseline object."""
    values = normalize(spec)
    if not any(values.values()):
        return params
    basis = master_meta["colour_basis"]
    old = np.asarray(basis["gains_rgb"], dtype=np.float64)
    gains = np.asarray(params.wb_gains_rgb or basis["gains_rgb"], dtype=np.float64)
    # Warm: red up/blue down. Positive tint: magenta relative to green.
    warmth, tint = values["warmth"] / 100., values["tint"] / 200.
    gains = gains * np.exp2([warmth + tint, 0., -warmth + tint])
    if basis.get("master_matrix") is not None:
        transform = np.asarray(basis["master_matrix"], dtype=np.float64) / old[None, :]
        matrix = transform @ np.diag(gains / old) @ np.linalg.inv(transform)
    else:
        from .master_export import switch_white_balance_matrix
        matrix = switch_white_balance_matrix(basis, gains, master_meta["gamut"])
    exposure = 2. ** values["exposure_ev"]
    matrix *= exposure
    return replace(params, wb_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
                   wb_gains_rgb=tuple(float(v) for v in gains),
                   noise_sigma=None if params.noise_sigma is None else params.noise_sigma * exposure,
                   shadow_smooth_start=params.shadow_smooth_start * exposure,
                   shadow_smooth_end=params.shadow_smooth_end * exposure).validate()
