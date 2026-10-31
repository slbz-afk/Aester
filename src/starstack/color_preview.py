"""Fixed display transform for the experimental linear-sRGB derivative.

All clipping here belongs to the display preview. The caller retains its
signed float master. No frame percentiles, automatic WB or brightness fitting.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageCms


def preview_recipe(exposure_ev=2.0) -> dict:
    if (isinstance(exposure_ev, bool) or not isinstance(exposure_ev, (float, int))
            or not math.isfinite(exposure_ev) or not -12 <= exposure_ev <= 12):
        raise ValueError('预览曝光需在 -12 到 +12 EV 之间')
    return {'schema_version': 1, 'input_space': 'linear_sRGB_D65_approximation',
            'exposure_ev': float(exposure_ev), 'asinh_strength': 8.0,
            'negative_handling': 'clip_only_in_preview_after_area_resize',
            'tone_curve': 'luminance_asinh_preserve_rgb_ratios_then_clip',
            'encoding': 'sRGB', 'automatic_brightness': False,
            'automatic_white_balance': False, 'row_order': 'TOP-DOWN'}


def render_preview(rgb: np.ndarray, recipe: dict, *, max_side=1600) -> np.ndarray:
    if recipe != preview_recipe(recipe.get('exposure_ev')):
        raise ValueError('未知的显示配方')
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.float32 or min(rgb.shape[:2]) < 1):
        raise ValueError('预览需要 TOP-DOWN float32 线性 sRGB')
    if type(max_side) is not int or not 1 <= max_side <= 4096:
        raise ValueError('预览长边超出支持范围')
    h, w = rgb.shape[:2]
    factor = min(1., max_side / max(h, w))
    size = (max(1, round(w*factor)), max(1, round(h*factor)))
    small = np.empty((size[1], size[0], 3), np.float64)
    for c in range(3):
        # Average the signed linear samples before display clipping/encoding.
        if not np.isfinite(rgb[..., c]).all():
            raise ValueError('工作图存在 NaN 或无穷数值')
        small[..., c] = np.asarray(Image.fromarray(rgb[..., c]).resize(size, Image.Resampling.BOX))
    x = np.maximum(small, 0) * 2**recipe['exposure_ev']
    luminance = x @ np.array([.2126, .7152, .0722])
    strength = recipe['asinh_strength']
    ratio = np.full_like(luminance, strength / np.arcsinh(strength))
    np.divide(np.arcsinh(strength*luminance), np.arcsinh(strength)*luminance,
              out=ratio, where=luminance > 0)
    x = np.clip(x * ratio[..., None], 0, 1)
    encoded = np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)
    return np.round(encoded*255).astype(np.uint8)


def save_preview(path, rgb: np.ndarray, recipe: dict, *, max_side=1600) -> dict:
    pixels = render_preview(rgb, recipe, max_side=max_side)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(pixels).save(path, quality=94, icc_profile=icc)
    return {'dimensions': [pixels.shape[1], pixels.shape[0]], 'icc_profile': 'sRGB',
            'display_only': True, 'recipe': recipe}
