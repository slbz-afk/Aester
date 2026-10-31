"""Whole-image colour candidates, separate from the unmodified stack master.

Camera cache conversion is explicitly an approximation: old Siril stacks have
already undergone data-dependent interpolation, channel normalization and
rejection. A fixed affine map cannot undo those decisions. No spatial weights
or separately estimated sky/ground colour are accepted by this module.
"""
import math

import numpy as np
from PIL import Image

from .background import image_array, number
from .raw_color import validate_recipe

VERSION = 'whole-color-linked-tone-1'
CACHE_STATE = 'siril_camera_RGB_DN_div_65535_with_common_black_approximation'


def camera_from_cache(data, raw_recipe, *, black_dn, input_state):
    """Global common-black, fixed WB and matrix; no clipping or display curve.

Only use after auditing the original CFA and old conversion/stack path. The
caller must bind that evidence and the actual reference frame to the output.
This is not a general camera-space detector or a new RAW decoder.
"""
    image_array(data)
    if input_state != CACHE_STATE:
        raise ValueError('Explicit audited Siril camera cache state is required')
    recipe = validate_recipe(raw_recipe)
    black = number(black_dn, 0, 65534, 'black_dn')
    if recipe['reference']['common_black_for_scale_DN'] != black:
        raise ValueError('Common black differs from the reference recipe')
    gains = np.asarray(recipe['white_balance']['gains_rgb'], dtype=np.float64)
    matrix = np.asarray(recipe['matrix']['values'], dtype=np.float64)
    scale = recipe['scale']['common_denominator_DN']
    out = np.empty_like(data)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for y in range(0, len(data), 128):
                tile = (data[y:y+128].astype(np.float64)*65535.-black)/scale
                out[y:y+128] = ((tile*gains) @ matrix.T).astype(np.float32)
    except FloatingPointError as exc:
        raise ValueError('Camera cache conversion exceeds float32') from exc
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite camera cache conversion')
    return out


def resize_signed(data, max_side=1600):
    """Reduce signed linear values before any lossy display operation."""
    image_array(data)
    if type(max_side) is not int or not 2 <= max_side <= 4096:
        raise ValueError('Invalid preview size')
    h, w = data.shape[:2]
    factor = min(1., max_side/max(h, w))
    size = max(2, round(w*factor)), max(2, round(h*factor))
    return np.stack([np.asarray(Image.fromarray(data[..., c]).resize(size, Image.Resampling.BOX))
                     for c in range(3)], axis=2).astype(np.float32)


def srgb_decode(value):
    a = np.asarray(value, dtype=np.float64)
    return np.where(a <= .04045, a/12.92, ((a+.055)/1.055)**2.4)


def midtone_for_anchor(anchor, target):
    """Solve the rational curve for an interior, display-linear anchor."""
    x = number(anchor, 1e-9, 1-1e-9, 'anchor')
    t = number(target, 1e-9, 1-1e-9, 'target')
    m = x*(t-1)/(2*x*t-x-t)
    return number(m, 1e-4, 1-1e-4, 'derived midtone')


def make_display_recipe(*, black, white, anchor, target_srgb,
                        gains_rgb=(1., 1., 1.), asinh_strength=12.):
    """Freeze an explicit brightness anchor; no auto fit or scene recognition.

All values refer to input after the fixed optional display gains. The target
is the encoded sRGB value of the largest channel at this scalar anchor, not a
guarantee about the median of an entire background area.
"""
    black = number(black, -1e20, 1e20, 'black')
    white = number(white, -1e20, 1e20, 'white')
    if white-black < 1e-20:
        raise ValueError('Display white must exceed black')
    anchor = number(anchor, black, white, 'anchor')
    k = number(asinh_strength, 0, 100, 'asinh_strength')
    target = number(target_srgb, .01, .95, 'target_srgb')
    if type(gains_rgb) not in (tuple, list) or len(gains_rgb) != 3:
        raise ValueError('Three fixed display gains required')
    gains = [number(v, .25, 4, 'display gain') for v in gains_rgb]
    x = (anchor-black)/(white-black)
    a = math.asinh(k*x)/math.asinh(k) if k else x
    m = midtone_for_anchor(a, float(srgb_decode(target)))
    return {'version': VERSION, 'method': 'linked_maxRGB_asinh_MTF_then_sRGB',
            'black': black, 'white': white, 'asinh_strength': k, 'midtone': m,
            'gains_rgb': gains, 'anchor': anchor, 'target_srgb': target,
            'spatial_scope': 'whole_image', 'display_only': True}


def render_display(data, recipe, *, max_side=1600):
    from .linked_tone import linked_tone
    if (not isinstance(recipe, dict) or recipe.get('version') != VERSION
            or recipe.get('method') != 'linked_maxRGB_asinh_MTF_then_sRGB'
            or recipe.get('spatial_scope') != 'whole_image'
            or recipe.get('display_only') is not True):
        raise ValueError('Unknown whole-image display recipe')
    # Reconstruct only to validate; use stored coefficients for replay.
    expected = make_display_recipe(black=recipe['black'], white=recipe['white'],
                                  anchor=recipe['anchor'], target_srgb=recipe['target_srgb'],
                                  gains_rgb=recipe['gains_rgb'], asinh_strength=recipe['asinh_strength'])
    m = number(recipe['midtone'], 1e-4, 1-1e-4, 'midtone')
    if not math.isclose(m, expected['midtone'], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError('Stored tone curve is inconsistent with its anchor')
    small = resize_signed(data, max_side)
    try:
        with np.errstate(over='raise', invalid='raise'):
            balanced = (small.astype(np.float64)*np.array(expected['gains_rgb'])).astype(np.float32)
    except FloatingPointError as exc:
        raise ValueError('Display gains exceed float32') from exc
    linear = linked_tone(balanced, black=recipe['black'], white=recipe['white'],
                         asinh_strength=recipe['asinh_strength'], midtone=m).astype(np.float64)
    encoded = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return np.round(np.clip(encoded, 0, 1)*255).astype(np.uint8)
