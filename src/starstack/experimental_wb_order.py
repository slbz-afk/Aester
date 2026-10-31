"""Opt-in WB-order diagnostics; no changes to the default RAW pipeline.

All arrays are TOP-DOWN float32, already black-subtracted. Gains are supplied
explicitly and are never normalized or estimated here. Arithmetic uses float64
row blocks and rounds to newly allocated float32 storage, preserving signed and
over-one values without clipping. Array values cannot prove processing history.

Both experiment orders return white-balanced camera RGB DN. Color conversion
then uses only the stored matrix/common scale, never a second WB. Removing WB
is a diagnostic coordinate change; it cannot undo a nonlinear demosaic's WB
dependence. The fixed-pedestal RCD backend retains its own experimental limits.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import raw_color
from .experimental_rcd import rcd_fixed_range
from .float_demosaic import rgb_pattern


ALGORITHM = 'experimental-fixed-wb-order-v1'
_BLOCK_ROWS = 128


def _numeric_leaves(value, *, integers=False):
    if isinstance(value, np.ndarray):
        return value.dtype.kind in ('iu' if integers else 'iuf')
    if isinstance(value, (tuple, list)):
        return all(_numeric_leaves(v, integers=integers) for v in value)
    types = (int, np.integer) if integers else (int, float, np.integer, np.floating)
    return isinstance(value, types) and not isinstance(value, (bool, np.bool_))


def _gains(gains_rgb):
    if not _numeric_leaves(gains_rgb):
        raise ValueError('WB gains must be real numbers, not booleans or strings')
    try:
        with np.errstate(over='raise', invalid='raise'):
            gains = np.asarray(gains_rgb, dtype=np.float64)
        if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError('WB gains must be three finite positive RGB coefficients')
    except (TypeError, OverflowError, FloatingPointError) as exc:
        raise ValueError('WB gains cannot be represented as float64') from exc
    return gains


def _pattern(pattern):
    # Validate before NumPy can hide a mixed bool/int list by coercion.
    if not _numeric_leaves(pattern, integers=True):
        raise ValueError('CFA pattern must contain integer indices, not booleans')
    return rgb_pattern(pattern)


def _input(array, *, cfa=False):
    if (not isinstance(array, np.ndarray) or array.dtype != np.float32
            or array.ndim != (2 if cfa else 3)
            or min(array.shape[:2]) < (2 if cfa else 1)
            or array.shape[0]*array.shape[1] > 120_000_000
            or (not cfa and array.shape[2] != 3)):
        raise ValueError('Expected TOP-DOWN float32 2D CFA or HWC RGB with valid dimensions')
    for y in range(0, array.shape[0], _BLOCK_ROWS):
        if not np.isfinite(array[y:y+_BLOCK_ROWS]).all():
            raise ValueError('Input contains nonfinite values')
    return array


def _float32(value):
    # The surrounding operation also uses errstate; underflow follows ordinary
    # float32 rounding, whereas overflow/NaN must never become a saved result.
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Result is not finite in float32')
    return result


def apply_cfa_wb(cfa, pattern, gains_rgb):
    """Multiply each CFA site by its fixed RGB gain; green slot 3 uses G1."""
    source, p, gains = _input(cfa, cfa=True), _pattern(pattern), _gains(gains_rgb)
    result = np.empty(source.shape, np.float32)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            for top in range(0, source.shape[0], _BLOCK_ROWS):
                bottom = min(source.shape[0], top+_BLOCK_ROWS)
                for y in range(2):
                    # Use absolute CFA parity, not a tile-relative phase.
                    first = top + (y-top) % 2
                    for x in range(2):
                        tile = source[first:bottom:2, x::2].astype(np.float64)
                        result[first:bottom:2, x::2] = _float32(tile*gains[p[y, x]])
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('CFA WB overflows float32 or produces invalid arithmetic') from exc
    return result


def _rgb_gain(rgb, gains_rgb, *, inverse):
    source, gains = _input(rgb), _gains(gains_rgb)
    result = np.empty(source.shape, np.float32)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            for y in range(0, source.shape[0], _BLOCK_ROWS):
                tile = source[y:y+_BLOCK_ROWS].astype(np.float64)
                transformed = tile/gains if inverse else tile*gains
                result[y:y+_BLOCK_ROWS] = _float32(transformed)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('RGB WB operation overflows float32 or produces invalid arithmetic') from exc
    return result


def apply_rgb_wb(rgb, gains_rgb):
    """Apply exactly the supplied gains once to unbalanced camera RGB DN."""
    return _rgb_gain(rgb, gains_rgb, inverse=False)


def remove_rgb_wb(balanced_rgb, gains_rgb):
    """Divide gains for camera-space diagnostics; not an inverse demosaicer."""
    return _rgb_gain(balanced_rgb, gains_rgb, inverse=True)


def color_from_balanced(balanced_rgb, recipe):
    """Stored M @ (already-WB camera RGB / common DN scale); no WB or gamma.

    validate_recipe checks the recipe but its unbalanced-input condition belongs
    to raw_color.apply_color. This separate function requires already-balanced
    input and deliberately does not call apply_color or alter the recipe.
    """
    source = _input(balanced_rgb)
    fixed = raw_color.validate_recipe(recipe)
    matrix = np.asarray(fixed['matrix']['values'], dtype=np.float64)
    scale = fixed['scale']['common_denominator_DN']
    result = np.empty(source.shape, np.float32)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            for y in range(0, source.shape[0], _BLOCK_ROWS):
                tile = source[y:y+_BLOCK_ROWS].astype(np.float64)/scale
                result[y:y+_BLOCK_ROWS] = _float32(tile @ matrix.T)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('Balanced color transform overflows float32 or produces invalid arithmetic') from exc
    return result


def _digest(array):
    digest = hashlib.sha256()
    for y in range(0, array.shape[0], _BLOCK_ROWS):
        digest.update(np.ascontiguousarray(array[y:y+_BLOCK_ROWS], dtype='<f4').tobytes())
    return digest.hexdigest()


def run_wb_order(cfa, pattern, output_root, recipe, *, order,
                 lower_dn, upper_dn, engine=None):
    """Run 'before' (CFA WB -> RCD) or 'after' (RCD -> RGB WB).

    Identical explicit bounds must be supplied by the caller to both trials;
    they apply to the actual CFA presented to RCD and are never auto-expanded.
    The returned summary is not persisted here: caller must save it. The nested
    backend record retains its own separate experiment.json unchanged.
    """
    if not isinstance(order, str) or order not in ('before', 'after'):
        raise ValueError("order must be 'before' or 'after'")
    source, p = _input(cfa, cfa=True), _pattern(pattern)
    fixed = raw_color.validate_recipe(recipe)
    gains = _gains(fixed['white_balance']['gains_rgb'])
    original_hash = _digest(source)
    engine_input = apply_cfa_wb(source, pattern, gains) if order == 'before' else source
    rgb, backend = rcd_fixed_range(engine_input, pattern, output_root,
        lower_dn=lower_dn, upper_dn=upper_dn, engine=engine)
    _input(rgb)
    if rgb.shape != (*source.shape, 3):
        raise ValueError('RCD output shape differs from the input CFA')
    if backend.get('state') != 'succeeded':
        raise ValueError('RCD backend did not report success')
    balanced = np.array(rgb, copy=True) if order == 'before' else apply_rgb_wb(rgb, gains)
    if _digest(source) != original_hash:
        raise ValueError('Input CFA changed during the WB-order experiment')
    record = {'schema_version': 1, 'state': 'succeeded', 'algorithm': ALGORITHM,
              'order': order, 'operations': ['CFA_WB', 'fixed_RCD'] if order == 'before'
                  else ['fixed_RCD', 'RGB_WB'],
              'gains_rgb': gains.tolist(), 'pattern_rgb': p.tolist(),
              'white_balance_applications': 1, 'matrix_applied': False,
              'common_scale_applied': False, 'output_units': 'white_balanced_camera_RGB_DN',
              'row_order': 'TOP-DOWN', 'channel_order': 'RGB',
              'input_sha256_float32_le': original_hash, 'input_unchanged': True,
              'balanced_sha256_float32_le': _digest(balanced),
              'recipe_sha256': hashlib.sha256(json.dumps(fixed, sort_keys=True,
                  separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
              'range_policy': 'caller_supplied_common_bounds_no_automatic_estimation',
              'lower_dn': float(lower_dn), 'upper_dn': float(upper_dn),
              'rcd_record': backend, 'record_persistence': 'returned_to_caller_only',
              'limitations': ['Nonlinear RCD need not commute with WB',
                  'Fixed positive pedestal changes RCD ratios; this remains an experimental backend',
                  'Input black subtraction, crop/CFA phase, and physical units are caller obligations',
                  'Removing output WB does not reverse WB effects inside nonlinear interpolation',
                  'Use color_from_balanced for subsequent matrix/common scale, without applying WB again']}
    return balanced, record
