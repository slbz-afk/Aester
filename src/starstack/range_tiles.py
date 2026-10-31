"""Exact float32 power-of-two arithmetic for the opt-in positive adapter.

Common RGB zeros are existing no-data sentinels. No additive offset or clipping
is permitted; the inverse operation must preserve every input bit.
"""
import math

import numpy as np


def choose_divisor(maximum):
    if isinstance(maximum, (bool, np.bool_)) or not (type(maximum) in (int, float) or isinstance(maximum, (np.integer, np.floating))):
        raise ValueError('maximum must be a finite scalar in [0, 8]')
    try:
        value = float(maximum)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError('maximum is out of range') from exc
    if not math.isfinite(value) or not 0 <= value <= 8:
        raise ValueError('maximum must be in [0, 8]')
    divisor = 1
    while value / divisor > .75:
        divisor *= 2
    return divisor


def rescale_positive_tile(data, divisor, *, restore=False):
    if (type(data) is not np.ndarray or data.dtype != np.dtype('float32')
            or data.ndim != 3 or data.shape[0] != 3 or min(data.shape) <= 0):
        raise ValueError('expected native float32 RGB tile')
    if type(divisor) is not int or not 1 <= divisor <= 65536 or divisor & (divisor - 1):
        raise ValueError('divisor must be an integer power of two in [1, 65536]')
    if type(restore) is not bool:
        raise ValueError('restore must be bool')
    positive = (data > 0).all(axis=0)
    zero = (data == 0).all(axis=0)
    if not np.isfinite(data).all() or not (positive | zero).all():
        raise ValueError('expected common RGB zeros or positive finite samples')
    if restore and np.any(data[:, positive] >= 1):
        raise ValueError('encoded positive samples must be below one')
    exponent = divisor.bit_length() - 1
    if not restore:
        exponent = -exponent
    with np.errstate(all='ignore'):
        result = np.ldexp(data, exponent).astype(np.float32, order='C', copy=True)
        inverse = np.ldexp(result, -exponent)
    if (not np.isfinite(result).all() or np.any(result[:, positive] <= 0)
            or (not restore and np.any(result[:, positive] >= 1))
            or not np.array_equal(inverse.view(np.uint32), data.view(np.uint32))):
        raise ValueError('scaling cannot preserve every float32 input bit')
    return result


def rescale_finite_tile(data, divisor, *, restore=False):
    """Contract 1.1.0: any finite float32 sample is carried exactly.

    Negative values, signed zeros and single-channel zeros are samples; only
    common RGB zero means no data, which scaling keeps as zero anyway. The
    engine clips its output to [0, 1], so every encoded sample must stay below
    one (the low end is Siril's, counted by the caller). The inverse must
    restore every input bit, so a subnormal that underflows is refused.
    """
    if (type(data) is not np.ndarray or data.dtype != np.dtype('float32')
            or data.ndim != 3 or data.shape[0] != 3 or min(data.shape) <= 0):
        raise ValueError('expected native float32 RGB tile')
    if type(divisor) is not int or not 1 <= divisor <= 65536 or divisor & (divisor - 1):
        raise ValueError('divisor must be an integer power of two in [1, 65536]')
    if type(restore) is not bool:
        raise ValueError('restore must be bool')
    if not np.isfinite(data).all():
        raise ValueError('expected finite samples')
    if restore and np.any(data >= 1):
        raise ValueError('encoded samples must be below one')
    exponent = divisor.bit_length() - 1
    if not restore:
        exponent = -exponent
    with np.errstate(all='ignore'):
        result = np.ldexp(data, exponent).astype(np.float32, order='C', copy=True)
        inverse = np.ldexp(result, -exponent)
    if (not np.isfinite(result).all()
            or (not restore and np.any(result >= 1))
            or not np.array_equal(inverse.view(np.uint32), data.view(np.uint32))):
        raise ValueError('scaling cannot preserve every float32 input bit')
    return result
