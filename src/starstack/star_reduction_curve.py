"""Bounded star-residual attenuation kernel.

The module exposes one function, :func:`reduction_delta`.  It computes a
nonnegative integer attenuation delta for a single tile and does not return a
composite.  Callers apply the same delta to both paired composites::

    edited_source = source - delta
    edited_base = base - delta

Only the positive residual ``P = max(base - diffuse, 0)`` is reduced.  The
delta is bounded so that no positive channel is pushed below the existing
nonzero code of either ``base`` or ``source``.  Negative signed residual
components are left unchanged.

Inputs must be exact ``numpy.ndarray`` or ``numpy.memmap`` instances with the
documented dtypes.  Lists and other ndarray subclasses are rejected with
``ValueError`` rather than coerced.
"""

import math

import numpy as np

__all__ = ["reduction_delta"]

_MAX_U16 = 65535.0
_MAX_PIXELS = 65536


def _require_plain_array(name, value):
    """Reject lists and ndarray subclasses; accept only ndarray or memmap."""
    if type(value) is np.ndarray or type(value) is np.memmap:
        return
    raise ValueError(
        f"{name} must be an ndarray or memmap, not {type(value).__name__}"
    )


def _require_u16(name, value, shape):
    """Validate a uint16 array with exactly *shape*."""
    _require_plain_array(name, value)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype.kind != "u" or value.dtype.itemsize != 2:
        raise ValueError(
            f"{name} must have an unsigned 16-bit dtype, got {value.dtype}"
        )


def _require_weight(name, value, shape):
    """Validate a float32/float64 weight array with exactly *shape*."""
    _require_plain_array(name, value)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8):
        raise ValueError(
            f"{name} must have a 32- or 64-bit floating dtype, got {value.dtype}"
        )


def _unit_parameter(name, value):
    """Validate a builtin int/float in [0, 1] and return it as float."""
    if type(value) not in (int, float):
        raise ValueError(
            f"{name} must be a builtin int or float, not {type(value).__name__}"
        )
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or value > 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    try:
        return float(value)
    except OverflowError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} is too large to convert to float") from exc


def reduction_delta(
    base,
    source,
    diffuse,
    brightness,
    weight,
    *,
    strength=0.55,
    bright_start=0.10,
    bright_end=0.45,
):
    """Return the bounded nonnegative attenuation delta for one tile.

    Parameters
    ----------
    base, source, diffuse : ndarray or memmap, shape (H, W, 3), uint16
        Paired composites and the diffuse/background estimate.  ``source`` may
        differ from ``base`` by a signed transient residual.
    brightness : ndarray or memmap, shape (H, W), uint16
        Supplied neighborhood maximum of the positive residual.  Every value
        must be at least ``max(maximum(base - diffuse, 0), axis=2)``.
    weight : ndarray or memmap, shape (H, W), float32 or float64
        Precomputed sky/Milky-Way permission/relief, finite in [0, 1].
    strength : builtin int or float, default 0.55
        Overall attenuation strength in [0, 1].
    bright_start, bright_end : builtin int or float, default 0.10, 0.45
        Positive-residual brightness ramp in normalised [0, 1] units with
        ``0 <= bright_start < bright_end <= 1``.

    Returns
    -------
    delta : ndarray, shape (H, W, 3), native int32, C-contiguous
        Independent array of nonnegative integer attenuations.

    Notes
    -----
    All inputs are validated even when ``strength`` is zero.  Inputs are never
    modified.  The result is computed in int32/float64 and is not the final
    composite.
    """
    strength_f = _unit_parameter("strength", strength)
    bright_start_f = _unit_parameter("bright_start", bright_start)
    bright_end_f = _unit_parameter("bright_end", bright_end)
    if not (bright_start_f < bright_end_f):
        raise ValueError("bright_start must be strictly less than bright_end")
    span = bright_end_f - bright_start_f
    if not (span > 0.0):
        raise ValueError("bright_end - bright_start must be positive")

    _require_plain_array("base", base)
    if base.ndim != 3:
        raise ValueError("base must have shape (H, W, 3)")
    H, W, C = base.shape
    if C != 3:
        raise ValueError("base must have exactly 3 channels")
    if H <= 0 or W <= 0:
        raise ValueError("base must have positive height and width")
    if H * W > _MAX_PIXELS:
        raise ValueError(
            f"base tile must contain at most {_MAX_PIXELS} pixels, got {H * W}"
        )
    if base.dtype.kind != "u" or base.dtype.itemsize != 2:
        raise ValueError(
            f"base must have an unsigned 16-bit dtype, got {base.dtype}"
        )

    shape_2d = (H, W)
    shape_3d = (H, W, 3)
    _require_u16("source", source, shape_3d)
    _require_u16("diffuse", diffuse, shape_3d)
    _require_u16("brightness", brightness, shape_2d)
    _require_weight("weight", weight, shape_2d)

    # Signed int32 working copies.  astype copies, so readonly,
    # noncontiguous, negative-stride and opposite-endian inputs are accepted
    # without mutating them.
    base_i = base.astype(np.int32, copy=True)
    source_i = source.astype(np.int32, copy=True)
    diffuse_i = diffuse.astype(np.int32, copy=True)

    # Positive residual.  Channels where base <= diffuse stay zero and are
    # therefore never attenuated.
    P = np.maximum(base_i - diffuse_i, 0)

    bright_i = brightness.astype(np.int32, copy=True)
    max_residual = P.max(axis=2)
    if np.any(bright_i < max_residual):
        raise ValueError(
            "brightness must be at least the pixel's maximum positive residual"
        )

    weight_f = weight.astype(np.float64, copy=True)
    if not np.isfinite(weight_f).all():
        raise ValueError("weight must be finite")
    if np.any(weight_f < 0.0) or np.any(weight_f > 1.0):
        raise ValueError("weight must lie in [0, 1]")

    # Safe integer room: preserve existing nonzero codes in both composites.
    room = np.minimum(
        np.maximum(base_i - 1, 0),
        np.maximum(source_i - 1, 0),
    )

    # Smoothstep brightness ramp on the normalised positive residual.
    x = bright_i.astype(np.float64) / _MAX_U16
    numerator = np.clip(x - bright_start_f, 0.0, span)
    u = numerator / span
    np.clip(u, 0.0, 1.0, out=u)
    smooth = u * u * (3.0 - 2.0 * u)
    amount = strength_f * weight_f * (1.0 - smooth)

    # One common per-pixel cap from the channels that still have room.
    P_f = P.astype(np.float64)
    room_f = room.astype(np.float64)
    ratio = np.ones_like(P_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(room_f, P_f, out=ratio, where=(P > 0))
    cap = ratio.min(axis=2)
    np.clip(cap, 0.0, 1.0, out=cap)
    np.minimum(amount, cap, out=amount)
    np.clip(amount, 0.0, 1.0, out=amount)

    # Round once to nearest even, then bound any floating-point overshoot.
    delta = np.rint(P_f * amount[..., None])
    np.minimum(delta, room_f, out=delta)
    np.minimum(delta, P_f, out=delta)
    return np.ascontiguousarray(delta, dtype=np.int32)
