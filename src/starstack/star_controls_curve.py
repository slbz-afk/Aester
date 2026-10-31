"""Full-range signed star-residual delta kernel.

The module exposes one function, :func:`full_range_delta`.  It computes a
signed integer delta for a single tile; callers apply the same delta to both
paired composites::

    edited_source = source - delta
    edited_base = base - delta

For strength values at or below 0.75 the delta is a nonnegative attenuation of
the positive residual ``P = max(base - diffuse, 0)``.  Above 0.75 it blends
toward the full signed residual ``R = int32(base) - int32(diffuse)``, so
negative residual components can be restored; restoration can increase values.
A single common per-pixel headroom cap limits the delta so that positive
deltas keep both composites nonnegative and negative deltas keep both
composites within uint16.

The function returns only the delta.  Outside the allowed/safe interior an
endpoint may be partial: when it cannot be reached because of the common
headroom cap, the signed residual is retained for the caller to count.
Mixed-sign pixels limited by the common cap are not claimed to be monotone.
All inputs are validated, and inputs are never modified.
"""

import numpy as np

from .star_reduction_curve import reduction_delta, _require_weight

__all__ = ["full_range_delta"]

_MAX_U16 = 65535.0


def full_range_delta(
    base,
    source,
    diffuse,
    brightness,
    weight,
    strength_map,
    *,
    bright_start=0.10,
    bright_end=0.45,
):
    """Return the signed full-range attenuation/restoration delta.

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
    strength_map : ndarray or memmap, shape (H, W), float32 or float64
        Per-pixel strength in [0, 1].  Values at or below 0.75 only attenuate
        positive residual; values above that blend toward full signed
        residual restoration.
    bright_start, bright_end : builtin int or float, default 0.10, 0.45
        Positive-residual brightness ramp in normalised [0, 1] units with
        ``0 <= bright_start < bright_end <= 1``.

    Returns
    -------
    delta : ndarray, shape (H, W, 3), native int32, C-contiguous
        Signed integer delta.  It is not a composite; callers subtract it
        from both ``base`` and ``source``.

    Notes
    -----
    The desired delta is computed in float64 and then subjected to one common
    per-pixel headroom cap.  The final integer clamp is a rounding guard and
    is not an additional per-channel gain.  When a full endpoint is
    unreachable, the signed residual is retained for the caller to count.
    Positive-star attenuation and negative-residual restoration can both be
    limited by other channels' headroom; mixed-sign results are not claimed to
    be monotone.  Since callers subtract the same delta from ``base`` and
    ``source``, their signed difference is preserved exactly.
    """
    # Reuse the existing kernel/validator with zero strength.  It validates
    # base, source, diffuse, brightness, weight, the tile limits and the
    # scalar brightness thresholds without mutating any input.
    reduction_delta(
        base,
        source,
        diffuse,
        brightness,
        weight,
        strength=0,
        bright_start=bright_start,
        bright_end=bright_end,
    )

    height, width, _ = base.shape
    _require_weight("strength_map", strength_map, (height, width))

    s = strength_map.astype(np.float64, copy=True)
    if not np.isfinite(s).all():
        raise ValueError("strength_map must be finite")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("strength_map must lie in [0, 1]")

    bright_start_f = float(bright_start)
    bright_end_f = float(bright_end)
    span = bright_end_f - bright_start_f

    # Signed int32 working copies.  astype copies, so readonly,
    # noncontiguous, negative-stride and opposite-endian inputs are accepted
    # without mutating them.
    base_i = base.astype(np.int32, copy=True)
    source_i = source.astype(np.int32, copy=True)
    diffuse_i = diffuse.astype(np.int32, copy=True)
    weight_f = weight.astype(np.float64, copy=True)

    signed = base_i - diffuse_i
    positive = np.maximum(signed, 0)

    # Smoothstep brightness ramp on the normalised positive residual.  Clip
    # the numerator into [0, span] before dividing so an extremely small
    # threshold interval cannot overflow.
    x = brightness.astype(np.float64, copy=True)
    x /= _MAX_U16
    np.subtract(x, bright_start_f, out=x)
    np.clip(x, 0.0, span, out=x)
    x /= span
    np.clip(x, 0.0, 1.0, out=x)
    bright_ramp = x * x * (3.0 - 2.0 * x)

    # Smoothstep blend from positive-only attenuation (q=0) to full signed
    # residual (q=1).
    q = (s - 0.75) / 0.25
    np.clip(q, 0.0, 1.0, out=q)
    q = q * q * (3.0 - 2.0 * q)

    positive_f = positive.astype(np.float64)
    signed_f = signed.astype(np.float64)
    attenuation = (1.0 - q) * s * (1.0 - bright_ramp)
    desired = weight_f[..., None] * (
        attenuation[..., None] * positive_f + q[..., None] * signed_f
    )

    # One common per-pixel cap.  Positive desired channels are limited by the
    # room that keeps both composites nonnegative; negative desired channels
    # are limited by the room that keeps both composites within uint16.  A
    # zero desired channel imposes ratio 1.
    base_f = base_i.astype(np.float64)
    source_f = source_i.astype(np.float64)
    pos_room = np.minimum(base_f, source_f)
    neg_room = np.minimum(_MAX_U16 - base_f, _MAX_U16 - source_f)
    room = np.where(
        desired > 0.0,
        pos_room,
        np.where(desired < 0.0, neg_room, 1.0),
    )
    ratio = np.ones_like(desired)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        np.divide(room, np.abs(desired), out=ratio, where=(desired != 0.0))
    cap = ratio.min(axis=2)
    np.clip(cap, 0.0, 1.0, out=cap)

    delta = np.rint(desired * cap[..., None])

    # Guard against integer rounding overshoot.  These bounds are exactly
    # [max(B-65535, S-65535), min(B, S)]; no independent gain is applied here.
    np.clip(delta, -neg_room, pos_room, out=delta)
    return np.ascontiguousarray(delta, dtype=np.int32)
