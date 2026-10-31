"""Signed display-layer codec for small image blocks.

This module implements two pure functions for display-code blocks shaped
``(H, W, 3)`` with positive ``H`` and ``W``, and ``1 <= H*W <= 65536``.

Only exact ``numpy.ndarray`` and ``numpy.memmap`` containers are accepted.
Inputs may be read-only, noncontiguous, negative-stride, or opposite-endian;
they are never modified.  All returned arrays are new, independent,
C-contiguous native ``uint16`` ndarrays.

The functions enforce arithmetic bounds only; they do not assess photographic
quality or physical reconstruction.
"""

import numpy as np

__all__ = ["split_signed_codes", "compose_layer_codes"]

_MAX_PIXELS = 65536
_UINT16_MAX = 65535
_SIGNED_MIN = -65535
_SIGNED_MAX = 65535


def _require_supported_container(arr, name):
    """Reject any container that is not exactly ndarray or memmap."""
    if type(arr) is not np.ndarray and type(arr) is not np.memmap:
        raise ValueError(
            f"{name} must be a numpy.ndarray or numpy.memmap, "
            f"not {type(arr).__name__}"
        )


def _validate_block_shape(arr, name):
    """Require shape (H, W, 3) with positive H, W and H*W <= 65536."""
    shape = arr.shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3); got {shape}")
    h, w = shape[0], shape[1]
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must have positive H and W; got {shape}")
    if h * w > _MAX_PIXELS:
        raise ValueError(
            f"{name} has H*W={h * w}, exceeding the limit {_MAX_PIXELS}"
        )


def _validate_signed_int32_dtype(arr, name):
    """Require a signed 4-byte integer dtype, accepting either byte order."""
    if arr.dtype.kind != "i" or arr.dtype.itemsize != 4:
        raise ValueError(
            f"{name} must use a signed 4-byte integer dtype "
            f"(kind 'i', itemsize 4); got {arr.dtype}"
        )


def _validate_uint16_dtype(arr, name):
    """Require an unsigned 2-byte integer dtype, accepting either byte order."""
    if arr.dtype.kind != "u" or arr.dtype.itemsize != 2:
        raise ValueError(
            f"{name} must use an unsigned 2-byte integer dtype "
            f"(kind 'u', itemsize 2); got {arr.dtype}"
        )


def _validate_int32_block(arr, name):
    """Validate container, shape, and dtype for an int32 block."""
    _require_supported_container(arr, name)
    _validate_block_shape(arr, name)
    _validate_signed_int32_dtype(arr, name)


def _validate_uint16_block(arr, name, expected_shape=None):
    """Validate container, shape, dtype, and optional shape match."""
    _require_supported_container(arr, name)
    _validate_block_shape(arr, name)
    _validate_uint16_dtype(arr, name)
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{name} shape {arr.shape} does not match background shape "
            f"{expected_shape}"
        )


def _validate_signed_range(arr, name):
    """Require every value in [-65535, 65535]."""
    lo = int(arr.min())
    hi = int(arr.max())
    if lo < _SIGNED_MIN or hi > _SIGNED_MAX:
        raise ValueError(
            f"{name} values must lie in [{_SIGNED_MIN}, {_SIGNED_MAX}]; "
            f"got [{lo}, {hi}]"
        )


def _validate_intermediate(value, step):
    """Require every int32 intermediate value in [0, 65535]."""
    lo = int(value.min())
    hi = int(value.max())
    if lo < 0 or hi > _UINT16_MAX:
        raise ValueError(
            f"intermediate after {step} is outside [0, {_UINT16_MAX}]; "
            f"got [{lo}, {hi}]"
        )


def _as_native_int32_block(arr):
    """Return a new C-contiguous ordinary native int32 copy of arr."""
    return np.array(arr, dtype=np.int32, order="C", copy=True)


def _as_uint16_output(values):
    """Return a new C-contiguous ordinary native uint16 array of values."""
    return np.array(values, dtype=np.uint16, order="C", copy=True)


def split_signed_codes(block):
    """Split a signed display-code block into positive and negative planes.

    Parameters
    ----------
    block : numpy.ndarray or numpy.memmap
        Block with shape ``(H, W, 3)``, positive ``H`` and ``W``,
        ``1 <= H*W <= 65536``, and a signed 4-byte integer dtype
        (native or opposite-endian).  Every value must lie in
        ``[-65535, 65535]``.  The input is never modified.

    Returns
    -------
    positive : numpy.ndarray
        New C-contiguous native ``uint16`` array equal to ``max(block, 0)``.
    negative : numpy.ndarray
        New C-contiguous native ``uint16`` array equal to ``max(-block, 0)``.

    Raises
    ------
    ValueError
        If the container, shape, dtype, or value range is invalid.
    """
    _validate_int32_block(block, "block")
    _validate_signed_range(block, "block")

    block_i32 = _as_native_int32_block(block)

    positive = _as_uint16_output(np.maximum(block_i32, 0))
    negative = _as_uint16_output(np.maximum(-block_i32, 0))
    return positive, negative


def compose_layer_codes(
    background,
    stars_positive,
    stars_negative,
    trails_positive=None,
    trails_negative=None,
):
    """Compose display-code layers with strict per-step range checking.

    The exact arithmetic order is::

        value = background - stars_negative
        value = value + stars_positive
        value = value - trails_negative   # only when trails are provided
        value = value + trails_positive   # only when trails are provided

    After each subtraction or addition, every intermediate value must lie
    in ``[0, 65535]``.  An underflow or overflow raises ``ValueError``
    immediately and is never clipped, wrapped, or compensated by a later
    layer.  Inputs are never modified.

    Parameters
    ----------
    background, stars_positive, stars_negative : numpy.ndarray or numpy.memmap
        Blocks with identical shape ``(H, W, 3)``, positive ``H`` and ``W``,
        ``1 <= H*W <= 65536``, and an unsigned 2-byte integer dtype
        (native or opposite-endian).
    trails_positive, trails_negative : numpy.ndarray or numpy.memmap, optional
        Optional trail layers.  They must either both be ``None`` or both be
        valid blocks matching the other inputs.

    Returns
    -------
    numpy.ndarray
        New C-contiguous native ``uint16`` array with the composed codes.

    Raises
    ------
    ValueError
        If any input is invalid, trail arguments are mismatched, or an
        arithmetic step leaves ``[0, 65535]``.
    """
    _validate_uint16_block(background, "background")
    shape = background.shape
    _validate_uint16_block(stars_positive, "stars_positive", shape)
    _validate_uint16_block(stars_negative, "stars_negative", shape)

    if (trails_positive is None) != (trails_negative is None):
        raise ValueError(
            "trails_positive and trails_negative must both be provided "
            "or both be None"
        )

    use_trails = trails_positive is not None
    if use_trails:
        _validate_uint16_block(trails_positive, "trails_positive", shape)
        _validate_uint16_block(trails_negative, "trails_negative", shape)

    background_i32 = _as_native_int32_block(background)
    stars_positive_i32 = _as_native_int32_block(stars_positive)
    stars_negative_i32 = _as_native_int32_block(stars_negative)

    if use_trails:
        trails_positive_i32 = _as_native_int32_block(trails_positive)
        trails_negative_i32 = _as_native_int32_block(trails_negative)

    value = background_i32 - stars_negative_i32
    _validate_intermediate(value, "subtracting stars_negative")

    value = value + stars_positive_i32
    _validate_intermediate(value, "adding stars_positive")

    if use_trails:
        value = value - trails_negative_i32
        _validate_intermediate(value, "subtracting trails_negative")

        value = value + trails_positive_i32
        _validate_intermediate(value, "adding trails_positive")

    return _as_uint16_output(value)
