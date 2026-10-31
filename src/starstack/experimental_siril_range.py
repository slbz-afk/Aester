import math
import numpy as np

_FLOAT32 = np.dtype(np.float32)
_BOOL = np.dtype(np.bool_)


def _require_exact_array(value, name):
    if type(value) is not np.ndarray and type(value) is not np.memmap:
        raise ValueError(f"{name} must be an exact np.ndarray or np.memmap")


def _require_cube(value, name):
    _require_exact_array(value, name)
    if value.dtype != _FLOAT32:
        raise ValueError(f"{name} must have dtype float32")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")
    h, w = value.shape[0], value.shape[1]
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} must have positive height and width")
    return h, w


def _require_mask(value, h, w, name):
    _require_exact_array(value, name)
    if value.dtype != _BOOL:
        raise ValueError(f"{name} must have dtype bool")
    if value.ndim != 2 or value.shape != (h, w):
        raise ValueError(f"{name} must have shape (H, W) matching the cube")
    return value


def _finite_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an int or float scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be an int or float scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be converted to float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def encode_for_siril(data, valid, *, offset, scale):
    h, w = _require_cube(data, "data")
    _require_mask(valid, h, w, "valid")
    offset = _finite_float(offset, "offset")
    scale = _finite_float(scale, "scale")
    if not scale > 0.0:
        raise ValueError("scale must be strictly positive")
    _require_finite(data, "data")

    out = np.zeros(data.shape, dtype=np.float32, order="C")
    if not valid.any():
        return out

    samples = data[valid]
    x = samples.astype(np.float64, copy=False)

    with np.errstate(all="ignore"):
        shifted = x + offset
        scaled = shifted / scale
        encoded = scaled.astype(np.float32)

    if not np.isfinite(shifted).all():
        raise ValueError("encode produced a non-finite intermediate")
    if not np.isfinite(scaled).all():
        raise ValueError("encode produced a non-finite intermediate")
    if not np.isfinite(encoded).all():
        raise ValueError("encode produced a non-finite float32 value")
    if np.any(encoded <= np.float32(0.0)) or np.any(encoded >= np.float32(1.0)):
        raise ValueError("encoded valid samples must lie strictly between 0 and 1")

    out[valid] = encoded
    return out


def decode_from_siril(encoded, valid, *, offset, scale):
    h, w = _require_cube(encoded, "encoded")
    _require_mask(valid, h, w, "valid")
    offset = _finite_float(offset, "offset")
    scale = _finite_float(scale, "scale")
    if not scale > 0.0:
        raise ValueError("scale must be strictly positive")
    _require_finite(encoded, "encoded")

    invalid = ~valid
    if invalid.any() and np.any(encoded[invalid] != 0.0):
        raise ValueError("invalid encoded samples must be zero")

    out = np.zeros(encoded.shape, dtype=np.float32, order="C")
    if not valid.any():
        return out

    samples = encoded[valid]
    x = samples.astype(np.float64, copy=False)
    if np.any(x <= 0.0) or np.any(x >= 1.0):
        raise ValueError("valid encoded samples must lie strictly between 0 and 1")

    with np.errstate(all="ignore"):
        product = x * scale
        decoded = product - offset
        result = decoded.astype(np.float32)

    if not np.isfinite(product).all():
        raise ValueError("decode produced a non-finite intermediate")
    if not np.isfinite(decoded).all():
        raise ValueError("decode produced a non-finite intermediate")
    if not np.isfinite(result).all():
        raise ValueError("decode produced a non-finite float32 value")

    out[valid] = result
    return out
