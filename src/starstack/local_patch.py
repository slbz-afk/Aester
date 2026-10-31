import numpy as np

__all__ = ["compose_local_patch"]

_MAX_ROI_DIM = 1024
_ALLOWED_DTYPE_TYPES = (np.uint8, np.uint16, np.float32)


def _require_plain_ndarray(value, name):
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be a plain numpy.ndarray")


def _check_float_data(arr, name):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _validate_original(original):
    _require_plain_ndarray(original, "original")
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original must have shape (H, W, 3)")
    H = int(original.shape[0])
    W = int(original.shape[1])
    if H < 2 or W < 2:
        raise ValueError("original height and width must each be at least 2")
    if original.dtype.type not in _ALLOWED_DTYPE_TYPES:
        raise ValueError("original dtype must be uint8, uint16, or float32")
    if original.dtype.type is np.float32:
        _check_float_data(original, "original")
    return H, W


def _validate_roi(roi, H, W):
    if type(roi) is not tuple:
        raise ValueError("roi must be a builtin tuple")
    if len(roi) != 4:
        raise ValueError("roi must contain exactly four builtin integers")
    if any(type(v) is not int for v in roi):
        raise ValueError("roi values must be builtin integers")
    x, y, w, h = roi
    if x < 0 or y < 0:
        raise ValueError("roi coordinates must be non-negative")
    if w <= 0 or h <= 0:
        raise ValueError("roi width and height must be positive")
    if w > _MAX_ROI_DIM or h > _MAX_ROI_DIM:
        raise ValueError("roi width and height must be at most 1024")
    if x + w > W or y + h > H:
        raise ValueError("roi must fit inside the image")
    if w * h * 100 > H * W * 4:
        raise ValueError("roi area must be at most 4% of the image")
    return x, y, w, h


def _validate_patch(patch, original, h, w):
    _require_plain_ndarray(patch, "patch")
    if patch.shape != (h, w, 3):
        raise ValueError("patch must have shape (h, w, 3)")
    if patch.dtype != original.dtype:
        raise ValueError("patch must have the same dtype as original")
    if original.dtype.type is np.float32:
        _check_float_data(patch, "patch")


def _validate_alpha(alpha, h, w, H, W):
    _require_plain_ndarray(alpha, "alpha")
    if alpha.dtype.type is not np.float32:
        raise ValueError("alpha must have dtype float32")
    if alpha.shape != (h, w):
        raise ValueError("alpha must have shape (h, w)")
    if not np.isfinite(alpha).all():
        raise ValueError("alpha must be finite")
    if alpha.min() < 0.0 or alpha.max() > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    active = alpha > np.float32(0.0)
    positive_count = int(np.count_nonzero(active))
    if positive_count <= 0:
        raise ValueError("alpha must contain at least one positive sample")
    if positive_count * 100 > H * W:
        raise ValueError("alpha positive sample count must be at most 1% of the image")
    return active, positive_count


def _validate_protected(protected, H, W, x, y, h, w, active):
    if protected is None:
        return
    _require_plain_ndarray(protected, "protected")
    if protected.dtype != np.dtype(bool):
        raise ValueError("protected must have dtype bool")
    if protected.shape != (H, W):
        raise ValueError("protected must have shape (H, W)")
    protected_region = protected[y:y+h, x:x+w]
    if np.any(active & protected_region):
        raise ValueError("positive alpha samples must not intersect protected pixels")


def compose_local_patch(original, patch, roi, alpha, *, protected=None):
    """Composite a bounded local patch with alpha blending.

    Returns ``(output, audit)``. Only positive-alpha samples inside ``roi``
    are written; every other sample remains bitwise identical to ``original``.
    """
    H, W = _validate_original(original)
    x, y, w, h = _validate_roi(roi, H, W)
    _validate_patch(patch, original, h, w)
    active, active_pixels = _validate_alpha(alpha, h, w, H, W)
    _validate_protected(protected, H, W, x, y, h, w, active)

    output = original.copy()
    orig_region = original[y:y+h, x:x+w]
    out_region = output[y:y+h, x:x+w]

    alpha_active = alpha[active].astype(np.float64)
    orig_active = orig_region[active].astype(np.float64)
    patch_active = patch[active].astype(np.float64)

    blend = (1.0 - alpha_active)[:, None] * orig_active
    blend += alpha_active[:, None] * patch_active

    exact_one = alpha_active == 1.0
    if np.any(exact_one):
        blend[exact_one] = patch_active[exact_one]

    if original.dtype.type is np.float32:
        result = blend.astype(original.dtype)
    else:
        result = np.rint(blend).astype(original.dtype)

    out_region[active] = result

    changed = np.any(out_region != orig_region, axis=2)
    changed_pixels = int(np.count_nonzero(changed))

    audit = {
        "version": "local-patch-v1",
        "roi": [int(x), int(y), int(w), int(h)],
        "image_shape": [int(H), int(W), 3],
        "active_pixels": int(active_pixels),
        "changed_pixels": int(changed_pixels),
        "max_allowed_pixels": int((H * W) // 100),
        "outside_mask_unchanged": True,
        "generative_content_possible": True,
        "source_preserved": True,
    }

    return output, audit
