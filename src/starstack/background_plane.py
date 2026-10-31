"""Fit an empirical plane per RGB channel to explicitly supplied samples."""

import numpy as np

_ALGORITHM = "least_squares_plane_v1"
_MIN_SAMPLES = 12
_MAX_CONDITION = 100.0


def _as_float64_samples(name, arr, columns):
    """Validate one input ndarray and return an independent float64 copy."""
    if type(arr) is not np.ndarray:
        raise ValueError(
            f"{name} must be a NumPy ndarray, got {type(arr).__name__}"
        )
    if arr.dtype.kind != "f" or arr.dtype.itemsize not in (4, 8):
        raise ValueError(
            f"{name} must have float32 or float64 dtype, got {arr.dtype!r}"
        )
    if arr.ndim != 2 or arr.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {arr.shape}")

    out = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def fit_plane(xy, rgb) -> dict:
    """Fit an empirical plane per RGB channel to the supplied samples.

    The plane is only an empirical summary of these samples; it does not
    establish that they represent background.
    """
    xy64 = _as_float64_samples("xy", xy, 2)
    rgb64 = _as_float64_samples("rgb", rgb, 3)

    sample_count = xy64.shape[0]
    if rgb64.shape[0] != sample_count:
        raise ValueError(
            "xy and rgb must have the same number of rows, got "
            f"{sample_count} and {rgb64.shape[0]}"
        )
    if sample_count < _MIN_SAMPLES:
        raise ValueError(
            f"at least {_MIN_SAMPLES} samples are required, got {sample_count}"
        )
    if np.any(xy64 < -1.0) or np.any(xy64 > 1.0):
        raise ValueError(
            "every xy coordinate must lie in the inclusive range [-1, 1]"
        )

    A = np.empty((sample_count, 3), dtype=np.float64)
    A[:, 0] = 1.0
    A[:, 1] = xy64[:, 0]
    A[:, 2] = xy64[:, 1]

    try:
        with np.errstate(all="ignore"):
            if np.linalg.matrix_rank(A) < 3:
                raise ValueError("design matrix A has rank < 3")

            condition_number = float(np.linalg.cond(A, 2))
            if not np.isfinite(condition_number):
                raise ValueError("condition number of A is not finite")
            if condition_number > _MAX_CONDITION:
                raise ValueError(
                    f"condition number of A exceeds {_MAX_CONDITION}: "
                    f"{condition_number!r}"
                )

            coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
                A, rgb64, rcond=None
            )
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("least-squares coefficients are not finite")

            fitted = A @ coefficients
            residual_rms = np.sqrt(np.mean(np.square(rgb64 - fitted), axis=0))
            if not np.all(np.isfinite(residual_rms)):
                raise ValueError("residual RMS values are not finite")
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"linear algebra failed: {exc}") from exc
    except FloatingPointError as exc:
        raise ValueError(f"numerical computation failed: {exc}") from exc

    return {
        "coefficients": [
            [float(value) for value in row] for row in coefficients.T
        ],
        "condition_number": condition_number,
        "sample_count": int(sample_count),
        "residual_rms_rgb": [float(value) for value in residual_rms],
        "algorithm": _ALGORITHM,
    }
