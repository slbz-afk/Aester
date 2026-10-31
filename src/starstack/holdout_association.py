"""Exclusive holdout association for a frozen evaluation matcher."""

import numpy as np

__all__ = ["associate_holdout"]

_CHUNK = 128
_MAX_COORD = 1.0e9
_MAX_COORD_INT = 1_000_000_000
_MAX_RADIUS = 1.0e9


def _as_points(value, name):
    """Return a validated float64 view/copy with shape (N, 2)."""
    try:
        arr = np.asarray(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must be convertible to a real numeric array of shape (N, 2)"
        ) from exc

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")

    kind = arr.dtype.kind
    if kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must contain real numeric values")

    if kind == "u":
        if np.any(arr > _MAX_COORD_INT):
            raise ValueError(
                f"{name} coordinates must not exceed 1e9 in absolute value"
            )
        try:
            out = np.asarray(arr, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} could not be converted to float64") from exc
    elif kind == "i":
        if np.any(arr > _MAX_COORD_INT) or np.any(arr < -_MAX_COORD_INT):
            raise ValueError(
                f"{name} coordinates must not exceed 1e9 in absolute value"
            )
        try:
            out = np.asarray(arr, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} could not be converted to float64") from exc
    else:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        if arr.dtype.itemsize > 8:
            if np.any(arr > _MAX_COORD) or np.any(arr < -_MAX_COORD):
                raise ValueError(
                    f"{name} coordinates must not exceed 1e9 in absolute value"
                )
        try:
            out = np.asarray(arr, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} could not be converted to float64") from exc
        if not np.all(np.isfinite(out)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(out > _MAX_COORD) or np.any(out < -_MAX_COORD):
            raise ValueError(
                f"{name} coordinates must not exceed 1e9 in absolute value"
            )

    return out


def _as_radius(radius):
    """Return a validated builtin float radius."""
    if type(radius) is bool:
        raise ValueError("radius must be a builtin int or float, not bool")
    if type(radius) is int:
        try:
            r = float(radius)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("radius could not be converted to a finite float") from exc
    elif type(radius) is float:
        r = radius
    else:
        raise ValueError("radius must be a builtin int or float")

    if not bool(np.isfinite(r)):
        raise ValueError("radius must be finite")
    if not (r > 0.0):
        raise ValueError("radius must be greater than zero")
    if r > _MAX_RADIUS:
        raise ValueError("radius must not exceed 1e9")
    return r


def associate_holdout(reference_xy, predicted_xy, *, radius):
    """Associate reference points to predicted coordinates with a fixed radius.

    ``predicted_xy`` are coordinates from a separate identity matcher, not from
    the tested model.  The function retains one result entry for every
    reference point.  It does not fit a transform, select a model, resolve
    ambiguity greedily, or reject entries based on candidate residuals.  The
    finite-radius outcome stays visible: failures are reported as ``missing``
    or ``ambiguous``, and a ``matched`` value records only a geometric radius
    coincidence, not proof of physical identity.

    Parameters
    ----------
    reference_xy, predicted_xy : array-like
        Real numeric arrays of shape ``(N, 2)``.  Empty ``(0, 2)`` inputs are
        accepted.
    radius : builtin int or float
        Positive finite association radius no larger than ``1e9``.

    Returns
    -------
    list of dict
        One builtin dict per reference row, in reference-index order.  Each
        dict has exactly ``reference_index``, ``source_index``, ``status``,
        ``candidate_count`` and ``distance_px``.
    """
    ref = _as_points(reference_xy, "reference_xy")
    src = _as_points(predicted_xy, "predicted_xy")
    r = _as_radius(radius)
    r2 = r * r

    n_ref = int(ref.shape[0])
    n_src = int(src.shape[0])

    ref_counts = np.empty(n_ref, dtype=np.int64)
    single_src = np.full(n_ref, -1, dtype=np.int64)
    single_d2 = np.full(n_ref, np.nan, dtype=np.float64)
    global_counts = np.zeros(n_src, dtype=np.int64)

    for start in range(0, n_ref, _CHUNK):
        end = min(start + _CHUNK, n_ref)
        m = end - start
        chunk = ref[start:end]

        d2 = np.empty((m, n_src), dtype=np.float64)
        tmp = np.empty((m, n_src), dtype=np.float64)

        np.subtract(chunk[:, 0:1], src[None, :, 0], out=d2)
        np.multiply(d2, d2, out=d2)
        np.subtract(chunk[:, 1:2], src[None, :, 1], out=tmp)
        np.multiply(tmp, tmp, out=tmp)
        d2 += tmp
        del tmp

        eligible = d2 <= r2
        row_counts = eligible.sum(axis=1, dtype=np.int64)
        ref_counts[start:end] = row_counts
        global_counts += eligible.sum(axis=0, dtype=np.int64)

        one = row_counts == 1
        if np.any(one):
            best = np.argmax(eligible, axis=1)
            rows = np.flatnonzero(one)
            single_src[start + rows] = best[rows]
            single_d2[start + rows] = d2[rows, best[rows]]

        del d2, eligible

    results = []
    for i in range(n_ref):
        count = int(ref_counts[i])
        if count == 0:
            results.append(
                {
                    "reference_index": int(i),
                    "source_index": None,
                    "status": "missing",
                    "candidate_count": 0,
                    "distance_px": None,
                }
            )
        elif count == 1:
            j = int(single_src[i])
            if int(global_counts[j]) == 1:
                results.append(
                    {
                        "reference_index": int(i),
                        "source_index": j,
                        "status": "matched",
                        "candidate_count": 1,
                        "distance_px": float(np.sqrt(single_d2[i])),
                    }
                )
            else:
                results.append(
                    {
                        "reference_index": int(i),
                        "source_index": None,
                        "status": "ambiguous",
                        "candidate_count": 1,
                        "distance_px": None,
                    }
                )
        else:
            results.append(
                {
                    "reference_index": int(i),
                    "source_index": None,
                    "status": "ambiguous",
                    "candidate_count": count,
                    "distance_px": None,
                }
            )

    return results
