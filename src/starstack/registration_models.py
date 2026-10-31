"""Bounded model comparison for frame-to-frame star correspondences.

Given matched star positions in two frames of the same camera, fit a few
transform families on a *pre-declared, spatially stratified* fit subset and
report residuals separately on the validation subset.  Nothing here touches
Siril's own homography except to evaluate it as-is; a fitted model that beats
it on validation is evidence about the *model family*, not an acceptance of
any product transform (the correspondences were built through Siril's H, so
residuals beyond the association radius are invisible).

Models (source -> destination, column vectors ``[x, y, 1]``):

* ``similarity``  rotation + isotropic scale + translation (4 dof)
* ``affine``      6 dof
* ``homography``  8 dof, normalised DLT
* ``homography_radial``  homography between *undistorted* coordinates of both
  frames, with a common radial distortion ``r' = r (1 + k1 r^2 + k2 r^4)``
  about the frame centre (10 dof, Gauss-Newton).  Residuals are measured in
  the undistorted space of the destination frame; for the small ``k`` values
  of a camera lens this differs from pixel residuals by a factor
  ``1 + O(k r^2)``, which is stated in the output.
* ``polynomial2`` / ``polynomial3``  full 2-D polynomial maps (12 / 20 dof),
  a bounded stand-in for "local" corrections.

The split is deterministic: inside each cell of a ``columns x rows`` grid the
stars are sorted by (x, y) and alternate fit / validation, so both subsets
cover every region and no residual value influences membership.  Pure NumPy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["MODELS_VERSION", "stratified_split", "fit_similarity", "fit_affine", "fit_homography",
           "fit_polynomial", "apply_polynomial", "fit_homography_radial", "apply_homography_radial",
           "region_of", "evaluate_model", "compare_models"]

MODELS_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Split and regions                                                           #
# --------------------------------------------------------------------------- #

def stratified_split(points_xy, width: int, height: int, *, columns: int = 4, rows: int = 3) -> np.ndarray:
    """Boolean mask, True = fit, False = validation; alternating inside each grid cell."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    fit = np.zeros(len(pts), dtype=bool)
    col = np.clip((pts[:, 0] * columns / width).astype(int), 0, columns - 1)
    row = np.clip((pts[:, 1] * rows / height).astype(int), 0, rows - 1)
    for c in range(columns):
        for r in range(rows):
            idx = np.where((col == c) & (row == r))[0]
            if len(idx) == 0:
                continue
            order = idx[np.lexsort((pts[idx, 1], pts[idx, 0]))]
            fit[order[0::2]] = True
    return fit


def region_of(x: float, y: float, width: int, height: int) -> str:
    """Same 30 % / 70 % rule as registration_diagnostics; labels are in the
    coordinate frame given (Siril-native here, so top/bottom are *native* labels)."""
    u, v = (x + 0.5) / width, (y + 0.5) / height
    u_low, u_high = u < 0.3, u >= 0.7
    v_low, v_high = v < 0.3, v >= 0.7
    if not (u_low or u_high) and not (v_low or v_high):
        return "center"
    if (u_low or u_high) and (v_low or v_high):
        return "corner"
    return "other"


# --------------------------------------------------------------------------- #
# Linear model fits                                                           #
# --------------------------------------------------------------------------- #

def _check_pairs(src, dst, minimum: int):
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) != len(d):
        raise ValueError("source and destination must have the same number of points")
    if len(s) < minimum:
        raise ValueError(f"need at least {minimum} correspondences, got {len(s)}")
    if not (np.isfinite(s).all() and np.isfinite(d).all()):
        raise ValueError("correspondences must be finite")
    return s, d


def _normalise_points(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = p.mean(axis=0)
    scale = math.sqrt(2.0) / max(1e-12, np.mean(np.hypot(*(p - mean).T)))
    t = np.array([[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]])
    return t, (p - mean) * scale


def fit_similarity(src, dst) -> np.ndarray:
    s, d = _check_pairs(src, dst, 2)
    # x' = a x - b y + tx ; y' = b x + a y + ty
    n = len(s)
    a_mat = np.zeros((2 * n, 4))
    a_mat[0::2, 0], a_mat[0::2, 1], a_mat[0::2, 2] = s[:, 0], -s[:, 1], 1.0
    a_mat[1::2, 0], a_mat[1::2, 1], a_mat[1::2, 3] = s[:, 1], s[:, 0], 1.0
    rhs = d.reshape(-1)
    a, b, tx, ty = np.linalg.lstsq(a_mat, rhs, rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def fit_affine(src, dst) -> np.ndarray:
    s, d = _check_pairs(src, dst, 3)
    design = np.hstack([s, np.ones((len(s), 1))])
    coeff = np.linalg.lstsq(design, d, rcond=None)[0]          # 3x2
    return np.vstack([coeff.T, [0.0, 0.0, 1.0]])


def fit_homography(src, dst) -> np.ndarray:
    s, d = _check_pairs(src, dst, 4)
    ts, sn = _normalise_points(s)
    td, dn = _normalise_points(d)
    n = len(sn)
    a_mat = np.zeros((2 * n, 9))
    x, y = sn[:, 0], sn[:, 1]
    u, v = dn[:, 0], dn[:, 1]
    a_mat[0::2, 0], a_mat[0::2, 1], a_mat[0::2, 2] = -x, -y, -1.0
    a_mat[0::2, 6], a_mat[0::2, 7], a_mat[0::2, 8] = u * x, u * y, u
    a_mat[1::2, 3], a_mat[1::2, 4], a_mat[1::2, 5] = -x, -y, -1.0
    a_mat[1::2, 6], a_mat[1::2, 7], a_mat[1::2, 8] = v * x, v * y, v
    # Only the right singular vectors are used. Tall systems need no square U;
    # four pairs form an 8x9 system and require full Vt to retain its nullspace.
    _, _, vt = np.linalg.svd(a_mat, full_matrices=a_mat.shape[0] < a_mat.shape[1])
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    if abs(h[2, 2]) < 1e-15:
        raise ValueError("degenerate homography")
    return h / h[2, 2]


def apply_matrix(h, xy) -> np.ndarray:
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    v = np.hstack([p, np.ones((len(p), 1))]) @ np.asarray(h, dtype=np.float64).T
    return v[:, :2] / v[:, 2:3]


# --------------------------------------------------------------------------- #
# Exact fixed-radius neighbour candidates (P1a G-5)                            #
# --------------------------------------------------------------------------- #

GRID_COORD_LIMIT = 1.0e8     # |x|, |y| bound of the cell search: cell indices < 1e8 exactly, keys < 4.1e16 < 2**63
GRID_RADIUS_LIMIT = 1.0e6    # larger radii run the dense rule (the product uses 3..20 px)
GRID_CELL_MARGIN = 1.0625    # cell edge / threshold radius: a 6.25 % margin over every rounding error (~1e-8 cell)
GRID_PAIR_FRACTION = 32      # clustered input: more than n*m/32 candidates (and more than the floor) -> dense rule
GRID_MIN_PAIR_BUDGET = 65536
_GRID_RADIUS_TYPES = (int, float, np.float64)   # exact types; bool, subclasses and other NumPy scalars run dense


def grid_threshold(radius):
    """``(rr, cell)`` for the exact cell search, or None when the dense rule must run.

    ``rr`` is ``radius * radius`` -- the dense rule's own expression, so it has the dense rule's value and type.  Only
    radii of exact type ``int``, ``float`` or ``np.float64`` in ``[0, GRID_RADIUS_LIMIT]`` take the cell search.

    The cell search also requires NumPy's underflow handling to be ``"ignore"`` (the default).  Inside the grid domain
    the underflow flag is the only floating-point condition either rule can raise (finite coordinates within 1e8 cannot
    overflow, divide by zero or produce NaN), but the dense rule squares every pair and the cell search only the
    candidates (plus ``xy / cell``), so under ``np.errstate(under="warn"/"raise"/"call"/"print"/"log")`` the two would
    report underflow differently.  In that case the dense rule runs and reports exactly what it always did.
    """
    if np.geterr()["under"] != "ignore":
        return None
    if type(radius) not in _GRID_RADIUS_TYPES:
        return None
    if not 0.0 <= radius <= GRID_RADIUS_LIMIT:            # NaN fails every comparison
        return None
    rr = radius * radius
    cell = max(GRID_CELL_MARGIN * float(radius), GRID_CELL_MARGIN * math.sqrt(rr), 1.0)
    return rr, cell


def grid_points_ok(points) -> bool:
    """True for a non-empty float64 ``(n, 2)`` ndarray whose coordinates are all finite and within GRID_COORD_LIMIT.

    NaN makes ``max`` NaN and the comparison False; +-inf exceeds the limit.  Anything else runs the dense rule.
    Raises no floating-point condition for any float64 input (checked under ``np.errstate(all="raise")``).
    """
    return (type(points) is np.ndarray and points.dtype == np.float64 and points.ndim == 2 and points.shape[1] == 2
            and len(points) > 0 and bool(np.abs(points).max() <= GRID_COORD_LIMIT))


def pair_budget(n: int, m: int) -> int:
    """Most candidate pairs the cell search may materialise before the dense rule is cheaper (both are exact)."""
    return max(n * m // GRID_PAIR_FRACTION, GRID_MIN_PAIR_BUDGET)


def cell_pairs(a: np.ndarray, b: np.ndarray, cell: float, max_pairs: int):
    """All index pairs ``(i, j)`` whose cells ``floor(xy / cell)`` differ by at most one on both axes (3x3 block),
    or None when there would be more than ``max_pairs`` of them (clustered input: the dense rule is cheaper).

    Every pair appears exactly once (a point lies in one cell, the nine offsets are distinct cells); the order of the
    pairs is irrelevant to every caller.  Requires ``grid_points_ok`` for both sets and ``cell >= 1``.
    """
    ca = np.floor(a / cell).astype(np.int64)
    cb = np.floor(b / cell).astype(np.int64)
    x0 = min(int(ca[:, 0].min()), int(cb[:, 0].min())) - 1
    y0 = min(int(ca[:, 1].min()), int(cb[:, 1].min())) - 1
    stride = max(int(ca[:, 0].max()), int(cb[:, 0].max())) - x0 + 2     # column index + dx stays in [0, stride)
    key_b = (cb[:, 1] - y0) * stride + (cb[:, 0] - x0)
    order_b = np.argsort(key_b, kind="stable")
    sorted_b = key_b[order_b]
    key_a = (ca[:, 1] - y0) * stride + (ca[:, 0] - x0)
    runs, total = [], 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            key = key_a + (dy * stride + dx)
            lo = np.searchsorted(sorted_b, key, side="left")
            count = np.searchsorted(sorted_b, key, side="right") - lo
            runs.append((lo, count))
            total += int(count.sum())
    if total > max_pairs:
        return None
    rows = np.arange(len(a), dtype=np.intp)
    ii, jj = [], []
    for lo, count in runs:
        size = int(count.sum())
        if size == 0:
            continue
        start = np.repeat(lo - (np.cumsum(count) - count), count)     # sorted slot of run k's first entry, minus k's offset
        ii.append(np.repeat(rows, count))
        jj.append(order_b[start + np.arange(size)])
    if not ii:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    return np.concatenate(ii), np.concatenate(jj)


def first_min(owner: np.ndarray, other: np.ndarray, d2: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Per owner id in ``range(count)``: the ``other`` with the smallest ``d2``, the lowest ``other`` on ties (the
    ``argmin`` rule), and that ``d2``; ``-1`` / ``inf`` for owners without a candidate.  ``d2`` must be NaN-free."""
    best_other = np.full(count, -1, dtype=np.intp)
    best_d2 = np.full(count, np.inf)
    if len(owner):
        order = np.lexsort((other, d2, owner))        # primary owner, then d2, then index: first per owner = argmin
        sorted_owner = owner[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = sorted_owner[1:] != sorted_owner[:-1]
        pick = order[first]
        best_other[owner[pick]] = other[pick]
        best_d2[owner[pick]] = d2[pick]
    return best_other, best_d2


def pair_d2(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """``((a[ia] - b[ib]) ** 2).sum(axis=1)`` with the dense rule's bits: fl(square(dx) + square(dy)).

    ``np.square`` is the ufunc ``ndarray ** 2`` dispatches to, so each square is the dense rule's own operation; the
    dense ``.sum`` over a length-2 axis is one IEEE addition.
    """
    dx = a[ia, 0] - b[ib, 0]
    d2 = np.square(dx)
    dy = a[ia, 1] - b[ib, 1]
    d2 += np.square(dy)
    return d2


def _poly_terms(p: np.ndarray, degree: int, scale: float, centre: np.ndarray) -> np.ndarray:
    q = (p - centre) / scale
    cols = []
    for total in range(degree + 1):
        for i in range(total, -1, -1):
            cols.append(q[:, 0] ** i * q[:, 1] ** (total - i))
    return np.stack(cols, axis=1)


def fit_polynomial(src, dst, degree: int, *, centre=None, scale: float | None = None) -> dict:
    terms = (degree + 1) * (degree + 2) // 2
    s, d = _check_pairs(src, dst, terms + 2)
    centre = np.asarray(centre if centre is not None else s.mean(axis=0), dtype=np.float64)
    scale = float(scale or max(1.0, np.max(np.abs(s - centre))))
    design = _poly_terms(s, degree, scale, centre)
    coeff = np.linalg.lstsq(design, d, rcond=None)[0]
    return {"degree": degree, "centre": centre.tolist(), "scale": scale, "coefficients": coeff.tolist()}


def apply_polynomial(model: dict, xy) -> np.ndarray:
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    design = _poly_terms(p, model["degree"], model["scale"], np.asarray(model["centre"]))
    return design @ np.asarray(model["coefficients"])


# --------------------------------------------------------------------------- #
# Homography with common radial distortion                                    #
# --------------------------------------------------------------------------- #

def _undistort(p: np.ndarray, centre: np.ndarray, scale: float, k1: float, k2: float) -> np.ndarray:
    q = (p - centre) / scale
    r2 = (q ** 2).sum(axis=1, keepdims=True)
    return centre + scale * q * (1.0 + k1 * r2 + k2 * r2 * r2)


def apply_homography_radial(model: dict, xy) -> np.ndarray:
    """Map source points into the *undistorted* destination space."""
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    centre = np.asarray(model["centre"], dtype=np.float64)
    u = _undistort(p, centre, model["scale"], model["k1"], model["k2"])
    return apply_matrix(np.asarray(model["H"]), u)


def fit_homography_radial(src, dst, centre, *, scale: float | None = None, iterations: int = 30) -> dict:
    """Gauss-Newton (with damping) over 8 homography parameters and k1, k2.

    The residual for each pair is ``U(dst) - H U(src)`` with the same
    distortion in both frames.  Starts from the plain homography fit.
    """
    s, d = _check_pairs(src, dst, 12)
    centre = np.asarray(centre, dtype=np.float64)
    scale = float(scale or max(np.max(np.abs(s - centre)), 1.0))
    h0 = fit_homography(s, d)
    theta = np.concatenate([(h0 / h0[2, 2]).reshape(-1)[:8], [0.0, 0.0]])

    def residual(t):
        h = np.append(t[:8], 1.0).reshape(3, 3)
        us = _undistort(s, centre, scale, t[8], t[9])
        ud = _undistort(d, centre, scale, t[8], t[9])
        return (apply_matrix(h, us) - ud).reshape(-1)

    lam = 1e-3
    cost = float(np.sum(residual(theta) ** 2))
    for _ in range(iterations):
        r = residual(theta)
        jac = np.empty((len(r), len(theta)))
        for j in range(len(theta)):
            step = 1e-6 * max(1.0, abs(theta[j]))
            tp = theta.copy()
            tp[j] += step
            jac[:, j] = (residual(tp) - r) / step
        jtj = jac.T @ jac
        jtr = jac.T @ r
        improved = False
        for _ in range(8):
            delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), -jtr)
            candidate = theta + delta
            new_cost = float(np.sum(residual(candidate) ** 2))
            if new_cost < cost:
                theta, cost = candidate, new_cost
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved or float(np.max(np.abs(delta))) < 1e-12:
            break
    h = np.append(theta[:8], 1.0).reshape(3, 3)
    return {"H": h.tolist(), "k1": float(theta[8]), "k2": float(theta[9]), "centre": centre.tolist(), "scale": scale,
            "residual_space": "undistorted destination coordinates (pixel residuals differ by a factor 1 + O(k r^2))"}


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def _stats(dist: np.ndarray, dxy: np.ndarray) -> dict:
    if len(dist) == 0:
        return {"count": 0, "rms_px": None, "median_px": None, "p95_px": None, "max_px": None,
                "mean_dx_px": None, "mean_dy_px": None}
    return {"count": int(len(dist)), "rms_px": float(np.sqrt(np.mean(dist ** 2))), "median_px": float(np.median(dist)),
            "p95_px": float(np.percentile(dist, 95)), "max_px": float(np.max(dist)),
            "mean_dx_px": float(dxy[:, 0].mean()), "mean_dy_px": float(dxy[:, 1].mean())}


def evaluate_model(predicted: np.ndarray, dst: np.ndarray, fit_mask: np.ndarray, width: int, height: int) -> dict:
    """Residual statistics by subset (fit / validation) and region of the destination point."""
    predicted = np.asarray(predicted, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    dxy = predicted - d
    dist = np.hypot(dxy[:, 0], dxy[:, 1])
    regions = np.array([region_of(x, y, width, height) for x, y in d])
    out = {}
    for name, mask in (("fit", fit_mask), ("validation", ~fit_mask)):
        entry = {"overall": _stats(dist[mask], dxy[mask])}
        for region in ("center", "corner", "other"):
            m = mask & (regions == region)
            entry[region] = _stats(dist[m], dxy[m])
        out[name] = entry
    return out


def compare_models(src, dst, width: int, height: int, *, siril_matrix=None, columns: int = 4, rows: int = 3,
                   polynomial_degrees=(2, 3)) -> dict:
    """Fit every model family on the stratified fit subset; report both subsets.

    ``siril_matrix`` (source -> destination, same coordinates as the points)
    is evaluated as-is under the name ``siril_H_as_is`` — it was fitted by
    Siril on unknown stars, so neither subset is independent for it.
    """
    s, d = _check_pairs(src, dst, 30)
    fit_mask = stratified_split(s, width, height, columns=columns, rows=rows)
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    models = {}
    if siril_matrix is not None:
        models["siril_H_as_is"] = {"dof": 8, "independent_validation": False,
                                   "prediction": apply_matrix(siril_matrix, s), "parameters": np.asarray(siril_matrix).tolist()}
    sim = fit_similarity(s[fit_mask], d[fit_mask])
    models["similarity"] = {"dof": 4, "independent_validation": True, "prediction": apply_matrix(sim, s), "parameters": sim.tolist()}
    aff = fit_affine(s[fit_mask], d[fit_mask])
    models["affine"] = {"dof": 6, "independent_validation": True, "prediction": apply_matrix(aff, s), "parameters": aff.tolist()}
    hom = fit_homography(s[fit_mask], d[fit_mask])
    models["homography"] = {"dof": 8, "independent_validation": True, "prediction": apply_matrix(hom, s), "parameters": hom.tolist()}
    rad = fit_homography_radial(s[fit_mask], d[fit_mask], centre)
    # residual space for the radial model: undistorted destination
    ud = _undistort(d, centre, rad["scale"], rad["k1"], rad["k2"])
    models["homography_radial"] = {"dof": 10, "independent_validation": True,
                                   "prediction": apply_homography_radial(rad, s), "destination_override": ud,
                                   "parameters": rad}
    for degree in polynomial_degrees:
        poly = fit_polynomial(s[fit_mask], d[fit_mask], degree, centre=centre)
        models[f"polynomial{degree}"] = {"dof": (degree + 1) * (degree + 2), "independent_validation": True,
                                         "prediction": apply_polynomial(poly, s), "parameters": poly}
    results = {}
    for name, model in models.items():
        target = model.get("destination_override", d)
        results[name] = {"dof": model["dof"], "independent_validation": model["independent_validation"],
                         "parameters": model["parameters"],
                         "residuals": evaluate_model(model["prediction"], target, fit_mask, width, height)}
        if "destination_override" in model:
            results[name]["residual_space"] = model["parameters"]["residual_space"]
    return {"models_version": MODELS_VERSION, "count": int(len(s)), "fit_count": int(fit_mask.sum()),
            "validation_count": int((~fit_mask).sum()), "split": {"columns": columns, "rows": rows,
            "rule": "per grid cell, points sorted by (x, y) alternate fit/validation; declared before any fit"},
            "width": width, "height": height, "centre": centre.tolist(), "models": results,
            "fit_mask": fit_mask.tolist(),
            "caveats": ["correspondences were built through Siril's homography within the association radius; "
                        "residuals larger than that radius are invisible to every model here",
                        "validation stars are independent of the fits made here, not of Siril's own H",
                        "a better validation fit shows what the transform family can explain, not that the stack should use it"]}
