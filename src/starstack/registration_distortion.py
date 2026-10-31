"""Bounded, shared correction terms fitted jointly over many frames (pure NumPy).

After two-pass registration the per-frame homography leaves a small residual
field that is *linear in the span* between frame and reference (RUN_LOG §16:
frame 6 at span -5 and frame 16 at span +5 show mirror-image grid means).  A
global homography cannot absorb it.  Two bounded explanations are fitted
here on the same correspondences, with one parameter set shared by all
frames and one homography per frame:

* ``radial``  a sensor-fixed radial distortion ``u = c + (p - c)(1 + k1 r^2 + k2 r^4)``
  (``r`` = |p - c| / scale) applied to both frame and reference points; the
  homography acts between undistorted points.  Predicted positions are
  mapped back through the inverse distortion so every residual is in
  reference *pixel* space.  4 shared parameters (k1, k2, cx, cy).
* ``field``   a polynomial displacement field ``v(x, y)`` in the reference
  frame scaled by the span: ``p_ref = H_f(p_frame) + s_f v(H_f(p_frame))``.
  It is the first-order form of any effect that grows with the sky
  displacement (differential distortion, differential refraction, ...).
  ``degree`` 1 -> 6, 2 -> 12 shared parameters.

Everything is evaluated on the pre-declared stratified validation subset
(``registration_models.stratified_split``); the *baseline* is the per-frame
homography fitted on the same fit subset without any shared term, and
Siril's own H is evaluated as-is (not independent).  A transfer test fits
the shared parameters on some frames only and evaluates the others with
fresh per-frame homographies: that is the claim that matters for a product
term — the term must predict frames it was not fitted on.

Nothing here is a product transform; it bounds what a shared term could
recover and which physical story is consistent with the data.
"""
from __future__ import annotations

import numpy as np

from .registration_models import (_poly_terms, apply_matrix, cell_pairs, evaluate_model, first_min, fit_homography,
                                  grid_points_ok, grid_threshold, pair_budget, pair_d2, region_of, stratified_split)

__all__ = ["DISTORTION_VERSION", "RadialDistortion", "frames_from_document", "span_of", "fit_joint_radial",
           "fit_joint_field", "predict", "evaluate_frames", "grid_signed_means", "transfer_test", "analyse",
           "mutual_nearest", "extrapolate_homography", "reassociate_frame", "reassociate_sequence"]

DISTORTION_VERSION = "1.3.1"


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

def span_of(frame_id: str, reference_id: str) -> int:
    """Signed number of images between frame and reference from the trailing digits (light_00016 - light_00011 = +5)."""
    def number(s: str) -> int:
        digits = ""
        for ch in reversed(s):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise ValueError(f"no trailing number in {s!r}")
        return int(digits)
    return number(frame_id) - number(reference_id)


def frames_from_document(document: dict, *, exclude: tuple[str, ...] = ()) -> dict:
    """Matched pairs per frame from a diagnostics document (adapter output).

    Returns ``{frame_id: {"span", "src", "dst", "H_siril"}}`` with TOP-DOWN
    reference/source coordinates exactly as the document carries them.
    """
    reference = document["reference"]["frame_id"]
    frames = {}
    for frame in document["frames"]:
        fid = frame["frame_id"]
        if fid in exclude or fid == reference:
            continue
        if frame.get("source_to_reference") is None or not frame.get("matches"):
            continue   # frame Siril could not register (zero matrix, no association): nothing to fit
        src = np.array([m["source_xy"] for m in frame["matches"]], dtype=np.float64).reshape(-1, 2)
        dst = np.array([m["reference_xy"] for m in frame["matches"]], dtype=np.float64).reshape(-1, 2)
        frames[fid] = {"span": span_of(fid, reference), "src": src, "dst": dst,
                       "H_siril": np.asarray(frame["source_to_reference"], dtype=np.float64)}
    return frames


# --------------------------------------------------------------------------- #
# Radial distortion                                                           #
# --------------------------------------------------------------------------- #

DIRECTIONS = ("p_to_u", "u_to_p")


class RadialDistortion:
    """Radial lens model between native pixels ``p`` and the undistorted grid ``u`` (both TOP-DOWN, pixel centres).

    ``direction="p_to_u"`` (the original form): ``u = c + m (p - c) (1 + k1 r^2 + k2 r^4)``
    with ``r = |p - c| / scale``; ``inverse`` by Newton on r.

    ``direction="u_to_p"`` (RUN_LOG §22, the product default since then): the same
    polynomial acts on the *undistorted* side, ``p = c + (u - c) (1 + k1 r^2 +
    k2 r^4)`` with ``r = |u - c| / scale`` and ``u`` the unmagnified undistorted
    position; ``apply`` (p -> u) is then the Newton step.  Siril exports frames
    through the SIP *inverse* polynomials (AP/BP, u -> p) and its
    implementation crashes above order 5, so this direction makes the exact
    lens model representable: AP/BP are exact, only the forward A/B (which the
    product does not use) are a least-squares fit.  With the original
    direction an order-5 AP/BP fit of the Z8 20 mm model leaves 0.96 px at
    the extreme corners.

    ``magnification`` ``m`` is a global scale of the undistorted grid (1 by
    default): the grid Siril works in is ``u' = c + m (u - c)``.  A product fit
    sets ``m = 1 / max(1, max |u - c| / |p - c|)`` over the image outline so
    that every native pixel maps *inside* the native box after undistortion:
    a barrel lens then loses no corner when Siril exports on the reference
    frame's own size (``-framing=current``).
    """

    def __init__(self, k1: float, k2: float, cx: float, cy: float, scale: float, magnification: float = 1.0, direction: str = "p_to_u"):
        self.k1, self.k2, self.cx, self.cy, self.scale = float(k1), float(k2), float(cx), float(cy), float(scale)
        self.magnification = float(magnification)
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        self.direction = direction
        if not (self.scale > 0 and self.magnification > 0):
            raise ValueError("scale and magnification must be positive")

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    def radial_factor(self, r2: np.ndarray) -> np.ndarray:
        return 1.0 + self.k1 * r2 + self.k2 * r2 * r2

    def _poly(self, q: np.ndarray) -> np.ndarray:
        """``q (1 + k1 r^2 + k2 r^4)`` for normalised offsets ``q`` (r = |q|)."""
        r2 = (q ** 2).sum(axis=1, keepdims=True)
        return q * self.radial_factor(r2)

    def _poly_inverse(self, q: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Solve ``x (1 + k1 |x|^2 + k2 |x|^4) = q`` for x (radial Newton on |x|).

        P1a G-2: the float64 operations, operands, operand order and first-occurrence order are exactly those of
        ``f = r * (1.0 + self.k1 * r ** 2 + self.k2 * r ** 4) - rq``,
        ``df = 1.0 + 3.0 * self.k1 * r ** 2 + 5.0 * self.k2 * r ** 4`` and
        ``r = r - f / np.where(np.abs(df) < 1e-12, 1e-12, df)``; only the second evaluation of ``r ** 2`` and of
        ``r ** 4`` (in ``df``) is replaced by the first one, and every ufunc writes into a buffer allocated once per
        call.  ``r ** 2`` is numpy's exponent-2 fast path ``np.square``; ``r ** 4`` is ``np.power(r, 4)`` and must
        NOT become ``r2 * r2`` (one rounding versus two would change the model's numbers).
        """
        rq = np.hypot(q[:, 0], q[:, 1])
        r = rq.copy()
        k1, k2 = self.k1, self.k2
        r2, r4, f, df, t = (np.empty_like(r) for _ in range(5))
        small = np.empty(r.shape, dtype=bool)
        for _ in range(iterations):
            np.square(r, out=r2)                  # r ** 2
            np.multiply(k1, r2, out=f)            # self.k1 * r ** 2
            np.add(1.0, f, out=f)                 # 1.0 + ...
            np.power(r, 4, out=r4)                # r ** 4
            np.multiply(k2, r4, out=t)            # self.k2 * r ** 4
            np.add(f, t, out=f)                   # (1.0 + k1 r^2) + k2 r^4
            np.multiply(r, f, out=f)              # r * (...)
            np.subtract(f, rq, out=f)             # ... - rq
            np.multiply(3.0 * k1, r2, out=df)     # (3.0 * self.k1) * r ** 2   (the second r ** 2 is the first one)
            np.add(1.0, df, out=df)               # 1.0 + ...
            np.multiply(5.0 * k2, r4, out=t)      # (5.0 * self.k2) * r ** 4   (likewise r ** 4)
            np.add(df, t, out=df)                 # (...) + (...)
            np.abs(df, out=t)                     # np.where(np.abs(df) < 1e-12, 1e-12, df)
            np.less(t, 1e-12, out=small)
            np.copyto(df, 1e-12, where=small)
            np.divide(f, df, out=f)               # f / ...
            np.subtract(r, f, out=r)              # r - ...
        del r2, r4, f, df, t, small               # free the work buffers before the tail (peak as before)
        factor = np.where(rq > 0, r / np.where(rq > 0, rq, 1.0), 1.0)
        return q * factor[:, None]

    def apply(self, p) -> np.ndarray:
        """Native pixel -> (magnified) undistorted grid."""
        q = (np.asarray(p, dtype=np.float64).reshape(-1, 2) - self.centre) / self.scale
        u = self._poly(q) if self.direction == "p_to_u" else self._poly_inverse(q)
        return self.centre + self.scale * self.magnification * u

    def inverse(self, u, iterations: int = 8) -> np.ndarray:
        """(Magnified) undistorted grid -> native pixel."""
        q = (np.asarray(u, dtype=np.float64).reshape(-1, 2) - self.centre) / (self.scale * self.magnification)
        p = self._poly_inverse(q, iterations) if self.direction == "p_to_u" else self._poly(q)
        return self.centre + self.scale * p

    def corner_shift_px(self, width: int, height: int) -> float:
        """How far the mapping moves the farthest image corner (pixels)."""
        corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
        return float(np.max(np.hypot(*(self.apply(corners) - corners).T)))

    def fitting_magnification(self, width: int, height: int) -> float:
        """The largest ``m`` that keeps the whole native box inside itself after undistortion.

        Sampled on the box outline: ``1 / max(1, max |u - c| / |p - c|)`` with
        ``u`` unmagnified, so a pincushion lens keeps ``m = 1``.
        """
        xs = np.linspace(0, width - 1, 65)
        ys = np.linspace(0, height - 1, 65)
        outline = np.vstack([np.column_stack([xs, np.zeros_like(xs)]), np.column_stack([xs, np.full_like(xs, height - 1)]),
                             np.column_stack([np.zeros_like(ys), ys]), np.column_stack([np.full_like(ys, width - 1), ys])])
        q = (outline - self.centre) / self.scale
        u = self._poly(q) if self.direction == "p_to_u" else self._poly_inverse(q)
        rp, ru = np.hypot(q[:, 0], q[:, 1]), np.hypot(u[:, 0], u[:, 1])
        factor = ru[rp > 0] / rp[rp > 0]
        return float(1.0 / max(1.0, float(factor.max()))) if len(factor) else 1.0

    def with_magnification(self, magnification: float) -> "RadialDistortion":
        return RadialDistortion(self.k1, self.k2, self.cx, self.cy, self.scale, magnification, self.direction)

    def as_dict(self) -> dict:
        record = {"k1": self.k1, "k2": self.k2, "cx": self.cx, "cy": self.cy, "scale": self.scale, "magnification": self.magnification}
        if self.direction != "p_to_u":
            record["direction"] = self.direction
        return record


def _fit_homographies(frames: dict, masks: dict, transform) -> dict:
    return {fid: fit_homography(transform(f["src"][masks[fid]]), transform(f["dst"][masks[fid]])) for fid, f in frames.items()}


def fit_joint_radial(frames: dict, width: int, height: int, *, masks: dict, iterations: int = 40,
                     fit_centre: bool = True, direction: str = "p_to_u", check=None) -> dict:
    """Shared (k1, k2[, cx, cy]) + one homography per frame.

    ``check`` (optional callable) is called before every Gauss-Newton
    iteration; it may raise to abort the fit (job cancellation).

    Variable projection: for every trial of the shared parameters the
    per-frame homographies are re-fitted (DLT between undistorted fit
    points), so the damped Gauss-Newton runs on the reduced cost in the
    shared parameters only — the homographies are strongly coupled to the
    distortion and alternating the two converges poorly.  The centre is only
    identifiable once k is non-zero, so k1, k2 are fitted first about the
    image centre and the centre is released afterwards.
    """
    scale = float(np.hypot(width, height) / 2.0)
    theta = np.array([0.0, 0.0, (width - 1) / 2.0, (height - 1) / 2.0])

    def model(t):
        return RadialDistortion(t[0], t[1], t[2], t[3], scale, direction=direction)

    def solve(t):
        d = model(t)
        hs = _fit_homographies(frames, masks, d.apply)
        parts = []
        for fid, f in frames.items():
            m = masks[fid]
            pred_u = apply_matrix(hs[fid], d.apply(f["src"][m]))
            parts.append((d.inverse(pred_u) - f["dst"][m]).reshape(-1))
        return np.concatenate(parts), hs

    r, hs = solve(theta)
    cost = float(np.sum(r ** 2))
    lam = 1e-3
    stages = [[0, 1], [0, 1, 2, 3]] if fit_centre else [[0, 1]]
    for active in stages:
        for _ in range(iterations):
            if check is not None:
                check()
            jac = np.empty((len(r), len(active)))
            for col, j in enumerate(active):
                step = 1e-5 if j < 2 else 2.0
                tp = theta.copy()
                tp[j] += step
                jac[:, col] = (solve(tp)[0] - r) / step
            jtj, jtr = jac.T @ jac, jac.T @ r
            improved = False
            for _ in range(10):
                delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), -jtr)
                candidate = theta.copy()
                candidate[active] += delta
                r_new, hs_new = solve(candidate)
                new_cost = float(np.sum(r_new ** 2))
                if new_cost < cost:
                    relative = (cost - new_cost) / max(cost, 1e-12)
                    theta, cost, r, hs, lam, improved = candidate, new_cost, r_new, hs_new, max(lam / 3.0, 1e-9), True
                    break
                lam *= 10.0
            if not improved or relative < 1e-7:
                break
    d = model(theta)
    return {"kind": "radial", "distortion": d, "H": hs, "fit_cost": cost, "corner_shift_px": d.corner_shift_px(width, height),
            "shared_dof": 4 if fit_centre else 2, "direction": direction}


# --------------------------------------------------------------------------- #
# Span-scaled displacement field                                              #
# --------------------------------------------------------------------------- #

def _field_design(p: np.ndarray, degree: int, width: int, height: int) -> np.ndarray:
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    return _poly_terms(p, degree, float(max(width, height) / 2.0), centre)


def fit_joint_field(frames: dict, width: int, height: int, *, masks: dict, degree: int = 2, rounds: int = 6) -> dict:
    """Shared polynomial field ``v`` (span-scaled) + one homography per frame, alternating linear fits."""
    terms = (degree + 1) * (degree + 2) // 2
    coeff = np.zeros((terms, 2))
    hs = _fit_homographies(frames, masks, lambda p: p)
    cost = None
    for _ in range(rounds):
        rows, rhs = [], []
        for fid, f in frames.items():
            m = masks[fid]
            base = apply_matrix(hs[fid], f["src"][m])
            rows.append(f["span"] * _field_design(base, degree, width, height))
            rhs.append(f["dst"][m] - base)
        design, target = np.vstack(rows), np.vstack(rhs)
        coeff = np.linalg.lstsq(design, target, rcond=None)[0]
        new_hs = {}
        for fid, f in frames.items():
            m = masks[fid]
            base = apply_matrix(hs[fid], f["src"][m])
            corrected = f["dst"][m] - f["span"] * (_field_design(base, degree, width, height) @ coeff)
            new_hs[fid] = fit_homography(f["src"][m], corrected)
        hs = new_hs
        new_cost = 0.0
        for fid, f in frames.items():
            m = masks[fid]
            base = apply_matrix(hs[fid], f["src"][m])
            new_cost += float(np.sum((base + f["span"] * (_field_design(base, degree, width, height) @ coeff) - f["dst"][m]) ** 2))
        if cost is not None and new_cost >= cost * (1 - 1e-9):
            cost = min(cost, new_cost)
            break
        cost = new_cost
    return {"kind": "field", "degree": degree, "coefficients": coeff, "H": hs, "fit_cost": cost, "shared_dof": 2 * terms}


def field_at(model: dict, xy, width: int, height: int) -> np.ndarray:
    """Displacement per unit span at reference positions ``xy``."""
    return _field_design(np.asarray(xy, dtype=np.float64).reshape(-1, 2), model["degree"], width, height) @ model["coefficients"]


# --------------------------------------------------------------------------- #
# Prediction and evaluation                                                   #
# --------------------------------------------------------------------------- #

def predict(model: dict, frame: dict, h, width: int, height: int) -> np.ndarray:
    """Reference-pixel prediction for every pair of ``frame`` under ``model`` with homography ``h``."""
    if model["kind"] == "baseline":
        return apply_matrix(h, frame["src"])
    if model["kind"] == "radial":
        d = model["distortion"]
        return d.inverse(apply_matrix(h, d.apply(frame["src"])))
    if model["kind"] == "field":
        base = apply_matrix(h, frame["src"])
        return base + frame["span"] * field_at(model, base, width, height)
    raise ValueError(model["kind"])


def grid_signed_means(predicted: np.ndarray, dst: np.ndarray, mask: np.ndarray, width: int, height: int,
                      *, columns: int = 4, rows: int = 3) -> dict:
    """Mean signed residual (predicted - dst) per grid cell over the masked points."""
    d = dst[mask]
    r = (predicted - dst)[mask]
    col = np.clip((d[:, 0] * columns / width).astype(int), 0, columns - 1)
    row = np.clip((d[:, 1] * rows / height).astype(int), 0, rows - 1)
    cells = {}
    for c in range(columns):
        for rr in range(rows):
            sel = (col == c) & (row == rr)
            cells[f"xcol_{c}/yrow_{rr}"] = {"count": int(sel.sum()),
                                            "mean_dx_px": float(r[sel, 0].mean()) if sel.any() else None,
                                            "mean_dy_px": float(r[sel, 1].mean()) if sel.any() else None,
                                            "rms_px": float(np.sqrt(np.mean(np.hypot(*r[sel].T) ** 2))) if sel.any() else None}
    return cells


def evaluate_frames(model: dict, frames: dict, masks: dict, width: int, height: int, hs: dict | None = None) -> dict:
    """Per-frame residual statistics (fit / validation, by region) and validation grid means."""
    out = {}
    hs = hs or model["H"]
    for fid, f in frames.items():
        pred = predict(model, f, hs[fid], width, height)
        out[fid] = {"span": f["span"], "count": int(len(f["src"])),
                    "residuals": evaluate_model(pred, f["dst"], masks[fid], width, height),
                    "grid_validation": grid_signed_means(pred, f["dst"], ~masks[fid], width, height)}
    return out


def _corner_table(evaluation: dict) -> dict:
    return {fid: e["residuals"]["validation"]["corner"]["rms_px"] for fid, e in evaluation.items()}


def transfer_test(kind: str, frames: dict, fit_ids: list[str], width: int, height: int, *, masks: dict, degree: int = 2) -> dict:
    """Fit the shared term on ``fit_ids`` only; evaluate every other frame with a fresh per-frame homography."""
    subset = {fid: frames[fid] for fid in fit_ids}
    if kind == "radial":
        model = fit_joint_radial(subset, width, height, masks=masks)
    else:
        model = fit_joint_field(subset, width, height, masks=masks, degree=degree)
    held = {fid: f for fid, f in frames.items() if fid not in fit_ids}
    hs = {}
    for fid, f in held.items():
        m = masks[fid]
        if kind == "radial":
            d = model["distortion"]
            hs[fid] = fit_homography(d.apply(f["src"][m]), d.apply(f["dst"][m]))
        else:
            base_h = fit_homography(f["src"][m], f["dst"][m])
            for _ in range(3):
                base = apply_matrix(base_h, f["src"][m])
                corrected = f["dst"][m] - f["span"] * field_at(model, base, width, height)
                base_h = fit_homography(f["src"][m], corrected)
            hs[fid] = base_h
    evaluation = evaluate_frames(model, held, masks, width, height, hs=hs)
    return {"kind": kind, "fit_frames": list(fit_ids), "held_out_frames": list(held), "shared_dof": model["shared_dof"],
            "held_out_validation_corner_rms_px": _corner_table(evaluation),
            "held_out_validation_overall_rms_px": {fid: e["residuals"]["validation"]["overall"]["rms_px"] for fid, e in evaluation.items()}}


def analyse(frames: dict, width: int, height: int, *, degrees=(1, 2), columns: int = 4, rows: int = 3) -> dict:
    """Baseline vs shared terms on every frame, plus transfer tests across the reference and across spans."""
    masks = {fid: stratified_split(f["dst"], width, height, columns=columns, rows=rows) for fid, f in frames.items()}
    results = {"distortion_version": DISTORTION_VERSION, "width": width, "height": height, "frame_count": len(frames),
               "spans": {fid: f["span"] for fid, f in frames.items()},
               "split": {"columns": columns, "rows": rows, "rule": "per grid cell of the reference positions, alternate fit/validation"},
               "models": {}}
    siril = {"kind": "baseline", "H": {fid: f["H_siril"] for fid, f in frames.items()}}
    results["models"]["siril_H_as_is"] = {"shared_dof": 0, "independent_validation": False,
                                          "frames": evaluate_frames(siril, frames, masks, width, height)}
    baseline = {"kind": "baseline", "H": _fit_homographies(frames, masks, lambda p: p)}
    results["models"]["homography_per_frame"] = {"shared_dof": 0, "independent_validation": True,
                                                 "frames": evaluate_frames(baseline, frames, masks, width, height)}
    radial = fit_joint_radial(frames, width, height, masks=masks)
    results["models"]["homography_plus_shared_radial"] = {
        "shared_dof": 4, "independent_validation": True, "parameters": radial["distortion"].as_dict(),
        "corner_shift_px": radial["corner_shift_px"], "frames": evaluate_frames(radial, frames, masks, width, height),
        "H_undistorted": {fid: np.asarray(h).tolist() for fid, h in radial["H"].items()}}
    for degree in degrees:
        field = fit_joint_field(frames, width, height, masks=masks, degree=degree)
        probes = np.array([[0, 0], [width - 1, 0], [(width - 1) / 2, (height - 1) / 2], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
        results["models"][f"homography_plus_shared_field_deg{degree}"] = {
            "shared_dof": field["shared_dof"], "independent_validation": True,
            "field_per_unit_span_px": {name: v.tolist() for name, v in zip(("top_left", "top_right", "centre", "bottom_left", "bottom_right"),
                                                                            field_at(field, probes, width, height))},
            "coefficients": field["coefficients"].tolist(), "frames": evaluate_frames(field, frames, masks, width, height)}
    # transfer tests: one side of the reference predicts the other; short spans predict long spans
    ids = list(frames)
    negative = [fid for fid in ids if frames[fid]["span"] < 0]
    positive = [fid for fid in ids if frames[fid]["span"] > 0]
    short = sorted(ids, key=lambda fid: abs(frames[fid]["span"]))[: max(3, len(ids) // 2)]
    results["transfer"] = {}
    for name, fit_ids in (("fit_negative_span_predict_positive", negative), ("fit_positive_span_predict_negative", positive),
                          ("fit_short_spans_predict_long", short)):
        if len(fit_ids) < 3 or len(fit_ids) == len(ids):
            results["transfer"][name] = {"skipped": "not enough frames on one side"}
            continue
        results["transfer"][name] = {"radial": transfer_test("radial", frames, fit_ids, width, height, masks=masks),
                                     "field_deg2": transfer_test("field", frames, fit_ids, width, height, masks=masks, degree=2),
                                     "baseline_corner_rms_px": {fid: results["models"]["homography_per_frame"]["frames"][fid]["residuals"]["validation"]["corner"]["rms_px"]
                                                                for fid in ids if fid not in fit_ids}}
    results["caveats"] = [
        "correspondences come through Siril's H within the association radius; residuals beyond it are invisible",
        "validation stars are independent of the fits made here, not of Siril's own H",
        "the shared terms bound what a term of that family explains on these frames; the transfer tests are the predictive claim",
        "spans are image counts; with a uniform cadence they are proportional to time and to sky displacement",
    ]
    return results


# --------------------------------------------------------------------------- #
# Re-association under the fitted model (uncensored evaluation)                #
# --------------------------------------------------------------------------- #

def mutual_nearest(predicted: np.ndarray, reference: np.ndarray, radius: float) -> np.ndarray:
    """Index pairs (i_pred, j_ref) that are mutual nearest neighbours within ``radius``.

    Bit for bit the all-pairs rule ``_mutual_nearest_dense`` (P1a G-5): a pair that rule can accept has every competitor
    within ``radius``, so candidates from the 3x3 block of cells of edge ``1.0625 * radius`` (at least 1 px) decide it;
    distances carry the dense bits and ties go to the lowest index like ``argmin``.  Non-finite or out-of-range
    coordinates, other radius types, clustered input and non-default underflow handling run the dense rule itself.
    """
    if len(predicted) == 0 or len(reference) == 0:
        return np.zeros((0, 2), dtype=int)
    grid = grid_threshold(radius)
    candidates = None
    if grid is not None and grid_points_ok(predicted) and grid_points_ok(reference):
        candidates = cell_pairs(predicted, reference, grid[1], pair_budget(len(predicted), len(reference)))
    if candidates is None:
        return _mutual_nearest_dense(predicted, reference, radius)
    rr = grid[0]
    ci, cj = candidates
    d2 = pair_d2(predicted, reference, ci, cj)
    nearest_ref, nearest_d2 = first_min(ci, cj, d2, len(predicted))
    nearest_pred, _ = first_min(cj, ci, d2, len(reference))
    i = np.arange(len(predicted))
    ok = nearest_d2 <= rr
    ok[ok] = nearest_pred[nearest_ref[ok]] == i[ok]
    return np.column_stack([i[ok], nearest_ref[ok]])


def _mutual_nearest_dense(predicted: np.ndarray, reference: np.ndarray, radius: float) -> np.ndarray:
    """The all-pairs rule of DISTORTION_VERSION 1.3.1, verbatim (non-empty inputs); also the oracle of G-5."""
    d2 = ((predicted[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    nearest_ref = d2.argmin(axis=1)
    nearest_pred = d2.argmin(axis=0)
    i = np.arange(len(predicted))
    ok = (nearest_pred[nearest_ref] == i) & (d2[i, nearest_ref] <= radius * radius)
    return np.column_stack([i[ok], nearest_ref[ok]])


def extrapolate_homography(known: dict, target_span: int, *, neighbours: int = 6, degree: int = 2) -> np.ndarray:
    """Predict a frame's (undistorted-space) homography from the nearest registered frames, element-wise in span.

    ``known`` maps span -> 3x3 matrix.  Sky motion is smooth in time, so each
    normalised matrix element is fitted with a low-order polynomial in span
    over the ``neighbours`` nearest spans and evaluated at ``target_span``.
    """
    spans = sorted(known, key=lambda s: abs(s - target_span))[:neighbours]
    if len(spans) < 2:
        raise ValueError("need at least two registered frames to extrapolate")
    degree = min(degree, len(spans) - 1)
    x = np.array(spans, dtype=np.float64)
    x0 = x.mean()
    design = np.vander((x - x0), degree + 1, increasing=True)
    mats = np.array([np.asarray(known[s], dtype=np.float64) / np.asarray(known[s], dtype=np.float64)[2, 2] for s in spans])
    coeff = np.linalg.lstsq(design, mats.reshape(len(spans), 9), rcond=None)[0]
    h = (np.vander(np.array([target_span - x0]), degree + 1, increasing=True) @ coeff).reshape(3, 3)
    return h / h[2, 2]


def reassociate_frame(distortion: RadialDistortion, h_init, reference_xy, frame_xy, width: int, height: int, *,
                      radii=(20.0, 10.0, 5.0, 3.0), min_pairs: int = 30, columns: int = 4, rows: int = 3) -> dict:
    """Associate *full* star lists under the shared distortion with a refined per-frame homography.

    Starting from ``h_init`` (undistorted space), predict every frame star's
    reference pixel position, pair mutual nearest neighbours within a
    shrinking radius, and refit the homography on the pairs each time.  The
    final pairs are split with the stratified rule and the validation half is
    evaluated, so the reported residuals are not the fit's own.
    """
    ref = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    src = np.asarray(frame_xy, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(h_init, dtype=np.float64)
    pairs = np.zeros((0, 2), dtype=int)
    history = []
    for radius in radii:
        pred = distortion.inverse(apply_matrix(h, distortion.apply(src)))
        pairs = mutual_nearest(pred, ref, radius)
        history.append({"radius_px": float(radius), "pairs": int(len(pairs))})
        if len(pairs) < min_pairs:
            return {"status": "too_few_pairs", "history": history, "matched": int(len(pairs)), "frame_count": int(len(src)),
                    "matched_fraction": float(len(pairs) / max(1, len(src)))}
        h = fit_homography(distortion.apply(src[pairs[:, 0]]), distortion.apply(ref[pairs[:, 1]]))
    s_pairs, d_pairs = src[pairs[:, 0]], ref[pairs[:, 1]]
    mask = stratified_split(d_pairs, width, height, columns=columns, rows=rows)
    h_fit = fit_homography(distortion.apply(s_pairs[mask]), distortion.apply(d_pairs[mask]))
    pred = distortion.inverse(apply_matrix(h_fit, distortion.apply(s_pairs)))
    return {"status": "associated", "history": history, "matched": int(len(pairs)), "frame_count": int(len(src)),
            "reference_count": int(len(ref)), "matched_fraction": float(len(pairs) / max(1, len(src))),
            "H_undistorted": h_fit.tolist(), "residuals": evaluate_model(pred, d_pairs, mask, width, height),
            "grid_validation": grid_signed_means(pred, d_pairs, ~mask, width, height, columns=columns, rows=rows)}


def reassociate_sequence(distortion: RadialDistortion, h_by_frame: dict, spans: dict, star_lists_xy: dict, reference_id: str,
                         width: int, height: int, *, radii=(20.0, 10.0, 5.0, 3.0), neighbours: int = 6, check=None) -> dict:
    """Re-associate every frame's full star list; frames without a homography get one extrapolated from neighbours.

    ``check`` (optional callable) is called before every frame; it may raise to abort (job cancellation).
    """
    ref_xy = star_lists_xy[reference_id]
    known = {spans[fid]: np.asarray(h, dtype=np.float64) for fid, h in h_by_frame.items() if fid in spans}
    out = {}
    for fid, xy in star_lists_xy.items():
        if fid == reference_id:
            continue
        if check is not None:
            check()
        span = spans.get(fid)
        if span is None:
            span = span_of(fid, reference_id)
        if fid in h_by_frame:
            h0, origin = np.asarray(h_by_frame[fid], dtype=np.float64), "joint_fit"
        else:
            h0, origin = extrapolate_homography(known, span, neighbours=neighbours), "extrapolated_from_neighbours"
        entry = reassociate_frame(distortion, h0, ref_xy, xy, width, height, radii=radii)
        entry.update({"span": int(span), "initial_homography": origin})
        out[fid] = entry
    return out
