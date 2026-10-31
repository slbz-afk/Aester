"""Adapter from Siril registration products to the registration diagnostics core.

Inputs are the *original* sequence file (``light_.seq`` with per-image
homographies) and Siril's cached star lists.  Outputs are documents for
``registration_diagnostics.analyze_sequence`` plus adapter-level summaries.

Three things are kept explicit and separate:

1. **Coordinate hypothesis.**  Siril's star coordinates and homographies live
   in Siril's own frame.  The diagnostics core needs TOP-DOWN, zero-based
   pixel-centre coordinates with column-vector source->reference matrices.  The
   conversion is a declared :class:`CoordinateHypothesis`; whether it is
   *verified* comes from the synthetic Siril experiment
   (``scripts/s2_siril_coordinate_proof.py``), never from fitting real data.
   Residual magnitudes and centre-versus-corner comparisons do not depend on
   the vertical flip or a constant offset; the top/bottom labels and any
   half-pixel offset do, so every report carries the hypothesis and its
   verification state.

2. **Star association.**  Siril's cached lists carry no cross-frame star
   identities.  Correspondences are built by projecting each frame star with
   the frame's homography and taking the mutual nearest reference star within
   a radius.  This *is* a matching step with a known bias: only stars that
   land within the radius are evaluated, so the association radius, the
   matched fraction and the ambiguity counts are always reported next to the
   residuals.

3. **Roles.**  Cached lists are Siril's own registration candidates; they may
   have been used to fit the homography.  Every associated star is therefore
   declared ``unknown``.  Nothing here ever declares ``holdout``.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from . import registration_diagnostics as diagnostics
from . import siril_sequence
from .registration_models import cell_pairs, first_min, grid_points_ok, grid_threshold, pair_budget, pair_d2

__all__ = ["ADAPTER_VERSION", "DEFAULT_HYPOTHESIS", "CoordinateHypothesis", "HYPOTHESES", "convert_point", "native_point", "to_siril_matrix",
           "convert_matrix", "associate_stars", "build_document", "run_sequence_diagnostics",
           "pooled_summary", "grid_summary", "fixed_position_screen"]

ADAPTER_VERSION = "0.2.2"
DEFAULT_HYPOTHESIS = "siril_1_4_4_proved_20260917"


@dataclasses.dataclass(frozen=True)
class CoordinateHypothesis:
    """How Siril coordinates map onto the diagnostics convention.

    ``x_td = x_siril + x_offset``; ``y_td = y_siril + y_offset`` when
    ``row_order`` is ``"TOP-DOWN"`` and ``y_td = (height - 1) - (y_siril +
    y_offset)`` when it is ``"BOTTOM-UP"``.  ``matrix_direction`` says what
    Siril's stored H means: ``frame_to_reference`` (documented) or
    ``reference_to_frame`` (inverted before use).
    """

    name: str
    row_order: str
    x_offset: float
    y_offset: float
    matrix_direction: str
    verified: bool
    basis: str

    def __post_init__(self):
        if self.row_order not in ("TOP-DOWN", "BOTTOM-UP"):
            raise ValueError("row_order must be TOP-DOWN or BOTTOM-UP")
        if self.matrix_direction not in ("frame_to_reference", "reference_to_frame"):
            raise ValueError("matrix_direction must be frame_to_reference or reference_to_frame")
        for value in (self.x_offset, self.y_offset):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("offsets must be finite numbers")

    def affine(self, height: int) -> np.ndarray:
        """3x3 matrix T with [x_td, y_td, 1] = T [x_siril, y_siril, 1]."""
        if self.row_order == "TOP-DOWN":
            return np.array([[1.0, 0.0, self.x_offset], [0.0, 1.0, self.y_offset], [0.0, 0.0, 1.0]])
        return np.array([[1.0, 0.0, self.x_offset], [0.0, -1.0, (height - 1) - self.y_offset], [0.0, 0.0, 1.0]])

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ``siril_1_4_4_proved_20260917`` is the convention established by the real
# Siril 1.4.4 run of scripts/s2_siril_coordinate_proof.py on 2026-09-17
# (Windows 11, product venv): star/matrix y grows downwards like the diagnostics
# convention, coordinates carry a +0.5 px pixel-corner origin (Siril (0.5, 0.5)
# is the centre of the top-left pixel), H maps frame -> reference.  The two
# ``*_unverified`` entries are kept for comparison runs only.
HYPOTHESES = {
    "siril_1_4_4_proved_20260917": CoordinateHypothesis(
        name="siril_1_4_4_proved_20260917", row_order="TOP-DOWN", x_offset=-0.5, y_offset=-0.5,
        matrix_direction="frame_to_reference", verified=True,
        basis="real Siril 1.4.4 synthetic run outputs/fable-s2-20260917/coordinate-proof-siril-20260917T092505Z-970195 "
              "(coordinate_proof.json sha256 31e9eb1451edc48d, light_.seq sha256 c3f81f06870cd091): 168/170 stars matched "
              "under TOP-DOWN with MAD 0.002 px (BOTTOM-UP 3/170), offsets -0.5/-0.5 exact, direction errors <=0.026 px vs >=16 px"),
    "siril_native_as_topdown_unverified": CoordinateHypothesis(
        name="siril_native_as_topdown_unverified", row_order="TOP-DOWN", x_offset=0.0, y_offset=0.0,
        matrix_direction="frame_to_reference", verified=False,
        basis="sirilpy Homography docstring (frame onto reference); origin and row order not documented"),
    "siril_native_as_bottomup_unverified": CoordinateHypothesis(
        name="siril_native_as_bottomup_unverified", row_order="BOTTOM-UP", x_offset=0.0, y_offset=0.0,
        matrix_direction="frame_to_reference", verified=False,
        basis="alternative row order; FITS files are stored BOTTOM-UP, which by itself proves nothing about API coordinates"),
}


def convert_point(hypothesis: CoordinateHypothesis, x: float, y: float, height: int) -> tuple[float, float]:
    t = hypothesis.affine(height)
    v = t @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


def convert_matrix(hypothesis: CoordinateHypothesis, h_siril, source_height: int, reference_height: int) -> np.ndarray:
    """Siril H -> diagnostics source->reference matrix under the hypothesis."""
    h = np.asarray(h_siril, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("H must be a finite 3x3 matrix")
    if hypothesis.matrix_direction == "reference_to_frame":
        h = np.linalg.inv(h)
    t_ref = hypothesis.affine(reference_height)
    t_src = hypothesis.affine(source_height)
    return t_ref @ h @ np.linalg.inv(t_src)


def to_siril_matrix(hypothesis: CoordinateHypothesis, h_source_to_reference, source_height: int, reference_height: int) -> np.ndarray:
    """Diagnostics source->reference matrix (top-down pixel centres) -> the H Siril stores; inverse of ``convert_matrix``.

    Used when the product writes its own registration into the sequence file
    (``siril_sequence.write_registration``).  The result is normalised so that
    ``H[2, 2] == 1`` like Siril's own lines.
    """
    h = np.asarray(h_source_to_reference, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("H must be a finite 3x3 matrix")
    t_ref = hypothesis.affine(reference_height)
    t_src = hypothesis.affine(source_height)
    h_siril = np.linalg.inv(t_ref) @ h @ t_src
    if hypothesis.matrix_direction == "reference_to_frame":
        h_siril = np.linalg.inv(h_siril)
    if abs(h_siril[2, 2]) < 1e-12:
        raise ValueError("degenerate homography")
    return h_siril / h_siril[2, 2]


def _project(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    ones = np.ones((xy.shape[0], 1))
    v = np.hstack([xy, ones]) @ h.T
    return v[:, :2] / v[:, 2:3]


def associate_stars(reference_stars, frame_stars, h_frame_to_reference, *, radius: float,
                    mutual: bool = True) -> dict:
    """Associate frame stars with reference stars through the given homography.

    Both star lists are sequences of dicts with ``x``/``y`` (same coordinate
    frame as ``h``).  Returns index pairs plus bookkeeping needed to judge the
    association itself: matched fraction, ambiguous candidates, unmatched.
    The candidates come from an exact cell search (P1a G-5; identical to the
    all-pairs rule ``_associate_dense``, which runs outside its domain).
    """
    if not (isinstance(radius, (int, float)) and math.isfinite(radius) and radius > 0):
        raise ValueError("radius must be a positive finite number")
    h = np.asarray(h_frame_to_reference, dtype=np.float64)
    ref = np.array([[s["x"], s["y"]] for s in reference_stars], dtype=np.float64).reshape(-1, 2)
    src = np.array([[s["x"], s["y"]] for s in frame_stars], dtype=np.float64).reshape(-1, 2)
    result = {"radius_px": float(radius), "mutual": bool(mutual), "reference_count": int(len(ref)),
              "frame_count": int(len(src)), "pairs": [], "unmatched_frame": [], "unmatched_reference": [],
              "ambiguous_frame": [], "matched_fraction_of_frame": 0.0}
    if len(ref) == 0 or len(src) == 0:
        result["unmatched_frame"] = list(range(len(src)))
        result["unmatched_reference"] = list(range(len(ref)))
        return result
    projected = _project(h, src)
    decided = None
    grid = grid_threshold(radius)
    if grid is not None and grid_points_ok(projected) and grid_points_ok(ref):
        decided = _associate_grid(projected, ref, radius, mutual, *grid)       # None: clustered beyond the budget
    pairs, ambiguous, matched_ref = decided if decided is not None else _associate_dense(projected, ref, radius, mutual)
    matched_src = {p["frame_index"] for p in pairs}
    ambiguous_set = set(ambiguous)
    result["pairs"] = pairs
    result["ambiguous_frame"] = ambiguous
    result["unmatched_frame"] = [i for i in range(len(src)) if i not in matched_src and i not in ambiguous_set]
    result["unmatched_reference"] = [j for j in range(len(ref)) if j not in matched_ref]
    result["matched_fraction_of_frame"] = len(pairs) / len(src)
    return result


def _associate_grid(projected, ref, radius, mutual, rr, cell):
    """The dense loop's decisions from the cell candidates (same rows, order and values); None over the pair budget."""
    candidates = cell_pairs(projected, ref, cell, pair_budget(len(projected), len(ref)))
    if candidates is None:
        return None
    ci, cj = candidates
    d2 = pair_d2(projected, ref, ci, cj)
    nearest_ref, nearest_d2 = first_min(ci, cj, d2, len(projected))
    nearest_src, _ = first_min(cj, ci, d2, len(ref))
    nearest_d = np.sqrt(nearest_d2)                                  # inf for rows without a candidate
    within = np.bincount(ci[d2 <= rr], minlength=len(projected))
    pairs, ambiguous, matched_ref = [], [], set()
    for i in np.flatnonzero(~(nearest_d > radius)).tolist():        # exactly the rows the dense loop does not skip
        j = int(nearest_ref[i])
        if int(within[i]) > 1:
            ambiguous.append(i)
            continue
        if mutual and int(nearest_src[j]) != i:
            continue
        if j in matched_ref:
            continue
        matched_ref.add(j)
        pairs.append({"frame_index": i, "reference_index": j, "distance_px": float(nearest_d[i]),
                      "projected_xy": [float(projected[i, 0]), float(projected[i, 1])]})
    return pairs, ambiguous, matched_ref


def _associate_dense(projected, ref, radius, mutual):
    """The all-pairs association of ADAPTER_VERSION 0.2.2, verbatim; also the oracle of G-5."""
    # squared distances, frame x reference
    d2 = ((projected[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    nearest_ref = d2.argmin(axis=1)
    nearest_d = np.sqrt(d2[np.arange(len(projected)), nearest_ref])
    nearest_src = d2.argmin(axis=0)
    pairs = []
    ambiguous = []
    matched_ref = set()
    for i in range(len(projected)):
        j = int(nearest_ref[i])
        if nearest_d[i] > radius:
            continue
        # a second candidate within the radius makes the association ambiguous
        row = d2[i]
        within = int(np.count_nonzero(row <= radius * radius))
        if within > 1:
            ambiguous.append(i)
            continue
        if mutual and int(nearest_src[j]) != i:
            continue
        if j in matched_ref:
            continue
        matched_ref.add(j)
        pairs.append({"frame_index": i, "reference_index": j, "distance_px": float(nearest_d[i]),
                      "projected_xy": [float(projected[i, 0]), float(projected[i, 1])]})
    return pairs, ambiguous, matched_ref


def build_document(*, sequence: dict, star_lists: dict, width: int, height: int,
                   hypothesis: CoordinateHypothesis, positions, layer: int, radius: float,
                   elapsed_seconds=None, space_prefix: str = "siril", reference_position: int | None = None) -> tuple[dict, dict]:
    """Build a diagnostics document for the given image positions.

    ``star_lists`` maps image stem -> parsed star list (``siril_sequence.read_lst``).
    ``positions`` are 0-based sequence positions to analyse (the reference
    position is never a frame of its own).  ``elapsed_seconds`` maps stem ->
    seconds or is None (times are unknown unless the caller provides them).
    ``reference_position`` selects another image as the comparison reference:
    frames are then mapped through ``inv(H_ref) @ H_frame`` (frame -> Siril
    reference -> chosen image), which lets two frames whose star lists were
    detected the same way be compared directly.  Returns (document,
    association records per stem).
    """
    header = sequence["header"]
    images = sequence["images"]
    regdata = sequence["regdata"].get(layer)
    if regdata is None:
        raise ValueError(f"sequence has no registration data for layer {layer}")
    if sequence["all_identity_per_layer"].get(layer):
        raise ValueError("all homographies are identity: this looks like a registered output sequence, "
                         "use the original sequence file")
    seq_ref_position = header["reference_image"]
    seq_ref_h = np.asarray(regdata[seq_ref_position]["H"])
    if not np.allclose(seq_ref_h, np.eye(3), atol=1e-9):
        raise ValueError("reference image homography is not identity; sequence and reference disagree")
    ref_position = seq_ref_position if reference_position is None else int(reference_position)
    if not 0 <= ref_position < len(images):
        raise ValueError(f"reference_position {ref_position} outside the sequence")
    ref_stem = images[ref_position]["stem"]
    if ref_stem not in star_lists:
        raise ValueError(f"star list for the reference image {ref_stem} is required")
    ref_stars = star_lists[ref_stem]["stars"]
    # Siril H of the chosen reference (identity for the sequence reference); frames are chained through it.
    ref_chain = np.asarray(regdata[ref_position]["H"], dtype=np.float64)
    if hypothesis.matrix_direction == "reference_to_frame":
        ref_chain = np.linalg.inv(ref_chain)
    ref_chain_inv = np.linalg.inv(ref_chain)
    convention = {"row_order": "TOP-DOWN", "origin": "top_left_pixel_center", "units": "pixel"}
    reference = {"frame_id": ref_stem, "width": int(width), "height": int(height),
                 "coordinate_space_id": f"{space_prefix}:{hypothesis.name}:reference:{ref_stem}"}
    frames = []
    associations = {}
    for position in positions:
        if not 0 <= position < len(images):
            raise ValueError(f"position {position} outside the sequence")
        image = images[position]
        stem = image["stem"]
        if position == ref_position:
            raise ValueError("the reference image cannot be analysed as a frame of itself")
        h_siril = np.asarray(regdata[position]["H"], dtype=np.float64)
        if reference_position is not None:
            # compose in the frame_to_reference sense, then re-express in Siril's stored direction
            h_f2r = h_siril if hypothesis.matrix_direction == "frame_to_reference" else np.linalg.inv(h_siril)
            h_f2r = ref_chain_inv @ h_f2r
            h_siril = h_f2r if hypothesis.matrix_direction == "frame_to_reference" else np.linalg.inv(h_f2r)
        elapsed = None if elapsed_seconds is None else elapsed_seconds.get(stem)
        frame = {"frame_id": stem, "width": int(width), "height": int(height),
                 "coordinate_space_id": f"{space_prefix}:{hypothesis.name}:frame:{stem}",
                 "elapsed_seconds": elapsed, "source_to_reference": None, "matches": []}
        if abs(np.linalg.det(h_siril)) < 1e-12 or not np.isfinite(h_siril).all():
            # Siril writes a zero matrix for a frame it could not register (excluded from the sequence)
            frame["source_to_reference"] = None
            frames.append(frame)
            associations[stem] = {"status": "not_registered", "selected": bool(image.get("selected", True))}
            continue
        if stem not in star_lists:
            frame["source_to_reference"] = convert_matrix(hypothesis, h_siril, height, height).tolist()
            frames.append(frame)
            associations[stem] = {"status": "no_star_list"}
            continue
        stars = star_lists[stem]["stars"]
        h_direction = h_siril if hypothesis.matrix_direction == "frame_to_reference" else np.linalg.inv(h_siril)
        association = associate_stars(ref_stars, stars, h_direction, radius=radius)
        association["status"] = "associated"
        matches = []
        for pair in association["pairs"]:
            src = stars[pair["frame_index"]]
            ref = ref_stars[pair["reference_index"]]
            sx, sy = convert_point(hypothesis, src["x"], src["y"], height)
            rx, ry = convert_point(hypothesis, ref["x"], ref["y"], height)
            matches.append({"star_id": f"ref{pair['reference_index'] + 1}<-{stem}#{pair['frame_index'] + 1}",
                            "source_xy": [sx, sy], "reference_xy": [rx, ry], "role": "unknown"})
        frame["source_to_reference"] = convert_matrix(hypothesis, h_siril, height, height).tolist()
        frame["matches"] = matches
        frames.append(frame)
        associations[stem] = association
    document = {"schema_version": 1, "coordinate_convention": convention, "reference": reference, "frames": frames}
    if reference_position is not None:
        document["adapter_note"] = (f"comparison reference is sequence position {ref_position} ({ref_stem}); "
                                    f"frame matrices are chained through the Siril reference {images[seq_ref_position]['stem']}")
    return document, associations


def _pool(distances: list[float]) -> dict:
    if not distances:
        return {"count": 0, "rms_px": None, "median_px": None, "p95_px": None, "max_px": None}
    d = np.asarray(distances, dtype=np.float64)
    return {"count": int(len(d)), "rms_px": float(np.sqrt(np.mean(d * d))), "median_px": float(np.median(d)),
            "p95_px": float(np.percentile(d, 95, method="linear")), "max_px": float(np.max(d))}


def pooled_summary(frame_result: dict, star_flags: dict | None = None) -> dict:
    """Hypothesis-robust view of one succeeded frame result.

    Corners are pooled (top/bottom labels depend on the unverified row order),
    signed mean residuals are given per pooled region, and optionally the
    statistics are repeated for unsaturated stars only (``star_flags`` maps
    star_id -> {"saturated": bool}).
    """
    if frame_result.get("state") != "succeeded":
        return {"available": False, "reason": frame_result.get("error")}
    residuals = frame_result["residuals"]
    groups = {"center": [], "corners_pooled": [], "other": []}
    signed = {key: [] for key in groups}
    unsaturated = {key: [] for key in groups}
    for r in residuals:
        key = "center" if r["region"] == "center" else ("other" if r["region"] == "other" else "corners_pooled")
        groups[key].append(r["distance_px"])
        signed[key].append((r["dx_px"], r["dy_px"]))
        if star_flags is not None and not star_flags.get(r["star_id"], {}).get("saturated", False):
            unsaturated[key].append(r["distance_px"])
    summary = {"available": True, "all_roles_pooled": {}, "unsaturated_only": {} if star_flags is not None else None}
    for key, values in groups.items():
        entry = _pool(values)
        if values:
            arr = np.asarray(signed[key])
            entry["mean_dx_px"] = float(arr[:, 0].mean())
            entry["mean_dy_px"] = float(arr[:, 1].mean())
        else:
            entry["mean_dx_px"] = entry["mean_dy_px"] = None
        summary["all_roles_pooled"][key] = entry
        if star_flags is not None:
            summary["unsaturated_only"][key] = _pool(unsaturated[key])
    summary["all_roles_pooled"]["overall"] = _pool([r["distance_px"] for r in residuals])
    return summary


def native_point(hypothesis: CoordinateHypothesis, x_td: float, y_td: float, height: int) -> tuple[float, float]:
    """Inverse of :func:`convert_point`: diagnostics coordinates back to Siril's frame."""
    v = np.linalg.inv(hypothesis.affine(height)) @ np.array([x_td, y_td, 1.0])
    return float(v[0]), float(v[1])


def grid_summary(frame_result: dict, hypothesis: CoordinateHypothesis, width: int, height: int,
                 *, columns: int = 4, rows: int = 3) -> dict:
    """Residual statistics on a coarse grid labelled in Siril-native coordinates.

    Column bands are unaffected by the row-order question; row bands are
    named ``ynative_0`` (Siril y near 0) upwards so that the labels stay true
    whichever row order the proof establishes.  ``mean_dx_px`` keeps its sign
    under both row orders, ``mean_dy_px`` flips sign with the row order and is
    reported as computed under the given hypothesis.
    """
    if frame_result.get("state") != "succeeded":
        return {"available": False, "reason": frame_result.get("error")}
    cells = {}
    for r in frame_result["residuals"]:
        rx, ry = r["reference_xy"]
        nx, ny = native_point(hypothesis, rx, ry, height)
        col = min(columns - 1, max(0, int(nx * columns / width)))
        row = min(rows - 1, max(0, int(ny * rows / height)))
        cells.setdefault((col, row), []).append(r)
    table = {}
    for (col, row), items in sorted(cells.items()):
        entry = _pool([i["distance_px"] for i in items])
        arr = np.asarray([[i["dx_px"], i["dy_px"]] for i in items])
        entry["mean_dx_px"] = float(arr[:, 0].mean())
        entry["mean_dy_px"] = float(arr[:, 1].mean())
        table[f"xcol_{col}/ynative_{row}"] = entry
    return {"available": True, "columns": columns, "rows": rows, "labels": "xcol_<0..columns-1 from x=0>/ynative_<0..rows-1 from Siril y=0>",
            "dy_sign_note": "mean_dy_px is in diagnostics coordinates under the stated hypothesis; it changes sign if the row order is the other one",
            "cells": table}


def fixed_position_screen(star_lists: dict, expected_shift_px: dict | None = None, *, tolerance_px: float = 1.0) -> dict:
    """Screen for detections that sit at the same *frame* pixel position in two
    lists (hot pixels, fixed pattern, dust): real stars move with the sky.

    ``star_lists`` maps stem -> parsed list; ``expected_shift_px`` maps
    ``"a|b"`` -> expected star displacement between the two frames (from the
    homographies at the frame centre).  A pair is only informative when that
    displacement clearly exceeds ``tolerance_px``; the result says so instead
    of reporting a clean screen.
    """
    stems = list(star_lists)
    pairs = {}
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            pa = np.array([[s["x"], s["y"]] for s in star_lists[a]["stars"]], dtype=np.float64).reshape(-1, 2)
            pb = np.array([[s["x"], s["y"]] for s in star_lists[b]["stars"]], dtype=np.float64).reshape(-1, 2)
            if len(pa) == 0 or len(pb) == 0:
                pairs[f"{a}|{b}"] = {"informative": False, "reason": "empty list", "candidates": []}
                continue
            d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
            ii, jj = np.where(d <= tolerance_px)
            known = expected_shift_px is not None and f"{a}|{b}" in expected_shift_px
            shift = expected_shift_px.get(f"{a}|{b}") if known else None
            unknown_shift = known and shift is None   # e.g. one frame's homography is singular (not registered)
            informative = (not unknown_shift) and (shift is None or shift > 3.0 * tolerance_px)
            pairs[f"{a}|{b}"] = {
                "informative": bool(informative), "expected_shift_px": shift, "tolerance_px": float(tolerance_px),
                "candidate_count": int(len(ii)),
                "candidates": [{"a_index": int(i_), "b_index": int(j_), "xy_a": [float(pa[i_, 0]), float(pa[i_, 1])],
                                "xy_b": [float(pb[j_, 0]), float(pb[j_, 1])], "distance_px": float(d[i_, j_])}
                               for i_, j_ in zip(ii, jj)],
                "reason": None if informative else ("expected displacement unknown: a homography is singular (frame not registered)"
                                                   if unknown_shift else "expected displacement too small to separate fixed pixels from stars")}
    return {"tolerance_px": float(tolerance_px), "pairs": pairs,
            "note": "candidates are same-position detections in frame coordinates; zero candidates on an informative pair "
                    "means no fixed-position detection in these lists, not that the frames have no hot pixels"}


def run_sequence_diagnostics(*, seq_path, lst_paths: dict, width: int, height: int,
                             hypothesis: CoordinateHypothesis, positions, layer: int = 1,
                             radius: float = 6.0, elapsed_seconds=None, reference_position: int | None = None) -> dict:
    """End-to-end: parse, associate, convert, run the diagnostics core, summarise."""
    sequence = siril_sequence.read_seq(seq_path)
    star_lists = {stem: siril_sequence.read_lst(path) for stem, path in lst_paths.items()}
    document, associations = build_document(sequence=sequence, star_lists=star_lists, width=width, height=height,
                                            hypothesis=hypothesis, positions=positions, layer=layer, radius=radius,
                                            elapsed_seconds=elapsed_seconds, reference_position=reference_position)
    report = diagnostics.analyze_sequence(document)
    flags = {}
    for frame in document["frames"]:
        stars = star_lists.get(frame["frame_id"], {}).get("stars", [])
        for match in frame["matches"]:
            frame_index = int(match["star_id"].rsplit("#", 1)[1]) - 1
            flags[match["star_id"]] = {"saturated": bool(stars[frame_index].get("sat", 0))}
    # fixed-position screen (hot pixels) with the smallest expected star displacement
    # (centre and four corners) between each pair of frames, from the homographies
    probes = np.array([[width / 2.0, height / 2.0], [0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]])
    positions_by_stem = {img["stem"]: img["position"] for img in sequence["images"]}
    regdata = sequence["regdata"][layer]

    def _shift(a, b):
        ha = np.asarray(regdata[positions_by_stem[a]]["H"], dtype=np.float64)
        hb = np.asarray(regdata[positions_by_stem[b]]["H"], dtype=np.float64)
        # a frame Siril could not register carries a zero/singular matrix: no expected shift
        if abs(np.linalg.det(ha)) < 1e-12 or abs(np.linalg.det(hb)) < 1e-12:
            return None
        if hypothesis.matrix_direction == "reference_to_frame":
            ha, hb = np.linalg.inv(ha), np.linalg.inv(hb)
        # where the same sky point lands in frame b for each probe pixel of frame a
        moved = _project(np.linalg.inv(hb) @ ha, probes)
        value = float(np.min(np.hypot(*(moved - probes).T)))
        return value if math.isfinite(value) else None

    stems = list(star_lists)
    shifts = {f"{a}|{b}": _shift(a, b) for i, a in enumerate(stems) for b in stems[i + 1:]}
    screen = fixed_position_screen(star_lists, shifts)
    summaries = {}
    for frame_result in report["frames"]:
        stem = frame_result["frame_id"]
        assoc = associations.get(stem, {})
        summaries[stem] = {"association": {k: v for k, v in assoc.items() if k != "pairs"},
                           "pooled": pooled_summary(frame_result, flags),
                           "grid": grid_summary(frame_result, hypothesis, width, height)}
        if assoc.get("status") == "associated":
            summaries[stem]["association"]["ambiguous_count"] = len(assoc["ambiguous_frame"])
            summaries[stem]["association"]["unmatched_frame_count"] = len(assoc["unmatched_frame"])
            summaries[stem]["association"]["unmatched_reference_count"] = len(assoc["unmatched_reference"])
            summaries[stem]["association"]["matched_count"] = len(assoc["pairs"])
            for key in ("ambiguous_frame", "unmatched_frame", "unmatched_reference"):
                summaries[stem]["association"].pop(key, None)
    return {"adapter_version": ADAPTER_VERSION, "hypothesis": hypothesis.as_dict(),
            "comparison_reference": document["reference"]["frame_id"], "adapter_note": document.get("adapter_note"),
            "sequence_source": sequence["source"], "sequence_header": sequence["header"],
            "star_list_sources": {stem: lists["source"] for stem, lists in star_lists.items()},
            "star_list_parameters": {stem: lists["parameters"] for stem, lists in star_lists.items()},
            "layer": layer, "association_radius_px": radius, "roles": "all unknown (Siril registration candidates)",
            "fixed_position_screen": screen,
            "document": document, "report": report, "summaries": summaries,
            "caveats": [
                ("coordinate hypothesis verified by the synthetic Siril proof: region labels and offsets are definitive "
                 "for this Siril version and command sequence" if hypothesis.verified else
                 "coordinate hypothesis NOT verified: vertical region labels and sub-pixel offsets may be wrong until "
                 "the synthetic Siril proof has run"),
                "associations are built through the tested homography itself; residuals larger than the radius are invisible",
                "all stars are Siril registration candidates and may have been used to fit H: no independent precision claim",
            ]}
