"""Refit a frame's homography from full-field star correspondences.

Evidence from the Z8 short group (outputs/fable-s2-20260917, RUN_LOG §8):
Siril's matcher keeps only ~half of the available pairs for frames 15 frames
away from the reference and the resulting H extrapolates poorly to the
corners.  This module builds candidate matrices for one frame and evaluates
them on the *same* reference correspondences:

* ``siril``          Siril's stored H, unchanged (no independence claim);
* ``direct``         homography refitted on the stratified fit half of the
                     reference<->frame pairs (reference list on Siril's
                     16-bit detection);
* ``chained``        homography refitted on frame<->intermediate pairs (two
                     32-bit-detected lists, more and cleaner pairs) and composed
                     with Siril's intermediate->reference H, which for an
                     adjacent frame is as good as a refit.

All coordinates are Siril-native; callers convert with the proved hypothesis
for the diagnostics core or for pixel warping.  Nothing here rejects frames
or edits sequence files.
"""
from __future__ import annotations

import numpy as np

from . import registration_models as models
from .registration_adapter import associate_stars

__all__ = ["REFIT_VERSION", "candidate_matrices", "evaluate_candidates", "refit_frame"]

REFIT_VERSION = "1.0.0"


def _xy(stars) -> np.ndarray:
    return np.array([[s["x"], s["y"]] for s in stars], dtype=np.float64).reshape(-1, 2)


def _pairs(reference_stars, frame_stars, h_frame_to_reference, radius):
    assoc = associate_stars(reference_stars, frame_stars, h_frame_to_reference, radius=radius)
    fi = np.array([p["frame_index"] for p in assoc["pairs"]], dtype=int)
    ri = np.array([p["reference_index"] for p in assoc["pairs"]], dtype=int)
    return assoc, fi, ri


def candidate_matrices(reference_stars, frame_stars, h_frame_siril, width, height, *,
                       intermediate_stars=None, h_intermediate_siril=None, radius: float = 6.0,
                       columns: int = 4, rows: int = 3) -> dict:
    """Build the candidate frame->reference matrices (Siril coordinates).

    Returns ``{name: {"H": 3x3 list, "fit_frame_indices": [...], "detail": {...}}}``.
    ``fit_frame_indices`` are the frame-star indices whose positions entered
    the fit, so an evaluation can exclude them.
    """
    h_siril = np.asarray(h_frame_siril, dtype=np.float64)
    out = {"siril": {"H": h_siril.tolist(), "fit_frame_indices": [], "independent_validation": False,
                     "detail": {"note": "Siril's stored matrix; its own fit used unknown stars"}}}
    # direct refit on reference <-> frame pairs
    assoc, fi, ri = _pairs(reference_stars, frame_stars, h_siril, radius)
    if len(fi) >= 30:
        src, dst = _xy(frame_stars)[fi], _xy(reference_stars)[ri]
        mask = models.stratified_split(src, width, height, columns=columns, rows=rows)
        h_direct = models.fit_homography(src[mask], dst[mask])
        out["direct"] = {"H": h_direct.tolist(), "fit_frame_indices": fi[mask].tolist(), "independent_validation": True,
                         "detail": {"pairs": int(len(fi)), "fit_pairs": int(mask.sum()), "radius_px": radius,
                                    "ambiguous": len(assoc["ambiguous_frame"])}}
    else:
        out["direct"] = {"H": None, "fit_frame_indices": [], "independent_validation": True,
                         "detail": {"pairs": int(len(fi)), "reason": "fewer than 30 correspondences"}}
    # chained refit through an intermediate frame
    if intermediate_stars is not None and h_intermediate_siril is not None:
        h_mid = np.asarray(h_intermediate_siril, dtype=np.float64)
        h_frame_to_mid = np.linalg.inv(h_mid) @ h_siril
        assoc_m, fi_m, mi = _pairs(intermediate_stars, frame_stars, h_frame_to_mid, radius)
        if len(fi_m) >= 30:
            src, dst = _xy(frame_stars)[fi_m], _xy(intermediate_stars)[mi]
            mask_m = models.stratified_split(src, width, height, columns=columns, rows=rows)
            h_f2m = models.fit_homography(src[mask_m], dst[mask_m])
            h_chain = h_mid @ h_f2m
            h_chain = h_chain / h_chain[2, 2]
            out["chained"] = {"H": h_chain.tolist(), "fit_frame_indices": fi_m[mask_m].tolist(), "independent_validation": True,
                              "detail": {"pairs_frame_intermediate": int(len(fi_m)), "fit_pairs": int(mask_m.sum()),
                                         "radius_px": radius, "ambiguous": len(assoc_m["ambiguous_frame"]),
                                         "intermediate_matrix": "Siril H of the intermediate frame, used as-is"}}
        else:
            out["chained"] = {"H": None, "fit_frame_indices": [], "independent_validation": True,
                              "detail": {"pairs_frame_intermediate": int(len(fi_m)), "reason": "fewer than 30 correspondences"}}
    return out


def evaluate_candidates(reference_stars, frame_stars, candidates: dict, h_assoc, width, height, *,
                        radius: float = 6.0) -> dict:
    """Residuals of every candidate on the same reference<->frame pairs.

    Pairs are associated once through ``h_assoc`` (Siril's H, so that all
    candidates see identical pairs).  For each candidate two views are
    reported: ``all_pairs`` and ``excluding_fit_inputs`` (pairs whose frame
    star entered that candidate's fit are left out).
    """
    assoc, fi, ri = _pairs(reference_stars, frame_stars, h_assoc, radius)
    src, dst = _xy(frame_stars)[fi], _xy(reference_stars)[ri]
    regions = np.array([models.region_of(x, y, width, height) for x, y in dst])
    result = {"pairs": int(len(fi)), "association_radius_px": radius, "ambiguous": len(assoc["ambiguous_frame"]),
              "candidates": {}}
    for name, cand in candidates.items():
        if cand.get("H") is None:
            result["candidates"][name] = {"available": False, "reason": cand["detail"].get("reason")}
            continue
        pred = models.apply_matrix(np.asarray(cand["H"]), src)
        d = pred - dst
        dist = np.hypot(d[:, 0], d[:, 1])
        used = np.isin(fi, np.asarray(cand.get("fit_frame_indices", []), dtype=int))
        views = {}
        for view, sel in (("all_pairs", np.ones(len(fi), bool)), ("excluding_fit_inputs", ~used)):
            entry = {"count": int(sel.sum())}
            for region in ("overall", "center", "corner", "other"):
                m = sel if region == "overall" else (sel & (regions == region))
                if m.any():
                    entry[region] = {"count": int(m.sum()), "rms_px": float(np.sqrt(np.mean(dist[m] ** 2))),
                                     "median_px": float(np.median(dist[m])), "max_px": float(dist[m].max()),
                                     "mean_dx_px": float(d[m, 0].mean()), "mean_dy_px": float(d[m, 1].mean())}
                else:
                    entry[region] = {"count": 0}
            views[view] = entry
        result["candidates"][name] = {"available": True, "independent_validation": cand.get("independent_validation", False),
                                      "fit_inputs_among_pairs": int(used.sum()), "views": views}
    return result


def refit_frame(sequence: dict, star_lists: dict, frame_position: int, *, intermediate_position=None,
                width: int, height: int, layer: int = 1, radius: float = 6.0) -> dict:
    """Convenience: candidates + evaluation for one sequence position."""
    header = sequence["header"]
    images = sequence["images"]
    regdata = sequence["regdata"][layer]
    ref_pos = header["reference_image"]
    ref_stem, frame_stem = images[ref_pos]["stem"], images[frame_position]["stem"]
    ref_stars, frame_stars = star_lists[ref_stem]["stars"], star_lists[frame_stem]["stars"]
    h_frame = np.asarray(regdata[frame_position]["H"], dtype=np.float64)
    mid_stars = mid_h = None
    mid_stem = None
    if intermediate_position is not None:
        mid_stem = images[intermediate_position]["stem"]
        mid_stars = star_lists[mid_stem]["stars"]
        mid_h = np.asarray(regdata[intermediate_position]["H"], dtype=np.float64)
    candidates = candidate_matrices(ref_stars, frame_stars, h_frame, width, height, intermediate_stars=mid_stars,
                                    h_intermediate_siril=mid_h, radius=radius)
    evaluation = evaluate_candidates(ref_stars, frame_stars, candidates, h_frame, width, height, radius=radius)
    return {"refit_version": REFIT_VERSION, "reference": ref_stem, "frame": frame_stem, "intermediate": mid_stem,
            "candidates": candidates, "evaluation": evaluation,
            "caveats": ["pairs are associated through Siril's own H within the radius; residuals beyond it are invisible",
                        "'excluding_fit_inputs' removes frame stars that entered the candidate's fit; the reference "
                        "positions and Siril's own fit remain shared",
                        "the reference star list was detected by Siril on the 16-bit read of the reference frame"]}
