"""Metrics for the S3 horizon-band acceptance (RUN_LOG §25): does the sky stack stay clean above the ground?

All arrays are TOP-DOWN.  ``mask`` is the job's sky mask as 0..1 floats,
``master``/``frame`` are (H, W, 3) linear RGB, ``coverage`` is the sampled
coverage map (``sky_validity.coverage_map``) with its ``step``.

* ``colour_band``: the master's R/G and B/G above the ground, relative to
  the reference frame's own colour at the same place and normalised at the
  top of the range — a "ratio of ratios" that cancels the sky's own colour
  gradient and the night's colour drift; ground contamination shows as an
  excess growing towards the horizon (before the fix: 1.09–1.27 at the edge).
* ``master_coverage_consistency``: the master must be exactly zero where no
  frame observed the sky and non-zero where at least one did.
* ``seam``: full-resolution same-location composition against an actual
  effective weight; missing evidence is explicit. Neighbour brightness
  ratios remain observations, not proof of an erroneous seam.
"""
from __future__ import annotations

import numpy as np

from .seam_validation import seam, judge_seam

HORIZON_ACCEPTANCE_VERSION = "2.0.0"
CRITERIA = {
    "H2": {"consistency_min": 0.99},
    # H3 (1.0.0, kept as a diagnostic): master colour relative to the reference frame.  It cannot separate ground
    # contamination from the sky's own colour drift over the night combined with the changing frame composition near the
    # horizon (RUN_LOG §25 first round: 0.043 with the sign flipped), so the criterion moved to H3c.
    "H3": {"max_abs_deviation": 0.02, "h_min": 4, "h_max": 400, "normalise_rows": 100},
    # H3c: master colour relative to the colour of the frames that actually contribute at that height (their exported
    # r_ frames, valid pixels only, weighted by how many pixels each contributes) — "the stack's colour is its
    # contributors' colour": ground of other frames would show as a deviation, the night's drift would not.
    "H3c": {"max_abs_deviation": 0.02, "h_min": 4, "h_max": 300, "band_px": 25, "min_frame_pixels": 200},
    # H5 2.0.0: recorded effective weight, same-location composition error,
    # and explicit missing/N/A states. Tolerance unchanged; historical
    # brightness ratios are observations and no longer determine a verdict.
    "H5": {"rim_min_ratio": 0.85, "rim_rows": 12, "max_zero_fraction": 1e-5, "compose_tolerance": 1e-5, "band_min_ratio": 0.85},
}


def mask_edge_rows(mask: np.ndarray) -> np.ndarray:
    """Per column, the first top-down row whose mask weight is below 1 (the start of the ground / feather); H if none."""
    below = mask < 1.0
    edge = np.where(below.any(axis=0), below.argmax(axis=0), mask.shape[0])
    return edge.astype(int)


def column_ranges(width: int, count: int = 3, span: int = 700) -> list[tuple[int, int]]:
    """``count`` column ranges spread over the width (scene-independent placement)."""
    centres = np.linspace(width * 0.125, width * 0.875, count)
    out = []
    for c in centres:
        x0 = int(max(0, min(width - span, round(c - span / 2))))
        out.append((x0, x0 + span))
    return out


def colour_band(master: np.ndarray, frame: np.ndarray, mask: np.ndarray, ranges, *, h_min: int = 4, h_max: int = 400,
                normalise_rows: int = 100, column_step: int = 4) -> dict:
    """Ratio-of-ratios colour profiles above the mask edge; returns per-range profiles, maxima and the verdict inputs."""
    edge = mask_edge_rows(mask)
    height = mask.shape[0]
    out = {"h_min": h_min, "h_max": h_max, "ranges": {}}
    for x0, x1 in ranges:
        prof = {"RG": [], "BG": []}
        cols = [x for x in range(x0, x1, column_step) if edge[x] - h_max >= 0 and edge[x] < height]
        if not cols:
            out["ranges"][f"x{x0}-{x1}"] = {"status": "no_ground_edge_in_range"}
            continue
        rows_m, rows_f = {"RG": [], "BG": []}, {"RG": [], "BG": []}
        for x in cols:
            e = edge[x]
            sl = slice(e - h_max, e)
            m = master[sl, x, :][::-1]; f = frame[sl, x, :][::-1]           # index 0 = the row just above the edge
            g_m = np.maximum(m[:, 1], 1e-9); g_f = np.maximum(f[:, 1], 1e-9)
            rows_m["RG"].append(m[:, 0] / g_m); rows_f["RG"].append(f[:, 0] / g_f)
            rows_m["BG"].append(m[:, 2] / g_m); rows_f["BG"].append(f[:, 2] / g_f)
        entry = {"columns": len(cols)}
        for key in ("RG", "BG"):
            pm = np.median(np.array(rows_m[key]), axis=0); pf = np.median(np.array(rows_f[key]), axis=0)
            ratio = pm / np.maximum(pf, 1e-9)
            top = np.median(ratio[h_max - normalise_rows:h_max])
            rel = ratio / top
            window = rel[h_min:h_max - normalise_rows]
            entry[key] = {"max_abs_deviation": float(np.max(np.abs(window - 1.0))), "at_edge": float(rel[h_min]),
                          "samples": {str(h): round(float(rel[h]), 4) for h in (h_min, 10, 25, 50, 100, 150, 200, 300) if h < h_max - normalise_rows},
                          "height_above_2pct_px": int(np.max(np.where(np.abs(window - 1.0) > 0.02)[0]) + h_min + 1) if np.any(np.abs(window - 1.0) > 0.02) else 0}
        out["ranges"][f"x{x0}-{x1}"] = entry
    values = [e[key]["max_abs_deviation"] for e in out["ranges"].values() if "RG" in e for key in ("RG", "BG")]
    out["max_abs_deviation"] = float(max(values)) if values else None
    return out


def judge_colour_band(result: dict, criteria: dict) -> dict:
    failing = []
    for name, e in result["ranges"].items():
        if "RG" not in e:
            failing.append(f"{name}: {e.get('status')}")
            continue
        for key in ("RG", "BG"):
            if e[key]["max_abs_deviation"] > criteria["max_abs_deviation"]:
                failing.append(f"{name} {key}: max |dev| {e[key]['max_abs_deviation']:.3f} > {criteria['max_abs_deviation']}")
    return {"passed": not failing, "failing": failing}


def master_coverage_consistency(master_green: np.ndarray, coverage: np.ndarray, step: int) -> dict:
    """Agreement between "master is exactly zero" and "coverage is zero" on the coverage grid."""
    sampled = master_green[::step, ::step][:coverage.shape[0], :coverage.shape[1]]
    zero_master = sampled == 0.0
    zero_cov = coverage == 0
    agree = zero_master == zero_cov
    return {"samples": int(agree.size), "agreement": float(agree.mean()),
            "master_zero_fraction": float(zero_master.mean()), "coverage_zero_fraction": float(zero_cov.mean()),
            "zero_master_with_coverage": int(np.sum(zero_master & ~zero_cov)), "nonzero_master_without_coverage": int(np.sum(~zero_master & zero_cov))}


def contributor_colour(master_u: np.ndarray, frame_crops, band_edges: list[int], *, min_frame_pixels: int = 200) -> dict:
    """Master colour against a plain mean of its (normalised) contributors, band by band.

    ``master_u``: (h_max, ncols, 3) crop of the master with row 0 the row just above the ground edge (already
    reversed); ``frame_crops``: iterable of the same crops for every stacked frame, already normalised the way the
    stack normalises them (zeros = the frame did not observe the pixel); ``band_edges``: row bounds of the bands.
    The expected stack is the per-pixel mean over the frames that observed the pixel; per band the median R/G and
    B/G of the master and of the expected stack over the same pixels, and their ratio.
    """
    h, n, _ = master_u.shape
    acc = np.zeros((h, n, 3), dtype=np.float64)
    count = np.zeros((h, n), dtype=np.int32)
    frames = 0
    for crop in frame_crops:
        frames += 1
        ok = crop[..., 1] > 0
        acc[ok] += crop[ok]
        count[ok] += 1
    expected = np.where(count[..., None] > 0, acc / np.maximum(count, 1)[..., None], 0.0)
    bands = [(band_edges[i], band_edges[i + 1]) for i in range(len(band_edges) - 1)]
    out = {"frames": frames, "bands": []}
    for b in bands:
        m = master_u[b[0]:b[1]]; e = expected[b[0]:b[1]]; c = count[b[0]:b[1]]
        ok = (m[..., 1] > 0) & (c > 0)
        entry = {"h0": int(b[0]), "h1": int(b[1]), "pixels": int(ok.sum()), "contributors_mean": float(c[ok].mean()) if ok.any() else 0.0,
                 "contributors_min": int(c[ok].min()) if ok.any() else 0}
        if ok.sum() >= min_frame_pixels:
            for key, idx in (("RG", 0), ("BG", 2)):
                master_val = float(np.median(m[..., idx][ok] / m[..., 1][ok]))
                expected_val = float(np.median(e[..., idx][ok] / e[..., 1][ok]))
                entry[key] = {"master": master_val, "expected": expected_val, "ratio": master_val / expected_val if expected_val else None}
        out["bands"].append(entry)
    ratios = [e[key]["ratio"] for e in out["bands"] for key in ("RG", "BG") if key in e and e[key]["ratio"] is not None]
    out["max_abs_deviation"] = float(max(abs(r - 1.0) for r in ratios)) if ratios else None
    out["bands_without_data"] = [(e["h0"], e["h1"]) for e in out["bands"] if "RG" not in e]
    return out


def normalise_like_stack(crop: np.ndarray, stats: dict, ref_stats: dict) -> np.ndarray:
    """Siril ``-norm=addscale`` on a crop: per channel ``(v - median_k) * (scale_ref / scale_k) + median_ref`` for the
    frame's non-zero pixels (its global median and MAD-based scale, the reference's the same); zeros stay zero."""
    out = np.zeros_like(crop, dtype=np.float32)
    ok = crop[..., 1] > 0
    for c in range(3):
        scale = ref_stats["mad"][c] / stats["mad"][c] if stats["mad"][c] > 0 else 1.0
        out[..., c][ok] = (crop[..., c][ok] - stats["median"][c]) * scale + ref_stats["median"][c]
    return out


def global_stats(planes_sampled) -> dict:
    """Per-channel median and MAD over a frame's non-zero sampled pixels (``planes_sampled``: three 2-D arrays)."""
    medians, mads = [], []
    for a in planes_sampled:
        v = np.asarray(a, dtype=np.float64).ravel()
        v = v[v > 0]
        if v.size == 0:
            medians.append(0.0); mads.append(0.0); continue
        med = float(np.median(v))
        medians.append(med); mads.append(float(np.median(np.abs(v - med))))
    return {"median": medians, "mad": mads}


def judge_contributor_colour(results: dict, criteria: dict) -> dict:
    """``results``: range name -> contributor_colour output."""
    failing = []
    for name, r in results.items():
        if r.get("max_abs_deviation") is None:
            failing.append(f"{name}: no band with data")
            continue
        if r["bands_without_data"]:
            failing.append(f"{name}: bands without data {r['bands_without_data']}")
        if r["max_abs_deviation"] > criteria["max_abs_deviation"]:
            failing.append(f"{name}: max |dev| {r['max_abs_deviation']:.3f} > {criteria['max_abs_deviation']}")
    return {"passed": not failing, "failing": failing}
