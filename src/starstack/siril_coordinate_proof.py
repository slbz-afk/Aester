"""Prove Siril's coordinate conventions from a synthetic run with known truth.

Given the truth written by :mod:`starstack.synthetic_starfield` and the
products of a real Siril ``register`` run on those frames (the original
``light_.seq`` with homographies and the ``cache/light_0000k.lst`` star
lists), this module determines, with evidence:

* **row order** of Siril's star/H coordinates relative to the TOP-DOWN truth,
* the **pixel-origin offset** (0, 0.5 or 1 pixel) on each axis,
* the **direction** of the stored homography (frame->reference, as the
  sirilpy docstring says, or the inverse).

The estimate uses only the synthetic data.  Real photographs are never used
to pick a convention.  ``simulate_siril_outputs`` writes files in Siril's
formats under a chosen convention so that the analysis itself can be tested
off-Windows; a simulated run is labelled as such in every output.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import siril_sequence
from .registration_adapter import CoordinateHypothesis

__all__ = ["PROOF_VERSION", "analyze_run", "simulate_siril_outputs", "hypothesis_from_proof"]

PROOF_VERSION = "1.0.0"
SNAP_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _nearest(points: np.ndarray, targets: np.ndarray):
    """For each point, index and distance of the nearest target (brute force)."""
    if len(points) == 0 or len(targets) == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    d2 = ((points[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    return idx, np.sqrt(d2[np.arange(len(points)), idx])


def _to_top_down(xy: np.ndarray, row_order: str, height: int, x_offset: float = 0.0, y_offset: float = 0.0) -> np.ndarray:
    out = np.empty_like(xy)
    out[:, 0] = xy[:, 0] + x_offset
    if row_order == "TOP-DOWN":
        out[:, 1] = xy[:, 1] + y_offset
    else:
        out[:, 1] = (height - 1) - (xy[:, 1] + y_offset)
    return out


def _estimate_row_order_and_offsets(detected_xy: np.ndarray, truth_xy: np.ndarray, height: int) -> dict:
    """Try both row orders with zero offsets, then measure the residual offsets."""
    candidates = {}
    for row_order in ("TOP-DOWN", "BOTTOM-UP"):
        mapped = _to_top_down(detected_xy, row_order, height)
        idx, dist = _nearest(mapped, truth_xy)
        keep = dist <= 3.0
        matched = int(keep.sum())
        entry = {"matched": matched, "detected": int(len(detected_xy)), "truth": int(len(truth_xy)),
                 "matched_fraction_of_truth": matched / max(1, len(truth_xy))}
        if matched >= 3:
            delta = truth_xy[idx[keep]] - mapped[keep]      # what must be added to reach the truth
            x_off = float(np.median(delta[:, 0]))
            y_med = float(np.median(delta[:, 1]))
            # for BOTTOM-UP, y_td = (h-1) - (y_s + y_offset)  =>  y_offset = -delta_y
            y_off = y_med if row_order == "TOP-DOWN" else -y_med
            spread = float(np.median(np.abs(delta - np.median(delta, axis=0)), axis=0).max())
            entry.update(x_offset_raw=x_off, y_offset_raw=y_off, spread_mad_px=spread)
        candidates[row_order] = entry
    return candidates


def _snap(value: float) -> tuple[float, bool]:
    nearest = min(SNAP_VALUES, key=lambda v: abs(v - value))
    return (nearest, True) if abs(nearest - value) <= 0.1 else (value, False)


def _normalise(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if abs(h[2, 2]) < 1e-15:
        return h / np.max(np.abs(h))
    return h / h[2, 2]


def _corner_transfer_error(h_a: np.ndarray, h_b: np.ndarray, width: int, height: int) -> float:
    corners = np.array([[-0.5, -0.5, 1.0], [width - 0.5, -0.5, 1.0], [width - 0.5, height - 0.5, 1.0], [-0.5, height - 0.5, 1.0]])
    pa = corners @ h_a.T
    pb = corners @ h_b.T
    pa = pa[:, :2] / pa[:, 2:3]
    pb = pb[:, :2] / pb[:, 2:3]
    return float(np.max(np.hypot(*(pa - pb).T)))


def analyze_run(truth: dict, seq_path: Path, cache_dir: Path, *, layer: int = 1, simulated: bool = False,
                siril_version: str | None = None) -> dict:
    """Determine Siril's conventions from truth + a register run; never from photographs."""
    width, height = truth["width"], truth["height"]
    sequence = siril_sequence.read_seq(seq_path)
    # A sequence written by simulate_siril_outputs is marked in its first comment; such
    # files can never yield a verified hypothesis, whatever the caller passed.
    first_line = Path(seq_path).read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if first_line and "SIMULATED" in first_line[0]:
        simulated = True
    header = sequence["header"]
    if header["reference_image"] != 0:
        raise ValueError("the synthetic run must use the first image as reference (setref 1)")
    regdata = sequence["regdata"].get(layer)
    if regdata is None:
        raise ValueError(f"no registration data for layer {layer} in {seq_path}")
    if sequence["all_identity_per_layer"].get(layer):
        raise ValueError("all homographies are identity: pass the original sequence, not the registered one")
    truth_frames = {f["frame"]: f for f in truth["frames"]}
    lists = {}
    for image in sequence["images"]:
        path = Path(cache_dir) / f"{image['stem']}.lst"
        if not path.exists():
            raise ValueError(f"missing star list {path}")
        lists[image["position"]] = siril_sequence.read_lst(path)
    result = {"proof_version": PROOF_VERSION, "simulated": simulated, "siril_version": siril_version,
              "sequence_source": sequence["source"], "sequence_header": header,
              "star_list_sources": {pos: l["source"] for pos, l in lists.items()},
              "layer": layer, "width": width, "height": height, "status": "inconclusive",
              "row_order": None, "x_offset": None, "y_offset": None, "matrix_direction": None,
              "evidence": {}}
    # 1. row order and offsets from the reference frame (identity transform)
    ref_truth = np.array([[s["x"], s["y"]] for s in truth_frames[1]["rendered_stars"]])
    ref_detected = np.array([[s["x"], s["y"]] for s in lists[0]["stars"]])
    candidates = _estimate_row_order_and_offsets(ref_detected, ref_truth, height)
    result["evidence"]["row_order_candidates"] = candidates
    good = [ro for ro, c in candidates.items()
            if c["matched_fraction_of_truth"] >= 0.8 and c.get("spread_mad_px", 9) <= 0.15]
    bad = [ro for ro, c in candidates.items() if c["matched_fraction_of_truth"] <= 0.2]
    if len(good) != 1 or len(bad) != 1:
        result["status"] = "inconclusive"
        result["reason"] = "row order not separable: neither exactly one candidate matches >=80 % of stars nor the other <=20 %"
        return result
    row_order = good[0]
    x_off, x_snapped = _snap(candidates[row_order]["x_offset_raw"])
    y_off, y_snapped = _snap(candidates[row_order]["y_offset_raw"])
    result.update(row_order=row_order, x_offset=x_off, y_offset=y_off)
    result["evidence"]["offsets"] = {"x_raw": candidates[row_order]["x_offset_raw"], "x_snapped": x_snapped,
                                     "y_raw": candidates[row_order]["y_offset_raw"], "y_snapped": y_snapped,
                                     "snap_tolerance_px": 0.1}
    hypothesis = CoordinateHypothesis(name="proof_candidate", row_order=row_order, x_offset=x_off, y_offset=y_off,
                                      matrix_direction="frame_to_reference", verified=False, basis="synthetic proof")
    t_siril_to_td = hypothesis.affine(height)
    t_td_to_siril = np.linalg.inv(t_siril_to_td)
    # 2. per-frame star binding under the candidate convention
    frames_evidence = {}
    binding_ok = True
    for image in sequence["images"]:
        pos = image["position"]
        frame_no = pos + 1
        detected = np.array([[s["x"], s["y"]] for s in lists[pos]["stars"]])
        mapped = _to_top_down(detected, row_order, height, x_off, y_off)
        best = None
        for candidate_frame, tf in truth_frames.items():
            tr = np.array([[s["x"], s["y"]] for s in tf["rendered_stars"]])
            idx, dist = _nearest(mapped, tr)
            fraction = float((dist <= 1.0).sum() / max(1, len(tr)))
            if best is None or fraction > best[1]:
                best = (candidate_frame, fraction, float(np.median(dist[dist <= 1.0])) if (dist <= 1.0).any() else None)
        expected_fraction = None
        tr = np.array([[s["x"], s["y"]] for s in truth_frames[frame_no]["rendered_stars"]])
        idx, dist = _nearest(mapped, tr)
        expected_fraction = float((dist <= 1.0).sum() / max(1, len(tr)))
        ok = expected_fraction >= 0.9 and best[0] == frame_no
        binding_ok &= ok
        frames_evidence[image["stem"]] = {"expected_truth_frame": frame_no, "best_truth_frame": best[0],
                                          "matched_fraction_expected": expected_fraction,
                                          "matched_fraction_best": best[1], "median_residual_px": best[2],
                                          "binding_ok": ok, "detected_count": int(len(detected))}
    result["evidence"]["frame_binding"] = frames_evidence
    # 3. homography direction from every non-reference frame
    direction_votes = {"frame_to_reference": 0, "reference_to_frame": 0}
    h_evidence = {}
    for image in sequence["images"][1:]:
        pos = image["position"]
        f_td = np.asarray(truth_frames[pos + 1]["frame_to_reference"], dtype=np.float64)
        h_pred = _normalise(t_td_to_siril @ f_td @ t_siril_to_td)
        h_siril = _normalise(np.asarray(regdata[pos]["H"], dtype=np.float64))
        err_forward = _corner_transfer_error(h_siril, h_pred, width, height)
        err_inverse = _corner_transfer_error(h_siril, _normalise(np.linalg.inv(h_pred)), width, height)
        verdict = None
        if err_forward <= 0.1 and err_inverse > 1.0:
            verdict = "frame_to_reference"
        elif err_inverse <= 0.1 and err_forward > 1.0:
            verdict = "reference_to_frame"
        if verdict:
            direction_votes[verdict] += 1
        h_evidence[image["stem"]] = {"corner_error_if_frame_to_reference_px": err_forward,
                                     "corner_error_if_reference_to_frame_px": err_inverse, "verdict": verdict,
                                     "siril_H_normalised": h_siril.tolist(), "predicted_H_normalised": h_pred.tolist()}
    result["evidence"]["homography_direction"] = h_evidence
    decided = [d for d, votes in direction_votes.items() if votes == len(sequence["images"]) - 1]
    if len(decided) == 1 and binding_ok and x_snapped and y_snapped:
        result["matrix_direction"] = decided[0]
        result["status"] = "proved"
    else:
        result["status"] = "inconclusive"
        reasons = []
        if len(decided) != 1:
            reasons.append("homography direction not unanimous across frames")
        if not binding_ok:
            reasons.append("frame/star-list binding failed for at least one frame")
        if not (x_snapped and y_snapped):
            reasons.append("pixel offsets are not within 0.1 px of a multiple of 0.5")
        result["reason"] = "; ".join(reasons)
    return result


def hypothesis_from_proof(proof: dict, name: str = "siril_1_4_4_proved") -> CoordinateHypothesis:
    if proof.get("status") != "proved":
        raise ValueError("proof is not conclusive")
    return CoordinateHypothesis(name=name, row_order=proof["row_order"], x_offset=float(proof["x_offset"]),
                                y_offset=float(proof["y_offset"]), matrix_direction=proof["matrix_direction"],
                                verified=not proof.get("simulated", False),
                                basis=("synthetic Siril run " + str(proof.get("sequence_source", {}).get("sha256"))
                                       if not proof.get("simulated") else "SIMULATED outputs only, not a proof"))


# --------------------------------------------------------------------------- #
# Simulation of Siril outputs (self-test of the analysis, never a proof)      #
# --------------------------------------------------------------------------- #

def _fmt(value: float) -> str:
    return f"{value:g}"


def simulate_siril_outputs(truth: dict, work_dir: Path, *, row_order: str, x_offset: float, y_offset: float,
                           matrix_direction: str, noise_px: float = 0.03, seed: int = 5,
                           name: str = "light_") -> dict:
    """Write ``light_.seq`` and ``cache/<stem>.lst`` as Siril would under a convention.

    Offsets follow the adapter definition (Siril -> TOP-DOWN adds x_offset /
    y_offset).  Star positions get Gaussian centroid noise and two-decimal
    rounding like real Siril lists.  Returns the paths written.
    """
    work_dir = Path(work_dir)
    (work_dir / "cache").mkdir(parents=True, exist_ok=True)
    height = truth["height"]
    hypothesis = CoordinateHypothesis(name="sim", row_order=row_order, x_offset=x_offset, y_offset=y_offset,
                                      matrix_direction="frame_to_reference", verified=False, basis="simulation")
    t_siril_to_td = hypothesis.affine(height)
    t_td_to_siril = np.linalg.inv(t_siril_to_td)
    rng = np.random.default_rng(seed)
    frames = truth["frames"]
    lines = ["#Siril sequence file (SIMULATED by starstack.siril_coordinate_proof, not Siril output)",
             f"S '{name}' 1 {len(frames)} {len(frames)} 5 0 6 0 0 0", "L 3"]
    for i in range(len(frames)):
        lines.append(f"I {i + 1} 1")
    for frame in frames:
        f_td = np.asarray(frame["frame_to_reference"], dtype=np.float64)
        h = _normalise(t_td_to_siril @ f_td @ t_siril_to_td)
        if matrix_direction == "reference_to_frame":
            h = _normalise(np.linalg.inv(h))
        coeff = " ".join(_fmt(v) for v in h.reshape(-1))
        lines.append(f"R1 3.5 3.5 0.8 0 0.02 {len(frame['rendered_stars'])} H {coeff}")
    (work_dir / f"{name}.seq").write_text("\n".join(lines) + "\n", encoding="utf-8")
    written = {"seq": str(work_dir / f"{name}.seq"), "lists": []}
    for frame in frames:
        stem = f"{name}{frame['frame']:05d}"
        rows = ["# %d stars found using the following parameters:" % len(frame["rendered_stars"]),
                "# sigma=1.00 roundness=0.50 radius=5 relax=0 profile=0 minbeta=1.5 max_stars=2000 layer=1 minA=0.00 maxA=0.00 maxR=1.00 (SIMULATED)",
                "# star#\tlayer\tB\tA\tbeta\tX\tY\tFWHMx [px]\tFWHMy [px]\tFWHMx [\"]\tFWHMy [\"]\tangle\tRMSE\tmag\tSat\tProfile\tRA\tDec"]
        order = sorted(frame["rendered_stars"], key=lambda s: -s["amplitude"])
        for n, star in enumerate(order, 1):
            v = t_td_to_siril @ np.array([star["x"], star["y"], 1.0])
            xs, ys = v[0] + rng.normal(0, noise_px), v[1] + rng.normal(0, noise_px)
            rows.append(f"{n}\t1\t800.0\t{star['amplitude']:.3f}\t-1.00\t{xs:10.2f}\t{ys:10.2f}\t3.50\t3.40\t0.0\t0.0\t0.0\t1.0e+002\t-10.00\t0\tGaussian\t0.0\t0.0")
        path = work_dir / "cache" / f"{stem}.lst"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        written["lists"].append(str(path))
    return written
