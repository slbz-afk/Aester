"""Prove whether Siril 1.4.4 accepts a SIP distortion we write and applies it in ``register``/``seqapplyreg``.

Route question (RUN_LOG §17): the shared radial term fitted on the real Z8
job would enter the product most cleanly through Siril's own
``register ... -disto`` (star positions corrected before the fit, images
undistorted once on export) — *if* Siril accepts a distortion solution that
we synthesise instead of one from a plate solve.  This module builds the
synthetic evidence for that, in the same spirit as the coordinate proof:

* ``generate_dataset`` renders frames whose stars sit at *distorted*
  positions ``p = D^-1(u)`` of undistorted truth ``u`` (frames related by
  exact homographies in undistorted space), and writes ``disto_solution.fit``
  — the reference frame carrying a WCS with SIP polynomials that encode the
  same radial ``D`` (forward A/B and fitted inverse AP/BP).
* ``analyze_run`` reads what Siril produced (``light_.seq`` homographies,
  ``cache/*.lst`` star lists, ``r_light_*.fit`` output frames) and decides,
  frame by frame, whether the homographies are the undistorted truth, whether
  the star lists are raw or corrected, and whether the exported frames are
  undistorted (stars at ``u``) or not (stars at ``p``).

* ``own_registration_sequence`` (RUN_LOG §22) rewrites the sequence file
  Siril left after a plain ``register -2pass`` with homographies *we* compute
  (the truth, optionally composed with a known shift) plus the ``D`` line for
  the SIP file, so that ``seqapplyreg`` alone exports the frames — the route
  the product takes when Siril's own star matching fails on long spans.  A
  known shift proves that the exported frames follow the written matrices and
  not a registration Siril did itself.

Conventions: truth is TOP-DOWN zero-based pixel centres.  SIP works on FITS
1-based pixel coordinates relative to CRPIX with rows counted from the
bottom; the radial polynomial is invariant under that flip, so only the
centre is converted.  Siril's own star/H convention is taken from the
proved adapter hypothesis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import fits_crop, siril_sequence
from . import registration_adapter as adapter
from .registration_distortion import RadialDistortion
from .sip_header import (sip_apply_forward, sip_cards, sip_forward_coefficients, sip_inverse_coefficients,  # noqa: F401
                         write_wcs_header_file)
from .synthetic_starfield import _card, default_layout, render_frame, rotation_scale_about, translation, write_fits_rgb16

__all__ = ["SIP_PROOF_VERSION", "generate_dataset", "analyze_run", "simulate_siril_outputs", "own_registration_sequence",
           "simulate_seqapplyreg", "default_distortion", "z8_like_distortion", "z8_like_inverse_distortion"]

SIP_PROOF_VERSION = "1.5.0"


def default_distortion(width: int = 1200, height: int = 901) -> RadialDistortion:
    """Off-centre barrel-like distortion: 2 % at the corner, centre 15/11 px from the image centre."""
    return RadialDistortion(0.02, -0.005, (width - 1) / 2.0 - 15.0, (height - 1) / 2.0 + 11.0, float(np.hypot(width, height) / 2.0))


def z8_like_distortion(width: int = 1200, height: int = 901) -> RadialDistortion:
    """The Z8 20 mm job's radial term (RUN_LOG §22 fit: k1 0.0537, k2 -0.0176) at the synthetic frame size: 3.6 % at the
    corner.  With it the order of the AP/BP inverse matters (order 5: ~0.5 % of the corner shift), unlike the mild default."""
    return RadialDistortion(0.0537, -0.0176, (width - 1) / 2.0 + 13.0, (height - 1) / 2.0 - 12.0, float(np.hypot(width, height) / 2.0))


def z8_like_inverse_distortion(width: int = 1200, height: int = 901) -> RadialDistortion:
    """Z8-strength barrel lens parameterised on the undistorted side (``u_to_p``, the product's form since RUN_LOG §22):
    its AP/BP inverse is exact at order 5, the forward A/B are a fit (~1 % of the corner shift)."""
    return RadialDistortion(-0.0497, 0.0125, (width - 1) / 2.0 + 13.0, (height - 1) / 2.0 - 12.0, float(np.hypot(width, height) / 2.0),
                            direction="u_to_p")


def proof_transforms() -> list[dict]:
    """Four frames in *undistorted* space (frame -> reference), larger motions than the coordinate proof."""
    return [
        {"frame": 1, "kind": "identity", "matrix": np.eye(3).tolist(), "description": "reference"},
        {"frame": 2, "kind": "translation", "matrix": translation(25.25, -12.5).tolist(), "description": "translation (+25.25, -12.5)"},
        {"frame": 3, "kind": "rotation", "matrix": rotation_scale_about(1.5, 1.0, 600.0, 450.0, -18.0, 30.0).tolist(),
         "description": "rotation +1.5 deg about (600, 450) then translation (-18, +30)"},
        {"frame": 4, "kind": "rotation", "matrix": rotation_scale_about(-1.0, 1.0, 300.0, 200.0, 30.5, 21.75).tolist(),
         "description": "rotation -1.0 deg about (300, 200) then translation (+30.5, +21.75)"},
    ]


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

def _fits_from_top_down(xy_td: np.ndarray, height: int) -> np.ndarray:
    return np.column_stack([xy_td[:, 0] + 1.0, height - xy_td[:, 1]])


def generate_dataset(output_dir: Path, *, distortion: RadialDistortion | None = None, layout: dict | None = None,
                     transforms: list[dict] | None = None, stem: str = "light") -> dict:
    """Write ``light_0000k.fit`` with distorted stars, ``disto_solution.fit`` and ``truth.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    layout = layout or default_layout()
    width, height = layout["width"], layout["height"]
    distortion = distortion or default_distortion(width, height)
    transforms = transforms or proof_transforms()
    u_ref = np.array([[s["x"], s["y"]] for s in layout["stars"]], dtype=np.float64)
    frames = []
    for spec in transforms:
        m = np.asarray(spec["matrix"], dtype=np.float64)          # frame -> reference, undistorted
        v = np.linalg.inv(m) @ np.vstack([u_ref.T, np.ones(len(u_ref))])
        u_frame = (v[:2] / v[2]).T
        p_frame = distortion.inverse(u_frame)                    # what the sensor records
        frame_layout = {"width": width, "height": height,
                        "stars": [{**star, "x": float(px), "y": float(py)} for star, (px, py) in zip(layout["stars"], p_frame)]}
        image, rendered = render_frame(frame_layout, np.eye(3), seed=spec["frame"])
        rendered_ids = {r["star_id"] for r in rendered}
        filename = f"{stem}_{spec['frame']:05d}.fit"
        extra = sip_cards(distortion, width, height, inverse_order=5) if spec["frame"] == 1 else []
        digest = write_fits_rgb16(output_dir / filename, image, extra_cards=[],
                                  comments=[f"synthetic distorted frame {spec['frame']}: {spec['description']}",
                                            "stars rendered at distorted positions p = D^-1(u); truth in truth.json"])
        if spec["frame"] == 1:
            write_fits_rgb16(output_dir / "disto_solution.fit", image, extra_cards=extra,
                             comments=["reference frame with a synthetic WCS+SIP solution encoding D (starstack.siril_sip_proof)",
                                       "CRPIX2 in the standard FITS bottom-up convention"])
            write_fits_rgb16(output_dir / "disto_solution_topdown.fit", image,
                             extra_cards=sip_cards(distortion, width, height, inverse_order=5, crpix2_convention="top_down"),
                             comments=["same solution with CRPIX2 written as a top-down row (Siril display grid)"])
            write_wcs_header_file(output_dir / "disto_solution.wcs", extra, width, height)
            magnified = distortion.with_magnification(distortion.fitting_magnification(width, height))
            write_wcs_header_file(output_dir / "disto_solution_mag.wcs", sip_cards(magnified, width, height, inverse_order=5), width, height)
        frames.append({"frame": spec["frame"], "filename": filename, "sha256": digest, "kind": spec["kind"],
                       "description": spec["description"], "frame_to_reference_undistorted": spec["matrix"],
                       "rendered_stars": rendered,
                       "undistorted_stars": [{"star_id": s["star_id"], "x": float(ux), "y": float(uy)}
                                             for s, (ux, uy) in zip(layout["stars"], u_frame) if s["star_id"] in rendered_ids]})
    truth = {"proof_version": SIP_PROOF_VERSION, "coordinate_convention": {"row_order": "TOP-DOWN", "origin": "top_left_pixel_center", "units": "pixel"},
             "width": width, "height": height, "layout_seed": layout["seed"], "distortion": distortion.as_dict(),
             "distortion_corner_shift_px": distortion.corner_shift_px(width, height),
             "sip": {"forward_order": sip_forward_coefficients(distortion, width, height)["order"],
                     "forward_max_fit_error_px": sip_forward_coefficients(distortion, width, height)["max_fit_error_px"],
                     "inverse_max_fit_error_px": sip_inverse_coefficients(distortion, width, height, 5)["max_fit_error_px"],
                     "inverse_max_fit_error_px_by_order": {str(o): sip_inverse_coefficients(distortion.with_magnification(distortion.fitting_magnification(width, height)), width, height, o)["max_fit_error_px"] for o in (5, 7, 9)},
                     "direction": distortion.direction,
                     "crpix_fits": [distortion.cx + 1.0, height - distortion.cy],
                     "crpix_topdown": [distortion.cx + 1.0, distortion.cy + 1.0]},
             "solution_file": "disto_solution.fit", "solution_file_topdown": "disto_solution_topdown.fit",
             "solution_wcs_header_only": "disto_solution.wcs",
             "solution_wcs_magnified": {"file": "disto_solution_mag.wcs", "magnification": distortion.fitting_magnification(width, height), "inverse_order": 5},
             "reference_stars_undistorted": layout["stars"], "frames": frames}
    (output_dir / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return truth


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #

def _nearest_fraction(points: np.ndarray, targets: np.ndarray, tol: float) -> tuple[float, float | None]:
    if len(points) == 0 or len(targets) == 0:
        return 0.0, None
    d = np.hypot(points[:, None, 0] - targets[None, :, 0], points[:, None, 1] - targets[None, :, 1])
    best = d.min(axis=0)
    hit = best <= tol
    return float(hit.mean()), (float(np.median(best[hit])) if hit.any() else None)


def _stats(arr: np.ndarray) -> dict:
    dist = np.hypot(arr[:, 0], arr[:, 1]) if len(arr) else np.zeros(0)
    return {"count": int(len(arr)),
            "rms_px": float(np.sqrt(np.mean(dist ** 2))) if len(dist) else None,
            "median_px": float(np.median(dist)) if len(dist) else None,
            "p95_px": float(np.percentile(dist, 95)) if len(dist) else None,
            "max_px": float(dist.max()) if len(dist) else None,
            "mean_dx_px": float(arr[:, 0].mean()) if len(arr) else None, "mean_dy_px": float(arr[:, 1].mean()) if len(arr) else None}


def _centroid_residuals(frame_path: Path, expected_td: np.ndarray, *, radius: int = 4, plane: int = 1, edge_margin: int = 30) -> dict:
    """Centroid each expected star in the frame (pixel measurement) and report offsets to the expectation.

    Stars closer than ``edge_margin`` to the frame edge are reported
    separately (``interior``): after a shift or rotation the resampled frame
    has no data there, and the centroid window then touches the padding.
    """
    header, _ = fits_crop.read_header(frame_path)
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    plane_view, _, _ = fits_crop.plane_memmap(frame_path, plane)
    scale = 1.0 if int(header["BITPIX"]) == 16 else 65535.0
    image = np.asarray(plane_view, dtype=np.float32)
    if int(header["BITPIX"]) == 16:
        image = image + 32768.0
    image = image[::-1, :] * scale if header.get("ROWORDER", "BOTTOM-UP").upper() == "BOTTOM-UP" else image * scale
    offsets, interior = [], []
    for x, y in expected_td:
        if not (radius + 2 <= x < width - radius - 2 and radius + 2 <= y < height - radius - 2):
            continue
        c = fits_crop._centroid(image, float(x), float(y), radius)
        if c is None or c[3] - c[2] < 50.0:
            continue
        offsets.append([c[0] - x, c[1] - y])
        if edge_margin <= x < width - edge_margin and edge_margin <= y < height - edge_margin:
            interior.append([c[0] - x, c[1] - y])
    all_stats = _stats(np.asarray(offsets, dtype=np.float64).reshape(-1, 2))
    return {**all_stats, "expected": int(len(expected_td)), "interior": _stats(np.asarray(interior, dtype=np.float64).reshape(-1, 2)),
            "edge_margin_px": edge_margin, "width": width, "height": height, "bitpix": header["BITPIX"]}


def analyze_run(truth: dict, work: Path, *, variant: str, simulated: bool = False, siril_version: str | None = None,
                hypothesis: adapter.CoordinateHypothesis | None = None, layer: int = 1, magnification: float = 1.0,
                expected_shift_px: tuple[float, float] | None = None, own_registration: bool = False) -> dict:
    """Decide from Siril's products whether the SIP solution was used for H, star lists and exported frames.

    With ``own_registration`` the sequence's homographies were written by us
    (``own_registration_sequence``): every non-reference frame's stars must
    then land at the reference's undistorted positions *plus*
    ``expected_shift_px`` (and, when a shift is given, not at the unshifted
    ones), which shows that ``seqapplyreg`` used the written matrices.  The
    statuses are then ``own_registration_applied`` / ``own_registration_ignored``.
    """
    work = Path(work)
    hypothesis = hypothesis or adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
    width, height = truth["width"], truth["height"]
    distortion = RadialDistortion(**truth["distortion"])
    shift = np.array(expected_shift_px if expected_shift_px is not None else (0.0, 0.0), dtype=np.float64)
    shift_mat = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
    # a solution written with a global magnification m places the undistorted grid at c + m (u - c)
    scale_mat = np.array([[magnification, 0.0, (1 - magnification) * distortion.cx],
                          [0.0, magnification, (1 - magnification) * distortion.cy], [0.0, 0.0, 1.0]])

    def scaled(xy: np.ndarray) -> np.ndarray:
        return distortion.centre + magnification * (xy - distortion.centre)

    seq_path = work / "light_.seq"
    sequence = siril_sequence.read_seq(seq_path)
    first_line = seq_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if first_line and "SIMULATED" in first_line[0]:
        simulated = True
    regdata = sequence["regdata"].get(layer)
    if regdata is None:
        raise ValueError(f"no registration data for layer {layer}")
    ref_pos = sequence["header"]["reference_image"]
    ref_no = sequence["images"][ref_pos]["index"]
    truth_frames = {f["frame"]: f for f in truth["frames"]}
    m_ref = np.asarray(truth_frames[ref_no]["frame_to_reference_undistorted"], dtype=np.float64)
    result = {"proof_version": SIP_PROOF_VERSION, "variant": variant, "simulated": simulated, "siril_version": siril_version,
              "magnification": magnification, "own_registration": own_registration,
              "expected_shift_px": list(map(float, shift)) if expected_shift_px is not None else None,
              "sequence_source": sequence["source"], "sequence_header": sequence["header"], "reference_frame": ref_no,
              "hypothesis": hypothesis.as_dict(), "distortion_truth": truth["distortion"], "frames": {}, "status": "inconclusive"}
    votes = {"H_undistorted_truth": 0, "H_distorted_fit": 0, "lists_raw": 0, "lists_corrected": 0,
             "output_undistorted": 0, "output_distorted": 0}
    for image in sequence["images"]:
        pos, no, stem = image["position"], image["index"], image["stem"]
        tf = truth_frames[no]
        p_frame = np.array([[s["x"], s["y"]] for s in tf["rendered_stars"]], dtype=np.float64)
        u_frame = scaled(np.array([[s["x"], s["y"]] for s in tf["undistorted_stars"]], dtype=np.float64))
        entry = {"frame": no}
        # star lists: raw (distorted) or corrected (undistorted)?
        lst_path = work / "cache" / f"{stem}.lst"
        if lst_path.exists():
            detected = np.array([[s["x"], s["y"]] for s in siril_sequence.read_lst(lst_path)["stars"]], dtype=np.float64)
            det_td = np.array([adapter.convert_point(hypothesis, x, y, height) for x, y in detected]) if len(detected) else detected
            raw_frac, raw_med = _nearest_fraction(det_td, p_frame, 0.5)
            cor_frac, cor_med = _nearest_fraction(det_td, u_frame, 0.5)
            entry["star_list"] = {"count": int(len(detected)), "fraction_at_distorted_truth": raw_frac, "median_px_distorted": raw_med,
                                  "fraction_at_undistorted_truth": cor_frac, "median_px_undistorted": cor_med}
            if raw_frac >= 0.8 and cor_frac < 0.5:
                votes["lists_raw"] += 1
            elif cor_frac >= 0.8 and raw_frac < 0.5:
                votes["lists_corrected"] += 1
        # homography: undistorted truth (frame -> reference in undistorted space) or a fit on distorted stars?
        if pos != ref_pos:
            h_td = adapter.convert_matrix(hypothesis, regdata[pos]["H"], height, height)
            m_frame = np.asarray(tf["frame_to_reference_undistorted"], dtype=np.float64)
            # frame -> global reference -> chosen reference in the (possibly magnified) undistorted space
            h_truth = shift_mat @ scale_mat @ (np.linalg.inv(m_ref) @ m_frame) @ np.linalg.inv(scale_mat)
            corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float64)
            err_undist = float(np.max(np.hypot(*(_project(h_td, corners) - _project(h_truth, corners)).T)))
            # what a plain homography fit on the distorted positions would give
            from .registration_models import fit_homography
            p_ref = np.array([[s["x"], s["y"]] for s in truth_frames[ref_no]["rendered_stars"]], dtype=np.float64)
            common = [i for i, s in enumerate(tf["rendered_stars"]) if s["star_id"] in {r["star_id"] for r in truth_frames[ref_no]["rendered_stars"]}]
            ids_ref = {s["star_id"]: i for i, s in enumerate(truth_frames[ref_no]["rendered_stars"])}
            src = np.array([[tf["rendered_stars"][i]["x"], tf["rendered_stars"][i]["y"]] for i in common])
            dst = np.array([p_ref[ids_ref[tf["rendered_stars"][i]["star_id"]]] for i in common])
            h_fit = fit_homography(src, dst)
            err_dist = float(np.max(np.hypot(*(_project(h_td, corners) - _project(h_fit, corners)).T)))
            entry["homography"] = {"corner_error_vs_undistorted_truth_px": err_undist, "corner_error_vs_distorted_fit_px": err_dist,
                                   "siril_H_top_down": h_td.tolist()}
            if err_undist <= 0.15 and err_dist > 0.5:
                votes["H_undistorted_truth"] += 1
            elif err_dist <= 0.15 and err_undist > 0.5:
                votes["H_distorted_fit"] += 1
        # exported frame: stars at the chosen reference's undistorted positions, or at its distorted ones?
        out_path = work / f"r_{stem}.fit"
        if out_path.exists():
            u_ref_chosen = scaled(np.array([[s["x"], s["y"]] for s in truth_frames[ref_no]["undistorted_stars"]], dtype=np.float64))
            p_ref_chosen = np.array([[s["x"], s["y"]] for s in truth_frames[ref_no]["rendered_stars"]], dtype=np.float64)
            frame_shift = shift if pos != ref_pos else np.zeros(2)      # the reference keeps the identity
            und = _centroid_residuals(out_path, u_ref_chosen + frame_shift)
            dis = _centroid_residuals(out_path, p_ref_chosen)
            entry["output"] = {"at_undistorted_reference": und, "at_distorted_reference": dis}
            ui, di = und["interior"], dis["interior"]

            def _clean(stats, n_expected):
                return stats["count"] >= 0.6 * n_expected and (stats["median_px"] or 9) <= 0.2 and (stats["p95_px"] or 9) <= 0.5

            def _off(stats, n_expected):
                return (stats["median_px"] or 9) > 0.6 or stats["count"] < 0.5 * n_expected

            shifted_ok = True
            if expected_shift_px is not None and pos != ref_pos:
                unshifted = _centroid_residuals(out_path, u_ref_chosen)
                entry["output"]["at_unshifted_undistorted_reference"] = unshifted
                shifted_ok = _off(unshifted["interior"], len(u_ref_chosen))
            if _clean(ui, len(u_ref_chosen)) and _off(di, len(p_ref_chosen)) and shifted_ok:
                votes["output_undistorted"] += 1
            elif _clean(di, len(p_ref_chosen)) and _off(ui, len(u_ref_chosen)):
                votes["output_distorted"] += 1
        result["frames"][stem] = entry
    result["votes"] = votes
    n_other = len(sequence["images"]) - 1
    n_out = sum(1 for e in result["frames"].values() if "output" in e)
    applied, ignored = ("own_registration_applied", "own_registration_ignored") if own_registration else ("sip_applied", "sip_ignored")
    if votes["H_undistorted_truth"] == n_other and n_out and votes["output_undistorted"] == n_out:
        result["status"] = applied
    elif votes["H_distorted_fit"] == n_other and (n_out == 0 or votes["output_distorted"] == n_out):
        result["status"] = ignored
    elif own_registration and n_out and votes["output_undistorted"] == 0:
        result["status"] = ignored
    result["reason"] = f"H votes {votes['H_undistorted_truth']}/{votes['H_distorted_fit']} of {n_other}; output votes {votes['output_undistorted']}/{votes['output_distorted']} of {n_out}"
    if simulated:
        result["status"] += "_SIMULATED"
    return result


def _project(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    v = np.hstack([xy, np.ones((len(xy), 1))]) @ h.T
    return v[:, :2] / v[:, 2:3]


# --------------------------------------------------------------------------- #
# Own registration written into the sequence (RUN_LOG §22)                     #
# --------------------------------------------------------------------------- #

def own_registration_sequence(work: Path, truth: dict, *, distortion_file: str, magnification: float = 1.0,
                              shift_px: tuple[float, float] | None = None, hypothesis: adapter.CoordinateHypothesis | None = None,
                              layer: int = 1, zero_quality_frames: tuple[int, ...] = ()) -> dict:
    """Rewrite ``work/light_.seq`` with the truth homographies (frame -> Siril's chosen reference, in the magnified
    undistorted space, optionally composed with ``shift_px``) in Siril's convention, plus the ``D`` line.

    ``zero_quality_frames`` lists image numbers whose six quality fields are
    written as zeros (like a frame Siril could not match) to check that
    ``seqapplyreg`` still exports them.  Returns the written matrices (top-down,
    frame -> reference) keyed by image stem and the re-parsed sequence.
    """
    work = Path(work)
    hypothesis = hypothesis or adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
    height = truth["height"]
    distortion = RadialDistortion(**truth["distortion"])
    scale_mat = np.array([[magnification, 0.0, (1 - magnification) * distortion.cx],
                          [0.0, magnification, (1 - magnification) * distortion.cy], [0.0, 0.0, 1.0]])
    shift = np.array(shift_px if shift_px is not None else (0.0, 0.0), dtype=np.float64)
    shift_mat = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
    sequence = siril_sequence.read_seq(work / "light_.seq")
    ref_pos = sequence["header"]["reference_image"]
    ref_no = sequence["images"][ref_pos]["index"]
    truth_frames = {f["frame"]: f for f in truth["frames"]}
    m_ref = np.asarray(truth_frames[ref_no]["frame_to_reference_undistorted"], dtype=np.float64)
    homographies, written, quality = {}, {}, {}
    for image in sequence["images"]:
        pos, no = image["position"], image["index"]
        if pos == ref_pos:
            h_td = np.eye(3)
        else:
            m_frame = np.asarray(truth_frames[no]["frame_to_reference_undistorted"], dtype=np.float64)
            h_td = shift_mat @ scale_mat @ (np.linalg.inv(m_ref) @ m_frame) @ np.linalg.inv(scale_mat)
        homographies[pos] = adapter.to_siril_matrix(hypothesis, h_td, height, height)
        written[image["stem"]] = h_td.tolist()
        if no in zero_quality_frames:
            quality[pos] = [0, 0, 0, 0, 0, 0]
    parsed = siril_sequence.write_registration(work / "light_.seq", homographies, layer=layer, distortion_file=distortion_file,
                                               quality=quality, selected={pos: True for pos in homographies})
    return {"reference_frame": ref_no, "shift_px": shift.tolist(), "magnification": magnification, "homographies_top_down": written,
            "sequence": {k: parsed[k] for k in ("header", "images")}, "distortion_file": distortion_file}


def simulate_seqapplyreg(truth: dict, work: Path, *, magnification: float = 1.0) -> None:
    """OFFLINE self-check of the own-registration analysis: render ``r_light_*.fit`` from the sequence's H
    (frame stars in the magnified undistorted space mapped by the written matrices).  Never evidence."""
    work = Path(work)
    hypothesis = adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
    height, width = truth["height"], truth["width"]
    distortion = RadialDistortion(**truth["distortion"])
    sequence = siril_sequence.read_seq(work / "light_.seq")
    truth_frames = {f["frame"]: f for f in truth["frames"]}
    for image in sequence["images"]:
        h_td = adapter.convert_matrix(hypothesis, sequence["regdata"][1][image["position"]]["H"], height, height)
        tf = truth_frames[image["index"]]
        u = np.array([[s["x"], s["y"]] for s in tf["undistorted_stars"]], dtype=np.float64)
        u = distortion.centre + magnification * (u - distortion.centre)
        v = np.hstack([u, np.ones((len(u), 1))]) @ h_td.T
        out = v[:, :2] / v[:, 2:3]
        layout = {"width": width, "height": height,
                  "stars": [{"star_id": s["star_id"], "x": float(x), "y": float(y), "amplitude": 20000.0, "sigma": 1.6}
                            for s, (x, y) in zip(tf["undistorted_stars"], out)]}
        frame, _ = render_frame(layout, np.eye(3), seed=200 + image["index"])
        write_fits_rgb16(work / f"r_{image['stem']}.fit", frame, comments=["SIMULATED seqapplyreg output from the sequence's own H"])


# --------------------------------------------------------------------------- #
# Offline simulation (labelled; never a proof)                                 #
# --------------------------------------------------------------------------- #

def simulate_siril_outputs(truth: dict, work: Path, *, applied: bool, noise_px: float = 0.03, seed: int = 9) -> None:
    """Write ``light_.seq``, ``cache/*.lst`` and ``r_light_*.fit`` as Siril would with (``applied``) or without the SIP."""
    work = Path(work)
    (work / "cache").mkdir(parents=True, exist_ok=True)
    hypothesis = adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
    height, width = truth["height"], truth["width"]
    t_siril_to_td = hypothesis.affine(height)
    t_td_to_siril = np.linalg.inv(t_siril_to_td)
    rng = np.random.default_rng(seed)
    frames = truth["frames"]
    from .registration_models import fit_homography
    lines = ["#Siril sequence file (SIMULATED by starstack.siril_sip_proof, not Siril output)",
             f"S 'light_' 1 {len(frames)} {len(frames)} 5 0 6 0 0 0", "L 3"] + [f"I {i + 1} 1" for i in range(len(frames))]
    ref = frames[0]
    ids_ref = {s["star_id"]: (s["x"], s["y"]) for s in ref["rendered_stars"]}
    for frame in frames:
        if applied:
            h_td = np.asarray(frame["frame_to_reference_undistorted"], dtype=np.float64)
        else:
            common = [s for s in frame["rendered_stars"] if s["star_id"] in ids_ref]
            src = np.array([[s["x"], s["y"]] for s in common])
            dst = np.array([ids_ref[s["star_id"]] for s in common])
            h_td = fit_homography(src, dst)
        h = t_td_to_siril @ h_td @ t_siril_to_td
        h = h / h[2, 2]
        lines.append("R1 3.5 3.5 0.8 0 0.02 %d H %s" % (len(frame["rendered_stars"]), " ".join(f"{v:.9g}" for v in h.reshape(-1))))
    (work / "light_.seq").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for frame in frames:
        stem = f"light_{frame['frame']:05d}"
        rows = ["# %d stars found using the following parameters:" % len(frame["rendered_stars"]),
                "# sigma=1.00 roundness=0.50 radius=5 relax=0 profile=0 minbeta=1.5 max_stars=2000 layer=1 minA=0.00 maxA=0.00 maxR=1.00 (SIMULATED)",
                "# star#\tlayer\tB\tA\tbeta\tX\tY\tFWHMx [px]\tFWHMy [px]\tFWHMx [\"]\tFWHMy [\"]\tangle\tRMSE\tmag\tSat\tProfile\tRA\tDec"]
        for n, star in enumerate(sorted(frame["rendered_stars"], key=lambda s: -s["amplitude"]), 1):
            v = t_td_to_siril @ np.array([star["x"], star["y"], 1.0])
            rows.append(f"{n}\t1\t800.0\t{star['amplitude']:.3f}\t-1.00\t{v[0] + rng.normal(0, noise_px):10.2f}\t{v[1] + rng.normal(0, noise_px):10.2f}"
                        "\t3.50\t3.40\t0.0\t0.0\t0.0\t1.0e+002\t-10.00\t0\tGaussian\t0.0\t0.0")
        (work / "cache" / f"{stem}.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")
        # exported frame: every frame's stars land on the reference's positions (undistorted if applied)
        key = "undistorted_stars" if applied else "rendered_stars"
        layout = {"width": width, "height": height,
                  "stars": [{"star_id": s["star_id"], "x": s["x"], "y": s["y"], "amplitude": 20000.0, "sigma": 1.6} for s in ref[key]]}
        image, _ = render_frame(layout, np.eye(3), seed=100 + frame["frame"])
        write_fits_rgb16(work / f"r_{stem}.fit", image, comments=["SIMULATED registered output"])
