"""Per-job lens-distortion calibration for the product flow (never raises).

After a transforms-only two-pass registration (``register light -2pass``,
which also writes the star lists), this module fits one sensor-fixed radial
distortion shared by all frames (``registration_distortion``), checks that
it *improves* the conditional corner residuals over plain per-frame
homographies, re-associates every frame's full star list under the model
(which is also how frames Siril could not match get their homography), and
writes the model as a Siril distortion file (``distortion.wcs``,
``sip_header``) with the global magnification that keeps a barrel lens'
corners inside the native frame.  It also gives every frame its homography
under the final model (``registration``) and, through
``write_registration_sequence``, writes them into the sequence file in
Siril's convention together with the ``D`` line, so that the product only
runs ``seqapplyreg`` + ``stack``: Siril undistorts and resamples, the
registration itself is ours.  The stacked sky is warped back to the native
geometry afterwards (``core``).

Evidence: RUN_LOG §17–§19 — on the Z8 20 mm sequence the term removes the
span-dependent residual (corners 2–6 px -> ≤0.5 px over 21 minutes) and
Siril 1.4.4 honours the SIP file; §22 — Siril's own star matching still
fails on frames 10+ minutes from the reference even with the SIP file
(50–160 pairs, 1–23 px corner errors, dropped frames), while the
re-association under the model registers all 48 frames to ≤0.5 px, and
``seqapplyreg`` follows matrices written into the sequence file (synthetic
proof with a known shift).  Decisions (user, 2026-09-17/18): fit per job
rather than keep lens calibrations; output geometry stays native; the
product writes its own registration.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import fits_crop, siril_sequence
from . import registration_adapter as adapter
from . import registration_distortion as rd
from .registration_models import evaluate_model, stratified_split, fit_homography, apply_matrix
from .sip_header import DEFAULT_INVERSE_ORDER, sip_cards, sip_forward_coefficients, sip_inverse_coefficients, write_wcs_header_file
from .siril import Cancelled

__all__ = ["CALIBRATION_VERSION", "DISTORTION_WCS", "DISTORTION_JSON", "calibrate", "load_calibration",
           "write_registration_sequence", "native_from_undistorted"]

CALIBRATION_VERSION = "1.4.1"
QUALITY_GATE_VERSION = "1.0.1"
# RUN_LOG §27: frames whose detection-stage quality is far off the sequence's own median do not enter the sky stack.
# Thresholds are relative (median of the sequence), so they carry to other scenes; the gate acts only with enough frames
# for a median to mean something, otherwise it only reports.  The reference frame is never excluded.
DEFAULT_QUALITY_GATE = {"enabled": True, "min_frames": 8, "wfwhm_ratio": 1.5, "fwhm_ratio": 1.3, "nstars_ratio": 0.5, "holdout_rms_ratio": 2.0}


def quality_gate(template: dict, ref_stem: str, *, thresholds: dict | None = None, holdout_rms: dict | None = None) -> dict:
    """Flag frames whose Siril quality fields (FWHM, weighted FWHM, star count) or conditional registration RMS are far off the
    sequence median; returns the per-frame metrics, ratios, flags and the frames to exclude from the sky stack."""
    thresholds = dict(DEFAULT_QUALITY_GATE, **(thresholds or {}))
    metrics = {}
    for image in template["images"]:
        q = template["quality"].get(str(image["position"]))
        if q and q[5] > 0:
            metrics[image["stem"]] = {"fwhm": float(q[0]), "wfwhm": float(q[1]), "roundness": float(q[2]), "nstars": int(q[5])}
    holdout_rms = {k: float(v) for k, v in (holdout_rms or {}).items() if v is not None and np.isfinite(v)}
    medians = {}
    for key in ("fwhm", "wfwhm", "nstars"):
        values = [m[key] for m in metrics.values() if m[key] > 0]
        medians[key] = float(np.median(values)) if values else None
    medians["holdout_rms"] = float(np.median(list(holdout_rms.values()))) if holdout_rms else None
    frames, excluded = {}, []
    enabled = bool(thresholds.get("enabled", True)) and len(metrics) >= int(thresholds["min_frames"])
    for stem, m in metrics.items():
        ratios, flags = {}, []
        if medians["wfwhm"]:
            ratios["wfwhm"] = m["wfwhm"] / medians["wfwhm"]
            if ratios["wfwhm"] > thresholds["wfwhm_ratio"]:
                flags.append(f"wfwhm {ratios['wfwhm']:.2f}x median")
        if medians["fwhm"]:
            ratios["fwhm"] = m["fwhm"] / medians["fwhm"]
            if ratios["fwhm"] > thresholds["fwhm_ratio"]:
                flags.append(f"fwhm {ratios['fwhm']:.2f}x median")
        if medians["nstars"]:
            ratios["nstars"] = m["nstars"] / medians["nstars"]
            if ratios["nstars"] < thresholds["nstars_ratio"]:
                flags.append(f"stars {ratios['nstars']:.2f}x median")
        if stem in holdout_rms and medians["holdout_rms"]:
            ratios["holdout_rms"] = holdout_rms[stem] / medians["holdout_rms"]
            if ratios["holdout_rms"] > thresholds["holdout_rms_ratio"]:
                flags.append(f"conditional validation rms {ratios['holdout_rms']:.2f}x median")
        frames[stem] = {**m, "holdout_rms_px": holdout_rms.get(stem), "ratios": {k: round(v, 3) for k, v in ratios.items()}, "flags": flags}
        if flags and enabled and stem != ref_stem:
            excluded.append({"stem": stem, "reasons": flags})
    reference_flagged = bool(frames.get(ref_stem, {}).get("flags"))
    return {"version": QUALITY_GATE_VERSION, "residual_semantics": "conditional_validation_not_independent",
            "enabled": bool(thresholds.get("enabled", True)), "applied": enabled, "min_frames": int(thresholds["min_frames"]),
            "thresholds": {k: thresholds[k] for k in ("wfwhm_ratio", "fwhm_ratio", "nstars_ratio", "holdout_rms_ratio")}, "frames_with_quality": len(metrics),
            "medians": medians, "frames": frames, "excluded": excluded, "excluded_frames": [e["stem"] for e in excluded],
            "flagged_not_applied": [stem for stem, f in frames.items() if f["flags"] and not enabled], "reference_flagged": reference_flagged}
DISTORTION_WCS = "distortion.wcs"
DISTORTION_JSON = "distortion.json"


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _spread(items: list, count: int) -> list:
    """Up to ``count`` items evenly spread over the (span-sorted) list."""
    if len(items) <= count:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, count)).astype(int))
    return [items[i] for i in idx]


def _corner_medians(model: dict, frames: dict, masks: dict, width: int, height: int, hs=None) -> tuple[float | None, float | None]:
    evaluation = rd.evaluate_frames(model, frames, masks, width, height, hs=hs)
    corner = _median([e["residuals"]["validation"]["corner"]["rms_px"] for e in evaluation.values()])
    overall = _median([e["residuals"]["validation"]["overall"]["rms_px"] for e in evaluation.values()])
    return corner, overall


def calibrate(work: str | Path, *, hypothesis: adapter.CoordinateHypothesis | None = None, radius: float = 8.0,
              max_fit_frames: int = 24, min_frames: int = 3, min_pairs: int = 200, required_improvement: float = 0.2,
              reassociation_radii=(20.0, 10.0, 5.0, 3.0), registration_max_rms_px: float = 1.0,
              sip_inverse_order: int = DEFAULT_INVERSE_ORDER, distortion_direction: str = "u_to_p", log=None, check=None,
              quality_gate_thresholds: dict | None = None) -> dict:
    """Fit, validate and write the distortion file; returns a JSON-serialisable summary (``status`` says what happened).

    ``quality_gate_thresholds`` (``DEFAULT_QUALITY_GATE`` by default; ``{"enabled": False}``
    to report only) excludes frames far off the sequence's own detection
    quality from the fit and from the sky stack (RUN_LOG §27); the summary's
    ``quality_gate`` lists every frame's metrics and the excluded ones.

    ``check`` (optional callable) is the job's cancellation probe: it is
    called between the stages and inside the per-frame loops and fit
    iterations (about every few seconds on a 144-frame job), and whatever it
    raises propagates out of ``calibrate`` unchanged — a cancellation is not
    a calibration failure, so it is neither swallowed into ``status``
    ``unavailable`` nor recorded in ``distortion.json`` (nothing is written
    before the fit is complete).

    When the model is ``fitted`` the summary also carries ``registration``
    (every frame's homography under the final model, top-down frame ->
    reference in the magnified undistorted grid, plus Siril's stored form and
    conditional validation residuals; frames with fewer than ``min_pairs`` pairs or a
    conditional RMS above ``registration_max_rms_px`` are ``not_registered``)
    and ``sequence_template`` (what ``write_registration_sequence`` needs to
    rebuild ``light_.seq`` without the probe's file).  ``distortion_direction``
    ``u_to_p`` (default, RUN_LOG §22) parameterises the radial polynomial on
    the undistorted side so that the AP/BP inverse Siril applies on export is
    exact at order 5; ``p_to_u`` is the original form (AP/BP least-squares,
    0.96 px at the extreme corners of the Z8 model).
    """
    started = time.monotonic()
    work = Path(work)
    log = log or (lambda *_: None)
    check = check or (lambda: None)
    summary = {"calibration_version": CALIBRATION_VERSION, "distortion_version": rd.DISTORTION_VERSION, "status": "unavailable",
               "file": None, "reason": None}
    try:
        check()
        hypothesis = hypothesis or adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
        seq_path = work / "light_.seq"
        if not seq_path.exists():
            summary["reason"] = "no sequence file after the probe registration"
            return _finish(work, summary, started)
        sequence = siril_sequence.read_seq(seq_path)
        images = sequence["images"]
        frames_fits = sorted(work.glob("light_*.fit"))
        if not frames_fits:
            summary["reason"] = "no converted frames"
            return _finish(work, summary, started)
        header, _ = fits_crop.read_header(frames_fits[0])
        width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
        regdata = sequence["regdata"].get(1)
        if regdata is None:
            summary["reason"] = "sequence has no registration data"
            return _finish(work, summary, started)
        ref_pos = sequence["header"]["reference_image"]
        ref_stem = images[ref_pos]["stem"]
        dropped = [im["stem"] for im in images if not im["selected"] or abs(np.linalg.det(np.asarray(regdata[im["position"]]["H"]))) < 1e-12]
        lists = {}
        for image in images:
            path = work / "cache" / f"{image['stem']}.lst"
            if path.exists():
                lists[image["stem"]] = siril_sequence.read_lst(path)
        if ref_stem not in lists:
            summary["reason"] = "no star list for the reference frame"
            return _finish(work, summary, started)
        positions = [im["position"] for im in images if im["position"] != ref_pos and im["stem"] in lists and im["stem"] not in dropped]
        template = _sequence_template(sequence, regdata, lists)
        gate = quality_gate(template, ref_stem, thresholds=quality_gate_thresholds)
        gate_excluded = set(gate["excluded_frames"])
        if gate_excluded:
            log(f"quality gate: {len(gate_excluded)} frame(s) kept out of the fit and the stack: " + "; ".join(f"{e['stem']} ({', '.join(e['reasons'])})" for e in gate["excluded"]))
        positions = [pos for pos in positions if images[pos]["stem"] not in gate_excluded]
        summary.update({"reference": ref_stem, "width": width, "height": height, "frames_registered": len(positions) + 1,
                        "frames_dropped_by_siril": dropped, "star_lists": len(lists),
                        "hypothesis": hypothesis.name, "sequence_template": template, "quality_gate": gate})
        if len(positions) < min_frames:
            summary["reason"] = f"only {len(positions)} registered frames with star lists"
            return _finish(work, summary, started)
        document, _ = adapter.build_document(sequence=sequence, star_lists=lists, width=width, height=height, hypothesis=hypothesis,
                                             positions=positions, layer=1, radius=radius)
        frames = {fid: f for fid, f in rd.frames_from_document(document).items() if len(f["src"]) >= min_pairs}
        summary["frames_with_pairs"] = len(frames)
        if len(frames) < min_frames:
            summary["reason"] = f"only {len(frames)} frames with at least {min_pairs} associated pairs"
            return _finish(work, summary, started)
        # round 1: fit on a span-spread subset of the Siril-associated pairs
        ordered = sorted(frames, key=lambda fid: frames[fid]["span"])
        subset_ids = _spread(ordered, max_fit_frames)
        subset = {fid: frames[fid] for fid in subset_ids}
        masks = {fid: stratified_split(f["dst"], width, height) for fid, f in subset.items()}
        check()
        baseline = {"kind": "baseline", "H": rd._fit_homographies(subset, masks, lambda p: p)}
        base_corner, base_overall = _corner_medians(baseline, subset, masks, width, height)
        radial = rd.fit_joint_radial(subset, width, height, masks=masks, direction=distortion_direction, check=check)
        r1_corner, r1_overall = _corner_medians(radial, subset, masks, width, height)
        log(f"calibration round 1: {len(subset)} frames, corner median {base_corner:.3f} -> {r1_corner:.3f} px")
        # round 2: re-associate full star lists under the model (also recovers frames Siril dropped), refit on them
        lists_xy = {stem: np.array([adapter.convert_point(hypothesis, s["x"], s["y"], height) for s in lst["stars"]], dtype=np.float64).reshape(-1, 2)
                    for stem, lst in lists.items()}
        spans = {stem: rd.span_of(stem, ref_stem) for stem in lists_xy}
        reassoc = rd.reassociate_sequence(radial["distortion"], radial["H"], spans, lists_xy, ref_stem, width, height, radii=reassociation_radii,
                                          check=check)
        associated = {fid: e for fid, e in reassoc.items() if e["status"] == "associated" and e["matched"] >= min_pairs}
        frames2 = {}
        ref_xy = lists_xy[ref_stem]
        for fid in _spread(sorted((f for f in associated if f not in gate_excluded), key=lambda f: associated[f]["span"]), max_fit_frames):
            entry = associated[fid]
            h = np.asarray(entry["H_undistorted"])
            d = radial["distortion"]
            pred = d.inverse(apply_matrix(h, d.apply(lists_xy[fid])))
            pairs = rd.mutual_nearest(pred, ref_xy, reassociation_radii[-1])
            if len(pairs) >= min_pairs:
                frames2[fid] = {"span": entry["span"], "src": lists_xy[fid][pairs[:, 0]], "dst": ref_xy[pairs[:, 1]], "H_siril": h}
        check()
        if len(frames2) >= min_frames:
            masks2 = {fid: stratified_split(f["dst"], width, height) for fid, f in frames2.items()}
            final = rd.fit_joint_radial(frames2, width, height, masks=masks2, direction=distortion_direction, check=check)
            base2 = {"kind": "baseline", "H": rd._fit_homographies(frames2, masks2, lambda p: p)}
            base2_corner, base2_overall = _corner_medians(base2, frames2, masks2, width, height)
            fin_corner, fin_overall = _corner_medians(final, frames2, masks2, width, height)
            rounds = 2
        else:
            final, base2_corner, base2_overall, fin_corner, fin_overall, rounds = radial, base_corner, base_overall, r1_corner, r1_overall, 1
        distortion = final["distortion"]
        half_diag = float(np.hypot(width, height) / 2.0)
        corner_shift = distortion.corner_shift_px(width, height)
        improved = (base2_corner is not None and fin_corner is not None and fin_corner <= (1.0 - required_improvement) * base2_corner)
        sane = np.isfinite(corner_shift) and 0.0 < corner_shift < 0.25 * half_diag and abs(distortion.cx - (width - 1) / 2) < 0.25 * width \
            and abs(distortion.cy - (height - 1) / 2) < 0.25 * height
        magnification = distortion.fitting_magnification(width, height)
        model = distortion.with_magnification(magnification)
        check()
        registration = _per_frame_registration(distortion, magnification, reassoc, lists_xy, ref_stem, images, width, height,
                                               hypothesis, min_pairs=min_pairs, radius=reassociation_radii[-1],
                                               max_rms_px=registration_max_rms_px, check=check)
        check()
        holdout = {stem: (f.get("validation") or {}).get("overall_rms_px") for stem, f in registration["frames"].items() if f.get("status") == "registered" and stem != ref_stem}
        gate = quality_gate(template, ref_stem, thresholds=quality_gate_thresholds, holdout_rms=holdout)
        summary["quality_gate"] = gate
        _apply_quality_gate(registration, gate)
        if gate["excluded_frames"]:
            log(f"quality gate: excluded from the sky stack {gate['excluded_frames']}")
        per_frame = {fid: {"span": e["span"], "status": e["status"], "initial_homography": e.get("initial_homography"),
                           "matched": e.get("matched"), "frame_count": e.get("frame_count"), "matched_fraction": e.get("matched_fraction"),
                           "corner_rms_px": e.get("residuals", {}).get("validation", {}).get("corner", {}).get("rms_px"),
                           "overall_rms_px": e.get("residuals", {}).get("validation", {}).get("overall", {}).get("rms_px")}
                     for fid, e in reassoc.items()}
        summary.update({
            "parameters": model.as_dict(), "corner_shift_px": corner_shift, "magnification": magnification,
            "fit_rounds": rounds, "fit_frames": sorted(frames2) if rounds == 2 else sorted(subset),
            "validation_corner_rms_median_px": {"homography_only": base2_corner, "with_distortion": fin_corner, "round1_homography_only": base_corner, "round1_with_distortion": r1_corner},
            "validation_overall_rms_median_px": {"homography_only": base2_overall, "with_distortion": fin_overall},
            "reassociation": {"frames": per_frame,
                              "matched_fraction_median": _median([e.get("matched_fraction") for e in reassoc.values()]),
                              "corner_rms_median_px": _median([e.get("corner_rms_px") for e in per_frame.values()]),
                              "recovered_dropped_frames": [fid for fid in dropped if reassoc.get(fid, {}).get("status") == "associated"]},
            "required_improvement": required_improvement, "improved": bool(improved), "sane": bool(sane),
            "registration": registration})
        if not sane:
            summary.update(status="unavailable", reason=f"fitted distortion is not plausible (corner shift {corner_shift:.1f} px, centre {distortion.cx:.0f},{distortion.cy:.0f})")
            return _finish(work, summary, started)
        if not improved:
            summary.update(status="no_improvement", reason=f"corner median {base2_corner:.3f} -> {fin_corner:.3f} px is less than {required_improvement:.0%} better")
            return _finish(work, summary, started)
        write_wcs_header_file(work / DISTORTION_WCS, sip_cards(model, width, height, inverse_order=sip_inverse_order), width, height)
        inverse_terms = sip_inverse_coefficients(model, width, height, sip_inverse_order)
        forward_terms = sip_forward_coefficients(model, width, height, sip_inverse_order)
        summary.update(status="fitted", file=DISTORTION_WCS, reason=None, distortion_direction=model.direction,
                       sip_inverse={"order": int(inverse_terms["order"]), "exact": bool(inverse_terms["exact"]), "max_fit_error_px": inverse_terms["max_fit_error_px"]},
                       sip_forward={"order": int(forward_terms["order"]), "exact": bool(forward_terms["exact"]), "max_fit_error_px": forward_terms["max_fit_error_px"]})
        log(f"calibration: corner median {base2_corner:.3f} -> {fin_corner:.3f} px, magnification {magnification:.4f}, {DISTORTION_WCS} written")
        return _finish(work, summary, started)
    except Cancelled:  # the job was cancelled: not a calibration outcome, nothing recorded
        raise
    except Exception as exc:  # calibration is an improvement, never a reason for a job to fail
        summary.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}", file=None)
        return _finish(work, summary, started)


def _apply_quality_gate(registration: dict, gate: dict) -> None:
    """Turn the gate's exclusions into ``excluded_quality`` frames of the registration (not stacked, matrices kept)."""
    excluded = {e["stem"]: e["reasons"] for e in gate.get("excluded", [])}
    for stem, reasons in excluded.items():
        frame = registration["frames"].get(stem)
        if frame is None:
            continue
        frame["status"] = "excluded_quality"
        frame["reason"] = "quality gate: " + "; ".join(reasons)
    registration["registered_frames"] = [s for s in registration["registered_frames"] if s not in excluded]
    registration["registered"] = len(registration["registered_frames"])
    registration["unregistered_frames"] = sorted(set(registration["unregistered_frames"]) | set(excluded))
    registration["quality_excluded_frames"] = sorted(excluded)
    registered = [f["validation"] for f in registration["frames"].values() if f.get("status") == "registered" and f.get("validation")]
    registration["validation_overall_rms_median_px"] = _median([v["overall_rms_px"] for v in registered])
    registration["validation_corner_rms_median_px"] = _median([v["corner_rms_px"] for v in registered])


def _sequence_template(sequence: dict, regdata: list, lists: dict) -> dict:
    """What is needed to rebuild the sequence file: header, layers, image list and per-position quality fields.

    Frames Siril could not match have all-zero quality fields; those get
    plausible values from their star list (median FWHM, roundness,
    background, count) so the file looks like Siril's own for every frame.
    """
    quality = {}
    for image in sequence["images"]:
        pos = image["position"]
        r = regdata[pos] if pos < len(regdata) else None
        fields = [r["fwhm"], r["weighted_fwhm"], r["roundness"], r["quality"], r["background_lvl"], r["number_of_stars"]] if r else [0] * 6
        if (not r or r["number_of_stars"] == 0) and image["stem"] in lists:
            stars = lists[image["stem"]]["stars"]
            fx = np.array([s.get("fwhmx_px") or 0.0 for s in stars], dtype=np.float64)
            fy = np.array([s.get("fwhmy_px") or 0.0 for s in stars], dtype=np.float64)
            ok = (fx > 0) & (fy > 0)
            fwhm = float(np.median(np.maximum(fx[ok], fy[ok]))) if ok.any() else 0.0
            roundness = float(np.median(np.minimum(fx[ok], fy[ok]) / np.maximum(fx[ok], fy[ok]))) if ok.any() else 0.0
            background = float(np.median([s.get("B") or 0.0 for s in stars])) if stars else 0.0
            fields = [fwhm, fwhm, roundness, 0.0, background, len(stars)]
        quality[str(pos)] = [float(v) for v in fields]
    header = dict(sequence["header"])
    return {"header": header, "layers": sequence.get("layers"), "images": [{"index": im["index"], "position": im["position"], "stem": im["stem"]}
                                                                          for im in sequence["images"]], "quality": quality}


def _per_frame_registration(distortion: rd.RadialDistortion, magnification: float, reassoc: dict, lists_xy: dict, ref_stem: str,
                            images: list, width: int, height: int, hypothesis, *, min_pairs: int, radius: float, max_rms_px: float,
                            check=None) -> dict:
    """Every frame's homography under the final model, from the model's own star pairs (RUN_LOG §22).

    Pairs: mutual nearest neighbours at ``radius`` under the re-association's
    matrix; the homography is refitted in the final model's undistorted
    space, once on the stratified fit half (conditional residuals reported) and
    once on all pairs (the matrix used).  Stored both as top-down frame ->
    reference in the *magnified* undistorted grid (what Siril's ``r_`` frames
    are in) and in Siril's own convention for the sequence file.
    """
    scale_mat = np.array([[magnification, 0.0, (1 - magnification) * distortion.cx],
                          [0.0, magnification, (1 - magnification) * distortion.cy], [0.0, 0.0, 1.0]])
    ref_xy = lists_xy[ref_stem]
    frames, registered, unregistered = {}, [], []
    for image in images:
        if check is not None:
            check()
        stem = image["stem"]
        entry = {"span": int(rd.span_of(stem, ref_stem)), "status": "not_registered", "reason": None, "pairs": 0, "validation": None}
        if stem == ref_stem:
            h_mag = np.eye(3)
            entry.update(status="registered", reason="reference", pairs=int(len(ref_xy)))
        else:
            e = reassoc.get(stem)
            if e is None or e.get("status") != "associated" or "H_undistorted" not in e or stem not in lists_xy:
                entry["reason"] = (e or {}).get("status", "no star list")
                frames[stem] = entry
                unregistered.append(stem)
                continue
            src = lists_xy[stem]
            pred = distortion.inverse(apply_matrix(np.asarray(e["H_undistorted"]), distortion.apply(src)))
            pairs = rd.mutual_nearest(pred, ref_xy, radius)
            entry["pairs"] = int(len(pairs))
            if len(pairs) < max(min_pairs, 12):
                entry["reason"] = f"only {len(pairs)} pairs at {radius:g} px"
                frames[stem] = entry
                unregistered.append(stem)
                continue
            s_xy, d_xy = src[pairs[:, 0]], ref_xy[pairs[:, 1]]
            u_src, u_dst = distortion.apply(s_xy), distortion.apply(d_xy)
            mask = stratified_split(d_xy, width, height)
            h_fit = fit_homography(u_src[mask], u_dst[mask])
            residual = evaluate_model(distortion.inverse(apply_matrix(h_fit, u_src)), d_xy, mask, width, height)["validation"]
            entry["validation"] = {"count": residual["overall"]["count"], "overall_rms_px": residual["overall"]["rms_px"],
                                   "corner_rms_px": residual["corner"]["rms_px"], "corner_count": residual["corner"]["count"],
                                   "max_px": residual["overall"]["max_px"]}
            rms = residual["overall"]["rms_px"]
            if rms is None or not np.isfinite(rms) or rms > max_rms_px:
                entry["reason"] = f"conditional validation RMS {rms} px above {max_rms_px} px"
                frames[stem] = entry
                unregistered.append(stem)
                continue
            h_all = fit_homography(u_src, u_dst)
            h_mag = scale_mat @ h_all @ np.linalg.inv(scale_mat)
            h_mag = h_mag / h_mag[2, 2]
            entry.update(status="registered", reason=None)
        entry["H_top_down_magnified"] = h_mag.tolist()
        entry["H_siril"] = adapter.to_siril_matrix(hypothesis, h_mag, height, height).tolist()
        frames[stem] = entry
        registered.append(stem)
    return {"reference": ref_stem, "hypothesis": hypothesis.name, "magnification": float(magnification),
            "min_pairs": int(min_pairs), "max_rms_px": float(max_rms_px), "pair_radius_px": float(radius),
            "registered": len(registered), "registered_frames": registered, "unregistered_frames": unregistered,
            "validation_overall_rms_median_px": _median([f["validation"]["overall_rms_px"] for f in frames.values() if f.get("validation")]),
            "validation_corner_rms_median_px": _median([f["validation"]["corner_rms_px"] for f in frames.values() if f.get("validation")]),
            "frames": frames}


def write_registration_sequence(work: str | Path, calibration: dict, *, layer: int = 1) -> dict:
    """Write ``work/light_.seq`` from the calibration's ``registration`` and ``sequence_template`` (RUN_LOG §22).

    Rebuilds the whole file (header, layers, image list, ``D<layer> 2
    distortion.wcs``, one ``R<layer>`` line per image): registered frames get
    their Siril-convention matrix and quality fields, the others Siril's own
    all-zero form and a cleared selection flag.  Raises ``ValueError`` when the
    calibration has no registration.  Returns what was written.
    """
    work = Path(work)
    registration = calibration.get("registration") or {}
    template = calibration.get("sequence_template")
    if not registration.get("frames") or not template:
        raise ValueError("calibration has no registration to write")
    if calibration.get("status") != "fitted" or not (work / DISTORTION_WCS).is_file():
        raise ValueError("distortion file missing: nothing to register against")
    header = template["header"]
    images = template["images"]
    homographies, selected, quality = {}, {}, {}
    for image in images:
        pos, stem = int(image["position"]), image["stem"]
        frame = registration["frames"].get(stem) or {}
        if frame.get("status") == "registered" and frame.get("H_siril"):
            homographies[pos] = frame["H_siril"]
            selected[pos] = True
            quality[pos] = template["quality"].get(str(pos), [0] * 6)
        else:
            selected[pos] = False
    n_selected = sum(1 for v in selected.values() if v)
    lines = ["#Siril sequence file. Contains list of images, selection, registration data and statistics",
             "#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version variable_size fz_flag drizzle",
             f"S '{header['name']}' {header['start_index']} {header['nb_images']} {n_selected} {header['fixed_len']} {header['reference_image']} "
             f"{header['version']} {header['variable_size']} {header['fz_flag']} {header['drizzle']}"]
    if template.get("layers"):
        lines.append(f"L {template['layers']}")
    lines += [f"I {image['index']} {1 if selected[int(image['position'])] else 0}" for image in images]
    lines.append(f"D{layer} 2 {DISTORTION_WCS}")
    for image in images:
        pos = int(image["position"])
        if pos in homographies:
            h = [float(v) for row in homographies[pos] for v in row]
            fields = " ".join(siril_sequence._fmt_number(v) for v in quality[pos])
        else:
            h = [0.0] * 9
            fields = "0 0 0 0 0 0"
        lines.append(f"R{layer} {fields} H " + " ".join(siril_sequence._fmt_number(v) for v in h))
    text = "\n".join(lines) + "\n"
    parsed = siril_sequence.parse_seq_text(text)      # what Siril will read must parse the way we read Siril's own files
    path = work / "light_.seq"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    import os
    os.replace(tmp, path)
    return {"file": str(path), "registered": n_selected, "images": len(images), "reference_position": parsed["header"]["reference_image"],
            "reference": registration.get("reference"), "unregistered_frames": list(registration.get("unregistered_frames") or []),
            "distortion_file": DISTORTION_WCS, "layer": layer}


def _finish(work: Path, summary: dict, started: float) -> dict:
    summary["validation_independence"] = {
        "status": "insufficient_data", "independent": False,
        "scope": "conditional validation after association; no globally isolated star identities",
        "reason": "probe initialization, reassociation and model/frame selection may reuse validation observations",
        "legacy_field_names": "validation and holdout_rms retained for compatibility; not independent test evidence",
    }
    summary["seconds"] = round(time.monotonic() - started, 1)
    try:
        (work / DISTORTION_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return summary


def load_calibration(work: str | Path) -> dict | None:
    path = Path(work) / DISTORTION_JSON
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def native_from_undistorted(undistorted_rgb_bottom_up: np.ndarray, parameters: dict, *, kernel: str | None = None) -> np.ndarray:
    """Warp a stacked sky (undistorted reference grid, BOTTOM-UP (H, W, 3) as ``imaging.read_rgb`` returns) back to native geometry.

    Native pixel ``p`` shows the sky point whose undistorted coordinate is
    ``D(p)``; the distortion works in TOP-DOWN coordinates, so rows are
    flipped around the warp.  Pixels whose source position lies outside the
    stack (beyond the magnified corners) are filled with 0 like Siril's own
    registration borders; positions inside are resampled with edge-clamped
    kernel taps, so no empty rim is left (``resample`` 1.0.1).

    All planes are resampled in one pass (``resample.warp_planes_fused``: one
    mapping and one set of kernel weights per block) straight into the
    returned C-contiguous BOTTOM-UP array; the result is bit-identical to
    warping each plane with ``warp_image`` (P1a G-1).
    """
    from . import resample
    model = rd.RadialDistortion(**parameters)
    bottom_up = np.asarray(undistorted_rgb_bottom_up, dtype=np.float32)
    top_down = np.moveaxis(bottom_up[::-1], -1, 0)
    kwargs = {"kernel": kernel} if kernel else {}
    if bottom_up.ndim != 3 or bottom_up.shape[-1] == 0:   # not an (H, W, planes>0) image: the former path verbatim
        warped = resample.warp_planes(top_down, model.apply, fill=0.0, **kwargs)
        return np.ascontiguousarray(np.moveaxis(warped, 0, -1)[::-1])
    native = np.empty(bottom_up.shape, dtype=np.float32)          # final (H, W, planes) BOTTOM-UP, C-contiguous
    resample.warp_planes_fused(top_down, model.apply, fill=0.0, out=np.moveaxis(native[::-1], -1, 0), **kwargs)
    return native
