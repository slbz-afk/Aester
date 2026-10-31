#!/usr/bin/env python3
"""S4 step 1 acceptance: bright-trail protection on the real subset (RUN_LOG §28, pre-registered R1-R7).

2026-09-25 current defaults: curated 48 inputs, start=0/every=1. DSC_2269
is frame00022.dng in that folder. The full-DNG usage examples below are
historical; use the current baseline job sky_mask.png for new runs.

Runs the product job through ``s3_accept_horizon`` (mask + quality gate, T1-T4 + H1/H2/H3c/H5) on the 3rd, 6th, ...,
144th frame of the Z8 night (``--start 2 --every 3``: 48 frames including DSC_2269, the known bright trail, which is
the subset's 22nd frame), then judges the transient stage of that job:

  R1  the job succeeded, work/transients/candidates.json exists, every candidate is traceable (input path + sha256,
      frame index, frame name; DATE-OBS reported when Siril carried it)
  R2  the DSC_2269 frame has an automatically detected candidate (otherwise FAIL, and a manual mark can be given with
      --mark for a second evaluation of the same job) of length >= 500 px whose along-trail core contrast on the
      source frame is >= 1.0 (the 2026-09-16 measurement gave +154 %), and whose line crosses the old ROI
      (x 2450-3350, y 0-1150 in that job's registered frame; different registration -> expanded by 200 px, sanity only)
  R3  the base master along the same line: |core contrast| <= 2 % of the single-frame contrast (the rejection stack
      keeps the trail out)
  R4  with the candidate kept (its suggested class, or a per-candidate override when the suggestion is not meteor —
      recorded), the undistorted "master + layer" along the trail reaches >= 80 % of the normalised single-frame
      contrast and the native with-transients sky >= 75 % (the warp-back resamples a thin line)
  R5  the layer is confined: the with-transients composite differs from the base composite only inside the kept
      footprints (dilated by the feather + 1 px), never inside the ground (mask < 1 dilated 3 px), and each footprint
      has <= length x (2 x (half width + 2) + 5) x 1.1 pixels; with the candidate omitted (a recompose) no
      with-transients composite is produced and the base composite is unchanged; restoring the decision reproduces the
      same with-transients composite bit-exactly.  REVISED FORM (recorded, RUN_LOG §28/§31): the pre-registration said
      "with nothing kept, the with-transients composite is bit-identical to the base"; the product emits no
      with-transients composite at all in that case, so the check is "no such product + base unchanged + restore
      bit-exact".
  R6  outside the footprints the composite is bit-exact with the base (part of R5's confinement check)
  R7  every measured candidate carries a suggested class from the fixed class list, at least one reason, the
      cross-frame evidence fields and a light-curve judgement for both ends (falsifiable listing check, 1.1.0 — the
      1.0.x script hard-coded this as passed); the availability of the continuation evidence on a subset whose
      adjacent frames are three exposures apart is stated

The verdict is "passed under the revised criteria": R2's input lookup (1.0.1), R5's form and R7's check (1.1.0) were
changed after results were seen and are recorded as such (RUN_LOG §29, §31); the numeric thresholds are unchanged.

Usage (Windows, from the project root):

    .venv\\Scripts\\python.exe scripts\\s4_accept_transients.py --start 2 --every 3 --sky-limit-y 4540 ^
        --sky-mask outputs\\z8-meteor-20260916\\full-jobs\\job-5a2213e524204fb4\\sky_mask.png ^
        --output-root outputs\\fable-s2-20260917
    .venv\\Scripts\\python.exe scripts\\s4_accept_transients.py --job-dir <job> --run <accept-transients run folder>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from starstack import fits_crop  # noqa: E402
from starstack import sky_validity as sv  # noqa: E402
from starstack import transient_stage as ts  # noqa: E402
from starstack import transient_trails as tt  # noqa: E402
import s3_accept_horizon as horizon  # noqa: E402
from development_assets import BASELINE  # noqa: E402

ACCEPTANCE_VERSION = "1.1.0"
KNOWN_TRAIL = {"input": "DSC_2269.dng", "min_length_px": 500.0, "min_source_contrast": 1.0,
               "old_roi_xyxy": [2450, 0, 3350, 1150], "old_roi_expand_px": 200, "old_job": "job-5a2213e524204fb4"}
CRITERIA = {"R3": {"master_contrast_max_fraction": 0.02},
            "R4": {"grid_min_fraction": 0.80, "native_min_fraction": 0.75},
            "R5": {"footprint_dilate_px": tt.DEFAULT_LAYER["feather_px"] + 1, "footprint_factor": 1.1}}


def line_contrast(path: Path, x0: float, y0: float, x1: float, y1: float, half: int, *, step: float = 4.0) -> dict:
    """Along-line core-vs-side contrast of an image (not a residual): median over positions of (core - side) / side per
    channel, plus the side levels; the same core/side geometry as the candidate measurement."""
    inner, outer = tt.DEFAULT_DETECTION["side_inner_px"], tt.DEFAULT_DETECTION["side_outer_px"]
    p0 = np.array([x0, y0], dtype=np.float64)
    p1 = np.array([x1, y1], dtype=np.float64)
    length = float(np.hypot(*(p1 - p0)))
    u = (p1 - p0) / length
    n_vec = np.array([-u[1], u[0]])
    reach = half + outer + 2
    xs = [p0[0], p1[0]]
    ys = [p0[1], p1[1]]
    box = (int(math.floor(min(xs) - reach)), int(math.floor(min(ys) - reach)), int(math.ceil(max(xs) + reach)) + 1, int(math.ceil(max(ys) + reach)) + 1)
    crops = []
    for c in range(3):
        crop = tt._crop(path, c, box[0], box[2], box[1], box[3])
        crop = crop.copy()
        crop[crop == 0] = np.nan
        crops.append(crop)
    origin = (max(0, box[0]), max(0, box[1]))
    t_values = np.arange(0.0, length + 1e-6, step)
    d_values = np.arange(-reach + 1, reach, 1.0)
    profiles = tt._sample_profiles(crops, origin, p0, u, n_vec, t_values, d_values)
    core_sel = np.abs(d_values) <= half
    side_sel = (np.abs(d_values) >= half + inner) & (np.abs(d_values) <= half + outer)
    out = {"positions": int(t_values.size), "contrast": {}, "side_level": {}, "core_level": {}}
    with tt._quiet():
        for c, name in enumerate("RGB"):
            core = np.nanmedian(profiles[c][:, core_sel], axis=1)
            side = np.nanmedian(profiles[c][:, side_sel], axis=1)
            ratio = (core - side) / side
            out["contrast"][name] = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None
            out["side_level"][name] = float(np.nanmedian(side)) if np.isfinite(side).any() else None
            out["core_level"][name] = float(np.nanmedian(core)) if np.isfinite(core).any() else None
    return out


def _stem_of(job: dict, basename: str, run: Path | None = None) -> str | None:
    """The job stem of the known trail's input: by the job's input names, else by the run manifest ``inputs.json`` that
    ``s3_accept_horizon`` writes when it hard-links the subset as ``frame000NN.dng`` (source name -> frame), else a
    ``light_000NN`` / ``frame000NN`` spelling."""
    name = basename.lower()
    # Curated inputs were renamed frame000NN. Prefer their immutable identity,
    # which still works after further sub-selection or input reordering.
    original_name = Path(name).with_suffix('.nef').name
    matches = [r for r in BASELINE['frames'] if r['original_name'].lower() == original_name]
    if matches:
        for i, entry in enumerate(job['inputs'], 1):
            if entry.get('sha256') == matches[0]['sha256']:
                return f"light_{i:05d}"
    for i, entry in enumerate(job["inputs"], 1):
        if Path(entry["path"]).name.lower() == name:
            return f"light_{i:05d}"
    manifest = run / "inputs.json" if run is not None else None
    if manifest is not None and manifest.is_file():
        try:
            for frame in json.loads(manifest.read_text(encoding="utf-8")).get("frames", []):
                source = re.split(r"[\\/]", str(frame.get("source", "")))[-1]      # a Windows path read anywhere
                if source.lower() == name:
                    return frame.get("stem") or f"light_{int(frame['frame']):05d}"
        except (ValueError, KeyError, TypeError):
            pass
    m = re.fullmatch(r"(?:light_|frame)0*(\d+)(?:\.\w+)?", name)
    if m and 0 < int(m.group(1)) <= len(job["inputs"]):
        return f"light_{int(m.group(1)):05d}"
    return None


def _segment_crosses_box(x0, y0, x1, y1, box, expand) -> bool:
    bx0, by0, bx1, by1 = box[0] - expand, box[1] - expand, box[2] + expand, box[3] + expand
    for t in np.linspace(0.0, 1.0, 400):
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            return True
    return False


def _footprint_mask(shape: tuple[int, int], segments: list[tuple], dilate: float, model=None) -> np.ndarray:
    """Native mask of the kept footprints: the segments are straight on the stack grid, so every native pixel in the
    segment's bounding box is mapped to the grid (``model.apply``) and tested there (a straight grid line is a curve in
    native geometry).  ``dilate`` allows for the feather and the resampling kernel."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for (x0, y0, x1, y1, half) in segments:
        reach = half + tt.DEFAULT_LAYER["footprint_margin_px"] + tt.DEFAULT_LAYER["feather_px"] + dilate
        ext = tt.DEFAULT_LAYER["end_extension_px"] + dilate
        corners = np.array([[min(x0, x1) - reach - ext, min(y0, y1) - reach - ext], [max(x0, x1) + reach + ext, min(y0, y1) - reach - ext],
                            [min(x0, x1) - reach - ext, max(y0, y1) + reach + ext], [max(x0, x1) + reach + ext, max(y0, y1) + reach + ext]], dtype=np.float64)
        native = model.inverse(corners) if model is not None else corners
        bx0, bx1 = int(max(0, math.floor(native[:, 0].min()) - 4)), int(min(width, math.ceil(native[:, 0].max()) + 5))
        by0, by1 = int(max(0, math.floor(native[:, 1].min()) - 4)), int(min(height, math.ceil(native[:, 1].max()) + 5))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float64)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        q = model.apply(pts) if model is not None else pts
        ux, uy = x1 - x0, y1 - y0
        length = math.hypot(ux, uy)
        ux, uy = ux / length, uy / length
        along = (q[:, 0] - x0) * ux + (q[:, 1] - y0) * uy
        across = np.abs(-(q[:, 0] - x0) * uy + (q[:, 1] - y0) * ux)
        inside = ((across <= reach) & (along >= -ext) & (along <= length + ext)).reshape(by1 - by0, bx1 - bx0)
        mask[by0:by1, bx0:bx1] |= inside
    return mask


def _read_top_down(path: Path) -> np.ndarray:
    planes = []
    for c in range(3):
        view, header, roworder = fits_crop.plane_memmap(path, c)
        planes.append(tt._rows(view, header, roworder, 0, view.shape[0]))
    return np.stack(planes)


def listing_check(record: dict) -> dict:
    """R7 (1.1.0): every measured candidate has a suggested class from the fixed list, a non-empty reason list, the three
    cross-frame evidence fields and a light-curve judgement for both ends; returns the incomplete ids."""
    incomplete = []
    measured = 0
    for c in record.get("candidates", []):
        if (c.get("measurement") or {}).get("status") != "measured":
            continue
        measured += 1
        s = c.get("suggestion") or {}
        ev = c.get("evidence") or {}
        ends = ((ev.get("light_curve") or {}).get("ends") or {})
        ok = (s.get("class") in tt.CLASSES and bool(s.get("reasons"))
              and all(k in ev for k in ("continues", "persistent", "periodic"))
              and all(ends.get(k) in ("faded", "abrupt", "between", "unjudged") for k in ("start", "stop")))
        if not ok:
            incomplete.append(c.get("id"))
    return {"measured": measured, "incomplete": incomplete, "passed": measured > 0 and not incomplete}


def transient_checks(job_dir: Path, mask_path: Path, run: Path, *, mark=None, log=print, known: dict | None = None) -> dict:
    from starstack import core
    known = dict(KNOWN_TRAIL, **(known or {}))
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    work = job_dir / "work"
    report = {"acceptance_version": ACCEPTANCE_VERSION, "criteria": CRITERIA, "known_trail": known, "job": str(job_dir)}
    failures = []
    record = ts.read_record(work)
    diag = (job.get("diagnostics") or {}).get("transients") or {}
    # R1
    r1 = {"state": job.get("state"), "record_present": record is not None, "candidates": len(record["candidates"]) if record else 0,
          "by_status": (record.get("summary") or {}).get("by_status") if record else None, "workers": record.get("workers") if record else None,
          "frames_scanned": record.get("frames_scanned") if record else None, "seconds": record.get("seconds") if record else None,
          "traceable": None, "date_obs_present": None, "stage_diag_status": diag.get("status")}
    if record:
        r1["traceable"] = all(((c.get("provenance") or {}).get("input") or {}).get("path") and ((c.get("provenance") or {}).get("input") or {}).get("sha256")
                              and (c.get("provenance") or {}).get("frame_index") for c in record["candidates"])
        r1["date_obs_present"] = sum(1 for c in record["candidates"] if (c.get("provenance") or {}).get("date_obs"))
    r1["passed"] = bool(job.get("state") == "succeeded" and record is not None and (r1["traceable"] is not False))
    report["R1"] = r1
    log(f"R1 {'ok' if r1['passed'] else 'FAIL'}: state {r1['state']}, record {r1['record_present']}, candidates {r1['candidates']} "
        f"(by status {r1['by_status']}), frames {r1['frames_scanned']}, {r1['seconds']} s, workers {r1['workers']}")
    if not r1["passed"]:
        failures.append("R1 job/record")
        report["verdict"] = {"passed": False, "failures": failures}
        return report
    stem = _stem_of(job, known["input"], run)
    if stem is None:
        log(f"R2 FAIL: input {known['input']} is not among the job inputs nor in the run manifest {run / 'inputs.json'}")
    used = (job.get("diagnostics") or {}).get("registration_source") == core.REGISTRATION_SOURCE_OWN
    model = None
    if used:
        from starstack.registration_distortion import RadialDistortion
        cal = json.loads((work / "distortion.json").read_text(encoding="utf-8"))
        model = RadialDistortion(**cal["parameters"])
    grid_master = ts.grid_master_path(work, used)
    # R2
    if mark is not None:
        log(f"manual mark on {stem}: {mark}")
        core.recompose_transients(str(job_dir), marks=[{"input": known["input"], **mark}])
        record = ts.read_record(work)
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    on_frame = [c for c in record["candidates"] if c["stem"] == stem and c["measurement"].get("status") == "measured"]
    auto = [c for c in on_frame if c["source"] == "auto"]
    chosen = max(auto or on_frame, key=lambda c: c["measurement"]["length_px"]) if on_frame else None
    r2 = {"stem": stem, "candidates_on_frame": [c["id"] for c in on_frame], "auto_detected": bool(auto), "chosen": chosen["id"] if chosen else None}
    if chosen:
        m = chosen["measurement"]
        src = line_contrast(work / f"r_{stem}.fit", m["x0"], m["y0"], m["x1"], m["y1"], m["half_width_px"])
        native = chosen["geometry"]["native_reference"]
        r2.update({"length_px": m["length_px"], "half_width_px": m["half_width_px"], "measured_contrast": m["contrast"],
                   "source_frame_contrast": src["contrast"], "source_side_level": src["side_level"], "total_snr": m["total_snr"],
                   "suggestion": chosen["suggestion"], "evidence": chosen.get("evidence"), "decision": chosen["decision"],
                   "endpoints_grid": [[m["x0"], m["y0"]], [m["x1"], m["y1"]]], "endpoints_native_reference": native,
                   "crosses_old_roi": _segment_crosses_box(native[0][0], native[0][1], native[1][0], native[1][1], known["old_roi_xyxy"], known["old_roi_expand_px"])})
        r2["passed"] = bool(auto and m["length_px"] >= known["min_length_px"] and (src["contrast"]["G"] or 0) >= known["min_source_contrast"] and r2["crosses_old_roi"])
    else:
        r2["passed"] = False
    report["R2"] = r2
    log(f"R2 {'ok' if r2['passed'] else 'FAIL'}: {stem} candidates {r2['candidates_on_frame']} auto {r2['auto_detected']}"
        + (f", length {r2['length_px']:.0f} px, source contrast G {r2['source_frame_contrast']['G']:.3f}, suggestion {r2['suggestion']['class']}, crosses old ROI {r2['crosses_old_roi']}" if chosen else ""))
    if not r2["passed"]:
        failures.append("R2 known trail")
    if chosen is None:
        report["verdict"] = {"passed": False, "failures": failures}
        return report
    m = chosen["measurement"]
    # R3: the base master along the same line (grid)
    base_grid = line_contrast(grid_master, m["x0"], m["y0"], m["x1"], m["y1"], m["half_width_px"])
    frame_contrast = r2["source_frame_contrast"]["G"]
    r3 = {"master_contrast": base_grid["contrast"], "single_frame_contrast_G": frame_contrast,
          "fraction_G": abs(base_grid["contrast"]["G"]) / frame_contrast if frame_contrast else None}
    r3["passed"] = bool(r3["fraction_G"] is not None and r3["fraction_G"] <= CRITERIA["R3"]["master_contrast_max_fraction"])
    report["R3"] = r3
    log(f"R3 {'ok' if r3['passed'] else 'FAIL'}: master contrast G {base_grid['contrast']['G']:+.5f} = {r3['fraction_G']:.4f} of the single frame")
    if not r3["passed"]:
        failures.append("R3 base master")
    # R4: keep the candidate (override when the suggestion is not meteor), grid master + layer and the native composite
    overridden = False
    if chosen["decision"] != "keep":
        log(f"R4: the candidate's decision is {chosen['decision']} ({chosen['decision_reason']}); keeping it by override for the protection check (recorded)")
        core.recompose_transients(str(job_dir), overrides={chosen["id"]: "keep"})
        overridden = True
        record = ts.read_record(work)
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        chosen = next(c for c in record["candidates"] if c["id"] == chosen["id"])
    stats = record["frame_stats"]
    ref_stats = stats.get(record.get("reference_stem")) or stats[stem]
    scale = tt.normalisation(stats[stem], ref_stats)
    box = tt.layer_box(work / f"r_{stem}.fit", grid_master, chosen, scale, stats[stem]["median"], ref_stats["median"], params=record["params"]["layer"])
    # grid: a region of the master around the trail (wide enough for the side bands) plus the layer box, measured with
    # the same sampler through a small scratch FITS
    view, header, roworder = fits_crop.plane_memmap(grid_master, 0)
    gh, gw = view.shape
    reach2 = m["half_width_px"] + tt.DEFAULT_DETECTION["side_outer_px"] + 4
    rx0, ry0 = int(max(0, math.floor(min(m["x0"], m["x1"]) - reach2))), int(max(0, math.floor(min(m["y0"], m["y1"]) - reach2)))
    rx1, ry1 = int(min(gw, math.ceil(max(m["x0"], m["x1"]) + reach2) + 1)), int(min(gh, math.ceil(max(m["y0"], m["y1"]) + reach2) + 1))
    region = np.stack([tt._crop(grid_master, c, rx0, rx1, ry0, ry1) for c in range(3)])
    bx, by = box["x0"] - rx0, box["y0"] - ry0
    bh, bw = box["data"].shape[1:]
    sy0, sx0 = max(0, by), max(0, bx)
    sy1, sx1 = min(region.shape[1], by + bh), min(region.shape[2], bx + bw)
    if sy1 > sy0 and sx1 > sx0:
        region[:, sy0:sy1, sx0:sx1] += box["data"][:, sy0 - by:sy1 - by, sx0 - bx:sx1 - bx]
    tmp = run / "r4_grid_master_plus_layer_region.fit"
    from starstack.synthetic_starfield import write_fits_rgb32
    write_fits_rgb32(tmp, region.astype(np.float32))
    with_grid = line_contrast(tmp, m["x0"] - rx0, m["y0"] - ry0, m["x1"] - rx0, m["y1"] - ry0, m["half_width_px"])
    tmp.unlink(missing_ok=True)
    expected_grid = frame_contrast * scale[1] * (r2["source_side_level"]["G"] or 0) / (base_grid["side_level"]["G"] or 1)
    artifacts = job.get("artifacts", [])
    with_native_path = next((a["path"] for a in artifacts if a.get("group") == "transients" and a["kind"] == "fits"), None)
    base_native_path = next((a["path"] for a in artifacts if a["label"] == "合成线性母版"), None)
    native = chosen["geometry"]["native_reference"]
    with_native = line_contrast(Path(with_native_path), native[0][0], native[0][1], native[1][0], native[1][1], m["half_width_px"]) if with_native_path else None
    base_native = line_contrast(Path(base_native_path), native[0][0], native[0][1], native[1][0], native[1][1], m["half_width_px"]) if base_native_path else None
    expected_native = frame_contrast * scale[1] * (r2["source_side_level"]["G"] or 0) / ((base_native or {}).get("side_level", {}).get("G") or 1)
    r4 = {"overridden_to_keep": overridden, "scale_addscale": scale, "grid_with_layer_contrast": with_grid["contrast"], "expected_grid_contrast_G": expected_grid,
          "grid_fraction": (with_grid["contrast"]["G"] / expected_grid) if expected_grid else None,
          "native_with_contrast": (with_native or {}).get("contrast"), "native_base_contrast": (base_native or {}).get("contrast"),
          "expected_native_contrast_G": expected_native,
          "native_fraction": ((with_native["contrast"]["G"] / expected_native) if (with_native and expected_native) else None),
          "with_transients_composite": with_native_path}
    r4["passed"] = bool(r4["grid_fraction"] is not None and r4["grid_fraction"] >= CRITERIA["R4"]["grid_min_fraction"]
                        and r4["native_fraction"] is not None and r4["native_fraction"] >= CRITERIA["R4"]["native_min_fraction"])
    report["R4"] = r4
    log(f"R4 {'ok' if r4['passed'] else 'FAIL'}: grid master+layer contrast G {with_grid['contrast']['G']:.3f} = {r4['grid_fraction'] if r4['grid_fraction'] is not None else float('nan'):.3f} of expected {expected_grid:.3f}; "
        f"native {(with_native or {}).get('contrast', {}).get('G')} = {r4['native_fraction']} of expected {expected_native:.3f}")
    if not r4["passed"]:
        failures.append("R4 protection")
    # R5/R6: confinement, ground, footprint size, omit -> no composite, restore -> bit-exact
    r5 = {"kept": [k["id"] for k in diag.get("kept", [])] if diag else None}
    if with_native_path and base_native_path:
        base_arr = _read_top_down(Path(base_native_path))
        with_arr = _read_top_down(Path(with_native_path))
        diff = np.abs(with_arr - base_arr).max(axis=0)
        height, width = diff.shape
        kept_segments = []
        for c in record["candidates"]:
            if c.get("decision") == "keep" and c["measurement"].get("status") == "measured":
                mm = c["measurement"]
                kept_segments.append((mm["x0"], mm["y0"], mm["x1"], mm["y1"], mm["half_width_px"]))     # stack-grid coordinates
        footprint = _footprint_mask((height, width), kept_segments, CRITERIA["R5"]["footprint_dilate_px"], model)
        mask_td = sv.load_mask_top_down(mask_path, width, height)
        ground = ~sv.static_validity(mask_td, dilation_px=sv.DEFAULT_DILATION_PX)
        outside = diff[~footprint]
        r5.update({"kept_segments": len(kept_segments), "changed_pixels": int((diff > 0).sum()), "changed_outside_footprints": int((outside > 0).sum()),
                   "max_diff_outside_footprints": float(outside.max()) if outside.size else 0.0, "changed_in_ground": int((diff[ground] > 0).sum()),
                   "footprint_pixels_bound_ok": all(k["footprint_pixels"] <= (next(c for c in record["candidates"] if c["id"] == k["id"])["measurement"]["length_px"]
                                                                              * (2 * (next(c for c in record["candidates"] if c["id"] == k["id"])["measurement"]["half_width_px"] + tt.DEFAULT_LAYER["footprint_margin_px"]) + 5)
                                                                              * CRITERIA["R5"]["footprint_factor"]) for k in (job["diagnostics"]["transients"].get("kept") or []))})
        # omit -> no with-transients composite; base unchanged; restore -> same composite
        before_base = _read_top_down(Path(base_native_path))
        omitted = core.recompose_transients(str(job_dir), overrides={c["id"]: "omit" for c in record["candidates"]})
        r5["omit_no_composite"] = not any(a.get("group") == "transients" for a in omitted["artifacts"])
        r5["base_unchanged_after_omit"] = bool(np.array_equal(_read_top_down(Path(base_native_path)), before_base))
        restored = core.recompose_transients(str(job_dir), overrides={c["id"]: ("keep" if overridden and c["id"] == chosen["id"] else None) for c in record["candidates"]})
        again = next((a["path"] for a in restored["artifacts"] if a.get("group") == "transients" and a["kind"] == "fits"), None)
        r5["restored_composite"] = again
        r5["restored_bit_exact"] = bool(again and np.array_equal(_read_top_down(Path(again)), with_arr))
        r5["passed"] = bool(r5["changed_outside_footprints"] == 0 and r5["changed_in_ground"] == 0 and r5["footprint_pixels_bound_ok"]
                            and r5["omit_no_composite"] and r5["base_unchanged_after_omit"] and r5["restored_bit_exact"])
        # before/after crop for the record
        try:
            g = chosen["geometry"]["native_reference"]
            cx, cy = (g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2
            half_box = max(abs(g[1][0] - g[0][0]), abs(g[1][1] - g[0][1])) * 0.6 + 60
            x0, x1 = int(max(0, cx - half_box)), int(min(width, cx + half_box))
            y0, y1 = int(max(0, cy - half_box)), int(min(height, cy + half_box))
            step = max(1, int(math.ceil((x1 - x0) / 700)))
            left = fits_crop.asinh_stretch(base_arr[1, y0:y1:step, x0:x1:step])
            right = fits_crop.asinh_stretch(with_arr[1, y0:y1:step, x0:x1:step])
            gap = np.zeros((left.shape[0], 6), dtype=np.uint8)
            fits_crop.write_png_gray(run / "transients_before_after.png", np.hstack([left, gap, right]))
            r5["before_after_png"] = str(run / "transients_before_after.png")
        except Exception as exc:   # the picture is for the record only
            r5["before_after_png"] = f"not written: {exc}"
    else:
        r5["passed"] = False
        r5["reason"] = "no with-transients composite or no base composite registered"
    report["R5"] = r5
    log(f"R5/R6 {'ok' if r5['passed'] else 'FAIL'}: {json.dumps({k: v for k, v in r5.items() if k not in ('kept',)}, ensure_ascii=False)[:600]}")
    if not r5["passed"]:
        failures.append("R5/R6 layer confinement / omit / restore")
    # R7: the census
    census = []
    for c in record["candidates"]:
        mm = c["measurement"]
        census.append({"id": c["id"], "stem": c["stem"], "source": c["source"], "status": mm.get("status"), "length_px": mm.get("length_px"),
                       "contrast_median": mm.get("contrast_median"), "total_snr": mm.get("total_snr"), "suggestion": c.get("suggestion"),
                       "continues": [l["stem"] for l in (c.get("evidence") or {}).get("continues", [])],
                       "persistent": (c.get("evidence") or {}).get("persistent"), "periodic": ((c.get("evidence") or {}).get("periodic") or {}).get("periodic"),
                       "decision": c.get("decision"), "input": ((c.get("provenance") or {}).get("input") or {}).get("path")})
    r7 = {"candidates": census, "adjacent_frame_spacing_exposures": 3,
          "note": "the subset's adjacent frames are three exposures apart: a satellite may have left the field between them, so the continuation evidence is weaker than on the full sequence",
          **listing_check(record)}
    report["R7"] = r7
    if not r7["passed"]:
        failures.append("R7 listing")
    log(f"R7 {'ok' if r7['passed'] else 'FAIL'}: {len(census)} candidates listed, {r7['measured']} measured, {len(r7['incomplete'])} incomplete; "
        f"by suggested class {json.dumps(record['summary']['by_effective_label'], ensure_ascii=False)}")
    report["verdict"] = {"passed": not failures, "failures": failures}
    return report


def render(report: dict, long_report: dict | None, horizon_report: dict | None) -> str:
    v = report["verdict"]
    lines = [f"# S4 亮迹保护验收（{report.get('run', '')}）", "",
             f"**结论：{'通过' if v['passed'] else '未通过'}**" + (f" — 未通过项：{'；'.join(v['failures'])}" if v["failures"] else ""), "",
             f"作业：`{report['job']}` · 判据版本 {report['acceptance_version']}（RUN_LOG §28 预登记；R2 查找 1.0.1、R5 形式与 R7 可证伪检查 1.1.0 为看结果后的修订，见 §29/§31）—— "
             f"结论口径为**按修订后判据**{'通过' if v['passed'] else '未通过'}，数值阈值未改", ""]
    if long_report:
        lv = long_report.get("verdict", {})
        lines.append(f"长跨度 T1–T4：{'通过' if lv.get('passed') else '未通过'}（{lv.get('failures')}）")
    if horizon_report:
        hv = horizon_report.get("verdict", {})
        lines.append(f"地平线 H1/H2/H3c/H5：{'通过' if hv.get('passed') else '未通过'}（{hv.get('failures')}）")
    lines.append("")
    for key in ("R1", "R2", "R3", "R4", "R5", "R7"):
        section = report.get(key)
        if not section:
            continue
        lines.append(f"## {key}：{'通过' if section.get('passed') else '未通过'}")
        lines.append("")
        if key == "R7":
            lines.append(section["note"])
            lines.append("")
            lines.append("| 候选 | 帧 | 来源 | 长度 px | 对比度中位 | 总 S/N | 建议 | 证据 | 处置 |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for c in section["candidates"]:
                sug = c["suggestion"] or {}
                lines.append(f"| {c['id']} | {c['stem']} | {c['source']} | {c['length_px'] if c['length_px'] is None else round(c['length_px'])} | "
                             f"{'' if c['contrast_median'] is None else round(c['contrast_median'], 3)} | {'' if c['total_snr'] is None else round(c['total_snr'], 1)} | "
                             f"{sug.get('class')} | {'；'.join(sug.get('reasons', []))} | {c['decision']} |")
        else:
            lines.append("```json")
            lines.append(json.dumps({k: v for k, v in section.items() if k not in ("evidence",)}, ensure_ascii=False, indent=1, default=str)[:6000])
            lines.append("```")
        lines.append("")
    if (report.get("R5") or {}).get("before_after_png"):
        lines.append(f"![before/after]({Path(report['R5']['before_after_png']).name})")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(horizon.long_span.DEFAULT_RAW_DIR))
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--siril", default=None)
    parser.add_argument("--sky-limit-y", type=float, default=None)
    parser.add_argument("--sky-mask", required=True)
    parser.add_argument("--label", default="accept-transients")
    parser.add_argument("--support-mode", choices=("legacy", "strict_siril_1_4_4"), default="legacy")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    parser.add_argument("--job-dir", default=None, help="evaluate an existing job instead of running one")
    parser.add_argument("--run", default=None, help="the run folder to write into (with --job-dir); default: a new folder")
    parser.add_argument("--mark", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                        help="manual mark on the DSC_2269 frame (native source pixels, top-down) when the trail was not detected automatically")
    parser.add_argument("--known-input", default=None, help="input file name of the known-trail frame (tests: a synthetic frame)")
    parser.add_argument("--known-min-length", type=float, default=KNOWN_TRAIL["min_length_px"])
    parser.add_argument("--old-roi", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"), help="override the old-ROI sanity box")
    args = parser.parse_args(argv)
    if args.known_input is None:
        args.known_input = KNOWN_TRAIL["input"]
    known = {"input": args.known_input, "min_length_px": args.known_min_length}
    if args.old_roi:
        known["old_roi_xyxy"] = list(args.old_roi)
    output_root = Path(args.output_root).resolve()
    long_report = horizon_report = None
    if args.job_dir:
        job_dir = Path(args.job_dir).resolve()
        run = Path(args.run).resolve() if args.run else output_root / f"{args.label}-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-eval"
        run.mkdir(parents=True, exist_ok=True)
    else:
        if args.sky_limit_y is None:
            parser.error("--sky-limit-y is required to run the job")
        started = _dt.datetime.now().timestamp()
        h_argv = ["--raw-dir", args.raw_dir, "--every", str(args.every), "--start", str(args.start), "--sky-limit-y", str(args.sky_limit_y),
                  "--sky-mask", args.sky_mask, "--label", args.label, "--output-root", str(output_root)]
        if args.limit is not None:
            h_argv += ["--limit", str(args.limit)]
        h_argv += ["--support-mode", args.support_mode]
        if args.siril:
            h_argv += ["--siril", args.siril]
        horizon.main(h_argv)               # T1–T4 + H1/H2/H3c/H5 on the new job; folded into the verdict below
        run = horizon._find_run(output_root, args.label, started)
        long_report = json.loads((run / "acceptance.json").read_text(encoding="utf-8"))
        horizon_report = json.loads((run / "horizon.json").read_text(encoding="utf-8")) if (run / "horizon.json").is_file() else None
        if "T1" not in long_report:
            print(f"FAIL: the product job did not succeed: {long_report.get('verdict')}")
            return 2
        job_dir = Path(long_report["job"]["dir"])
    mark = None
    if args.mark:
        mark = {"x0": args.mark[0], "y0": args.mark[1], "x1": args.mark[2], "y1": args.mark[3]}
    report = transient_checks(job_dir, Path(args.sky_mask).resolve(), run, mark=mark, known=known)
    report["run"] = run.name
    if long_report and not long_report["verdict"]["passed"]:
        report["verdict"]["passed"] = False
        report["verdict"]["failures"].append(f"long-span T1-T4: {long_report['verdict']['failures']}")
    if horizon_report and not horizon_report["verdict"]["passed"]:
        report["verdict"]["passed"] = False
        report["verdict"]["failures"].append(f"horizon: {horizon_report['verdict']['failures']}")
    (run / "transients_acceptance.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (run / "TRANSIENTS.md").write_text(render(report, long_report, horizon_report), encoding="utf-8")
    print(f"{'PASS' if report['verdict']['passed'] else 'FAIL'}: {run / 'TRANSIENTS.md'}")
    for f in report["verdict"]["failures"]:
        print(f"  - {f}")
    return 0 if report["verdict"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
