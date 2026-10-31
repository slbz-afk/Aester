"""S2 long-span acceptance of the product's distortion flow (one command).

Current entry (2026-09-25): defaults now use the already-selected 48-frame
registry, every=1/start=0. The 144-frame examples below describe historical
validation only; their live pixel cache and full DNG folder are retired.
Do not reconstruct all 144 frames for routine development.

Runs the *product* job (``core``: convert -> probe registration -> per-job
distortion calibration -> ``register -2pass -disto=file`` -> ``seqapplyreg``
-> stack -> warp back to native geometry) on every N-th RAW of a folder
(default every 3rd of the 144 Z8 frames: 48 frames over 31 minutes, the same
selection as the registration-only run of RUN_LOG §18), then judges it with
the pre-registered criteria below.  Helpers are shared with
``s2_accept_two_pass.py`` and ``s2_accept_distortion.py``; the RAW files are
read only and nothing outside the run directory is modified.

    <output-root>/accept-long-span-<UTC stamp>-<6hex>/
        inputs/, inputs.json              hard links (frame00001.dng ...) + which RAW is which
        jobs/job-<id>/                    the product job (work/: light_*, r_light_*, both masters)
        job_summary.json                  state, diagnostics, timeline
        rframes/<frame>_<box>.json        Siril's own r_ frames vs the reference r_ frame (undistorted grid)
        native/<box>.json                 native sky master vs the native reference light frame
        master/<box>.json                 same stars: native master vs the native reference frame (shapes)
        acceptance.json / ACCEPTANCE.md

Why a separate run: on the 16-frame / 3-minute set the distortion term can
only be judged against the centroid noise floor (RUN_LOG §21).  The term
exists for spans of tens of minutes, where a plain homography was measured
at 2-6 px in the corners and Siril dropped the farthest frames (§18); this
run asks whether the *product* achieves what the offline analysis claimed.

Pre-registered criteria (fixed before the run; thresholds come from the §18
claims, not from expected results):
  T1 job         succeeded; inputs are the every-N-th selection; at least ``--min-registered-fraction`` of the
                 frames registered by the calibration (default 1.0: 48/48 on the every-3rd set; the full 144-frame
                 night is pre-registered at 0.97, i.e. at most 4 excluded, every excluded frame listed with its
                 reason) and every registered frame stacked (STACKCNT = registered); since RUN_LOG §22 the matrices
                 are the product's own, written into the sequence file; scripts match the product constants;
                 reference agrees between light_.seq, the Siril log and diagnostics;
                 distortion fitted and applied; distortion.wcs and sky_master_undistorted.fit present; native
                 master float32 with the frame geometry; STACKCNT carried.
  T2 calibration from work/distortion.json: status fitted; every frame Siril dropped in the probe is recovered
                 by the model; every frame re-associated with matched fraction >= 0.75 and validation corner
                 RMS <= 0.5 px, at most 3 frames tolerated (fraction >= 0.4, RMS <= 1.0: the detection-anomaly
                 frames of §18); corner median with the term <= 0.5 x the homography-only median (§18: 1.0-2.3
                 -> 0.16-0.42 px).
  T3 r_ frames   Siril's exported frames 1, 8, 16, 24, 32, 40, 47, 48 (plus the sequence ends and the farthest
                 frame) vs the reference r_ frame in the four boxes: star centroid RMS <= 0.5 px in every box;
                 at most 2 frames tolerated (<= 1.0 px).  The RMS and mean offset vs span are reported as the
                 measurement of the residual span term beyond the radial model (§18: 0.16 -> 0.43 px at 20 min).
  Boxes          (rule fixed before the third run, RUN_LOG §22: the first two runs used the 16-frame job's fixed
                 boxes, and on a 31-minute sequence the lower-right one maps outside the far frames and into the
                 ground band, so its numbers measured missing data, not registration).  With ``--sky-limit-y``
                 given, the four boxes are placed by rule inside the *common sky coverage*: the part of the
                 reference grid that every registered frame covers with sky (its native footprint above the
                 ground line, mapped through the job's own distortion model and matrices), shrunk by a 200 px
                 margin for the native/undistorted difference; the boxes keep the 16-frame sizes (1400x1000
                 corners, 1100x1000 centre) and are anchored at the region's top-left, top-right, centre and
                 lower-right.  Without ``--sky-limit-y`` the fixed boxes are used (comparability with runs 1-2).
  T3b warp-back  native sky master vs the native reference light frame: centroid median <= 0.2 px and RMS
                 <= 0.4 px in every box.
  T4 master      same stars, native master vs the native reference frame (position-matched, so field-dependent
                 optics cancel): median ratio master/frame of sigma_major and sigma_minor <= 1.05 in every box.
                 Context only (not judged): the single-frame sigma of the checked r_ frames vs the reference's.
Verdict: pass only if all of T1..T4 pass; tolerated exceptions are listed.

Resources: ~40 GB in the job's work directory (48 x 274 MB light_ + 48 x 548 MB r_light_ + two masters);
about 7-8 minutes.  Nothing is deleted by this script.

Example (Windows, product venv; the Z8 set's ground line is at native row ~4540 of the reference frame):

    .venv\\Scripts\\python.exe scripts\\s2_accept_long_span.py --raw-dir outputs\\z8-meteor-20260916\\dng ^
        --every 3 --sky-limit-y 4540 --output-root outputs\\fable-s2-20260917

The whole night (144 frames, ~120 GB, 10-20 min), with the prototype's sky mask so the job also composes the ground
(the mask is in native geometry, which the product keeps):

    .venv\\Scripts\\python.exe scripts\\s2_accept_long_span.py --raw-dir outputs\\z8-meteor-20260916\\dng ^
        --every 1 --sky-limit-y 4540 --min-registered-fraction 0.97 --frames 1 24 48 72 96 120 143 144 ^
        --sky-mask outputs\\z8-meteor-20260916\\full-jobs\\job-5a2213e524204fb4\\sky_mask.png --label accept-full-night ^
        --output-root outputs\\fable-s2-20260917
    .venv\\Scripts\\python.exe scripts\\s2_accept_long_span.py --job-dir <existing job>   (re-evaluate only)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from starstack import core, fits_crop, siril, siril_sequence  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402
import s2_accept_distortion as dist_accept  # noqa: E402
import s2_accept_two_pass as base  # noqa: E402
from development_assets import INPUT_DIR  # noqa: E402

ACCEPTANCE_VERSION = "1.2.0"
BOXES = dict(base.BOXES)
CRITERIA = {
    "T1": {"distortion_status": "fitted", "sky_geometry": "native_via_undistorted_stack"},
    "T2": {"matched_fraction": 0.75, "corner_rms_px": 0.5, "tolerated_frames": 3, "tolerated_fraction": 0.4, "tolerated_rms_px": 1.0,
           "max_corner_median_ratio": 0.5, "dropped_frames_all_recovered": True},
    "T3": {"rms_px": 0.5, "tolerated_rms_px": 1.0, "tolerated_frames": 2},
    "T3b": {"median_px": 0.2, "rms_px": 0.4},
    "T4": {"max_ratio_master_over_frame": 1.05, "min_paired": 5},
}
DEFAULT_FRAMES = [1, 8, 16, 24, 32, 40, 47, 48]
DEFAULT_RAW_DIR = INPUT_DIR
_fmt, _write_json, _now = base._fmt, base._write_json, base._now


def select_frames(raw_dir: Path, every: int, limit: int | None, start: int = 0) -> list[Path]:
    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in core.RAW)
    chosen = files[start::every]
    return chosen[:limit] if limit else chosen


def link_inputs(files: list[Path], target: Path, manifest: Path, *, every: int, start: int, raw_dir: Path) -> dict:
    """Hard-link (or copy) the selected RAW files as frame00001.dng ...; the manifest lives *beside* the inputs folder."""
    target.mkdir(parents=True, exist_ok=False)
    method, records = None, []
    for i, src in enumerate(files, 1):
        dst = target / f"frame{i:05d}{src.suffix.lower()}"
        try:
            os.link(src, dst)
            method = method or "hardlink"
        except OSError:
            shutil.copyfile(src, dst)
            method = "copy"
        records.append({"frame": i, "stem": f"light_{i:05d}", "source": str(src), "size": src.stat().st_size})
    payload = {"method": method, "every": every, "start": start, "raw_dir": str(raw_dir), "count": len(files), "frames": records}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def job_checks(job: dict, job_dir: Path, sequence: dict, manifest: dict | None, *, min_registered_fraction: float = 1.0) -> dict:
    """T1: the distortion-flow checks of s2_accept_distortion, minus the reference-job comparison, plus the long-span ones."""
    result = dist_accept.job_checks(job, job_dir, {"inputs": job["inputs"]}, sequence)
    checks = dict(result["checks"])
    checks.pop("same_inputs_as_reference_job", None)
    checks.pop("stacked_equals_inputs", None)        # replaced below: the product may exclude frames it could not register
    checks.pop("r_frames_written", None)             # Siril exports the selected frames only
    diagnostics = job.get("diagnostics") or {}
    regdata = sequence["regdata"].get(1) or []
    registered = [im for im in sequence["images"]
                  if im["selected"] and im["position"] < len(regdata) and abs(np.linalg.det(np.asarray(regdata[im["position"]]["H"]))) > 1e-12]
    detail = diagnostics.get("registration_detail") or {}
    n_inputs = len(job["inputs"])
    n_registered = detail.get("registered") if detail.get("registered") is not None else len(registered)
    expected = (manifest or {}).get("count")
    r_frames = len(list((job_dir / "work").glob("r_light_*.fit")))
    # RUN_LOG §27: frames the quality gate kept out are expected exclusions — listed with reasons, deselected, not exported
    gate = diagnostics.get("quality_gate") or {}
    gate_excluded = list(gate.get("excluded_frames") or []) if gate.get("in_effect") else []
    frames_detail = detail.get("frames") or {}
    selected = {im["stem"]: im["selected"] for im in sequence["images"]}
    checks.update({
        "inputs_are_the_selected_raws": (n_inputs == expected == len((manifest or {}).get("frames", []))) if manifest else None,
        "registered_fraction_at_least": n_registered >= min_registered_fraction * (n_inputs - len(gate_excluded)) - 1e-9,
        "registered_frames_selected_in_sequence": len(registered) == n_registered,
        "stacked_count_equals_registered": diagnostics.get("stacked_count") == n_registered,
        "r_frames_written_for_registered": r_frames == n_registered,
        "quality_excluded_listed_with_reasons": all(e.get("reasons") and frames_detail.get(e["stem"], {}).get("status") == "excluded_quality"
                                                   for e in gate.get("excluded", [])) if gate_excluded else None,
        "quality_excluded_deselected_and_not_exported": all(not selected.get(stem, True) and not (job_dir / "work" / f"r_{stem}.fit").exists()
                                                           for stem in gate_excluded) if gate_excluded else None,
    })
    passed = all(v for v in checks.values() if v is not None)
    return {**result, "passed": passed, "checks": checks, "registered_in_stack_sequence": len(registered), "sequence_images": len(sequence["images"]),
            "registered": n_registered, "inputs": n_inputs, "min_registered_fraction": min_registered_fraction,
            "quality_gate": {"in_effect": bool(gate.get("in_effect")), "excluded": gate.get("excluded", []), "thresholds": gate.get("thresholds"), "medians": gate.get("medians")},
            "excluded_frames": {stem: (f.get("reason") or f.get("status")) for stem, f in frames_detail.items() if f.get("status") != "registered"}}


def calibration_checks(work: Path, criteria: dict, *, quality_excluded=()) -> dict:
    result = dist_accept.calibration_checks(work, criteria)
    # frames the quality gate kept out of the stack are not judged (RUN_LOG §27); they stay in the table
    excluded = set(quality_excluded)
    dropped = set(result.get("frames_dropped_by_siril") or [])
    recovered = set(result.get("recovered") or [])
    all_recovered = (dropped - excluded) <= recovered
    failing = [f for f in result["failing"] if f not in excluded]
    tolerated = [f for f in result["tolerated"] if f not in excluded]
    ratio = result.get("corner_median_ratio")
    passed = bool(result.get("status") == "fitted" and not failing and len(tolerated) <= criteria["tolerated_frames"]
                  and ratio is not None and ratio <= criteria["max_corner_median_ratio"]
                  and (all_recovered or not criteria.get("dropped_frames_all_recovered")))
    return {**result, "passed": passed, "failing": failing, "tolerated": tolerated, "quality_excluded": sorted(excluded),
            "dropped_all_recovered": all_recovered, "dropped_not_recovered": sorted(dropped - recovered)}


BOX_SIZES = {"topright": (1400, 1000), "centre": (1100, 1000), "topleft": (1400, 1000), "rightlower": (1400, 1000)}


def common_sky_coverage(calibration: dict, width: int, height: int, sky_limit_y: float, *, step: int = 40) -> dict:
    """Axis-aligned extent of the reference-grid points every registered frame covers with sky.

    A reference-grid point ``u`` is valid for frame k when its native position
    ``p_k = D^-1(H_k^-1 u)`` lies inside the frame and above ``sky_limit_y``
    (native row of the ground line).  Sampled on a ``step`` px lattice; the
    result is the largest axis-aligned rectangle of the lattice whose rows and
    columns are valid everywhere (a conservative inner box).
    """
    registration = calibration.get("registration") or {}
    frames = {stem: f for stem, f in (registration.get("frames") or {}).items() if f.get("status") == "registered" and f.get("H_top_down_magnified")}
    if not frames:
        raise ValueError("calibration has no registered frames")
    model = RadialDistortion(**calibration["parameters"])
    xs = np.arange(0, width, step, dtype=np.float64)
    ys = np.arange(0, height, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    u = np.column_stack([gx.ravel(), gy.ravel()])
    valid = np.ones(len(u), dtype=bool)
    per_frame = {}
    for stem, f in frames.items():
        h_inv = np.linalg.inv(np.asarray(f["H_top_down_magnified"], dtype=np.float64))
        p = model.inverse(_apply(h_inv, u))
        ok = (p[:, 0] >= 0) & (p[:, 0] <= width - 1) & (p[:, 1] >= 0) & (p[:, 1] <= min(height - 1, sky_limit_y))
        per_frame[stem] = float(ok.mean())
        valid &= ok
    grid = valid.reshape(len(ys), len(xs))

    def centre_run(row: np.ndarray, centre: int):
        """(first, last) lattice column of the valid run containing ``centre``; None if the centre column is invalid."""
        if not row[centre]:
            return None
        a = centre
        while a > 0 and row[a - 1]:
            a -= 1
        b = centre
        while b < len(row) - 1 and row[b + 1]:
            b += 1
        return a, b

    # largest-area inscribed lattice rectangle whose rows all contain the centre column: for every row the valid run around
    # the centre, then the best contiguous row range with the intersection of those runs as width (O(rows^2))
    runs = [centre_run(grid[i], len(xs) // 2) for i in range(len(ys))]
    best, best_area = None, 0.0
    for r0 in range(len(ys)):
        if runs[r0] is None:
            continue
        a, b = runs[r0]
        for r1 in range(r0, len(ys)):
            if runs[r1] is None:
                break
            a, b = max(a, runs[r1][0]), min(b, runs[r1][1])
            if b < a:
                break
            area = (xs[b] - xs[a]) * (ys[r1] - ys[r0])
            if area > best_area:
                best, best_area = (a, b, r0, r1), area
    if best is None:
        raise ValueError("no reference-grid rectangle is covered by every frame")
    a, b, r0, r1 = best
    extent = {"x0": float(xs[a]), "x1": float(xs[b]), "y0": float(ys[r0]), "y1": float(ys[r1]), "rectangular": True}
    return {**extent, "valid_fraction_by_frame": per_frame, "frames": len(frames), "sky_limit_y": float(sky_limit_y), "step": step,
            "valid_fraction_all": float(valid.mean())}


def _apply(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    v = np.hstack([xy, np.ones((len(xy), 1))]) @ h.T
    return v[:, :2] / v[:, 2:3]


def boxes_by_rule(coverage: dict, *, margin: int = 200, sizes: dict | None = None) -> dict:
    """Place the four boxes inside the common coverage shrunk by ``margin``: corners at the region's corners, centre at its centre."""
    sizes = sizes or BOX_SIZES
    x0, x1 = coverage["x0"] + margin, coverage["x1"] - margin
    y0, y1 = coverage["y0"] + margin, coverage["y1"] - margin
    if x1 - x0 < max(w for w, _ in sizes.values()) or y1 - y0 < max(h for _, h in sizes.values()):
        raise ValueError(f"common sky coverage too small for the boxes: x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    def box(ax, ay, name):
        w, h = sizes[name]
        bx = {"left": x0, "right": x1 - w, "centre": cx - w / 2}[ax]
        by = {"top": y0, "bottom": y1 - h, "centre": cy - h / 2}[ay]
        return (int(round(bx)), int(round(bx + w)), int(round(by)), int(round(by + h)))
    return {"topright": box("right", "top", "topright"), "centre": box("centre", "centre", "centre"),
            "topleft": box("left", "top", "topleft"), "rightlower": box("right", "bottom", "rightlower")}


def span_table(sequence: dict, results: dict, cadence_seconds: float | None) -> list[dict]:
    """Per checked frame: span in images (and minutes when the cadence is known) with the four-box RMS and mean offsets."""
    ref = sequence["images"][sequence["header"]["reference_image"]]
    by_stem = {im["stem"]: im for im in sequence["images"]}
    rows = {}
    for key, e in results.items():
        if "offsets" not in e:
            continue
        stem, box = key.split("|")
        span = by_stem[stem]["index"] - ref["index"]
        row = rows.setdefault(stem, {"frame": stem, "span_images": span,
                                     "span_minutes": (span * cadence_seconds / 60.0) if cadence_seconds else None, "boxes": {}})
        s = e["offsets"]
        row["boxes"][box] = {"rms_px": s["rms_px"], "mean_dx_px": s["mean_dx_px"], "mean_dy_px": s["mean_dy_px"], "count": s["count"],
                             "sigma_major_frame": e["shape_frame"].get("median_sigma_major_px"), "sigma_major_reference": e["shape_reference"].get("median_sigma_major_px")}
    for row in rows.values():
        values = [b["rms_px"] for b in row["boxes"].values() if b["rms_px"] is not None]
        row["max_rms_px"] = max(values) if values else None
    return sorted(rows.values(), key=lambda r: r["span_images"])


def master_vs_reference_frame(work: Path, sequence: dict, hypothesis, boxes: dict, plane: int, run: Path, *, log=print) -> dict:
    """T4: same stars, native master vs the native reference light frame (shapes; position-matched)."""
    out = run / "master"
    out.mkdir()
    ref_pos = sequence["header"]["reference_image"]
    ref_stem = sequence["images"][ref_pos]["stem"]
    frame_path, master_path = work / f"{ref_stem}.fit", work / "sky_master.fit"
    header, _ = fits_crop.read_header(frame_path)
    height = int(header["NAXIS2"])
    seeds = base._reference_seed_stars(work / "cache" / f"{ref_stem}.lst", hypothesis, height)
    results = {}
    for name, box in boxes.items():
        x0, x1, y0, y1 = box
        xy = [(x, y) for x, y, _ in base._in_box(seeds, box)]
        frame_crop = fits_crop.crop_top_down(frame_path, plane, x0, x1, y0, y1)
        master_crop = fits_crop.crop_top_down(master_path, plane, x0, x1, y0, y1) * base.MASTER_SCALE
        shape_frame = fits_crop.measure_star_shapes(frame_crop, xy)
        shape_master = fits_crop.measure_star_shapes(master_crop, xy)
        paired = base.paired_shapes(shape_frame, shape_master)      # "old" = single reference frame, "new" = master
        entry = {"box": list(box), "stars_in_box": len(xy), "paired_comparison": paired,
                 "frame_summary": {k: v for k, v in shape_frame.items() if k != "stars"},
                 "master_summary": {k: v for k, v in shape_master.items() if k != "stars"}, "reference": ref_stem}
        fits_crop.write_png_gray(out / f"{name}_master.png", fits_crop.asinh_stretch(master_crop, **base.STRETCH))
        fits_crop.write_png_gray(out / f"{name}_frame.png", fits_crop.asinh_stretch(frame_crop, **base.STRETCH))
        _write_json(out / f"{name}.json", {**entry, "frame_stars": shape_frame.get("stars", []), "master_stars": shape_master.get("stars", [])})
        results[name] = entry
        log(f"  master {name:10s} paired={paired.get('paired', 0):3d} sigma_major frame={_fmt(paired.get('old_median_sigma_major_px'))} "
            f"master={_fmt(paired.get('new_median_sigma_major_px'))} ratio={_fmt(paired.get('median_ratio_new_over_old'))} "
            f"minor ratio={_fmt(paired.get('median_ratio_sigma_minor'))}")
    return {"dir": str(out), "plane": plane, "reference": ref_stem, "results": results}


def judge_master(results: dict, criteria: dict) -> dict:
    failing = []
    for name, entry in results.items():
        p = entry.get("paired_comparison", {})
        if p.get("paired", 0) < criteria["min_paired"]:
            failing.append(f"{name}: fewer than {criteria['min_paired']} paired stars")
            continue
        ratios = {"sigma_major": p["new_median_sigma_major_px"] / p["old_median_sigma_major_px"], "sigma_minor": p["median_ratio_sigma_minor"]}
        for key, value in ratios.items():
            if value > criteria["max_ratio_master_over_frame"]:
                failing.append(f"{name}: {key} master/frame {value:.3f} > {criteria['max_ratio_master_over_frame']}")
    return {"passed": not failing, "failing": failing}


def render(report: dict) -> str:
    v = report["verdict"]
    t1, t2, t3, t3b, t4 = report["T1"], report["T2"], report["T3"], report["T3b"], report["T4"]
    lines = [f"# S2 长跨度产品验收（{report['run']}）", "",
             f"生成：{report['environment']['timestamp']} · 作业 `{report['job']['dir']}` · 输入 {report['inputs'].get('count')} 帧"
             f"（每 {report['inputs'].get('every')} 帧取 1，{report['inputs'].get('raw_dir')}）",
             f"结论：**{'通过' if v['passed'] else '未通过'}**（T1 {t1['passed']} · T2 {t2['passed']} · T3 {t3['judgement']['passed']} · T3b {t3b['judgement']['passed']} · T4 {t4['judgement']['passed']}）", ""]
    if v["tolerated"]:
        lines += [f"容许例外（列出，不隐藏）：{v['tolerated']}", ""]
    if v["failures"]:
        lines += ["未通过项："] + [f"- {f}" for f in v["failures"]] + [""]
    rule = report.get("box_rule") or {}
    if rule.get("rule") == "common_sky_coverage":
        cov = rule["coverage"]
        lines += [f"测量框（规则：全部帧公共天空覆盖区 x {cov['x0']:.0f}–{cov['x1']:.0f}、y {cov['y0']:.0f}–{cov['y1']:.0f}（地平线 y<{cov['sky_limit_y']:.0f}，{cov['frames']} 帧），"
                  f"内缩 {rule['margin_px']} px）：" + "；".join(f"{k} x {v[0]}–{v[1]} y {v[2]}–{v[3]}" for k, v in report["boxes"].items()), ""]
    else:
        lines += [f"测量框（{rule.get('rule', 'fixed')}）：" + "；".join(f"{k} x {v[0]}–{v[1]} y {v[2]}–{v[3]}" for k, v in report["boxes"].items()), ""]
    lines += ["## T1 作业", "", "| 检查 | 结果 |", "|---|---|"]
    for key, value in t1["checks"].items():
        lines.append(f"| {key} | {'✔' if value else ('–' if value is None else '✘')} |")
    ref = t1["reference"]
    lines += ["", f"参考帧：{ref['stem']}（位置 {ref['position']}，图像 {ref['index']}，日志选择 {ref['log_choice']}）→ 原片 `{report['inputs'].get('reference_source')}`",
              f"探测阶段 Siril 丢弃：{t2.get('frames_dropped_by_siril')} · 模型救回：{t2.get('recovered')} · 正式配准登记 {t1.get('registered_in_stack_sequence')}/{t1.get('sequence_images')} 帧，堆栈 {t1['diagnostics'].get('stacked_count')}/{t1['diagnostics'].get('input_count')}"
              f" · 要求登记 ≥ {t1.get('min_registered_fraction', 1.0):.0%}（像质门槛排除的帧不计）· 未登记帧：{t1.get('excluded_frames') or '无'}",
              (lambda g: (f"像质门槛（相对本序列中位，{g.get('thresholds')}）：排除 " + ("、".join(f"{e['stem']}（{'，'.join(e['reasons'])}）" for e in g.get('excluded', [])) or "无")
                          + f"；中位 {g.get('medians')}") if g.get('in_effect') else "像质门槛：未生效（退回 Siril 两段式或无拟合）")(t1.get('quality_gate') or {}),
              f"配准来源：{t1.get('registration_source')} · " + (lambda d: (f"本产品写入 {d.get('registered')} 帧矩阵，未配准 {d.get('unregistered_frames')}，"
                                                                   f"留出 RMS 中位 总体 {_fmt(d.get('validation_overall_rms_median_px'))} / 四角 {_fmt(d.get('validation_corner_rms_median_px'))} px"
                                                                   if d else "Siril 自行匹配"))(t1.get("registration_detail") or {}),
              f"Siril 用时 convert {_fmt(t1['siril_seconds']['convert'], 1)} s，probe {_fmt(t1['siril_seconds']['probe'], 1)} s，stack {_fmt(t1['siril_seconds']['stack'], 1)} s · 拟合 {_fmt(t1.get('calibration_seconds'), 1)} s · 作业总用时 {report['job']['wall_seconds']:.0f} s · work 目录 {_fmt(report['job'].get('work_gib'), 1)} GiB",
              f"母版（原生）：{t1['master']} · 去畸变堆栈：{t1.get('undistorted_master')}", ""]
    lines += ["## T2 作业自拟的畸变项（work/distortion.json）", "",
              f"状态 {t2['status']} · 参数 {t2['parameters']} · 最远角位移 {_fmt(t2['corner_shift_px'], 1)} px · 放大率 {_fmt(t2['magnification'], 4)}",
              f"留出四角 RMS 中位：纯单应 {_fmt((t2['validation_corner_rms_median_px'] or {}).get('homography_only'))} → 加畸变 {_fmt((t2['validation_corner_rms_median_px'] or {}).get('with_distortion'))}（比 {_fmt(t2['corner_median_ratio'])}）· 拟合帧 {len(t2.get('fit_frames') or [])} · 丢弃帧全部救回 {t2.get('dropped_all_recovered')}", "",
              "| 帧 | 跨度 | 初值 | 状态 | 匹配/帧星数 | 匹配率 | 四角 RMS | 总体 RMS |", "|---|---|---|---|---|---|---|---|"]
    for stem, e in sorted(t2["frames"].items(), key=lambda kv: kv[1].get("span") or 0):
        lines.append(f"| {stem} | {e.get('span')} | {e.get('initial_homography')} | {e.get('status')} | {e.get('matched')}/{e.get('frame_count')} | "
                     f"{_fmt(e.get('matched_fraction'), 2)} | {_fmt(e.get('corner_rms_px'))} | {_fmt(e.get('overall_rms_px'))} |")
    lines += ["", f"判定：passed {t2['passed']}，failing {t2['failing']}，tolerated {t2['tolerated']}，未救回 {t2.get('dropped_not_recovered')}", ""]
    lines += ["## T3 Siril 导出的 r_ 帧对参考 r_ 帧（去畸变网格，绿通道）", "",
              "| 帧 | 跨度（帧 / 分钟） | 框 | 星数 | RMS | 中位 | max | 平均 dx,dy | 参考 σ_major | 帧 σ_major |", "|---|---|---|---|---|---|---|---|---|---|"]
    spans = {row["frame"]: row for row in t3.get("span_table", [])}
    for key, e in t3["results"].items():
        if "offsets" not in e:
            lines.append(f"| {e.get('frame')} | | – | {e.get('status')} | | | | | | |")
            continue
        s = e["offsets"]
        row = spans.get(e["frame"], {})
        minutes = row.get("span_minutes")
        lines.append(f"| {e['frame']} | {row.get('span_images')} / {_fmt(minutes, 1)} | {key.split('|')[1]} | {s['count']} | {_fmt(s['rms_px'])} | {_fmt(s['median_px'])} | {_fmt(s['max_px'], 2)} | "
                     f"{_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)} | {_fmt(e['shape_reference'].get('median_sigma_major_px'))} | {_fmt(e['shape_frame'].get('median_sigma_major_px'))} |")
    lines += ["", "跨度趋势（每帧四框最大 RMS）：" + "；".join(f"{r['frame']} {r['span_images']:+d}（{_fmt(r['span_minutes'], 1)} min）{_fmt(r['max_rms_px'])}" for r in t3.get("span_table", [])),
              "", f"判定：{t3['judgement']}", ""]
    lines += ["## T3b 原生母版对原生参考帧（回变换几何）", "", "| 框 | 星数 | 中位 | RMS | max | 平均 dx,dy | 参考单帧 σ_major | 母版 σ_major |", "|---|---|---|---|---|---|---|---|"]
    for name, e in t3b["results"].items():
        s = e["offsets"]
        lines.append(f"| {name} | {s['count']} | {_fmt(s['median_px'])} | {_fmt(s['rms_px'])} | {_fmt(s['max_px'], 2)} | {_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)} | "
                     f"{_fmt(e['shape_frame'].get('median_sigma_major_px'))} | {_fmt(e['shape_master'].get('median_sigma_major_px'))} |")
    lines += ["", f"判定：{t3b['judgement']}", ""]
    lines += ["## T4 原生母版对原生参考帧同一批星（母版/单帧）", "",
              "| 框 | 配对星 | σ_major 母版/帧 | σ_minor 母版/帧 | 面积 母版/帧 | 流量 母版/帧（仅参考） | 帧/母版 σ_major | 帧/母版 σ_minor |", "|---|---|---|---|---|---|---|---|"]
    for name, e in t4["results"].items():
        p = e.get("paired_comparison", {})
        if not p.get("paired"):
            lines.append(f"| {name} | 0 | – | – | – | – | – | – |")
            continue
        lines.append(f"| {name} | {p['paired']} | {_fmt(p['new_median_sigma_major_px'] / p['old_median_sigma_major_px'])} | {_fmt(p['median_ratio_sigma_minor'])} | "
                     f"{_fmt(p['median_ratio_area'])} | {_fmt(p['median_ratio_flux'])} | {_fmt(p['old_median_sigma_major_px'])}/{_fmt(p['new_median_sigma_major_px'])} | "
                     f"{_fmt(p['old_median_sigma_minor_px'])}/{_fmt(p['new_median_sigma_minor_px'])} |")
    lines += ["", f"判定：{t4['judgement']}", "", "## 判据（运行前固定）", "", "```json", json.dumps(CRITERIA, ensure_ascii=False, indent=1), "```", "",
              "## 注意事项", "",
              "- T3 直接用 Siril 写出的 r_ 帧（去畸变网格），种子星为参考星表经畸变模型（含放大率）映射的位置；测量只用像素。跨度趋势是径向项之外剩余项的实测。",
              "- T3b 检查回变换的几何：原生母版里参考时刻的星应回到原片位置。",
              "- T4 与参考单帧同位置比较，视场相关的光学差异抵消；母版是 48 帧 winsorized 剔除堆栈，单帧是 Siril 按质量自选的参考帧（通常是最锐的一帧），所以这是偏严的判据；流量比只作参考（归一化后母版流量 ≈ 参考帧流量）。",
              "- 本脚本不改动原片；输入硬链接、作业与全部产物都在本运行目录内；不删除任何东西。", ""]
    return "\n".join(lines)


def _dir_gib(path: Path) -> float | None:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2 ** 30
    except OSError:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="folder with the RAW frames (read-only)")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cadence-seconds", type=float, default=None, help="seconds between folder inputs; curated 48 defaults to 39, other folders to historical 13 (display only)")
    parser.add_argument("--job-dir", default=None, help="an already-run product job to evaluate (skips Siril)")
    parser.add_argument("--siril", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=list(DEFAULT_FRAMES))
    parser.add_argument("--plane", type=int, default=1)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--sky-limit-y", type=float, default=None,
                        help="native row of the ground line in the reference frame; with it the boxes are placed by the common-coverage rule")
    parser.add_argument("--box-margin", type=int, default=200)
    parser.add_argument("--min-registered-fraction", type=float, default=1.0, help="T1: fraction of the inputs the calibration must register")
    parser.add_argument("--sky-mask", default=None, help="sky mask PNG (native geometry) handed to the product job so it also composes the ground")
    parser.add_argument("--label", default="accept-long-span")
    parser.add_argument("--support-mode", choices=("legacy", "strict_siril_1_4_4"), default="legacy")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)
    if args.cadence_seconds is None:
        args.cadence_seconds = (39.0 if not args.job_dir and
            Path(args.raw_dir).resolve() == DEFAULT_RAW_DIR.resolve() else 13.0)

    started = time.monotonic()
    hypothesis = adapter.HYPOTHESES[args.hypothesis or adapter.DEFAULT_HYPOTHESIS]
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"{args.label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    print(f"run: {run}")
    manifest = None
    if args.job_dir:
        job_dir = Path(args.job_dir).resolve()
        job = core.load_job(job_dir)
        candidate = job_dir.parent.parent / "inputs.json"
        if candidate.exists():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
        job_run = {"job": job, "timeline": None, "wall_seconds": 0.0, "reused": True}
    else:
        status = siril.doctor(args.siril)
        if not status.get("available"):
            raise SystemExit(f"Siril not available: {status.get('message')}")
        raw_dir = Path(args.raw_dir).resolve()
        files = select_frames(raw_dir, args.every, args.limit, args.start)
        if len(files) < 3:
            raise SystemExit("fewer than 3 RAW frames selected")
        if len({p.suffix.lower() for p in files}) != 1:
            raise SystemExit("mixed RAW extensions")
        manifest = link_inputs(files, run / "inputs", run / "inputs.json", every=args.every, start=args.start, raw_dir=raw_dir)
        print(f"inputs: {manifest['count']} RAW ({manifest['method']}), every {args.every} from {raw_dir}")
        print("running the product job (convert, probe, calibration, distortion-aware registration, stack, warp-back) ...")
        job_run = base.run_product_job(run / "inputs", run / "jobs", status["executable"], key=run.name, log=print,
                                       sky_mask_path=str(Path(args.sky_mask).resolve()) if args.sky_mask else None,
                                       support_mode=args.support_mode)
        job_run.update(reused=False)
        job = job_run["job"]
        job_dir = Path(job["job_dir"])
    work = job_dir / "work"
    job_run["work_gib"] = _dir_gib(work)
    _write_json(run / "job_summary.json", {"job": job, "timeline": job_run["timeline"], "wall_seconds": job_run["wall_seconds"],
                                            "reused": job_run["reused"], "work_gib": job_run["work_gib"], "inputs": manifest,
                                            "artifacts": job.get("artifacts"), "sky_mask": args.sky_mask})
    inputs_info = {"count": (manifest or {}).get("count", len(job["inputs"])), "every": (manifest or {}).get("every"), "raw_dir": (manifest or {}).get("raw_dir")}
    if job["state"] != "succeeded":
        report = {"acceptance_version": ACCEPTANCE_VERSION, "run": run.name, "job": {"dir": str(job_dir), "state": job["state"], "error": job.get("error")},
                  "inputs": inputs_info, "verdict": {"passed": False, "failures": [f"job {job['state']}: {job.get('error')}"], "tolerated": {}},
                  "environment": {"timestamp": _now()}}
        _write_json(run / "acceptance.json", report)
        (run / "ACCEPTANCE.md").write_text(f"# 验收未通过：作业 {job['state']}\n\n{job.get('error')}\n", encoding="utf-8")
        print(f"FAIL: job {job['state']}: {job.get('error')}")
        return 2
    sequence = siril_sequence.read_seq(work / core.SEQUENCE_FILE)
    print("T1 job checks ...")
    t1 = job_checks(job, job_dir, sequence, manifest, min_registered_fraction=args.min_registered_fraction)
    ref_index = t1["reference"]["index"]
    if manifest:
        source = next((f["source"] for f in manifest["frames"] if f["frame"] == ref_index), None)
        inputs_info["reference_source"] = source
    print("T2 calibration checks ...")
    gate_excluded = [e["stem"] for e in t1.get("quality_gate", {}).get("excluded", [])] if t1.get("quality_gate", {}).get("in_effect") else []
    t2 = calibration_checks(work, CRITERIA["T2"], quality_excluded=gate_excluded)
    frames = base.plan_check_frames(sequence, args.frames)
    excluded_indices = {im["index"] for im in sequence["images"] if im["stem"] in gate_excluded}
    if excluded_indices:
        print(f"T3 skips the frames the quality gate kept out of the stack: {sorted(excluded_indices)}")
        frames = [i for i in frames if i not in excluded_indices]
    failures = []
    boxes, box_rule = dict(BOXES), {"rule": "fixed_16_frame_boxes"}
    if args.sky_limit_y is not None:
        calibration = json.loads((work / "distortion.json").read_text(encoding="utf-8")) if (work / "distortion.json").exists() else {}
        try:
            header, _ = fits_crop.read_header(work / "sky_master.fit")
            coverage = common_sky_coverage(calibration, int(header["NAXIS1"]), int(header["NAXIS2"]), args.sky_limit_y)
            boxes = boxes_by_rule(coverage, margin=args.box_margin)
            box_rule = {"rule": "common_sky_coverage", "coverage": coverage, "margin_px": args.box_margin, "boxes": {k: list(v) for k, v in boxes.items()}}
            print(f"boxes by rule (common sky coverage x {coverage['x0']:.0f}-{coverage['x1']:.0f}, y {coverage['y0']:.0f}-{coverage['y1']:.0f}, margin {args.box_margin}):")
            for name, b in boxes.items():
                print(f"  {name:10s} x {b[0]}-{b[1]} y {b[2]}-{b[3]}")
        except (ValueError, KeyError, OSError) as exc:
            box_rule = {"rule": "common_sky_coverage_failed", "error": f"{type(exc).__name__}: {exc}"}
            failures.append(f"boxes: {box_rule['error']}")
            print(f"box rule failed: {exc}; using the fixed boxes")
    if t2.get("parameters"):
        model = RadialDistortion(**t2["parameters"])
        print(f"T3 r_ frames {frames} ...")
        t3 = dist_accept.rframe_checks(work, sequence, model, hypothesis, frames, boxes, args.plane, run, log=print)
        t3["judgement"] = base.judge_crops({k: v for k, v in t3["results"].items() if "offsets" in v}, CRITERIA["T3"])
        if any("offsets" not in v for v in t3["results"].values()):
            t3["judgement"]["passed"] = False
            t3["judgement"]["failing"] = t3["judgement"]["failing"] + [v["frame"] for v in t3["results"].values() if "offsets" not in v]
        cadence = args.cadence_seconds * ((manifest or {}).get("every") or 1)
        t3["span_table"] = span_table(sequence, t3["results"], cadence)
        t3["span_unit_seconds"] = cadence
    else:
        t3 = {"results": {}, "judgement": {"passed": False, "failing": ["no distortion parameters"], "tolerated": []}, "span_table": []}
    print("T3b native master vs native reference frame ...")
    t3b = dist_accept.native_checks(work, sequence, hypothesis, boxes, args.plane, run, log=print)
    t3b["judgement"] = dist_accept.judge_native(t3b["results"], CRITERIA["T3b"])
    print("T4 master vs reference frame shapes ...")
    t4 = master_vs_reference_frame(work, sequence, hypothesis, boxes, args.plane, run, log=print)
    t4["judgement"] = judge_master(t4["results"], CRITERIA["T4"])
    if not t1["passed"]:
        failures += [f"T1 {k}" for k, v in t1["checks"].items() if v is False]
    if not t2["passed"]:
        failures.append(f"T2 status {t2['status']} failing {t2['failing']} tolerated {t2['tolerated']} ratio {_fmt(t2['corner_median_ratio'])} "
                        f"dropped_not_recovered {t2.get('dropped_not_recovered')}")
    if not t3["judgement"]["passed"]:
        failures.append(f"T3 failing {t3['judgement']['failing']} tolerated {t3['judgement']['tolerated']}")
    if not t3b["judgement"]["passed"]:
        failures.append(f"T3b {t3b['judgement']['failing']}")
    if not t4["judgement"]["passed"]:
        failures.append(f"T4 {t4['judgement']['failing']}")
    tolerated = {k: v for k, v in {"T2": t2["tolerated"], "T3": t3["judgement"].get("tolerated", [])}.items() if v}
    report = {"acceptance_version": ACCEPTANCE_VERSION, "run": run.name, "criteria": CRITERIA, "boxes": {k: list(v) for k, v in boxes.items()},
              "box_rule": box_rule, "hypothesis": hypothesis.as_dict(),
              "inputs": inputs_info, "job": {"dir": str(job_dir), "wall_seconds": job_run["wall_seconds"], "reused": job_run["reused"], "work_gib": job_run["work_gib"]},
              "T1": t1, "T2": t2, "T3": t3, "T3b": t3b, "T4": t4,
              "verdict": {"passed": not failures, "failures": failures, "tolerated": tolerated},
              "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__, "timestamp": _now(),
                              "elapsed_seconds": time.monotonic() - started, "argv": sys.argv[1:] if argv is None else list(argv)}}
    _write_json(run / "acceptance.json", report)
    (run / "ACCEPTANCE.md").write_text(render(report), encoding="utf-8")
    print(f"{'PASS' if not failures else 'FAIL'}: {run / 'ACCEPTANCE.md'}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
