"""S2 acceptance of the per-job lens-distortion flow in the product (one command).

Runs the *product* job (``core``: convert -> probe registration -> distortion
calibration -> ``register -2pass -disto=file`` -> ``seqapplyreg`` -> stack ->
warp back to native geometry) on the same 16 DNGs as the reference one-pass
job, then judges it with the pre-registered criteria below.  Helpers are
shared with ``s2_accept_two_pass.py``; nothing outside the run directory is
modified.

    <output-root>/accept-distortion-<UTC stamp>-<6hex>/
        inputs/, jobs/job-<id>/          hard links + the product job
        job_summary.json                 state, diagnostics, timeline
        rframes/<frame>_<box>.json       Siril's own r_ frames vs the reference r_ frame (undistorted grid)
        native/<box>.json                native sky master vs the native reference light frame (warp-back geometry)
        master/<box>.json, *.png         old one-pass master vs new native master, same stars
        acceptance.json / ACCEPTANCE.md

Pre-registered criteria (fixed before the run):
  T1 job         succeeded; 16/16 stacked; distortion status "fitted" and applied; sky_geometry
                 native_via_undistorted_stack; 02_probe.ssf / 02_stack.ssf (and, since efficiency P1b-3,
                 02_export.ssf) match the product constants;
                 reference agrees between light_.seq, the Siril log and diagnostics; distortion.wcs and
                 sky_master_undistorted.fit present; native master float32 with the frame geometry.
  T2 calibration from work/distortion.json: every registered frame re-associated with matched
                 fraction >= 0.8 and validation corner RMS <= 0.5 px; at most one frame tolerated
                 (fraction >= 0.4, RMS <= 1.0: the frame-1 detection anomaly); corner median with
                 the term <= 0.7 x the homography-only median.
  T3 r_ frames   Siril's exported frames 1, 2, 16 (and the sequence ends) vs the reference r_ frame
                 in the four boxes: star centroid RMS <= 0.5 px (same single tolerated frame <= 1.0).
  T3b warp-back  native sky master vs the native reference light frame, same boxes: centroid median
                 <= 0.2 px and RMS <= 0.4 px in every box (the stars of the reference frame's own
                 time are put back where the raw frame has them).
  T4 master      same stars, old one-pass master vs new native master: median ratios new/old of
                 sigma_major, sigma_minor and area <= 1.03 in every box; flux ratio >= 0.97; in the
                 top-right box sigma_minor or area < 1.0 (RUN_LOG §16 located the old smear there).
Verdict: pass only if all of T1..T4 pass; tolerated exceptions are listed.

Outcome on 2026-09-17 (accept-distortion-20260917T132256Z-2ce03a, RUN_LOG §21): T1/T3/T3b/T4 passed
(T3 with no tolerated frame); T2 failed only on the corner-median ratio (0.733 > 0.70).  That
sub-criterion was mis-specified for a 3-minute set: frames within a few spans of the reference are
already at the centroid noise floor (0.15–0.20 px) with a plain homography, so the median over all
frames cannot improve by 30% whatever the model.  The criteria above are left as pre-registered; a
ratio criterion belongs to a long-span (tens of minutes) acceptance run, which has not been done.

Example (Windows, product venv, ~5 min):

    .venv\\Scripts\\python.exe scripts\\s2_accept_distortion.py --output-root outputs\\fable-s2-20260917
    .venv\\Scripts\\python.exe scripts\\s2_accept_distortion.py --job-dir <existing job>   (re-evaluate only)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from starstack import core, fits_crop, siril_sequence  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack.registration_calibration import DISTORTION_JSON, DISTORTION_WCS  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402
import s2_accept_two_pass as base  # noqa: E402

ACCEPTANCE_VERSION = "2.0.0"
BOXES = dict(base.BOXES)
CRITERIA = {
    "T1": {"distortion_status": "fitted", "sky_geometry": "native_via_undistorted_stack"},
    "T2": {"matched_fraction": 0.8, "corner_rms_px": 0.5, "tolerated_frames": 1, "tolerated_fraction": 0.4, "tolerated_rms_px": 1.0,
           "max_corner_median_ratio": 0.7},
    "T3": {"rms_px": 0.5, "tolerated_rms_px": 1.0, "tolerated_frames": 1},
    "T3b": {"median_px": 0.2, "rms_px": 0.4},
    "T4": {"max_ratio": 1.03, "min_flux_ratio": 0.97, "must_improve_box": "topright"},
}
_fmt, _write_json, _now = base._fmt, base._write_json, base._now


def job_checks(job: dict, job_dir: Path, reference_job: dict, sequence: dict) -> dict:
    checks_v1 = base.job_checks(job, job_dir, reference_job, sequence)
    work = job_dir / "work"
    diagnostics = job.get("diagnostics") or {}
    distortion = diagnostics.get("distortion") or {}
    applied = bool(distortion.get("applied"))
    own = diagnostics.get("registration_source") == getattr(core, "REGISTRATION_SOURCE_OWN", "starstack_calibration")
    expected_stack = core.stack_script(DISTORTION_WCS if applied else None, own_registration=own) if own else core.stack_script(DISTORTION_WCS if applied else None)
    from starstack import registered_support as rs
    support = job.get("support_mode") == rs.MODE
    if support:
        expected_stack = ["cd work", "seqclean r_light -stat", core.STACK_COMMAND]
    expected_registration = (core.registration_commands(DISTORTION_WCS, own_registration=True) if own
                             else core.registration_commands(DISTORTION_WCS if applied else None))
    if support:
        expected_registration = [rs.EXPORT_COMMAND] if own else [core.PROBE_SCRIPT[1], rs.EXPORT_COMMAND]
    stack_lines = (job_dir / "02_stack.ssf").read_text(encoding="utf-8").split("\n") if (job_dir / "02_stack.ssf").exists() else []
    stack_lines = base.stack_settings.without_settings(stack_lines)[0]   # EFF-D4 contract jobs: settings lines dropped
    probe_lines = (job_dir / "02_probe.ssf").read_text(encoding="utf-8").split("\n") if (job_dir / "02_probe.ssf").exists() else []
    checks = dict(checks_v1["checks"])
    checks.update({
        "registration_commands": diagnostics.get("registration") == expected_registration,
        # strict support's export/stack layout is unchanged; L accepts its pre-P1b and its split layout (P1b-3)
        "stack_script_matches_product": ((stack_lines[1:1 + len(expected_stack)] == expected_stack if stack_lines else None)
                                         if support else base.stack_scripts_match(job_dir, expected_stack)),
        "probe_script_matches_product": probe_lines[1:1 + len(core.PROBE_SCRIPT)] == core.PROBE_SCRIPT if probe_lines else False,
        "distortion_fitted_and_applied": distortion.get("status") == CRITERIA["T1"]["distortion_status"] and applied,
        "sky_geometry_native_via_undistorted_stack": diagnostics.get("sky_geometry") == CRITERIA["T1"]["sky_geometry"],
        "distortion_files_present": (work / DISTORTION_WCS).is_file() and (work / DISTORTION_JSON).is_file(),
        "undistorted_master_present": (work / "sky_master_undistorted.fit").is_file(),
        # a job with a sky mask also records "foreground" before "convert" (RUN_LOG §25); the three stages must follow in order
        "checkpoints_convert_distortion_stack": [c for c in (job.get("checkpoints") or []) if c not in ("foreground", "transients")] == ["convert", "distortion", "stack"],
        "stackcnt_carried_to_native_master": (checks_v1.get("master") or {}).get("STACKCNT") is not None,
    })
    if support:
        export_lines = (job_dir / "02_export.ssf").read_text(encoding="utf-8").splitlines()
        checks["export_script_matches_product"] = export_lines[1:] == ["cd work", rs.EXPORT_COMMAND]
        checks["support_checkpoint_present"] = (work / rs.RECORD).is_file() and (work / rs.NATIVE_VALID).is_file()
    passed = all(v for v in checks.values() if v is not None)
    probe_log = (job_dir / "02_probe.log").read_text(encoding="utf-8", errors="replace") if (job_dir / "02_probe.log").exists() else ""
    return {**checks_v1, "passed": passed, "checks": checks, "distortion": distortion, "registration_source": diagnostics.get("registration_source"),
            "registration_detail": diagnostics.get("registration_detail"),
            "undistorted_master": base.fits_summary(work / "sky_master_undistorted.fit") if (work / "sky_master_undistorted.fit").exists() else None,
            "siril_seconds": {**checks_v1["siril_seconds"], "probe": base.parse_execution_seconds(probe_log)},
            "calibration_seconds": distortion.get("seconds")}


def calibration_checks(work: Path, criteria: dict) -> dict:
    record = json.loads((work / DISTORTION_JSON).read_text(encoding="utf-8")) if (work / DISTORTION_JSON).exists() else {}
    frames = (record.get("reassociation") or {}).get("frames") or {}
    failing, tolerated = [], []
    for stem, e in frames.items():
        fraction, rms = e.get("matched_fraction") or 0.0, e.get("corner_rms_px")
        if e.get("status") != "associated" or rms is None:
            failing.append(stem)
        elif fraction >= criteria["matched_fraction"] and rms <= criteria["corner_rms_px"]:
            continue
        elif fraction >= criteria["tolerated_fraction"] and rms <= criteria["tolerated_rms_px"]:
            tolerated.append(stem)
        else:
            failing.append(stem)
    medians = record.get("validation_corner_rms_median_px") or {}
    ratio = (medians.get("with_distortion") / medians.get("homography_only")) if medians.get("homography_only") else None
    passed = (record.get("status") == "fitted" and not failing and len(tolerated) <= criteria["tolerated_frames"]
              and ratio is not None and ratio <= criteria["max_corner_median_ratio"])
    return {"passed": passed, "failing": failing, "tolerated": tolerated, "status": record.get("status"), "reason": record.get("reason"),
            "parameters": record.get("parameters"), "corner_shift_px": record.get("corner_shift_px"), "magnification": record.get("magnification"),
            "validation_corner_rms_median_px": medians, "corner_median_ratio": ratio, "fit_frames": record.get("fit_frames"),
            "frames_dropped_by_siril": record.get("frames_dropped_by_siril"), "recovered": (record.get("reassociation") or {}).get("recovered_dropped_frames"),
            "frames": {stem: {k: e.get(k) for k in ("span", "status", "initial_homography", "matched", "frame_count", "matched_fraction", "corner_rms_px", "overall_rms_px")}
                       for stem, e in frames.items()},
            "seconds": record.get("seconds")}


def rframe_checks(work: Path, sequence: dict, model: RadialDistortion, hypothesis, frames: list[int], boxes: dict, plane: int, run: Path,
                  *, log=print) -> dict:
    """Siril's own registered frames against the reference r_ frame: what actually goes into the stack (undistorted grid)."""
    out = run / "rframes"
    out.mkdir()
    ref_pos = sequence["header"]["reference_image"]
    ref_stem = sequence["images"][ref_pos]["stem"]
    header, _ = fits_crop.read_header(work / f"r_{ref_stem}.fit")
    scale = 65535.0 if header["BITPIX"] == "-32" else 1.0
    height = int(header["NAXIS2"])
    raw = base._reference_seed_stars(work / "cache" / f"{ref_stem}.lst", hypothesis, height)
    seeds = [tuple(p) for p in model.apply(np.array(raw, dtype=np.float64))]          # where those stars sit in the undistorted grid
    by_index = {im["index"]: im for im in sequence["images"]}
    results = {}
    for index in frames:
        stem = by_index[index]["stem"]
        if not (work / f"r_{stem}.fit").exists():
            results[f"{stem}|all"] = {"frame": stem, "status": "no_r_frame"}
            continue
        for name, box in boxes.items():
            x0, x1, y0, y1 = box
            xy = [(x, y) for x, y, _ in base._in_box(seeds, box)]
            ref_crop = fits_crop.crop_top_down(work / f"r_{ref_stem}.fit", plane, x0, x1, y0, y1) * scale
            crop = fits_crop.crop_top_down(work / f"r_{stem}.fit", plane, x0, x1, y0, y1) * scale
            offsets = fits_crop.measure_star_offsets(ref_crop, crop, xy, radius=4)
            entry = {"frame": stem, "box": list(box), "stars_in_box": len(xy), "offsets": offsets["summary"],
                     "shape_reference": {k: v for k, v in fits_crop.measure_star_shapes(ref_crop, xy).items() if k != "stars"},
                     "shape_frame": {k: v for k, v in fits_crop.measure_star_shapes(crop, xy).items() if k != "stars"}}
            _write_json(out / f"{stem}_{name}.json", {**entry, "offsets_detail": offsets})
            results[f"{stem}|{name}"] = entry
            s = offsets["summary"]
            log(f"  r_frame {stem} {name:10s} n={s['count']:3d} rms={_fmt(s['rms_px'])} mean=({_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)})")
    return {"dir": str(out), "reference": ref_stem, "grid": "undistorted reference grid (magnification applied)", "results": results}


def native_checks(work: Path, sequence: dict, hypothesis, boxes: dict, plane: int, run: Path, *, log=print) -> dict:
    """Native sky master vs the native reference light frame: the warp-back must put the stars back where the raw frame has them."""
    out = run / "native"
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
        offsets = fits_crop.measure_star_offsets(frame_crop, master_crop, xy, radius=4)
        entry = {"box": list(box), "stars_in_box": len(xy), "offsets": offsets["summary"],
                 "shape_frame": {k: v for k, v in fits_crop.measure_star_shapes(frame_crop, xy).items() if k != "stars"},
                 "shape_master": {k: v for k, v in fits_crop.measure_star_shapes(master_crop, xy).items() if k != "stars"}}
        _write_json(out / f"{name}.json", {**entry, "offsets_detail": offsets})
        results[name] = entry
        s = offsets["summary"]
        log(f"  native {name:10s} n={s['count']:3d} median={_fmt(s['median_px'])} rms={_fmt(s['rms_px'])} mean=({_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)})")
    return {"dir": str(out), "reference": ref_stem, "results": results}


def judge_native(results: dict, criteria: dict) -> dict:
    failing = []
    for name, e in results.items():
        s = e["offsets"]
        if s.get("median_px") is None or s["median_px"] > criteria["median_px"] or (s.get("rms_px") or 9) > criteria["rms_px"]:
            failing.append(f"{name}: median {_fmt(s.get('median_px'))} rms {_fmt(s.get('rms_px'))}")
    return {"passed": not failing, "failing": failing}


def judge_master(results: dict, criteria: dict) -> dict:
    failing = []
    improved = None
    for name, entry in results.items():
        p = entry.get("paired_comparison", {})
        if p.get("paired", 0) < 5:
            failing.append(f"{name}: fewer than 5 paired stars")
            continue
        ratios = {"sigma_major": p["new_median_sigma_major_px"] / p["old_median_sigma_major_px"], "sigma_minor": p["median_ratio_sigma_minor"],
                  "area": p["median_ratio_area"]}
        for key, value in ratios.items():
            if value > criteria["max_ratio"]:
                failing.append(f"{name}: {key} new/old {value:.3f} > {criteria['max_ratio']}")
        if p["median_ratio_flux"] < criteria["min_flux_ratio"]:
            failing.append(f"{name}: flux new/old {p['median_ratio_flux']:.3f} < {criteria['min_flux_ratio']}")
        if name == criteria["must_improve_box"]:
            improved = ratios["sigma_minor"] < 1.0 or ratios["area"] < 1.0
            if not improved:
                failing.append(f"{name}: neither sigma_minor ({ratios['sigma_minor']:.3f}) nor area ({ratios['area']:.3f}) improved")
    return {"passed": not failing and improved is True, "failing": failing, "must_improve_box_improved": improved}


def render(report: dict) -> str:
    v = report["verdict"]
    t1, t2, t3, t3b, t4 = report["T1"], report["T2"], report["T3"], report["T3b"], report["T4"]
    lines = [f"# S2 镜头畸变流程产品验收（{report['run']}）", "",
             f"生成：{report['environment']['timestamp']} · 参考作业 `{report['reference_job']['dir']}` · 新作业 `{report['job']['dir']}`",
             f"结论：**{'通过' if v['passed'] else '未通过'}**（T1 {t1['passed']} · T2 {t2['passed']} · T3 {t3['judgement']['passed']} · T3b {t3b['judgement']['passed']} · T4 {t4['judgement']['passed']}）", ""]
    if v["tolerated"]:
        lines += [f"容许例外（列出，不隐藏）：{v['tolerated']}", ""]
    if v["failures"]:
        lines += ["未通过项："] + [f"- {f}" for f in v["failures"]] + [""]
    lines += ["## T1 作业", "", "| 检查 | 结果 |", "|---|---|"]
    for key, value in t1["checks"].items():
        lines.append(f"| {key} | {'✔' if value else ('–' if value is None else '✘')} |")
    ref = t1["reference"]
    lines += ["", f"参考帧：{ref['stem']}（位置 {ref['position']}，图像 {ref['index']}，日志选择 {ref['log_choice']}）→ 原片 `{ref.get('original_input')}`",
              f"Siril 用时 convert {_fmt(t1['siril_seconds']['convert'], 1)} s，probe {_fmt(t1['siril_seconds']['probe'], 1)} s，stack {_fmt(t1['siril_seconds']['stack'], 1)} s · 拟合 {_fmt(t1.get('calibration_seconds'), 1)} s · 作业总用时 {report['job']['wall_seconds']:.0f} s",
              f"母版（原生）：{t1['master']} · 去畸变堆栈：{t1.get('undistorted_master')}", ""]
    lines += ["## T2 作业自拟的畸变项（work/distortion.json）", "",
              f"状态 {t2['status']} · 参数 {t2['parameters']} · 最远角位移 {_fmt(t2['corner_shift_px'], 1)} px · 放大率 {_fmt(t2['magnification'], 4)}",
              f"留出四角 RMS 中位：纯单应 {_fmt((t2['validation_corner_rms_median_px'] or {}).get('homography_only'))} → 加畸变 {_fmt((t2['validation_corner_rms_median_px'] or {}).get('with_distortion'))}（比 {_fmt(t2['corner_median_ratio'])}）· Siril 丢弃 {t2['frames_dropped_by_siril']} · 救回 {t2['recovered']}", "",
              "| 帧 | 跨度 | 初值 | 状态 | 匹配/帧星数 | 匹配率 | 四角 RMS | 总体 RMS |", "|---|---|---|---|---|---|---|---|"]
    for stem, e in sorted(t2["frames"].items(), key=lambda kv: kv[1].get("span") or 0):
        lines.append(f"| {stem} | {e.get('span')} | {e.get('initial_homography')} | {e.get('status')} | {e.get('matched')}/{e.get('frame_count')} | "
                     f"{_fmt(e.get('matched_fraction'), 2)} | {_fmt(e.get('corner_rms_px'))} | {_fmt(e.get('overall_rms_px'))} |")
    lines += ["", f"判定：passed {t2['passed']}，failing {t2['failing']}，tolerated {t2['tolerated']}", ""]
    lines += ["## T3 Siril 导出的 r_ 帧对参考 r_ 帧（去畸变网格，绿通道）", "",
              "| 帧 | 框 | 星数 | RMS | 中位 | max | 平均 dx,dy | 参考 σ_major | 帧 σ_major |", "|---|---|---|---|---|---|---|---|---|"]
    for key, e in t3["results"].items():
        if "offsets" not in e:
            lines.append(f"| {e.get('frame')} | – | {e.get('status')} | | | | | | |")
            continue
        s = e["offsets"]
        lines.append(f"| {e['frame']} | {key.split('|')[1]} | {s['count']} | {_fmt(s['rms_px'])} | {_fmt(s['median_px'])} | {_fmt(s['max_px'], 2)} | "
                     f"{_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)} | {_fmt(e['shape_reference'].get('median_sigma_major_px'))} | {_fmt(e['shape_frame'].get('median_sigma_major_px'))} |")
    lines += ["", f"判定：{t3['judgement']}", ""]
    lines += ["## T3b 原生母版对原生参考帧（回变换几何）", "", "| 框 | 星数 | 中位 | RMS | max | 平均 dx,dy | 参考单帧 σ_major | 母版 σ_major |", "|---|---|---|---|---|---|---|---|"]
    for name, e in t3b["results"].items():
        s = e["offsets"]
        lines.append(f"| {name} | {s['count']} | {_fmt(s['median_px'])} | {_fmt(s['rms_px'])} | {_fmt(s['max_px'], 2)} | {_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)} | "
                     f"{_fmt(e['shape_frame'].get('median_sigma_major_px'))} | {_fmt(e['shape_master'].get('median_sigma_major_px'))} |")
    lines += ["", f"判定：{t3b['judgement']}", ""]
    lines += ["## T4 新旧母版同一批星（旧 = 一段式；新 = 畸变流程原生母版）", "",
              "| 框 | 配对星 | σ_major 新/旧 | σ_minor 新/旧 | 面积 新/旧 | 流量 新/旧 | 旧/新 σ_major | 旧/新 σ_minor |", "|---|---|---|---|---|---|---|---|"]
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
              "- T3 直接用 Siril 写出的 r_ 帧（去畸变网格），种子星为参考星表经畸变模型（含放大率）映射的位置；测量只用像素。",
              "- T3b 检查回变换的几何：原生母版里参考时刻的星应回到原片位置；母版比单帧信噪比高、经过两次重采样，σ 的差异分开看。",
              "- T4 同一批星的配对比较；旧母版为一段式流程（参考 light_00001），种子经旧 H 映射。",
              "- 本脚本不改动参考作业、原片与旧输出；新作业与输入硬链接都在本运行目录内。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-job", default=str(base.DEFAULT_REFERENCE_JOB))
    parser.add_argument("--job-dir", default=None, help="an already-run product job to evaluate (skips Siril)")
    parser.add_argument("--siril", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=[1, 2, 16])
    parser.add_argument("--plane", type=int, default=1)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--label", default="accept-distortion")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    started = time.monotonic()
    reference_dir = Path(args.reference_job).resolve()
    reference_job = json.loads((reference_dir / "job.json").read_text(encoding="utf-8"))
    hypothesis = adapter.HYPOTHESES[args.hypothesis or adapter.DEFAULT_HYPOTHESIS]
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"{args.label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    print(f"run: {run}")
    if args.job_dir:
        job_dir = Path(args.job_dir).resolve()
        job = core.load_job(job_dir)
        job_run = {"job": job, "timeline": None, "wall_seconds": 0.0, "inputs": None, "reused": True}
    else:
        inputs = base.link_inputs(reference_job, run / "inputs")
        print(f"inputs: {inputs['count']} files ({inputs['method']})")
        print("running the product job (probe, calibration, distortion-aware registration, stack, warp-back) ...")
        job_run = base.run_product_job(run / "inputs", run / "jobs", args.siril or reference_job["engine"]["executable"], key=run.name, log=print)
        job_run.update(inputs=inputs, reused=False)
        job = job_run["job"]
        job_dir = Path(job["job_dir"])
    _write_json(run / "job_summary.json", {"job": job, "timeline": job_run["timeline"], "wall_seconds": job_run["wall_seconds"],
                                            "inputs": job_run["inputs"], "reused": job_run["reused"]})
    if job["state"] != "succeeded":
        report = {"acceptance_version": ACCEPTANCE_VERSION, "run": run.name, "job": {"dir": str(job_dir), "state": job["state"], "error": job.get("error")},
                  "verdict": {"passed": False, "failures": [f"job {job['state']}: {job.get('error')}"], "tolerated": {}}, "environment": {"timestamp": _now()}}
        _write_json(run / "acceptance.json", report)
        (run / "ACCEPTANCE.md").write_text(f"# 验收未通过：作业 {job['state']}\n\n{job.get('error')}\n", encoding="utf-8")
        print(f"FAIL: job {job['state']}: {job.get('error')}")
        return 2
    work = job_dir / "work"
    new_seq = siril_sequence.read_seq(work / core.SEQUENCE_FILE)
    old_seq = siril_sequence.read_seq(reference_dir / "work" / core.SEQUENCE_FILE)
    print("T1 job checks ...")
    t1 = job_checks(job, job_dir, reference_job, new_seq)
    print("T2 calibration checks ...")
    t2 = calibration_checks(work, CRITERIA["T2"])
    frames = base.plan_check_frames(new_seq, args.frames)
    failures = []
    if t2.get("parameters"):
        model = RadialDistortion(**t2["parameters"])
        print(f"T3 r_ frames {frames} ...")
        t3 = rframe_checks(work, new_seq, model, hypothesis, frames, BOXES, args.plane, run, log=print)
        t3["judgement"] = base.judge_crops({k: v for k, v in t3["results"].items() if "offsets" in v}, CRITERIA["T3"])
        if any("offsets" not in v for v in t3["results"].values()):
            t3["judgement"]["passed"] = False
            t3["judgement"]["failing"] = t3["judgement"]["failing"] + [v["frame"] for v in t3["results"].values() if "offsets" not in v]
    else:
        t3 = {"results": {}, "judgement": {"passed": False, "failing": ["no distortion parameters"], "tolerated": []}}
    print("T3b native master vs native reference frame ...")
    t3b = native_checks(work, new_seq, hypothesis, BOXES, args.plane, run, log=print)
    t3b["judgement"] = judge_native(t3b["results"], CRITERIA["T3b"])
    print("T4 master comparison ...")
    t4 = base.master_comparison(reference_dir, work, old_seq, new_seq, hypothesis, BOXES, args.plane, run, log=print)
    t4["judgement"] = judge_master(t4["results"], CRITERIA["T4"])
    if not t1["passed"]:
        failures += [f"T1 {k}" for k, v in t1["checks"].items() if v is False]
    if not t2["passed"]:
        failures.append(f"T2 status {t2['status']} failing {t2['failing']} tolerated {t2['tolerated']} ratio {_fmt(t2['corner_median_ratio'])}")
    if not t3["judgement"]["passed"]:
        failures.append(f"T3 failing {t3['judgement']['failing']} tolerated {t3['judgement']['tolerated']}")
    if not t3b["judgement"]["passed"]:
        failures.append(f"T3b {t3b['judgement']['failing']}")
    if not t4["judgement"]["passed"]:
        failures.append(f"T4 {t4['judgement']['failing']}")
    tolerated = {k: v for k, v in {"T2": t2["tolerated"], "T3": t3["judgement"].get("tolerated", [])}.items() if v}
    report = {"acceptance_version": ACCEPTANCE_VERSION, "run": run.name, "criteria": CRITERIA, "boxes": BOXES, "hypothesis": hypothesis.as_dict(),
              "reference_job": {"dir": str(reference_dir)}, "job": {"dir": str(job_dir), "wall_seconds": job_run["wall_seconds"], "reused": job_run["reused"], "inputs": job_run["inputs"]},
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
