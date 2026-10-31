"""S3 acceptance: the sky stack stays clean above the ground (RUN_LOG §25).

2026-09-25: default input is the curated 48-frame folder (every=1);
old full-DNG examples below are historical. Use that baseline job sky_mask.png.


Runs the product job on a RAW subset *with the sky mask* through the long-span
acceptance (T1–T4 unchanged: registration, Siril r_ frames, warp-back, star
shapes), then adds the horizon criteria, fixed before the run:

  H1  the job masked every frame (diagnostics.sky_validity applied, source job_mask_static or job_mask_per_frame (1.2.0),
      dilation = the product's SKY_VALIDITY_DILATION_PX),
      wrote the ground master before masking and the coverage map after the stack
  H2  the sky master is exactly zero where no frame observed the sky and non-zero where one did (agreement >= 0.99)
  H3c colour band gone: in three column ranges spread over the width, band by band (25 px) from 4 to 300 px above the
      mask edge, the master's R/G and B/G equal the pixel-weighted colour of the frames that contribute there (their
      exported r_ frames, valid pixels only) within 0.02 — the stack's colour is its contributors' colour
      (H3 of 1.0.0, the master against the reference frame, is still reported: it cannot separate ground contamination
      from the sky's own colour drift over the night, RUN_LOG §25)
  H5  same-position blend against the recorded effective sky weight, tolerance 1e-5.
      Full-resolution by default; missing evidence is insufficient_data. Natural dark
      structure and cross-location brightness ratios are observations, not blend failures.

    <output-root>/accept-horizon-<stamp>-<hex>/     the long-span run folder (ACCEPTANCE.md, acceptance.json, jobs/...)
        HORIZON.md, horizon.json                    the S3 verdict and numbers
        horizon_before_after.png                    sky masters at the horizon, unmasked job vs this job, + coverage

With --run, the old acceptance.json and job are read only. New HORIZON.md,
horizon.json and comparison images go into a fresh evaluation folder.

Example (Windows, product venv):

    .venv\\Scripts\\python.exe scripts\\s3_accept_horizon.py --every 3 --sky-limit-y 4540 \\
        --sky-mask outputs\\z8-meteor-20260916\\full-jobs\\job-5a2213e524204fb4\\sky_mask.png \\
        --compare-job outputs\\fable-s2-20260917\\accept-long-span-20260918T054628Z-779ea6\\jobs\\job-25bedce91526c43f5e126822
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import hashlib
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from starstack import fits_crop, sky_validity as sv  # noqa: E402
from starstack import horizon_acceptance as ha  # noqa: E402
from starstack import core  # noqa: E402
from starstack.acceptance_status import summarize_checks, render_checks  # noqa: E402
from starstack.h5_job_validation import evaluate_job  # noqa: E402
from starstack.registration_distortion import RadialDistortion  # noqa: E402
import s2_accept_long_span as long_span  # noqa: E402


def _load_json(text):
    data = json.loads(text)
    json.dumps(data, allow_nan=False)  # reject non-finite historical metadata explicitly
    return data


def _rgb_top_down(path: Path) -> np.ndarray:
    planes = []
    for p in range(3):
        view, header, roworder = fits_crop.plane_memmap(path, p)
        a = np.asarray(view, dtype=np.float32)
        if int(header["BITPIX"]) == 16:
            a = (a + 32768.0) / 65535.0
        planes.append(a[::-1] if roworder == "BOTTOM-UP" else a)
    return np.stack(planes, axis=-1)


def _edge_aligned_crop(path: Path, ux: np.ndarray, uy: np.ndarray, h_max: int) -> np.ndarray:
    """(h_max, ncols, 3) crop of a frame in its own grid: column ``ux[i]``, rows ``uy[i]-h_max .. uy[i]-1`` reversed
    (row 0 = just above the ground edge)."""
    x0, x1 = int(np.floor(ux.min())), int(np.ceil(ux.max())) + 1
    y0, y1 = int(np.floor(uy.min())) - h_max, int(np.ceil(uy.max())) + 1
    header, _ = fits_crop.read_header(path)
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    x0, x1, y0, y1 = max(0, x0), min(width, x1), max(0, y0), min(height, y1)
    box = np.stack([fits_crop.crop_top_down(path, p, x0, x1, y0, y1) for p in range(3)], axis=-1)
    if int(header["BITPIX"]) == 16:
        box = box / 65535.0
    out = np.zeros((h_max, len(ux), 3), dtype=np.float32)
    for i, (cx, cy) in enumerate(zip(ux, uy)):
        c = int(round(cx)) - x0
        r1 = int(round(cy)) - y0
        r0 = r1 - h_max
        if 0 <= c < box.shape[1] and r0 >= 0 and r1 <= box.shape[0]:
            out[:, i, :] = box[r0:r1, c, :][::-1]
    return out


def contributor_checks(work: Path, mask: np.ndarray, ranges, own_registration: bool, criteria: dict, log=print) -> dict:
    """H3c on the grid the stack was made in (undistorted when the job used its own registration, native otherwise)."""
    edge = ha.mask_edge_rows(mask)
    model = None
    master_path = work / "sky_master.fit"
    if own_registration:
        cal = _load_json((work / "distortion.json").read_text(encoding="utf-8"))
        model = RadialDistortion(**cal["parameters"])
        master_path = work / "sky_master_undistorted.fit"
    frames = sorted(work.glob("r_light_*.fit"))
    if not frames:
        return _stage("insufficient_data", "no exported contributor frames", grid="undistorted" if own_registration else "native", frames=0, ranges={})
    h_max, h_min, band = criteria["h_max"], criteria["h_min"], criteria["band_px"]
    edges = list(range(h_min, h_max, band)) + [h_max]
    # the stack normalises every frame to the reference (-norm=addscale); do the same to the crops from each frame's
    # global per-channel median and MAD over its non-zero pixels (sampled every 16 px)
    ref_stem = _load_json((work / "distortion.json").read_text(encoding="utf-8")).get("reference") if (work / "distortion.json").is_file() else None
    stats = {}
    for f in frames:
        planes = []
        for pl in range(3):
            view, header, roworder = fits_crop.plane_memmap(f, pl)
            sample = np.asarray(view[::16, ::16], dtype=np.float32)
            if int(header["BITPIX"]) == 16:
                sample = (sample + 32768.0) / 65535.0
            planes.append(sample)
        stats[f.name] = ha.global_stats(planes)
    ref_name = f"r_{ref_stem}.fit" if ref_stem and (work / f"r_{ref_stem}.fit").is_file() else frames[0].name
    ref_stats = stats[ref_name]
    log(f"  H3c normalisation reference {ref_name}: median {[round(v, 5) for v in ref_stats['median']]} mad {[round(v, 6) for v in ref_stats['mad']]}")
    results = {}
    for x0, x1 in ranges:
        cols = np.arange(x0, x1, 4)
        pts = np.column_stack([cols, edge[cols]]).astype(np.float64)
        keep = (edge[cols] < mask.shape[0]) & (edge[cols] - h_max >= 0)
        pts = pts[keep]
        if len(pts) == 0:
            results[f"x{x0}-{x1}"] = {"max_abs_deviation": None, "bands": [], "bands_without_data": [], "status": "no_ground_edge_in_range"}
            continue
        u = model.apply(pts) if model is not None else pts
        master_crop = _edge_aligned_crop(master_path, u[:, 0], u[:, 1], h_max)
        crops = (ha.normalise_like_stack(_edge_aligned_crop(f, u[:, 0], u[:, 1], h_max), stats[f.name], ref_stats) for f in frames)
        results[f"x{x0}-{x1}"] = ha.contributor_colour(master_crop, crops, edges, min_frame_pixels=criteria["min_frame_pixels"])
        results[f"x{x0}-{x1}"]["columns"] = int(len(pts))
        log(f"  H3c {f'x{x0}-{x1}'}: max |dev| {results[f'x{x0}-{x1}']['max_abs_deviation']} over {len(edges) - 1} bands, {len(frames)} frames")
    output = {"grid": "undistorted" if own_registration else "native", "frames": len(frames), "band_edges": edges, "ranges": results,
            "normalisation": {"reference": ref_name, "reference_stats": ref_stats, "kind": "addscale_like (median, MAD; non-zero pixels sampled every 16 px)"},
            "judgement": _judge_contributors(results, criteria)}
    json.dumps(output, allow_nan=False)
    return output


def _find_run(output_root: Path, label: str, after: float) -> Path:
    runs = [p for p in output_root.glob(f"{label}-*") if p.is_dir() and p.stat().st_mtime >= after - 5]
    if not runs:
        raise SystemExit("the long-span acceptance produced no run folder")
    return max(runs, key=lambda p: p.stat().st_mtime)


_GATES = ("H1", "H2", "H3c", "H5")
_EXIT_CODES = {"passed": 0, "failed": 2, "insufficient_data": 3, "not_applicable": 4}


def _stage(status, reason, *, samples=0, **evidence):
    summary = summarize_checks([{"name": "evidence", "status": status, "samples": samples, "reason": reason}])
    return dict(evidence, judgement=summary, passed=summary["passed"])


def _capture(fn):
    """Only known input/data errors become reported outcomes; coding errors still surface."""
    try:
        return fn()
    except (FileNotFoundError, PermissionError) as exc:
        return _stage("insufficient_data", f"{type(exc).__name__}: {exc}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return _stage("failed", f"{type(exc).__name__}: {exc}")


def _judge_contributors(results, criteria):
    checks = []
    for name, result in results.items():
        bands = result.get("bands", [])
        measured = 0
        missing = not bands or bool(result.get("bands_without_data"))
        invalid = False
        for band in bands:
            ratios = [band.get(key, {}).get("ratio") for key in ("RG", "BG")]
            if any(value is None for value in ratios):
                missing = True
            elif not all(isinstance(value, (float, int)) and math.isfinite(value) for value in ratios):
                invalid = True
            else:
                measured += 1
        deviation = result.get("max_abs_deviation")
        if invalid or (deviation is not None and not math.isfinite(deviation)):
            status, reason = "failed", "non-finite colour metric"
        elif deviation is not None and deviation > criteria["max_abs_deviation"]:
            status, reason = "failed", f"max |dev| {deviation:.3f} > {criteria['max_abs_deviation']}"
            if missing:
                reason += "; other bands lack data"
        elif measured == 0 or missing or deviation is None:
            status, reason = "insufficient_data", "one or more requested colour bands lack measurable data"
        else:
            status, reason = "passed", "all requested bands satisfy the existing colour tolerance"
        checks.append({"name": name, "status": status, "samples": measured, "reason": reason})
    return summarize_checks(checks)


def _gate_check(name, stage):
    judgement = stage.get("judgement", {})
    status = judgement.get("status", "insufficient_data")
    details = (judgement.get("failing", []) + judgement.get("incomplete", []) + judgement.get("not_applicable", []))
    reason = "; ".join(details) if details else "completed stage evidence; see detailed measurements below"
    # This overview counts assessed stage records, not independent pixels/scenes.
    return {"name": name, "status": status, "samples": 1 if status == "passed" else 0, "reason": reason}


def _verdict(report, extra=()):
    summary = summarize_checks([_gate_check(name, report.get(name, {})) for name in _GATES] + list(extra))
    # Keep the old fields for callers; incomplete is deliberately not failures.
    return dict(summary, failures=list(summary["failing"]))


def _empty_report(job_dir, reason, status="insufficient_data"):
    report = {"horizon_acceptance_version": ha.HORIZON_ACCEPTANCE_VERSION, "criteria": ha.CRITERIA, "job": str(job_dir)}
    for name in (*_GATES, "H3"):
        report[name] = _stage(status, reason)
    report["verdict"] = _verdict(report)
    return report


def _h1(job, validity, work):
    h1 = {"applied": validity.get("applied"), "source": validity.get("source"), "dilation_px": validity.get("dilation_px"),
          "frames_masked": validity.get("frames_masked"), "checkpoints": job.get("checkpoints"),
          "ground_master_present": (work / "foreground_master.fit").is_file(), "coverage_present": (work / sv.COVERAGE_PNG).is_file(),
          "coverage": validity.get("coverage"), "compose": validity.get("compose")}
    checks = []
    expected = {"applied": True, "source": ("job_mask_static", "job_mask_per_frame"), "dilation_px": core.SKY_VALIDITY_DILATION_PX}
    for key, target in expected.items():
        value = h1[key]
        ok = value in target if isinstance(target, tuple) else value == target
        status = "insufficient_data" if value is None else "passed" if ok else "failed"
        checks.append({"name": key, "status": status, "samples": int(status == "passed"), "reason": f"recorded {value!r}; required {target!r}"})
    checkpoints = h1["checkpoints"]
    inputs = job.get("inputs")
    for key, known, ok in (("checkpoints", isinstance(checkpoints, list) and bool(checkpoints), isinstance(checkpoints, list) and checkpoints[:2] == ["foreground", "convert"]),
                           ("frames_masked", isinstance(inputs, list) and bool(inputs) and h1["frames_masked"] is not None, isinstance(inputs, list) and h1["frames_masked"] == len(inputs))):
        status = "insufficient_data" if not known else "passed" if ok else "failed"
        checks.append({"name": key, "status": status, "samples": int(status == "passed"), "reason": "existing masking/order criterion satisfied" if ok and known else "missing or inconsistent masking/order evidence"})
    for key in ("ground_master_present", "coverage_present"):
        status = "passed" if h1[key] else "insufficient_data"
        checks.append({"name": key, "status": status, "samples": int(h1[key]), "reason": "required asset exists" if h1[key] else "required asset missing"})
    h1["judgement"] = summarize_checks(checks)
    h1["passed"] = h1["judgement"]["passed"]
    return h1


def horizon_checks(job_dir: Path, mask_path: Path, *, compare_job: Path | None, run: Path, log=print) -> dict:
    report = _empty_report(job_dir, "stage has not been measured")
    try:
        raw = (job_dir / "job.json").read_bytes()
        job = _load_json(raw)
        if not isinstance(job, dict):
            raise ValueError("job metadata must be an object")
        artifacts = job.get("artifacts", [])
        diagnostics = job.get("diagnostics") or {}
        if not isinstance(artifacts, list) or any(not isinstance(item, dict) for item in artifacts):
            raise ValueError("job artifacts must be a list of objects")
        if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("sky_validity") or {}, dict):
            raise ValueError("invalid sky validity metadata")
    except (FileNotFoundError, PermissionError) as exc:
        return _empty_report(job_dir, f"{type(exc).__name__}: {exc}")
    except (ValueError, OSError) as exc:
        return _empty_report(job_dir, f"{type(exc).__name__}: {exc}", "failed")
    report["job_json_sha256"] = hashlib.sha256(raw).hexdigest()
    report["H5"] = _capture(lambda: evaluate_job(job_dir, job=job, mask_path=mask_path))
    report["H5"]["passed"] = report["H5"]["judgement"]["passed"]
    if not job.get("mask"):
        for name in ("H1", "H2", "H3", "H3c"):
            report[name] = _stage("not_applicable", "job has no sky/ground mask")
        report["verdict"] = _verdict(report)
        return report
    if job.get("state") != "succeeded":
        report["verdict"] = _verdict(report)
        return report
    work = job_dir / "work"
    diagnostics = job.get("diagnostics") or {}
    if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("sky_validity", {}), dict):
        report["H1"] = _stage("failed", "invalid sky validity metadata")
        report["verdict"] = _verdict(report)
        return report
    validity = diagnostics.get("sky_validity") or {}
    report["H1"] = _capture(lambda: _h1(job, validity, work))

    def geometry():
        header, _ = fits_crop.read_header(work / "sky_master.fit")
        width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
        if min(width, height) <= 0:
            raise ValueError("empty master geometry")
        mask = sv.load_mask_top_down(mask_path, width, height)
        master = _rgb_top_down(work / "sky_master.fit")
        if master.shape != (height, width, 3) or not np.isfinite(master).all():
            raise ValueError("invalid sky master samples")
        return {"master": master, "mask": mask, "width": width}
    assets = _capture(geometry)
    if "master" not in assets:
        for name in ("H2", "H3", "H3c"):
            report[name] = assets
        report["verdict"] = _verdict(report)
        return report
    master, mask, width = assets["master"], assets["mask"], assets["width"]
    ranges = ha.column_ranges(width, span=min(700, width))
    cov, step = None, 4

    def coverage():
        nonlocal cov, step
        metadata = _load_json((work / "sky_coverage.json").read_text(encoding="utf-8"))
        step = metadata["step"]
        if type(step) is not int or step <= 0:
            raise ValueError("invalid coverage step")
        cov = sv.read_coverage_png(work / sv.COVERAGE_PNG)
        if cov.shape != master[::step, ::step, 1].shape:
            raise ValueError("coverage shape does not match its sampling grid")
        if cov.size == 0:
            return _stage("insufficient_data", "coverage has no samples")
        evidence = ha.master_coverage_consistency(master[..., 1], cov, step)
        status = "passed" if evidence["agreement"] >= ha.CRITERIA["H2"]["consistency_min"] else "failed"
        return _stage(status, "existing coverage consistency criterion", samples=evidence["samples"], **{key: value for key, value in evidence.items() if key != "samples"}, measured_samples=evidence["samples"])
    report["H2"] = _capture(coverage)
    c3 = ha.CRITERIA["H3"]

    def reference_diagnostic():
        ref_stem = diagnostics.get("reference_stem")
        if not ref_stem:
            return _stage("insufficient_data", "reference frame identity missing", ranges={})
        frame = _rgb_top_down(work / f"{ref_stem}.fit")
        if frame.shape != master.shape or not np.isfinite(frame).all():
            raise ValueError("invalid reference frame samples")
        result = ha.colour_band(master, frame, mask, ranges, h_min=c3["h_min"], h_max=c3["h_max"], normalise_rows=c3["normalise_rows"])
        json.dumps(result, allow_nan=False)
        return dict(result, legacy_judgement=ha.judge_colour_band(result, c3))
    report["H3"] = _capture(reference_diagnostic)  # diagnostic only, never a gate
    own = diagnostics.get("registration_source") == "starstack_calibration" and (work / "sky_master_undistorted.fit").is_file()
    report["H3c"] = _capture(lambda: contributor_checks(work, mask, ranges, own, ha.CRITERIA["H3c"], log=log))
    report["H3c"]["passed"] = report["H3c"]["judgement"]["passed"]
    if compare_job is not None:
        def comparison():
            cjob = _load_json((compare_job / "job.json").read_text(encoding="utf-8"))
            if not isinstance(cjob, dict):
                raise ValueError("comparison job metadata must be an object")
            cdiag = cjob.get("diagnostics") or {}
            if not isinstance(cdiag, dict):
                raise ValueError("comparison diagnostics must be an object")
            cref = cdiag.get("reference_stem")
            cmaster = _rgb_top_down(compare_job / "work" / "sky_master.fit")
            cframe = _rgb_top_down(compare_job / "work" / f"{cref}.fit")
            if cmaster.shape != master.shape or cframe.shape != master.shape:
                raise ValueError("comparison image geometry differs")
            before = ha.colour_band(cmaster, cframe, mask, ranges, h_min=c3["h_min"], h_max=c3["h_max"], normalise_rows=c3["normalise_rows"])
            json.dumps(before, allow_nan=False)
            _before_after_png(cmaster, master, cov, step, mask, ranges, run / "horizon_before_after.png")
            return dict(before, job=str(compare_job))
        report["H3_before"] = _capture(comparison)
    report["verdict"] = _verdict(report)
    for name in _GATES:
        log(f"{name}: {report[name]['judgement']['status']}")
    return report


def _before_after_png(before: np.ndarray, after: np.ndarray, cov, step: int, mask: np.ndarray, ranges, path: Path) -> None:
    """Sky masters at the horizon (RGB, one stretch per pair) for the three column ranges, plus the coverage map crop."""
    from PIL import Image, ImageDraw
    edge = ha.mask_edge_rows(mask)
    tiles = []
    for x0, x1 in ranges:
        e = int(np.median(edge[x0:x1]))
        y0, y1 = max(0, e - 400), min(mask.shape[0], e + 100)
        a = before[y0:y1, x0:x1]; b = after[y0:y1, x0:x1]
        lo = np.percentile(a[..., 1], 1); hi = np.percentile(a[..., 1], 99.5); hi = max(hi, lo + 1e-6)
        def stretch(img):
            x = np.clip((img - lo) / (hi - lo), 0, 1)
            return (255 * np.arcsinh(10 * x) / np.arcsinh(10)).astype(np.uint8)
        ta, tb = Image.fromarray(stretch(a)), Image.fromarray(stretch(b))
        c = cov[y0 // step:(y1 + step - 1) // step, x0 // step:(x1 + step - 1) // step] if cov is not None else np.zeros((1, 1))
        tc = Image.fromarray((255 * c / max(1, c.max())).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.NEAREST).convert("RGB")
        tiles.append((f"x{x0}-{x1} rows {y0}-{y1}", ta, tb, tc))
    w = tiles[0][1].width
    pad, head = 6, 22
    canvas = Image.new("RGB", (3 * w + 4 * pad, head + sum(t[1].height + pad + 18 for t in tiles) + pad), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 4), "BEFORE: unmasked job sky master", fill=(255, 255, 255)); d.text((w + 2 * pad, 4), "AFTER: masked job sky master", fill=(255, 255, 255))
    d.text((2 * w + 3 * pad, 4), "coverage (frames per pixel, white = all)", fill=(255, 255, 255))
    y = head
    for name, ta, tb, tc in tiles:
        d.text((pad, y), name, fill=(200, 200, 200)); y += 18
        canvas.paste(ta, (pad, y)); canvas.paste(tb, (w + 2 * pad, y)); canvas.paste(tc, (2 * w + 3 * pad, y))
        y += ta.height + pad
    canvas.save(path)


def render(report: dict, long_span_report: dict | None) -> str:
    overview = summarize_checks(report["verdict"]["checks"])
    lines = [f"# S3 地平线验收（{report.get('run', '')}）", "",
             f"判据版本：{ha.HORIZON_ACCEPTANCE_VERSION}。H1／H2／H3c 的算法与阈值保持；H3 仅作参考帧诊断。", "",
             render_checks(overview), "", "上表样本数按已完成的阶段记录计数，不代表独立像素、星点或场景；详细数量见各节。", "",
             "H5 2.0.0 检查同位置天空、地景与已记录实际权重的混合，绝对容差仍为 1e-5。亮度邻域比只作观察，天然暗地景不因此判错；缺输入、缺权重或漏采不计通过。", "",
             "本报告不证明完整插值支持、独立配准留出或摄影质量；旧 T1–T4 仍按其原版本和取样条件解释。", ""]
    titles = {"H1": "作业屏蔽", "H2": "母版零值与覆盖", "H3": "参考帧颜色诊断（不参与总判定）", "H3c": "贡献帧颜色", "H5": "同位置混合与接边"}
    for name, title in titles.items():
        stage = report.get(name, {})
        lines += [f"## {name} {title}", ""]
        if "judgement" in stage:
            lines += [render_checks(stage["judgement"]), ""]
        evidence = {key: value for key, value in stage.items() if key not in ("judgement", "passed")}
        lines += ["```json", json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), "```", ""]
    if "H3_before" in report:
        lines += ["## 未屏蔽作业比较（诊断）", "", "```json", json.dumps(report["H3_before"], indent=2, ensure_ascii=False, allow_nan=False), "```", ""]
    if long_span_report is not None:
        lines += ["## 长跨度 T1–T4 既有记录", "", "```json", json.dumps(long_span_report.get("verdict"), indent=2, ensure_ascii=False, allow_nan=False), "```", ""]
    return "\n".join(lines) + "\n"


def _long_span_check(report, error=None):
    name = "long_span_T1_T4"
    if error is not None:
        return {"name": name, "status": error[0], "samples": 0, "reason": error[1]}
    if not isinstance(report, dict):
        return {"name": name, "status": "insufficient_data", "samples": 0, "reason": "long-span report missing"}
    verdict = report.get("verdict") or {}
    if not isinstance(verdict, dict):
        return {"name": name, "status": "failed", "samples": 0, "reason": "malformed long-span verdict"}
    stages = ("T1", "T2", "T3", "T3b", "T4")
    explicit_failure = verdict.get("passed") is False
    incomplete = []
    for stage in stages:
        item = report.get(stage)
        if not isinstance(item, dict):
            incomplete.append(stage)
            continue
        assessed = item.get("judgement", item)
        if not isinstance(assessed, dict) or type(assessed.get("passed")) is not bool:
            incomplete.append(stage)
        elif not assessed["passed"]:
            explicit_failure = True
    if explicit_failure:
        status, reason = "failed", "existing T1–T4 report records a failure; original criteria unchanged"
    elif incomplete or verdict.get("passed") is not True:
        status, reason = "insufficient_data", f"missing T1–T4 stage or verdict evidence: {', '.join(incomplete)}"
    else:
        status, reason = "passed", "all existing T1–T4 stage results present and passed under their recorded criteria"
    return {"name": name, "status": status, "samples": 1 if status == "passed" else 0, "reason": reason}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(long_span.DEFAULT_RAW_DIR))
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--siril", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=list(long_span.DEFAULT_FRAMES))
    parser.add_argument("--sky-limit-y", type=float, required=True)
    parser.add_argument("--sky-mask", required=True)
    parser.add_argument("--compare-job", default=None, help="an unmasked job on the same frames, measured with the same colour metric for the record")
    parser.add_argument("--min-registered-fraction", type=float, default=1.0)
    parser.add_argument("--label", default="accept-horizon")
    parser.add_argument("--support-mode", choices=("legacy", "strict_siril_1_4_4"), default="legacy")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "h5-validation-20260919"))
    parser.add_argument("--run", default=None, help="evaluate an existing accept-horizon run folder instead of running the job")
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    if args.run:
        source_run = Path(args.run).resolve()
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = output_root / f"horizon-evaluation-{stamp}-{uuid.uuid4().hex[:8]}"
        run.mkdir(parents=True, exist_ok=False)
    else:
        started = _dt.datetime.now().timestamp()
        long_argv = ["--raw-dir", args.raw_dir, "--every", str(args.every), "--start", str(args.start), "--frames", *map(str, args.frames),
                     "--sky-limit-y", str(args.sky_limit_y), "--sky-mask", args.sky_mask, "--min-registered-fraction", str(args.min_registered_fraction),
                     "--label", args.label, "--output-root", str(output_root)]
        if args.limit is not None:
            long_argv += ["--limit", str(args.limit)]
        long_argv += ["--support-mode", args.support_mode]
        if args.siril:
            long_argv += ["--siril", args.siril]
        long_span.main(long_argv)          # T1–T4; its exit code is folded into the verdict below
        run = _find_run(output_root, args.label, started)
        source_run = run
    long_report, long_error = None, None
    try:
        long_report = _load_json((source_run / "acceptance.json").read_text(encoding="utf-8"))
        if not isinstance(long_report, dict):
            raise ValueError("long-span report must be an object")
    except (FileNotFoundError, PermissionError) as exc:
        long_error = ("insufficient_data", f"{type(exc).__name__}: {exc}")
    except (ValueError, OSError) as exc:
        long_error = ("failed", f"{type(exc).__name__}: {exc}")
    job_info = long_report.get("job") if isinstance(long_report, dict) else None
    job_path = job_info.get("dir") if isinstance(job_info, dict) else None
    if isinstance(job_path, str) and job_path:
        job_dir = Path(job_path)
        if not job_dir.is_absolute():
            job_dir = source_run / job_dir
        report = horizon_checks(job_dir.resolve(), Path(args.sky_mask).resolve(), compare_job=Path(args.compare_job).resolve() if args.compare_job else None, run=run)
    else:
        report = _empty_report("unknown", "job directory is missing from the existing long-span report")
    report["run"] = run.name
    report["source_run"] = str(source_run)
    report["long_span_verdict"] = long_report.get("verdict") if isinstance(long_report, dict) else None
    report["verdict"] = _verdict(report, [_long_span_check(long_report, long_error)])
    (run / "horizon.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (run / "HORIZON.md").write_text(render(report, long_report), encoding="utf-8")
    status = report["verdict"]["status"]
    print(f"{status}: {run / 'HORIZON.md'}")
    for issue in report["verdict"]["failures"] + report["verdict"]["incomplete"]:
        print(f"  - {issue}")
    return _EXIT_CODES[status]


if __name__ == "__main__":
    sys.exit(main())
