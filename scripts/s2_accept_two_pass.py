"""S2 acceptance of the two-pass registration in the product flow (one command).

Runs the *product* job (``starstack.core``: stage -> ``01_convert`` -> ``02_stack``
with ``register light -2pass -transf=homography`` / ``seqapplyreg light`` /
``stack r_light ...``; since efficiency P1b-3 ``02_export`` runs the registration
commands and ``02_stack`` ``seqclean`` + the stack, and both layouts are accepted
here) on the same 16 DNGs as the reference one-pass job, then
judges it with the pre-registered criteria below, all on the new job's own
files.  Nothing in the reference job, the DNGs or older outputs is modified;
inputs are hard-linked (or copied) into the run directory.

    <output-root>/accept-twopass-<UTC stamp>-<6hex>/
        inputs/                     hard links to the 16 DNGs (same bytes, verified by sha256)
        jobs/job-<id>/              the new product job (stage/, work/, products/, logs)
        job_summary.json            state, diagnostics, timings, r_ frame and master headers
        diagnostics-new/            adapter diagnostics on the new light_.seq (all frames)
        diagnostics-baseline/       the same on the reference job's one-pass light_.seq
        crops/<frame>_<box>.json    pixel-level crop checks (frame warped onto the new reference)
        crops/montage_<frame>_<box>.png
        master/<box>.json           old vs new sky_master star shapes on the same stars
        master/<box>_{old,new}.png, master/montage_<box>.png
        acceptance.json / ACCEPTANCE.md   criteria, measurements, verdict

Pre-registered criteria (T1..T4), fixed before the run:
  T1 job      succeeded; stacked == inputs; master float32 with the reference frame's
              geometry; diagnostics.reference_selection == siril_2pass_auto and the
              reference stem agrees with light_.seq and with the Siril log line.
  T2 adapter  every non-reference frame: corner-pooled RMS <= 0.5 px, >= 1000 matched
              pairs; at most one frame may instead be "tolerated" (RMS <= 1.0 px,
              >= 300 pairs) - the documented frame-1 detection anomaly (RUN_LOG §14).
  T3 pixels   for every checked frame and box (top-right, centre, top-left, right-lower):
              star centroid RMS <= 0.5 px after warping with Siril's H; the same single
              tolerated frame may reach 1.0 px.
  T4 master   on the same stars in each box: new median sigma_major <= 1.02 x old, and
              in the top-right box (where the one-pass smear was measured) new < old.
Verdict: pass only if T1..T4 all pass; "tolerated" exceptions are listed, never hidden.

Example (Windows, product venv, ~10 min including Siril):

    .venv\\Scripts\\python.exe scripts\\s2_accept_two_pass.py --output-root outputs\\fable-s2-20260917

Re-evaluate an existing new job without re-running Siril:

    .venv\\Scripts\\python.exe scripts\\s2_accept_two_pass.py --job-dir outputs\\...\\jobs\\job-<id>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from starstack import fits_crop  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack import siril_sequence  # noqa: E402
from starstack import core  # noqa: E402
from starstack import stack_settings  # noqa: E402

ACCEPTANCE_VERSION = "1.0.0"
DEFAULT_REFERENCE_JOB = ROOT / "outputs" / "z8-meteor-20260916" / "smoke-jobs" / "job-9dcccd46202d4c0b"
# Diagnostics coordinates (TOP-DOWN, zero-based pixel centres) on the new reference grid.
BOXES = {"topright": (6800, 8200, 0, 1000), "centre": (3600, 4700, 2300, 3300),
         "topleft": (0, 1400, 0, 1000), "rightlower": (6800, 8200, 3400, 4400)}
CRITERIA = {"T1": {"stacked_equals_inputs": True, "reference_selection": core.REFERENCE_SELECTION},
            "T2": {"corner_rms_px": 0.5, "min_pairs": 1000, "tolerated_frames": 1, "tolerated_rms_px": 1.0, "tolerated_min_pairs": 300},
            "T3": {"rms_px": 0.5, "tolerated_rms_px": 1.0, "tolerated_frames": 1},
            "T4": {"max_ratio_new_over_old": 1.02, "must_improve_box": "topright"}}
STRETCH = {"low_percentile": 5.0, "high_percentile": 99.7, "softness": 20.0}
MASTER_SCALE = 65535.0  # float masters are 0..1; shape thresholds are in 16-bit DN like the light frames

_REFERENCE_CHOICE = [re.compile(r"choosing image (\d+) as new reference", re.I),
                     re.compile(r"选择图像\s*(\d+)\s*作为新参考")]
_EXEC_TIME = [re.compile(r"Total execution time:\s*([0-9.]+)\s*(ms|s|min)", re.I),
              re.compile(r"总执行时间[:：]\s*([0-9.]+)\s*(ms|s|min)")]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def parse_reference_choice(log_text: str) -> int | None:
    """Image number Siril's two-pass log says it chose as the new reference (last trial wins)."""
    chosen = None
    for line in log_text.splitlines():
        for pattern in _REFERENCE_CHOICE:
            match = pattern.search(line)
            if match:
                chosen = int(match.group(1))
    return chosen


def stage_log(job_dir: Path, name: str) -> str:
    path = job_dir / f"{name}.log"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def registration_log(job_dir: Path) -> str:
    """The log of the Siril process that ran the stack-phase registration: 02_export (efficiency P1b-3 split
    layout) when present, else 02_stack (the pre-P1b single process)."""
    return stage_log(job_dir, "02_export") if (job_dir / "02_export.log").exists() else stage_log(job_dir, "02_stack")


def stack_scripts_match(job_dir: Path, combined: list[str]) -> bool | None:
    """Whether the job's stack-phase Siril scripts are the product's, in either L layout.

    ``combined`` is the pre-P1b single-process script (export commands + core.STACK_COMMAND).  Split layout
    (efficiency P1b-3, 02_export.ssf present): 02_export runs ``combined[:-1]`` and 02_stack runs
    core.LEGACY_STACK_SCRIPT, with the EFF-D4 settings lines before seqclean in a job admitted under the
    stack-settings contract.  Pre-P1b layout: 02_stack runs ``combined``.  None without a 02_stack.ssf.
    """
    def body(name, count):
        lines = (job_dir / f"{name}.ssf").read_text(encoding="utf-8").split("\n")[1:]
        return stack_settings.without_settings(lines)[0][:count]
    if not (job_dir / "02_stack.ssf").exists():
        return None
    combined = list(combined)
    if (job_dir / "02_export.ssf").exists():
        stack = list(core.LEGACY_STACK_SCRIPT)
        return (combined[-1] == core.STACK_COMMAND and body("02_export", len(combined) - 1) == combined[:-1]
                and body("02_stack", len(stack)) == stack)
    return body("02_stack", len(combined)) == combined


def parse_execution_seconds(log_text: str) -> float | None:
    for pattern in _EXEC_TIME:
        found = pattern.findall(log_text)
        if found:
            value, unit = found[-1]
            return float(value) * {"ms": 1e-3, "s": 1.0, "min": 60.0}[unit.lower()]
    return None


def plan_check_frames(sequence: dict, requested: list[int] | None) -> list[int]:
    """Image numbers to check at pixel level: both sequence ends, the farthest frame from
    the reference, plus any requested ones; never the reference itself."""
    images = sequence["images"]
    ref = sequence["header"]["reference_image"]
    farthest = max(range(len(images)), key=lambda p: abs(p - ref))
    positions = [0, len(images) - 1, farthest]
    indices = [images[p]["index"] for p in positions] + list(requested or [])
    ref_index = images[ref]["index"]
    return [i for i in dict.fromkeys(indices) if i != ref_index and any(im["index"] == i for im in images)]


def link_inputs(reference_job: dict, target: Path) -> dict:
    """Hard-link (or copy) the reference job's inputs into ``target`` in the same order."""
    target.mkdir(parents=True, exist_ok=False)
    method = None
    for i, source in enumerate(reference_job["inputs"], 1):
        src = Path(source["path"])
        if not src.is_file():
            raise SystemExit(f"reference input missing: {src}")
        dst = target / f"{i:03d}_{src.name}"
        try:
            os.link(src, dst)
            method = method or "hardlink"
        except OSError:
            shutil.copyfile(src, dst)
            method = "copy"
    return {"dir": str(target), "count": len(reference_job["inputs"]), "method": method}


def run_product_job(inputs_dir: Path, jobs_root: Path, siril_path: str, *, key: str, log=print, sky_mask_path: str | None = None,
                    support_mode: str = "legacy") -> dict:
    """Submit and run the product job in this process (same code path as the worker)."""
    original_spawn = core._spawn_worker
    core._spawn_worker = lambda directory: None  # run synchronously below; the worker only calls run_job
    try:
        job = core.submit_job(str(inputs_dir), str(jobs_root), siril_path, sky_mask_path=sky_mask_path, idempotency_key=key,
                              support_mode=support_mode)
    finally:
        core._spawn_worker = original_spawn
    job_dir = Path(job["job_dir"])
    timeline = [{"time": _now(), "state": job["state"], "stage": job["stage"], "progress": job["progress"], "message": job["message"]}]
    started = time.monotonic()
    worker = threading.Thread(target=core.run_job, args=(str(job_dir),), daemon=True)
    worker.start()
    last = (job["state"], job["stage"], job["progress"])
    while worker.is_alive():
        time.sleep(2.0)
        try:
            current = core.load_job(job_dir)
        except (OSError, ValueError, KeyError):
            continue
        key_now = (current["state"], current["stage"], current["progress"])
        if key_now != last:
            last = key_now
            timeline.append({"time": _now(), "state": current["state"], "stage": current["stage"],
                             "progress": current["progress"], "message": current["message"]})
            log(f"  [{time.monotonic() - started:7.1f}s] {current['state']}/{current['stage']} {current['progress']}% {current['message']}")
    worker.join()
    final = core.load_job(job_dir)
    return {"job": final, "timeline": timeline, "wall_seconds": time.monotonic() - started}


def fits_summary(path: Path) -> dict:
    header, _ = fits_crop.read_header(path)
    keys = ("BITPIX", "NAXIS1", "NAXIS2", "NAXIS3", "ROWORDER", "STACKCNT")
    return {"file": str(path), "size_bytes": path.stat().st_size, **{k: header.get(k) for k in keys if k in header}}


def job_checks(job: dict, job_dir: Path, reference_job: dict, sequence: dict) -> dict:
    work = job_dir / "work"
    stack_log = stage_log(job_dir, "02_stack")
    export_log = stage_log(job_dir, "02_export")   # efficiency P1b-3: L's export process (absent before P1b)
    convert_log = (job_dir / "01_convert.log").read_text(encoding="utf-8", errors="replace") if (job_dir / "01_convert.log").exists() else ""
    diagnostics = job.get("diagnostics") or {}
    ref_position = sequence["header"]["reference_image"]
    ref_stem = sequence["images"][ref_position]["stem"]
    ref_index = sequence["images"][ref_position]["index"]
    log_choice = parse_reference_choice(registration_log(job_dir))
    same_inputs = [a["sha256"] == b["sha256"] for a, b in zip(job["inputs"], reference_job["inputs"])]
    r_frames = sorted(work.glob("r_light_*.fit"))
    master = work / "sky_master.fit"
    checks = {
        "state_succeeded": job["state"] == "succeeded",
        "same_inputs_as_reference_job": len(job["inputs"]) == len(reference_job["inputs"]) and all(same_inputs),
        "stacked_equals_inputs": diagnostics.get("stacked_count") == diagnostics.get("input_count") == len(job["inputs"]),
        "reference_selection_is_two_pass": diagnostics.get("reference_selection") == core.REFERENCE_SELECTION,
        "reference_stem_matches_sequence": diagnostics.get("reference_stem") == ref_stem,
        "reference_matches_siril_log": log_choice == ref_index if log_choice is not None else None,
        "reference_input_is_that_frame": diagnostics.get("reference_input") == job["inputs"][ref_index - 1]["path"],
        "registration_commands": diagnostics.get("registration") == list(core.REGISTRATION_COMMANDS),
        "stack_script_matches_product": stack_scripts_match(job_dir, core.STACK_SCRIPT),
        "r_frames_written": len(r_frames) == len(job["inputs"]),
    }
    master_info = fits_summary(master) if master.exists() else None
    frame_info = fits_summary(sorted(work.glob("light_*.fit"))[0]) if list(work.glob("light_*.fit")) else None
    r_info = fits_summary(r_frames[0]) if r_frames else None
    if master_info and frame_info:
        checks["master_float32_reference_geometry"] = (master_info["BITPIX"] == "-32" and master_info["NAXIS1"] == frame_info["NAXIS1"]
                                                       and master_info["NAXIS2"] == frame_info["NAXIS2"])
    passed = all(v for v in checks.values() if v is not None)
    in_range = 1 <= ref_index <= len(job["inputs"])
    return {"passed": passed, "checks": checks, "diagnostics": diagnostics, "reference": {"position": ref_position, "stem": ref_stem,
            "index": ref_index, "log_choice": log_choice, "input": job["inputs"][ref_index - 1]["path"] if in_range else None,
            "original_input": reference_job["inputs"][ref_index - 1]["path"] if in_range and ref_index <= len(reference_job["inputs"]) else None},
            "master": master_info, "light_frame": frame_info, "r_frame": r_info, "r_frame_count": len(r_frames),
            "siril_seconds": {"convert": parse_execution_seconds(convert_log), "stack": parse_execution_seconds(stack_log),
                              **({"export": parse_execution_seconds(export_log)} if export_log else {})},
            "sequence_header": sequence["header"], "limitations": job.get("limitations")}


def adapter_diagnostics(seq_path: Path, lst_dir: Path, width: int, height: int, hypothesis, run: Path, label: str,
                        *, radius: float = 6.0) -> dict:
    import s2_real_sequence_diagnostics as diag  # noqa: E402  (scripts/ on sys.path)
    sequence = siril_sequence.read_seq(seq_path)
    ref = sequence["header"]["reference_image"]
    positions = [p for p in range(len(sequence["images"])) if p != ref]
    needed = [sequence["images"][ref]["stem"]] + [sequence["images"][p]["stem"] for p in positions]
    lst_paths = {}
    for stem in needed:
        path = lst_dir / f"{stem}.lst"
        if not path.exists():
            raise SystemExit(f"missing star list {path}")
        lst_paths[stem] = path
    result = adapter.run_sequence_diagnostics(seq_path=seq_path, lst_paths=lst_paths, width=width, height=height,
                                              hypothesis=hypothesis, positions=positions, layer=1, radius=radius)
    result["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                             "timestamp": _now(), "argv": ["(s2_accept_two_pass)", label]}
    result["frames_requested"] = {"indices": None, "positions": positions, "reference_frame_index": None, "reference_position": ref}
    out = run / f"diagnostics-{label}"
    out.mkdir()
    _write_json(out / "diagnostics_input.json", result["document"])
    _write_json(out / "diagnostics_report.json", result["report"])
    _write_json(out / "association_summary.json", {k: v for k, v in result.items() if k not in ("document", "report")})
    (out / "SUMMARY.md").write_text(diag.render_summary(result, out), encoding="utf-8")
    frames = {}
    for frame in result["report"]["frames"]:
        stem = frame["frame_id"]
        s = result["summaries"].get(stem, {})
        pooled = s.get("pooled", {}).get("all_roles_pooled", {}) if s.get("pooled", {}).get("available") else {}
        frames[stem] = {"state": frame["state"], "matched": s.get("association", {}).get("matched_count"),
                        "frame_stars": s.get("association", {}).get("frame_count"),
                        "center_rms_px": pooled.get("center", {}).get("rms_px"),
                        "corners_rms_px": pooled.get("corners_pooled", {}).get("rms_px"),
                        "corners_max_px": pooled.get("corners_pooled", {}).get("max_px"),
                        "overall_rms_px": pooled.get("overall", {}).get("rms_px")}
    return {"dir": str(out), "reference": sequence["images"][ref]["stem"], "frames": frames}


def judge_adapter(frames: dict, criteria: dict) -> dict:
    failing, tolerated = [], []
    for stem, f in frames.items():
        rms, matched = f.get("corners_rms_px"), f.get("matched") or 0
        if f.get("state") != "succeeded" or rms is None:
            failing.append(stem)
        elif rms <= criteria["corner_rms_px"] and matched >= criteria["min_pairs"]:
            continue
        elif rms <= criteria["tolerated_rms_px"] and matched >= criteria["tolerated_min_pairs"]:
            tolerated.append(stem)
        else:
            failing.append(stem)
    passed = not failing and len(tolerated) <= criteria["tolerated_frames"]
    return {"passed": passed, "failing": failing, "tolerated": tolerated}


def _reference_seed_stars(lst_path: Path, hypothesis, height: int) -> list[tuple[float, float]]:
    stars = siril_sequence.read_lst(lst_path)["stars"]
    return [adapter.convert_point(hypothesis, s["x"], s["y"], height) for s in stars]


def _in_box(xy, box, margin=8):
    x0, x1, y0, y1 = box
    return [(x - x0, y - y0, k) for k, (x, y) in enumerate(xy) if x0 + margin <= x < x1 - margin and y0 + margin <= y < y1 - margin]


def _montage(panels_by_star: list[list[np.ndarray]], factor: int = 6) -> np.ndarray | None:
    rows = []
    for panels in panels_by_star:
        lo, hi = np.percentile(panels[0][np.isfinite(panels[0])], [5, 99.9]) if np.isfinite(panels[0]).any() else (0.0, 1.0)
        hi = max(hi, lo + 1.0)
        cells = []
        for panel in panels:
            x = np.clip((np.nan_to_num(panel, nan=lo) - lo) / (hi - lo), 0, None)
            y = np.arcsinh(20.0 * x) / np.arcsinh(20.0)
            cell = np.clip(np.round(255 * y), 0, 255).astype(np.uint8)
            cell = np.repeat(np.repeat(cell, factor, axis=0), factor, axis=1)
            cell[:, -1] = 255
            cells.append(cell)
        row = np.concatenate(cells, axis=1)
        row[-1, :] = 255
        rows.append(row)
    return np.concatenate(rows, axis=0) if rows else None


def pixel_crop_checks(work: Path, sequence: dict, hypothesis, frames: list[int], boxes: dict, plane: int, run: Path,
                      *, zoom_stars: int = 6, window: int = 24, log=print) -> dict:
    out = run / "crops"
    out.mkdir()
    ref_pos = sequence["header"]["reference_image"]
    ref_stem = sequence["images"][ref_pos]["stem"]
    ref_path = work / f"{ref_stem}.fit"
    header, _ = fits_crop.read_header(ref_path)
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    seeds = _reference_seed_stars(work / "cache" / f"{ref_stem}.lst", hypothesis, height)
    regdata = sequence["regdata"][1]
    by_index = {im["index"]: im for im in sequence["images"]}
    results = {}
    half = window // 2
    for index in frames:
        image = by_index[index]
        h_td = adapter.convert_matrix(hypothesis, regdata[image["position"]]["H"], height, height)
        frame_path = work / f"{image['stem']}.fit"
        for name, box in boxes.items():
            x0, x1, y0, y1 = box
            local = _in_box(seeds, box)
            xy = [(x, y) for x, y, _ in local]
            reference = fits_crop.crop_top_down(ref_path, plane, x0, x1, y0, y1)
            warped = fits_crop.warp_to_reference(frame_path, plane, h_td, box)
            offsets = fits_crop.measure_star_offsets(reference, warped["crop"], xy, radius=4)
            stack = fits_crop.stack_two(reference, warped["crop"])
            entry = {"frame": image["stem"], "box": list(box), "stars_in_box": len(xy), "coverage_fraction": warped["coverage_fraction"],
                     "source_window": warped["source_window"], "offsets": offsets["summary"],
                     "shape_reference": {k: v for k, v in fits_crop.measure_star_shapes(reference, xy).items() if k != "stars"},
                     "shape_warped": {k: v for k, v in fits_crop.measure_star_shapes(warped["crop"], xy).items() if k != "stars"},
                     "shape_stack2": {k: v for k, v in fits_crop.measure_star_shapes(stack, xy).items() if k != "stars"},
                     "H_diagnostics": h_td.tolist()}
            measured = offsets["offsets"]
            if zoom_stars > 0 and measured:
                def peak(o):
                    cx, cy = int(round(o["reference_xy"][0])), int(round(o["reference_xy"][1]))
                    return float(reference[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].max())
                panels = []
                for o in sorted(measured, key=lambda o: -peak(o))[:zoom_stars]:
                    cx, cy = int(round(o["reference_xy"][0])), int(round(o["reference_xy"][1]))
                    if half <= cx < reference.shape[1] - half and half <= cy < reference.shape[0] - half:
                        sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
                        panels.append([reference[sl], warped["crop"][sl], stack[sl]])
                montage = _montage(panels)
                if montage is not None:
                    fits_crop.write_png_gray(out / f"montage_{image['stem']}_{name}.png", montage)
                    entry["montage"] = {"file": f"montage_{image['stem']}_{name}.png", "columns": ["reference", "frame warped", "stack2"],
                                        "rows": len(panels), "window_px": window}
            _write_json(out / f"{image['stem']}_{name}.json", {**entry, "offsets_detail": offsets})
            results[f"{image['stem']}|{name}"] = entry
            s = offsets["summary"]
            log(f"  crop {image['stem']} {name:10s} n={s['count']:3d} rms={_fmt(s['rms_px'])} mean=({_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)})")
    return {"dir": str(out), "reference": ref_stem, "plane": plane, "frames": frames, "results": results}


def judge_crops(results: dict, criteria: dict) -> dict:
    per_frame = {}
    for key, entry in results.items():
        stem = key.split("|")[0]
        rms = entry["offsets"].get("rms_px")
        per_frame.setdefault(stem, []).append(rms)
    failing, tolerated = [], []
    for stem, values in per_frame.items():
        if any(v is None for v in values):
            failing.append(stem)
        elif max(values) <= criteria["rms_px"]:
            continue
        elif max(values) <= criteria["tolerated_rms_px"]:
            tolerated.append(stem)
        else:
            failing.append(stem)
    return {"passed": not failing and len(tolerated) <= criteria["tolerated_frames"], "failing": failing, "tolerated": tolerated}


def paired_shapes(old_shapes: dict, new_shapes: dict) -> dict:
    """Per-star comparison of two ``measure_star_shapes`` results keyed by seed index."""
    old = {s["index"]: s for s in old_shapes.get("stars", [])}
    new = {s["index"]: s for s in new_shapes.get("stars", [])}
    common = sorted(set(old) & set(new))
    if not common:
        return {"paired": 0}
    o_maj = np.array([old[k]["sigma_major_px"] for k in common])
    n_maj = np.array([new[k]["sigma_major_px"] for k in common])
    o_el = np.array([old[k]["elongation"] for k in common])
    n_el = np.array([new[k]["elongation"] for k in common])
    ratio = n_maj / o_maj
    o_min = np.array([old[k]["sigma_minor_px"] for k in common])
    n_min = np.array([new[k]["sigma_minor_px"] for k in common])
    o_flux = np.array([old[k]["flux"] for k in common])
    n_flux = np.array([new[k]["flux"] for k in common])
    return {"paired": len(common), "old_median_sigma_major_px": float(np.median(o_maj)), "new_median_sigma_major_px": float(np.median(n_maj)),
            "median_ratio_new_over_old": float(np.median(ratio)), "ratio_p25": float(np.percentile(ratio, 25)), "ratio_p75": float(np.percentile(ratio, 75)),
            "fraction_new_sharper": float(np.mean(n_maj < o_maj)),
            "old_median_elongation": float(np.median(o_el)), "new_median_elongation": float(np.median(n_el)),
            "old_median_fwhm_major_px": float(2.3548 * np.median(o_maj)), "new_median_fwhm_major_px": float(2.3548 * np.median(n_maj)),
            # RUN_LOG §16: the one-pass smear went into the minor axis and into rejected flux, so report both axes, area and flux
            "old_median_sigma_minor_px": float(np.median(o_min)), "new_median_sigma_minor_px": float(np.median(n_min)),
            "median_ratio_sigma_minor": float(np.median(n_min / o_min)),
            "median_ratio_area": float(np.median((n_maj * n_min) / (o_maj * o_min))),
            "median_ratio_flux": float(np.median(n_flux / np.where(o_flux > 0, o_flux, np.nan)))}


def master_comparison(old_job_dir: Path, new_work: Path, old_sequence: dict, new_sequence: dict, hypothesis, boxes: dict, plane: int,
                      run: Path, *, zoom_stars: int = 6, window: int = 24, log=print) -> dict:
    """Same stars, two masters: seeds from the new reference list, mapped into the old master through the old one-pass H."""
    out = run / "master"
    out.mkdir()
    old_master, new_master = old_job_dir / "work" / "sky_master.fit", new_work / "sky_master.fit"
    header, _ = fits_crop.read_header(new_master)
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    new_ref = new_sequence["images"][new_sequence["header"]["reference_image"]]
    seeds = _reference_seed_stars(new_work / "cache" / f"{new_ref['stem']}.lst", hypothesis, height)
    # the new reference frame as seen by the old job: its one-pass H maps that frame onto the old reference grid
    old_position = next(im["position"] for im in old_sequence["images"] if im["index"] == new_ref["index"])
    h_new_ref_to_old = adapter.convert_matrix(hypothesis, old_sequence["regdata"][1][old_position]["H"], height, height)
    pts = np.array([[x, y, 1.0] for x, y in seeds]).T
    proj = h_new_ref_to_old @ pts
    seeds_old = list(zip(proj[0] / proj[2], proj[1] / proj[2]))
    results = {}
    half = window // 2
    for name, box in boxes.items():
        x0, x1, y0, y1 = box
        local = _in_box(seeds, box)
        if not local:
            results[name] = {"box": list(box), "stars_in_box": 0, "paired": 0}
            continue
        keys = [k for _, _, k in local]
        ox = np.array([seeds_old[k][0] for k in keys])
        oy = np.array([seeds_old[k][1] for k in keys])
        ob = (max(0, int(np.floor(ox.min())) - 16), min(width, int(np.ceil(ox.max())) + 17),
              max(0, int(np.floor(oy.min())) - 16), min(height, int(np.ceil(oy.max())) + 17))
        new_crop = fits_crop.crop_top_down(new_master, plane, x0, x1, y0, y1) * MASTER_SCALE
        old_crop = fits_crop.crop_top_down(old_master, plane, *ob) * MASTER_SCALE
        xy_new = [(x, y) for x, y, _ in local]
        xy_old = [(seeds_old[k][0] - ob[0], seeds_old[k][1] - ob[2]) for k in keys]
        shape_new = fits_crop.measure_star_shapes(new_crop, xy_new)
        shape_old = fits_crop.measure_star_shapes(old_crop, xy_old)
        paired = paired_shapes(shape_old, shape_new)
        entry = {"box": list(box), "old_box": list(ob), "stars_in_box": len(local), "paired_comparison": paired,
                 "old_summary": {k: v for k, v in shape_old.items() if k != "stars"},
                 "new_summary": {k: v for k, v in shape_new.items() if k != "stars"},
                 "old_master": str(old_master), "new_master": str(new_master), "old_reference": old_sequence["images"][old_sequence["header"]["reference_image"]]["stem"],
                 "new_reference": new_ref["stem"], "h_new_reference_to_old_grid": h_new_ref_to_old.tolist()}
        fits_crop.write_png_gray(out / f"{name}_new.png", fits_crop.asinh_stretch(new_crop, **STRETCH))
        fits_crop.write_png_gray(out / f"{name}_old.png", fits_crop.asinh_stretch(old_crop, **STRETCH))
        new_by = {s["index"]: s for s in shape_new.get("stars", [])}
        old_by = {s["index"]: s for s in shape_old.get("stars", [])}
        panels = []
        for k in sorted(set(new_by) & set(old_by), key=lambda k: -new_by[k]["flux"])[:zoom_stars]:
            nx, ny = int(round(new_by[k]["xy"][0])), int(round(new_by[k]["xy"][1]))
            oxk, oyk = int(round(old_by[k]["xy"][0])), int(round(old_by[k]["xy"][1]))
            if (half <= nx < new_crop.shape[1] - half and half <= ny < new_crop.shape[0] - half
                    and half <= oxk < old_crop.shape[1] - half and half <= oyk < old_crop.shape[0] - half):
                panels.append([old_crop[oyk - half:oyk + half, oxk - half:oxk + half], new_crop[ny - half:ny + half, nx - half:nx + half]])
        montage = _montage(panels)
        if montage is not None:
            fits_crop.write_png_gray(out / f"montage_{name}.png", montage)
            entry["montage"] = {"file": f"montage_{name}.png", "columns": ["old master (one-pass)", "new master (two-pass)"], "rows": len(panels), "window_px": window}
        _write_json(out / f"{name}.json", {**entry, "old_stars": shape_old.get("stars", []), "new_stars": shape_new.get("stars", [])})
        results[name] = entry
        log(f"  master {name:10s} paired={paired.get('paired', 0):3d} sigma_major old={_fmt(paired.get('old_median_sigma_major_px'))} "
            f"new={_fmt(paired.get('new_median_sigma_major_px'))} ratio={_fmt(paired.get('median_ratio_new_over_old'))}")
    return {"dir": str(out), "plane": plane, "results": results}


def judge_master(results: dict, criteria: dict) -> dict:
    failing = []
    improved_box = None
    for name, entry in results.items():
        paired = entry.get("paired_comparison", {})
        if paired.get("paired", 0) < 5:
            failing.append(f"{name}: fewer than 5 paired stars")
            continue
        ratio = paired["new_median_sigma_major_px"] / paired["old_median_sigma_major_px"]
        if ratio > criteria["max_ratio_new_over_old"]:
            failing.append(f"{name}: new/old sigma_major {ratio:.3f} > {criteria['max_ratio_new_over_old']}")
        if name == criteria["must_improve_box"]:
            improved_box = ratio < 1.0
            if not improved_box:
                failing.append(f"{name}: new master not sharper than old ({ratio:.3f})")
    return {"passed": not failing and improved_box is True, "failing": failing, "must_improve_box_improved": improved_box}


def render(report: dict) -> str:
    v = report["verdict"]
    t1, t2, t3, t4 = report["T1"], report["T2"], report["T3"], report["T4"]
    lines = [f"# S2 两段式配准产品验收（{report['run']}）", "",
             f"生成：{report['environment']['timestamp']} · 参考作业 `{report['reference_job']['dir']}` · 新作业 `{report['job']['dir']}`",
             f"结论：**{'通过' if v['passed'] else '未通过'}**（T1 {t1['passed']} · T2 {t2['judgement']['passed']} · T3 {t3['judgement']['passed']} · T4 {t4['judgement']['passed']}）", ""]
    if v["tolerated"]:
        lines += [f"容许例外（列出，不隐藏）：{v['tolerated']}", ""]
    if v["failures"]:
        lines += ["未通过项："] + [f"- {f}" for f in v["failures"]] + [""]
    lines += ["## T1 作业", "", "| 检查 | 结果 |", "|---|---|"]
    for key, value in t1["checks"].items():
        lines.append(f"| {key} | {'✔' if value else ('–' if value is None else '✘')} |")
    ref = t1["reference"]
    lines += ["", f"参考帧：{ref['stem']}（位置 {ref['position']}，图像 {ref['index']}，日志选择 {ref['log_choice']}）→ `{ref['input']}`（原片 `{ref.get('original_input')}`）",
              f"母版：{t1['master']} · r_ 帧：{t1['r_frame']}（{t1['r_frame_count']} 帧）· Siril 用时 convert {_fmt(t1['siril_seconds']['convert'], 1)} s，stack {_fmt(t1['siril_seconds']['stack'], 1)} s · 作业总用时 {report['job']['wall_seconds']:.0f} s", ""]
    lines += ["## T2 适配器诊断（新作业全帧，参考 " + t2["new"]["reference"] + "）", "",
              "| 帧 | 匹配 | 中心 RMS | 四角 RMS | 四角 max | 总体 RMS | 基线（一段式，参考 " + t2["baseline"]["reference"] + "）四角 RMS |", "|---|---|---|---|---|---|---|"]
    for stem, f in t2["new"]["frames"].items():
        b = t2["baseline"]["frames"].get(stem, {})
        lines.append(f"| {stem} | {f.get('matched')}/{f.get('frame_stars')} | {_fmt(f.get('center_rms_px'))} | {_fmt(f.get('corners_rms_px'))} | "
                     f"{_fmt(f.get('corners_max_px'), 2)} | {_fmt(f.get('overall_rms_px'))} | {_fmt(b.get('corners_rms_px'))} |")
    lines += ["", f"判定：{t2['judgement']}", ""]
    lines += ["## T3 像素级裁块（帧经 Siril H 重采样到参考 " + t3["reference"] + "，绿通道）", "",
              "| 帧 | 框 | 星数 | RMS | 中位 | max | 平均 dx,dy | 参考单帧 σ_major | 两帧叠加 σ_major |", "|---|---|---|---|---|---|---|---|---|"]
    for key, e in t3["results"].items():
        s = e["offsets"]
        lines.append(f"| {e['frame']} | {key.split('|')[1]} | {s['count']} | {_fmt(s['rms_px'])} | {_fmt(s['median_px'])} | {_fmt(s['max_px'], 2)} | "
                     f"{_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)} | {_fmt(e['shape_reference'].get('median_sigma_major_px'))} | "
                     f"{_fmt(e['shape_stack2'].get('median_sigma_major_px'))} |")
    lines += ["", f"判定：{t3['judgement']}", ""]
    lines += ["## T4 新旧母版同一批星的星形（旧 = 一段式参考 light_00001；新 = 两段式参考 " + t2["new"]["reference"] + "）", "",
              "| 框 | 配对星 | 旧中位 σ_major | 新中位 σ_major | 中位比 新/旧 (p25–p75) | 新更锐比例 | 旧/新中位拉长比 |", "|---|---|---|---|---|---|---|"]
    for name, e in t4["results"].items():
        p = e.get("paired_comparison", {})
        if not p.get("paired"):
            lines.append(f"| {name} | 0 | – | – | – | – | – |")
            continue
        lines.append(f"| {name} | {p['paired']} | {_fmt(p['old_median_sigma_major_px'])} | {_fmt(p['new_median_sigma_major_px'])} | "
                     f"{_fmt(p['median_ratio_new_over_old'])} ({_fmt(p['ratio_p25'], 2)}–{_fmt(p['ratio_p75'], 2)}) | {_fmt(p['fraction_new_sharper'], 2)} | "
                     f"{_fmt(p['old_median_elongation'], 2)}/{_fmt(p['new_median_elongation'], 2)} |")
    lines += ["", f"判定：{t4['judgement']}", "",
              "## 判据（运行前固定）", "", "```json", json.dumps(CRITERIA, ensure_ascii=False, indent=1), "```", "",
              "## 注意事项", "",
              "- 单帧星形（参考单帧 σ_major）是光学；只有叠加后的增量才是配准。角部单帧拉长比 1.4–1.9 与配准无关。",
              "- 母版对比的种子星来自新参考帧星表，旧母版位置经旧作业的一段式 H 映射；测量本身只用像素。两母版归一化相同（rej w 3 3, addscale, 32b）。",
              "- T2/T3 的关联/质心都以 Siril 自己的 H 为起点，反映该 H 的一致性；这是验收产品流程，不是独立真值。",
              "- 本脚本不改动参考作业、原片与旧输出；新作业与输入硬链接都在本运行目录内。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-job", default=str(DEFAULT_REFERENCE_JOB), help="one-pass product job to compare against")
    parser.add_argument("--job-dir", default=None, help="an already-run two-pass product job to evaluate (skips Siril)")
    parser.add_argument("--siril", default=None, help="siril-cli executable (default: the reference job's engine)")
    parser.add_argument("--frames", type=int, nargs="*", default=[2, 16], help="extra image numbers for pixel crops")
    parser.add_argument("--plane", type=int, default=1)
    parser.add_argument("--hypothesis", default=None, help=f"one of {sorted(adapter.HYPOTHESES)} (default: adapter default, proved)")
    parser.add_argument("--label", default="accept-twopass")
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
        inputs = link_inputs(reference_job, run / "inputs")
        print(f"inputs: {inputs['count']} files ({inputs['method']}) -> {inputs['dir']}")
        siril = args.siril or reference_job["engine"]["executable"]
        print("running the product job (Siril two-pass) ...")
        job_run = run_product_job(run / "inputs", run / "jobs", siril, key=run.name, log=print)
        job_run["inputs"] = inputs
        job_run["reused"] = False
        job = job_run["job"]
        job_dir = Path(job["job_dir"])
    _write_json(run / "job_summary.json", {"job": job, "timeline": job_run["timeline"], "wall_seconds": job_run["wall_seconds"],
                                            "inputs": job_run["inputs"], "reused": job_run["reused"]})
    if job["state"] != "succeeded":
        report = {"acceptance_version": ACCEPTANCE_VERSION, "run": run.name, "job": {"dir": str(job_dir), "state": job["state"], "error": job.get("error")},
                  "verdict": {"passed": False, "failures": [f"job {job['state']}: {job.get('error')}"], "tolerated": []},
                  "environment": {"timestamp": _now()}}
        _write_json(run / "acceptance.json", report)
        (run / "ACCEPTANCE.md").write_text(f"# 验收未通过：作业 {job['state']}\n\n{job.get('error')}\n", encoding="utf-8")
        print(f"FAIL: job {job['state']}: {job.get('error')}")
        return 2

    work = job_dir / "work"
    new_seq = siril_sequence.read_seq(work / core.SEQUENCE_FILE)
    old_seq = siril_sequence.read_seq(reference_dir / "work" / core.SEQUENCE_FILE)
    header, _ = fits_crop.read_header(work / "sky_master.fit")
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])

    print("T1 job checks ...")
    t1 = job_checks(job, job_dir, reference_job, new_seq)
    print("T2 adapter diagnostics (new, all frames) ...")
    t2_new = adapter_diagnostics(work / core.SEQUENCE_FILE, work / "cache", width, height, hypothesis, run, "new")
    print("T2 adapter diagnostics (baseline one-pass job) ...")
    t2_base = adapter_diagnostics(reference_dir / "work" / core.SEQUENCE_FILE, reference_dir / "work" / "cache", width, height, hypothesis, run, "baseline")
    t2 = {"new": t2_new, "baseline": t2_base, "judgement": judge_adapter(t2_new["frames"], CRITERIA["T2"])}
    frames = plan_check_frames(new_seq, args.frames)
    print(f"T3 pixel crops for frames {frames} ...")
    t3 = pixel_crop_checks(work, new_seq, hypothesis, frames, BOXES, args.plane, run, log=print)
    t3["judgement"] = judge_crops(t3["results"], CRITERIA["T3"])
    print("T4 master comparison ...")
    t4 = master_comparison(reference_dir, work, old_seq, new_seq, hypothesis, BOXES, args.plane, run, log=print)
    t4["judgement"] = judge_master(t4["results"], CRITERIA["T4"])

    failures = []
    if not t1["passed"]:
        failures += [f"T1 {k}" for k, v in t1["checks"].items() if v is False]
    if not t2["judgement"]["passed"]:
        failures.append(f"T2 frames failing {t2['judgement']['failing']} tolerated {t2['judgement']['tolerated']}")
    if not t3["judgement"]["passed"]:
        failures.append(f"T3 frames failing {t3['judgement']['failing']} tolerated {t3['judgement']['tolerated']}")
    if not t4["judgement"]["passed"]:
        failures.append(f"T4 {t4['judgement']['failing']}")
    tolerated = {"T2": t2["judgement"]["tolerated"], "T3": t3["judgement"]["tolerated"]}
    report = {"acceptance_version": ACCEPTANCE_VERSION, "run": run.name, "criteria": CRITERIA, "boxes": BOXES,
              "hypothesis": hypothesis.as_dict(), "reference_job": {"dir": str(reference_dir), "inputs_sha256": [i["sha256"] for i in reference_job["inputs"]]},
              "job": {"dir": str(job_dir), "wall_seconds": job_run["wall_seconds"], "reused": job_run["reused"], "inputs": job_run["inputs"]},
              "T1": t1, "T2": t2, "T3": t3, "T4": t4,
              "verdict": {"passed": not failures, "failures": failures, "tolerated": {k: v for k, v in tolerated.items() if v}},
              "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                              "timestamp": _now(), "elapsed_seconds": time.monotonic() - started,
                              "argv": sys.argv[1:] if argv is None else list(argv)}}
    _write_json(run / "acceptance.json", report)
    (run / "ACCEPTANCE.md").write_text(render(report), encoding="utf-8")
    print(f"{'PASS' if not failures else 'FAIL'}: {run / 'ACCEPTANCE.md'}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
