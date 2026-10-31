"""S2 next package: pixel-level check of candidate homographies on a fixed crop.

Takes a refit run (``refit_long_span.json``), reads the reference frame and
the long-span frame directly from the sequence's FITS files (only the rows
the crop needs), warps the frame into reference coordinates under every
candidate matrix, measures star centroid offsets on the raw pixels and
writes fixed-stretch PNG crops plus two-frame stacks:

    <output-root>/crop-check-<label>-<UTC stamp>-<6hex>/
        crop_check.json                 offsets per candidate, box, stretch, sources
        CROP_CHECK.md
        reference.png                   the reference crop
        <candidate>_warped.png          frame resampled into reference coordinates
        <candidate>_stack2.png          mean of reference and warped frame (misregistration = elongation)
        detail_<candidate>_stack2.png   3x nearest-neighbour zoom of a sub-box of the stack

Example (Windows):

    .venv\\Scripts\\python.exe scripts\\s2_pixel_crop_check.py ^
        --refit outputs\\fable-s2-20260917\\refit-f16-via2-<stamp>\\refit_long_span.json ^
        --box 6800 8200 0 1000 --detail 7500 7900 250 650 --output-root outputs\\fable-s2-20260917

Coordinates of ``--box``/``--detail`` are diagnostics coordinates (TOP-DOWN
display rows, zero-based pixel centres).  Read-only on the FITS files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from starstack import fits_crop  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack import siril_sequence  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def zoom_nn(image8: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image8, factor, axis=0), factor, axis=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit", required=True, help="refit_long_span.json from scripts/s2_refit_long_span.py")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), required=True)
    parser.add_argument("--detail", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), default=None,
                        help="sub-box of --box for the 3x zoomed stacks")
    parser.add_argument("--plane", type=int, default=1, help="FITS plane (0 R, 1 G, 2 B); Siril registered on 1")
    parser.add_argument("--centroid-radius", type=int, default=4)
    parser.add_argument("--zoom-stars", type=int, default=6, help="brightest measured stars shown in the zoom montage (0 = none)")
    parser.add_argument("--zoom-window", type=int, default=24, help="window side in px around each star in the montage")
    parser.add_argument("--zoom-factor", type=int, default=6)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    refit = json.loads(Path(args.refit).read_text(encoding="utf-8"))
    seq_path = Path(refit["sequence_source"]["path"])
    work = seq_path.parent
    sequence = siril_sequence.read_seq(seq_path)
    if sequence["source"]["sha256"] != refit["sequence_source"]["sha256"]:
        raise SystemExit("the sequence file changed since the refit run")
    ext = None
    for candidate_ext in (".fit", ".fits", ".fts"):
        if (work / f"{refit['reference']}{candidate_ext}").exists():
            ext = candidate_ext
            break
    if ext is None:
        raise SystemExit(f"no FITS file for {refit['reference']} next to {seq_path}")
    ref_path, frame_path = work / f"{refit['reference']}{ext}", work / f"{refit['frame']}{ext}"
    hypothesis = adapter.CoordinateHypothesis(**refit["hypothesis"])
    width, height = refit["width"], refit["height"]
    x0, x1, y0, y1 = args.box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise SystemExit("--box outside the image")

    reference = fits_crop.crop_top_down(ref_path, args.plane, x0, x1, y0, y1)
    # reference stars inside the box (Siril list -> diagnostics coordinates -> crop coordinates)
    lst_path = Path(refit["star_list_sources"][refit["reference"]]["path"])
    ref_list = siril_sequence.read_lst(lst_path)
    stars_xy = []
    for s in ref_list["stars"]:
        tx, ty = adapter.convert_point(hypothesis, s["x"], s["y"], height)
        if x0 + 8 <= tx < x1 - 8 and y0 + 8 <= ty < y1 - 8:
            stars_xy.append((tx - x0, ty - y0))
    label = args.label or f"{refit['frame']}-box{x0}-{y0}"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"crop-check-{label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    stretch = {"low_percentile": 5.0, "high_percentile": 99.7, "softness": 20.0}
    fits_crop.write_png_gray(run / "reference.png", fits_crop.asinh_stretch(reference, **stretch))
    results = {}
    reference_shape = fits_crop.measure_star_shapes(reference, stars_xy)
    detail = args.detail
    if detail is not None:
        dx0, dx1, dy0, dy1 = detail
        if not (x0 <= dx0 < dx1 <= x1 and y0 <= dy0 < dy1 <= y1):
            raise SystemExit("--detail must lie inside --box")
        fits_crop.write_png_gray(run / "detail_reference.png",
                                 zoom_nn(fits_crop.asinh_stretch(reference[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0], **stretch), 3))
    for name, cand in refit["candidates"].items():
        if cand.get("H_diagnostics") is None:
            results[name] = {"available": False, "reason": cand["detail"].get("reason")}
            continue
        warped = fits_crop.warp_to_reference(frame_path, args.plane, np.asarray(cand["H_diagnostics"]), (x0, x1, y0, y1))
        offsets = fits_crop.measure_star_offsets(reference, warped["crop"], stars_xy, radius=args.centroid_radius)
        stack = fits_crop.stack_two(reference, warped["crop"])
        shape_stack = fits_crop.measure_star_shapes(stack, stars_xy)
        shape_warped = fits_crop.measure_star_shapes(warped["crop"], stars_xy)
        fits_crop.write_png_gray(run / f"{name}_warped.png", fits_crop.asinh_stretch(warped["crop"], **stretch))
        fits_crop.write_png_gray(run / f"{name}_stack2.png", fits_crop.asinh_stretch(stack, **stretch))
        if detail is not None:
            fits_crop.write_png_gray(run / f"detail_{name}_stack2.png",
                                     zoom_nn(fits_crop.asinh_stretch(stack[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0], **stretch), 3))
        results[name] = {"available": True, "independent_validation": cand.get("independent_validation", False),
                         "source_window": warped["source_window"], "coverage_fraction": warped["coverage_fraction"],
                         "offsets": offsets,
                         "shape_stack2": {k: v for k, v in shape_stack.items() if k != "stars"},
                         "shape_warped_frame": {k: v for k, v in shape_warped.items() if k != "stars"}}
    # zoom montage: the brightest measured stars, one row each; columns = reference, then each candidate's two-frame stack
    montage_info = None
    if args.zoom_stars > 0 and any(r.get("available") for r in results.values()):
        stacks = {}
        for name, cand in refit["candidates"].items():
            if results.get(name, {}).get("available"):
                warped = fits_crop.warp_to_reference(frame_path, args.plane, np.asarray(cand["H_diagnostics"]), (x0, x1, y0, y1))
                stacks[name] = fits_crop.stack_two(reference, warped["crop"])
        first = next(r for r in results.values() if r.get("available"))
        measured = first["offsets"]["offsets"]
        half = args.zoom_window // 2

        def peak(o):
            cx, cy = int(round(o["reference_xy"][0])), int(round(o["reference_xy"][1]))
            return float(reference[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].max())

        chosen = sorted(measured, key=lambda o: -peak(o))[:args.zoom_stars]
        rows = []
        for o in chosen:
            cx, cy = int(round(o["reference_xy"][0])), int(round(o["reference_xy"][1]))
            if not (half <= cx < reference.shape[1] - half and half <= cy < reference.shape[0] - half):
                continue
            sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
            panels = [reference[sl]] + [stacks[name][sl] for name in stacks]
            lo, hi = np.percentile(reference[sl], [5, 99.9])
            hi = max(hi, lo + 1.0)
            cells = []
            for panel in panels:
                x = np.clip((np.nan_to_num(panel, nan=lo) - lo) / (hi - lo), 0, None)
                y = np.arcsinh(20.0 * x) / np.arcsinh(20.0)
                cell = np.clip(np.round(255 * y), 0, 255).astype(np.uint8)
                cell = zoom_nn(cell, args.zoom_factor)
                cell[:, -1] = 255
                cells.append(cell)
            row = np.concatenate(cells, axis=1)
            row[-1, :] = 255
            rows.append(row)
        if rows:
            fits_crop.write_png_gray(run / "zoom_montage.png", np.concatenate(rows, axis=0))
            montage_info = {"file": "zoom_montage.png", "columns": ["reference"] + [f"stack2:{name}" for name in stacks],
                            "rows": len(rows), "window_px": args.zoom_window, "zoom_factor": args.zoom_factor,
                            "stretch": "per-row asinh on the reference window (5–99.9 percentile), same for every column"}
    out = {"crop_check_version": fits_crop.FITS_CROP_VERSION, "refit_source": str(Path(args.refit).resolve()),
           "zoom_montage": montage_info,
           "reference_file": str(ref_path), "frame_file": str(frame_path), "plane": args.plane, "box": [x0, x1, y0, y1],
           "detail_box": detail, "stars_measured_from": str(lst_path), "stars_in_box": len(stars_xy),
           "stretch": stretch, "coordinate_convention": "TOP-DOWN, zero-based pixel centres (diagnostics); box in the same",
           "hypothesis": refit["hypothesis"], "candidates": results,
           "reference_shape": {k: v for k, v in reference_shape.items() if k != "stars"},
           "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                           "timestamp": _now(), "argv": sys.argv[1:] if argv is None else list(argv)},
           "caveats": ["centroids are measured on the raw green plane with a fixed window; saturated or blended stars are not excluded",
                       "the star positions used as seeds come from Siril's reference list; the measurement itself uses pixels only",
                       "PNG stretch is fixed and documented; it is for looking, not for judging photometry"]}
    (run / "crop_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# 像素级裁块检查：{refit['frame']} → {refit['reference']}，框 x{x0}–{x1} y{y0}–{y1}（绿通道）", "",
             f"生成：{out['environment']['timestamp']} · 参考文件 `{ref_path.name}` · 帧文件 `{frame_path.name}` · 框内参考星 {len(stars_xy)} 颗（种子来自参考星表，测量只用像素）", "",
             "| 候选 | 独立验证 | 测得星数 | 跳过 | RMS | 中位 | max | 平均 dx,dy | 覆盖 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if not r.get("available"):
            lines.append(f"| {name} | – | 不可用：{r.get('reason')} | | | | | | |")
            continue
        s = r["offsets"]["summary"]
        lines.append(f"| {name} | {'是' if r['independent_validation'] else '否'} | {s['count']} | {s['skipped']} | {_fmt(s['rms_px'])} | "
                     f"{_fmt(s['median_px'])} | {_fmt(s['max_px'], 2)} | {_fmt(s['mean_dx_px'], 2)},{_fmt(s['mean_dy_px'], 2)} | {r['coverage_fraction']:.3f} |")
    rs = reference_shape
    lines += ["", "## 星点形状（二阶矩，同一批星；单帧形状 = 光学，叠加后增量 = 配准拖散）", "",
              "| 图像 | n | 中位 σ_major | 中位 σ_minor | 中位拉长比 | 中位 FWHM_major |", "|---|---|---|---|---|---|",
              f"| 参考帧单帧 | {rs.get('count', 0)} | {_fmt(rs.get('median_sigma_major_px'))} | {_fmt(rs.get('median_sigma_minor_px'))} | "
              f"{_fmt(rs.get('median_elongation'))} | {_fmt(rs.get('median_fwhm_major_px'), 2)} |"]
    for name, r in results.items():
        if not r.get("available"):
            continue
        for tag, key in (("重采样帧单帧", "shape_warped_frame"), ("两帧叠加", "shape_stack2")):
            sh = r[key]
            lines.append(f"| {name}：{tag} | {sh.get('count', 0)} | {_fmt(sh.get('median_sigma_major_px'))} | {_fmt(sh.get('median_sigma_minor_px'))} | "
                         f"{_fmt(sh.get('median_elongation'))} | {_fmt(sh.get('median_fwhm_major_px'), 2)} |")
    lines += ["", "图：`reference.png`、`<候选>_warped.png`、`<候选>_stack2.png`（参考与重采样帧的两帧平均，配准误差表现为星点拉长/双像）"
              + ("、`detail_*.png`（3× 放大子框）" if detail is not None else "")
              + (f"、`zoom_montage.png`（最亮 {montage_info['rows']} 颗星，每行一颗，列 = {' | '.join(montage_info['columns'])}，{montage_info['zoom_factor']}× 放大）" if montage_info else ""), "",
              "## 注意事项", ""] + [f"- {c}" for c in out["caveats"]]
    (run / "CROP_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ok: {run / 'CROP_CHECK.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
