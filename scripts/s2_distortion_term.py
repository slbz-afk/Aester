"""S2 next package, step 1: bound the span-dependent residual left by two-pass registration.

Input is a diagnostics document (``diagnostics_input.json`` written by
``scripts/s2_real_sequence_diagnostics.py`` or by the acceptance script's
``diagnostics-new/``): matched star pairs per frame in reference pixel
coordinates.  Fits, on the pre-declared stratified fit subset:

* per-frame homography (baseline, same subset, no shared term)
* homography + one sensor-fixed radial distortion shared by all frames
* homography + one span-scaled polynomial displacement field (degree 1, 2)

and reports validation residuals per frame and region, validation grid
means for the extreme frames, and transfer tests (fit one side of the
reference / short spans, predict the other side / long spans).  Read-only.

    <output-root>/distortion-term-<label>-<UTC stamp>-<6hex>/
        distortion_term.json
        DISTORTION.md

Example (Windows, product venv):

    .venv\\Scripts\\python.exe scripts\\s2_distortion_term.py ^
        --document outputs\\fable-s2-20260917\\accept-twopass-<stamp>\\diagnostics-new\\diagnostics_input.json ^
        --output-root outputs\\fable-s2-20260917
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

from starstack import registration_distortion as rd  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def render(result: dict, run: Path, source: str) -> str:
    models = result["models"]
    order = list(models)
    frames = sorted(result["spans"], key=lambda fid: result["spans"][fid])
    lines = [f"# 跨度相关残差的有界共享项（{run.name}）", "",
             f"生成：{result['environment']['timestamp']} · 文档 `{source}` · {result['frame_count']} 帧 · 画幅 {result['width']}×{result['height']} · "
             f"分层 {result['split']['columns']}×{result['split']['rows']}（拟合/验证交替，先声明后拟合）", "",
             "## 每帧验证集四角 RMS（px）", "",
             "| 帧 | 跨度 | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for fid in frames:
        cells = [_fmt(models[m]["frames"][fid]["residuals"]["validation"]["corner"]["rms_px"]) for m in order]
        lines.append(f"| {fid} | {result['spans'][fid]:+d} | " + " | ".join(cells) + " |")
    lines += ["", "## 每帧验证集总体 RMS（px）", "", "| 帧 | 跨度 | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for fid in frames:
        cells = [_fmt(models[m]["frames"][fid]["residuals"]["validation"]["overall"]["rms_px"]) for m in order]
        lines.append(f"| {fid} | {result['spans'][fid]:+d} | " + " | ".join(cells) + " |")
    lines += ["", "## 共享参数", ""]
    rad = models["homography_plus_shared_radial"]
    p = rad["parameters"]
    lines.append(f"- 径向畸变（传感器固定）：k1 {p['k1']:.5g}，k2 {p['k2']:.5g}，中心 ({p['cx']:.1f}, {p['cy']:.1f})，尺度 {p['scale']:.0f} px；"
                 f"最远角的位移 {rad['corner_shift_px']:.2f} px（共享自由度 {rad['shared_dof']}）")
    for name, m in models.items():
        if name.startswith("homography_plus_shared_field"):
            probes = m["field_per_unit_span_px"]
            lines.append(f"- {name}：每单位跨度的位移场 (dx, dy) px —— " + "，".join(f"{k} ({v[0]:+.3f}, {v[1]:+.3f})" for k, v in probes.items())
                         + f"（共享自由度 {m['shared_dof']}）")
    lines += ["", "## 验证集网格有符号均值（最长跨度两帧；基线 vs 共享项）", ""]
    extremes = [frames[0], frames[-1]]
    for fid in extremes:
        lines += [f"### {fid}（跨度 {result['spans'][fid]:+d}）", "", "| 单元 | n | 基线 dx,dy | 径向 dx,dy | 场2 dx,dy |", "|---|---|---|---|---|"]
        base = models["homography_per_frame"]["frames"][fid]["grid_validation"]
        radg = models["homography_plus_shared_radial"]["frames"][fid]["grid_validation"]
        fld = models.get("homography_plus_shared_field_deg2", models.get("homography_plus_shared_field_deg1"))["frames"][fid]["grid_validation"]
        for cell in base:
            b, r, f = base[cell], radg[cell], fld[cell]
            lines.append(f"| {cell} | {b['count']} | {_fmt(b['mean_dx_px'], 2)},{_fmt(b['mean_dy_px'], 2)} | "
                         f"{_fmt(r['mean_dx_px'], 2)},{_fmt(r['mean_dy_px'], 2)} | {_fmt(f['mean_dx_px'], 2)},{_fmt(f['mean_dy_px'], 2)} |")
        lines.append("")
    lines += ["## 迁移测试（共享项只在部分帧上拟合，其余帧只重拟合各自单应；验证集四角 RMS）", "",
              "| 测试 | 留出帧 | 基线 | 径向 | 场2 |", "|---|---|---|---|---|"]
    for name, t in result["transfer"].items():
        if "skipped" in t:
            lines.append(f"| {name} | – | 跳过：{t['skipped']} | | |")
            continue
        for fid in t["radial"]["held_out_frames"]:
            lines.append(f"| {name} | {fid} ({result['spans'][fid]:+d}) | {_fmt(t['baseline_corner_rms_px'][fid])} | "
                         f"{_fmt(t['radial']['held_out_validation_corner_rms_px'][fid])} | {_fmt(t['field_deg2']['held_out_validation_corner_rms_px'][fid])} |")
    lines += ["", "## 注意事项", ""] + [f"- {c}" for c in result["caveats"]]
    lines += ["", f"环境：{result['environment']['platform']} · Python {result['environment']['python']} · NumPy {result['environment']['numpy']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document", required=True, help="diagnostics_input.json (adapter document with matches)")
    parser.add_argument("--exclude-frames", nargs="*", default=[], help="frame ids to leave out (e.g. light_00001)")
    parser.add_argument("--degrees", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--label", default="run")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    document = json.loads(Path(args.document).read_text(encoding="utf-8"))
    width, height = int(document["reference"]["width"]), int(document["reference"]["height"])
    frames = rd.frames_from_document(document, exclude=tuple(args.exclude_frames))
    if len(frames) < 3:
        raise SystemExit("need at least 3 frames with matches")
    result = rd.analyse(frames, width, height, degrees=tuple(args.degrees))
    result["source"] = {"document": str(Path(args.document).resolve()), "reference": document["reference"]["frame_id"],
                        "coordinate_convention": document.get("coordinate_convention"), "excluded": list(args.exclude_frames)}
    result["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                             "timestamp": _now(), "argv": sys.argv[1:] if argv is None else list(argv)}
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"distortion-term-{args.label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "distortion_term.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "DISTORTION.md").write_text(render(result, run, args.document), encoding="utf-8")
    print(f"ok: {run / 'DISTORTION.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
