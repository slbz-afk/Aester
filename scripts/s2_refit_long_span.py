"""S2 next package: candidate homographies for a long-span frame, evaluated on
the same reference correspondences (Siril as-is / direct refit / chained refit).

    <output-root>/refit-<label>-<UTC stamp>-<6hex>/
        refit_long_span.json   candidates (Siril and diagnostics coordinates), evaluation, caveats
        REFIT.md

Example (Windows, Z8 smoke job, frame 16 with frame 2 as the intermediate):

    .venv\\Scripts\\python.exe scripts\\s2_refit_long_span.py ^
        --seq "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work\\light_.seq" ^
        --lst-dir "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work\\cache" ^
        --frame 16 --via 2 --width 8280 --height 5520 --output-root outputs\\fable-s2-20260917

Read-only on inputs.  The candidates are evidence for the next decision, not
matrices the product stacks with.
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

from starstack import registration_adapter as adapter  # noqa: E402
from starstack import registration_refit as refit  # noqa: E402
from starstack import siril_sequence  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def render_markdown(result: dict) -> str:
    ev = result["evaluation"]
    lines = [f"# 长跨度帧单应重拟合：{result['frame']} → {result['reference']}"
             + (f"（中间帧 {result['intermediate']}）" if result.get("intermediate") else ""), "",
             f"生成：{result['environment']['timestamp']} · 序列 sha256 {result['sequence_source']['sha256'][:12]}… · "
             f"对应 {ev['pairs']} 对（半径 {ev['association_radius_px']} px，歧义 {ev['ambiguous']}），全部经 Siril 原 H 关联",
             f"坐标假设：{result['hypothesis']['name']}（{'已实证' if result['hypothesis']['verified'] else '未实证'}）", "",
             "## 同一批参考对应上的残差（px；区域按参考帧诊断坐标）", "",
             "| 候选 | 视图 | n | 总体 RMS | 中心 | 四角（n） | 四角平均 dx,dy | 其他 | max |", "|---|---|---|---|---|---|---|---|---|"]
    for name, cand in ev["candidates"].items():
        if not cand.get("available"):
            lines.append(f"| {name} | – | – | 不可用：{cand.get('reason')} | | | | | |")
            continue
        for view, entry in cand["views"].items():
            o, c, k, t = entry["overall"], entry["center"], entry["corner"], entry["other"]
            lines.append(f"| {name} | {view} | {entry['count']} | {_fmt(o.get('rms_px'))} | {_fmt(c.get('rms_px'))} | "
                         f"{_fmt(k.get('rms_px'))}（{k['count']}） | {_fmt(k.get('mean_dx_px'), 2)},{_fmt(k.get('mean_dy_px'), 2)} | "
                         f"{_fmt(t.get('rms_px'))} | {_fmt(o.get('max_px'), 2)} |")
    lines += ["", "## 候选矩阵（Siril 原生坐标，frame→reference）", ""]
    for name, cand in result["candidates"].items():
        lines.append(f"- **{name}**：{json.dumps(cand['detail'], ensure_ascii=False)}")
        if cand.get("H") is not None:
            h = np.asarray(cand["H"])
            lines.append("  `" + " ".join(f"{v:.6g}" for v in h.reshape(-1)) + "`")
    lines += ["", "## 注意事项", ""] + [f"- {c}" for c in result["caveats"]]
    lines += ["", f"环境：{result['environment']['platform']} · Python {result['environment']['python']} · NumPy {result['environment']['numpy']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--lst-dir", required=True)
    parser.add_argument("--frame", type=int, required=True, help="Siril image index of the long-span frame")
    parser.add_argument("--via", type=int, default=None, help="Siril image index of the intermediate frame (chained refit)")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    hypothesis = adapter.HYPOTHESES[args.hypothesis or adapter.DEFAULT_HYPOTHESIS]
    sequence = siril_sequence.read_seq(args.seq)
    if sequence["all_identity_per_layer"].get(args.layer):
        raise SystemExit("all homographies are identity: pass the original light_.seq")
    by_index = {img["index"]: img["position"] for img in sequence["images"]}
    for idx in (args.frame, args.via):
        if idx is not None and idx not in by_index:
            raise SystemExit(f"image index {idx} not in sequence")
    ref_pos = sequence["header"]["reference_image"]
    frame_pos = by_index[args.frame]
    via_pos = by_index[args.via] if args.via is not None else None
    if frame_pos == ref_pos or via_pos == ref_pos or via_pos == frame_pos:
        raise SystemExit("frame, intermediate and reference must be three different images")
    lst_dir = Path(args.lst_dir)
    lists = {}
    for pos in [ref_pos, frame_pos] + ([via_pos] if via_pos is not None else []):
        stem = sequence["images"][pos]["stem"]
        path = lst_dir / f"{stem}.lst"
        if not path.exists():
            raise SystemExit(f"missing star list {path}")
        lists[stem] = siril_sequence.read_lst(path)
    result = refit.refit_frame(sequence, lists, frame_pos, intermediate_position=via_pos, width=args.width,
                               height=args.height, layer=args.layer, radius=args.radius)
    # diagnostics-convention copies of the candidate matrices (for warping / the core)
    for cand in result["candidates"].values():
        if cand.get("H") is not None:
            cand["H_diagnostics"] = adapter.convert_matrix(hypothesis, np.asarray(cand["H"]), args.height, args.height).tolist()
    result["hypothesis"] = hypothesis.as_dict()
    result["sequence_source"] = sequence["source"]
    result["star_list_sources"] = {stem: l["source"] for stem, l in lists.items()}
    result["width"], result["height"] = args.width, args.height
    result["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                             "timestamp": _now(), "argv": sys.argv[1:] if argv is None else list(argv)}
    label = args.label or f"f{args.frame}" + (f"-via{args.via}" if args.via is not None else "")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"refit-{label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "refit_long_span.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "REFIT.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"ok: {run / 'REFIT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
