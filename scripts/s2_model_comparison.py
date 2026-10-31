"""S2 bounded model comparison on real frame-to-frame star correspondences.

Builds correspondences between two cached Siril star lists through the
sequence homographies (chained ``inv(H_dst) @ H_src``), splits them by a
pre-declared spatial rule, fits similarity / affine / homography /
homography+radial / polynomial models on the fit half and reports the
validation half by region.  Siril's own chained matrix is evaluated as-is.

    <output-root>/model-comparison-<label>-<UTC stamp>-<6hex>/
        model_comparison.json
        MODEL_COMPARISON.md

Example (Windows, Z8 smoke job, frame 16 mapped onto frame 2):

    .venv\\Scripts\\python.exe scripts\\s2_model_comparison.py ^
        --seq "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work\\light_.seq" ^
        --lst-dir "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work\\cache" ^
        --source-frame 16 --destination-frame 2 --width 8280 --height 5520 --output-root outputs\\fable-s2-20260917

Coordinates stay in Siril's native frame (no hypothesis needed: the comparison
is invariant to a flip or a constant offset, only the region labels are
native).  Read-only on inputs; a new directory per run.
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
from starstack import registration_models as models  # noqa: E402
from starstack import siril_sequence  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def build_correspondences(seq_path, lst_dir, source_frame: int, destination_frame: int, *, layer: int = 1,
                          radius: float = 6.0) -> dict:
    sequence = siril_sequence.read_seq(seq_path)
    if sequence["all_identity_per_layer"].get(layer):
        raise SystemExit("all homographies are identity: pass the original light_.seq, not r_light_.seq")
    by_index = {img["index"]: img for img in sequence["images"]}
    for index in (source_frame, destination_frame):
        if index not in by_index:
            raise SystemExit(f"image index {index} not in sequence")
    if source_frame == destination_frame:
        raise SystemExit("source and destination must differ")
    src_img, dst_img = by_index[source_frame], by_index[destination_frame]
    regdata = sequence["regdata"][layer]
    h_src = np.asarray(regdata[src_img["position"]]["H"], dtype=np.float64)
    h_dst = np.asarray(regdata[dst_img["position"]]["H"], dtype=np.float64)
    chained = np.linalg.inv(h_dst) @ h_src          # source frame -> destination frame (documented direction)
    lists = {}
    for img in (src_img, dst_img):
        path = Path(lst_dir) / f"{img['stem']}.lst"
        if not path.exists():
            raise SystemExit(f"missing star list {path}")
        lists[img["stem"]] = siril_sequence.read_lst(path)
    assoc = adapter.associate_stars(lists[dst_img["stem"]]["stars"], lists[src_img["stem"]]["stars"], chained, radius=radius)
    src_stars, dst_stars = lists[src_img["stem"]]["stars"], lists[dst_img["stem"]]["stars"]
    src = np.array([[src_stars[p["frame_index"]]["x"], src_stars[p["frame_index"]]["y"]] for p in assoc["pairs"]])
    dst = np.array([[dst_stars[p["reference_index"]]["x"], dst_stars[p["reference_index"]]["y"]] for p in assoc["pairs"]])
    return {"sequence_source": sequence["source"], "sequence_header": sequence["header"],
            "source_stem": src_img["stem"], "destination_stem": dst_img["stem"],
            "star_list_sources": {stem: lst["source"] for stem, lst in lists.items()},
            "siril_matrix_source_to_destination": chained.tolist(),
            "association": {k: v for k, v in assoc.items() if k not in ("pairs", "unmatched_frame", "unmatched_reference", "ambiguous_frame")}
            | {"matched_count": len(assoc["pairs"]), "ambiguous_count": len(assoc["ambiguous_frame"]),
               "unmatched_source_count": len(assoc["unmatched_frame"]), "unmatched_destination_count": len(assoc["unmatched_reference"])},
            "src": src, "dst": dst}


def render_markdown(result: dict) -> str:
    c = result["comparison"]
    lines = [f"# 有界模型比较：{result['source_stem']} → {result['destination_stem']}", "",
             f"生成：{result['environment']['timestamp']} · 序列 `{Path(result['sequence_source']['path']).name}`（sha256 {result['sequence_source']['sha256'][:12]}…）",
             f"对应关系：经 Siril 链式矩阵、半径 {result['association']['radius_px']} px 互最近邻，匹配 {result['association']['matched_count']} / 源 {result['association']['frame_count']} 颗，歧义 {result['association']['ambiguous_count']}",
             f"分层拆分：{c['split']['columns']}×{c['split']['rows']} 网格内按 (x,y) 排序交替 → 拟合 {c['fit_count']}，验证 {c['validation_count']}（拆分先于任何拟合，不按残差挑选）",
             "", "## 验证子集残差（px，区域按 Siril 原生坐标）", "",
             "| 模型 | 自由度 | 独立验证 | 总体 RMS | 中心 | 四角合并（n） | 四角平均 dx,dy | 其他 | max |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in c["models"].items():
        v = m["residuals"]["validation"]
        lines.append(f"| {name} | {m['dof']} | {'是' if m['independent_validation'] else '否（Siril 自行拟合）'} | {_fmt(v['overall']['rms_px'])} | "
                     f"{_fmt(v['center']['rms_px'])} | {_fmt(v['corner']['rms_px'])}（{v['corner']['count']}） | "
                     f"{_fmt(v['corner']['mean_dx_px'], 2)},{_fmt(v['corner']['mean_dy_px'], 2)} | {_fmt(v['other']['rms_px'])} | {_fmt(v['overall']['max_px'], 2)} |")
    lines += ["", "## 拟合子集残差（同一模型，供对照过拟合）", "", "| 模型 | 总体 RMS | 中心 | 四角合并 | 其他 |", "|---|---|---|---|---|"]
    for name, m in c["models"].items():
        f = m["residuals"]["fit"]
        lines.append(f"| {name} | {_fmt(f['overall']['rms_px'])} | {_fmt(f['center']['rms_px'])} | {_fmt(f['corner']['rms_px'])} | {_fmt(f['other']['rms_px'])} |")
    rad = c["models"].get("homography_radial", {}).get("parameters", {})
    if rad:
        lines += ["", f"homography_radial：k1 = {rad['k1']:.4g}，k2 = {rad['k2']:.4g}（r 以 {rad['scale']:.0f} px 归一，残差在去畸变坐标中；"
                  "两帧共用同一畸变，只受帧间差分运动约束，不是镜头标定）"]
    lines += ["", "## 注意事项", ""] + [f"- {x}" for x in c["caveats"]]
    lines += ["", f"环境：{result['environment']['platform']} · Python {result['environment']['python']} · NumPy {result['environment']['numpy']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--lst-dir", required=True)
    parser.add_argument("--source-frame", type=int, required=True, help="Siril image index mapped onto the destination")
    parser.add_argument("--destination-frame", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    data = build_correspondences(args.seq, args.lst_dir, args.source_frame, args.destination_frame, layer=args.layer, radius=args.radius)
    comparison = models.compare_models(data.pop("src"), data.pop("dst"), args.width, args.height,
                                       siril_matrix=np.asarray(data["siril_matrix_source_to_destination"]),
                                       columns=args.columns, rows=args.rows)
    result = dict(data)
    result["comparison"] = comparison
    result["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                             "timestamp": _now(), "argv": sys.argv[1:] if argv is None else list(argv)}
    label = args.label or f"f{args.source_frame}-to-f{args.destination_frame}"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"model-comparison-{label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "model_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "MODEL_COMPARISON.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"ok: {run / 'MODEL_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
