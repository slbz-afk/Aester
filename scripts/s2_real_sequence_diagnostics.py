"""S2 real-sequence registration diagnostics (read-only on Siril products).

Feeds an *original* Siril sequence file and its cached star lists through the
registration adapter and the diagnostics core, and writes one run directory:

    <output-root>/real-diagnostics-<label>-<UTC stamp>-<6hex>/
        diagnostics_input.json     document handed to registration_diagnostics
        diagnostics_report.json    core report (unchanged contract)
        association_summary.json   sources, hypothesis, association + pooled/grid summaries, caveats
        SUMMARY.md                 one-page human summary

Example (Windows, product venv, Z8 smoke job, frames 2 and 16 against the reference 1):

    .venv\\Scripts\\python.exe scripts\\s2_real_sequence_diagnostics.py ^
        --seq "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work\\light_.seq" ^
        --lst-dir "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work\\cache" ^
        --frames 2 16 --width 8280 --height 5520 --output-root outputs\\fable-s2-20260917

The default hypothesis is ``registration_adapter.DEFAULT_HYPOTHESIS`` (the
convention proved on the real Siril 1.4.4 on 2026-09-17); ``--proof
<coordinate_proof.json>`` from a real (non-simulated, proved) run overrides
it, ``--hypothesis`` selects a named one (including the ``*_unverified``
comparison entries).  Every output carries the hypothesis and its verified
flag.  Nothing under ``--seq``/``--lst-dir`` is modified; a registered
``r_*.seq`` is rejected.
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
from starstack import siril_sequence  # noqa: E402
from starstack.siril_coordinate_proof import hypothesis_from_proof  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_run_dir(root: Path, label: str) -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = root / f"real-diagnostics-{label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def resolve_hypothesis(name: str | None, proof_path: str | None) -> adapter.CoordinateHypothesis:
    if proof_path:
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        hypothesis = hypothesis_from_proof(proof)
        if not hypothesis.verified:
            raise SystemExit("the given proof is SIMULATED or inconclusive and cannot verify a hypothesis; "
                             "run scripts/s2_siril_coordinate_proof.py run on the real Siril first")
        return hypothesis
    key = name or adapter.DEFAULT_HYPOTHESIS
    if key not in adapter.HYPOTHESES:
        raise SystemExit(f"unknown hypothesis {key!r}; choose from {sorted(adapter.HYPOTHESES)}")
    return adapter.HYPOTHESES[key]


def select_positions(sequence: dict, frames: list[int] | None, positions: list[int] | None,
                     ref_position: int | None = None) -> list[int]:
    by_index = {image["index"]: image["position"] for image in sequence["images"]}
    chosen: list[int] = []
    if frames:
        for index in frames:
            if index not in by_index:
                raise SystemExit(f"image index {index} is not in the sequence (indices {sorted(by_index)})")
            chosen.append(by_index[index])
    if positions:
        chosen.extend(positions)
    ref = sequence["header"]["reference_image"] if ref_position is None else ref_position
    chosen = [p for p in dict.fromkeys(chosen) if p != ref]
    if not chosen:
        chosen = [p for p in range(len(sequence["images"])) if p != ref]
    return chosen


def _fmt(value, digits=3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(result: dict, run: Path) -> str:
    hyp = result["hypothesis"]
    header = result["sequence_header"]
    lines = ["# S2 真实序列配准诊断（只读 Siril 产物）", "",
             f"运行目录：`{run.name}`  ·  生成时间：{result['environment']['timestamp']}",
             f"序列：`{result['sequence_source']['path']}`（sha256 {result['sequence_source']['sha256'][:12]}…，"
             f"{header['nb_images']} 帧，参考位置 {header['reference_image']}，seq 版本 {header['version']}）",
             f"坐标假设：**{hyp['name']}**（行序 {hyp['row_order']}，偏移 x{hyp['x_offset']:+.1f}/y{hyp['y_offset']:+.1f}，"
             f"矩阵方向 {hyp['matrix_direction']}，{'已实证' if hyp['verified'] else '未实证'}）",
             f"关联半径 {result['association_radius_px']} px · 角色：{result['roles']} · 层 {result['layer']} · "
             f"比较参考：{result['comparison_reference']}" + (f"（{result['adapter_note']}）" if result.get('adapter_note') else ""), ""]
    if not hyp["verified"]:
        lines += ["> 未实证假设：残差幅度、中心/四角对比与本页数值不依赖上下翻转；上下标签与半像素偏移需等 "
                  "`scripts/s2_siril_coordinate_proof.py run` 在真实 Siril 上跑过才能确认。", ""]
    lines += ["## 每帧汇总", "",
              "| 帧 | 匹配/帧星数 | 匹配率 | 歧义 | 中心 RMS | 四角合并 RMS | 四角 p95/max | 其他 RMS | 总体 RMS | 几何交集 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for frame in result["report"]["frames"]:
        stem = frame["frame_id"]
        s = result["summaries"].get(stem, {})
        assoc = s.get("association", {})
        pooled = s.get("pooled", {}).get("all_roles_pooled", {}) if s.get("pooled", {}).get("available") else {}
        if frame["state"] != "succeeded":
            lines.append(f"| {stem} | – | – | – | 失败：{frame.get('error')} | | | | | |")
            continue
        corners = pooled.get("corners_pooled", {})
        overlap = frame.get("geometric_overlap", {})
        frac = overlap.get("reference_area_fraction") if isinstance(overlap, dict) else None
        lines.append(f"| {stem} | {assoc.get('matched_count', '–')}/{assoc.get('frame_count', '–')} | "
                     f"{_fmt(assoc.get('matched_fraction_of_frame'))} | {assoc.get('ambiguous_count', '–')} | "
                     f"{_fmt(pooled.get('center', {}).get('rms_px'))} | {_fmt(corners.get('rms_px'))} | "
                     f"{_fmt(corners.get('p95_px'))}/{_fmt(corners.get('max_px'))} | "
                     f"{_fmt(pooled.get('other', {}).get('rms_px'))} | {_fmt(pooled.get('overall', {}).get('rms_px'))} | "
                     f"{_fmt(frac)} |")
    lines += ["", "## 粗网格（Siril 原生坐标带，x 列 0→3 自左起，y 带 0→2 自 Siril y=0 起）", ""]
    for frame in result["report"]["frames"]:
        stem = frame["frame_id"]
        grid = result["summaries"].get(stem, {}).get("grid", {})
        if not grid.get("available"):
            continue
        lines += [f"### {stem}", "", "| 单元 | n | RMS | 中位 | mean dx | mean dy |", "|---|---|---|---|---|---|"]
        for cell, entry in grid["cells"].items():
            lines.append(f"| {cell} | {entry['count']} | {_fmt(entry['rms_px'])} | {_fmt(entry['median_px'])} | "
                         f"{_fmt(entry['mean_dx_px'], 2)} | {_fmt(entry['mean_dy_px'], 2)} |")
        lines.append("")
    screen = result.get("fixed_position_screen", {})
    lines += ["## 固定位置检测筛查（坏点候选：两表中帧坐标相同的检测）", "",
              "| 表对 | 预期星点位移 (px) | 容差 (px) | 候选数 | 有效 |", "|---|---|---|---|---|"]
    for pair, entry in screen.get("pairs", {}).items():
        lines.append(f"| {pair} | {_fmt(entry.get('expected_shift_px'), 1)} | {entry.get('tolerance_px', '–')} | "
                     f"{entry.get('candidate_count', '–')} | {'是' if entry.get('informative') else '否：' + str(entry.get('reason'))} |")
    lines += ["", f"> {screen.get('note', '')}", ""]
    lines += ["## 注意事项", ""] + [f"- {c}" for c in result["caveats"]]
    lines += ["", "## 来源", ""]
    for stem, src in result["star_list_sources"].items():
        lines.append(f"- `{stem}.lst`：sha256 {src['sha256'][:12]}…，{src['size_bytes']} B，参数 {result['star_list_parameters'][stem]}")
    lines += ["", f"环境：{result['environment']['platform']} · Python {result['environment']['python']} · NumPy {result['environment']['numpy']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", required=True, help="original sequence file (e.g. work/light_.seq)")
    parser.add_argument("--lst-dir", required=True, help="directory with cache/<stem>.lst star lists")
    parser.add_argument("--frames", type=int, nargs="*", help="Siril image indices to analyse (e.g. 2 16)")
    parser.add_argument("--positions", type=int, nargs="*", help="0-based sequence positions to analyse")
    parser.add_argument("--reference-frame", type=int, default=None,
                        help="Siril image index to compare against instead of the sequence reference "
                             "(frames are chained through inv(H_ref) @ H_frame); use it when the reference's cached "
                             "star list was detected differently from the others")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--hypothesis", default=None, help=f"one of {sorted(adapter.HYPOTHESES)}")
    parser.add_argument("--proof", default=None, help="coordinate_proof.json from a real Siril proof run")
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    hypothesis = resolve_hypothesis(args.hypothesis, args.proof)
    sequence = siril_sequence.read_seq(args.seq)
    if sequence["all_identity_per_layer"].get(args.layer):
        raise SystemExit("all homographies are identity: this is a registered output sequence (r_*.seq); pass the original light_.seq")
    ref_position = sequence["header"]["reference_image"]
    if args.reference_frame is not None:
        by_index = {image["index"]: image["position"] for image in sequence["images"]}
        if args.reference_frame not in by_index:
            raise SystemExit(f"reference image index {args.reference_frame} is not in the sequence")
        ref_position = by_index[args.reference_frame]
    positions = select_positions(sequence, args.frames, args.positions, ref_position)
    if not positions:
        raise SystemExit("no frame left to analyse after excluding the comparison reference")
    needed = [sequence["images"][ref_position]["stem"]] + [sequence["images"][p]["stem"] for p in positions]
    lst_dir = Path(args.lst_dir)
    lst_paths = {}
    for stem in needed:
        path = lst_dir / f"{stem}.lst"
        if not path.exists():
            raise SystemExit(f"missing star list {path}")
        lst_paths[stem] = path
    result = adapter.run_sequence_diagnostics(seq_path=args.seq, lst_paths=lst_paths, width=args.width, height=args.height,
                                              hypothesis=hypothesis, positions=positions, layer=args.layer, radius=args.radius,
                                              reference_position=None if args.reference_frame is None else ref_position)
    result["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                             "timestamp": _now(), "argv": sys.argv[1:] if argv is None else list(argv)}
    result["frames_requested"] = {"indices": args.frames, "positions": positions, "reference_frame_index": args.reference_frame,
                                  "reference_position": ref_position}
    run = _new_run_dir(Path(args.output_root).resolve(), args.label)
    (run / "diagnostics_input.json").write_text(json.dumps(result["document"], ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "diagnostics_report.json").write_text(json.dumps(result["report"], ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("document", "report")}
    (run / "association_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "SUMMARY.md").write_text(render_summary(result, run), encoding="utf-8")
    failed = [f["frame_id"] for f in result["report"]["frames"] if f["state"] != "succeeded"]
    print(f"{'partial' if failed else 'ok'}: {run / 'SUMMARY.md'}" + (f" (failed frames: {failed})" if failed else ""))
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
