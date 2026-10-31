"""Registration-only run over a long time span: does the shared radial term still hold?

Selects every N-th RAW of a folder. The current 48-frame development folder
is already sampled from every third original exposure across the whole 31 min;
use ``--every 1`` to retain all 48, with about 39 seconds between folder inputs.
The script hard-links the selected files into a new run directory, converts them
with the product's conversion commands and runs Siril two-pass registration
*only* (transforms + star lists; no r_ frames, no stack), then runs the
adapter diagnostics on every frame and the shared-term analysis
(``registration_distortion``).  Disk: the 16-bit converted frames only
(~274 MB per frame; 48 frames ~13 GB).  Nothing outside the run directory
is modified; the RAW files are read only.

    <output-root>/register-only-<label>-<UTC stamp>-<6hex>/
        inputs/                   hard links (frame00001.dng ...) + inputs.json (which RAW is which)
        work/                     light_*.fit, light_.seq, cache/*.lst
        01_convert.log/.ssf, 02_register.log/.ssf, siril.ini
        diagnostics-all/          adapter diagnostics, all frames vs the Siril-chosen reference
        distortion-term-*/        shared radial / field terms, transfer tests
        REGISTER_ONLY.md, register_only.json

Spans in the distortion analysis are in *selected-frame* units (1 unit =
``--every`` folder inputs = ``--every`` x cadence seconds). For the curated
48-frame folder, cadence is about 39 seconds, not the original 13 seconds.

Example (Windows, product venv, ~5 min):

    .venv\\Scripts\\python.exe scripts\\s2_register_only_long_span.py ^
        --raw-dir outputs\\efficiency-20260919\\accept-efficient-48-20260919T165004Z-87d540\\inputs ^
        --every 1 --output-root outputs\\fable-s2-20260917
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

from starstack import core  # noqa: E402
from starstack import registration_adapter as adapter  # noqa: E402
from starstack import registration_distortion as rd  # noqa: E402
from starstack import siril  # noqa: E402
from starstack import siril_sequence  # noqa: E402

RAW = core.RAW


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def select_frames(raw_dir: Path, every: int, limit: int | None, start: int = 0) -> list[Path]:
    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in RAW)
    chosen = files[start::every]
    return chosen[:limit] if limit else chosen


def link_inputs(files: list[Path], target: Path) -> dict:
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
    (target / "inputs.json").write_text(json.dumps({"method": method, "frames": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"method": method, "count": len(files), "frames": records}


def reassociation_stage(work: Path, sequence: dict, hypothesis, width: int, height: int, distortion: dict, document: dict,
                        out: Path, *, radii=(20.0, 10.0, 5.0, 3.0), log=print) -> dict:
    """Re-associate every frame's full star list under the fitted shared radial term (uncensored evaluation).

    Frames Siril dropped (zero matrix) get an initial homography extrapolated
    from their neighbours; all others start from the joint fit.  Writes
    ``reassociation.json`` into ``out`` and returns the per-frame table.
    """
    from starstack.registration_models import fit_homography
    model = distortion["models"]["homography_plus_shared_radial"]
    dist = rd.RadialDistortion(**model["parameters"])
    ref_stem = document["reference"]["frame_id"]
    lists = {}
    for image in sequence["images"]:
        path = work / "cache" / f"{image['stem']}.lst"
        if path.exists():
            stars = siril_sequence.read_lst(path)["stars"]
            lists[image["stem"]] = np.array([adapter.convert_point(hypothesis, s["x"], s["y"], height) for s in stars], dtype=np.float64).reshape(-1, 2)
    h_by_frame = {fid: np.asarray(h) for fid, h in model.get("H_undistorted", {}).items()}
    if not h_by_frame:   # older analysis output: refit each frame's homography under the distortion from the document pairs
        for frame in document["frames"]:
            if frame.get("source_to_reference") is None or not frame.get("matches"):
                continue
            src = np.array([m["source_xy"] for m in frame["matches"]], dtype=np.float64)
            dst = np.array([m["reference_xy"] for m in frame["matches"]], dtype=np.float64)
            h_by_frame[frame["frame_id"]] = fit_homography(dist.apply(src), dist.apply(dst))
    spans = {stem: rd.span_of(stem, ref_stem) for stem in lists}
    result = rd.reassociate_sequence(dist, h_by_frame, spans, lists, ref_stem, width, height, radii=radii)
    for fid, e in sorted(result.items(), key=lambda kv: kv[1]["span"]):
        r = e.get("residuals", {}).get("validation", {})
        log(f"  reassoc {fid} span {e['span']:+d} {e['status']:14s} {e['initial_homography'][:12]:12s} matched {e['matched']}/{e['frame_count']} "
            f"corner {_fmt(r.get('corner', {}).get('rms_px'))} overall max {_fmt(r.get('overall', {}).get('max_px'), 2)}")
    payload = {"distortion": model["parameters"], "radii_px": list(radii), "reference": ref_stem, "frames": result,
               "note": "full star lists (Siril cache) paired by mutual nearest neighbour under D^-1(H D(p)); validation half of the final pairs"}
    (out / "reassociation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def render(report: dict) -> str:
    frames = report["diagnostics"]["frames"]
    spans = report["distortion"]["spans"]
    models = report["distortion"]["models"]
    lines = [f"# 长跨度只配准运行（{report['run']}）", "",
             f"生成：{report['environment']['timestamp']} · {report['inputs']['count']} 帧（每 {report['every']} 张取 1，跨度单位 = {report['every']} 张原始帧）· "
             f"参考 {report['reference']} · Siril convert {_fmt(report['siril_seconds']['convert'], 1)} s，register {_fmt(report['siril_seconds']['register'], 1)} s", "",
             f"Siril 完成配准 {report.get('registered_count', '–')}/{report['inputs']['count']} 帧；无法匹配而被 Siril 排除（零矩阵、未入选）：{report.get('excluded_by_siril') or '无'}", "",
             "## 每帧：适配器诊断（对 Siril 参考）与共享项后的验证集四角 RMS（px）", "",
             "| 帧 | 跨度 | 原始帧 | 匹配 | 四角 RMS（Siril H） | 基线单应 | +共享径向 | +场 deg2 |", "|---|---|---|---|---|---|---|---|"]
    by_stem = {r["stem"]: r["source"] for r in report["inputs"]["frames"]}
    for stem in sorted(spans, key=lambda s: spans[s]):
        d = frames.get(stem, {})
        cells = [_fmt(models[m]["frames"][stem]["residuals"]["validation"]["corner"]["rms_px"])
                 for m in ("homography_per_frame", "homography_plus_shared_radial", "homography_plus_shared_field_deg2")]
        lines.append(f"| {stem} | {spans[stem]:+d} | {Path(by_stem.get(stem, '')).name} | {d.get('matched')}/{d.get('frame_stars')} | "
                     f"{_fmt(d.get('corners_rms_px'))} | " + " | ".join(cells) + " |")
    rad = models["homography_plus_shared_radial"]
    p = rad["parameters"]
    lines += ["", "## 共享径向项", "",
              f"k1 {p['k1']:.5g}，k2 {p['k2']:.5g}，中心 ({p['cx']:.1f}, {p['cy']:.1f})，最远角位移 {rad['corner_shift_px']:.1f} px；"
              f"16 帧作业（§17）为 k1 0.0536、k2 −0.0194、中心 (4055, 2807)、176 px。", "",
              "## 迁移测试（验证集四角 RMS）", "", "| 测试 | 留出帧 | 跨度 | 基线 | 径向 | 场2 |", "|---|---|---|---|---|---|"]
    for name, t in report["distortion"]["transfer"].items():
        if "skipped" in t:
            lines.append(f"| {name} | – | – | 跳过 | | |")
            continue
        for fid in t["radial"]["held_out_frames"]:
            lines.append(f"| {name} | {fid} | {spans[fid]:+d} | {_fmt(t['baseline_corner_rms_px'][fid])} | "
                         f"{_fmt(t['radial']['held_out_validation_corner_rms_px'][fid])} | {_fmt(t['field_deg2']['held_out_validation_corner_rms_px'][fid])} |")
    reassoc = report.get("reassociation") or {}
    if reassoc:
        lines += ["", "## 用共享径向项对完整星表重新关联（不经 Siril 的 H；验证半集）", "",
                  "| 帧 | 跨度 | 初值 | 状态 | 匹配/帧星数 | 匹配率 | 中心 RMS | 四角 RMS | 四角 max | 总体 max |", "|---|---|---|---|---|---|---|---|---|---|"]
        for fid, e in sorted(reassoc.items(), key=lambda kv: kv[1]["span"]):
            r = e.get("residuals", {}).get("validation", {})
            lines.append(f"| {fid} | {e['span']:+d} | {e['initial_homography']} | {e['status']} | {e['matched']}/{e['frame_count']} | "
                         f"{e['matched_fraction']:.2f} | {_fmt(r.get('center', {}).get('rms_px'))} | {_fmt(r.get('corner', {}).get('rms_px'))} | "
                         f"{_fmt(r.get('corner', {}).get('max_px'), 2)} | {_fmt(r.get('overall', {}).get('max_px'), 2)} |")
    lines += ["", "## 注意事项", "",
              "- 只算变换与星表，不写 r_ 帧、不堆栈；work 内只有 16 位转换帧、序列文件与缓存星表。",
              "- 跨度单位是抽样后的帧数；乘以抽样间隔与拍摄节奏才是时间。",
              "- 对应仍经 Siril 自己的 H 在关联半径内建立；长跨度上若 Siril 的 H 本身偏差超过关联半径，匹配数会先掉，这一列要同时看。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=None, help="folder with the RAW frames (read-only)")
    parser.add_argument("--reuse-run", default=None, help="an existing run directory whose Siril products are complete: "
                                                         "skip linking/convert/register and only (re)run the analysis")
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--siril", default=None)
    parser.add_argument("--radius", type=float, default=8.0, help="association radius; wider than 6 for long spans")
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--label", default="long-span")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    import s2_accept_two_pass as accept  # noqa: E402  (reuse the diagnostics wrapper and log parsers)
    if args.reuse_run:
        run = Path(args.reuse_run).resolve()
        work = run / "work"
        if not (work / core.SEQUENCE_FILE).exists() or not (run / "inputs" / "inputs.json").exists():
            raise SystemExit(f"{run} has no complete Siril products (work/light_.seq, inputs/inputs.json)")
        recorded = json.loads((run / "inputs" / "inputs.json").read_text(encoding="utf-8"))
        inputs = {"method": recorded["method"], "count": len(recorded["frames"]), "frames": recorded["frames"]}
        raw_dir = Path(recorded["frames"][0]["source"]).parent if recorded["frames"] else None
        status = {"reused": True}
        t_convert = accept.parse_execution_seconds((run / "01_convert.log").read_text(encoding="utf-8", errors="replace")) if (run / "01_convert.log").exists() else None
        t_register = accept.parse_execution_seconds((run / "02_register.log").read_text(encoding="utf-8", errors="replace")) if (run / "02_register.log").exists() else None
        converted = sorted(work.glob("light_*.fit"))
        print(f"reusing {run}: {len(converted)} converted frames")
    else:
        if not args.raw_dir:
            raise SystemExit("--raw-dir is required unless --reuse-run is given")
        status = siril.doctor(args.siril)
        if not status.get("available"):
            raise SystemExit(f"Siril not available: {status.get('message')}")
        raw_dir = Path(args.raw_dir).resolve()
        files = select_frames(raw_dir, args.every, args.limit, args.start)
        if len(files) < 3:
            raise SystemExit("fewer than 3 RAW frames selected")
        if len({p.suffix.lower() for p in files}) != 1:
            raise SystemExit("mixed RAW extensions")
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = Path(args.output_root).resolve() / f"register-only-{args.label}-{stamp}-{uuid.uuid4().hex[:6]}"
        run.mkdir(parents=True, exist_ok=False)
        print(f"run: {run}")
        inputs = link_inputs(files, run / "inputs")
        print(f"inputs: {inputs['count']} RAW ({inputs['method']}), every {args.every} from {raw_dir}")
        work = run / "work"
        work.mkdir()

        def no_cancel():
            return None

        started = time.monotonic()
        siril.run_script(status["executable"], run, "01_convert", ["setext fit", "set32bits", "setcompress 0", "cd inputs",
                                                                 "convertraw light -debayer -out=../work"], no_cancel)
        t_convert = time.monotonic() - started
        converted = sorted(work.glob("light_*.fit"))
        if len(converted) != len(files):
            raise SystemExit(f"Siril converted {len(converted)} of {len(files)} frames; see {run / '01_convert.log'}")
        print(f"converted {len(converted)} frames in {t_convert:.0f} s")
        started = time.monotonic()
        siril.run_script(status["executable"], run, "02_register", ["cd work", "register light -2pass -transf=homography"], no_cancel)
        t_register = time.monotonic() - started
        print(f"registered in {t_register:.0f} s")

    sequence = siril_sequence.read_seq(work / core.SEQUENCE_FILE)
    ref = sequence["header"]["reference_image"]
    ref_stem = sequence["images"][ref]["stem"]
    from starstack import fits_crop
    header, _ = fits_crop.read_header(converted[0])
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    hypothesis = adapter.HYPOTHESES[args.hypothesis or adapter.DEFAULT_HYPOTHESIS]
    print("adapter diagnostics on all frames ...")
    diag_label = "all" if not (run / "diagnostics-all").exists() else f"all-{_dt.datetime.now(_dt.timezone.utc).strftime('%H%M%S')}"
    diagnostics = accept.adapter_diagnostics(work / core.SEQUENCE_FILE, work / "cache", width, height, hypothesis, run, diag_label,
                                             radius=args.radius)
    document = json.loads((Path(diagnostics["dir"]) / "diagnostics_input.json").read_text(encoding="utf-8"))
    frames = rd.frames_from_document(document)
    print("shared-term analysis ...")
    distortion = rd.analyse(frames, width, height, degrees=(1, 2))
    distortion["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__, "timestamp": _now()}
    out = run / f"distortion-term-{args.label}"
    if out.exists():   # a re-run of the analysis on the same run directory keeps the earlier attempt
        out = run / f"distortion-term-{args.label}-{_dt.datetime.now(_dt.timezone.utc).strftime('%H%M%S')}"
    out.mkdir()
    (out / "distortion_term.json").write_text(json.dumps(distortion, ensure_ascii=False, indent=1), encoding="utf-8")
    import s2_distortion_term as dt_script  # noqa: E402
    (out / "DISTORTION.md").write_text(dt_script.render(distortion, out, str(Path(diagnostics["dir"]) / "diagnostics_input.json")), encoding="utf-8")
    print("re-association of full star lists under the shared radial term ...")
    reassociation = reassociation_stage(work, sequence, hypothesis, width, height, distortion, document, out)

    excluded = [im["stem"] for im in sequence["images"] if not im["selected"]]
    report = {"run": run.name, "raw_dir": str(raw_dir), "every": args.every, "start": args.start, "inputs": inputs, "siril": status,
              "excluded_by_siril": excluded, "registered_count": len(sequence["images"]) - len(excluded),
              "reassociation": {fid: {k: v for k, v in e.items() if k != "grid_validation"} for fid, e in reassociation.items()},
              "reference": ref_stem, "sequence_header": sequence["header"], "width": width, "height": height,
              "siril_seconds": {"convert": t_convert, "register": t_register}, "hypothesis": hypothesis.as_dict(),
              "diagnostics": diagnostics, "distortion": {k: v for k, v in distortion.items() if k != "models"} | {
                  "models": {name: {"parameters": m.get("parameters"), "corner_shift_px": m.get("corner_shift_px"),
                                    "field_per_unit_span_px": m.get("field_per_unit_span_px"),
                                    "frames": {fid: {"residuals": {"validation": {"corner": e["residuals"]["validation"]["corner"],
                                                                                   "overall": e["residuals"]["validation"]["overall"]}}}
                                               for fid, e in m["frames"].items()}}
                             for name, m in distortion["models"].items()}},
              "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                              "timestamp": _now(), "argv": sys.argv[1:] if argv is None else list(argv)}}
    (run / "register_only.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "REGISTER_ONLY.md").write_text(render(report), encoding="utf-8")
    print(f"ok: {run / 'REGISTER_ONLY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
