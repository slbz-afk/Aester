"""S2 coordinate proof: run Siril on an asymmetric synthetic star field and decide
its row order, pixel origin and homography direction from the results.

Usage (Windows, real Siril; one command, new output directory each time):

    .venv\\Scripts\\python.exe scripts\\s2_siril_coordinate_proof.py run --output-root outputs\\fable-s2-20260917

Steps: ``generate`` writes four 1200x901 16-bit RGB FITS frames plus
``truth.json``; ``run`` converts and registers them with the same Siril
commands the product uses (``convert`` / ``setref light 1`` /
``register light -transf=homography``) inside an isolated job directory,
then ``analyze`` compares Siril's ``light_.seq`` and ``cache/*.lst`` with the
truth and writes ``coordinate_proof.json`` and ``PROOF.md``.

``simulate --convention ...`` writes fake Siril outputs from the truth so that
the analysis can be exercised without Siril; such runs are labelled
SIMULATED everywhere and are not a proof of anything about Siril.

Nothing outside ``--output-root`` is written.  Existing run directories are
never reused.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from starstack import synthetic_starfield as sf  # noqa: E402
from starstack import siril_coordinate_proof as cp  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_run_dir(root: Path, label: str) -> Path:
    # Siril resolves the -s/-i/-d arguments itself and, after changing its CWD to the job
    # directory, a relative script path no longer exists ("命令行使用绝对路径"); the run
    # directory is therefore always absolute, whatever --output-root was given.
    root = Path(root).resolve()
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = root / f"coordinate-proof-{label}-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def _environment(extra=None) -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
            "timestamp": _now()}
    info.update(extra or {})
    return info


def generate(run: Path) -> dict:
    stage = run / "job" / "stage"
    truth = sf.generate_dataset(stage)
    (run / "job" / "work").mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": _now(), "frames": [{"filename": f["filename"], "sha256": f["sha256"]} for f in truth["frames"]],
                "truth_sha256": _sha(stage / "truth.json")}
    (run / "generate.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return truth


def run_siril(run: Path, siril_path: str | None) -> dict:
    from starstack import siril
    status = siril.doctor(siril_path)
    if not status.get("available"):
        raise RuntimeError(f"Siril not available: {status.get('message')}")
    job = (run / "job").resolve()

    def no_cancel():
        return None

    log = {"siril": status, "started_at": _now(), "commands": {}}
    convert = ["setext fit", "set32bits", "setcompress 0", "cd stage", "convert light -out=../work"]
    siril.run_script(status["executable"], job, "01_convert", convert, no_cancel)
    log["commands"]["01_convert"] = convert
    register = ["cd work", "setref light 1", "register light -transf=homography"]
    siril.run_script(status["executable"], job, "02_register", register, no_cancel)
    log["commands"]["02_register"] = register
    log["finished_at"] = _now()
    work = job / "work"
    log["products"] = {p.name: _sha(p) for p in sorted(work.glob("*.seq"))}
    log["products"].update({f"cache/{p.name}": _sha(p) for p in sorted((work / "cache").glob("*.lst"))} if (work / "cache").exists() else {})
    (run / "siril_run.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return log


def analyze(run: Path, *, simulated: bool, siril_version: str | None) -> dict:
    truth = json.loads((run / "job" / "stage" / "truth.json").read_text(encoding="utf-8"))
    work = run / "job" / "work"
    proof = cp.analyze_run(truth, work / "light_.seq", work / "cache", simulated=simulated, siril_version=siril_version)
    proof["environment"] = _environment()
    (run / "coordinate_proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "PROOF.md").write_text(render_markdown(proof), encoding="utf-8")
    return proof


def render_markdown(proof: dict) -> str:
    kind = "SIMULATED（非 Siril 实际输出，只检验分析逻辑）" if proof["simulated"] else "真实 Siril 运行"
    lines = [f"# Siril 坐标约定实证 — {kind}", "",
             f"状态：**{proof['status']}**  ·  Siril 版本：{proof.get('siril_version') or '未记录'}  ·  分析时间：{proof['environment']['timestamp']}", ""]
    if proof["status"] == "proved":
        lines += [f"- 行序：Siril 星点/矩阵坐标相对 TOP-DOWN 真值为 **{proof['row_order']}**",
                  f"- 像素原点偏移（Siril → TOP-DOWN 零基像素中心）：x {proof['x_offset']:+.1f}，y {proof['y_offset']:+.1f}",
                  f"- 单应矩阵方向：**{proof['matrix_direction']}**", ""]
    else:
        lines += [f"- 未能判定：{proof.get('reason')}", ""]
    c = proof["evidence"].get("row_order_candidates", {})
    lines += ["## 行序候选（参考帧星点对真值）", "", "| 候选 | 匹配/真值 | 原始 x 偏移 | 原始 y 偏移 | 离散(MAD) |", "|---|---|---|---|---|"]
    for name, e in c.items():
        lines.append(f"| {name} | {e['matched']}/{e['truth']} | {e.get('x_offset_raw', float('nan')):+.3f} | {e.get('y_offset_raw', float('nan')):+.3f} | {e.get('spread_mad_px', float('nan')):.3f} |")
    lines += ["", "## 每帧绑定与矩阵方向", "", "| 帧 | 绑定 | 星点匹配率 | H 前向误差(px) | H 反向误差(px) | 判定 |", "|---|---|---|---|---|---|"]
    binding = proof["evidence"].get("frame_binding", {})
    direction = proof["evidence"].get("homography_direction", {})
    for stem, b in binding.items():
        d = direction.get(stem, {})
        lines.append(f"| {stem} | {'ok' if b['binding_ok'] else 'FAIL'} (truth {b['best_truth_frame']}) | {b['matched_fraction_expected']:.3f} | "
                     f"{d.get('corner_error_if_frame_to_reference_px', float('nan')):.4f} | {d.get('corner_error_if_reference_to_frame_px', float('nan')):.3f} | {d.get('verdict') or '参考帧'} |")
    lines += ["", "判据：行序候选之一匹配 ≥80 % 且离散 ≤0.15 px、另一候选 ≤20 %；偏移在 0.1 px 内落到 0.5 的倍数；"
              "每个非参考帧的矩阵方向一致（角点传递误差 ≤0.1 px，另一方向 >1 px）；星表按内容绑定到真值帧。",
              "此结论只对本次 Siril 版本与命令序列成立；不涉及任何实拍数据。"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
        if name == "run":
            p.add_argument("--siril", default=None, help="siril-cli path; default: doctor() search")
        if name == "simulate":
            p.add_argument("--row-order", choices=("TOP-DOWN", "BOTTOM-UP"), default="BOTTOM-UP")
            p.add_argument("--offset", type=float, default=0.0)
            p.add_argument("--direction", choices=("frame_to_reference", "reference_to_frame"), default="frame_to_reference")
    p = sub.add_parser("analyze")
    p.add_argument("run_dir")
    p.add_argument("--simulated", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        run = Path(args.run_dir).resolve()
        version = None
        log = run / "siril_run.json"
        if log.exists():
            version = json.loads(log.read_text(encoding="utf-8")).get("siril", {}).get("version")
        proof = analyze(run, simulated=args.simulated, siril_version=version)
        print(f"{proof['status']}: {run / 'coordinate_proof.json'}")
        return 0 if proof["status"] == "proved" else 2
    root = Path(args.output_root)
    if args.command == "generate":
        run = _new_run_dir(root, "generate")
        generate(run)
        print(f"generated: {run / 'job' / 'stage'}")
        return 0
    if args.command == "simulate":
        run = _new_run_dir(root, "SIMULATED")
        truth = generate(run)
        cp.simulate_siril_outputs(truth, run / "job" / "work", row_order=args.row_order, x_offset=args.offset,
                                  y_offset=args.offset, matrix_direction=args.direction)
        proof = analyze(run, simulated=True, siril_version=None)
        print(f"SIMULATED {proof['status']}: {run / 'coordinate_proof.json'}")
        return 0 if proof["status"] == "proved" else 2
    run = _new_run_dir(root, "siril")
    generate(run)
    try:
        log = run_siril(run, args.siril)
    except Exception as exc:  # keep the generated dataset for a manual run
        (run / "siril_run.json").write_text(json.dumps({"error": str(exc), "environment": _environment()}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"error: Siril step failed: {exc}\n  dataset kept at {run / 'job' / 'stage'}", file=sys.stderr)
        return 1
    proof = analyze(run, simulated=False, siril_version=log["siril"].get("version"))
    print(f"{proof['status']}: {run / 'coordinate_proof.json'}")
    return 0 if proof["status"] == "proved" else 2


if __name__ == "__main__":
    sys.exit(main())
