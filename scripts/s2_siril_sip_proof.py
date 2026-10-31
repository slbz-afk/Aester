"""Does Siril 1.4.4 accept a SIP distortion solution we write, and apply it in register/seqapplyreg?

Same shape as the coordinate proof: ``generate`` writes a synthetic distorted
sequence plus ``disto_solution.fit`` (the reference frame carrying a WCS+SIP
header that encodes the same radial distortion); ``run`` executes four Siril
variants on copies of the frames — ``disto_file`` (``-disto=file
disto_solution.fit``, CRPIX2 bottom-up), ``disto_file_topdown`` (CRPIX2 as a
top-down row), ``sip_in_frames`` (every frame carries the WCS+SIP header and
``-disto=image`` refers to the sequence reference) and ``nodisto`` (the
product commands, for contrast), each followed by ``seqapplyreg light`` — and
``analyze`` decides from Siril's products whether the homographies are the
undistorted truth, whether the star lists were corrected, and whether the
exported frames are undistorted.  ``simulate`` writes labelled fake outputs
so the analysis can be exercised off-Windows (never a proof).

Own-registration variants (RUN_LOG §22, ``ownreg_mag`` / ``ownreg_mag_shift``):
Siril registers once without distortion (the product's probe), we rewrite the
sequence file with *our* homographies plus the ``D`` line for the magnified
SIP file, and Siril only runs ``seqapplyreg``.  The shifted variant adds a
known (+7.3, -4.1) px translation and zeroes frame 4's quality fields; the
exported frames must follow the written matrices for ``own_registration_applied``.

    <output-root>/sip-proof-<UTC stamp>-<6hex>/
        work/             light_0000k.fit, disto_solution.fit, truth.json
        v_<variant>/      copies + Siril products of each variant (disto_image, disto_file, sip_in_frames, nodisto)
        *.ssf / *.log     Siril scripts and logs
        sip_proof.json, SIP_PROOF.md

Example (Windows, product venv):

    .venv\\Scripts\\python.exe scripts\\s2_siril_sip_proof.py run --output-root outputs\\fable-s2-20260917
    .venv\\Scripts\\python.exe scripts\\s2_siril_sip_proof.py analyze --run outputs\\fable-s2-20260917\\sip-proof-<stamp>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from starstack import siril_sip_proof as sp  # noqa: E402

VARIANTS = {
    # "-disto=file <path>": Siril reads the WCS+SIP of that file (validate_disto_params in distorsion.c requires a
    # wcslib solution with SIP terms and the sequence's frame size); CRPIX2 in the standard bottom-up convention
    "disto_file": ["register light -2pass -transf=homography -disto=file disto_solution.fit", "seqapplyreg light"],
    # same, CRPIX2 written as a top-down row (Siril treats crpix as being in its own grid, siril_wcs.c pix2wcs2)
    "disto_file_topdown": ["register light -2pass -transf=homography -disto=file disto_solution_topdown.fit", "seqapplyreg light"],
    # header-only .wcs file in Siril's own distortion-file form (NAXIS 0 + IMAGEW/IMAGEH), what the product would write
    "disto_wcs": ["register light -2pass -transf=homography -disto=file disto_solution.wcs", "seqapplyreg light"],
    # the same with the global magnification (linear SIP terms) the product uses to keep barrel corners inside the box
    "disto_wcs_mag": ["register light -2pass -transf=homography -disto=file disto_solution_mag.wcs", "seqapplyreg light"],
    # NOTE (RUN_LOG §22): a solution file with AP/BP order 9 is accepted ("畸变数据有效") and then crashes Siril 1.4.4 in both
    # register and seqapplyreg (sip-proof-20260918T053047Z-4559e1); orders above 5 are not offered here any more.  The product
    # instead fits the lens on the undistorted side (``--distortion z8inv``) so that the order-5 AP/BP are exact.
    # every frame carries the WCS+SIP itself, as a plate-solved sequence would, and "-disto=image" refers to the
    # sequence *reference* image (command.c: sequence_find_refimage), not to a loaded file
    # (init_disto_data then copies the WCS of the *loaded* image, gfit, so the reference frame is loaded first)
    "sip_in_frames": ["load light_00001.fit", "register light -2pass -transf=homography -disto=image", "seqapplyreg light"],
    # the product commands, for contrast
    "nodisto": ["register light -2pass -transf=homography", "seqapplyreg light"],
}

# Own registration (RUN_LOG §22): Siril only detects/registers once without distortion (the product's probe), then
# *we* rewrite the sequence file with our homographies (truth, in the magnified undistorted space) plus the D line,
# and Siril only runs ``seqapplyreg``.  The shifted variant proves the exported frames follow the written matrices:
# every non-reference frame must land at the reference's undistorted positions + shift.  Frame 4 of that variant
# gets zeroed quality fields (as a frame Siril could not match would have) to show ``seqapplyreg`` still exports it.
OWN_REGISTRATION_VARIANTS = {
    "ownreg_mag": {"shift_px": None, "zero_quality_frames": (), "solution": "solution_wcs_magnified"},
    "ownreg_mag_shift": {"shift_px": (7.3, -4.1), "zero_quality_frames": (4,), "solution": "solution_wcs_magnified"},
}
OWN_REGISTRATION_STEPS = (["register light -2pass -transf=homography"], ["seqapplyreg light"])


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fmt(v, digits=3):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def _new_run_dir(root: Path) -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = root.resolve() / f"sip-proof-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    return run


DISTORTIONS = {"default": sp.default_distortion, "z8": sp.z8_like_distortion, "z8inv": sp.z8_like_inverse_distortion}


def generate(run: Path, distortion: str = "default") -> dict:
    return sp.generate_dataset(run / "work", distortion=DISTORTIONS[distortion]())


def _copy_frames(work: Path, target: Path, *, sip_in_frames: bool = False) -> None:
    target.mkdir(exist_ok=False)
    if sip_in_frames:
        from starstack.registration_distortion import RadialDistortion
        from starstack.synthetic_starfield import read_fits_minimal, write_fits_rgb16
        truth = json.loads((work / "truth.json").read_text(encoding="utf-8"))
        cards = sp.sip_cards(RadialDistortion(**truth["distortion"]), truth["width"], truth["height"])
        for path in sorted(work.glob("light_*.fit")):
            _, plane = read_fits_minimal(path)
            write_fits_rgb16(target / path.name, plane, extra_cards=cards, comments=["frame rewritten with the synthetic WCS+SIP header"])
    else:
        for path in sorted(work.glob("light_*.fit")):
            shutil.copyfile(path, target / path.name)
    for name in ("disto_solution.fit", "disto_solution_topdown.fit", "disto_solution.wcs", "disto_solution_mag.wcs"):
        if (work / name).exists():
            shutil.copyfile(work / name, target / name)


def run_variants(run: Path, siril_path: str | None, variants: list[str]) -> dict:
    from starstack import siril
    status = siril.doctor(siril_path)
    if not status.get("available"):
        raise SystemExit(f"Siril not available: {status.get('message')}")
    work = run / "work"

    def no_cancel():
        return None

    outcome = {"siril": status, "variants": {}}
    truth = json.loads((work / "truth.json").read_text(encoding="utf-8"))
    for name in variants:
        target = run / f"v_{name}"
        _copy_frames(work, target, sip_in_frames=(name == "sip_in_frames"))
        if name in OWN_REGISTRATION_VARIANTS:
            spec = OWN_REGISTRATION_VARIANTS[name]
            step_a = [f"cd v_{name}", *OWN_REGISTRATION_STEPS[0]]
            step_b = [f"cd v_{name}", *OWN_REGISTRATION_STEPS[1]]
            entry = {"commands": {"a_siril": step_a, "b_python": "own_registration_sequence", "c_siril": step_b},
                     "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}}
            try:
                siril.run_script(status["executable"], run, f"register_{name}_a", step_a, no_cancel)
                magnified = truth[spec.get("solution", "solution_wcs_magnified")]
                entry["written"] = sp.own_registration_sequence(target, truth, distortion_file=magnified["file"], magnification=magnified["magnification"],
                                                                shift_px=spec["shift_px"], zero_quality_frames=tuple(spec["zero_quality_frames"]))
                shutil.copyfile(target / "light_.seq", target / "light_written_by_us.seq")
                siril.run_script(status["executable"], run, f"register_{name}_b", step_b, no_cancel)
                entry["status"] = "ran"
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = str(exc)[-2000:]
            entry["files"] = sorted(p.name for p in target.iterdir())
            outcome["variants"][name] = entry
            continue
        commands = [f"cd v_{name}", *VARIANTS[name]]
        entry = {"commands": commands}
        try:
            siril.run_script(status["executable"], run, f"register_{name}", commands, no_cancel)
            entry["status"] = "ran"
        except Exception as exc:  # keep the log; the analysis will say what is missing
            entry["status"] = "failed"
            entry["error"] = str(exc)[-2000:]
        entry["files"] = sorted(p.name for p in target.iterdir())
        outcome["variants"][name] = entry
    return outcome


def analyze(run: Path, *, simulated: bool = False, siril_version: str | None = None) -> dict:
    truth = json.loads((run / "work" / "truth.json").read_text(encoding="utf-8"))
    results = {"proof_version": sp.SIP_PROOF_VERSION, "run": run.name, "truth": {k: v for k, v in truth.items() if k not in ("frames", "reference_stars_undistorted")},
               "variants": {}, "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__, "timestamp": _now()}}
    for target in sorted(run.glob("v_*")):
        name = target.name[2:]
        if not (target / "light_.seq").exists():
            results["variants"][name] = {"status": "no_sequence_file", "files": sorted(p.name for p in target.iterdir())}
            continue
        own = OWN_REGISTRATION_VARIANTS.get(name)
        magnification = truth.get("solution_wcs_magnified", {}).get("magnification", 1.0) if (name.endswith("_mag") or own is not None) else 1.0
        try:
            results["variants"][name] = sp.analyze_run(truth, target, variant=name, simulated=simulated, siril_version=siril_version,
                                                       magnification=magnification, own_registration=own is not None,
                                                       expected_shift_px=(own or {}).get("shift_px"))
        except Exception as exc:
            results["variants"][name] = {"status": "analysis_error", "error": f"{type(exc).__name__}: {exc}"}
    return results


def render(results: dict) -> str:
    t = results["truth"]
    lines = [f"# Siril SIP 畸变实证（{results['run']}）", "",
             f"生成：{results['environment']['timestamp']} · 真值畸变 k1 {t['distortion']['k1']}，k2 {t['distortion']['k2']}，中心 ({t['distortion']['cx']:.1f}, {t['distortion']['cy']:.1f})，"
             f"最远角位移 {t['distortion_corner_shift_px']:.1f} px · SIP 正向阶 {t['sip']['forward_order']}，逆向拟合最大误差 {t['sip']['inverse_max_fit_error_px']:.4f} px"
             + (f"（放大后按阶数：{ {k: round(v, 4) for k, v in t['sip'].get('inverse_max_fit_error_px_by_order', {}).items()} }）" if t['sip'].get('inverse_max_fit_error_px_by_order') else "")
             + (f" · 模型方向 {t['sip'].get('direction')}，正向 A/B 拟合最大误差 {t['sip'].get('forward_max_fit_error_px', 0.0):.4f} px" if t['sip'].get('direction') else ""), ""]
    for name, v in results["variants"].items():
        lines += [f"## 变体 `{name}`：**{v.get('status')}**", ""]
        if "frames" not in v:
            lines += [f"{v}", ""]
            continue
        own_note = (f" · 我们写入的 H（期望位移 {v.get('expected_shift_px')}）" if v.get("own_registration") else "")
        lines += [f"参考帧 {v['reference_frame']} · 放大率 {v.get('magnification', 1.0):.4f}{own_note} · {v.get('reason')}", "",
                  "| 帧 | 星表落在畸变真值/去畸变真值的比例 | H 对去畸变真值角误差 | H 对畸变拟合角误差 | 输出帧星点在去畸变参考位置（离边 ≥30 px 的星） | 在畸变参考位置 | 输出尺寸/位深 |",
                  "|---|---|---|---|---|---|---|"]
        for stem, e in v["frames"].items():
            sl = e.get("star_list", {})
            h = e.get("homography", {})
            o = e.get("output", {})
            und, dis = o.get("at_undistorted_reference", {}), o.get("at_distorted_reference", {})
            ui, di = und.get("interior", {}), dis.get("interior", {})
            lines.append(f"| {stem} | {_fmt(sl.get('fraction_at_distorted_truth'), 2)}/{_fmt(sl.get('fraction_at_undistorted_truth'), 2)} | "
                         f"{_fmt(h.get('corner_error_vs_undistorted_truth_px'))} | {_fmt(h.get('corner_error_vs_distorted_fit_px'))} | "
                         f"{ui.get('count', '–')}: 中位 {_fmt(ui.get('median_px'))} p95 {_fmt(ui.get('p95_px'))}（全部 {und.get('count', '–')} 星 RMS {_fmt(und.get('rms_px'))}） | "
                         f"{di.get('count', '–')}: 中位 {_fmt(di.get('median_px'))}（RMS {_fmt(dis.get('rms_px'))}） | "
                         f"{und.get('width', '–')}×{und.get('height', '–')} / {und.get('bitpix', '–')} |")
        lines.append("")
    lines += ["## 读法", "",
              "- `sip_applied`：每个非参考帧的 H 都等于去畸变空间里的真值（角误差 ≤0.15 px）且导出帧离边 ≥30 px 的星点落在参考的去畸变位置（中位 ≤0.2、p95 ≤0.5 px）——Siril 接受并使用了我们写的 SIP。贴边星在平移/旋转后落进无数据区，单独列出不计入判定。",
              "- `sip_ignored`：H 等于在畸变星点上拟合的单应，导出帧星点仍在畸变位置——SIP 被忽略（或语法/头未被接受，看日志）。",
              "- `own_registration_applied`（`ownreg_*` 变体）：seq 里的 H 是我们按已实证约定写入的（真值，`_shift` 变体再叠加已知位移），Siril 只跑 `seqapplyreg`；导出帧的星点落在参考去畸变位置 + 期望位移（且不在未位移处）——Siril 用的是我们写的矩阵和 D 行指定的 SIP 文件。`own_registration_ignored`：导出帧不随写入的矩阵走。",
              "- 星表列说明 `cache/*.lst` 存的是原始检测还是校正后的位置；两者都可能，只影响适配器读法。",
              "- 带 `_SIMULATED` 的结论来自离线模拟，不是证据。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
        p.add_argument("--distortion", choices=sorted(DISTORTIONS), default="default",
                       help="synthetic lens: mild default, Z8-like (3.6 %, p_to_u) or Z8-like on the undistorted side (z8inv, exact AP/BP)")
        if name == "run":
            p.add_argument("--siril", default=None)
            p.add_argument("--variants", nargs="*", default=list(VARIANTS) + list(OWN_REGISTRATION_VARIANTS))
        if name == "simulate":
            p.add_argument("--applied", action="store_true", help="simulate Siril honouring the SIP")
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--siril-version", default=None)
    args = parser.parse_args(argv)

    if args.command == "generate":
        run = _new_run_dir(Path(args.output_root))
        generate(run, args.distortion)
        print(f"generated: {run}")
        return 0
    if args.command == "run":
        run = _new_run_dir(Path(args.output_root))
        generate(run, args.distortion)
        outcome = run_variants(run, args.siril, args.variants)
        (run / "run_outcome.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=1), encoding="utf-8")
        results = analyze(run, siril_version=outcome["siril"].get("version"))
        results["run_outcome"] = outcome
        (run / "sip_proof.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        (run / "SIP_PROOF.md").write_text(render(results), encoding="utf-8")
        statuses = {name: v.get("status") for name, v in results["variants"].items()}
        print(f"{statuses}: {run / 'SIP_PROOF.md'}")
        wanted = {name: ("own_registration_applied" if name in OWN_REGISTRATION_VARIANTS else "sip_applied") for name in args.variants}
        return 0 if any(statuses.get(name) == status for name, status in wanted.items()) else 2
    if args.command == "simulate":
        run = _new_run_dir(Path(args.output_root))
        truth = generate(run, args.distortion)
        for name, applied in (("disto_file", args.applied), ("nodisto", False)):
            target = run / f"v_{name}"
            _copy_frames(run / "work", target)
            sp.simulate_siril_outputs(truth, target, applied=applied)
        # own registration, offline: a plain (no-distortion) sequence rewritten with our matrices, then a fake seqapplyreg
        target = run / "v_ownreg_mag_shift"
        _copy_frames(run / "work", target)
        sp.simulate_siril_outputs(truth, target, applied=False)
        spec = OWN_REGISTRATION_VARIANTS["ownreg_mag_shift"]
        magnified = truth["solution_wcs_magnified"]
        sp.own_registration_sequence(target, truth, distortion_file=magnified["file"], magnification=magnified["magnification"],
                                     shift_px=spec["shift_px"], zero_quality_frames=tuple(spec["zero_quality_frames"]))
        sp.simulate_seqapplyreg(truth, target, magnification=magnified["magnification"])
        results = analyze(run, simulated=True)
        (run / "sip_proof.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        (run / "SIP_PROOF.md").write_text(render(results), encoding="utf-8")
        print(f"simulated: {run / 'SIP_PROOF.md'}")
        return 0
    run = Path(args.run).resolve()
    results = analyze(run, siril_version=args.siril_version)
    (run / "sip_proof.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (run / "SIP_PROOF.md").write_text(render(results), encoding="utf-8")
    print({name: v.get("status") for name, v in results["variants"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
