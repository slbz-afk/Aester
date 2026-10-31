"""S2 next package (b): does a Siril registration option recover the lost pairs
for long-span frames?  Runs ``register`` variants on hard links of the real
frames (no copies, no output frames when the build offers ``-noout``) and
evaluates every resulting sequence with the adapter on the same criteria.

    <output-root>/siril-variants-<UTC stamp>-<6hex>/
        help_register.log        output of Siril's own ``help register`` (documents the options of this build)
        variants.json / VARIANTS.md
        v_<name>/                hard links to the frames, Siril's light_.seq, cache/*.lst, logs

Windows (project root; ~1–2 min per variant, a few MB of disk):

    .venv\\Scripts\\python.exe scripts\\s2_siril_register_variants.py ^
        --work "outputs\\z8-meteor-20260916\\smoke-jobs\\job-9dcccd46202d4c0b\\work" --output-root outputs\\fable-s2-20260917

Variants are only run when every option they need appears in ``help register``;
a variant that would write registered frames is skipped unless
``--allow-output-frames`` is given.  The original job directory is never written.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from starstack import registration_models as models  # noqa: E402
from starstack import siril_sequence  # noqa: E402
from starstack.registration_adapter import associate_stars  # noqa: E402

VARIANTS = [
    {"name": "baseline", "needs": [], "commands": ["register light -transf=homography"],
     "note": "product command; -noout appended when the build has it"},
    {"name": "twopass", "needs": ["-2pass"], "commands": ["register light -2pass -transf=homography"],
     "note": "two-pass registration (Siril picks the reference and computes transforms only)"},
    {"name": "twopass_maxstars1500", "needs": ["-2pass", "-maxstars"],
     "commands": ["register light -2pass -transf=homography -maxstars=1500"],
     "note": "two-pass with a lower star cap: does any frame still hit the cap and re-adjust its threshold?"},
    {"name": "minpairs4", "needs": ["-minpairs"], "commands": ["register light -transf=homography -minpairs=4"],
     "note": "lower pair threshold; tests whether the matcher, not the threshold, limits the pairs"},
]
HELP_COMMANDS = ["register", "seqapplyreg"]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def link_frames(work: Path, target: Path, stems) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    method = None
    for stem in stems:
        src = work / f"{stem}.fit"
        if not src.exists():
            raise SystemExit(f"missing frame {src}")
        dst = target / src.name
        try:
            os.link(src, dst)
            method = method or "hardlink"
        except OSError:
            raise SystemExit(f"cannot hard-link {src} -> {dst}; refusing to copy 274 MB frames silently")
    return {"method": method, "count": len(stems)}


def parse_help_options(text: str) -> list[str]:
    """Option tokens mentioned by ``help register`` (e.g. -2pass, -noout, -minpairs); pure numbers excluded."""
    tokens = re.findall(r"(?<![\w-])(-[A-Za-z0-9][\w]*)", text)
    return sorted({t for t in tokens if re.search(r"[A-Za-z]", t)})


def evaluate_sequence(seq_path: Path, cache: Path, width: int, height: int, radius: float) -> dict:
    seq = siril_sequence.read_seq(seq_path)
    header = seq["header"]
    regdata = seq["regdata"].get(1)
    if regdata is None:
        return {"error": "no layer-1 registration data"}
    ref_pos = header["reference_image"]
    ref_stem = seq["images"][ref_pos]["stem"]
    ref_list = siril_sequence.read_lst(cache / f"{ref_stem}.lst")
    out = {"reference": ref_stem, "reference_position": ref_pos, "frames": {}}
    ref_stars = ref_list["stars"]
    for image in seq["images"]:
        pos = image["position"]
        if pos == ref_pos:
            continue
        lst = cache / f"{image['stem']}.lst"
        if not lst.exists():
            out["frames"][image["stem"]] = {"error": "no star list"}
            continue
        stars = siril_sequence.read_lst(lst)["stars"]
        h = np.asarray(regdata[pos]["H"], dtype=np.float64)
        assoc = associate_stars(ref_stars, stars, h, radius=radius)
        fi = [p["frame_index"] for p in assoc["pairs"]]
        ri = [p["reference_index"] for p in assoc["pairs"]]
        if not fi:
            out["frames"][image["stem"]] = {"pairs": 0}
            continue
        src = np.array([[stars[i]["x"], stars[i]["y"]] for i in fi])
        dst = np.array([[ref_stars[j]["x"], ref_stars[j]["y"]] for j in ri])
        d = models.apply_matrix(h, src) - dst
        dist = np.hypot(d[:, 0], d[:, 1])
        regions = np.array([models.region_of(x, y, width, height) for x, y in dst])
        entry = {"pairs": int(len(fi)), "overall_rms_px": float(np.sqrt(np.mean(dist ** 2))), "max_px": float(dist.max()),
                 "siril_number_of_stars": regdata[pos]["number_of_stars"]}
        for region in ("center", "corner", "other"):
            m = regions == region
            entry[region] = {"count": int(m.sum()), "rms_px": float(np.sqrt(np.mean(dist[m] ** 2))) if m.any() else None,
                             "mean_dx_px": float(d[m, 0].mean()) if m.any() else None, "mean_dy_px": float(d[m, 1].mean()) if m.any() else None}
        out["frames"][image["stem"]] = entry
    return out


def parse_pairs_from_log(text: str) -> dict:
    """Per-image pair counts from a register log; Siril prints English or localized (zh-CN) lines."""
    pairs = {}
    current = None
    for line in text.splitlines():
        m = re.search(r"(?:Matching stars in image|匹配星点于图像)\s*(\d+)", line)
        if m:
            current = int(m.group(1))
        m2 = re.search(r"(?:Initial pair matches|初始星点匹配)[:：]\s*(\d+)", line)
        if m2 and current is not None:
            pairs.setdefault(current, {})["initial"] = int(m2.group(1))
        m3 = re.search(r"(?:Pair matches after fitting|拟合后星点匹配)[:：]\s*(\d+)", line)
        if m3 and current is not None:
            pairs.setdefault(current, {})["after_fitting"] = int(m3.group(1))
        m4 = re.search(r"(?:Inliers|内点)[:：]?\s*([\d.]+)", line)
        if m4 and current is not None:
            pairs.setdefault(current, {})["inliers"] = float(m4.group(1))
    return pairs


def parse_detection_from_log(text: str) -> dict:
    """Per-image detection FWHM in processing order (English or zh-CN 'Found 2000 ... (FWHM x)')."""
    fwhm = []
    for line in text.splitlines():
        m = re.search(r"\(FWHM\s*([\d.]+)\)|（FWHM\s*([\d.]+)）", line)
        if m:
            fwhm.append(float(m.group(1) or m.group(2)))
    threshold_readjust = sum(1 for line in text.splitlines() if "重新调整阈值" in line or "adjusting threshold" in line.lower())
    return {"detection_fwhm_in_log_order": fwhm, "threshold_readjust_messages": threshold_readjust}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, help="directory with light_0000k.fit of the real job (read-only)")
    parser.add_argument("--width", type=int, default=8280)
    parser.add_argument("--height", type=int, default=5520)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--siril", default=None)
    parser.add_argument("--variants", nargs="*", default=[v["name"] for v in VARIANTS])
    parser.add_argument("--allow-output-frames", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "fable-s2-20260917"))
    args = parser.parse_args(argv)

    from starstack import siril
    status = siril.doctor(args.siril)
    if not status.get("available"):
        raise SystemExit(f"Siril not available: {status.get('message')}")
    work = Path(args.work).resolve()
    stems = sorted(p.stem for p in work.glob("light_*.fit") if re.fullmatch(r"light_\d{5}", p.stem))
    if len(stems) < 3:
        raise SystemExit(f"fewer than 3 light_ frames in {work}")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output_root).resolve() / f"siril-variants-{stamp}-{uuid.uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)

    def no_cancel():
        return None

    # 0. what does this build document for `register` (and `seqapplyreg`, needed to apply 2-pass transforms)?
    siril.run_script(status["executable"], run, "help_register", ["help register"], no_cancel)
    help_text = (run / "help_register.log").read_text(encoding="utf-8", errors="replace")
    options = parse_help_options(help_text)
    for extra in HELP_COMMANDS[1:]:
        try:
            siril.run_script(status["executable"], run, f"help_{extra}", [f"help {extra}"], no_cancel)
        except Exception as exc:  # documentation only; never blocks the variants
            (run / f"help_{extra}.error.txt").write_text(str(exc), encoding="utf-8")
    report = {"siril": status, "started_at": _now(), "work": str(work), "frames": stems, "help_options": options,
              "variants": {}}
    for spec in VARIANTS:
        if spec["name"] not in args.variants:
            continue
        missing = [o for o in spec["needs"] if o not in options]
        entry = {"note": spec["note"], "needs": spec["needs"]}
        if missing:
            entry["status"] = "skipped"
            entry["reason"] = f"options not in help register: {missing}"
            report["variants"][spec["name"]] = entry
            continue
        commands = list(spec["commands"])
        writes_frames = "-2pass" not in commands[-1]
        if writes_frames and "-noout" in options:
            commands[-1] += " -noout"
            writes_frames = False
        if writes_frames and not args.allow_output_frames:
            entry["status"] = "skipped"
            entry["reason"] = "would write registered frames (no -noout in this build); pass --allow-output-frames"
            report["variants"][spec["name"]] = entry
            continue
        vdir = run / f"v_{spec['name']}"
        entry["link"] = link_frames(work, vdir, stems)
        entry["commands"] = ["cd " + vdir.name] + commands
        try:
            siril.run_script(status["executable"], run, f"register_{spec['name']}", entry["commands"], no_cancel)
            entry["status"] = "ran"
        except Exception as exc:  # keep going with the other variants
            entry["status"] = "failed"
            entry["error"] = str(exc)
            report["variants"][spec["name"]] = entry
            continue
        log = (run / f"register_{spec['name']}.log").read_text(encoding="utf-8", errors="replace")
        entry["log_pairs"] = parse_pairs_from_log(log)
        entry["log_detection"] = parse_detection_from_log(log)
        seq_path = vdir / "light_.seq"
        if seq_path.exists():
            entry["evaluation"] = evaluate_sequence(seq_path, vdir / "cache", args.width, args.height, args.radius)
        else:
            entry["evaluation"] = {"error": "no light_.seq produced"}
        report["variants"][spec["name"]] = entry
    report["finished_at"] = _now()
    report["environment"] = {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__}
    (run / "variants.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Siril register 变体实验（{report['started_at']}，Siril {status.get('version')}）", "",
             f"帧：{len(stems)} 个硬链接自 `{work}`；`help register` 列出的选项：{' '.join(options) or '（无法解析）'}", ""]
    for name, entry in report["variants"].items():
        lines.append(f"## {name} — {entry['status']}")
        lines.append(f"{entry['note']}；命令：`{' / '.join(entry.get('commands', []))}`" + (f"；原因：{entry.get('reason') or entry.get('error')}" if entry["status"] != "ran" else ""))
        ev = entry.get("evaluation") or {}
        if ev.get("frames"):
            lines += ["", f"参考：{ev['reference']}（位置 {ev['reference_position']}）", "",
                      "| 帧 | 日志初始配对 | 拟合后 | 适配器对应 | 总体 RMS | 中心 | 四角（n） | 四角平均 dx,dy | max |", "|---|---|---|---|---|---|---|---|---|"]
            for stem, fr in ev["frames"].items():
                idx = int(stem[-5:])
                lp = entry.get("log_pairs", {}).get(idx, {})
                if "pairs" not in fr or fr["pairs"] == 0:
                    lines.append(f"| {stem} | {lp.get('initial', '–')} | {lp.get('after_fitting', '–')} | {fr.get('pairs', '–')} | {fr.get('error', '')} | | | | |")
                    continue
                c = fr["corner"]
                centre_rms = "–" if fr["center"]["rms_px"] is None else f"{fr['center']['rms_px']:.3f}"
                corner_rms = "–" if c["rms_px"] is None else f"{c['rms_px']:.3f}"
                corner_dx = "–" if c["mean_dx_px"] is None else f"{c['mean_dx_px']:+.2f}"
                corner_dy = "–" if c["mean_dy_px"] is None else f"{c['mean_dy_px']:+.2f}"
                lines.append(f"| {stem} | {lp.get('initial', '–')} | {lp.get('after_fitting', '–')} | {fr['pairs']} | {fr['overall_rms_px']:.3f} | "
                             f"{centre_rms} | {corner_rms}（{c['count']}） | {corner_dx},{corner_dy} | {fr['max_px']:.2f} |")
        lines.append("")
    lines += ["注意：适配器对应经各变体自己的 H 在 6 px 内关联，因此只反映各自 H 的一致性；`-2pass` 可能更换参考帧，比较时以参考列为准。"
              "硬链接的帧与原作业共享内容，Siril 只读它们；原作业目录未写入。"]
    (run / "VARIANTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ok: {run / 'VARIANTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
