"""EFF-D4 KD-5 (efficiency P1c, docs/EFFICIENCY_P1C_PREREG_20261005.md §4.2): consecutive contract L stacks, real Siril.

Development and validation tool only (not shipped).  It copies the first --frames registered frames (r_light_NNNNN.fit)
of a finished job's work folder, e.g. the F48 reference job, into a fresh folder, writes a sequence file of exactly
those frames (the source's S/L/I/R lines cut to N images, its reference kept when it is among them, no statistics
lines), and then runs the stack-settings contract stack --runs times in a row, each exactly as the job worker runs
L's 02_stack: core._contract_stack (plan from the sequence and the first selected frame, settings lines before
seqclean, work/siril-stack.ini, OMP_NUM_THREADS=1, log and STACKCNT checks) with siril.run_script inside
siril.job_stage_launches(), AESTER_SIRIL_PIN unset.  The sequence file is restored before every run.  With
--affinity N this process is first confined to the first N logical processors of its processor group (Siril
inherits it), as the bench's o16 profile.

The source is only read.  Any failure (a non-zero Siril exit, a failed check) stops the series: nothing is retried.
Passes (exit 0) when every run succeeded and every master has the same data digest; 1 otherwise; 2 for usage errors.

  kd5_contract_stack_repeat.py --source <job>/work --folder <scratch>/kd5 --frames 16 --runs 20 [--affinity 16]
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import core, siril, stack_settings  # noqa: E402

SCHEMA = "aester-kd5-contract-stack-repeat-1"
SIRIL = ROOT / "tools" / "siril" / "bin" / "siril-cli.exe"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sequence_subset(text: str, frames: int) -> str:
    """The source sequence cut to its first ``frames`` images, all selected, without statistics lines."""
    out, images, regs = [], 0, {}
    header = None
    for line in text.splitlines():
        tag = line.split(None, 1)[0] if line.strip() else ""
        if tag == "S":
            header = line
            continue
        if tag == "I":
            images += 1
            if images <= frames:
                out.append(("I", f"I {line.split()[1]} 1"))
            continue
        if tag.startswith("R") and tag[1:].isdigit():
            regs[tag] = regs.get(tag, 0) + 1
            if regs[tag] <= frames:
                out.append(("R", line))
            continue
        if tag.startswith("M"):
            continue
        out.append(("", line))
    if header is None or images < frames:
        raise ValueError(f"the source sequence has {images} images, fewer than {frames}")
    fields = header.split("'")
    rest = fields[2].split()
    reference = int(rest[4]) if int(rest[4]) < frames else 0
    rest[1], rest[2], rest[4] = str(frames), str(frames), str(reference)   # nb_images, nb_selected, reference
    lines = [line for kind, line in out if kind == ""]
    body = [line for kind, line in out if kind in ("I", "R")]
    head = [line for line in lines if line.startswith("#")]
    tail = [line for line in lines if not line.startswith("#")]
    return "\n".join(head + [f"{fields[0]}'{fields[1]}' {' '.join(rest)}"] + tail + body) + "\n"


def prepare(source: Path, folder: Path, frames: int) -> dict:
    if folder.exists():
        raise ValueError(f"{folder} exists; choose a new folder")
    work = folder / "work"
    work.mkdir(parents=True)
    names = [f"r_light_{i:05d}.fit" for i in range(1, frames + 1)]
    for name in names:
        if not (source / name).is_file():
            raise ValueError(f"{source / name} is missing")
    for name in names:
        shutil.copyfile(source / name, work / name)
    text = sequence_subset((source / "r_light_.seq").read_text(encoding="utf-8"), frames)
    (folder / "pristine.seq").write_text(text, encoding="utf-8", newline="\n")
    ini = source.parent / "siril.ini"
    if ini.is_file():
        shutil.copyfile(ini, folder / "siril.ini")
    else:
        (folder / "siril.ini").write_bytes(b"")
    return {"source": str(source), "frames": names, "bytes": sum((work / n).stat().st_size for n in names)}


def master_digest(path: Path) -> str:
    from astropy.io import fits
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdus:
        return hashlib.sha256(hdus[0].data.tobytes()).hexdigest()


def set_affinity(logical: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    mask = (1 << logical) - 1
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        raise OSError(f"SetProcessAffinityMask failed ({ctypes.get_last_error()})")
    return mask


def run_series(folder: Path, executable: str, runs: int) -> list:
    work = folder / "work"
    results = []
    for index in range(runs):
        (work / "sky_master.fit").unlink(missing_ok=True)
        shutil.copyfile(folder / "pristine.seq", work / "r_light_.seq")
        plans, started = [], time.perf_counter()
        offset = stack_settings.log_offset(folder / "02_stack.log")
        entry = {"index": index, "started_at": now()}
        try:
            with siril.job_stage_launches():
                core._contract_stack(siril.run_script, executable, folder, core.LEGACY_STACK_SCRIPT, lambda: None,
                                     note=plans.append)
            entry.update(ok=True, master=master_digest(work / "sky_master.fit"))
        except Exception as exc:          # a Siril failure or a failed check: the series stops, nothing is retried
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:3000])
        entry.update(seconds=round(time.perf_counter() - started, 3), plan=plans[-1] if plans else None,
                     log=stack_settings.log_lines(folder / "02_stack.log", offset))
        results.append(entry)
        if not entry["ok"]:
            break
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="a finished job's work folder (read only)")
    parser.add_argument("--folder", type=Path, required=True, help="a new folder for the copies and the runs")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--affinity", type=int, help="confine this process (and Siril) to N logical processors")
    parser.add_argument("--siril", type=Path, default=SIRIL)
    parser.add_argument("--reuse", action="store_true", help="reuse a folder prepared by an earlier call")
    args = parser.parse_args(argv)
    if args.frames < 2 or args.runs < 1 or (args.affinity is not None and not 1 <= args.affinity <= 64):
        parser.error("--frames >= 2, --runs >= 1, --affinity 1..64")
    folder = args.folder.resolve()
    try:
        if not Path(args.siril).is_file():
            raise ValueError(f"missing Siril: {args.siril}")
        prepared = (json.loads((folder / "prepared.json").read_text(encoding="utf-8")) if args.reuse
                    else prepare(args.source.resolve(), folder, args.frames))
        if not args.reuse:
            (folder / "prepared.json").write_text(json.dumps(prepared, indent=1), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"kd5_contract_stack_repeat: {exc}", file=sys.stderr)
        return 2
    os.environ.pop(siril.SIRIL_PIN_ENV, None)
    os.environ.pop(core.L_LAYOUT_ENV, None)
    report = {"schema": SCHEMA, "contract": stack_settings.contract(), "folder": str(folder),
              "siril": str(Path(args.siril).resolve()), "frames": args.frames, "runs_requested": args.runs,
              "affinity_mask": hex(set_affinity(args.affinity)) if args.affinity else None, "started_at": now()}
    results = run_series(folder, str(Path(args.siril).resolve()), args.runs)
    masters = {r["master"] for r in results if r.get("ok")}
    report.update(results=results, finished_at=now(), runs=len(results),
                  succeeded=sum(1 for r in results if r["ok"]), masters=sorted(masters),
                  plans=sorted({json.dumps(r["plan"], sort_keys=True) for r in results if r["plan"]}))
    report["passed"] = report["succeeded"] == args.runs and len(masters) == 1
    name = f"kd5-{'a' + str(args.affinity) if args.affinity else 'unpinned'}.json"
    (folder / name).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "runs", "succeeded", "masters", "plans", "affinity_mask")},
                     ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
