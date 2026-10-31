"""Efficiency P1b-3 phase replay: L's Siril export + stack, combined (A) against split (B), on a kept real job.

Development and validation tool only (not shipped).  It replays the registration_stack phase of a finished,
unpinned L job with own registration (path 3a, the one every L bench takes) in a separate replay folder:

  A  02_stack  = core.stack_script(DISTORTION_WCS, own_registration=True)   (the pre-P1b single process)
  B  02_export = core.export_script(DISTORTION_WCS, own_registration=True), then
     02_stack  = core.LEGACY_STACK_SCRIPT                                  (the P1b-3 layout)

in the order ABBA x --blocks, every launch through siril.run_script inside siril.job_stage_launches(), exactly
as the job worker launches them.  Before each run the folder is reset as the job worker resets it between
attempts: r_light* and sky_master* deleted, light_.seq rewritten by the product's
registration_calibration.write_registration_sequence, siril.ini restored.

  replay_legacy_stack.py --replay DIR --job JOB_DIR --profile dev     # prepares DIR (copies the light frames)
  replay_legacy_stack.py --replay DIR --profile o16                    # reuses DIR
  replay_legacy_stack.py --replay DIR --profile hx --expect-manifest results/<D3 run>/manifest.json

Preparation copies work/light_*.fit, distortion.json, distortion.wcs and the job's siril.ini into DIR/work
(about 13 GB for 48 frames) and fingerprints the job's sky_master_undistorted.fit; it never modifies or deletes
the job folder (delete the kept bench slot afterwards with the bench's own tools).  Profiles: dev (no
affinity, unpinned), o16 (the first 16 logical processors of this process's group, unpinned), hx (o16 plus
AESTER_SIRIL_PIN=mem=9).  The affinity is set on this process before the first launch; the guarded Siril
children inherit it.

Recorded per run: the phase wall time; per Siril process its Job Object wall, CPU, peak job memory, process
count and exit code; digests (verifier recipe: header cards without DATE/CHECKSUM/DATASUM, data block digest)
of the master and, in the first block, of every r_light frame; text digests of r_light_.seq, light_.seq and
siril.ini; from the stack log, Siril's untranslated "Memory per image: X MB. Max memory: Y MB" and
"allocating data for N threads" lines and its processor count.

Acceptance (P1b prereg, lsplit A4): (i) one master over all runs, equal to the source job's (or
--expect-manifest's) sky_master_undistorted.fit data digest, one r_light_.seq / light_.seq / siril.ini, r_light
frames identical over the first block; (ii) thread and normalization-worker lines identical in A and B;
(iii) median paired phase_wall(B) - phase_wall(A) <= +1.5 s (dev), +2.0 s (o16, hx); (iv) memory: hx every pair
B peak <= 1.05 x A peak, dev/o16 median of B/A <= 1.10.  Any Siril failure stops the replay (no retry) and
fails it.  Exit code 0 when every criterion passes, 1 otherwise, 2 for usage errors.

--contract d4 (efficiency P1c KD-2, docs/EFFICIENCY_P1C_PREREG_20261005.md §4.2): the EFF-D4 stack-settings replay.
The source may be a job admitted under the contract (its 02_stack.ssf carries exactly the settings lines before
seqclean) or one admitted before it.  Each run resets the folder as above (siril.ini restored to the job's pre-stack
copy, work/siril-stack.ini removed), runs 02_export (the 3a export script), then the contract 02_stack: the
settings lines ``setcpu T'`` / ``set core.mem_mode=1`` / ``set core.mem_amount=B`` for one --settings pair, run
with work/siril-stack.ini and the product's contract environment (OMP_NUM_THREADS=1), never with AESTER_SIRIL_PIN;
a pair that is not a contract plan for the exported sequence (T' != min(T', n, B*1024//p)) is refused.  The
control group (--control T:B, default the first pair; 'none' for none) runs the P1b stack script pinned with
AESTER_SIRIL_PIN=cpu=T,mem=B.  The --settings pairs (default KD-2's 48:153,16:153,10:9,8:8,4:4) and the control
run round-robin --repeat times (default 2).  Only the profile's affinity is used.

Every pair (and the control) is checked against n and p before any Siril run: preparation records them from the
source job's work/r_light_.seq and its first selected frame (``stack_sequence`` in prepared.json; else an earlier
replay's export in the folder is used), and a pair that is not a contract plan is then a usage error (exit 2).  When
neither is there, the first run's export decides, as each run's own check does.  KD-2's default tiers are the
48-frame plans; with fewer frames T' is at most n, so the 8-frame fixture takes the same tiers at T' = min(T', 8):
dev/o16 (8, 153), hx (8, 9), a8 (8, 8) and (4, 4).

  replay_legacy_stack.py --contract d4 --replay DIR --job K-L48-JOB --profile dev
  replay_legacy_stack.py --contract d4 --replay DIR8 --job JOB8 --profile dev --settings 8:153,8:9,8:8,4:4 \
      --expect-data-digest b4f7d0b6 --reference-master OLD8/sky_master_undistorted.fit --expect-differing 16

Criteria: (one_master) every contract run gives one master, equal to the expected one (--expect-data-digest prefix,
else --expect-manifest, else the source job's master); (own_plan) each contract run's log section holds exactly one
'Memory per image: p MB. Max memory: B*1024 MB' and one 'allocating data for T' threads' line equal to its own plan,
and its STACKCNT equals n; (no_wild_difference) no contract master differs from the first by more than 4 ULP or in
a NaN/Inf: otherwise the report says stop (``stop_p1c``); (control) a control whose pair is itself a contract plan
gives the same master; (reference) with --expect-differing N, the master differs from --reference-master in exactly
N values by at most 1 ULP.  The log lines of different pairs are not required to agree: T' and B depend on the
machine by design, and the master must not.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from starstack import core, process_guard, siril, stack_settings  # noqa: E402
from starstack.registration_calibration import (DISTORTION_JSON, DISTORTION_WCS, load_calibration,  # noqa: E402
                                                write_registration_sequence)
import verify_workflow_equivalence as verifier  # noqa: E402

SCHEMA = "aester-legacy-stack-replay-1"
D4_SCHEMA = "aester-legacy-stack-replay-d4-1"
D4_DEFAULT_SETTINGS = "48:153,16:153,10:9,8:8,4:4"     # KD-2: dev, o16, hx, a8 and (4, 4)
D4_MAX_ULP = 4                                          # a larger difference (or one in a NaN/Inf) stops P1c
SIRIL = ROOT / "tools" / "siril" / "bin" / "siril-cli.exe"
PROFILES = {"dev": {"affinity": None, "pin": None, "max_delta_seconds": 1.5},
            "o16": {"affinity": 16, "pin": None, "max_delta_seconds": 2.0},
            "hx": {"affinity": 16, "pin": "mem=9", "max_delta_seconds": 2.0}}
HX_MAX_PAIR_RATIO = 1.05
MEDIAN_MAX_RATIO = 1.10
PREPARED = "prepared.json"
MASTER_GLOB = "{WF}/stack/job-*/work/sky_master_undistorted.fit"
_MEMORY = re.compile(r"Memory per image: (\d+) MB\. Max memory: (\d+) MB")
_THREADS = re.compile(r"allocating data for (\d+) threads")
_PROCESSORS = re.compile(r"(?:using|使用)\s*(\d+)\s*(?:logical processors|个逻辑处理器)", re.IGNORECASE)


class ReplayError(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def script_a() -> list:
    return core.stack_script(DISTORTION_WCS, own_registration=True)


def script_b() -> tuple:
    return core.export_script(DISTORTION_WCS, own_registration=True), list(core.LEGACY_STACK_SCRIPT)


def order(blocks: int) -> list:
    return [layout for _ in range(blocks) for layout in "ABBA"]


# --------------------------------------------------------------------------- digests

def fits_digest(path: Path) -> dict:
    """The verifier's FITS recipe: header cards without DATE/CHECKSUM/DATASUM, and the data block digest."""
    entry = verifier.fingerprint_file(path, path.name)
    if "error" in entry:
        raise ReplayError(f"{path}: {entry['error']}")
    hdus = entry["content"]["hdus"]
    cards = json.dumps([h["cards"] for h in hdus], ensure_ascii=False, sort_keys=True)
    return {"header": hashlib.sha256(cards.encode("utf-8")).hexdigest(),
            "data": [h["data"]["digest"] if h["data"] else None for h in hdus]}


def text_digest(path: Path, folder: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace(str(folder).replace("\\", "\\\\"), "{REPLAY}").replace(str(folder), "{REPLAY}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_master(manifest_path: Path) -> list:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    found = [entry for key, entry in manifest["files"].items() if fnmatch.fnmatchcase(_canonical(key), MASTER_GLOB)]
    if len(found) != 1:
        raise ReplayError(f"{manifest_path}: expected one {MASTER_GLOB}, found {len(found)}")
    return [h["data"]["digest"] if h["data"] else None for h in found[0]["content"]["hdus"]]


def _canonical(key: str) -> str:
    return re.sub(r"job-[0-9a-f]{8,40}", "job-x", key)


# --------------------------------------------------------------------------- preparation

def _source_checks(job_dir: Path, contract: str = "none") -> dict:
    record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    problems = []
    if record.get("state") != "succeeded":
        problems.append(f"state {record.get('state')}")
    if record.get("support_mode", "legacy") != "legacy" or record.get("stack_range_mode", "legacy") != "legacy":
        problems.append("not an L job (support/stack range mode)")
    diagnostics = record.get("diagnostics") or {}
    if diagnostics.get("registration_source") != core.REGISTRATION_SOURCE_OWN:
        problems.append("not own registration (path 3a)")
    notes = ((diagnostics.get("timings") or {}).get("notes") or {})
    if notes.get("siril_pin"):
        problems.append("a pinned run (AESTER_SIRIL_PIN)")
    try:
        admitted = stack_settings.job_contract(record) is not None
    except ValueError as exc:
        admitted = None
        problems.append(str(exc))
    if admitted and contract != "d4":
        problems.append("a job admitted under the stack-settings contract (replay it with --contract d4)")
    for name in ("02_export.ssf", "02_stack.ssf"):
        path = job_dir / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if contract == "d4" and name == "02_stack.ssf":
            lines, found = stack_settings.without_settings(lines)   # the contract's own lines, before seqclean
            if (found is not None) != bool(admitted):
                problems.append(f"{name} {'lacks' if admitted else 'carries'} the stack-settings lines")
        if any(line.startswith(("setcpu", "set core.mem")) for line in lines):
            problems.append(f"{name} carries pin lines")
    if problems:
        raise ReplayError(f"{job_dir} cannot be replayed: " + "; ".join(problems))
    return record


def prepare(job_dir: Path, replay: Path, *, check_disk: bool = True, contract: str = "none") -> dict:
    job_dir, replay = job_dir.resolve(), replay.resolve()
    if replay.exists():
        raise ReplayError(f"{replay} exists; reuse it without --job, or choose a new folder")
    if replay == job_dir or job_dir in replay.parents or replay in job_dir.parents:
        raise ReplayError("the replay folder must not overlap the job folder")
    record = _source_checks(job_dir, contract)
    source = job_dir / "work"
    frames = sorted(p for p in source.glob("light_*.fit") if p.stem[6:].isdigit())
    if not frames:
        raise ReplayError(f"{source} has no light frames")
    for name in (DISTORTION_JSON, DISTORTION_WCS):
        if not (source / name).is_file():
            raise ReplayError(f"{source / name} is missing")
    need = 2 * sum(p.stat().st_size for p in frames) + (2 << 30)
    anchor = next(p for p in [replay, *replay.parents] if p.exists())
    free = shutil.disk_usage(anchor).free
    if check_disk and free < need:
        raise ReplayError(f"free disk {free / 2**30:.1f} GiB < needed {need / 2**30:.1f} GiB")
    work = replay / "work"
    work.mkdir(parents=True)
    (replay / "pristine").mkdir()
    for path in [*frames, source / DISTORTION_JSON, source / DISTORTION_WCS]:
        shutil.copyfile(path, work / path.name)
    shutil.copyfile(job_dir / "siril.ini", replay / "pristine" / "siril.ini")
    master = source / "sky_master_undistorted.fit"
    prepared = {"schema": SCHEMA, "prepared_at": now(), "source_job": str(job_dir), "frames": len(frames),
                "frame_bytes": sum(p.stat().st_size for p in frames), "signature": record.get("signature"),
                "source_master": fits_digest(master)["data"] if master.is_file() else None,
                "source_siril_ini": text_digest(job_dir / "siril.ini", job_dir),
                "stack_sequence": sequence_shape(source)}
    (replay / PREPARED).write_text(json.dumps(prepared, indent=1), encoding="utf-8")
    return prepared


def reset(replay: Path) -> None:
    work = replay / "work"
    for path in [*work.glob("r_light*"), *work.glob("sky_master*")]:
        path.unlink()
    (work / stack_settings.CONFIG_NAME).unlink(missing_ok=True)   # a contract stack's own configuration
    calibration = load_calibration(work)
    if calibration is None:
        raise ReplayError(f"{work / DISTORTION_JSON} is unreadable")
    write_registration_sequence(work, calibration)
    shutil.copyfile(replay / "pristine" / "siril.ini", replay / "siril.ini")   # the job's configuration before its stack


# --------------------------------------------------------------------------- one run

class Accounting:
    def __init__(self):
        self.entries = []
        process_guard.add_accounting_listener(self.entries.append)

    def since(self, mark: int) -> list:
        return self.entries[mark:]


def _stack_lines(replay: Path, offset: int) -> dict:
    path = replay / "02_stack.log"
    with path.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", errors="replace")
    memory, threads, processors = _MEMORY.findall(text), _THREADS.findall(text), _PROCESSORS.findall(text)
    found = {"memory_per_image_mb": int(memory[-1][0]) if memory else None,
             "max_memory_mb": int(memory[-1][1]) if memory else None,
             "threads": int(threads[-1]) if threads else None,
             "processors": int(processors[-1]) if processors else None}
    return found


def _log_size(replay: Path, name: str) -> int:
    path = replay / f"{name}.log"
    return path.stat().st_size if path.exists() else 0


def run_once(replay: Path, layout: str, executable: str, accounting: Accounting, *, frames_digests: bool) -> dict:
    reset(replay)
    offset = _log_size(replay, "02_stack")
    mark = len(accounting.entries)
    launches = [("02_stack", script_a())] if layout == "A" else list(zip(("02_export", "02_stack"), script_b()))
    started = time.perf_counter()
    with siril.job_stage_launches():
        for name, commands in launches:
            siril.run_script(executable, replay, name, list(commands), lambda: None)
    wall = time.perf_counter() - started
    processes = []
    for (name, _), entry in zip(launches, accounting.since(mark) + [None] * len(launches)):
        entry = entry or {}
        processes.append({"name": name, "wall_seconds": entry.get("wall_seconds"),
                          "cpu_seconds": ((entry.get("cpu_user_seconds") or 0) + (entry.get("cpu_kernel_seconds") or 0))
                          if entry else None,
                          "peak_job_memory_bytes": entry.get("peak_job_memory_bytes"),
                          "descendant_processes": entry.get("descendant_processes"),
                          "returncode": entry.get("returncode")})
    work = replay / "work"
    r_light = sorted(work.glob("r_light_*.fit"))
    lines = _stack_lines(replay, offset)
    selected = len(r_light)
    if lines["memory_per_image_mb"] and lines["max_memory_mb"] is not None and lines["processors"]:
        lines["normalization_workers"] = min(lines["max_memory_mb"] // lines["memory_per_image_mb"],
                                             lines["processors"], selected)
    else:
        lines["normalization_workers"] = None
    result = {"layout": layout, "phase_wall_seconds": wall, "processes": processes,
              "peak_job_memory_bytes": max((p["peak_job_memory_bytes"] or 0 for p in processes), default=0) or None,
              "master": fits_digest(work / "sky_master.fit"), "r_light_count": selected,
              "texts": {name: text_digest(work / name, replay) for name in ("r_light_.seq", "light_.seq")}
              | {"siril.ini": text_digest(replay / "siril.ini", replay)},
              "stack_log": lines}
    if frames_digests:
        combined = hashlib.sha256(json.dumps([fits_digest(p) for p in r_light], sort_keys=True).encode()).hexdigest()
        result["r_light_digest"] = combined
    return result


# --------------------------------------------------------------------------- criteria

def evaluate(runs: list, profile: str, expected_master) -> dict:
    limits = PROFILES[profile]
    masters = {json.dumps(r["master"], sort_keys=True) for r in runs}
    data = {json.dumps(r["master"]["data"]) for r in runs}
    texts = {name: {r["texts"][name] for r in runs} for name in ("r_light_.seq", "light_.seq", "siril.ini")}
    frames = {r["r_light_digest"] for r in runs if "r_light_digest" in r}
    criterion_i = {"masters": len(masters), "texts": {k: len(v) for k, v in texts.items()}, "r_light_digests": len(frames),
                   "expected_master_data": expected_master,
                   "matches_expected": None if expected_master is None else data == {json.dumps(expected_master)}}
    criterion_i["passed"] = (len(masters) == 1 and all(len(v) == 1 for v in texts.values()) and len(frames) == 1
                             and criterion_i["matches_expected"] is not False)
    keys = ("threads", "normalization_workers", "memory_per_image_mb", "processors")
    by_layout = {layout: {json.dumps({k: r["stack_log"][k] for k in keys}) for r in runs if r["layout"] == layout}
                 for layout in "AB"}
    criterion_ii = {"A": sorted(by_layout["A"]), "B": sorted(by_layout["B"])}
    criterion_ii["passed"] = (len(by_layout["A"]) == 1 and by_layout["A"] == by_layout["B"]
                              and all(r["stack_log"]["threads"] is not None for r in runs))
    pairs = []
    for start in range(0, len(runs) - 1, 2):
        first, second = runs[start], runs[start + 1]
        a, b = (first, second) if first["layout"] == "A" else (second, first)
        ratio = (b["peak_job_memory_bytes"] / a["peak_job_memory_bytes"]
                 if a["peak_job_memory_bytes"] and b["peak_job_memory_bytes"] else None)
        pairs.append({"runs": [start, start + 1], "delta_wall_seconds": b["phase_wall_seconds"] - a["phase_wall_seconds"],
                      "peak_ratio": ratio,
                      "b_export_peak": next((p["peak_job_memory_bytes"] for p in b["processes"] if p["name"] == "02_export"), None),
                      "b_stack_peak": next((p["peak_job_memory_bytes"] for p in b["processes"] if p["name"] == "02_stack"), None)})
    deltas = [p["delta_wall_seconds"] for p in pairs]
    median_delta = statistics.median(deltas) if deltas else None
    criterion_iii = {"median_delta_seconds": median_delta, "limit_seconds": limits["max_delta_seconds"],
                     "passed": median_delta is not None and median_delta <= limits["max_delta_seconds"]}
    ratios = [p["peak_ratio"] for p in pairs]
    if profile == "hx":
        criterion_iv = {"rule": f"every pair B/A <= {HX_MAX_PAIR_RATIO}", "ratios": ratios,
                        "passed": bool(ratios) and all(r is not None and r <= HX_MAX_PAIR_RATIO for r in ratios)}
    else:
        median_ratio = statistics.median(ratios) if ratios and None not in ratios else None
        criterion_iv = {"rule": f"median B/A <= {MEDIAN_MAX_RATIO}", "ratios": ratios, "median_ratio": median_ratio,
                        "passed": median_ratio is not None and median_ratio <= MEDIAN_MAX_RATIO}
    return {"pairs": pairs, "i": criterion_i, "ii": criterion_ii, "iii": criterion_iii, "iv": criterion_iv,
            "passed": all(c["passed"] for c in (criterion_i, criterion_ii, criterion_iii, criterion_iv))}


# --------------------------------------------------------------------------- --contract d4 (EFF-D4, KD-2)

def parse_settings(text: str) -> list:
    """'T:B,T:B' -> [(T, B), ...] (positive integers)."""
    pairs = []
    for part in str(text).split(","):
        threads, separator, memory = part.strip().partition(":")
        if not separator or not threads.isdigit() or not memory.isdigit() or int(threads) < 1 or int(memory) < 1:
            raise ValueError(f"settings must look like 'T:B,T:B' with positive integers, not {text!r}")
        pairs.append((int(threads), int(memory)))
    return pairs


def sequence_shape(work: Path) -> dict | None:
    """{'frames': n, 'frame_mb': p} of an exported work/r_light_.seq and its first selected frame (the inputs of the
    product's plan), or None when there is no readable export."""
    try:
        frames, first = stack_settings.selected_frames(work / stack_settings.SEQUENCE_NAME)
        per_frame = stack_settings.frame_mb(*stack_settings.frame_geometry(work / f"{first}.fit"))
    except (OSError, ValueError, KeyError):
        return None
    return {"frames": frames, "frame_mb": per_frame}


def _not_a_plan(threads: int, memory: int, frames: int, per_frame: int) -> str | None:
    """Why (T', B) is not a contract plan for n frames of p MB, or None when it is."""
    try:
        planned = stack_settings.threads(threads, frames, memory, per_frame)
    except ValueError:             # the budget holds less than one frame
        return f"{threads}:{memory} is not a contract plan for n={frames}, p={per_frame} MB (B holds no frame)"
    if planned != threads:
        return f"{threads}:{memory} is not a contract plan for n={frames}, p={per_frame} MB (T' would be {planned})"
    return None


def check_pairs(pairs, shape: dict) -> None:
    """Every (T', B) must be a contract plan for the sequence ``shape``; else ReplayError naming each pair that is not."""
    problems = [problem for problem in (_not_a_plan(t, b, shape["frames"], shape["frame_mb"]) for t, b in pairs)
                if problem]
    if problems:
        raise ReplayError("; ".join(problems) + "; pass --settings / --control pairs with T' = min(T', n, B*1024//p)")


def contract_plan(work: Path, threads: int, memory: int) -> dict:
    """The product's plan for this export with T' and B given; refused unless the pair is a contract plan."""
    frames, first = stack_settings.selected_frames(work / stack_settings.SEQUENCE_NAME)
    per_frame = stack_settings.frame_mb(*stack_settings.frame_geometry(work / f"{first}.fit"))
    problem = _not_a_plan(threads, memory, frames, per_frame)
    if problem:
        raise ReplayError(problem)
    return {"version": stack_settings.VERSION, "script": "02_stack", "frames": frames, "frame_mb": per_frame,
            "threads": threads, "memory_gib": memory, "thread_cap": threads, "thread_cap_source": "replay",
            "memory_source": "replay"}


@contextmanager
def _pin(value):
    previous = os.environ.get(siril.SIRIL_PIN_ENV)
    if value is None:
        os.environ.pop(siril.SIRIL_PIN_ENV, None)
    else:
        os.environ[siril.SIRIL_PIN_ENV] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(siril.SIRIL_PIN_ENV, None)
        else:
            os.environ[siril.SIRIL_PIN_ENV] = previous


def _sequence_without_statistics(path: Path, folder: Path) -> dict:
    """r_light_.seq without its M lines (Siril's statistics cache: its parallel workers can drop an entry) and the
    number of M lines."""
    if not path.exists():
        return {"digest": None, "statistics_lines": None}
    lines = path.read_text(encoding="utf-8", errors="replace").replace(str(folder), "{REPLAY}").splitlines()
    kept = "\n".join(line for line in lines if not line.startswith("M"))
    return {"digest": hashlib.sha256(kept.encode("utf-8")).hexdigest(),
            "statistics_lines": sum(line.startswith("M") for line in lines)}


def run_d4(replay: Path, executable: str, accounting: Accounting, threads: int, memory: int, *,
           control: bool = False) -> dict:
    """One replay run: reset, 02_export, then the contract 02_stack (or the pinned P1b stack for the control)."""
    reset(replay)
    work = replay / "work"
    mark = len(accounting.entries)
    started = time.perf_counter()
    with siril.job_stage_launches(), _pin(None):
        siril.run_script(executable, replay, "02_export", core.export_script(DISTORTION_WCS, own_registration=True),
                         lambda: None)
        plan = contract_plan(work, threads, memory)
        offset = _log_size(replay, "02_stack")
        if control:      # the fixed-parameter control: the P1b script, AESTER_SIRIL_PIN=cpu=T,mem=B
            with _pin(f"cpu={threads},mem={memory}"):
                siril.run_script(executable, replay, "02_stack", list(core.LEGACY_STACK_SCRIPT), lambda: None)
        else:            # as core._contract_stack: settings lines, own configuration, contract environment
            config = work / stack_settings.CONFIG_NAME
            shutil.copyfile(replay / "siril.ini", config)
            try:
                siril.run_script(executable, replay, "02_stack",
                                 stack_settings.with_settings(core.LEGACY_STACK_SCRIPT, plan), lambda: None,
                                 config=config, stack_settings=True)
            finally:
                config.unlink(missing_ok=True)
    wall = time.perf_counter() - started
    found = stack_settings.log_lines(replay / "02_stack.log", offset)
    from astropy.io import fits
    stackcnt = fits.getheader(work / "sky_master.fit").get("STACKCNT")
    processes = [{"name": name, "wall_seconds": (entry or {}).get("wall_seconds"),
                  "peak_job_memory_bytes": (entry or {}).get("peak_job_memory_bytes"),
                  "returncode": (entry or {}).get("returncode")}
                 for name, entry in zip(("02_export", "02_stack"), accounting.since(mark) + [None, None])]
    return {"kind": "control" if control else "contract", "settings": [threads, memory], "plan": plan,
            "phase_wall_seconds": wall, "processes": processes, "master": fits_digest(work / "sky_master.fit"),
            "stack_log": found, "stackcnt": stackcnt,
            "own_plan": (found == {"memory": [(plan["frame_mb"], memory * 1024)], "threads": [threads]}
                         and stackcnt == plan["frames"]),
            "r_light_seq": _sequence_without_statistics(work / "r_light_.seq", replay),
            "siril_ini": text_digest(replay / "siril.ini", replay)}


def _ordered(bits):
    import numpy as np
    value = bits.astype(np.int64)
    return np.where(value < 0, np.int64(-(2 ** 31)) - value, value)


def ulp_difference(a: Path, b: Path) -> dict:
    """Values that differ between two float32 FITS masters, the largest difference in ULP, and differences that
    involve a NaN or an infinity."""
    import numpy as np
    from astropy.io import fits
    with fits.open(a, memmap=False, do_not_scale_image_data=True) as ha, \
            fits.open(b, memmap=False, do_not_scale_image_data=True) as hb:
        x, y = np.asarray(ha[0].data), np.asarray(hb[0].data)
        if x.shape != y.shape or x.dtype.kind != "f" or x.dtype.itemsize != 4 or y.dtype != x.dtype:
            return {"comparable": False, "shapes": [list(x.shape), list(y.shape)]}
        xb, yb = x.astype(np.float32).view(np.int32), y.astype(np.float32).view(np.int32)
        differ = xb != yb
        nonfinite = differ & ~(np.isfinite(x) & np.isfinite(y))
        ulp = np.abs(_ordered(xb[differ & ~nonfinite]) - _ordered(yb[differ & ~nonfinite]))
        return {"comparable": True, "values": int(x.size), "differing": int(differ.sum()),
                "max_ulp": int(ulp.max()) if ulp.size else 0, "nonfinite_differences": int(nonfinite.sum())}


def evaluate_d4(runs: list, expected_master, *, reference=None, expect_differing=None) -> dict:
    contract = [r for r in runs if r["kind"] == "contract"]
    data = {json.dumps(r["master"]["data"]) for r in contract}
    if isinstance(expected_master, str):         # --expect-data-digest: a prefix of the master's data digest
        matches = bool(contract) and all(str(r["master"]["data"][0]).startswith(expected_master) for r in contract)
    else:
        matches = None if expected_master is None else data == {json.dumps(expected_master)}
    one = {"masters": len(data), "expected_master_data": expected_master, "matches_expected": matches}
    one["passed"] = len(data) == 1 and one["matches_expected"] is not False
    own = {"failed_runs": [i for i, r in enumerate(runs) if r["kind"] == "contract" and not r["own_plan"]]}
    own["passed"] = bool(contract) and not own["failed_runs"]
    wild = {"limit_ulp": D4_MAX_ULP, "pairs": [r.get("ulp_to_first") for r in contract[1:]]}
    wild["stop_p1c"] = any(p and (not p.get("comparable", True) or p.get("max_ulp", 0) > D4_MAX_ULP
                                  or p.get("nonfinite_differences", 0)) for p in wild["pairs"])
    wild["passed"] = not wild["stop_p1c"]
    controls = [r for r in runs if r["kind"] == "control"]
    control = {"runs": len(controls),
               "equal_to_contract": [json.dumps(r["master"]["data"]) in data for r in controls]}
    control["passed"] = all(control["equal_to_contract"])
    result = {"one_master": one, "own_plan": own, "no_wild_difference": wild, "control": control}
    if reference is not None:
        check = {"comparison": reference, "expect_differing": expect_differing}
        check["passed"] = (expect_differing is None or (reference.get("comparable") is True
                           and reference["differing"] == expect_differing and reference["max_ulp"] <= 1
                           and reference["nonfinite_differences"] == 0))
        result["reference"] = check
    result["passed"] = all(c["passed"] for c in result.values())
    result["stop_p1c"] = wild["stop_p1c"]
    return result


def replay_d4(replay: Path, profile: str, settings: list, control, repeat: int, executable: str, expected_master,
              *, reference_master=None, expect_differing=None) -> dict:
    accounting = Accounting()
    plan = [(pair, False) for pair in settings] + ([(control, True)] if control else [])
    order = [entry for _ in range(repeat) for entry in plan]
    report = {"schema": D4_SCHEMA, "profile": profile, "contract": stack_settings.contract(), "replay": str(replay),
              "siril": executable, "started_at": now(), "settings": [list(p) for p in settings],
              "control": list(control) if control else None, "repeat": repeat,
              "order": [{"settings": list(p), "control": c} for p, c in order], "runs": [], "error": None}
    first_master = None
    settings_profile = {**PROFILES[profile], "pin": None}      # D4 lines replace any pin; only the affinity is used
    with profile_environment(profile, settings_profile) as environment:
        report["environment"] = environment
        try:
            for index, ((threads, memory), is_control) in enumerate(order):
                run = run_d4(replay, executable, accounting, threads, memory, control=is_control)
                master = replay / "work" / "sky_master.fit"
                if not is_control:
                    if first_master is None:
                        first_master = replay / "first_contract_master.fit"
                        shutil.copyfile(master, first_master)
                    else:
                        run["ulp_to_first"] = ulp_difference(first_master, master)
                if reference_master is not None and index == 0:
                    report["reference_comparison"] = ulp_difference(Path(reference_master), master)
                report["runs"].append({"index": index, **run})
                wild = run.get("ulp_to_first") or {}
                if wild and (not wild.get("comparable", True) or wild.get("max_ulp", 0) > D4_MAX_ULP
                             or wild.get("nonfinite_differences", 0)):
                    report["error"] = "a contract master differs by more than 4 ULP or in a NaN/Inf: stop P1c"
                    break
        except Exception as exc:     # a Siril failure (any non-zero exit) stops the replay: never retried
            report["error"] = f"{type(exc).__name__}: {exc}"[:4000]
    report["finished_at"] = now()
    report.update(evaluate_d4(report["runs"], expected_master, reference=report.get("reference_comparison"),
                              expect_differing=expect_differing))
    if report["error"] is not None:
        report["passed"] = False
    return report


# --------------------------------------------------------------------------- profile

@contextmanager
def profile_environment(profile: str, settings: dict | None = None):
    settings = PROFILES[profile] if settings is None else settings
    previous = os.environ.get(siril.SIRIL_PIN_ENV)
    if settings["pin"]:
        os.environ[siril.SIRIL_PIN_ENV] = settings["pin"]
    else:
        os.environ.pop(siril.SIRIL_PIN_ENV, None)
    os.environ.pop(core.L_LAYOUT_ENV, None)
    try:
        yield {"affinity_mask": hex(set_affinity(settings["affinity"])) if settings["affinity"] else None,
               siril.SIRIL_PIN_ENV: settings["pin"]}
    finally:
        if previous is None:
            os.environ.pop(siril.SIRIL_PIN_ENV, None)
        else:
            os.environ[siril.SIRIL_PIN_ENV] = previous


def set_affinity(logical: int) -> int:
    """The first ``logical`` processors of this process's group, inherited by its guarded children (as the bench)."""
    if os.name != "nt":
        raise ReplayError("affinity profiles need Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    mask = (1 << logical) - 1
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
        raise ReplayError(f"SetProcessAffinityMask failed ({ctypes.get_last_error()})")
    return mask


# --------------------------------------------------------------------------- main

def replay_profile(replay: Path, profile: str, blocks: int, executable: str, expected_master) -> dict:
    accounting = Accounting()
    report = {"schema": SCHEMA, "profile": profile, "blocks": blocks, "order": order(blocks), "replay": str(replay),
              "siril": executable, "started_at": now(), "scripts": {"A": script_a(), "B": [list(s) for s in script_b()]},
              "runs": [], "error": None}
    with profile_environment(profile) as environment:
        report["environment"] = environment
        try:
            for index, layout in enumerate(report["order"]):
                run = run_once(replay, layout, executable, accounting, frames_digests=index < 4)
                report["runs"].append({"index": index, "block": index // 4, **run})
        except Exception as exc:     # a Siril failure (any non-zero exit) stops the replay: never retried
            report["error"] = f"{type(exc).__name__}: {exc}"[:4000]
    report["finished_at"] = now()
    if report["error"] is None:
        report.update(evaluate(report["runs"], profile, expected_master))
    else:
        report["passed"] = False
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=Path, required=True, help="the replay folder (prepared from --job if new)")
    parser.add_argument("--job", type=Path, help="a finished, unpinned L job folder (needed to prepare --replay)")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--blocks", type=int, default=3, help="ABBA blocks (default 3 = 6 pairs)")
    parser.add_argument("--siril", type=Path, default=SIRIL)
    parser.add_argument("--expect-manifest", type=Path,
                        help="a whole-workflow manifest whose sky_master_undistorted.fit data digest the masters must "
                             "equal (default: the source job's, recorded at preparation)")
    parser.add_argument("--report", type=Path, help="default: <replay>/replay-<profile>.json")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; prepare and run nothing")
    parser.add_argument("--contract", choices=("none", "d4"), default="none",
                        help="d4: the EFF-D4 stack-settings replay (KD-2) instead of the P1b A/B layouts")
    parser.add_argument("--settings", default=D4_DEFAULT_SETTINGS, help="d4: T:B pairs (default %(default)s)")
    parser.add_argument("--control", default=None,
                        help="d4: the pinned control pair T:B, or 'none' (default: the first --settings pair)")
    parser.add_argument("--repeat", type=int, default=2, help="d4: rounds over the pairs (default 2)")
    parser.add_argument("--expect-data-digest", help="d4: a prefix of the master's data digest every run must have")
    parser.add_argument("--reference-master", type=Path, help="d4: a FITS master to count differing values against")
    parser.add_argument("--expect-differing", type=int,
                        help="d4: the number of values that must differ from --reference-master (each <= 1 ULP)")
    args = parser.parse_args(argv)
    if args.blocks < 1:
        parser.error("--blocks must be at least 1")
    if args.contract == "d4":
        try:
            args.settings = parse_settings(args.settings)
            args.control = (None if args.control == "none" else
                            args.settings[0] if args.control is None else parse_settings(args.control)[0])
        except ValueError as exc:
            parser.error(str(exc))
        if args.repeat < 1:
            parser.error("--repeat must be at least 1")
        if args.expect_differing is not None and args.reference_master is None:
            parser.error("--expect-differing needs --reference-master")
    elif any(value is not None for value in (args.control, args.expect_data_digest, args.reference_master,
                                             args.expect_differing)):
        parser.error("--control, --expect-data-digest, --reference-master and --expect-differing need --contract d4")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    replay = args.replay.resolve()
    d4 = args.contract == "d4"
    if d4:
        plan = {"replay": str(replay), "contract": "d4", "profile": args.profile,
                "affinity": PROFILES[args.profile]["affinity"], "settings": [list(p) for p in args.settings],
                "control": list(args.control) if args.control else None, "repeat": args.repeat,
                "stack_script": stack_settings.with_settings(core.LEGACY_STACK_SCRIPT,
                                                             {"threads": "T'", "memory_gib": "B"}),
                "prepare_from": str(args.job) if args.job else None}
    else:
        plan = {"replay": str(replay), "profile": args.profile, "order": order(args.blocks),
                "settings": PROFILES[args.profile], "scripts": {"A": script_a(), "B": [list(s) for s in script_b()]},
                "prepare_from": str(args.job) if args.job else None}
    if args.dry_run:
        print(json.dumps(plan, indent=1, ensure_ascii=False))
        return 0
    pairs = (args.settings + ([args.control] if args.control else [])) if d4 else []
    try:
        if (replay / PREPARED).exists():
            if args.job is not None:
                prepared = json.loads((replay / PREPARED).read_text(encoding="utf-8"))
                if Path(prepared["source_job"]) != args.job.resolve():
                    raise ReplayError(f"{replay} was prepared from {prepared['source_job']}")
        elif args.job is None:
            raise ReplayError(f"{replay} is not prepared; pass --job")
        else:
            source_shape = sequence_shape(args.job.resolve() / "work") if d4 else None
            if source_shape is not None:      # before the frames are copied
                check_pairs(pairs, source_shape)
            prepare(args.job, replay, contract=args.contract)
        prepared = json.loads((replay / PREPARED).read_text(encoding="utf-8"))
        expected = manifest_master(args.expect_manifest) if args.expect_manifest else prepared.get("source_master")
        if d4 and args.expect_data_digest:
            expected = args.expect_data_digest
        if not Path(args.siril).is_file():
            raise ReplayError(f"missing Siril: {args.siril}")
        if d4 and args.reference_master is not None and not args.reference_master.is_file():
            raise ReplayError(f"missing reference master: {args.reference_master}")
        shape = None
        if d4:            # every pair against n and p before any Siril run (else the first export decides)
            shape = prepared.get("stack_sequence") or sequence_shape(replay / "work")
            if shape is not None:
                check_pairs(pairs, shape)
    except ReplayError as exc:
        print(f"replay_legacy_stack: {exc}", file=sys.stderr)
        return 2
    if d4:
        report = replay_d4(replay, args.profile, args.settings, args.control, args.repeat, str(args.siril), expected,
                           reference_master=args.reference_master, expect_differing=args.expect_differing)
    else:
        report = replay_profile(replay, args.profile, args.blocks, str(args.siril), expected)
    report["prepared"] = prepared
    if d4:
        report["pairs_checked_before_runs"] = shape
    target = args.report or replay / f"replay-{'d4-' if d4 else ''}{args.profile}.json"
    target.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    summary = {key: report.get(key) for key in ("profile", "passed", "error")}
    for key in ("i", "ii", "iii", "iv", "one_master", "own_plan", "no_wild_difference", "control", "reference"):
        if key in report:
            summary[key] = report[key]["passed"]
    if d4:
        summary["stop_p1c"] = report.get("stop_p1c")
    summary["report"] = str(target)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
