"""Siril footprint sweep for the resource governor (efficiency P1b-2, bench-only; run in P2 preparation).

Measures Siril's peak job commit and its direct child's peak working set per launch over a grid of thread counts and
memory allowances, so P2 can recalibrate ``starstack.resources`` (governor final.md §2.6, cross-check M10/X12):

  t   in --threads (default 2,4,8,16): Siril worker threads;
  mem in --mem (default none,4,8): Siril's memory allowance in GiB ('none' = Siril's own 0.9 x free RAM);
  route A (default): this tool's own process affinity is set to a spread mask of t logical processors (one per
        physical core first, highest efficiency class first: resources.affinity_mask); guarded Siril children
        inherit it. A memory allowance adds the pin's script lines (AESTER_SIRIL_PIN mem=);
  route B: AESTER_SIRIL_PIN cpu=t[,mem=m] (setcpu / set core.mem_* script lines, and the pin's pool variable).
  pool: Siril's OpenMP/OpenBLAS pool is never set by this tool. siril.siril_environment builds every Siril run's
        environment block and sets the pool itself (U-G3; a cpu pin sets its own), so a pool exported here would not
        reach Siril. Each configuration and each point records the pool of the block Siril actually receives
        (resources.siril_pool_setting: its OMP_NUM_THREADS, else the LPs of Siril's group; this Siril's OpenBLAS is an
        OpenMP build, so OPENBLAS_NUM_THREADS does not size it and is only recorded) and the prediction uses that
        pool. --omp is kept for old command lines and accepts only 1.
  stages (--stages, default convert,probe,export,stack): the product's commands (core.PROBE_SCRIPT, the legacy
        export 'seqapplyreg light -filter-included', and the stack-only launch 'seqclean r_light -stat' + the
        legacy stack command), run in that order in one folder per configuration.

Siril is started ONLY through starstack.siril.run_script (X12): its own environment block, guard, timeouts and log.
Each launch's Job Object accounting (peak job commit, direct-child peak working set / commit) is collected through
process_guard.add_accounting_listener. Results go to --out (JSON, one point per launch) with the governor's
prediction for the same (t, p, frames, allowance) next to each measurement.

Disk: a configuration holds the converted frames and, from 'export' on, the registered frames: about
frames x 2 x 12 B/px (48 Z8 frames: about 53 GB) until it is deleted after its last stage (--keep keeps the last).
The inputs are verified against the development baseline manifest and hard-linked from one copy in the sweep root.
Never writes to C:\\Aester or %LOCALAPPDATA%\\Aester; refuses links in the sweep root.

    PY scripts\\bench_footprints.py --dry-run --frames 48
    PY scripts\\bench_footprints.py --frames 48 --threads 4,8 --mem none,4 --out outputs\\footprint-sweep\\t48.json
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import itertools
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
_spec = importlib.util.spec_from_file_location("bench_workflow_for_footprints", SCRIPTS / "bench_workflow.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

SCHEMA = "aester-footprint-sweep-1"
DEFAULT_ROOT = REPO / "outputs" / "footprint-sweep"
MARKER = ".aester-footprint-sweep.json"
STAGES = ("convert", "probe", "export", "stack")
BYTES_PER_PIXEL_FRAME = 12          # float32 RGB, converted and registered frames alike
POOL_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")   # recorded per point; resources.SIRIL_POOL_ENV decides
DEFAULT_FRAME_PIXELS = 5520 * 8280  # the Z8 development baseline


class SweepError(RuntimeError):
    pass


def parse_list(text, kind):
    items = [part.strip().lower() for part in str(text).split(",") if part.strip()]
    if not items:
        raise SweepError(f"empty --{kind}")
    out = []
    for item in items:
        if kind == "mem" and item == "none":
            out.append(None)
            continue
        if not item.isdigit() or not 1 <= int(item) <= (64 if kind == "threads" else 1024):
            raise SweepError(f"bad --{kind} item {item!r}")
        out.append(int(item))
    return out


def _starstack():
    """This checkout's starstack (bench_workflow._import_starstack: its src on the path, another copy refused)."""
    return bench._import_starstack(REPO / "src")


def stage_commands():
    """The product's commands for each stage (imported, so the sweep follows the product)."""
    _starstack()
    from starstack import core
    return {"convert": ["setext fit", "set32bits", "setcompress 0", "cd stage",
                        "convertraw light -debayer -out=../work"],
            "probe": list(core.PROBE_SCRIPT),
            "export": ["cd work", "seqapplyreg light -filter-included"],
            "stack": ["cd work", "seqclean r_light -stat", core.STACK_COMMAND]}


def plan_sweep(args) -> dict:
    root = bench.ensure_allowed(args.root, "sweep root")
    bench._no_links_on_path(root, "sweep root")
    threads, mems = parse_list(args.threads, "threads"), parse_list(args.mem, "mem")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages or any(s not in STAGES for s in stages) or stages != [s for s in STAGES if s in stages]:
        raise SweepError(f"--stages must be an ordered subset of {','.join(STAGES)}")
    if args.route not in ("A", "B"):
        raise SweepError("--route must be A or B")
    if args.omp != 1:
        raise SweepError("--omp: only 1. Siril's OpenMP/OpenBLAS pool comes from siril.siril_environment (U-G3) or, "
                         "with --route B, from the cpu pin; a value set here would never reach Siril")
    configs = []
    for t, m in itertools.product(threads, mems):
        pin = (f"cpu={t}" + (f",mem={m}" if m else "")) if args.route == "B" else (f"mem={m}" if m else None)
        config = {"id": f"{args.route}-t{t}-m{m or 'none'}", "route": args.route, "threads": t, "mem_gib": m,
                  "pin": pin}
        config.update(siril_pool(root / f"cfg-{config['id']}", config["pin"]))
        configs.append(config)
    pixels = args.frame_pixels or DEFAULT_FRAME_PIXELS
    per_config = args.frames * BYTES_PER_PIXEL_FRAME * pixels * (2 if {"export", "stack"} & set(stages) else 1)
    return {"schema": SCHEMA, "root": str(root), "frames": args.frames, "frame_pixels": pixels,
            "stages": stages, "configs": configs, "siril": str(args.siril),
            "inputs": {"manifest": str(bench._abs(args.manifest)), "inputs_dir": str(bench._abs(args.inputs_dir))},
            "disk_per_config_bytes": per_config, "launches": len(configs) * len(stages),
            "out": str(bench._abs(args.out)) if args.out else str(root / "sweep.json"), "keep": bool(args.keep)}


def siril_pool(folder, pin_text, environment=None) -> dict:
    """{'pool', 'pool_source', 'pool_environment'}: the OpenMP/OpenBLAS pool of the environment block a Siril run in
    ``folder`` with this pin receives (siril.siril_environment, read-only), or of ``environment`` when given."""
    _starstack()
    from starstack import resources, siril
    if environment is None:
        environment = siril.siril_environment(folder, siril.siril_pin(pin_text))
    pool, source = resources.siril_pool_setting(environment)
    if pool is None:
        pool, source = resources.cpu_topology().siril_group_lps, "group"
    return {"pool": pool, "pool_source": source,
            "pool_environment": {name: environment.get(name) for name in POOL_VARIABLES}}


def _marked(root: Path) -> bool:
    try:
        return json.loads((root / MARKER).read_text(encoding="utf-8")).get("schema") == SCHEMA
    except (OSError, ValueError):
        return False


def remove_config(folder: Path, root: Path) -> None:
    """Delete one configuration folder: a direct child of the marked sweep root, containing no links."""
    folder, root = bench._abs(folder), bench._abs(root)
    if os.path.normcase(str(folder.parent)) != os.path.normcase(str(root)) or not folder.name.startswith("cfg-"):
        raise SweepError(f"refusing to delete {folder}: not a configuration folder of {root}")
    if not _marked(root):
        raise SweepError(f"refusing to delete inside {root}: no sweep marker")
    for directory, dirnames, filenames in os.walk(folder):
        for name in dirnames + filenames:
            if bench._is_link(Path(directory) / name):
                raise SweepError(f"refusing to delete {folder}: contains a link {Path(directory) / name}")

    def writable(function, path, _error):
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(folder, onexc=writable)


class _Affinity:
    """Route A: this process's affinity (children inherit it); restored on exit. Windows only."""

    def __init__(self):
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.GetCurrentProcess.restype = ctypes.c_void_p
        self.k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                    ctypes.POINTER(ctypes.c_size_t)]
        self.k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        self.k32.GetProcessAffinityMask(self.k32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system))
        self.original = process.value

    def set(self, mask: int) -> None:
        if not mask or not self.k32.SetProcessAffinityMask(self.k32.GetCurrentProcess(), mask):
            raise SweepError(f"SetProcessAffinityMask({hex(mask)}) failed ({ctypes.get_last_error()})")

    def restore(self) -> None:
        if self.original:
            self.k32.SetProcessAffinityMask(self.k32.GetCurrentProcess(), self.original)


def run_sweep(plan, *, run_script=None, link=os.link) -> dict:
    _starstack()
    from starstack import process_guard, resources, siril
    run_script = run_script or siril.run_script
    root = Path(plan["root"])
    inputs = bench.verify_inputs(plan["inputs"]["manifest"], plan["inputs"]["inputs_dir"], plan["frames"])
    if shutil.disk_usage(root.parent if not root.exists() else root).free < plan["disk_per_config_bytes"] * 1.1:
        raise SweepError("not enough free disk for one configuration")
    root.mkdir(parents=True, exist_ok=True)
    if not _marked(root):
        if any(root.iterdir()):
            raise SweepError(f"{root} is not empty and has no sweep marker")
        (root / MARKER).write_text(json.dumps({"schema": SCHEMA}), encoding="utf-8")
    sources = root / "inputs"
    sources.mkdir(exist_ok=True)
    for index, frame in enumerate(inputs["frames"], 1):
        target = sources / f"frame{index:05d}{Path(frame['path']).suffix.lower()}"
        if not target.exists():
            shutil.copyfile(frame["path"], target)
    commands = stage_commands()
    topology = resources.cpu_topology()
    collected = []
    process_guard.add_accounting_listener(collected.append)
    affinity = _Affinity() if os.name == "nt" and plan["configs"] and plan["configs"][0]["route"] == "A" else None
    saved = {name: os.environ.get(name) for name in (siril.SIRIL_PIN_ENV,)}
    points = []
    try:
        for config in plan["configs"]:
            folder = root / f"cfg-{config['id']}"
            if folder.exists():
                remove_config(folder, root)
            (folder / "stage").mkdir(parents=True)
            (folder / "work").mkdir()
            for source in sorted(sources.iterdir()):
                link(source, folder / "stage" / source.name)
            mask = None
            if config["route"] == "A":
                group = (topology.process_affinity or {}).get("group", 0)
                mask = resources.affinity_mask(topology.cores, config["threads"], group=group)
                if affinity is not None:
                    affinity.set(mask)
            if config["pin"]:
                os.environ[siril.SIRIL_PIN_ENV] = config["pin"]
            else:
                os.environ.pop(siril.SIRIL_PIN_ENV, None)
            for stage in plan["stages"]:
                # the block run_script will give Siril (siril._started): its pool, recorded and predicted
                pool = siril_pool(folder, None, siril.siril_environment(folder, siril.siril_pin(),
                                                                        commands=commands[stage]))
                del collected[:]
                started = time.monotonic()
                error = None
                try:
                    run_script(plan["siril"], folder, f"sweep_{stage}", commands[stage], lambda: None)
                except Exception as exc:     # recorded, never retried (P1b no-retry rule)
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                accounting = dict(collected[-1]) if collected else {}
                parts = resources.siril_parts(commands[stage])
                allowance = (config["mem_gib"] * resources.GiB if config["mem_gib"] else
                             int(resources.SIRIL_DEFAULT_MEM_RATIO * (resources.memory_snapshot().available_physical or 0)))
                predicted = resources.siril_demand(parts, threads=config["threads"], pool=pool["pool"],
                                                   frames=plan["frames"], frame_pixels=plan["frame_pixels"],
                                                   allowance=allowance)
                points.append({"config": config["id"], "stage": stage, "parts": list(parts), "mask": hex(mask) if mask else None,
                               **pool,
                               "wall_seconds": round(time.monotonic() - started, 3), "error": error,
                               "returncode": accounting.get("returncode"),
                               "peak_job_commit_bytes": accounting.get("peak_job_memory_bytes"),
                               "child_peak_working_set_bytes": accounting.get("child_peak_working_set_bytes"),
                               "child_peak_commit_bytes": accounting.get("child_peak_commit_bytes"),
                               "descendant_processes": accounting.get("descendant_processes"),
                               "predicted": {"commit": predicted["commit"], "ws": predicted["ws"],
                                             "bound_by": predicted["bound_by"], "allowance": allowance}})
                if error:
                    break
            if not plan["keep"] or config is not plan["configs"][-1]:
                remove_config(folder, root)
    finally:
        if affinity is not None:
            affinity.restore()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            process_guard._ACCOUNTING_LISTENERS.remove(collected.append)
        except (AttributeError, ValueError):
            pass
    result = {**plan, "footprints_version": resources.FOOTPRINTS_VERSION, "points": points,
              "finished_utc": bench.utc_now().isoformat()}
    out = Path(plan["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--threads", default="2,4,8,16")
    parser.add_argument("--mem", default="none,4,8", help="GiB list; 'none' = Siril's own allowance")
    parser.add_argument("--route", default="A", help="A (affinity) or B (setcpu pin)")
    parser.add_argument("--omp", type=int, default=1,
                        help="only 1 (old command lines): Siril's pool comes from siril.siril_environment or a cpu pin")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--frame-pixels", type=int, help="pixels per frame for the predictions (default Z8)")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--inputs-dir", default=str(bench.DEFAULT_INPUTS))
    parser.add_argument("--manifest", default=str(bench.DEFAULT_MANIFEST))
    parser.add_argument("--siril", default=str(bench.SIRIL))
    parser.add_argument("--out", help="result JSON (default <root>/sweep.json)")
    parser.add_argument("--keep", action="store_true", help="keep the last configuration's folder")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; verify inputs; create nothing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = plan_sweep(args)
        if args.dry_run:
            inputs = bench.verify_inputs(plan["inputs"]["manifest"], plan["inputs"]["inputs_dir"], plan["frames"])
            print(json.dumps({"dry_run": True, "plan": plan, "inputs_verified": inputs["verified"]}, indent=1))
            return 0
        if not Path(plan["siril"]).is_file():
            raise SweepError(f"missing Siril: {plan['siril']}")
        result = run_sweep(plan)
        print(json.dumps({"points": len(result["points"]), "out": plan["out"],
                          "errors": sum(1 for p in result["points"] if p["error"])}))
        return 0
    except (SweepError, bench.BenchError) as exc:
        print(f"bench_footprints: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
