"""The P1b run check (efficiency P1b prereg section 4, CROSS_CHECK X7/X9): one acceptance bench run against its baseline.

Development and validation tool only.  Reads two bench result folders (plan.json, summary.json, manifest.json,
records/job.json, records/workflow.json) and the verify_workflow_equivalence.py compare report between them
(a = the baseline, b = the run), and checks, each section reported separately:

  plan     parity with the baseline: mode, frames, slot, profile, simulation profile, the baseline's pins,
           submit.backend_options (ONNX Runtime threads included), submit.stack_range_mode, and the effective
           knob environment (the inherited AESTER_* / OMP_* / OPENSSL_* ... values with the plan's settings
           applied: OPENSSL_ia32cap, AESTER_SIRIL_PIN, AESTER_RENDER_WORKERS, every P1b knob, MSYSTEM).  A value
           that is meant to differ from the baseline is named with --expect-env KEY=VALUE ("unset" = absent);
           a pins key the baseline does not have must be named with --expect-pin KEY=JSON.  Slot paths are not
           compared; MSYSTEM / XDG_DATA_HOME are skipped when either run used --siril-desktop (X8).
  compare  the report passed, its sides are these two manifests (a = baseline), and the allowed set is exactly the
           mode's (X7): L (default, against a pre-P1b baseline) the two Siril scripts of the L split,
           {WF}/stack/job-<job#1>/02_export.ssf only in b and {WF}/stack/job-<job#1>/02_stack.ssf different;
           F and R nothing.  only_in_listed is exactly {WF}/stack/job-<job#1>/02_export.log in b for that L case,
           nothing otherwise, plus --extra-listed SIDE:NAME.
  scripts  (L) the stack-phase script texts (CRLF read as LF): the baseline's 02_stack.ssf is the pre-P1b single
           script of path 3a, the run's 02_export.ssf and 02_stack.ssf are the split layout (pin lines included).
  notes    (prereg 4.3) L and R: the job's frame_pool.coverage note with --expect-coverage's workers/serial/reason;
           F: no such note; every run: the hash_ledger notes carry miss_bytes and largest_misses and version
           --hash-version, a governor note exists in the job and workflow receipts iff AESTER_GOVERNOR is not 'off',
           no l_siril_layout note, and no frame_pool / governor / hash_ledger key outside timings blocks.
  siril    the run and its job succeeded in one attempt (any non-zero Siril exit fails the job; nothing retries a
           launch), and every Siril operation of the job had at most 2 processes per launch (descendant_processes,
           siril-cli and its console host; a gdbus.exe would be the third).

Exit code 0 when every section passes, 1 otherwise, 2 for usage errors.  --json writes the full result.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starstack import core, siril  # noqa: E402

SCHEMA = "aester-p1b-run-check-1"
RELEVANT_ENV = re.compile(r"^(AESTER_|STARSTACK_|OMP_|OPENBLAS_|MKL_|OPENSSL_|ORT_|PYTHON|MSYSTEM$|XDG_DATA_HOME$)",
                          re.IGNORECASE)   # the bench's own filter (bench_workflow.RELEVANT_ENV)
PATH_KEYS = {"AESTER_CACHE_ROOT", "AESTER_COMPONENT_ROOT", "XDG_DATA_HOME", "PYTHONPATH", "AESTER_BENCH_HOOK_LOG",
             "AESTER_BENCH_PIN_SHIM"}
DESKTOP_KEYS = {"MSYSTEM", "XDG_DATA_HOME"}
# Bench harness hygiene, not compared (P1b review SIRIL-2): the bench always disables Siril's session bus, which never
# changes an output; runs before that change (the P1a baselines) have no value, and new code sets it for Siril itself.
HYGIENE_KEYS = {"DBUS_SESSION_BUS_ADDRESS"}
# Knobs whose absent or empty value means a documented default: compared after normalising (the P1a baselines
# ran with AESTER_HASH_LEDGER=on inherited from the shell, the S-4b runs without it: the same setting).
KNOB_DEFAULTS = {"AESTER_HASH_LEDGER": "on",          # hashing.ledger_mode
                 "AESTER_L_SIRIL_LAYOUT": "split",     # core.legacy_stack_layout
                 "AESTER_GOVERNOR": "shadow",          # P1b prereg 8 (U-G1): the product default
                 "AESTER_FRAME_WORKERS": "auto"}       # P1b-1 frame pool default
EXPORT_SSF, STACK_SSF = "{WF}/stack/job-<job#1>/02_export.ssf", "{WF}/stack/job-<job#1>/02_stack.ssf"
EXPORT_LOG = "{WF}/stack/job-<job#1>/02_export.log"
L_ALLOW_PATTERNS = ["*/stack/job-*/02_export.ssf", "*/stack/job-*/02_stack.ssf"]
OUTSIDE_TIMINGS = ("frame_pool", "governor", "hash_ledger")
TIMING_KEYS = ("timings", "timing_attempts")


class UsageError(Exception):
    pass


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(folder: Path) -> dict:
    folder = Path(folder)
    missing = [name for name in ("plan.json", "summary.json", "manifest.json", "records/job.json")
               if not (folder / name).is_file()]
    if missing:
        raise UsageError(f"{folder}: missing {', '.join(missing)}")
    run = {"folder": str(folder), "plan": _read(folder / "plan.json"), "summary": _read(folder / "summary.json"),
           "manifest": _read(folder / "manifest.json"), "job": _read(folder / "records" / "job.json")}
    workflow = folder / "records" / "workflow.json"
    run["workflow"] = _read(workflow) if workflow.is_file() else None
    return run


# --------------------------------------------------------------------------- plan parity

def effective_environment(plan: dict, summary: dict) -> dict:
    """What the run's processes saw: the inherited relevant variables with the plan's settings applied."""
    env = {k: v for k, v in (summary.get("inherited_environment") or {}).items() if RELEVANT_ENV.match(k)}
    for key, value in (plan.get("environment") or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def knob_value(key: str, value):
    """A knob's effective value: absent or empty means its default (KNOB_DEFAULTS); other values unchanged."""
    if key in KNOB_DEFAULTS:
        return (value or "").strip().lower() or KNOB_DEFAULTS[key]
    return value


def _desktop(plan: dict) -> bool:
    return (plan.get("siril_launch") or {}).get("mode") == "desktop-sentinel"


def check_plan(run: dict, base: dict, expect_env: dict, expect_pin: dict) -> dict:
    a, b = base["plan"], run["plan"]
    problems = []
    for key in ("mode", "frames", "slot", "profile"):
        if a.get(key) != b.get(key):
            problems.append(f"{key}: baseline {a.get(key)!r}, run {b.get(key)!r}")
    if (a.get("simulation") or {}).get("profile") != (b.get("simulation") or {}).get("profile"):
        problems.append("simulation.profile differs")
    for key in ("backend_options", "stack_range_mode"):
        if (a.get("submit") or {}).get(key) != (b.get("submit") or {}).get(key):
            problems.append(f"submit.{key}: baseline {(a.get('submit') or {}).get(key)!r}, "
                            f"run {(b.get('submit') or {}).get(key)!r}")
    pins_a, pins_b = a.get("pins") or {}, b.get("pins") or {}
    for key in sorted(set(pins_a) | set(pins_b)):
        if key in expect_pin:
            if pins_b.get(key) != expect_pin[key]:
                problems.append(f"pins.{key}: run {pins_b.get(key)!r}, expected {expect_pin[key]!r}")
        elif key not in pins_a:
            problems.append(f"pins.{key} is new in the run ({pins_b.get(key)!r}): name it with --expect-pin")
        elif pins_a.get(key) != pins_b.get(key):
            problems.append(f"pins.{key}: baseline {pins_a.get(key)!r}, run {pins_b.get(key)!r}")
    env_a, env_b = effective_environment(a, base["summary"]), effective_environment(b, run["summary"])
    skipped = set(PATH_KEYS) | HYGIENE_KEYS | (DESKTOP_KEYS if _desktop(a) or _desktop(b) else set())
    compared = {}
    for key in sorted((set(env_a) | set(env_b) | set(expect_env)) - skipped):
        expected = knob_value(key, expect_env[key] if key in expect_env else env_a.get(key))
        actual = knob_value(key, env_b.get(key))
        compared[key] = {"baseline": env_a.get(key), "run": env_b.get(key), "expected": expected}
        if actual != expected:
            problems.append(f"environment {key}: run {env_b.get(key)!r}, expected {expected!r}"
                            + ("" if key in expect_env else " (the baseline's)"))
    return {"passed": not problems, "problems": problems, "environment": compared,
            "skipped_environment": sorted(skipped)}


# --------------------------------------------------------------------------- compare report

def check_compare(report: dict, run: dict, base: dict, mode: str, layout: str, extra_listed: list) -> dict:
    problems = []
    if not report.get("pass"):
        problems.append("the compare report did not pass")
    for side, source in (("a", base), ("b", run)):
        if (report.get(side) or {}).get("created_utc") != source["manifest"].get("created_utc"):
            problems.append(f"report side {side} is not {source['folder']}'s manifest (a = baseline, b = run)")
    if mode == "L" and layout == "combined":
        expected_allowed = {EXPORT_SSF: "ALLOWED_ONLY_IN_B", STACK_SSF: "ALLOWED_DIFFERENT"}
        expected_patterns = sorted(L_ALLOW_PATTERNS)
        listed = {"a": [], "b": [EXPORT_LOG]}
    else:
        expected_allowed, expected_patterns, listed = {}, [], {"a": [], "b": []}
    for side, name in extra_listed:
        listed[side].append(name)
    allowed = {name: entry.get("verdict") for name, entry in (report.get("allowed") or {}).items()}
    if allowed != expected_allowed:
        problems.append(f"allowed {allowed} != {expected_allowed}")
    if sorted(report.get("allow_patterns") or []) != expected_patterns:
        problems.append(f"allow patterns {report.get('allow_patterns')} != {expected_patterns}")
    if report.get("allow_fields") or report.get("tolerances"):
        problems.append("the comparison used --allow-field or a tolerance")
    actual_listed = {side: sorted((report.get("only_in_listed") or {}).get(side) or []) for side in ("a", "b")}
    if actual_listed != {side: sorted(names) for side, names in listed.items()}:
        problems.append(f"only_in_listed {actual_listed} != {listed}")
    return {"passed": not problems, "problems": problems, "allowed": allowed, "only_in_listed": actual_listed}


# --------------------------------------------------------------------------- scripts (L)

def manifest_text(manifest: dict, canonical: str) -> str | None:
    pattern = canonical.replace("<job#1>", "*")
    found = [entry for key, entry in manifest["files"].items() if fnmatch.fnmatchcase(key.replace("\\", "/"), pattern)]
    if len(found) != 1:
        return None
    return ((found[0].get("content") or {}).get("text") or "").replace("\r\n", "\n")


def check_scripts(run: dict, base: dict, layout: str) -> dict:
    pin_a, pin_b = (base["plan"].get("pins") or {}).get("siril"), (run["plan"].get("pins") or {}).get("siril")
    combined = core.stack_script(core.DISTORTION_WCS, own_registration=True)
    split = (core.export_script(core.DISTORTION_WCS, own_registration=True), list(core.LEGACY_STACK_SCRIPT))
    expected = {("b", EXPORT_SSF): siril.script_text(split[0], pin_b), ("b", STACK_SSF): siril.script_text(split[1], pin_b)}
    if layout == "combined":
        expected[("a", STACK_SSF)] = siril.script_text(combined, pin_a)
        expected[("a", EXPORT_SSF)] = None
    else:
        expected[("a", EXPORT_SSF)] = siril.script_text(split[0], pin_a)
        expected[("a", STACK_SSF)] = siril.script_text(split[1], pin_a)
    problems = []
    for (side, name), text in expected.items():
        actual = manifest_text((base if side == "a" else run)["manifest"], name)
        if actual != text:
            problems.append(f"{side}: {name} is {actual!r}, expected {text!r}")
    return {"passed": not problems, "problems": problems}


# --------------------------------------------------------------------------- notes

def keys_outside_timings(value, names=OUTSIDE_TIMINGS, path="") -> list:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TIMING_KEYS:
                continue
            if any(key == n or key.startswith(n + ".") for n in names):
                found.append(f"{path}/{key}")
            found.extend(keys_outside_timings(item, names, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(keys_outside_timings(item, names, f"{path}[{index}]"))
    return found


def check_notes(run: dict, mode: str, coverage: dict | None, hash_version: str) -> dict:
    problems = []
    job_notes = (((run["job"].get("diagnostics") or {}).get("timings") or {}).get("notes") or {})
    workflow = run["workflow"]
    workflow_notes = (((workflow or {}).get("timings") or {}).get("notes") or {}) if workflow else None
    receipts = {"job": job_notes, **({"workflow": workflow_notes} if workflow is not None else {})}
    note = job_notes.get("frame_pool.coverage")
    if mode in ("L", "R"):
        if coverage is None:
            raise UsageError("L and R runs need --expect-coverage workers=N,serial=BOOL[,reason=TEXT]")
        if not isinstance(note, dict):
            problems.append("job note frame_pool.coverage is missing")
        else:
            for key, value in coverage.items():
                if note.get(key) != value:
                    problems.append(f"frame_pool.coverage {key}: {note.get(key)!r}, expected {value!r}")
    elif note is not None:
        problems.append("an F run has a frame_pool.coverage note (F uses the support-map path)")
    env = effective_environment(run["plan"], run["summary"])
    governor_mode = knob_value("AESTER_GOVERNOR", env.get("AESTER_GOVERNOR"))     # default shadow (prereg 8, U-G1)
    for label, notes in receipts.items():
        ledger = notes.get("hash_ledger")
        if not isinstance(ledger, dict):
            problems.append(f"{label}: hash_ledger note missing")
        else:
            for key in ("miss_bytes", "largest_misses"):
                if key not in ledger:
                    problems.append(f"{label}: hash_ledger has no {key}")
            if ledger.get("version") != hash_version:
                problems.append(f"{label}: hash_ledger version {ledger.get('version')!r}, expected {hash_version!r}")
        if ("governor" in notes) != (governor_mode != "off"):
            problems.append(f"{label}: governor note {'present' if 'governor' in notes else 'absent'} with "
                            f"AESTER_GOVERNOR={governor_mode}")
        if "l_siril_layout" in notes:
            problems.append(f"{label}: l_siril_layout note (a non-default L layout)")
    for label, record in (("job.json", run["job"]), ("workflow.json", workflow)):
        if record is not None:
            for path in keys_outside_timings(record):
                problems.append(f"{label}: {path} outside the timing receipts")
    return {"passed": not problems, "problems": problems, "governor_mode": governor_mode,
            "frame_pool.coverage": note}


# --------------------------------------------------------------------------- Siril

def check_siril(run: dict) -> dict:
    problems = []
    job = run["job"]
    if run["summary"].get("state") != "succeeded":
        problems.append(f"bench state {run['summary'].get('state')!r}")
    if job.get("state") != "succeeded":
        problems.append(f"job state {job.get('state')!r}: {job.get('error')}")
    diagnostics = job.get("diagnostics") or {}
    if diagnostics.get("timing_attempts"):
        problems.append(f"the job ran {len(diagnostics['timing_attempts']) + 1} attempts")
    if run["workflow"] is not None and (run["workflow"].get("timing_attempts") or []):
        problems.append("the workflow ran more than one attempt")
    operations = ((diagnostics.get("timings") or {}).get("resources") or {}).get("operations") or {}
    launches = {}
    for name, operation in sorted(operations.items()):
        if not name.startswith("script:"):
            continue
        children = operation.get("children") or {}
        processes, descendants = children.get("processes") or 0, children.get("descendant_processes")
        launches[name] = {"processes": processes, "descendant_processes": descendants}
        if not processes or descendants is None:
            problems.append(f"{name}: no child accounting")
        elif descendants > 2 * processes:
            problems.append(f"{name}: {descendants} processes for {processes} Siril launch(es) (at most 2 each)")
    if not launches:
        problems.append("the job receipt has no Siril operations")
    return {"passed": not problems, "problems": problems, "launches": launches}


# --------------------------------------------------------------------------- main

def _pairs(items, what):
    result = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise UsageError(f"{what} must look like KEY=VALUE: {item!r}")
        result[key] = value
    return result


def _coverage(text):
    if text is None:
        return None
    values = _pairs(text.split(","), "--expect-coverage")
    result = {}
    for key, value in values.items():
        if key == "workers":
            result[key] = int(value)
        elif key == "serial":
            if value not in ("true", "false"):
                raise UsageError("--expect-coverage serial must be true or false")
            result[key] = value == "true"
        elif key == "reason":
            result[key] = None if value == "none" else value
        else:
            raise UsageError(f"--expect-coverage knows workers, serial and reason, not {key!r}")
    return result


def check(run_dir, baseline_dir, compare_path, *, expect_env=(), expect_pin=(), coverage=None,
          hash_version="aester-hashing-3", extra_listed=(), baseline_layout="combined") -> dict:
    run, base = load_run(run_dir), load_run(baseline_dir)
    report = _read(compare_path)
    env = {k: (None if v == "unset" else v) for k, v in _pairs(expect_env, "--expect-env").items()}
    pins = {k: json.loads(v) for k, v in _pairs(expect_pin, "--expect-pin").items()}
    listed = []
    for item in extra_listed:
        side, separator, name = item.partition(":")
        if not separator or side not in ("a", "b"):
            raise UsageError(f"--extra-listed must look like b:NAME, not {item!r}")
        listed.append((side, name))
    mode = run["plan"].get("mode")
    if mode not in ("L", "F", "R"):
        raise UsageError(f"unknown run mode {mode!r}")
    sections = {"plan": check_plan(run, base, env, pins),
                "compare": check_compare(report, run, base, mode, baseline_layout, listed),
                "notes": check_notes(run, mode, _coverage(coverage), hash_version),
                "siril": check_siril(run)}
    if mode == "L":
        sections["scripts"] = check_scripts(run, base, baseline_layout)
    return {"schema": SCHEMA, "run": run["folder"], "baseline": base["folder"], "compare": str(compare_path),
            "mode": mode, "baseline_layout": baseline_layout, **sections,
            "passed": all(section["passed"] for section in sections.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the P1b bench result folder")
    parser.add_argument("--baseline", type=Path, required=True, help="the baseline bench result folder")
    parser.add_argument("--compare", type=Path, required=True, help="compare report, a = baseline, b = run")
    parser.add_argument("--expect-env", action="append", default=[], metavar="KEY=VALUE",
                        help="a knob value meant to differ from the baseline ('unset' = absent)")
    parser.add_argument("--expect-pin", action="append", default=[], metavar="KEY=JSON",
                        help="a plan pins value the baseline does not have, e.g. frame_workers='\"auto\"'")
    parser.add_argument("--expect-coverage", metavar="workers=N,serial=BOOL[,reason=TEXT]",
                        help="the job's frame_pool.coverage note (required for L and R)")
    parser.add_argument("--hash-version", default="aester-hashing-3")
    parser.add_argument("--extra-listed", action="append", default=[], metavar="SIDE:NAME",
                        help="another listed-only file a P1b item adds (named by its own prereg)")
    parser.add_argument("--baseline-layout", choices=("combined", "split"), default="combined",
                        help="the baseline's L layout: combined (pre-P1b, default) or split (a P1b baseline)")
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args(argv)
    try:
        result = check(args.run, args.baseline, args.compare, expect_env=args.expect_env,
                       expect_pin=args.expect_pin, coverage=args.expect_coverage, hash_version=args.hash_version,
                       extra_listed=args.extra_listed, baseline_layout=args.baseline_layout)
    except UsageError as exc:
        print(f"check_p1b_run: {exc}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    sections = [name for name in ("plan", "compare", "scripts", "notes", "siril") if name in result]
    print(json.dumps({"passed": result["passed"], **{name: result[name]["passed"] for name in sections},
                      "problems": {name: result[name]["problems"] for name in sections if result[name]["problems"]}},
                     ensure_ascii=False, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
