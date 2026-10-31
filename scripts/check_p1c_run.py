"""The P1c run check (efficiency P1c prereg section 6, "运行检查"): one acceptance bench run against its K reference.

Development and validation tool only (scripts/check_p1b_run.py stays as it was, so the P1b evidence still reproduces).
``check`` reads two bench result folders (plan.json, summary.json, manifest.json, records/job.json,
records/workflow.json; a --job run of the bench, e.g. the strict L CLI job of K-SL20 and W19-W21, has no workflow
record and its manifest names the job folder {JOB}) and the verify_workflow_equivalence.py compare report between
them (a = the reference, b = the run), and checks, each section reported separately:

  plan      parity with the reference: mode, frames, slot, profile, job kind, simulation profile, the submitted
            parameters (backend_options, stack_range_mode, trails ...; paths aside), every pins value but the
            parallelism ones, and the effective environment (inherited relevant variables with the plan applied; slot
            paths, bench hygiene and the parallelism variables aside).  A value meant to differ is named with
            --expect-env / --expect-pin, the parallel mode with --expect-parallel.  The AESTER_* / STARSTACK_* set is
            the expected one: the bench's own, AESTER_HASH_LEDGER and the --expect-env names.  No fixed parameters
            beyond the profile's (--pin-siril, --pin-ort, single parallelism knobs) unless --allow-pins (the contract
            comparisons KN-2, A1b).  With a memory view, no process may have wrapped the public available_memory /
            available_commit without deducting the code root's in-process reservations (the hook's record says).
  parallel  the parallelism variables equal PARALLEL_TABLE for the run's mode (a profile's own knob under
            'default' as the plan records it).  --parallel serial: every frame-pool note entry is serial with reason
            'workers' (or an --allow-serial-reason), the trail scan used 1 process, O-1 is off.  default / max: every entry with 2 or more units is
            parallel unless its reason is a pre-registered one (budget, units, route; --allow-serial-reason adds);
            max: the trail scan used the bench's explicit number (or every frame).  No frame_pool.native_mask note
            when the support record names the reference route (G-4t).  The overlap.o1 note (O-1) has core's shape
            (``overlap_problems``: schema, eligible, a known reason, an eligible note's plan, headrooms and outcome).
            The per-site maxima go to the result for ``coverage``.
  notes     R19: each frame_pool.<site> note is a list, one entry per call (frame_pool.coverage may keep P1b's
            single dict), each entry with unit_bytes, user and kernel CPU and the longest unit (all entries) and the
            floating-point environment (parallel entries); --note-field NAME=KEY|KEY renames, --optional-note-field
            NAME drops one.  hash_ledger in the job and workflow receipts with the reference's version, a governor
            note iff AESTER_GOVERNOR is not 'off' (the compact form accepted), no l_siril_layout note, and no
            frame_pool / governor / hash_ledger / overlap / siril_stack_settings / nox_threads / pool_stats key
            outside the timing receipts.
  threads   the hook's thread audit: at most 16 frame-pool threads in any process at any 1 Hz sample; the
            BLAS-capable threads and the sampler's per-process OS thread maxima are reported.
  contract  EFF-D4: L and R jobs carry admission_options.stack_settings (siril-stack-settings-1), a
            siril_stack_settings note and exactly one D4 block (setcpu / set core.mem_mode=1 / set
            core.mem_amount) after the pin lines and before the stack command of 02_stack.ssf (R: each range-trial
            stack.ssf); F has none of the three.  EFF-NOX: a run with stars_enabled froze backend_options
            intra_op_threads = 8 in records/workflow.json.  --before-contract d4|nox for code that predates K2/K1
            (A1b, KN-2, X2): d4 then expects none of the three in L/R too; nox skips the request check.
  siril     the run and its job succeeded (a resumed run: one attempt per segment), and every Siril operation of
            the job had at most 2 processes per launch.
  compare   the report passed, its sides are these two manifests (a = reference), with strict ids, no allowance,
            pattern, tolerance or field allowance, and no listed-only file but --extra-listed SIDE:NAME and the
            scan worker ledgers and logs a different trail-scan worker count explains (scan_worker_listing).
            A cancel and resume run (C1-C8, X1, X2, X2b: the second segment, compared with the uninterrupted
            reference) passes --resume-allowance R_RESUME.json, the set frozen from the K calibration of that cancel
            point (``resume-allowance`` below, prereg 6.4): the report may then fail, but only by differences inside
            that set (files and directories on one side, differing files, and for JSON files the difference paths);
            strict-id, binding, derived-digest and preimage failures never pass; every FITS / TIFF / JPEG / PNG / NPY
            present on both sides has the same data digests; and the final tree has no .partial-* entry and nothing
            under work/transients (O-1's marker work/transients.o1-pending included) that the uninterrupted reference
            lacks -- a trails-on job's own scan output, which the reference has too, is not left over, nor a scan
            run's worker ledgers and logs (their count follows the workers), nor, when the cancelled attempt did not
            run O-1 (no eligible overlap.o1 note in job.json's timing_attempts), the residue K's serial scan left at
            the same cancel point (the calibration's k_leftovers, by scan-runs folder).
  cancel    with --resume-allowance only (criteria 1-3 of prereg 6.4): one resume (two segments), the first
            segment's cancel valid, to 'cancelled' within max(3 s, K's latency + 1 s) on this host or 5 s on the
            6900HX proxy (--cancel-limit-s for a cancel in O-1's overlap window), no live descendant 5 s after
            'cancelled'; and the allowance belongs to the run (allowance_binding: --cancel-point, the profile,
            the trigger text unless --cancel-when-changed, the reference unless --allowance-reference-differs).
            --cross-version (X1, X2, X2b: the first segment ran other code) reports the latencies without judging them.
  bytes     what the comparison normalises away: each pair of JSON files (by canonical name) has the same key
            order at every level outside the timing receipts, and each pair of TIFF pages the same strip / tile
            layout (offsets and byte counts by digest, rows per strip).
  info      never fails: the foreign CPU, the governor's shadow recommendation against each pool's actual workers,
            the hook's memory views and OpenBLAS configurations.

``resume-allowance --run K_RESUMED --reference K --compare REPORT --cancel-point C1 --out R_RESUME.json`` freezes the
R_resume set of one cancel point from K (prereg 6.4: calibrate all eight before any P1c cancel run): every difference
of the report (as above), K's own cancel latencies and its cancel spec.  It refuses a calibration that criterion 4
would refuse anyway (a strict-id, binding, derived-digest or data-digest difference, a .partial-* leftover, an
allowance); K's serial scan cancelled while it ran (C7 on K) leaves its scan output under work/transients, which is
recorded as ``k_leftovers`` (the residue O-1's marker protocol removes; a P1c run may show only that same residue,
and only when its cancelled attempt did not run O-1).

``kd3 --run K --reference P1B --compare REPORT`` (prereg 4.2 KD-3, the report made with --no-strict-ids): the only
deterministic-id difference is the stack job's signature; the JSON differences of job.json (and of its copies in
workflow.json) are the contract field stack_settings, admission_options.stack_settings among them; the text
differences are in 02_stack.ssf (only its D4 and pin lines) and the Siril configuration files siril*.ini (a new
work/siril-stack.ini included); any other Siril script may differ only by pin lines (a pinned P1B-L20); nothing else
differs, data digests included.

``coverage RESULT.json ...`` reads the --json results of passed checks and requires, per site (SITES; --site-max
SITE=N for a measured cap such as strict_support), stack mode (L, F, R, and SL for the strict L job runs) and parallel
mode (default, max), that at least one accepted run reached the planned maximum (default: the site's cap; max: 16
frame-pool threads, the trail scan's explicit number), bounded by the call's units.  A site that never ran is a
problem unless the table marks it conditional.  Every expected combination (EXPECTED_COMBOS, the W table; or each
--expect-combo MODE:PARALLEL) must have an accepted run.

Exit code 0 when every section passes, 1 otherwise, 2 for usage errors.  --json writes the full result.
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

SCHEMA = "aester-p1c-run-check-1"
COVERAGE_SCHEMA = "aester-p1c-coverage-1"
RESUME_SCHEMA = "aester-p1c-resume-allowance-1"
KD3_SCHEMA = "aester-p1c-kd3-1"
RELEVANT_ENV = re.compile(r"^(AESTER_|STARSTACK_|OMP_|OPENBLAS_|MKL_|OPENSSL_|ORT_|PYTHON|MSYSTEM$|XDG_DATA_HOME$)",
                          re.IGNORECASE)   # the bench's own filter (bench_workflow.RELEVANT_ENV)
# Slot paths (the same bench slot gives the same values, but a reference may come from another bench root) and the
# bench's own bookkeeping; never parity.
PATH_KEYS = {"AESTER_CACHE_ROOT", "AESTER_COMPONENT_ROOT", "XDG_DATA_HOME", "PYTHONPATH", "AESTER_BENCH_HOOK_LOG",
             "AESTER_BENCH_PIN_SHIM", "AESTER_BENCH_HOOK_EVENTS", "AESTER_BENCH_THREAD_LOG"}
HYGIENE_KEYS = {"DBUS_SESSION_BUS_ADDRESS", "AESTER_BENCH_REEXEC", "AESTER_BENCH_BLAS_LOG"}   # bench bookkeeping
DESKTOP_KEYS = {"MSYSTEM", "XDG_DATA_HOME"}
# Knobs whose absent or empty value means a documented default (compared after normalising).
KNOB_DEFAULTS = {"AESTER_HASH_LEDGER": "on", "AESTER_L_SIRIL_LAYOUT": "split", "AESTER_GOVERNOR": "shadow",
                 "AESTER_FRAME_WORKERS": "auto", "AESTER_RENDER_WORKERS": "auto", "AESTER_OVERLAP": "auto"}
INHERITED_KNOBS = {"AESTER_HASH_LEDGER"}          # set from the shell for W5/W6/W7 (shadow ledger); not the bench's
SURVEILLED = ("AESTER_", "STARSTACK_")
PARALLEL_VARIABLES = ("AESTER_FRAME_WORKERS", "STARSTACK_TRANSIENT_WORKERS", "AESTER_RENDER_WORKERS",
                      "AESTER_HASH_WORKERS", "AESTER_OVERLAP")
PARALLEL_PINS = ("parallel", "overlap")
# Pins a P1b-era bench did not record, and the value its runs had (A1b compares a new-bench run with P1B-L20).
PIN_DEFAULTS = {"trails": "on", "memory_view_gib": None, "commit_view_gib": None, "python_openblas": None}
SUBMIT_PATHS = ("siril_path", "star_model_path", "stack_engine_path", "regions_path", "delivery_root", "sky_mask_path")
ALLOWED_SERIAL_REASONS = ("budget", "units", "route")
# R19 note fields: the prereg names unit_bytes and fp_env_applied; the CPU and longest-unit names are the accepted
# spellings until the implementation settles them (--note-field renames).
NOTE_FIELDS = {"unit_bytes": ("unit_bytes",),
               "cpu_user": ("cpu_user_s", "cpu_user_seconds", "user_cpu_s", "user_seconds"),
               "cpu_kernel": ("cpu_kernel_s", "cpu_kernel_seconds", "kernel_cpu_s", "kernel_seconds"),
               "longest_unit": ("max_unit_s", "max_unit_seconds", "longest_unit_s", "longest_unit_seconds"),
               "fp_env": ("fp_env_applied",)}
PARALLEL_ONLY_FIELDS = ("fp_env",)               # a serial entry runs no worker thread
LEGACY_DICT_NOTES = ("frame_pool.coverage",)     # P1b's single-call form
OUTSIDE_TIMINGS = ("frame_pool", "governor", "hash_ledger", "overlap", "siril_stack_settings", "nox_threads",
                   "pool_stats")
TIMING_KEYS = ("timings", "timing_attempts")
MAX_FRAME_POOL_THREADS = 16                      # frame_pool.MAX_FRAME_WORKERS
D4_FIELD, D4_VERSION, D4_NOTE = "stack_settings", "siril-stack-settings-1", "siril_stack_settings"
NOX_THREADS = 8                                  # starstack_runtime.NOX_INTRA_OP_THREADS (EFF-NOX)
OVERLAP_NOTE = "overlap.o1"
# core.O1_NOTE_SCHEMA and core._o1_decide's reasons in its order (prereg §3.6), plus overlap_error:<type>
OVERLAP_SCHEMA = "overlap-o1-1"
OVERLAP_REASONS = ("trails_off", "fallback_registration", "stack_checkpointed", "overlap_off", "serial_workers",
                   "in_pool", "stackcnt", "naxis2", "marks_invalid", "memory_unknown", "memory_tight", "commit_tight",
                   "slower_than_serial")
OVERLAP_PLAN_FIELDS = ("mode", "pixels", "frames", "n_o1", "n_serial", "cov_plan", "crop_cap", "scan_unit_bytes",
                       "a_reserve_bytes", "b_inproc_bytes", "headroom_bytes", "predicted_s", "pieces")
SUPPORT_RECORD = "interpolation_support.json"    # registered_support.RECORD
QUICK_SUPPORT_IMPLEMENTATION = "numpy_buffered_kernel_2"   # registered_support.SUPPORT_IMPLEMENTATION (G-4t's route)
# Planned maxima per site (prereg 2.3): cap under 'default'; under 'max' every frame-pool site may reach 16.
# modes: the stack modes in which the site runs (SL: the strict L CLI job of W19-W21, prereg 3.2: its job-level sites,
# strict support instead of coverage counting, no workflow); conditional: may legitimately not run in an accepted run.
SITES = {
    "median": {"modes": ("L", "F", "R", "SL"), "default": 4},
    "sky_zero": {"modes": ("L", "F", "R", "SL"), "default": 4},
    "coverage": {"modes": ("L", "R"), "default": 8},
    "strict_support": {"modes": ("F", "SL"), "default": 8,
                       "note": "STRICT_WORKER_CAP is measured (4/6/8): --site-max"},
    "native_mask": {"modes": ("F", "SL"), "default": 4, "conditional": True,
                    "note": "measured 2-4; absent when the record label picks the reference route"},
    "finite_write": {"modes": ("F",), "default": 4},
    "finite_inventory": {"modes": ("F",), "default": 4},
    "trail_crops": {"modes": ("L", "F", "R", "SL"), "default": 8, "conditional": True,
                    "note": "trails on, with candidates"},
    "milkyway_render": {"modes": ("L", "F", "R"), "default": 8},
    "star_reduction": {"modes": ("L", "F", "R"), "default": 8},
    "trail_pair": {"modes": ("L", "F", "R"), "default": 8, "conditional": True, "note": "automatic add-back only"},
    "addback_boxes": {"modes": ("L", "F", "R"), "default": 2, "conditional": True,
                      "note": "T-7 only if measured worth it"},
    "transient_scan": {"modes": ("L", "F", "R", "SL"), "default": 8, "conditional": True, "processes": True,
                       "note": "T-2 scan processes (candidates.json workers); trails on"},
}
# The bench's --job kinds and the stack mode coverage counts each under.
JOB_KINDS = {"strict-legacy": "SL"}
# The (stack mode, parallel mode) combinations of the W table (prereg 6.3) that must each have an accepted run:
# L20/L48 default and max (W2, W3, W9/W10), F default and max (W7, W8, W11/W12), R48 default (W13, W18), strict L20
# default and max (W19, W21).  Serial runs (W1, W6, W20) have no maximum to reach.
EXPECTED_COMBOS = (("L", "default"), ("L", "max"), ("F", "default"), ("F", "max"), ("R", "default"),
                   ("SL", "default"), ("SL", "max"))
# Cancel and resume (prereg 6.4)
DATA_CLASSES = ("fits", "tiff", "image", "npy")      # the verifier's classes of FITS, TIFF, JPEG/PNG, NPY files
LEFTOVER = re.compile(r"(?:^|/)(?:work/transients(?:[/.]|$)|\.partial-)")
SCAN_OUTPUT = re.compile(r"(?:^|/)work/transients(?:/|$)")   # the job's own scan output when the reference has it too
SCAN_RUN_DIR = re.compile(r"^(.*?(?:^|/)work/transients/scan-runs/[^/]+)(?:/|$)")   # group 1: a scan run's folder
# a scan run's worker ledgers and logs: their count follows the scan's worker count (the verifier lists them and never
# compares them, verify_workflow_equivalence.LISTED_ONLY), so they are never leftovers by name; their folder is
SCAN_RUN_WORKER_FILE = re.compile(r"(?:^|/)work/transients/scan-runs/[^/]+/worker-\d+\.(?:json|log)$")
CANCEL_LIMIT_S, CANCEL_K_MARGIN_S, HX2_CANCEL_LIMIT_S, QUIET_LIMIT_S = 3.0, 1.0, 5.0, 5.0
HX2_PROFILES = ("6900hx-proxy-v2",)
RESULT_META = ("verdict", "class", "one_sided_diagnostic_fields", "deep", "allowed_by", "tolerance")
# KD-3 (prereg 4.2): the contract field, the Siril configuration files and the lines a D4 block or a pin consists of
D4_FIELD_SEGMENT = "/" + "stack_settings"
SIRIL_INI = re.compile(r"siril.*\.ini", re.IGNORECASE)
SETTING_LINE = re.compile(r"setcpu [0-9]+|set core\.mem_(?:mode|amount|ratio)=[0-9.]+")


class UsageError(Exception):
    pass


def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"aester_check_p1c_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULES = {}


def bench_module():
    if "bench" not in _MODULES:
        _MODULES["bench"] = _load_script("bench_workflow")
    return _MODULES["bench"]


def verifier_module():
    if "verifier" not in _MODULES:
        _MODULES["verifier"] = _load_script("verify_workflow_equivalence")
    return _MODULES["verifier"]


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


# --------------------------------------------------------------------------- shared readers

def effective_environment(plan: dict, summary: dict) -> dict:
    """What the run's processes saw: the inherited relevant variables with the plan's settings applied (upper case:
    Windows variables ignore case)."""
    env = {k.upper(): v for k, v in (summary.get("inherited_environment") or {}).items() if RELEVANT_ENV.match(k)}
    for key, value in (plan.get("environment") or {}).items():
        if value is None:
            env.pop(key.upper(), None)
        else:
            env[key.upper()] = value
    return env


def knob_value(key: str, value):
    """A knob's effective value: absent or empty means its default (KNOB_DEFAULTS); other values unchanged."""
    if key in KNOB_DEFAULTS:
        return (value or "").strip().lower() or KNOB_DEFAULTS[key]
    return value


def _desktop(plan: dict) -> bool:
    return (plan.get("siril_launch") or {}).get("mode") == "desktop-sentinel"


def job_kind(run: dict) -> str | None:
    """The bench's --job kind of a run (a stack job without a workflow), else None."""
    return run["plan"].get("job")


def job_glob(run: dict, rel: str) -> str:
    """Manifest glob of a file of the run's stack job: {WF}/stack/job-<id>/rel in a workflow, {JOB}/rel in a --job
    run (the bench's job-directory fingerprint names the job folder {JOB})."""
    return ("{JOB}/" if job_kind(run) else "*/stack/job-*/") + rel


def parallel_mode(plan: dict) -> str:
    """The run's effective parallel mode: the plan's record (bench P1c), else 'default' when no parallelism knob was
    given (or only at its documented default: frame workers auto, overlap on), else 'custom'."""
    recorded = (plan.get("parallel") or {}).get("effective")
    if recorded:
        return recorded
    pins = plan.get("pins") or {}
    knobs = [None if pins.get("frame_workers") == "auto" else pins.get("frame_workers"),
             None if pins.get("overlap") == "on" else pins.get("overlap"), pins.get("parallel")]
    profile_render = ((bench_module().PROFILES.get(plan.get("profile")) or {}).get("knobs") or {}).get("render_workers")
    if pins.get("render_workers") not in (None, profile_render):
        knobs.append(pins.get("render_workers"))
    return "default" if all(k is None for k in knobs) else "custom"


def _entries(value) -> list:
    """A note's call entries: a list (R19), P1b's single dict, or either written as one-line JSON (compact form)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except ValueError:
                    pass
            out.append(item)
        return out
    return [value]


def receipts(run: dict) -> list:
    """(label, notes) of every timing receipt of the run: the job, the workflow and each trail apply."""
    out = [("job", (((run["job"].get("diagnostics") or {}).get("timings") or {}).get("notes") or {}))]
    workflow = run.get("workflow")
    if workflow is not None:
        out.append(("workflow", ((workflow.get("timings") or {}).get("notes") or {})))
        for index, delivery in enumerate(workflow.get("review_deliveries") or []):
            notes = (((delivery or {}).get("timings") or {}).get("notes")) if isinstance(delivery, dict) else None
            if notes is not None:
                out.append((f"workflow.review_deliveries[{index}]", notes))
    return out


def pool_notes(run: dict) -> list:
    """[{receipt, key, site, raw, entries}] for every frame_pool.<site> note."""
    found = []
    for label, notes in receipts(run):
        for key, value in notes.items():
            if isinstance(key, str) and key.startswith("frame_pool."):
                found.append({"receipt": label, "key": key, "site": key[len("frame_pool."):], "raw": value,
                              "entries": _entries(value)})
    return found


def manifest_docs(manifest: dict, pattern: str) -> list:
    """(name, document) of the manifest's JSON files whose key matches the glob (forward slashes)."""
    return [(key, ((entry.get("content") or {}).get("doc"))) for key, entry in sorted(manifest.get("files", {}).items())
            if fnmatch.fnmatchcase(key.replace("\\", "/"), pattern) and entry.get("class") == "json"]


def manifest_texts(manifest: dict, pattern: str) -> list:
    return [(key, ((entry.get("content") or {}).get("text") or "").replace("\r\n", "\n"))
            for key, entry in sorted(manifest.get("files", {}).items())
            if fnmatch.fnmatchcase(key.replace("\\", "/"), pattern) and entry.get("class") == "text"]


def scan_workers(run: dict) -> list:
    """(name, workers, frames) of every trail record (candidates.json) in the run's manifest."""
    out = []
    for name, doc in manifest_docs(run["manifest"], "*/transients/candidates.json"):
        if isinstance(doc, dict) and "workers" in doc:
            frames = len(doc.get("frame_order") or []) or doc.get("frames_scanned")
            out.append((name, doc.get("workers"), frames if isinstance(frames, int) else None))
    return out


def overlap_problems(note) -> list:
    """The shape of the overlap.o1 note core writes (prereg §3.6; core._run_locked's o1_begin): the schema, ``eligible``
    (a bool), ``reason`` (None when eligible, else one of OVERLAP_REASONS or ``overlap_error:<type>``: the first
    condition that failed); an eligible note carries the plan (OVERLAP_PLAN_FIELDS, both headrooms) and the outcome
    (joined / recorded / guard_mismatch:<guard> / discarded:<why>), a joined one the join wait and the windows.
    A plan's figures may also accompany an ineligible note (the memory and time criteria). No note: no problem."""
    if note is None:
        return []
    problems = []
    for index, entry in enumerate(_entries(note)):
        where = f"[{index}]" if isinstance(note, list) else ""
        if not isinstance(entry, dict):
            problems.append(f"{where} is not a record")
            continue
        if entry.get("schema") != OVERLAP_SCHEMA:
            problems.append(f"{where} schema {entry.get('schema')!r}, not {OVERLAP_SCHEMA}")
        eligible, reason = entry.get("eligible"), entry.get("reason")
        if not isinstance(eligible, bool):
            problems.append(f"{where} eligible is {eligible!r}, not a bool")
            continue
        if eligible:
            if reason is not None:
                problems.append(f"{where} eligible with the reason {reason!r}")
            missing = [k for k in OVERLAP_PLAN_FIELDS if k not in entry]
            if missing:
                problems.append(f"{where} eligible without {', '.join(missing)}")
            headroom = entry.get("headroom_bytes")
            if not (isinstance(headroom, dict) and {"physical", "commit"} <= set(headroom)):
                problems.append(f"{where} headroom_bytes lacks the physical / commit pair: {headroom!r}")
            outcome = entry.get("outcome")
            if not isinstance(outcome, str):
                problems.append(f"{where} eligible without an outcome")
            elif outcome == "joined" and not ("join_wait_s" in entry and isinstance(entry.get("window_s"), dict)):
                problems.append(f"{where} joined without join_wait_s / window_s")
        elif not (reason in OVERLAP_REASONS or (isinstance(reason, str) and reason.startswith("overlap_error:"))):
            problems.append(f"{where} not eligible with an unknown reason {reason!r}")
    return problems


def overlap_on(note) -> bool | None:
    """O-1's state from the overlap.o1 note: None when there is no note (O-1 not reached, or code before O-1)."""
    entries = _entries(note) if note is not None else []
    if not entries:                            # no note, or an empty one (check_parallel reports the latter)
        return None
    note = entries[-1]
    if not isinstance(note, dict):
        return True
    for key in ("eligible", "enabled", "used", "overlapped"):
        if key in note:
            return bool(note[key])
    if "reason" in note:
        return note["reason"] in (None, "", "ok", "eligible")
    return True


# --------------------------------------------------------------------------- plan

def _profile_knobs(plan: dict) -> dict:
    return ((bench_module().PROFILES.get(plan.get("profile")) or {}).get("knobs") or {})


def check_plan(run: dict, ref: dict, expect_env: dict, expect_pin: dict, expect_parallel: str | None,
               allow_pins: bool) -> dict:
    a, b = ref["plan"], run["plan"]
    problems = []
    for key in ("mode", "frames", "slot", "profile", "job"):
        if a.get(key) != b.get(key):
            problems.append(f"{key}: reference {a.get(key)!r}, run {b.get(key)!r}")
    if (a.get("simulation") or {}).get("profile") != (b.get("simulation") or {}).get("profile"):
        problems.append("simulation.profile differs")
    submit_a = {k: v for k, v in (a.get("submit") or {}).items() if k not in SUBMIT_PATHS}
    submit_b = {k: v for k, v in (b.get("submit") or {}).items() if k not in SUBMIT_PATHS}
    for key in sorted(set(submit_a) | set(submit_b)):
        if submit_a.get(key, "<absent>") != submit_b.get(key, "<absent>"):
            problems.append(f"submit.{key}: reference {submit_a.get(key, '<absent>')!r}, "
                            f"run {submit_b.get(key, '<absent>')!r}")
    pins_a, pins_b = a.get("pins") or {}, b.get("pins") or {}
    for key in sorted((set(pins_a) | set(pins_b)) - set(PARALLEL_PINS)):
        if key in expect_pin:
            if pins_b.get(key) != expect_pin[key]:
                problems.append(f"pins.{key}: run {pins_b.get(key)!r}, expected {expect_pin[key]!r}")
        elif key not in pins_a:
            if key in PIN_DEFAULTS and pins_b.get(key) == PIN_DEFAULTS[key]:
                continue                       # what a run of the P1b-era bench had
            problems.append(f"pins.{key} is new in the run ({pins_b.get(key)!r}): name it with --expect-pin")
        elif pins_a.get(key) != pins_b.get(key):
            problems.append(f"pins.{key}: reference {pins_a.get(key)!r}, run {pins_b.get(key)!r}")
    mode_a, mode_b = parallel_mode(a), parallel_mode(b)
    wanted = expect_parallel or mode_a
    if mode_b != wanted:
        problems.append(f"parallel mode: run {mode_b!r}, expected {wanted!r}"
                        + ("" if expect_parallel else " (the reference's; name another with --expect-parallel)"))
    if not allow_pins:
        profile = _profile_knobs(b)
        siril = bench_module().parse_pin(profile.get("pin_siril")) if profile.get("pin_siril") else None
        if pins_b.get("siril") != siril:
            problems.append(f"fixed Siril parameters {pins_b.get('siril')!r} beyond the profile's {siril!r} "
                            "(a contract comparison passes --allow-pins)")
        if pins_b.get("ort_intra_op_threads") != profile.get("pin_ort"):
            problems.append(f"--pin-ort {pins_b.get('ort_intra_op_threads')!r} beyond the profile's "
                            f"{profile.get('pin_ort')!r} (a contract comparison passes --allow-pins)")
        if mode_b == "custom":
            problems.append("single parallelism knobs (--frame-workers / --render-workers / --overlap) without "
                            "--parallel: a fixed parameter (--allow-pins)")
    env_a, env_b = effective_environment(a, ref["summary"]), effective_environment(b, run["summary"])
    skipped = PATH_KEYS | HYGIENE_KEYS | set(PARALLEL_VARIABLES) | (DESKTOP_KEYS if _desktop(a) or _desktop(b) else set())
    compared = {}
    for key in sorted((set(env_a) | set(env_b) | set(expect_env)) - skipped):
        expected = knob_value(key, expect_env[key] if key in expect_env else env_a.get(key))
        actual = knob_value(key, env_b.get(key))
        compared[key] = {"reference": env_a.get(key), "run": env_b.get(key), "expected": expected}
        if actual != expected:
            problems.append(f"environment {key}: run {env_b.get(key)!r}, expected {expected!r}"
                            + ("" if key in expect_env else " (the reference's)"))
    managed = {k.upper() for k in (b.get("environment") or {})}
    unexpected = sorted(k for k in env_b if k.startswith(SURVEILLED) and k not in managed
                        and k not in INHERITED_KNOBS and k not in HYGIENE_KEYS and k not in expect_env)
    for key in unexpected:
        problems.append(f"environment {key}={env_b[key]!r} is not one the bench sets: an inherited diagnostics "
                        "switch? name it with --expect-env")
    problems.extend(view_problems(run))
    return {"passed": not problems, "problems": problems, "parallel": {"reference": mode_a, "run": mode_b},
            "environment": compared, "skipped_environment": sorted(skipped)}


# What the hook records about a wrapped view (bench HOOK_SOURCE patch_performance)
VIEW_RESERVATIONS_OK = ("below the view", "deducted from the view", "none")


def view_problems(run: dict) -> list:
    """A memory / commit view must keep the code root's in-process reservations (R20): applied to the reading before
    reservations, or around the public function with the reservations deducted from the view.  A process whose view
    record says otherwise (or predates the record: no 'reservations') saw min(real - reserved, view), in which the
    reservations of T-2, O-1 and the pools vanish under the 6900HX proxy's 9 GiB."""
    pins = run["plan"].get("pins") or {}
    if pins.get("memory_view_gib") is None and pins.get("commit_view_gib") is None:
        return []
    problems = []
    views = ((run["summary"].get("hook_events") or {}).get("views") or {})
    for process in views.get("processes") or []:
        for name, applied in sorted((process.get("applied") or {}).items()):
            applied = applied if isinstance(applied, dict) else {}
            if applied.get("status") != "wrapped":
                continue
            reservations = applied.get("reservations")
            hidden = reservations == "none" and applied.get("frame_pool_reserved")
            if reservations not in VIEW_RESERVATIONS_OK or hidden:
                problems.append(f"memory view {name} in pid {process.get('pid')}: {applied.get('function')!r} wrapped "
                                f"with reservations {reservations!r} (frame_pool.reserved_bytes "
                                f"{'present' if applied.get('frame_pool_reserved') else 'unknown'}): the view hides "
                                "the code root's in-process reservations")
    return problems


# --------------------------------------------------------------------------- parallel

def expected_parallel_environment(plan: dict, mode: str) -> dict | None:
    table = bench_module().PARALLEL_TABLE.get(mode)
    if table is None:
        return None
    expected = dict(table)
    if plan.get("parallel") is None:            # a bench before P1c: a profile's render workers are its tier
        render = (plan.get("pins") or {}).get("render_workers")
        if render is not None and render == _profile_knobs(plan).get("render_workers"):
            expected["AESTER_RENDER_WORKERS"] = str(render)
        return expected
    recorded = plan["parallel"]
    tier = set(recorded.get("tier") or ())
    names = {"render_workers": "AESTER_RENDER_WORKERS", "frame_workers": "AESTER_FRAME_WORKERS",
             "overlap": "AESTER_OVERLAP"}
    for knob in tier:
        name = names.get(knob)
        if name and name in (recorded.get("environment") or {}):
            expected[name] = recorded["environment"][name]
    return expected


def _site_max(mode: str, site: str, plan_env: dict, site_caps: dict) -> int | None:
    entry = SITES.get(site) or {}
    if mode == "serial":
        return 1
    if mode == "max":
        if entry.get("processes"):
            try:
                return int(plan_env.get("STARSTACK_TRANSIENT_WORKERS"))
            except (TypeError, ValueError):
                return None
        return MAX_FRAME_POOL_THREADS
    if mode == "default":
        return site_caps.get(site, entry.get("default"))
    return None


def check_parallel(run: dict, allowed_reasons, site_caps: dict, serial_reasons=("workers",)) -> dict:
    plan = run["plan"]
    mode = parallel_mode(plan)
    env = effective_environment(plan, run["summary"])
    problems, info = [], []
    expected = expected_parallel_environment(plan, mode)
    if expected is not None:
        for name in PARALLEL_VARIABLES:
            want, have = knob_value(name, expected.get(name)), knob_value(name, env.get(name))
            if want != have:
                problems.append(f"{name}={env.get(name)!r} but --parallel {mode} sets {expected.get(name)!r}")
    sites = {}
    for note in pool_notes(run):
        record = sites.setdefault(note["site"], {"calls": 0, "max_workers": 0, "max_units": 0, "serial_calls": 0,
                                                 "reasons": [], "receipts": [], "reached": False})
        record["receipts"] = sorted(set(record["receipts"]) | {note["receipt"]})
        planned = _site_max(mode, note["site"], env, site_caps)
        record["planned_max"] = planned
        for index, entry in enumerate(note["entries"]):
            where = f"{note['receipt']}: {note['key']}[{index}]"
            if not isinstance(entry, dict):
                problems.append(f"{where} is not a call record")
                continue
            record["calls"] += 1
            workers, units = entry.get("workers"), entry.get("units")
            workers = workers if isinstance(workers, int) else 0
            units = units if isinstance(units, int) else 0
            record["max_workers"] = max(record["max_workers"], workers)
            record["max_units"] = max(record["max_units"], units)
            serial = bool(entry.get("serial")) or workers <= 1
            if serial:
                record["serial_calls"] += 1
                if entry.get("reason") not in record["reasons"]:
                    record["reasons"].append(entry.get("reason"))
            if planned is not None and units and workers >= min(planned, units):
                record["reached"] = True
            if mode == "serial":
                if not entry.get("serial") or entry.get("reason") not in serial_reasons:
                    problems.append(f"{where}: --parallel serial, but serial={entry.get('serial')!r} "
                                    f"reason={entry.get('reason')!r} workers={entry.get('workers')!r}")
            elif mode in ("default", "max") and serial and units >= 2:
                if entry.get("reason") not in allowed_reasons:
                    problems.append(f"{where}: {units} units ran serially, reason {entry.get('reason')!r} "
                                    f"(pre-registered: {', '.join(allowed_reasons)})")
    scans = scan_workers(run)
    if scans:
        values = sorted({w for _, w, _ in scans if isinstance(w, int)})
        frames = max((f for _, _, f in scans if isinstance(f, int)), default=None)
        planned = _site_max(mode, "transient_scan", env, site_caps)
        sites["transient_scan"] = {"calls": len(scans), "max_workers": max(values) if values else 0,
                                   "max_units": frames or 0, "serial_calls": sum(1 for _, w, _ in scans if w == 1),
                                   "reasons": [], "receipts": ["candidates.json"], "planned_max": planned,
                                   "reached": bool(values) and planned is not None
                                   and max(values) >= min(planned, frames or planned)}
        if len(values) > 1:
            problems.append(f"trail records disagree on the scan workers: {values}")
        if mode == "serial" and values != [1]:
            problems.append(f"--parallel serial, but the trail scan used {values} processes")
        if mode == "max" and planned is not None and values and values[-1] != min(planned, frames or planned):
            problems.append(f"--parallel max, but the trail scan used {values} processes "
                            f"(STARSTACK_TRANSIENT_WORKERS={planned}, {frames} frames)")
    elif (plan.get("submit") or {}).get("trails_enabled"):
        info.append("trails on, but the manifest has no trail record with 'workers'")
    o1_note = (((run["job"].get("diagnostics") or {}).get("timings") or {}).get("notes") or {}).get(OVERLAP_NOTE)
    if o1_note is not None and not _entries(o1_note):
        problems.append(f"the {OVERLAP_NOTE} note is empty ({o1_note!r})")
    else:
        problems.extend(f"{OVERLAP_NOTE}{p}" for p in overlap_problems(o1_note))
    o1 = overlap_on(o1_note)
    if mode == "serial" and o1:
        problems.append("--parallel serial, but the overlap.o1 note says O-1 ran")
    # G-4t (prereg 3.2): the native stage follows the support record's label; the reference route runs no pool
    routes = sorted({doc.get("implementation") for _, doc in manifest_docs(run["manifest"],
                                                                           job_glob(run, "work/" + SUPPORT_RECORD))
                     if isinstance(doc, dict) and doc.get("implementation")})
    if routes and QUICK_SUPPORT_IMPLEMENTATION not in routes and "native_mask" in sites:
        problems.append(f"the support record names the reference route {routes}, but a frame_pool.native_mask note "
                        "exists")
    return {"passed": not problems, "problems": problems, "info": info, "mode": mode, "overlap_o1": o1,
            "support_routes": routes, "sites": sites}


# --------------------------------------------------------------------------- notes

def keys_outside_timings(value, names=OUTSIDE_TIMINGS, path="") -> list:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TIMING_KEYS:
                continue
            if any(key == n or str(key).startswith(n + ".") for n in names):
                found.append(f"{path}/{key}")
            found.extend(keys_outside_timings(item, names, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(keys_outside_timings(item, names, f"{path}[{index}]"))
    return found


def _field(entry: dict, aliases) -> str | None:
    return next((alias for alias in aliases if alias in entry), None)


def check_notes(run: dict, ref: dict, note_fields: dict) -> dict:
    problems, info = [], []
    for note in pool_notes(run):
        if not isinstance(note["raw"], list):
            if note["key"] in LEGACY_DICT_NOTES and isinstance(note["raw"], dict):
                info.append(f"{note['receipt']}: {note['key']} keeps P1b's single-call form (fields not checked)")
                continue
            problems.append(f"{note['receipt']}: {note['key']} is not a list of calls (R19: note_pool accumulates)")
        for index, entry in enumerate(note["entries"]):
            if not isinstance(entry, dict):
                continue
            serial = bool(entry.get("serial")) or (entry.get("workers") or 0) <= 1
            missing = [name for name, aliases in note_fields.items()
                       if not (serial and name in PARALLEL_ONLY_FIELDS) and _field(entry, aliases) is None]
            if missing:
                problems.append(f"{note['receipt']}: {note['key']}[{index}] lacks {', '.join(missing)}")
    env = effective_environment(run["plan"], run["summary"])
    governor_mode = knob_value("AESTER_GOVERNOR", env.get("AESTER_GOVERNOR"))
    ref_ledger = dict(receipts(ref)).get("job", {}).get("hash_ledger")
    ref_version = ref_ledger.get("version") if isinstance(ref_ledger, dict) else None
    if ref_version is None:
        info.append("the reference has no job hash_ledger note; the ledger version is not compared")
    for label, notes in receipts(run):
        if label.startswith("workflow.review_deliveries"):
            continue
        ledger = notes.get("hash_ledger")
        if not isinstance(ledger, dict):
            problems.append(f"{label}: hash_ledger note missing")
        else:
            for key in ("miss_bytes", "largest_misses"):
                if key not in ledger:
                    problems.append(f"{label}: hash_ledger has no {key}")
            if ref_version is not None and ledger.get("version") != ref_version:
                problems.append(f"{label}: hash_ledger version {ledger.get('version')!r}, the reference's "
                                f"{ref_version!r}")
        governor = notes.get("governor")
        if (governor is not None) != (governor_mode != "off"):
            problems.append(f"{label}: governor note {'present' if governor is not None else 'absent'} with "
                            f"AESTER_GOVERNOR={governor_mode}")
        elif governor is not None and not isinstance(governor, dict):
            problems.append(f"{label}: governor note is not an object")
        if "l_siril_layout" in notes:
            problems.append(f"{label}: l_siril_layout note (a non-default L layout)")
    for label, record in (("job.json", run["job"]), ("workflow.json", run.get("workflow"))):
        if record is not None:
            for path in keys_outside_timings(record):
                problems.append(f"{label}: {path} outside the timing receipts")
    return {"passed": not problems, "problems": problems, "info": info, "governor_mode": governor_mode,
            "pool_notes": sorted({n["key"] for n in pool_notes(run)})}


# --------------------------------------------------------------------------- threads

def check_threads(run: dict) -> dict:
    problems, info = [], []
    summary = run["summary"]
    audit = summary.get("thread_audit")
    if not isinstance(audit, dict) or not audit.get("processes"):
        problems.append("the summary has no thread audit (a bench before P1c, or the hook did not load)")
        audit = {"processes": []}
    worst = []
    for process in audit["processes"]:
        value = process.get("frame_pool_max")
        if isinstance(value, int) and value > MAX_FRAME_POOL_THREADS:
            problems.append(f"pid {process.get('pid')} ({(process.get('argv') or ['?'])[0]}): {value} frame-pool "
                            f"threads at once (at most {MAX_FRAME_POOL_THREADS})")
        worst.append({"pid": process.get("pid"), "argv0": (process.get("argv") or [None])[0],
                      "frame_pool_max": value, "blas_capable_max": process.get("blas_capable_max"),
                      "python_threads_max": process.get("python_threads_max"), "samples": process.get("samples")})
    per_exe = ((summary.get("sampler") or {}).get("per_exe") or {})
    os_threads = {exe: slot.get("max_threads") for exe, slot in sorted(per_exe.items())}
    blas = [p.get("blas_capable_max") for p in audit["processes"] if isinstance(p.get("blas_capable_max"), int)]
    if blas and max(blas) > 24:
        info.append(f"up to {max(blas)} BLAS-capable Python threads in one process (OpenBLAS MAX_THREADS=24)")
    return {"passed": not problems, "problems": problems, "info": info, "processes": worst,
            "frame_pool_max": audit.get("frame_pool_max"), "blas_capable_max": max(blas) if blas else None,
            "os_threads_max_per_exe": os_threads}


# --------------------------------------------------------------------------- contract

_D4_LINES = (re.compile(r"setcpu [1-9][0-9]*"), re.compile(r"set core\.mem_mode=1"),
             re.compile(r"set core\.mem_amount=[1-9][0-9]*"))


def d4_blocks(text: str, pin: dict | None) -> dict:
    """The D4 blocks of one Siril script: the pin lines (the bench's AESTER_SIRIL_PIN) at the head are not D4."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    head = 2 if len(lines) > 1 and lines[1] == "offline" else 1
    pin_block = bench_module().pin_lines(pin)
    if pin_block and lines[head:head + len(pin_block)] == pin_block:
        head += len(pin_block)
    body = lines[head:]
    blocks, stray = [], []
    index = 0
    while index < len(body):
        if all(index + k < len(body) and pattern.fullmatch(body[index + k]) for k, pattern in enumerate(_D4_LINES)):
            blocks.append({"line": head + index, "setcpu": int(body[index].split()[1]),
                           "mem_amount": int(body[index + 2].split("=")[1])})
            index += 3
            continue
        if body[index].startswith(("setcpu", "set core.mem_")):
            stray.append(head + index)
        index += 1
    stack_line = next((head + i for i, line in enumerate(body) if line.startswith("stack ")), None)
    return {"blocks": blocks, "stray": stray, "stack_line": stack_line}


def check_contract(run: dict, d4_version: str, before=()) -> dict:
    """before: contracts the run's code predates ('d4': no D4 anywhere, as in F; 'nox': the request is not checked)
    -- A1b, KN-2 and X2 run code from before K1/K2."""
    problems, info = [], []
    mode = run["plan"].get("mode")
    pin = (run["plan"].get("pins") or {}).get("siril")
    settings = (run["job"].get("admission_options") or {}).get(D4_FIELD)
    job_notes = dict(receipts(run)).get("job", {})
    note, note_source = job_notes.get(D4_NOTE), "final attempt"
    if note is None:
        # 事后 10-07 (window D review): a run resumed after the stack checkpoint stacked in an earlier attempt, whose
        # note travels in timing_attempts; take the latest attempt that has it (the one that wrote 02_stack.ssf)
        for prefix, timings in reversed(job_attempt_timings(run["job"])[:-1]):
            earlier = ((timings.get("notes") or {}) if isinstance(timings, dict) else {}).get(D4_NOTE)
            if earlier is not None:
                note, note_source = earlier, prefix.rstrip(":")
                info.append(f"{D4_NOTE} from {note_source}: the final attempt started past the stack")
                break
    scripts = {}
    if mode in ("L", "R") and "d4" not in before:
        if not isinstance(settings, dict) or settings.get("version") != d4_version:
            problems.append(f"job admission_options.{D4_FIELD} is {settings!r}, expected version {d4_version!r}")
        if note is None:
            problems.append(f"job note {D4_NOTE} is missing")
        # L (and strict L): 02_stack.ssf (paths 3a/3b/3c); R: each range trial's stack.ssf (run_protected_stack)
        targets = manifest_texts(run["manifest"], job_glob(run, "02_stack.ssf") if mode == "L"
                                 else job_glob(run, "range-*/trial*/stack.ssf"))
        if not targets:
            problems.append("no stack script in the manifest (" + ("02_stack.ssf)" if mode == "L"
                                                                   else "range-*/trial*/stack.ssf)"))
        for name, text in targets:
            found = d4_blocks(text, pin)
            scripts[name] = found
            if len(found["blocks"]) != 1:
                problems.append(f"{name}: {len(found['blocks'])} D4 blocks (exactly one after the pin lines)")
            elif found["stack_line"] is not None and found["blocks"][0]["line"] > found["stack_line"]:
                problems.append(f"{name}: the D4 block follows the stack command")
            if found["stray"]:
                problems.append(f"{name}: stray thread/memory lines at {found['stray']}")
    else:                                      # F, or code from before EFF-D4: none of the three
        what = "an F job" if mode == "F" else "a job before EFF-D4"
        if settings is not None:
            problems.append(f"{what} has admission_options.{D4_FIELD}")
        if note is not None:
            problems.append(f"{what} has a {D4_NOTE} note")
        for name, text in manifest_texts(run["manifest"], "*.ssf"):
            found = d4_blocks(text, pin)
            if found["blocks"] or found["stray"]:
                scripts[name] = found
                problems.append(f"{name}: thread/memory lines beyond the pin in {what}")
    workflow = run.get("workflow") or {}
    request = workflow.get("request") or {}
    nox = (request.get("backend_options") or {}).get("intra_op_threads") if isinstance(request, dict) else None
    if job_kind(run):
        info.append(f"a {job_kind(run)} job run: no workflow request, no separation, no nox thread check")
    elif "nox" in before:
        info.append(f"code before EFF-NOX: the request's intra_op_threads {nox!r} is not checked (plan parity is)")
    elif request.get("stars_enabled"):
        if nox != NOX_THREADS:
            problems.append(f"the frozen request's backend_options.intra_op_threads is {nox!r}, not {NOX_THREADS} "
                            "(EFF-NOX)")
    elif workflow:
        info.append("stars_enabled is off: no separation, no nox thread check")
    nox_note = dict(receipts(run)).get("workflow", {}).get("nox_threads")
    return {"passed": not problems, "problems": problems, "info": info, "stack_settings": settings,
            "siril_stack_settings_note": note, "note_source": note_source, "scripts": scripts, "nox_request_threads": nox,
            "nox_threads_note": nox_note}


# --------------------------------------------------------------------------- Siril

def check_siril(run: dict) -> dict:
    problems = []
    job = run["job"]
    summary = run["summary"]
    segments = len((summary.get("resume") or {}).get("segments") or []) or 1
    if summary.get("state") != "succeeded":
        problems.append(f"bench state {summary.get('state')!r}")
    if job.get("state") != "succeeded":
        problems.append(f"job state {job.get('state')!r}: {job.get('error')}")
    diagnostics = job.get("diagnostics") or {}
    attempts = len(diagnostics.get("timing_attempts") or [])
    if attempts > segments - 1:
        problems.append(f"the job ran {attempts + 1} attempts in {segments} segment(s)")
    if run.get("workflow") is not None and len(run["workflow"].get("timing_attempts") or []) > segments - 1:
        problems.append("the workflow ran more attempts than segments")
    # 事后 10-07 (window D review): a run resumed after the stack checkpoint (C2-hx2, C7) launches no Siril in its
    # final attempt; every attempt's launches count (timing_attempts, then the final timings), each judged on its own
    launches = {}
    for prefix, timings in job_attempt_timings(job):
        operations = ((timings.get("resources") or {}).get("operations") or {}) if isinstance(timings, dict) else {}
        for name, operation in sorted(operations.items()):
            if not name.startswith("script:"):
                continue
            children = operation.get("children") or {}
            processes, descendants = children.get("processes") or 0, children.get("descendant_processes")
            launches[prefix + name] = {"processes": processes, "descendant_processes": descendants}
            if not processes or descendants is None:
                problems.append(f"{prefix}{name}: no child accounting")
            elif descendants > 2 * processes:
                problems.append(f"{prefix}{name}: {descendants} processes for {processes} Siril launch(es) "
                                "(at most 2 each)")
    if not launches:
        problems.append("the job receipt has no Siril operations (in any attempt)")
    return {"passed": not problems, "problems": problems, "launches": launches, "segments": segments}


def job_attempt_timings(job: dict) -> list:
    """(prefix, timings) of every attempt of the stack job: earlier attempts ('attempt<k>:'), then the final ('')."""
    diagnostics = job.get("diagnostics") or {}
    earlier = [(f"attempt{index + 1}:", timings) for index, timings in enumerate(diagnostics.get("timing_attempts") or [])]
    return earlier + [("", diagnostics.get("timings") or {})]


# --------------------------------------------------------------------------- compare report

def _report_basics(report: dict, run: dict, ref: dict, allow_no_strict_ids: bool) -> tuple[list, bool]:
    """Problems every comparison shares: the report's sides are these manifests, no allowance, strict ids."""
    problems = []
    for side, source in (("a", ref), ("b", run)):
        if (report.get(side) or {}).get("created_utc") != source["manifest"].get("created_utc"):
            problems.append(f"report side {side} is not {source['folder']}'s manifest (a = reference, b = run)")
    for key in ("allowed", "allow_patterns", "allow_fields", "tolerances", "tolerated"):
        if report.get(key):
            problems.append(f"the comparison used {key}: {report.get(key)!r}")
    strict = (report.get("strict_ids") or {}).get("on")
    if not strict and not allow_no_strict_ids:
        problems.append("the comparison ran without strict ids")
    return problems, bool(strict)


def _extra_listed(extra_listed) -> dict:
    listed = {"a": [], "b": []}
    for side, name in extra_listed:
        listed[side].append(name)
    return listed


SCAN_WORKER_INDEX = re.compile(r"/worker-(\d+)\.(json|log)$")


def scan_worker_listing(actual: dict, run: dict, ref: dict) -> dict:
    """The listed-only names a different trail-scan worker count explains (事后 10-07: W1 serial and W3 max against
    K's planner count, and any default whose planner chose another count): with n_a and n_b the one count each side's
    trail records give (scan_workers), the side with more workers lists the ledger and the log of every worker k,
    min <= k < max, in one scan-runs folder - exactly those, or nothing is explained."""
    counts = {}
    for side, folder in (("a", ref), ("b", run)):
        values = {workers for _, workers, _ in scan_workers(folder) if isinstance(workers, int)}
        if len(values) != 1:
            return {"a": [], "b": []}
        counts[side] = values.pop()
    explained = {"a": [], "b": []}
    if counts["a"] == counts["b"]:
        return explained
    more = "a" if counts["a"] > counts["b"] else "b"
    low, high = sorted(counts.values())
    names = [name for name in actual[more] if SCAN_RUN_WORKER_FILE.search(name)]
    folders = {SCAN_RUN_DIR.match(name).group(1) for name in names}
    found = {SCAN_WORKER_INDEX.search(name).groups() for name in names}
    wanted = {(f"{k:03d}", extension) for k in range(low, high) for extension in ("json", "log")}
    if len(folders) == 1 and found == wanted and len(names) == len(wanted):
        explained[more] = sorted(names)
    return explained


def check_compare(report: dict, run: dict, ref: dict, extra_listed: list, allow_no_strict_ids: bool) -> dict:
    problems = []
    if not report.get("pass"):
        problems.append("the compare report did not pass"
                        + (" (a resumed run: pass --resume-allowance with the K calibration of its cancel point)"
                           if (run["summary"].get("resume") or {}).get("segments") else ""))
    basics, strict = _report_basics(report, run, ref, allow_no_strict_ids)
    problems.extend(basics)
    listed = _extra_listed(extra_listed)
    actual = {side: sorted((report.get("only_in_listed") or {}).get(side) or []) for side in ("a", "b")}
    workers = scan_worker_listing(actual, run, ref)
    rest = {side: [name for name in actual[side] if name not in workers[side]] for side in ("a", "b")}
    if rest != {side: sorted(names) for side, names in listed.items()}:
        problems.append(f"only_in_listed {rest} != {listed}"
                        + (f" (scan worker files explained by the worker counts: {workers})"
                           if workers["a"] or workers["b"] else ""))
    return {"passed": not problems, "problems": problems, "strict_ids": strict, "only_in_listed": actual,
            "scan_worker_files": workers}


# --------------------------------------------------------------------------- cancel and resume (prereg 6.4)

def fatal_report_items(report: dict, *, strict_ids=True) -> list:
    """What no allowance may cover: deterministic ids, file errors, links, missing deliveries, derived digests that
    did not verify, excluded keys in a signature preimage, record bindings that do not match their files."""
    problems = []
    if strict_ids and (report.get("strict_ids") or {}).get("differ"):
        problems.append(f"deterministic ids differ: {report['strict_ids']['differ'][:5]}")
    for key in ("errors", "links_different"):
        if report.get(key):
            problems.append(f"the report has {key}: {sorted(report[key])[:10]}")
    for side in ("a", "b"):
        if ((report.get("missing_deliveries") or {}).get(side)):
            problems.append(f"side {side}: missing deliveries {report['missing_deliveries'][side][:5]}")
        if (((report.get("derived_digests") or {}).get(side) or {}).get("failed")):
            problems.append(f"side {side}: derived digests not verified "
                            f"{report['derived_digests'][side]['failed'][:5]}")
        if ((report.get("excluded_keys_in_preimages") or {}).get(side)):
            problems.append(f"side {side}: excluded keys in a preimage "
                            f"{report['excluded_keys_in_preimages'][side][:5]}")
        if (((report.get("bindings") or {}).get(side) or {}).get("mismatch")):
            problems.append(f"side {side}: record bindings that do not match their files "
                            f"{report['bindings'][side]['mismatch'][:5]}")
    return problems


def _sides(run: dict, ref: dict):
    verifier = verifier_module()
    return verifier, verifier.Side(ref["manifest"], "a"), verifier.Side(run["manifest"], "b")


def json_difference_paths(sa, sb, name: str, verifier) -> list:
    """Every difference path of one JSON file present on both sides, as the report writes them ('/json/...'), in
    full: the report itself lists at most the verifier's LIST_LIMIT."""
    ca, _ = sa.content(sa.by_name[name][0])
    cb, _ = sb.content(sb.by_name[name][0])
    found = []
    verifier._json_diff(ca, cb, "", found, 10 ** 9)
    return sorted({item["path"] for item in found})


def report_differences(report: dict, sa, sb, verifier) -> dict:
    """The differences of a compare report in R_resume's terms: canonical names of files and directories on one side
    (listed-only files apart), and per differing file its class and either its JSON difference paths (one-sided
    excluded fields included) or the kinds of difference the report names (text, header, pixels, ...)."""
    different = {}
    for name, result in sorted((report.get("different") or {}).items()):
        item = {"class": result.get("class")}
        if result.get("class") == "json" and name in sa.by_name and name in sb.by_name:
            item["paths"] = sorted(set(json_difference_paths(sa, sb, name, verifier))
                                   | {f.get("path") for f in result.get("one_sided_fields") or []})
        else:
            item["kinds"] = sorted(k for k in result if k not in RESULT_META)
        different[name] = item
    return {"different": different,
            **{key: sorted(report.get(key) or []) for key in ("only_in_a", "only_in_b", "dirs_only_in_a",
                                                               "dirs_only_in_b")},
            "only_in_listed": {side: sorted((report.get("only_in_listed") or {}).get(side) or []) for side in "ab"}}


def data_digests(entry: dict):
    """Per HDU / page / image / array: (dtype, shape, digest of the data) of a FITS, TIFF, JPEG/PNG or NPY entry."""
    content, cls = entry.get("content") or {}, entry.get("class")
    if cls == "fits":
        return [[h.get("dtype"), h.get("shape"), (h.get("data") or {}).get("digest")]
                for h in content.get("hdus") or []]
    if cls == "tiff":
        return [[p.get("dtype"), p.get("shape"), (p.get("pixels") or {}).get("digest")]
                for p in content.get("pages") or []]
    if cls == "image":
        return [[content.get("dtype"), content.get("shape"), (content.get("pixels") or {}).get("digest")]]
    if cls == "npy":
        return [[content.get("dtype"), content.get("shape"), (content.get("data") or {}).get("digest")]]
    return None


def data_differences(sa, sb) -> tuple[list, int]:
    """(canonical names whose data digests differ, how many data files were compared): FITS / TIFF / JPEG / PNG /
    NPY files present on both sides, whatever else the report says about them."""
    out, compared = [], 0
    for name in sorted(set(sa.by_name) & set(sb.by_name)):
        ea, eb = sa.files[sa.by_name[name][0]], sb.files[sb.by_name[name][0]]
        if ea.get("class") not in DATA_CLASSES and eb.get("class") not in DATA_CLASSES:
            continue
        compared += 1
        if ea.get("class") != eb.get("class") or data_digests(ea) != data_digests(eb):
            out.append(name)
    return out, compared


def _tree_names(side) -> set:
    return set(side.names.values()) | {side.canon.string(rel) for rel in side.m.get("dirs", {})}


def leftovers(side, reference=None) -> list:
    """Canonical names of work/transients and .partial-* files and directories in a side's tree.  With the
    uninterrupted ``reference`` side, a work/transients entry the reference has too is the job's own scan output (a
    trails-on job keeps work/transients/candidates.json, its crops and its one scan-runs folder): only the rest is
    left over -- O-1's marker work/transients.o1-pending, a second scan-runs folder, any .partial-* entry.  A scan
    run's worker ledgers and logs (worker-NNN.json / .log) are never leftovers by name: their count follows the scan's
    worker count (a resumed serial scan against an overlapped reference, n_serial against n_o1), which the listed-only
    check owns; a scan-runs folder the reference lacks is still left over by its directory entry."""
    names = _tree_names(side)
    kept = _tree_names(reference) if reference is not None else set()
    return sorted(name for name in names if LEFTOVER.search(name)
                  and not (name in kept and SCAN_OUTPUT.search(name))
                  and not SCAN_RUN_WORKER_FILE.search(name))


def scan_run_folder(name: str) -> str | None:
    """The canonical scan-runs folder (work/transients/scan-runs/<id>) a name is or lies in, else None."""
    match = SCAN_RUN_DIR.match(name)
    return match.group(1) if match else None


def overlap_in_earlier_attempts(run: dict) -> bool | None:
    """Whether O-1 ran (an eligible overlap.o1 note) in an earlier attempt of the run's job -- for a cancel run, the
    cancelled segment's: job.json's timing_attempts keep each earlier attempt's timings.  None: no earlier attempt
    has the note (O-1 not reached, or code before O-1, such as the first segment of X1)."""
    states = []
    for timings in (((run.get("job") or {}).get("diagnostics") or {}).get("timing_attempts") or []):
        note = ((timings.get("notes") or {}) if isinstance(timings, dict) else {}).get(OVERLAP_NOTE)
        entries = _entries(note) if note is not None else []
        if entries:
            states.append(any(overlap_on(entry) for entry in entries))
    return any(states) if states else None


def first_cancel(run: dict) -> dict | None:
    """The first segment's cancel record of a resumed run (the bench copies it into the second's summary)."""
    resume = run["summary"].get("resume") or {}
    if isinstance(resume.get("first_cancel"), dict):
        return resume["first_cancel"]
    path = Path(run["folder"]) / "segments" / "1-summary.json"
    return (_read(path).get("cancel") if path.is_file() else None) or None


def resumed_problems(run: dict) -> list:
    """Criteria 2-3 shared by the calibration and the check: one resume (two segments) that succeeded, after a valid
    cancel."""
    problems = []
    segments = (run["summary"].get("resume") or {}).get("segments") or []
    if len(segments) != 2:
        problems.append(f"{len(segments)} segment(s): a cancel run resumed once has two (prereg 6.4 criterion 3)")
    cancel = first_cancel(run)
    if not cancel:
        problems.append("no cancel record of the first segment")
    elif not cancel.get("valid"):
        problems.append(f"the first segment's cancel was invalid: {cancel.get('invalid_reasons')}")
    return problems


def resume_allowance(run_dir, reference_dir, compare_path, cancel_point: str) -> dict:
    """R_resume of one cancel point, frozen from K (prereg 6.4): the K run cancelled there and resumed once, compared
    with the uninterrupted K reference."""
    run, ref = load_run(run_dir), load_run(reference_dir)
    report = _read(compare_path)
    if not cancel_point or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", cancel_point):
        raise UsageError("--cancel-point must name the cancel point, e.g. C1 or C2-hx2")
    problems, _ = _report_basics(report, run, ref, False)
    problems.extend(fatal_report_items(report))
    problems.extend(resumed_problems(run))
    if run["summary"].get("state") != "succeeded":
        problems.append(f"the resumed K run ended {run['summary'].get('state')!r}")
    verifier, sa, sb = _sides(run, ref)
    data, compared = data_differences(sa, sb)
    if data:
        problems.append(f"data digests differ in K itself (criterion 4 cannot hold): {data[:10]}")
    left = leftovers(sb, sa)
    # K's serial scan, cancelled while it ran (C7 on K, prereg 6.4: "串行扫描进行中、子进程存活时取消"), keeps its
    # scan-runs folder and the resume adds another: the residue O-1's marker protocol removes. Recorded with the
    # calibration (k_leftovers): a P1c run may show only that same residue (O-1 off: the serial scan unchanged). A
    # .partial-* entry refuses the calibration.
    k_scan_left = [name for name in left if SCAN_OUTPUT.search(name)]
    if len(k_scan_left) < len(left):
        problems.append(f"leftovers in K's resumed tree: {[n for n in left if n not in k_scan_left][:10]}")
    cancel = first_cancel(run) or {}
    latency = cancel.get("latency") or {}
    if latency.get("to_cancelled_s") is None:
        problems.append("K's first segment has no latency to 'cancelled'")
    allowance = {"schema": RESUME_SCHEMA, "cancel_point": cancel_point,
                 "cancel_when": (cancel.get("spec") or {}).get("text"), "profile": run["plan"].get("profile"),
                 "k_run": run["folder"], "k_reference": ref["folder"], "compare": str(compare_path),
                 "k_latency": {key: latency.get(key) for key in ("to_cancelled_s", "to_job_cancelled_s",
                                                                 "to_no_descendants_s",
                                                                 "cancelled_to_no_descendants_s")},
                 "data_files_compared": compared, "k_leftovers": k_scan_left,
                 **report_differences(report, sa, sb, verifier)}
    return {"passed": not problems, "problems": problems, "allowance": allowance}


def load_allowance(path) -> dict:
    allowance = _read(path)
    if allowance.get("schema") != RESUME_SCHEMA:
        raise UsageError(f"{path} is not a {RESUME_SCHEMA} file (check_p1c_run.py resume-allowance writes one)")
    return allowance


def check_compare_resume(report: dict, run: dict, ref: dict, allowance: dict, extra_listed: list,
                         allow_no_strict_ids: bool) -> dict:
    """Criterion 4 of prereg 6.4 against the uninterrupted reference: differences only inside R_resume, the same data
    digests, no leftovers."""
    problems, strict = _report_basics(report, run, ref, allow_no_strict_ids)
    problems.extend(fatal_report_items(report))
    verifier, sa, sb = _sides(run, ref)
    found = report_differences(report, sa, sb, verifier)
    frozen = allowance.get("different") or {}
    for name, item in found["different"].items():
        known = frozen.get(name)
        if known is None:
            problems.append(f"{name} differs ({item.get('paths') or item.get('kinds')}), outside R_resume")
        elif "paths" in item:
            extra = sorted(set(item["paths"]) - set(known.get("paths") or ()))
            if extra:
                problems.append(f"{name}: difference paths outside R_resume {extra[:10]}")
        else:
            extra = sorted(set(item["kinds"]) - set(known.get("kinds") or ()))
            if extra or known.get("class") != item.get("class"):
                problems.append(f"{name}: {extra or item.get('class')} differences outside R_resume")
    for key in ("only_in_a", "only_in_b", "dirs_only_in_a", "dirs_only_in_b"):
        extra = sorted(set(found[key]) - set(allowance.get(key) or ()))
        if extra:
            problems.append(f"{key} outside R_resume: {extra[:10]}")
    listed = _extra_listed(extra_listed)
    rest = {side: sorted(set(found["only_in_listed"][side]) - set((allowance.get("only_in_listed") or {}).get(side) or ())
                         - set(listed[side])) for side in ("a", "b")}
    # 事后 10-07: as in check_compare, the worker ledgers and logs a different scan worker count explains (P1c's O-1
    # scan runs 8 workers, the K references 4; K's own calibrations compare 4 with 4, so R_resume never holds them)
    workers = scan_worker_listing(rest, run, ref)
    for side in ("a", "b"):
        extra = [name for name in rest[side] if name not in workers[side]]
        if extra:
            problems.append(f"listed-only files of side {side} outside R_resume: {extra[:10]}")
    data, compared = data_differences(sa, sb)
    for name in data[:20]:
        problems.append(f"{name}: data digests differ (criterion 4: every FITS/TIFF/JPEG/PNG/NPY the same)")
    # the residue K's own serial scan left at this cancel point (k_leftovers) is the serial path's, unchanged when O-1
    # does not run (prereg §3.6 "没有标记时，串行的语义完全不变"): excused only when the cancelled attempt did not run
    # O-1 (no eligible overlap.o1 note; with O-1 the marker protocol leaves nothing), and then by its scan-runs
    # folders (which frames a cancelled scan had finished, its .pending names, depends on timing); anything else --
    # O-1's marker, a residue K did not leave, a .partial-* entry -- fails
    o1_cancelled = overlap_in_earlier_attempts(run)
    k_residue = set() if o1_cancelled else set(allowance.get("k_leftovers") or ())
    k_folders = {folder for folder in map(scan_run_folder, k_residue) if folder}
    left = [name for name in leftovers(sb, sa) if name not in k_residue and scan_run_folder(name) not in k_folders]
    if left:
        problems.append(f"leftovers in the final tree (work/transients, .partial-*): {left[:10]}"
                        + ("; O-1 ran in the cancelled attempt, so K's serial-scan residue is not excused"
                           if o1_cancelled and allowance.get("k_leftovers") else ""))
    return {"passed": not problems, "problems": problems, "mode": "resume", "strict_ids": strict,
            "cancel_point": allowance.get("cancel_point"), "data_files_compared": compared,
            "scan_worker_files": workers, "overlap_o1_cancelled_attempt": o1_cancelled,
            "differences": {"files": len(found["different"]),
                            **{key: len(found[key]) for key in ("only_in_a", "only_in_b", "dirs_only_in_a",
                                                                "dirs_only_in_b")}}}


def allowance_binding(run: dict, allowance: dict, *, reference: dict | None = None, cancel_point: str | None = None,
                      cancel_when_changed: bool = False, allowance_reference_differs: bool = False) -> tuple:
    """事后 10-07 (window D review): the R_resume file belongs to this run -- the expected cancel point (--cancel-point),
    the run's profile, the K calibration's trigger text (unless --cancel-when-changed names a registered P1c-only
    change) and the calibration's reference folder (unless --allowance-reference-differs, X2b's reuse of X2-sep's)."""
    problems, info = [], []
    if cancel_point is None:
        info.append("no --cancel-point: the allowance's cancel point was not checked")
    elif allowance.get("cancel_point") != cancel_point:
        problems.append(f"the allowance is for cancel point {allowance.get('cancel_point')!r}, not {cancel_point!r}")
    if "profile" in allowance and allowance["profile"] != run["plan"].get("profile"):
        problems.append(f"the allowance was calibrated with profile {allowance.get('profile')!r}, the run has "
                        f"{run['plan'].get('profile')!r}")
    spec = ((first_cancel(run) or {}).get("spec") or {}).get("text")
    if allowance.get("cancel_when") and spec != allowance.get("cancel_when"):
        (info if cancel_when_changed else problems).append(
            f"the run cancelled at {spec!r}, K's calibration at {allowance.get('cancel_when')!r}"
            + (" (--cancel-when-changed)" if cancel_when_changed else ""))
    if reference is not None and allowance.get("k_reference"):
        frozen, given = Path(allowance["k_reference"]).name, Path(reference["folder"]).name
        if frozen != given:
            (info if allowance_reference_differs else problems).append(
                f"the allowance was calibrated against {frozen}, the run is compared with {given}"
                + (" (--allowance-reference-differs)" if allowance_reference_differs else ""))
    return problems, info


def check_cancel(run: dict, allowance: dict, cross_version: bool, cancel_limit_s: float | None, *,
                 reference: dict | None = None, cancel_point: str | None = None, cancel_when_changed: bool = False,
                 allowance_reference_differs: bool = False) -> dict:
    """Criteria 1-3 of prereg 6.4 from the first segment's cancel record, and the allowance's binding to the run."""
    problems, info = resumed_problems(run), []
    bound, notes = allowance_binding(run, allowance, reference=reference, cancel_point=cancel_point,
                                     cancel_when_changed=cancel_when_changed,
                                     allowance_reference_differs=allowance_reference_differs)
    problems.extend(bound)
    info.extend(notes)
    cancel = first_cancel(run) or {}
    latency = cancel.get("latency") or {}
    k = (allowance.get("k_latency") or {}).get("to_cancelled_s")
    hx2 = run["plan"].get("profile") in HX2_PROFILES
    if cancel_limit_s is not None:
        limit, source = float(cancel_limit_s), "--cancel-limit-s"
    elif hx2:
        limit, source = HX2_CANCEL_LIMIT_S, "6900HX proxy v2: 5 s"
    elif isinstance(k, (int, float)):
        limit, source = max(CANCEL_LIMIT_S, k + CANCEL_K_MARGIN_S), f"max(3 s, K {k} s + 1 s)"
    else:
        limit, source = None, "no K latency in the allowance"
    spec = (cancel.get("spec") or {}).get("text")
    to_cancelled, quiet = latency.get("to_cancelled_s"), latency.get("cancelled_to_no_descendants_s")
    if cross_version:
        info.append("cross-version run (X1, X2, X2b): latencies reported, not judged (criterion 4 only)")
    else:
        if limit is None:
            problems.append("no cancel limit: the allowance has no K latency (recalibrate) and no --cancel-limit-s")
        elif to_cancelled is None or to_cancelled > limit:
            problems.append(f"request to 'cancelled' {to_cancelled!r} s > {limit} s ({source})")
        if quiet is None or quiet > QUIET_LIMIT_S:
            problems.append(f"descendants alive {quiet!r} s after 'cancelled' (at most {QUIET_LIMIT_S} s)")
    return {"passed": not problems, "problems": problems, "info": info, "limit_s": limit, "limit_source": source,
            "to_cancelled_s": to_cancelled, "cancelled_to_no_descendants_s": quiet, "k_to_cancelled_s": k,
            "cancel_when": spec, "cross_version": bool(cross_version)}


# --------------------------------------------------------------------------- KD-3 (prereg 4.2)

def _without_settings(text: str) -> list:
    """A Siril script's lines without its thread and memory setting lines (D4 block, pin)."""
    return [line.strip() for line in text.replace("\r\n", "\n").split("\n") if not SETTING_LINE.fullmatch(line.strip())]


def kd3(run_dir, reference_dir, compare_path) -> dict:
    """KD-3: a whole workflow of code with EFF-D4 (b) against one without it (a), compared with --no-strict-ids."""
    run, ref = load_run(run_dir), load_run(reference_dir)
    report = _read(compare_path)
    problems, _ = _report_basics(report, run, ref, True)
    problems.extend(fatal_report_items(report, strict_ids=False))
    verifier, sa, sb = _sides(run, ref)
    ids = verifier._deterministic_id_differences(sa, sb)
    kinds = sorted({item["kind"] for item in ids})
    if kinds != ["job-signature"]:
        problems.append(f"deterministic-id differences of kinds {kinds}: KD-3 expects exactly the job signature")
    found = report_differences(report, sa, sb, verifier)
    texts = {}
    for name, item in found["different"].items():
        base = name.rsplit("/", 1)[-1]
        # stack_snapshot-<id>.json is the workflow's frozen copy of the stack job's admission options (it carries the
        # contract field like job.json; found by the first real KD-3 runs, P1c window A)
        snapshot = base.startswith("stack_snapshot") and base.endswith(".json")
        if item["class"] == "json" and (base in ("job.json", "workflow.json") or snapshot):
            other = [p for p in item["paths"] if not p.endswith(D4_FIELD_SEGMENT)]
            if other:
                problems.append(f"{name}: JSON differences beyond the contract field: {other[:10]}")
            if base == "job.json" and "/json/admission_options" + D4_FIELD_SEGMENT not in item["paths"]:
                problems.append(f"{name}: admission_options{D4_FIELD_SEGMENT.replace('/', '.')} does not differ")
        elif item["class"] == "text" and (base.endswith(".ssf") or SIRIL_INI.fullmatch(base)):
            texts[name] = base
            if base.endswith(".ssf"):
                ea, eb = sa.files[sa.by_name[name][0]], sb.files[sb.by_name[name][0]]
                lines_a = _without_settings(((ea.get("content") or {}).get("text") or ""))
                lines_b = _without_settings(((eb.get("content") or {}).get("text") or ""))
                if [sa.canon.string(x) for x in lines_a] != [sb.canon.string(x) for x in lines_b]:
                    problems.append(f"{name}: differs beyond its thread / memory setting lines")
        else:
            problems.append(f"{name}: {item.get('paths') or item.get('kinds')} differ ({item['class']}); KD-3 expects "
                            "only the contract field, 02_stack.ssf, siril*.ini and pin lines")
    if not any(base == "02_stack.ssf" for base in texts.values()):
        problems.append("no 02_stack.ssf difference: the D4 block is missing from one side's stack script?")
    for key in ("only_in_a", "only_in_b"):
        other = [name for name in found[key] if not SIRIL_INI.fullmatch(name.rsplit("/", 1)[-1])]
        if other:
            problems.append(f"{key}: {other[:10]} (only a Siril configuration file such as siril-stack.ini may be new)")
    for key in ("dirs_only_in_a", "dirs_only_in_b"):
        if found[key]:
            problems.append(f"{key}: {found[key][:10]}")
    data, compared = data_differences(sa, sb)
    for name in data[:20]:
        problems.append(f"{name}: data digests differ (KD-3 expects bit-identical pixels)")
    return {"schema": KD3_SCHEMA, "run": run["folder"], "reference": ref["folder"], "compare": str(compare_path),
            "passed": not problems, "problems": problems, "id_differences": ids, "text_differences": sorted(texts),
            "json_differences": {name: item["paths"] for name, item in found["different"].items() if "paths" in item},
            "data_files_compared": compared}


# --------------------------------------------------------------------------- byte-level supplements

def _skip_path(name: str, path: str, verifier) -> bool:
    return verifier._diagnostic_slot(name, path, verifier.DIAGNOSTIC_SLOTS) is not None


def key_order_differences(a, b, name: str, canon_a, canon_b, verifier, path: str = "", out=None, limit=40) -> list:
    """Paths where two JSON documents order their common keys differently (timing receipts and diagnostic slots
    aside; a key on one side only is the comparison's business, e.g. a resumed run's R_resume)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        def keys(doc, canon):
            return [(canon.string(str(k)), k) for k in doc if k not in TIMING_KEYS
                    and not _skip_path(name, f"{path}/{k}", verifier)]
        ka, kb = keys(a, canon_a), keys(b, canon_b)
        common = {k for k, _ in ka} & {k for k, _ in kb}
        if [k for k, _ in ka if k in common] != [k for k, _ in kb if k in common]:
            out.append({"file": name, "path": path or "/", "a": [k for k, _ in ka][:20], "b": [k for k, _ in kb][:20]})
        raw_b = dict(kb)
        for canonical, raw in ka:
            if canonical in raw_b:
                key_order_differences(a[raw], b[raw_b[canonical]], name, canon_a, canon_b, verifier,
                                      f"{path}/{raw}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        for index, (x, y) in enumerate(zip(a, b)):
            key_order_differences(x, y, name, canon_a, canon_b, verifier, f"{path}[{index}]", out, limit)
    return out


LAYOUT_TAGS = ("StripOffsets", "StripByteCounts", "RowsPerStrip", "TileOffsets", "TileByteCounts")


def tiff_layout_differences(ea: dict, eb: dict, name: str) -> list:
    pages_a = (ea.get("content") or {}).get("pages") or []
    pages_b = (eb.get("content") or {}).get("pages") or []
    out = []
    if len(pages_a) != len(pages_b):
        out.append({"file": name, "pages": [len(pages_a), len(pages_b)]})
    for index, (x, y) in enumerate(zip(pages_a, pages_b)):
        da, db = x.get("dropped") or {}, y.get("dropped") or {}
        for tag in LAYOUT_TAGS:
            if da.get(tag) != db.get(tag):
                out.append({"file": name, "page": index, "tag": tag, "a": da.get(tag), "b": db.get(tag)})
    return out


def check_bytes(run: dict, ref: dict) -> dict:
    verifier = verifier_module()
    sa, sb = verifier.Side(ref["manifest"], "a"), verifier.Side(run["manifest"], "b")
    problems, json_pairs, tiff_pairs, ambiguous = [], 0, 0, []
    for name in sorted(set(sa.by_name) & set(sb.by_name)):
        rels_a, rels_b = sa.by_name[name], sb.by_name[name]
        if len(rels_a) != 1 or len(rels_b) != 1:
            ambiguous.append(name)
            continue
        ea, eb = sa.files[rels_a[0]], sb.files[rels_b[0]]
        if ea.get("class") != eb.get("class"):
            continue
        if ea.get("class") == "json":
            doc_a, doc_b = (ea.get("content") or {}).get("doc"), (eb.get("content") or {}).get("doc")
            if doc_a is None or doc_b is None:
                continue
            json_pairs += 1
            for difference in key_order_differences(doc_a, doc_b, name, sa.canon, sb.canon, verifier):
                problems.append(f"{name} {difference['path']}: key order {difference['a']} vs {difference['b']}")
        elif ea.get("class") == "tiff":
            tiff_pairs += 1
            for difference in tiff_layout_differences(ea, eb, name):
                if "pages" in difference:
                    problems.append(f"{name}: {difference['pages'][0]} vs {difference['pages'][1]} pages")
                else:
                    problems.append(f"{name} page {difference['page']}: {difference['tag']} differs")
    return {"passed": not problems, "problems": problems, "json_pairs": json_pairs, "tiff_pairs": tiff_pairs,
            "ambiguous_names": ambiguous[:40]}


# --------------------------------------------------------------------------- info (never fails)

def info_section(run: dict) -> dict:
    summary = run["summary"]
    foreign = summary.get("foreign_cpu") or {}
    governor = {}
    for label, notes in receipts(run):
        note = notes.get("governor")
        if not isinstance(note, dict):
            continue
        for stage, entry in (note.get("pools") or {}).items():
            if isinstance(entry, str):
                try:
                    entry = json.loads(entry)
                except ValueError:
                    entry = {}
            site = stage[3:] if stage.startswith("py.") else stage
            actual_notes = [e for n in pool_notes(run) if n["site"] == site for e in n["entries"] if isinstance(e, dict)]
            governor[f"{label}:{stage}"] = {"recommended": entry.get("recommended"), "actual": entry.get("actual"),
                                            "bound_by": entry.get("bound_by"),
                                            "pool_workers": [e.get("workers") for e in actual_notes] or None}
    hooks = summary.get("hook_events") or {}
    warnings = []
    if foreign.get("over_one_core"):
        warnings.append(f"foreign CPU {foreign.get('foreign_cores_mean')} cores on average: rerun once (prereg 6.6)")
    views = hooks.get("views") or {}
    if views and not views.get("ok", True):
        warnings.append(f"memory views: {views.get('never_seen')} never applied")
    return {"foreign_cpu": foreign, "governor_shadow": governor, "views": views,
            "openblas": sorted({(b.get("corename"), b.get("num_threads")) for b in hooks.get("openblas") or []},
                               key=str),
            "warnings": warnings}


# --------------------------------------------------------------------------- main

def _pairs(items, what):
    result = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise UsageError(f"{what} must look like KEY=VALUE: {item!r}")
        result[key] = value
    return result


def note_fields(renames=(), optional=()) -> dict:
    fields = {name: tuple(aliases) for name, aliases in NOTE_FIELDS.items()}
    for name, keys in _pairs(renames, "--note-field").items():
        if name not in fields:
            raise UsageError(f"--note-field knows {', '.join(fields)}, not {name!r}")
        fields[name] = tuple(k for k in keys.split("|") if k)
    for name in optional or ():
        if name not in fields:
            raise UsageError(f"--optional-note-field knows {', '.join(fields)}, not {name!r}")
        fields.pop(name)
    return fields


def site_caps(items) -> dict:
    caps = {}
    for site, value in _pairs(items, "--site-max").items():
        if site not in SITES or not value.isdigit() or int(value) < 1:
            raise UsageError(f"--site-max needs a site of {', '.join(SITES)} and a positive number: {site}={value}")
        caps[site] = int(value)
    return caps


def check(run_dir, reference_dir, compare_path, *, expect_env=(), expect_pin=(), expect_parallel=None,
          allow_pins=False, allow_serial_reasons=(), extra_listed=(), allow_no_strict_ids=False, note_renames=(),
          optional_note_fields=(), site_max=(), d4_version=D4_VERSION, before_contracts=(), resume_allowance_path=None,
          cross_version=False, cancel_limit_s=None, cancel_point=None, cancel_when_changed=False,
          allowance_reference_differs=False) -> dict:
    run, ref = load_run(run_dir), load_run(reference_dir)
    report = _read(compare_path)
    env = {k.upper(): (None if v == "unset" else v) for k, v in _pairs(expect_env, "--expect-env").items()}
    pins = {k: json.loads(v) for k, v in _pairs(expect_pin, "--expect-pin").items()}
    if expect_parallel is not None and expect_parallel not in ("serial", "default", "max", "custom"):
        raise UsageError("--expect-parallel must be serial, default, max or custom")
    listed = []
    for item in extra_listed:
        side, separator, name = item.partition(":")
        if not separator or side not in ("a", "b"):
            raise UsageError(f"--extra-listed must look like b:NAME, not {item!r}")
        listed.append((side, name))
    mode = run["plan"].get("mode")
    if mode not in ("L", "F", "R"):
        raise UsageError(f"unknown run mode {mode!r}")
    kind = job_kind(run)
    if kind is not None and kind not in JOB_KINDS:
        raise UsageError(f"unknown --job kind {kind!r} (known: {', '.join(JOB_KINDS)})")
    caps = site_caps(site_max)
    if any(item not in ("d4", "nox") for item in before_contracts):
        raise UsageError("--before-contract knows d4 and nox")
    if (cross_version or cancel_limit_s is not None or cancel_point is not None or cancel_when_changed
            or allowance_reference_differs) and resume_allowance_path is None:
        raise UsageError("--cross-version, --cancel-limit-s, --cancel-point, --cancel-when-changed and "
                         "--allowance-reference-differs go with --resume-allowance (a cancel and resume run)")
    allowance = load_allowance(resume_allowance_path) if resume_allowance_path is not None else None
    reasons = tuple(ALLOWED_SERIAL_REASONS) + tuple(allow_serial_reasons)
    sections = {"plan": check_plan(run, ref, env, pins, expect_parallel, allow_pins),
                "parallel": check_parallel(run, reasons, caps, ("workers", *allow_serial_reasons)),
                "notes": check_notes(run, ref, note_fields(note_renames, optional_note_fields)),
                "threads": check_threads(run),
                "contract": check_contract(run, d4_version, tuple(before_contracts)),
                "siril": check_siril(run),
                "compare": check_compare(report, run, ref, listed, allow_no_strict_ids) if allowance is None
                else check_compare_resume(report, run, ref, allowance, listed, allow_no_strict_ids),
                "bytes": check_bytes(run, ref)}
    if allowance is not None:
        sections["cancel"] = check_cancel(run, allowance, cross_version, cancel_limit_s, reference=ref,
                                          cancel_point=cancel_point, cancel_when_changed=cancel_when_changed,
                                          allowance_reference_differs=allowance_reference_differs)
    passed = all(section["passed"] for section in sections.values())
    coverage = {"mode": JOB_KINDS.get(kind, mode), "parallel": sections["parallel"]["mode"], "passed": passed,
                "run_id": run["summary"].get("run_id"), "sites": sections["parallel"]["sites"]}
    return {"schema": SCHEMA, "run": run["folder"], "reference": ref["folder"], "compare": str(compare_path),
            "mode": mode, "job": kind, **sections, "info": info_section(run), "coverage": coverage,
            "sections": list(sections), "passed": passed}


def expected_combos(items=None) -> list:
    """--expect-combo MODE:PARALLEL values (default EXPECTED_COMBOS) as (stack mode, parallel mode) pairs."""
    if not items:
        return list(EXPECTED_COMBOS)
    modes = sorted({mode for entry in SITES.values() for mode in entry["modes"]})
    combos = []
    for item in items:
        stack_mode, separator, parallel = item.partition(":")
        if not separator or stack_mode not in modes or parallel not in ("default", "max"):
            raise UsageError(f"--expect-combo must look like MODE:PARALLEL with MODE one of {', '.join(modes)} and "
                             f"PARALLEL default or max: {item!r}")
        combos.append((stack_mode, parallel))
    return combos


def coverage(results, site_max=(), expect_combos=None) -> dict:
    """Across accepted runs: did every site reach its planned maximum at least once per stack and parallel mode, and
    did every expected combination have an accepted run at all?"""
    caps = site_caps(site_max)
    expected = expected_combos(expect_combos)
    accepted = [r for r in results if r.get("passed")]
    ran = {(r["mode"], r["parallel"]) for r in accepted if r.get("parallel") in ("default", "max")}
    table, problems, info = {}, [], []
    for stack_mode, parallel in sorted(set(expected) - ran):
        failed = sum(1 for r in results if (r.get("mode"), r.get("parallel")) == (stack_mode, parallel)
                     and not r.get("passed"))
        problems.append(f"{stack_mode}/{parallel}: no accepted run" + (f" ({failed} failed)" if failed else "")
                        + ", so its sites were never shown at their planned maxima")
    for site, entry in SITES.items():
        for stack_mode, parallel in sorted(ran):
            if stack_mode not in entry["modes"]:
                continue
            runs = [r for r in accepted if (r["mode"], r["parallel"]) == (stack_mode, parallel)]
            seen = [((r.get("sites") or {}).get(site), r.get("run_id")) for r in runs]
            seen = [(s, run_id) for s, run_id in seen if s]
            key = f"{site}/{stack_mode}/{parallel}"
            reached = [run_id for s, run_id in seen if s.get("reached")
                       or (parallel == "default" and site in caps and s.get("max_workers", 0)
                           >= min(caps[site], s.get("max_units") or caps[site]))]
            table[key] = {"runs": len(runs), "ran_in": [run_id for _, run_id in seen], "reached_in": reached,
                          "max_workers": max((s.get("max_workers", 0) for s, _ in seen), default=None),
                          "planned_max": caps.get(site, entry["default"]) if parallel == "default"
                          else ("bench explicit" if entry.get("processes") else MAX_FRAME_POOL_THREADS)}
            if not seen:
                (info if entry.get("conditional") else problems).append(
                    f"{key}: never ran in {len(runs)} accepted run(s)"
                    + (f" ({entry.get('note')})" if entry.get("note") else ""))
            elif not reached:
                problems.append(f"{key}: ran in {len(seen)} run(s) but never at its planned maximum "
                                f"{table[key]['planned_max']} (max workers {table[key]['max_workers']})")
    unregistered = sorted({site for r in accepted for site in (r.get("sites") or {}) if site not in SITES})
    if unregistered:
        info.append(f"sites outside the table (reported only): {', '.join(unregistered)}")
    return {"schema": COVERAGE_SCHEMA, "accepted_runs": len(accepted), "ignored_runs": len(results) - len(accepted),
            "expected_combinations": [list(c) for c in expected], "combinations": [list(c) for c in sorted(ran)],
            "sites": table, "problems": problems, "info": info, "passed": not problems}


def _write_result(path, result):
    if path:
        Path(path).write_bytes(json.dumps(result, indent=1, ensure_ascii=False).encode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("check", help="one run against its K reference")
    one.add_argument("--run", type=Path, required=True, help="the P1c bench result folder")
    one.add_argument("--reference", type=Path, required=True, help="the K reference bench result folder")
    one.add_argument("--compare", type=Path, required=True, help="compare report, a = reference, b = run")
    one.add_argument("--expect-env", action="append", default=[], metavar="KEY=VALUE",
                     help="a variable meant to differ from the reference ('unset' = absent)")
    one.add_argument("--expect-pin", action="append", default=[], metavar="KEY=JSON",
                     help="a plan pins value meant to differ from the reference")
    one.add_argument("--expect-parallel", help="the run's parallel mode when it differs from the reference's")
    one.add_argument("--allow-pins", action="store_true",
                     help="fixed parameters are intended (contract comparisons KN-2, A1b)")
    one.add_argument("--allow-serial-reason", action="append", default=[], metavar="REASON",
                     help="another pre-registered reason for a serial pool")
    one.add_argument("--extra-listed", action="append", default=[], metavar="SIDE:NAME",
                     help="a listed-only file an item adds (named by its prereg)")
    one.add_argument("--allow-no-strict-ids", action="store_true",
                     help="the report ran with --no-strict-ids (KD-3 has its own subcommand, kd3)")
    one.add_argument("--note-field", action="append", default=[], metavar="NAME=KEY[|KEY]",
                     help=f"the keys of a required note field ({', '.join(NOTE_FIELDS)})")
    one.add_argument("--optional-note-field", action="append", default=[], metavar="NAME")
    one.add_argument("--site-max", action="append", default=[], metavar="SITE=N",
                     help="a site's measured cap under 'default' (e.g. strict_support=6)")
    one.add_argument("--d4-version", default=D4_VERSION)
    one.add_argument("--before-contract", action="append", default=[], choices=("d4", "nox"),
                     help="the run's code predates this contract (A1b, KN-2, X2): d4 = no D4 anywhere, nox = the "
                          "request's thread count is not checked")
    one.add_argument("--resume-allowance", type=Path, metavar="R_RESUME.json",
                     help="a cancel and resume run (C1-C8, X1, X2, X2b): R_resume of its cancel point, frozen on K "
                          "with the resume-allowance subcommand")
    one.add_argument("--cross-version", action="store_true",
                     help="with --resume-allowance: the first segment ran other code (X1, X2, X2b); the latencies "
                          "are reported, not judged")
    one.add_argument("--cancel-limit-s", type=float,
                     help="with --resume-allowance: the pre-registered limit to 'cancelled' of this cancel point "
                          "(a cancel in O-1's overlap window: the rest of G-3, prereg 5 item 5) instead of "
                          "max(3 s, K + 1 s), or 5 s on hx2")
    one.add_argument("--cancel-point", help="with --resume-allowance: the allowance must be for this cancel point")
    one.add_argument("--cancel-when-changed", action="store_true",
                     help="with --resume-allowance: the run's trigger text may differ from K's calibration (a "
                          "registered P1c-only change; reported, not judged)")
    one.add_argument("--allowance-reference-differs", action="store_true",
                     help="with --resume-allowance: the allowance was calibrated against another reference "
                          "(registered reuse, e.g. X2b with X2-sep's)")
    one.add_argument("--json", type=Path, help="write the full result here (input of 'coverage')")
    many = commands.add_parser("coverage", help="planned maxima reached across accepted runs")
    many.add_argument("results", nargs="+", type=Path, help="--json results of 'check'")
    many.add_argument("--site-max", action="append", default=[], metavar="SITE=N")
    many.add_argument("--expect-combo", action="append", default=[], metavar="MODE:PARALLEL",
                      help="a (stack mode, parallel mode) that needs an accepted run; given, it replaces the W "
                           f"table's {', '.join(f'{m}:{p}' for m, p in EXPECTED_COMBOS)}")
    many.add_argument("--json", type=Path)
    calibrate = commands.add_parser("resume-allowance", help="freeze R_resume of one cancel point from K")
    calibrate.add_argument("--run", type=Path, required=True, help="K's resumed bench result folder (second segment)")
    calibrate.add_argument("--reference", type=Path, required=True, help="the uninterrupted K bench result folder")
    calibrate.add_argument("--compare", type=Path, required=True, help="compare report, a = reference, b = run")
    calibrate.add_argument("--cancel-point", required=True, help="C1 ... C8, X1, X2, X2b (with a host suffix if any)")
    calibrate.add_argument("--out", type=Path, required=True, help="the R_resume file (refused if it exists)")
    contract = commands.add_parser("kd3", help="prereg 4.2 KD-3: EFF-D4 code against code without it")
    contract.add_argument("--run", type=Path, required=True, help="the bench result folder of the code with EFF-D4")
    contract.add_argument("--reference", type=Path, required=True, help="the bench result folder without it")
    contract.add_argument("--compare", type=Path, required=True,
                          help="compare report made with --no-strict-ids, a = reference, b = run")
    contract.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "coverage":
            results = []
            for path in args.results:
                loaded = _read(path)
                if loaded.get("schema") != SCHEMA or "coverage" not in loaded:
                    raise UsageError(f"{path} is not a {SCHEMA} result")
                results.append(loaded["coverage"])
            result = coverage(results, args.site_max, args.expect_combo)
            _write_result(args.json, result)
            print(json.dumps({"passed": result["passed"], "problems": result["problems"], "info": result["info"]},
                             ensure_ascii=False, indent=1))
            return 0 if result["passed"] else 1
        if args.command == "resume-allowance":
            if args.out.exists():
                raise UsageError(f"{args.out} exists: an R_resume set is frozen once")
            result = resume_allowance(args.run, args.reference, args.compare, args.cancel_point)
            if result["passed"]:
                _write_result(args.out, result["allowance"])
            print(json.dumps({"passed": result["passed"], "problems": result["problems"],
                              "out": str(args.out) if result["passed"] else None}, ensure_ascii=False, indent=1))
            return 0 if result["passed"] else 1
        if args.command == "kd3":
            result = kd3(args.run, args.reference, args.compare)
            _write_result(args.json, result)
            print(json.dumps({"passed": result["passed"], "problems": result["problems"]}, ensure_ascii=False,
                             indent=1))
            return 0 if result["passed"] else 1
        result = check(args.run, args.reference, args.compare, expect_env=args.expect_env,
                       expect_pin=args.expect_pin, expect_parallel=args.expect_parallel, allow_pins=args.allow_pins,
                       allow_serial_reasons=args.allow_serial_reason, extra_listed=args.extra_listed,
                       allow_no_strict_ids=args.allow_no_strict_ids, note_renames=args.note_field,
                       optional_note_fields=args.optional_note_field, site_max=args.site_max,
                       d4_version=args.d4_version, before_contracts=args.before_contract,
                       resume_allowance_path=args.resume_allowance, cross_version=args.cross_version,
                       cancel_limit_s=args.cancel_limit_s, cancel_point=args.cancel_point,
                       cancel_when_changed=args.cancel_when_changed,
                       allowance_reference_differs=args.allowance_reference_differs)
    except UsageError as exc:
        print(f"check_p1c_run: {exc}", file=sys.stderr)
        return 2
    _write_result(args.json, result)
    sections = result["sections"]
    print(json.dumps({"passed": result["passed"], **{name: result[name]["passed"] for name in sections},
                      "problems": {name: result[name]["problems"] for name in sections if result[name]["problems"]},
                      "warnings": result["info"]["warnings"]}, ensure_ascii=False, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
