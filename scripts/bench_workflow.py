"""Aester whole-workflow benchmark driver (efficiency P0, item 4).

Runs one complete workflow through the Python API, exactly as the GUI does
(``workflow.submit_workflow`` + its detached worker processes), inside an
isolated bench root. It never writes to C:\\Aester or %LOCALAPPDATA%\\Aester;
the user's cache is only read, once, by ``--prepare-assets``.

Layout (bench root, default outputs/perf-baseline-20260928):
  assets/                  read-only copies of the user's region drawing and its
                           reference FITS, plus per-frame-count region projects
  runs/<slot>/             one working slot per mode+frames (e.g. L20, F48):
                           c/ cache (output_root), d/ delivery_root,
                           k/ component root, h/ import hook (--code-root only),
                           x/ Siril's private XDG_DATA_HOME (see below).
                           A fixed slot path and idempotency key give repeated
                           runs identical absolute paths, so their fingerprints
                           compare. Deleted after each run unless --keep.
  results/<run-id>/        plan.json, timeline.jsonl (1 Hz), summary.json,
                           manifest.json (whole-workflow fingerprint), records/

Steps:
  --prepare-assets         copy the user's regions json, its region-assets and
                           the reference FITS it points to; rewrite paths in the
                           copy; check the copy reproduces the user's masks.
  --mode L|F|R --frames N  run (L legacy Siril; F finite_validity_v1 engine;
                           R legacy + positive_power2_v1). Inputs are the first N
                           frames of the 48-frame development baseline, each
                           verified against its recorded SHA-256 first.
  --check-environment      verify affinity inheritance and the SHA mask, cheaply.
  --dry-run                print the resolved plan (inputs are still verified).

Siril launch hygiene (efficiency P1a, harness only; recorded in the plan and
summary): every run sets XDG_DATA_HOME to the slot's x/ and keeps an inherited
MSYSTEM, else sets MINGW64. Each Siril 1.4.4 launch checks, repairs or creates
a Python venv under XDG_DATA_HOME\\siril before its script runs; with MSYSTEM
set and no venv there it stops at once. The P0/P1 runs inherited MINGW64 from
Git Bash, so this keeps their conditions from any shell, and no bench run
touches the user's %LOCALAPPDATA%\\siril (the finite build deletes a desktop
Siril's venv there). Every run also sets DBUS_SESSION_BUS_ADDRESS=disabled:
(efficiency P1b review, SF-2): a Siril of an older --code-root (before S-4d)
then neither starts nor joins the user's D-Bus session bus, whose daemon would
die with a stage's Job Object and end every GLib program on it (exit code 3);
newer code sets the same value for Siril itself. Pixels do not depend on any of
these values. --siril-desktop DIR
(efficiency P1b W5, an end-to-end check of the product's own Siril environment)
instead removes MSYSTEM and sets XDG_DATA_HOME=DIR, a sentinel desktop-Siril
folder (never the real one, under any spelling: an extended-length, device or
UNC path is refused, and so is any DIR that is, or whose siril\\ is, the same
folder as the real one); the summary lists DIR before and after the run.

Test-only knobs (all recorded in the summary): --pin-siril 'cpu=16,mem=16'
(AESTER_SIRIL_PIN), --pin-ort N (backend_options.intra_op_threads),
--affinity N (first N logical processors of this process's group; children
inherit, verified per process), --mask-sha (OPENSSL_ia32cap hides the SHA
extension; the bench re-executes itself so its own hashing is masked too),
--render-workers N (AESTER_RENDER_WORKERS; otherwise an inherited value is
cleared), --frame-workers auto|1..16|max (AESTER_FRAME_WORKERS, the P1b
frame pool; likewise cleared when not given), --code-root PATH (another
checkout, e.g. a git worktree of an older commit), --governor
shadow|shadow-full|off (AESTER_GOVERNOR, efficiency P1b-2; the bench default
is shadow-full, the product default shadow) and --governor-facts TEXT
(AESTER_GOVERNOR_FACTS: what-if machine facts for the governor's recorded
decisions only; nothing changes). Both governor keys are always set or
cleared, so an inherited value never leaks into a run (X9).
--profile NAME fills those knobs for a named consumer-PC simulation
(PROFILES); an explicit knob that disagrees with the profile is refused.
Every simulated run records plan/summary "simulation": the label for the E3
table, what is and is not limited, and the 事后 notes.

Efficiency P1c knobs (docs/EFFICIENCY_P1C_PREREG_20261005.md 6.1; recorded in the plan pins and the summary):
--parallel serial|default|max sets every parallelism variable of PARALLEL_TABLE at once (AESTER_FRAME_WORKERS,
STARSTACK_TRANSIENT_WORKERS, AESTER_RENDER_WORKERS, AESTER_HASH_WORKERS, AESTER_OVERLAP); an explicit knob for one
of them that disagrees is refused (a profile's own knob stands under 'default': it is the simulated machine's
default). Without --parallel those variables are cleared unless a knob sets them (X9). --overlap on|off
(AESTER_OVERLAP auto|off), --trails off (trails_enabled False and no trail_default: the GUI/MCP default).
--memory-view GiB / --commit-view GiB: performance.available_memory / available_commit return min(real, view) in
the bench process and in every Python process of the run (the import hook below; available_commit is wrapped when
the code root has it, its absence is logged). --python-openblas CORETYPE:THREADS sets OPENBLAS_CORETYPE and
OPENBLAS_NUM_THREADS for the Python processes only (the hook removes them from Siril's environment block).
--cancel-when RECORD:FIELD=VALUE[+DELAY] cancels the workflow DELAY seconds after the field of workflow.json or of
the stack job's job.json first equals VALUE ('*' = present and not null); it polls every 0.1 s through
records.read_json and records two latencies: to the 'cancelled' state and to no live descendant. A cancel whose
workflow never reaches 'cancelled', or whose descendants outlive --quiesce-seconds (the bench's Job Object kills
them when it exits), is invalid, and --resume refuses its slot. --resume continues the slot kept by a valid
--cancel-when run (workflow.resume_workflow), possibly with another --code-root: it replays the first segment's
knobs and refuses a run whose plan, pins or effective environment differ (only the code root may). Both segments
keep their own results folder; the second copies the first's plan and summary into segments/.
--job strict-legacy (K-SL20, W19-W21) submits one stack job the way the CLI and MCP do (core.submit_job: strict
support, legacy stack, the copied sky mask, the first N frames as input_files) in its own slot S<mode><frames>, with
every other knob (--parallel, views, hook, governor) as for a workflow; records/ then holds job.json only, and the
manifest is the verifier's fingerprint of the job folder as {JOB} (fingerprint_job), which its compare and
scripts/check_p1c_run.py read like a workflow's. No --cancel-when / --resume / --pin-ort with --job.
The import hook (sitecustomize.py in the slot's h/, on PYTHONPATH) is installed in every run: it records the
starstack each process imports, applies the memory views, logs each process's OpenBLAS configuration after numpy
loads (openblas_get_config, corename, threads) and samples each process's Python threads once a second (frame-pool
threads aester-<name>_<k>, render threads, BLAS-capable threads; scripts/check_p1c_run.py checks them). The summary
also records the foreign CPU: busy CPU time of the whole machine (every processor group) minus the run tree's CPU
time (Job Object accounting) over the run, and the timeline the same per second.

P0 runs (docs/EFFICIENCY_P0_PREREG_20260928.md 1.5), from the repo root with
PY = .venv\\Scripts\\python.exe -X utf8 and S = scripts\\bench_workflow.py:
  once   PY S --prepare-assets
  R1     PY S --mode L --frames 20 --pin-siril cpu=16,mem=16 --pin-ort 8 --label R1a   (then --label R1b)
  R2     PY S --mode F --frames 48 --pin-siril cpu=16,mem=16 --pin-ort 8 --label R2a   (then --label R2b)
  R3     git worktree add --detach "..\\aester-wt-0eba79b" 0eba79b
         PY S --mode L --frames 20 --pin-siril cpu=16,mem=16 --pin-ort 8 --code-root "..\\aester-wt-0eba79b" --label R3-0eba79b
  R4     PY S --mode L --frames 48 --label R4-L1 ; --mode F --frames 48 --label R4-F1 ;
         --mode L --frames 48 --label R4-L2 ; --mode F --frames 48 --label R4-F2
  R5     PY S --mode L --frames 20 --label R5 ; --mode F --frames 20 --label R5 ; --mode R --frames 48 --label R5
  R6     PY S --mode L --frames 48 --profile ordinary16 --label R6 ; --mode F --frames 48 --profile ordinary16 --label R6
         (= --affinity 16 --mask-sha --pin-ort 8; "16 logical processors, no SHA, unlimited memory".
         事后: optimistic for a 16 GB PC and not a 6900HX stand-in; see PROFILES)
  R6x    事后, optional 6900HX proxy (SHA on, Siril mem 9 GiB, render workers 3):
         PY S --mode L --frames 48 --profile 6900hx-proxy --label R6x ; --mode F --frames 48 --profile 6900hx-proxy --label R6x
         The user's laptop run stays the memory truth.

P1c runs (prereg 6.1, 6.3, 6.4), e.g.:
  K-L20  PY S --mode L --frames 20 --label K-L20                         (no pins; shadow-full governor)
  W1     PY S --mode L --frames 20 --parallel serial --code-root <P1c worktree> --label W1
  W4     PY S --mode L --frames 20 --trails off --label W4
  hx2    PY S --mode L --frames 48 --profile 6900hx-proxy-v2 --label K-L48-hx2
  C5     PY S --mode F --frames 20 --cancel-when job:stage=stack+40 --label C5a     (calibrated on K first)
         PY S --mode F --frames 20 --resume --code-root <P1c worktree> --label C5b
  W21    PY S --mode L --frames 20 --job strict-legacy --parallel max --code-root <P1c worktree> --label W21
then scripts/check_p1c_run.py check --run results/<W1> --reference results/<K-L20> --compare <report> ...
(a cancel run: check ... --resume-allowance <R_resume of its cancel point>, frozen first with resume-allowance)
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_BENCH_ROOT = REPO / "outputs" / "perf-baseline-20260928"
DEFAULT_MANIFEST = REPO / "docs" / "testdata" / "DEVELOPMENT_BASELINE_48_20260925.json"
DEFAULT_INPUTS = REPO / "outputs" / "efficiency-20260919" / "accept-efficient-48-20260919T165004Z-87d540" / "inputs"
USER_REGIONS_REVISION = "f17619dd0ec34b9ef9ed78fc10c025bc0b6fb961269649082d80bfa3a9409d6f"
SIRIL = REPO / "tools" / "siril" / "bin" / "siril-cli.exe"
FINITE_ENGINE = REPO / "tools" / "finite-siril" / "finite-validity-v1" / "siril-cli.exe"
NOX_MODEL = REPO / "outputs" / "starless-acceptance-20260923" / "models" / "nox_color.onnx"
VERIFIER = SCRIPTS / "verify_workflow_equivalence.py"

SUMMARY_SCHEMA = "aester-bench-summary-1"
ASSETS_SCHEMA = "aester-bench-assets-1"
SLOT_MARKER = ".aester-bench-slot.json"
SLOT_NAME = re.compile(r"(?:SL|[LFR])[0-9]{1,2}")   # SL: the --job strict-legacy slot (e.g. SL20)
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
SHA_MASK = ":~0x20000000"      # CPUID leaf 7 EBX bit 29 (SHA extensions) hidden from OpenSSL
SIRIL_PIN_ENV = "AESTER_SIRIL_PIN"
REEXEC_MARK = "AESTER_BENCH_REEXEC"
HOOK_LOG_ENV = "AESTER_BENCH_HOOK_LOG"
HOOK_SHIM_ENV = "AESTER_BENCH_PIN_SHIM"
GIB = 1024 ** 3
MAX_CACHE_CHARS = 120
# Measured working-set sizes per frame (user's L20 29.4 GB, F48 80.8 GB) plus margin.
DISK_PER_FRAME = {"L": 1.6 * GIB, "R": 1.6 * GIB, "F": 1.85 * GIB}

MODES = {
    "L": {"name": "legacy", "stack_range_mode": "legacy", "stack_engine_path": None},
    "F": {"name": "finite", "stack_range_mode": "finite_validity_v1", "stack_engine_path": FINITE_ENGINE},
    "R": {"name": "range", "stack_range_mode": "positive_power2_v1", "stack_engine_path": None},
}
# --job (P1c prereg 6.1 K-SL20, 6.3 W19-W21): one stack job submitted the way the CLI ('run') and MCP ('submit_job')
# submit it, core.submit_job, instead of a workflow. Its own slot (S + mode + frames, e.g. SL20) and idempotency key;
# the first N baseline frames as input_files (the CLI's folder form would take all 48); the user's sky mask from the
# prepared assets (verified to equal the drawing's rasterisation); transient_mode as the CLI leaves it, or 'off' with
# --trails off. strict-legacy: strict support with the legacy stack, so retain_frame_support stays False (prereg 3.2).
JOB_KINDS = {"strict-legacy": {"mode": "L", "support_mode": "strict_siril_1_4_4", "stack_range_mode": "legacy"}}
JOB_SLOT_PREFIX = "S"
# The user's 2026-09-28 run (GUI): the nox adapter as the GUI sends it, plus
# the visible settings. compose_feather_mode follows from the region project
# ('canvas-inward-feather-1'); it is not a submit parameter.
# jpeg_quality is deliberately not sent although the GUI always sends its slider value (2026-10-09): the existing
# reference runs were made without it, and with it unset the request digest, child job and every output byte stay
# those references'. Setting it would only re-encode the delivered JPEGs (a new reference set would be needed).
NOX_OPTIONS = {"tile": 512, "stride": 256, "batch": 1, "layout": "nhwc", "channel_order": "bgr",
               "value_range": "-1..1", "stitch": "crop", "intra_op_threads": None}
USER_SETTINGS = {"background_strength": 0.85, "milkyway_offset": -0.1, "trails_enabled": True,
                 "trail_default": "meteor", "stars_enabled": True, "milkyway_enabled": True,
                 "milkyway_strength": 0.75, "save_starless": True, "save_star_layer": True,
                 "color_adjustments": {"warmth": -5.0, "tint": -6.0, "exposure_ev": 0.3},
                 "delivery_format": "aester-delivery-2", "image_rotation": 0}
RELEVANT_ENV = re.compile(r"^(AESTER_|STARSTACK_|OMP_|OPENBLAS_|MKL_|OPENSSL_|ORT_|PYTHON|MSYSTEM$|XDG_DATA_HOME$)",
                          re.IGNORECASE)
# Siril launch hygiene (module docstring): an inherited MSYSTEM is kept, else this default.
DEFAULT_MSYSTEM = "MINGW64"
# ... and no D-Bus session bus for any Siril of the run, old code roots included (siril._NO_SESSION_BUS, S-4d).
NO_SESSION_BUS = "disabled:"
RENDER_WORKERS_ENV = "AESTER_RENDER_WORKERS"
MAX_RENDER_WORKERS = 8          # performance.MAX_RENDER_WORKERS
FRAME_WORKERS_ENV = "AESTER_FRAME_WORKERS"      # frame_pool.ENV_WORKERS (P1b-1)
MAX_FRAME_WORKERS = 16          # frame_pool.MAX_FRAME_WORKERS
# Resource governor (efficiency P1b-2, shadow only): bench runs record the full machine facts (U-G1: the product
# default is "shadow", which writes no machine inventory, U-G4). The what-if facts change no actual setting.
GOVERNOR_ENV = "AESTER_GOVERNOR"
GOVERNOR_FACTS_ENV = "AESTER_GOVERNOR_FACTS"
GOVERNOR_MODES = ("shadow", "shadow-full", "off")
BENCH_GOVERNOR_DEFAULT = "shadow-full"
# The governor's what-if laptop (A6): a 16 GiB laptop as Windows shows it, 9 GiB free, 8 cores / 16 logical
# processors, Siril's pool 16, commit limit 24 GiB with 14 GiB available, NVMe (governor final.md §3.3, §6 A6).
LAPTOP_GOVERNOR_FACTS = "total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,io=nvme"
_GOVERNOR_FACT = re.compile(r"[a-z_]+=[0-9A-Za-z.]+")
FRAME_HW = (5520, 8280)         # Siril-converted Z8 mosaic (devices_registry shape_source); render-worker estimate only

# Efficiency P1c (prereg 6.1): one switch for every parallelism variable. None = the variable is cleared (the
# product default); 'auto' is the same default written out, as the prereg's table has it.
TRANSIENT_WORKERS_ENV = "STARSTACK_TRANSIENT_WORKERS"   # transient_stage.WORKERS_ENV (T-2 scan processes)
HASH_WORKERS_ENV = "AESTER_HASH_WORKERS"                 # hashing.ENV_WORKERS
OVERLAP_ENV = "AESTER_OVERLAP"                           # frame_pool (P1c O-1): auto | off
MAX_SCAN_WORKERS = 8            # T-2 SCAN_WORKER_CAP: 'max' names this explicit number (the override is not capped)
MAX_HASH_WORKERS = 32           # hashing.MAX_WORKERS: hashing reads integers only, so 'max' is written as this number
PARALLEL_MODES = ("serial", "default", "max")
PARALLEL_VARIABLES = (FRAME_WORKERS_ENV, TRANSIENT_WORKERS_ENV, RENDER_WORKERS_ENV, HASH_WORKERS_ENV, OVERLAP_ENV)
PARALLEL_TABLE = {
    "serial": {FRAME_WORKERS_ENV: "1", TRANSIENT_WORKERS_ENV: "1", RENDER_WORKERS_ENV: "1", HASH_WORKERS_ENV: "1",
               OVERLAP_ENV: "off"},
    "default": {FRAME_WORKERS_ENV: None, TRANSIENT_WORKERS_ENV: None, RENDER_WORKERS_ENV: "auto", HASH_WORKERS_ENV: None,
                OVERLAP_ENV: "auto"},
    "max": {FRAME_WORKERS_ENV: "max", TRANSIENT_WORKERS_ENV: str(MAX_SCAN_WORKERS),
            RENDER_WORKERS_ENV: str(MAX_RENDER_WORKERS), HASH_WORKERS_ENV: str(MAX_HASH_WORKERS), OVERLAP_ENV: "auto"},
}
# A variable's unset value and its written-out default are the same setting.
_PARALLEL_DEFAULTS = {FRAME_WORKERS_ENV: "auto", RENDER_WORKERS_ENV: "auto", OVERLAP_ENV: "auto"}
OVERLAP_VALUES = {"on": "auto", "off": "off"}
# The import hook's switches (every run installs the hook; see HOOK_SOURCE).
MEMORY_VIEW_ENV = "AESTER_BENCH_MEMORY_VIEW"             # bytes: performance.available_memory -> min(real, view)
COMMIT_VIEW_ENV = "AESTER_BENCH_COMMIT_VIEW"             # bytes: performance.available_commit -> min(real, view)
HOOK_EVENTS_ENV = "AESTER_BENCH_HOOK_EVENTS"             # folder: one JSON per hook event (views, OpenBLAS, Siril block)
THREAD_LOG_ENV = "AESTER_BENCH_THREAD_LOG"               # folder: one JSON per Python process (thread audit, 1 Hz)
PYTHON_ONLY_ENV = "AESTER_BENCH_PYTHON_ONLY"             # names removed from Siril's environment block
BLAS_LOG_ENV = "AESTER_BENCH_BLAS_LOG"                   # "1": log openblas_get_config once numpy is loaded
PYTHON_OPENBLAS_KEYS = ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS")
_PYTHON_OPENBLAS = re.compile(r"([A-Za-z][A-Za-z0-9]{1,31}):([1-9][0-9]{0,2})")
# --cancel-when RECORD:FIELD=VALUE[+DELAY] (prereg 6.1, 6.4)
_CANCEL_WHEN = re.compile(r"(?P<record>workflow|job):(?P<field>[A-Za-z0-9_][A-Za-z0-9_.\-]*)=(?P<value>[^+]*)"
                          r"(?:\+(?P<delay>[0-9]{1,5}(?:\.[0-9]{1,3})?))?")
CANCEL_POLL_SECONDS = 0.1
# The knobs a --resume replays from the first segment (those given there on the command line; --profile refills its
# own): argparse destinations.
RESUME_KNOBS = ("profile", "pin_siril", "pin_ort", "affinity", "mask_sha", "render_workers", "frame_workers",
                "governor", "governor_facts", "memory_view", "commit_view", "python_openblas", "parallel", "overlap",
                "trails", "siril_desktop")
# Environment entries that depend on the code root, not on the run (--resume may change the code root).
CODE_ROOT_KEYS = ("PYTHONPATH", HOOK_SHIM_ENV)

# Consumer-PC simulations. The prereg (1.4) defines only "ordinary16"; its memory is not limited, so
# it is optimistic for a 16 GB PC and is not a stand-in for the user's AMD 6900HX laptop, which has
# SHA. "6900hx-proxy" was added after review (事后): Siril's unpinned memory is 0.9 x free RAM, about
# 9 GiB on a 16 GB laptop with about 10 GiB free, and performance.render_workers then allows 3
# (min(avail/3, avail - 1 GiB) at about 0.95 GiB per 45.7 MP worker). The laptop run stays the truth.
PROFILES = {
    "ordinary16": {"knobs": {"affinity": 16, "mask_sha": True, "pin_ort": 8},
                   "status": "prereg R6 (1.4, 1.5); --pin-ort 8 added 事后"},
    "6900hx-proxy": {"knobs": {"affinity": 16, "mask_sha": False, "pin_ort": 8, "pin_siril": "mem=9",
                               "render_workers": 3, "governor_facts": LAPTOP_GOVERNOR_FACTS},
                     "status": "事后: not pre-registered; added after review of R6"},
    # P1c prereg 6.1: v1's affinity, Siril mem 9, render workers 3 and governor facts, plus the memory and commit
    # views of LAPTOP_GOVERNOR_FACTS (avail=9G, commit_avail=14G) and the laptop's OpenBLAS kernel and thread count
    # (Zen/Haswell, 16) in the Python processes; this host's OpenBLAS picks SkylakeX and 24. No --pin-ort: after
    # EFF-NOX (K1) the frozen request carries 8 anyway, and the K references run without fixed parameters.
    "6900hx-proxy-v2": {"knobs": {"affinity": 16, "mask_sha": False, "pin_siril": "mem=9", "render_workers": 3,
                                  "governor_facts": LAPTOP_GOVERNOR_FACTS, "memory_view": 9.0, "commit_view": 14.0,
                                  "python_openblas": "Haswell:16"},
                        "status": "prereg P1c 6.1 (6900HX proxy v2: v1 + memory views + OpenBLAS Haswell/16)"},
}


class BenchError(RuntimeError):
    """A refused or failed bench step; the message says what to fix."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def _abs(path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _inside(path, root) -> bool:
    p = os.path.normcase(str(_abs(path))).rstrip("\\/")
    r = os.path.normcase(str(_abs(root))).rstrip("\\/")
    return p == r or p.startswith(r + os.sep)


def forbidden_roots() -> list[Path]:
    """The user's installation and per-user Aester data; never written by the bench."""
    roots = [Path("C:/Aester")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Aester")
    return [_abs(r) for r in roots]


def ensure_allowed(path, what: str) -> Path:
    path = _abs(path)
    for root in forbidden_roots():
        if _inside(path, root) or _inside(root, path):
            raise BenchError(f"{what} must stay outside {root}: {path}")
    return path


# --siril-desktop (efficiency P1b W5): the product-entry check that Siril never touches a desktop Siril's folder.
def known_local_appdata() -> Path | None:
    """FOLDERID_LocalAppData, where GLib puts Siril's data folder when XDG_DATA_HOME is unset; None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes
    import uuid

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                    ("Data4", ctypes.c_ubyte * 8)]

    known = uuid.UUID("F1B32785-6FBA-4FCF-9D55-7B8E7F157091")
    guid = GUID(known.fields[0], known.fields[1], known.fields[2], (ctypes.c_ubyte * 8)(*known.bytes[8:]))
    path = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path)) != 0:
        return None
    try:
        return Path(path.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path)


def _same_folder(a, b) -> bool:
    """True when both exist and are the same file-system object (st_dev + st_ino: any spelling, case, 8.3 name,
    junction, symbolic link or subst drive); False when either is missing or cannot be read."""
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return False


def siril_desktop_folder(value) -> Path:
    """The sentinel "desktop Siril" data folder; never the user's real one (its siril\\ is what Siril would use).

    Refused (P1b review SIRIL-3): an extended-length, device or UNC spelling (\\\\?\\..., \\\\.\\..., \\\\server\\share,
    which a sentinel never needs and which the path comparisons below cannot see through); DIR when it is the real
    LocalAppData by name or by identity; DIR\\siril when it is the real siril folder (checked when both exist); and
    DIR inside the real siril folder or inside a forbidden root (C:\\Aester, %LOCALAPPDATA%\\Aester) by identity."""
    text = os.fspath(value)
    if text.replace("/", "\\").startswith("\\\\"):
        raise BenchError(f"--siril-desktop must be a plain local folder (no \\\\?\\, \\\\.\\ or UNC spelling): {text}")
    desktop = ensure_allowed(value, "--siril-desktop")
    real = [p for p in (known_local_appdata(), os.environ.get("LOCALAPPDATA")) if p]
    for root in real:
        if (_abs(desktop) == _abs(root) or Path(desktop).resolve() == Path(root).resolve()
                or (Path(desktop) / "siril").resolve() == (Path(root) / "siril").resolve()
                or _same_folder(desktop, root) or _same_folder(Path(desktop) / "siril", Path(root) / "siril")):
            raise BenchError(f"--siril-desktop must be a sentinel folder, not the user's data folder {root}")
    guarded = [*forbidden_roots(), *(Path(root) / "siril" for root in real)]
    for part in (desktop, *desktop.parents):
        for root in guarded:
            if _same_folder(part, root):
                raise BenchError(f"--siril-desktop must stay outside {root}: {desktop} (the same folder as {part})")
    return desktop


def folder_listing(root) -> dict:
    """Relative path -> ["dir"] or [size, sha256] of everything below root ({} when it does not exist)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): ["dir"] if p.is_dir() else [p.stat().st_size, sha256_file(p)]
            for p in sorted(root.rglob("*"))} if root.is_dir() else {}


def _is_link(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _no_links_on_path(path: Path, what: str):
    for part in (path, *path.parents):
        if _is_link(part):
            raise BenchError(f"{what}: {part} is a link or junction; refused")


def sha256_file(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False, default=str)
    except ValueError:      # a non-finite number copied from a record must not lose the whole summary
        text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=True, default=str)
    if exclusive:
        with open(path, "x", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
        return
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def parse_pin(value) -> dict | None:
    """Same rules as starstack.siril.siril_pin (kept in step by tests)."""
    if value is None or not str(value).strip():
        return None
    pin = {}
    for part in str(value).strip().split(","):
        key, separator, number = part.partition("=")
        key, number = key.strip().lower(), number.strip()
        if not separator or key not in ("cpu", "mem") or key in pin or not re.fullmatch(r"[1-9][0-9]{0,4}", number):
            raise BenchError(f"--pin-siril must look like 'cpu=16,mem=16' (either part optional): {value!r}")
        pin[key] = int(number)
    return pin


def pin_text(pin: dict) -> str:
    return ",".join(f"{k}={pin[k]}" for k in ("cpu", "mem") if k in pin)


def pin_lines(pin: dict | None) -> list[str]:
    lines = []
    if pin and "cpu" in pin:
        lines.append(f"setcpu {pin['cpu']}")
    if pin and "mem" in pin:
        lines += ["set core.mem_mode=1", f"set core.mem_amount={pin['mem']}"]
    return lines


# --------------------------------------------------------------------------- inputs

def verify_inputs(manifest_path, inputs_dir, count: int) -> dict:
    """The first ``count`` baseline frames, each checked for size and SHA-256."""
    manifest_path, inputs_dir = _abs(manifest_path), _abs(inputs_dir)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = sorted(manifest["frames"], key=lambda f: int(f["frame"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BenchError(f"cannot read the baseline manifest {manifest_path}: {exc}") from exc
    if [int(f["frame"]) for f in frames] != list(range(1, len(frames) + 1)):
        raise BenchError("baseline manifest frames are not numbered 1..N")
    if not 1 <= count <= len(frames):
        raise BenchError(f"--frames must be 1..{len(frames)}")
    selected, problems = [], []
    for frame in frames[:count]:
        path = inputs_dir / Path(frame["dng_path"]).name
        entry = {"path": str(path), "sha256": frame["sha256"], "bytes": int(frame["bytes"])}
        if not path.is_file():
            problems.append(f"missing {path}")
        elif path.stat().st_size != entry["bytes"]:
            problems.append(f"size differs {path}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"SHA-256 differs {path}")
        selected.append(entry)
    if problems:
        raise BenchError("input verification failed against the baseline manifest: " + "; ".join(problems[:6])
                         + (f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""))
    return {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "inputs_dir": str(inputs_dir), "verified": len(selected), "frames": selected}


# --------------------------------------------------------------------------- assets

def _import_starstack(src: Path):
    """Import starstack from ``src`` only; refuse a different, already imported copy."""
    src = _abs(src)
    loaded = sys.modules.get("starstack")
    if loaded is not None:
        if not _inside(Path(loaded.__file__).resolve(), src):
            raise BenchError(f"starstack is already imported from {Path(loaded.__file__).parent}, not {src}")
    elif str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import starstack  # noqa: F401  (import root checked below)
    if not _inside(Path(starstack.__file__).resolve(), src):
        raise BenchError(f"starstack imported from {starstack.__file__}, expected under {src}")
    return starstack


def _copy_readonly(source: Path, target: Path) -> dict:
    """Plain byte copy (no links); the source is opened read-only and re-hashed."""
    before = source.stat()
    digest = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256_file(target) != digest:
        raise BenchError(f"source changed while copying or the copy differs: {source}")
    return {"source": str(source), "sha256": digest, "bytes": before.st_size}


def derive_project(user_project: dict, frames: list[dict], reference_copy: Path, revision_fn) -> dict:
    """The user's drawing bound to ``frames`` and the local reference copy (revision resealed)."""
    body = copy.deepcopy({k: v for k, v in user_project.items() if k not in ("revision", "project_path")})
    binding = body["source_binding"]
    if binding["reference"]["source_sha256"] != frames[0]["sha256"]:
        raise BenchError("the drawing's reference frame is not the first baseline frame")
    binding["input_files"] = [{"path": f["path"], "sha256": f["sha256"]} for f in frames]
    binding["reference"]["source_path"] = frames[0]["path"]
    binding["reference"]["linear_path"] = str(reference_copy)
    body["revision"] = revision_fn(body)
    return body


def prepare_assets(bench_root, *, source_cache=None, revision=USER_REGIONS_REVISION,
                   manifest_path=DEFAULT_MANIFEST, inputs_dir=DEFAULT_INPUTS, src=REPO / "src",
                   project_frames=(20, 48)) -> dict:
    """Copy the user's region drawing once; never writes below ``source_cache``."""
    bench_root = ensure_allowed(bench_root, "bench root")
    source_cache = _abs(source_cache or Path(os.environ.get("LOCALAPPDATA", "")) / "Aester" / "Cache")
    assets = bench_root / "assets"
    if _inside(assets, source_cache) or _inside(source_cache, assets):
        raise BenchError("assets and the source cache must not overlap")
    if (assets / "manifest.json").exists():
        return verify_assets(bench_root)
    if assets.exists():
        raise BenchError(f"{assets} exists without manifest.json; remove it by hand after checking it")
    _import_starstack(src)
    from starstack import regions
    bench_root.mkdir(parents=True, exist_ok=True)
    user_json = source_cache / f"regions-{revision}.json"
    user_masks = source_cache / "region-assets" / f"region-assets-{revision}"
    if not user_masks.is_dir():
        user_masks = source_cache / f"region-assets-{revision}"
    project = json.loads(user_json.read_text(encoding="utf-8"))
    if project.get("revision") != revision or regions._revision(project) != revision:
        raise BenchError("the user's regions json does not match its revision")
    baseline = verify_inputs(manifest_path, inputs_dir, max(len(project["source_binding"]["input_files"]),
                                                            max(project_frames)))
    for index, bound in enumerate(project["source_binding"]["input_files"]):
        if bound["sha256"] != baseline["frames"][index]["sha256"]:
            raise BenchError(f"the drawing is bound to a frame that is not baseline frame {index + 1}")
    reference = Path(project["source_binding"]["reference"]["linear_path"])
    stage = bench_root / (".partial-assets-" + os.urandom(6).hex())
    files = {}
    try:
        files["source/" + user_json.name] = _copy_readonly(user_json, stage / "source" / user_json.name)
        for name in ("sky.png", "milkyway.tif"):
            if (user_masks / name).is_file():
                files[f"source/{user_masks.name}/{name}"] = _copy_readonly(user_masks / name,
                                                                          stage / "source" / user_masks.name / name)
        files["reference/" + reference.name] = _copy_readonly(reference, stage / "reference" / reference.name)
        if files["reference/" + reference.name]["sha256"] != project["source_binding"]["reference"]["linear_sha256"]:
            raise BenchError("the reference FITS differs from the drawing's recorded digest")
        mask_check = _check_masks(regions, project, stage / "source" / user_masks.name)
        record = {"schema": ASSETS_SCHEMA, "prepared_utc": utc_now().isoformat(), "source_cache": str(source_cache),
                  "source_revision": revision, "files": files, "mask_check": mask_check,
                  "note": "Read-only copies; source files re-hashed after copying. Region projects regions-fNN.json "
                          "are the user's drawing bound to the first NN baseline frames and the local reference copy."}
        os.rename(stage, assets)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    record["files"] = {k: {**v, "copy": str(assets / k)} for k, v in files.items()}
    write_json(assets / "manifest.json", record, exclusive=True)
    for count in project_frames:
        project_file(bench_root, count, manifest_path=manifest_path, inputs_dir=inputs_dir, revision_fn=regions._revision,
                     baseline=baseline)
    return verify_assets(bench_root)


def _check_masks(regions, project: dict, masks: Path) -> dict:
    """Does the current rasteriser reproduce the user's saved masks from the copied drawing?"""
    import numpy as np
    from PIL import Image
    sky, weight = regions.rasterize(project)
    result = {"sky_png": None, "milkyway_tif": None}
    if (masks / "sky.png").is_file():
        result["sky_png"] = bool(np.array_equal(np.asarray(Image.open(masks / "sky.png")), sky))
    if (masks / "milkyway.tif").is_file():
        import tifffile
        result["milkyway_tif"] = bool(np.array_equal(tifffile.imread(masks / "milkyway.tif"), weight))
    if False in result.values():
        raise BenchError(f"the copied drawing does not reproduce the user's masks: {result}")
    return result


def verify_assets(bench_root) -> dict:
    assets = _abs(bench_root) / "assets"
    try:
        record = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"assets are not prepared ({exc}); run --prepare-assets first") from exc
    if record.get("schema") != ASSETS_SCHEMA:
        raise BenchError("unknown assets manifest")
    for rel, entry in record["files"].items():
        path = assets / rel
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise BenchError(f"asset copy missing or changed: {path}")
    return record


def reference_copy(bench_root, record=None) -> Path:
    record = record or verify_assets(bench_root)
    refs = [k for k in record["files"] if k.startswith("reference/")]
    if len(refs) != 1:
        raise BenchError("assets must hold exactly one reference FITS")
    return _abs(bench_root) / "assets" / refs[0]


def sky_mask_copy(bench_root) -> Path:
    """The user's sky mask in the prepared assets (source/region-assets-<revision>/sky.png): a --job's sky mask."""
    try:
        record = json.loads((_abs(bench_root) / "assets" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"--job needs the prepared assets' sky mask ({exc}); run --prepare-assets first") from exc
    masks = sorted(k for k in record.get("files") or {} if k.startswith("source/region-assets-")
                   and k.endswith("/sky.png"))
    if len(masks) != 1:
        raise BenchError(f"--job needs exactly one copied sky mask in the assets, found {masks}")
    return _abs(bench_root) / "assets" / masks[0]


def project_file(bench_root, count: int, *, manifest_path=DEFAULT_MANIFEST, inputs_dir=DEFAULT_INPUTS,
                 revision_fn=None, baseline=None) -> Path:
    """assets/regions-fNN.json, created once and afterwards only re-checked."""
    bench_root = _abs(bench_root)
    record = verify_assets(bench_root)
    path = bench_root / "assets" / f"regions-f{count:02d}.json"
    if revision_fn is None:
        from starstack import regions
        revision_fn = regions._revision
    source = next(bench_root / "assets" / k for k in record["files"] if k.startswith("source/regions-"))
    user_project = json.loads(source.read_text(encoding="utf-8"))
    baseline = baseline or verify_inputs(manifest_path, inputs_dir, count)
    expected = derive_project(user_project, baseline["frames"][:count], reference_copy(bench_root, record), revision_fn)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise BenchError(f"{path} differs from the derivation of the copied drawing; check it by hand")
        return path
    write_json(path, expected, exclusive=True)
    return path


# --------------------------------------------------------------------------- plan

def _git_commit(root: Path) -> dict:
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True, timeout=60)
        return {"commit": head.stdout.strip() or None, "src_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "src_dirty": None}


def native_pin_support(code_src: Path) -> bool:
    try:
        return SIRIL_PIN_ENV in (code_src / "starstack" / "siril.py").read_text(encoding="utf-8")
    except OSError:
        return False


def _same_knob(knob, given, wanted) -> bool:
    if knob == "pin_siril":
        return parse_pin(given) == parse_pin(wanted)
    if knob in ("memory_view", "commit_view"):
        try:
            return float(given) == float(wanted)
        except (TypeError, ValueError):
            return False
    if knob == "python_openblas":
        return parse_python_openblas(given) == parse_python_openblas(wanted)
    return given == wanted


def apply_profile(args):
    """Fill the knobs of ``--profile``; an explicitly given knob must agree with it.  The knobs it filled are kept
    in ``args.profile_filled`` (--parallel default lets a profile's own knob stand)."""
    name = getattr(args, "profile", None)
    args.profile_filled = []
    if not name:
        return args
    if name not in PROFILES:
        raise BenchError(f"--profile must be one of {', '.join(PROFILES)}")
    for knob, wanted in PROFILES[name]["knobs"].items():
        given = getattr(args, knob, None)
        if given is None:
            setattr(args, knob, wanted)
            args.profile_filled.append(knob)
            continue
        if not _same_knob(knob, given, wanted):
            raise BenchError(f"--{knob.replace('_', '-')} {given!r} conflicts with --profile {name} ({wanted!r})")
    return args


def parse_python_openblas(value) -> dict | None:
    """'Haswell:16' -> {'OPENBLAS_CORETYPE': 'Haswell', 'OPENBLAS_NUM_THREADS': '16'} (Python processes only)."""
    if value is None:
        return None
    match = _PYTHON_OPENBLAS.fullmatch(str(value).strip())
    if not match:
        raise BenchError(f"--python-openblas must look like CORETYPE:THREADS, e.g. Haswell:16: {value!r}")
    return dict(zip(PYTHON_OPENBLAS_KEYS, match.groups()))


def view_bytes(value, what: str) -> int | None:
    """A memory view in GiB (as given) -> bytes."""
    if value is None:
        return None
    try:
        gib = float(value)
    except (TypeError, ValueError):
        gib = float("nan")
    if not 0.25 <= gib <= 4096:
        raise BenchError(f"--{what} must be a number of GiB, 0.25..4096: {value!r}")
    return int(round(gib * GIB))


def parse_cancel_when(value) -> dict | None:
    """RECORD:FIELD=VALUE[+DELAY]; RECORD is workflow (workflow.json) or job (the stack job's job.json), FIELD a
    dotted path (list items by index), VALUE compared with the field's text (JSON text for numbers, true/false/null;
    '*' = present and not null), DELAY seconds after the condition first holds."""
    if value is None:
        return None
    match = _CANCEL_WHEN.fullmatch(str(value).strip())
    if not match:
        raise BenchError("--cancel-when must look like RECORD:FIELD=VALUE[+DELAY] with RECORD workflow or job, "
                         f"e.g. job:stage=stack+2.5: {value!r}")
    return {"text": str(value).strip(), "record": match["record"], "field": match["field"], "value": match["value"],
            "delay_s": float(match["delay"] or 0)}


_MISSING = object()


def record_field(record, dotted: str):
    """The value at a dotted path of a JSON record (list items by index), or _MISSING."""
    node = record
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return _MISSING
    return node


def field_matches(value, wanted: str) -> bool:
    if value is _MISSING:
        return False
    if wanted == "*":
        return value is not None
    return (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) == wanted


def _parallel_default(name, value):
    """A parallelism variable's value with its unset form and its written-out default made one."""
    return _PARALLEL_DEFAULTS.get(name) if value is None or value == _PARALLEL_DEFAULTS.get(name) else value


def resolve_parallel(args) -> dict:
    """The parallelism variables of a run (prereg 6.1 table) and its effective mode.

    --parallel MODE sets all five; an explicit knob that disagrees is refused, except that under 'default' a knob
    filled by --profile stands (the simulated machine's default, e.g. render workers 3 on the 6900HX proxy).
    Without --parallel each variable is its knob or cleared; the effective mode is then 'default' when no knob (other
    than a profile's) sets one to anything but its documented default (--frame-workers auto, --overlap on), else
    'custom'."""
    mode = getattr(args, "parallel", None)
    filled = set(getattr(args, "profile_filled", ()) or ())
    knobs = {FRAME_WORKERS_ENV: ("frame_workers", getattr(args, "frame_workers", None)),
             RENDER_WORKERS_ENV: ("render_workers", None if getattr(args, "render_workers", None) is None
                                  else str(args.render_workers)),
             OVERLAP_ENV: ("overlap", OVERLAP_VALUES.get(getattr(args, "overlap", None)))}
    if mode is None:
        environment = {name: None for name in PARALLEL_VARIABLES}
        custom = []
        for name, (knob, value) in knobs.items():
            environment[name] = value
            if _parallel_default(name, value) != _parallel_default(name, None) and knob not in filled:
                custom.append(knob)
        return {"mode": None, "effective": "custom" if custom else "default", "environment": environment,
                "knobs": sorted(custom), "tier": sorted(k for k, v in knobs.values() if v is not None and k in filled)}
    if mode not in PARALLEL_MODES:
        raise BenchError(f"--parallel must be one of {', '.join(PARALLEL_MODES)}")
    environment = dict(PARALLEL_TABLE[mode])
    tier = []
    for name, (knob, value) in knobs.items():
        if value is None or _parallel_default(name, value) == _parallel_default(name, environment[name]):
            continue
        if mode == "default" and knob in filled:
            environment[name] = value
            tier.append(knob)
            continue
        source = f"--profile {args.profile}" if knob in filled else f"--{knob.replace('_', '-')}"
        raise BenchError(f"{source} ({name}={value}) conflicts with --parallel {mode} ({name}="
                         f"{environment[name] if environment[name] is not None else 'unset'}); "
                         "drop one of them" + ("" if mode != "default" else " (or omit --parallel)"))
    return {"mode": mode, "effective": mode, "environment": environment, "knobs": [], "tier": sorted(tier)}


def _gib_text(value) -> str:
    return f"{float(value):g}"


def simulation_descriptor(profile, mode, *, affinity, mask_sha, pin, render_workers, pin_ort, memory_view=None,
                          commit_view=None, python_openblas=None) -> dict | None:
    """What a simulated run limits and what it leaves to this host; the label is the E3 table's row name."""
    if not (profile or affinity or mask_sha or render_workers or memory_view or commit_view or python_openblas):
        return None
    memory = ([f"Siril mem {pin['mem']} GiB"] if pin and "mem" in pin else []) + \
             ([f"render workers {render_workers}"] if render_workers else []) + \
             ([f"memory view {_gib_text(memory_view)} GiB"] if memory_view else []) + \
             ([f"commit view {_gib_text(commit_view)} GiB"] if commit_view else [])
    label = ", ".join([f"{affinity} logical processors" if affinity else "all host logical processors",
                       "no SHA" if mask_sha else "SHA on",
                       ", ".join(memory) + " (other memory unlimited)" if memory else "unlimited memory"])
    notes = []
    if mask_sha:
        notes.append("SHA extensions masked: models a PC without SHA (the prereg's 'ordinary PC'). The reference "
                     "laptop (AMD 6900HX) has SHA, so this is not a 6900HX stand-in.")
    if not memory:
        notes.append("事后: memory is not limited, so this is optimistic for a 16 GB PC. performance.render_workers "
                     "also caps by free RAM (budget min(avail/3, avail - 1 GiB), about 0.95 GiB per 45.7 MP worker): "
                     "this host keeps the CPU cap of 8, a 16 GB laptop with about 10 GiB free gets about 3. Unpinned "
                     "Siril takes 0.9 x this host's free RAM, so convert, export and stack run more images at once "
                     "than on the laptop.")
    elif memory_view or commit_view:
        notes.append("Memory limited where the product reads it: Siril's memory limit, the render worker count and "
                     "performance.available_memory / available_commit (min(real, view) in the bench and every Python "
                     "process; the summary's hook_events list them). Other readers (the governor's machine facts, "
                     "ONNX Runtime, Siril's own 0.9 x free RAM when unpinned), the page cache and paging see this "
                     "host; T-2 and O-1 decide from the views.")
    else:
        notes.append("事后: memory limited only where the product reads it: Siril's memory limit and the render "
                     "worker count. Python and ONNX Runtime memory, the page cache and paging are not limited or "
                     "modelled; the laptop may pick fewer render workers as its free RAM changes.")
    if python_openblas:
        notes.append(f"OpenBLAS in the Python processes: OPENBLAS_CORETYPE={python_openblas['OPENBLAS_CORETYPE']}, "
                     f"OPENBLAS_NUM_THREADS={python_openblas['OPENBLAS_NUM_THREADS']} (this host: SkylakeX, 24); "
                     "Siril's environment block does not get them; the hook logs each process's openblas_get_config. "
                     "Results hold for this kernel type and thread count; ONNX Runtime across instruction sets is "
                     "not covered.")
    if pin and "mem" in pin and mode in ("L", "R"):
        notes.append("Legacy stack: a Siril memory limit can change its block layout (FINDINGS GAP3-4), so masters "
                     "may differ by a few float32 ULP from unpinned runs. Use this run for timings, not fingerprints.")
    if affinity and pin_ort:
        notes.append(f"事后: --pin-ort {pin_ort} is not in the prereg. Under --affinity ONNX Runtime's automatic "
                     "thread count may follow this host; a 16-thread PC would pick 8. The timeline's per-process "
                     "thread counts show what ran.")
    notes.append("The user's laptop run (AMD 6900HX, 16 GB, NVMe) is the memory truth; clocks, disk speed, a cold "
                 "page cache and paging are not simulated here.")
    descriptor = {"profile": profile, "status": PROFILES[profile]["status"] if profile else "ad hoc knobs",
                  "label": label, "logical_processors": affinity, "sha_extensions": "masked" if mask_sha else "host",
                  "memory_limits": memory, "notes": notes}
    if python_openblas:
        descriptor["python_openblas"] = dict(python_openblas)
    return descriptor


def resolve_governor(args) -> tuple[str, str | None]:
    """(AESTER_GOVERNOR, AESTER_GOVERNOR_FACTS) of a run; the bench default records the full facts."""
    governor = (getattr(args, "governor", None) or BENCH_GOVERNOR_DEFAULT).strip().lower()
    if governor not in GOVERNOR_MODES:
        raise BenchError(f"--governor must be one of {', '.join(GOVERNOR_MODES)}")
    facts = getattr(args, "governor_facts", None)
    facts = facts.strip() if isinstance(facts, str) and facts.strip() else None
    if facts is not None:
        if len(facts) > 200 or not all(_GOVERNOR_FACT.fullmatch(part.strip()) for part in facts.split(",")):
            raise BenchError("--governor-facts must look like 'total=16G,avail=9G,lps=16,cores=8,...'")
        if governor == "off":
            raise BenchError("--governor-facts needs --governor shadow or shadow-full")
    return governor, facts


def resolve_plan(args) -> dict:
    bench = ensure_allowed(args.bench_root, "bench root")
    _no_links_on_path(bench, "bench root")
    if args.mode not in MODES:
        raise BenchError("--mode must be L, F or R")
    if args.label and not LABEL.fullmatch(args.label):
        raise BenchError("--label must be 1-32 characters [A-Za-z0-9_-]")
    if args.pin_ort is not None and not 1 <= args.pin_ort <= 256:
        raise BenchError("--pin-ort must be 1..256")
    if args.affinity is not None and not 1 <= args.affinity <= 64:
        raise BenchError("--affinity must be 1..64 (one processor group)")
    render_workers = getattr(args, "render_workers", None)
    if render_workers is not None and not 1 <= render_workers <= MAX_RENDER_WORKERS:
        raise BenchError(f"--render-workers must be 1..{MAX_RENDER_WORKERS}")
    frame_workers = getattr(args, "frame_workers", None)
    if frame_workers is not None and frame_workers not in ("auto", "max", *map(str, range(1, MAX_FRAME_WORKERS + 1))):
        raise BenchError(f"--frame-workers must be auto, max or 1..{MAX_FRAME_WORKERS}")
    governor, governor_facts = resolve_governor(args)
    memory_view, commit_view = getattr(args, "memory_view", None), getattr(args, "commit_view", None)
    memory_view_bytes, commit_view_bytes = view_bytes(memory_view, "memory-view"), view_bytes(commit_view, "commit-view")
    python_openblas = parse_python_openblas(getattr(args, "python_openblas", None))
    trails = getattr(args, "trails", None) or "on"
    if trails not in ("on", "off"):
        raise BenchError("--trails must be on or off")
    if getattr(args, "overlap", None) not in (None, *OVERLAP_VALUES):
        raise BenchError("--overlap must be on or off")
    parallel = resolve_parallel(args)
    cancel_when = parse_cancel_when(getattr(args, "cancel_when", None))
    if cancel_when and getattr(args, "keep", False):
        raise BenchError("--cancel-when keeps its slot for --resume anyway; drop --keep")
    job = getattr(args, "job", None)
    if job is not None:
        if job not in JOB_KINDS:
            raise BenchError(f"--job must be one of {', '.join(JOB_KINDS)}")
        if args.mode != JOB_KINDS[job]["mode"]:
            raise BenchError(f"--job {job} runs --mode {JOB_KINDS[job]['mode']}")
        if cancel_when or getattr(args, "resume", False):
            raise BenchError("--job runs have no --cancel-when / --resume (the cancel points of prereg 6.4 are "
                             "workflow runs)")
        if args.pin_ort is not None:
            raise BenchError("--pin-ort sets the workflow's separation; a --job runs none")
    mode = MODES[args.mode]
    frames_ok = isinstance(args.frames, int) and 1 <= args.frames <= 99
    slot = (JOB_SLOT_PREFIX if job else "") + f"{args.mode}{args.frames}"
    if not frames_ok or not SLOT_NAME.fullmatch(slot):
        raise BenchError("--frames must be 1..99")
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{slot}" + (f"-{args.label}" if args.label else "")
    runs = bench / "runs"
    slot_dir = runs / slot
    code_root = _abs(args.code_root) if args.code_root else REPO
    code_src = code_root / "src"
    if not (code_src / "starstack" / "__init__.py").is_file():
        raise BenchError(f"--code-root has no src/starstack package: {code_root}")
    pin = parse_pin(args.pin_siril)
    hook = True             # P1c: every run (import record, memory views, OpenBLAS log, thread audit)
    shim = bool(pin) and not native_pin_support(code_src)
    if shim and not args.code_root:
        raise BenchError(f"{code_src / 'starstack' / 'siril.py'} does not support {SIRIL_PIN_ENV}")
    paths = {"bench_root": bench, "runs": runs, "slot": slot_dir, "cache": slot_dir / "c",
             "delivery": slot_dir / "d", "components": slot_dir / "k", "hook": slot_dir / "h",
             "siril_xdg": slot_dir / "x",
             "assets": bench / "assets", "project": bench / "assets" / f"regions-f{args.frames:02d}.json",
             "results": bench / "results" / run_id}
    for name, path in paths.items():
        ensure_allowed(path, name)
        if not _inside(path, bench):
            raise BenchError(f"{name} escapes the bench root: {path}")
    if len(str(paths["cache"])) > MAX_CACHE_CHARS:
        # Finite stacks fail on Windows MAX_PATH about 113 characters below the cache root (GAP3-6).
        raise BenchError(f"cache path {paths['cache']} is longer than {MAX_CACHE_CHARS} characters; use a shorter --bench-root")
    backend = dict(NOX_OPTIONS, intra_op_threads=args.pin_ort)
    engine = mode["stack_engine_path"]
    for tool in (SIRIL, *([] if job else [NOX_MODEL]), *([engine] if engine else [])):
        if not Path(tool).is_file():
            raise BenchError(f"missing tool: {tool}")
    if job:
        # core.submit_job's own parameters (input_dir, output_root and input_files are added at submit time)
        submit = {"siril_path": str(SIRIL), "sky_mask_path": str(sky_mask_copy(bench)),
                  "idempotency_key": f"aester-bench-{slot}", "support_mode": JOB_KINDS[job]["support_mode"],
                  "stack_range_mode": JOB_KINDS[job]["stack_range_mode"],
                  **({"transient_mode": "off"} if trails == "off" else {})}
    else:
        submit = {"siril_path": str(SIRIL), "star_model_path": str(NOX_MODEL),
                  "stack_engine_path": str(engine) if engine else None,
                  "stack_range_mode": mode["stack_range_mode"], "backend_options": backend,
                  "idempotency_key": f"aester-bench-{slot}", "regions_path": str(paths["project"]),
                  "delivery_root": str(paths["delivery"]), **copy.deepcopy(USER_SETTINGS)}
    if trails == "off" and not job:
        # the GUI/MCP default: no trail scan, and (as the GUI sends it) no trail_default at all
        submit["trails_enabled"] = False
        submit.pop("trail_default", None)
    parallel_env = parallel["environment"]
    environment = {"AESTER_CACHE_ROOT": str(paths["cache"]), "AESTER_COMPONENT_ROOT": str(paths["components"]),
                   SIRIL_PIN_ENV: pin_text(pin) if pin else None,
                   "OPENSSL_ia32cap": SHA_MASK if args.mask_sha else None,
                   RENDER_WORKERS_ENV: parallel_env[RENDER_WORKERS_ENV],
                   FRAME_WORKERS_ENV: parallel_env[FRAME_WORKERS_ENV],
                   TRANSIENT_WORKERS_ENV: parallel_env[TRANSIENT_WORKERS_ENV],
                   HASH_WORKERS_ENV: parallel_env[HASH_WORKERS_ENV],
                   OVERLAP_ENV: parallel_env[OVERLAP_ENV],
                   MEMORY_VIEW_ENV: str(memory_view_bytes) if memory_view_bytes else None,
                   COMMIT_VIEW_ENV: str(commit_view_bytes) if commit_view_bytes else None,
                   **{key: (python_openblas or {}).get(key) for key in PYTHON_OPENBLAS_KEYS},
                   PYTHON_ONLY_ENV: ",".join(PYTHON_OPENBLAS_KEYS) if python_openblas else None,
                   # P1b X9: the diagnostics-only L layout switch never leaks in from the shell (default split layout)
                   "AESTER_L_SIRIL_LAYOUT": None,
                   GOVERNOR_ENV: governor, GOVERNOR_FACTS_ENV: governor_facts,
                   "MSYSTEM": os.environ.get("MSYSTEM") or DEFAULT_MSYSTEM,
                   "XDG_DATA_HOME": str(paths["siril_xdg"]),
                   # P1b review (SF-2): an old --code-root's stock Siril must not start or join the user's session
                   # bus either; GLib reads it, nothing else in a run does. Kept under --siril-desktop too.
                   "DBUS_SESSION_BUS_ADDRESS": NO_SESSION_BUS}
    siril_launch = {"MSYSTEM": environment["MSYSTEM"],
                    "MSYSTEM_source": "inherited" if os.environ.get("MSYSTEM") else "bench default",
                    "XDG_DATA_HOME": environment["XDG_DATA_HOME"],
                    "DBUS_SESSION_BUS_ADDRESS": NO_SESSION_BUS,
                    "note": "harness only: Siril skips its Python venv work, never uses the user's "
                            "%LOCALAPPDATA%\\siril and never starts or joins the user's D-Bus session bus; "
                            "pixels do not depend on it"}
    if getattr(args, "siril_desktop", None):
        # P1b W5 (E2E): this process and its workers look like a GUI started from Explorer (no MSYSTEM) whose
        # Siril data folder is a sentinel; the product's own Siril environment block must keep Siril away from it.
        desktop = siril_desktop_folder(args.siril_desktop)
        environment.update(MSYSTEM=None, XDG_DATA_HOME=str(desktop))
        siril_launch = {"mode": "desktop-sentinel", "MSYSTEM": None, "MSYSTEM_source": "removed (--siril-desktop)",
                        "XDG_DATA_HOME": str(desktop), "DBUS_SESSION_BUS_ADDRESS": NO_SESSION_BUS,
                        "note": "end-to-end check: the summary's siril_desktop listing must not change"}
    if hook:
        environment.update({"PYTHONPATH": os.pathsep.join([str(code_src), str(paths["hook"])]),
                            HOOK_LOG_ENV: str(paths["hook"] / "imports"), HOOK_SHIM_ENV: "1" if shim else None,
                            HOOK_EVENTS_ENV: str(paths["hook"] / "events"),
                            THREAD_LOG_ENV: str(paths["hook"] / "threads"), BLAS_LOG_ENV: "1"})
    estimate = int(DISK_PER_FRAME[args.mode] * args.frames)
    explicit = {knob: getattr(args, knob) for knob in RESUME_KNOBS if getattr(args, knob, None) is not None
                and knob not in (getattr(args, "profile_filled", None) or ())}
    return {"run_id": run_id, "slot": slot, "mode": args.mode, "mode_name": mode["name"], "frames": args.frames,
            "job": job, "label": args.label, "paths": {k: str(v) for k, v in paths.items()},
            "code_root": str(code_root), "code_src": str(code_src), "code": _git_commit(code_root),
            "code_root_given": bool(args.code_root),
            "native_pin_support": native_pin_support(code_src), "hook": hook, "pin_shim": shim,
            "pins": {"siril": pin, "ort_intra_op_threads": args.pin_ort, "affinity_logical": args.affinity,
                     "mask_sha": bool(args.mask_sha), "render_workers": render_workers,
                     "frame_workers": frame_workers,
                     "governor": governor, "governor_facts": governor_facts,
                     "parallel": parallel["mode"], "overlap": getattr(args, "overlap", None), "trails": trails,
                     "memory_view_gib": None if memory_view is None else float(memory_view),
                     "commit_view_gib": None if commit_view is None else float(commit_view),
                     "python_openblas": python_openblas},
            "parallel": parallel, "explicit_knobs": explicit,
            "profile": getattr(args, "profile", None),
            "simulation": simulation_descriptor(getattr(args, "profile", None), args.mode, affinity=args.affinity,
                                                mask_sha=bool(args.mask_sha), pin=pin,
                                                render_workers=render_workers, pin_ort=args.pin_ort,
                                                memory_view=memory_view, commit_view=commit_view,
                                                python_openblas=python_openblas),
            "submit": submit, "environment": environment, "siril_launch": siril_launch,
            "cancel_when": cancel_when, "resume": bool(getattr(args, "resume", False)),
            "disk_estimate_bytes": estimate,
            "keep": bool(args.keep), "replace": bool(args.replace),
            "inputs": {"manifest": str(_abs(args.baseline_manifest)), "inputs_dir": str(_abs(args.inputs_dir))}}


# --------------------------------------------------------------------------- safe deletion

def safe_remove(target, runs_root) -> None:
    """Delete one bench slot: a marked direct child of runs/, containing no links."""
    runs, target = _abs(runs_root), _abs(target)
    if os.path.normcase(str(target.parent)) != os.path.normcase(str(runs)) or not SLOT_NAME.fullmatch(target.name):
        raise BenchError(f"refusing to delete {target}: not a slot directly inside {runs}")
    for root in forbidden_roots():
        if _inside(target, root):
            raise BenchError(f"refusing to delete inside {root}")
    _no_links_on_path(target, "delete target")
    if runs.name != "runs" or not target.is_dir():
        raise BenchError(f"refusing to delete {target}: not a bench slot directory")
    try:
        marker = json.loads((target / SLOT_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"refusing to delete {target}: no bench slot marker ({exc})") from exc
    if marker.get("slot") != target.name:
        raise BenchError(f"refusing to delete {target}: marker names slot {marker.get('slot')!r}")
    for directory, dirnames, filenames in os.walk(target):
        for name in dirnames + filenames:
            if _is_link(Path(directory) / name):
                raise BenchError(f"refusing to delete {target}: contains a link or junction {Path(directory) / name}")

    def writable(function, path, _error):
        os.chmod(path, stat.S_IWRITE)
        function(path)

    for attempt in range(4):
        try:
            shutil.rmtree(target, onexc=writable)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def disk_usage(root: Path) -> dict:
    files = size = linked = 0
    for directory, _dirs, names in os.walk(root):
        for name in names:
            try:
                info = os.lstat(os.path.join(directory, name))
            except OSError:
                continue
            files += 1
            size += info.st_size
            linked += info.st_nlink > 1
    return {"files": files, "bytes": size, "hard_linked_files": linked}


# --------------------------------------------------------------------------- Windows measurement

class Win:
    """Minimal read-only Win32 helpers (no admin rights needed)."""

    STILL_ACTIVE = 259
    QUERY = 0x1000          # PROCESS_QUERY_LIMITED_INFORMATION
    VM_READ = 0x0010

    def __init__(self):
        import ctypes
        import ctypes.wintypes as wt
        self.ctypes, self.wt = ctypes, wt
        k = self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        H = ctypes.c_void_p

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "total_page", "avail_page",
                                                  "total_virtual", "avail_virtual", "ext")]

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                               "PrivateUsage")]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps",
                                                          "ReadBytes", "WriteBytes", "OtherBytes")]

        class PE32(ctypes.Structure):
            _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                        ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]

        class BASIC_ACCOUNTING(ctypes.Structure):
            _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                        ("ThisPeriodTotalUserTime", ctypes.c_longlong), ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                        ("TotalPageFaultCount", wt.DWORD), ("TotalProcesses", wt.DWORD),
                        ("ActiveProcesses", wt.DWORD), ("TotalTerminatedProcesses", wt.DWORD)]

        class BASIC_AND_IO(ctypes.Structure):
            _fields_ = [("BasicInfo", BASIC_ACCOUNTING), ("IoInfo", IO)]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.MEMORYSTATUSEX, self.PMC, self.IO, self.PE32 = MEMORYSTATUSEX, PMC, IO, PE32
        self.BASIC_AND_IO, self.EXTENDED_LIMIT = BASIC_AND_IO, EXTENDED_LIMIT
        k.GetCurrentProcess.restype = H
        k.OpenProcess.restype = H
        k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k.CloseHandle.argtypes = [H]
        k.CreateToolhelp32Snapshot.restype = H
        k.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        k.Process32FirstW.argtypes = [H, ctypes.c_void_p]
        k.Process32NextW.argtypes = [H, ctypes.c_void_p]
        k.GetProcessTimes.argtypes = [H] + [ctypes.c_void_p] * 4
        k.GetProcessIoCounters.argtypes = [H, ctypes.c_void_p]
        k.K32GetProcessMemoryInfo.argtypes = [H, ctypes.c_void_p, wt.DWORD]
        k.GetExitCodeProcess.argtypes = [H, ctypes.c_void_p]
        k.GetProcessAffinityMask.argtypes = [H, ctypes.c_void_p, ctypes.c_void_p]
        k.SetProcessAffinityMask.argtypes = [H, ctypes.c_size_t]
        k.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        k.CreateJobObjectW.restype = H
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k.SetInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        k.QueryInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
        k.AssignProcessToJobObject.argtypes = [H, H]
        k.SetHandleInformation.argtypes = [H, wt.DWORD, wt.DWORD]
        k.GetSystemTimes.argtypes = [ctypes.c_void_p] * 3
        k.GetActiveProcessorGroupCount.restype = wt.WORD
        k.GetActiveProcessorCount.argtypes = [wt.WORD]
        k.GetActiveProcessorCount.restype = wt.DWORD

        class SPPI(ctypes.Structure):           # SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION (100 ns units)
            _fields_ = [("IdleTime", ctypes.c_longlong), ("KernelTime", ctypes.c_longlong),
                        ("UserTime", ctypes.c_longlong), ("DpcTime", ctypes.c_longlong),
                        ("InterruptTime", ctypes.c_longlong), ("InterruptCount", ctypes.c_ulong)]

        self.SPPI = SPPI
        try:
            self.nt = ctypes.WinDLL("ntdll")
            self.nt.NtQuerySystemInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p, wt.ULONG, ctypes.c_void_p,
                                                           wt.ULONG, ctypes.POINTER(wt.ULONG)]
            self.nt.NtQuerySystemInformationEx.restype = ctypes.c_long
        except (OSError, AttributeError):
            self.nt = None

    def system_times(self) -> dict | None:
        """Idle, kernel (idle included) and user seconds summed over every processor of every active group.
        GetSystemTimes covers only one processor group (64 of the development PC's 192), so this reads
        SystemProcessorPerformanceInformation per group; GetSystemTimes is the fallback (method says which)."""
        ctypes, wt = self.ctypes, self.wt
        if self.nt is not None:
            idle = kernel = user = processors = 0
            try:
                for group in range(self.k32.GetActiveProcessorGroupCount()):
                    count = self.k32.GetActiveProcessorCount(group)
                    buffer = (self.SPPI * max(1, count))()
                    number, returned = wt.USHORT(group), wt.ULONG()
                    if self.nt.NtQuerySystemInformationEx(8, ctypes.byref(number), ctypes.sizeof(number), buffer,
                                                          ctypes.sizeof(buffer), ctypes.byref(returned)) != 0:
                        raise OSError("NtQuerySystemInformationEx failed")
                    for entry in buffer[:returned.value // ctypes.sizeof(self.SPPI)]:
                        idle, kernel, user = idle + entry.IdleTime, kernel + entry.KernelTime, user + entry.UserTime
                        processors += 1
                return {"idle": idle / 1e7, "kernel": kernel / 1e7, "user": user / 1e7, "processors": processors,
                        "method": "per-group"}
            except OSError:
                pass
        values = [ctypes.c_ulonglong() for _ in range(3)]
        if not self.k32.GetSystemTimes(*(ctypes.byref(v) for v in values)):
            return None
        return {"idle": values[0].value / 1e7, "kernel": values[1].value / 1e7, "user": values[2].value / 1e7,
                "processors": None, "method": "GetSystemTimes (one processor group)"}

    @staticmethod
    def busy_seconds(times: dict | None) -> float | None:
        return None if not times else times["kernel"] - times["idle"] + times["user"]

    def memory_status(self) -> dict:
        m = self.MEMORYSTATUSEX()
        m.dwLength = self.ctypes.sizeof(m)
        self.k32.GlobalMemoryStatusEx(self.ctypes.byref(m))
        return {"total": m.total, "avail": m.avail, "commit": m.total_page - m.avail_page, "commit_limit": m.total_page}

    def snapshot(self) -> dict:
        """pid -> (parent pid, exe name, thread count)."""
        handle = self.k32.CreateToolhelp32Snapshot(2, 0)
        if handle in (None, self.ctypes.c_void_p(-1).value):
            return {}
        out = {}
        entry = self.PE32()
        entry.dwSize = self.ctypes.sizeof(entry)
        try:
            ok = self.k32.Process32FirstW(handle, self.ctypes.byref(entry))
            while ok:
                out[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile, entry.cntThreads)
                ok = self.k32.Process32NextW(handle, self.ctypes.byref(entry))
        finally:
            self.k32.CloseHandle(handle)
        return out

    def open(self, pid: int):
        return self.k32.OpenProcess(self.QUERY | self.VM_READ, False, pid) or self.k32.OpenProcess(self.QUERY, False, pid)

    def close(self, handle):
        if handle:
            self.k32.CloseHandle(handle)

    def times(self, handle):
        c, e, kt, ut = (self.ctypes.c_ulonglong() for _ in range(4))
        if not self.k32.GetProcessTimes(handle, *(self.ctypes.byref(x) for x in (c, e, kt, ut))):
            return None, None
        return c.value, (kt.value + ut.value) / 1e7

    def alive(self, handle) -> bool:
        code = self.wt.DWORD()
        return bool(self.k32.GetExitCodeProcess(handle, self.ctypes.byref(code))) and code.value == self.STILL_ACTIVE

    def memory(self, handle):
        p = self.PMC()
        p.cb = self.ctypes.sizeof(p)
        if not self.k32.K32GetProcessMemoryInfo(handle, self.ctypes.byref(p), p.cb):
            return None
        return {"private": p.PrivateUsage, "ws": p.WorkingSetSize, "peak_ws": p.PeakWorkingSetSize,
                "peak_private": p.PeakPagefileUsage}

    def io(self, handle):
        c = self.IO()
        if not self.k32.GetProcessIoCounters(handle, self.ctypes.byref(c)):
            return None
        return c.ReadBytes, c.WriteBytes

    def affinity(self, handle):
        process, system = self.ctypes.c_size_t(), self.ctypes.c_size_t()
        if not self.k32.GetProcessAffinityMask(handle, self.ctypes.byref(process), self.ctypes.byref(system)):
            return None
        return process.value

    def set_own_affinity(self, logical: int) -> int:
        mask = (1 << logical) - 1
        me = self.k32.GetCurrentProcess()
        if not self.k32.SetProcessAffinityMask(me, mask):
            raise BenchError(f"SetProcessAffinityMask failed ({self.ctypes.get_last_error()})")
        if self.affinity(me) != mask:
            raise BenchError("process affinity was not applied as requested")
        return mask

    def own_job(self):
        """A KILL_ON_JOB_CLOSE job holding this process and all its descendants (for accounting)."""
        job = self.k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(self.ctypes.get_last_error(), "CreateJobObjectW")
        limits = self.EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x2000
        if not (self.k32.SetInformationJobObject(job, 9, self.ctypes.byref(limits), self.ctypes.sizeof(limits))
                and self.k32.SetHandleInformation(job, 1, 0)
                and self.k32.AssignProcessToJobObject(job, self.k32.GetCurrentProcess())):
            error = self.ctypes.get_last_error()
            self.k32.CloseHandle(job)
            raise OSError(error, "job accounting unavailable")
        return job

    def job_accounting(self, job) -> dict:
        acc, ext = self.BASIC_AND_IO(), self.EXTENDED_LIMIT()
        self.k32.QueryInformationJobObject(job, 8, self.ctypes.byref(acc), self.ctypes.sizeof(acc), None)
        self.k32.QueryInformationJobObject(job, 9, self.ctypes.byref(ext), self.ctypes.sizeof(ext), None)
        return {"cpu_user_s": acc.BasicInfo.TotalUserTime / 1e7, "cpu_kernel_s": acc.BasicInfo.TotalKernelTime / 1e7,
                "processes_total": acc.BasicInfo.TotalProcesses, "processes_active": acc.BasicInfo.ActiveProcesses,
                "io_read_bytes": acc.IoInfo.ReadBytes, "io_write_bytes": acc.IoInfo.WriteBytes,
                "io_read_ops": acc.IoInfo.ReadOps, "io_write_ops": acc.IoInfo.WriteOps,
                "peak_job_commit_bytes": ext.PeakJobMemoryUsed, "peak_process_commit_bytes": ext.PeakProcessMemoryUsed}


class _Proc:
    __slots__ = ("pid", "ppid", "exe", "handle", "created", "cpu", "io", "memory", "affinity", "threads",
                 "alive", "first_t", "last_t")

    def __init__(self, pid, ppid, exe, handle, created, t):
        self.pid, self.ppid, self.exe, self.handle, self.created = pid, ppid, exe, handle, created
        self.cpu, self.io, self.memory, self.affinity, self.threads = 0.0, (0, 0), None, None, 0
        self.alive, self.first_t, self.last_t = True, t, t


class Sampler:
    """1 Hz timeline of the system and of this process's descendant tree.

    Descendants are adopted by parent pid with a creation-time check (a child
    is never older than its parent), so a reused pid cannot pull in strangers.
    Handles are held until stop(), so exited processes still report final CPU
    and I/O. Processes living less than one interval may be missed; the job
    accounting in the summary covers those.
    """

    def __init__(self, win: Win, out_path: Path, *, interval=1.0, stage=lambda: None, disk_path=None,
                 expected_affinity=None):
        self.win, self.out_path, self.interval, self.stage = win, out_path, interval, stage
        self.disk_path, self.expected_affinity = disk_path, expected_affinity
        self.procs: dict[int, _Proc] = {}
        self.finished: list[_Proc] = []
        self.rejected: set[tuple[int, int]] = set()
        self.rows = 0
        self.peaks = {"tree_private": (0, None), "tree_ws": (0, None), "sys_commit": (0, None)}
        self.min_avail = (None, None)
        self.min_disk_free = (None, None)
        self.last_cpu, self.last_t, self.last_busy = None, None, None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.t0 = time.perf_counter()
        root = os.getpid()
        handle = win.open(root)
        if not handle:
            raise BenchError("cannot open this process for measurement")
        created, _cpu = win.times(handle)
        self.procs[root] = _Proc(root, None, Path(sys.executable).name, handle, created, 0.0)

    def start(self):
        self.out = open(self.out_path, "a", encoding="utf-8", newline="\n")
        self._thread = threading.Thread(target=self._loop, name="aester-bench-sampler", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        self.sample()
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:  # a sampling glitch must not end the benchmark
                self.out.write(json.dumps({"t": round(time.perf_counter() - self.t0, 3), "error": repr(exc)}) + "\n")

    def _adopt(self, snapshot, t):
        """Track every new process whose parent is tracked (repeat until closed under parenthood)."""
        changed = True
        while changed:
            changed = False
            for pid, (ppid, exe, threads) in snapshot.items():
                current = self.procs.get(pid)
                if current is not None and current.alive:
                    current.threads = max(current.threads, threads)
                    continue
                parent = self.procs.get(ppid)
                if parent is None or pid == ppid or (pid, ppid) in self.rejected:
                    continue
                handle = self.win.open(pid)
                if not handle:
                    continue
                created, _cpu = self.win.times(handle)
                if current is not None and created == current.created:
                    self.win.close(handle)          # the same, already recorded process
                    continue
                if parent.created is None or created is None or created < parent.created:
                    self.win.close(handle)          # older than its supposed parent: a reused pid
                    self.rejected.add((pid, ppid))
                    continue
                if current is not None:             # an exited tracked pid reused by a new child
                    self.finished.append(current)
                proc = _Proc(pid, ppid, exe, handle, created, t)
                proc.threads = threads
                self.procs[pid] = proc
                changed = True

    def sample(self):
        with self._lock:
            t = time.perf_counter() - self.t0
            snapshot = self.win.snapshot()
            self._adopt(snapshot, t)
            tree = {"private": 0, "ws": 0, "cpu": 0.0, "read": 0, "write": 0, "alive": 0}
            by_exe, threads = {}, {}
            for proc in list(self.procs.values()) + self.finished:
                if proc.alive:
                    _created, cpu = self.win.times(proc.handle)
                    proc.cpu = cpu if cpu is not None else proc.cpu
                    io = self.win.io(proc.handle)
                    proc.io = io if io is not None else proc.io
                    if self.win.alive(proc.handle):
                        proc.memory = self.win.memory(proc.handle) or proc.memory
                        if proc.affinity is None:
                            proc.affinity = self.win.affinity(proc.handle)
                        proc.last_t = t
                        tree["alive"] += 1
                        if proc.memory:
                            tree["private"] += proc.memory["private"]
                            tree["ws"] += proc.memory["ws"]
                        current = snapshot.get(proc.pid)
                        if current is not None and (proc.ppid is None or current[0] == proc.ppid):
                            threads[str(proc.pid)] = current[2]     # OS threads of each live process, this second
                    else:
                        proc.alive = False
                tree["cpu"] += proc.cpu
                tree["read"] += proc.io[0]
                tree["write"] += proc.io[1]
                name = proc.exe.lower()
                slot = by_exe.setdefault(name, {"alive": 0, "private": 0, "ws": 0, "cpu_s": 0.0})
                slot["cpu_s"] += proc.cpu
                if proc.alive and proc.memory:
                    slot["alive"] += 1
                    slot["private"] += proc.memory["private"]
                    slot["ws"] += proc.memory["ws"]
            system = self.win.memory_status()
            disk_free = shutil.disk_usage(self.disk_path).free if self.disk_path else None
            busy = self.win.busy_seconds(self.win.system_times())
            cores = busy_cores = foreign = None         # busy logical cores since the previous sample
            if self.last_t is not None and t > self.last_t:
                cores = round((tree["cpu"] - self.last_cpu) / (t - self.last_t), 3)
                if busy is not None and self.last_busy is not None:
                    busy_cores = round((busy - self.last_busy) / (t - self.last_t), 3)
                    foreign = round(busy_cores - cores, 3)   # an over-estimate when a short process was missed
            self.last_cpu, self.last_t, self.last_busy = tree["cpu"], t, busy
            row = {"t": round(t, 3), "utc": utc_now().isoformat(timespec="milliseconds"), "stage": self.stage(),
                   "sys_avail": system["avail"], "sys_commit": system["commit"], "tree_private": tree["private"],
                   "tree_ws": tree["ws"], "tree_cpu_s": round(tree["cpu"], 3), "cpu_cores": cores,
                   "sys_busy_cores": busy_cores, "foreign_cores": foreign,
                   "io_read": tree["read"], "io_write": tree["write"], "procs": tree["alive"],
                   "disk_free": disk_free, "by_exe": by_exe, "threads": threads}
            self.out.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.out.flush()
            self.rows += 1
            for key, value in (("tree_private", tree["private"]), ("tree_ws", tree["ws"]), ("sys_commit", system["commit"])):
                if value > self.peaks[key][0]:
                    self.peaks[key] = (value, round(t, 3))
            if self.min_avail[0] is None or system["avail"] < self.min_avail[0]:
                self.min_avail = (system["avail"], round(t, 3))
            if disk_free is not None and (self.min_disk_free[0] is None or disk_free < self.min_disk_free[0]):
                self.min_disk_free = (disk_free, round(t, 3))
            self.system_total = system["total"]

    def descendants_alive(self) -> int:
        with self._lock:
            me = os.getpid()
            self._adopt(self.win.snapshot(), time.perf_counter() - self.t0)
            return sum(1 for p in self.procs.values() if p.pid != me and self.win.alive(p.handle))

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        self.sample()
        self.out.close()
        me = os.getpid()
        everyone = list(self.procs.values()) + self.finished
        per_exe = {}
        for proc in everyone:
            slot = per_exe.setdefault(proc.exe.lower(), {"processes": 0, "cpu_s": 0.0, "io_read": 0, "io_write": 0,
                                                         "max_peak_ws": 0, "max_peak_private": 0, "max_threads": 0})
            slot["processes"] += 1
            slot["cpu_s"] = round(slot["cpu_s"] + proc.cpu, 3)
            slot["io_read"] += proc.io[0]
            slot["io_write"] += proc.io[1]
            slot["max_threads"] = max(slot["max_threads"], proc.threads)
            if proc.memory:
                slot["max_peak_ws"] = max(slot["max_peak_ws"], proc.memory["peak_ws"])
                slot["max_peak_private"] = max(slot["max_peak_private"], proc.memory["peak_private"])
        mismatched = []
        if self.expected_affinity is not None:
            mismatched = [{"pid": p.pid, "exe": p.exe, "affinity": p.affinity} for p in everyone
                          if p.affinity is not None and p.affinity != self.expected_affinity]
        processes = [{"pid": p.pid, "ppid": p.ppid, "exe": p.exe, "self": p.pid == me, "cpu_s": round(p.cpu, 3),
                      "io_read": p.io[0], "io_write": p.io[1], "threads_max": p.threads,
                      "peak_ws": (p.memory or {}).get("peak_ws"), "peak_private": (p.memory or {}).get("peak_private"),
                      "affinity": p.affinity, "first_t": round(p.first_t, 3), "last_t": round(p.last_t, 3)}
                     for p in everyone]
        for proc in everyone:
            self.win.close(proc.handle)
        return {"interval_s": self.interval, "samples": self.rows, "timeline": str(self.out_path),
                "peak_tree_private_bytes": self.peaks["tree_private"][0], "peak_tree_private_t": self.peaks["tree_private"][1],
                "peak_tree_ws_bytes": self.peaks["tree_ws"][0], "peak_tree_ws_t": self.peaks["tree_ws"][1],
                "peak_sys_commit_bytes": self.peaks["sys_commit"][0], "min_sys_avail_bytes": self.min_avail[0],
                "min_sys_avail_t": self.min_avail[1], "sys_total_bytes": getattr(self, "system_total", None),
                "min_disk_free_bytes": self.min_disk_free[0],
                "tree_cpu_s": round(sum(p.cpu for p in everyone), 3),
                "bench_process_cpu_s": round(sum(p.cpu for p in everyone if p.pid == me), 3),
                "per_exe": per_exe, "processes_seen": len(everyone),
                "affinity_check": None if self.expected_affinity is None else
                {"expected_mask": hex(self.expected_affinity),
                 "checked": sum(1 for p in everyone if p.affinity is not None), "mismatched": mismatched},
                "processes": processes}


# --------------------------------------------------------------------------- SHA mask and affinity checks

_SHA_PROBE = ("import hashlib,json,time;d=b'\\x5a'*(256<<20);t=time.perf_counter();"
              "h=hashlib.sha256(d).hexdigest();s=time.perf_counter()-t;"
              "print(json.dumps({'digest':h,'GBps':round(len(d)/s/1e9,3),"
              "'abc':hashlib.sha256(b'abc').hexdigest()}))")
_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def sha_probe(mask: bool) -> dict:
    env = dict(os.environ)
    env.pop("OPENSSL_ia32cap", None)
    if mask:
        env["OPENSSL_ia32cap"] = SHA_MASK
    run = subprocess.run([sys.executable, "-c", _SHA_PROBE], env=env, capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        raise BenchError("SHA probe failed: " + run.stderr[-500:])
    result = json.loads(run.stdout)
    result["hashlib_ok"] = result.pop("abc") == _ABC
    return result


def sha_check() -> dict:
    plain, masked = sha_probe(False), sha_probe(True)
    return {"mask": SHA_MASK, "unmasked": plain, "masked": masked,
            "digests_match": plain["digest"] == masked["digest"] and plain["hashlib_ok"] and masked["hashlib_ok"],
            "slowdown": round(plain["GBps"] / masked["GBps"], 2) if masked["GBps"] else None,
            "note": "A slowdown well above 1 shows OpenSSL stopped using the SHA extension; about 1 means the CPU "
                    "has none or the mask had no effect."}


_AFFINITY_PROBE = r"""
import ctypes, json, subprocess, sys
k = ctypes.WinDLL('kernel32')
k.GetCurrentProcess.restype = ctypes.c_void_p
p, s = ctypes.c_size_t(), ctypes.c_size_t()
k.GetProcessAffinityMask(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.byref(p), ctypes.byref(s))
depth = int(sys.argv[1])
out = {'depth': depth, 'mask': hex(p.value)}
if depth < 2:
    child = subprocess.run([sys.executable, '-c', sys.argv[2], str(depth + 1), sys.argv[2]],
                           capture_output=True, text=True, creationflags=0x08000000)
    out['child'] = json.loads(child.stdout)
print(json.dumps(out))
"""


def check_environment(args) -> dict:
    report = {"cpu_count": os.cpu_count(), "python": sys.version, "platform": sys.platform,
              "inherited_environment": {k: v for k, v in os.environ.items() if RELEVANT_ENV.match(k)}}
    if args.affinity:
        win = Win()
        mask = win.set_own_affinity(args.affinity)
        run = subprocess.run([sys.executable, "-c", _AFFINITY_PROBE, "1", _AFFINITY_PROBE],
                             capture_output=True, text=True, timeout=120)
        chain = json.loads(run.stdout)
        masks = []
        node = chain
        while node:
            masks.append(node["mask"])
            node = node.get("child")
        report["affinity"] = {"requested_logical": args.affinity, "mask": hex(mask), "descendant_masks": masks,
                              "inherited": all(m == hex(mask) for m in masks)}
    if args.mask_sha:
        report["sha"] = sha_check()
    return report


# --------------------------------------------------------------------------- import hook (every run)

HOOK_SOURCE = r'''"""Aester bench hook, generated by scripts/bench_workflow.py for one bench run.

Loaded by site.py in every Python process of the run (PYTHONPATH). It records
which starstack package each process imports, and for code roots that predate
AESTER_SIRIL_PIN it applies the same pin starstack.siril applies natively
(identical script lines, log note and Siril-only OMP_NUM_THREADS). Silent in
unrelated interpreters.

Efficiency P1c (each part only when its variable is set): memory views
(performance.available_memory / available_commit return min(real, view)). The
view goes on the reading before in-process reservations, so that they are still
deducted from it (R20): _memory_status (P1c-0: the (physical, commit) pair), or
_available_memory_raw / _available_commit_raw; a code root with neither gets the
public function wrapped with frame_pool.reserved_bytes() deducted from the view
(min(raw, view) - reserved = min(raw - reserved, view - reserved)); each wrap's
event says which ('reservations'); a function the code root lacks is logged as
absent and skipped); names kept out of Siril's environment block
(siril.siril_environment); the process's OpenBLAS configuration once numpy is
loaded; and a 1 Hz audit of the process's Python threads by name. Nothing here
changes what a process computes.
"""
import os
import sys

_LOG = os.environ.get("AESTER_BENCH_HOOK_LOG")
_SHIM = os.environ.get("AESTER_BENCH_PIN_SHIM") == "1"
_EVENTS = os.environ.get("AESTER_BENCH_HOOK_EVENTS")
_VIEWS = (("available_memory", "_available_memory_raw", os.environ.get("AESTER_BENCH_MEMORY_VIEW")),
          ("available_commit", "_available_commit_raw", os.environ.get("AESTER_BENCH_COMMIT_VIEW")))
_PYTHON_ONLY = [name for name in (os.environ.get("AESTER_BENCH_PYTHON_ONLY") or "").split(",") if name]
_BLAS_LOG = os.environ.get("AESTER_BENCH_BLAS_LOG") == "1"
_THREAD_LOG = os.environ.get("AESTER_BENCH_THREAD_LOG")
_THREAD_INTERVAL = 1.0
# Thread names (threading.enumerate): a ThreadPoolExecutor's threads are '<prefix>_<k>'. Frame-pool threads are
# 'aester-<pool name>_<k>'; these executors are not the frame pool. Render executors run row ranges (BLAS-capable);
# the BLAS-free ones only hash, read back or wait.
_OTHER_EXECUTORS = ("aester-stage-readback", "aester-finishing", "aester-hash", "aester-array-hash",
                    "aester-validity-hash", "aester-range-hash", "aester-render", "aester-master")
_RENDER_EXECUTORS = ("aester-render", "aester-master")
_BLAS_FREE = ("aester-stage-readback", "aester-finishing", "aester-hash", "aester-array-hash", "aester-validity-hash",
              "aester-range-hash", "aester-volume-query", "aester-peak-sampler", "aester-bench")
_STOPPED = []           # stop(): no more writes (the bench process deletes the slot that holds the logs)
_AUDIT = {}


def stop():
    """For the bench process before it deletes the slot: a last thread-audit write, then no writes at all."""
    write = _AUDIT.get("write")
    if write is not None and not _STOPPED:
        write()
    _STOPPED.append(True)


def _event(kind, **fields):
    """One JSON file per event in AESTER_BENCH_HOOK_EVENTS; never raises."""
    if not _EVENTS or _STOPPED:
        return
    import json
    import time
    try:
        os.makedirs(_EVENTS, exist_ok=True)
        name = os.path.join(_EVENTS, "%d-%d-%s.json" % (os.getpid(), time.time_ns(), kind))
        with open(name, "x", encoding="utf-8") as stream:
            json.dump(dict(fields, event=kind, pid=os.getpid(), argv=list(getattr(sys, "argv", [])[:3]),
                           executable=sys.executable), stream, default=str)
    except Exception:
        pass


def _reserved_reader(module):
    """frame_pool.reserved_bytes of the code root (P1c-0), or None: through the performance module or sys.modules."""
    pool = getattr(module, "frame_pool", None) or sys.modules.get("starstack.frame_pool")
    reader = getattr(pool, "reserved_bytes", None)
    return reader if callable(reader) else None


def patch_performance(module):
    """Memory views on the reading before in-process reservations: _memory_status (both views, one wrap), else
    _available_*_raw, else the public function with the reservations deducted from the view."""
    if getattr(module, "_aester_bench_viewed", False):
        return None
    applied = {}
    limits = [int(view) if view else None for _, _, view in _VIEWS]
    status = getattr(module, "_memory_status", None)
    if callable(status) and any(limit is not None for limit in limits):

        def memory_status(*args, _original=status, **kwargs):
            readings = _original(*args, **kwargs)
            try:
                values = list(readings)
            except TypeError:
                return readings
            for index, limit in enumerate(limits[:len(values)]):
                if limit is not None and values[index] is not None:
                    values[index] = min(int(values[index]), limit)
            return tuple(values)

        memory_status.__name__, memory_status.__doc__, memory_status.__wrapped__ = \
            "_memory_status", status.__doc__, status
        module._memory_status = memory_status
        for index, (public, _, view) in enumerate(_VIEWS):
            if view:
                applied[public] = {"status": "wrapped", "function": "_memory_status[%d]" % index,
                                   "view_bytes": limits[index], "reservations": "below the view"}
        module._aester_bench_viewed = True
        _event("view", module=module.__name__, file=getattr(module, "__file__", None), applied=applied)
        return applied
    for (public, raw, view), limit in zip(_VIEWS, limits):
        if not view:
            continue
        name = raw if callable(getattr(module, raw, None)) else public
        original = getattr(module, name, None)
        if not callable(original):
            applied[public] = {"status": "absent"}
            continue
        on_public = name == public
        reader = _reserved_reader(module) if on_public else None

        def viewed(*args, _original=original, _limit=limit, _public=on_public, **kwargs):
            real = _original(*args, **kwargs)
            if real is None:
                return real
            reserved = _reserved_reader(module) if _public else None
            # the public reading is net of reservations already: deduct them from the view as well
            return min(int(real), max(0, _limit - int(reserved())) if reserved is not None else _limit)

        viewed.__name__, viewed.__doc__, viewed.__wrapped__ = name, original.__doc__, original
        setattr(module, name, viewed)
        applied[public] = {"status": "wrapped", "function": name, "view_bytes": limit,
                           "reservations": "below the view" if not on_public else
                           "deducted from the view" if reader is not None else "none",
                           "frame_pool_reserved": reader is not None}
    module._aester_bench_viewed = True
    if applied:
        _event("view", module=module.__name__, file=getattr(module, "__file__", None), applied=applied)
    return applied


def patch_siril_environment(module):
    """Keep _PYTHON_ONLY names (OpenBLAS settings meant for the Python processes) out of Siril's environment."""
    if not _PYTHON_ONLY or getattr(module, "_aester_bench_python_only", False):
        return False
    original = getattr(module, "siril_environment", None)
    module._aester_bench_python_only = True
    if not callable(original):
        _event("siril_environment", status="absent", names=_PYTHON_ONLY)
        return False

    def siril_environment(*args, **kwargs):
        env = original(*args, **kwargs)
        for name in _PYTHON_ONLY:
            env.pop(name, None)
        return env

    siril_environment.__doc__, siril_environment.__wrapped__ = original.__doc__, original
    module.siril_environment = siril_environment
    _event("siril_environment", status="wrapped", names=_PYTHON_ONLY)
    return True


def log_openblas():
    """openblas_get_config / corename / num_threads of the OpenBLAS numpy loaded (numpy.libs or numpy/.libs)."""
    if getattr(log_openblas, "done", False):
        return None
    log_openblas.done = True
    result = {}
    try:
        import ctypes
        import glob
        numpy = sys.modules.get("numpy")
        base = os.path.dirname(getattr(numpy, "__file__", "") or "")
        paths = sorted(glob.glob(os.path.join(os.path.dirname(base), "numpy.libs", "*openblas*"))
                       + glob.glob(os.path.join(base, ".libs", "*openblas*")))
        result["library"] = paths[0] if paths else None
        if paths:
            library = ctypes.CDLL(paths[0])
            for key, names, restype in (
                    ("config", ("openblas_get_config", "scipy_openblas_get_config64_", "openblas_get_config64_"),
                     ctypes.c_char_p),
                    ("corename", ("openblas_get_corename", "scipy_openblas_get_corename64_",
                                  "openblas_get_corename64_"), ctypes.c_char_p),
                    ("num_threads", ("openblas_get_num_threads", "scipy_openblas_get_num_threads64_",
                                     "openblas_get_num_threads64_"), ctypes.c_int)):
                for name in names:
                    function = getattr(library, name, None)
                    if function is None:
                        continue
                    function.restype, function.argtypes = restype, []
                    value = function()
                    result[key] = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
                    break
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    result["environment"] = {name: os.environ.get(name) for name in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS")}
    _event("openblas", **result)
    return result


def thread_counts(names):
    """(counts, per-prefix counts) of one sample of Python thread names."""
    import re
    counts = {"total": len(names), "frame_pool": 0, "render": 0, "aester_other": 0, "blas_capable": 1}
    prefixes = {}
    for name in names:
        if not name.startswith("aester-") or name.startswith("aester-bench"):
            continue
        match = re.fullmatch(r"(aester-.+)_([0-9]+)", name)
        prefix = match.group(1) if match else name
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if match and prefix not in _OTHER_EXECUTORS:
            counts["frame_pool"] += 1
            counts["blas_capable"] += 1
        elif prefix in _RENDER_EXECUTORS:
            counts["render"] += 1
            counts["blas_capable"] += 1
        else:
            counts["aester_other"] += 1
            if not prefix.startswith(_BLAS_FREE):
                counts["blas_capable"] += 1
    return counts, prefixes


def start_thread_audit():
    """A daemon thread ('bench-thread-audit') that samples threading.enumerate() every second and keeps the maxima
    in AESTER_BENCH_THREAD_LOG/<pid>-<start ns>.json (rewritten when a maximum rises, every 30 samples and at exit)."""
    import atexit
    import json
    import threading
    import time
    started_ns = time.time_ns()
    path = os.path.join(_THREAD_LOG, "%d-%d.json" % (os.getpid(), started_ns))
    state = {"pid": os.getpid(), "started_ns": started_ns, "interval_s": _THREAD_INTERVAL, "samples": 0,
             "max": {}, "max_t": {}, "max_by_prefix": {}, "first_t": None, "last_t": None}
    t0 = time.perf_counter()
    lock = threading.Lock()

    def write():
        with lock:
            if _STOPPED:
                return
            try:
                os.makedirs(_THREAD_LOG, exist_ok=True)
                temp = path + ".tmp"
                with open(temp, "w", encoding="utf-8") as stream:
                    json.dump(dict(state, argv=list(getattr(sys, "argv", [])[:3]), executable=sys.executable), stream)
                os.replace(temp, path)
            except Exception:
                pass

    def sample():
        counts, prefixes = thread_counts([thread.name for thread in threading.enumerate()])
        t = round(time.perf_counter() - t0, 3)
        risen = False
        with lock:
            state["samples"] += 1
            state["first_t"] = t if state["first_t"] is None else state["first_t"]
            state["last_t"] = t
            for key, value in counts.items():
                if value > state["max"].get(key, -1):
                    state["max"][key], state["max_t"][key] = value, t
                    risen = True
            for prefix, value in prefixes.items():
                if value > state["max_by_prefix"].get(prefix, 0):
                    state["max_by_prefix"][prefix] = value
                    risen = True
        return risen

    def loop():
        while not _STOPPED:
            try:
                if sample() or state["samples"] % 30 == 1:
                    write()
            except Exception:
                pass
            time.sleep(_THREAD_INTERVAL)

    _AUDIT["write"] = write
    atexit.register(write)
    threading.Thread(target=loop, name="bench-thread-audit", daemon=True).start()
    return path


def _record(module):
    if not _LOG:
        return
    import json
    import time
    try:
        os.makedirs(_LOG, exist_ok=True)
        name = os.path.join(_LOG, "%d-%d.json" % (os.getpid(), time.time_ns()))
        with open(name, "x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "module": module.__name__, "file": module.__file__,
                       "argv": sys.argv[:3], "executable": sys.executable}, stream)
    except OSError:
        pass


def _parse(value):
    import re
    text = (value or "").strip()
    if not text:
        return None
    pin = {}
    for part in text.split(","):
        key, separator, number = part.partition("=")
        key, number = key.strip().lower(), number.strip()
        if not separator or key not in ("cpu", "mem") or key in pin or not re.fullmatch(r"[1-9][0-9]{0,4}", number):
            raise ValueError("AESTER_SIRIL_PIN must look like 'cpu=16,mem=16' "
                             "(either part optional, positive integers, mem in GiB)")
        pin[key] = int(number)
    return pin


def _commands(pin):
    lines = []
    if pin and "cpu" in pin:
        lines.append("setcpu %d" % pin["cpu"])
    if pin and "mem" in pin:
        lines += ["set core.mem_mode=1", "set core.mem_amount=%d" % pin["mem"]]
    return lines


def _environment(pin):
    return {"OMP_NUM_THREADS": str(pin["cpu"])} if pin and "cpu" in pin else {}


def patch_siril(module):
    """Wrap an older starstack.siril; no-op when it supports the pin natively."""
    if getattr(module, "SIRIL_PIN_ENV", None) == "AESTER_SIRIL_PIN" or getattr(module, "_aester_bench_pinned", False):
        return False
    import contextlib
    import threading
    original_run, original_guard = module.run_script, module.guarded_process
    lock = threading.Lock()

    @contextlib.contextmanager
    def guarded_process(args, **kwargs):
        pin = _parse(os.environ.get("AESTER_SIRIL_PIN"))
        overrides = _environment(pin)
        if pin and kwargs.get("stdout") is not None and hasattr(kwargs["stdout"], "write"):
            note = "; ".join(_commands(pin) + ["%s=%s" % item for item in overrides.items()])
            kwargs["stdout"].write(("aester: AESTER_SIRIL_PIN applied: %s\n" % note).encode("utf-8"))
        with contextlib.ExitStack() as stack:
            if overrides:
                with lock:
                    saved = {name: os.environ.get(name) for name in overrides}
                    os.environ.update(overrides)
                    try:
                        process = stack.enter_context(original_guard(args, **kwargs))
                    finally:
                        for name, previous in saved.items():
                            if previous is None:
                                os.environ.pop(name, None)
                            else:
                                os.environ[name] = previous
            else:
                process = stack.enter_context(original_guard(args, **kwargs))
            yield process

    def run_script(executable, job_dir, name, commands, cancel_check, **kwargs):
        pin = _parse(os.environ.get("AESTER_SIRIL_PIN"))
        commands = list(commands)
        if pin:
            head = 1 if commands and commands[0] == "offline" else 0
            commands[head:head] = _commands(pin)
        return original_run(executable, job_dir, name, commands, cancel_check, **kwargs)

    module.guarded_process = guarded_process
    module.run_script = run_script
    module._aester_bench_pinned = True
    return True


def _after(fullname, module):
    """What the hook does once a watched module has executed (each step only when its variable is set)."""
    if fullname == "starstack":
        _record(module)
    elif fullname == "starstack.siril":
        if _SHIM:
            patch_siril(module)
        if _PYTHON_ONLY:
            patch_siril_environment(module)
    elif fullname == "starstack.performance":
        patch_performance(module)
    elif fullname == "numpy" and _BLAS_LOG:
        log_openblas()


def _watched():
    names = set()
    if _LOG:
        names.add("starstack")
    if _SHIM or _PYTHON_ONLY:
        names.add("starstack.siril")
    if any(view for _, _, view in _VIEWS):
        names.add("starstack.performance")
    if _BLAS_LOG:
        names.add("numpy")
    return names


def _install():
    import importlib.abc
    import importlib.machinery
    watched = _watched()

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in watched:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                _after(fullname, module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())
    for fullname in sorted(watched):           # already imported (the bench process loads this file late)
        if fullname in sys.modules and fullname != "starstack":
            _after(fullname, sys.modules[fullname])


if _watched():
    try:
        _install()
    except Exception:
        pass
if _THREAD_LOG:
    try:
        start_thread_audit()
    except Exception:
        pass
'''


def write_hook(hook_dir: Path) -> Path:
    hook_dir.mkdir(parents=True, exist_ok=True)
    path = hook_dir / "sitecustomize.py"
    path.write_text(HOOK_SOURCE, encoding="utf-8", newline="\n")
    return path


def load_hook(path: Path):
    spec = importlib.util.spec_from_file_location("aester_bench_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hook_imports(log_dir: Path, code_src: Path, since_ns: int | None = None) -> dict:
    entries = _json_files(Path(log_dir), since_ns)
    outside = [e for e in entries if not _inside(Path(e.get("file", "")).resolve(), code_src)]
    return {"processes": len(entries), "outside_code_root": outside, "ok": bool(entries) and not outside,
            "argv0": sorted({str((e.get("argv") or [""])[0]) for e in entries})}


def _json_files(folder: Path, since_ns: int | None = None) -> list:
    """The hook's JSON files in folder, named <pid>-<time ns>[-kind].json; with since_ns only those written from
    then on (a resumed slot also holds the first segment's)."""
    entries = []
    for item in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if since_ns is not None:
            parts = item.stem.split("-")
            if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) < since_ns:
                continue
        try:
            entries.append(json.loads(item.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return entries


def hook_events(events_dir: Path, plan: dict | None = None, since_ns: int | None = None) -> dict:
    """The hook's events of one run: which processes applied the memory views (and which view functions the code
    root lacks), each process's OpenBLAS configuration, and the Siril environment wrapper."""
    entries = _json_files(Path(events_dir), since_ns)
    views, blas, siril = ([e for e in entries if e.get("event") == kind]
                          for kind in ("view", "openblas", "siril_environment"))
    pins = (plan or {}).get("pins") or {}
    requested = {name: pins.get(key) for name, key in (("available_memory", "memory_view_gib"),
                                                        ("available_commit", "commit_view_gib"))
                 if pins.get(key) is not None}
    status = {}
    for event in views:
        for name, applied in (event.get("applied") or {}).items():
            status.setdefault(name, {}).setdefault(applied.get("status"), set()).add(event.get("pid"))
    view = {name: {state: sorted(pids) for state, pids in sorted(found.items())} for name, found in sorted(status.items())}
    missing = sorted(name for name in requested if "wrapped" not in view.get(name, {}) and "absent" not in view.get(name, {}))
    return {"events": len(entries),
            "views": {"requested_gib": requested, "by_function": view,
                      "processes": [{"pid": e.get("pid"), "argv": e.get("argv"), "applied": e.get("applied")}
                                    for e in views],
                      "absent": sorted(name for name, found in view.items() if "absent" in found),
                      "never_seen": missing,
                      "ok": not missing and all(set(found) <= {"wrapped", "absent"} for found in view.values())},
            "openblas": [{"pid": e.get("pid"), "argv": e.get("argv"), "config": e.get("config"),
                          "corename": e.get("corename"), "num_threads": e.get("num_threads"),
                          "environment": e.get("environment"), "error": e.get("error")} for e in blas],
            "siril_environment": [{"pid": e.get("pid"), "status": e.get("status"), "names": e.get("names")}
                                  for e in siril]}


def thread_audit(log_dir: Path, since_ns: int | None = None) -> dict:
    """The hook's 1 Hz thread audit: per Python process the maxima of frame-pool, render, BLAS-capable and all
    Python threads (and per name prefix)."""
    processes = []
    for entry in _json_files(Path(log_dir), since_ns):
        maxima = entry.get("max") or {}
        processes.append({"pid": entry.get("pid"), "argv": entry.get("argv"), "samples": entry.get("samples"),
                          "frame_pool_max": maxima.get("frame_pool"), "blas_capable_max": maxima.get("blas_capable"),
                          "render_max": maxima.get("render"), "python_threads_max": maxima.get("total"),
                          "max_t": entry.get("max_t"), "max_by_prefix": entry.get("max_by_prefix") or {}})

    def peak(key):
        values = [p[key] for p in processes if isinstance(p.get(key), int)]
        return max(values) if values else None
    return {"processes": processes, "frame_pool_max": peak("frame_pool_max"), "blas_capable_max": peak("blas_capable_max"),
            "python_threads_max": peak("python_threads_max"), "interval_s": 1.0,
            "note": "Python threads by name (threading.enumerate); native OpenBLAS/ORT/OpenMP pools are in the "
                    "sampler's per-process OS thread counts"}


# --------------------------------------------------------------------------- after-run collection

def collect_timings(value, path="$", out=None, limit=4000):
    """Every timing block (keys containing 'timing') and every *seconds number, by JSON path."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            where = f"{path}.{key}"
            if "timing" in str(key).lower() and isinstance(item, (dict, list)):
                out.append({"path": where, "value": item})
            elif str(key).endswith("seconds") and isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append({"path": where, "value": item})
            else:
                collect_timings(item, where, out, limit)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            collect_timings(item, f"{path}[{index}]", out, limit)
    return out


def pin_check(workflow_dir: Path, pin: dict | None) -> dict:
    """Did every Siril script of the run carry exactly the requested header pin?"""
    expected = pin_lines(pin)
    scripts = sorted(workflow_dir.rglob("*.ssf"))
    pinned, unpinned = [], []
    for script in scripts:
        lines = script.read_text(encoding="utf-8", errors="replace").splitlines()
        head = 2 if len(lines) > 1 and lines[1] == "offline" else 1
        has = bool(expected) and lines[head:head + len(expected)] == expected
        header_pin = any(line.startswith(("setcpu ", "set core.mem_")) for line in lines[head:head + 3])
        (pinned if (has if expected else header_pin) else unpinned).append(str(script.relative_to(workflow_dir)))
    notes = 0
    for log in workflow_dir.rglob("*.log"):
        with open(log, "rb") as stream:
            notes += b"aester: AESTER_SIRIL_PIN applied" in stream.read(4096)
    ok = (not unpinned and bool(scripts)) if expected else not pinned
    return {"expected_lines": expected, "scripts": len(scripts), "pinned": pinned if not expected else len(pinned),
            "unpinned": unpinned if expected else len(unpinned), "logs_with_note": notes, "ok": ok}


def siril_reported_processors(workflow_dir: Path) -> dict:
    pattern = re.compile("(?:using|使用)\\s*(\\d+)\\s*(?:logical processors|个逻辑处理器)".encode("utf-8"), re.IGNORECASE)
    counts = {}
    for log in workflow_dir.rglob("*.log"):
        try:
            for match in pattern.finditer(log.read_bytes()):
                counts[match.group(1).decode()] = counts.get(match.group(1).decode(), 0) + 1
        except OSError:
            continue
    return counts


def render_workers_view() -> dict:
    """The product's own render-worker choice for a 45.7 MP frame, as the bench process sees it at submit.
    The workflow's worker decides again later with its own free RAM; the timeline shows its threads."""
    try:
        from starstack import performance
        available = performance.available_memory()
        return {"frame_hw": list(FRAME_HW), "available_bytes": available, "override": os.environ.get(RENDER_WORKERS_ENV),
                "workers": performance.render_workers(*FRAME_HW)}
    except (ImportError, AttributeError, ValueError) as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def submitter_hashing(before=None) -> dict:
    """Hashing done in this process by ``submit_workflow`` (the submitter: a GUI/MCP process in production).

    The workflow and job records hold only their own workers' hashing, so efficiency P1 criterion H4
    (real reads and ledger hits against the P0 trace, which includes the submitter) adds this part.
    Call it once before the submit (the running totals) and once after with those (the change).  A code
    root without the P1 counters reports hashed_* only.
    """
    try:
        from starstack import profiling
    except ImportError as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    now = {}
    for name in ("hashed_totals", "hash_saved_totals", "hashed_memory_totals"):
        reader = getattr(profiling, name, None)
        if reader is None:
            continue
        try:
            now.update(reader())
        except Exception as exc:  # diagnostics only
            now[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    if before is None:
        return now

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return {key: (value - before.get(key, 0) if number(value) and number(before.get(key, 0)) else value)
            for key, value in now.items()}


def _tail_copy(source: Path, target: Path, limit=2 * 1024 * 1024):
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as stream:
        stream.seek(max(0, source.stat().st_size - limit))
        target.write_bytes(stream.read())
    return str(target)


def fingerprint_run(workflow_dir: Path, out_path: Path, tool: Path | None = None) -> dict:
    """Whole-workflow fingerprint via scripts/verify_workflow_equivalence.py, imported lazily."""
    tool = Path(tool or VERIFIER)
    if not tool.is_file():
        return {"status": "skipped", "note": f"{tool.name} is not present; no manifest was written"}
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("aester_verify_workflow_equivalence", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entry = getattr(module, "main", None)
        if callable(entry):
            # The tool's own command line, in this process: its manifest format and its checks.
            import contextlib
            import io
            output, errors = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    code = entry(["fingerprint", str(workflow_dir), "--out", str(out_path)])
            except SystemExit as exc:
                code = exc.code
            if code in (0, None) and out_path.is_file():
                return {"status": "ok", "method": "main(['fingerprint', ...])", "manifest": str(out_path),
                        "report": output.getvalue()[-4000:], "seconds": round(time.perf_counter() - started, 1)}
            return {"status": "error", "method": "main", "returncode": code, "stderr": errors.getvalue()[-4000:],
                    "stdout": output.getvalue()[-2000:]}
        function = getattr(module, "fingerprint", None)
        if callable(function):
            kwargs = {"progress": None} if "progress" in inspect.signature(function).parameters else {}
            try:
                manifest = function(str(workflow_dir), **kwargs)
            except TypeError:
                manifest = None
            if isinstance(manifest, dict):
                write_json(out_path, manifest, exclusive=True)
                return {"status": "ok", "method": "fingerprint()", "manifest": str(out_path),
                        "files": len(manifest.get("files", {})), "seconds": round(time.perf_counter() - started, 1)}
        run = subprocess.run([sys.executable, "-X", "utf8", str(tool), "fingerprint", str(workflow_dir),
                              "--out", str(out_path)], capture_output=True, text=True, timeout=6 * 3600)
        if run.returncode == 0 and out_path.is_file():
            return {"status": "ok", "method": "cli", "manifest": str(out_path), "stdout": run.stdout[-2000:],
                    "seconds": round(time.perf_counter() - started, 1)}
        return {"status": "error", "method": "cli", "returncode": run.returncode, "stderr": run.stderr[-4000:]}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def fingerprint_job(job_dir: Path, out_path: Path, tool: Path | None = None, *, input_dir=None) -> dict:
    """A stack job's fingerprint (--job; prereg 6.3 W19-W21 "the comparison tool's job-directory mode"): the verifier's
    own fingerprint() and manifest format, with the job folder walked as the {JOB} root (its parent {CACHE}, the
    input folder {INPUT}) where a workflow would be; its compare then reads the manifest like any other (the job.json
    timing receipts are its {JOB}/job.json diagnostic slot; strict ids from the shared {CACHE} and {INPUT}).  The
    verifier file is not changed (A16): this module's copy of it gets a job-folder discover()."""
    tool = Path(tool or VERIFIER)
    if not tool.is_file():
        return {"status": "skipped", "note": f"{tool.name} is not present; no manifest was written"}
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location("aester_verify_workflow_equivalence_job", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        job = Path(job_dir).resolve(strict=True)
        record = module._read_json_file(job / "job.json")
        out_path = Path(out_path).resolve()
        if module._inside(out_path, job):
            raise ValueError("the manifest must not be written inside the job folder")

        def discover(_folder, extra_roots=None):
            roots = {"{JOB}": str(job)}
            if module._usable_root(job.parent):
                roots["{CACHE}"] = str(job.parent)
            if input_dir and module._usable_root(input_dir):
                roots["{INPUT}"] = os.path.normpath(str(input_dir))
            roots["{WF}"] = str(job)          # fingerprint() records it as workflow_dir; dropped below
            return record, roots, [("{JOB}", job)], []

        module.discover = discover
        manifest = module.fingerprint(str(job))
        manifest["roots"].pop("{WF}", None)
        manifest.update(workflow_dir=None, workflow_state=None, job_dir=str(job), job_state=record.get("state"),
                        fingerprint_mode="job directory (bench_workflow.fingerprint_job)")
        module._write_json_file(out_path, manifest)
        return {"status": "ok", "method": "fingerprint() with the job folder as {JOB} (job-directory mode)",
                "manifest": str(out_path), "files": len(manifest["files"]),
                "bindings_mismatch": len(manifest["bindings"]["mismatch"]),
                "seconds": round(time.perf_counter() - started, 1)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- the run

class _Shared:
    stage = None
    hook = None             # the import hook loaded into this process (stopped before the slot is deleted)


class _Driven:
    """What a run submits and follows: the workflow (workflow.submit_workflow, the GUI's path; cancel and resume), or
    with --job one stack job (core.submit_job, the CLI/MCP path; job.json is its record)."""

    def __init__(self, plan: dict):
        self.job = plan.get("job")
        if self.job:
            from starstack import core as product
            self.submit, self.status = product.submit_job, product.load_job
            self.cancel = lambda directory: product.cancel_job(str(directory))
            self.dir_key, self.record_name = "job_dir", "job.json"
        else:
            from starstack import workflow as product
            self.submit, self.status, self.cancel = (product.submit_workflow, product.workflow_status,
                                                     product.cancel_workflow)
            self.dir_key, self.record_name = "workflow_dir", "workflow.json"
        self.product, self.terminal = product, product.TERMINAL


def _stop_hook():
    hook, _Shared.hook = _Shared.hook, None
    if hook is not None:
        try:
            hook.stop()
        except Exception:
            pass


def _apply_environment(environment: dict) -> dict:
    """Set/unset the plan's variables for this process and every child."""
    applied = {}
    for name, value in environment.items():
        if value is None:
            if name in os.environ:
                applied[name] = f"<removed {os.environ[name]!r}>"
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
            applied[name] = value
    return applied


def effective_environment(plan: dict, inherited: dict) -> dict:
    """What a run's processes see: the inherited relevant variables with the plan's settings applied (names upper
    case: Windows variables ignore case, and os.environ reports OPENSSL_ia32cap as OPENSSL_IA32CAP)."""
    env = {k.upper(): v for k, v in (inherited or {}).items() if RELEVANT_ENV.match(k)}
    for name, value in (plan.get("environment") or {}).items():
        if value is None:
            env.pop(name.upper(), None)
        else:
            env[name.upper()] = value
    return env


# --------------------------------------------------------------------------- --cancel-when and --resume

class CancelWatch:
    """--cancel-when: when the record field first matches, wait DELAY seconds, then cancel_workflow; afterwards the
    first moments the workflow (and its stack job) say 'cancelled' and the first moment no descendant lives.
    Times are seconds since the submit (or the resume) call, like the run's other wall times."""

    def __init__(self, spec: dict, read_json):
        self.spec, self.read_json = spec, read_json
        self.condition_t = self.requested_t = self.cancel_call_s = None
        self.workflow_cancelled_t = self.job_cancelled_t = self.quiet_t = None
        self.descendants_at_cancelled = None
        self.value_seen = _MISSING
        self.reads = self.read_errors = 0
        self.cancel_error = None

    def _read(self, path: Path):
        try:
            record = self.read_json(path)
        except (OSError, ValueError):
            self.read_errors += 1
            return None
        self.reads += 1
        return record

    def _path(self, directory, status) -> Path | None:
        if self.spec["record"] == "workflow":
            return Path(directory) / "workflow.json"
        job = status.get("stack_job_dir")
        return Path(job) / "job.json" if job else None

    def due(self, directory, status, now: float) -> bool:
        """True once: the condition has held (first seen at condition_t) for DELAY seconds."""
        if self.requested_t is not None:
            return False
        if self.condition_t is None:
            path = self._path(directory, status)
            record = self._read(path) if path is not None and path.is_file() else None
            if record is None:
                return False
            value = record_field(record, self.spec["field"])
            if not field_matches(value, self.spec["value"]):
                return False
            self.condition_t, self.value_seen = now, value
        return now - self.condition_t >= self.spec["delay_s"]

    def observe(self, status, now: float, descendants) -> None:
        if self.requested_t is None:
            return
        if self.workflow_cancelled_t is None and status.get("state") == "cancelled":
            self.workflow_cancelled_t = now
            self.descendants_at_cancelled = descendants()
        if self.job_cancelled_t is None and status.get("stack_job_dir"):
            job = self._read(Path(status["stack_job_dir"]) / "job.json")
            if job is not None and job.get("state") == "cancelled":
                self.job_cancelled_t = now

    def report(self, final_state) -> dict:
        def at(t):
            return None if t is None else round(t, 3)

        def since(t):
            return None if t is None or self.requested_t is None else round(t - self.requested_t, 3)
        reasons = []
        if self.requested_t is None:
            reasons.append("the condition never held before the run ended; nothing was cancelled")
        else:
            if self.cancel_error:
                reasons.append(f"cancel_workflow failed: {self.cancel_error}")
            if final_state != "cancelled":
                reasons.append(f"the workflow ended {final_state!r}, not 'cancelled'")
            if self.quiet_t is None:
                reasons.append("descendants were still alive when --quiesce-seconds ran out (the bench's Job Object "
                               "kills them when it exits)")
        return {"spec": self.spec, "triggered": self.requested_t is not None,
                "value_seen": None if self.value_seen is _MISSING else self.value_seen,
                "condition_at_s": at(self.condition_t), "requested_at_s": at(self.requested_t),
                "cancel_call_s": self.cancel_call_s, "workflow_cancelled_at_s": at(self.workflow_cancelled_t),
                "job_cancelled_at_s": at(self.job_cancelled_t), "no_descendants_at_s": at(self.quiet_t),
                "latency": {"to_cancelled_s": since(self.workflow_cancelled_t),
                            "to_job_cancelled_s": since(self.job_cancelled_t),
                            "to_no_descendants_s": since(self.quiet_t),
                            "cancelled_to_no_descendants_s": None if self.quiet_t is None
                            or self.workflow_cancelled_t is None else round(self.quiet_t - self.workflow_cancelled_t, 3)},
                "descendants_at_cancelled": self.descendants_at_cancelled, "poll_s": CANCEL_POLL_SECONDS,
                "record_reads": self.reads, "record_read_errors": self.read_errors, "final_state": final_state,
                "valid": not reasons, "invalid_reasons": reasons}


def _read_json_retrying(path: Path, retry_seconds: float = 2.0):
    """records.read_json's rule for a code root without it: retry only Windows sharing denials, to a deadline."""
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(.02)


def read_slot_marker(slot: Path) -> dict:
    try:
        marker = json.loads((slot / SLOT_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"{slot}: no readable bench slot marker ({exc})") from exc
    if marker.get("slot") != slot.name:
        raise BenchError(f"{slot}: the marker names slot {marker.get('slot')!r}")
    return marker


def load_resume_source(bench_root, mode: str, frames: int) -> dict:
    """The first segment of a --resume: the slot kept by a valid --cancel-when run, its plan and its summary."""
    bench = ensure_allowed(bench_root, "bench root")
    slot = bench / "runs" / f"{mode}{frames}"
    if not slot.is_dir():
        raise BenchError(f"--resume: there is no kept slot {slot}")
    marker = read_slot_marker(slot)
    if not marker.get("resumable"):
        reasons = (marker.get("cancel") or {}).get("invalid_reasons") or ["not left by a valid --cancel-when run"]
        raise BenchError(f"--resume: slot {slot} is not resumable: {'; '.join(reasons)}")
    run_id = (marker.get("segments") or [None])[-1]
    results = bench / "results" / str(run_id)
    try:
        plan = json.loads((results / "plan.json").read_text(encoding="utf-8"))
        summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"--resume: cannot read the first segment's plan/summary in {results} ({exc})") from exc
    if not (summary.get("cancel") or {}).get("valid"):
        raise BenchError(f"--resume: the first segment {run_id} has no valid cancel record")
    directory = summary.get("workflow_dir")
    if not directory or not _inside(directory, slot) or not (Path(directory) / "workflow.json").is_file():
        raise BenchError(f"--resume: the first segment's workflow {directory!r} is not in {slot}")
    return {"slot": str(slot), "marker": marker, "run_id": run_id, "results": str(results), "plan": plan,
            "summary": summary, "workflow_dir": directory}


def replay_knobs(args, first_plan: dict) -> dict:
    """Fill the knobs the first segment was given on its command line; a knob given again must agree."""
    replayed = {}
    for knob, value in (first_plan.get("explicit_knobs") or {}).items():
        if knob not in RESUME_KNOBS:
            continue
        current = getattr(args, knob, None)
        if current is None:
            setattr(args, knob, value)
            replayed[knob] = value
        elif not _same_knob(knob, current, value):
            raise BenchError(f"--resume: --{knob.replace('_', '-')} {current!r} differs from the first segment's "
                             f"{value!r}")
    return replayed


def resume_differences(first_plan: dict, first_summary: dict, plan: dict, inherited: dict) -> list:
    """Why a second segment would not run in the first segment's environment ([] = the same). Only the code root
    (and what follows from it: PYTHONPATH's code entry, the pin shim) may differ."""
    problems = []
    for key in ("mode", "frames", "slot", "profile", "pins", "submit", "siril_launch", "hook"):
        if first_plan.get(key) != plan.get(key):
            problems.append(f"{key}: first {first_plan.get(key)!r}, now {plan.get(key)!r}")
    first = effective_environment(first_plan, first_summary.get("inherited_environment") or {})
    now = effective_environment(plan, inherited)
    for name in sorted((set(first) | set(now)) - set(CODE_ROOT_KEYS) - {REEXEC_MARK}):
        if first.get(name) != now.get(name):
            problems.append(f"environment {name}: first {first.get(name)!r}, now {now.get(name)!r}")
    if (first.get("PYTHONPATH") or "").split(os.pathsep)[1:] != (now.get("PYTHONPATH") or "").split(os.pathsep)[1:]:
        problems.append("PYTHONPATH: the hook entries differ")
    return problems


def update_slot_marker(slot: Path, run_id: str, cancel: dict | None) -> dict:
    """Record a kept segment in the slot marker; resumable only after a valid cancel."""
    marker = read_slot_marker(slot)
    segments = [s for s in (marker.get("segments") or [marker.get("run_id")]) if s]
    if run_id not in segments:
        segments.append(run_id)
    marker.update(segments=segments, resumable=bool(cancel and cancel.get("valid")),
                  cancel=None if cancel is None else {k: cancel.get(k) for k in ("valid", "invalid_reasons",
                                                                                 "latency", "spec")},
                  updated_utc=utc_now().isoformat())
    write_json(slot / SLOT_MARKER, marker)
    return marker


# --------------------------------------------------------------------------- foreign CPU

def cpu_reading(win, job, own_handle) -> dict:
    return {"system": win.system_times(), "job": win.job_accounting(job) if job is not None else None,
            "bench_cpu_s": win.times(own_handle)[1] if own_handle else None, "t": time.perf_counter()}


def foreign_cpu(start: dict, end: dict, sampler_tree_cpu_s=None) -> dict:
    """Busy CPU time of the whole machine minus the run tree's CPU time over the run (prereg 6.6: a run whose mean
    foreign CPU exceeds one core is rerun once and both are reported)."""
    wall = end["t"] - start["t"]
    busy0, busy1 = Win.busy_seconds(start["system"]), Win.busy_seconds(end["system"])
    if busy0 is None or busy1 is None or wall <= 0:
        return {"status": "unavailable"}
    if start.get("job") and end.get("job"):
        def job_cpu(reading):
            return reading["job"]["cpu_user_s"] + reading["job"]["cpu_kernel_s"]
        tree, source = job_cpu(end) - job_cpu(start), "job accounting"
    elif sampler_tree_cpu_s is not None and start.get("bench_cpu_s") is not None:
        tree, source = sampler_tree_cpu_s - start["bench_cpu_s"], "sampler (short-lived processes may be missed)"
    else:
        return {"status": "unavailable", "note": "no tree CPU reading"}
    system = busy1 - busy0
    foreign = system - tree
    cores = foreign / wall
    return {"status": "ok", "method": end["system"].get("method"), "processors": end["system"].get("processors"),
            "wall_s": round(wall, 3), "system_busy_cpu_s": round(system, 3), "tree_cpu_s": round(tree, 3),
            "tree_source": source, "foreign_cpu_s": round(foreign, 3), "foreign_cores_mean": round(cores, 3),
            "over_one_core": cores > 1.0,
            "rule": "prereg 6.6: a run whose mean foreign CPU exceeds one core is rerun once; both are reported"}


# --------------------------------------------------------------------------- the run

def run(plan: dict, args, resume: dict | None = None) -> int:
    if os.name != "nt":
        raise BenchError("the bench measures Windows processes; run it on Windows")
    paths = {k: Path(v) for k, v in plan["paths"].items()}
    runs, slot = paths["runs"], paths["slot"]
    if resume is not None:
        if _abs(resume["slot"]) != _abs(slot):
            raise BenchError(f"--resume: the first segment's slot {resume['slot']} is not {slot}")
        problems = resume_differences(resume["plan"], resume["summary"], plan, dict(os.environ))
        if problems:
            raise BenchError("--resume: this run's environment differs from the first segment's: "
                             + "; ".join(problems[:8]) + (f" (+{len(problems) - 8} more)" if len(problems) > 8 else ""))
    elif slot.exists():
        if not plan["replace"]:
            raise BenchError(f"{slot} exists (an earlier --keep run?); pass --replace to delete it first")
        safe_remove(slot, runs)
    inputs = verify_inputs(plan["inputs"]["manifest"], plan["inputs"]["inputs_dir"], plan["frames"])
    verify_assets(paths["bench_root"])
    if resume is None:
        free = shutil.disk_usage(paths["bench_root"] if paths["bench_root"].exists() else REPO).free
        if free < plan["disk_estimate_bytes"]:
            raise BenchError(f"free disk {free / GIB:.1f} GiB < estimated {plan['disk_estimate_bytes'] / GIB:.1f} GiB")
    results = paths["results"]
    try:
        results.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BenchError(f"{results} already exists; wait a second or change --label") from exc
    if resume is None:
        runs.mkdir(parents=True, exist_ok=True)
        slot.mkdir(exist_ok=False)
        write_json(slot / SLOT_MARKER, {"schema": "aester-bench-slot-1", "slot": slot.name, "run_id": plan["run_id"],
                                        "created_utc": utc_now().isoformat(), "pid": os.getpid()}, exclusive=True)
    summary = {"schema": SUMMARY_SCHEMA, "run_id": plan["run_id"], "plan": plan, "started_utc": utc_now().isoformat(),
               "host": {"cpu_count": os.cpu_count(), "python": sys.version, "executable": sys.executable},
               "inherited_environment": {k: v for k, v in os.environ.items() if RELEVANT_ENV.match(k)}}
    if resume is not None:
        # both segments on record: the first's plan and summary travel with the second's results
        segments = results / "segments"
        segments.mkdir()
        write_json(segments / "1-plan.json", resume["plan"], exclusive=True)
        write_json(segments / "1-summary.json", resume["summary"], exclusive=True)
        summary["resume"] = {"first_run_id": resume["run_id"], "first_results": resume["results"],
                             "workflow_dir": resume["workflow_dir"], "first_code_root": resume["plan"].get("code_root"),
                             "code_root": plan["code_root"], "first_code": resume["plan"].get("code"),
                             "replayed_knobs": plan.get("replayed_knobs") or {},
                             "first_cancel": resume["summary"].get("cancel"),
                             "segments": list(resume["marker"].get("segments") or []) + [plan["run_id"]]}
    write_json(results / "plan.json", plan, exclusive=True)
    try:
        return _run_started(plan, args, paths, inputs, summary, resume)
    except BaseException as exc:
        _stop_hook()
        if "sampler" not in summary:          # failed before the workflow was submitted: nothing to keep
            summary.update(state=None, setup_error=f"{type(exc).__name__}: {exc}", finished_utc=utc_now().isoformat())
            write_json(results / "summary.json", summary)
            if resume is None:                # a resumed slot is the first segment's: never deleted here
                safe_remove(slot, runs)
        raise


def _run_started(plan, args, paths, inputs, summary, resume=None) -> int:
    runs, slot, results = paths["runs"], paths["slot"], paths["results"]
    segment_ns = time.time_ns()               # hook logs of this segment (a resumed slot holds the first's too)
    if resume is None:
        for name in ("cache", "delivery", "components", "siril_xdg"):
            paths[name].mkdir()
    win = Win()
    summary["affinity_mask"] = hex(win.set_own_affinity(plan["pins"]["affinity_logical"])) \
        if plan["pins"]["affinity_logical"] else None
    summary["environment_applied"] = _apply_environment(plan["environment"])
    desktop = plan["siril_launch"]["XDG_DATA_HOME"] if plan["siril_launch"].get("mode") == "desktop-sentinel" else None
    if desktop:
        summary["siril_desktop"] = {"path": desktop, "before": folder_listing(desktop)}
    hook = load_hook(write_hook(paths["hook"])) if plan["hook"] else None   # here too; children via PYTHONPATH
    _Shared.hook = hook
    if plan["pins"]["mask_sha"]:
        summary["sha_check"] = sha_check()
    starstack = _import_starstack(Path(plan["code_src"]))
    driven = _Driven(plan)
    try:
        from starstack import records as product_records
        read_json = getattr(product_records, "read_json", None) or _read_json_retrying
    except ImportError:
        read_json = _read_json_retrying
    summary["starstack_file"] = starstack.__file__
    summary["simulation"] = plan.get("simulation")
    summary["render_workers_at_start"] = render_workers_view()
    if plan["pins"]["siril"] and plan["native_pin_support"]:
        from starstack import siril
        summary["siril_pin_diagnostics"] = siril.pin_diagnostics()
    project = project_file(paths["bench_root"], plan["frames"], manifest_path=plan["inputs"]["manifest"],
                           inputs_dir=plan["inputs"]["inputs_dir"], baseline=inputs)
    summary["inputs"] = {k: v for k, v in inputs.items() if k != "frames"} | {"files": [Path(f["path"]).name for f in inputs["frames"]]}
    summary["project"] = {"path": str(project), "sha256": sha256_file(project)}
    try:
        job = win.own_job()
        summary["job_accounting"] = {"state": "active"}
    except OSError as exc:
        job = None
        summary["job_accounting"] = {"state": "unavailable", "error": str(exc)}
    shared = _Shared()
    sampler = Sampler(win, results / "timeline.jsonl", interval=args.sample_seconds, stage=lambda: shared.stage,
                      disk_path=str(paths["bench_root"]), expected_affinity=int(summary["affinity_mask"], 16)
                      if summary["affinity_mask"] else None).start()
    watch = CancelWatch(plan["cancel_when"], read_json) if plan.get("cancel_when") else None
    transitions, wall = [], {}
    state, directory, interrupted = None, None, False
    own = sampler.procs[os.getpid()].handle
    cpu_start = cpu_reading(win, job, own)
    t0 = time.perf_counter()
    try:
        submitter_before = submitter_hashing()
        if resume is not None:
            status = driven.product.resume_workflow(resume["workflow_dir"])
        else:
            status = driven.submit(inputs["inputs_dir"], str(paths["cache"]),
                                   input_files=[f["path"] for f in inputs["frames"]], **plan["submit"])
        wall["submit_call_s"] = round(time.perf_counter() - t0, 3)
        summary["submitter_hashing"] = submitter_hashing(submitter_before)
        directory = Path(status[driven.dir_key])
        summary[driven.dir_key] = str(directory)
        deadline = t0 + args.timeout_minutes * 60
        last, failures, full_read = None, 0, time.perf_counter()
        while True:
            key = (status.get("state"), status.get("stage"))
            now = round(time.perf_counter() - t0, 3)
            if key != last:
                transitions.append({"t": now, "state": key[0], "stage": key[1], "message": status.get("message")})
                shared.stage = f"{key[0]}/{key[1]}"
                last = key
            if "submit_to_base_s" not in wall and status.get("base_delivery_dir"):
                wall["submit_to_base_s"] = now
            if watch is not None:
                watch.observe(status, time.perf_counter() - t0, sampler.descendants_alive)
                if status.get("state") not in driven.terminal and watch.due(directory, status,
                                                                             time.perf_counter() - t0):
                    watch.requested_t = time.perf_counter() - t0
                    try:
                        status = driven.product.cancel_workflow(directory)
                    except Exception as exc:  # recorded; the run then ends as it would have
                        watch.cancel_error = f"{type(exc).__name__}: {exc}"
                    watch.cancel_call_s = round(time.perf_counter() - t0 - watch.requested_t, 3)
                    transitions.append({"t": round(watch.requested_t, 3), "state": "cancel_requested_by_bench",
                                        "stage": key[1], "message": watch.spec["text"]})
                    continue
            if status.get("state") in driven.terminal and not status.get("trail_addback_pending"):
                break
            if time.perf_counter() > deadline:
                interrupted = "timeout"
                driven.cancel(directory)
                break
            time.sleep(CANCEL_POLL_SECONDS if watch is not None else args.poll_seconds)
            try:
                if watch is not None and time.perf_counter() - full_read < args.poll_seconds:
                    # between the full status reads (stale-worker detection probes worker.lock, which a worker
                    # starting at that instant would find taken) only the record itself, at 0.1 s
                    status = read_json(directory / "workflow.json")
                else:
                    status, full_read = driven.status(directory), time.perf_counter()
                failures = 0
            except (OSError, ValueError) as exc:     # record replaced mid-read; retry
                failures += 1
                if failures > 40:
                    raise BenchError(f"cannot read the {driven.record_name} record: {exc}") from exc
        wall["submit_to_final_s"] = round(time.perf_counter() - t0, 3)
        if "submit_to_base_s" in wall:
            wall["addback_s"] = round(wall["submit_to_final_s"] - wall["submit_to_base_s"], 3)
        state = status.get("state")
    except (KeyboardInterrupt, Exception) as exc:
        interrupted = "keyboard" if isinstance(exc, KeyboardInterrupt) else "error"
        summary["bench_error"] = f"{type(exc).__name__}: {exc}"
        if directory is not None:
            try:
                driven.cancel(directory)
            except Exception:
                pass
    finally:
        quiet_started = time.perf_counter()
        fine = watch is not None and watch.requested_t is not None
        while True:
            alive = sampler.descendants_alive()
            if not alive:
                if fine:
                    watch.quiet_t = time.perf_counter() - t0
                break
            if time.perf_counter() - quiet_started >= args.quiesce_seconds:
                break
            time.sleep(CANCEL_POLL_SECONDS if fine else 1)
        wall["quiesce_s"] = round(time.perf_counter() - quiet_started, 3)
        cpu_end = cpu_reading(win, job, own)
        if desktop:
            after = folder_listing(desktop)
            summary["siril_desktop"].update(after=after, identical=after == summary["siril_desktop"]["before"])
        summary["sampler"] = sampler.stop()
        if job is not None:
            summary["job_accounting"] = {"state": "ok", **win.job_accounting(job),
                                         "note": "whole tree incl. the bench process; see sampler.bench_process_cpu_s"}
        summary["foreign_cpu"] = foreign_cpu(cpu_start, cpu_end, summary["sampler"].get("tree_cpu_s"))
    summary.update(state=state, interrupted=interrupted, wall=wall, transitions=transitions,
                   finished_utc=utc_now().isoformat())
    if watch is not None:
        summary["cancel"] = watch.report(state)
    _stop_hook()                              # this process's last audit write; nothing more goes into the slot
    if plan["hook"]:
        summary["code_root_check"] = hook_imports(paths["hook"] / "imports", Path(plan["code_src"]), since_ns=segment_ns)
        summary["hook_events"] = hook_events(paths["hook"] / "events", plan, since_ns=segment_ns)
        summary["thread_audit"] = thread_audit(paths["hook"] / "threads", since_ns=segment_ns)
    keep_reason = None
    if directory is not None and (directory / driven.record_name).is_file():
        records = results / "records"
        records.mkdir()
        if driven.job:                        # a --job run: the stack job is the run's directory, no workflow
            record, job_dir = None, directory
        else:
            record = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
            shutil.copyfile(directory / "workflow.json", records / "workflow.json")
            _tail_copy(directory / "worker.log", records / "workflow-worker.log")
            job_dir = Path(record["stack_job_dir"]) if record.get("stack_job_dir") else None
        job_record = None
        if job_dir is not None and (job_dir / "job.json").is_file():
            shutil.copyfile(job_dir / "job.json", records / "job.json")
            _tail_copy(job_dir / "worker.log", records / "job-worker.log")
            job_record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        summary["error"] = (record if record is not None else job_record or {}).get("error")
        summary["record_times"] = {
            **({"workflow": {k: v for k, v in record.items() if k.endswith("_at")}} if record is not None else {}),
            "job": {k: v for k, v in (job_record or {}).items() if k.endswith("_at")}}
        summary["record_timings"] = {**({"workflow": collect_timings(record)} if record is not None else {}),
                                     "job": collect_timings(job_record or {})}
        if record is not None:
            summary["trail_review"] = {k: (record.get("trail_review") or {}).get(k) for k in
                                       ("state", "kept_count", "kept_ids", "automatic_addback",
                                        "automatic_addback_error", "message")}
        summary["siril_pin_check"] = pin_check(directory, plan["pins"]["siril"])
        summary["siril_reported_processors"] = siril_reported_processors(directory)
        summary["disk_usage"] = disk_usage(slot)
        if state == "succeeded" and not interrupted:
            summary["fingerprint"] = (fingerprint_job(directory, results / "manifest.json",
                                                      input_dir=plan["inputs"]["inputs_dir"]) if driven.job
                                      else fingerprint_run(directory, results / "manifest.json"))
            if summary["fingerprint"]["status"] == "error":
                keep_reason = "fingerprint failed; rerun the verifier on the kept run"
        else:
            summary["fingerprint"] = {"status": "skipped",
                                      "note": f"{'job' if driven.job else 'workflow'} state {state}, "
                                              f"interrupted={interrupted}"}
    cancelled = watch is not None and watch.requested_t is not None
    if cancelled:
        keep_reason = ("--cancel-when: kept for --resume" if summary["cancel"]["valid"] else
                       "--cancel-when: invalid cancel, not resumable (" + "; ".join(summary["cancel"]["invalid_reasons"]) + ")")
    if interrupted:
        keep_reason = f"interrupted ({interrupted}); processes may still hold files"
    summary["deleted"] = False
    if plan["keep"] or keep_reason:
        summary["kept"] = {"path": str(slot), "reason": "--keep" if plan["keep"] else keep_reason}
        if slot.is_dir():
            try:
                summary["slot_marker"] = update_slot_marker(slot, plan["run_id"], summary.get("cancel") if cancelled
                                                            else None)
            except (OSError, BenchError) as exc:
                summary["slot_marker"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        write_json(results / "summary.json", summary)
        try:
            safe_remove(slot, runs)
            summary["deleted"] = True
        except (OSError, BenchError) as exc:
            summary["kept"] = {"path": str(slot), "reason": f"deletion failed: {type(exc).__name__}: {exc}"}
    write_json(results / "summary.json", summary)
    print(json.dumps({"run_id": plan["run_id"], "state": state, "wall": wall, "results": str(results),
                      "fingerprint": (summary.get("fingerprint") or {}).get("status"), "deleted": summary["deleted"],
                      "simulation": (plan.get("simulation") or {}).get("label"),
                      **({"cancel": {k: summary["cancel"][k] for k in ("triggered", "valid", "latency")}}
                         if watch is not None else {}),
                      "foreign_cores_mean": (summary.get("foreign_cpu") or {}).get("foreign_cores_mean")},
                     ensure_ascii=False))
    if watch is not None:
        return 0 if cancelled and summary["cancel"]["valid"] and not interrupted else 1
    return 0 if state == "succeeded" and not interrupted else 1


# --------------------------------------------------------------------------- CLI

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-assets", action="store_true", help="copy the user's region drawing once (read-only)")
    action.add_argument("--check-environment", action="store_true", help="verify affinity inheritance / SHA mask")
    parser.add_argument("--mode", choices=sorted(MODES), help="L legacy, F finite_validity_v1, R positive_power2_v1")
    parser.add_argument("--frames", type=int, default=48, help="first N baseline frames (default 48)")
    parser.add_argument("--label", help="suffix of the run id, e.g. R1a")
    parser.add_argument("--pin-siril", help="AESTER_SIRIL_PIN value, e.g. cpu=16,mem=16")
    parser.add_argument("--pin-ort", type=int, help="backend_options.intra_op_threads")
    parser.add_argument("--affinity", type=int, help="restrict to the first N logical processors (1..64)")
    parser.add_argument("--mask-sha", action="store_true", default=None,
                        help=f"OPENSSL_ia32cap={SHA_MASK} for every process")
    parser.add_argument("--render-workers", type=int, help=f"{RENDER_WORKERS_ENV}=N (1..{MAX_RENDER_WORKERS})")
    parser.add_argument("--frame-workers", help=f"{FRAME_WORKERS_ENV}=auto|1..{MAX_FRAME_WORKERS}|max (P1b frame pool)")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="consumer-PC simulation preset (see PROFILES)")
    parser.add_argument("--governor", help=f"{GOVERNOR_ENV}: shadow, shadow-full (bench default) or off")
    parser.add_argument("--governor-facts", help=f"{GOVERNOR_FACTS_ENV}: what-if machine for the governor's "
                                                 "recorded decisions, e.g. the laptop of --profile 6900hx-proxy")
    parser.add_argument("--parallel", choices=PARALLEL_MODES,
                        help="P1c: set every parallelism variable (PARALLEL_TABLE); a disagreeing knob is refused")
    parser.add_argument("--overlap", choices=sorted(OVERLAP_VALUES), help=f"{OVERLAP_ENV}=auto (on) or off (O-1)")
    parser.add_argument("--trails", choices=("on", "off"),
                        help="off: no trail scan and no trail_default (the GUI/MCP default); default on")
    parser.add_argument("--memory-view", metavar="GIB",
                        help=f"performance.available_memory -> min(real, GIB) in every Python process ({MEMORY_VIEW_ENV})")
    parser.add_argument("--commit-view", metavar="GIB",
                        help=f"performance.available_commit -> min(real, GIB) where the code root has it ({COMMIT_VIEW_ENV})")
    parser.add_argument("--python-openblas", metavar="CORETYPE:THREADS",
                        help="OPENBLAS_CORETYPE / OPENBLAS_NUM_THREADS for the Python processes only, e.g. Haswell:16")
    parser.add_argument("--cancel-when", metavar="RECORD:FIELD=VALUE[+DELAY]",
                        help="cancel the workflow DELAY s after workflow.json or job.json FIELD first equals VALUE; "
                             "the slot is kept for --resume")
    parser.add_argument("--resume", action="store_true",
                        help="continue the slot kept by a valid --cancel-when run (another --code-root allowed); the "
                             "first segment's knobs are replayed and any other difference is refused")
    parser.add_argument("--job", choices=sorted(JOB_KINDS),
                        help="submit one stack job as the CLI/MCP do (core.submit_job) instead of a workflow: "
                             "strict-legacy = strict support + legacy stack (K-SL20, W19-W21); slot S<mode><frames>")
    parser.add_argument("--code-root", help="run starstack from another checkout (its src/)")
    parser.add_argument("--siril-desktop", metavar="DIR",
                        help="end-to-end check (P1b W5): run without MSYSTEM and with XDG_DATA_HOME=DIR, a sentinel "
                             "desktop-Siril data folder (never the real one); the summary lists DIR before and after")
    parser.add_argument("--keep", action="store_true", help="do not delete the run slot")
    parser.add_argument("--replace", action="store_true", help="delete a slot left by an earlier --keep run first")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan; do not run")
    parser.add_argument("--bench-root", default=str(DEFAULT_BENCH_ROOT))
    parser.add_argument("--inputs-dir", default=str(DEFAULT_INPUTS))
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--source-cache", help="prepare-assets source (default %%LOCALAPPDATA%%\\Aester\\Cache)")
    parser.add_argument("--regions-revision", default=USER_REGIONS_REVISION)
    parser.add_argument("--timeout-minutes", type=float, default=360.)
    parser.add_argument("--quiesce-seconds", type=float, default=300.)
    parser.add_argument("--poll-seconds", type=float, default=.5)
    parser.add_argument("--sample-seconds", type=float, default=1.)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        resume = None
        if args.resume:
            if args.replace or args.prepare_assets or args.check_environment or not args.mode:
                raise BenchError("--resume needs --mode and --frames of the kept slot, and no --replace")
            if args.job:
                raise BenchError("--job runs have no --cancel-when / --resume (the cancel points of prereg 6.4 are "
                                 "workflow runs)")
            resume = load_resume_source(args.bench_root, args.mode, args.frames)
            args.replayed_knobs = replay_knobs(args, resume["plan"])
        apply_profile(args)
        if args.prepare_assets:
            record = prepare_assets(args.bench_root, source_cache=args.source_cache, revision=args.regions_revision,
                                    manifest_path=args.baseline_manifest, inputs_dir=args.inputs_dir)
            print(json.dumps({"assets": str(_abs(args.bench_root) / "assets"), "files": len(record["files"]),
                              "mask_check": record.get("mask_check")}, ensure_ascii=False))
            return 0
        if args.check_environment:
            print(json.dumps(check_environment(args), ensure_ascii=False, indent=1))
            return 0
        if not args.mode:
            raise BenchError("choose --mode L|F|R (or --prepare-assets / --check-environment)")
        wanted = SHA_MASK if args.mask_sha else ""
        inherited = os.environ.get("OPENSSL_ia32cap", "")
        if not args.dry_run and args.mask_sha and inherited != SHA_MASK and not os.environ.get(REEXEC_MARK):
            # OpenSSL reads the mask once per process: restart so this process hashes masked too.
            command = [sys.executable] + (["-X", "utf8"] if sys.flags.utf8_mode else []) + [str(Path(__file__).resolve()), *argv]
            child = subprocess.Popen(command, env=dict(os.environ, OPENSSL_ia32cap=SHA_MASK, **{REEXEC_MARK: "1"}))
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:   # the child handles Ctrl+C itself (cancel, record, keep)
                    continue
        if not args.dry_run and inherited != wanted:
            raise BenchError(f"inherited OPENSSL_ia32cap={inherited!r} does not match this run; unset it or use --mask-sha")
        plan = resolve_plan(args)
        if resume is not None:
            plan.update(resume_of=resume["run_id"], replayed_knobs=getattr(args, "replayed_knobs", {}))
            # the explicit knobs of a resumed segment are the first segment's (replayed or given again)
            plan["explicit_knobs"] = resume["plan"].get("explicit_knobs") or {}
            problems = resume_differences(resume["plan"], resume["summary"], plan, dict(os.environ))
            if problems:
                raise BenchError("--resume: this run's environment differs from the first segment's: "
                                 + "; ".join(problems[:8]))
        if args.dry_run:
            inputs = verify_inputs(plan["inputs"]["manifest"], plan["inputs"]["inputs_dir"], plan["frames"])
            try:
                verify_assets(plan["paths"]["bench_root"])
                assets = "prepared"
            except BenchError as exc:
                assets = f"not prepared: {exc}"
            project = Path(plan["paths"]["project"])
            print(json.dumps({"dry_run": True, "plan": plan, "assets": assets,
                              "project": "exists" if project.exists() else "created at run time",
                              "inputs_verified": inputs["verified"],
                              "slot_exists": Path(plan["paths"]["slot"]).exists()}, ensure_ascii=False, indent=1))
            return 0
        return run(plan, args, resume)
    except BenchError as exc:
        print(f"bench_workflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
