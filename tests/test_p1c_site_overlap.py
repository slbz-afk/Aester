"""Efficiency P1c O-1 site tests: the trail scan beside the post-stack work (docs/EFFICIENCY_P1C_PREREG_20261005.md
§3.6, §5 items 2/5/6/7/9, §6.2 (j)).

Jobs run through ``core.run_job`` on a controlled engine (never Siril): the job's own registration (a fitted
calibration with a real SIP distortion.wcs), L with a sky mask (the coverage pool runs in A) or strict support (G-4t in
A), Siril's fallback registration for the ineligible routes. The scan children are real guarded processes: a fast
stand-in child (``CHILD``) that publishes the results the real scan gives on these frames (taken once from a serial run
with the real children), so that B's timing, failures and holds can be steered; one test runs the real children
throughout. Records are compared through scripts/verify_workflow_equivalence.py's canonicalisation (its job-directory
mode, as bench_workflow.fingerprint_job uses it).

- eligibility: every branch of prereg §3.6 in order (trails off, fallback registration, past the stack, STACKCNT 2,
  invalid marks, overlap off / AESTER_FRAME_WORKERS=1 / in a pool unit, memory unknown, memory_tight, commit_tight,
  slower_than_serial, an override under memory pressure) and each guard raising while it is evaluated
  (overlap_error:<type>); the plan's figures (prereg §3.6 and the 6900HX / ordinary-PC what-ifs of §0 item 3);
- C1 equality: O-1 on, AESTER_OVERLAP=off, AESTER_FRAME_WORKERS=1 and max, with B ending before the native remap,
  during the diagnostics and after checkpoint('stack'), at 1, 2 and 4 scan processes; L and strict;
- the join guards (each mismatch, a comparison that raises): B discarded, its output removed, the serial body runs;
- cross-stage failures and their priority (C3): A fails (native remap, master check, coverage) while the scan runs or
  after it failed; STACKCNT < 3; B fails first (scan error, ledger mismatch) and A later: A's error wins; resumed, the
  job ends as an uninterrupted one;
- cancellation (C5): during the scan, during the join wait, after checkpoint('stack') with B running, at the pre-join
  check; within the bounds of prereg §5 item 5, no scan process left, the resumed job equals an uninterrupted one;
- the single writer: checkpoint(), list-field _update and progress only from the main thread; B forced to finish
  during checkpoint('stack');
- reservations and pools: A's pieces released in order, the crop pool on B's thread shares the 16-slot budget with the
  coverage pool, nothing reserved afterwards; the note overlap.o1 only under timings;
- reclamation: B's failed outcome is collected once logged (weakref);
- the forced kill of the job worker during O-1: no scan process survives; the recovered job equals an uninterrupted
  one;
- fallback registration (with and without strict support, with per-frame masks): ineligible, and the scan's grid
  master path and SHA-256 at its start equal the serial reference while B's start is jittered across the in-place
  rewrite;
- C9: C1, C3 and C5 with 50 random jitters each: canonical equality, no thread or process leak, no lock timeout.

The Beside helper's own semantics (value, exception, abandon, repeatable, context, pools inside sharing the budget and
abandoned midway, reclamation) are tests/test_p1c_frame_pool.py's (P1c-0); this file tests them through O-1.
"""
from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from starstack import core, frame_pool, hashing, imaging, performance, profiling, siril
from starstack import registered_support as rs
from starstack import registration_calibration as rc
from starstack import transient_scan as scan
from starstack import transient_stage as ts
import d4_fakes
import p1c_site_harness as h

pytestmark = pytest.mark.usefixtures("pre_contract_jobs")

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
VERIFIER = ROOT / "scripts" / "verify_workflow_equivalence.py"
H, W, FRAMES = 128, 192, 6
SKY_ROWS = 96
GiB = 1 << 30
REGISTER = "register light -2pass -transf=homography"
APPLY = "seqapplyreg light -filter-included"
# frame index -> x0, y0, x1, y1 (in the sky rows); the detector finds and measures both at this size
TRAILS = {2: (10.0, 8.0, 0.85 * W, 0.4 * H), 4: (0.15 * W, 0.6 * H, 0.95 * W, 0.1 * H)}
REAL_ESTIMATES = core._o1_estimates
REAL_WORKER_COMMAND = scan._worker_command


# ================================================================== the controlled world

def write_sources(folder: Path, *, height=H, width=W, frames=FRAMES) -> Path:
    """Frames with a star and, in two of them, a bright trail across the sky rows."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261006)
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    for k in range(frames):
        data = (0.03 + rng.normal(0, 0.002, (height, width, 3))).astype(np.float64)
        data[5:8, 10:13] += 0.2
        if k in TRAILS:
            x0, y0, x1, y1 = TRAILS[k]
            ux, uy = x1 - x0, y1 - y0
            length = float(np.hypot(ux, uy))
            along = ((xx - x0) * ux + (yy - y0) * uy) / length
            across = (-(xx - x0) * uy + (yy - y0) * ux) / length
            trail = np.where((along >= 0) & (along <= length), 0.25 * np.exp(-across ** 2 / 2.0), 0.0)
            data += trail[..., None]
        imaging.write_linear(folder / f"frame {k}.fit", data.astype(np.float32), "Test fixture")
    return folder


def write_mask(path: Path, *, height=H, width=W) -> Path:
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[SKY_ROWS, :] = 128
    mask[SKY_ROWS + 1:, :] = 0
    Image.fromarray(mask).save(path)
    return path


def write_frame_masks(folder: Path, *, height=H, width=W, frames=FRAMES) -> Path:
    """Per-frame sky masks (RUN_LOG §26) for two of the frames: a hole each."""
    folder.mkdir(parents=True, exist_ok=True)
    for k in (1, 3):
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[SKY_ROWS + 1:, :] = 0
        mask[10:16, 30 + 10 * k:40 + 10 * k] = 0
        Image.fromarray(mask).save(folder / f"frame {k}.png")
    return folder


def fitted_calibration(work, **kwargs):
    """What registration_calibration.calibrate returns for a fitted term (the own route): a benign radial model and one
    matrix per frame (tests/test_core_two_pass_reference), with a real SIP distortion.wcs (strict exports read it)."""
    from test_core_two_pass_reference import JobDiagnosticsTest
    from test_strict_support_kernel import _sip_header
    stems = {f"light_{i:05d}" for i in range(1, FRAMES + 1)}
    summary = JobDiagnosticsTest._fake_calibration(work, stems, count=FRAMES)
    summary["parameters"] = {"k1": 2e-3, "k2": 0.0, "cx": (W - 1) / 2, "cy": H / 2 - .5, "scale": float(H),
                             "magnification": .99, "direction": "u_to_p"}
    (Path(work) / rc.DISTORTION_JSON).write_text(json.dumps(summary), encoding="utf-8")
    (Path(work) / rc.DISTORTION_WCS).unlink(missing_ok=True)
    header = _sip_header(np.random.default_rng(31), W, H, 3, 1e-3)
    fits.PrimaryHDU(header=fits.Header(header)).writeto(Path(work) / rc.DISTORTION_WCS)
    return summary


def unavailable_calibration(work, **kwargs):
    summary = {"status": "unavailable", "reason": "test: Siril's own registration"}
    (Path(work) / rc.DISTORTION_JSON).write_text(json.dumps(summary), encoding="utf-8")
    return summary


class Engine:
    """Siril's file contract for the scripts these jobs launch (never Siril). ``stackcnt``: the STACKCNT the stack
    writes (default: the frames stacked); ``hooks[name]``: called after that script."""

    def __init__(self, stackcnt=None):
        self.names, self.stackcnt, self.hooks = [], stackcnt, {}

    def __call__(self, executable, directory, name, commands, cancel_check, **options):
        from test_core_two_pass_reference import _seq_text
        commands, work = list(commands), Path(directory) / "work"
        self.names.append(name)
        (Path(directory) / f"{name}.ssf").write_text(siril.script_text(commands), encoding="utf-8")
        cancel_check()
        if name == "01_convert":
            for i, path in enumerate(sorted((Path(directory) / "stage").glob("*")), 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif name == "02_probe":
            assert commands == core.PROBE_SCRIPT
        elif name == "02_register_fallback":
            (work / core.SEQUENCE_FILE).write_text(_seq_text(FRAMES, 1), encoding="utf-8")
        elif name == "02_export":
            if REGISTER in commands:                      # Siril's own two-pass registration writes the sequence
                (work / core.SEQUENCE_FILE).write_text(_seq_text(FRAMES, 1), encoding="utf-8")
            selected = [image["stem"] for image in core.read_seq(work / core.SEQUENCE_FILE)["images"]
                        if image["selected"]]
            for number, stem in enumerate(selected, 1):
                shutil.copyfile(work / f"{stem}.fit", work / f"r_light_{number:05d}.fit")
            (work / "r_light_.seq").write_text(d4_fakes.sequence_text(len(selected), name="r_light_"),
                                               encoding="utf-8")
        elif name == "02_stack":
            frames = sorted(work.glob("r_light_*.fit"))
            data = np.mean([imaging.read_rgb(p) for p in frames], axis=0).astype(np.float32)
            imaging.write_linear(work / "sky_master.fit", data, "Controlled engine, not Siril",
                                 extra={"STACKCNT": len(frames) if self.stackcnt is None else self.stackcnt})
        else:
            raise AssertionError(name)
        hook = self.hooks.get(name)
        if hook is not None:
            hook(Path(directory))


# ------------------------------------------------------------------ the stand-in scan child

CHILD = r'''
import json, os, sys, time
from pathlib import Path
library, request = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads(library.read_text(encoding="utf-8"))
payload = json.loads(request.read_text(encoding="utf-8"))
signals = Path(plan["signals"])
import ctypes
def created(pid):
    if os.name != "nt":
        return 0
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    k.OpenProcess.restype = ctypes.c_void_p
    k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.OpenProcess(0x1000, False, pid)
    values = [ctypes.c_uint64() for _ in range(4)]
    try:
        k.GetProcessTimes(h, *(ctypes.byref(v) for v in values))
    finally:
        k.CloseHandle(h)
    return values[0].value
name = request.parent.name + "-" + request.stem
staged = signals / (name + ".tmp")
staged.write_text(json.dumps({"pid": os.getpid(), "created": created(os.getpid())}), encoding="utf-8")
os.replace(staged, signals / (name + ".started"))
def publish(path, value):
    tmp = path.with_suffix(".pending")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
deadline = time.monotonic() + 60
for job in payload["jobs"]:
    stem = job["stem"]
    delay = plan.get("delays", {}).get(stem, plan.get("delay", 0))
    if delay:
        time.sleep(delay)
    gate = plan.get("gate")
    while gate and not Path(gate).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if stem in plan.get("fail", []):
        publish(Path(job["error"]), {"version": 1, "index": job["index"], "stem": stem, "type": "ValueError",
                                     "message": "injected scan failure"})
        sys.exit(1)
    publish(Path(job["result"]), {"version": 1, "stem": stem, "scanned": plan["scanned"][stem]})
'''


class Children:
    """The scan children's command (scan._worker_command) as the stand-in, steered through a plan file; each child
    reports its identity in ``signals`` (outside the job, so the job's tree is the real scan's)."""

    def __init__(self, folder: Path, scanned: dict):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.script = self.folder / "child.py"
        self.script.write_text(CHILD, encoding="utf-8")
        self.library = self.folder / "plan.json"
        self.signals = self.folder / "signals"
        self.scanned = scanned
        self.set()

    def set(self, **plan):
        if self.signals.exists():
            shutil.rmtree(self.signals)
        self.signals.mkdir()
        gate = plan.pop("gate", None)
        self.gate = None if gate is None else self.folder / "gate"
        if self.gate is not None:
            self.gate.unlink(missing_ok=True)
        self.library.write_text(json.dumps({"signals": str(self.signals), "scanned": self.scanned,
                                            "gate": None if self.gate is None else str(self.gate), **plan}),
                                encoding="utf-8")

    @classmethod
    def attach(cls, folder: Path) -> "Children":
        """The same children's command in another process (the forced-kill test's job worker); writes nothing."""
        self = cls.__new__(cls)
        self.folder = Path(folder)
        self.script, self.library, self.signals = self.folder / "child.py", self.folder / "plan.json", \
            self.folder / "signals"
        self.scanned, self.gate = None, None
        return self

    def open_gate(self):
        if self.gate is not None:
            self.gate.write_text("open", encoding="utf-8")

    def command(self, request):
        return [sys.executable, "-I", str(self.script), str(self.library), str(request)]

    def started(self) -> list:
        out = []
        for path in sorted(self.signals.glob("*.started")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return out

    def wait_started(self, count=1, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.started()) >= count:
                return True
            time.sleep(0.01)
        raise AssertionError("no scan child started")

    def alive(self) -> list:
        """Children (by pid and creation time) still running."""
        return [identity for identity in self.started() if _alive(identity)]


def _alive(identity) -> bool:
    if os.name != "nt":
        try:
            os.kill(identity["pid"], 0)
            return True
        except OSError:
            return False
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    k.OpenProcess.restype = ctypes.c_void_p
    k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.WaitForSingleObject.restype = ctypes.c_uint32
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k.OpenProcess(0x00101000, False, identity["pid"])
    if not handle:
        return False
    try:
        values = [ctypes.c_uint64() for _ in range(4)]
        if not k.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)) or values[0].value != identity["created"]:
            return False                                  # a recycled pid is another process
        return k.WaitForSingleObject(handle, 0) == 258
    finally:
        k.CloseHandle(handle)


def wait_until(predicate, timeout=15.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    raise AssertionError("bounded test wait expired")


# ------------------------------------------------------------------ the verifier's job-directory comparison

_VERIFIER = None


def verifier():
    global _VERIFIER
    if _VERIFIER is None:
        spec = importlib.util.spec_from_file_location("aester_verify_o1_tests", VERIFIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _VERIFIER = module
    return _VERIFIER


_MANIFESTS = {}


def fingerprint(job_dir, input_dir=None) -> dict:
    """bench_workflow.fingerprint_job's manifest: the verifier's fingerprint() with the job folder as {JOB} (kept per
    folder and record state: a finished reference is compared many times)."""
    job = Path(job_dir).resolve()
    stat = (job / "job.json").stat()
    key = (str(job), str(input_dir), stat.st_mtime_ns, stat.st_size)
    if key not in _MANIFESTS:
        _MANIFESTS[key] = _fingerprint(job, input_dir)
    return _MANIFESTS[key]


def _fingerprint(job: Path, input_dir) -> dict:
    module = verifier()
    record = module._read_json_file(job / "job.json")

    def discover(_folder, extra_roots=None):
        roots = {"{JOB}": str(job)}
        if module._usable_root(job.parent):
            roots["{CACHE}"] = str(job.parent)
        if input_dir is not None and module._usable_root(input_dir):
            roots["{INPUT}"] = os.path.normpath(str(input_dir))
        roots["{WF}"] = str(job)
        return record, roots, [("{JOB}", job)], []
    saved = module.discover
    module.discover = discover
    try:
        manifest = module.fingerprint(str(job), workers=2)
    finally:
        module.discover = saved
    manifest["roots"].pop("{WF}", None)
    return manifest


RESUMED_JOB_FIELD = {"path": "/worker_pid", "rule": "process id of the worker process"}
# What any resumed job's record has that an uninterrupted one lacks, serial or overlapped (the R_resume of prereg §6.4,
# here for the controlled engine): the attempt resumed past the stack checkpoint reports the sequence it found
# (core's ``sequence_registration is None`` branch) instead of the one it wrote.
RESUMED_JOB_PATHS = ("/json/diagnostics/registration_detail/sequence",)


def _resume_only(result) -> bool:
    """job.json differs only as every resumed job does: the worker_pid key core.resume_job writes (None here: the
    tests' _spawn_worker starts nothing, so an uninterrupted job never has it; a real run has it on both sides) and
    RESUMED_JOB_PATHS."""
    if result.get("class") != "json" or set(result) - {"verdict", "class", "json", "one_sided_fields",
                                                       "one_sided_diagnostic_fields"}:
        return False
    fields = result.get("one_sided_fields") or []
    return (all({k: f.get(k) for k in RESUMED_JOB_FIELD} == RESUMED_JOB_FIELD for f in fields)
            and all(d.get("path", "").startswith(RESUMED_JOB_PATHS) for d in result.get("json") or []))


def canonical_problems(reference_dir, job_dir, input_dir=None, *, resumed=False) -> list:
    """What the verifier's comparison finds between two job folders (empty: equal under its canonicalisation).
    ``resumed``: the second job was resumed (its worker_pid key, RESUMED_JOB_FIELD, is not a difference)."""
    ma, mb = fingerprint(reference_dir, input_dir), fingerprint(job_dir, input_dir)
    report = verifier().compare(ma, mb, strict_ids=False)
    if report["pass"]:
        return []
    problems = []
    # A FITS-input job's signature preimage carries fits_input.frames[*].exposure_seconds, a key the verifier excludes
    # everywhere (it flags that for any FITS job, built for RAW workflows): the same on both sides, it is not O-1's.
    exclusions = report["excluded_keys_in_preimages"]
    if exclusions["a"] != exclusions["b"] or any(not item["path"].startswith("/fits_input/")
                                                 for item in exclusions["a"]):
        problems.append(f"excluded keys in preimages: {exclusions['a'][:2]} / {exclusions['b'][:2]}")
    report["excluded_keys_in_preimages"] = {"a": [], "b": []}
    for key in ("only_in_a", "only_in_b", "dirs_only_in_a", "dirs_only_in_b", "links_different"):
        if report[key]:
            problems.append(f"{key}: {report[key][:8]}")
    for key in ("different", "errors"):
        for name, result in list(report[key].items())[:8]:
            if resumed and key == "different" and name.endswith("job.json") and _resume_only(result):
                continue
            problems.append(f"{key}: {name}: {json.dumps(result, default=str)[:600]}")
    for side in ("a", "b"):
        if report["derived_digests"][side]["failed"]:
            problems.append(f"derived digests {side}: {report['derived_digests'][side]['failed'][:4]}")
        if report["missing_deliveries"][side]:
            problems.append(f"missing deliveries {side}")
    for side, manifest in (("a", ma), ("b", mb)):
        if manifest["bindings"]["mismatch"]:
            problems.append(f"binding mismatch {side}: {manifest['bindings']['mismatch'][:4]}")
    return problems                       # every condition of the verifier's verdict is listed above


# ------------------------------------------------------------------ the world fixture

class World:
    def __init__(self, tmp_path: Path, monkeypatch, scanned: dict | None):
        self.tmp, self.monkeypatch = tmp_path, monkeypatch
        self.source = write_sources(tmp_path / "sources")
        self.mask = write_mask(tmp_path / "sky_mask.png")
        self.frame_masks = write_frame_masks(tmp_path / "frame_masks")
        info = {"available": True, "executable": str(tmp_path / "engine.exe"), "version": "1.4.4",
                "message": "controlled"}
        monkeypatch.setattr(core, "doctor", lambda *_: dict(info))
        monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
        monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 ** 12))
        monkeypatch.setattr(rs, "require_engine", lambda engine: None)
        monkeypatch.setattr(rc, "calibrate", fitted_calibration)
        self.engine = Engine()
        monkeypatch.setattr(core, "run_script", self.engine)
        self.children = None
        if scanned is not None:
            self.children = Children(tmp_path / "children", scanned)
            monkeypatch.setattr(scan, "_worker_command", self.children.command)
        self.machine()
        self.o1_criterion(True)
        for name in (frame_pool.ENV_WORKERS, frame_pool.ENV_OVERLAP, ts.WORKERS_ENV):
            monkeypatch.delenv(name, raising=False)
        self.started = []
        monkeypatch.setattr(core, "_o1_started", self.started.append)

    def machine(self, physical=64 * GiB, commit=128 * GiB):
        """performance's readings (net of in-process reservations, R20)."""
        self.monkeypatch.setattr(performance, "_memory_status", lambda: (physical, commit))

    def o1_criterion(self, passes: bool):
        """The time criterion on these tiny frames: A's estimate scales with the pixels, so it never passes unaided."""
        self.monkeypatch.setattr(core, "_o1_estimates", (lambda mode, pixels, frames: (frames * 0.01, 1000.0))
                                 if passes else REAL_ESTIMATES)

    def env(self, **values):
        for name, value in values.items():
            if value is None:
                self.monkeypatch.delenv(name, raising=False)
            else:
                self.monkeypatch.setenv(name, str(value))

    def submit(self, label, *, strict=False, own=True, frame_masks=False, marks=None, mode="review",
               key=None):
        self.monkeypatch.setattr(rc, "calibrate", fitted_calibration if own else unavailable_calibration)
        options = {"sky_mask_path": str(self.mask), "transient_mode": mode}
        if strict:
            options["support_mode"] = rs.MODE
        if frame_masks:
            options["frame_masks_dir"] = str(self.frame_masks)
        if marks is not None:
            options["transient_marks"] = marks
        job = core.submit_job(str(self.source), str(self.tmp / label), idempotency_key=key or "o1", **options)
        return job["job_dir"]

    def run(self, label, **submit):
        directory = self.submit(label, **submit)
        core.run_job(directory)
        return core.load_job(directory)

    def resume(self, directory):
        core.resume_job(directory)
        core.run_job(directory)
        return core.load_job(directory)


def note_of(job) -> dict:
    return (((job.get("diagnostics") or {}).get("timings") or {}).get("notes") or {}).get(core.O1_NOTE)


def succeeded(job):
    assert job["state"] == "succeeded", job.get("error")
    return job


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """The real scan's per-frame results on these frames (one serial job with the real scan children)."""
    from _pytest.monkeypatch import MonkeyPatch
    patch = MonkeyPatch()
    tmp = tmp_path_factory.mktemp("o1-scanned")
    try:
        from starstack import stack_settings
        patch.setattr(stack_settings, "admission_field", lambda *a, **k: None)
        world = World(tmp, patch, None)
        world.env(AESTER_OVERLAP="off")
        job = succeeded(world.run("reference"))
        runs = sorted((Path(job["job_dir"]) / "work" / ts.TRANSIENTS_DIR / "scan-runs").iterdir())
        assert len(runs) == 1
        results = {}
        for path in sorted(runs[0].glob("result-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            results[payload["stem"]] = payload["scanned"]
        assert len(results) == FRAMES
        assert sum(len(r["result"]["segments"]) for r in results.values()) >= 2, "the frames must give candidates"
        return results
    finally:
        patch.undo()


@pytest.fixture
def world(tmp_path, monkeypatch, scanned):
    w = World(tmp_path, monkeypatch, scanned)
    yield w
    if w.children is not None:
        w.children.open_gate()
    assert frame_pool._budget_used == 0 and frame_pool.reserved_bytes() == 0
    leftover = [t.name for t in threading.enumerate() if t.name.startswith("aester-beside")]
    assert not leftover, leftover


# ================================================================== helpers

def wrap(monkeypatch, target, name, *, before=None, after=None):
    """Replace target.name with a wrapper calling ``before(*args)`` and ``after(result, *args)`` around it."""
    real = getattr(target, name)

    def wrapper(*args, **kwargs):
        if before is not None:
            before(*args, **kwargs)
        result = real(*args, **kwargs)
        if after is not None:
            after(result, *args, **kwargs)
        return result
    monkeypatch.setattr(target, name, wrapper)
    return real


def reference(world, label="reference", **submit):
    """The serial run (AESTER_OVERLAP=off) of the same job."""
    saved = os.environ.get(frame_pool.ENV_OVERLAP)
    world.env(AESTER_OVERLAP="off")
    try:
        return world.run(label, **submit)
    finally:
        world.env(AESTER_OVERLAP=saved)


def overlap_of(world):
    return world.started[-1] if world.started else None


def b_done(world) -> bool:
    o1 = overlap_of(world)
    return o1 is not None and o1.helper is not None and o1.helper.done()


def no_trace_of_b(job_dir) -> bool:
    work = Path(job_dir) / "work"
    return not (work / ts.TRANSIENTS_DIR).exists() and not (work / core.O1_MARKER).exists()


def scan_runs(job_dir) -> list:
    folder = Path(job_dir) / "work" / ts.TRANSIENTS_DIR / "scan-runs"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def equal_to(reference_job, job, world, *, resumed=False):
    problems = canonical_problems(reference_job["job_dir"], job["job_dir"], world.source, resumed=resumed)
    assert problems == [], problems


def aester_threads() -> list:
    return sorted(t.name for t in threading.enumerate() if t.name.startswith("aester-") and t.is_alive()
                  and not t.name.startswith("aester-peak-sampler"))


def quiet(world, timeout=10.0):
    """No helper or pool thread and no scan child left."""
    wait_until(lambda: not aester_threads(), timeout)
    if world.children is not None:
        wait_until(lambda: not world.children.alive(), timeout)


class Cancel:
    """Writes the job's cancel.request when ``when()`` holds (polled on a thread), recording the moment."""

    def __init__(self, directory, when, timeout=30.0):
        self.directory, self.when, self.at = Path(directory), when, None
        self.thread = threading.Thread(target=self._run, args=(timeout,), daemon=True)
        self.thread.start()

    def _run(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.when():
                    self.at = time.monotonic()
                    (self.directory / "cancel.request").touch()
                    return
            except Exception:
                pass
            time.sleep(0.005)


def stage_is(directory, stage):
    def check():
        return json.loads((Path(directory) / "job.json").read_text(encoding="utf-8")).get("stage") == stage
    return check


# ================================================================== the plan on what-if machines (§3.6, §0 item 3)

def what_if(monkeypatch, tmp_path, *, lps, cores, physical, commit, frames=48, height=5520, width=8280):
    """_o1_decide's readings for a 45.7 MP job of ``frames`` frames on a what-if machine (nothing of the test host)."""
    from starstack import resources
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "sky_master_undistorted.fit").write_bytes(b"")
    header = {"NAXIS1": width, "NAXIS2": height, "STACKCNT": frames}
    monkeypatch.setattr(fits, "getheader", lambda path, *a, **k: dict(header))
    stems = [f"light_{i:05d}" for i in range(1, frames + 1)]
    monkeypatch.setattr(ts, "frame_order", lambda folder: list(stems))
    monkeypatch.setattr(core, "registration_transforms", lambda *a, **k: ({s: np.eye(3) for s in stems}, None))
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: lps)
    monkeypatch.setattr(resources, "cpu_topology", lambda **_: SimpleNamespace(python_usable_cores=cores))
    monkeypatch.setattr(performance, "_memory_status", lambda: (physical, commit))
    for name in (frame_pool.ENV_WORKERS, frame_pool.ENV_OVERLAP, ts.WORKERS_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(core, "_o1_estimates", REAL_ESTIMATES)
    job = {"inputs": [{"path": f"in {i}.nef", "size": 1, "sha256": "0" * 64} for i in range(frames)],
           "transient_marks": []}
    return dict(transient_mode="review", own_registration=True, stack_done=False, work=work, job=job,
                calibration={"status": "fitted"}, support_enabled=False, validity_record={"dilation_px": 3})


# (name, lps, cores, physical, commit, mode, reason, n_o1, n_serial); prereg §3.6 "结果：本机和普通电脑档开启；6900HX
# 代理和 16 GB 笔记本关闭"
MACHINES = [
    ("development PC", 192, 96, 200 * GiB, 300 * GiB, "lr", None, 8, 8),
    ("ordinary-PC simulation (memory unlimited)", 16, 8, 1 << 40, 1 << 40, "lr", None, 6, 7),
    ("6900HX proxy v2 (9 GiB view, 14 GiB commit)", 16, 8, 9 * GiB, 14 * GiB, "lr", "slower_than_serial", 1, 4),
    ("16 GB laptop, commit tighter", 16, 8, 9 * GiB, 5 * GiB, "lr", "commit_tight", 1, 4),
    ("16 GB laptop, physical tighter", 16, 8, 3 * GiB, 14 * GiB, "lr", "memory_tight", 1, 4),
    ("32 GiB free", 16, 8, 32 * GiB, 64 * GiB, "lr", None, 5, 6),
    ("development PC, F", 192, 96, 200 * GiB, 300 * GiB, "strict", None, 8, 8),
    ("6900HX proxy v2, F", 16, 8, 9 * GiB, 14 * GiB, "strict", "slower_than_serial", 1, 4),
]


@pytest.mark.parametrize("name, lps, cores, physical, commit, mode, reason, n_o1, n_serial", MACHINES,
                         ids=[m[0] for m in MACHINES])
def test_the_plan_on_what_if_machines(monkeypatch, tmp_path, name, lps, cores, physical, commit, mode, reason,
                                      n_o1, n_serial):
    kwargs = what_if(monkeypatch, tmp_path, lps=lps, cores=cores, physical=physical, commit=commit)
    if mode == "strict":
        kwargs.update(support_enabled=True, native_workers=4, native_unit_bytes=rs.NATIVE_UNIT_BYTES)
    got, plan, frozen = core._o1_decide(**kwargs)
    assert got == reason, plan
    assert (plan["n_o1"], plan["n_serial"]) == (n_o1, n_serial), plan
    assert (frozen is not None) is (reason is None)
    pixels = 5520 * 8280
    unit = ts.scan_unit_bytes(pixels)
    assert plan["scan_unit_bytes"] == unit and plan["pieces"]["scan"] == n_o1 * unit
    if mode == "lr":
        cov = plan["cov_plan"]
        share = cov * (core._coverage_unit_bytes(5520, 8280) + frame_pool.BLAS_SLOT_BYTES)
        assert plan["pieces"]["coverage"] == share and plan["pieces"]["native_mask"] == 0
        assert plan["a_reserve_bytes"] == max(24 * pixels, 12 * pixels + share)
        if cov == 8:                             # the planned maximum covers the measured L48 rise (2.65 GB)
            assert plan["a_reserve_bytes"] >= 2_650_000_000
        assert plan["crop_cap"] == max(1, min(ts.TRAIL_CROP_CAP, frame_pool.MAX_FRAME_WORKERS - cov))
    else:
        assert plan["cov_plan"] == 0 and plan["pieces"]["coverage"] == 0
        assert plan["pieces"]["native_mask"] == 4 * rs.NATIVE_UNIT_BYTES
        assert plan["a_reserve_bytes"] == core.O1_A_STRICT_BYTES + 4 * rs.NATIVE_UNIT_BYTES
    assert plan["b_inproc_bytes"] == plan["crop_cap"] * ts.TRAIL_CROP_UNIT_BYTES + frame_pool.BLAS_SLOT_BYTES
    scan_cpu, a_seconds = REAL_ESTIMATES(mode, pixels, 48)
    assert plan["predicted_s"]["scan_cpu"] == round(scan_cpu, 2) == round(3.6 * 48, 2)
    assert plan["predicted_s"]["a"] == core.O1_A_SECONDS[mode]
    # the two headrooms of the note: what the scan processes leave of each budget after A and B's own part
    for kind, available in (("physical", physical), ("commit", commit)):
        net = available - plan["a_reserve_bytes"] - plan["b_inproc_bytes"]
        assert plan["headroom_bytes"][kind] == core._o1_budget(net) - n_o1 * unit


def test_an_override_still_has_to_fit_the_memory(monkeypatch, tmp_path):
    """prereg §3.6: STARSTACK_TRANSIENT_WORKERS (or AESTER_FRAME_WORKERS) is checked like the planner's count."""
    kwargs = what_if(monkeypatch, tmp_path, lps=16, cores=8, physical=32 * GiB, commit=64 * GiB)
    assert core._o1_decide(**kwargs)[0] is None                         # automatic: 5 processes fit
    monkeypatch.setenv(ts.WORKERS_ENV, "8")
    reason, plan, _ = core._o1_decide(**kwargs)
    assert reason == "memory_tight" and plan["n_o1"] == 8 and plan["override"] == ts.WORKERS_ENV
    monkeypatch.setenv(ts.WORKERS_ENV, "3")
    reason, plan, _ = core._o1_decide(**kwargs)
    assert reason is None and plan["n_o1"] == plan["n_serial"] == 3
    monkeypatch.delenv(ts.WORKERS_ENV)
    monkeypatch.setenv(frame_pool.ENV_WORKERS, "max")                   # the scan's own maximum: 7 on 8 cores
    reason, plan, _ = core._o1_decide(**kwargs)
    assert reason == "memory_tight" and plan["n_o1"] == 7 and plan["override"] == frame_pool.ENV_WORKERS


def test_memory_unknown_turns_o1_off(monkeypatch, tmp_path):
    for physical, commit in ((None, None), (64 * GiB, None), (None, 64 * GiB)):
        kwargs = what_if(monkeypatch, tmp_path, lps=16, cores=8, physical=physical, commit=commit)
        assert core._o1_decide(**kwargs)[0] == "memory_unknown"


# ================================================================== eligibility, branch by branch (§3.6, (j))

@pytest.fixture
def decision(world):
    """_o1_decide's arguments for a finished own-registration L job of the world (its work folder, record and
    calibration), eligible as they are."""
    from starstack import sky_validity as sv
    job = succeeded(world.run("decide"))
    work = Path(job["job_dir"]) / "work"
    kwargs = dict(transient_mode="review", own_registration=True, stack_done=False, work=work, job=job,
                  calibration=rc.load_calibration(work), support_enabled=False, validity_record=sv.read_record(work))
    assert core._o1_decide(**kwargs)[0] is None
    return kwargs


def _header_with(**cards):
    real = fits.getheader

    def getheader(path, *args, **kwargs):
        header = real(path, *args, **kwargs)
        if Path(path).name == "sky_master_undistorted.fit":
            header = fits.Header(header)
            for key, value in cards.items():
                header[key] = value
        return header
    return getheader


BRANCHES = [
    ("trails_off", {"transient_mode": "off"}, None),
    ("fallback_registration", {"own_registration": False}, None),
    ("stack_checkpointed", {"stack_done": True}, None),
    ("overlap_off", {}, {"AESTER_OVERLAP": "off"}),
    ("overlap_off", {}, {"AESTER_OVERLAP": " OFF "}),
    ("serial_workers", {}, {"AESTER_FRAME_WORKERS": "1"}),
    ("stackcnt", "STACKCNT=2", None),
    ("naxis2", "NAXIS2=12.5", None),
    ("marks_invalid", "marks", None),
    ("memory_unknown", "memory", None),
    ("memory_tight", "physical", None),
    ("commit_tight", "commit", None),
    ("slower_than_serial", "estimates", None),
]


@pytest.mark.parametrize("reason, change, env", BRANCHES, ids=[f"{b[0]}-{i}" for i, b in enumerate(BRANCHES)])
def test_each_eligibility_branch(world, decision, monkeypatch, reason, change, env):
    kwargs = dict(decision)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    if isinstance(change, dict):
        kwargs.update(change)
    elif change == "STACKCNT=2":
        monkeypatch.setattr(fits, "getheader", _header_with(STACKCNT=2))
    elif change == "NAXIS2=12.5":
        real = fits.getheader
        monkeypatch.setattr(fits, "getheader", lambda path, *a, **k: (
            {**dict(real(path, *a, **k)), "NAXIS2": 12.5} if Path(path).name.endswith("undistorted.fit")
            else real(path, *a, **k)))
    elif change == "marks":
        kwargs["job"] = {**kwargs["job"], "transient_marks": [{"stem": "not an input", "x0": 1, "y0": 1, "x1": 9,
                                                               "y1": 9}]}
    elif change == "memory":
        world.machine(64 * GiB, None)
    elif change == "physical":
        world.machine(int(1.5 * GiB), 128 * GiB)
    elif change == "commit":
        world.machine(64 * GiB, int(1.5 * GiB))
    elif change == "estimates":
        world.o1_criterion(False)                     # the real estimates: 5 s of margin on a 24.6 kpx job
    got, plan, frozen = core._o1_decide(**kwargs)
    assert got == reason and frozen is None, plan


def test_inside_a_pool_unit_o1_is_off(decision):
    seen = frame_pool.run_ordered(lambda x, unit: core._o1_decide(**decision)[0], range(2), workers=2, name="t-o1")
    assert seen == ["in_pool", "in_pool"]


def test_the_first_failing_condition_is_the_reason(decision, monkeypatch):
    monkeypatch.setenv(frame_pool.ENV_OVERLAP, "off")
    assert core._o1_decide(**{**decision, "transient_mode": "off", "own_registration": False})[0] == "trails_off"
    assert core._o1_decide(**{**decision, "own_registration": False, "stack_done": True})[0] == \
        "fallback_registration"
    assert core._o1_decide(**{**decision, "stack_done": True})[0] == "stack_checkpointed"
    assert core._o1_decide(**decision)[0] == "overlap_off"


GUARDS = [
    ("getheader", lambda m: m.setattr(fits, "getheader", _raise(OSError("header unreadable"))), "OSError"),
    ("available_commit", lambda m: m.setattr(performance, "available_commit", _raise(RuntimeError("x"))),
     "RuntimeError"),
    ("frame_order", lambda m: m.setattr(ts, "frame_order", _raise(ValueError("seq"))), "ValueError"),
    ("registration_transforms", lambda m: m.setattr(core, "registration_transforms", _raise(KeyError("H"))),
     "KeyError"),
    ("coverage decision", lambda m: m.setattr(performance, "frame_workers", _raise(TypeError("x"))), "TypeError"),
    ("scan planner", lambda m: m.setattr(ts, "plan_scan_workers", _raise(ZeroDivisionError())),
     "ZeroDivisionError"),
    ("scan override", lambda m: m.setattr(ts, "scan_override", _raise(ValueError("knob"))), "ValueError"),
    ("grid master", lambda m: m.setattr(ts, "grid_master_path", lambda work, own: Path(work) / "sky_master.fit"),
     "grid_master"),
]


def _raise(exc):
    def raising(*args, **kwargs):
        raise exc
    return raising


@pytest.mark.parametrize("name, patch, kind", GUARDS, ids=[g[0] for g in GUARDS])
def test_a_guard_that_raises_while_deciding_makes_o1_ineligible(decision, monkeypatch, name, patch, kind):
    patch(monkeypatch)
    reason, plan, frozen = core._o1_decide(**decision)
    assert reason == f"overlap_error:{kind}" and frozen is None


def test_ineligible_jobs_note_why_and_run_the_serial_code(world):
    """Job level: trails off (also the GUI/MCP default), fallback registration, a job resumed past the stack, overlap
    off, AESTER_FRAME_WORKERS=1; each notes its reason under timings and starts no helper."""
    cases = {}
    cases["trails_off"] = succeeded(world.run("off", mode="off"))
    cases["fallback_registration"] = succeeded(world.run("fallback", own=False))
    world.env(AESTER_OVERLAP="off")
    cases["overlap_off"] = succeeded(world.run("knob"))
    world.env(AESTER_OVERLAP=None, AESTER_FRAME_WORKERS=1)
    cases["serial_workers"] = succeeded(world.run("serial"))
    world.env(AESTER_FRAME_WORKERS=None)
    assert world.started == []
    for reason, job in cases.items():
        note = note_of(job)
        assert note["eligible"] is False and note["reason"] == reason and note["schema"] == core.O1_NOTE_SCHEMA
    # a job resumed after its stack checkpoint: no overlap, the reason says why
    real_compose, calls = core._compose_sky, []

    def compose(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("stop after the scan")
        return real_compose(*args, **kwargs)
    world.monkeypatch.setattr(core, "_compose_sky", compose)
    failed = world.run("resumed")
    assert failed["state"] == "failed" and "stop after the scan" in failed["error"]
    assert note_of(failed)["eligible"] is True and note_of(failed)["outcome"] == "joined"
    assert failed["checkpoints"][-2:] == ["stack", "transients"]
    resumed = succeeded(world.resume(failed["job_dir"]))
    assert note_of(resumed)["reason"] == "stack_checkpointed" and len(world.started) == 1


# ================================================================== C1: equality, whatever B's schedule (§6.2 (j))

def steer(world, schedule, m):
    """Make B end before the native remap, during the diagnostics, or after checkpoint('stack') (``m``: the
    monkeypatch of this run's wrappers)."""
    if schedule == "before_remap":
        world.children.set()
        wrap(m, rc, "native_from_undistorted", before=lambda *a, **k: wait_until(lambda: b_done(world), 30))
    elif schedule == "in_diagnostics":
        world.children.set(gate=True)

        def diagnostics(*args, **kwargs):
            world.children.open_gate()
            wait_until(lambda: b_done(world), 30)
        wrap(m, core, "exposure_diagnostics", before=diagnostics)
    elif schedule == "after_stack":
        world.children.set(gate=True)

        def stack_checkpoint(result, directory, files):
            if any(Path(f).name == "sky_master.fit" for f in files):
                world.children.open_gate()
        wrap(m, core, "_checkpoint_records", after=stack_checkpoint)
    else:
        world.children.set()


@pytest.mark.parametrize("mode", ["L", "strict"])
def test_c1_overlap_on_equals_the_serial_job_whatever_b_s_schedule_and_count(world, mode):
    """C1 (§6.2 (j)): O-1 on, with B ending before the native remap, during the diagnostics or after
    checkpoint('stack'), at 1, 2 and 4 scan processes; AESTER_FRAME_WORKERS=1 (serial) and max; each equal to the
    serial job (AESTER_OVERLAP=off) under the verifier's canonicalisation."""
    strict = mode == "strict"
    serial = succeeded(reference(world, strict=strict))
    assert note_of(serial)["reason"] == "overlap_off"
    runs = 0
    for schedule in ("before_remap", "in_diagnostics", "after_stack"):
        for workers in (1, 2, 4):
            with pytest.MonkeyPatch.context() as m:
                steer(world, schedule, m)
                m.setenv(ts.WORKERS_ENV, str(workers))
                job = succeeded(world.run(f"{schedule}-{workers}", strict=strict))
            runs += 1
            note = note_of(job)
            assert note["eligible"] is True and note["outcome"] == "joined" and note["n_o1"] == workers, note
            if schedule != "after_stack":
                assert note["b_done_at_join"] is True
            equal_to(serial, job, world)
            assert len(scan_runs(job["job_dir"])) == 1
            assert not (Path(job["job_dir"]) / "work" / core.O1_MARKER).exists()
            quiet(world)
    world.children.set()
    for knob, value, eligible in ((frame_pool.ENV_WORKERS, "1", False), (frame_pool.ENV_WORKERS, "max", True),
                                  (frame_pool.ENV_OVERLAP, "auto", True)):
        with pytest.MonkeyPatch.context() as m:
            m.setenv(knob, value)
            job = succeeded(world.run(f"{knob}-{value}", strict=strict))
        assert note_of(job)["eligible"] is eligible, note_of(job)
        equal_to(serial, job, world)
        quiet(world)
    assert len(world.started) == runs + 2


def test_the_pools_of_an_overlapped_job_leave_no_context_change(world, monkeypatch):
    """§6.2 (e): the frame pools of an O-1 job (coverage on the main thread, crops on B's) under _AUDIT_CONTEXT."""
    serial = succeeded(reference(world))
    monkeypatch.setattr(frame_pool, "_AUDIT_CONTEXT", True)
    job = succeeded(world.run("audited"))
    assert note_of(job)["outcome"] == "joined"
    equal_to(serial, job, world)


def _equality_target(folder, scanned_path):
    """In a fresh interpreter (§6.2 (g) numeric variants, (a) PYTHONHASHSEED): the world's serial job and its O-1 job,
    compared through the verifier; returns its problems and O-1's outcome."""
    from starstack import stack_settings
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(stack_settings, "admission_field", lambda *a, **k: None)
        world = World(Path(folder), patch, json.loads(Path(scanned_path).read_text(encoding="utf-8")))
        serial = succeeded(reference(world))
        job = succeeded(world.run("o1"))
        return {"problems": canonical_problems(serial["job_dir"], job["job_dir"], world.source),
                "outcome": note_of(job)["outcome"]}
    finally:
        patch.undo()


NUMERIC_AND_SEEDS = (*h.NUMERIC_ENVS, {"PYTHONHASHSEED": "1"}, {"PYTHONHASHSEED": "4242"})


@pytest.mark.parametrize("env", NUMERIC_AND_SEEDS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()))
def test_o1_equals_serial_under_numeric_variants_and_hash_seeds(tmp_path, scanned, env):
    """B's in-process work (the crop pool, the evidence) moves from the main thread to the helper thread, never out of
    or into the scan children: under each numeric variant and two hash seeds the O-1 job equals the serial one."""
    path = tmp_path / "scanned.json"
    path.write_text(json.dumps(scanned), encoding="utf-8")
    digest = h.subprocess_digest("test_p1c_site_overlap:_equality_target", str(tmp_path / "w"), str(path), env=env)
    assert digest == h.canonical_digest({"problems": [], "outcome": "joined"})


def test_f48_b_s_in_process_part_beside_the_real_native_remap(tmp_path):
    """(b) real data, the F48 reference job (read only): B's in-process part -- the crop phase on its frame pool with
    O-1's crop cap, then the evidence -- on a Beside helper while the main thread runs A's native remap of the
    45.7 MP undistorted master: B rebuilds the job's candidates record and its 299 crops (written into the basetemp),
    A the job's native master, bit for bit; nothing stays reserved, no helper or pool thread is left."""
    import test_p1c_site_trail_crops as t3
    from starstack.imaging import read_rgb
    f48 = t3.load_f48()
    work = f48["work"]
    calibration = json.loads((work / "distortion.json").read_text(encoding="utf-8"))
    support = json.loads((work / rs.RECORD).read_text(encoding="utf-8"))
    crop_cap = max(1, min(ts.TRAIL_CROP_CAP, frame_pool.MAX_FRAME_WORKERS - 0))      # F: no coverage pool
    crops = tmp_path / "crops"
    seen = {}

    def b(probe):
        seen["thread"] = threading.current_thread().name
        record = t3.f48_record(f48)
        ts._crop_phase(record, f48["scanned"], f48["stems"], f48["master"], crops, probe, work=work,
                       inputs=f48["job"]["inputs"], transforms=f48["transforms"], model=f48["model"],
                       own_registration=True, cap=crop_cap)
        record["seconds"] = f48["fixture"]["seconds"]
        return ts._finish(record, work, persist=False, check=probe)
    started = time.perf_counter()
    helper = frame_pool.Beside.start(b, "o1-f48", check=None)
    try:
        native = rc.native_from_undistorted(read_rgb(work / "sky_master_undistorted.fit"), calibration["parameters"])
        remap_seconds = time.perf_counter() - started
        record = helper.result()
    finally:
        helper.abandon()
    native[~rs.load_array(work, support["native_valid"])[::-1]] = 0.0
    expected = read_rgb(work / "sky_master.fit")
    assert native.dtype == expected.dtype and np.array_equal(native.view(np.uint32), expected.view(np.uint32))
    assert t3.assert_rebuilds_the_fixture(f48, record, crops) == 299
    assert seen["thread"] == "aester-beside-o1-f48"
    assert frame_pool.reserved_bytes() == 0 and frame_pool._budget_used == 0
    wait_until(lambda: not aester_threads(), 10)
    print(f"F48: native remap {remap_seconds:.1f} s beside B's crops and evidence; "
          f"together {time.perf_counter() - started:.1f} s")


def test_a_real_scan_overlapped_equals_the_serial_job(world, scanned):
    """The real scan children (transient_scan_worker) under O-1 and serially: equal, and the candidates' crops were
    written on B's thread through the crop pool (T-3) within O-1's cap."""
    world.monkeypatch.setattr(scan, "_worker_command", REAL_WORKER_COMMAND)
    serial = succeeded(reference(world))
    job = succeeded(world.run("real"))
    note = note_of(job)
    assert note["eligible"] is True and note["outcome"] == "joined"
    equal_to(serial, job, world)
    crops = job["diagnostics"]["timings"]["notes"]["frame_pool.trail_crops"]
    assert crops[0]["decision"]["cap"] == note["crop_cap"]
    record = json.loads((Path(job["job_dir"]) / "work" / ts.TRANSIENTS_DIR / ts.CANDIDATES_FILE)
                        .read_text(encoding="utf-8"))
    assert len(record["candidates"]) >= 2 and all(c.get("review_png") for c in record["candidates"]
                                                    if c["measurement"].get("status") == "measured")
    assert record["workers"] == note["n_o1"]


# ================================================================== the join guards (§3.6 "在原位置汇合")

def test_each_join_guard_and_a_failing_comparison():
    frozen = {"used": True, "height_px": 128, "calibration": {"a": [1, 2]}, "inputs": [{"path": "x"}],
              "marks": [{"stem": "light_00001", "x0": 1.0}]}
    o1 = core._Overlap(Path("."), Path("."), {"pieces": {}}, frozen, None, {})
    live = {key: json.loads(json.dumps(value)) for key, value in frozen.items()}
    assert o1.mismatch(**live) is None
    for name, value in (("used", False), ("height_px", 127), ("calibration", {"a": [1, 3]}),
                        ("inputs", [{"path": "y"}]), ("marks", [])):
        assert o1.mismatch(**{**live, name: value}) == name

    class Unequal:
        def __eq__(self, other):
            raise TypeError("cannot compare")
        __hash__ = None
    assert o1.mismatch(**{**live, "calibration": Unequal()}) == "compare:TypeError"
    assert o1.mismatch(**{**live, "marks": float("nan")}) == "marks"


def test_a_guard_mismatch_discards_b_and_runs_the_serial_scan(world):
    """B (held by the gate, still scanning) is abandoned, its output removed with the marker, and the serial body runs
    in its place: the job equals the serial one (one scan run folder, the serial body's)."""
    serial = succeeded(reference(world))
    world.children.set(gate=True)
    world.monkeypatch.setattr(core._Overlap, "mismatch", lambda self, **live: "calibration")
    calls = []
    real_detect = core._detect_transients

    def detect(*args, **kwargs):
        calls.append((threading.current_thread() is threading.main_thread(), kwargs.get("workers")))
        if calls[-1][0]:
            world.children.open_gate()              # the serial scan must not wait for the gate
            world.children.set()
        return real_detect(*args, **kwargs)
    world.monkeypatch.setattr(core, "_detect_transients", detect)
    job = succeeded(world.run("guard"))
    note = note_of(job)
    assert note["eligible"] is True and note["outcome"] == "guard_mismatch:calibration"
    assert calls == [(False, note["n_o1"]), (True, None)]
    equal_to(serial, job, world)
    assert len(scan_runs(job["job_dir"])) == 1
    quiet(world)


def test_a_guard_mismatch_whose_output_cannot_be_removed_fails_the_attempt(world):
    world.children.set()
    world.monkeypatch.setattr(core._Overlap, "mismatch", lambda self, **live: "marks")
    world.monkeypatch.setattr(core, "_o1_clear", lambda work, directory=None: False)
    job = world.run("stuck")
    assert job["state"] == "failed" and "无法删除上次中断的亮迹扫描输出" in job["error"]
    assert "transients" not in job["checkpoints"]
    quiet(world)


# ================================================================== the marker protocol's own failures (§3.6)

def test_a_failed_start_whose_output_cannot_be_removed_yet_falls_back_to_the_serial_code(world):
    """B cannot start (the helper thread fails to start) after the marker was written, and the folder cannot be
    removed at that moment: O-1's machinery never fails the attempt there -- the marker stays, the serial code runs,
    and the marker rule at the trail position removes the output before the serial scan (here, A's native remap
    later, the folder can be removed)."""
    serial = succeeded(reference(world))
    world.monkeypatch.setattr(frame_pool.Beside, "start", classmethod(_raise(RuntimeError("no thread for B"))))
    stuck, calls = [True], []
    real_clear = core._o1_clear

    def clear(work, directory=None):
        calls.append(stuck[0])
        return False if stuck[0] else real_clear(work, directory)
    world.monkeypatch.setattr(core, "_o1_clear", clear)
    wrap(world.monkeypatch, rc, "native_from_undistorted", before=lambda *a, **k: stuck.__setitem__(0, False))
    job = succeeded(world.run("start"))
    note = note_of(job)
    assert note["eligible"] is False and note["reason"] == "overlap_error:RuntimeError"
    assert calls == [True, False]                  # close() at the start, then the marker rule at the trail position
    assert not (Path(job["job_dir"]) / "work" / core.O1_MARKER).exists()
    assert len(scan_runs(job["job_dir"])) == 1
    equal_to(serial, job, world)
    quiet(world)


def test_a_stale_marker_beside_a_transients_checkpoint_that_cannot_go_is_kept_not_fatal(world):
    """The marker could not be removed after the join (kept, logged) and the attempt failed later; the resumed attempt
    finds it beside the transients checkpoint and still cannot remove it: logged, the marker stays, the job goes on
    (as _Overlap.keep does) instead of failing on the stale marker alone."""
    import pathlib
    real_unlink = pathlib.Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == core.O1_MARKER:
            raise PermissionError(13, "the marker is held (test)", str(self))
        return real_unlink(self, *args, **kwargs)
    world.monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    failing = [True]

    def compose(*args, **kwargs):
        if failing[0]:
            failing[0] = False
            raise InjectedA("the base composition failed (test)")
    wrap(world.monkeypatch, core, "_compose_sky", before=compose)
    directory = world.submit("stale")
    core.run_job(directory)
    first = core.load_job(directory)
    assert first["state"] == "failed" and "transients" in first["checkpoints"]
    assert note_of(first)["outcome"] == "joined"
    work = Path(directory) / "work"
    assert (work / core.O1_MARKER).exists()
    job = succeeded(world.resume(directory))
    assert (work / core.O1_MARKER).exists()          # stays: a stale marker beside a checkpoint (visible to checks)
    log = (Path(directory) / "worker.log").read_text(encoding="utf-8")
    assert f"could not remove the marker {core.O1_MARKER}" in log
    assert f"could not remove the stale marker {core.O1_MARKER}" in log
    assert len(scan_runs(directory)) == 1 and job["diagnostics"]["transients"]["status"] != "failed"
    quiet(world)


@pytest.mark.skipif(os.name != "nt", reason="Windows error codes")
def test_the_marker_protocol_retries_a_delete_pending_entry(tmp_path, monkeypatch):
    """rmtree of work/transients meets ERROR_DIR_NOT_EMPTY (145) while a just-ended process's handle closes: retried
    within O1_REMOVE_SECONDS like the sharing denials; any other error is not, and the marker stays (logged)."""
    work = tmp_path / "work"
    (work / ts.TRANSIENTS_DIR / "scan-runs" / "run").mkdir(parents=True)
    (work / ts.TRANSIENTS_DIR / "scan-runs" / "run" / "worker-000.log").write_text("x", encoding="utf-8")
    (work / core.O1_MARKER).write_bytes(b"")
    real_rmtree, calls = shutil.rmtree, []

    def rmtree(path, *args, **kwargs):
        calls.append(str(path))
        if len(calls) < 3:
            raise OSError(41, "The directory is not empty", str(path), 145)
        return real_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(core.shutil, "rmtree", rmtree)
    assert core._o1_clear(work, tmp_path) is True
    assert len(calls) == 3 and not (work / ts.TRANSIENTS_DIR).exists() and not (work / core.O1_MARKER).exists()
    (work / ts.TRANSIENTS_DIR).mkdir()
    (work / core.O1_MARKER).write_bytes(b"")
    calls.clear()

    def other(path, *args, **kwargs):
        calls.append(str(path))
        raise OSError(13, "The requested operation cannot be performed on a mapped file", str(path), 1224)
    monkeypatch.setattr(core.shutil, "rmtree", other)
    assert core._o1_clear(work, tmp_path) is False
    assert len(calls) == 1 and (work / core.O1_MARKER).exists()
    assert "the marker transients.o1-pending stays" in (tmp_path / "worker.log").read_text(encoding="utf-8")


def test_a_join_interrupted_by_keyboardinterrupt_leaves_no_helper_running_before_the_removal(tmp_path, monkeypatch):
    """A KeyboardInterrupt inside Beside.result's join neither abandons B nor collects its outcome, yet _Overlap.result
    marks the outcome taken: close() -- the worker's finally -- still abandons B and waits it out before it removes
    work/transients, and releases every piece."""
    work = tmp_path / "work"
    (work / ts.TRANSIENTS_DIR).mkdir(parents=True)
    (work / core.O1_MARKER).write_bytes(b"")

    class Timer:
        started = time.perf_counter()

        def attach_note(self, *args):
            pass
    o1 = core._Overlap(tmp_path, work, {"pieces": {}}, {}, Timer(), {})
    running = threading.Event()

    def scan(probe):
        while True:
            probe()
            (work / ts.TRANSIENTS_DIR / "b.txt").write_text("B writes here", encoding="utf-8")
            running.set()
            time.sleep(0.01)
    before = frame_pool.reserved_bytes()
    o1.reserve("scan", 1 << 20)
    o1.helper = frame_pool.Beside.start(scan, "o1-ki")
    try:
        assert running.wait(10)
        monkeypatch.setattr(o1.helper, "result", _raise(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            o1.result(lambda: None)
        assert o1.taken and not o1.helper.done()
        assert frame_pool.reserved_bytes() == before + (1 << 20)   # B still runs: its processes' bytes stay promised
        at_clear = []
        real_clear = core._o1_clear

        def clear(target, directory=None):
            at_clear.append(o1.helper.done())
            return real_clear(target, directory)
        monkeypatch.setattr(core, "_o1_clear", clear)
        assert o1.close("the worker is stopping") is True
        assert at_clear == [True] and o1.helper.abandoned and o1.helper.done()
        assert frame_pool.reserved_bytes() == before
        assert not (work / ts.TRANSIENTS_DIR).exists() and not (work / core.O1_MARKER).exists()
        assert o1.close("again") is True and at_clear == [True]           # repeatable: nothing left to do
    finally:
        o1.helper.abandon()
        o1.release_all()


# ================================================================== C3: failures across the stages and their priority

class InjectedA(RuntimeError):
    pass


def inject_a_failure(world, m, where, wait=None):
    """A (main thread) fails at ``where`` after ``wait()`` (B's progress), with the same message serially."""
    def fail(*args, **kwargs):
        if wait is not None:
            wait()
        raise InjectedA(f"injected A failure: {where}")
    if where == "native_remap":
        wrap(m, rc, "native_from_undistorted", before=fail)
    elif where == "native_master_check":                # the second _validate_master: the native master's
        calls, real = [], core._validate_master

        def validate(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                fail()
            return real(*args, **kwargs)
        m.setattr(core, "_validate_master", validate)
    elif where == "coverage":
        wrap(m, core, "_write_coverage", before=fail)
    elif where == "finish_stack":
        wrap(m, rs, "finish_stack", before=fail)
    elif where == "diagnostics":
        wrap(m, core, "exposure_diagnostics", before=fail)
    else:
        raise AssertionError(where)


C3_CASES = [
    # (A's failure, mode, B's state when A fails); "cropping": B's crop pool (a frame pool started on the helper) is
    # running, so B is abandoned midway through it
    ("native_remap", "L", "scanning"), ("native_remap", "L", "scan_failed"), ("native_remap", "L", "ledger_mismatch"),
    ("native_master_check", "L", "scanning"), ("coverage", "L", "scanning"), ("coverage", "L", "scan_failed"),
    ("diagnostics", "L", "scanning"), ("diagnostics", "L", "cropping"), ("finish_stack", "strict", "scanning"),
    ("native_remap", "strict", "scanning"),
]


@pytest.mark.parametrize("where, mode, b_state", C3_CASES, ids=["-".join(c) for c in C3_CASES])
def test_c3_a_failure_wins_ends_b_and_resumes_like_an_uninterrupted_job(world, where, mode, b_state):
    """§6.2 (j) cross-stage failures: A (the native remap, the native master check, the coverage count, the strict
    finish, the diagnostics) fails while the scan runs, after it failed, or after it hit a ledger mismatch: the job's
    error is the serial one (A's error wins), no scan process is left, no transients diagnostics, no trace of B; the
    failed folders are equal under the verifier's canonicalisation; resumed, the job equals an uninterrupted one."""
    strict = mode == "strict"
    clean = succeeded(reference(world, "clean", strict=strict))
    with pytest.MonkeyPatch.context() as m:
        inject_a_failure(world, m, where)
        serial = reference(world, "serial-failed", strict=strict)
    assert serial["state"] == "failed" and f"injected A failure: {where}" in serial["error"]
    with pytest.MonkeyPatch.context() as m:
        if b_state == "scanning":
            world.children.set(gate=True)
            wait = lambda: world.children.wait_started(1)                      # noqa: E731
        elif b_state == "scan_failed":
            world.children.set(fail=["light_00001"])
            wait = lambda: wait_until(lambda: b_done(world), 30)              # noqa: E731
        elif b_state == "cropping":
            world.children.set()
            cropping, real_crop = threading.Event(), ts.tt.write_crop_png

            def slow_crop(*args, **kwargs):
                cropping.set()
                time.sleep(0.5)
                return real_crop(*args, **kwargs)
            m.setattr(ts.tt, "write_crop_png", slow_crop)
            wait = lambda: cropping.wait(30) and time.sleep(0.05)              # noqa: E731
        else:
            world.children.set()

            def mismatch(*args, **kwargs):
                raise hashing.HashLedgerMismatch("r_light_00001.fit", "test", "ledger", "0" * 64, "1" * 64)
            m.setattr(ts, "detect_job", mismatch)
            wait = lambda: wait_until(lambda: b_done(world), 30)              # noqa: E731
        inject_a_failure(world, m, where, wait)
        job = world.run("o1-failed", strict=strict)
    note = note_of(job)
    assert job["state"] == "failed" and job["error"] == serial["error"], job.get("error")
    assert note["eligible"] is True and note["outcome"] == f"discarded:the main thread failed: InjectedA", note
    assert note.get("failure") is None                  # B's own outcome was never taken: A's error decides
    assert "transients" not in (job.get("diagnostics") or {}) and job["checkpoints"] == serial["checkpoints"]
    quiet(world)
    assert no_trace_of_b(job["job_dir"])
    equal_to(serial, job, world)
    assert "O-1: the trail scan was discarded" in (Path(job["job_dir"]) / "worker.log").read_text(encoding="utf-8")
    world.children.set()
    equal_to(clean, succeeded(world.resume(job["job_dir"])), world, resumed=True)
    quiet(world)


def test_stackcnt_below_three_raises_the_same_error_and_never_starts_b(world):
    world.engine.stackcnt = 2
    serial = reference(world, "serial")
    job = world.run("o1")
    assert serial["state"] == job["state"] == "failed"
    assert job["error"] == serial["error"] == "实际参与堆栈不足 3 帧，请检查配准日志"
    assert note_of(job)["reason"] == "stackcnt" and world.started == [] and no_trace_of_b(job["job_dir"])
    equal_to(serial, job, world)


def test_a_failed_scan_with_a_succeeding_a_records_the_serial_transients_error(world):
    """B fails (frame 3, in frame order) and A succeeds: the job goes on with diagnostics.transients failed and the
    same text as the serial run (T-2's frame-order message, independent of the process count); B's files stay as the
    serial scan's do, the marker goes."""
    world.children.set(fail=["light_00003"])
    serial = succeeded(reference(world))
    world.children.set(fail=["light_00003"])
    world.env(STARSTACK_TRANSIENT_WORKERS=2)               # another process count than the serial run's
    job = succeeded(world.run("o1"))
    error = job["diagnostics"]["transients"]
    assert error["status"] == "failed" and error == serial["diagnostics"]["transients"]
    assert "第 3 帧（light_00003）" in error["error"]
    note = note_of(job)
    assert note["outcome"] == "recorded" and note["failure"] == "ScanWorkerError"
    assert "transients" not in job["checkpoints"]
    assert not (Path(job["job_dir"]) / "work" / core.O1_MARKER).exists()
    assert len(scan_runs(job["job_dir"])) == 1
    equal_to(serial, job, world)
    quiet(world)


# ================================================================== C5: cancellation (§5 item 5, §6.4 C7, U-C3)

CANCEL_LIMIT_S = 3.0          # prereg §6.4 criterion 1 on this host: max(3 s, K + 1 s); these frames are tiny


def cancel_at(world, m, point, directory_box):
    """Arrange the cancel at ``point``; returns a callable giving the moment it was requested."""
    if point == "during_scan":
        world.children.set(gate=True)
        box = {}

        def start(*args, **kwargs):
            box["cancel"] = Cancel(directory_box[0], lambda: len(world.children.started()) >= 1)
        wrap(m, core, "_o1_started", before=start)
        return lambda: box["cancel"].at
    if point == "join_wait":
        world.children.set(gate=True)
        box = {}

        def start(*args, **kwargs):
            box["cancel"] = Cancel(directory_box[0], stage_is(directory_box[0], "transients"))
        wrap(m, core, "_o1_started", before=start)
        return lambda: box["cancel"].at
    if point == "after_stack":
        world.children.set(gate=True)
        box = {}

        def stack_checkpoint(result, directory, files):
            if any(Path(f).name == "sky_master.fit" for f in files):
                box["at"] = time.monotonic()
                (Path(directory) / "cancel.request").touch()
        wrap(m, core, "_checkpoint_records", after=stack_checkpoint)
        return lambda: box.get("at")
    if point == "pre_join_check":
        world.children.set()
        calls, box = [], {}

        def marks(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2 and world.started:      # the transient position (the first call is O-1's decision)
                wait_until(lambda: b_done(world), 30)
                box["at"] = time.monotonic()
                (Path(directory_box[0]) / "cancel.request").touch()
        wrap(m, core, "_normalise_marks", before=marks)
        return lambda: box.get("at")
    raise AssertionError(point)


@pytest.mark.parametrize("point", ["during_scan", "join_wait", "after_stack", "pre_join_check"])
@pytest.mark.parametrize("mode", ["L", "strict"])
def test_c5_a_cancel_in_the_overlap_window_stops_promptly_and_resumes_like_an_uninterrupted_job(world, point, mode):
    strict = mode == "strict"
    clean = succeeded(reference(world, "clean", strict=strict))
    directory = world.submit("o1-cancel", strict=strict)
    with pytest.MonkeyPatch.context() as m:
        requested = cancel_at(world, m, point, [directory])
        core.run_job(directory)
        returned = time.monotonic()
    job = core.load_job(directory)
    assert job["state"] == "cancelled", (job["state"], job.get("error"))
    at = requested()
    assert at is not None and returned - at <= CANCEL_LIMIT_S, returned - (at or returned)
    note = note_of(job)
    assert note["eligible"] is True and note["outcome"] == "discarded:the main thread was cancelled", note
    assert note.get("failure") in (None, "Cancelled")    # Cancelled: taken at the join (B's probe or the wait saw it)
    quiet(world)
    assert no_trace_of_b(directory)
    expected_checkpoints = ["foreground", "convert", "distortion"] + (["stack"] if point != "during_scan" else [])
    if point == "during_scan":
        assert job["checkpoints"] in (expected_checkpoints, expected_checkpoints + ["stack"])
    else:
        assert job["checkpoints"] == expected_checkpoints
    world.children.set()
    resumed = succeeded(world.resume(directory))
    equal_to(clean, resumed, world, resumed=True)
    assert len(scan_runs(directory)) == 1
    quiet(world)


# ================================================================== the single writer (§3.6 "单一写者")

def test_only_the_main_thread_writes_the_record_even_when_b_ends_during_checkpoint_stack(world):
    serial_updates = []
    real_update = core._update

    def record(store):
        def update(job_dir, **changes):
            store.append((threading.current_thread() is threading.main_thread(), sorted(changes),
                          changes.get("progress"), changes.get("stage")))
            return real_update(job_dir, **changes)
        return update
    world.monkeypatch.setattr(core, "_update", record(serial_updates))
    serial = succeeded(reference(world))
    updates, checkpoints = [], []
    world.monkeypatch.setattr(core, "_update", record(updates))
    world.children.set(gate=True)
    real_records = core._checkpoint_records

    def checkpoint_records(directory, files):
        checkpoints.append(threading.current_thread() is threading.main_thread())
        if any(Path(f).name == "sky_master.fit" for f in files):
            world.children.open_gate()                  # B ends while checkpoint('stack') is being written
            wait_until(lambda: b_done(world), 30)
        return real_records(directory, files)
    world.monkeypatch.setattr(core, "_checkpoint_records", checkpoint_records)
    job = succeeded(world.run("o1"))
    assert note_of(job)["outcome"] == "joined" and note_of(job)["b_done_at_join"] is True
    assert all(checkpoints) and len(checkpoints) == len(job["checkpoints"])
    assert all(main for main, *_ in updates), [u for u in updates if not u[0]]
    sequence = lambda log: [(keys, progress, stage) for _, keys, progress, stage in log]       # noqa: E731
    assert sequence(updates) == sequence(serial_updates)
    equal_to(serial, job, world)


def test_b_writes_the_record_only_through_check_on_a_cancel(world):
    updates = []
    real_update = core._update

    def update(job_dir, **changes):
        updates.append((threading.current_thread().name, sorted(changes)))
        return real_update(job_dir, **changes)
    world.monkeypatch.setattr(core, "_update", update)
    world.children.set(gate=True)
    directory = world.submit("o1")

    def hold_main(*args, **kwargs):                    # the main thread stays in A: only B can see the cancel
        Cancel(directory, lambda: len(world.children.started()) >= 1)
        wait_until(lambda: b_done(world), 30)
    wrap(world.monkeypatch, rc, "native_from_undistorted", before=hold_main)
    core.run_job(directory)
    assert core.load_job(directory)["state"] == "cancelled"
    helpers = [keys for name, keys in updates if name != threading.main_thread().name]
    assert helpers and all(keys == ["message", "state"] for keys in helpers), updates
    assert all(name in (threading.main_thread().name, "aester-beside-o1-scan") for name, _ in updates)
    quiet(world)


# ================================================================== reservations, budget and the note

def held_pieces(world) -> list:
    o1 = overlap_of(world)
    with o1._held_lock:
        return sorted(o1.held)


def test_the_plan_s_reservations_are_released_as_a_reaches_each_part(world):
    """A's piece goes once its native master is written and checked, the coverage piece as its pool starts (which
    reserves its own workers), the scan processes' bytes on B's thread as soon as scan_frames has returned (the
    processes have ended: their memory is no longer promised), not at the join; nothing is reserved afterwards. The
    scan is held by the gate until A's coverage pool has decided."""
    seen = {}
    from starstack import sky_validity as sv
    world.children.set(gate=True)

    def at_remap(*args, **kwargs):
        seen["remap"] = frame_pool.reserved_bytes()

    def at_coverage(*args, workers, **kwargs):
        seen["coverage"] = (frame_pool.reserved_bytes(), workers, held_pieces(world))
        world.children.open_gate()

    def at_crops(*args, **kwargs):
        seen["crops"] = (held_pieces(world), threading.current_thread().name)
    wrap(world.monkeypatch, rc, "native_from_undistorted", before=at_remap)
    wrap(world.monkeypatch, sv, "coverage_map", before=at_coverage)
    wrap(world.monkeypatch, ts, "_crop_phase", before=at_crops)
    real_result = core._Overlap.result

    def result(self, wait_check):
        value = real_result(self, wait_check)
        seen["joined"] = frame_pool.reserved_bytes()
        return value
    world.monkeypatch.setattr(core._Overlap, "result", result)
    job = succeeded(world.run("o1"))
    pieces = note_of(job)["pieces"]
    assert pieces["coverage"] > 0 and pieces["scan"] > 0 and pieces["native_mask"] == 0
    assert seen["remap"] == pieces["a"] + pieces["coverage"] + pieces["scan"]
    held, workers, names = seen["coverage"]
    unit = core._coverage_unit_bytes(H, W)
    assert workers == note_of(job)["cov_plan"] and names == ["scan"]          # A's own part is allocated by now
    assert held == pieces["scan"] + workers * (unit + frame_pool.BLAS_SLOT_BYTES)
    crops_held, crops_thread = seen["crops"]
    assert crops_thread == "aester-beside-o1-scan" and "scan" not in crops_held   # gone with the scan processes
    assert seen["joined"] == 0 and frame_pool.reserved_bytes() == 0


def test_a_failed_scan_releases_the_scan_processes_bytes_before_the_join(world):
    """scan_frames raising (a scan child failed) releases the scan piece on B's thread as well: A, still running,
    no longer counts the ended processes; B's failure is recorded at the join as before."""
    seen = {}
    from starstack import sky_validity as sv
    world.children.set(fail=["light_00002"])
    ended = threading.Event()
    real_scan = ts.scan_frames

    def scan_frames(*args, **kwargs):
        try:
            return real_scan(*args, **kwargs)
        finally:
            ended.set()
    world.monkeypatch.setattr(ts, "scan_frames", scan_frames)

    def at_coverage(*args, **kwargs):
        assert ended.wait(30)
        wait_until(lambda: "scan" not in held_pieces(world))
        seen["coverage"] = held_pieces(world)
    wrap(world.monkeypatch, sv, "coverage_map", before=at_coverage)
    job = succeeded(world.run("o1"))
    assert seen["coverage"] == [] and note_of(job)["outcome"] == "recorded"
    assert job["diagnostics"]["transients"]["status"] == "failed"
    assert frame_pool.reserved_bytes() == 0


def test_a_strict_job_releases_g4t_s_share_before_the_finish(world):
    seen = {}
    world.children.set(gate=True)                  # the scan (and its reservation) runs past the finish

    def at_finish(*args, **kwargs):
        seen["finish"] = frame_pool.reserved_bytes()
        world.children.open_gate()
    wrap(world.monkeypatch, rs, "finish_stack", before=at_finish)
    job = succeeded(world.run("o1", strict=True))
    note = note_of(job)
    pieces = note["pieces"]
    assert note["eligible"] is True and pieces["native_mask"] > 0 and pieces["coverage"] == 0
    assert seen["finish"] == pieces["a"] + pieces["scan"]
    assert note["a_reserve_bytes"] == pieces["a"] + pieces["native_mask"]


def test_the_crop_pool_on_b_and_the_coverage_pool_share_the_16_slots(world):
    """B's crop pool and A's coverage pool run at the same time: together they never hold more than the process's 16
    frame-pool threads, both still fan out, and the crop pool's cap leaves the coverage plan's slots (prereg §3.6)."""
    crops_started = threading.Event()
    real_crop = ts.tt.write_crop_png

    def slow_crop(*args, **kwargs):
        crops_started.set()
        time.sleep(0.2)
        return real_crop(*args, **kwargs)
    world.monkeypatch.setattr(ts.tt, "write_crop_png", slow_crop)
    peak = {"n": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            n = sum(1 for t in threading.enumerate() if t.name.startswith(("aester-trail_crops", "aester-coverage")))
            peak["n"] = max(peak["n"], n)
            time.sleep(0.002)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    from starstack import sky_validity as sv
    real_map = sv.coverage_map
    overlapped = []

    def coverage_map(*args, **kwargs):
        assert crops_started.wait(30)
        overlapped.append(any(t.name.startswith("aester-trail_crops") for t in threading.enumerate()))
        return real_map(*args, **kwargs)
    world.monkeypatch.setattr(sv, "coverage_map", coverage_map)
    try:
        job = succeeded(world.run("o1"))
    finally:
        stop.set()
        sampler.join(5)
    notes = job["diagnostics"]["timings"]["notes"]
    note = note_of(job)
    crops, coverage = notes["frame_pool.trail_crops"][0], notes["frame_pool.coverage"][0]
    assert crops["decision"]["cap"] == note["crop_cap"] == min(ts.TRAIL_CROP_CAP, 16 - note["cov_plan"])
    assert coverage["decision"]["o1_cap"] == note["cov_plan"]
    assert not crops["serial"] and not coverage["serial"]
    assert overlapped == [True] and peak["n"] <= frame_pool.MAX_FRAME_WORKERS
    assert crops["workers"] + coverage["workers"] <= frame_pool.MAX_FRAME_WORKERS


def test_the_note_is_in_the_timings_only_and_the_run_check_reads_it(world):
    spec = importlib.util.spec_from_file_location("aester_check_p1c_o1", ROOT / "scripts" / "check_p1c_run.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    job = succeeded(world.run("o1"))
    off = succeeded(world.run("off", mode="off"))
    for record, eligible in ((job, True), (off, False)):
        note = note_of(record)
        assert checker.overlap_on(note) is eligible
        assert checker.keys_outside_timings(record) == []
        assert len(json.dumps(note)) < 4096
        assert checker.overlap_problems(note) == [], checker.overlap_problems(note)
    def without_timings(value):
        if isinstance(value, dict):
            return {k: without_timings(v) for k, v in value.items() if k not in ("timings", "timing_attempts")}
        if isinstance(value, list):
            return [without_timings(v) for v in value]
        return value
    stored = json.loads((Path(job["job_dir"]) / "job.json").read_text(encoding="utf-8"))
    text = json.dumps(without_timings(stored), ensure_ascii=False)
    for word in (core.O1_NOTE, core.O1_MARKER, "o1-scan", "n_o1", "cov_plan", "headroom_bytes"):
        assert word not in text, word
    timings = job["diagnostics"]["timings"]
    assert set(note_of(job)["window_s"]) == {"b_start_s", "b_end_s", "join_start_s", "join_end_s"}
    assert "transient_detect" in timings["operations_seconds"] and "transient_join" in timings["operations_seconds"]


# ================================================================== reclamation (§3.6 "交出结果后清空所存的结果")

def test_b_s_failed_outcome_is_collected_once_logged(world):
    """B fails holding a large intermediate: after the join logs it (transients_error), nothing keeps it -- checked at
    the base composition, with the garbage collector off (reference counts alone)."""
    refs, seen = [], {}

    class Big:
        def __init__(self):
            self.buffer = bytearray(16 << 20)

    def detect(*args, **kwargs):
        big = Big()
        refs.append(weakref.ref(big))
        raise ValueError("the scan failed holding a large intermediate", big)
    world.monkeypatch.setattr(ts, "detect_job", detect)
    wrap(world.monkeypatch, core, "_compose_sky", before=lambda *a, **k: seen.update(alive=refs[0]() is not None))
    gc.collect()
    gc.disable()
    try:
        job = succeeded(world.run("o1"))
    finally:
        gc.enable()
    assert note_of(job)["outcome"] == "recorded" and job["diagnostics"]["transients"]["status"] == "failed"
    assert seen == {"alive": False}


# ================================================================== the forced kill of the job worker (§6.2 (j))

def _driver(argv):
    """The job worker of the forced-kill test, in a process of its own: the world's stand-ins (engine, calibration,
    scan children, an eligible O-1), its identity for the test, then core.run_job. ``when``: 'remap' blocks the main
    thread in the native remap (before checkpoint('stack')); 'join' lets it reach the join while B is held."""
    job_dir, children_dir, when = argv
    from test_process_guard import IDENTITY
    namespace = {}
    exec(IDENTITY, namespace)
    folder = Path(children_dir)
    staged = folder / "driver.identity-tmp"
    staged.write_text(json.dumps(namespace["identity"](os.getpid())), encoding="utf-8")
    os.replace(staged, folder / "driver.identity")
    core.run_script = Engine()
    rc.calibrate = fitted_calibration
    scan._worker_command = Children.attach(folder).command
    core._o1_estimates = lambda mode, pixels, frames: (frames * 0.01, 1000.0)
    performance._memory_status = lambda: (64 * GiB, 128 * GiB)
    if when == "remap":
        def blocked(*args, **kwargs):
            (folder / "remap.blocked").write_text("blocked", encoding="utf-8")
            time.sleep(120)
        rc.native_from_undistorted = blocked
    core.run_job(job_dir)


@pytest.mark.skipif(os.name != "nt", reason="the Job Object kill-on-close is the Windows product guarantee")
@pytest.mark.parametrize("when", ["remap", "join"])
def test_a_forced_kill_during_o1_leaves_no_scan_process_and_the_recovered_job_equals_an_uninterrupted_one(world, when):
    """The job worker is killed (TerminateProcess) inside the O-1 window -- in the native remap before
    checkpoint('stack'), or at the join after it -- while B's scan children run: none survives (the guards' Job
    Objects close with their owner). Its record stays 'running' without a lock (recovery_blocked); once a supervisor
    has confirmed that nothing of it runs (here: the record set to failed), the resumed job removes B's leftovers (the
    stack stage's cleanup, or the marker rule at the transient position) and equals an uninterrupted job."""
    from test_process_guard import OwnedProcess
    clean = succeeded(reference(world, "clean"))
    world.children.set(gate=True)
    directory = world.submit("killed")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TESTS), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    log = world.tmp / f"driver-{when}.log"
    owner, children = None, []
    with log.open("wb") as stream:
        launcher = subprocess.Popen(
            [sys.executable, "-c", "import sys, test_p1c_site_overlap as t; t._driver(sys.argv[1:])", directory,
             str(world.children.folder), when], cwd=str(TESTS), env=env, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            identity = wait_until(lambda: _read_json(world.children.folder / "driver.identity"), 90)
            owner = OwnedProcess(identity)
            world.children.wait_started(1, 90)
            if when == "remap":
                wait_until(lambda: (world.children.folder / "remap.blocked").exists(), 90)
            else:
                wait_until(stage_is(directory, "transients"), 90)
                time.sleep(0.3)                             # the main thread waits at the join
            children = world.children.started()
            assert children and all(_alive(c) for c in children)
            work = Path(directory) / "work"
            assert (work / core.O1_MARKER).exists() and (work / ts.TRANSIENTS_DIR).exists()
            owner.kill()
            wait_until(lambda: not owner.alive(), 30)
            wait_until(lambda: not any(_alive(c) for c in children), 30)
        finally:
            world.children.open_gate()
            if owner is not None:
                owner.kill()
                owner.close()
            if launcher.poll() is None:
                launcher.kill()
            launcher.wait(timeout=30)
    job = core.load_job(directory)
    assert job["state"] == "failed" and job.get("recovery_blocked") is True       # derived: a crashed worker
    assert ("stack" in job["checkpoints"]) is (when == "join") and "transients" not in job["checkpoints"]
    work = Path(directory) / "work"
    assert (work / core.O1_MARKER).exists() and (work / ts.TRANSIENTS_DIR).exists()     # B's output, to be removed
    stored = json.loads((Path(directory) / "job.json").read_text(encoding="utf-8"))
    stored.update(state="failed", error="the worker was killed (test)")              # the supervisor's recovery
    core.atomic_json(Path(directory) / "job.json", stored)
    world.children.set()
    resumed = succeeded(world.resume(directory))
    equal_to(clean, resumed, world, resumed=True)
    assert len(scan_runs(directory)) == 1 and not (work / core.O1_MARKER).exists()
    quiet(world)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ================================================================== fallback registration (§3.6, §6.2 (j))

@pytest.mark.parametrize("strict, frame_masks", [(False, False), (False, True), (True, False), (True, True)],
                         ids=["L", "L-frame-masks", "strict", "strict-frame-masks"])
def test_fallback_registration_is_ineligible_and_the_scan_sees_the_serial_master(world, strict, frame_masks):
    """Siril's two-pass fallback: the grid master is sky_master.fit, which strict support rewrites in place
    (sv.apply_validity_to_fits), so O-1 is not eligible. With the in-place rewrite, the strict finish and the
    diagnostics jittered (5 schedules), the scan always starts after them, on the main thread, with the serial run's
    grid master path and SHA-256, and the jobs are equal."""
    import hashlib
    from starstack import sky_validity as sv
    seen = []
    real_detect = ts.detect_job

    def detect(work, *, own_registration, **kwargs):
        master = ts.grid_master_path(work, own_registration)
        seen.append((master.name, hashlib.sha256(master.read_bytes()).hexdigest(),
                     threading.current_thread() is threading.main_thread()))
        return real_detect(work, own_registration=own_registration, **kwargs)
    world.monkeypatch.setattr(ts, "detect_job", detect)
    options = dict(own=False, strict=strict, frame_masks=frame_masks)
    serial = succeeded(reference(world, **options))
    assert seen and seen[0][0] == "sky_master.fit" and seen[0][2] is True
    for seed in range(5):
        r = random.Random(seed)
        jitter = lambda *a, **k: time.sleep(r.uniform(0, 0.05))                   # noqa: E731
        with pytest.MonkeyPatch.context() as m:
            wrap(m, rs, "finish_stack", before=jitter, after=jitter)
            wrap(m, sv, "apply_validity_to_fits", before=jitter, after=jitter)
            wrap(m, core, "exposure_diagnostics", before=jitter)
            job = succeeded(world.run(f"fallback-{seed}", **options))
        assert note_of(job)["eligible"] is False and note_of(job)["reason"] == "fallback_registration"
        assert seen[-1] == seen[0]
        equal_to(serial, job, world)
    assert world.started == [] and len(seen) == 6


# ================================================================== C9: C1, C3 and C5 under 50 random schedules each

JITTER_RUNS = int(os.environ.get("AESTER_O1_JITTER_RUNS", "50"))       # prereg §6.2 (j): 50


def jitter_everything(m, r):
    """Random delays in the frame pools' units, in A's steps and in the scan children (per frame)."""
    m.setattr(frame_pool, "_before_unit", lambda index: time.sleep(r.uniform(0, 0.004)))
    for target, name in ((rc, "native_from_undistorted"), (core, "_write_coverage"),
                         (core, "exposure_diagnostics"), (core, "_checkpoint_records")):
        wrap(m, target, name, before=lambda *a, **k: time.sleep(r.uniform(0, 0.06)))
    m.setenv(ts.WORKERS_ENV, str(r.choice([1, 2, 3, 6])))


@pytest.mark.parametrize("case", ["C1", "C3", "C5"])
def test_c9_fifty_random_schedules_leak_nothing_and_stay_equal(world, case):
    """§6.2 (j): C1 (success), C3 (A fails at a random step) and C5 (a cancel at a random moment of the window) with
    random delays everywhere, JITTER_RUNS times each: canonical equality with the serial outcome (C5: after the
    resume), no helper or pool thread, no scan process, no metadata lock timeout (every run ends in its expected
    state)."""
    clean = succeeded(reference(world, "clean"))
    failed = {}
    if case == "C3":
        for where in ("native_remap", "coverage", "diagnostics"):
            with pytest.MonkeyPatch.context() as m:
                inject_a_failure(world, m, where)
                failed[where] = reference(world, f"serial-{where}")
            assert failed[where]["state"] == "failed"
    master = random.Random(f"o1-{case}")
    outcomes = {}
    for run in range(JITTER_RUNS):
        seed = master.randrange(1 << 30)
        r = random.Random(seed)
        world.children.set(delays={f"light_{i:05d}": round(r.uniform(0, 0.08), 3) for i in range(1, FRAMES + 1)})
        label = f"{case}-{run}"
        with pytest.MonkeyPatch.context() as m:
            jitter_everything(m, r)
            if case == "C1":
                job = world.run(label)
                assert job["state"] == "succeeded", (seed, job.get("error"))
                equal_to(clean, job, world)
            elif case == "C3":
                where = r.choice(sorted(failed))
                inject_a_failure(world, m, where)
                job = world.run(label)
                assert job["state"] == "failed" and job["error"] == failed[where]["error"], (seed, job.get("error"))
                assert no_trace_of_b(job["job_dir"]), seed
                equal_to(failed[where], job, world)
            else:
                directory = world.submit(label)
                pause = r.uniform(0, 0.4)
                canceller = Cancel(directory, lambda: bool(world.started) and time.sleep(pause) is None)
                core.run_job(directory)
                canceller.thread.join(35)
                job = core.load_job(directory)
                (Path(directory) / "cancel.request").unlink(missing_ok=True)     # a cancel after the end: dropped
                outcomes["first:" + job["state"]] = outcomes.get("first:" + job["state"], 0) + 1
                if job["state"] == "cancelled":
                    after_window = "transients" in job["checkpoints"]
                    if after_window:                          # the cancel came after the join: B's output is the job's
                        assert note_of(job)["outcome"] == "joined", seed
                        assert not (Path(directory) / "work" / core.O1_MARKER).exists(), seed
                    else:
                        assert no_trace_of_b(directory), seed
                    quiet(world)
                    with pytest.MonkeyPatch.context() as calm:
                        calm.setattr(frame_pool, "_before_unit", lambda index: None)
                        job = world.resume(directory)
                    if after_window:
                        # a cancel in the composition keeps the cancelled attempt's products/<uuid> (serial too: the
                        # R_resume of prereg §6.4); O-1 is over by then, so only the outcome is checked here
                        outcomes["after the window"] = outcomes.get("after the window", 0) + 1
                    else:
                        equal_to(clean, job, world, resumed=True)
                else:
                    equal_to(clean, job, world)
                assert job["state"] == "succeeded", (seed, job.get("error"))
            outcomes[job["state"]] = outcomes.get(job["state"], 0) + 1
        quiet(world)
        assert frame_pool._budget_used == 0 and frame_pool.reserved_bytes() == 0, seed
        world.started.clear()
    print(f"C9 {case}: {JITTER_RUNS} runs, final states {outcomes}")
    if case == "C5":                                       # most cancels land inside the O-1 window
        assert outcomes.get("first:cancelled", 0) - outcomes.get("after the window", 0) >= max(1, JITTER_RUNS // 5)


