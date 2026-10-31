"""Efficiency P1c site tests for the trail scan processes (prereg §3.4 T-2, §6.2 A5).

- The process count: overrides in the T-2 order (explicit, STARSTACK_TRANSIENT_WORKERS, AESTER_FRAME_WORKERS 1-16 or
  max), and the automatic count max(floor, n_mem) with n_mem <= the ceiling, on what-if machines (frame_pool.usable_cpus,
  resources.cpu_topology and the memory readings patched: the results do not depend on the test machine); the governor's
  shadow reads the same overrides (resources._override_value, 'max' included).
- The worst-case child (the peak test that fixes SCAN_UNIT_BYTES): a trail across the whole 45.7 MP frame, found and
  marked; and a tiny frame for the fixed part.
- Failures in frame order: real scanner children running a fake ``_scan_frame`` (raise, crash, a failed error record,
  a late non-zero exit) at frame k in {0, 1, middle, last}, at 1-8 processes: the same exception and message, the same
  result / error files left; later frames' children stopped at once; earlier failures win however late they come.
- An in-process fake guard and clock (as tests/test_scan_result_read_retry.py) for the owner's read rules: a corrupt or
  foreign result, an invalid or unreadable error record, the cleanup after ``stop``.
- Cancellation within 1 s with live children, every child ended.
- Real data: F48 frames scanned at 1, the default and 8 processes give the job's own scan results.
- The scan children's import closure is unchanged (frame_pool, performance, resources stay out).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import p1c_site_harness as h
import p1c_trail_fixtures as fx
from starstack import frame_pool, performance, process_guard, profiling, resources
from starstack import transient_scan as scan
from starstack import transient_stage as ts

GiB = 1 << 30
SRC = Path(__file__).resolve().parents[1] / "src"
REAL_SCAN_CHILD = scan._worker_command


# ================================================================== the process count

def machine(monkeypatch, *, lps, cores, physical, commit=None):
    """A what-if machine: usable logical processors, physical cores in the affinity mask, memory readings."""
    monkeypatch.setattr(frame_pool, "usable_cpus", lambda: lps)
    monkeypatch.setattr(resources, "cpu_topology", lambda **_: SimpleNamespace(python_usable_cores=cores))
    monkeypatch.setattr(performance, "_memory_status", lambda: (physical, commit))
    for name in (ts.WORKERS_ENV, ts.FRAME_WORKERS_ENV):
        monkeypatch.delenv(name, raising=False)


# (name, lps, cores, available physical, available commit, frames, expected processes, decided by)
MACHINES = [
    ("development PC", 192, 96, 200 * GiB, 300 * GiB, 48, 8, "cap"),
    ("ordinary-PC simulation (16 LPs, memory unlimited)", 16, 8, 1 << 40, 1 << 40, 48, 7, "cap"),
    ("6900HX proxy v2 (9 GiB view, 14 GiB commit)", 16, 8, 9 * GiB, 14 * GiB, 48, 4, "floor"),
    ("16 GB laptop, commit tighter", 16, 8, 9 * GiB, 5 * GiB, 48, 4, "floor"),
    ("32 GiB free", 16, 8, 32 * GiB, 64 * GiB, 48, 6, "memory"),
    ("memory unknown", 16, 8, None, None, 48, 4, "floor"),
    ("four LPs, two cores", 4, 2, 64 * GiB, 64 * GiB, 48, 3, "floor"),
    ("one LP", 1, 1, 64 * GiB, 64 * GiB, 48, 1, "cap"),
    ("cores unknown: the LPs stand in", 6, None, 64 * GiB, 64 * GiB, 48, 4, "floor"),
    ("few frames", 192, 96, 200 * GiB, 300 * GiB, 5, 5, "units"),
]


@pytest.mark.parametrize("name, lps, cores, physical, commit, frames, want, bound", MACHINES,
                         ids=[m[0] for m in MACHINES])
def test_the_automatic_count_on_what_if_machines(monkeypatch, name, lps, cores, physical, commit, frames, want,
                                                 bound):
    machine(monkeypatch, lps=lps, cores=cores, physical=physical, commit=commit)
    decision = {}
    assert ts.default_workers(None, frames=frames, decision=decision) == want, decision
    assert decision["bound"] == bound, decision
    floor, ceiling = resources.scan_process_bounds(ts.SCAN_WORKER_CAP)
    assert floor == max(1, min(4, lps - 1)) and decision["floor"] == floor and decision["ceiling"] == ceiling
    assert floor <= want <= max(floor, ceiling)
    assert decision["unit_bytes"] == ts.scan_unit_bytes(None) == ts.SCAN_UNIT_BYTES


def test_memory_reservations_pixels_reserved_cores_and_the_o1_form(monkeypatch):
    machine(monkeypatch, lps=16, cores=12, physical=40 * GiB, commit=80 * GiB)
    assert ts.plan_scan_workers() == 7                          # 40 GiB: budget 13.3 GiB / 1.8 GB -> 7 (< ceiling 8)
    with frame_pool.reserved(10 * GiB):                         # deducted by performance.available_memory (R20)
        assert ts.plan_scan_workers() == 5
    assert ts.plan_scan_workers(102_000_000) == 4               # a 102 MP frame: 3.95 GB a child -> 3, floor 4
    assert ts.plan_scan_workers(102_000_000, floor=False) == 3  # O-1's planner: no floor
    assert ts.plan_scan_workers(reserved_cores=6) == 5          # ceiling 12 - 1 - 6
    assert ts.plan_scan_workers(available=3 * GiB, floor=False) == 1
    assert ts.plan_scan_workers(available=None) == 4            # unknown memory: one, then the floor
    assert ts.plan_scan_workers(available=None, floor=False) == 1


def test_unit_bytes_scale_with_pixels():
    assert ts.scan_unit_bytes(ts.REFERENCE_PIXELS) == ts.SCAN_UNIT_BYTES
    assert ts.scan_unit_bytes(0) == ts.SCAN_UNIT_BYTES          # 0 / None: the reference frame
    small, large = ts.scan_unit_bytes(64 * 96), ts.scan_unit_bytes(2 * ts.REFERENCE_PIXELS)
    assert ts.SCAN_BASE_BYTES < small < ts.SCAN_BASE_BYTES + 1_000_000
    assert large == 2 * ts.SCAN_UNIT_BYTES - ts.SCAN_BASE_BYTES


def test_frame_pixels_of_the_master(tmp_path):
    path = fx.write_fits(tmp_path / "m.fit", 30, 40, lambda c, y0, y1: [[0.5] * 40] * (y1 - y0))
    assert ts.frame_pixels(path) == 1200
    assert ts.frame_pixels(tmp_path / "missing.fit") is None                       # OSError
    (tmp_path / "short.fit").write_bytes(b"SIMPLE  =                    T")
    assert ts.frame_pixels(tmp_path / "short.fit") is None                         # ValueError
    (tmp_path / "keyless.fit").write_bytes(fx.header_bytes(30, 40, bitpix=-32, roworder="TOP-DOWN")
                                           .replace(b"NAXIS1  =", b"NAXISX  ="))
    assert ts.frame_pixels(tmp_path / "keyless.fit") is None                       # KeyError


# (environment, expected processes or "max" / "auto", the variable that decided)
OVERRIDES = [
    ({"STARSTACK_TRANSIENT_WORKERS": "6"}, 6, "STARSTACK_TRANSIENT_WORKERS"),
    ({"STARSTACK_TRANSIENT_WORKERS": "12", "AESTER_FRAME_WORKERS": "2"}, 12, "STARSTACK_TRANSIENT_WORKERS"),
    ({"STARSTACK_TRANSIENT_WORKERS": " auto ", "AESTER_FRAME_WORKERS": "2"}, 2, "AESTER_FRAME_WORKERS"),
    ({"AESTER_FRAME_WORKERS": "16"}, 16, "AESTER_FRAME_WORKERS"),
    ({"AESTER_FRAME_WORKERS": "1"}, 1, "AESTER_FRAME_WORKERS"),
    ({"AESTER_FRAME_WORKERS": "max"}, "max", "AESTER_FRAME_WORKERS"),
    ({"AESTER_FRAME_WORKERS": "auto"}, "auto", None),
    ({"STARSTACK_TRANSIENT_WORKERS": "abc", "AESTER_FRAME_WORKERS": "2"}, "auto", None),   # first set decides
    ({"STARSTACK_TRANSIENT_WORKERS": "0", "AESTER_FRAME_WORKERS": "2"}, "auto", None),
    ({}, "auto", None),
]


@pytest.mark.parametrize("env, want, variable", OVERRIDES, ids=[str(o[0]) for o in OVERRIDES])
@pytest.mark.parametrize("box", [(16, 8), (6, 3), (192, 96)], ids=["16x8", "6x3", "192x96"])
def test_overrides_follow_the_t2_order_and_agree_with_the_governor(monkeypatch, env, want, variable, box):
    lps, cores = box
    machine(monkeypatch, lps=lps, cores=cores, physical=9 * GiB, commit=14 * GiB)     # memory never limits these
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    floor, ceiling = resources.scan_process_bounds(ts.SCAN_WORKER_CAP)
    expected = max(floor, ceiling) if want == "max" else (ts.plan_scan_workers() if want == "auto" else want)
    decision = {}
    assert ts.default_workers(frames=48, decision=decision) == expected
    assert ts.scan_override() == ((None, None) if want == "auto" else (expected, variable))
    assert decision["bound"] == ("override" if want != "auto" else decision["bound"])
    governor, error = resources._override(resources.FOOTPRINTS[resources.SCAN_STAGE])
    if want == "auto":
        assert governor is None                 # the shadow computes its own recommendation (an error is noted)
    else:
        assert governor == expected and error is None


def test_a_malformed_frame_knob_fails_and_the_caps_agree(monkeypatch):
    monkeypatch.delenv(ts.WORKERS_ENV, raising=False)
    monkeypatch.setenv("AESTER_FRAME_WORKERS", "17")
    with pytest.raises(ValueError, match="AESTER_FRAME_WORKERS"):
        ts.default_workers()
    assert resources.FOOTPRINTS[resources.SCAN_STAGE].cap == ts.SCAN_WORKER_CAP
    assert resources.FOOTPRINTS[resources.SCAN_STAGE].override_env == (ts.WORKERS_ENV, ts.FRAME_WORKERS_ENV)
    assert ts.FRAME_WORKERS_ENV == frame_pool.ENV_WORKERS


def test_the_scan_children_import_closure_is_unchanged():
    """frame_pool, performance and resources are imported lazily by transient_stage (prereg §3.4)."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import starstack.transient_scan_worker; "
            "from starstack.transient_stage import _scan_frame; "
            "print(sorted(m for m in sys.modules if m.startswith('starstack.')))")
    out = subprocess.run([sys.executable, "-B", "-c", code, str(SRC)], capture_output=True, text=True, check=True)
    modules = set(eval(out.stdout))                                    # noqa: S307 -- our own printed list
    assert not modules & {"starstack.frame_pool", "starstack.performance", "starstack.resources"}, modules
    assert {"starstack.transient_stage", "starstack.transient_trails", "starstack.fits_crop"} <= modules


def test_detect_job_notes_the_plan_and_records_the_count(monkeypatch, tmp_path):
    """detect_job: the decision goes to the timings note transient_scan.plan, the count to the record's 'workers'
    (as before), clamped to the frames."""
    machine(monkeypatch, lps=16, cores=8, physical=9 * GiB, commit=14 * GiB)
    work = tmp_path / "work"
    work.mkdir()
    for i in (1, 2):
        fx.write_fits(work / f"r_light_{i:05d}.fit", 40, 60, lambda c, y0, y1: [[0.02 + 0.001 * c] * 60] * (y1 - y0))
    fx.write_fits(work / "sky_master.fit", 40, 60, lambda c, y0, y1: [[0.02 + 0.001 * c] * 60] * (y1 - y0))
    timer = profiling.StageTimer()
    record = ts.detect_job(work, own_registration=False, transforms={}, model=None, inputs=[], reference_stem=None,
                           log=lambda *_: None)
    report = timer.finish()
    assert record["workers"] == 2                                      # floor 4, clamped to two frames
    plan, = report["notes"]["transient_scan.plan"]
    assert plan["bound"] == "floor" and plan["workers"] == 4 and plan["processes"] == 2 and plan["frames"] == 2
    assert plan["pixels"] == 2400 and plan["unit_bytes"] == ts.scan_unit_bytes(2400)


# ================================================================== the worst-case child (fixes SCAN_UNIT_BYTES)

@contextmanager
def child_accounting():
    seen = []
    process_guard.add_accounting_listener(seen.append)
    try:
        yield seen
    finally:
        process_guard._ACCOUNTING_LISTENERS.remove(seen.append)


@pytest.mark.parametrize("shape", [(64, 96), fx.Z8_SHAPE], ids=["tiny", "z8-45.7MP"])
def test_worst_case_child_peak_is_within_the_charged_unit(tmp_path, shape):
    """The peak test of prereg §3.4: one scan child on a trail across the whole frame's diagonal, found and marked by
    hand (both measurements read the whole frame). Its peak commit (the job's largest process: the venv launcher is
    the direct child) stays within scan_unit_bytes for the frame's pixels."""
    if process_guard.guard_kind() != process_guard.GUARD_JOB_OBJECT:
        pytest.skip("the job's peak process memory is read on Windows")
    args = fx.worst_case_pair(tmp_path, shape)
    with child_accounting() as seen:
        result = scan.scan_frames({"light_00001": args}, workers=1, output_dir=tmp_path, log=lambda *_: None)
    segments = result["light_00001"]["result"]["segments"]
    assert [s["source"] for s in segments] == ["auto", "manual"]
    assert all(s["measurement"]["status"] == "measured" for s in segments)
    diagonal = (shape[0] ** 2 + shape[1] ** 2) ** 0.5
    assert all(s["measurement"]["length_px"] > 0.85 * diagonal for s in segments)     # the ends fade into noise
    peak = max(a["peak_process_memory_bytes"] for a in seen if a.get("peak_process_memory_bytes"))
    unit = ts.scan_unit_bytes(shape[0] * shape[1])
    print(f"worst-case scan child {shape}: peak commit {peak / 1e6:.1f} MB, charged {unit / 1e6:.1f} MB")
    assert peak <= unit


# ================================================================== failures in frame order (real children)

FAKE_CHILD = r'''
import json, os, sys, time
from pathlib import Path
request = Path(sys.argv[2])
if os.environ.get("P1C_IDENTITY"):             # first, before the imports: the pid file is complete once visible
    staged = request.with_suffix(".pid-tmp")
    staged.write_text(str(os.getpid()))
    os.replace(staged, request.with_suffix(".pid"))
sys.path.insert(0, sys.argv[1])
from starstack import transient_stage, transient_scan_worker
def fake(frame_path, master_path, detection, grid_marks, stats_step):
    time.sleep(detection.get("sleep", 0.0))
    behaviour = detection.get("behaviour") or (      # a job's request (no test keys): P1C_FAIL_STEMS names the frames
        "raise" if Path(frame_path).stem[2:] in os.environ.get("P1C_FAIL_STEMS", "").split(",") else "ok")
    if behaviour == "raise":
        raise ValueError(f"injected scan failure of {Path(frame_path).name}")
    if behaviour == "crash":
        os._exit(3)
    if behaviour == "record-fails":            # the error record cannot be written: the frame's error still raises
        (request.parent / f"error-{detection['index']:06d}.pending").write_text("in the way")
        raise ValueError("injected failure without a record")
    return {"stats": {"frame": Path(frame_path).name}, "result": {"segments": [], "sigma": [0.0, 0.0, 0.0],
            "saturated": False, "foreground_pixels": None}, "seconds": 0.0}
transient_stage._scan_frame = fake
code = transient_scan_worker.main([str(request)])
sys.exit(int(os.environ.get("P1C_EXIT_AFTER", code)))
'''


@pytest.fixture
def fake_children(monkeypatch):
    if process_guard.guard_kind() == process_guard.GUARD_NONE:
        pytest.skip("no supported process guard")
    monkeypatch.setattr(scan, "_worker_command", lambda request: [sys.executable, "-c", FAKE_CHILD, str(SRC),
                                                                  str(request)])


def fake_jobs(count, behaviours=None, sleeps=None):
    behaviours, sleeps = behaviours or {}, sleeps or {}
    return {f"light_{i + 1:05d}": {"frame_path": f"r_light_{i + 1:05d}.fit", "master_path": "m.fit",
                                   "detection": {"behaviour": behaviours.get(i, "ok"), "sleep": sleeps.get(i, 0.0),
                                                 "index": i},
                                   "grid_marks": [], "stats_step": 16} for i in range(count)}


def run_scan(tmp_path, jobs, workers, **kwargs):
    folder = tmp_path / f"w{workers}-{time.monotonic_ns()}"
    logs = []
    started = time.perf_counter()
    try:
        result = scan.scan_frames(jobs, workers=workers, output_dir=folder, log=logs.append, **kwargs)
        error = None
    except Exception as exc:                   # noqa: BLE001 -- compared below
        result, error = None, exc
    seconds = time.perf_counter() - started
    runs = list((folder / "scan-runs").iterdir())
    assert len(runs) == 1
    files = sorted(p.name for p in runs[0].iterdir() if not p.name.startswith("worker-"))
    cleanup = [line for line in logs if "could not remove" in line]        # in assertion messages
    return SimpleNamespace(result=result, error=error, seconds=seconds, files=files, logs=logs, run=runs[0],
                           cleanup=cleanup)


COUNT = 9
POSITIONS = h.failure_positions(COUNT)                         # 0, 1, 4, 8


def frame_files(k, *, error=True, pending=False):
    names = [scan.result_name(i) for i in range(k)]
    if error:
        names.append(scan.error_name(k))
    if pending:
        names.append(f"error-{k:06d}.pending")
    return sorted(names)


@pytest.mark.parametrize("k", POSITIONS)
def test_a_raising_frame_is_reported_alike_at_every_process_count(tmp_path, fake_children, k):
    jobs = fake_jobs(COUNT, {k: "raise"})
    outcomes = {w: run_scan(tmp_path, jobs, w) for w in (1, 2, 3, 4, 8)}
    message = f"亮迹扫描失败：第 {k + 1} 帧（light_{k + 1:05d}）：ValueError: injected scan failure of r_light_{k + 1:05d}.fit"
    for workers, outcome in outcomes.items():
        assert type(outcome.error) is scan.ScanWorkerError and str(outcome.error) == message, (workers, outcome.error)
        assert "scan-runs" not in str(outcome.error) and "worker-" not in str(outcome.error)
        assert outcome.files == frame_files(k), (workers, outcome.files, outcome.cleanup)
        assert any("run folder" in line and "ValueError" in line for line in outcome.logs)    # details: worker.log
    record = json.loads((outcomes[3].run / scan.error_name(k)).read_text(encoding="utf-8"))
    assert record == {"version": 1, "index": k, "stem": f"light_{k + 1:05d}", "type": "ValueError",
                      "message": f"injected scan failure of r_light_{k + 1:05d}.fit"}


@pytest.mark.parametrize("k", [0, 4, 8])
def test_a_crash_without_a_record_is_the_childs_first_missing_frame(tmp_path, fake_children, k):
    jobs = fake_jobs(COUNT, {k: "crash"})
    for workers in (1, 3, 8):
        outcome = run_scan(tmp_path, jobs, workers)
        assert str(outcome.error) == f"亮迹扫描失败：第 {k + 1} 帧（light_{k + 1:05d}）：扫描进程异常退出（3），没有错误记录"
        assert outcome.files == frame_files(k, error=False), (workers, outcome.files, outcome.cleanup)


def test_an_error_record_that_cannot_be_written_still_fails_the_frame(tmp_path, fake_children):
    jobs = fake_jobs(COUNT, {2: "record-fails"})
    for workers in (1, 4):
        outcome = run_scan(tmp_path, jobs, workers)
        assert str(outcome.error) == "亮迹扫描失败：第 3 帧（light_00003）：扫描进程异常退出（1），没有错误记录"
        assert outcome.files == frame_files(2, error=False, pending=True), (workers, outcome.files, outcome.cleanup)
        assert any("injected failure without a record" in line for line in outcome.logs)   # the child's traceback


def test_an_earlier_frame_failing_late_outranks_a_later_one_failing_first(tmp_path, fake_children):
    jobs = fake_jobs(COUNT, {2: "raise", 5: "crash"}, sleeps={2: 1.0})
    for workers in (1, 3, 4, 8):
        outcome = run_scan(tmp_path, jobs, workers)
        assert "第 3 帧（light_00003）：ValueError" in str(outcome.error), (workers, outcome.error)
        assert outcome.files == frame_files(2), (workers, outcome.files, outcome.cleanup)


def test_children_with_only_later_frames_are_stopped_at_once(tmp_path, fake_children):
    jobs = fake_jobs(COUNT, {1: "raise"}, sleeps={i: 20.0 for i in range(2, COUNT)})
    for workers in (1, 4, 8):
        outcome = run_scan(tmp_path, jobs, workers)
        assert "第 2 帧（light_00002）" in str(outcome.error)
        assert outcome.seconds < 10.0, (workers, outcome.seconds)            # not the 20 s of the later frames
        assert outcome.files == frame_files(1), (workers, outcome.files, outcome.cleanup)


def test_a_late_non_zero_exit_ranks_after_every_frame(tmp_path, fake_children, monkeypatch):
    monkeypatch.setenv("P1C_EXIT_AFTER", "5")
    for workers in (1, 3):
        outcome = run_scan(tmp_path, fake_jobs(4), workers)
        assert str(outcome.error) == "亮迹扫描失败：扫描进程在发布全部结果后异常退出（5）"
        assert outcome.files == sorted(scan.result_name(i) for i in range(4))
    monkeypatch.delenv("P1C_EXIT_AFTER")
    outcome = run_scan(tmp_path, fake_jobs(4, {3: "raise"}), 2)               # a frame failure outranks it
    assert "第 4 帧" in str(outcome.error)


@pytest.fixture
def controlled_job(tmp_path, monkeypatch, pre_contract_jobs, fake_children):
    """Jobs on four controlled 16 x 24 exposures with trail review on (tests/test_core_selected_inputs.py's stand-in
    engine, whose export also writes the r_light frames), so that core runs the real detect_job and scan_frames with
    the fake scanner children. ``run(fail, scan_workers)`` runs one job in this process and returns (job.json, the
    job folder, what core printed)."""
    import shutil

    import numpy as np

    from starstack import core, imaging
    source = tmp_path / "sources"
    source.mkdir()
    paths = []
    for i in range(4):
        path = source / f"frame-{i}.fit"
        image = np.full((16, 24, 3), .02 + .001 * i, dtype=np.float32)
        image[3:6, 5:8] = .3
        imaging.write_linear(path, image, "Controlled numerical exposure")
        paths.append(str(path))
    engine = {"available": True, "executable": "test-engine", "version": "1.4.4", "message": "controlled"}
    monkeypatch.setattr(core, "doctor", lambda *_: engine)
    monkeypatch.setattr(core, "_spawn_worker", lambda *_: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12))

    def run_script(executable, directory, name, commands, check):
        check()
        work = directory / "work"
        if name == "01_convert":
            for i, path in enumerate(sorted((directory / "stage").glob("*")), 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif name == "02_export":                # identity registration: r_light = light
            for path in sorted(work.glob("light_*.fit")):
                shutil.copyfile(path, work / f"r_{path.name}")
        elif name == "02_stack":
            pixels = np.mean([imaging.read_rgb(p) for p in sorted(work.glob("light_*.fit"))], axis=0)
            imaging.write_linear(work / "sky_master.fit", pixels.astype(np.float32), "controlled stack")
        elif name != "02_probe":
            raise AssertionError(name)
    monkeypatch.setattr(core, "run_script", run_script)
    printed = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append(" ".join(map(str, args))))
    runs = []

    def run(fail, scan_workers):
        monkeypatch.setenv(ts.WORKERS_ENV, str(scan_workers))
        monkeypatch.setenv("P1C_FAIL_STEMS", fail or "")
        printed.clear()
        job = core.submit_job(None, str(tmp_path / f"out-{len(runs)}"), input_files=paths, transient_mode="review")
        runs.append(job)
        core.run_job(job["job_dir"])
        return core.load_job(job["job_dir"]), Path(job["job_dir"]), "\n".join(printed)
    return run


@pytest.mark.parametrize("k", [0, 2, 3])
def test_a_scan_failure_reaches_job_json_alike_at_every_process_count(controlled_job, k):
    """§6.2 (c) at job level: a frame's scan failure leaves the job succeeded with diagnostics.transients failed; its
    error is the U-C5 text (frame number, stem, type, message; no run folder or log tail), the same at 1, 3 and 4 scan
    processes, as are the files the scan leaves; the details go to the job's log; a clean run then reviews."""
    stem = f"light_{k + 1:05d}"
    seen = []
    for workers in (1, 3, 4):
        job, folder, printed = controlled_job(stem, workers)
        assert job["state"] == "succeeded", job.get("error")
        transients = job["diagnostics"]["transients"]
        assert transients == {"status": "failed", "error": f"ScanWorkerError: 亮迹扫描失败：第 {k + 1} 帧（{stem}）："
                                                           f"ValueError: injected scan failure of r_{stem}.fit"}, workers
        assert "transients" not in job["checkpoints"] and job["transient_revision"] is None
        assert "run folder" in printed and f"frame {k + 1} raised ValueError" in printed
        run, = (folder / "work" / ts.TRANSIENTS_DIR / "scan-runs").iterdir()
        files = sorted(p.name for p in run.iterdir() if not p.name.startswith("worker-"))
        assert files == frame_files(k), (workers, files)
        seen.append((job["state"], job.get("error"), transients, job["message"], files))
    assert all(outcome == seen[0] for outcome in seen[1:]), seen
    job, _, _ = controlled_job(None, 3)
    assert job["state"] == "succeeded" and job["diagnostics"]["transients"]["status"] == "needs_review"


def test_success_at_every_process_count(tmp_path, fake_children):
    jobs = fake_jobs(COUNT, sleeps={0: 0.3})
    for workers in (1, 2, 8, 16):
        outcome = run_scan(tmp_path, jobs, workers)
        assert outcome.error is None and list(outcome.result) == list(jobs)
        assert outcome.files == sorted(scan.result_name(i) for i in range(COUNT))
        assert [outcome.result[s]["stats"]["frame"] for s in jobs] == [j["frame_path"] for j in jobs.values()]


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_cancel_with_live_children_returns_within_one_second(tmp_path, fake_children, monkeypatch, workers):
    monkeypatch.setenv("P1C_IDENTITY", "1")
    folder = tmp_path / "out"
    expected = min(workers, COUNT)
    switch = h.CancelSwitch()

    def trip_once_every_child_runs():          # not a fixed delay: child start-up takes longer on a busy machine
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(list(folder.glob("scan-runs/*/*.pid"))) < expected:
            time.sleep(0.02)
        time.sleep(0.3)                        # past the pid file: importing, or in the frame's 30 s sleep
        switch.trip()
    threading.Thread(target=trip_once_every_child_runs, daemon=True).start()
    latency = h.assert_cancels_in_time(
        lambda check: scan.scan_frames(fake_jobs(COUNT, sleeps={i: 30.0 for i in range(COUNT)}), workers=workers,
                                       output_dir=folder, check=check, log=lambda *_: None), switch)
    pids = [int(p.read_text()) for p in folder.glob("scan-runs/*/*.pid")]
    assert len(pids) == expected
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.05)
    assert not any(_alive(pid) for pid in pids)
    print(f"scan cancel latency, {workers} processes: {latency:.3f} s")


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_uint32()
        return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259     # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


# ================================================================== the owner's read rules (fake guard and clock)

class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Children that publish what ``state.plan`` says per frame index ('ok', 'omit', 'foreign', 'corrupt',
    'error', 'bad-error', 'error-only'), exiting with ``state.status(child number)`` (None: running)."""
    clock = Clock()
    state = SimpleNamespace(clock=clock, plan={}, status=lambda number: 1 if state.failing(number) else 0,
                            children=[], logs=[])
    state.failing = lambda number: any(state.plan.get(job["index"]) in ("error", "bad-error", "omit")
                                       for job in state.jobs_of[number])
    state.jobs_of = {}

    class Child:
        def __init__(self, number):
            self.number, self.killed = number, False

        def poll(self):
            return state.status(self.number)

        def kill(self):
            self.killed = True

    @contextmanager
    def guard(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        child = Child(len(state.children))
        state.children.append(child)
        state.jobs_of[child.number] = request["jobs"]
        for job in request["jobs"]:
            what = state.plan.get(job["index"], "ok")
            if what in ("ok", "foreign", "corrupt"):
                stem = "someone-else" if what == "foreign" else job["stem"]
                text = "{" if what == "corrupt" else json.dumps({"version": 1, "stem": stem, "scanned": {
                    "result": {"segments": []}, "stats": {}, "seconds": 0.0}})
                Path(job["result"]).write_text(text, encoding="utf-8")
                continue
            if what == "error":
                Path(job["error"]).write_text(json.dumps({"version": 1, "index": job["index"], "stem": job["stem"],
                                                          "type": "OSError", "message": "disk gone"}),
                                              encoding="utf-8")
            elif what == "bad-error":
                Path(job["error"]).write_text(json.dumps({"version": 1, "index": job["index"] + 1, "stem": "x",
                                                          "type": "OSError", "message": "?"}), encoding="utf-8")
            break                                   # a failing child publishes nothing further
        (Path(command[-1]).parent / f"worker-{child.number:03d}.log").write_text("child log tail line")
        yield child

    monkeypatch.setattr(scan, "guarded_process", guard)
    monkeypatch.setattr(scan, "time", clock)
    state.run = lambda count, workers, **kwargs: scan.scan_frames(
        {f"s{i}": {} for i in range(count)}, workers=workers, output_dir=tmp_path / f"w{workers}-{len(state.logs)}",
        log=state.logs.append, **kwargs)
    return state


def test_a_foreign_result_waits_for_an_earlier_pending_frame(world, monkeypatch):
    """Frame 2's result is someone else's (a failure of frame 2); frame 0's read is blocked until 0.4 s, so the
    failure is reported once frame 0 is read -- and frame 0 is fine, so frame 2's failure it is."""
    world.plan = {2: "foreign"}
    real = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == scan.result_name(0) and world.clock.now < .4:
            raise PermissionError(13, "injected denial")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(scan.ScanWorkerError, match="第 3 帧（s2）的结果身份不符（result-000002.json）"):
        world.run(4, 2)
    assert world.clock.now == pytest.approx(.4)


def test_a_corrupt_later_result_does_not_hide_an_earlier_crash(world):
    world.plan = {1: "omit", 2: "corrupt"}       # child 0: frames 0, 2 (2 does not parse); child 1: 1 (crashes), 3
    with pytest.raises(scan.ScanWorkerError, match="第 2 帧（s1）：扫描进程异常退出（1），没有错误记录"):
        world.run(4, 2)
    world.plan = {2: "corrupt"}
    with pytest.raises(json.JSONDecodeError):    # alone, the corrupt result is its frame's failure, unchanged
        world.run(4, 2)


def test_an_error_record_names_the_frame(world):
    world.plan = {2: "error"}
    for workers in (1, 2, 3):
        with pytest.raises(scan.ScanWorkerError) as error:
            world.run(5, workers)
        assert str(error.value) == "亮迹扫描失败：第 3 帧（s2）：OSError: disk gone"


def test_an_invalid_error_record_is_reported_with_the_log_tail(world):
    world.plan = {1: "bad-error"}
    with pytest.raises(scan.ScanWorkerError) as error:
        world.run(3, 1)
    assert "第 2 帧（s1）：扫描进程退出（1），错误记录无效（ValueError）" in str(error.value)
    assert "child log tail line" in str(error.value)


def test_an_unreadable_error_record_waits_then_is_reported_with_the_log_tail(world, monkeypatch):
    world.plan = {0: "error"}
    real = Path.read_text

    def read(path, *args, **kwargs):
        if path.name.startswith("error-"):
            error = PermissionError(13, "injected denial")
            raise error
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(scan.ScanWorkerError) as error:
        world.run(2, 1)
    assert "错误记录无法读取（PermissionError）" in str(error.value) and "child log tail line" in str(error.value)
    assert 2.0 <= world.clock.now <= 2.11


def test_the_files_after_stop_are_removed_once_the_children_ended(world, tmp_path):
    world.plan = {3: "error"}
    with pytest.raises(scan.ScanWorkerError):
        world.run(8, 4)                              # children 0-3 publish frames 0-2 and 4-7; 3 fails
    run, = (tmp_path / "w4-0" / "scan-runs").iterdir()
    names = sorted(p.name for p in run.iterdir() if not p.name.startswith("worker-"))
    assert names == sorted([scan.result_name(i) for i in range(3)] + [scan.error_name(3)])
    assert all(child.killed is False for child in world.children)     # all had exited: nothing to stop


def _left(world, tmp_path, workers=2):
    """Run a scan in ``world`` that fails with a ScanWorkerError; the result / error / pending files it leaves."""
    before = set(tmp_path.iterdir())
    with pytest.raises(scan.ScanWorkerError, match="第 2 帧"):
        world.run(4, workers)
    folder, = set(tmp_path.iterdir()) - before
    run, = (folder / "scan-runs").iterdir()
    return sorted(p.name for p in run.iterdir() if not p.name.startswith("worker-"))


def test_the_cleanup_watches_for_the_last_write_of_a_child_still_ending(world, tmp_path, monkeypatch):
    """A child's job that still held a process at its guard's close (the venv launcher's interpreter ending after
    it): the folder is watched for REMOVAL_SETTLE_SECONDS, so a file the ending child completes is removed too."""
    world.plan = {1: "error"}                        # child 0: frames 0, 2 published; child 1: frame 1 fails
    inner = scan.guarded_process

    @contextmanager
    def ending(command, **kwargs):
        with inner(command, **kwargs) as child:
            child.accounting = {"active_processes": 1}
            yield child
    monkeypatch.setattr(scan, "guarded_process", ending)
    sleeps = []
    advance = world.clock.sleep

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:                         # the ending child's last write, after the first removal pass
            run, = tmp_path.glob("w2-*/scan-runs/*")
            (run / "result-000003.pending").write_text("{")
        advance(seconds)
    monkeypatch.setattr(world.clock, "sleep", sleep)
    assert _left(world, tmp_path) == [scan.error_name(1), scan.result_name(0)]
    assert len(sleeps) > 3
    assert world.clock.now >= 3 * .02 + scan.REMOVAL_SETTLE_SECONDS - 1e-9
    assert world.clock.now < scan.REMOVAL_SECONDS


def test_the_cleanup_retries_locked_files_and_logs_what_stays(world, tmp_path, monkeypatch):
    """No child still ending: one pass, no wait (the fake clock stays at 0) -- unless a file is locked: a sharing
    denial is retried within REMOVAL_SECONDS; a file locked longer is logged and kept."""
    world.plan = {1: "error"}
    assert _left(world, tmp_path) == [scan.error_name(1), scan.result_name(0)] and world.clock.now == 0
    real = Path.unlink
    locked = {scan.result_name(2): 1.0}              # file -> locked until this clock time

    def unlink(path, *args, **kwargs):
        if world.clock.now < locked.get(path.name, -1):
            raise PermissionError(13, "injected sharing denial")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    assert _left(world, tmp_path) == [scan.error_name(1), scan.result_name(0)]
    assert 1.0 <= world.clock.now <= 1.05
    locked[scan.result_name(2)] = 1e9
    assert _left(world, tmp_path) == [scan.error_name(1), scan.result_name(0), scan.result_name(2)]
    assert 1.0 + scan.REMOVAL_SECONDS <= world.clock.now <= 1.1 + scan.REMOVAL_SECONDS
    assert any("could not remove result-000002.json after the failure (still locked)" in line for line in world.logs)


# ================================================================== real data: F48 frames

def test_f48_frames_scan_alike_at_one_default_and_eight_processes(tmp_path):
    job_dir = h.f48_job()
    work = job_dir / "work"
    run, = sorted(p for p in (work / ts.TRANSIENTS_DIR / "scan-runs").iterdir() if p.is_dir())
    requests = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run.glob("worker-*.json"))]
    jobs = {job["stem"]: job for request in requests for job in request["jobs"]}
    stored = {}
    for path in sorted(run.glob("result-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored[payload["stem"]] = payload["scanned"]
    stems = sorted(jobs, key=lambda s: -len(stored[s]["result"]["segments"]))[:4] + sorted(jobs)[:2]
    stems = sorted(set(stems))
    chosen = {stem: jobs[stem]["args"] for stem in stems}
    for workers in (1, None, 8):
        count = workers or ts.default_workers(ts.frame_pixels(Path(chosen[stems[0]]["master_path"])),
                                              frames=len(stems))
        started = time.perf_counter()
        result = scan.scan_frames(chosen, workers=min(count, len(stems)), output_dir=tmp_path / f"w{workers}",
                                  log=lambda *_: None)
        print(f"F48 {len(stems)} frames at {count} processes: {time.perf_counter() - started:.1f} s")
        for stem in stems:
            got = {key: result[stem][key] for key in ("stats", "result")}
            want = {key: stored[stem][key] for key in ("stats", "result")}
            assert json.dumps(got, allow_nan=False) == json.dumps(want, allow_nan=False), stem
