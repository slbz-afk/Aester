"""Bounded, real Windows/Siril lifecycle check using owned synthetic files.

Run from the project root with .venv/Scripts/python.exe. This does not kill
workers, touch unrelated processes, or change production modules.

Cancellations of one job, each followed by a resume (efficiency P1b-3, P1c O-1):
  1. "export": an in-flight Siril export -- the stack-phase registration commands
     (register / seqapplyreg) running in 02_export (02_stack in the pre-P1b
     single-process layout);
  2. "stack": an in-flight Siril stack after 02_export finished (the boundary
     the L split adds; skipped under AESTER_L_SIRIL_LAYOUT=combined);
  3. "overlap": the trail scan running beside the post-stack work (efficiency
     P1c O-1, prereg §3.6 / §6.4 C7): the marker work/transients.o1-pending is
     present and the scan's children were launched. Skipped, with the
     attempt's overlap.o1 reason, when O-1 does not run (on the small
     synthetic fixture its time criterion turns it off); a cancel there may
     wait for the native remap's remaining part (prereg §5 item 5, U-C3).
Each must reach "cancelled" with no living owned engine (and, for "overlap",
no living Python descendant 5 s after it, B's output removed with the marker);
the last resume must succeed.  Siril's progress is matched on its untranslated
stdout lines ("running command <name>"); the translated ones follow the system
language (for example "正在运行命令：register" on a Chinese system).
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from starstack.core import submit_job, load_job, cancel_job, resume_job, legacy_stack_layout
from starstack.siril import doctor, run_script

# Siril's untranslated stdout lines (the translated ones, such as "Running command: register", follow the language).
EXPORT_MARKERS = ("running command register", "running command seqapplyreg")
STACK_MARKER = "running command stack"
TERMINAL = {"succeeded", "failed", "cancelled"}
O1_MARKER = "transients.o1-pending"            # starstack.core.O1_MARKER (efficiency P1c O-1)
O1_NOTE = "overlap.o1"                         # starstack.core.O1_NOTE
PYTHON_IMAGES = {"python.exe", "pythonw.exe"}  # the job worker (and its venv launcher) and the scan children
QUIET_SECONDS = 5.0                            # prereg §6.4 criterion 2: no living descendant 5 s after 'cancelled'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir()) if p.is_file()}


class OwnedProcessObserver:
    """Keep read-only handles to exact Siril descendants of this job worker."""

    def __init__(self, executable):
        if os.name != "nt":
            raise RuntimeError("This bounded acceptance harness currently targets Windows")
        self.executable = os.path.normcase(str(executable.resolve()))
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

        self.Entry = Entry
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Process32FirstW", "Process32NextW"):
            function = getattr(self.kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
            function.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                         wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handles = {}
        self.records = []
        self.python_handles = {}                  # Python descendants (the worker, the trail scan's children)
        self.python_records = []

    def observe(self, worker_pid):
        if not worker_pid:
            return
        handle = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        rows = []
        try:
            item = self.Entry()
            item.dwSize = ctypes.sizeof(item)
            found = self.kernel.Process32FirstW(handle, ctypes.byref(item))
            while found:
                rows.append((int(item.th32ProcessID), int(item.th32ParentProcessID), item.szExeFile))
                found = self.kernel.Process32NextW(handle, ctypes.byref(item))
        finally:
            self.kernel.CloseHandle(handle)
        owned = {int(worker_pid)}
        for _ in range(5):
            expanded = owned | {pid for pid, parent, _ in rows if parent in owned}
            if expanded == owned:
                break
            owned = expanded
        for pid, parent, name in rows:
            if pid in owned and pid not in self.python_handles and name.lower() in PYTHON_IMAGES:
                process = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
                if process:                   # by handle: a recycled pid is never mistaken for it
                    self.python_handles[pid] = process
                    self.python_records.append({"pid": pid, "parent_pid": parent, "worker_pid": worker_pid,
                                                "image": name, "observed_at": timestamp()})
                continue
            if pid not in owned or pid in self.handles or name.lower() != "siril-cli.exe":
                continue
            process = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
            if not process:  # The process may have exited since the snapshot.
                continue
            buffer = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buffer))
            if (not self.kernel.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size))
                    or os.path.normcase(buffer.value) != self.executable):
                self.kernel.CloseHandle(process)
                continue
            self.handles[pid] = process
            self.records.append({"pid": pid, "parent_pid": parent, "worker_pid": worker_pid,
                                 "executable": buffer.value, "observed_at": timestamp()})

    def living(self, handles=None):
        # An open process handle identifies the observed process even if its PID
        # is later recycled. This never sends a signal or terminates a process.
        result = []
        for pid, handle in (self.handles if handles is None else handles).items():
            status = self.kernel.WaitForSingleObject(handle, 0)
            if status == 258:
                result.append(pid)
            elif status != 0:
                raise RuntimeError(f"Could not query process handle {pid}: wait result {status}")
        return result

    def living_python(self):
        """Observed Python descendants still running (the job worker and the trail scan's children)."""
        return self.living(self.python_handles)

    def close(self):
        for handle in [*self.handles.values(), *self.python_handles.values()]:
            self.kernel.CloseHandle(handle)


def log_tail(job_dir, name, offset=0):
    """A Siril log of the job from byte ``offset`` on (logs are appended across attempts); '' when absent."""
    path = Path(job_dir) / f"{name}.log"
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read().decode("utf-8", errors="replace")


def log_size(job_dir, name):
    path = Path(job_dir) / f"{name}.log"
    return path.stat().st_size if path.exists() else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", type=Path, default=ROOT / "data" / "demo" / "synthetic-fbd621c4",
                        help="the synthetic FITS fixture (read only; default: data/demo/synthetic-fbd621c4)")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs",
                        help="parent of the run folder lifecycle-<id> (default: outputs)")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "diagnostics" / "lifecycle.json",
                        help="report file (default: docs/diagnostics/lifecycle.json)")
    parser.add_argument("--siril", type=Path, default=ROOT / "tools" / "siril" / "bin" / "siril-cli.exe",
                        help="the pinned siril-cli.exe (default: tools/siril/bin/siril-cli.exe; a git worktree "
                             "names the main checkout's)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    demo = args.demo
    executable = args.siril
    run_root = args.outputs / f"lifecycle-{uuid.uuid4().hex[:10]}"
    run_root.mkdir(parents=True)
    owned_input = run_root / "input"
    owned_input.mkdir()
    report_file = args.report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    layout = legacy_stack_layout()     # the job worker inherits this environment
    report = {"started_at": timestamp(), "real_siril": True, "synthetic_only": True,
              "strong_worker_kill_tested": False, "run_root": str(run_root), "passed": False,
              "engine": doctor(str(executable)), "l_siril_layout": layout, "transitions": [], "cancel_points": []}
    observer = OwnedProcessObserver(executable)
    job_dir = None
    original_before = hashes(demo)
    original_owned = None

    def event(job):
        value = (job["state"], job["stage"])
        if not report["transitions"] or tuple(report["transitions"][-1]["value"]) != value:
            report["transitions"].append({"at": timestamp(), "value": list(value)})

    def current():
        job = load_job(job_dir)
        observer.observe(job.get("worker_pid"))
        event(job)
        return job

    def export_running(offsets):
        # Cancel point 1: Siril is running the stack-phase registration commands (an export).  Split layout:
        # 02_export; the pre-P1b single process (no 02_export.log): 02_stack.
        text = log_tail(job_dir, "02_export", offsets["02_export"])
        if not text and not (Path(job_dir) / "02_export.log").exists():
            text = log_tail(job_dir, "02_stack", offsets["02_stack"])
        return any(marker in text for marker in EXPORT_MARKERS)

    def stack_running(offsets):
        # Cancel point 2: the stack process of this attempt runs, after this attempt's 02_export finished.
        return (log_size(job_dir, "02_export") > offsets["02_export"]
                and STACK_MARKER in log_tail(job_dir, "02_stack", offsets["02_stack"]))

    def cancel_at(point, ready, offsets, timeout):
        # Wait for the Siril command and an observed live child, not merely the scheduler's pre-launch "stack" state.
        request = None
        deadline = time.monotonic() + timeout
        job = current()
        while time.monotonic() < deadline:
            job = current()
            if job["state"] in TERMINAL:
                break
            if job["stage"] == "stack" and ready(offsets) and observer.living():
                request = {"point": point, "at": timestamp(), "stage": job["stage"],
                           "owned_engine_pids": observer.living()}
                cancel_job(job_dir)
                break
            time.sleep(.04)
        if request is None:
            raise RuntimeError(f"Could not observe an in-flight Siril {point} to cancel: {job['state']}")
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            job = current()
            if job["state"] in TERMINAL:
                break
            time.sleep(.1)
        after = {"state": job["state"], "stage": job["stage"], "checkpoints": job.get("checkpoints"),
                 "living_owned_engine_pids": observer.living()}
        report["cancel_points"].append({**request, "after_cancel": after})
        if point == "export":       # the report keys of the single cancel point before P1b
            report["cancel_request"], report["after_cancel"] = request, after
        if job["state"] != "cancelled" or observer.living():
            raise RuntimeError(f"Cancellation at the {point} point did not reach cancelled with the observed "
                               "Siril process stopped")

    def offsets_now():
        return {name: log_size(job_dir, name) for name in ("02_export", "02_stack")}

    def overlap_window():
        # Cancel point 3 (efficiency P1c O-1): B started (the marker) and launched its scan children (their ledgers)
        work = Path(job_dir) / "work"
        return (work / O1_MARKER).exists() and any(work.glob("transients/scan-runs/*/worker-*.json"))

    def cancel_overlap(timeout):
        """Cancel in the O-1 window of the running attempt; False (the point skipped, with the attempt's overlap.o1
        reason) when the attempt ends or passes the window without one."""
        request = None
        deadline = time.monotonic() + timeout
        job = current()
        while time.monotonic() < deadline:
            job = current()
            if job["state"] in TERMINAL or job["stage"] in ("compose", "complete"):
                break
            if job["state"] == "running" and overlap_window():
                request = {"point": "overlap", "at": timestamp(), "stage": job["stage"],
                           "living_python_pids": observer.living_python()}
                requested_at = time.monotonic()
                cancel_job(job_dir)
                break
            time.sleep(.02)
        if request is None:
            while time.monotonic() < deadline and job["state"] not in TERMINAL:
                time.sleep(.1)
                job = current()
            note = (((job.get("diagnostics") or {}).get("timings") or {}).get("notes") or {}).get(O1_NOTE)
            note = note[-1] if isinstance(note, list) and note else note
            report["cancel_points"].append({
                "point": "overlap", "skipped": "the O-1 window was not reached", "state": job["state"],
                "overlap_o1": ({key: note.get(key) for key in ("eligible", "reason", "outcome")}
                               if isinstance(note, dict) else None)})
            return False
        cancelled_at = None
        # prereg §5 item 5 / U-C3: a cancel in the window may wait for the native remap's remaining part
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            job = current()
            if job["state"] in TERMINAL:
                cancelled_at = time.monotonic()
                break
            time.sleep(.1)
        quiet = time.monotonic() + QUIET_SECONDS
        while time.monotonic() < quiet and (observer.living_python() or observer.living()):
            time.sleep(.1)
        work = Path(job_dir) / "work"
        after = {"state": job["state"], "stage": job["stage"], "checkpoints": job.get("checkpoints"),
                 "living_owned_engine_pids": observer.living(), "living_python_pids": observer.living_python(),
                 "marker_present": (work / O1_MARKER).exists(),
                 "transients_present_without_checkpoint": ((work / "transients").exists()
                                                           and "transients" not in (job.get("checkpoints") or [])),
                 "seconds_to_terminal": None if cancelled_at is None else round(cancelled_at - requested_at, 3)}
        report["cancel_points"].append({**request, "after_cancel": after})
        if (job["state"] != "cancelled" or after["living_owned_engine_pids"] or after["living_python_pids"]
                or after["marker_present"] or after["transients_present_without_checkpoint"]):
            raise RuntimeError("Cancellation in the O-1 window did not reach cancelled with every descendant stopped "
                               "and the overlapped scan's output removed")
        return True

    try:
        frames = sorted(demo.glob("*.fit"))
        if not frames:
            raise RuntimeError("Named synthetic fixture does not contain FITS images")
        # Repeat the eight synthetic frames twice, retaining the source files.
        for index in range(16):
            shutil.copyfile(frames[index % len(frames)], owned_input / f"frame_{index + 1:03d}.fit")
        original_owned = hashes(owned_input)
        job = submit_job(str(owned_input), str(run_root / "jobs"), str(executable),
                         idempotency_key="lifecycle-acceptance")
        job_dir = job["job_dir"]
        report["job_dir"] = job_dir
        cancel_at("export", export_running, offsets_now(), 70)
        if layout == "split":
            offsets = offsets_now()
            job = resume_job(job_dir)
            report["resume_requested_at"] = [timestamp()]
            cancel_at("stack", stack_running, offsets, 150)
        else:
            report["resume_requested_at"] = []
            report["cancel_points"].append({"point": "stack", "skipped": f"AESTER_L_SIRIL_LAYOUT={layout}"})
        job = resume_job(job_dir)
        report["resume_requested_at"].append(timestamp())
        if cancel_overlap(150):          # point 3: cancelled in the O-1 window; then the last resume
            job = resume_job(job_dir)
            report["resume_requested_at"].append(timestamp())
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            job = current()
            if job["state"] in TERMINAL:
                break
            time.sleep(.1)
        report["after_resume"] = {"state": job["state"], "stage": job["stage"],
                                  "error": job.get("error"), "artifacts": job.get("artifacts", []),
                                  "living_owned_engine_pids": observer.living()}
        if job["state"] != "succeeded" or observer.living():
            raise RuntimeError("Resumed job did not finish successfully with its observed engine stopped")
        if not job.get("artifacts") or not all(Path(a["path"]).is_file() for a in job["artifacts"]):
            raise RuntimeError("Completed job has missing published artifacts")
        fault_dir = run_root / "missing-file-fault"
        fault_dir.mkdir()
        fault_start = time.monotonic()

        def fault_timeout():
            if time.monotonic() - fault_start > 25:
                raise TimeoutError("Missing-file check exceeded its deadline")

        try:
            run_script(str(executable), fault_dir, "missing_file", ["load missing_input.fit"], fault_timeout)
        except RuntimeError as exc:
            match = re.search(r"（(-?\d+)）", str(exc))
            report["missing_file"] = {"exception_type": type(exc).__name__, "error": str(exc),
                                      "return_code": int(match.group(1)) if match else None,
                                      "log": str(fault_dir / "missing_file.log")}
            if not match or int(match.group(1)) == 0:
                raise RuntimeError("Missing-file failure lacked confirmed nonzero Siril exit") from exc
        else:
            raise RuntimeError("Missing-file script returned without an error")
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if job_dir:
            job = load_job(job_dir)
            if job["state"] not in {"succeeded", "failed", "cancelled"}:
                cancel_job(job_dir)
                stop = time.monotonic() + 25
                while time.monotonic() < stop:
                    job = current()
                    if job["state"] in {"succeeded", "failed", "cancelled"}:
                        break
                    time.sleep(.1)
                report["cleanup_job_state"] = job["state"]
    finally:
        report["original_demo_hashes"] = original_before
        report["original_demo_unchanged"] = original_before == hashes(demo)
        report["owned_input_unchanged"] = original_owned is not None and original_owned == hashes(owned_input)
        report["observed_owned_engine_processes"] = observer.records
        report["living_owned_engine_pids_at_end"] = observer.living()
        report["observed_python_descendants"] = observer.python_records
        report["living_python_pids_at_end"] = observer.living_python()
        observer.close()
        report["finished_at"] = timestamp()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["passed"] = bool(report["passed"] and report["original_demo_unchanged"]
                                and report["owned_input_unchanged"]
                                and not report["living_owned_engine_pids_at_end"])
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "elapsed_seconds": report["elapsed_seconds"],
                      "report": str(report_file), "job_dir": job_dir,
                      "error": report.get("error")}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
