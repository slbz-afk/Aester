"""Real Siril owner-death acceptance; only owned synthetic files/processes."""
from pathlib import Path
from ctypes import wintypes
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from starstack.siril import run_script, Cancelled
from starstack.process_guard import guarded_process
from validate_lifecycle import OwnedProcessObserver, hashes, timestamp

EXE = ROOT / "tools/siril/bin/siril-cli.exe"


def native(observer):
    kernel = observer.kernel
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    return kernel


def identity(kernel, handle):
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    name = ctypes.create_unicode_buffer(32768)
    size = wintypes.DWORD(len(name))
    if not kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"creation_time": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime,
            "image_path": os.path.normcase(name.value)}


def owner(directory, nonce):
    observer = OwnedProcessObserver(EXE)
    kernel = native(observer)
    info = identity(kernel, kernel.GetCurrentProcess())
    info.update(pid=os.getpid(), parent_pid=os.getppid(), nonce=nonce)
    temporary = directory / "owner.tmp"
    temporary.write_text(json.dumps(info), encoding="utf-8")
    temporary.replace(directory / "owner.json")

    def stop():
        if (directory / "stop.request").exists():
            raise Cancelled("Acceptance cleanup requested")

    try:
        run_script(str(EXE), directory, "crash_check", [
            "setext fit", "set32bits", "setcompress 0", "cd input", "convert light -out=../work",
            "cd ../work", "setref light 1", "register light -transf=homography",
            "register light -transf=homography", "register light -transf=homography"], stop)
    except Cancelled:
        return 0
    return 0


def snapshot(directory):
    return {str(p.relative_to(directory)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(directory.rglob("*")) if p.is_file()}


def main():
    root = ROOT / "outputs" / ("guard-crash-" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True)
    (root / "input").mkdir()
    (root / "work").mkdir()
    demo = ROOT / "data/demo/synthetic-fbd621c4"
    original = hashes(demo)
    for path in sorted(demo.glob("*.fit")):
        shutil.copyfile(path, root / "input" / path.name)
    owned_original = hashes(root / "input")
    nonce = uuid.uuid4().hex
    report = {"passed": False, "real_siril": True, "synthetic_only": True,
              "test": "hard termination of verified real owner during Siril registration",
              "started_at": timestamp(), "run_root": str(root), "resumed": False}
    observer = OwnedProcessObserver(EXE)
    kernel = native(observer)
    owner_handle = None
    siril_handles = {}
    launcher = None
    cleanup = ExitStack()
    killed_owner = False
    started = time.monotonic()
    try:
        with (root / "owner.log").open("ab", buffering=0) as log:
            # A separate outer Job bounds failed acceptance cleanup. Keep it
            # open throughout the observation so it does not cause the tested
            # Siril exit: only the verified owner is killed during the test.
            launcher = cleanup.enter_context(guarded_process(
                                        [sys.executable, str(Path(__file__).resolve()), "--owner", str(root), nonce],
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        deadline = time.monotonic() + 20
        while not (root / "owner.json").exists() and time.monotonic() < deadline:
            if launcher.poll() is not None:
                raise RuntimeError("Owner ended before identity handshake")
            time.sleep(.02)
        handshake = json.loads((root / "owner.json").read_text(encoding="utf-8"))
        if handshake["nonce"] != nonce or not (handshake["pid"] == launcher.pid or
                                               handshake["parent_pid"] == launcher.pid):
            raise RuntimeError("Owner handshake is not attributable to this launcher")
        candidate = kernel.OpenProcess(0x100000 | 0x1000 | 1, False, handshake["pid"])
        expected = {key: handshake[key] for key in ("creation_time", "image_path")}
        try:
            if not candidate or identity(kernel, candidate) != expected:
                raise RuntimeError("Owner PID no longer identifies the handshaken process")
        except BaseException:
            if candidate:
                kernel.CloseHandle(candidate)
            raise
        owner_handle = candidate  # Cleanup receives only verified identities.
        report["owner"] = handshake
        report["launcher_pid"] = launcher.pid
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            observer.observe(handshake["pid"])
            log_path = root / "crash_check.log"
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            alive = observer.living()
            if "Running command: register" in log_text and alive:
                for pid in alive:
                    handle = kernel.OpenProcess(0x100000 | 0x1000 | 1, False, pid)
                    try:
                        if not handle or identity(kernel, handle) != identity(kernel, observer.handles[pid]):
                            raise RuntimeError("Siril process identity changed before capture")
                    except BaseException:
                        if handle:
                            kernel.CloseHandle(handle)
                        raise
                    siril_handles[pid] = handle
                report["kill_requested_at"] = timestamp()
                report["live_siril_at_kill"] = {str(pid): identity(kernel, h) for pid, h in siril_handles.items()}
                if kernel.WaitForSingleObject(owner_handle, 0) != 258:
                    raise RuntimeError("Owner already exited before the intended hard kill")
                if not kernel.TerminateProcess(owner_handle, 73):
                    raise ctypes.WinError(ctypes.get_last_error())
                killed_owner = True
                break
            if kernel.WaitForSingleObject(owner_handle, 0) != 258:
                raise RuntimeError("Owner finished before in-flight registration was observable")
            time.sleep(.02)
        if not killed_owner:
            raise RuntimeError("Could not observe registration within bounded time")
        if kernel.WaitForSingleObject(owner_handle, 10000) != 0:
            raise RuntimeError("Actual owner did not exit")
        for pid, handle in siril_handles.items():
            if kernel.WaitForSingleObject(handle, 10000) != 0:
                raise RuntimeError(f"Observed Siril process {pid} survived actual owner death")
        report["owner_and_siril_exited_at"] = timestamp()
        launcher.wait(timeout=10)
        # All writes monitored here are from the owned job, including its logs.
        snapshots = []
        for _ in range(3):
            time.sleep(.7)
            snapshots.append(snapshot(root))
        report["output_stable"] = snapshots[0] == snapshots[1] == snapshots[2]
        report["stability_observation_seconds"] = 2.1
        report["stable_snapshot"] = snapshots[-1]
        if not report["output_stable"]:
            raise RuntimeError("Owned output files kept changing after owner/Siril exit")
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        # Prefer the real adapter's normal guard cleanup while its owner lives.
        (root / "stop.request").touch()
        if owner_handle and kernel.WaitForSingleObject(owner_handle, 0) == 258:
            kernel.WaitForSingleObject(owner_handle, 15000)
        # Last-resort cleanup is restricted to handles with verified identity.
        if owner_handle and kernel.WaitForSingleObject(owner_handle, 0) == 258:
            kernel.TerminateProcess(owner_handle, 74)
            kernel.WaitForSingleObject(owner_handle, 10000)
            report["cleanup_forced_owner"] = True
        for pid, handle in siril_handles.items():
            if kernel.WaitForSingleObject(handle, 0) == 258:
                kernel.TerminateProcess(handle, 74)
                kernel.WaitForSingleObject(handle, 10000)
                report.setdefault("cleanup_forced_siril", []).append(pid)
    finally:
        # On success this runs only after all exit/stability assertions; on a
        # failed handshake it also stops unreported descendants of our launcher.
        cleanup.close()
        report["original_demo_hashes"] = original
        report["original_demo_unchanged"] = hashes(demo) == original
        report["owned_input_unchanged"] = hashes(root / "input") == owned_original
        report["observed_siril_descendants"] = observer.records
        report["living_observed_siril_at_end"] = observer.living()
        report["actual_owner_alive_at_end"] = bool(owner_handle and kernel.WaitForSingleObject(owner_handle, 0) == 258)
        for handle in siril_handles.values():
            kernel.CloseHandle(handle)
        if owner_handle:
            kernel.CloseHandle(owner_handle)
        observer.close()
        report["passed"] = bool(report["passed"] and report["original_demo_unchanged"] and
                                report["owned_input_unchanged"] and not report["living_observed_siril_at_end"])
        report["finished_at"] = timestamp()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        path = ROOT / "docs/diagnostics/guard_crash.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(path), "error": report.get("error"),
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--owner":
        raise SystemExit(owner(Path(sys.argv[2]), sys.argv[3]))
    raise SystemExit(main())
