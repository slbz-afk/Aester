"""Tests for process_guard.

Run with ``pytest test_process_guard.py`` (preferred) or, without pytest,
``python -m unittest -v test_process_guard``: the cases are plain
``unittest.TestCase`` classes so that pytest collects them natively and no
third-party package is needed.

Three groups:

* ``StructLayout``      -- the ctypes definitions match the Windows SDK sizes
                           (runs on any platform).
* ``FakeKernel32Tests`` -- the Windows orchestration (call order, cleanup on
                           every failure path, handle ownership) against a
                           recording fake of kernel32 (runs on any platform).
* ``RealProcessTests``  -- real processes under the guard the current platform
                           provides: success, cancel, init failure, and the
                           owner being force-killed while the child runs.
                           Children are ``python -c`` snippets that only
                           sleep or append bytes to a marker file.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from starstack import process_guard as pg  # noqa: E402

IS_WINDOWS = os.name == "nt"
NATIVE_GUARD = pg.guard_kind()


# --------------------------------------------------------------------------- #
# helper child programs (all harmless)                                        #
# --------------------------------------------------------------------------- #

def py(*code_lines: str) -> list[str]:
    return [sys.executable, "-c", "\n".join(code_lines)]


SLEEP_CHILD = py("import time", "time.sleep(60)")

# Small child programs contain no shell/network operations. Every writer has
# both a finite lifetime and a per-test stop marker for failure-path cleanup.
IDENTITY = '''
import ctypes, json, os
def identity(pid):
    if os.name == "nt":
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        k.OpenProcess.restype = ctypes.c_void_p
        k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            raise ctypes.WinError(ctypes.get_last_error())
        values = [ctypes.c_uint64() for _ in range(4)]
        try:
            if not k.GetProcessTimes(h, *(ctypes.byref(v) for v in values)):
                raise ctypes.WinError(ctypes.get_last_error())
            created = values[0].value
        finally:
            k.CloseHandle(h)
    else:
        with open(f"/proc/{pid}/stat") as f:
            created = int(f.read().rsplit(")", 1)[1].split()[19])
    return {"pid": pid, "created": created}
'''

HEARTBEAT = (
    "import os, sys, time\n"
    "f = open(sys.argv[1], 'ab', buffering=0)\n"
    "deadline = time.monotonic() + 60\n"
    "while time.monotonic() < deadline and not os.path.exists(sys.argv[1] + '.stop'):\n"
    "    f.write(b'x'); time.sleep(0.05)\n"
)

# Starts a heartbeat grandchild with a plain Popen, prints its pid, sleeps.
HEARTBEAT_PARENT = (
    IDENTITY + "\nimport sys, subprocess, time\n"
    "child = subprocess.Popen([sys._base_executable, '-c', %r, sys.argv[1]],\n"
    "                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "print(json.dumps(identity(child.pid)), flush=True)\n"
    "deadline = time.monotonic() + 60\n"
    "try:\n"
    "    while time.monotonic() < deadline and not os.path.exists(sys.argv[1] + '.stop'):\n"
    "        time.sleep(0.05)\n"
    "finally:\n"
    "    if child.poll() is None: child.kill()\n"
    "    child.wait(timeout=8)\n"
) % HEARTBEAT

# The "owner": starts a child (heartbeat writer, or a heartbeat *parent*)
# either under the guard or with a plain Popen, prints the pid(s), then sleeps
# until the test kills it.  argv: mode(guarded|plain) marker payload(direct|parent)
OWNER = (
    IDENTITY + "\nimport sys, subprocess, time\n"
    "from starstack import process_guard as pg\n"
    "mode, marker, payload = sys.argv[1:4]\n"
    # The Windows venv redirector's Popen.pid can differ from this real owner.
    "print(json.dumps(identity(os.getpid())), flush=True)\n"
    "if sys.stdin.readline().strip() != 'start': sys.exit(0)\n"
    "snippet = %r if payload == 'direct' else %r\n"
    "cmd = [sys._base_executable, '-c', snippet, marker]\n"
    "log = open(marker + '.owner.log', 'ab', buffering=0)\n"
    "cm = None\n"
    "if mode == 'guarded':\n"
    "    cm = pg.guarded_process(cmd, stdout=log, stderr=subprocess.STDOUT)\n"
    "    proc = cm.__enter__()\n"
    "else:\n"
    "    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)\n"
    "print(json.dumps(identity(proc.pid)), flush=True)\n"
    "deadline = time.monotonic() + 60\n"
    "try:\n"
    "    while time.monotonic() < deadline and not os.path.exists(marker + '.stop'):\n"
    "        time.sleep(0.05)\n"
    "finally:\n"
    "    if proc.poll() is None: proc.kill()\n"
    "    proc.wait(timeout=8)\n"
    "    if cm is not None: cm.__exit__(None, None, None)\n"
    "    log.close()\n"
) % (HEARTBEAT, HEARTBEAT_PARENT)


def pid_alive(pid: int) -> bool:
    """Same semantics as locking.pid_alive, kept local so the tests stand alone."""
    if IS_WINDOWS:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie waiting for init to reap it is dead for our purposes
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


class OwnedProcess:
    """Pin the exact reported process; never reacquire a PID during cleanup.

    The reporting process stays alive (owner handshake or a held child handle)
    while we open it. Creation identity is checked before any termination.
    Windows handles and Linux pidfds remain bound if the numeric PID is reused.
    """

    def __init__(self, reported: dict):
        self.pid = reported["pid"]
        created = reported["created"]
        if type(self.pid) is not int or self.pid <= 0 or type(created) is not int:
            raise ValueError("invalid process identity")
        self._closed = False
        self._kill_requested = False
        if IS_WINDOWS:
            self.api = ctypes.WinDLL("kernel32", use_last_error=True)
            self.api.OpenProcess.restype = ctypes.c_void_p
            self.api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
            self.api.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
            self.api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self.api.WaitForSingleObject.restype = ctypes.c_uint32
            self.api.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self.api.CloseHandle.argtypes = [ctypes.c_void_p]
            self.handle = self.api.OpenProcess(0x00101001, False, self.pid)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                values = [ctypes.c_uint64() for _ in range(4)]
                if not self.api.GetProcessTimes(self.handle, *(ctypes.byref(v) for v in values)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if values[0].value != created:
                    raise ValueError("process creation identity changed")
            except BaseException:
                self.api.CloseHandle(self.handle)
                raise
        else:
            if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
                raise unittest.SkipTest("safe owner-death tests require Linux pidfds")
            self.handle = os.pidfd_open(self.pid)
            try:
                text = Path(f"/proc/{self.pid}/stat").read_text()
                if int(text.rsplit(")", 1)[1].split()[19]) != created:
                    raise ValueError("process creation identity changed")
            except BaseException:
                os.close(self.handle)
                raise

    def alive(self):
        if self._closed:
            return False
        if IS_WINDOWS:
            state = self.api.WaitForSingleObject(self.handle, 0)
            if state == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            return state == 258
        return not select.select([self.handle], [], [], 0)[0]

    def kill(self):
        if self._closed or self._kill_requested or not self.alive():
            return
        if IS_WINDOWS:
            if not self.api.TerminateProcess(self.handle, 1) and self.alive():
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            try:
                signal.pidfd_send_signal(self.handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._kill_requested = True

    def wait(self, timeout=8):
        if not wait_until(lambda: not self.alive(), timeout):
            raise TimeoutError(f"owned process {self.pid} did not exit")

    def close(self):
        if self._closed:
            return
        try:
            self.kill()
            self.wait()
        finally:
            if IS_WINDOWS:
                self.api.CloseHandle(self.handle)
            else:
                os.close(self.handle)
            self._closed = True


def bounded_readline(stream, timeout=10):
    """Bound startup reporting; owner cleanup closes the pipe on failure."""
    result = queue.Queue(maxsize=1)

    def read():
        try:
            result.put(stream.readline(4096))
        except BaseException as exc:
            result.put(exc)

    threading.Thread(target=read, daemon=True).start()
    try:
        line = result.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError("owner did not report its identity in time") from exc
    if isinstance(line, BaseException):
        raise line
    if not line.endswith("\n"):
        raise ValueError(f"incomplete owner report: {line!r}")
    return json.loads(line)


def wait_until(predicate, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def heartbeat_stopped(marker: Path, quiet: float = 0.6, timeout: float = 8.0) -> bool:
    """True once the marker file has not grown for ``quiet`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        before = file_size(marker)
        time.sleep(quiet)
        if file_size(marker) == before:
            return True
    return False


def heartbeat_running(marker: Path, timeout: float = 5.0) -> bool:
    start = file_size(marker)
    return wait_until(lambda: file_size(marker) > start, timeout)


# --------------------------------------------------------------------------- #
# 1. ctypes layouts                                                           #
# --------------------------------------------------------------------------- #

class StructLayout(unittest.TestCase):
    def test_sizes_match_windows_sdk(self):
        expected = pg.EXPECTED_STRUCT_SIZES[ctypes.sizeof(ctypes.c_void_p)]
        for name, size in expected.items():
            with self.subTest(struct=name):
                self.assertEqual(ctypes.sizeof(getattr(pg, name)), size)

    def test_field_offsets(self):
        # Spot checks on the fields the guard actually writes/reads.
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("offsets listed for 64-bit builds")
        self.assertEqual(pg.STARTUPINFOW.hStdInput.offset, 80)
        self.assertEqual(pg.STARTUPINFOW.hStdOutput.offset, 88)
        self.assertEqual(pg.STARTUPINFOW.hStdError.offset, 96)
        self.assertEqual(pg.STARTUPINFOEXW.lpAttributeList.offset, 104)
        self.assertEqual(pg.PROCESS_INFORMATION.dwProcessId.offset, 16)
        self.assertEqual(pg.JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags.offset, 16)


# --------------------------------------------------------------------------- #
# 2. Windows orchestration against a fake kernel32                            #
# --------------------------------------------------------------------------- #

def read_environment_block(address: int) -> list[str]:
    """The entries of a CreateProcessW environment block at ``address`` (it ends with an empty entry)."""
    width = ctypes.sizeof(ctypes.c_wchar)
    entries = []
    while True:
        entry = ctypes.wstring_at(address)
        if not entry:
            return entries
        entries.append(entry)
        units = len(entry.encode("utf-16-le")) // 2 if width == 2 else len(entry)
        address += (units + 1) * width


class FakeKernel32:
    """Records every call, models handle lifetimes and job membership.

    Failure knobs: ``create_ok``, ``resume_ok``, ``init_probe_ok``, ``init_ok``,
    ``handle_list_supported``,
    ``job_list_supported`` (UpdateProcThreadAttribute(JOB_LIST) accepted),
    ``job_list_honoured`` (CreateProcess actually places the child in the job).
    """

    STD = {pg.STD_INPUT_HANDLE: 11, pg.STD_OUTPUT_HANDLE: 12, pg.STD_ERROR_HANDLE: 13}

    def __init__(self, *, create_ok=True, resume_ok=True, init_probe_ok=True,
                 init_ok=True, handle_list_supported=True,
                 job_list_supported=True, job_list_honoured=True):
        self.create_ok, self.resume_ok = create_ok, resume_ok
        self.init_probe_ok, self.init_ok = init_probe_ok, init_ok
        self.handle_list_supported = handle_list_supported
        self.job_list_supported, self.job_list_honoured = job_list_supported, job_list_honoured
        self.calls: list[tuple] = []
        self.open: set[int] = set()
        self.error = 0
        self.next_handle = 1000
        self.jobs: dict[int, set[int]] = {}          # job handle -> process handles
        self.processes: dict[int, dict] = {}         # process handle -> state
        self.attribute_lists: dict[int, dict] = {}   # buffer address -> attributes
        self.inherit_flags: dict[int, bool] = {}
        self.environments: list = []                 # per CreateProcessW: None, or the block's entries

    # -- bookkeeping ----------------------------------------------------------
    def _new(self) -> int:
        handle, self.next_handle = self.next_handle, self.next_handle + 1
        self.open.add(handle)
        return handle

    def _fail(self, code: int) -> int:
        self.error = code
        return 0

    def last_error(self) -> int:
        return self.error

    def get_osfhandle(self, fd: int) -> int:
        return 500 + fd

    def _kill_job(self, hjob: int):
        for hprocess in self.jobs[hjob]:
            self.processes[hprocess].update(alive=False, exit_code=pg.KILL_EXIT_STATUS)

    # -- job -------------------------------------------------------------------
    def CreateJobObjectW(self, attributes, name):
        self.calls.append(("CreateJobObjectW",))
        hjob = self._new()
        self.jobs[hjob] = set()
        self.inherit_flags[hjob] = False
        return hjob

    def SetInformationJobObject(self, hjob, klass, info, size):
        self.calls.append(("SetInformationJobObject", klass, info._obj.BasicLimitInformation.LimitFlags))
        return 1

    def SetHandleInformation(self, handle, mask, flags):
        self.calls.append(("SetHandleInformation", handle, mask, flags))
        self.inherit_flags[handle] = bool(flags & pg.HANDLE_FLAG_INHERIT)
        return 1

    def IsProcessInJob(self, hprocess, hjob, result):
        self.calls.append(("IsProcessInJob", hprocess, hjob))
        result._obj.value = int(hprocess in self.jobs[hjob])
        return 1

    def AssignProcessToJobObject(self, hjob, hprocess):
        self.calls.append(("AssignProcessToJobObject", hjob, hprocess))
        raise AssertionError("strict guard must never assign a job after creation")

    def TerminateJobObject(self, hjob, code):
        self.calls.append(("TerminateJobObject", hjob, code))
        self._kill_job(hjob)
        return 1

    # -- handles ---------------------------------------------------------------
    def GetCurrentProcess(self):
        return -1

    def GetStdHandle(self, which):
        return self.STD[which]

    def DuplicateHandle(self, sp, source, tp, target, access, inherit, options):
        self.calls.append(("DuplicateHandle", source, bool(inherit)))
        target._obj.value = self._new()
        self.inherit_flags[target._obj.value] = bool(inherit)
        return 1

    def CloseHandle(self, handle):
        self.calls.append(("CloseHandle", handle))
        assert handle in self.open, f"CloseHandle on unknown/closed handle {handle}"
        self.open.remove(handle)
        if handle in self.jobs:
            self._kill_job(handle)  # KILL_ON_JOB_CLOSE: last handle closes -> members die
        return 1

    # -- attribute list --------------------------------------------------------
    def InitializeProcThreadAttributeList(self, buffer, count, flags, size):
        if buffer is None:
            if not self.init_probe_ok:
                return self._fail(5)
            size._obj.value = 64
            return self._fail(pg.ERROR_INSUFFICIENT_BUFFER)
        if not self.init_ok:
            return self._fail(5)
        self.attribute_lists[ctypes.addressof(buffer)] = {}
        self.calls.append(("InitializeProcThreadAttributeList", count))
        return 1

    def UpdateProcThreadAttribute(self, buffer, flags, attribute, value, size, prev, ret):
        self.calls.append(("UpdateProcThreadAttribute", attribute, list(value)))
        if attribute == pg.PROC_THREAD_ATTRIBUTE_JOB_LIST and not self.job_list_supported:
            return self._fail(87)
        if attribute == pg.PROC_THREAD_ATTRIBUTE_HANDLE_LIST and not self.handle_list_supported:
            return self._fail(87)
        self.attribute_lists[ctypes.addressof(buffer)][attribute] = list(value)
        return 1

    def DeleteProcThreadAttributeList(self, buffer):
        self.calls.append(("DeleteProcThreadAttributeList",))
        del self.attribute_lists[ctypes.addressof(buffer)]

    # -- process ---------------------------------------------------------------
    def CreateProcessW(self, app, cmdline, pa, ta, inherit, flags, env, cwd, startup, info):
        self.calls.append(("CreateProcessW", cmdline.value, flags, cwd))
        self.environments.append(None if env is None else read_environment_block(env))
        if not self.create_ok:
            return self._fail(2)
        assert flags & pg.CREATE_SUSPENDED, "child must be created suspended"
        si = startup._obj.StartupInfo
        assert si.dwFlags & pg.STARTF_USESTDHANDLES
        for handle in (si.hStdInput, si.hStdOutput, si.hStdError):
            assert handle in self.open and self.inherit_flags[handle], "std handles must be live and inheritable"
        attributes = {}
        if flags & pg.EXTENDED_STARTUPINFO_PRESENT:
            attributes = self.attribute_lists[startup._obj.lpAttributeList]
        hprocess, hthread = self._new(), self._new()
        self.processes[hprocess] = {"alive": True, "running": False, "exit_code": None,
                                    "thread": hthread, "inherited": attributes.get(pg.PROC_THREAD_ATTRIBUTE_HANDLE_LIST)}
        for hjob in attributes.get(pg.PROC_THREAD_ATTRIBUTE_JOB_LIST, []):
            if self.job_list_honoured:
                self.jobs[hjob].add(hprocess)
        info._obj.hProcess, info._obj.hThread = hprocess, hthread
        info._obj.dwProcessId, info._obj.dwThreadId = 4242, 4243
        return 1

    def ResumeThread(self, hthread):
        self.calls.append(("ResumeThread", hthread))
        if not self.resume_ok:
            self.error = 6
            return pg.RESUME_FAILED
        for state in self.processes.values():
            if state["thread"] == hthread and state["alive"]:
                state["running"] = True
        return 1

    def TerminateProcess(self, hprocess, code):
        self.calls.append(("TerminateProcess", hprocess, code))
        state = self.processes[hprocess]
        if not state["alive"]:
            return self._fail(5)
        state.update(alive=False, exit_code=code)
        return 1

    def WaitForSingleObject(self, handle, millis):
        self.calls.append(("WaitForSingleObject", handle, millis))
        return pg.WAIT_TIMEOUT if self.processes[handle]["alive"] else pg.WAIT_OBJECT_0

    def GetExitCodeProcess(self, hprocess, code):
        state = self.processes[hprocess]
        code._obj.value = pg.STILL_ACTIVE if state["alive"] else state["exit_code"]
        return 1

    # -- test conveniences -----------------------------------------------------
    def exit_child(self, code: int):
        (state,) = self.processes.values()
        state.update(alive=False, exit_code=code)

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


class FakeKernel32Tests(unittest.TestCase):
    ARGS = ["C:\\Program Files\\Siril\\bin\\siril-cli.exe", "-s", "a b.ssf"]

    def start(self, fake: FakeKernel32):
        # stdout: a real file (its fd goes through get_osfhandle); stderr merged; stdin inherited
        with tempfile.TemporaryFile() as log:
            return pg._JobObjectProcess.start(
                self.ARGS, stdout=log, stderr=subprocess.STDOUT, stdin=None,
                cwd="C:\\job", creationflags=0x08000000, api=fake)

    def test_success_path_and_handle_lifecycle(self):
        fake = FakeKernel32()
        process = self.start(fake)
        names = fake.names()
        # job configured before the child exists, child created suspended, in job before resume
        self.assertLess(names.index("SetInformationJobObject"), names.index("CreateProcessW"))
        self.assertLess(names.index("CreateProcessW"), names.index("ResumeThread"))
        self.assertIn(("SetInformationJobObject", pg.JobObjectExtendedLimitInformation,
                       pg.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE), fake.calls)
        self.assertEqual(process.details, {"job_list": True, "handle_list": True, "assigned_after_create": False})
        self.assertNotIn("AssignProcessToJobObject", names)
        (state,) = fake.processes.values()
        self.assertTrue(state["running"])
        self.assertEqual(len(state["inherited"]), 2)  # stdin + shared stdout/stderr
        # command line quoting and flags passed through
        create = next(call for call in fake.calls if call[0] == "CreateProcessW")
        self.assertEqual(create[1], '"C:\\Program Files\\Siril\\bin\\siril-cli.exe" -s "a b.ssf"')
        self.assertEqual(create[2] & 0x08000000, 0x08000000)
        self.assertEqual(create[3], "C:\\job")
        # job handle explicitly non-inheritable; thread handle and std duplicates already closed
        self.assertFalse(fake.inherit_flags[process._hjob])
        self.assertEqual(fake.open, {process._hjob, process._hprocess})
        self.assertEqual(process.pid, 4242)
        self.assertTrue(process.in_job())
        self.assertIsNone(process.poll())
        with self.assertRaises(subprocess.TimeoutExpired):
            process.wait(timeout=0.01)
        fake.exit_child(7)
        self.assertEqual(process.poll(), 7)
        self.assertEqual(process.wait(), 7)
        process.terminate()  # already finished: no TerminateProcess call
        self.assertNotIn("TerminateProcess", fake.names())
        hjob, hprocess = process._hjob, process._hprocess
        process.close()
        self.assertEqual(fake.open, set())
        closes = [call[1] for call in fake.calls if call[0] == "CloseHandle"]
        self.assertLess(closes.index(hjob), closes.index(hprocess))  # job handle first
        process.close()  # idempotent
        self.assertEqual(process.returncode, 7)

    def test_create_process_failure_releases_everything(self):
        fake = FakeKernel32(create_ok=False)
        with self.assertRaises(OSError) as raised:
            self.start(fake)
        self.assertNotIsInstance(raised.exception, pg.ProcessGuardError)
        self.assertEqual(raised.exception.filename, self.ARGS[0])
        self.assertEqual(fake.open, set())  # job, std duplicates all closed
        self.assertEqual(fake.attribute_lists, {})
        self.assertEqual(fake.processes, {})

    def test_job_list_unsupported_fails_before_spawn(self):
        fake = FakeKernel32(job_list_supported=False)
        with self.assertRaises(pg.ProcessGuardError):
            self.start(fake)
        self.assertNotIn("CreateProcessW", fake.names())
        self.assertNotIn("AssignProcessToJobObject", fake.names())
        self.assertEqual(fake.processes, {})
        self.assertEqual(fake.attribute_lists, {})
        self.assertEqual(fake.open, set())

    def test_job_list_ignored_kills_suspended_child_without_fallback(self):
        fake = FakeKernel32(job_list_honoured=False)
        with self.assertRaises(pg.ProcessGuardError):
            self.start(fake)
        (state,) = fake.processes.values()
        self.assertFalse(state["running"], "child must never have been resumed")
        self.assertFalse(state["alive"])
        names = fake.names()
        self.assertNotIn("AssignProcessToJobObject", names)
        self.assertNotIn("ResumeThread", names)
        self.assertLess(names.index("TerminateProcess"), names.index("WaitForSingleObject"))
        self.assertEqual(fake.open, set())

    def test_handle_list_unsupported_fails_before_spawn(self):
        fake = FakeKernel32(handle_list_supported=False)
        with self.assertRaises(pg.ProcessGuardError):
            self.start(fake)
        self.assertNotIn("CreateProcessW", fake.names())
        self.assertEqual(fake.processes, {})
        self.assertEqual(fake.attribute_lists, {})
        self.assertEqual(fake.open, set())

    def test_attribute_initialization_failure_prevents_spawn(self):
        for options in [{"init_probe_ok": False}, {"init_ok": False}]:
            with self.subTest(options=options):
                fake = FakeKernel32(**options)
                with self.assertRaises(pg.ProcessGuardError):
                    self.start(fake)
                self.assertNotIn("CreateProcessW", fake.names())
                self.assertEqual(fake.processes, {})
                self.assertEqual(fake.attribute_lists, {})
                self.assertEqual(fake.open, set())

    def test_resume_failure_kills_child(self):
        fake = FakeKernel32(resume_ok=False)
        with self.assertRaises(pg.ProcessGuardError):
            self.start(fake)
        (state,) = fake.processes.values()
        self.assertFalse(state["alive"])
        self.assertEqual(fake.open, set())

    def test_close_kills_running_job_then_releases(self):
        fake = FakeKernel32()
        process = self.start(fake)
        hjob, hprocess = process._hjob, process._hprocess
        process.close()
        names = fake.names()
        self.assertIn("TerminateJobObject", names)
        closes = [call[1] for call in fake.calls if call[0] == "CloseHandle"]
        self.assertLess(closes.index(hjob), closes.index(hprocess))
        self.assertEqual(process.returncode, pg.KILL_EXIT_STATUS)
        self.assertEqual(fake.open, set())
        self.assertFalse(process.in_job())

    def test_terminate_hits_child_only_kill_hits_job(self):
        fake = FakeKernel32()
        process = self.start(fake)
        process.terminate()
        self.assertIn(("TerminateProcess", process._hprocess, pg.KILL_EXIT_STATUS), fake.calls)
        self.assertEqual(process.wait(timeout=1), pg.KILL_EXIT_STATUS)
        process.kill()
        self.assertIn(("TerminateJobObject", process._hjob, pg.KILL_EXIT_STATUS), fake.calls)
        process.close()

    def test_devnull_and_inherit_sources(self):
        fake = FakeKernel32()
        process = pg._JobObjectProcess.start(
            self.ARGS, stdout=subprocess.DEVNULL, stderr=None, stdin=subprocess.DEVNULL,
            cwd=None, creationflags=0, api=fake)
        (state,) = fake.processes.values()
        self.assertEqual(len(state["inherited"]), 3)
        # stderr=None inherits the owner's STD_ERROR_HANDLE (fake handle 13) via a duplicate
        self.assertIn(("DuplicateHandle", 13, True), fake.calls)
        process.close()
        self.assertEqual(fake.open, set())

    def test_pipe_and_bad_stdout_spec_rejected(self):
        with self.assertRaises(ValueError):
            with pg.guarded_process(SLEEP_CHILD, stdout=subprocess.PIPE):
                pass
        with self.assertRaises(ValueError):
            pg._JobObjectProcess.start(self.ARGS, stdout=subprocess.STDOUT, stderr=None,
                                       stdin=None, cwd=None, creationflags=0, api=FakeKernel32())

    # -- env (efficiency P1a): None inherits exactly as before; a mapping is the child's whole block --
    def test_no_env_passes_no_block_and_the_same_flags(self):
        fake = FakeKernel32()
        process = self.start(fake)
        create = next(call for call in fake.calls if call[0] == "CreateProcessW")
        self.assertEqual(create[2], 0x08000000 | pg.CREATE_SUSPENDED | pg.EXTENDED_STARTUPINFO_PRESENT)
        self.assertEqual(fake.environments, [None])
        process.close()
        self.assertEqual(fake.open, set())

    def test_env_block_is_sorted_unicode_and_flagged(self):
        fake = FakeKernel32()
        env = {"b": "2", "Path": "C:\\x y", "A": "星空 缓存 \U0001F320", "=C:": "C:\\"}
        with tempfile.TemporaryFile() as log:
            process = pg._JobObjectProcess.start(
                self.ARGS, stdout=log, stderr=subprocess.STDOUT, stdin=None,
                cwd="C:\\job", creationflags=0x08000000, api=fake, env=env)
        create = next(call for call in fake.calls if call[0] == "CreateProcessW")
        self.assertEqual(create[2], 0x08000000 | pg.CREATE_SUSPENDED | pg.EXTENDED_STARTUPINFO_PRESENT
                         | pg.CREATE_UNICODE_ENVIRONMENT)
        self.assertEqual(fake.environments, [["=C:=C:\\", "A=星空 缓存 \U0001F320", "b=2", "Path=C:\\x y"]])
        process.close()
        self.assertEqual(fake.open, set())

    def test_illegal_env_is_refused_before_any_handle_exists(self):
        for env in ({"": "1"}, {"A=B": "1"}, {"A\0": "1"}, {"A": "x\0y"}):
            with self.subTest(env=env):
                fake = FakeKernel32()
                with self.assertRaises(ValueError):
                    pg._JobObjectProcess.start(self.ARGS, stdout=subprocess.DEVNULL, stderr=None, stdin=None,
                                               cwd=None, creationflags=0, api=fake, env=env)
                self.assertEqual(fake.calls, [])
                self.assertEqual(fake.open, set())


class EnvironmentBlockTests(unittest.TestCase):
    def test_block_format(self):
        self.assertIsNone(pg._environment_block(None))
        self.assertEqual(pg._environment_block({})[:], "\0\0")
        self.assertEqual(pg._environment_block({"b": "2", "A": "1", "c": ""})[:], "A=1\0b=2\0c=\0\0\0")
        block = pg._environment_block({"Z": "1", "y": "2"})   # keep the buffer alive while it is read
        self.assertEqual(read_environment_block(ctypes.addressof(block)), ["y=2", "Z=1"])

    def test_popen_path_passes_env_only_when_given(self):
        with mock.patch.object(pg.subprocess, "Popen") as popen:
            popen.return_value.pid, popen.return_value.args = 1, ["x"]
            pg._PopenProcess.start(["x"], stdout=None, stderr=None, stdin=None, cwd=None, pdeathsig=False)
            self.assertNotIn("env", popen.call_args.kwargs)
            source = {"A": "1"}
            pg._PopenProcess.start(["x"], stdout=None, stderr=None, stdin=None, cwd=None, pdeathsig=False,
                                   env=source)
            self.assertEqual(popen.call_args.kwargs["env"], {"A": "1"})
            self.assertIsNot(popen.call_args.kwargs["env"], source)


# --------------------------------------------------------------------------- #
# 3. real processes under the native guard                                    #
# --------------------------------------------------------------------------- #

@unittest.skipIf(NATIVE_GUARD == pg.GUARD_NONE, "no guard on this platform")
class RealProcessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # unittest cleanups are LIFO; remove the directory after all processes
        # and their marker/log handles have been stopped and closed.
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.env = dict(os.environ, PYTHONPATH=str(SRC))

    def test_success_collects_output_and_returncode(self):
        log = self.tmp / "run.log"
        child = py("import sys", "print('out', flush=True)",
                   "print('err', file=sys.stderr, flush=True)", "sys.exit(3)")
        with log.open("ab", buffering=0) as output:
            with pg.guarded_process(child, stdout=output, stderr=subprocess.STDOUT, cwd=self.tmp) as process:
                self.assertEqual(process.guard, NATIVE_GUARD)
                self.assertGreater(process.pid, 0)
                process.wait(timeout=10)
                self.assertEqual(process.returncode, 3)
                self.assertEqual(process.wait(), 3)
                process.terminate()  # no-op after exit
                process.kill()
        text = log.read_text()
        self.assertIn("out", text)
        self.assertIn("err", text)
        self.assertFalse(pid_alive(process.pid))

    def test_cancel_terminate_then_wait(self):
        with pg.guarded_process(SLEEP_CHILD, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) as process:
            with self.assertRaises(subprocess.TimeoutExpired):
                process.wait(timeout=0.2)
            started = time.monotonic()
            process.terminate()
            process.wait(timeout=8)
            self.assertLess(time.monotonic() - started, 8)
            self.assertIsNotNone(process.returncode)
            self.assertEqual(process.returncode, 1 if IS_WINDOWS else -signal.SIGTERM)
        self.assertFalse(pid_alive(process.pid))

    def test_kill_then_wait(self):
        with pg.guarded_process(SLEEP_CHILD, stdout=subprocess.DEVNULL) as process:
            process.kill()
            process.wait(timeout=8)
            self.assertEqual(process.returncode, 1 if IS_WINDOWS else -signal.SIGKILL)

    def test_leaving_block_kills_running_child(self):
        marker = self.tmp / "beat"
        try:
            with pg.guarded_process([sys.executable, "-c", HEARTBEAT, str(marker)],
                                    stdout=subprocess.DEVNULL) as process:
                self.assertTrue(heartbeat_running(marker))
                raise RuntimeError("simulated failure inside the with block")
        except RuntimeError:
            pass
        self.assertIsNotNone(process.returncode)
        self.assertTrue(heartbeat_stopped(marker))
        if IS_WINDOWS:
            self.assertFalse(pid_alive(process.pid))

    def test_env_reaches_the_child_and_this_process_is_unchanged(self):
        out = self.tmp / "env.json"
        names = ("AESTER_PG_OVERRIDE", "AESTER_PG_INHERITED", "AESTER_PG_REMOVED")
        child = py("import json, os, sys",
                   f"json.dump({{k: os.environ.get(k) for k in {names!r}}}, open(sys.argv[1], 'w', encoding='utf-8'),"
                   " ensure_ascii=False)") + [str(out)]
        with mock.patch.dict(os.environ, {"AESTER_PG_INHERITED": "kept", "AESTER_PG_REMOVED": "gone"}):
            before = dict(os.environ)
            env = {k: v for k, v in os.environ.items() if k != "AESTER_PG_REMOVED"}
            env["AESTER_PG_OVERRIDE"] = "星空 缓存 \U0001F320"
            with pg.guarded_process(child, stdout=subprocess.DEVNULL, env=env) as process:
                self.assertEqual(process.wait(timeout=60), 0)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8")),
                         {"AESTER_PG_OVERRIDE": "星空 缓存 \U0001F320", "AESTER_PG_INHERITED": "kept",
                          "AESTER_PG_REMOVED": None})

    def test_missing_executable_raises_and_leaves_nothing(self):
        missing = str(self.tmp / "no-such-siril-cli.exe")
        with self.assertRaises(FileNotFoundError):
            with pg.guarded_process([missing, "-v"], stdout=subprocess.DEVNULL):
                self.fail("body must not run")

    def test_missing_cwd_raises(self):
        with self.assertRaises(OSError):
            with pg.guarded_process(SLEEP_CHILD, stdout=subprocess.DEVNULL, cwd=self.tmp / "absent"):
                self.fail("body must not run")

    @unittest.skipUnless(NATIVE_GUARD == pg.GUARD_PDEATHSIG, "Linux preexec path")
    def test_linux_preexec_failure_leaves_no_child(self):
        marker = self.tmp / "beat"
        failing = mock.Mock(return_value=-1)
        with mock.patch.object(pg, "_prctl_function", failing):
            with self.assertRaises(subprocess.SubprocessError):
                with pg.guarded_process([sys.executable, "-c", HEARTBEAT, str(marker)], stdout=subprocess.DEVNULL):
                    self.fail("body must not run")
        time.sleep(0.3)
        self.assertFalse(marker.exists(), "child must not have run")

    def test_unguarded_platform_refuses_unless_allowed(self):
        with mock.patch.object(pg, "guard_kind", return_value=pg.GUARD_NONE):
            with self.assertRaises(pg.ProcessGuardError):
                with pg.guarded_process(SLEEP_CHILD, stdout=subprocess.DEVNULL):
                    self.fail("body must not run")
            with pg.guarded_process(py("pass"), stdout=subprocess.DEVNULL, allow_unguarded=True) as process:
                self.assertEqual(process.guard, pg.GUARD_NONE)
                self.assertEqual(process.wait(timeout=30), 0)

    # -- owner force-killed ------------------------------------------------------

    def run_owner(self, mode: str, payload: str, marker: Path):
        if not IS_WINDOWS and not (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")):
            self.skipTest("safe owner-death tests require Linux pidfds")
        launcher = subprocess.Popen([sys.executable, "-c", OWNER, mode, str(marker), payload],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, cwd=self.tmp, env=self.env,
                                    text=True, **({"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}))
        owned = []
        cleaned = False

        def finish():
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            errors = []
            # This signal affects only the snippets writing this test's marker.
            # It also bounds cleanup if startup failed before a child report.
            try:
                marker.with_name(marker.name + ".stop").touch()
            except OSError as exc:
                errors.append(exc)
            try:
                launcher.stdin.close()
            except BrokenPipeError:
                pass  # an already-dead owner may have closed its reader
            except OSError as exc:
                errors.append(exc)
            if owned:
                try:
                    owned[0].wait(timeout=2)
                except TimeoutError:
                    pass
                except Exception as exc:
                    errors.append(exc)
            for process in reversed(owned):
                try:
                    process.close()
                except Exception as exc:
                    errors.append(exc)
            try:
                if launcher.poll() is None:
                    launcher.kill()  # Popen retains its own exact process handle.
                launcher.wait(timeout=8)
            except Exception as exc:
                errors.append(exc)
            finally:
                for stream in (launcher.stdout, launcher.stderr):
                    try:
                        stream.close()
                    except OSError as exc:
                        errors.append(exc)
            if errors:
                raise errors[0]

        # Register before reading, parsing, opening handles or asserting.
        self.addCleanup(finish)

        def retain(reported):
            process = OwnedProcess(reported)
            owned.append(process)
            return process

        owner = retain(bounded_readline(launcher.stdout))
        launcher.stdin.write("start\n")
        launcher.stdin.flush()
        child = retain(bounded_readline(launcher.stdout))
        grandchild = None
        if payload == "parent":
            owner_log = marker.with_name(marker.name + ".owner.log")

            def report_ready():
                return owner_log.exists() and "\n" in owner_log.read_text()

            if not wait_until(report_ready, 10):
                self.fail("grandchild identity not reported")
            grandchild = retain(json.loads(owner_log.read_text().splitlines()[0]))
        self.assertTrue(heartbeat_running(marker), "payload heartbeat never started")
        self.assertTrue(owner.alive())
        return owner, child, grandchild

    def force_kill(self, owner: OwnedProcess):
        # Kill the identity reported by os.getpid(), not a venv redirector.
        # TerminateProcess / pidfd SIGKILL prevents OWNER's finally from running.
        owner.kill()
        owner.wait(timeout=10)

    def test_owner_force_killed_child_dies(self):
        marker = self.tmp / "beat"
        owner, child, _ = self.run_owner("guarded", "direct", marker)
        self.force_kill(owner)
        self.assertTrue(heartbeat_stopped(marker), "guarded child kept running after owner death")
        self.assertTrue(wait_until(lambda: not child.alive(), 5), "child still alive")

    def test_negative_control_plain_popen_child_survives(self):
        """Proves the detector: without the guard the child outlives its owner."""
        marker = self.tmp / "beat"
        owner, child, _ = self.run_owner("plain", "direct", marker)
        self.force_kill(owner)
        self.assertTrue(heartbeat_running(marker), "unguarded child unexpectedly died with owner")
        self.assertTrue(child.alive())
        child.kill()
        child.wait()
        self.assertTrue(heartbeat_stopped(marker))

    def test_owner_force_killed_grandchild(self):
        """Windows: the job takes the grandchild too. Linux: documented gap."""
        marker = self.tmp / "beat"
        owner, child, grandchild = self.run_owner("guarded", "parent", marker)
        self.force_kill(owner)
        if NATIVE_GUARD == pg.GUARD_JOB_OBJECT:
            self.assertTrue(heartbeat_stopped(marker), "grandchild survived owner death on Windows")
            self.assertTrue(wait_until(lambda: not child.alive(), 5))
            self.assertTrue(wait_until(lambda: not grandchild.alive(), 5))
        else:
            # pdeathsig reaches only the direct child; the grandchild is orphaned.
            self.assertTrue(wait_until(lambda: not child.alive(), 5), "direct child should have died")
            self.assertTrue(heartbeat_running(marker), "expected documented Linux gap: grandchild survives")
            grandchild.kill()
            grandchild.wait()
            self.assertTrue(heartbeat_stopped(marker))

    @unittest.skipUnless(NATIVE_GUARD == pg.GUARD_JOB_OBJECT, "Windows job object only")
    def test_windows_child_is_in_job_and_details(self):
        with pg.guarded_process(SLEEP_CHILD, stdout=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW) as process:
            self.assertTrue(process.in_job())
            self.assertTrue(process.details["handle_list"])
            self.assertTrue(process.details["job_list"])
            self.assertFalse(process.details["assigned_after_create"])
            process.kill()
            process.wait(timeout=8)
        self.assertFalse(process.in_job())


if __name__ == "__main__":
    unittest.main(verbosity=2)
