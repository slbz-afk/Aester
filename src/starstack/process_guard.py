"""Keep a long-running child process from outliving the worker that started it.

Why this exists
---------------
``siril.run_script`` starts ``siril-cli`` with ``subprocess.Popen``.  Normal
cancellation is orderly (terminate -> wait -> kill -> wait), but when the
*worker* is killed from outside (task manager, crash, a GUI that gives up),
nothing tears Siril down: it keeps writing into the job directory while a
recovery run may already be writing to the same paths.

This module starts the child so that the operating system, not Python code,
ends it when the owner process disappears.

What is guaranteed, per platform  (see ``guard_kind()``)
--------------------------------------------------------
Windows -- ``"windows_job_object"``
    The child is created *suspended* and placed in a fresh, anonymous Job
    Object configured with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` before its
    first instruction runs; only then is its initial thread resumed.  The
    owner holds the single, non-inheritable handle to the job.  When the
    owner exits for any reason (clean exit, unhandled exception,
    TerminateProcess from outside, crash), the kernel closes that handle,
    the job's handle count drops to zero and the kernel terminates every
    process still in the job: the child and all of its descendants (they
    inherit job membership at creation and cannot break away because the
    job grants no breakaway).

    This application requires atomic assignment through
    ``PROC_THREAD_ATTRIBUTE_JOB_LIST`` and an explicit inherited-handle list.
    Attribute setup failures abort before CreateProcess; there is no legacy
    fallback.  After creation, job membership is checked before resuming the
    child.  This path targets current Windows 10/11 desktop installations.

    Not covered, by any user-mode mechanism: a process the kernel cannot
    terminate (wedged in a driver call), and an owner that is *suspended*
    rather than terminated (a suspended owner still holds the handle).

    The owner is the process executing this context manager. A Windows venv
    launcher can introduce another process, so a launcher's Popen.pid is not
    necessarily that owner. Tests and recovery code must establish identity.

Linux -- ``"linux_pdeathsig"``
    Best-effort fallback so the same code path can be exercised off Windows.
    It has not been accepted as a production protection path for this app.
    The child requests ``SIGKILL`` on the death of its parent thread
    (``prctl(PR_SET_PDEATHSIG)``) and exits at once if the parent already
    died before the request was registered (closing the fork -> prctl
    window).  Scope: only the *direct* child, only on Linux, and only while
    the thread that called ``guarded_process`` is alive -- call it from a
    thread that outlives the child (the worker's main thread).
    Grandchildren are not covered.

Anything else -- ``"none"``
    No owner-death protection is available from the standard library.
    ``guarded_process`` refuses to start the child unless the caller passes
    ``allow_unguarded=True`` explicitly.

Non-goals
---------
This module acts only on the process it created (and, on Windows, on that
process's descendants through the job).  It never looks processes up by
name and never touches a Siril started by anything else.

Interface
---------
``guarded_process(args, *, stdout=None, stderr=None, stdin=None, cwd=None,
                  creationflags=0, allow_unguarded=False)``
    Context manager.  Yields an object with the subset of the ``Popen``
    interface the adapter's poll/cancel loop uses:

    ``pid``, ``args``, ``returncode``, ``guard``, ``details``,
    ``poll()``, ``wait(timeout=None)`` (raises ``subprocess.TimeoutExpired``),
    ``terminate()``, ``kill()``.

    Leaving the ``with`` block requests termination if needed, waits a
    bounded time for the direct child, and releases all handles.  On Windows
    closing the job also requests termination of remaining descendants.
    Termination and I/O cancellation can complete asynchronously: return from
    the block does not itself prove every descendant has fully exited.

    After the block, ``process.accounting`` holds the Windows Job Object
    accounting of the child and its descendants, read just before the job
    handle was closed (user/kernel CPU seconds, I/O transfer bytes, peak job
    and per-process committed memory, wall seconds), or ``None`` when it is
    unavailable (other platforms, a test double, a failed query).  Reading it
    never raises and never changes how the child is started or ended.
    ``add_accounting_listener(callback)`` receives each such dict as well
    (diagnostics only; a failing callback is ignored).

``guard_kind()``  -> which mechanism this platform provides.
``ProcessGuardError`` (``OSError``) -> the guard could not be established;
    startup cleanup requests termination of any created child and closes the
    job.  The bounded wait cannot guarantee completion of stalled kernel I/O.

Only the standard library is used.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import sys
import time
from typing import Any

__all__ = [
    "GUARD_JOB_OBJECT",
    "GUARD_NONE",
    "GUARD_PDEATHSIG",
    "ProcessGuardError",
    "add_accounting_listener",
    "guard_kind",
    "guarded_process",
]

GUARD_JOB_OBJECT = "windows_job_object"
GUARD_PDEATHSIG = "linux_pdeathsig"
GUARD_NONE = "none"

# How long leaving the ``with`` block waits for a killed child before releasing
# handles anyway.  On Windows the job handle close re-kills whatever is left.
CLOSE_KILL_TIMEOUT = 10.0

# Exit status used by the Linux fallback when the child discovers, before exec,
# that its owner is already gone.
ORPHANED_BEFORE_GUARD_STATUS = 122


_ACCOUNTING_LISTENERS: list = []


def add_accounting_listener(callback) -> None:
    """Also hand every ended child's accounting dict to ``callback`` (diagnostics only)."""
    if callback not in _ACCOUNTING_LISTENERS:
        _ACCOUNTING_LISTENERS.append(callback)


def _publish_accounting(accounting) -> None:
    if accounting is None:
        return
    for callback in list(_ACCOUNTING_LISTENERS):
        try:
            callback(dict(accounting))
        except Exception:
            pass  # accounting is diagnostic; it never affects the child's lifetime


class ProcessGuardError(OSError):
    """The child could not be placed under the guard (or the guard is unavailable).

    Guard setup errors never deliberately resume an unprotected child.
    ``code`` carries the Windows error code (0 when not applicable).
    """

    def __init__(self, message: str, code: int = 0):
        # OSError(errno, strerror, filename, winerror): on Windows the 4th
        # argument also populates ``winerror``; elsewhere it is ignored.
        super().__init__(0, message, None, code)
        self.code = code

    def __str__(self) -> str:  # keep the message readable in logs
        return self.args[1] if len(self.args) > 1 else super().__str__()


def guard_kind() -> str:
    """Return the mechanism ``guarded_process`` will use on this platform."""
    if os.name == "nt":
        return GUARD_JOB_OBJECT
    if sys.platform.startswith("linux"):
        try:
            _linux_prctl()
        except (OSError, AttributeError):
            return GUARD_NONE
        return GUARD_PDEATHSIG
    return GUARD_NONE


@contextlib.contextmanager
def guarded_process(args, *, stdout=None, stderr=None, stdin=None, cwd=None,
                    creationflags: int = 0, allow_unguarded: bool = False, env=None):
    """Start ``args`` under the platform guard and yield a Popen-like object.

    ``stdout``/``stderr``/``stdin`` accept what ``run_script`` uses: ``None``
    (inherit), an open file object or file descriptor, ``subprocess.DEVNULL``,
    and ``subprocess.STDOUT`` for ``stderr``.  ``subprocess.PIPE`` is not
    supported -- the guard does not manage pipes.  ``creationflags`` is
    honoured on Windows and ignored elsewhere.  ``env``: ``None`` inherits
    this process's environment (unchanged behaviour); a mapping is the
    child's complete environment, as for ``subprocess.Popen(env=...)``.
    This process's own environment is never changed.

    Raises ``ProcessGuardError`` when the guard cannot be established, and
    ordinary ``OSError`` (e.g. ``FileNotFoundError``) when
    the executable cannot be started, exactly like ``Popen``.
    """
    args = [os.fspath(a) for a in args]
    if not args:
        raise ValueError("args must name an executable")
    if any(_is_marker(spec, subprocess.PIPE) for spec in (stdin, stdout, stderr)):
        raise ValueError("subprocess.PIPE is not supported by guarded_process")
    kind = guard_kind()
    if kind == GUARD_NONE and not allow_unguarded:
        raise ProcessGuardError(
            f"no owner-death guard is available on {sys.platform}; "
            "pass allow_unguarded=True to run the child unprotected")
    if kind == GUARD_JOB_OBJECT:
        process: Any = _JobObjectProcess.start(
            args, stdout=stdout, stderr=stderr, stdin=stdin, cwd=cwd,
            creationflags=creationflags, env=env)
    else:
        process = _PopenProcess.start(
            args, stdout=stdout, stderr=stderr, stdin=stdin, cwd=cwd,
            pdeathsig=(kind == GUARD_PDEATHSIG), env=env)
    try:
        yield process
    finally:
        process.close()


def _is_marker(spec, marker: int) -> bool:
    """True when ``spec`` is the subprocess constant ``marker`` (PIPE/STDOUT/DEVNULL)."""
    return isinstance(spec, int) and not isinstance(spec, bool) and spec == marker


# --------------------------------------------------------------------------- #
# POSIX path: plain Popen, plus PR_SET_PDEATHSIG on Linux                      #
# --------------------------------------------------------------------------- #

_PR_SET_PDEATHSIG = 1
_SIGKILL = 9
_prctl_function = None


def _linux_prctl():
    """Resolve libc ``prctl`` once, in the parent, so the child only calls it."""
    global _prctl_function
    if _prctl_function is None:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.prctl  # AttributeError on libcs without prctl
        function.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                             ctypes.c_ulong, ctypes.c_ulong]
        function.restype = ctypes.c_int
        _prctl_function = function
    return _prctl_function


def _pdeathsig_preexec(owner_pid: int):
    """Build the ``preexec_fn`` that arms the parent-death signal in the child."""
    prctl = _linux_prctl()

    def preexec() -> None:
        # Runs in the child between fork() and exec(): two syscalls, no locks.
        if prctl(_PR_SET_PDEATHSIG, _SIGKILL, 0, 0, 0) != 0:
            # Popen reports this as SubprocessError and the child never execs.
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != owner_pid:
            # The owner died between fork() and prctl(): the death signal
            # will never come, so do not start the payload at all.
            os._exit(ORPHANED_BEFORE_GUARD_STATUS)

    return preexec


class _PopenProcess:
    """Popen-backed guarded process (Linux pdeathsig, or explicitly unguarded)."""

    @classmethod
    def start(cls, args, *, stdout, stderr, stdin, cwd, pdeathsig: bool, env=None):
        extra = {} if env is None else {"env": dict(env)}
        if pdeathsig:
            extra["preexec_fn"] = _pdeathsig_preexec(os.getpid())
        popen = subprocess.Popen(args, stdin=stdin, stdout=stdout, stderr=stderr,
                                 cwd=cwd, **extra)
        return cls(popen, GUARD_PDEATHSIG if pdeathsig else GUARD_NONE)

    def __init__(self, popen: subprocess.Popen, guard: str):
        self._popen = popen
        self._closed = False
        self.guard = guard
        self.details = {"pdeathsig": guard == GUARD_PDEATHSIG}
        self.accounting = None  # no job accounting off Windows
        self.pid = popen.pid
        self.args = list(popen.args)

    @property
    def returncode(self):
        return self._popen.returncode

    def poll(self):
        return self._popen.poll()

    def wait(self, timeout=None):
        return self._popen.wait(timeout=timeout)

    def terminate(self) -> None:
        self._popen.terminate()  # Popen skips the signal if already finished

    def kill(self) -> None:
        self._popen.kill()

    def close(self) -> None:
        """Kill the child if it is still running; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._popen.poll() is None:
            self._popen.kill()
            try:
                self._popen.wait(timeout=CLOSE_KILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass  # only a process stuck in the kernel survives SIGKILL


# --------------------------------------------------------------------------- #
# Windows path: Job Object with KILL_ON_JOB_CLOSE                              #
# --------------------------------------------------------------------------- #
#
# The structures and the orchestration below are defined on every platform so
# that the orchestration (call order, cleanup on each failure path, handle
# ownership) can be unit-tested against a fake kernel32 anywhere; only
# ``_kernel32()`` needs Windows.  Fixed-width ctypes aliases are used instead
# of ``ctypes.wintypes`` so that the struct layouts computed off-Windows are
# the real 64-bit Windows layouts (``wintypes.DWORD`` is 8 bytes on Linux).

BOOL = ctypes.c_int32
DWORD = ctypes.c_uint32
WORD = ctypes.c_uint16
HANDLE = ctypes.c_void_p
LPVOID = ctypes.c_void_p
LPBYTE = ctypes.c_void_p
LPWSTR = ctypes.c_wchar_p
LPCWSTR = ctypes.c_wchar_p
SIZE_T = ctypes.c_size_t
ULONG_PTR = ctypes.c_size_t
LARGE_INTEGER = ctypes.c_int64
ULONGLONG = ctypes.c_uint64

INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
STARTF_USESTDHANDLES = 0x00000100
STD_INPUT_HANDLE = 0xFFFFFFF6   # (DWORD)-10
STD_OUTPUT_HANDLE = 0xFFFFFFF5  # (DWORD)-11
STD_ERROR_HANDLE = 0xFFFFFFF4   # (DWORD)-12
DUPLICATE_SAME_ACCESS = 0x00000002
HANDLE_FLAG_INHERIT = 0x00000001
JobObjectBasicAndIoAccountingInformation = 8
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002  # ProcThreadAttributeHandleList | INPUT
PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D     # ProcThreadAttributeJobList | INPUT
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF
INFINITE = 0xFFFFFFFF
STILL_ACTIVE = 259
ERROR_INSUFFICIENT_BUFFER = 122
RESUME_FAILED = 0xFFFFFFFF  # ResumeThread returns (DWORD)-1 on failure
# Exit status handed to TerminateProcess/TerminateJobObject; same value Popen uses.
KILL_EXIT_STATUS = 1


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD),
        ("lpReserved", LPWSTR),
        ("lpDesktop", LPWSTR),
        ("lpTitle", LPWSTR),
        ("dwX", DWORD),
        ("dwY", DWORD),
        ("dwXSize", DWORD),
        ("dwYSize", DWORD),
        ("dwXCountChars", DWORD),
        ("dwYCountChars", DWORD),
        ("dwFillAttribute", DWORD),
        ("dwFlags", DWORD),
        ("wShowWindow", WORD),
        ("cbReserved2", WORD),
        ("lpReserved2", LPBYTE),
        ("hStdInput", HANDLE),
        ("hStdOutput", HANDLE),
        ("hStdError", HANDLE),
    ]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", STARTUPINFOW),
        ("lpAttributeList", LPVOID),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", HANDLE),
        ("hThread", HANDLE),
        ("dwProcessId", DWORD),
        ("dwThreadId", DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ULONGLONG),
        ("WriteOperationCount", ULONGLONG),
        ("OtherOperationCount", ULONGLONG),
        ("ReadTransferCount", ULONGLONG),
        ("WriteTransferCount", ULONGLONG),
        ("OtherTransferCount", ULONGLONG),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", LARGE_INTEGER),
        ("PerJobUserTimeLimit", LARGE_INTEGER),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", ULONG_PTR),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", LARGE_INTEGER),
        ("TotalKernelTime", LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", LARGE_INTEGER),
        ("TotalPageFaultCount", DWORD),
        ("TotalProcesses", DWORD),
        ("ActiveProcesses", DWORD),
        ("TotalTerminatedProcesses", DWORD),
    ]


class JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicInfo", JOBOBJECT_BASIC_ACCOUNTING_INFORMATION),
        ("IoInfo", IO_COUNTERS),
    ]


# Known sizeof() values from the Windows SDK, keyed by pointer width; the test
# suite checks the definitions above against them on any platform.
EXPECTED_STRUCT_SIZES = {
    8: {"STARTUPINFOW": 104, "STARTUPINFOEXW": 112, "PROCESS_INFORMATION": 24,
        "JOBOBJECT_EXTENDED_LIMIT_INFORMATION": 144,
        "JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION": 96},
    4: {"STARTUPINFOW": 68, "STARTUPINFOEXW": 72, "PROCESS_INFORMATION": 16,
        "JOBOBJECT_EXTENDED_LIMIT_INFORMATION": 112,
        "JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION": 96},
}


class _Kernel32:
    """The exact set of kernel32 entry points the guard uses, with argtypes set.

    A test double only needs to provide these same attributes.  Every
    function keeps its Win32 signature; ``last_error()`` returns the error
    code of the most recent failing call and ``get_osfhandle(fd)`` maps a CRT
    file descriptor to its OS handle.
    """

    def __init__(self):
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        def bind(name, restype, *argtypes):
            function = getattr(kernel, name)
            function.restype = restype
            function.argtypes = list(argtypes)
            return function

        self.CreateJobObjectW = bind("CreateJobObjectW", HANDLE, LPVOID, LPCWSTR)
        self.SetInformationJobObject = bind("SetInformationJobObject", BOOL,
                                            HANDLE, ctypes.c_int, LPVOID, DWORD)
        self.AssignProcessToJobObject = bind("AssignProcessToJobObject", BOOL, HANDLE, HANDLE)
        self.IsProcessInJob = bind("IsProcessInJob", BOOL, HANDLE, HANDLE, ctypes.POINTER(BOOL))
        self.TerminateJobObject = bind("TerminateJobObject", BOOL, HANDLE, DWORD)
        self.SetHandleInformation = bind("SetHandleInformation", BOOL, HANDLE, DWORD, DWORD)
        self.GetCurrentProcess = bind("GetCurrentProcess", HANDLE)
        self.GetStdHandle = bind("GetStdHandle", HANDLE, DWORD)
        self.DuplicateHandle = bind("DuplicateHandle", BOOL, HANDLE, HANDLE, HANDLE,
                                    ctypes.POINTER(HANDLE), DWORD, BOOL, DWORD)
        self.CloseHandle = bind("CloseHandle", BOOL, HANDLE)
        self.InitializeProcThreadAttributeList = bind(
            "InitializeProcThreadAttributeList", BOOL, LPVOID, DWORD, DWORD, ctypes.POINTER(SIZE_T))
        self.UpdateProcThreadAttribute = bind(
            "UpdateProcThreadAttribute", BOOL, LPVOID, DWORD, ULONG_PTR, LPVOID, SIZE_T, LPVOID, LPVOID)
        self.DeleteProcThreadAttributeList = bind("DeleteProcThreadAttributeList", None, LPVOID)
        # lpCommandLine is LPVOID rather than LPWSTR on purpose: CreateProcessW may
        # write into it, so a mutable unicode buffer is passed, never a str.
        self.CreateProcessW = bind("CreateProcessW", BOOL, LPCWSTR, LPVOID, LPVOID, LPVOID,
                                   BOOL, DWORD, LPVOID, LPCWSTR, LPVOID,
                                   ctypes.POINTER(PROCESS_INFORMATION))
        self.ResumeThread = bind("ResumeThread", DWORD, HANDLE)
        self.TerminateProcess = bind("TerminateProcess", BOOL, HANDLE, DWORD)
        self.WaitForSingleObject = bind("WaitForSingleObject", DWORD, HANDLE, DWORD)
        self.GetExitCodeProcess = bind("GetExitCodeProcess", BOOL, HANDLE, ctypes.POINTER(DWORD))
        # Read-only accounting; optional for test doubles (see job_accounting).
        self.QueryInformationJobObject = bind("QueryInformationJobObject", BOOL,
                                              HANDLE, ctypes.c_int, LPVOID, DWORD, LPVOID)
        # Read-only direct-child peaks (efficiency P1b-2); optional for test doubles (see child_peak_memory).
        self.K32GetProcessMemoryInfo = bind("K32GetProcessMemoryInfo", BOOL, HANDLE, LPVOID, DWORD)
        self.get_osfhandle = msvcrt.get_osfhandle

    @staticmethod
    def last_error() -> int:
        return ctypes.get_last_error()


_kernel32_instance = None


def _kernel32() -> _Kernel32:
    global _kernel32_instance
    if _kernel32_instance is None:
        _kernel32_instance = _Kernel32()
    return _kernel32_instance


def _describe_error(code: int) -> str:
    try:
        return ctypes.FormatError(code).strip()  # type: ignore[attr-defined]  # Windows only
    except (AttributeError, OSError):
        return f"error {code}"


def _guard_error(what: str, api, code: int | None = None) -> ProcessGuardError:
    code = api.last_error() if code is None else code
    return ProcessGuardError(f"{what} failed: [WinError {code}] {_describe_error(code)}", code)


def _spawn_error(api, args) -> OSError:
    """CreateProcess failed: raise what Popen would (FileNotFoundError etc.)."""
    code = api.last_error()
    # errno=None + winerror lets OSError pick the subclass on Windows.
    return OSError(None, f"[WinError {code}] {_describe_error(code)}", args[0], code)


class _StdHandles:
    """Inheritable duplicates of the child's three standard handles.

    Owned by this object only until CreateProcess has returned; the child has
    its own copies from then on and ``close()`` drops ours.  Sources:
    ``None`` -> the owner's corresponding standard handle when it is usable,
    otherwise NUL; ``subprocess.DEVNULL`` -> NUL; a file object or fd -> its
    OS handle; ``subprocess.STDOUT`` (stderr only) -> the same handle as
    stdout.
    """

    def __init__(self, api, stdin, stdout, stderr):
        self._api = api
        self._owned: list[int] = []
        try:
            self.stdin = self._inheritable(stdin, STD_INPUT_HANDLE, os.O_RDONLY)
            self.stdout = self._inheritable(stdout, STD_OUTPUT_HANDLE, os.O_WRONLY)
            if _is_marker(stderr, subprocess.STDOUT):
                self.stderr = self.stdout
            else:
                self.stderr = self._inheritable(stderr, STD_ERROR_HANDLE, os.O_WRONLY)
        except BaseException:
            self.close()
            raise

    def distinct(self) -> list[int]:
        return sorted({self.stdin, self.stdout, self.stderr})

    def close(self) -> None:
        owned, self._owned = self._owned, []
        for handle in owned:
            self._api.CloseHandle(handle)

    def _inheritable(self, spec, std_id: int, devnull_flags: int) -> int:
        if spec is None:
            source = self._api.GetStdHandle(std_id)
            if source not in (None, 0, INVALID_HANDLE_VALUE):
                try:
                    return self._duplicate(source)
                except ProcessGuardError:
                    pass  # e.g. a stale handle inherited from a console-less parent
            return self._devnull(devnull_flags)
        if isinstance(spec, int):
            if spec == subprocess.DEVNULL:
                return self._devnull(devnull_flags)
            if spec in (subprocess.PIPE, subprocess.STDOUT):
                raise ValueError("subprocess.PIPE is unsupported and STDOUT is only valid for stderr")
            fd = spec
        else:
            fd = spec.fileno()
        return self._duplicate(self._api.get_osfhandle(fd))

    def _devnull(self, flags: int) -> int:
        fd = os.open(os.devnull, flags)
        try:
            return self._duplicate(self._api.get_osfhandle(fd))
        finally:
            os.close(fd)

    def _duplicate(self, source: int) -> int:
        target = HANDLE()
        current = self._api.GetCurrentProcess()
        if not self._api.DuplicateHandle(current, source, current, ctypes.byref(target),
                                         0, True, DUPLICATE_SAME_ACCESS):
            raise _guard_error("DuplicateHandle", self._api)
        self._owned.append(target.value)
        return target.value


class _AttributeList:
    """PROC_THREAD_ATTRIBUTE_LIST requiring HANDLE_LIST and JOB_LIST.

    HANDLE_LIST limits inheritance to exactly the standard handles (what
    ``Popen`` does since 3.7). JOB_LIST binds the child atomically. Failure of
    either attribute is fatal; CreateProcess must not run without both.
    """

    def __init__(self, api, buffer, values, job_list: bool):
        self._api = api
        self.buffer = buffer
        self._values = values  # ctypes objects referenced by the list; keep alive
        self.job_list = job_list

    @classmethod
    def build(cls, api, hjob: int, inherit: list[int]):
        size = SIZE_T(0)
        probe = api.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        if probe or size.value == 0 or api.last_error() != ERROR_INSUFFICIENT_BUFFER:
            raise _guard_error("InitializeProcThreadAttributeList(size)", api)
        buffer = ctypes.create_string_buffer(size.value)
        if not api.InitializeProcThreadAttributeList(buffer, 2, 0, ctypes.byref(size)):
            raise _guard_error("InitializeProcThreadAttributeList", api)
        try:
            handles = (HANDLE * len(inherit))(*inherit)
            if not api.UpdateProcThreadAttribute(buffer, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                                 handles, ctypes.sizeof(handles), None, None):
                raise _guard_error("UpdateProcThreadAttribute(HANDLE_LIST)", api)
            jobs = (HANDLE * 1)(hjob)
            if not api.UpdateProcThreadAttribute(buffer, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST,
                                                 jobs, ctypes.sizeof(jobs), None, None):
                raise _guard_error("UpdateProcThreadAttribute(JOB_LIST)", api)
        except BaseException:
            api.DeleteProcThreadAttributeList(buffer)
            raise
        return cls(api, buffer, (handles, jobs), True)

    def close(self) -> None:
        buffer, self.buffer = self.buffer, None
        if buffer is not None:
            self._api.DeleteProcThreadAttributeList(buffer)


def _environment_block(env):
    """A CreateProcessW environment block, or ``None`` to inherit this process's environment.

    UTF-16 ``NAME=value`` entries, each NUL-terminated and sorted
    case-insensitively (as ``subprocess.Popen`` builds them), then a final NUL.
    A name that is empty, holds ``=`` after its first character or holds NUL,
    or a value holding NUL, raises ``ValueError``.
    """
    if env is None:
        return None
    items = []
    for key, value in env.items():
        key, value = os.fsdecode(key), os.fsdecode(value)
        if not key or "=" in key[1:] or "\0" in key or "\0" in value:
            raise ValueError(f"illegal environment variable name or value: {key!r}")
        items.append((key, value))
    items.sort(key=lambda item: item[0].upper())
    return ctypes.create_unicode_buffer("".join(f"{key}={value}\0" for key, value in items) + "\0")


def _configure_job(api, hjob: int) -> None:
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not api.SetInformationJobObject(hjob, JobObjectExtendedLimitInformation,
                                       ctypes.byref(limits), ctypes.sizeof(limits)):
        raise _guard_error("SetInformationJobObject", api)
    # The whole scheme rests on the child *not* holding a handle to its own job:
    # an unnamed job handle is non-inheritable already; make that explicit.
    if not api.SetHandleInformation(hjob, HANDLE_FLAG_INHERIT, 0):
        raise _guard_error("SetHandleInformation", api)


def job_accounting(api, hjob) -> dict | None:
    """Job Object totals of every process ever in the job; ``None`` when unavailable.

    Read-only (QueryInformationJobObject) and best effort: it never raises.
    CPU times are seconds (the kernel reports 100 ns units); memory values
    are committed bytes as reported in JOBOBJECT_EXTENDED_LIMIT_INFORMATION.
    """
    query = getattr(api, "QueryInformationJobObject", None)
    if query is None or not hjob:
        return None
    try:
        basic = JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION()
        if not query(hjob, JobObjectBasicAndIoAccountingInformation, ctypes.byref(basic),
                     ctypes.sizeof(basic), None):
            return None
        info, io = basic.BasicInfo, basic.IoInfo
        result = {"source": "windows_job_object",
                  "cpu_user_seconds": info.TotalUserTime / 1e7,
                  "cpu_kernel_seconds": info.TotalKernelTime / 1e7,
                  "descendant_processes": int(info.TotalProcesses),
                  "active_processes": int(info.ActiveProcesses),
                  "page_faults": int(info.TotalPageFaultCount),
                  "io_read_operations": int(io.ReadOperationCount),
                  "io_write_operations": int(io.WriteOperationCount),
                  "io_other_operations": int(io.OtherOperationCount),
                  "io_read_bytes": int(io.ReadTransferCount),
                  "io_write_bytes": int(io.WriteTransferCount),
                  "io_other_bytes": int(io.OtherTransferCount),
                  "peak_job_memory_bytes": None, "peak_process_memory_bytes": None}
        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if query(hjob, JobObjectExtendedLimitInformation, ctypes.byref(limits), ctypes.sizeof(limits), None):
            result["peak_job_memory_bytes"] = int(limits.PeakJobMemoryUsed)
            result["peak_process_memory_bytes"] = int(limits.PeakProcessMemoryUsed)
        return result
    except Exception:
        return None


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD),
        ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", SIZE_T),
        ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T),
        ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T),
        ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T),
        ("PeakPagefileUsage", SIZE_T),
    ]


def child_peak_memory(api, hprocess) -> dict | None:
    """Peak working set and peak commit of the direct child (readable after it exited, while its handle is
    open); ``None`` when unavailable. Read-only diagnostics (efficiency P1b-2); never raises."""
    query = getattr(api, "K32GetProcessMemoryInfo", None)
    if query is None or not hprocess:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not query(hprocess, ctypes.byref(counters), counters.cb):
            return None
        return {"child_peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "child_peak_commit_bytes": int(counters.PeakPagefileUsage)}
    except Exception:
        return None


def _in_job(api, hprocess: int, hjob: int) -> bool:
    result = BOOL(0)
    if not api.IsProcessInJob(hprocess, hjob, ctypes.byref(result)):
        raise _guard_error("IsProcessInJob", api)
    return bool(result.value)


class _JobObjectProcess:
    """A child living inside a KILL_ON_JOB_CLOSE job owned by this process.

    Handle ownership: this object owns the job handle and the process handle
    until ``close()``.  The thread handle and the inheritable standard-handle
    duplicates are closed inside ``start()`` before it returns.

    Close order in ``close()``: kill (whole job) if still running -> bounded
    wait -> CloseHandle(job) -> CloseHandle(process).  Closing the job handle
    is the kernel-side kill switch, so it comes before the process handle and
    happens even when the wait timed out.
    """

    guard = GUARD_JOB_OBJECT

    @classmethod
    def start(cls, args, *, stdout, stderr, stdin, cwd, creationflags: int, api=None, env=None):
        api = _kernel32() if api is None else api
        # Validated before any handle exists; referenced here until CreateProcessW has returned.
        environment = _environment_block(env)
        args = [os.fspath(a) for a in args]
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(args))
        cwd = None if cwd is None else os.fspath(cwd)
        details = {"job_list": False, "handle_list": False, "assigned_after_create": False}

        hjob = api.CreateJobObjectW(None, None)
        if not hjob:
            raise _guard_error("CreateJobObjectW", api)
        hprocess = hthread = None
        try:
            _configure_job(api, hjob)
            std = _StdHandles(api, stdin, stdout, stderr)
            try:
                attributes = _AttributeList.build(api, hjob, std.distinct())
                try:
                    startup = STARTUPINFOEXW()
                    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
                    startup.StartupInfo.hStdInput = std.stdin
                    startup.StartupInfo.hStdOutput = std.stdout
                    startup.StartupInfo.hStdError = std.stderr
                    # CREATE_SUSPENDED: nothing runs until the job membership is confirmed.
                    flags = creationflags | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT
                    if environment is not None:
                        flags |= CREATE_UNICODE_ENVIRONMENT
                    startup.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
                    startup.lpAttributeList = ctypes.addressof(attributes.buffer)
                    details["handle_list"] = True
                    details["job_list"] = True
                    info = PROCESS_INFORMATION()
                    if not api.CreateProcessW(None, command_line, None, None, True, flags,
                                              None if environment is None else ctypes.addressof(environment),
                                              cwd, ctypes.byref(startup), ctypes.byref(info)):
                        raise _spawn_error(api, args)
                    # Own the returned handles immediately, before later cleanup
                    # callbacks can fail. Every following error takes the same
                    # child/process/thread cleanup path below.
                    hprocess, hthread, pid = info.hProcess, info.hThread, info.dwProcessId
                finally:
                    if attributes is not None:
                        attributes.close()
            finally:
                std.close()

            # From here on a (suspended) child exists; every failure must end it.
            if not _in_job(api, hprocess, hjob):
                raise ProcessGuardError("atomic JOB_LIST assignment was not established; child will not be resumed")
            previous_suspend_count = api.ResumeThread(hthread)
            if previous_suspend_count == RESUME_FAILED:
                raise _guard_error("ResumeThread", api)
            if previous_suspend_count != 1:
                raise ProcessGuardError(
                    f"unexpected primary-thread suspend count: {previous_suspend_count}")
            api.CloseHandle(hthread)
            hthread = None
            process = cls(api, hjob, hprocess, pid, args, details)
        except BaseException:
            try:
                if hprocess is not None:
                    try:
                        api.TerminateProcess(hprocess, KILL_EXIT_STATUS)
                    finally:
                        api.WaitForSingleObject(hprocess, int(CLOSE_KILL_TIMEOUT * 1000))
            finally:
                try:
                    if hthread is not None:
                        api.CloseHandle(hthread)
                finally:
                    try:
                        if hprocess is not None:
                            api.CloseHandle(hprocess)
                    finally:
                        api.CloseHandle(hjob)
            raise
        return process

    def __init__(self, api, hjob: int, hprocess: int, pid: int, args, details: dict):
        # Allocate the argument copy before taking ownership, so an allocation
        # failure is handled by start() without a partially owning finalizer.
        copied_args = list(args)
        self._api = api
        self._hjob: int | None = hjob
        self._hprocess: int | None = hprocess
        self._closed = False
        self.pid = pid
        self.args = copied_args
        self.details = details
        self.returncode: int | None = None
        self.accounting: dict | None = None
        self._started = time.monotonic()

    # -- Popen-compatible subset -------------------------------------------

    def poll(self):
        if self.returncode is None and self._hprocess is not None:
            state = self._api.WaitForSingleObject(self._hprocess, 0)
            if state == WAIT_OBJECT_0:
                self.returncode = self._exit_code()
            elif state != WAIT_TIMEOUT:
                raise _guard_error("WaitForSingleObject", self._api)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if self._hprocess is None:
                raise ProcessGuardError("process handle already closed")
            millis = INFINITE if timeout is None else max(0, int(timeout * 1000))
            state = self._api.WaitForSingleObject(self._hprocess, millis)
            if state == WAIT_TIMEOUT:
                raise subprocess.TimeoutExpired(self.args, timeout)
            if state != WAIT_OBJECT_0:
                raise _guard_error("WaitForSingleObject", self._api)
            self.returncode = self._exit_code()
        return self.returncode

    def terminate(self) -> None:
        """TerminateProcess on the child (what ``Popen.terminate`` does on Windows).

        Descendants are left to ``kill()``/``close()``, which act on the job.
        """
        if self.poll() is not None:
            return
        if not self._api.TerminateProcess(self._hprocess, KILL_EXIT_STATUS):
            code = self._api.last_error()
            # Like Popen: the call is refused once the process is already on
            # its way out; treat that as "already finished".
            exit_code = self._exit_code()
            if exit_code == STILL_ACTIVE:
                raise _guard_error("TerminateProcess", self._api, code)
            self.returncode = exit_code

    def kill(self) -> None:
        """Terminate everything in the job (child and descendants); idempotent."""
        if self._hjob is None:
            return
        if not self._api.TerminateJobObject(self._hjob, KILL_EXIT_STATUS):
            raise _guard_error("TerminateJobObject", self._api)

    # -- lifetime -----------------------------------------------------------

    def in_job(self) -> bool:
        """True while the child is (still) a member of the guard's job."""
        if self._hjob is None or self._hprocess is None:
            return False
        return _in_job(self._api, self._hprocess, self._hjob)

    def close(self) -> None:
        """Request cleanup, bounded-wait for the child, then close both handles.

        Job close also terminates descendants even if the direct child has
        already exited. It is not an all-descendants-exited wait primitive.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self.poll() is None:
                self.kill()
                try:
                    self.wait(timeout=CLOSE_KILL_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass  # closing the job handle below re-kills what is left
        finally:
            try:
                self._read_accounting()
            finally:
                self._release_handles()
        _publish_accounting(self.accounting)

    def _read_accounting(self) -> None:
        """Keep the job's accounting before its handle goes; diagnostics never raise."""
        try:
            accounting = job_accounting(self._api, self._hjob)
            if accounting is not None:
                accounting.update(wall_seconds=max(0.0, time.monotonic() - self._started),
                                  executable=os.path.basename(self.args[0]) if self.args else None,
                                  returncode=self.returncode,
                                  child_peak_working_set_bytes=None, child_peak_commit_bytes=None)
                accounting.update(child_peak_memory(self._api, self._hprocess) or {})
            self.accounting = accounting
        except Exception:
            self.accounting = None

    def _release_handles(self) -> None:
        hjob, self._hjob = self._hjob, None
        hprocess, self._hprocess = self._hprocess, None
        if hjob is not None:
            self._api.CloseHandle(hjob)  # last handle -> KILL_ON_JOB_CLOSE fires
        if hprocess is not None:
            self._api.CloseHandle(hprocess)

    def __del__(self):
        # Never block in a finaliser; the job handle close is the kill switch.
        try:
            self._release_handles()
        except Exception:
            pass

    def _exit_code(self) -> int:
        code = DWORD()
        if not self._api.GetExitCodeProcess(self._hprocess, ctypes.byref(code)):
            raise _guard_error("GetExitCodeProcess", self._api)
        return code.value
