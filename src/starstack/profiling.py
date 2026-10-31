"""Monotonic worker timing receipts with cheap per-stage process accounting.

No image sampling or process tracing: every reading is an O(1) operating-system
counter taken when a phase, stage or operation starts or ends.

``monotonic-stages-2`` keeps every ``monotonic-stages-1`` key with the same
meaning (``phases_seconds`` disjoint wall time that adds up to
``total_seconds``, ``operations_seconds`` overlapping subtotals) and adds,
alongside them:

* ``phase_entries`` -- how often each phase was entered;
* ``resources`` -- per phase and per operation: CPU seconds of this process
  (user/kernel), its I/O transfer counters, the bytes hashed through
  :func:`note_hashed` / :func:`sha256_file`, the guarded child processes that
  ended (Job Object accounting published by :mod:`starstack.process_guard`)
  and, at phase end, the working set / private bytes with the process-lifetime
  peaks. ``hashed_*`` is a lower bound: hash helpers that do not report are
  not included, and ``resources.hashed_sources`` names the helpers that did;
* ``hash_saved_*`` (per phase and operation) and ``resources.hash_saved_sources``
  -- bytes whose digest a helper took from the per-run hash ledger of
  :mod:`starstack.hashing` instead of reading the file (kind ``ledger``,
  ``handover`` or ``write``) or from an earlier item of the same batch (kind
  ``batch``), reported through :func:`note_hash_saved`;
* ``hashed_memory_*`` (per phase and operation) and
  ``resources.hashed_memory_sources`` -- bytes hashed from memory rather than
  from a file (array identities, encoded buffers, digests computed while
  writing), reported through :func:`note_hashed_memory`; not part of
  ``hashed_*``;
* ``host`` (when requested) -- logical processors / processor groups and a
  measured single-thread SHA-256 rate with a SHA-extension guess;
* ``notes`` (when any) -- small values attached with :func:`note` (every
  live timer), e.g. a pinned test configuration that explains the
  measurement, or with :meth:`StageTimer.attach_note` (that timer only, e.g.
  the resource governor's shadow decisions); :func:`note_append` (efficiency
  P1c, rule R19) accumulates a list instead, one entry per call (frame-pool
  site notes ``frame_pool.<site>``);
* with ``StageTimer(peak_sampler_hz=...)`` (efficiency P1b-2, ``process-resources-2``):
  per phase and per operation ``local_peak_private_bytes`` /
  ``local_peak_working_set_bytes`` from a low-rate sampler thread plus the
  phase and operation boundaries (sampled lower bounds, unlike the
  process-lifetime ``peak_*``), the sampler's own cost under
  ``resources.peak_sampler``, and ``operation_windows`` (start/end seconds of
  each operation, at most 64 per name, then counted). The machine-wide
  ``min_system_available_bytes`` is added only when the timer's
  ``record_system_memory`` is true, which the resource governor sets for its
  ``shadow-full`` level alone (prereg section 8, U-G4: no raw RAM values under
  the default ``shadow`` or ``off``). A sampler thread that cannot start
  leaves the timer without local peaks and says so under
  ``resources.peak_sampler.error``; it never fails the worker. Guarded
  children add ``child_peak_working_set_bytes`` / ``child_peak_commit_bytes``
  (direct child, read before its handle closes) to ``children``.

These are diagnostics. They never enter a signature, a data digest, an
idempotency key or a delivered receipt, and readers must accept records
without them (older workers wrote only the v1 keys, or nothing).

Nested stages
-------------
Workers that call deep helpers use :func:`activate` for their timer and mark
work with :func:`stage` / :func:`staged` / :func:`switch`. These act on every
timer activated in the current thread and are no-ops when none is active.
Stage time is exclusive: a nested stage pauses its parent, so phases stay
disjoint. A name starting with ``"."`` is relative to the innermost open stage
of each timer (``stage("publish")`` then ``switch(".verify")`` is
``publish.verify``; with no open stage it is just ``verify``).
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import functools
import hashlib
import math
import os
import sys
import threading
import time
import weakref
from time import perf_counter

VERSION = "monotonic-stages-2"
LEGACY_VERSION = "monotonic-stages-1"
RESOURCES_VERSION = "process-resources-2"   # -2 adds local peaks, windows, child peaks (additive)
HOST_VERSION = "host-profile-1"
# Single-thread OpenSSL SHA-256 is about 2 GB/s with SHA extensions (SHA-NI)
# on current desktop/server cores and about 0.4-0.7 GB/s without them.
SHA_EXT_THRESHOLD_GBPS = 1.0
SHA_PROBE_CHUNK = 1 << 20
SHA_PROBE_CHUNKS = 64

HASH_SAVED_KINDS = ("ledger", "handover", "write", "batch")
_DELTA_KEYS = ("cpu_user_seconds", "cpu_kernel_seconds",
               "io_read_operations", "io_write_operations", "io_other_operations",
               "io_read_bytes", "io_write_bytes", "io_other_bytes",
               "hashed_bytes", "hashed_files",
               "hash_saved_bytes", "hash_saved_files") + tuple(f"hash_saved_{kind}_bytes" for kind in HASH_SAVED_KINDS) + (
               "hashed_memory_bytes", "hashed_memory_calls")
_LEVEL_KEYS = ("working_set_bytes", "private_bytes")
_PEAK_KEYS = ("peak_working_set_bytes", "peak_private_bytes")
_CHILD_SUM_KEYS = ("cpu_user_seconds", "cpu_kernel_seconds", "io_read_bytes", "io_write_bytes",
                   "io_other_bytes", "wall_seconds", "descendant_processes")
_CHILD_MAX_KEYS = ("peak_job_memory_bytes", "peak_process_memory_bytes",
                   "child_peak_working_set_bytes", "child_peak_commit_bytes")

RESOURCE_SEMANTICS = (
    "resources are process-wide counter deltas charged to whichever phase/operation was current in the timer; "
    "cpu_* is this process only (all threads); io_* are GetProcessIoCounters transfers (memory-mapped paging "
    "reads are not counted); hashed_* is a lower bound: it counts only hashing reported through "
    "starstack.profiling.note_hashed or sha256_file (hashed_sources names the reporting helpers over the whole "
    "receipt) and omits hash helpers that do not report; hash_saved_* counts bytes whose digest came from the "
    "per-run hash ledger (starstack.hashing) instead of a read, by kind ledger/handover/write, or from an earlier "
    "item of the same batch (kind batch) (hash_saved_sources names the helpers; in shadow mode the verifying reads "
    "are in hashed_* as well); hashed_memory_* counts bytes hashed from memory, not from a file (array identities, "
    "encoded buffers, digests computed while writing; hashed_memory_sources names the helpers) and is not part of "
    "hashed_*; children are "
    "guarded child processes (Job Object accounting, including their descendants) that ended during the "
    "phase/operation; memory levels are taken at phase end, peak_* are process-lifetime peaks at phase end and "
    "*_rise_bytes is how much the lifetime peak grew inside the phase")
LOCAL_PEAK_SEMANTICS = (
    "sampled lower bound: local_peak_* are the highest private / working-set bytes of this process seen inside the "
    "phase or operation by the peak sampler thread (peak_sampler.hz) and at its boundaries; "
    "min_system_available_bytes (only when the resource governor runs at shadow-full, prereg U-G4) is the lowest "
    "system available physical memory seen by the sampler there")
WINDOW_LIMIT = 64
PEAK_SAMPLER_THREAD = "aester-peak-sampler"


# --------------------------------------------------------------------------- #
# Process-wide counters                                                        #
# --------------------------------------------------------------------------- #

_counter_lock = threading.Lock()
_hashed = [0, 0]  # bytes, files
_hashed_by_source = {}  # label -> [bytes, files]
_saved = {kind: [0, 0] for kind in HASH_SAVED_KINDS}  # kind -> [bytes, files]
_saved_by_source = {}  # label -> {kind: [bytes, files]}
_memory = [0, 0]  # bytes, calls
_memory_by_source = {}  # label -> [bytes, calls]
UNLABELLED_SOURCE = "unlabelled"


def note_hashed(nbytes, files=1, source=None):
    """Count bytes hashed for integrity checks; never raises.

    ``source`` is a short label of the reporting helper (e.g. ``"core._sha"``)
    so a receipt shows which hashing its lower-bound totals include.
    """
    try:
        nbytes, files = max(0, int(nbytes)), max(0, int(files))
        label = UNLABELLED_SOURCE if source is None else str(source)[:80]
    except Exception:
        return
    with _counter_lock:
        _hashed[0] += nbytes
        _hashed[1] += files
        entry = _hashed_by_source.setdefault(label, [0, 0])
        entry[0] += nbytes
        entry[1] += files


def note_hash_saved(nbytes, files=1, source=None, kind="ledger"):
    """Count bytes whose digest came from the hash ledger instead of a read; never raises.

    ``kind`` is ``"ledger"`` (same-process repeat), ``"handover"`` (digest
    handed over by another process of the same run) or ``"write"`` (digest
    computed while this process wrote the file); other kinds are ignored.
    """
    try:
        nbytes, files = max(0, int(nbytes)), max(0, int(files))
        label = UNLABELLED_SOURCE if source is None else str(source)[:80]
    except Exception:
        return
    if kind not in HASH_SAVED_KINDS:
        return
    with _counter_lock:
        _saved[kind][0] += nbytes
        _saved[kind][1] += files
        entry = _saved_by_source.setdefault(label, {name: [0, 0] for name in HASH_SAVED_KINDS})
        entry[kind][0] += nbytes
        entry[kind][1] += files


def note_hashed_memory(nbytes, calls=1, source=None):
    """Count bytes hashed from memory (not read from a file); never raises."""
    try:
        nbytes, calls = max(0, int(nbytes)), max(0, int(calls))
        label = UNLABELLED_SOURCE if source is None else str(source)[:80]
    except Exception:
        return
    with _counter_lock:
        _memory[0] += nbytes
        _memory[1] += calls
        entry = _memory_by_source.setdefault(label, [0, 0])
        entry[0] += nbytes
        entry[1] += calls


def _memory_reading():
    with _counter_lock:
        return ({"hashed_memory_bytes": _memory[0], "hashed_memory_calls": _memory[1]},
                {label: {"hashed_memory_bytes": value[0], "hashed_memory_calls": value[1]}
                 for label, value in _memory_by_source.items()})


def hashed_memory_totals():
    return _memory_reading()[0]


def hashed_memory_sources():
    """Process-wide in-memory hashed bytes/calls per reporting helper since start."""
    return _memory_reading()[1]


def _saved_view(by_kind):
    out = {"hash_saved_bytes": sum(value[0] for value in by_kind.values()),
           "hash_saved_files": sum(value[1] for value in by_kind.values())}
    for kind in HASH_SAVED_KINDS:
        out[f"hash_saved_{kind}_bytes"] = by_kind[kind][0]
    return out


def _hashed_reading():
    """Totals and the per-helper split, from one consistent reading."""
    with _counter_lock:
        return ({"hashed_bytes": _hashed[0], "hashed_files": _hashed[1]},
                {label: {"hashed_bytes": value[0], "hashed_files": value[1]}
                 for label, value in _hashed_by_source.items()})


def _saved_reading():
    """Saved totals and the per-helper split (see :func:`note_hash_saved`), from one consistent reading."""
    with _counter_lock:
        return _saved_view(_saved), {label: _saved_view(value) for label, value in _saved_by_source.items()}


def hash_saved_totals():
    return _saved_reading()[0]


def hash_saved_sources():
    """Process-wide saved bytes/files per reporting helper since start."""
    return _saved_reading()[1]


def hashed_totals():
    return _hashed_reading()[0]


def hashed_sources():
    """Process-wide hashed bytes/files per reporting helper since start."""
    return _hashed_reading()[1]


def sha256_file(path, source="profiling.sha256_file"):
    """Hex SHA-256 of a file, counted through :func:`note_hashed`.

    The digest is exactly ``hashlib.file_digest(handle, "sha256")``; hash
    helpers elsewhere can use this (or add one ``note_hashed`` line) so the
    per-stage hashed bytes include them.
    """
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        note_hashed(handle.tell(), source=source)
    return digest


_timers = weakref.WeakSet()
_timers_lock = threading.Lock()


def note_child_accounting(accounting):
    """Receive one ended guarded child's Job Object accounting (see process_guard)."""
    if not isinstance(accounting, dict):
        return
    with _timers_lock:
        live = list(_timers)
    for timer in live:
        try:
            timer._on_child(accounting)
        except Exception:
            pass


def note(key, value):
    """Attach a small diagnostic value to every running timer's receipt (``notes``).

    For settings that explain a measurement, e.g. a pinned Siril thread/memory
    configuration or a resource decision. The value is copied as JSON-safe
    data; like all timings it must never feed a signature or digest.
    """
    safe = _json_safe(value)
    with _timers_lock:
        live = list(_timers)
    for timer in live:
        try:
            timer._note(str(key), copy.deepcopy(safe))
        except Exception:
            pass


NOTE_LIST_LIMIT = 64          # entries kept per accumulated note; later calls are only counted in "<key>.dropped"


def note_append(key, value, *, limit=NOTE_LIST_LIMIT):
    """Append a small diagnostic value to the list note ``key`` of every running timer: one entry per call, so a
    site that runs several times never overwrites itself (efficiency P1c rule R19, e.g. ``frame_pool.<site>``). After
    ``limit`` entries further calls are counted in ``<key>.dropped``. Never raises; diagnostics only, like :func:`note`.
    """
    try:
        safe = _json_safe(value)
        limit = max(0, int(limit))
    except Exception:
        return
    with _timers_lock:
        live = list(_timers)
    for timer in live:
        try:
            timer._note_append(str(key), copy.deepcopy(safe), limit)
        except Exception:
            pass


def _register_with_process_guard():
    try:
        from . import process_guard
        process_guard.add_accounting_listener(note_child_accounting)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Operating-system readings                                                    #
# --------------------------------------------------------------------------- #

_windows = None


def _windows_api():
    """Lazily bound kernel32 entry points, or False when unavailable."""
    global _windows
    if _windows is None:
        try:
            import ctypes

            class MemoryCounters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                            ("PrivateUsage", ctypes.c_size_t)]

            class IoCounters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            current = kernel.GetCurrentProcess
            current.restype, current.argtypes = ctypes.c_void_p, []
            memory = kernel.K32GetProcessMemoryInfo
            memory.restype = ctypes.c_int32
            memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_uint32]
            io = kernel.GetProcessIoCounters
            io.restype, io.argtypes = ctypes.c_int32, [ctypes.c_void_p, ctypes.POINTER(IoCounters)]

            class MemoryStatus(ctypes.Structure):  # MEMORYSTATUSEX
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                         "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                         "ullAvailExtendedVirtual")]

            status = kernel.GlobalMemoryStatusEx
            status.restype, status.argtypes = ctypes.c_int32, [ctypes.POINTER(MemoryStatus)]
            _windows = {"ctypes": ctypes, "kernel": kernel, "current": current, "memory": memory, "io": io,
                        "MemoryCounters": MemoryCounters, "IoCounters": IoCounters,
                        "status": status, "MemoryStatus": MemoryStatus}
        except Exception:
            _windows = False
    return _windows


def process_resources():
    """One snapshot of this process's counters (``None`` where unavailable)."""
    out = dict.fromkeys(_DELTA_KEYS + _LEVEL_KEYS + _PEAK_KEYS)
    try:
        times = os.times()
        out["cpu_user_seconds"], out["cpu_kernel_seconds"] = float(times.user), float(times.system)
    except Exception:
        pass
    totals, out["hashed_sources"] = _hashed_reading()
    out.update(totals)
    saved, out["hash_saved_sources"] = _saved_reading()
    out.update(saved)
    memory, out["hashed_memory_sources"] = _memory_reading()
    out.update(memory)
    api = _windows_api() if os.name == "nt" else False
    if api:
        ctypes = api["ctypes"]
        handle = api["current"]()
        counters = api["MemoryCounters"]()
        counters.cb = ctypes.sizeof(counters)
        if api["memory"](handle, ctypes.byref(counters), counters.cb):
            out.update(working_set_bytes=int(counters.WorkingSetSize),
                       peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                       private_bytes=int(counters.PrivateUsage),
                       peak_private_bytes=int(counters.PeakPagefileUsage))
        io = api["IoCounters"]()
        if api["io"](handle, ctypes.byref(io)):
            out.update(io_read_operations=int(io.ReadOperationCount), io_write_operations=int(io.WriteOperationCount),
                       io_other_operations=int(io.OtherOperationCount), io_read_bytes=int(io.ReadTransferCount),
                       io_write_bytes=int(io.WriteTransferCount), io_other_bytes=int(io.OtherTransferCount))
    elif os.name != "nt":
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            out["peak_working_set_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except Exception:
            pass
        try:
            with open("/proc/self/io", encoding="ascii") as handle:
                fields = dict(line.split(":", 1) for line in handle.read().splitlines() if ":" in line)
            out["io_read_bytes"], out["io_write_bytes"] = int(fields["read_bytes"]), int(fields["write_bytes"])
        except Exception:
            pass
    return out


def peak_reading():
    """(private bytes, working set bytes, system available physical bytes) now; None where unavailable.

    The peak sampler's one reading: two O(1) counters (K32GetProcessMemoryInfo, GlobalMemoryStatusEx).
    """
    api = _windows_api() if os.name == "nt" else False
    if not api:
        return None
    try:
        ctypes = api["ctypes"]
        counters = api["MemoryCounters"]()
        counters.cb = ctypes.sizeof(counters)
        private = working = available = None
        if api["memory"](api["current"](), ctypes.byref(counters), counters.cb):
            private, working = int(counters.PrivateUsage), int(counters.WorkingSetSize)
        status = api["MemoryStatus"]()
        status.dwLength = ctypes.sizeof(status)
        if api["status"](ctypes.byref(status)):
            available = int(status.ullAvailPhys)
        return private, working, available
    except Exception:
        return None


def _peak_loop(ref, stop, interval, reader):
    """Sampler thread body; holds the timer only weakly so an unfinished timer can still be collected."""
    started, cpu, ticks = perf_counter(), time.thread_time(), 0
    while not stop.wait(interval):
        timer = ref()
        if timer is None:
            return
        try:
            reading = reader()
            if reading is not None:
                timer._peak_sample(reading)
        except Exception:
            pass
        ticks += 1
        del timer
    timer = ref()
    if timer is not None:
        timer._peak_done(ticks, perf_counter() - started, time.thread_time() - cpu)


_host = None
_host_lock = threading.Lock()


def host_profile():
    """Logical processors and a measured SHA-256 rate; measured once per process."""
    global _host
    with _host_lock:
        if _host is None:
            _host = _measure_host()
        return copy.deepcopy(_host)


def _measure_host():
    info = {"version": HOST_VERSION, "logical_processors": os.cpu_count(),
            "openssl_ia32cap": os.environ.get("OPENSSL_ia32cap"),
            "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER")}
    try:
        chunk = bytes(SHA_PROBE_CHUNK)
        hashlib.sha256(chunk).digest()  # warm the implementation and the buffer
        started = perf_counter()
        digest = hashlib.sha256()
        for _ in range(SHA_PROBE_CHUNKS):
            digest.update(chunk)
        digest.digest()
        seconds = max(perf_counter() - started, 1e-9)
        rate = SHA_PROBE_CHUNK * SHA_PROBE_CHUNKS / seconds / 1e9
        info.update(sha256_probe_bytes=SHA_PROBE_CHUNK * SHA_PROBE_CHUNKS, sha256_seconds=round(seconds, 6),
                    sha256_rate_gbps=round(rate, 3), sha_ext_guess=bool(rate >= SHA_EXT_THRESHOLD_GBPS),
                    sha_ext_threshold_gbps=SHA_EXT_THRESHOLD_GBPS,
                    sha_ext_method="single-thread in-memory hashlib.sha256 rate against the threshold; not cpuid")
    except Exception as exc:
        info.update(sha256_rate_gbps=None, sha_ext_guess=None, sha_probe_error=f"{type(exc).__name__}: {exc}")
    if os.name == "nt":
        try:
            info.update(_windows_processors())
        except Exception:
            pass
    return info


def _windows_processors():
    import ctypes
    import winreg
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    group_count = kernel.GetActiveProcessorGroupCount
    group_count.restype, group_count.argtypes = ctypes.c_uint16, []
    per_group = kernel.GetActiveProcessorCount
    per_group.restype, per_group.argtypes = ctypes.c_uint32, [ctypes.c_uint16]
    current = kernel.GetCurrentProcess
    current.restype, current.argtypes = ctypes.c_void_p, []
    out = {"processor_groups": [int(per_group(g)) for g in range(int(group_count()))]}
    affinity = kernel.GetProcessAffinityMask
    affinity.restype = ctypes.c_int32
    affinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if affinity(current(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        out["process_affinity_processors"] = bin(process_mask.value).count("1")
    groups = kernel.GetProcessGroupAffinity
    groups.restype = ctypes.c_int32
    groups.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16)]
    count, array = ctypes.c_uint16(64), (ctypes.c_uint16 * 64)()
    if groups(current(), ctypes.byref(count), array):
        out["process_groups"] = [int(array[i]) for i in range(count.value)]
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            out["cpu_name"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------- #
# Accumulators                                                                 #
# --------------------------------------------------------------------------- #

def _finite(value):
    return value if not isinstance(value, float) or math.isfinite(value) else None


def _delta(before, after):
    out = {}
    for key in _DELTA_KEYS:
        a = (before or {}).get(key)
        b = (after or {}).get(key)
        out[key] = None if a is None or b is None else max(0, b - a)
    return out


def _add(target, delta):
    for key, value in delta.items():
        if value is not None:
            target[key] = (target.get(key) or 0) + value


_SAVED_SOURCE_KEYS = ("hash_saved_bytes", "hash_saved_files") + tuple(
    f"hash_saved_{kind}_bytes" for kind in HASH_SAVED_KINDS)


def _source_delta(before, after, field="hashed_sources", keys=("hashed_bytes", "hashed_files")):
    """Per-helper hashed (or saved) bytes/files between two samples; None without source readings."""
    first = (before or {}).get(field)
    last = (after or {}).get(field)
    if not isinstance(first, dict) or not isinstance(last, dict):
        return None
    out = {}
    for label, value in sorted(last.items()):
        old = first.get(label) or {}
        change = {key: max(0, int(value.get(key) or 0) - int(old.get(key) or 0)) for key in keys}
        if any(change.values()):
            out[label] = change
    return out


def _new_children():
    return {"processes": 0, **dict.fromkeys(_CHILD_SUM_KEYS, 0), **dict.fromkeys(_CHILD_MAX_KEYS)}


def _add_child(target, accounting):
    target["processes"] += 1
    for key in _CHILD_SUM_KEYS:
        value = accounting.get(key)
        if isinstance(value, (int, float)) and _finite(value) is not None:
            target[key] += value
    for key in _CHILD_MAX_KEYS:
        value = accounting.get(key)
        if isinstance(value, int):
            target[key] = value if target[key] is None else max(target[key], value)


def _round(value):
    return round(value, 6) if isinstance(value, float) else value


def _view(acc, children=None):
    out = {key: _round(acc.get(key)) for key in _DELTA_KEYS}
    user, kernel = acc.get("cpu_user_seconds"), acc.get("cpu_kernel_seconds")
    out["cpu_seconds"] = None if user is None or kernel is None else _round(user + kernel)
    for key in _LEVEL_KEYS + _PEAK_KEYS + ("peak_working_set_rise_bytes", "peak_private_rise_bytes"):
        if key in acc:
            out[key] = acc[key]
    if children and children["processes"]:
        child = {key: _round(value) for key, value in children.items()}
        child["cpu_seconds"] = _round(children["cpu_user_seconds"] + children["cpu_kernel_seconds"])
        out["children"] = child
    return out


# --------------------------------------------------------------------------- #
# Timer                                                                        #
# --------------------------------------------------------------------------- #

class StageTimer:
    """Non-overlapping phases plus explicitly overlapping operation subtotals.

    A receipt describes one worker attempt, including reused-checkpoint overhead.
    Wall time includes child-process waits; the resources block separates this
    process's CPU from the CPU of guarded children.
    """
    def __init__(self, clock=None, *, sampler=None, host=False, initial="setup", peak_sampler_hz=None,
                 peak_reader=None):
        self.clock = clock or perf_counter
        self._sampler = sampler or process_resources
        self.host = None
        if host:
            try:
                self.host = host_profile()  # measured before the timer starts
            except Exception as exc:
                self.host = {"version": HOST_VERSION, "error": f"{type(exc).__name__}: {exc}"}
        self._lock = threading.Lock()
        self.started = self.clock()
        self.phase_started = self.started
        self.current = initial
        self.phases = {}
        self.operations = {}
        self.counts = {}
        self.entries = {initial: 1}
        self.finished = None
        self._report = None
        self._stack = []           # (return_to, stage_key) of open stages
        self._phase_resources = {}
        self._operation_resources = {}
        self._phase_children = {}
        self._operation_children = {}
        self._open_operations = {}
        self.notes = {}
        # efficiency P1b-2: local peaks per phase/operation from a low-rate sampler thread (off by default)
        self._peak_hz = None
        self._local_peaks = {}     # ("phase"|"operation", name) -> [private, working set, min system available]
        self._windows = {}         # operation -> [[start_s, end_s], ...] (sampler-enabled timers only)
        self._windows_dropped = {}
        self._peak_stats = None
        self._peak_thread = self._peak_stop = None
        self._peak_error = None
        self._peak_reader = peak_reader or peak_reading
        # U-G4 (prereg section 8): the machine-wide available RAM is a raw system-memory value, written only when the
        # resource governor runs at shadow-full; resources.shadow_start sets this (profiling never imports resources).
        self.record_system_memory = False
        self._first = self._last = self._sample()
        with _timers_lock:
            _timers.add(self)
        if peak_sampler_hz:
            try:
                self._start_peak_sampler(float(peak_sampler_hz))
            except Exception as exc:     # no sampler (diagnostics only): never fails the worker, finish() included
                self._peak_hz = None
                self._peak_thread = self._peak_stop = None
                self._peak_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def _start_peak_sampler(self, hz):
        if not math.isfinite(hz) or hz <= 0:
            return
        self._peak_hz = hz
        if self._first:
            self._peak_update(("phase", self.current), self._first)
        self._peak_stop = threading.Event()
        self._peak_thread = threading.Thread(target=_peak_loop, name=PEAK_SAMPLER_THREAD, daemon=True,
                                             args=(weakref.ref(self), self._peak_stop, 1.0 / hz, self._peak_reader))
        self._peak_thread.start()

    def _peak_update(self, key, reading):
        """Raise the local peaks of ``key``; ``reading`` is a peak_reading() tuple or a process_resources() sample."""
        if isinstance(reading, dict):
            reading = (reading.get("private_bytes"), reading.get("working_set_bytes"), None)
        private, working, available = reading
        entry = self._local_peaks.setdefault(key, [None, None, None])
        if private is not None:
            entry[0] = private if entry[0] is None else max(entry[0], private)
        if working is not None:
            entry[1] = working if entry[1] is None else max(entry[1], working)
        if available is not None:
            entry[2] = available if entry[2] is None else min(entry[2], available)

    def _peak_sample(self, reading):
        with self._lock:
            if self._report is not None:
                return
            self._peak_update(("phase", self.current), reading)
            for name in self._open_operations:
                self._peak_update(("operation", name), reading)

    def _peak_done(self, ticks, seconds, cpu_seconds):
        self._peak_stats = {"hz": self._peak_hz, "ticks": ticks, "seconds": round(seconds, 3),
                            "effective_hz": round(ticks / seconds, 2) if seconds > 0 else None,
                            "cpu_seconds": round(cpu_seconds, 4),
                            "cpu_share": round(cpu_seconds / seconds, 5) if seconds > 0 else None,
                            "thread": PEAK_SAMPLER_THREAD}

    def _stop_peak_sampler(self):
        """Stop and join a started sampler thread; never raises (finish() must not fail the worker)."""
        thread, stop = self._peak_thread, self._peak_stop
        self._peak_thread = None
        try:
            if stop is not None:
                stop.set()
            if thread is not None and thread.ident is not None and thread is not threading.current_thread():
                thread.join(1.0)
        except Exception:
            pass

    # -- readings --------------------------------------------------------------
    def _sample(self):
        try:
            return self._sampler()
        except Exception:
            return None

    def _charge(self, key, sample):
        acc = self._phase_resources.setdefault(key, {})
        _add(acc, _delta(self._last, sample))
        if sample:
            for level in _LEVEL_KEYS:
                if sample.get(level) is not None:
                    acc[level] = sample[level]
            for peak in _PEAK_KEYS:
                value = sample.get(peak)
                if value is not None:
                    acc[peak] = max(acc.get(peak) or 0, value)
                    before = (self._last or {}).get(peak)
                    rise = peak.replace("_bytes", "_rise_bytes")
                    acc[rise] = (acc.get(rise) or 0) + (max(0, value - before) if before is not None else 0)
        self._last = sample

    def _note(self, key, value):
        with self._lock:
            if self._report is None:
                self.notes[key] = value

    def _note_append(self, key, value, limit):
        with self._lock:
            if self._report is not None:
                return
            entries = self.notes.get(key)
            if not isinstance(entries, list):
                entries = self.notes[key] = []
            if len(entries) < limit:
                entries.append(value)
            else:
                dropped = key + ".dropped"
                self.notes[dropped] = self.notes.get(dropped, 0) + 1

    def attach_note(self, key, value):
        """Attach a small JSON-safe value to this timer's receipt only (``notes``); never raises.

        The public form of the per-timer half of :func:`note`, which reaches every live timer. Ignored once the
        receipt is built. Diagnostics only, like all timings.
        """
        try:
            self._note(str(key), copy.deepcopy(_json_safe(value)))
        except Exception:
            pass

    def _on_child(self, accounting):
        with self._lock:
            if self._report is not None:
                return
            _add_child(self._phase_children.setdefault(self.current, _new_children()), accounting)
            for name in self._open_operations:
                _add_child(self._operation_children.setdefault(name, _new_children()), accounting)

    # -- phases and stages -----------------------------------------------------
    def _switch(self, key, entry=True):
        if self._report is not None:
            return
        instant = self.clock()
        with self._lock:
            self.phases[self.current] = self.phases.get(self.current, 0.0) + max(0.0, instant - self.phase_started)
            sample = self._sample()
            self._charge(self.current, sample)
            if self._peak_hz and sample:
                self._peak_update(("phase", self.current), sample)
                self._peak_update(("phase", key), sample)
            self.current, self.phase_started = key, instant
            if entry:
                self.entries[key] = self.entries.get(key, 0) + 1

    def _qualify(self, name):
        if not name.startswith("."):
            return name
        base = self._stack[-1][1] if self._stack else ""
        return base + name if base else name[1:]

    def phase(self, name):
        self._switch(name)

    def switch(self, name):
        """Move the current time charge to a sibling key inside the innermost open stage."""
        self._switch(self._qualify(name))

    def _enter(self, name):
        if self._report is not None:
            return False
        key = self._qualify(name)
        self._stack.append((self.current, key))
        self._switch(key)
        return True

    def _leave(self):
        if self._stack and self._report is None:
            return_to, _ = self._stack.pop()
            self._switch(return_to, entry=False)

    @contextmanager
    def stage(self, name):
        entered = self._enter(name)
        try:
            yield
        finally:
            if entered:
                self._leave()

    # -- overlapping operations ------------------------------------------------
    @contextmanager
    def operation(self, name):
        start = self.clock()
        before = self._sample()
        with self._lock:
            self._open_operations[name] = self._open_operations.get(name, 0) + 1
            if self._peak_hz and before:
                self._peak_update(("operation", name), before)
        try:
            yield
        finally:
            after = self._sample()
            with self._lock:
                end = self.clock()
                self.operations[name] = self.operations.get(name, 0.0) + max(0.0, end - start)
                self.counts[name] = self.counts.get(name, 0) + 1
                _add(self._operation_resources.setdefault(name, {}), _delta(before, after))
                if self._peak_hz:
                    if after:
                        self._peak_update(("operation", name), after)
                    if self._report is None:
                        windows = self._windows.setdefault(name, [])
                        if len(windows) < WINDOW_LIMIT:
                            windows.append([round(start - self.started, 3), round(end - self.started, 3)])
                        else:
                            self._windows_dropped[name] = self._windows_dropped.get(name, 0) + 1
                self._open_operations[name] -= 1
                if not self._open_operations[name]:
                    del self._open_operations[name]

    def call(self, name, function, *args, **kwargs):
        with self.operation(name):
            return function(*args, **kwargs)

    # -- receipt ---------------------------------------------------------------
    def finish(self):
        self._stop_peak_sampler()   # joined before the receipt is built (it takes the lock per sample)
        with self._lock:
            first = self._report is None
            if first:
                self.finished = self.clock()
                self.phases[self.current] = self.phases.get(self.current, 0.0) + max(0.0, self.finished - self.phase_started)
                final = self._sample()
                self._charge(self.current, final)
                self._report = self._build(final)
        if first:
            with _timers_lock:
                _timers.discard(self)
        return copy.deepcopy(self._report)

    def _build(self, final):
        report = {"version": VERSION, "scope": "worker_attempt", "unit": "seconds",
                  "total_seconds": max(0.0, self.finished - self.started),
                  "phases_seconds": dict(self.phases), "operations_seconds": dict(self.operations),
                  "operation_counts": dict(self.counts),
                  "semantics": "phases are disjoint wall time; operations overlap phases and must not be added to them",
                  "compatible_with": LEGACY_VERSION, "phase_entries": dict(self.entries)}
        try:
            children = _new_children()
            for child in self._phase_children.values():
                for key in ("processes",) + _CHILD_SUM_KEYS:
                    children[key] += child[key]
                for key in _CHILD_MAX_KEYS:
                    if child[key] is not None:
                        children[key] = child[key] if children[key] is None else max(children[key], child[key])
            total = _delta(self._first, final)
            memory = {key: (final or {}).get(key) for key in _LEVEL_KEYS + _PEAK_KEYS}
            report["resources"] = {
                "version": RESOURCES_VERSION, "platform": sys.platform, "pid": os.getpid(),
                "phases": {key: _view(acc, self._phase_children.get(key))
                           for key, acc in self._phase_resources.items()},
                "operations": {key: _view(acc, self._operation_children.get(key))
                               for key, acc in self._operation_resources.items()},
                "total": _view({**total, **memory}, children),
                "semantics": RESOURCE_SEMANTICS}
            if self._peak_hz:
                self._attach_local_peaks(report)
            elif self._peak_error:
                report["resources"]["peak_sampler"] = {"hz": None, "error": self._peak_error,
                                                       "thread": PEAK_SAMPLER_THREAD}
            sources = _source_delta(self._first, final)
            if sources is not None:
                report["resources"]["hashed_sources"] = sources
            saved = _source_delta(self._first, final, "hash_saved_sources", _SAVED_SOURCE_KEYS)
            if saved is not None:
                report["resources"]["hash_saved_sources"] = saved
            in_memory = _source_delta(self._first, final, "hashed_memory_sources",
                                      ("hashed_memory_bytes", "hashed_memory_calls"))
            if in_memory is not None:
                report["resources"]["hashed_memory_sources"] = in_memory
        except Exception as exc:  # diagnostics must never fail the worker
            report["resources"] = {"version": RESOURCES_VERSION, "error": f"{type(exc).__name__}: {exc}"}
        if self.host is not None:
            report["host"] = self.host
        if self._windows:
            report["operation_windows"] = {name: [list(w) for w in windows] for name, windows in self._windows.items()}
            if self._windows_dropped:
                report["operation_windows_dropped"] = dict(self._windows_dropped)
        if self.notes:
            report["notes"] = dict(self.notes)
        return _json_safe(report)

    def _attach_local_peaks(self, report):
        resources = report["resources"]
        system_memory = self.record_system_memory is True        # U-G4: raw system RAM only at shadow-full
        for kind, section in (("phase", "phases"), ("operation", "operations")):
            for name, view in resources[section].items():
                private, working, available = self._local_peaks.get((kind, name), (None, None, None))
                view.update(local_peak_private_bytes=private, local_peak_working_set_bytes=working)
                if system_memory:
                    view["min_system_available_bytes"] = available
        resources["local_peak_semantics"] = LOCAL_PEAK_SEMANTICS
        resources["peak_sampler"] = self._peak_stats or {"hz": self._peak_hz, "ticks": None,
                                                         "thread": PEAK_SAMPLER_THREAD}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Thread-local activation for deep helpers                                     #
# --------------------------------------------------------------------------- #

_local = threading.local()


def active():
    return getattr(_local, "timers", ())


@contextmanager
def activate(timer):
    """Make ``timer`` receive :func:`stage` / :func:`switch` marks in this thread."""
    previous = active()
    _local.timers = previous + (timer,)
    try:
        yield timer
    finally:
        _local.timers = previous


@contextmanager
def stage(name):
    """Exclusive nested stage on every active timer; a no-op without one."""
    entered = []
    for timer in active():
        try:
            if timer._enter(name):
                entered.append(timer)
        except Exception:
            pass
    try:
        yield
    finally:
        for timer in reversed(entered):
            try:
                timer._leave()
            except Exception:
                pass


def switch(name):
    """Charge following time to ``name`` inside the innermost stage of every active timer."""
    for timer in active():
        try:
            timer.switch(name)
        except Exception:
            pass


def staged(name):
    """Decorator form of :func:`stage`; the wrapped function keeps its signature."""
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return wrapper
    return decorate


_register_with_process_guard()
