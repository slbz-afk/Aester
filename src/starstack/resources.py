"""Resource governor core (efficiency P1b-2): machine facts, two budgets, footprints, Siril plans -- SHADOW ONLY.

docs/EFFICIENCY_P1B_PREREG_20261001.md (P1b-2, §8 U-G1/U-G4), design p1b/design/governor/final.md, cross-check X1-X17.

What this module does in P1b:
- reads machine facts as seen by this process (memory and commit, processor groups, cores and efficiency classes,
  page-file mode, volume types, power state, the Aester process tree's resident memory);
- computes two budgets, RAM (working set) and commit, because Siril's per-launch OpenMP/OpenBLAS pool is commit only;
- predicts per-stage footprints from a calibrated table (Siril: pool + min(units x per-unit, k x allowance));
- recommends worker counts for Python pools and threads/affinity/environment for Siril launches;
- in **shadow mode** writes those recommendations, with today's settings and predictions, into a ``governor`` note of
  its scope's StageTimer, i.e. only under ``timings``.

What it never does: change an environment variable, an affinity, a priority, a thread or worker count, a script line,
a child's environment or arguments, or write a file. Every shadow entry point swallows its own failures into the note's
``errors`` and returns None, so a job can never fail because of it. Recommendations are applied from P2 on.

Privacy (U-G4, user decision 2026-10-01, conservative): the product default ``shadow`` writes only recommendations and
rounded budget numbers -- no machine inventory (raw RAM / commit values, page-file mode, volume types or names, power
state, process lists). ``shadow-full`` (bench and development) also writes the full facts.

Native safety: variable-length Win32 structures are copied into ``bytes`` and parsed with ``struct`` under explicit
bounds checks; nothing is cast from a raw pointer. A malformed buffer yields "unknown". Standard library + ctypes only.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import threading
import time
import weakref
from dataclasses import asdict, dataclass, field, replace

from . import frame_pool   # usable_cpus / MAX_FRAME_WORKERS: one definition of usable CPUs (X2); stdlib only

VERSION = "aester-resources-1"
# Siril rows: the design's calibration "footprints-2026-09-30b" (178 points, unchanged); Python rows updated by the
# cross-check X3 (py.coverage measured 230 MiB, new py.blas_slot), then by P1c-0 (docs/EFFICIENCY_P1C_PREREG_20261005.md
# §2.2/§2.3: rows for the P1c pool sites with provisional unit sizes, py.convert_phase without U-M14's two copies, the
# T-2 override order of py.transient_scan). The version only reaches timings (the governor note).
FOOTPRINTS_VERSION = "footprints-2026-10-05a"

KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3
MB, GB = 10 ** 6, 10 ** 9
REFERENCE_PIXELS = 5520 * 8280          # 45.7 MP Z8: every pixel-scaled coefficient was measured at this size

MODE_ENV = "AESTER_GOVERNOR"            # shadow (default) | shadow-full | off ; anything else -> off with a note
FACTS_ENV = "AESTER_GOVERNOR_FACTS"     # bench/tests: what-if machine, e.g. total=15.4G,avail=9G,lps=16,cores=8,...
MODES = ("shadow", "shadow-full", "off")
DEFAULT_MODE = "shadow"                 # U-G1
NOTE_KEY = "governor"
JOB_SCOPE = "stack-job"
NOTE_CAP_BYTES = 32 * KiB               # per scope
JOB_NOTE_CAP_BYTES = 8 * KiB            # X15: the job note is copied into workflow.json checkpoints and snapshots
# X15: a cap holds for the note as stored at its deepest copy. records.atomic_json writes indent 2 in text mode
# (CRLF on Windows), so each line but the first carries 2 x depth more indent bytes than at depth 0, plus a CR.
# depth = the containers around the note's value:
#   stack-job  workflow.json checkpoints.stack.data.job.diagnostics.timing_attempts[i].notes.governor  9
#              (the current attempt's ...job.diagnostics.timings.notes.governor is 8; job.json and snapshots 4-5)
#   workflow   workflow.json timing_attempts[i].notes.governor  4  (timings.notes.governor 3)
NOTE_DEPTH = {JOB_SCOPE: 9, "workflow": 4}
BYTES_FIELD_ROOM = 64                   # the trailing "bytes" line at depth 9 is about 40 bytes
MAX_ERRORS = 12


# =============================================================================================================== #
# Windows API (bound lazily; tests replace _kernel32 with failing stubs)                                          #
# =============================================================================================================== #

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


class _PMC_EX2(ctypes.Structure):   # PROCESS_MEMORY_COUNTERS_EX2 (the EX prefix is PROCESS_MEMORY_COUNTERS_EX)
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t), ("PrivateWorkingSetSize", ctypes.c_size_t),
                ("SharedCommitUsage", ctypes.c_uint64)]


_PMC_EX_SIZE = _PMC_EX2.PrivateWorkingSetSize.offset


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", ctypes.c_uint32), ("cntUsage", ctypes.c_uint32), ("th32ProcessID", ctypes.c_uint32),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", ctypes.c_uint32),
                ("cntThreads", ctypes.c_uint32), ("th32ParentProcessID", ctypes.c_uint32),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", ctypes.c_uint32), ("szExeFile", ctypes.c_wchar * 260)]


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


class _Api:
    """kernel32 entry points with explicit prototypes, on a private WinDLL instance (no shared argtypes)."""

    def __init__(self):
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        HANDLE, BOOL, DWORD = ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32

        def bind(name, restype, *argtypes):
            function = getattr(k, name)
            function.restype, function.argtypes = restype, list(argtypes)
            return function

        self.GlobalMemoryStatusEx = bind("GlobalMemoryStatusEx", BOOL, ctypes.POINTER(_MEMORYSTATUSEX))
        self.GetCurrentProcess = bind("GetCurrentProcess", HANDLE)
        self.K32GetProcessMemoryInfo = bind("K32GetProcessMemoryInfo", BOOL, HANDLE, ctypes.c_void_p, DWORD)
        self.GetLogicalProcessorInformationEx = bind("GetLogicalProcessorInformationEx", BOOL, ctypes.c_int,
                                                     ctypes.c_void_p, ctypes.POINTER(DWORD))
        self.GetActiveProcessorGroupCount = bind("GetActiveProcessorGroupCount", ctypes.c_uint16)
        self.GetActiveProcessorCount = bind("GetActiveProcessorCount", DWORD, ctypes.c_uint16)
        self.GetProcessGroupAffinity = bind("GetProcessGroupAffinity", BOOL, HANDLE, ctypes.POINTER(ctypes.c_uint16),
                                            ctypes.POINTER(ctypes.c_uint16))
        self.GetProcessAffinityMask = bind("GetProcessAffinityMask", BOOL, HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t))
        self.GetDriveTypeW = bind("GetDriveTypeW", ctypes.c_uint32, ctypes.c_wchar_p)
        self.GetVolumeInformationW = bind("GetVolumeInformationW", BOOL, ctypes.c_wchar_p, ctypes.c_wchar_p, DWORD,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, DWORD)
        self.GetDiskFreeSpaceExW = bind("GetDiskFreeSpaceExW", BOOL, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint64),
                                        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64))
        self.CreateFileW = bind("CreateFileW", HANDLE, ctypes.c_wchar_p, DWORD, DWORD, ctypes.c_void_p, DWORD, DWORD,
                                HANDLE)
        self.DeviceIoControl = bind("DeviceIoControl", BOOL, HANDLE, DWORD, ctypes.c_void_p, DWORD, ctypes.c_void_p,
                                    DWORD, ctypes.POINTER(DWORD), ctypes.c_void_p)
        self.CloseHandle = bind("CloseHandle", BOOL, HANDLE)
        self.GetSystemPowerStatus = bind("GetSystemPowerStatus", BOOL, ctypes.POINTER(_SYSTEM_POWER_STATUS))
        self.CreateToolhelp32Snapshot = bind("CreateToolhelp32Snapshot", HANDLE, DWORD, DWORD)
        self.Process32FirstW = bind("Process32FirstW", BOOL, HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
        self.Process32NextW = bind("Process32NextW", BOOL, HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
        self.OpenProcess = bind("OpenProcess", HANDLE, DWORD, BOOL, DWORD)
        self.GetProcessTimes = bind("GetProcessTimes", BOOL, HANDLE, *([ctypes.POINTER(ctypes.c_uint64)] * 4))


_api_lock = threading.Lock()
_api_instance = None


def _kernel32():
    """The bound API, or None off Windows / when binding failed (every reader then returns unknowns)."""
    global _api_instance
    if os.name != "nt":
        return None
    with _api_lock:
        if _api_instance is None:
            try:
                _api_instance = _Api()
            except Exception:
                _api_instance = False
    return _api_instance or None


_INVALID_HANDLE = ctypes.c_void_p(-1).value


# =============================================================================================================== #
# Machine facts                                                                                                    #
# =============================================================================================================== #

@dataclass(frozen=True)
class MemorySnapshot:
    """GlobalMemoryStatusEx and this process, 'as seen by this process' (commit values are the process view)."""
    total_physical: int | None
    available_physical: int | None
    commit_limit: int | None          # ullTotalPageFile: min(system, job) commit limit
    commit_available: int | None      # ullAvailPageFile: what this process may still commit
    own_private: int | None
    own_working_set: int | None
    taken_at: float


@dataclass(frozen=True)
class CpuTopology:
    logical_total: int | None
    groups: tuple                     # LPs per active processor group
    physical_cores: int | None
    smt: bool | None
    efficiency_classes: dict          # class -> cores (higher = faster)
    cores: tuple                      # ((group, efficiency class, (LP numbers within the group, ...)), ...)
    process_affinity: dict | None     # {"group", "mask"} when explicit and narrower than the group, else None
    python_usable_lps: int | None     # frame_pool.usable_cpus() (X2)
    python_usable_cores: int | None
    siril_group_lps: int | None       # LPs of the group a child starts in (Siril's OpenMP sees <= 64)
    siril_threads_today: int | None   # popcount(affinity mask) if explicit, else siril_group_lps
    siril_pool_today: int | None      # this process's pool knob (siril_pool_setting) if set, else siril_group_lps


@dataclass(frozen=True)
class VolumeFacts:
    role: str                         # source | work | output
    root: str | None
    drive_type: str                   # fixed | removable | remote | cdrom | ramdisk | unknown
    file_system: str | None
    free_bytes: int | None
    bus_type: str | None              # "full" level only (storage IOCTL)
    seek_penalty: bool | None         # "full" level only
    io_class: str                     # nvme | ssd | hdd | removable | network | fixed | unknown
    status: str                       # ok | skipped_remote | timeout | error | no_path


@dataclass(frozen=True)
class MachineFacts:
    memory: MemorySnapshot
    cpu: CpuTopology
    pagefile: str                     # system_managed | fixed | none | unknown
    power: dict                       # {"on_mains": bool|None, "has_battery": bool|None}
    volumes: tuple
    level: str                        # light | full | whatif
    override: dict | None = None      # what-if facts only


def memory_snapshot() -> MemorySnapshot:
    """Fresh each call (a few microseconds); unknown fields are None."""
    values = dict.fromkeys(("total_physical", "available_physical", "commit_limit", "commit_available",
                            "own_private", "own_working_set"))
    api = _kernel32()
    if api is not None:
        try:
            status = _MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if api.GlobalMemoryStatusEx(ctypes.byref(status)):
                values.update(total_physical=int(status.ullTotalPhys), available_physical=int(status.ullAvailPhys),
                              commit_limit=int(status.ullTotalPageFile), commit_available=int(status.ullAvailPageFile))
        except Exception:
            pass
        try:
            own = _process_memory(api, api.GetCurrentProcess())
        except Exception:
            own = None
        if own:
            values.update(own_private=own["private"], own_working_set=own["ws"])
    elif os.name != "nt":
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            values.update(total_physical=os.sysconf("SC_PHYS_PAGES") * page,
                          available_physical=os.sysconf("SC_AVPHYS_PAGES") * page)
        except (AttributeError, OSError, ValueError):
            pass
    return MemorySnapshot(taken_at=time.monotonic(), **values)


def _process_memory(api, handle) -> dict | None:
    """{"private", "ws", "peak_ws", "peak_commit", "shared_commit"} of one process handle, or None."""
    if not handle:
        return None
    try:
        counters = _PMC_EX2()
        for size in (ctypes.sizeof(_PMC_EX2), _PMC_EX_SIZE):
            counters.cb = size
            if api.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), size):
                return {"private": int(counters.PrivateUsage), "ws": int(counters.WorkingSetSize),
                        "peak_ws": int(counters.PeakWorkingSetSize), "peak_commit": int(counters.PeakPagefileUsage),
                        "shared_commit": int(counters.SharedCommitUsage) if size == ctypes.sizeof(_PMC_EX2) else None}
    except Exception:
        pass
    return None


# ---- processor topology ---------------------------------------------------------------------------------------- #

RELATION_PROCESSOR_CORE = 0
LTP_PC_SMT = 1


def parse_processor_cores(raw) -> list | None:
    """[(group, efficiency class, (LP numbers in the group), smt flag)] from the bytes of
    GetLogicalProcessorInformationEx(RelationProcessorCore); None when the buffer is malformed (never raises)."""
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    raw = bytes(raw)
    cores, offset = [], 0
    try:
        while offset < len(raw):
            if offset + 8 > len(raw):
                return None
            relation, size = struct.unpack_from("<II", raw, offset)
            if size < 8 or offset + size > len(raw):
                return None
            if relation == RELATION_PROCESSOR_CORE:
                # PROCESSOR_RELATIONSHIP: Flags BYTE, EfficiencyClass BYTE, Reserved[20], GroupCount WORD,
                # GroupMask[GroupCount] of GROUP_AFFINITY {KAFFINITY Mask (8 bytes), WORD Group, WORD Reserved[3]}
                if size < 32 + 16:
                    return None
                flags, efficiency = raw[offset + 8], raw[offset + 9]
                group_count = struct.unpack_from("<H", raw, offset + 30)[0]
                if group_count < 1 or 32 + 16 * group_count > size:
                    return None
                for index in range(group_count):
                    mask, group = struct.unpack_from("<QH", raw, offset + 32 + 16 * index)
                    lps = tuple(bit for bit in range(64) if mask >> bit & 1)
                    if not lps:
                        return None
                    cores.append((int(group), int(efficiency), lps, bool(flags & LTP_PC_SMT)))
            offset += size
    except struct.error:
        return None
    return cores or None


def _read_core_buffer(api) -> bytes | None:
    needed = ctypes.c_uint32(0)
    api.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None, ctypes.byref(needed))
    if not needed.value or needed.value > 64 * MiB:
        return None
    buffer = ctypes.create_string_buffer(needed.value)
    if not api.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buffer, ctypes.byref(needed)):
        return None
    return buffer.raw[:needed.value]


def _process_group_affinity(api) -> tuple[list[int] | None, int | None]:
    """(groups of this process, its affinity mask within its group when it has exactly one), Nones when unknown."""
    try:
        process = api.GetCurrentProcess()
        count = ctypes.c_uint16(64)
        array = (ctypes.c_uint16 * 64)()
        groups = None
        if api.GetProcessGroupAffinity(process, ctypes.byref(count), array) and 0 < count.value <= 64:
            groups = [int(array[i]) for i in range(count.value)]
        mask = None
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if api.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            mask = int(process_mask.value)
        return groups, mask
    except Exception:
        return None, None


def build_topology(cores, group_lps, process_groups, process_mask, *, usable_lps, omp=None) -> CpuTopology:
    """Pure: the topology record from parsed cores, LPs per group and this process's group affinity."""
    cores = list(cores or [])
    groups = tuple(int(n) for n in (group_lps or ()))
    logical_total = sum(groups) if groups else (sum(len(c[2]) for c in cores) or None)
    classes = {}
    for _, efficiency, _, _ in cores:
        classes[efficiency] = classes.get(efficiency, 0) + 1
    smt = any(c[3] or len(c[2]) > 1 for c in cores) if cores else None
    home = process_groups[0] if process_groups else 0
    group_size = groups[home] if home < len(groups) else (groups[0] if groups else None)
    group_full = ((1 << group_size) - 1) if group_size else None
    affinity = None
    if process_groups and len(process_groups) == 1 and process_mask and group_full and process_mask != group_full:
        affinity = {"group": home, "mask": int(process_mask)}
    if affinity is not None:
        mask = affinity["mask"]
        usable_cores = sum(1 for g, _, lps, _ in cores if g == home and any(mask >> lp & 1 for lp in lps)) or None
        siril_threads = bin(mask).count("1")
    else:
        if usable_lps and logical_total and usable_lps >= logical_total:
            usable_cores = len(cores) or None
        elif cores and process_groups and len(process_groups) == 1:
            usable_cores = sum(1 for g, _, _, _ in cores if g == home) or None
        elif cores and usable_lps and logical_total:
            usable_cores = max(1, round(len(cores) * usable_lps / logical_total))
        else:
            usable_cores = len(cores) or None
        siril_threads = group_size
    siril_group = min(64, group_size) if group_size else None
    siril_threads = min(64, siril_threads) if siril_threads else siril_group
    pool = _positive_int(omp)
    return CpuTopology(logical_total=logical_total, groups=groups, physical_cores=len(cores) or None, smt=smt,
                       efficiency_classes=dict(sorted(classes.items())),
                       cores=tuple((g, e, lps) for g, e, lps, _ in cores), process_affinity=affinity,
                       python_usable_lps=usable_lps, python_usable_cores=usable_cores, siril_group_lps=siril_group,
                       siril_threads_today=siril_threads, siril_pool_today=pool or siril_group)


def _positive_int(value) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


# The variables that size Siril's OpenMP/OpenBLAS pool, first match wins. Only OMP_NUM_THREADS: the bundled
# libopenblas.dll is an OpenMP build (it imports libgomp-1.dll, GOMP_parallel), whose threads follow OpenMP, and
# OPENBLAS_NUM_THREADS=1 alone left the pool at 64 (P1b review SIRIL-1 fix, measured through run_script: 15.06 GB
# commit for 8 Z8 frames, as with nothing set, against 6.58 GB with OMP_NUM_THREADS=1; siril.py U-G3 comment).
# A Siril built against a pthreads OpenBLAS would read OPENBLAS_NUM_THREADS first: put it in front here.
SIRIL_POOL_ENV = ("OMP_NUM_THREADS",)


def siril_pool_setting(environment) -> tuple[int | None, str | None]:
    """(pool, variable) of a Siril process started with ``environment``: the OpenMP/OpenBLAS pool (128 MiB commit per
    pool thread, F1) is the first positive SIRIL_POOL_ENV variable (OMP_NUM_THREADS; OPENBLAS_NUM_THREADS does not
    size this Siril's pool); (None, None) when none is set (Siril's default: the logical processors of its group).
    Read it from the block Siril actually receives (siril.siril_environment), never from this process's own."""
    for name in SIRIL_POOL_ENV:
        value = _positive_int((environment or {}).get(name))
        if value:
            return value, name
    return None, None


_topology_cache = None
_topology_lock = threading.Lock()


def cpu_topology(*, refresh=False) -> CpuTopology:
    """Cached per process (about 0.2 ms the first time); unknowns are None."""
    global _topology_cache
    with _topology_lock:
        if _topology_cache is not None and not refresh:
            return _topology_cache
        cores, group_lps, process_groups, process_mask = None, None, None, None
        api = _kernel32()
        if api is not None:
            try:
                cores = parse_processor_cores(_read_core_buffer(api))
            except Exception:
                cores = None
            try:
                count = int(api.GetActiveProcessorGroupCount())
                group_lps = [int(api.GetActiveProcessorCount(g)) for g in range(count)] if 0 < count <= 64 else None
            except Exception:
                group_lps = None
            process_groups, process_mask = _process_group_affinity(api)
        elif os.name != "nt":
            group_lps = [os.cpu_count()] if os.cpu_count() else None
        try:
            usable = int(frame_pool.usable_cpus())
        except Exception:
            usable = None
        _topology_cache = build_topology(cores, group_lps, process_groups, process_mask, usable_lps=usable,
                                         omp=siril_pool_setting(os.environ)[0])
        return _topology_cache


def affinity_mask(cores, n, *, group) -> int:
    """Route A mask (pure): ``n`` LPs of ``group`` -- cores sorted by efficiency class (highest first), then core
    order; the first LP of every core before any SMT sibling. Never more than 64 LPs or outside the group."""
    members = [(index, efficiency, tuple(lps)) for index, (g, efficiency, lps) in enumerate(cores) if g == group]
    members.sort(key=lambda item: (-item[1], item[0]))
    order = []
    depth = max((len(lps) for _, _, lps in members), default=0)
    for level in range(depth):
        order += [lps[level] for _, _, lps in members if level < len(lps) and 0 <= lps[level] < 64]
    mask = 0
    for lp in order[:max(0, min(int(n), 64))]:
        mask |= 1 << lp
    return mask


# ---- volumes, page file, power ---------------------------------------------------------------------------------- #

_DRIVE_TYPES = {0: "unknown", 1: "unknown", 2: "removable", 3: "fixed", 4: "remote", 5: "cdrom", 6: "ramdisk"}
DRIVE_REMOTE = 4
IOCTL_STORAGE_QUERY_PROPERTY = 0x2D1400
_BUS_TYPES = {0: "unknown", 1: "scsi", 2: "atapi", 3: "ata", 4: "1394", 5: "ssa", 6: "fibre", 7: "usb", 8: "raid",
              9: "iscsi", 10: "sas", 11: "sata", 12: "sd", 13: "mmc", 14: "virtual", 15: "file_backed_virtual",
              16: "spaces", 17: "nvme", 18: "scm", 19: "ufs"}
IO_WORKERS = {"nvme": 4, "ssd": 2, "hdd": 1, "removable": 1, "network": 1, "fixed": 2, "unknown": 2}   # SYS-IO-7
VOLUME_TIMEOUT = 0.2
VOLUME_CACHE_SECONDS = 30.0


def volume_root(path) -> str | None:
    """'C:\\' or '\\\\server\\share\\' of a path (string operation only; no I/O)."""
    try:
        text = os.path.abspath(os.fspath(path)) if os.name == "nt" else os.fspath(path)
    except (TypeError, ValueError, OSError):
        return None
    if os.name != "nt":
        return "/"
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\") or text.startswith("\\\\.\\"):
        text = text[4:]
    if text.startswith("\\\\"):
        parts = text[2:].split("\\")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return "\\\\" + parts[0] + "\\" + parts[1] + "\\"
        return None
    if len(text) >= 2 and text[1] == ":":
        return text[0].upper() + ":\\"
    return None


def parse_storage_descriptor(raw) -> str | None:
    """STORAGE_DEVICE_DESCRIPTOR bytes -> bus type name (BusType at offset 28), None when malformed."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 32:
        return None
    version, size = struct.unpack_from("<II", raw, 0)
    if size < 32 or version < 32:
        return None
    return _BUS_TYPES.get(struct.unpack_from("<I", raw, 28)[0], "other")


def parse_seek_penalty(raw) -> bool | None:
    """DEVICE_SEEK_PENALTY_DESCRIPTOR bytes -> IncursSeekPenalty (offset 8), None when malformed."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 9:
        return None
    version, size = struct.unpack_from("<II", raw, 0)
    if size < 9 or version < 9:
        return None
    return bool(raw[8])


def io_class(drive_type: str, bus_type: str | None = None, seek_penalty: bool | None = None) -> str:
    if drive_type == "remote":
        return "network"
    if drive_type in ("removable", "cdrom") or bus_type in ("usb", "sd", "mmc"):
        return "removable"
    if bus_type == "nvme":
        return "nvme"
    if seek_penalty is False or drive_type == "ramdisk":
        return "ssd"
    if seek_penalty is True:
        return "hdd"
    return "fixed" if drive_type == "fixed" else "unknown"


def _drive_type(root: str) -> int | None:
    if root.startswith("\\\\"):
        return DRIVE_REMOTE       # a share: never queried (its server may sleep or be gone)
    api = _kernel32()
    if api is None:
        return None
    try:
        return int(api.GetDriveTypeW(root))
    except Exception:
        return None


def drive_type(path) -> str:
    """'removable' | 'fixed' | 'remote' | 'cdrom' | 'ramdisk' | 'unknown' of the volume holding ``path``, from
    GetDriveType alone: no device handle, no helper thread, never raises (a share is 'remote' without a query).
    Efficiency P1c F-2 caps its pool on removable and network volumes."""
    root = volume_root(path)
    code = _drive_type(root) if root is not None else None
    return _DRIVE_TYPES.get(code, "unknown") if code is not None else "unknown"


def _storage_property(api, root: str, prop: int) -> bytes | None:
    if len(root) != 3 or root[1] != ":":
        return None
    handle = api.CreateFileW("\\\\.\\" + root[:2], 0, 3, None, 3, 0, None)   # no access rights, share r/w, open
    if not handle or handle == _INVALID_HANDLE:
        return None
    try:
        query = struct.pack("<II4x", prop, 0)                              # STORAGE_PROPERTY_QUERY
        buffer = ctypes.create_string_buffer(1024)
        returned = ctypes.c_uint32(0)
        if not api.DeviceIoControl(handle, IOCTL_STORAGE_QUERY_PROPERTY, query, len(query), buffer, 1024,
                                   ctypes.byref(returned), None):
            return None
        return buffer.raw[:min(returned.value, 1024)]
    finally:
        api.CloseHandle(handle)


def _volume_query(root: str, level: str) -> dict:
    """Runs on a helper thread: file system and free space; at level 'full' also the storage IOCTL."""
    out = {"file_system": None, "free_bytes": None, "bus_type": None, "seek_penalty": None}
    api = _kernel32()
    if api is None:
        return out
    name = ctypes.create_unicode_buffer(64)
    if api.GetVolumeInformationW(root, None, 0, None, None, None, name, 64):
        out["file_system"] = name.value or None
    free, total, total_free = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
    if api.GetDiskFreeSpaceExW(root, ctypes.byref(free), ctypes.byref(total), ctypes.byref(total_free)):
        out["free_bytes"] = int(free.value)
    if level == "full":
        out["bus_type"] = parse_storage_descriptor(_storage_property(api, root, 0))      # StorageDeviceProperty
        out["seek_penalty"] = parse_seek_penalty(_storage_property(api, root, 7))        # ...SeekPenaltyProperty
    return out


def _bounded(function, timeout: float):
    """(status, value): runs ``function`` on a daemon helper thread; on timeout the thread is abandoned (it only reads)."""
    box = {}

    def run():
        try:
            box["value"] = function()
        except BaseException as exc:   # noqa: BLE001 - reported as status
            box["error"] = exc

    thread = threading.Thread(target=run, name="aester-volume-query", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return "timeout", None
    if "error" in box:
        return "error", None
    return "ok", box.get("value")


_volume_cache = {}
_volume_lock = threading.Lock()


def volume_facts(path, role, *, level="light", timeout=VOLUME_TIMEOUT) -> VolumeFacts:
    """Facts of the volume holding ``path`` (cached per root and level for 30 s). Level 'light' never opens a device
    handle; a remote volume is never queried beyond its drive type."""
    if path is None:
        return VolumeFacts(role, None, "unknown", None, None, None, None, "unknown", "no_path")
    root = volume_root(path)
    if root is None:
        return VolumeFacts(role, None, "unknown", None, None, None, None, "unknown", "error")
    key = (root, level)
    now = time.monotonic()
    with _volume_lock:
        cached = _volume_cache.get(key)
    if cached is not None and now - cached[0] < VOLUME_CACHE_SECONDS:
        return replace(cached[1], role=role)
    code = _drive_type(root)
    drive = _DRIVE_TYPES.get(code, "unknown") if code is not None else "unknown"
    if code == DRIVE_REMOTE:
        facts = VolumeFacts(role, root, "remote", None, None, None, None, "network", "skipped_remote")
    else:
        status, value = _bounded(lambda: _volume_query(root, level), timeout)
        value = value or {}
        facts = VolumeFacts(role, root, drive, value.get("file_system"), value.get("free_bytes"), value.get("bus_type"),
                            value.get("seek_penalty"), io_class(drive, value.get("bus_type"), value.get("seek_penalty")),
                            status)
    if facts.status in ("ok", "skipped_remote"):
        with _volume_lock:
            _volume_cache[key] = (now, facts)
    return facts


def pagefile_mode(entries) -> str:
    """'system_managed' | 'fixed' | 'none' | 'unknown' from the PagingFiles REG_MULTI_SZ strings."""
    if entries is None:
        return "unknown"
    if isinstance(entries, str):
        entries = [entries]
    items = [str(item).strip() for item in entries if str(item).strip()]
    if not items:
        return "none"
    kinds = set()
    for item in items:
        parts = item.split()
        if parts[0].startswith("?:"):
            kinds.add("system_managed")
            continue
        sizes = [_positive_int(p) if p != "0" else 0 for p in parts[1:3]]
        if len(sizes) < 2 or sizes == [0, 0] or None in sizes:
            kinds.add("system_managed")
        else:
            kinds.add("fixed")
    return "system_managed" if "system_managed" in kinds else "fixed"


def read_pagefile_mode() -> str:
    if os.name != "nt":
        return "unknown"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management") as key:
            return pagefile_mode(winreg.QueryValueEx(key, "PagingFiles")[0])
    except (OSError, ImportError):
        return "unknown"


def power_state() -> dict:
    out = {"on_mains": None, "has_battery": None}
    api = _kernel32()
    if api is None:
        return out
    try:
        status = _SYSTEM_POWER_STATUS()
        if api.GetSystemPowerStatus(ctypes.byref(status)):
            out["on_mains"] = None if status.ACLineStatus == 255 else status.ACLineStatus == 1
            out["has_battery"] = None if status.BatteryFlag == 255 else not (status.BatteryFlag & 128)
    except Exception:
        pass
    return out


# ---- the Aester process tree ------------------------------------------------------------------------------------ #

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TH32CS_SNAPPROCESS = 0x2


def is_aester_image(name) -> bool:
    name = str(name or "").lower()
    return name in ("python.exe", "pythonw.exe") or (name.startswith("aester") and name.endswith(".exe"))


def tree_members(snapshot: dict, me: int, created) -> tuple[int, list[int]]:
    """Pure: (root, members) of the Aester tree around ``me``.

    ``snapshot`` maps pid -> (parent pid, image name); ``created(pid)`` is a creation time or None. The root is found
    by walking parents while the parent is an Aester image (python.exe / pythonw.exe / Aester*.exe) that started no
    later than its child (guards PID reuse). Members are the root and every descendant whose creation is not earlier
    than its parent's.
    """
    root, seen = me, {me}
    while root in snapshot:
        parent = snapshot[root][0]
        if parent in seen or parent not in snapshot or not is_aester_image(snapshot[parent][1]):
            break
        parent_created, child_created = created(parent), created(root)
        if parent_created is None or child_created is None or parent_created > child_created:
            break
        seen.add(parent)
        root = parent
    children = {}
    for pid, (ppid, _) in snapshot.items():
        if pid != ppid:
            children.setdefault(ppid, []).append(pid)
    members, todo, visited = [], [root], set()
    while todo:
        pid = todo.pop()
        if pid in visited:
            continue
        visited.add(pid)
        members.append(pid)
        parent_created = created(pid)
        for child in children.get(pid, ()):
            child_created = created(child)
            if parent_created is not None and child_created is not None and child_created >= parent_created:
                todo.append(child)
    return root, members


def _process_snapshot(api) -> dict:
    handle = api.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not handle or handle == _INVALID_HANDLE:
        return {}
    out = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = api.Process32FirstW(handle, ctypes.byref(entry))
        while ok and len(out) < 100_000:
            out[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), str(entry.szExeFile))
            ok = api.Process32NextW(handle, ctypes.byref(entry))
    finally:
        api.CloseHandle(handle)
    return out


def aester_resident(*, detail=False) -> dict:
    """Working set and commit (private bytes) of the whole Aester tree (about 12 ms); ``detail`` adds members with
    their SharedCommitUsage, unaggregated (X17). Unknown -> {"ws": None, "commit": None}."""
    api = _kernel32()
    out = {"ws": None, "commit": None, "members": 0, "root_exe": None, "root_is_self": None}
    if api is None:
        return out
    try:
        snapshot = _process_snapshot(api)
        info = {}

        def probe(pid):
            if pid not in info:
                handle = api.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                created = memory = None
                if handle:
                    try:
                        times = [ctypes.c_uint64() for _ in range(4)]
                        if api.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                            created = int(times[0].value)
                        memory = _process_memory(api, handle)
                    finally:
                        api.CloseHandle(handle)
                info[pid] = (created, memory)
            return info[pid]

        me = os.getpid()
        if me not in snapshot:
            return out
        root, members = tree_members(snapshot, me, lambda pid: probe(pid)[0])
        ws = commit = 0
        rows = []
        for pid in members:
            memory = probe(pid)[1]
            if memory:
                ws += memory["ws"]
                commit += memory["private"]
            if detail:
                rows.append({"pid": pid, "exe": snapshot.get(pid, (0, None))[1],
                             "ws": memory and memory["ws"], "commit": memory and memory["private"],
                             "shared_commit": memory and memory["shared_commit"]})
        out.update(ws=ws, commit=commit, members=len(members), root_exe=snapshot.get(root, (0, None))[1],
                   root_is_self=root == me)
        if detail:
            out["detail"] = rows[:32]
    except Exception:
        return {"ws": None, "commit": None, "members": 0, "root_exe": None, "root_is_self": None}
    return out


def machine_facts(*, level="light", paths=()) -> MachineFacts:
    """All facts as seen by this process. ``paths``: [(role, path), ...]; level 'full' adds the storage IOCTL."""
    volumes = tuple(volume_facts(path, role, level=level) for role, path in paths)
    return MachineFacts(memory=memory_snapshot(), cpu=cpu_topology(), pagefile=read_pagefile_mode(),
                        power=power_state(), volumes=volumes, level=level)


# ---- what-if facts (bench / tests) ------------------------------------------------------------------------------ #

_SIZE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([KMGT]?)$", re.IGNORECASE)
_WHATIF_KEYS = {"total", "avail", "lps", "cores", "siril_pool", "commit", "commit_avail", "io", "resident_ws",
                "resident_commit"}


def _size(text) -> int:
    match = _SIZE.match(str(text).strip())
    if not match:
        raise ValueError(f"bad size {text!r}")
    return int(float(match.group(1)) * {"": 1, "K": KiB, "M": MiB, "G": GiB, "T": 1024 * GiB}[match.group(2).upper()])


def parse_whatif(text) -> MachineFacts | None:
    """What-if machine from 'total=15.4G,avail=9G,lps=16,cores=8,siril_pool=16,commit=24G,commit_avail=14G,io=nvme'
    (sizes in GiB/MiB/KiB with G/M/K, bytes without). None for an empty value; ValueError when malformed."""
    if text is None or not str(text).strip():
        return None
    values = {}
    for part in str(text).split(","):
        key, separator, value = part.partition("=")
        key = key.strip().lower()
        if not separator or key not in _WHATIF_KEYS or key in values or not value.strip():
            raise ValueError(f"{FACTS_ENV}: bad item {part.strip()!r}")
        values[key] = value.strip()
    for required in ("total", "avail", "lps", "cores"):
        if required not in values:
            raise ValueError(f"{FACTS_ENV}: missing {required}")
    total, avail = _size(values["total"]), _size(values["avail"])
    lps, cores = _positive_int(values["lps"]), _positive_int(values["cores"])
    if not lps or not cores or cores > lps or avail > total:
        raise ValueError(f"{FACTS_ENV}: inconsistent lps/cores/total/avail")
    commit = _size(values["commit"]) if "commit" in values else total
    commit_avail = _size(values["commit_avail"]) if "commit_avail" in values else avail
    io = values.get("io", "unknown").lower()
    if io not in IO_WORKERS:
        raise ValueError(f"{FACTS_ENV}: io must be one of {', '.join(IO_WORKERS)}")
    pool = _positive_int(values["siril_pool"]) if "siril_pool" in values else None
    per_core = max(1, lps // cores)
    synthetic = tuple((0, 0, tuple(range(i * per_core, i * per_core + per_core))) for i in range(cores))
    group = min(64, lps)
    cpu = CpuTopology(logical_total=lps, groups=(lps,), physical_cores=cores, smt=lps > cores,
                      efficiency_classes={0: cores}, cores=synthetic, process_affinity=None, python_usable_lps=lps,
                      python_usable_cores=cores, siril_group_lps=group, siril_threads_today=group,
                      siril_pool_today=pool or group)
    memory = MemorySnapshot(total, avail, commit, commit_avail, None, None, time.monotonic())
    volumes = tuple(VolumeFacts(role, None, "fixed", None, None, None, None, io, "whatif")
                    for role in ("source", "work", "output"))
    override = {"text": str(text).strip()[:200]}
    for key in ("resident_ws", "resident_commit"):
        if key in values:
            override[key] = _size(values[key])
    return MachineFacts(memory=memory, cpu=cpu, pagefile="unknown", power={"on_mains": None, "has_battery": None},
                        volumes=volumes, level="whatif", override=override)


# =============================================================================================================== #
# Budgets                                                                                                          #
# =============================================================================================================== #

PRESETS = {"save": 0.35, "balanced": 0.50, "full": None}       # share of physical RAM (plan decision 5, U5)
RAM_RESERVE_MIN, RAM_RESERVE_SHARE = 3 * GiB, 0.12
COMMIT_RESERVE_MIN, COMMIT_RESERVE_SHARE = 2 * GiB, 0.10       # of the commit limit [est]


@dataclass(frozen=True)
class Budget:
    preset: str
    ram_total: int | None             # working set the whole Aester tree may hold at once
    commit_total: int | None          # commit the whole Aester tree may hold at once
    ram_reserve: int | None
    commit_reserve: int | None
    resident_ws: int
    resident_commit: int
    cpu_lps: int | None               # Siril: LPs of its group Aester may keep busy
    python_cpu_cores: int | None      # Python pools: usable physical cores
    basis: dict = field(default_factory=dict)

    def ram_allowance(self, resident_ws=None) -> int | None:
        return None if self.ram_total is None else self.ram_total - (self.resident_ws if resident_ws is None else resident_ws)

    def commit_allowance(self, resident_commit=None) -> int | None:
        if self.commit_total is None:
            return None
        return self.commit_total - (self.resident_commit if resident_commit is None else resident_commit)


def budget(facts: MachineFacts, preset="balanced", *, resident=None) -> Budget:
    """RAM and commit budgets of the whole Aester tree (no floors: an allowance may be negative)."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}")
    resident = resident or {}
    resident_ws, resident_commit = int(resident.get("ws") or 0), int(resident.get("commit") or 0)
    m, cpu = facts.memory, facts.cpu
    basis = {"preset": preset}
    ram_total = ram_reserve = None
    if m.total_physical and m.available_physical is not None:
        ram_reserve = max(RAM_RESERVE_MIN, int(RAM_RESERVE_SHARE * m.total_physical))
        by_available = m.available_physical + resident_ws - ram_reserve
        share = PRESETS[preset]
        by_share = None if share is None else int(share * m.total_physical)
        ram_total = by_available if by_share is None else min(by_share, by_available)
        basis["ram_bound"] = "available" if by_share is None or by_available <= by_share else "share"
    commit_total = commit_reserve = None
    if m.commit_limit and m.commit_available is not None:
        commit_reserve = max(COMMIT_RESERVE_MIN, int(COMMIT_RESERVE_SHARE * m.commit_limit))
        commit_total = m.commit_available + resident_commit - commit_reserve
        basis["commit_bound"] = "available"
    lps = cpu.siril_group_lps
    cpu_lps = None if not lps else max(1, lps - (2 if lps >= 8 else 1))
    cores = cpu.python_usable_cores
    python_cores = None if not cores else max(1, cores - 1)
    return Budget(preset, ram_total, commit_total, ram_reserve, commit_reserve, resident_ws, resident_commit,
                  cpu_lps, python_cores, basis)


# =============================================================================================================== #
# Footprints                                                                                                       #
# =============================================================================================================== #

# py.transient_scan: one scan child's charge, the same numbers as transient_stage.SCAN_UNIT_BYTES / SCAN_BASE_BYTES
# (that module cannot be imported here; tests/test_resources_footprints.py pins the two equal at several frame sizes).
SCAN_UNIT_BYTES = 1_800_000_000        # at the 45.7 MP reference
SCAN_BASE_BYTES = 70_000_000           # not scaled by pixels


def _bpp(gigabytes: float) -> float:
    """Bytes per pixel of a value measured at the 45.7 MP reference frame."""
    return gigabytes * GB / REFERENCE_PIXELS


@dataclass(frozen=True)
class Footprint:
    key: str                          # "siril.convert", ..., "py.render_rows"
    unit: str                         # thread_or_frame | frame | thread | process | worker | file | session | caller
    commit_per_unit: tuple            # (central, high): bytes per pixel when scale == "pixels", per column when
                                      # "width", per frame x column when "frames_width", bytes otherwise
    ws_ratio: tuple                   # (central, high): WS / (commit - pool) for Siril parts; WS / commit for Python
    fixed: tuple                      # (commit, ws) per stage, same scale as commit_per_unit
    allowance_k: tuple | None         # Siril parts bounded by k x Siril's memory allowance: (central, high)
    scale: str                        # pixels | width | frames_width | blas_threads | none
    cap: int | None                   # never more units (speed plateau / Siril limit / pool design cap)
    kind: str                         # cpu | gil | io | siril  (gil is a P1d signal only; it never forces 1, X1)
    provenance: str
    pool: str = "none"                # frame (frame_pool) | rows | process | hash | session | siril | none
    units_rule: str = "items"         # Siril: min_tf (min(threads, frames)) | frames ; Python: items
    per_thread: tuple = (0.0, 0.0)    # stack_finite: bytes/px per frame per thread
    thread_fixed: tuple = (0.0, 0.0)  # stack_rej: bytes/px x min(1, threads/64)
    override_env: str | tuple | None = None   # a knob, or knobs in precedence order (the first one set decides)
    unit_base: tuple = (0.0, 0.0)     # (central, high) bytes per unit, not scaled: a process's interpreter and modules


SIRIL_POOL_BYTES = 128 * MiB          # per OpenMP/OpenBLAS pool thread, commit only (F1)
SIRIL_POOL_FIXED = 30 * MB
SIRIL_POOL_WS = 40 * MB
SIRIL_PARTS = ("convert", "probe", "export", "export_stack", "stack_finite", "stack_rej")
_S = "measured: 47 P0/P1H/P1A bench receipts + 16-frame F1/F3 runs (governor final.md §2.4, F8)"
FOOTPRINTS = {fp.key: fp for fp in (
    Footprint("siril.convert", "thread_or_frame", (_bpp(.86), _bpp(.95)), (.90, .95), (0, 0), (1.05, 1.20), "pixels",
              16, "siril", _S + "; in-sample central -5.5..+2.1 %", "siril", "min_tf"),
    Footprint("siril.probe", "thread_or_frame", (_bpp(.46), _bpp(.50)), (.98, 1.0), (0, 0), None, "pixels",
              32, "siril", _S + "; in-sample central -0.7..+11.3 %", "siril", "min_tf"),
    Footprint("siril.export", "thread_or_frame", (_bpp(2.0), _bpp(2.7)), (.955, 1.0), (0, 0), (1.0, 1.20), "pixels",
              64, "siril", _S + "; 26 export-only receipts + F1; high/measured 1.03..1.62", "siril", "min_tf"),
    Footprint("siril.export_stack", "thread_or_frame", (_bpp(2.0), _bpp(2.7)), (.95, 1.0), (0, 0), (1.0, 1.20),
              "pixels", 64, "siril", _S + "; 21 combined L receipts; high/measured 1.04..1.54", "siril", "min_tf"),
    Footprint("siril.stack_finite", "frame", (_bpp(.30), _bpp(.30) * 1.2), (1.0, 1.02), (0, 0), (1.0, 1.20),
              "pixels", 64, "siril", _S + "; frames x (0.30 + 0.0025 t) GB; high 1.08..1.42", "siril", "frames",
              per_thread=(_bpp(.0025), _bpp(.0025) * 1.2)),
    Footprint("siril.stack_rej", "thread_or_frame", (_bpp(.55), _bpp(.55) * 1.25), (.95, 1.0), (0, 0), (1.0, 1.20),
              "pixels", 64, "siril", _S + "; 13.6 GB min(1, t/64) + 0.55 GB min(t, F); 9 points (64-thread fit)",
              "siril", "min_tf", thread_fixed=(_bpp(13.6), _bpp(13.6) * 1.25)),
    Footprint("py.worker_base", "process", (0, 0), (1.0, 1.0), (36 * MB, 50 * MB), None, "blas_threads", None, "cpu",
              "measured F6: 0.036 / 0.541 / 0.811 GB commit at 1 / 16 / 64 BLAS threads (OpenBLAS MAX_THREADS 24); "
              "WS 0.05 GB", "none"),
    Footprint("py.blas_slot", "caller", (32 * MiB, 32 * MiB), (.0078, .0078), (0, 0), None, "none", None, "cpu",
              "measured: pool design final/f1 (32 MiB commit, 0.25 MB WS per extra peak concurrent BLAS caller, "
              "persistent) [X3]", "none"),
    Footprint("py.convert_phase", "stage", (0, 0), (1.0, 1.0), (_bpp(.84), _bpp(.84)), None, "pixels", None, "cpu",
              "est: P1a receipts (1.88 GB: C-1 median +0.61 GB, write_linear +1.04 GB two copies) less the two copies "
              "U-M14 removes (P1c §2.2); C-1's extra median threads are py.median", "none"),
    Footprint("py.median", "thread", (60 * MB / (48 * 8280), 80 * MB / (48 * 8280)), (1.0, 1.0), (0, 0), None,
              "frames_width", 4, "gil",
              "est: P1c §2.3 C-1 provisional (60 MB per 64-row block thread at n=48, W=8280, scaled by n x W); "
              "cap 4, the site also limits open handles to 1024 // n", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.sky_zero", "frame", (256 * 16, 256 * 30), (1.0, 1.0), (0, 0), None, "width", 4, "io",
              "est: P1c §2.3 C-3 provisional: 256 rows x W x (4 x (2 + digest queue 4 + 1) + 2) B = 63.6 MB for "
              "float32 frames (high), 256 x W x 16 B = 34 MB for 16-bit frames (central); cap 4", "frame",
              override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.native_remap", "rows", (50 * MB, 60 * MB), (1.0, 1.0), (24.6, 24.6), None, "pixels", 8, "gil",
              "est: geometry design §9 (50 MB per row range, 2.05 S fixed)", "process"),
    Footprint("py.native_mask", "rows", (190 * MiB / 8280, 200 * MiB / 8280), (1.0, 1.0), (0, 0), None, "width", 4,
              "cpu", "measured: P1c G-4t worst case (every footprint mixed, ~360 B per point): 183-196 MiB per 64-row "
              "block at W=8280, scaled by W (declared 200 MiB); cap 4 [provisional: swept 2/3/4 on a shared host, the "
              "idle 2/4/8 rescan of P1c §3.2 is open]",
              "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.strict_exports", "frame", (226 * MiB / REFERENCE_PIXELS, 256 * MiB / REFERENCE_PIXELS),
              (1.0, 1.0), (0, 0), None, "pixels", 4, "gil",
              "measured: P1c F-2 unit peak 175-182 MiB per export at 45.7 MP plus its held result (43.6 MiB, R17) = "
              "226 MiB, declared 256 MiB (BLAS slot: py.blas_slot); cap 4 [provisional: the 4/6/8 sweep ran on a shared "
              "host, the idle sweep and rule of P1c §3.2 are open]",
              "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.finite_write", "frame", (_bpp(.68), _bpp(.68)), (1.0, 1.0), (0, 0), None, "pixels", 4, "io",
              "est: P1c §3.3 F-3a streamed unit F + 2P + 6T = 0.68 GB at 45.7 MP (the reference body runs on one "
              "thread); cap 4", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.finite_inventory", "frame", (2.0 + 8 * MiB / REFERENCE_PIXELS, 2.0 + 8 * MiB / REFERENCE_PIXELS),
              (1.0, 1.0), (0, 0), None, "pixels", 4, "io",
              "est: P1c §2.3 F-3b provisional 2P + 8 MiB per frame (~100 MB at 45.7 MP); cap 4", "frame",
              override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.coverage", "frame", (230 * MiB / REFERENCE_PIXELS, 230 * MiB / REFERENCE_PIXELS), (1.0, 1.0),
              (0, 0), None, "pixels", 8, "cpu",
              "measured: pool design exp6 (218 MB peak + 10.9 MB result per unit, unit_bytes 230 MiB) [X3]; cap 8 (G-6)",
              "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.transient_scan", "process", ((SCAN_UNIT_BYTES - SCAN_BASE_BYTES) / REFERENCE_PIXELS,) * 2,
              (1.0, 1.0), (0, 0), None, "pixels", 8, "cpu",
              "measured: P1c T-2 worst case (tests/test_p1c_site_transient_scan.py::"
              "test_worst_case_child_peak_is_within_the_charged_unit, 2026-10-06: peak commit 1.635 GB at 45.7 MP, "
              "61.4-61.6 MB at 64 x 96 px), charged as transient_stage.scan_unit_bytes: 70 MB a process plus 1.73 GB "
              "per 45.7 MP (1.8 GB at 45.7 MP, about 10 % margin; typical P1a 0.55-0.56 GB); cap 8; overrides in the "
              "T-2 order (P1c §3.4)",
              "process", override_env=("STARSTACK_TRANSIENT_WORKERS", "AESTER_FRAME_WORKERS"),
              unit_base=(SCAN_BASE_BYTES, SCAN_BASE_BYTES)),
    Footprint("py.render_rows", "worker", ((128 + 32) * 234, (128 + 32) * 234), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "measured at one sensor width (3 vs 8 workers: 4.94 vs 6.24-6.51 GB): W x 160 rows x 234 B; "
              "cap 8 (MAX_RENDER_WORKERS, T-5)", "rows", override_env="AESTER_RENDER_WORKERS"),
    Footprint("py.trail_crops", "frame", (32 * MiB, 32 * MiB), (1.0, 1.0), (0, 0), None, "none", 8, "gil",
              "est: P1c §2.3 T-3 provisional 32 MiB per frame's candidates (after T-3m reads only sampled pixels); "
              "cap 8", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.milkyway_render", "rows", (130 * MB / 8280, 130 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "est: P1c §3.5 T-5a provisional ~130 MB per 128-row block at W=8280, kept as the bound over the "
              "F48 serial unit peak measured 110.1 MB; scaled by W; cap 8", "frame",
              override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.star_reduction", "rows", (45 * MB / 8280, 60 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "est: P1c §3.5 T-5b provisional 45-60 MB per 16-row block at W=8280 and r=8, kept as the bound "
              "over the F48 serial unit peak measured 28.6 MB (63.4 MB at r=32, within 3 x 45 MB); callers pass "
              "width = W x (16 + 4 r) / 48 for the halo; cap 8", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.trail_pair", "rows", (210 * MB / 8280, 210 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "est: P1c §3.5 T-5c provisional ~210 MB per 128-row block at W=8280, kept as the bound over the "
              "F48 serial unit peak measured 175.6 MB; scaled by W; cap 8", "frame",
              override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.star_layers", "rows", (16 * MB / 8280, 16 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "measured: P1c §3.5 T-5x (a) F48 serial unit peak 8.9 MB per 16-row layer block at W=8280 "
              "(bound 16 MB), scaled by W; cap 8", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.starless_scope", "rows", (45 * MB / 8280, 45 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "measured: P1c §3.5 T-5x (b) F48 serial unit peak 38.4 MB per 128-row starless scope block at "
              "W=8280 (bound 45 MB), scaled by W; the advice residual check's 64-row blocks (19.2 MB, bound 24 MB) "
              "pass half the width; cap 8", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.jpeg_rows", "rows", (32 * MB / 8280, 32 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "measured: P1c §3.5 T-5x (c) F48 serial unit peak 25.6 MB per 128-row 16-to-8-bit JPEG block "
              "at W=8280 (bound 32 MB), scaled by W; cap 8", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.inward_weight", "rows", (120 * MB / 8280, 120 * MB / 8280), (1.0, 1.0), (0, 0), None, "width", 8,
              "cpu", "measured: P1c §3.5 T-5x (d) F48-size serial unit peak 72.7 MB per 256-row tile at W=8280, "
              "width_px=128 (514 rows read; 181.0 MB at width_px=512; 106.8 / 266.1 MB before the float32 feather "
              "start) (bound 120 MB); callers pass width = W x (256 + 2 (width_px + 1)) / 514; cap 8", "frame",
              override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.addback_boxes", "frame", (12.0, 12.0), (1.0, 1.0), (0, 0), None, "pixels", 2, "gil",
              "est placeholder (one S per stem) until T-7's measure-first step sizes it by box area; cap 2 provisional "
              "(P1c §3.5: the measured best, most likely 2)", "frame", override_env="AESTER_FRAME_WORKERS"),
    Footprint("py.hash", "file", (8 * MiB, 8 * MiB), (1.0, 1.0), (0, 0), None, "none", 32, "io",
              "code: hashing.py 8 MiB buffer per file in flight; cap 32 (hashing.MAX_WORKERS)", "hash",
              override_env="AESTER_HASH_WORKERS"),
    Footprint("py.nox_session", "session", (800 * MB, 1000 * MB), (1.0, 1.0), (0, 0), None, "none", 1, "cpu",
              "est: plan findings (P3 owns nox threads and sessions)", "session"),
)}
SIRIL_FOOTPRINT = {part: FOOTPRINTS["siril." + part] for part in SIRIL_PARTS}


def blas_base_commit(blas_threads) -> int:
    """py.worker_base commit at a BLAS thread count (OpenBLAS keeps at most 24 thread buffers)."""
    threads = max(1, min(int(blas_threads or 1), 24))
    return 36 * MB + 32 * MiB * (threads - 1)


REFERENCE_WIDTH, REFERENCE_FRAMES = 8280, 48   # the 45.7 MP Z8 width and the 48-frame baseline


def footprint_bytes(fp: Footprint, units, *, frame_pixels=None, width=None, blas_threads=None, frames=None) -> dict:
    """{"commit": (central, high), "ws": (central, high)} for ``units`` of a Python stage (fixed part included)."""
    units = max(0, int(units))
    if fp.scale == "pixels":
        scale = frame_pixels or REFERENCE_PIXELS
    elif fp.scale == "width":
        scale = width or REFERENCE_WIDTH
    elif fp.scale == "frames_width":
        scale = (frames or REFERENCE_FRAMES) * (width or REFERENCE_WIDTH)
    else:
        scale = 1
    if fp.scale == "blas_threads":
        fixed_commit = blas_base_commit(blas_threads) * max(1, units)
        commit = (fixed_commit, fixed_commit)
        ws = (fp.fixed[1] * max(1, units),) * 2
        return {"commit": tuple(int(v) for v in commit), "ws": tuple(int(v) for v in ws)}
    # unit_base is added last: a row without one (0.0) keeps its former values bit for bit
    commit = tuple(fp.fixed[0] * scale + units * c * scale + units * b
                   for c, b in zip(fp.commit_per_unit, fp.unit_base))
    ws = tuple(fp.fixed[1] * scale + units * c * scale * r + units * b * r
               for c, b, r in zip(fp.commit_per_unit, fp.unit_base, fp.ws_ratio))
    return {"commit": tuple(int(v) for v in commit), "ws": tuple(int(v) for v in ws)}


def units_within(fp: Footprint, ram_allowance, commit_allowance, *, frame_pixels=None, width=None, cap_items=None,
                 blas_threads=None, frames=None) -> int:
    """The most units (central footprint) that fit both allowances, at most ``cap_items``; may return 0."""
    limit = cap_items if cap_items is not None else 4096
    if ram_allowance is None or commit_allowance is None:
        return max(0, limit)
    one = footprint_bytes(fp, 1, frame_pixels=frame_pixels, width=width, blas_threads=blas_threads, frames=frames)
    none = footprint_bytes(fp, 0, frame_pixels=frame_pixels, width=width, blas_threads=blas_threads, frames=frames)
    per_ws, per_commit = one["ws"][0] - none["ws"][0], one["commit"][0] - none["commit"][0]
    room_ws, room_commit = ram_allowance - none["ws"][0], commit_allowance - none["commit"][0]
    if room_ws < 0 or room_commit < 0:
        return 0
    fits = limit
    if per_ws > 0:
        fits = min(fits, int(room_ws // per_ws))
    if per_commit > 0:
        fits = min(fits, int(room_commit // per_commit))
    return max(0, fits)


# ---- Siril parts and demand ------------------------------------------------------------------------------------- #

def siril_parts(commands, executable=None) -> tuple:
    """Parts of one Siril launch, by command (not script name): convert, probe, export, export_stack, stack_finite,
    stack_rej; ('other',) when none is recognised. ``seqapplyreg`` followed by a stock ``stack`` in one launch is
    export_stack; a ``stack ... -nan_missing_v1`` (only the finite build accepts it) is stack_finite."""
    words = [str(c).strip() for c in commands or ()]
    parts = []
    for line in words:
        head = line.split(" ", 1)[0].lower()
        if head in ("convertraw", "convert"):
            part = "convert"
        elif head == "register":
            part = "probe"
        elif head == "seqapplyreg":
            part = "export"
        elif head == "stack":
            if "-nan_missing_v1" in line:
                part = "stack_finite"
            elif parts and parts[-1] == "export":
                parts[-1] = "export_stack"
                continue
            else:
                part = "stack_rej"
        else:
            continue
        if part not in parts:
            parts.append(part)
    return tuple(parts) or ("other",)


def _part_demand(part, threads, frames, frame_pixels, allowance):
    fp = SIRIL_FOOTPRINT[part]
    pixels = frame_pixels or REFERENCE_PIXELS
    t, frames = max(1, int(threads)), max(1, int(frames or 1))
    out = {}
    for index, label in enumerate(("central", "high")):
        if fp.units_rule == "frames":
            demand = frames * (fp.commit_per_unit[index] + fp.per_thread[index] * t) * pixels
        else:
            demand = min(t, frames) * fp.commit_per_unit[index] * pixels
            demand += fp.thread_fixed[index] * min(1.0, t / 64) * pixels
        bound = "units"
        if fp.allowance_k is not None and allowance is not None and fp.allowance_k[index] * allowance < demand:
            demand, bound = fp.allowance_k[index] * allowance, "allowance"
        out[label] = (demand, bound)
    return out


def siril_demand(parts, *, threads, pool, frames, frame_pixels, allowance) -> dict:
    """Predicted peak job commit and working set of one launch: pool(p) + max over parts of min(D, k x A)."""
    pool_commit = SIRIL_POOL_FIXED + SIRIL_POOL_BYTES * max(1, int(pool or 1))
    best = {"central": (0.0, "units", None), "high": (0.0, "units", None)}
    ws = {"central": 0.0, "high": 0.0}
    for part in parts:
        if part not in SIRIL_FOOTPRINT:
            continue
        fp = SIRIL_FOOTPRINT[part]
        demand = _part_demand(part, threads, frames, frame_pixels, allowance)
        for index, label in enumerate(("central", "high")):
            value, bound = demand[label]
            if value >= best[label][0]:
                best[label] = (value, bound, part)
            ws[label] = max(ws[label], fp.ws_ratio[index] * value)
    known = any(p in SIRIL_FOOTPRINT for p in parts)
    if not known:
        return {"commit": None, "ws": None, "bound_by": None, "parts": list(parts)}
    return {"commit": (int(pool_commit + best["central"][0]), int(pool_commit + best["high"][0])),
            "ws": (int(SIRIL_POOL_WS + ws["central"]), int(SIRIL_POOL_WS + ws["high"])),
            "bound_by": best["high"][1], "dominant": best["high"][2], "pool_commit": int(pool_commit),
            "parts": list(parts)}


# =============================================================================================================== #
# Pool decisions (the P1c interface; shadow only in P1b)                                                           #
# =============================================================================================================== #

PLANNED_ACTIONS = ("run", "run_reduced", "wait", "warn_run", "needs_U6", "refuse")


@dataclass(frozen=True)
class WorkerDecision:
    stage: str
    workers: int
    bound_by: str                     # override | items | cpu | ram | commit | io | cap
    over_budget: bool
    planned_action: str
    ram_allowance: int | None
    commit_allowance: int | None
    predicted: dict
    basis: dict


def _override(fp: Footprint) -> tuple[int | None, str | None]:
    """(workers, error) of the stage's existing override knob(s); never raises (shadow). With several knobs (in
    precedence order, e.g. py.transient_scan's T-2 order) the first one set decides, a malformed one included."""
    names = fp.override_env
    if not names:
        return None, None
    for name in ((names,) if isinstance(names, str) else names):
        raw = os.environ.get(name, "").strip()
        if raw and raw.lower() != "auto":
            return _override_value(name, raw, fp.key)
    return None, None


SCAN_STAGE = "py.transient_scan"


def scan_process_bounds(cap, *, reserved_cores=0) -> tuple[int, int]:
    """(floor, ceiling) of the trail scan's process count (efficiency P1c T-2, prereg §3.4; R20): the floor is today's
    min(4, usable logical processors - 1), the ceiling min(cap, physical cores in the affinity mask - 1 -
    ``reserved_cores``) (the usable logical processors when the cores are unknown); both at least 1. The scan's
    automatic count lies in [floor, max(floor, ceiling)]; ``AESTER_FRAME_WORKERS=max`` means max(floor, ceiling)
    for it (transient_stage and _override_value both use this one definition)."""
    usable = max(1, int(frame_pool.usable_cpus()))
    floor = max(1, min(4, usable - 1))
    try:
        cores = cpu_topology().python_usable_cores
    except Exception:
        cores = None
    cores = cores or usable
    ceiling = max(1, min(int(cap), int(cores) - 1 - int(reserved_cores)))
    return floor, ceiling


def _override_value(name, raw, stage=None) -> tuple[int | None, str | None]:
    try:
        if name == "AESTER_FRAME_WORKERS":
            if raw.lower() == "max":
                if stage == SCAN_STAGE:                     # scan processes: the scan's own maximum (T-2)
                    return max(scan_process_bounds(FOOTPRINTS[SCAN_STAGE].cap)), None
                return min(frame_pool.MAX_FRAME_WORKERS, int(frame_pool.usable_cpus())), None
            if not (raw.isascii() and raw.isdigit()):       # the same values frame_pool accepts (P1b review pool-3)
                raise ValueError
            value = int(raw)
            if not 1 <= value <= frame_pool.MAX_FRAME_WORKERS:
                raise ValueError
            return value, None
        if name == "AESTER_RENDER_WORKERS":
            value = int(raw)
            if not 1 <= value <= 8:
                raise ValueError
            return value, None
        value = int(raw)
        if value < 1:
            raise ValueError
        return (min(value, 32) if name == "AESTER_HASH_WORKERS" else value), None
    except (TypeError, ValueError):
        return None, f"{name}={raw[:24]!r} ignored (malformed)"


def _io_class_for(facts: MachineFacts, role="source") -> str:
    for volume in facts.volumes:
        if volume.role == role:
            return volume.io_class
    return "unknown"


def pool_workers(stage, *, items, frame_pixels=None, width=None, facts=None, budget=None, resident=None,
                 frames=None) -> WorkerDecision:
    """workers = min(override or cap, items, cpu_limit, units_within(ram, commit)); frame-pool stages are clamped to
    frame_pool.MAX_FRAME_WORKERS (X1). Existing overrides keep precedence and are reported as 'override'. ``frames``
    scales the ``frames_width`` rows (py.median)."""
    fp = FOOTPRINTS[stage]
    facts = facts or machine_facts()
    plan_budget = budget or globals()["budget"](facts, resident=resident)
    items = max(0, int(items or 0))
    cap = fp.cap
    if fp.pool == "frame":
        cap = min(cap or frame_pool.MAX_FRAME_WORKERS, frame_pool.MAX_FRAME_WORKERS)
    candidates = {"cap": cap if cap is not None else 4096, "items": items}
    if fp.kind == "io":
        role = "source" if stage in ("py.hash",) else "work"
        candidates["io"] = IO_WORKERS.get(_io_class_for(facts, role), 2)
    else:
        candidates["cpu"] = plan_budget.python_cpu_cores or 1
    ram_allow, commit_allow = plan_budget.ram_allowance(), plan_budget.commit_allowance()
    fit = units_within(fp, ram_allow, commit_allow, frame_pixels=frame_pixels, width=width,
                       cap_items=min(candidates.values()), frames=frames)
    override, error = _override(fp)
    basis = {"kind": fp.kind, "cap": cap, "items": items}
    if error:
        basis["override_error"] = error
    if override is not None:
        workers, bound = min(override, max(1, items)) if items else override, "override"
        if fp.pool == "frame":
            workers = min(workers, frame_pool.MAX_FRAME_WORKERS)
    else:
        bound, workers = min(candidates.items(), key=lambda kv: (kv[1], kv[0] != "items"))
        if fit < workers:
            one = footprint_bytes(fp, 1, frame_pixels=frame_pixels, width=width, frames=frames)
            zero = footprint_bytes(fp, 0, frame_pixels=frame_pixels, width=width, frames=frames)
            ws_limited = ram_allow is not None and (one["ws"][0] - zero["ws"][0]) * (fit + 1) + zero["ws"][0] > ram_allow
            workers, bound = fit, ("ram" if ws_limited else "commit")
    over = items > 0 and workers < 1
    if over:
        workers = 1
    if items == 0:
        workers = 0
    action = "warn_run" if over else ("run_reduced" if bound in ("ram", "commit") else "run")
    predicted = footprint_bytes(fp, max(workers, 0), frame_pixels=frame_pixels, width=width, frames=frames)
    return WorkerDecision(stage, workers, bound, over, action, ram_allow, commit_allow, predicted, basis)


# =============================================================================================================== #
# Siril plans                                                                                                      #
# =============================================================================================================== #

@dataclass(frozen=True)
class SirilPlan:
    name: str
    parts: tuple
    route: str                        # today | A (affinity + env) | B (script lines) | A+B
    threads: int | None               # Siril worker threads
    pool: int | None                  # OpenMP/OpenBLAS pool threads
    affinity: dict | None             # {"group", "mask"}
    environment: dict                 # additions, e.g. {"OMP_NUM_THREADS": "1"}
    script_commands: tuple            # route B only
    allowance_bytes: int | None       # Siril's memory allowance under this plan
    predicted: dict
    record_neutral: bool | None       # None = unmeasured
    over_budget: bool | None
    planned_action: str
    basis: dict


SIRIL_DEFAULT_MEM_RATIO = 0.9
PART_THREAD_CAP = {"convert": 16, "probe": 32, "export": 64, "export_stack": 64, "stack_finite": 64, "stack_rej": 64}
ALLOWANCE_BOUND_PARTS = ("convert", "export", "stack_finite")          # route B can cap their in-flight units
NEUTRAL_UNDER_A = ("convert", "probe", "export", "stack_finite")       # measured record-neutral (F1, F3, P0)
UNCAPPABLE_PARTS = ("stack_rej", "export_stack")                       # thread/allowance dependent masters (U6)
FINITE_MIN_MEM_GIB = 4                                                 # 1 GiB fails, 2 GiB 2.2x slower (MEMEDGE-5)


def parse_siril_ini(text) -> dict:
    """{'mem_mode', 'mem_ratio', 'mem_amount'} of the [core] group of a siril.ini text (missing keys absent)."""
    out, section = {}, None
    for line in str(text or "").splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section != "core" or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        try:
            if key == "mem_mode":
                out[key] = int(value.strip())
            elif key in ("mem_ratio", "mem_amount"):
                out[key] = float(value.strip())
        except ValueError:
            continue
    return out


def _read_siril_ini(folder) -> dict:
    if folder is None:
        return {}
    try:
        path = os.path.join(os.fspath(folder), "siril.ini")
        if os.path.getsize(path) > 1 * MiB:
            return {}
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return parse_siril_ini(handle.read())
    except OSError:
        return {}


def _today_allowance(pin, ini, available) -> tuple[int | None, str]:
    if pin and pin.get("mem"):
        return int(pin["mem"] * GiB), "pin"
    if ini.get("mem_mode") == 1 and ini.get("mem_amount"):
        return int(ini["mem_amount"] * GiB), "siril.ini"
    ratio = ini.get("mem_ratio") if ini.get("mem_mode") == 0 and ini.get("mem_ratio") else SIRIL_DEFAULT_MEM_RATIO
    if available is None:
        return None, "unknown"
    return int(ratio * available), "ratio_of_available"


def _fits(predicted, ram_allowance, commit_allowance) -> bool | None:
    if not predicted or predicted.get("commit") is None or ram_allowance is None or commit_allowance is None:
        return None
    return predicted["ws"][0] <= ram_allowance and predicted["commit"][0] <= commit_allowance


def siril_today(commands, executable, facts: MachineFacts, *, frames, frame_pixels, name=None, folder=None,
                pin=..., environment=None, siril_ini=None, whatif=False) -> SirilPlan:
    """What runs now: t = setcpu if pinned else this process's Siril thread count; p = the pool of Siril's own
    environment block (siril.siril_environment, read-only; siril_pool_setting: its OMP_NUM_THREADS) else the group's
    LPs; A = pinned mem, else siril.ini, else 0.9 x available physical now. A proxy therefore predicts the proxy.
    ``whatif``: product defaults on the what-if machine (no pin, no inherited environment)."""
    from . import siril
    parts = siril_parts(commands, executable)
    basis = {}
    if whatif:
        pin = None
    elif pin is ...:
        try:
            pin = siril.siril_pin()
        except ValueError as exc:
            pin, basis["pin_error"] = None, str(exc)[:120]
    if environment is None:
        try:
            environment = siril.siril_environment(folder or ".", pin, base={} if whatif else None, commands=commands)
        except Exception as exc:
            environment, basis["environment_error"] = {}, f"{type(exc).__name__}"
    cpu = facts.cpu
    threads = pin["cpu"] if pin and pin.get("cpu") else cpu.siril_threads_today
    pool, pool_source = siril_pool_setting(environment)
    if pool is None:
        pool, pool_source = (cpu.siril_pool_today if whatif else cpu.siril_group_lps), "group"
    ini = siril_ini if siril_ini is not None else ({} if whatif else _read_siril_ini(folder))
    allowance, source = _today_allowance(pin, ini, facts.memory.available_physical)
    basis.update(allowance_source=source, pool_source=pool_source)
    predicted = siril_demand(parts, threads=threads or 1, pool=pool or 1, frames=frames, frame_pixels=frame_pixels,
                             allowance=allowance)
    return SirilPlan(name or "", parts, "today", threads, pool, cpu.process_affinity, {},
                     tuple(siril.pin_commands(pin)) if pin else (), allowance, predicted,
                     record_neutral(parts, "today"), None, "run", basis)


def record_neutral(parts, route) -> bool | None:
    """Whether a plan leaves every compared file and record as today (governor final.md §2.3, §3.6):
    'today' changes nothing; route B / A+B change script text (and siril.ini); route A changes only process affinity
    and environment, measured neutral for convert, probe, export and the finite stack, never for a launch with a stock
    rejection stack (its master depends on Siril's thread count, F3/E1). None = unmeasured (an unknown launch)."""
    parts = tuple(parts)
    if parts == ("other",) or not parts:
        return None
    if route == "today":
        return True
    if route in ("B", "A+B"):
        return False
    if any(p in UNCAPPABLE_PARTS for p in parts):
        return False
    return True if all(p in NEUTRAL_UNDER_A for p in parts) else None


def plan_siril(commands, executable, facts: MachineFacts, plan_budget: Budget, *, frames, frame_pixels, route="A",
               omp=1, name=None, today: SirilPlan | None = None, resident=None) -> SirilPlan:
    """A recommended plan (P2 applies it; P1b records only).

    route 'A': process affinity (one LP per physical core first, highest efficiency class first) + OMP_NUM_THREADS=omp;
    threads = the most that fit both allowances (central prediction) up to the part's cap and the budget's LPs.
    route 'B': ``setcpu t`` (t never above the LPs the process may use: Siril refuses more) with OMP_NUM_THREADS=t,
    plus ``set core.mem_mode=1`` / ``mem_amount`` when only an allowance cap fits.
    route 'A+B': A for threads plus the memory lines, for parts whose in-flight units Siril bounds by its allowance
    (convert, export, finite; finite >= 4 GiB). 'auto' = A, falling back to A+B.
    Launches with a stock rejection stack (stack_rej, export_stack) are never changed (U6): 'today', needs_U6 when
    they do not fit."""
    if route not in ("A", "B", "A+B", "auto"):
        raise ValueError("route must be A, B, A+B or auto")
    parts = siril_parts(commands, executable)
    today = today or siril_today(commands, executable, facts, frames=frames, frame_pixels=frame_pixels, name=name)
    ram_allow = plan_budget.ram_allowance(None if resident is None else resident.get("ws"))
    commit_allow = plan_budget.commit_allowance(None if resident is None else resident.get("commit"))
    basis = {"ram_allowance": ram_allow, "commit_allowance": commit_allow, "requested_route": route}
    if any(p in UNCAPPABLE_PARTS for p in parts) or parts == ("other",):
        fits = _fits(today.predicted, ram_allow, commit_allow)
        action = "run" if fits in (True, None) else ("needs_U6" if parts != ("other",) else "warn_run")
        return replace(today, record_neutral=record_neutral(parts, "today"), over_budget=fits is False,
                       planned_action=action,
                       basis={**today.basis, **basis, "reason": "stock rejection stack: thread/allowance dependent (U6)"
                              if parts != ("other",) else "unknown launch"})
    cpu = facts.cpu
    group = (cpu.process_affinity or {}).get("group", 0)
    by_setcpu = route == "B"
    cap = min([PART_THREAD_CAP.get(p, 64) for p in parts] + [plan_budget.cpu_lps or 1, cpu.siril_group_lps or 64, 64])
    if by_setcpu:
        cap = min(cap, cpu.siril_threads_today or cap)          # setcpu above the visible LPs is refused by Siril

    def pool_for(threads):
        return threads if by_setcpu else omp

    allowance = today.allowance_bytes
    memory_lines = ()
    chosen = None
    for threads in range(max(1, cap), 0, -1):
        predicted = siril_demand(parts, threads=threads, pool=pool_for(threads), frames=frames,
                                 frame_pixels=frame_pixels, allowance=allowance)
        if _fits(predicted, ram_allow, commit_allow) is not False:
            chosen = (threads, predicted)
            break
    if chosen is None and route in ("B", "A+B", "auto") and all(p in ALLOWANCE_BOUND_PARTS for p in parts):
        # cap Siril's allowance so that k x A fits both budgets; threads as the CPU allows
        threads = max(1, cap)
        k = max(SIRIL_FOOTPRINT[p].allowance_k[0] for p in parts)
        w = max(SIRIL_FOOTPRINT[p].ws_ratio[0] for p in parts)
        pool_commit = SIRIL_POOL_FIXED + SIRIL_POOL_BYTES * pool_for(threads)
        room = None if ram_allow is None or commit_allow is None else min(
            (ram_allow - SIRIL_POOL_WS) / (k * w), (commit_allow - pool_commit) / k)
        mem = int(room // GiB) if room is not None and room > 0 else 0
        floor = FINITE_MIN_MEM_GIB if "stack_finite" in parts else 1
        if mem >= floor:
            capped = siril_demand(parts, threads=threads, pool=pool_for(threads), frames=frames,
                                  frame_pixels=frame_pixels, allowance=mem * GiB)
            if _fits(capped, ram_allow, commit_allow) is not False:
                chosen, allowance = (threads, capped), mem * GiB
                memory_lines = ("set core.mem_mode=1", f"set core.mem_amount={mem}")
        else:
            basis["route_b_refused"] = f"allowance {mem} GiB below {floor} GiB"
    final_route = "B" if by_setcpu else ("A+B" if memory_lines else "A")
    over = chosen is None
    if over:
        threads = 1
        predicted = siril_demand(parts, threads=1, pool=pool_for(1), frames=frames, frame_pixels=frame_pixels,
                                 allowance=allowance)
        heavy = plan_budget.ram_total is not None and plan_budget.resident_ws > 0.25 * plan_budget.ram_total
        alone = _fits(predicted, plan_budget.ram_total, plan_budget.commit_total)
        action = "wait" if heavy and alone else "warn_run"
    else:
        threads, predicted = chosen
        action = "run" if threads >= min(cap, today.threads or cap) and not memory_lines else "run_reduced"
    script = ((f"setcpu {threads}",) if by_setcpu else ()) + memory_lines
    mask = None if by_setcpu or not cpu.cores else affinity_mask(cpu.cores, threads, group=group)
    return SirilPlan(name or today.name, parts, final_route, threads, pool_for(threads),
                     {"group": group, "mask": mask} if mask else None, {"OMP_NUM_THREADS": str(pool_for(threads))},
                     script, allowance, predicted, record_neutral(parts, final_route), over, action, basis)


def recommend_siril(commands, executable, facts, plan_budget, *, frames, frame_pixels, name=None, today=None,
                    resident=None, omp=1) -> SirilPlan:
    return plan_siril(commands, executable, facts, plan_budget, frames=frames, frame_pixels=frame_pixels,
                      route="auto", omp=omp, name=name, today=today, resident=resident)


# =============================================================================================================== #
# Shadow mode                                                                                                      #
# =============================================================================================================== #

def mode_setting() -> tuple[str, str | None]:
    """(mode, error): AESTER_GOVERNOR; unset/empty -> 'shadow'; malformed -> ('off', message). Never raises."""
    raw = os.environ.get(MODE_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MODE, None
    value = raw.strip().lower()
    if value in MODES:
        return value, None
    return "off", f"{MODE_ENV}={raw.strip()[:40]!r} is not one of {', '.join(MODES)}; governor off"


def mode() -> str:
    return mode_setting()[0]


class _Scope:
    def __init__(self, name, timer, level, mode_name, cap):
        self.name = name
        self.timer = weakref.ref(timer)
        self.level = level                    # light | full
        self.mode = mode_name
        self.cap = cap
        self.private = mode_name == "shadow"  # U-G4: no machine inventory
        self.note = {}
        self.decisions = []
        self.pools = {}
        self.errors = []
        self.dropped = 0
        self.frames = None
        self.frame_pixels = None
        self.facts = None
        self.whatif = None
        self.seconds = 0.0
        self.calls = 0

    def live_timer(self):
        timer = self.timer()
        if timer is None or getattr(timer, "_report", None) is not None:
            return None
        return timer


_scope = None
_scope_lock = threading.RLock()


def _round_gb(value, digits=2):
    return None if value is None else round(value / GB, digits)


def _error(scope, where, exc):
    if len(scope.errors) < MAX_ERRORS:
        scope.errors.append(f"{where}: {type(exc).__name__}: {str(exc)[:160]}")


def _facts_view(facts: MachineFacts) -> dict:
    """Full machine facts (shadow-full / what-if only; never under the private 'shadow' level)."""
    m, cpu = facts.memory, facts.cpu
    return {"level": facts.level,
            "memory": {"total_physical": m.total_physical, "available_physical": m.available_physical,
                       "commit_limit": m.commit_limit, "commit_available": m.commit_available},
            "cpu": {"logical_total": cpu.logical_total, "groups": list(cpu.groups), "physical_cores": cpu.physical_cores,
                    "smt": cpu.smt, "efficiency_classes": {str(k): v for k, v in cpu.efficiency_classes.items()},
                    "process_affinity": None if cpu.process_affinity is None else
                    {"group": cpu.process_affinity["group"], "mask": hex(cpu.process_affinity["mask"])},
                    "python_usable_lps": cpu.python_usable_lps, "python_usable_cores": cpu.python_usable_cores,
                    "siril_group_lps": cpu.siril_group_lps, "siril_threads_today": cpu.siril_threads_today,
                    "siril_pool_today": cpu.siril_pool_today},
            "pagefile": facts.pagefile, "power": dict(facts.power),
            "volumes": [{k: v for k, v in asdict(volume).items()} for volume in facts.volumes],
            **({"override": facts.override} if facts.override else {})}


def _budget_view(b: Budget, private: bool) -> dict:
    if private:
        return {"preset": b.preset, "ram_total_gb": None if b.ram_total is None else round(b.ram_total / GB),
                "commit_total_gb": None if b.commit_total is None else round(b.commit_total / GB),
                "cpu_lps": b.cpu_lps, "python_cpu_cores": b.python_cpu_cores}
    return {"preset": b.preset, "ram_total": b.ram_total, "commit_total": b.commit_total, "ram_reserve": b.ram_reserve,
            "commit_reserve": b.commit_reserve, "resident_ws": b.resident_ws, "resident_commit": b.resident_commit,
            "cpu_lps": b.cpu_lps, "python_cpu_cores": b.python_cpu_cores,
            "bound": f"ram {b.basis.get('ram_bound')}, commit {b.basis.get('commit_bound')}"}


def _resident_view(resident: dict, private: bool) -> dict:
    out = {"ws_gb": _round_gb(resident.get("ws"), 1 if private else 3),
           "commit_gb": _round_gb(resident.get("commit"), 1 if private else 3), "members": resident.get("members")}
    if not private:
        out["root_exe"] = resident.get("root_exe")
        out["root_is_self"] = resident.get("root_is_self")
        if resident.get("detail"):
            out["detail"] = resident["detail"]
    return out


# Siril decision keys (short: the job note is copied into workflow.json checkpoints and snapshots, X15):
#   t threads | p pool threads | A_gb Siril memory allowance | c_gb / ch_gb predicted peak job commit central / high
#   w_gb / wh_gb predicted peak working set central / high | by bound by (units | allowance) | omp OMP_NUM_THREADS
#   rec recommended plan | neutral record-neutral | over over budget | action planned P2 action
#   ram_gb / commit_gb the RAM / commit allowances at the decision | res_ws_gb / res_commit_gb the Aester tree then
#   Python pools are aggregated per stage (calls, actual / recommended workers as "n" or "low-high").


def _prediction_view(predicted, digits) -> dict:
    commit = (predicted or {}).get("commit") or (None, None)
    ws = (predicted or {}).get("ws") or (None, None)
    return {"c_gb": _round_gb(commit[0], digits), "ch_gb": _round_gb(commit[1], digits),
            "w_gb": _round_gb(ws[0], digits), "wh_gb": _round_gb(ws[1], digits),
            "by": (predicted or {}).get("bound_by")}


def _plan_view(plan: SirilPlan, private: bool, *, today=False) -> dict:
    """Flat short keys (one line each in the indented records): today's plan, or a recommendation."""
    digits = 1 if private else 2
    out = {} if today else {"route": plan.route}
    out.update(t=plan.threads, p=plan.pool,
               A_gb=None if plan.allowance_bytes is None else
               (round(plan.allowance_bytes / GB) if private else round(plan.allowance_bytes / GB, 2)),
               **_prediction_view(plan.predicted, digits))
    if plan.script_commands:
        out["script"] = list(plan.script_commands)
    if today:
        if not private and plan.basis.get("allowance_source"):
            out["src"] = plan.basis["allowance_source"]
    else:
        if plan.affinity:
            out["mask"] = hex(plan.affinity["mask"])
        if plan.environment.get("OMP_NUM_THREADS"):
            out["omp"] = plan.environment["OMP_NUM_THREADS"]
    return out


def _note_size(value, depth=0) -> int:
    """Bytes of the note as records.atomic_json stores it ``depth`` containers deep (NOTE_DEPTH): UTF-8, indent 2, so
    every line but the first carries 2 x depth more bytes than at depth 0, and CRLF line ends (text mode on Windows)."""
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return len(text.encode("utf-8")) + text.count("\n") * (1 + 2 * depth)


def _stored_bytes(note, depth) -> int:
    """Set note["bytes"] to the note's own stored size, that field included (fixed point; the digits settle at once)."""
    note["bytes"] = 0
    for _ in range(4):
        size = _note_size(note, depth)
        if size == note["bytes"]:
            break
        note["bytes"] = size
    return note["bytes"]


def _attach(scope):
    """Re-attach the whole object to the scope's own timer (profiling.StageTimer.attach_note), within the size cap
    measured as stored (X15: NOTE_DEPTH, CRLF; ``bytes`` is that size)."""
    timer = scope.live_timer()
    if timer is None:
        return
    depth = NOTE_DEPTH.get(scope.name, 0)
    note = dict(scope.note)
    note.update(decisions=list(scope.decisions), pools=dict(scope.pools), dropped=scope.dropped,
                errors=list(scope.errors), calls=scope.calls, seconds=round(scope.seconds, 4))
    room = scope.cap - BYTES_FIELD_ROOM       # room for the "bytes" field
    if _note_size(note, depth) > room:
        note = _shrink(scope, note, room, depth)
    _stored_bytes(note, depth)
    timer.attach_note(NOTE_KEY, note)


WHATIF_TRIMMED_KEYS = ("route", "t", "mask", "c_gb", "w_gb", "ram_gb", "commit_gb", "neutral", "over", "action")
REC_TRIMMED_KEYS = ("route", "t", "p", "A_gb", "mask", "script")
COMPACT_VIEWS = ("today", "rec", "whatif")      # per-decision views that a compacted note writes as one-line JSON text
# _shrink step 4b, in turn: key -> the value dropped (None: any value). An absent calls / over_budget / planned_action
# in a trimmed pool entry means 1 / 0 / "run" (POOL_TRIM_DEFAULTS; pool_view fills them back in).
POOL_TRIMS = ({"predicted_commit_gb": None, "calls": 1, "over_budget": 0, "planned_action": "run"},
              {"bound_by": None})
POOL_TRIM_DEFAULTS = {key: value for trims in POOL_TRIMS for key, value in trims.items() if value is not None}


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def note_view(value) -> dict:
    """A decision's today / rec / whatif view, or a pool entry, as a dict: notes over their size cap write these as
    one-line JSON text (``"compact": true``), others as dicts; anything unreadable is {}."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except ValueError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def pool_view(entry, trimmed=False) -> dict:
    """A pool entry as a new dict (either stored form, as note_view). In a note marked ``pools_trimmed`` the default
    values that step 4b dropped (POOL_TRIM_DEFAULTS: calls 1, over_budget 0, planned_action "run") are filled back in;
    predicted_commit_gb and bound_by, dropped whatever their value, stay absent."""
    view = dict(note_view(entry))
    if trimmed and view:
        for key, value in POOL_TRIM_DEFAULTS.items():
            view.setdefault(key, value)
    return view


def _shrink(scope, note, cap, depth=0) -> dict:
    """Keep the note within ``cap`` as stored (X15: NOTE_DEPTH, CRLF), dropping the least useful detail first and the
    oldest decisions last (counted in ``dropped``). Order:
    1. per-decision resident and what-if 'today', the pools' last_at_s;
    2. lossless: each decision's today / rec / whatif view and each pool entry become one-line JSON text
       (``"compact": true``; note_view reads either form) -- the indented form costs about 40 bytes per value as stored;
    3. machine facts, what-if facts (their small cpu block stays: bench_report's A6 mask rule);
    4. what-if detail (WHATIF_TRIMMED_KEYS stay: the A6 rule) and rec detail (REC_TRIMMED_KEYS stay);
    4b. (P1c-0: about eight more pool sites reach the job note, prereg §2.2) the pool entries' detail (POOL_TRIMS, in
       turn): predicted_commit_gb and the default values (calls 1, over_budget 0, planned_action "run"), then
       bound_by; actual, recommended and whatif always stay (the run check's M6); the note says ``pools_trimmed``;
    5. the oldest decisions."""
    decisions = [dict(d) for d in note["decisions"]]
    pools = {stage: dict(entry) for stage, entry in (note.get("pools") or {}).items()}
    note = dict(note, decisions=decisions, pools=pools)
    compact = False

    def rendered():
        if not compact:
            return note
        return dict(note, compact=True,
                    decisions=[{k: (_compact(v) if k in COMPACT_VIEWS and isinstance(v, dict) else v)
                                for k, v in decision.items()} for decision in decisions],
                    pools={stage: _compact(entry) for stage, entry in pools.items()})

    def fits():
        return _note_size(rendered(), depth) <= cap

    for decision in decisions:
        for key in ("res_ws_gb", "res_commit_gb", "whatif_today"):
            decision.pop(key, None)
    for entry in pools.values():
        entry.pop("last_at_s", None)
    if fits():
        return rendered()
    compact = True
    if fits():
        return rendered()
    for strip in ("facts", "whatif_facts"):
        if note.get(strip) is not None:
            kept = note[strip].get("cpu") if strip == "whatif_facts" and isinstance(note[strip], dict) else None
            note[strip] = {"omitted": "size cap", **({"cpu": kept} if kept else {})}
            if fits():
                return rendered()
    for decision in decisions:
        for key, kept in (("whatif", WHATIF_TRIMMED_KEYS), ("rec", REC_TRIMMED_KEYS)):
            if isinstance(decision.get(key), dict):
                decision[key] = {k: decision[key][k] for k in kept if k in decision[key]}
    if fits():
        return rendered()
    if pools:
        note["pools_trimmed"] = True
        for droppable in POOL_TRIMS:
            for entry in pools.values():
                for key, default in droppable.items():
                    if key in entry and (default is None or entry[key] == default):
                        del entry[key]
            if fits():
                return rendered()
    while decisions and not fits():
        decisions.pop(0)
        scope.decisions.pop(0)
        scope.dropped += 1
        note["dropped"] = scope.dropped
    if not fits():
        note = {k: note[k] for k in ("version", "mode", "scope", "timer", "dropped", "calls") if k in note}
        note["errors"] = ["note over the size cap; detail dropped"]
        return note
    return rendered()


def _resident_now(scope) -> dict:
    return aester_resident(detail=not scope.private and scope.name != JOB_SCOPE)


def _at(timer) -> float | None:
    try:
        return round(timer.clock() - timer.started, 3)
    except Exception:
        return None


def _request_paths(request):
    """(first input path, delivery root) of a workflow request, defensively (None when absent or odd)."""
    if not isinstance(request, dict):
        return None, None
    files = request.get("input_files")
    first = files[0] if isinstance(files, list) and files else None
    source = first.get("path") if isinstance(first, dict) else (first if isinstance(first, str) else None)
    output = request.get("delivery_root")
    return source, output if isinstance(output, str) else None


def shadow_start(scope_name, *, timer, job=None, work_dir=None, source=None, output=None, request=None) -> None:
    """Open a shadow scope bound to ``timer`` (weakly) for this process; a later call replaces it. Never raises.
    ``job``: the stack job record (frames, frame pixels, first input); ``request``: a workflow request (first input,
    delivery root)."""
    global _scope
    mode_name, error = mode_setting()
    try:   # U-G4: the sampler's raw system RAM (min_system_available_bytes) only at shadow-full; shadow/off omit it
        timer.record_system_memory = mode_name == "shadow-full"
    except Exception:
        pass
    try:
        if request is not None and (source is None or output is None):
            request_source, request_output = _request_paths(request)
            source = source if source is not None else request_source
            output = output if output is not None else request_output
        if mode_name == "off":
            with _scope_lock:
                _scope = None
            if error:      # a malformed knob: off, with a note saying so (never fails the job)
                timer.attach_note(NOTE_KEY, {"version": VERSION, "mode": "off", "scope": scope_name,
                                             "errors": [error]})
            return
        cap = JOB_NOTE_CAP_BYTES if scope_name == JOB_SCOPE else NOTE_CAP_BYTES
        scope = _Scope(scope_name, timer, "full" if mode_name == "shadow-full" else "light", mode_name, cap)
        with _scope_lock:
            _scope = scope
        started = time.perf_counter()
        with timer.operation("governor"):
            _start(scope, timer, job=job, work_dir=work_dir, source=source, output=output)
        scope.seconds += time.perf_counter() - started
        scope.calls += 1
        _attach(scope)
    except Exception as exc:   # never fails a job; a failed start leaves no live scope behind
        with _scope_lock:
            _scope = None
        try:
            timer.attach_note(NOTE_KEY, {"version": VERSION, "mode": mode_name, "scope": scope_name,
                                         "errors": [f"start: {type(exc).__name__}: {str(exc)[:160]}"]})
        except Exception:
            pass


def _start(scope, timer, *, job, work_dir, source, output):
    if job:
        scope.frames = len(job.get("inputs") or []) or None
        scope.frame_pixels = (job.get("disk_estimate") or {}).get("frame_pixels")
        if source is None and job.get("inputs"):
            source = (job["inputs"][0] or {}).get("path")
    paths = [(role, path) for role, path in (("source", source), ("work", work_dir), ("output", output)) if path]
    facts = machine_facts(level=scope.level, paths=paths)
    scope.facts = facts
    resident = _resident_now(scope)
    real_budget = budget(facts, resident=resident)
    note = {"version": VERSION, "mode": scope.mode, "scope": scope.name, "timer": scope.name,
            "footprints": FOOTPRINTS_VERSION, "level": scope.level,
            "frames": scope.frames, "frames_basis": "inputs" if scope.frames else None,
            "frame_pixels": scope.frame_pixels,
            "budget": _budget_view(real_budget, scope.private),
            "resident_at_start": _resident_view(resident, scope.private)}
    if scope.private:
        note["privacy"] = "U-G4: recommendations and rounded budgets only; no machine inventory"
    else:
        note["facts"] = _facts_view(facts)
    try:
        scope.whatif = parse_whatif(os.environ.get(FACTS_ENV))
    except ValueError as exc:
        scope.whatif = None
        _error(scope, "whatif", exc)
    if scope.whatif is not None:
        whatif_resident = _whatif_resident(scope, resident)
        cpu = scope.whatif.cpu       # bench_report's A6 mask rule reads it (sibling packing, > 64 LPs)
        note["whatif_facts"] = {"level": "whatif", **scope.whatif.override,
                                "cpu": {"logical_total": cpu.logical_total, "physical_cores": cpu.physical_cores}}
        whatif_budget = budget(scope.whatif, resident=whatif_resident)
        note["whatif_budget"] = {"ram_total": whatif_budget.ram_total, "commit_total": whatif_budget.commit_total,
                                 "cpu_lps": whatif_budget.cpu_lps, "python_cpu_cores": whatif_budget.python_cpu_cores}
    scope.note = note
    try:   # hashing: one decision against hashing.resolve_workers(None) (design §3.10)
        from . import hashing
        _pool_decision(scope, "py.hash", actual=hashing.resolve_workers(None), items=max(1, scope.frames or 1))
    except Exception as exc:
        _error(scope, "py.hash", exc)


def _whatif_resident(scope, resident):
    override = (scope.whatif.override or {}) if scope.whatif else {}
    return {"ws": override.get("resident_ws", resident.get("ws")),
            "commit": override.get("resident_commit", resident.get("commit"))}


def _live_scope():
    with _scope_lock:
        scope = _scope
    if scope is None or scope.live_timer() is None:
        return None
    return scope


def scope_timer():
    """The live timer of this process's shadow scope (the job worker's or the workflow worker's), or None. A P1c pool
    site called where no timer is activated in the thread (the job worker hands its timer around explicitly) opens
    its timer operation on it (performance.operation). Never raises; diagnostics only."""
    try:
        scope = _live_scope()
        return None if scope is None else scope.live_timer()
    except Exception:
        return None


def shadow_siril(name, folder, commands, executable, *, frames=None, frame_pixels=None) -> None:
    """Record today's plan and the recommended one for a Siril launch about to start. Never raises; no-op without a
    live scope. ``frames``/``frame_pixels`` default to the scope's job (admission count: an upper bound)."""
    scope = _live_scope()
    if scope is None:
        return
    timer = scope.live_timer()
    started = time.perf_counter()
    try:
        with _scope_lock, timer.operation("governor"):
            frames = frames or scope.frames
            frame_pixels = frame_pixels or scope.frame_pixels
            facts = replace(scope.facts, memory=memory_snapshot())
            resident = _resident_now(scope)
            plan_budget = budget(facts, resident=resident)
            today = siril_today(commands, executable, facts, frames=frames, frame_pixels=frame_pixels, name=name,
                                folder=folder)
            recommended = recommend_siril(commands, executable, facts, plan_budget, frames=frames,
                                          frame_pixels=frame_pixels, name=name, today=today)
            today_fits = _fits(today.predicted, plan_budget.ram_allowance(), plan_budget.commit_allowance())
            try:
                from . import siril
                priority = siril.siril_priority(job_stage=True)
            except Exception:
                priority = None
            digits = 0 if scope.private else 2
            rec_view = _plan_view(recommended, scope.private)
            if recommended.allowance_bytes == today.allowance_bytes:
                rec_view.pop("A_gb", None)            # only shown when the plan caps Siril's allowance
            decision = {"kind": "siril", "name": str(name), "parts": "+".join(today.parts), "at_s": _at(timer),
                        "today": {**_plan_view(today, scope.private, today=True), "fits": today_fits},
                        "rec": rec_view,
                        "neutral": recommended.record_neutral, "over": recommended.over_budget,
                        "action": recommended.planned_action,
                        "ram_gb": _round_gb(plan_budget.ram_allowance(), digits),
                        "commit_gb": _round_gb(plan_budget.commit_allowance(), digits),
                        "res_ws_gb": _round_gb(resident.get("ws"), 1),
                        "res_commit_gb": _round_gb(resident.get("commit"), 1)}
            if priority != "normal":
                decision["today"]["prio"] = priority
            if frames != scope.frames:
                decision["frames"] = frames
            if scope.whatif is not None:
                decision.update(_whatif_siril(scope, name, folder, commands, executable, frames, frame_pixels, resident))
            scope.decisions.append(decision)
    except Exception as exc:
        _error(scope, f"siril {name}", exc)
    scope.seconds += time.perf_counter() - started
    scope.calls += 1
    try:
        _attach(scope)
    except Exception:
        pass


def _whatif_siril(scope, name, folder, commands, executable, frames, frame_pixels, resident):
    facts = scope.whatif
    whatif_resident = _whatif_resident(scope, resident)
    plan_budget = budget(facts, resident=whatif_resident)
    today = siril_today(commands, executable, facts, frames=frames, frame_pixels=frame_pixels, name=name,
                        folder=folder, whatif=True)
    recommended = recommend_siril(commands, executable, facts, plan_budget, frames=frames, frame_pixels=frame_pixels,
                                  name=name, today=today)
    plan = _plan_view(recommended, False)
    view = {key: plan[key] for key in ("route", "t", "mask", "omp", "c_gb", "w_gb", "script") if key in plan}
    view.update(neutral=recommended.record_neutral, over=recommended.over_budget, action=recommended.planned_action,
                ram_gb=_round_gb(plan_budget.ram_allowance()), commit_gb=_round_gb(plan_budget.commit_allowance()))
    full = _plan_view(today, False, today=True)
    today_view = {key: full.get(key) for key in ("t", "p", "c_gb", "w_gb")}
    today_view["fits"] = _fits(today.predicted, plan_budget.ram_allowance(), plan_budget.commit_allowance())
    return {"whatif": view, "whatif_today": today_view}


def _pool_decision(scope, stage, *, actual, items, frame_pixels=None, width=None, frames=None):
    facts = replace(scope.facts, memory=memory_snapshot())
    resident = scope.note.get("resident_at_start") or {}
    resident = {"ws": int((resident.get("ws_gb") or 0) * GB), "commit": int((resident.get("commit_gb") or 0) * GB)}
    frames = frames or scope.frames
    decision = pool_workers(stage, items=items, frame_pixels=frame_pixels or scope.frame_pixels, width=width,
                            facts=facts, budget=budget(facts, resident=resident), frames=frames)
    entry = scope.pools.setdefault(stage, {"calls": 0, "actual": None, "recommended": None, "over_budget": 0})
    entry["calls"] += 1
    for key, value in (("actual", actual), ("recommended", decision.workers)):
        if value is not None:      # one number, or "lowest-highest" when calls differed
            low, high = _span(entry[key])
            low, high = (value, value) if low is None else (min(low, value), max(high, value))
            entry[key] = low if low == high else f"{low}-{high}"
    entry["over_budget"] += int(decision.over_budget)
    entry["bound_by"] = decision.bound_by
    entry["planned_action"] = decision.planned_action
    entry["predicted_commit_gb"] = max(entry.get("predicted_commit_gb") or 0,
                                       _round_gb(decision.predicted["commit"][1], 1 if scope.private else 3) or 0)
    entry["last_at_s"] = _at(scope.live_timer())
    if decision.basis.get("override_error"):
        entry["override_error"] = decision.basis["override_error"]
    if scope.whatif is not None:
        whatif = pool_workers(stage, items=items, frame_pixels=frame_pixels or scope.frame_pixels, width=width,
                              facts=scope.whatif, budget=budget(scope.whatif, resident=_whatif_resident(scope, resident)),
                              frames=frames)
        entry["whatif"] = f"{whatif.workers} {whatif.planned_action}" + (" over" if whatif.over_budget else "")


def _span(value):
    if value is None:
        return None, None
    if isinstance(value, int):
        return value, value
    low, _, high = str(value).partition("-")
    return int(low), int(high)


def shadow_pool(stage, *, actual, items, frame_pixels=None, width=None, frames=None) -> None:
    """Record the recommended worker count of a Python pool next to the ``actual`` one (aggregated per stage, X15).
    ``frames`` (default: the scope's job) scales the ``frames_width`` rows. Never raises; no-op without a live scope."""
    scope = _live_scope()
    if scope is None:
        return
    timer = scope.live_timer()
    started = time.perf_counter()
    try:
        with _scope_lock, timer.operation("governor"):
            _pool_decision(scope, stage, actual=actual, items=items, frame_pixels=frame_pixels, width=width,
                           frames=frames)
    except Exception as exc:
        _error(scope, f"pool {stage}", exc)
    scope.seconds += time.perf_counter() - started
    scope.calls += 1
    try:
        _attach(scope)
    except Exception:
        pass


def shadow_nox(*, actual_intra_op_threads) -> None:
    """Record the nox thread setting (never changed by the governor: it is recorded and feeds a signature, P3/P4)."""
    scope = _live_scope()
    if scope is None:
        return
    timer = scope.live_timer()
    started = time.perf_counter()
    try:
        with _scope_lock, timer.operation("governor"):
            today = "ort_auto" if actual_intra_op_threads is None else int(actual_intra_op_threads)
            scope.decisions.append({"kind": "nox", "name": "separation", "at_s": _at(timer),
                                    "today": {"intra_op_threads": today}, "rec": {"intra_op_threads": today},
                                    "neutral": True, "over": None, "action": "run",
                                    "reason": "nox threads are recorded and signed (P3/P4); never changed here"})
    except Exception as exc:
        _error(scope, "nox", exc)
    scope.seconds += time.perf_counter() - started
    scope.calls += 1
    try:
        _attach(scope)
    except Exception:
        pass


def current_scope_name() -> str | None:
    """Name of the live scope in this process (tests and diagnostics)."""
    scope = _live_scope()
    return None if scope is None else scope.name
