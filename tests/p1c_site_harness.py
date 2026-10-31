"""Shared scaffolding for the efficiency P1c pool-site tests (docs/EFFICIENCY_P1C_PREREG_20261005.md §6.2, A5).

Each site gets ``tests/test_p1c_site_<site>.py``; the pieces here are what every one of them needs:

(a) reference vs pool -- ``synthetic_cases()`` (workers {1,2,3,4,8,16} x in flight {w, 2w} x state policy
    {reuse, fresh} x 5 jitter seeds) and ``real_cases()`` (workers {1, default, 16}, one seed, reuse) for
    ``pytest.mark.parametrize``; ``pool_conditions(case)`` sets the frame pool's test seams (state policy, the
    seeded scheduling jitter, the R6 context audit) for one run; ``assert_same(expected, actual)`` compares arrays as
    uint8 views, bytes as bytes, files by bytes, records as JSON *with key order*, floats by bit pattern;
    ``hashseed_digests(target)`` reruns a site in fresh interpreters under two PYTHONHASHSEED values (and, through
    ``env``, the numeric variants of (g): NUMERIC_ENVS).
(b) real data -- ``fixture_root()`` honours AESTER_FIXTURE_ROOT (a worktree has no outputs/; point it at the main
    checkout) and defaults to the repository root; ``fixture_path()`` skips when an asset is missing, or fails when
    AESTER_REQUIRE_FIXTURES is set (acceptance runs: a skipped real-data test counts as a failure); ``f48_run()`` /
    ``f48_job()`` are the read-only F48 reference run and its one stack job folder;
    ``copy_subset()`` copies read-only fixture files into the test's basetemp; ``assert_short_path()`` keeps Siril
    run folders under 240 characters.
(c) failure -- ``failure_positions(n)`` ({0, 1, middle, last}); ``FailOn`` wraps a callable to raise at chosen calls;
    ``outcome(run)`` / ``assert_like_serial(serial, parallel)`` compare exception type and message and the delivered
    prefix; ``tree_state()`` / ``changed_files()`` give the files a failed run created or modified.
(d) cancellation -- ``CancelSwitch`` is a job ``check()`` that raises a fresh exception per call once tripped (as
    core.check does) and times the pool's return against CANCEL_LIMIT_SECONDS (1.0 s, §5 item 5).
(e) the audit -- ``pool_conditions`` turns on ``frame_pool._AUDIT_CONTEXT`` unless told not to.
(f) memory -- ``PeakSampler`` samples this process's private bytes on a thread (rise from the start of the block),
    with tracemalloc's peak as a lower-bound cross-check; ``pool_memory_limit(stats)`` is granted x (unit_bytes +
    BLAS slot) + in flight x result bytes; ``head_last=True`` jitter makes the first unit finish last so every other
    result is held; ``peak_pagefile_usage()`` is the lifetime peak for before/after steps in a fresh process.
Plus ``assert_pool_quiet(before)``: no ``aester-*`` thread left, the frame pool's slots and reservations released.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from starstack import frame_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests"
SRC = REPO_ROOT / "src"
FIXTURE_ENV = "AESTER_FIXTURE_ROOT"
REQUIRE_ENV = "AESTER_REQUIRE_FIXTURES"
F48_RUN = Path("outputs") / "perf-baseline-20260928" / "runs" / "F48"     # read-only reference job (§0, A1a)
F48_FILES = 758     # docs/WORKSPACE_COLD_ARCHIVE_20261005.md: restored from T: on 10-05, 758/758 files checked

SYNTHETIC_WORKERS = (1, 2, 3, 4, 8, 16)
STATE_POLICIES = ("reuse", "fresh")
JITTER_SEEDS = (0, 1, 2, 3, 4)
HASH_SEEDS = ("0", "2718281")
CANCEL_LIMIT_SECONDS = 1.0
SIRIL_PATH_LIMIT = 240
# (g): numeric environments a site that moves work out of the single-BLAS-thread scan children must be equal under
NUMERIC_ENVS = ({"NPY_DISABLE_CPU_FEATURES": "X86_V3"}, {"OPENBLAS_NUM_THREADS": "1"},
                {"OPENBLAS_NUM_THREADS": "16"}, {"OPENBLAS_NUM_THREADS": "24"}, {"OPENBLAS_CORETYPE": "Haswell"})


# ================================================================== (b) fixtures

def fixture_root() -> Path:
    """AESTER_FIXTURE_ROOT, else the repository root (untracked assets -- outputs/, tools/, raw*/ -- live only in the
    main checkout, so a worktree sets the variable to it)."""
    value = os.environ.get(FIXTURE_ENV, "").strip()
    return Path(value) if value else REPO_ROOT


def fixture_path(*parts, what: str | None = None) -> Path:
    """An asset under fixture_root(); a missing one skips the test, or fails it when AESTER_REQUIRE_FIXTURES is set."""
    path = fixture_root().joinpath(*parts)
    if not path.exists():
        message = f"fixture missing: {what or Path(*parts)} ({path}; set {FIXTURE_ENV} to the main checkout)"
        if os.environ.get(REQUIRE_ENV, "").strip() not in ("", "0"):
            pytest.fail(message)
        pytest.skip(message)
    return path


def f48_run() -> Path:
    return fixture_path(F48_RUN, what="the F48 reference job")


def f48_job() -> Path:
    """The F48 run's one stack job folder (c/workflow-*/stack/job-*: job.json, work/, products/)."""
    jobs = sorted(path.parent for path in f48_run().glob("c/workflow-*/stack/job-*/job.json"))
    assert len(jobs) == 1, f"the F48 reference run should hold one stack job, found {len(jobs)}"
    return jobs[0]


def copy_subset(source: Path, dest: Path, patterns=("*",)) -> list:
    """Copy the files of ``source`` matching ``patterns`` (relative globs, recursive with '**') into ``dest``, sorted,
    as plain writable copies (the fixture stays read-only; never a link). Returns the copies' paths."""
    source, dest = Path(source), Path(dest)
    chosen = sorted({p for pattern in patterns for p in source.glob(pattern) if p.is_file()})
    copies = []
    for path in chosen:
        target = dest / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        copies.append(target)
    return copies


def assert_short_path(path, limit: int = SIRIL_PATH_LIMIT) -> Path:
    """§6.2 (b): a Siril run folder must stay under ``limit`` characters; use a short basetemp root."""
    text = str(Path(path).resolve())
    assert len(text) <= limit, f"path too long for Siril ({len(text)} > {limit}): {text}"
    return Path(path)


# ================================================================== (a) cases and pool conditions

@dataclass(frozen=True)
class Case:
    workers: int | None          # None: the site's own default decision
    inflight: int | None         # None: the pool's default (2 x workers)
    policy: str = "reuse"
    seed: int | None = 0         # None: no jitter

    @property
    def id(self) -> str:
        workers = "default" if self.workers is None else f"w{self.workers}"
        inflight = "" if self.inflight is None else f"-i{self.inflight}"
        seed = "" if self.seed is None else f"-s{self.seed}"
        return f"{workers}{inflight}-{self.policy}{seed}"


def synthetic_cases(workers=SYNTHETIC_WORKERS, policies=STATE_POLICIES, seeds=JITTER_SEEDS) -> list:
    """§6.2 (a) synthetic sizes: workers x in flight {w, 2w} x state policy x jitter seeds."""
    return [Case(w, i, p, s) for w in workers for i in sorted({w, 2 * w}) for p in policies for s in seeds]


def real_cases(*, maximum: int = 16) -> list:
    """§6.2 (a) real sizes: workers {1, the site's default, 16}, one seed, reuse ("fresh" would hold dozens of states)."""
    return [Case(1, None, "reuse", 0), Case(None, None, "reuse", 0), Case(maximum, None, "reuse", 0)]


def params(cases) -> list:
    return [pytest.param(case, id=case.id) for case in cases]


def jitter(seed: int | None, *, max_seconds: float = 0.003, head_last: bool = False, head_seconds: float = 0.05):
    """A ``frame_pool._before_unit`` replacement: a seeded random sleep before each unit; ``head_last`` adds a long
    sleep before unit 0 so that it finishes last (every later result is held: the memory peak of §6.2 (f))."""
    rng = random.Random(seed)
    lock = threading.Lock()

    def before(index):
        with lock:
            delay = rng.random() * max_seconds if seed is not None else 0.0
        if head_last and index == 0:
            delay += head_seconds
        if delay:
            time.sleep(delay)
    return before


@contextlib.contextmanager
def pool_conditions(case: Case | None = None, *, audit: bool = True, max_jitter: float = 0.003,
                    head_last: bool = False, head_seconds: float = 0.05):
    """Set the frame pool's test seams for one run and restore them: the state policy (R12), seeded jitter, and the
    context audit (R6, §6.2 (e))."""
    saved = (frame_pool._STATE_POLICY, frame_pool._AUDIT_CONTEXT, frame_pool._before_unit)
    case = case or Case(None, None)
    try:
        frame_pool._STATE_POLICY = case.policy
        frame_pool._AUDIT_CONTEXT = audit
        if case.seed is not None or head_last:
            frame_pool._before_unit = jitter(case.seed, max_seconds=max_jitter, head_last=head_last,
                                             head_seconds=head_seconds)
        yield case
    finally:
        frame_pool._STATE_POLICY, frame_pool._AUDIT_CONTEXT, frame_pool._before_unit = saved


def aester_threads() -> set:
    return {t for t in threading.enumerate() if t.name.startswith("aester-")}


def assert_pool_quiet(before: set | None = None, *, settle: float = 5.0) -> None:
    """No pool thread left behind (compared with ``before``), every worker slot and reservation released."""
    before = before or set()
    deadline = time.monotonic() + settle
    while (aester_threads() - before) and time.monotonic() < deadline:
        time.sleep(0.01)
    left = sorted(t.name for t in aester_threads() - before)
    assert not left, f"pool threads left: {left}"
    assert frame_pool._budget_used == 0, f"frame-pool slots still taken: {frame_pool._budget_used}"
    assert frame_pool.reserved_bytes() == 0, f"memory still reserved: {frame_pool.reserved_bytes()}"


# ---- comparing ------------------------------------------------------------------------------------------------------

def _float_bits(value: float) -> bytes:
    return struct.pack("<d", value)


def assert_same(expected, actual, where: str = "result") -> None:
    """Bit identity: arrays (dtype, shape, uint8 view), bytes, files (Path: their bytes), records (dicts with the
    same key order), floats by bit pattern (-0.0 and NaN payloads included), everything else by ==."""
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray), f"{where}: array vs not"
        assert expected.dtype == actual.dtype, f"{where}: dtype {expected.dtype} != {actual.dtype}"
        assert expected.shape == actual.shape, f"{where}: shape {expected.shape} != {actual.shape}"
        a = np.ascontiguousarray(expected).view(np.uint8)
        b = np.ascontiguousarray(actual).view(np.uint8)
        if not np.array_equal(a, b):
            first = int(np.flatnonzero(a.reshape(-1) != b.reshape(-1))[0])
            raise AssertionError(f"{where}: bytes differ (first at byte {first} of {a.size})")
        return
    if isinstance(expected, Path) or isinstance(actual, Path):
        assert isinstance(expected, Path) and isinstance(actual, Path), f"{where}: path vs not"
        assert_same(expected.read_bytes(), actual.read_bytes(), f"{where} ({expected.name})")
        return
    if isinstance(expected, (bytes, bytearray, memoryview)):
        assert bytes(expected) == bytes(actual), f"{where}: bytes differ ({len(expected)} vs {len(actual)})"
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{where}: dict vs {type(actual).__name__}"
        assert list(expected) == list(actual), f"{where}: keys or key order differ: {list(expected)} != {list(actual)}"
        for key in expected:
            assert_same(expected[key], actual[key], f"{where}[{key!r}]")
        return
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual), f"{where}: {type(expected).__name__} vs {type(actual).__name__}"
        assert len(expected) == len(actual), f"{where}: length {len(expected)} != {len(actual)}"
        for index, (a, b) in enumerate(zip(expected, actual)):
            assert_same(a, b, f"{where}[{index}]")
        return
    if isinstance(expected, float) and isinstance(actual, float):
        assert _float_bits(expected) == _float_bits(actual), f"{where}: {expected!r} != {actual!r} (bits)"
        return
    if isinstance(expected, np.generic) and isinstance(actual, np.generic):
        assert_same(np.asarray(expected), np.asarray(actual), where)
        return
    assert type(expected) is type(actual) and expected == actual, f"{where}: {expected!r} != {actual!r}"


def json_bytes(record) -> bytes:
    """A record as the product writes JSON (key order kept): for byte comparisons of records."""
    return json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def tree_bytes(root: Path) -> dict:
    """{relative posix path: bytes} of every file under ``root``, sorted."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def tree_state(root: Path) -> dict:
    """{relative posix path: (size, mtime_ns, sha256)} under ``root``: what a failed run created or modified."""
    root = Path(root)
    state = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            info = path.stat()
            state[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns,
                                                         hashlib.sha256(path.read_bytes()).hexdigest())
    return state


def changed_files(before: dict, after: dict) -> dict:
    """{'created': [...], 'modified': [...], 'deleted': [...]} between two tree_state() readings (content changes:
    size or digest; an mtime-only change counts as modified too)."""
    return {"created": sorted(set(after) - set(before)),
            "modified": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
            "deleted": sorted(set(before) - set(after))}


# ---- PYTHONHASHSEED and numeric reruns ------------------------------------------------------------------------------

def canonical_digest(value) -> str:
    """sha256 of a canonical encoding: arrays (dtype, shape, bytes), bytes, str, int, float bits, bool, None, dicts
    (in key order), lists / tuples, Paths (file bytes)."""
    digest = hashlib.sha256()

    def feed(item):
        if isinstance(item, np.ndarray):
            data = np.ascontiguousarray(item)
            digest.update(b"A" + str(data.dtype.str).encode() + repr(data.shape).encode() + data.view(np.uint8).tobytes())
        elif isinstance(item, np.generic):
            feed(np.asarray(item))
        elif isinstance(item, Path):
            digest.update(b"P" + item.read_bytes())
        elif isinstance(item, (bytes, bytearray, memoryview)):
            digest.update(b"B" + struct.pack("<Q", len(item)) + bytes(item))
        elif isinstance(item, str):
            data = item.encode("utf-8")
            digest.update(b"S" + struct.pack("<Q", len(data)) + data)
        elif isinstance(item, bool) or item is None:
            digest.update(b"N" + repr(item).encode())
        elif isinstance(item, int):
            digest.update(b"I" + str(item).encode() + b";")
        elif isinstance(item, float):
            digest.update(b"F" + _float_bits(item))
        elif isinstance(item, dict):
            digest.update(b"D" + struct.pack("<Q", len(item)))
            for key, entry in item.items():
                feed(key)
                feed(entry)
        elif isinstance(item, (list, tuple)):
            digest.update((b"L" if isinstance(item, list) else b"T") + struct.pack("<Q", len(item)))
            for entry in item:
                feed(entry)
        else:
            raise TypeError(f"canonical_digest: unsupported {type(item).__name__}")
    feed(value)
    return digest.hexdigest()


_CHILD = r'''
import importlib, json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
module, _, name = sys.argv[3].partition(":")
target = getattr(importlib.import_module(module), name)
args = json.loads(sys.argv[4])
from p1c_site_harness import canonical_digest
print("P1C-DIGEST", canonical_digest(target(*args)), flush=True)
'''


def subprocess_digest(target: str, *args, env: dict | None = None, timeout: float = 900) -> str:
    """Run ``module:function`` (importable from tests/ or src/) in a fresh interpreter with ``args`` (JSON values)
    and extra environment ``env``; return canonical_digest() of what it returns."""
    environment = {**os.environ, **(env or {})}
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", _CHILD, str(TESTS), str(SRC), target,
                             json.dumps(list(args))], capture_output=True, text=True, timeout=timeout,
                            env=environment, encoding="utf-8", errors="replace")
    lines = [line for line in result.stdout.splitlines() if line.startswith("P1C-DIGEST ")]
    assert result.returncode == 0 and lines, (result.returncode, result.stdout[-2000:], result.stderr[-4000:])
    return lines[-1].split()[1]


def hashseed_digests(target: str, *args, seeds=HASH_SEEDS, env: dict | None = None, timeout: float = 900) -> dict:
    """{PYTHONHASHSEED: digest} of ``target(*args)`` run in a fresh interpreter per seed (§6.2 (a))."""
    return {seed: subprocess_digest(target, *args, env={**(env or {}), "PYTHONHASHSEED": seed}, timeout=timeout)
            for seed in seeds}


def assert_hashseed_stable(reference: str, pooled: str, *args, seeds=HASH_SEEDS, env: dict | None = None) -> str:
    """The reference side and the P1c side, each under two hash seeds: four equal digests (returned)."""
    digests = {("reference", seed): digest for seed, digest in hashseed_digests(reference, *args, seeds=seeds,
                                                                                env=env).items()}
    digests.update({("pool", seed): digest for seed, digest in hashseed_digests(pooled, *args, seeds=seeds,
                                                                               env=env).items()})
    assert len(set(digests.values())) == 1, f"digests differ: {digests}"
    return next(iter(digests.values()))


# ================================================================== (c) failure

def failure_positions(count: int) -> list:
    """§6.2 (c): fail at k in {0, 1, middle, last} (those that exist)."""
    return sorted({k for k in (0, 1, count // 2, count - 1) if 0 <= k < count})


class FailOn:
    """Wrap ``function`` so that the calls whose ``key(*args, **kwargs)`` is in ``at`` raise ``error(key)`` instead
    (thread-safe; ``key`` defaults to the call's ordinal, 0-based). ``calls`` lists the keys seen, in call order."""

    def __init__(self, function, at, error=lambda key: RuntimeError(f"injected failure at {key}"), key=None):
        self.function, self.at, self.error, self.key = function, set(at), error, key
        self.calls, self._lock, self._count = [], threading.Lock(), 0

    def __call__(self, *args, **kwargs):
        with self._lock:
            key = self.key(*args, **kwargs) if self.key is not None else self._count
            self._count += 1
            self.calls.append(key)
        if key in self.at:
            raise self.error(key)
        return self.function(*args, **kwargs)


@dataclass
class Outcome:
    error_type: type | None
    message: str | None
    delivered: list
    value: object = None


def outcome(run) -> Outcome:
    """``run(delivered)`` appends what it delivered (in order) to the list it is given; capture how it ended."""
    delivered = []
    try:
        value = run(delivered)
    except BaseException as exc:            # noqa: BLE001 -- the test compares how it ended
        return Outcome(type(exc), str(exc), delivered)
    return Outcome(None, None, delivered, value)


def assert_like_serial(serial: Outcome, parallel: Outcome) -> None:
    """§6.2 (c): the same exception type and message as the serial run; the delivered prefix never longer, and a
    prefix of the serial one."""
    assert parallel.error_type is serial.error_type, (serial.error_type, parallel.error_type, parallel.message)
    assert parallel.message == serial.message, (serial.message, parallel.message)
    assert len(parallel.delivered) <= len(serial.delivered), (serial.delivered, parallel.delivered)
    assert parallel.delivered == serial.delivered[:len(parallel.delivered)], (serial.delivered, parallel.delivered)


# ================================================================== (d) cancellation

class Cancelled(RuntimeError):
    """The default cancel of CancelSwitch (core's own Cancelled is a RuntimeError too; pass WorkflowCancelled, a
    KeyboardInterrupt, to test that path)."""


class CancelSwitch:
    """A job ``check()``: once tripped, every call raises a *fresh* ``exc_type(message)`` (as core.check does) and
    the first one is kept in ``raised[0]``. Trip it by time (``trip_after``), at the k-th call (``trip_at_call``,
    0-based: a submission-time cancel), or directly (``trip``)."""

    def __init__(self, exc_type=Cancelled, message="用户取消"):
        self.exc_type, self.message = exc_type, message
        self.raised, self.calls = [], 0
        self.tripped_at = None
        self._at_call = None
        self._lock = threading.Lock()
        self._timer = None

    def __call__(self):
        with self._lock:
            index = self.calls
            self.calls += 1
            if self._at_call is not None and index >= self._at_call and self.tripped_at is None:
                self.tripped_at = time.perf_counter()
            tripped = self.tripped_at is not None
        if tripped:
            exc = self.exc_type(self.message)
            self.raised.append(exc)
            raise exc

    def trip(self):
        with self._lock:
            if self.tripped_at is None:
                self.tripped_at = time.perf_counter()

    def trip_after(self, seconds: float):
        self._timer = threading.Timer(seconds, self.trip)
        self._timer.start()
        return self

    def trip_at_call(self, k: int):
        self._at_call = k
        return self

    def latency(self, returned_at: float) -> float | None:
        return None if self.tripped_at is None else returned_at - self.tripped_at

    def close(self):
        if self._timer is not None:
            self._timer.cancel()


def assert_cancels_in_time(run, switch: CancelSwitch, *, limit: float = CANCEL_LIMIT_SECONDS) -> float:
    """``run(switch)`` must raise the first exception ``switch`` raised, within ``limit`` seconds of the trip (R18,
    §6.2 (d)), and leave no pool thread, slot or reservation behind. Returns the latency."""
    before = aester_threads()
    try:
        with pytest.raises(switch.exc_type) as info:
            run(switch)
    finally:
        switch.close()
    returned = time.perf_counter()
    assert switch.raised and info.value is switch.raised[0], "not the first cancel object raised by check()"
    latency = switch.latency(returned)
    assert latency is not None and latency <= limit, f"cancel took {latency:.3f} s (limit {limit} s)"
    assert_pool_quiet(before)
    return latency


# ================================================================== (f) memory

def private_bytes() -> int | None:
    """This process's private bytes (commit) now; None off Windows."""
    counters = _process_counters()
    return None if counters is None else counters[0]


def peak_pagefile_usage() -> int | None:
    """This process's lifetime peak commit (PeakPagefileUsage); compare before/after a step in a fresh process."""
    counters = _process_counters()
    return None if counters is None else counters[1]


_PSAPI = None


def _process_counters():
    global _PSAPI
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    if _PSAPI is None:
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        get = k32.K32GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get.restype = wintypes.BOOL
        _PSAPI = (Counters, get, k32.GetCurrentProcess)
    Counters, get, current = _PSAPI
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not get(current(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PrivateUsage), int(counters.PeakPagefileUsage)


class PeakSampler:
    """``with PeakSampler() as peak:`` samples private bytes every ``interval`` seconds on a thread; afterwards
    ``peak.rise`` is the highest reading minus the one at the start (a sampled lower bound of the true peak) and
    ``peak.traced`` tracemalloc's peak (Python allocations only: a lower-bound cross-check, it cannot see native
    buffers). ``rise`` is None off Windows."""

    def __init__(self, interval: float = 0.001, *, trace: bool = True):
        self.interval, self.trace = interval, trace
        self.start = self.highest = self.rise = self.traced = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None
        self._tracing = False

    def _loop(self):
        while not self._stop.is_set():
            value = private_bytes()
            if value is not None:
                self.highest = value if self.highest is None else max(self.highest, value)
                self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.trace and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracing = True
        if self.trace:
            tracemalloc.reset_peak()
        self.start = private_bytes()
        self.highest = self.start
        self._thread = threading.Thread(target=self._loop, name="p1c-peak-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        value = private_bytes()
        if value is not None and self.highest is not None:
            self.highest = max(self.highest, value)
        if self.start is not None and self.highest is not None:
            self.rise = self.highest - self.start
        if self.trace:
            self.traced = tracemalloc.get_traced_memory()[1]
            if self._tracing:
                tracemalloc.stop()
        return False


def pool_memory_limit(stats: dict, *, result_bytes: int = 0, blas: bool = True) -> int:
    """§6.2 (f): granted workers x (unit_bytes + BLAS slot) + in flight x result bytes, from a run's pool_stats."""
    workers = int(stats.get("workers") or 1)
    unit = int(stats.get("unit_bytes") or 0)
    inflight = int(stats.get("inflight") or workers)
    slot = frame_pool.BLAS_SLOT_BYTES if blas else 0
    return workers * (unit + slot) + inflight * int(result_bytes)


def gib(value: float) -> int:
    return int(math.ceil(value * (1 << 30)))
