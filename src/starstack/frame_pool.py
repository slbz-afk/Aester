"""Bounded, ordered, cancellable thread fan-out for per-frame / per-block work (efficiency P1b-1).

docs/EFFICIENCY_P1B_PREREG_20261001.md (P1b-1, rules R1-R15).  Standard library only (it is part of the frozen
image-holdout closure, ``scripts/verify_image_holdout.py``).

Serial reference (``workers == 1``, fewer than two items, a call from inside a unit, an open ``exclusive`` section, or
no free worker slots): literally this loop in the calling thread, no threads, no context copy::

    state = init() if init else None
    try:
        for i, item in enumerate(items):
            if check_each and check: check()
            payload = prepare(i, item, check or no_check) if prepare else item
            on_result(i, fn(payload, Unit(i, check or no_check, state)))   # or results.append(...)
    finally:
        if finalize: finalize(state)

Parallel path: the coordinator (the calling thread) calls check(), prepare() and on_result(), always in item order and
never from a worker; fn runs on up to ``workers`` threads named ``aester-<name>_<k>``, each task in its own copy of the
caller's contextvars context (numpy errstate, the hashing scope and the Siril job-stage marker travel with it).  In a
worker, ``unit.check`` is an abort probe: it raises the private ``_Aborted`` (a BaseException) once the pool stops
above this unit and never has side effects.

``prepare`` receives a coordinator-side check (rule R15, P1b cross-check X18).  On the serial path it is the caller's
``check`` itself.  On the parallel path it calls ``check()`` and turns any exception into the pool's cancel: the first
object is kept, ``check()`` is never called again (a later call re-raises the kept object), and the run raises it.  A
``prepare`` therefore never calls the job's ``check()`` directly.

Stop rule (monotone): ``stop`` is the lowest index at which a failure is known (-1 = cancel: every unit).  It only
ever moves down.  Submission (check + prepare) ends as soon as any failure or cancel is known.

Raised (always after every worker thread has ended and every created state was finalized):
- a failure of prepare(i) / fn(i) / on_result(i): the one at the lowest index (the serial loop's);
- any exception of check() (at submission, at a wait poll or through prepare's check) is a cancel: the first object
  is kept, check() is never called again, every unit stops at its next probe, nothing more is delivered; the cancel is
  raised unless the unit next in line for delivery ended with a failure of its own (then that failure);
- if finalize() raises, its first exception is raised with the above as ``__context__``; the serial loop's
  ``finally: finalize(state)`` behaves the same way.

Process-wide worker budget: at most MAX_FRAME_WORKERS pool worker threads exist at once in the process, whoever starts
the pools; a pool that gets fewer than two slots runs the serial path.

Diagnostics (``stats``) belong in ``profiling.note(...)`` (timings notes only), never in a record or a digest; a P1c
site names the dict ``pool_stats`` and hands it to ``performance.note_pool(site, pool_stats)`` (rule R19).

Efficiency P1c-0 (docs/EFFICIENCY_P1C_PREREG_20261005.md §2.2, rules R16-R22) adds, still standard library only:
- in-process memory reservations: ``reserved(nbytes)``, and ``run_ordered(..., unit_bytes=...)`` holds
  ``granted x (unit_bytes + BLAS slot)`` while its worker threads run; ``reserved_bytes()`` is the outstanding total,
  which performance.available_memory() / available_commit() deduct (so every worker decision sees it, once);
- the floating-point environment (R21): every worker task (and the ``Beside`` helper) starts with the coordinator's
  denormal and rounding control bits (``_controlfp`` ``_MCW_DN | _MCW_RC`` on Windows); ``fp_env_applied`` counts the
  tasks whose thread had to be changed;
- ``exclusive(name)``: a process-wide count around the stock Siril stack launch and nox separation; a pool started
  while one is open runs serially (reason ``exclusive``; its stats name the section); under the test seam
  ``_EXCLUSIVE_RAISES`` it raises ExclusiveViolation instead;
- ``Beside``: one helper thread next to the calling thread (O-1), and the ``AESTER_OVERLAP`` knob (auto | off), which
  validate_environment() checks with AESTER_FRAME_WORKERS at job start;
- per-run diagnostics in ``stats``: unit_bytes, reserved_bytes, CPU (units; process user/kernel), the longest unit,
  fp_env.
"""
from __future__ import annotations

import contextvars
import operator
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Any, Callable, NamedTuple

MAX_FRAME_WORKERS = 16                   # OpenBLAS 0.3.34 MAX_THREADS=24 warns from 28 concurrent BLAS callers
ENV_WORKERS = "AESTER_FRAME_WORKERS"     # auto (default) | 1..16 | max -- measurements and verification only
ENV_OVERLAP = "AESTER_OVERLAP"           # auto (default) | off -- O-1 diagnostics; never in a signature or a record
POLL_SECONDS = 0.1                       # the coordinator calls check() at least this often while it waits
BLAS_SLOT_BYTES = 32 << 20               # OpenBLAS keeps one 32 MiB buffer (commit) per extra peak concurrent caller
_MAX_GROUPS = 64                         # GetProcessGroupAffinity buffer (Windows has at most a few dozen groups)
_MCW_DN, _MCW_RC = 0x03000000, 0x00000300   # float.h: denormal control (FTZ/DAZ on SSE) and rounding control
FP_ENV_MASK = _MCW_DN | _MCW_RC          # R21: the control bits a worker takes over from the coordinator
_IN_POOL = contextvars.ContextVar("aester_frame_pool_unit", default=False)

_budget_lock = threading.Lock()
_budget_used = 0
_reserved = 0                            # bytes held by reserved() and by running pools that declared unit_bytes
_exclusive = {}                          # name -> open count of exclusive() sections (process-wide)

# ---- test seams (never changed by product code) ------------------------------------------------------------------
_STATE_POLICY = "reuse"                  # "reuse": `workers` states, checked out per task; "fresh": init() per unit
_AUDIT_CONTEXT = False                   # True: raise ContextLeak when a unit or callback leaves a ContextVar changed
_EXCLUSIVE_RAISES = False                # True: a pool that would run in parallel inside exclusive() raises


def _before_unit(index):
    """Test seam (tests monkeypatch it to add scheduling jitter); a no-op in the product."""


class _Aborted(BaseException):
    """Raised by a worker's unit.check() once the pool is stopping; never escapes run_ordered.
    A BaseException so that a unit's ``except Exception`` fallback cannot swallow it (rule R14)."""


class _Abandoned(BaseException):
    """Raised by a ``Beside`` probe once the helper is abandoned. Its own class, not ``_Aborted``, so that a helper's
    abandonment is never mistaken for a pool's stop (and a BaseException, so ``except Exception`` cannot swallow it)."""


class ContextLeak(AssertionError):
    """Audit mode only: a unit or callback left a ContextVar (numpy errstate, hashing scope, ...) changed (rule R6)."""


class ExclusiveViolation(AssertionError):
    """Test seam ``_EXCLUSIVE_RAISES`` only: a pool that would have run in parallel was started inside exclusive()."""


def _no_check():
    return None


class Unit(NamedTuple):
    index: int                           # position in items
    check: Callable[[], None]            # serial path: the caller's check itself; parallel: an abort probe
    state: Any                           # a per-worker state from init() (exclusive while the unit runs), or None


def in_pool() -> bool:
    """True inside a frame-pool unit running on a worker thread (pools started there run serially)."""
    return _IN_POOL.get()


def usable_cpus() -> int:
    """Logical processors this process may run on.

    ``os.cpu_count()`` ignores the affinity mask (192 on the development PC under a 16-processor mask).  On Windows a
    process confined to one processor group counts the bits of its affinity mask; a process spanning several groups
    (Windows 11 default) may use every active processor.  Elsewhere ``sched_getaffinity``.
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)   # own instance: no shared argtypes are changed
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.GetProcessGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT),
                                                    ctypes.POINTER(wintypes.USHORT)]
            k32.GetProcessGroupAffinity.restype = wintypes.BOOL
            k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                   ctypes.POINTER(ctypes.c_size_t)]
            k32.GetProcessAffinityMask.restype = wintypes.BOOL
            me = k32.GetCurrentProcess()
            count = wintypes.USHORT(_MAX_GROUPS)
            groups = (wintypes.USHORT * _MAX_GROUPS)()
            if k32.GetProcessGroupAffinity(me, ctypes.byref(count), groups) and count.value == 1:
                process, system = ctypes.c_size_t(), ctypes.c_size_t()
                if k32.GetProcessAffinityMask(me, ctypes.byref(process), ctypes.byref(system)) and process.value:
                    return bin(process.value).count("1")
        except (AttributeError, OSError):
            pass
    elif hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _setting() -> str:
    setting = os.environ.get(ENV_WORKERS, "").strip().lower() or "auto"
    # ASCII digits only: str.isdigit() also accepts superscript digits ('²', which int() rejects without naming the
    # knob) and full-width ones ('８', which int() reads although the bench's --frame-workers refuses it)
    if setting in ("auto", "max") or (setting.isascii() and setting.isdigit()
                                      and 1 <= int(setting) <= MAX_FRAME_WORKERS):
        return setting
    raise ValueError(f"{ENV_WORKERS} must be auto, max or 1-{MAX_FRAME_WORKERS}")


def worker_setting() -> str:
    """AESTER_FRAME_WORKERS as the pool reads it: 'auto' (also when unset), 'max' or its ASCII digits; ValueError
    when malformed."""
    return _setting()


def _overlap_setting() -> str:
    setting = os.environ.get(ENV_OVERLAP, "").strip().lower() or "auto"
    if setting in ("auto", "off"):       # 'ｏｆｆ' (full width) lower-cases to itself and is refused like U-P2's knob
        return setting
    raise ValueError(f"{ENV_OVERLAP} must be auto or off")


def validate_environment() -> None:
    """Raise ValueError for a malformed AESTER_FRAME_WORKERS or AESTER_OVERLAP (the job worker calls it at job start,
    so a malformed value fails the job before any work, as U-P2 decided for AESTER_FRAME_WORKERS)."""
    _setting()
    _overlap_setting()


def overlap_enabled() -> bool:
    """O-1 may overlap (prereg §3.6): AESTER_OVERLAP is not ``off``, AESTER_FRAME_WORKERS is not 1 and the caller is
    not inside a pool unit. Raises ValueError for a malformed knob (validate_environment has refused it before)."""
    if _overlap_setting() == "off":
        return False
    setting = _setting()
    if setting not in ("auto", "max") and int(setting) == 1:
        return False
    return not _IN_POOL.get()


# ---- R21: the floating-point environment ----------------------------------------------------------------------------

_fp_api = None                           # the C runtime's _controlfp (own PyDLL instance), or False when unavailable


def _controlfp():
    global _fp_api
    if _fp_api is None:
        api = False
        if os.name == "nt":
            try:
                import ctypes
                # own instance (no shared argtypes are changed); PyDLL keeps the GIL across this tiny cdecl call, so a
                # worker task does not hand the GIL over and back just to read its control word
                api = ctypes.PyDLL("ucrtbase")._controlfp
                api.argtypes = [ctypes.c_uint, ctypes.c_uint]
                api.restype = ctypes.c_uint
            except (AttributeError, OSError):
                api = False
        _fp_api = api
    return _fp_api


def fp_environment() -> int | None:
    """The calling thread's denormal and rounding control bits (``_MCW_DN | _MCW_RC``), or None where the C runtime's
    ``_controlfp`` is not available (non-Windows: skipped, prereg §2.2)."""
    api = _controlfp()
    return None if not api else int(api(0, 0)) & FP_ENV_MASK


def _apply_fp_environment(wanted) -> bool:
    """Give the calling thread the control bits ``wanted`` (from fp_environment() in the coordinator); True when they
    differed and were set. A new Windows thread starts from the default control word, so a library that set FTZ/DAZ
    in the coordinator would otherwise make serial and parallel results differ on subnormal values."""
    if wanted is None:
        return False
    api = _controlfp()
    if int(api(0, 0)) & FP_ENV_MASK == wanted:
        return False
    api(wanted, FP_ENV_MASK)
    return True


# ---- per-run CPU (diagnostics only) -------------------------------------------------------------------------------

class _UnitClock:
    """A run's CPU and its longest unit, for the run check (prereg §6): ``unit_cpu_seconds`` sums each unit's thread
    CPU (``time.thread_time``: user + kernel of the thread that ran it); ``cpu_user_seconds`` / ``cpu_kernel_seconds``
    are this process's user / kernel CPU over the run (``os.times``: every thread, so a run that overlaps other work,
    O-1, is charged that work too). Two C-level clock reads per unit: no ctypes call (a ctypes call releases the GIL,
    which costs a pool of tiny units tens of microseconds each)."""
    __slots__ = ("unit_cpu", "longest", "process")

    def __init__(self):
        self.unit_cpu = self.longest = 0.0
        self.process = os.times()

    @staticmethod
    def start():
        return time.perf_counter(), time.thread_time()

    @staticmethod
    def measure(mark):
        return time.perf_counter() - mark[0], time.thread_time() - mark[1]

    def add(self, measured):                   # the parallel path calls it under the pool's lock
        wall, cpu = measured
        self.longest = max(self.longest, wall)
        self.unit_cpu += cpu

    def view(self) -> dict:
        end = os.times()
        return {"unit_cpu_seconds": round(self.unit_cpu, 4), "max_unit_seconds": round(self.longest, 4),
                "cpu_user_seconds": round(end.user - self.process.user, 4),
                "cpu_kernel_seconds": round(end.system - self.process.system, 4)}


# ---- in-process memory reservations ---------------------------------------------------------------------------------

def _byte_count(value, what) -> int:
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{what} must be a non-negative integer") from None
    if count < 0:
        raise ValueError(f"{what} must be a non-negative integer")
    return count


def _reserve(nbytes: int) -> None:
    global _reserved
    if nbytes:
        with _budget_lock:
            _reserved += nbytes


def _unreserve(nbytes: int) -> None:
    global _reserved
    if nbytes:
        with _budget_lock:
            _reserved -= nbytes


def reserved_bytes() -> int:
    """Bytes of in-process reservations outstanding now (``reserved`` sections and running pools' ``unit_bytes``)."""
    with _budget_lock:
        return _reserved


@contextmanager
def reserved(nbytes):
    """Hold ``nbytes`` of this process's memory as promised for the ``with`` block (O-1: the part of the main work not
    yet allocated, the scan processes' bytes). performance.available_memory() / available_commit() deduct every
    outstanding reservation, so pools deciding meanwhile see it. A reservation should not include a pool that declares
    ``unit_bytes`` itself: that pool reserves its own workers while they run."""
    count = _byte_count(nbytes, "nbytes")
    _reserve(count)
    try:
        yield count
    finally:
        _unreserve(count)


# ---- exclusive sections ---------------------------------------------------------------------------------------------

@contextmanager
def exclusive(name: str):
    """A process-wide exclusive section (the stock Siril stack launch, nox separation): every frame pool started
    anywhere in this process while one is open runs serially, reason ``exclusive``, its stats naming the section
    (``exclusive``), which the site's note records. Sections nest and count. Under the test seam _EXCLUSIVE_RAISES a
    pool that would have run in parallel raises ExclusiveViolation instead."""
    name = str(name)
    with _budget_lock:
        _exclusive[name] = _exclusive.get(name, 0) + 1
    try:
        yield
    finally:
        with _budget_lock:
            _exclusive[name] -= 1
            if not _exclusive[name]:
                del _exclusive[name]


def exclusive_sections() -> tuple:
    """Names of the exclusive sections open now (sorted)."""
    with _budget_lock:
        return tuple(sorted(_exclusive))


def resolve_workers(requested: int | None, *, units: int, cap: int = MAX_FRAME_WORKERS) -> int:
    """AESTER_FRAME_WORKERS if set (1..16, or 'max'), else ``requested`` (the caller's / governor's choice), else
    min(cap, usable CPUs); clamped to [1, min(16, units)]."""
    setting = _setting()
    if setting == "max":
        chosen = min(MAX_FRAME_WORKERS, usable_cpus())
    elif setting == "auto":
        chosen = requested if requested is not None else min(cap, usable_cpus())
    else:
        chosen = int(setting)
    if type(chosen) is not int or chosen < 1:
        raise ValueError("workers must be a positive integer")
    return max(1, min(chosen, MAX_FRAME_WORKERS, max(1, units)))


def default_workers(*, cap: int, unit_bytes: int, available_bytes: int | None, units: int,
                    blas: bool = True, why: dict | None = None) -> int:
    """Interim policy until the governor drives worker counts: <= cap, <= half the usable CPUs, memory like
    ``performance.render_workers`` (min(avail/3, avail - 1 GiB)) with each worker charged ``unit_bytes`` plus its
    persistent OpenBLAS slot.  Unknown memory: one worker.

    Outstanding in-process reservations (``reserved``, running pools' ``unit_bytes``) are deducted from the memory
    exactly once: ``available_bytes`` comes from performance.available_memory() / available_commit() (rule R20), which
    deduct them; subtracting them here as well would count them twice.  ``why`` (a dict) receives the bounds and the
    one that decided (cap | cpu | memory | units | memory-unknown), for timings notes only."""
    if available_bytes is None:
        if why is not None:
            why.update(bound="memory-unknown", available_bytes=None)
        return 1
    per_worker = max(1, int(unit_bytes) + (BLAS_SLOT_BYTES if blas else 0))
    budget = max(0, min(available_bytes // 3, available_bytes - (1 << 30)))
    bounds = {"cap": min(cap, MAX_FRAME_WORKERS), "cpu": max(1, usable_cpus() // 2), "memory": budget // per_worker,
              "units": units}
    workers = max(1, min(bounds.values()))
    if why is not None:
        why.update(bounds, bound=min(bounds, key=bounds.get), available_bytes=int(available_bytes),
                   per_worker_bytes=per_worker)
    return workers


def _acquire(n: int) -> int:
    global _budget_used
    with _budget_lock:
        granted = min(n, MAX_FRAME_WORKERS - _budget_used)
        if granted < 2:
            return 0
        _budget_used += granted
        return granted


def _release(n: int) -> None:
    global _budget_used
    if n:
        with _budget_lock:
            _budget_used -= n


def _snapshot() -> dict:
    return {var: value for var, value in contextvars.copy_context().items() if var is not _IN_POOL}


def _same(before: dict, where: str) -> None:
    after = _snapshot()
    changed = [v.name for v in set(before) | set(after)
               if v not in before or v not in after or before[v] is not after[v]]
    if changed:
        raise ContextLeak(f"frame_pool: {where} left ContextVar(s) changed: {sorted(changed)}")


def run_ordered(fn, items, *, workers: int, name: str, check=None, prepare=None, on_result=None, init=None,
                finalize=None, inflight: int | None = None, check_each: bool = True, stats: dict | None = None,
                unit_bytes: int | None = None, blas: bool = True):
    """``[fn(prepare(i, item, check), unit) for i, item in enumerate(items)]`` with serial-loop semantics (module
    docstring).

    Returns the list of results when ``on_result`` is None, else None.  ``inflight`` bounds the items submitted and
    not yet delivered (default 2 x workers, never below workers).  ``unit_bytes`` (P1c R10/R17: one unit's peak plus
    its share of in-flight payloads and results) makes the parallel path reserve ``granted x (unit_bytes + the BLAS
    slot when blas)`` while its worker threads run (``reserved_bytes``); the serial path runs in the caller's thread,
    whose memory the caller accounts for, and reserves nothing.  ``stats`` (a dict) receives diagnostics only:
    requested, workers, units, serial, reason (serial only), exclusive (when that is the reason), max_running,
    max_held, inflight, wall_seconds, unit_bytes, reserved_bytes, unit_cpu_seconds, max_unit_seconds,
    cpu_user_seconds, cpu_kernel_seconds (``_UnitClock``), fp_env (the coordinator's R21 control bits, hex) and
    fp_env_applied.
    """
    items = list(items)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if unit_bytes is not None:
        unit_bytes = _byte_count(unit_bytes, "unit_bytes")
    collect = [] if on_result is None else None
    deliver = on_result if on_result is not None else (lambda i, r: collect.append(r))
    wanted = min(workers, len(items))
    reason, granted, section = None, 0, None
    if wanted < 2:
        reason = "workers" if workers < 2 else "units"
    elif _IN_POOL.get():
        reason = "nested"
    else:
        sections = exclusive_sections()
        if sections:
            section = "+".join(sections)
            if _EXCLUSIVE_RAISES:
                raise ExclusiveViolation(f"frame_pool: pool {name!r} started inside exclusive section(s) {section}")
            reason = "exclusive"
        else:
            granted = _acquire(wanted)
            if not granted:
                reason = "budget"
    clock = _UnitClock()
    if reason is not None:
        started = time.perf_counter()
        _run_serial(fn, items, check=check, prepare=prepare, deliver=deliver, init=init, finalize=finalize,
                    check_each=check_each, clock=clock)
        if stats is not None:
            stats.update(requested=workers, workers=1, units=len(items), serial=True, reason=reason,
                         wall_seconds=round(time.perf_counter() - started, 4))
            if section is not None:
                stats["exclusive"] = section
            fp_env = fp_environment()          # the units ran in this thread: nothing to apply
            stats.update(unit_bytes=unit_bytes, reserved_bytes=0, **clock.view(),
                         fp_env=None if fp_env is None else hex(fp_env), fp_env_applied=0)
        return collect
    held = 0
    try:
        if unit_bytes is not None:
            held = granted * (unit_bytes + (BLAS_SLOT_BYTES if blas else 0))
            _reserve(held)
        err = _run_parallel(fn, items, workers=granted, name=name, check=check, init=init, finalize=finalize,
                            inflight=inflight, check_each=check_each, prepare=prepare, deliver=deliver,
                            stats=stats, requested=workers, clock=clock)
    finally:
        _unreserve(held)
        _release(granted)
    if stats is not None:
        stats.update(unit_bytes=unit_bytes, reserved_bytes=held, **clock.view())
    if err is not None:
        try:
            raise err
        finally:
            err = None                 # no frame -> exception cycle keeps states/results alive
    return collect


def _run_serial(fn, items, *, check, prepare, deliver, init, finalize, check_each, clock):
    audit = _AUDIT_CONTEXT
    base = _snapshot() if audit else None
    own_check = check if check is not None else _no_check
    state = init() if init is not None else None
    try:
        for i, item in enumerate(items):
            if check_each and check is not None:
                check()
            payload = prepare(i, item, own_check) if prepare is not None else item
            if audit:
                _same(base, f"prepare({i})")
            mark = clock.start()
            result = fn(payload, Unit(i, own_check, state))
            clock.add(clock.measure(mark))
            if audit:
                _same(base, f"unit {i}")
            deliver(i, result)
            if audit:
                _same(base, f"on_result({i})")
    finally:
        if finalize is not None:
            finalize(state)


def _run_parallel(fn, items, *, workers, name, check, init, finalize, inflight, check_each, prepare, deliver,
                  stats, requested, clock):
    """Runs the pool; returns the exception to raise (or None).  Never raises itself, so that its frame, which
    holds every state and result, is gone before the caller raises."""
    count = len(items)
    inflight = max(workers, 2 * workers if inflight is None else int(inflight))
    lock = threading.Lock()
    stop = [None]                              # lowest index with a known failure; -1: every unit (cancel)
    policy = _STATE_POLICY
    audit = _AUDIT_CONTEXT
    base = _snapshot() if audit else None
    states, idle, fresh = [], [], {}           # reuse: `workers` states, idle ones in `idle`; fresh: index -> state
    running = {"now": 0, "max": 0}
    pending, done, failures = {}, {}, {}       # future -> index; index -> finished future; index -> exception
    cancel = interrupt = None
    next_submit = next_deliver = max_held = 0
    executor = None
    started = time.perf_counter()
    last_check = time.monotonic()
    fp_env = fp_environment()                  # R21: the coordinator's control bits, given to every task's thread
    fp_applied = [0]

    def lower(i):
        with lock:
            stop[0] = i if stop[0] is None else min(stop[0], i)

    def task(i, payload, own_state):
        _IN_POOL.set(True)                     # this task's own context copy: never leaks to the caller
        if _apply_fp_environment(fp_env):
            with lock:
                fp_applied[0] += 1
        before = _snapshot() if audit else None

        def probe():
            s = stop[0]
            if s is not None and i > s:
                raise _Aborted()

        if own_state is None and states:
            with lock:
                own_state = idle.pop()         # exclusive while the task runs; returned below
            borrowed = True
        else:
            borrowed = False
        with lock:
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
        mark = clock.start()
        try:
            _before_unit(i)
            probe()
            result = fn(payload, Unit(i, probe, own_state))
            if audit:
                _same(before, f"unit {i}")
            return result
        finally:
            measured = clock.measure(mark)
            with lock:
                clock.add(measured)
                running["now"] -= 1
                if borrowed:
                    idle.append(own_state)

    def poll():
        nonlocal cancel, last_check
        try:
            check()
        except BaseException as exc:           # every check() exception is a cancel
            cancel = exc
            lower(-1)
        last_check = time.monotonic()
        if audit and cancel is None:
            _same(base, "check()")

    def guarded_check():
        """prepare's check (R15): the job's check() once per call until it raises; then only the kept cancel."""
        if cancel is None and check is not None:
            poll()
        if cancel is not None:
            raise cancel

    def harvest(finished):
        for future in finished:
            i = pending.pop(future)
            exc = future.exception()
            if isinstance(exc, _Aborted):
                continue
            done[i] = future
            if exc is not None:
                lower(i)                       # units above a known failure stop at their next probe

    try:
        if init is not None and policy != "fresh":
            for _ in range(workers):           # before check(0), as in the serial path
                states.append(init())          # inside the try: a failing init() finalizes the ones made
            idle.extend(states)
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"aester-{name}")
        while True:
            submitted = False
            if (cancel is None and stop[0] is None and next_submit < count
                    and next_submit - next_deliver < inflight):
                i = next_submit                # submit ONE unit, then deliver whatever is ready
                if check_each and check is not None:
                    poll()
                if cancel is None:
                    try:
                        payload = prepare(i, items[i], guarded_check) if prepare is not None else items[i]
                        if audit:
                            _same(base, f"prepare({i})")
                        own = None
                        if init is not None and policy == "fresh":
                            own = fresh[i] = init()
                    except BaseException as exc:
                        if cancel is None:     # a cancel seen through prepare's check stays a cancel (R15)
                            failures[i] = exc
                            lower(i)
                    else:
                        if cancel is None:     # a prepare that swallowed the cancel submits nothing
                            # a fresh context copy per task: one Context can be entered by one thread at a time
                            pending[executor.submit(contextvars.copy_context().run, task, i, payload, own)] = i
                            next_submit += 1
                            submitted = True
                        payload = own = None
            harvest([f for f in pending if f.done()])      # non-blocking
            max_held = max(max_held, len(done))
            while cancel is None and next_deliver in done:
                future = done.pop(next_deliver)
                exc = future.exception()
                if exc is None:
                    try:
                        deliver(next_deliver, future.result())
                        if audit:
                            _same(base, f"on_result({next_deliver})")
                    except BaseException as error:
                        exc = error
                future = None
                if exc is not None:
                    failures[next_deliver] = exc
                    lower(next_deliver)
                    exc = None
                    break
                next_deliver += 1
            if next_deliver == count or next_deliver in failures or cancel is not None:
                break
            if submitted:
                continue                       # keep the window full, delivering in between
            if not pending:                    # the head is submitted and unfinished whenever submission is closed
                if stop[0] is None and next_submit == next_deliver < count:
                    continue
                raise RuntimeError("frame_pool: internal error, nothing pending")
            timeout = POLL_SECONDS if check is None else max(0.0, POLL_SECONDS - (time.monotonic() - last_check))
            finished, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            harvest(finished)
            finished = None
            max_held = max(max_held, len(done))
            if next_deliver in done:
                continue                       # deliver first: a finished head is never turned into a cancel
            if check is not None and time.monotonic() - last_check >= POLL_SECONDS:
                poll()
    except BaseException as exc:               # KeyboardInterrupt in the coordinator, init() failure, pool machinery
        interrupt = exc
    complete = interrupt is None and next_deliver == count
    if not complete:
        lower(-1)
    try:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)   # every worker thread has ended after this line
    except BaseException as exc:
        interrupt = interrupt or exc
    err = None
    if interrupt is not None:
        err = interrupt
    elif not complete:
        if next_deliver in failures:
            err = failures[next_deliver]
        else:
            head = next((f for f, i in pending.items() if i == next_deliver), None) or done.get(next_deliver)
            head_exc = head.exception() if head is not None and head.done() and not head.cancelled() else None
            if head_exc is not None and not isinstance(head_exc, _Aborted):
                err = head_exc                 # the head unit ended with a failure of its own
            elif cancel is not None:
                err = cancel
            else:
                err = RuntimeError("frame_pool: stopped without a failure")
            head = head_exc = None
    fin_err = None
    if finalize is not None:
        for state in [*states, *(fresh[k] for k in sorted(fresh))]:
            try:
                finalize(state)                # every created state, each in its own try
            except BaseException as exc:
                if fin_err is None:
                    fin_err = exc
                else:
                    fin_err.add_note(f"frame_pool: another finalize() failed too: {exc!r}")
    if stats is not None:
        stats.update(requested=requested, workers=workers, units=count, serial=False, max_running=running["max"],
                     max_held=max_held, inflight=inflight, wall_seconds=round(time.perf_counter() - started, 4),
                     fp_env=None if fp_env is None else hex(fp_env), fp_env_applied=fp_applied[0])
    pending.clear(); done.clear(); failures.clear(); states.clear(); idle.clear(); fresh.clear()
    if fin_err is not None:
        if err is not None and fin_err.__context__ is None:
            fin_err.__context__ = err          # as an exception raised in the serial loop's `finally:`
        err = fin_err
    cancel = interrupt = fin_err = executor = None
    box = [err]                               # return it without a frame local that points at it: an exception
    err = None                                 # raised in this frame (prepare, on_result) has this frame in its
    return box.pop()                           # traceback, and a local reference would make a cycle


# ---- Beside: one helper thread next to the calling thread (O-1, prereg §3.6) ---------------------------------------

class Beside:
    """Runs ``fn(probe)`` on one helper thread named ``aester-beside-<name>`` while the calling thread goes on.

    - The helper runs in a copy of the caller's contextvars context (hash-ledger scope, numpy errstate travel with it)
      with ``in_pool()`` false, so pools it starts may run in parallel; they share the process-wide worker budget. The
      helper itself takes none of the MAX_FRAME_WORKERS slots. It starts with the caller's floating-point control bits
      (R21).
    - ``probe()`` (hand it to the helper's work as its check): once abandoned it raises ``_Abandoned``, a
      BaseException of its own; otherwise it calls the job's ``check`` (when given).
    - ``result(wait_check)`` waits, calling ``wait_check()`` at least every POLL_SECONDS; if that raises, the helper is
      abandoned and the exception re-raised. Returns the helper's value or raises its exception, and keeps no
      reference to either afterwards (a kept traceback would hold the helper's memory maps and intermediate results).
    - ``abandon()`` asks the helper to stop and by default waits until its thread has ended; the outcome is discarded,
      also one that arrives after ``abandon(wait=False)`` returned. Repeatable. It never raises except
      KeyboardInterrupt / SystemExit; anything else is recorded in ``errors`` and never replaces the caller's exception.
    """

    def __init__(self, fn, name: str, check=None):
        self._fn = fn
        self._check = check
        self._stop = threading.Event()
        self._lock = threading.Lock()          # orders storing the outcome against abandon() discarding it
        self._outcome = None                   # (value, exception) once the helper has ended; cleared when handed out
        self._taken = False
        self.errors = []
        self.fp_env_applied = False
        self.name = str(name)
        self._fp_env = fp_environment()
        context = contextvars.copy_context()
        self._thread = threading.Thread(target=context.run, args=(self._run,), name=f"aester-beside-{self.name}")

    @classmethod
    def start(cls, fn, name: str, check=None) -> "Beside":
        """Start ``fn(probe)`` on the helper thread and return the helper."""
        helper = cls(fn, name, check)
        helper._thread.start()
        return helper

    def _run(self):
        _IN_POOL.set(False)                    # this thread's own context copy: pools started here may fan out
        try:
            self.fp_env_applied = _apply_fp_environment(self._fp_env)
            value = self._fn(self.probe)
        except BaseException as exc:           # kept for result(); never escapes the thread
            self._keep((None, exc))
        else:
            self._keep((value, None))
        finally:
            self._fn = self._check = None

    def _keep(self, outcome) -> None:
        # Once abandoned nothing is stored: an outcome that arrives after abandon(wait=False) returned is dropped
        # here, one that arrived before is dropped by abandon(); the lock leaves no gap between the two.
        with self._lock:
            if not self._stop.is_set():
                self._outcome = outcome

    def probe(self) -> None:
        if self._stop.is_set():
            raise _Abandoned()
        check = self._check
        if check is not None:
            check()

    def done(self) -> bool:
        return not self._thread.is_alive()

    @property
    def abandoned(self) -> bool:
        return self._stop.is_set()

    def result(self, wait_check=None):
        if self._stop.is_set():
            raise RuntimeError(f"frame_pool.Beside {self.name}: result() after abandon()")
        if self._taken:
            raise RuntimeError(f"frame_pool.Beside {self.name}: result() was already taken")
        while True:
            self._thread.join(POLL_SECONDS)
            if not self._thread.is_alive():
                break
            if wait_check is not None:
                try:
                    wait_check()
                except BaseException:
                    self.abandon()
                    raise
        box, self._outcome, self._taken = [self._outcome], None, True
        value, exc = box.pop()
        if exc is not None:
            try:
                raise exc
            finally:
                exc = None                     # no frame -> exception cycle keeps the helper's results alive
        return value

    def abandon(self, *, wait: bool = True) -> None:
        try:
            with self._lock:
                self._stop.set()
                self._outcome = None           # discarded: a failed helper's traceback must not outlive it
            if wait and self._thread.ident is not None and self._thread is not threading.current_thread():
                while self._thread.is_alive():
                    self._thread.join(POLL_SECONDS)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self.errors.append(f"abandon: {type(exc).__name__}: {str(exc)[:200]}")
