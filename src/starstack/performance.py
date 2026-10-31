"""Small bounded resource decisions; no dependency on optional GPU packages.

Efficiency P1c-0 (docs/EFFICIENCY_P1C_PREREG_20261005.md §2.2, rules R19/R20): ``available_memory()`` and
``available_commit()`` are the one source of memory figures for worker decisions; both deduct this process's
outstanding in-process reservations (``frame_pool.reserved_bytes()``). The bench's memory / commit views (prereg §6.1)
belong on ``_memory_status`` (the readings before reservations), so that reservations are deducted from a view too; a
view wrapped around ``available_memory`` itself would hide them. ``frame_workers()`` is a frame-pool site's worker
decision and ``note_pool()`` its timings note (a list, one entry per call); ``operation()`` is the named timer
operation a site's pool runs in (the finishing sites, T-5/T-5x, whose callers hold no timer of their own).
"""
import os
from contextlib import ExitStack, contextmanager

from . import frame_pool


def _memory_status():
    """(available physical, available commit) bytes before reservations; either may be None."""
    try:
        if os.name == 'nt':
            import ctypes
            class Memory(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
            state = Memory()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.available), int(state.available_page)
        else:
            return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'), None
    except (AttributeError, OSError, ValueError):
        pass
    return None, None


def physical_total():
    """Installed physical memory in bytes (``ullTotalPhys`` / sysconf), or None when unreadable; never viewed or
    reduced by reservations (EFF-D4's memory budget uses it, P1c rule R20 keeps every memory figure here)."""
    try:
        if os.name == 'nt':
            import ctypes
            class Memory(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
            state = Memory()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)) and state.total:
                return int(state.total)
            return None
        return int(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')) or None
    except (AttributeError, OSError, ValueError):
        return None


def _net(value):
    return None if value is None else max(0, int(value) - frame_pool.reserved_bytes())


def available_memory():
    """Available physical memory minus this process's outstanding in-process reservations, or None when unknown."""
    return _net(_memory_status()[0])


def available_commit():
    """Available commit (``ullAvailPageFile``: what this process may still commit) minus this process's outstanding
    in-process reservations, or None when unknown (always None off Windows)."""
    return _net(_memory_status()[1])


def frame_workers(units, unit_bytes, cap, blas=True, *, extra_bytes=0, decision=None):
    """Worker count of a frame-pool site (rule R20): ``AESTER_FRAME_WORKERS`` when set, else
    ``frame_pool.default_workers`` over the smaller of available physical memory and available commit (both net of
    in-process reservations; an unknown commit is ignored, an unknown physical figure gives one worker) less
    ``extra_bytes`` (e.g. F-2's prepare-ahead decoding). ``decision`` (a dict) receives the bounds and the one that
    decided (override | cap | cpu | memory | units | memory-unknown), for timings notes only."""
    physical, commit = available_memory(), available_commit()
    available = None
    if physical is not None:
        available = max(0, min(physical, physical if commit is None else commit) - int(extra_bytes))
    why = {}
    auto = frame_pool.default_workers(cap=cap, unit_bytes=unit_bytes, available_bytes=available, units=units,
                                      blas=blas, why=why)
    workers = frame_pool.resolve_workers(auto, units=units, cap=cap)
    if decision is not None:
        overridden = frame_pool.worker_setting() != "auto"
        decision.update(why, workers=workers, unit_bytes=int(unit_bytes), blas=bool(blas),
                        extra_bytes=int(extra_bytes), bound="override" if overridden else why.get("bound"),
                        memory_source=None if physical is None else
                        ("commit" if commit is not None and commit < physical else "physical"))
    return workers


def note_pool(site, stats):
    """Append one pool run's diagnostics (the ``pool_stats`` dict given to ``frame_pool.run_ordered``) to the timings
    note ``frame_pool.<site>``: a list, one entry per call, so a site that runs twice never overwrites itself (R19).
    Diagnostics only: never a record, a signature or a digest."""
    from . import profiling
    profiling.note_append(f"frame_pool.{site}", stats)


@contextmanager
def operation(name):
    """``with performance.operation(name):`` -- the timer operation ``name`` on every profiling.StageTimer active in
    this thread (``profiling.activate``: the workflow worker's and a trail apply's timers), else on the live timer of
    the process's governor scope (``resources.scope_timer``: the job worker, which hands its timer around explicitly
    and does not activate it), so that the 10 Hz peak sampler charges a pool site's memory to it (P1c §6.2 (h), M3).
    A no-op without either (CLI, MCP, GUI). Diagnostics only: operations live under timings."""
    from . import profiling
    timers = profiling.active()
    if not timers:
        from . import resources
        timer = resources.scope_timer()
        timers = () if timer is None else (timer,)
    with ExitStack() as stack:
        for timer in timers:
            try:
                stack.enter_context(timer.operation(name))
            except Exception:          # diagnostics never fail the work they time
                pass
        yield


MAX_RENDER_WORKERS = 8      # docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md A: was 4
PREVIEW_MIN_PIXELS = 200_000  # previews and candidates: 640 px interactive views already gain from row parallelism


def render_workers(height, width, *, rows=128, cpu_count=None, memory_bytes=None, bytes_per_pixel=768,
                   min_pixels=4_000_000):
    """At most ``MAX_RENDER_WORKERS`` row groups and half the logical CPUs, with >=1 GiB left for UI and other operations.

    Estimate is deliberately conservative (``bytes_per_pixel`` of live float64
    temporaries per row pixel; looks pass a larger figure). Unknown memory falls
    back to one worker; images below ``min_pixels`` stay serial. Inside a frame-pool unit it is always one (P1c
    §6.2 (h): a unit never fans out again, rule R13).
    """
    override = os.environ.get('AESTER_RENDER_WORKERS', 'auto')
    if override not in ('auto',) + tuple(str(n) for n in range(1, MAX_RENDER_WORKERS + 1)):
        raise ValueError(f'AESTER_RENDER_WORKERS 只能为 auto 或 1–{MAX_RENDER_WORKERS}')
    ceiling = MAX_RENDER_WORKERS if override == 'auto' else int(override)
    if height * width < min_pixels or ceiling == 1 or frame_pool.in_pool():
        return 1
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    available = available_memory() if memory_bytes is None else memory_bytes
    if available is None:
        return 1
    budget = max(0, min(available // 3, available - 1024**3))
    per_worker = max(1, width * (rows + 32) * bytes_per_pixel)
    return max(1, min(ceiling, max(1, cpus // 2), budget // per_worker, (height + rows - 1) // rows))


def row_ranges(height, rows, workers):
    """Contiguous [start, stop) row ranges, each a whole number of ``rows`` blocks, at most ``workers`` of them.
    Every range starts on a block boundary, so a caller that walks each range in ``rows`` steps sees exactly the
    blocks a serial pass sees."""
    blocks = (height + rows - 1) // rows
    count = max(1, min(workers, blocks))
    stride = ((blocks + count - 1) // count) * rows
    return [(start, min(start + stride, height)) for start in range(0, height, stride)]


def run_row_ranges(fn, height, rows, workers, *, name='aester-render'):
    """``fn((start, stop))`` for each of ``row_ranges`` on a bounded thread pool; results in range order.
    Callers write disjoint output rows; an exception in any range is raised here."""
    ranges = row_ranges(height, rows, workers)
    if len(ranges) == 1:
        return [fn(ranges[0])]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix=name) as executor:
        return list(executor.map(fn, ranges))
