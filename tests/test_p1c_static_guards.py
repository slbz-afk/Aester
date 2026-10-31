"""Efficiency P1c static guards (docs/EFFICIENCY_P1C_PREREG_20261005.md §6.2 (h)), AST based.

- Thread pools, threads and process starts appear only at listed sites (POOL_SITES), including every
  ``frame_pool.run_ordered`` / ``performance.run_row_ranges`` / ``frame_pool.Beside.start`` call, every child
  process (``subprocess.*`` and ``guarded_process``; a Python child, ``sys.executable``, is its own kind), the GUI's
  Qt pools and threads (``QThreadPool``, ``QThread``, ``QRunnable`` subclasses) and the MCP servers' asyncio threads
  (``asyncio.to_thread``, ``run_in_executor``).
- CPU counts and memory figures come from the one source each (R20).
- R19: pool statistics stay in timings. The dict is always called ``pool_stats``, and every read of that name is the
  ``stats=`` of ``run_ordered`` (or of a listed wrapper that forwards it, POOL_STATS_FORWARDERS), an argument of a
  timings note (``performance.note_pool``, ``profiling.note`` / ``note_append``) or a ``.get()`` / ``[]`` read handed
  straight to ``resources.shadow_pool``; never a record value, an alias, a return value or a writer's argument.
- Every P1c pool site (P1C_SITES) declares unit_bytes, hands its pool statistics as ``pool_stats`` (R19), notes them
  with ``performance.note_pool('<site>', ...)``, records a ``resources.shadow_pool('py.<row>', ...)`` decision and runs
  inside a named timer operation (its run_ordered call sits in ``with <timer>.operation(name)``, or every call of its
  function does); its unit lists never iterate, unpack, list or add an unsorted set / glob / directory listing, also
  not through a local variable (R16).
- No algorithm or stage version constant changes (VERSIONS), apart from the D4 / NOX contract additions.
- ``performance.render_workers`` gives one worker inside a pool unit (checked at run time in test_p1c_frame_pool.py).

Sites are anchored by (file, enclosing function qualname, construct) with a count, not by line: the P1c items are
merged one after another into the same files, and a line anchor would break on every unrelated edit. A failure prints
``file:line`` of each offending call. A later P1c item adds its sites to POOL_SITES and P1C_SITES in its own commit.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "starstack"

# ------------------------------------------------------------------------------------------------ the scanner

THREAD_POOLS = {"ThreadPoolExecutor", "ProcessPoolExecutor"}
THREADS = {"Thread", "Timer"}                         # threading.Thread / threading.Timer (QTimer is another name)
# class bases that make a class a thread or a pool task: base name -> (kind, module it must come from or None)
THREAD_BASES = {"Thread": ("thread", "threading"), "QThread": ("qt_thread", None), "QRunnable": ("qt_runnable", None)}
SUBPROCESS_CALLS = {"Popen", "run", "call", "check_call", "check_output"}
OS_PROCESS_CALLS = {"system", "startfile", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp",
                    "spawnvpe", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
                    "posix_spawn", "posix_spawnp"}


def _dotted(node) -> str:
    """'a.b.c' for a Name / Attribute chain, '' otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _mentions_sys_executable(call) -> bool:
    return any(_dotted(node) == "sys.executable" for node in ast.walk(call))


def construct(call: ast.Call, imported: dict) -> str | None:
    """The kind of thread / pool / process start a call is, or None."""
    name = _dotted(call.func)
    last = name.rsplit(".", 1)[-1]
    head = name.split(".", 1)[0]
    origin = imported.get(head, head)
    if last in THREAD_POOLS:
        return "thread_pool"
    if last in THREADS and (name in ("threading.Thread", "threading.Timer") or
                            (name == last and imported.get(last) == "threading")):
        return "thread"
    if name in ("_thread.start_new_thread", "_thread.start_new"):
        return "thread"
    if last == "QThreadPool" or name.endswith("QThreadPool.globalInstance"):
        return "qt_pool"
    if last == "QThread":
        return "qt_thread"
    if name == "asyncio.to_thread" or (name == "to_thread" and imported.get("to_thread") == "asyncio"):
        return "async_thread"
    if last == "run_in_executor":
        return "async_thread"
    if origin.startswith("multiprocessing") or origin.startswith("concurrent.futures.process"):
        return "multiprocessing"
    if last == "run_ordered":
        return "frame_pool"
    if last == "run_row_ranges":
        return "row_pool"
    if name.endswith("Beside.start"):
        return "beside"
    if last == "guarded_process":
        return "python_process" if _mentions_sys_executable(call) else "process"
    if (name in {f"subprocess.{n}" for n in SUBPROCESS_CALLS} or
            (name == last and last in SUBPROCESS_CALLS and imported.get(last) == "subprocess")):
        return "python_process" if _mentions_sys_executable(call) else "process"
    if name in {f"os.{n}" for n in OS_PROCESS_CALLS}:
        return "process"
    return None


def _imports(tree) -> dict:
    """Local name -> module it came from (``from threading import Thread`` -> {'Thread': 'threading'})."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names[(alias.asname or alias.name).split(".")[0]] = alias.name
    return names


def class_construct(node: ast.ClassDef, imported: dict) -> str | None:
    """The kind of thread / pool task a class is by its bases (threading.Thread, QThread, QRunnable), or None."""
    for base in node.bases:
        name = _dotted(base)
        last = name.rsplit(".", 1)[-1]
        if last not in THREAD_BASES:
            continue
        kind, module = THREAD_BASES[last]
        if module is None or name == f"{module}.{last}" or (name == last and imported.get(last) == module):
            return kind
    return None


class _Calls(ast.NodeVisitor):
    """Every call with the qualname of its enclosing function or class ('<module>' at module level), and every class
    with its own qualname."""

    def __init__(self):
        self.stack, self.calls, self.classes = [], [], []

    def _scoped(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _scoped

    def visit_ClassDef(self, node):
        self.classes.append((".".join(self.stack + [node.name]), node))
        self._scoped(node)

    def visit_Call(self, node):
        self.calls.append((".".join(self.stack) or "<module>", node))
        self.generic_visit(node)


def calls_of(source: str):
    tree = ast.parse(source)
    visitor = _Calls()
    visitor.visit(tree)
    return tree, visitor.calls


def sites_in(source: str, relative: str) -> dict:
    """{(file, qualname, construct): [line, ...]} of one module's source (a thread class: its own qualname)."""
    tree = ast.parse(source)
    visitor = _Calls()
    visitor.visit(tree)
    imported = _imports(tree)
    found = {}
    for qualname, call in visitor.calls:
        kind = construct(call, imported)
        if kind is not None:
            found.setdefault((relative, qualname, kind), []).append(call.lineno)
    for qualname, node in visitor.classes:
        kind = class_construct(node, imported)
        if kind is not None:
            found.setdefault((relative, qualname, kind), []).append(node.lineno)
    return found


def package_files():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def package_sites() -> dict:
    found = {}
    for path in package_files():
        found.update(sites_in(path.read_text(encoding="utf-8"), path.relative_to(PACKAGE).as_posix()))
    return found


# ------------------------------------------------------------------------------------------------ the lists

# (file, qualname, construct): count. Prereg §6.2 (h) names core.py:805 (stage read-back), sky_validity.py:230,
# stack_range.py:64, finishing_service.py:24 and transient_scan.py:110; the rest are every other site that exists at
# the P1c base (41ce6f3) -- the pool machinery itself, hashing, render rows, the peak sampler, the volume probe,
# each child process the product starts, and the GUI's and MCP servers' threads.
POOL_SITES = {
    # pool machinery and its helpers
    ("frame_pool.py", "_run_parallel", "thread_pool"): 1,               # the frame pool's executor
    ("frame_pool.py", "Beside.__init__", "thread"): 1,                  # P1c-0: the O-1 helper thread
    ("performance.py", "run_row_ranges", "thread_pool"): 1,             # render row ranges
    ("hashing.py", "_many", "thread_pool"): 1,                          # P1 parallel file hashing
    ("hashing.py", "arrays_digest_many", "thread_pool"): 1,             # P1 parallel array digests
    ("components.py", "_probe", "thread"): 1,                           # 2026-10-09 download-source speed probe (I/O, daemon)
    ("profiling.py", "StageTimer._start_peak_sampler", "thread"): 1,    # P1b-2 peak sampler
    ("resources.py", "_bounded", "thread"): 1,                          # P1b-2 bounded volume query
    # single-thread helpers named by the prereg
    ("core.py", "_stage_inputs", "thread_pool"): 1,                     # prereg: core.py:805 read-back thread
    ("sky_validity.py", "_Digest.__init__", "thread_pool"): 1,          # prereg: sky_validity.py:230
    ("stack_range.py", "_ReadAhead.__init__", "thread_pool"): 1,        # prereg: stack_range.py:64
    ("finishing_service.py", "FinishingTools.__init__", "thread_pool"): 1,     # prereg: finishing_service.py:24
    # frame-pool and row-pool call sites
    ("sky_validity.py", "coverage_at", "frame_pool"): 1,                # P1b G-6 coverage
    ("imaging.py", "_median_explicit", "frame_pool"): 1,                # P1c C-1 foreground median blocks
    ("core.py", "_sky_zero", "frame_pool"): 1,                          # P1c C-3 sky zeroing per frame
    ("registered_support.py", "prepare_exports", "frame_pool"): 1,      # P1c F-2 strict-support exports
    ("registered_support.py", "native_mask", "frame_pool"): 1,          # P1c G-4t native mask row blocks
    ("finite_stack.py", "_source_inventory", "frame_pool"): 1,          # P1c F-3b finite_inventory
    ("finite_stack.py", "_write_inputs", "frame_pool"): 1,              # P1c F-3a finite_write
    ("transient_stage.py", "_crop_phase", "frame_pool"): 1,             # P1c T-3 review crops
    ("milkyway.py", "render", "frame_pool"): 1,                         # P1c T-5a milkyway_render
    ("star_reduction.py", "_reduce_rows", "frame_pool"): 1,             # P1c T-5b star_reduction
    ("workflow_edit.py", "add_signed_trails", "frame_pool"): 1,         # P1c T-5c trail_pair
    ("star_layer_export.py", "_export_rows", "frame_pool"): 1,          # P1c T-5x (a) star_layers
    ("photographic_job.py", "_starless_scope", "frame_pool"): 1,        # P1c T-5x (b) starless_scope.*
    ("star_reduction.py", "_advice_residuals", "frame_pool"): 1,        # P1c T-5x (b) starless_scope.advice_residuals
    ("photographic_job.py", "_jpeg_rows", "frame_pool"): 1,             # P1c T-5x (c) jpeg_rows.*
    ("region_blend.py", "inward_weight", "frame_pool"): 1,              # P1c T-5x (d) inward_weight
    ("core.py", "_Overlap.start", "beside"): 1,                         # P1c O-1: the trail scan beside the main thread
    ("display_transform.py", "render", "row_pool"): 1,
    ("master_export.py", "convert_camera_master", "row_pool"): 1,
    # child processes
    ("transient_scan.py", "scan_frames", "process"): 1,                 # prereg: transient_scan.py:110 (scan children)
    ("core.py", "_spawn_worker", "python_process"): 1,                  # the job worker
    ("workflow.py", "_spawn", "python_process"): 1,                     # the workflow worker
    ("process_guard.py", "_PopenProcess.start", "process"): 1,          # guarded_process itself
    ("siril.py", "_started", "process"): 1,
    ("siril.py", "doctor", "process"): 1,
    ("raw_import.py", "prepare_import", "process"): 1,
    ("finishing_backend.py", "render_recipe", "process"): 1,
    ("finishing_components.py", "_probe_version", "process"): 1,
    ("retouch_distribution.py", "_run_upstream_installer", "process"): 1,
    ("components.py", "_validate", "process"): 2,
    ("finishing_launch.py", "main", "process"): 1,
    # The desktop app's Qt threads and its restart now live in src/aester_desktop: tests/desktop/test_desktop_static_guards.py
}
# MCP threads: each tool runs its service call through asyncio.to_thread (asyncio's default executor), as does each
# server's lifespan when it closes. (file, enclosing function): the tools' function names.
ASYNC_THREAD_SITES = {
    ("finishing_mcp.py", "create_server"): ("lifespan",),
    ("finishing_mcp.py", "register_tools"): (
        "finishing_add_layers", "finishing_add_mask", "finishing_add_reference", "finishing_capabilities",
        "finishing_create", "finishing_draw_mask", "finishing_export", "finishing_feedback", "finishing_get",
        "finishing_preview", "finishing_recover", "finishing_revise", "finishing_tonal_mask"),
    ("mcp_server.py", "create_server"): (
        "apply_trail_review", "cancel_job", "cancel_workflow", "coordinate_background", "create_regions",
        "draw_regions", "export_master", "export_star_layers", "finish_milkyway", "finish_photo", "get_regions",
        "import_photos", "inspect_raw", "inspect_sources", "inspect_workflow_cache", "job_status", "lifespan",
        "list_transients",
        "open_trail_review", "prepare_reference", "preview", "preview_regions", "protect_trace", "reduce_stars",
        "render_job_preview", "render_photo", "resume_job", "resume_workflow", "retouch_setup", "rotate_reference",
        "save_trail_review", "set_transients", "submit_job", "submit_workflow", "suggest_star_reduction",
        "trail_candidate_context", "update_regions", "workflow_status"),
}
POOL_SITES.update({(file, f"{outer}.{name}", "async_thread"): 1
                   for (file, outer), names in ASYNC_THREAD_SITES.items() for name in names})

# R20: os.cpu_count() only here (frame_pool.usable_cpus is the source for new pools); memory figures only here
# (performance.available_memory / available_commit for new pools).
CPU_COUNT_SITES = {
    ("frame_pool.py", "usable_cpus"), ("hashing.py", "resolve_workers"), ("performance.py", "render_workers"),
    ("profiling.py", "_measure_host"), ("resources.py", "cpu_topology"), ("siril.py", "siril_omp_threads"),
    ("stack_settings.py", "processor_groups"),          # EFF-D4: a processor-group size off Windows, not a worker count
}
MEMORY_API = re.compile(r"GlobalMemoryStatusEx|SC_AVPHYS_PAGES|psutil|ullAvail|available_page")
MEMORY_SOURCE_FILES = {"performance.py", "profiling.py", "resources.py"}

# Legacy run_ordered calls (before R19 named the argument pool_stats): file, qualname.
LEGACY_STATS_NAMES = {("sky_validity.py", "coverage_at")}


@dataclass(frozen=True)
class P1cSite:
    """One P1c pool site (prereg §2.3): where its run_ordered call is, where it notes and shadows the decision, and
    the timer operation it runs in."""
    site: str                              # note key frame_pool.<site>
    pool: tuple                            # (file, qualname) of the run_ordered call
    footprint: str                         # resources row, e.g. "py.median"
    note: tuple                            # (file, qualname) calling performance.note_pool('<site>', ...)
    shadow: tuple                          # (file, qualname) calling resources.shadow_pool('<footprint>', ...)
    # The timer operation the pool runs inside (10 Hz sampler, M3): (file, name), or (file, qualname, name) to look in
    # one function only. Either the run_ordered call sits in ``with <timer>.operation(name)`` within ``pool``, or every
    # call of (reference to) the pool function in that file / function sits in such a with-block or in
    # ``<timer>.call(name, ...)``.
    operation: tuple
    unit_lists: tuple = ()                 # (file, qualname) of functions that build its unit lists or records (R16)


# Each P1c item adds its site here (C-1 median, C-3 sky_zero, F-2 strict_support, G-4t native_mask, F-3a/F-3b
# finite_write / finite_inventory, T-3 trail_crops, T-5a/b/c milkyway_render / star_reduction / trail_pair, T-5x,
# T-7 addback_boxes).
_STARLESS_SCOPE = ("photographic_job.py", "_starless_scope")
_JPEG_ROWS = ("photographic_job.py", "_jpeg_rows")
P1C_SITES: tuple = (
    # C-1: imaging._median_explicit runs its blocks in the job timer's operation foreground_median (core passes the
    # timer through median_foreground); the float / former routes stay serial and note reason 'route'.
    P1cSite("median", ("imaging.py", "_median_explicit"), "py.median", ("imaging.py", "_median_explicit"),
            ("imaging.py", "_median_explicit"), ("imaging.py", "_median_explicit", "foreground_median")),
    # C-3: core._sky_zero runs in the job timer's operation sky_zero (inside source_sky_validity); the re-zeroing
    # route of pre-sky-validity jobs keeps the serial loop (_sky_zero_reference).
    P1cSite("sky_zero", ("core.py", "_sky_zero"), "py.sky_zero", ("core.py", "_sky_zero"), ("core.py", "_sky_zero"),
            ("core.py", "_sky_zero", "sky_zero"), unit_lists=(("core.py", "_ensure_sky_validity"),)),
    # F-2: decided, noted and shadowed in core._supported_stack, whose prepare_exports call is in the
    # "registered_support" operation; R16: the export list and record (prepare_exports), F's checkpoint files
    P1cSite("strict_support", ("registered_support.py", "prepare_exports"), "py.strict_exports",
            ("core.py", "_supported_stack"), ("core.py", "_supported_stack"),
            ("core.py", "_supported_stack", "registered_support"), unit_lists=(("core.py", "_finite_stack_files"),)),
    # G-4t: decided, noted and shadowed in core._run_locked; finish_stack runs native_mask in the "native_mask"
    # operation (core hands it the timer's operation)
    P1cSite("native_mask", ("registered_support.py", "native_mask"), "py.native_mask",
            ("core.py", "_run_locked"), ("core.py", "_run_locked"),
            ("registered_support.py", "finish_stack", "native_mask")),
    # F-3b / F-3a (prereg §3.3): finite_stack is in the frozen holdout closure (R22), so core hands run() a Pools
    # (core._FinitePools) whose note_pool / shadow_pool / operation are performance.note_pool, resources.shadow_pool and
    # the job timer's operation; the calls sit in finite_stack itself.
    P1cSite("finite_inventory", ("finite_stack.py", "_source_inventory"), "py.finite_inventory",
            ("finite_stack.py", "_source_inventory"), ("finite_stack.py", "_source_inventory"),
            ("finite_stack.py", "_source_inventory", "finite_inventory"), (("finite_stack.py", "run"),)),
    P1cSite("finite_write", ("finite_stack.py", "_write_inputs"), "py.finite_write",
            ("finite_stack.py", "_write_inputs"), ("finite_stack.py", "_write_inputs"),
            ("finite_stack.py", "_write_inputs", "finite_write"), (("finite_stack.py", "run"),)),   # run(): the receipt
    # T-3: one frame's review crops a unit; the pool runs in the timer operation "trail_crops" (core passes its timer to
    # detect_job); detect_job builds the unit list (frame_order: sorted) and _crop_phase the record's candidates.
    P1cSite("trail_crops", ("transient_stage.py", "_crop_phase"), "py.trail_crops",
            ("transient_stage.py", "_crop_phase"), ("transient_stage.py", "_crop_phase"),
            ("transient_stage.py", "_crop_phase", "trail_crops"),
            unit_lists=(("transient_stage.py", "detect_job"), ("transient_stage.py", "frame_order"))),
    # T-5a / T-5b / T-5c (prereg §3.5): the pool, its note, its shadow decision and its timer operation in one function
    P1cSite("milkyway_render", ("milkyway.py", "render"), "py.milkyway_render", ("milkyway.py", "render"),
            ("milkyway.py", "render"), ("milkyway.py", "milkyway_render")),
    P1cSite("star_reduction", ("star_reduction.py", "_reduce_rows"), "py.star_reduction",
            ("star_reduction.py", "_reduce_rows"), ("star_reduction.py", "_reduce_rows"),
            ("star_reduction.py", "star_reduction")),
    P1cSite("trail_pair", ("workflow_edit.py", "add_signed_trails"), "py.trail_pair",
            ("workflow_edit.py", "add_signed_trails"), ("workflow_edit.py", "add_signed_trails"),
            ("workflow_edit.py", "trail_pair")),
    # T-5x (a)-(d): one note key per call site (prereg §3.5); shared pool functions forward pool_stats to the caller
    P1cSite("star_layers", ("star_layer_export.py", "_export_rows"), "py.star_layers",
            ("star_layer_export.py", "_export_rows"), ("star_layer_export.py", "_export_rows"),
            ("star_layer_export.py", "star_layers")),
    P1cSite("starless_scope.advice", _STARLESS_SCOPE, "py.starless_scope",
            ("star_reduction.py", "suggest_star_reduction"), ("star_reduction.py", "suggest_star_reduction"),
            ("photographic_job.py", "starless_scope")),
    P1cSite("starless_scope.reduction", _STARLESS_SCOPE, "py.starless_scope", ("star_reduction.py", "reduce_stars"),
            ("star_reduction.py", "reduce_stars"), ("photographic_job.py", "starless_scope")),
    P1cSite("starless_scope.milkyway", _STARLESS_SCOPE, "py.starless_scope", ("photographic_job.py", "_finish_display"),
            ("photographic_job.py", "_finish_display"), ("photographic_job.py", "starless_scope")),
    P1cSite("starless_scope.advice_residuals", ("star_reduction.py", "_advice_residuals"), "py.starless_scope",
            ("star_reduction.py", "_advice_residuals"), ("star_reduction.py", "_advice_residuals"),
            ("star_reduction.py", "advice_residuals")),
    P1cSite("jpeg_rows.reduction", _JPEG_ROWS, "py.jpeg_rows", ("star_reduction.py", "reduce_stars"),
            ("star_reduction.py", "reduce_stars"), ("photographic_job.py", "jpeg_rows")),
    P1cSite("jpeg_rows.trail_pair", _JPEG_ROWS, "py.jpeg_rows", ("workflow_edit.py", "_write_pair"),
            ("workflow_edit.py", "_write_pair"), ("photographic_job.py", "jpeg_rows")),
    # 2026-10-09 JPEG quality: a workflow that froze jpeg_quality re-encodes its delivered JPEGs at publication
    P1cSite("jpeg_rows.publish", _JPEG_ROWS, "py.jpeg_rows", ("workflow.py", "_encode_jpeg"),
            ("workflow.py", "_encode_jpeg"), ("photographic_job.py", "jpeg_rows")),
    P1cSite("inward_weight", ("region_blend.py", "inward_weight"), "py.inward_weight",
            ("region_blend.py", "inward_weight"), ("region_blend.py", "inward_weight"),
            ("region_blend.py", "inward_weight")),
)

# ------------------------------------------------------------------------------------------------ the tests


def _where(found: dict, key) -> str:
    return ", ".join(f"src/starstack/{key[0]}:{line}" for line in found.get(key, []))


def test_pool_and_process_sites_are_the_listed_ones():
    found = package_sites()
    unlisted = {key: lines for key, lines in found.items() if key not in POOL_SITES}
    miscounted = {key: (len(found[key]), POOL_SITES[key]) for key in found
                  if key in POOL_SITES and len(found[key]) != POOL_SITES[key]}
    gone = sorted(key for key in POOL_SITES if key not in found)
    assert not unlisted, "unlisted thread / pool / process sites (add them to POOL_SITES with their prereg item): " + \
        "; ".join(f"{key} at {_where(found, key)}" for key in sorted(unlisted))
    assert not miscounted, f"site counts changed (found, listed): {miscounted}"
    assert not gone, f"listed sites no longer exist (remove them from POOL_SITES): {gone}"


def test_the_scanner_sees_every_kind_of_start():
    source = '''
import asyncio, subprocess, sys, threading, os
from asyncio import to_thread
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import multiprocessing as mp
from PySide6.QtCore import QRunnable, QThread, QThreadPool, QTimer
from .process_guard import guarded_process
from . import frame_pool, performance

def f(timer):
    ThreadPoolExecutor(2)
    threading.Thread(target=f)
    Thread(target=f)
    threading.Timer(1, f)
    mp.Pool(2)
    subprocess.Popen([sys.executable, "-m", "x"])
    subprocess.run(["siril-cli"])
    guarded_process(["siril-cli"])
    os.startfile("x")
    frame_pool.run_ordered(f, [], workers=1, name="x")
    performance.run_row_ranges(f, 1, 1, 1)
    frame_pool.Beside.start(f, "x")

async def g(loop):
    QThreadPool(None)
    QThreadPool.globalInstance().start(None)
    QThread()
    await asyncio.to_thread(f)
    await to_thread(f)
    await loop.run_in_executor(None, f)

class C:
    def m(self):
        QTimer(self)          # Qt timers are not threads
        self.timer.start()

    class Task(QRunnable):
        pass

class Worker(QThread):
    pass

class Plain(threading.Thread):
    pass

class Other(Thread):
    pass

class NotAThread(object):
    pass
'''
    found = sites_in(source, "x.py")
    kinds = sorted((qualname, kind, len(lines)) for (_, qualname, kind), lines in found.items())
    assert kinds == [("C.Task", "qt_runnable", 1), ("Other", "thread", 1), ("Plain", "thread", 1),
                     ("Worker", "qt_thread", 1),
                     ("f", "beside", 1), ("f", "frame_pool", 1), ("f", "multiprocessing", 1), ("f", "process", 3),
                     ("f", "python_process", 1), ("f", "row_pool", 1), ("f", "thread", 3), ("f", "thread_pool", 1),
                     ("g", "async_thread", 3), ("g", "qt_pool", 2), ("g", "qt_thread", 1)]


def test_r20_cpu_counts_and_memory_figures_have_one_source():
    unlisted, memory = [], []
    for path in package_files():
        relative = path.relative_to(PACKAGE).as_posix()
        source = path.read_text(encoding="utf-8")
        _, calls = calls_of(source)
        for qualname, call in calls:
            if _dotted(call.func) == "os.cpu_count" and (relative, qualname) not in CPU_COUNT_SITES:
                unlisted.append(f"src/starstack/{relative}:{call.lineno} ({qualname})")
        if relative not in MEMORY_SOURCE_FILES:
            for number, line in enumerate(source.splitlines(), 1):
                if MEMORY_API.search(line) and not line.lstrip().startswith("#"):
                    memory.append(f"src/starstack/{relative}:{number}")
    assert not unlisted, "os.cpu_count() outside the listed sites (use frame_pool.usable_cpus(), R20): " + \
        ", ".join(unlisted)
    assert not memory, "memory figures read outside performance / resources / profiling (R20): " + ", ".join(memory)
    cpu_found = set()
    for path in package_files():
        relative = path.relative_to(PACKAGE).as_posix()
        _, calls = calls_of(path.read_text(encoding="utf-8"))
        cpu_found.update((relative, q) for q, call in calls if _dotted(call.func) == "os.cpu_count")
    assert cpu_found == CPU_COUNT_SITES, f"listed os.cpu_count sites that no longer exist: {CPU_COUNT_SITES - cpu_found}"


def _stats_argument(call):
    for keyword in call.keywords:
        if keyword.arg == "stats":
            return keyword.value
    return None


def test_r19_run_ordered_statistics_are_pool_stats():
    """R19: the statistics argument is always called pool_stats (never a dict that goes into a record)."""
    offenders = []
    for path in package_files():
        relative = path.relative_to(PACKAGE).as_posix()
        _, calls = calls_of(path.read_text(encoding="utf-8"))
        for qualname, call in calls:
            if _dotted(call.func).rsplit(".", 1)[-1] != "run_ordered" or relative == "frame_pool.py":
                continue
            value = _stats_argument(call)
            if value is None or (isinstance(value, ast.Name) and value.id == "pool_stats"):
                continue
            if (relative, qualname) not in LEGACY_STATS_NAMES:
                offenders.append(f"src/starstack/{relative}:{call.lineno} stats={ast.unparse(value)}")
    assert not offenders, "run_ordered(stats=...) must pass a dict named pool_stats (R19): " + ", ".join(offenders)


# R19 wrappers that take the pool statistics and hand them on to run_ordered: (file, qualname) -> parameter name. Their
# parameter is held to the same rules as pool_stats. A P1c item whose site calls a wrapper adds the wrapper here.
POOL_STATS_FORWARDERS = {
    ("sky_validity.py", "coverage_map"): "stats",       # P1b G-6: core._sky_coverage -> coverage_map -> coverage_at
    ("sky_validity.py", "coverage_at"): "stats",
    ("registered_support.py", "prepare_exports"): "pool_stats",   # P1c F-2: core._supported_stack -> prepare_exports
    ("registered_support.py", "finish_stack"): "pool_stats",      # P1c G-4t: core -> finish_stack -> native_mask
    ("registered_support.py", "native_mask"): "pool_stats",
    # P1c T-5x (b)/(c): one shared pool function per kind, one note key per call site
    ("photographic_job.py", "_starless_scope"): "pool_stats",
    ("star_reduction.py", "_prepare_inputs"): "pool_stats",     # advice and reduction -> _starless_scope
    ("photographic_job.py", "_jpeg_rows"): "pool_stats",
}
NOTE_CALLS = {"note_pool", "note", "note_append"}       # performance.note_pool, profiling.note / note_append: timings
FORWARD_KEYWORDS = {"stats", "pool_stats"}


def _last(node) -> str:
    return _dotted(node).rsplit(".", 1)[-1]


def _parents(tree) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def stats_leaks(scope, name: str, relative: str, forwarders=None) -> list:
    """R19: reads of ``name`` in ``scope`` (a module or a function) that could carry pool statistics outside timings.
    Allowed: ``stats=`` / ``pool_stats=`` of run_ordered or of a forwarder; an argument of a timings note (NOTE_CALLS);
    a ``.get()`` / ``[]`` read that is itself an argument of resources.shadow_pool. Anything else -- a record or dict
    value, a store, an alias, a return value, an f-string, a writer's argument -- is reported as ``file:line``."""
    forwarders = POOL_STATS_FORWARDERS if forwarders is None else forwarders
    callees = {"run_ordered"} | {qualname.rsplit(".", 1)[-1] for _, qualname in forwarders}
    parents = _parents(scope)
    found = []
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
            continue
        parent = parents.get(node)
        holder = parents.get(parent)
        if isinstance(parent, ast.keyword) and isinstance(holder, ast.Call):
            if parent.arg in FORWARD_KEYWORDS and _last(holder.func) in callees:
                continue
            if _last(holder.func) in NOTE_CALLS:
                continue
        if isinstance(parent, ast.Call) and node in parent.args and _last(parent.func) in NOTE_CALLS:
            continue
        read = None
        if isinstance(parent, ast.Attribute) and parent.attr == "get" and isinstance(holder, ast.Call) \
                and holder.func is parent:
            read = holder
        elif isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
            read = parent
        if read is not None:
            target = parents.get(read)
            if isinstance(target, ast.keyword):
                target = parents.get(target)
            if isinstance(target, ast.Call) and read is not target.func and _last(target.func) == "shadow_pool":
                continue
        context = parent if parent is not None else node
        found.append(f"src/starstack/{relative}:{node.lineno} {ast.unparse(context)[:100]}")
    return found


def _function_node(tree, qualname):
    def find(nodes, prefix):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                if name == qualname:
                    return node
                hit = find(node.body, name)
                if hit is not None:
                    return hit
        return None
    return find(tree.body, "")


def test_r19_pool_stats_stay_in_timings():
    """§6.2 (h) "no new keys outside timings": pool_stats (and a forwarder's parameter) only reach run_ordered, the
    timings notes and shadow_pool, never a record, a job dict or a writer."""
    leaks = []
    for path in package_files():
        relative = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        leaks += stats_leaks(tree, "pool_stats", relative)
        for (file, qualname), parameter in POOL_STATS_FORWARDERS.items():
            if file == relative:
                function = _function_node(tree, qualname)
                assert function is not None, f"listed forwarder {file}:{qualname} no longer exists"
                arguments = function.args
                names = [a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]
                assert parameter in names, f"forwarder {file}:{qualname} has no parameter {parameter!r}"
                leaks += stats_leaks(function, parameter, relative)
    assert not leaks, "pool statistics outside timings (R19): " + "; ".join(leaks)


def test_the_r19_checker_finds_each_leak():
    source = '''
from . import frame_pool, performance, profiling, records, resources, sky_validity as sv
def site(job, record, check):
    pool_stats = {}
    frame_pool.run_ordered(f, [1, 2], workers=2, name="demo", stats=pool_stats)
    sv.coverage_map(1, 2, stats=pool_stats)
    performance.note_pool("demo", pool_stats)
    profiling.note("frame_pool.demo", pool_stats)
    profiling.note_append("frame_pool.demo", value=pool_stats)
    resources.shadow_pool("py.demo", actual=pool_stats.get("workers"), items=pool_stats["units"])
    record["pool"] = pool_stats
    job["workers"] = pool_stats.get("workers")
    records.write(job, stats=pool_stats)
    alias = pool_stats
    log(f"{pool_stats['workers']} workers")
    resources.shadow_pool("py.demo", actual=4, items=3, extra=pool_stats)
    return pool_stats

def wrap(items, stats=None, other=None):
    frame_pool.run_ordered(f, items, workers=2, name="w", stats=stats)
    other["s"] = stats
'''
    tree = ast.parse(source)
    leaks = stats_leaks(tree, "pool_stats", "x.py")
    lines = sorted(int(leak.split(":")[1].split()[0]) for leak in leaks)
    assert lines == [11, 12, 13, 14, 15, 16, 17], leaks                 # lines 5-10 are the allowed uses
    assert any("record['pool'] = pool_stats" in leak for leak in leaks)
    forwarder = {("x.py", "wrap"): "stats"}
    wrap_leaks = stats_leaks(_function_node(tree, "wrap"), "stats", "x.py", forwarders=forwarder)
    assert [leak.split()[0] for leak in wrap_leaks] == ["src/starstack/x.py:21"], wrap_leaks
    unlisted = stats_leaks(tree, "pool_stats", "x.py", forwarders={})          # coverage_map is not a forwarder here
    assert sorted(int(leak.split(":")[1].split()[0]) for leak in unlisted) == [6] + lines


# ---- per P1c site --------------------------------------------------------------------------------------------------

def _literal_first_argument(call):
    return call.args[0].value if call.args and isinstance(call.args[0], ast.Constant) else None


def site_problems(site: P1cSite, read=lambda relative: (PACKAGE / relative).read_text(encoding="utf-8")) -> list:
    """What a P1c site lacks of §6.2 (h); empty when it complies. ``read`` maps a file to its source (tests)."""
    problems = []

    def function(where):
        return _function_node(ast.parse(read(where[0])), where[1])

    def calls(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call)] if node is not None else []

    pool = function(site.pool)
    pools = [c for c in calls(pool) if _last(c.func) == "run_ordered"]
    if pool is None or not pools:
        problems.append(f"{site.site}: no run_ordered call in {site.pool}")
    for call in pools:
        keywords = {k.arg: k.value for k in call.keywords}
        if "unit_bytes" not in keywords:
            problems.append(f"{site.site}: run_ordered at {site.pool[0]}:{call.lineno} declares no unit_bytes (R10)")
        stats = keywords.get("stats")
        if not (isinstance(stats, ast.Name) and stats.id == "pool_stats"):
            problems.append(f"{site.site}: run_ordered at {site.pool[0]}:{call.lineno} has no stats=pool_stats (R19)")
    notes = [c for c in calls(function(site.note)) if _last(c.func) == "note_pool"
             and _literal_first_argument(c) == site.site]
    if not notes:
        problems.append(f"{site.site}: no note_pool({site.site!r}, ...) in {site.note}")
    shadows = [c for c in calls(function(site.shadow)) if _last(c.func) == "shadow_pool"
               and _literal_first_argument(c) == site.footprint]
    if not shadows:
        problems.append(f"{site.site}: no shadow_pool({site.footprint!r}, ...) in {site.shadow}")
    problems.extend(f"{site.site}: {message}" for message in operation_problems(site, pool, pools, read))
    for where in (site.pool,) + tuple(site.unit_lists):
        problems.extend(f"{site.site}: {message}" for message in unsorted_iterations(function(where), where[0]))
    return problems


def _operation_regions(scope, name) -> list:
    """The code that runs inside the timer operation ``name`` within ``scope``: the bodies of
    ``with <timer>.operation(name)`` and the arguments of ``<timer>.call(name, function, ...)``."""
    regions = []
    for node in ast.walk(scope) if scope is not None else ():
        if isinstance(node, (ast.With, ast.AsyncWith)):
            if any(isinstance(item.context_expr, ast.Call) and _last(item.context_expr.func) == "operation"
                   and _literal_first_argument(item.context_expr) == name for item in node.items):
                regions.extend(node.body)
        elif isinstance(node, ast.Call) and _last(node.func) == "call" and _literal_first_argument(node) == name:
            regions.extend(node.args[1:])
            regions.extend(k.value for k in node.keywords)
    return regions


def _inside(regions, target) -> bool:
    return any(node is target for region in regions for node in ast.walk(region))


def operation_problems(site: P1cSite, pool, pools, read) -> list:
    """The site's run_ordered calls run inside its timer operation: lexically inside it within ``site.pool``, or
    the pool function is only ever called (or handed on) from inside it in ``site.operation``'s file / function."""
    file, name = site.operation[0], site.operation[-1]
    scope_name = site.operation[1] if len(site.operation) == 3 else None
    tree = ast.parse(read(file))
    scope = tree if scope_name is None else _function_node(tree, scope_name)
    where = f"{file}" + (f" {scope_name}" if scope_name else "")
    regions = _operation_regions(scope, name)
    if not regions:
        return [f"no timer operation {name!r} in {where}"]
    own = _operation_regions(pool, name) if file == site.pool[0] and scope_name in (None, site.pool[1]) else []
    outside = [call for call in pools if not _inside(own, call)]
    if not outside:
        return []
    function_name = site.pool[1].rsplit(".", 1)[-1]
    uses = [node for node in ast.walk(scope)
            if (isinstance(node, ast.Name) and node.id == function_name)
            or (isinstance(node, ast.Attribute) and node.attr == function_name)]
    if uses and all(_inside(regions, node) for node in uses):
        return []
    lines = ", ".join(f"{site.pool[0]}:{call.lineno}" for call in outside)
    stray = ", ".join(f"{file}:{node.lineno}" for node in uses if not _inside(regions, node)) or "none"
    return [f"run_ordered at {lines} runs outside timer operation {name!r}: not inside it in {site.pool[1]}, and "
            f"{function_name}() is not called only from inside it in {where} (calls outside: {stray})"]


# ---- R16 -------------------------------------------------------------------------------------------------------------

UNORDERED = {"glob", "rglob", "iterdir", "listdir", "scandir", "set", "frozenset", "walk"}
KEEPS_ORDER = {"list", "tuple", "enumerate", "reversed", "iter"}  # their first argument's order (or lack of one)
MATERIALISE = {"list", "tuple"}                                   # an unordered result fixed into a sequence
ADDERS = {"extend": 0, "append": 0, "update": 0, "insert": 1}     # method -> the argument added to a sequence / dict
ORDER_FREE = {"sorted", "set", "frozenset", "len", "any", "all"}  # consumers whose result does not depend on order


def _unordered_source(node, tainted=None) -> str | None:
    """The unordered producer an expression yields (not through sorted()), or None. ``tainted`` maps local names that
    hold such a result at this point to their producer."""
    tainted = tainted or {}
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "a set"
    if isinstance(node, ast.Name):
        return f"{node.id} (= {tainted[node.id]})" if node.id in tainted else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _unordered_source(node.left, tainted) or _unordered_source(node.right, tainted)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return next(filter(None, (_unordered_source(g.iter, tainted) for g in node.generators)), None)
    if isinstance(node, ast.Call):
        last = _last(node.func)
        if last == "sorted":
            return None
        if last in UNORDERED:
            return f"{last}()"
        if last in KEEPS_ORDER and node.args:
            return _unordered_source(node.args[0], tainted)
        if last in ("zip", "chain"):
            return next(filter(None, (_unordered_source(a, tainted) for a in node.args)), None)
        if last in ("map", "filter") and len(node.args) > 1:
            return next(filter(None, (_unordered_source(a, tainted) for a in node.args[1:])), None)
    return None


def _taint_timeline(function_node) -> dict:
    """Local name -> [(line, producer or None)] in source order: what each assignment leaves in the name (``x += y``
    keeps an unordered x unordered and makes x unordered when y is)."""
    assignments = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign):
            assignments += [(node, target, node.value, False) for target in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            assignments.append((node, node.target, node.value, False))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            assignments.append((node, node.target, node.value, True))
    timeline, now = {}, {}
    for node, target, value, augmented in sorted(assignments, key=lambda a: (a[0].lineno, a[0].col_offset)):
        if not isinstance(target, ast.Name):
            continue
        what = _unordered_source(value, now)
        if augmented:
            what = what or now.get(target.id)
        timeline.setdefault(target.id, []).append((node.lineno, what))
        if what:
            now[target.id] = what
        else:
            now.pop(target.id, None)
    return timeline


def _tainted_at(timeline, line) -> dict:
    """The names holding an unordered result just before ``line`` (the last assignment above it decides)."""
    tainted = {}
    for name, events in timeline.items():
        before = [what for at, what in events if at < line]
        if before and before[-1]:
            tainted[name] = before[-1]
    return tainted


def unsorted_iterations(function_node, relative: str) -> list:
    """R16: a set, glob, iterdir, listdir, scandir or walk result not passed through sorted() that a function iterates
    (loop, comprehension, run_ordered items), unpacks (``*x``), lists (``list(x)`` / ``tuple(x)``) or adds to a sequence
    or mapping (``extend`` / ``append`` / ``insert`` / ``update`` / ``+=``), directly or through a local variable.
    Order-free consumers (sorted, set, frozenset, len, any, all) and set comprehensions are fine."""
    if function_node is None:
        return []
    parents = _parents(function_node)
    timeline = _taint_timeline(function_node)
    found, seen = [], set()

    def order_free(node):
        holder = parents.get(node)
        return isinstance(holder, ast.Call) and node in holder.args and _last(holder.func) in ORDER_FREE

    def report(verb, node, line=None):
        line = line or node.lineno
        what = _unordered_source(node, _tainted_at(timeline, line))
        if what is not None and (line, what) not in seen:
            seen.add((line, what))
            found.append(f"{relative}:{line} {verb} {what} unsorted (R16)")

    nodes = list(ast.walk(function_node))
    for node in nodes:                                  # iterations first: their message wins on a shared line
        if isinstance(node, (ast.For, ast.AsyncFor)):
            report("iterates", node.iter)
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp)) and not order_free(node):
            for generator in node.generators:
                report("iterates", generator.iter)
        elif isinstance(node, ast.Call) and _last(node.func) == "run_ordered":
            items = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "items"), None)
            if items is not None:
                report("iterates", items)
    for node in nodes:
        if isinstance(node, ast.Starred) and not isinstance(parents.get(node), ast.Set):
            report("unpacks", node.value)
        elif isinstance(node, ast.Call):
            last = _last(node.func)
            if last in MATERIALISE and node.args and not order_free(node):
                report("lists", node.args[0], node.lineno)
            elif isinstance(node.func, ast.Attribute) and last in ADDERS and len(node.args) > ADDERS[last]:
                report("adds", node.args[ADDERS[last]])
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            report("adds", node.value)
    return found


def test_every_p1c_site_notes_shadows_and_declares_its_memory():
    problems = [problem for site in P1C_SITES for problem in site_problems(site)]
    assert not problems, "\n".join(problems)
    for site in P1C_SITES:
        from starstack import resources
        assert site.footprint in resources.FOOTPRINTS, site
        assert (site.pool[0], site.pool[1], "frame_pool") in POOL_SITES, f"{site.site}: list its pool in POOL_SITES"


def test_the_site_checker_finds_each_missing_piece():
    """The checker itself, on synthetic sources (the registry is empty until the first P1c item lands)."""
    good = {
        "a.py": '''
from . import frame_pool, performance, resources
def site(paths, timer):
    pool_stats = {}
    with timer.operation("demo_op"):
        frame_pool.run_ordered(f, sorted(paths.glob("*.npy")), workers=2, name="demo", unit_bytes=8,
                               stats=pool_stats)
    performance.note_pool("demo", pool_stats)
    resources.shadow_pool("py.demo", actual=pool_stats.get("workers"), items=3)
''',
        "b.py": '''
from . import frame_pool
def site(paths, check):
    stats = {}
    for p in paths.glob("*.npy"):
        pass
    frame_pool.run_ordered(f, list({1, 2}), workers=2, name="demo", stats=stats)
''',
    }
    spec = P1cSite("demo", ("a.py", "site"), "py.demo", ("a.py", "site"), ("a.py", "site"), ("a.py", "demo_op"))
    assert site_problems(spec, read=good.__getitem__) == []
    bad = P1cSite("demo", ("b.py", "site"), "py.demo", ("b.py", "site"), ("b.py", "site"), ("b.py", "demo_op"))
    problems = "\n".join(site_problems(bad, read=good.__getitem__))
    for expected in ("declares no unit_bytes", "no stats=pool_stats", "no note_pool('demo'",
                     "no shadow_pool('py.demo'", "no timer operation 'demo_op'", "b.py:5 iterates glob() unsorted",
                     "b.py:7 iterates a set unsorted"):
        assert expected in problems, (expected, problems)


def _site(file, operation=None):
    return P1cSite("demo", (file, "site"), "py.demo", (file, "site"), (file, "site"), operation or (file, "demo_op"))


SITE_BODY = '''
    pool_stats = {}
    frame_pool.run_ordered(f, sorted(paths), workers=2, name="demo", unit_bytes=8, stats=pool_stats)
    performance.note_pool("demo", pool_stats)
    resources.shadow_pool("py.demo", actual=pool_stats.get("workers"), items=3)
'''


def test_the_site_checker_relates_the_pool_to_its_timer_operation():
    """The run_ordered call runs inside the named operation: lexically, or every call of the pool function does."""
    sources = {
        "caller.py": "def site(paths):" + SITE_BODY + '''
def outer(timer, paths):
    with timer.operation("demo_op"):
        site(paths)
    timer.call("demo_op", site, paths)
''',
        "elsewhere.py": "def site(paths, timer):" + SITE_BODY + '''
def unrelated(timer):
    with timer.operation("demo_op"):
        pass
''',
        "partly.py": "def site(paths):" + SITE_BODY + '''
def outer(timer, paths):
    with timer.operation("demo_op"):
        site(paths)
    site(paths)
''',
        "sibling.py": '''
def site(paths, timer):
    with timer.operation("demo_op"):
        pass''' + SITE_BODY,
    }
    read = sources.__getitem__
    assert site_problems(_site("caller.py"), read=read) == []
    assert site_problems(_site("caller.py", ("caller.py", "outer", "demo_op")), read=read) == []
    for file, stray in (("elsewhere.py", "none"), ("partly.py", "partly.py:10"), ("sibling.py", "none")):
        problems = "\n".join(site_problems(_site(file), read=read))
        assert "runs outside timer operation 'demo_op'" in problems and f"calls outside: {stray}" in problems, \
            (file, problems)
    problems = "\n".join(site_problems(_site("caller.py", ("caller.py", "site", "demo_op")), read=read))
    assert "no timer operation 'demo_op' in caller.py site" in problems


def test_the_r16_checker_sees_unpacking_variables_and_listing():
    """R16 shapes beyond a plain loop: the core.py:1129 unpacking (prereg §2.1), a listing kept in a variable, a list()
    into a record, extend / += with a listing, a run_ordered over a variable; and the order-free ones that pass."""
    bad = ast.parse('''
def site(work, fs, stack_files, record, run_ordered):
    stack_files.extend([work / fs.NVALID, work / fs.NUSED, *work.glob('frame_support_light_*.npy')])
    names = list(work.glob('*.npy'))
    for n in names:
        pass
    out = [str(p) for p in work.glob('y')]
    record["files"] = tuple(os.listdir(work))
    stack_files += work.iterdir()
    later = work.glob("z")
    run_ordered(f, later, workers=2)
    pairs = zip(range(3), names)
    for i, n in pairs:
        pass
    stack_files.extend(p.name for p in work.rglob("*"))
    merged = sorted(work.glob("a")) + list({1, 2})
    record["merged"] = merged[0]
    stack_files.extend(merged)
''').body[0]
    found = unsorted_iterations(bad, "core.py")
    text = "\n".join(found)
    for expected in ("core.py:3 unpacks glob() unsorted", "core.py:4 lists glob() unsorted",
                     "core.py:5 iterates names (= glob()) unsorted", "core.py:7 iterates glob() unsorted",
                     "core.py:8 lists listdir() unsorted", "core.py:9 adds iterdir() unsorted",
                     "core.py:11 iterates later (= glob()) unsorted", "core.py:13 iterates pairs (= names (= glob()))",
                     "core.py:15 iterates rglob() unsorted", "core.py:16 lists a set unsorted",
                     "core.py:18 adds merged (= a set) unsorted"):
        assert expected in text, (expected, found)
    good = ast.parse('''
def site(work, stack_files, record, run_ordered):
    names = sorted(p.name for p in work.glob("*.npy"))
    for n in names:
        pass
    count = len(list(work.glob("*")))
    stems = {p.stem for p in work.glob("*")}
    again = work.glob("b")
    again = sorted(again)
    for p in again:
        pass
    stack_files.extend([work / "a", *sorted(work.glob("frame_support_light_*.npy"))])
    record["files"] = [str(p) for p in sorted(os.listdir(work))]
    run_ordered(f, sorted(stems), workers=2)
    present = any(p.suffix == ".npy" for p in work.iterdir())
    unique = set(work.iterdir()) | {*work.glob("c")}
''').body[0]
    assert unsorted_iterations(good, "core.py") == []


# ---- versions --------------------------------------------------------------------------------------------------------

VERSION_NAME = re.compile(r"^_?[A-Z0-9_]*VERSION$")
# Every module-level *VERSION constant at the P1c base (41ce6f3), FOOTPRINTS_VERSION at its P1c-0 value (it only
# reaches timings). P1c changes none of them (§6.2 (h), A16) except the contract items' own changes or additions
# (EFF-D4, EFF-NOX), listed in ALLOWED_VERSION_CHANGES when they land.
VERSIONS = {
    "background.py:VERSION": "background-plane-display-1",
    # REL-CACHE option A (in-app per-job cache cleanup, 2026-10-09): record formats of a new module, no algorithm
    "cache_cleanup.py:PLAN_VERSION": "aester-app-cache-plan-1",
    "cache_cleanup.py:RESULT_VERSION": "aester-app-cache-result-1",
    "cache_cleanup.py:MARKER_VERSION": "aester-app-cache-cleaned-1",
    "cache_cleanup.py:GC_VERSION": "app-cache-cleanup-1",
    "cache_cleanup.py:LISTING_VERSION": "aester-app-cache-listing-1",
    # Read-side copy of the literal the uninstaller writes (uninstall_options.py, execute_cache_cleanup).
    "cache_cleanup.py:UNINSTALL_MARKER_VERSION": "aester-cache-cleaned-1",
    "camera_looks.py:VERSION": "camera-looks-2",
    "color_adjustments.py:VERSION": "color-adjustments-1",
    "color_export.py:VERSION": "color-export-1",
    "devices.py:VERSION": "devices-1",
    "devices_registry.py:REGISTRY_VERSION": "devices-registry-5",
    "disk_budget.py:VERSION": "disk-estimate-2",
    "display_transform.py:VERSION": "display-transform-2",
    "experimental_dcp.py:VERSION": "experimental-z8-dcp-d65-float-1",
    "finishing_components.py:TESTED_GIMP_VERSION": "3.2.6",
    "finite_backend_assets.py:RUNTIME_VERSION": "finite-validity-v1",
    "finite_stack.py:VERSION": "1.0.0",
    "fits_crop.py:FITS_CROP_VERSION": "1.0.0",
    "fits_input.py:VERSION": "fits-exposure-1",
    "gc.py:VERSION": "gc-1",
    "ground_finish.py:VERSION": "ground-photographic-v2",
    "hashing.py:VERSION": "aester-hashing-3",
    "hashing.py:HANDOVER_VERSION": "aester-hash-handover-2",
    "holdout_catalog.py:CATALOG_VERSION": "1.0.0",
    "horizon_acceptance.py:HORIZON_ACCEPTANCE_VERSION": "2.0.0",
    "icc_profiles.py:VERSION": "icc-profiles-1",
    "interpolation_support.py:SUPPORT_VERSION": "1.0.0",
    "master_export.py:VERSION": "master-export-1",
    "milkyway.py:VERSION": "masked-milkyway-4",
    "milkyway_gate.py:VERSION": "milkyway-subject-evidence-1",
    "milkyway_metrics.py:_VERSION": "display-comparison-1",
    "photographic.py:VERSION": "display-photographic-prototype-4",
    "photographic_job.py:VERSION": "photographic-job-1",
    "photographic_job.py:MILKYWAY_VERSION": "milkyway-job-3",
    "photo_render.py:VERSION": "photo-render-2",
    "profiling.py:VERSION": "monotonic-stages-2",
    "profiling.py:LEGACY_VERSION": "monotonic-stages-1",
    "profiling.py:RESOURCES_VERSION": "process-resources-2",
    "profiling.py:HOST_VERSION": "host-profile-1",
    "registered_support.py:SUPPORT_VERSION": "1.0.0",
    "registration_adapter.py:ADAPTER_VERSION": "0.2.2",
    "registration_calibration.py:CALIBRATION_VERSION": "1.4.1",
    "registration_calibration.py:QUALITY_GATE_VERSION": "1.0.1",
    "registration_diagnostics.py:SCHEMA_VERSION": 1,
    "registration_diagnostics.py:MODULE_VERSION": "1.0.1",
    "registration_distortion.py:DISTORTION_VERSION": "1.3.1",
    "registration_holdout.py:HOLDOUT_VERSION": "1.0.0",
    "registration_models.py:MODELS_VERSION": "1.0.0",
    "registration_refit.py:REFIT_VERSION": "1.0.0",
    "resample.py:RESAMPLE_VERSION": "1.0.1",
    "resources.py:VERSION": "aester-resources-1",
    "resources.py:FOOTPRINTS_VERSION": "footprints-2026-10-05a",
    "retouch_distribution.py:GIMP_VERSION": "3.2.6",
    "retouch_distribution.py:SKILL_VERSION": "nl2-v5.1",
    "seam_validation.py:METRIC_VERSION": "2.0.0",
    "sip_header.py:SIP_HEADER_VERSION": "1.2.0",
    "siril_coordinate_proof.py:PROOF_VERSION": "1.0.0",
    "siril_nodata_proof.py:NODATA_PROOF_VERSION": "1.1.0",
    "siril_sequence.py:SEQ_PARSER_VERSION": "1.1.0",
    "siril_sip_proof.py:SIP_PROOF_VERSION": "1.5.0",
    "siril_support_map.py:SIRIL_SUPPORT_MAP_VERSION": "1.0.0",
    "sky_validity.py:SKY_VALIDITY_VERSION": "1.1.1",
    "stack_range.py:VERSION": "1.1.0",
    "starless_metrics.py:VERSION": "starless-metrics-1",
    "starless_metrics2.py:VERSION": "starless-metrics-2",
    "starless_metrics3.py:VERSION": "starless-metrics-3.1",
    "starless_metrics3.py:TRUTH_VERSION": "starless-injection-3",
    "starless_metrics3.py:E1_VERSION": "starless-e1-guards-3.1",
    "starless_runtime.py:VERSION": "starless-runtime-1.1",
    "star_density.py:VERSION": "compact-star-density-1",
    "star_layer_audit.py:_VERSION": "signed-star-layer-audit-1",
    "star_layer_export.py:VERSION": "editable-star-layers-1",
    "star_reduction.py:VERSION": "star-reduction-job-1",
    "star_separation.py:VERSION": "star-separation-experiment-3",
    "storage.py:VERSION": "aester-delivery-1",
    "storage.py:IMAGE_VERSION": "aester-delivery-2",
    "synthetic_starfield.py:GENERATOR_VERSION": "1.1.0",
    "trail_finish.py:VERSION": "trail-photographic-v3",
    "trail_review_session.py:VERSION": "trail-review-session-1",
    "trail_review_session.py:EXPLICIT_POLICY_VERSION": "explicit-keep-only-1",
    "trail_review_session.py:METEOR_POLICY_VERSION": "meteor-default-keep-1",
    "transient_review.py:VERSION": "transient-recall-review-1",
    "transient_stage.py:STAGE_VERSION": "1.4.1",
    "transient_support.py:VERSION": "1.0.0",
    "transient_trails.py:TRANSIENT_VERSION": "1.2.1",
    "transient_trails.py:MANUAL_ASSOCIATION_VERSION": "1.0.0",
    "uninstall_options.py:PLAN_VERSION": "aester-uninstall-cache-1",
    "whole_color.py:VERSION": "whole-color-linked-tone-1",
    "workflow.py:VERSION": "workflow-1",
    "workflow_edit.py:VERSION": "workflow-reviewed-trails-1",
    "workflow_edit.py:LAYER_VERSION": "reviewed-trails-layer-1",
}
# name -> reason: the only changes or additions P1c may make (EFF-D4 / EFF-NOX add theirs in their own commits).
ALLOWED_VERSION_CHANGES = {
    "stack_settings.py:VERSION": "EFF-D4 contract version (prereg 4.2), new with K2",
    "stack_settings.py:ENGINE_VERSION": "EFF-D4: the Siril build the contract applies to, new with K2",
}


def module_versions() -> dict:
    versions = {}
    for path in package_files():
        relative = path.relative_to(PACKAGE).as_posix()
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                if (isinstance(target, ast.Name) and VERSION_NAME.match(target.id) and node.value is not None
                        and isinstance(node.value, ast.Constant)):
                    versions[f"{relative}:{target.id}"] = node.value.value
    return versions


def test_no_algorithm_or_stage_version_changes():
    current = module_versions()
    changed = {key: (VERSIONS.get(key), value) for key, value in current.items()
               if VERSIONS.get(key, object()) != value and key not in ALLOWED_VERSION_CHANGES}
    removed = sorted(key for key in VERSIONS if key not in current and key not in ALLOWED_VERSION_CHANGES)
    assert not changed, f"version constants changed or added (snapshot, now): {changed}"
    assert not removed, f"version constants removed: {removed}"
