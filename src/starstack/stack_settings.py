"""EFF-D4: the stock Siril stack's thread and memory settings for L, strict L and R jobs (efficiency P1c, contract K2).

docs/EFFICIENCY_P1C_PREREG_20261005.md §4.2.  Standard library only at import (astropy is imported lazily for one
FITS header); siril.py, in the frozen holdout closure, never imports this module: core computes every value here and
passes them to the Siril runner.

Why (Siril 1.4.4, upstream source checked against the finite build, which leaves these functions unchanged):
  - the stack's normalization runs ``nb = min(max_memory_MB / per-frame MB, com.max_thread, frames)`` workers
    (stacking/normalization.c normalization_get_max_number_of_threads and compute_normalization);
  - each worker gets ``compute_thread_distribution(nb, com.max_thread)`` inner threads (core/processing.c);
  - with more than one inner thread a frame's statistics depend on the thread count (the bwmv OpenMP
    ``reduction(+)`` in algos/statistics_float.c) and, when the split is uneven, on the ``schedule(guided)`` order;
  - rejection and averaging are per pixel (stacking/median_and_mean.c), independent of threads and memory blocks.
So the contract ``setcpu T'`` with ``T' = nb`` gives every normalization worker exactly one inner thread: the master
then depends only on the data and the Siril 1.4.4 build.  T' and the budget B are machine-dependent schedule values,
never recorded except in the timings note ``siril_stack_settings``.

The rule:
  n   the selected frames, parsed from the text of r_light_.seq (never through the hash ledger);
  p   the per-frame MB of the first selected frame, as normalization.c computes it: float data
      w*h*(layers+2)*4 bytes, 16-bit data w*h*((layers+1)*2+8) bytes, divided by 2**20 (integer), at least 1;
  cap a pinned cpu (AESTER_SIRIL_PIN), else min(usable_cpus(), the smallest active processor group, 64): a child
      may start in another processor group, and Siril refuses a setcpu above the processors it sees (the groups
      are read once per process; the governor's resources module stays shadow-only and is not used);
  B   a pinned mem in GiB, else min(floor(0.6 x physical total), floor(0.9 x available now)) GiB, the available
      memory from performance.available_memory() (P1c R20); an unreadable total falls back to the total read
      earlier in this process, an unreadable total or available memory to 4 GiB (flagged "fallback");
  T' = min(cap, n, B*1024 // p); B*1024 // p < 1 is refused before Siril starts.
Script lines ``setcpu T'``, ``set core.mem_mode=1``, ``set core.mem_amount=B`` go right before ``seqclean`` (after any
pin lines, so they win); the Siril process runs with OMP_NUM_THREADS=1 and without OMP_DYNAMIC, OMP_THREAD_LIMIT and
OMP_MAX_ACTIVE_LEVELS (siril.run_script(stack_settings=True)).  After the stack, this attempt's log section must hold
exactly one ``Memory per image: p MB. Max memory: B*1024 MB`` line and one ``allocating data for T' threads`` line,
and (L, strict L) the master's STACKCNT must equal n: then max_thread = T' <= B*1024/p and STACKCNT = n, so every
normalization worker had one inner thread.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

VERSION = "siril-stack-settings-1"
ENGINE_VERSION = "1.4.4"
RULE = "setcpu equals normalization workers; explicit core.mem_amount"
APPLIES_TO = ("legacy", "strict-legacy", "positive_power2_v1")
FINITE_MODE = "finite_validity_v1"          # finite_stack.MODE: the finite engine has its own normalization
FIELD = "stack_settings"                    # admission_options.stack_settings (job signature preimage)
NOTE_KEY = "siril_stack_settings"           # timings note only
CONFIG_NAME = "siril-stack.ini"             # work/siril-stack.ini: L / strict-L 02_stack's own Siril configuration
SEQUENCE_NAME = "r_light_.seq"
THREAD_CEILING = 64                        # Siril's OpenMP sees at most one processor group (<= 64 processors)
FALLBACK_GIB = 4
MIB, GIB = 1 << 20, 1 << 30
REFUSED = "堆栈设置合同已改变，请新建作业"

# The untranslated stdout lines of Siril's stack (normalization.c, median_and_mean.c); the patterns of
# scripts/replay_legacy_stack.py.
MEMORY_LINE = re.compile(r"Memory per image: (\d+) MB\. Max memory: (\d+) MB")
THREADS_LINE = re.compile(r"allocating data for (\d+) threads")
_SCRIPT_LINES = (re.compile(r"setcpu [1-9][0-9]*"), re.compile(r"set core\.mem_mode=1"),
                 re.compile(r"set core\.mem_amount=[1-9][0-9]*"))

_last_total = None                          # the physical total read last in this process (fallback for a failed read)
_groups = None                              # processor groups, read once per process


def contract() -> dict:
    """A fresh copy of the contract written into ``admission_options.stack_settings``."""
    return {"version": VERSION, "engine_version": ENGINE_VERSION, "rule": RULE, "applies_to": list(APPLIES_TO)}


def admission_field(engine: dict | None, stack_range_mode: str) -> dict | None:
    """The field a new job gets: the contract for every non-finite mode on Siril 1.4.4, else None (no field)."""
    if stack_range_mode == FINITE_MODE or (engine or {}).get("version") != ENGINE_VERSION:
        return None
    return contract()


def job_contract(job: dict) -> dict | None:
    """The contract a job was admitted under, or None for a job without the field (it keeps its old scripts).

    A field that is not exactly the contract, or that sits on a finite job or a job whose engine is not 1.4.4, is
    refused (ValueError), like any other changed admission option."""
    options = job.get("admission_options")
    if not isinstance(options, dict) or FIELD not in options:
        return None
    if (options[FIELD] != contract() or job.get("stack_range_mode", "legacy") == FINITE_MODE
            or (job.get("engine") or {}).get("version") != ENGINE_VERSION):
        raise ValueError(REFUSED)
    return contract()


# --------------------------------------------------------------------------- the plan

def data_type(bitpix: int) -> str:
    """'ushort' or 'float' for a FITS BITPIX, as Siril's get_data_type reads it (io/image_format_fits.c)."""
    if bitpix in (8, 16):
        return "ushort"
    if bitpix in (32, -32, -64):
        return "float"
    raise ValueError(f"Siril 堆栈不支持 BITPIX={bitpix} 的帧")


def frame_mb(width: int, height: int, layers: int, bitpix: int) -> int:
    """normalization.c's per-frame memory in MB (integer division by 2**20, at least 1)."""
    pixels = int(width) * int(height)
    if data_type(bitpix) == "float":
        size = pixels * (int(layers) + 2) * 4
    else:
        size = pixels * ((int(layers) + 1) * 2 + 2 * 4)
    return max(1, size // MIB)


def selected_frames(sequence: Path) -> tuple[int, str]:
    """(n, stem of the first selected frame) from the text of a Siril sequence file."""
    from .siril_sequence import parse_seq_text
    parsed = parse_seq_text(Path(sequence).read_text(encoding="utf-8", errors="strict"))
    chosen = [image["stem"] for image in parsed["images"] if image["selected"]]
    if not chosen or len(chosen) != parsed["header"]["nb_selected"]:
        raise ValueError(f"{Path(sequence).name} 的选片数量无效（{len(chosen)}／{parsed['header']['nb_selected']}）")
    return len(chosen), chosen[0]


def frame_geometry(path: Path) -> tuple[int, int, int, int]:
    """(width, height, layers, BITPIX) from a FITS primary header."""
    from astropy.io import fits
    header = fits.getheader(path)
    naxis = int(header.get("NAXIS", 0))
    if naxis not in (2, 3):
        raise ValueError(f"{Path(path).name} 不是二维或三维图像")
    layers = int(header["NAXIS3"]) if naxis == 3 else 1
    return int(header["NAXIS1"]), int(header["NAXIS2"]), layers, int(header["BITPIX"])


def thread_cap(pin: dict | None, *, usable: int | None = None, groups=None) -> tuple[int, str]:
    """(cap, source): a pinned cpu, else min(usable CPUs, the smallest active processor group, 64)."""
    if pin and "cpu" in pin:
        return int(pin["cpu"]), "pin"
    if usable is None:
        from .frame_pool import usable_cpus
        usable = usable_cpus()
    if groups is None:
        groups = processor_groups()
    sizes = [int(g) for g in (groups or ()) if int(g) > 0]
    return max(1, min([int(usable), THREAD_CEILING] + ([min(sizes)] if sizes else []))), "machine"


def processor_groups() -> tuple:
    """Logical processors per active processor group, read once per process (elsewhere: one group of cpu_count);
    () when unknown.  Its own kernel32 instance: no shared argtypes are changed."""
    global _groups
    if _groups is None:
        found = ()
        if os.name == "nt":
            try:
                import ctypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                count, per_group = kernel32.GetActiveProcessorGroupCount, kernel32.GetActiveProcessorCount
                count.restype, count.argtypes = ctypes.c_uint16, []
                per_group.restype, per_group.argtypes = ctypes.c_uint32, [ctypes.c_uint16]
                groups = int(count())
                found = tuple(int(per_group(g)) for g in range(groups)) if 0 < groups <= 64 else ()
            except (AttributeError, OSError):
                found = ()
        elif os.cpu_count():
            found = (os.cpu_count(),)
        _groups = found
    return _groups


def _read_total_memory() -> int | None:
    """Physical memory in bytes, None when unreadable (performance.physical_total: rule R20's one memory source)."""
    from . import performance
    return performance.physical_total()


def _total_memory() -> int | None:
    """The physical total now, else the one read last in this process."""
    global _last_total
    total = _read_total_memory()
    if total:
        _last_total = total
        return total
    return _last_total


def memory_gib(pin: dict | None, *, total: int | None = None, available: int | None = None) -> tuple[int, str]:
    """(B, source): a pinned mem, else min(floor(0.6 total), floor(0.9 available)) GiB; 4 GiB ("fallback") when the
    total or the available memory cannot be read.  ``total``/``available`` (bytes) default to this machine now."""
    if pin and "mem" in pin:
        return int(pin["mem"]), "pin"
    if total is None:
        total = _total_memory()
    if available is None:
        from .performance import available_memory
        available = available_memory()
    if not total or available is None:
        return FALLBACK_GIB, "fallback"
    return min(6 * int(total) // (10 * GIB), 9 * int(available) // (10 * GIB)), "runtime"


def threads(cap: int, frames: int, memory: int, per_frame_mb: int) -> int:
    """T' = min(cap, n, B*1024 // p); refused (ValueError) when the budget holds less than one frame."""
    fit = int(memory) * 1024 // int(per_frame_mb)
    if fit < 1:
        raise ValueError(f"Siril 堆栈的内存额度 {memory} GiB 不足以容纳一帧（每帧 {per_frame_mb} MB），"
                         "请关闭其他程序释放内存后恢复作业")
    if frames < 1:
        raise ValueError("堆栈没有选中的帧")
    return max(1, min(int(cap), int(frames), fit))


def plan(folder: Path, *, pin: dict | None = None, script: str = "02_stack", usable: int | None = None, groups=None,
         total: int | None = None, available: int | None = None) -> dict:
    """The settings of one contract stack over ``folder``'s r_light_.seq and its first selected frame."""
    folder = Path(folder)
    frames, first = selected_frames(folder / SEQUENCE_NAME)
    width, height, layers, bitpix = frame_geometry(folder / f"{first}.fit")
    per_frame = frame_mb(width, height, layers, bitpix)
    cap, cap_source = thread_cap(pin, usable=usable, groups=groups)
    memory, memory_source = memory_gib(pin, total=total, available=available)
    return {"version": VERSION, "script": script, "frames": frames, "frame_mb": per_frame,
            "threads": threads(cap, frames, memory, per_frame), "memory_gib": memory,
            "thread_cap": cap, "thread_cap_source": cap_source, "memory_source": memory_source}


def script_lines(settings: dict) -> list[str]:
    return [f"setcpu {settings['threads']}", "set core.mem_mode=1", f"set core.mem_amount={settings['memory_gib']}"]


def with_settings(commands, settings: dict) -> list[str]:
    """``commands`` with the three lines right before the first ``seqclean`` (after any pin lines)."""
    commands = list(commands)
    index = next((i for i, c in enumerate(commands) if str(c).split()[:1] == ["seqclean"]), None)
    if index is None:
        raise ValueError("contract stack script has no seqclean command")
    return commands[:index] + script_lines(settings) + commands[index:]


def without_settings(commands) -> tuple[list[str], tuple | None]:
    """(commands without a contract triple right before ``seqclean``, (T', B) or None): for acceptance tools."""
    commands = list(commands)
    for index in range(len(commands) - 3):
        triple = commands[index:index + 3]
        if (all(p.fullmatch(str(c)) for p, c in zip(_SCRIPT_LINES, triple))
                and str(commands[index + 3]).split()[:1] == ["seqclean"]):
            return commands[:index] + commands[index + 3:], (int(triple[0].split()[1]), int(triple[2].split("=")[1]))
    return commands, None


# --------------------------------------------------------------------------- the checks after the stack

def log_offset(path: Path) -> int:
    """Where this attempt's section of an append-only Siril log starts."""
    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return 0


def log_lines(path: Path, offset: int = 0) -> dict:
    """{'memory': [(p, max MB), ...], 'threads': [T, ...]}: the matching lines from ``offset`` on (bytes, CRLF ok)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except FileNotFoundError:
        data = b""
    memory, workers = [], []
    for line in data.decode("utf-8", errors="replace").splitlines():
        found = MEMORY_LINE.search(line)
        if found:
            memory.append((int(found.group(1)), int(found.group(2))))
        found = THREADS_LINE.search(line)
        if found:
            workers.append(int(found.group(1)))
    return {"memory": memory, "threads": workers}


def verify_log(path: Path, offset: int, settings: dict) -> None:
    """Exactly one memory line equal to (p, B*1024) and one thread line equal to T' in this attempt; else RuntimeError."""
    found = log_lines(path, offset)
    expected_memory = (settings["frame_mb"], settings["memory_gib"] * 1024)
    if found["memory"] != [expected_memory] or found["threads"] != [settings["threads"]]:
        raise RuntimeError(
            f"Siril 堆栈设置核对失败（{VERSION}，{Path(path).name}）：预期 'Memory per image: {expected_memory[0]} MB. "
            f"Max memory: {expected_memory[1]} MB' 与 'allocating data for {settings['threads']} threads' 各一行，"
            f"本次日志为 {found['memory']} 与 {found['threads']}；请恢复作业重新导出并堆栈")


def verify_stack_count(master: Path, settings: dict) -> None:
    """The master's STACKCNT must equal n (L and strict L; R checks it itself)."""
    from astropy.io import fits
    value = fits.getheader(master).get("STACKCNT")
    if type(value) is not int or value != settings["frames"]:
        raise RuntimeError(f"Siril 堆栈设置核对失败（{VERSION}）：母版 STACKCNT={value!r}，"
                           f"选中 {settings['frames']} 帧；请恢复作业重新导出并堆栈")
