"""Siril 1.4.x command adapter, with isolated paths and retained logs."""
from contextlib import contextmanager
from pathlib import Path
import contextvars
import ctypes
import functools
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from .process_guard import guarded_process

ROOT = Path(__file__).resolve().parents[2]
# Opt-in resource pin for reproducibility tests and benchmarks only
# (docs/EFFICIENCY_P0_PREREG_20260928.md 1.3). Unset keeps today's scripts
# byte for byte; the product never sets it.
SIRIL_PIN_ENV = "AESTER_SIRIL_PIN"
_PIN_VALUE = re.compile(r"[1-9][0-9]{0,4}")


class Cancelled(RuntimeError):
    pass


DEFAULT_TOTAL_TIMEOUT = 24 * 60 * 60.0
DEFAULT_PROGRESS_TIMEOUT = 2 * 60 * 60.0
_UNSET = object()


class SirilTimeoutError(TimeoutError):
    """A deadline was reached; lack of log output is not proof of a hung engine."""

    def __init__(self, name: str, kind: str, limit: float, log_path: Path):
        self.stage = name
        self.kind = kind
        self.limit_seconds = limit
        self.log_path = log_path
        detail = "总运行" if kind == "total" else "无新增日志"
        super().__init__(f"Siril {name} 超过{detail}时限 {limit:g} 秒，已请求停止；日志：{log_path}。"
                         "静默计算也可能触发日志时限，可按素材调整时限后另行重试。")


def _timeout_setting(value, env_name: str, default: float) -> float | None:
    if value is _UNSET:
        value = os.environ.get(env_name, "").strip() or default
        if isinstance(value, str) and value.lower() in {"none", "off"}:
            return None
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{env_name} must be finite and positive, or off")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be finite and positive, or off") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{env_name} must be finite and positive, or off")
    return seconds


def siril_pin(value=_UNSET) -> dict | None:
    """Parse AESTER_SIRIL_PIN, e.g. 'cpu=16,mem=16'; either part is optional.

    cpu is Siril's thread count (``setcpu``, plus OMP_NUM_THREADS for the
    Siril process); mem fixes Siril's memory budget in GiB
    (``core.mem_mode=1`` with ``core.mem_amount``). None when unset or empty.
    A malformed value raises rather than silently running unpinned.
    """
    if value is _UNSET:
        value = os.environ.get(SIRIL_PIN_ENV, "")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{SIRIL_PIN_ENV} must be a string such as 'cpu=16,mem=16'")
    text = value.strip()
    if not text:
        return None
    pin = {}
    for part in text.split(","):
        key, separator, number = part.partition("=")
        key, number = key.strip().lower(), number.strip()
        if not separator or key not in ("cpu", "mem") or key in pin or not _PIN_VALUE.fullmatch(number):
            raise ValueError(f"{SIRIL_PIN_ENV} must look like 'cpu=16,mem=16' "
                             "(either part optional, positive integers, mem in GiB)")
        pin[key] = int(number)
    return pin


def pin_commands(pin: dict | None) -> list[str]:
    """Script lines for a parsed pin; empty for no pin."""
    commands = []
    if pin and "cpu" in pin:
        commands.append(f"setcpu {pin['cpu']}")
    if pin and "mem" in pin:
        commands += ["set core.mem_mode=1", f"set core.mem_amount={pin['mem']}"]
    return commands


def pin_environment(pin: dict | None) -> dict:
    """Extra environment for the pinned Siril process only."""
    return {"OMP_NUM_THREADS": str(pin["cpu"])} if pin and "cpu" in pin else {}


def pin_diagnostics(value=_UNSET) -> dict | None:
    """What a pin applies, for timing/diagnostic records; None when unpinned.

    Diagnostic only: it must never enter a signature or data digest.
    """
    pin = siril_pin(value)
    if pin is None:
        return None
    return {"source": SIRIL_PIN_ENV, "cpu": pin.get("cpu"), "mem_gb": pin.get("mem"),
            "script_commands": pin_commands(pin), "process_environment": pin_environment(pin)}


def script_text(commands, pin: dict | None = None) -> str:
    """The .ssf body. A pin goes right after the header lines ('requires' and,
    when first, 'offline'), so explicit later script settings still win."""
    commands = list(commands)
    if pin:
        head = 1 if commands and commands[0] == "offline" else 0
        commands[head:head] = pin_commands(pin)
    return "requires 1.4.4 1.5.0\n" + "\n".join(commands) + "\n"


# ---- Job-stage Siril runs yield the CPU to interactive work (efficiency P1a S-4c) ----------------------------
# BELOW_NORMAL kept a NORMAL UI thread's 2 ms work at p99 2 ms while Siril saturated its CPUs (6-21 ms at NORMAL)
# and cost nothing on an idle homogeneous machine on mains power. Siril does give up the whole share of any
# concurrent NORMAL load (machine throughput is unchanged). Hybrid (P/E-core) CPUs and battery power can lower a
# below-normal process's QoS (E-cores, efficient frequencies; learn.microsoft.com .../procthread/quality-of-service),
# which is unmeasured, so they keep NORMAL; so do interactive runs (the reference preview has a 5-minute
# deadline). The priority class changes scheduling only, never what Siril computes.
# Measured in the P1a acceptance (2026-09-29, 192 logical CPUs, Siril unpinned): the legacy 48-frame 02_stack took
# 91.5 / 50.1 s at BELOW_NORMAL against 38.2 / 40.5 s at NORMAL (interleaved), with 1.4-2.8x the CPU seconds: with
# one OpenMP thread per logical CPU, any preempted thread stalls the others at the next barrier.  Limited to 16
# logical CPUs there was no difference.  So 'auto' keeps NORMAL until the P2 governor caps Siril's threads
# (AUTO_BELOW_NORMAL); 'below_normal' still forces it for measurements.
AUTO_BELOW_NORMAL = False
SIRIL_PRIORITY_ENV = "AESTER_SIRIL_PRIORITY"   # auto (default) | normal | below_normal: for measurements only
_PRIORITY_FLAGS = {"normal": 0, "below_normal": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)}
_JOB_STAGE = contextvars.ContextVar("aester_siril_job_stage", default=False)


@contextmanager
def job_stage_launches():
    """Siril runs started inside this block are job stages (core._run_locked); all others are interactive."""
    token = _JOB_STAGE.set(True)
    try:
        yield
    finally:
        _JOB_STAGE.reset(token)


def siril_priority(job_stage: bool | None = None) -> str:
    """'below_normal' or 'normal' for the next Siril run. A malformed AESTER_SIRIL_PRIORITY raises."""
    setting = os.environ.get(SIRIL_PRIORITY_ENV, "").strip().lower() or "auto"
    if setting not in ("auto", "normal", "below_normal"):
        raise ValueError(f"{SIRIL_PRIORITY_ENV} must be auto, normal or below_normal")
    job_stage = _JOB_STAGE.get() if job_stage is None else job_stage
    if os.name != "nt" or not job_stage or setting == "normal":
        return "normal"
    if setting == "below_normal":
        return "below_normal"
    if not AUTO_BELOW_NORMAL:
        return "normal"
    return "below_normal" if _single_efficiency_class() and _on_mains_power() else "normal"


def _cpu_efficiency_classes(kernel32=None) -> frozenset | None:
    """EfficiencyClass values over all CPU sets (GetSystemCpuSetInformation, Windows 10 1607+); None if unknown."""
    try:
        query = (kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)).GetSystemCpuSetInformation
        query.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
                          ctypes.c_uint32]
        query.restype = ctypes.c_int32
        needed = ctypes.c_uint32(0)
        query(None, 0, ctypes.byref(needed), None, 0)      # fails with ERROR_INSUFFICIENT_BUFFER and sets needed
        if not needed.value:
            return None
        buffer = ctypes.create_string_buffer(needed.value)
        if not query(buffer, needed, ctypes.byref(needed), None, 0):
            return None
        raw = buffer.raw[:needed.value]
    except (OSError, AttributeError, ctypes.ArgumentError):
        return None
    classes, offset = set(), 0
    while offset + 8 <= len(raw):
        size, kind = struct.unpack_from("<II", raw, offset)  # SYSTEM_CPU_SET_INFORMATION.Size, .Type
        if size < 8 or offset + size > len(raw):
            return None
        if kind == 0 and size >= 19:                         # CpuSetInformation: EfficiencyClass at byte 18
            classes.add(raw[offset + 18])
        offset += size
    return frozenset(classes) or None


@functools.lru_cache(maxsize=1)
def _single_efficiency_class() -> bool:
    classes = _cpu_efficiency_classes()
    return classes is not None and len(classes) == 1


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


def _on_mains_power(kernel32=None) -> bool:
    """True on AC power or without a system battery; False on battery or when unknown (asked for every run)."""
    try:
        function = (kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)).GetSystemPowerStatus
        function.argtypes = [ctypes.POINTER(_SYSTEM_POWER_STATUS)]
        function.restype = ctypes.c_int32
        status = _SYSTEM_POWER_STATUS()
        if not function(ctypes.byref(status)):
            return False
    except (OSError, AttributeError, ctypes.ArgumentError):
        return False
    return status.ACLineStatus == 1 or (status.BatteryFlag != 255 and bool(status.BatteryFlag & 128))


# ---- No Python venv work in our Siril runs (efficiency P1a S-4b) --------------------------------------------
# Every Siril 1.4.4 launch starts a thread, before the script's first line, that checks, repairs or creates a Python
# venv under g_get_user_data_dir()\siril and pip-installs its sirilpy module (main-cli.c:212-213,
# siril_pythonmodule.c:2519-2610, 2772-2839); the 'offline' script command cannot stop it. On Windows, for the
# Siril process only:
#   XDG_DATA_HOME -> the run's own folder: GLib looks there instead of the user's %LOCALAPPDATA%\siril (shared with a
#                    desktop Siril, whose venv the finite build would otherwise delete) and finds no venv;
#   MSYSTEM       -> with no venv present, Siril stops at its msys2 check (Windows-only code) and starts no process.
# It then logs five msys2/Python lines, which launch_note explains. Elsewhere that check does not exist and a private
# XDG_DATA_HOME would make Siril build a venv inside the job folder, so nothing is overridden there.
_NO_PYTHON = {"MSYSTEM": "AESTER"}

# ---- No D-Bus session bus in our Siril runs (efficiency P1b S-4d, "SF-2") ------------------------------------
# On Windows every GLib program of the user's session shares one D-Bus session bus, found through session-wide named
# objects (GLib 2.90 gdbusprivate.c:2031-2034). The first program that needs it starts the daemon gdbus.exe from its
# own bin folder (:2316-2400) without job breakaway, so a stock-Siril stage puts it in the stage's KILL_ON_JOB_CLOSE
# Job Object (process_guard), and every stage end kills it. Every GApplication, even a non-unique one, connects
# (gapplicationimpl-dbus.c:647-651) with exit-on-close (gdbusconnection.c:7983, 929-934): a desktop Siril or GIMP, or
# another Aester stage, that joined the bus is ended with exit code 3 when the daemon dies (measured: 5/5 and 5/5;
# docs/EFFICIENCY_P1B_PREREG_20261001.md). "disabled:" is no known transport (gdbusaddress.c), so g_bus_get_sync fails
# and GApplication carries on as a plain non-unique application (gapplicationimpl-dbus.c:653-659): Siril then neither
# starts nor joins a bus. Siril 1.4.4 uses no D-Bus, GSettings or network monitor anywhere else, so nothing it
# computes changes. Windows only, like the reason; launch_note is unchanged (no new log lines).
_NO_SESSION_BUS = {"DBUS_SESSION_BUS_ADDRESS": "disabled:"}

# ---- A one-thread OpenMP/OpenBLAS pool in our Siril runs (efficiency P1b U-G3) --------------------------------
# Siril sets its own worker count for each parallel region (setcpu; default: the logical processors of its group), so
# OMP_NUM_THREADS does not change how many threads compute. It only sizes the pools that OpenMP and OpenBLAS create at
# start-up, each pool thread committing about 128 MiB: 1 instead of 64 here is 6.9-9.2 GB less commit per launch, the
# same working set and wall time, and identical outputs for all six stage types (P1b governor design F1). Every
# platform; an AESTER_SIRIL_PIN with cpu=N keeps OMP_NUM_THREADS=N (pin_environment wins).
# One exception to "same wall time" (P1b review SIRIL-1, measured 2026-10-02 through run_script, 8 rounds): LibRaw's
# Canon CR3 decoder runs its own OpenMP regions, which follow OMP_NUM_THREADS, so a CR3 convert takes about 1.23-1.26x
# as long (EOS R5 / R6 II single frame 1.22 -> 1.54 s / 0.68 -> 0.85 s; 8 R5 frames 3.60 -> 4.43 s); Z8 DNG is
# unchanged (1.88 -> 1.84 s) and every output is identical. OPENBLAS_NUM_THREADS=1 is no substitute: this OpenBLAS is
# an OpenMP build and the pool stays at 64 (15.06 GB commit for 8 Z8 frames, as with nothing set, against 6.58 GB).
# So runs that decode RAW files (convert / convertraw) get a pool of 8 (fewer on smaller machines): the CR3 decode is
# back to its old speed (R5 1.215 s at OMP_NUM_THREADS=8 against 1.220 s with nothing set) for about 0.94 GB more
# commit than a pool of 1, still about 7.6 GB below the old 64-thread pool; outputs identical in all three settings.
SIRIL_OMP_THREADS = "1"
SIRIL_DECODE_OMP_THREADS = 8
DECODE_COMMANDS = ("convert", "convertraw")


def siril_omp_threads(commands=None) -> str:
    """OMP_NUM_THREADS of a Siril run without a cpu pin: a small pool for RAW-decoding scripts, else one thread."""
    decodes = any((str(c).split() or [""])[0].lower() in DECODE_COMMANDS for c in (commands or ()))
    if not decodes:
        return SIRIL_OMP_THREADS
    return str(max(1, min(SIRIL_DECODE_OMP_THREADS, os.cpu_count() or 1)))


# ---- The stack-settings contract's OpenMP environment (efficiency P1c EFF-D4, stack_settings.py) ---------------------
# A contract stack (L, strict L and R jobs admitted under siril-stack-settings-1) sets its threads with setcpu, and its
# results must not depend on anything else: OMP_NUM_THREADS is 1 whatever a pin says (P1b measured identical masters at
# 1, unset and 16), and OpenMP variables that could make 'setcpu' end with fewer threads than asked (OMP_DYNAMIC,
# OMP_THREAD_LIMIT, OMP_MAX_ACTIVE_LEVELS) are removed from the Siril process's environment block.
STACK_SETTINGS_OMP_THREADS = "1"
STACK_SETTINGS_UNSET = ("OMP_DYNAMIC", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS")


def siril_environment(folder, pin: dict | None = None, base=None, *, windows: bool | None = None,
                      commands=None, stack_settings: bool = False) -> dict:
    """The complete environment block of one Siril run (this process's environment is never changed).
    ``commands``: the script's commands, which choose the OpenMP pool (siril_omp_threads).
    ``stack_settings``: a contract stack (EFF-D4): OMP_NUM_THREADS=1 and no STACK_SETTINGS_UNSET variable.
    ``windows`` defaults to this platform; tests pass it to cover both branches without touching os.name."""
    env = dict(os.environ if base is None else base)
    env["OMP_NUM_THREADS"] = siril_omp_threads(commands)
    env.update(pin_environment(pin))
    if stack_settings:
        names = {"OMP_NUM_THREADS", *STACK_SETTINGS_UNSET}
        env = {key: value for key, value in env.items() if key.upper() not in names}
        env["OMP_NUM_THREADS"] = STACK_SETTINGS_OMP_THREADS
    if (os.name == "nt") if windows is None else windows:
        env["XDG_DATA_HOME"] = os.path.abspath(folder)
        env.update(_NO_PYTHON)
        env.update(_NO_SESSION_BUS)
    return env


def launch_note(priority: str) -> str:
    """The log line written before each Windows Siril run; it travels with the msys2 lines into error tails."""
    return (f"aester: Siril launch priority={priority}; Siril's Python scripting is off for this run (Aester sets "
            "MSYSTEM and XDG_DATA_HOME for this process only), so msys2 and Python venv lines below are expected and "
            "unrelated to the result. 本次按设计关闭 Siril 的 Python 脚本环境，下面的 msys2／Python 虚拟环境提示属预期现象，"
            "与处理结果无关。\n")


@contextmanager
def _started(args, output, pin: dict | None, folder, priority: str = "normal", commands=None,
             stack_settings: bool = False):
    """guarded_process for one Siril run in ``folder``, with its own environment block (siril_environment).
    A pin writes its log note first; on Windows the launch note follows. ``priority`` is a siril_priority()
    value: the Siril process's priority class on Windows."""
    if pin:
        environment = dict(pin_environment(pin))
        if stack_settings and environment:
            environment["OMP_NUM_THREADS"] = STACK_SETTINGS_OMP_THREADS
        note = "; ".join(pin_commands(pin) + [f"{k}={v}" for k, v in environment.items()])
        output.write(f"aester: {SIRIL_PIN_ENV} applied: {note}\n".encode("utf-8"))
    if stack_settings:
        output.write((f"aester: stack settings contract: OMP_NUM_THREADS={STACK_SETTINGS_OMP_THREADS}, "
                      f"{', '.join(STACK_SETTINGS_UNSET)} unset; Siril configuration {Path(args[2]).name}\n")
                     .encode("utf-8"))
    if os.name == "nt":
        output.write(launch_note(priority).encode("utf-8"))
    with guarded_process(args, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | _PRIORITY_FLAGS[priority],
                         env=siril_environment(folder, pin, commands=commands, stack_settings=stack_settings)) as process:
        yield process


def conversion_failure_message(log_path: Path, fallback: str, *, start_offset: int = 0) -> str:
    """Explain a failed conversion from its own log, never from a file suffix.

    Siril can return zero even when LibRaw rejects every input. Callers must
    still validate conversion products. Logs are append-only across retries;
    the caller's pre-run byte offset excludes diagnostics from older attempts.
    Missing/unreadable logs must not replace the original conversion failure.
    """
    unsupported = False
    readable = True
    pattern = re.compile(r"error in libraw\s+unsupported file format or not raw file", re.IGNORECASE)
    try:
        with log_path.open("rb") as stream:
            stream.seek(start_offset)
            unsupported = any(pattern.search(line.decode("utf-8", errors="replace")) for line in stream)
    except OSError:
        readable = False
    detail = ""
    if unsupported:
        detail = ("当前 Siril／LibRaw 引擎无法解码此 RAW 文件的格式或压缩方式"
                  "（日志报告 Unsupported file format or not RAW file）。"
                  "请保留原片，尝试用兼容转换器导出无额外有损压缩的 DNG 副本后重试。")
    return (f"{detail}{fallback}；转换日志：{log_path}"
            + ("（暂时无法读取）" if not readable else ""))


# Windows MAX_PATH as both pinned Siril builds meet it (measured 2026-09-29, efficiency P1a S-8): a file whose
# full path is 259 UTF-16 code units long is written, 260 is not (finite and stock stack outputs, stock convert
# outputs; ASCII and CJK). Siril is not long-path aware, so the system's LongPathsEnabled setting does not lift it.
ENGINE_PATH_LIMIT = 259


def path_units(path) -> int:
    """Length of ``path`` in UTF-16 code units, the unit of the Windows path limit."""
    return len(os.fspath(path).encode("utf-16-le")) // 2


def engine_path_refusal(root, longest) -> str | None:
    """None when Siril can write ``longest``; otherwise the admission refusal, naming the folder the user chose."""
    length = path_units(longest)
    if length <= ENGINE_PATH_LIMIT:
        return None
    limit, excess = ENGINE_PATH_LIMIT, length - ENGINE_PATH_LIMIT
    return (f"缓存或结果文件夹路径过长：在 {root} 中，堆栈引擎需要写入长 {length} 个字符的文件路径，"
            f"比 Windows 路径上限（{limit} 个字符）多 {excess} 个字符。请换一个更短的缓存或结果文件夹后重新提交；"
            "本次提交尚未开始处理")


def doctor(siril_path: str | None = None) -> dict:
    candidates = [siril_path, os.environ.get("STARSTACK_SIRIL"),
                  ROOT / "tools" / "siril" / "bin" / "siril-cli.exe",
                  shutil.which("siril-cli"), Path("C:/Program Files/Siril/bin/siril-cli.exe")]
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if not p.is_file():
            if siril_path:
                break
            continue
        try:
            # The version probe exits during option parsing, before Siril looks at any data folder or bus
            # (main-cli.c:86-93), but it may be any 1.4.x it finds, a desktop Siril included: it gets the same
            # private environment block as a script run, in a throw-away folder (efficiency P1b W4).
            with tempfile.TemporaryDirectory(prefix="aester-siril-probe-", ignore_cleanup_errors=True) as probe:
                run = subprocess.run([str(p.resolve()), "-v"], stdin=subprocess.DEVNULL, capture_output=True,
                                     timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                     env=siril_environment(probe))
            output = (run.stdout + run.stderr).decode("utf-8", errors="replace")
            match = re.search(r"\b(1\.4\.\d+)\b", output)
            if run.returncode == 0 and match and int(match.group(1).split('.')[-1]) >= 4:
                return {"available": True, "executable": str(p.resolve()), "version": match.group(1),
                        "message": "Siril 已就绪"}
            return {"available": False, "executable": str(p), "version": None,
                    "message": "此原型要求 Siril 1.4.4 或更新的 1.4.x；无法确认版本：" + output[-400:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "message": str(exc), "executable": str(p), "version": None}
    return {"available": False, "executable": None, "version": None,
            "message": "尚未找到 Siril 1.4.x，请选择 siril-cli.exe"}


def run_script(executable: str, job_dir: Path, name: str, commands: list[str], cancel_check,
               *, total_timeout=_UNSET, progress_timeout=_UNSET, config=None, stack_settings: bool = False) -> None:
    """Run an isolated script with cancellation and conservative watchdogs.

    Optional timeouts are seconds; explicit None disables that watchdog.
    Otherwise the defaults are 24 hours total and 2 hours without changes to
    the log, overridable by STARSTACK_SIRIL_TOTAL_TIMEOUT_SECONDS and
    STARSTACK_SIRIL_PROGRESS_TIMEOUT_SECONDS (positive seconds, or 'off').
    Log activity is an observable proxy, not proof of useful progress. A
    timeout is always a failed operation and never enables unsafe recovery.
    The opt-in AESTER_SIRIL_PIN (see siril_pin) adds fixed thread/memory
    lines to the script and a note to the log; unset changes nothing.
    Runs inside job_stage_launches() may run below normal priority (see
    siril_priority); the script and its results are the same either way.
    On Windows, Siril gets its own environment block (siril_environment) that
    skips its Python venv work, and the log starts with launch_note.
    ``config``: Siril's configuration file (default ``job_dir/siril.ini``); Siril writes its settings back into it.
    ``stack_settings``: a stack under the EFF-D4 contract (its script carries the settings lines): the Siril
    process runs with OMP_NUM_THREADS=1 and without OMP_DYNAMIC / OMP_THREAD_LIMIT / OMP_MAX_ACTIVE_LEVELS.
    """
    total_timeout = _timeout_setting(total_timeout, "STARSTACK_SIRIL_TOTAL_TIMEOUT_SECONDS", DEFAULT_TOTAL_TIMEOUT)
    progress_timeout = _timeout_setting(progress_timeout, "STARSTACK_SIRIL_PROGRESS_TIMEOUT_SECONDS", DEFAULT_PROGRESS_TIMEOUT)
    pin = siril_pin()
    priority = siril_priority()   # like the pin: a malformed override fails before any file is written
    script = job_dir / f"{name}.ssf"
    script.write_text(script_text(commands, pin), encoding="utf-8")
    log = job_dir / f"{name}.log"
    # Siril gets its own config; no modification of the user's desktop settings.
    config = job_dir / "siril.ini" if config is None else Path(config)
    config.touch(exist_ok=True)  # 1.4.4 accepts an empty file, but rejects a missing -i path.
    args = [executable, "-i", str(config), "-d", str(job_dir), "-s", str(script)]
    started = time.monotonic()
    with log.open("ab", buffering=0) as output, _started(args, output, pin, job_dir, priority, commands,
                                                          stack_settings) as process:
        try:
            last_progress = time.monotonic()
            info = log.stat()
            signature = (info.st_size, info.st_mtime_ns)
            while process.poll() is None:
                cancel_check()
                now = time.monotonic()
                info = log.stat()
                current = (info.st_size, info.st_mtime_ns)
                if current != signature:
                    signature, last_progress = current, now
                if total_timeout is not None and now - started >= total_timeout:
                    raise SirilTimeoutError(name, "total", total_timeout, log)
                if progress_timeout is not None and now - last_progress >= progress_timeout:
                    raise SirilTimeoutError(name, "progress", progress_timeout, log)
                time.sleep(.2)
            cancel_check()
            if process.returncode != 0:
                tail = log.read_bytes()[-4000:].decode("utf-8", errors="replace")
                raise RuntimeError(f"Siril {name} 失败（{process.returncode}）\n{tail}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # The guard still closes its Job Object and performs
                        # its own bounded cleanup. Never add an infinite wait.
                        pass
