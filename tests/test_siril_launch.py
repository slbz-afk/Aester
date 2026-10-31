"""How Aester launches Siril (efficiency P1a S-4b and S-4c).

S-4c: job-stage Siril runs (core._run_locked's timed_script) are marked, and
may run below normal priority. By default ('auto') they keep NORMAL: the P1a
acceptance measured a slower unpinned legacy stack at BELOW_NORMAL on a
many-core machine (siril.AUTO_BELOW_NORMAL). With the guarded policy switched
on, only Windows machines whose CPUs report one efficiency class and that are
on mains power use it; interactive runs (the reference preview), hybrid CPUs,
battery power and other platforms keep NORMAL. AESTER_SIRIL_PRIORITY=below_normal
forces it for measurements.

S-4b: on Windows every Siril run gets its own environment block with
XDG_DATA_HOME = the run folder and MSYSTEM set, so Siril skips its Python venv
work and never uses the user's %LOCALAPPDATA%\\siril; the log starts with a
launch note that explains the msys2/Python lines that follow. Scripts are
unchanged. The real-Siril tests need the local runtimes.

Efficiency P1b: S-4d adds DBUS_SESSION_BUS_ADDRESS=disabled: (Windows), so Siril
neither starts nor joins the user's D-Bus session bus, whose daemon died with a
stage's Job Object and took every GLib program on it along (exit code 3); U-G3
adds OMP_NUM_THREADS=1 unless a pin sets cpu. The real tests plant a sentinel
"desktop Siril" folder (W2) and must leave it byte-identical, while a negative
control shows the old launch way really changes it; two bus tests (W3) run a
user's GLib program (gdbus monitor) next to a stage. The second one (V3) kills a
real session-bus daemon in the user's logon session, so it runs only with
AESTER_TEST_KILL_SESSION_BUS=1 (P1b review SIRIL-5).
"""
from contextlib import contextmanager
import ctypes
import hashlib
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import siril_guard
from starstack import core, process_guard, references, siril
from test_siril import ControlledProcess

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
windows = pytest.mark.skipif(os.name != "nt", reason="priority classes are Windows-only")


def recorder(monkeypatch):
    calls = []

    @contextmanager
    def launch(args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs, "job_stage": siril._JOB_STAGE.get()})
        yield ControlledProcess(finished=0)

    monkeypatch.setattr(siril, "guarded_process", launch)
    return calls


@pytest.fixture
def machine(monkeypatch):
    """One efficiency class on mains power unless a test says otherwise; no override, no pin."""
    state = SimpleNamespace(single=True, mains=True)
    monkeypatch.setattr(siril, "_single_efficiency_class", lambda: state.single)
    monkeypatch.setattr(siril, "_on_mains_power", lambda: state.mains)
    monkeypatch.delenv(siril.SIRIL_PRIORITY_ENV, raising=False)
    monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    return state


def run(folder, *, job_stage=False):
    if job_stage:
        with siril.job_stage_launches():
            return siril.run_script("siril-cli.exe", folder, "step", ["cd work"], lambda: None)
    return siril.run_script("siril-cli.exe", folder, "step", ["cd work"], lambda: None)


# --------------------------------------------------------------------------- priority policy

def launch_note(priority):
    return siril.launch_note(priority).encode("utf-8") if os.name == "nt" else b""


def test_interactive_run_keeps_normal_priority(tmp_path, monkeypatch, machine):
    calls = recorder(monkeypatch)
    before = dict(os.environ)
    run(tmp_path)
    (call,) = calls
    assert call["kwargs"]["creationflags"] == NO_WINDOW and not call["job_stage"]
    assert set(call["kwargs"]) == {"stdin", "stdout", "stderr", "creationflags", "env"}
    assert (tmp_path / "step.ssf").read_text(encoding="utf-8") == "requires 1.4.4 1.5.0\ncd work\n"
    assert (tmp_path / "step.log").read_bytes() == launch_note("normal") and dict(os.environ) == before


@windows
@pytest.mark.parametrize("single, mains, expected", [
    (True, True, NO_WINDOW | BELOW_NORMAL),
    (False, True, NO_WINDOW),        # hybrid (P/E-core) CPU
    (True, False, NO_WINDOW),        # on battery, or power state unknown
    (False, False, NO_WINDOW),
])
def test_job_stage_runs_below_normal_only_on_one_efficiency_class_on_mains(tmp_path, monkeypatch, machine,
                                                                            single, mains, expected):
    monkeypatch.setattr(siril, "AUTO_BELOW_NORMAL", True)      # the guarded policy, switched off by default
    machine.single, machine.mains = single, mains
    calls = recorder(monkeypatch)
    run(tmp_path, job_stage=True)
    (call,) = calls
    assert call["job_stage"] and call["kwargs"]["creationflags"] == expected
    assert (tmp_path / "step.ssf").read_text(encoding="utf-8") == "requires 1.4.4 1.5.0\ncd work\n"
    assert (tmp_path / "step.log").read_bytes() == launch_note("below_normal" if expected & BELOW_NORMAL else "normal")


@pytest.mark.parametrize("single, mains", [(True, True), (False, True), (True, False)])
def test_auto_keeps_normal_by_default(tmp_path, monkeypatch, machine, single, mains):
    # P1a acceptance: BELOW_NORMAL slowed the unpinned legacy stack on a many-core machine, so 'auto' is NORMAL.
    assert siril.AUTO_BELOW_NORMAL is False
    machine.single, machine.mains = single, mains
    monkeypatch.delenv(siril.SIRIL_PRIORITY_ENV, raising=False)
    assert siril.siril_priority(job_stage=True) == "normal"
    calls = recorder(monkeypatch)
    run(tmp_path, job_stage=True)
    assert calls[0]["kwargs"]["creationflags"] == NO_WINDOW


@pytest.mark.skipif(os.name == "nt", reason="the non-Windows branch")
def test_other_platforms_always_keep_normal(tmp_path, monkeypatch, machine):
    monkeypatch.setenv(siril.SIRIL_PRIORITY_ENV, "below_normal")
    assert siril.siril_priority(job_stage=True) == "normal"
    calls = recorder(monkeypatch)
    run(tmp_path, job_stage=True)
    assert calls[0]["kwargs"]["creationflags"] == NO_WINDOW


@windows
@pytest.mark.parametrize("setting, job_stage, expected", [
    ("normal", True, "normal"),
    (" Normal ", True, "normal"),
    ("below_normal", True, "below_normal"),      # forced, even on a hybrid CPU on battery
    ("BELOW_NORMAL", True, "below_normal"),
    ("below_normal", False, "normal"),           # never for interactive runs
    ("auto", True, "normal"),                    # auto on a hybrid CPU on battery
    ("", True, "normal"),
])
def test_override_applies_to_job_stages_only(tmp_path, monkeypatch, machine, setting, job_stage, expected):
    machine.single = machine.mains = False
    monkeypatch.setenv(siril.SIRIL_PRIORITY_ENV, setting)
    assert siril.siril_priority(job_stage=job_stage) == expected
    calls = recorder(monkeypatch)
    run(tmp_path, job_stage=job_stage)
    assert calls[0]["kwargs"]["creationflags"] == NO_WINDOW | (BELOW_NORMAL if expected == "below_normal" else 0)


@pytest.mark.parametrize("job_stage", [False, True])
@pytest.mark.parametrize("setting", ["low", "idle", "below-normal", "0", "normal,auto"])
def test_malformed_override_fails_before_any_file_is_written(tmp_path, monkeypatch, machine, setting, job_stage):
    monkeypatch.setenv(siril.SIRIL_PRIORITY_ENV, setting)
    calls = recorder(monkeypatch)
    with pytest.raises(ValueError, match=siril.SIRIL_PRIORITY_ENV):
        run(tmp_path, job_stage=job_stage)
    assert not calls and list(tmp_path.iterdir()) == []


def test_job_stage_marker_resets_after_the_block_and_after_an_exception():
    assert siril._JOB_STAGE.get() is False
    with siril.job_stage_launches():
        assert siril._JOB_STAGE.get() is True
        with siril.job_stage_launches():
            assert siril._JOB_STAGE.get() is True
        assert siril._JOB_STAGE.get() is True
    assert siril._JOB_STAGE.get() is False
    with pytest.raises(RuntimeError, match="inside"):
        with siril.job_stage_launches():
            raise RuntimeError("inside")
    assert siril._JOB_STAGE.get() is False
    assert siril.siril_priority() == "normal"


# --------------------------------------------------------------------------- S-4b: Siril's environment

def test_each_run_gets_its_own_block_without_changing_this_process(tmp_path, monkeypatch, machine):
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "autolaunch:")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    monkeypatch.setenv("AESTER_S4B_INHERITED", "kept")
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=5")
    calls = recorder(monkeypatch)
    before = dict(os.environ)
    run(tmp_path, job_stage=True)
    env = calls[0]["kwargs"]["env"]
    assert dict(os.environ) == before and env is not os.environ
    assert env["AESTER_S4B_INHERITED"] == "kept" and env["OMP_NUM_THREADS"] == "5"
    if os.name == "nt":
        assert env["XDG_DATA_HOME"] == os.path.abspath(tmp_path) and env["MSYSTEM"] == "AESTER"
        assert env["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"
    else:
        assert env["XDG_DATA_HOME"] == str(tmp_path / "user-data") and env["MSYSTEM"] == "MINGW64"
        assert env["DBUS_SESSION_BUS_ADDRESS"] == "autolaunch:"
    lines = (tmp_path / "step.log").read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[0] == "aester: AESTER_SIRIL_PIN applied: setcpu 5; OMP_NUM_THREADS=5\n"
    assert "".join(lines[1:]).encode("utf-8") == launch_note("normal")      # auto keeps NORMAL (AUTO_BELOW_NORMAL)
    assert (tmp_path / "step.ssf").read_text(encoding="utf-8") == "requires 1.4.4 1.5.0\nsetcpu 5\ncd work\n"


@pytest.mark.parametrize("pin, omp", [(None, "1"), ("", "1"), ("mem=4", "1"), ("cpu=7", "7"), ("mem=4,cpu=16", "16")])
def test_siril_gets_a_one_thread_openmp_pool_unless_a_pin_sets_cpu(tmp_path, monkeypatch, machine, pin, omp):
    """U-G3: Siril's own thread count is unchanged (setcpu / its processors); OMP_NUM_THREADS only sizes the
    OpenMP/OpenBLAS pools created at start-up. The log and the script are as before."""
    monkeypatch.setenv("OMP_NUM_THREADS", "12")                        # inherited: replaced for Siril only
    if pin is None:
        monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    else:
        monkeypatch.setenv(siril.SIRIL_PIN_ENV, pin)
    calls = recorder(monkeypatch)
    run(tmp_path)
    assert calls[0]["kwargs"]["env"]["OMP_NUM_THREADS"] == omp and os.environ["OMP_NUM_THREADS"] == "12"
    log = (tmp_path / "step.log").read_text(encoding="utf-8")
    assert ("OMP_NUM_THREADS" in log) is (omp != "1")                   # only a pin writes a note
    assert siril.pin_environment(siril.siril_pin(pin)) == ({} if omp == "1" else {"OMP_NUM_THREADS": omp})


@pytest.mark.parametrize("windows", [True, False])
def test_siril_environment_on_both_platform_branches(tmp_path, windows):
    base = {"PATH": "C:\\x", "XDG_DATA_HOME": "C:\\user", "OMP_NUM_THREADS": "3",
            "DBUS_SESSION_BUS_ADDRESS": "autolaunch:"}
    original = dict(base)
    env = siril.siril_environment(tmp_path, None, base, windows=windows)
    assert base == original                                               # the base is copied
    if windows:
        assert env == {"PATH": "C:\\x", "XDG_DATA_HOME": os.path.abspath(tmp_path), "OMP_NUM_THREADS": "1",
                       "MSYSTEM": "AESTER", "DBUS_SESSION_BUS_ADDRESS": "disabled:"}
    else:          # the msys2 early return and the autolaunched bus are Windows-only: only the OpenMP pool
        assert env == {**base, "OMP_NUM_THREADS": "1"}
    pinned = siril.siril_environment(tmp_path, {"cpu": 9}, {}, windows=windows)
    assert pinned["OMP_NUM_THREADS"] == "9" and ("MSYSTEM" in pinned) is windows
    assert ("DBUS_SESSION_BUS_ADDRESS" in pinned) is windows
    assert siril.siril_environment(tmp_path, {"mem": 2}, {}, windows=False) == {"OMP_NUM_THREADS": "1"}
    assert siril.siril_environment(tmp_path, None, {}, windows=True) == {
        "OMP_NUM_THREADS": "1", "XDG_DATA_HOME": os.path.abspath(tmp_path), "MSYSTEM": "AESTER",
        "DBUS_SESSION_BUS_ADDRESS": "disabled:"}


def test_launch_note_is_one_bilingual_line_without_misleading_words():
    for priority in ("normal", "below_normal"):
        note = siril.launch_note(priority)
        assert note.startswith(f"aester: Siril launch priority={priority}; ") and note.endswith("\n")
        assert note.count("\n") == 1 and "msys2" in note and "与处理结果无关" in note
        # conversion_failure_message and the conversion-error tests look for these in a failure's text
        assert not any(word in note for word in ("HE", "DNG", "无法解码", "libraw", "LibRaw"))


# --------------------------------------------------------------------------- who is a job stage

@pytest.fixture
def sources(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    for index in range(3):
        (source / f"frame{index}.nef").write_bytes(b"controlled placeholder, never decoded")
    monkeypatch.setattr(core, "doctor", lambda *_: {
        "available": True, "executable": "controlled-not-executed.exe", "version": "1.4.4"})
    monkeypatch.setattr(core, "_spawn_worker", lambda *_: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(references, "_optional_raw_metadata", lambda *_: {"available": False})
    # The real adapter: tests/test_result_consistency.py::test_fallback_exclusion_survives_register_and_stale_exports
    # leaves its fixture's fake core.run_script behind (the autouse monkeypatch is undone after its fixture).
    monkeypatch.setattr(core, "run_script", siril.run_script)
    return SimpleNamespace(source=source, output=tmp_path / "out")


def test_core_job_stages_are_marked_and_the_reference_preview_is_not(sources, monkeypatch, machine):
    monkeypatch.setenv(siril.SIRIL_PRIORITY_ENV, "below_normal")      # makes the job-stage marking visible
    calls = recorder(monkeypatch)
    job = core.submit_job(str(sources.source), str(sources.output))
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed"   # the controlled engine converts nothing
    assert calls and all(call["job_stage"] for call in calls), result["error"]
    assert calls[0]["args"][-1].endswith("01_convert.ssf")
    assert {call["kwargs"]["creationflags"] for call in calls} == {
        NO_WINDOW | (BELOW_NORMAL if os.name == "nt" else 0)}
    calls.clear()
    with pytest.raises(RuntimeError, match="参考图准备失败"):
        references.prepare_reference(str(sources.source), str(sources.output))
    assert calls and not any(call["job_stage"] for call in calls)
    assert {call["kwargs"]["creationflags"] for call in calls} == {NO_WINDOW}
    assert siril._JOB_STAGE.get() is False


def test_timed_script_marks_every_run_script_call(sources, monkeypatch, machine):
    monkeypatch.setenv(siril.SIRIL_PRIORITY_ENV, "below_normal")
    seen = []

    def fake_run_script(executable, folder, name, commands, cancel_check):
        seen.append((name, siril._JOB_STAGE.get(), siril.siril_priority()))
        raise RuntimeError("controlled stop")

    monkeypatch.setattr(core, "run_script", fake_run_script)
    job = core.submit_job(str(sources.source), str(sources.output))
    core.run_job(job["job_dir"])
    assert seen == [("01_convert", True, "below_normal" if os.name == "nt" else "normal")]
    assert siril._JOB_STAGE.get() is False


def test_every_job_stage_siril_launch_goes_through_timed_script():
    # The runtime tests above only reach 01_convert.  Pin the structure that marks every later launch too: the job
    # worker (_run_locked) starts Siril only through timed_script (job_stage_launches) or hands timed_script on as the
    # runner, and the stack helpers launch only through the runner they are given.
    import ast
    import inspect
    from starstack import finite_stack, stack_range

    def calls(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call)]

    def name(call):
        f = call.func
        return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None

    def function(module, fname):
        tree = ast.parse(inspect.getsource(module))
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fname)

    run_locked = function(core, "_run_locked")
    timed = next(n for n in ast.walk(run_locked) if isinstance(n, ast.FunctionDef) and n.name == "timed_script")
    inside_timed = {id(c) for c in calls(timed)}
    direct = [c for c in calls(run_locked) if name(c) == "run_script" and id(c) not in inside_timed]
    assert direct == []
    launches = [c for c in calls(run_locked) if name(c) == "timed_script"]
    assert {c.args[2].value for c in launches if isinstance(c.args[2], ast.Constant)} >= {
        "01_convert", "02_probe", "02_export", "02_stack", "02_register_fallback"}
    handed = [c for c in calls(run_locked)
              if name(c) in ("_supported_stack", "run_protected_stack", "convert_cfa", "_contract_stack")]
    assert len(handed) >= 6

    def handed_on(c):
        # a Name argument, or one wrapped by core._range_runner (EFF-D4: R's trial runner with the contract lines)
        args = list(c.args) + [k.value for k in c.keywords]
        wrapped = [a for a in args if isinstance(a, ast.Call) and name(a) == "_range_runner"]
        return [a.id for a in args + [w.args[0] for w in wrapped] if isinstance(a, ast.Name)]
    for c in handed:
        assert "timed_script" in handed_on(c), ast.unparse(c)
    # P1b-3: L's three paths (3a, 3b, 3c) launch through export_then_stack, a nested helper of _run_locked whose
    # launches are timed_script calls with literal names (EFF-D4: or _contract_stack, handed timed_script).
    helper = next(n for n in ast.walk(run_locked) if isinstance(n, ast.FunctionDef) and n.name == "export_then_stack")
    assert {c.args[2].value for c in calls(helper) if name(c) == "timed_script"} == {"02_export", "02_stack"}
    assert [ast.unparse(c.args[0]) for c in calls(helper) if name(c) == "_contract_stack"] == ["timed_script"]
    assert len([c for c in calls(run_locked) if name(c) == "export_then_stack"]) == 3
    for fname in ("_supported_stack", "_contract_stack", "_range_runner"):
        body = function(core, fname)
        assert [ast.unparse(c) for c in calls(body) if name(c) == "run_script"] == [], fname
    assert any(name(c) == "runner" for c in calls(function(core, "_contract_stack")))
    assert any(name(c) == "runner" for c in calls(function(core, "_range_runner")))
    for module, fname in ((finite_stack, "run"), (stack_range, "run_protected_stack")):
        body = function(module, fname)
        assert not [c for c in calls(body) if name(c) == "run_script"], fname
        assert any(name(c) == "runner" for c in calls(body)), fname


# --------------------------------------------------------------------------- platform state

def cpu_set(identifier, efficiency_class, *, size=32, kind=0):
    """One SYSTEM_CPU_SET_INFORMATION record (Size, Type, Id, Group, LP, Core, LLC, NUMA, EfficiencyClass, ...)."""
    head = struct.pack("<IIIHBBBBB", size, kind, 256 + identifier, 0, identifier, identifier, 0, 0,
                       efficiency_class)
    return (head + bytes(max(0, size - len(head))))[:max(size, 8)]


class FakeKernel32:
    """GetSystemCpuSetInformation / GetSystemPowerStatus doubles (plain functions, so argtypes can be set)."""

    def __init__(self, data=b"", *, fail=False, power=None):
        self.data, self.fail, self.calls = data, fail, []

        def cpu_sets(buffer, length, returned, process, flags):
            self.calls.append("GetSystemCpuSetInformation")
            returned._obj.value = len(self.data)
            if self.fail or buffer is None or int(getattr(length, "value", length)) < len(self.data):
                return 0
            ctypes.memmove(buffer, self.data, len(self.data))
            return 1

        def power_status(status):
            self.calls.append("GetSystemPowerStatus")
            if power is None:
                return 0
            status._obj.ACLineStatus, status._obj.BatteryFlag = power
            return 1

        self.GetSystemCpuSetInformation = cpu_sets
        self.GetSystemPowerStatus = power_status


@pytest.mark.parametrize("records, expected", [
    ([cpu_set(i, 0) for i in range(8)], frozenset({0})),
    ([cpu_set(i, 1 if i < 4 else 0) for i in range(12)], frozenset({0, 1})),      # hybrid
    ([cpu_set(0, 2), cpu_set(1, 2, kind=7), cpu_set(2, 2, size=40)], frozenset({2})),   # other types skipped
    ([cpu_set(0, 0), cpu_set(1, 0, size=4)], None),                                     # malformed Size
    ([cpu_set(0, 0), struct.pack("<II", 64, 0)], None),                                 # runs past the end
    ([], None),
])
def test_efficiency_classes_from_cpu_set_records(records, expected):
    fake = FakeKernel32(b"".join(records))
    assert siril._cpu_efficiency_classes(fake) == expected


def test_efficiency_classes_unknown_when_the_query_fails():
    assert siril._cpu_efficiency_classes(FakeKernel32(cpu_set(0, 0), fail=True)) is None
    assert siril._cpu_efficiency_classes(SimpleNamespace()) is None          # no such entry point


@pytest.mark.parametrize("classes, single", [(frozenset({0}), True), (frozenset({0, 1}), False), (None, False)])
def test_single_efficiency_class(monkeypatch, classes, single):
    siril._single_efficiency_class.cache_clear()
    monkeypatch.setattr(siril, "_cpu_efficiency_classes", lambda: classes)
    try:
        assert siril._single_efficiency_class() is single
    finally:
        siril._single_efficiency_class.cache_clear()


@pytest.mark.parametrize("power, mains", [
    ((1, 8), True),          # on AC, charging
    ((1, 255), True),
    ((0, 8), False),         # on battery
    ((0, 1), False),
    ((255, 128), True),      # AC unknown, no system battery (a desktop)
    ((255, 255), False),     # both unknown
    ((255, 8), False),
    (None, False),           # the call failed
])
def test_mains_power(power, mains):
    assert siril._on_mains_power(FakeKernel32(power=power)) is mains
    assert siril._on_mains_power(SimpleNamespace()) is False


@windows
def test_this_machine_reports_its_platform_state():
    classes = siril._cpu_efficiency_classes()
    assert isinstance(classes, frozenset) and classes and all(isinstance(value, int) for value in classes)
    assert siril._on_mains_power() in (True, False)


# --------------------------------------------------------------------------- a real Siril

STOCK_SIRIL = Path(__file__).resolve().parents[1] / "tools" / "siril" / "bin" / "siril-cli.exe"


FINITE_SIRIL = Path(__file__).resolve().parents[1] / "tools" / "finite-siril" / "finite-validity-v1" / "siril-cli.exe"


def observe(monkeypatch):
    """Keep every guarded Siril process so its Job Object accounting can be read after it ended."""
    processes = []
    real = siril.guarded_process

    @contextmanager
    def observing(args, **kwargs):
        with real(args, **kwargs) as process:
            processes.append(process)
            yield process

    monkeypatch.setattr(siril, "guarded_process", observing)
    return processes


def tree_listing(root: Path) -> dict:
    """Relative path -> ("dir",) or (size, sha256) for everything below root."""
    listing = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        listing[relative] = ("dir",) if path.is_dir() else (path.stat().st_size,
                                                            hashlib.sha256(path.read_bytes()).hexdigest())
    return listing


def plant_sentinel(desktop: Path) -> dict:
    """A desktop Siril's data folder as Siril 1.4.4 leaves it (<data dir>\\siril): a Python venv with a package the
    user installed, and the unpacked sirilpy module. venv\\Scripts\\python.exe is a 0-byte stand-in, which every
    Siril's venv health check rejects whether or not it can start processes, so a launch that reaches this folder
    deletes the venv (siril_pythonmodule.c:2519-2555). Returns the listing to compare with."""
    venv = desktop / "siril" / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_bytes(b"")
    (venv / "pyvenv.cfg").write_text("home = C:\\nowhere\n", encoding="utf-8")
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    (venv / "Lib" / "site-packages" / "user_module.txt").write_text("installed by a desktop script\n",
                                                                    encoding="utf-8")
    (desktop / "siril" / ".python_module").mkdir()
    (desktop / "siril" / ".python_module" / "METADATA").write_text("Version: 1.4.4\n", encoding="utf-8")
    return tree_listing(desktop)


def safety_net(tmp_path, monkeypatch):
    """This process's XDG_DATA_HOME points to a sentinel "desktop Siril" folder, and MSYSTEM and the bus address are
    unset (a GUI started from Explorer): if the per-run block ever failed to reach Siril, its venv work would change
    the sentinel (and be seen), never the user's %LOCALAPPDATA%; the session guard would also stop the launch."""
    desktop = tmp_path / "safety-net"
    before = plant_sentinel(desktop)
    monkeypatch.setenv("XDG_DATA_HOME", str(desktop))
    monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    return SimpleNamespace(path=desktop, before=before)


def assert_quiet_launch(process, job, net, priority):
    """Siril and its conhost only (no gdbus.exe since S-4d, no python.exe or pip), the sentinel byte-identical
    (no .python_module.tmp.* or *.error-* folder either), nothing of Siril's own in the run folder, and the log
    starts with the launch note followed by Siril's msys2/Python lines."""
    assert process.accounting and process.accounting["descendant_processes"] <= 2, process.accounting
    assert tree_listing(net.path) == net.before
    assert not (job / "siril").exists()
    log = next(job.glob("*.log")).read_text(encoding="utf-8")
    note = siril.launch_note(priority)
    # Siril localizes its lines (the stock build logs "错误：检测到 msys2 环境…" on a Chinese system)
    assert log.startswith(note) and "msys2" in log[len(note):]


@pytest.mark.skipif(os.name != "nt" or not STOCK_SIRIL.is_file(), reason="needs the local Siril runtime")
def test_real_stock_siril_skips_python_and_writes_the_same_frames(tmp_path, monkeypatch, machine):
    from astropy.io import fits
    import numpy as np
    net = safety_net(tmp_path, monkeypatch)
    rng = np.random.default_rng(5)
    frames = [(.1 + rng.random((3, 48, 64)) * .5).astype(np.float32) for _ in range(2)]
    commands = ["setext fit", "set32bits", "setcompress 0", "cd stage", "convert light -out=../work"]
    written = {}
    for label in ("reference", "product"):
        job = tmp_path / label
        (job / "stage").mkdir(parents=True)
        (job / "work").mkdir()
        for number, data in enumerate(frames, 1):
            fits.PrimaryHDU(data).writeto(job / "stage" / f"light{number:05d}.fit")
        if label == "reference":
            # A run as before S-4b and U-G3: inherited environment, no OMP_NUM_THREADS (a full OpenMP pool), the
            # venv skipped; the bus line only keeps the reference off the user's session bus (S-4d, guard R2).
            (job / "siril.ini").touch()
            (job / "01_convert.ssf").write_text(siril.script_text(commands), encoding="utf-8")
            reference = {k: v for k, v in os.environ.items() if k.upper() != "OMP_NUM_THREADS"}
            reference.update(MSYSTEM="MINGW64", XDG_DATA_HOME=str(tmp_path / "xdg"),
                             DBUS_SESSION_BUS_ADDRESS="disabled:")
            with (job / "01_convert.log").open("ab") as log, siril.guarded_process(
                    [str(STOCK_SIRIL), "-i", str(job / "siril.ini"), "-d", str(job), "-s", str(job / "01_convert.ssf")],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
                    env=reference) as process:
                assert process.wait(timeout=120) == 0
        else:
            processes = observe(monkeypatch)
            with siril.job_stage_launches():
                siril.run_script(str(STOCK_SIRIL), job, "01_convert", commands, lambda: None)
            assert_quiet_launch(processes[0], job, net, "normal")
        written[label] = [fits.getdata(job / "work" / f"light_{number:05d}.fit") for number in (1, 2)]
    for reference, product in zip(written["reference"], written["product"]):
        assert reference.dtype == product.dtype and reference.tobytes() == product.tobytes()
    assert sorted(p.name for p in (tmp_path / "product").iterdir()) == ["01_convert.log", "01_convert.ssf",
                                                                        "siril.ini", "stage", "work"]


@pytest.mark.skipif(os.name != "nt" or not FINITE_SIRIL.is_file(), reason="needs the local finite runtime")
def test_real_finite_siril_skips_python_too(tmp_path, monkeypatch, machine):
    from astropy.io import fits
    import numpy as np
    from starstack import finite_stack as fs
    net = safety_net(tmp_path, monkeypatch)
    attempt = tmp_path / "finite-000000000000"
    attempt.mkdir()
    rng = np.random.default_rng(6)
    for number in (1, 2, 3):
        data = (.2 + rng.normal(0, .01, (3, 64, 96))).astype(np.float32)
        data[:, 0, :2] = np.nan
        header = fits.Header()
        header["ROWORDER"] = "BOTTOM-UP"
        header["DATAMIN"], header["DATAMAX"] = float(np.nanmin(data)), float(np.nanmax(data))
        fits.PrimaryHDU(data, header).writeto(attempt / f"q_{number:05d}.fit")
    processes = observe(monkeypatch)
    with siril.job_stage_launches():
        siril.run_script(str(FINITE_SIRIL), attempt, "stack",
                         ["offline", "setext fit", "set32bits", "setcompress 0", "set gui.icc_autoassignment=0",
                          "set gui.icc_autoconversion=0", "setref q 1", fs.STACK], lambda: None)
    assert_quiet_launch(processes[0], attempt, net, "normal")
    assert all((attempt / name).is_file() for name in ("finite_master.fit", "finite_master_nused.fit",
                                                        "finite_master_nvalid.fit"))


@pytest.mark.skipif(os.name != "nt" or not STOCK_SIRIL.is_file(), reason="needs the local Siril runtime")
def test_real_siril_runs_below_normal_in_a_job_stage_and_writes_the_same_frames(tmp_path, monkeypatch):
    from astropy.io import fits
    import numpy as np
    # MSYSTEM and a private XDG_DATA_HOME keep Siril's Python venv work away from the user's %LOCALAPPDATA%\siril.
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(siril.SIRIL_PRIORITY_ENV, "below_normal")      # whatever this machine's CPUs and power
    monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint32
    classes = []
    real = siril.guarded_process

    @contextmanager
    def observing(args, **kwargs):
        with real(args, **kwargs) as process:
            classes.append(kernel32.GetPriorityClass(process._hprocess))
            yield process

    monkeypatch.setattr(siril, "guarded_process", observing)
    rng = np.random.default_rng(4)
    frames = [(.1 + rng.random((3, 48, 64)) * .5).astype(np.float32) for _ in range(2)]
    written = {}
    for label, job_stage in (("interactive", False), ("job", True)):
        job = tmp_path / label
        (job / "stage").mkdir(parents=True)
        (job / "work").mkdir()
        for number, data in enumerate(frames, 1):
            fits.PrimaryHDU(data).writeto(job / "stage" / f"light{number:05d}.fit")
        commands = ["setext fit", "set32bits", "setcompress 0", "cd stage", "convert light -out=../work"]
        if job_stage:
            with siril.job_stage_launches():
                siril.run_script(str(STOCK_SIRIL), job, "01_convert", commands, lambda: None)
        else:
            siril.run_script(str(STOCK_SIRIL), job, "01_convert", commands, lambda: None)
        written[label] = [fits.getdata(job / "work" / f"light_{number:05d}.fit") for number in (1, 2)]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    own = kernel32.GetPriorityClass(kernel32.GetCurrentProcess())
    # An interactive launch passes no class flag, as 63a49d8 did: CreateProcess then gives NORMAL, or the caller's class
    # when that is IDLE or BELOW_NORMAL.  The job-stage launch asks for BELOW_NORMAL_PRIORITY_CLASS explicitly.
    assert classes == [own if own in (0x40, 0x4000) else 0x20, 0x4000]
    for interactive, job in zip(written["interactive"], written["job"]):
        assert interactive.dtype == job.dtype and interactive.tobytes() == job.tobytes()


# --------------------------------------------------------------------------- W2: the negative control, the probes

def _finite_inputs(folder, count=3, shape=(3, 64, 96), seed=6):
    """Inputs as finite_stack._write_input leaves them (BOTTOM-UP, DATAMIN/DATAMAX, NaN = missing coverage)."""
    from astropy.io import fits
    import numpy as np
    rng = np.random.default_rng(seed)
    for number in range(1, count + 1):
        data = (.2 + rng.normal(0, .01, shape)).astype(np.float32)
        data[:, 0, :2] = np.nan
        header = fits.Header()
        header["ROWORDER"] = "BOTTOM-UP"
        header["DATAMIN"], header["DATAMAX"] = float(np.nanmin(data)), float(np.nanmax(data))
        fits.PrimaryHDU(data, header).writeto(folder / f"q_{number:05d}.fit")


def _finite_commands(cpu=None):
    from starstack import finite_stack as fs
    return (["offline"] + ([f"setcpu {cpu}"] if cpu else []) +
            ["setext fit", "set32bits", "setcompress 0", "set gui.icc_autoassignment=0",
             "set gui.icc_autoconversion=0", "setref q 1", fs.STACK])


def old_launch_environment(desktop: Path) -> dict:
    """How Aester started Siril before S-4b: the inherited environment of a GUI opened from Explorer (no MSYSTEM),
    whose XDG_DATA_HOME is unset, so GLib uses %LOCALAPPDATA% -- here the sentinel desktop folder instead. Two
    changes keep the control deterministic and harmless: the bus is disabled (S-4d; no sibling Siril may be hit),
    and PATH holds only the Windows folders, so once the venv is deleted no Siril build can find a Python to build a
    new one (or pip-install from the network), even when it can start processes (W7 gives the finite runtime the
    GLib spawn helpers)."""
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    drop = {"MSYSTEM", "XDG_DATA_HOME", "DBUS_SESSION_BUS_ADDRESS", "PATH", "OMP_NUM_THREADS", siril.SIRIL_PIN_ENV}
    env = {k: v for k, v in os.environ.items() if k.upper() not in drop}
    env.update(XDG_DATA_HOME=str(desktop), DBUS_SESSION_BUS_ADDRESS="disabled:",
               PATH=os.pathsep.join([str(system / "System32"), str(system)]))
    return env


@pytest.mark.skipif(os.name != "nt" or not FINITE_SIRIL.is_file(), reason="needs the local finite runtime")
def test_negative_control_the_old_launch_destroys_the_desktop_venv(tmp_path, monkeypatch, machine):
    """Launched the old way, the same Siril reaches the desktop folder and deletes its venv, so the sentinel checks
    above would see a regression. The sentinel's python.exe is rejected by any build's health check, so this holds
    with or without the GLib spawn helpers; the healthy-venv case (deleted only while the runtime cannot start
    processes) is W7's own test."""
    desktop = tmp_path / "desktop"
    before = plant_sentinel(desktop)
    env = old_launch_environment(desktop)
    assert Path(env["XDG_DATA_HOME"]).resolve().is_relative_to(tmp_path.resolve())      # never the user's folder
    assert os.path.normcase(os.path.abspath(env["XDG_DATA_HOME"])) != siril_guard.REAL_DATA
    job = tmp_path / "finite-000000000000"
    job.mkdir()
    # Siril's venv check runs in a thread next to the script; a 3-frame stack (about 50 ms) could end the process
    # before that thread ran on a loaded machine (seen once in 4 runs, 2026-10-02), so the stack lasts longer (one
    # thread, like V3: with 64, many frames of few rows leave empty row blocks, which the stack rejects).
    _finite_inputs(job, count=300, shape=(3, 32, 48))
    (job / "siril.ini").touch()
    (job / "stack.ssf").write_text(siril.script_text(_finite_commands(cpu=1)), encoding="utf-8")
    with (job / "stack.log").open("ab") as log, process_guard.guarded_process(
            [str(FINITE_SIRIL), "-i", str(job / "siril.ini"), "-d", str(job), "-s", str(job / "stack.ssf")],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
            env=env) as process:
        returncode = process.wait(timeout=120)
    text = (job / "stack.log").read_text(encoding="utf-8", errors="replace")
    after = tree_listing(desktop)
    assert returncode == 0, text[-2000:]
    assert not (desktop / "siril" / "venv").exists(), after                 # the desktop venv is gone,
    assert "unhealthy" in text.lower(), text[-2000:]                        # judged "unhealthy" by Siril
    module = {k: v for k, v in before.items() if k.startswith("siril/.python_module")}
    assert {k: v for k, v in after.items() if k.startswith("siril/.python_module")} == module
    assert not (job / "siril").exists()


@pytest.mark.parametrize("build", ["stock", "finite"])
def test_version_probe_never_reaches_the_venv(tmp_path, monkeypatch, machine, build):
    """W4: doctor probes with a private environment in a throw-away folder; and even with the desktop folder as its
    data folder, '-v' exits before Siril's venv thread (main-cli.c:86-93), for both builds."""
    executable = STOCK_SIRIL if build == "stock" else FINITE_SIRIL
    if os.name != "nt" or not executable.is_file():
        pytest.skip("needs the local Siril runtimes")
    net = safety_net(tmp_path, monkeypatch)
    real_run = subprocess.run
    seen = []

    def recording(args, **kwargs):
        seen.append(dict(kwargs.get("env") or {}))
        return real_run(args, **kwargs)

    monkeypatch.setattr(siril.subprocess, "run", recording)
    result = siril.doctor(str(executable))
    monkeypatch.setattr(siril.subprocess, "run", real_run)
    assert result["available"] and result["version"] == "1.4.4", result
    (env,) = seen
    probe = Path(env["XDG_DATA_HOME"])
    assert probe.name.startswith("aester-siril-probe-") and not probe.exists()          # removed afterwards
    assert env["MSYSTEM"] == "AESTER" and env["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"
    raw = real_run([str(executable), "-v"], stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
                   creationflags=NO_WINDOW, env=old_launch_environment(net.path))
    assert raw.returncode == 0 and b"1.4.4" in raw.stdout + raw.stderr
    assert tree_listing(net.path) == net.before


# --------------------------------------------------------------------------- W3: the user's session bus (S-4d)

GDBUS = STOCK_SIRIL.parent / "gdbus.exe"
BUS_PROGRAMS = ("siril", "gdbus", "gimp")
# V3 kills a real session-bus daemon: any GLib program of this logon session that joined it meanwhile would end with
# exit code 3, so the test is opt-in, and it still skips when a newcomer has GLib's gio loaded (P1b review SIRIL-5).
KILL_BUS_ENV = "AESTER_TEST_KILL_SESSION_BUS"
GIO_MODULE = re.compile(r"^(lib)?gio-2\.0(-0)?\.dll$", re.IGNORECASE)


class _ProcessEntry(ctypes.Structure):          # PROCESSENTRY32W
    _fields_ = [("dwSize", ctypes.c_uint32), ("cntUsage", ctypes.c_uint32), ("th32ProcessID", ctypes.c_uint32),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", ctypes.c_uint32),
                ("cntThreads", ctypes.c_uint32), ("th32ParentProcessID", ctypes.c_uint32),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", ctypes.c_uint32), ("szExeFile", ctypes.c_wchar * 260)]


def process_table() -> list:
    """(pid, parent pid, lower-case executable name) of every process (a Toolhelp snapshot)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.Process32FirstW.argtypes = kernel32.Process32NextW.argtypes = [ctypes.c_void_p,
                                                                            ctypes.POINTER(_ProcessEntry)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel32.CreateToolhelp32Snapshot(2, 0)                     # TH32CS_SNAPPROCESS
    if snapshot in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            rows.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile.lower()))
            more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return rows


def session_of(pid: int) -> int | None:
    """The logon (Terminal Services) session of ``pid``; None when it cannot be read (gone, or not ours to ask)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.ProcessIdToSessionId.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    session = ctypes.c_uint32()
    return session.value if kernel32.ProcessIdToSessionId(pid, ctypes.byref(session)) else None


def loaded_modules(pid: int):
    """Lower-case base names of the modules loaded in ``pid`` (EnumProcessModulesEx, 32- and 64-bit), "gone" when it
    has exited, or None when it cannot be inspected (access denied, or still starting)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.K32EnumProcessModulesEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                                 ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    kernel32.K32GetModuleBaseNameW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)    # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    if not handle:
        return "gone" if ctypes.get_last_error() == 87 else None     # ERROR_INVALID_PARAMETER: no such process
    try:
        exit_code = ctypes.c_uint32()
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) and exit_code.value != 259:   # STILL_ACTIVE
            return "gone"
        modules, needed = (ctypes.c_void_p * 1024)(), ctypes.c_uint32()
        for _ in range(3):
            if not kernel32.K32EnumProcessModulesEx(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed),
                                                    0x03):           # LIST_MODULES_ALL
                return None
            if needed.value <= ctypes.sizeof(modules):
                break
            modules = (ctypes.c_void_p * (needed.value // ctypes.sizeof(ctypes.c_void_p) + 64))()
        else:
            return None
        names, buffer = set(), ctypes.create_unicode_buffer(520)
        for module in modules[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
            if module and kernel32.K32GetModuleBaseNameW(handle, module, buffer, len(buffer)):
                names.add(buffer.value.lower())
        return names
    finally:
        kernel32.CloseHandle(handle)


def possible_bus_clients(present, ours) -> list:
    """Processes started after the ``present`` snapshot in this logon session (GLib finds the session bus through
    session-local named objects), other than ``ours`` and their children, that have GLib's gio loaded, so may have
    joined the bus, or that cannot be inspected: [(pid, name, "gio" | "uninspectable")]."""
    own_session, ours, found = session_of(os.getpid()), set(ours), []
    for pid, parent, name in process_table():
        if pid in present or pid in ours or parent in ours or session_of(pid) not in (own_session, None):
            continue
        modules = loaded_modules(pid)
        if modules == "gone":
            continue
        if modules is None:
            found.append((pid, name, "uninspectable"))
        elif any(GIO_MODULE.match(module) for module in modules):
            found.append((pid, name, "gio"))
    return found


def gdbus_children(parents) -> list:
    """A gdbus.exe whose parent is one of ``parents``: the session-bus daemon that program autolaunched."""
    parents = set(parents)
    return [pid for pid, parent, name in process_table() if name == "gdbus.exe" and parent in parents]


def wait_for(predicate, timeout, step=.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    return predicate()


def no_session_bus_or_skip():
    """A daemon whose last client just left quits by itself within moments; anything else makes the test skip."""
    running = wait_for(lambda: not [row for row in process_table() if row[2] == "gdbus.exe"], 10.0, step=.2)
    if not running:
        running = [row for row in process_table() if row[2] == "gdbus.exe"]
        pytest.skip(f"a session bus (or a client of one) is already running: {running}; this test needs its own")


def start_glib_app(folder):
    """A GLib program the user opened from Explorer, standing in for a desktop Siril or GIMP: stock gdbus.exe
    monitoring the bus (g_bus_get_sync, so exit-on-close, then g_main_loop_run like a GUI: gdbus-tool.c:463, 2213),
    outside any Job Object and with the user's bus address. With no bus running it autolaunches the daemon as its
    own child, which quits by itself once its last client is gone."""
    env = {k: v for k, v in os.environ.items() if k.upper() != "DBUS_SESSION_BUS_ADDRESS"}
    log = (folder / "glib-app.log").open("wb")
    app = subprocess.Popen([str(GDBUS), "monitor", "--session", "--dest", "org.freedesktop.DBus"], stdout=log,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW, env=env)
    return SimpleNamespace(process=app, log=log, path=folder / "glib-app.log")


def stop_glib_app(app, daemons=()):
    """Close the app; the bus daemon it started quits once its last client is gone (it is never killed here:
    another program may have joined it)."""
    if app is None:
        return
    if app.process.poll() is None:
        app.process.terminate()
    try:
        app.process.wait(timeout=10)
    finally:
        app.log.close()
    daemons = set(daemons)
    wait_for(lambda: not daemons & {pid for pid, _, _ in process_table()}, 10.0, step=.2)


def start_stage(executable, job, commands, processes):
    """One product job stage (siril.run_script in a thread), ended early through the product's cancel path when
    ``stage.cancel`` is set."""
    stage = SimpleNamespace(cancel=threading.Event(), error=None, process=None)

    def check():
        if stage.cancel.is_set():
            raise siril.Cancelled("test: stage end")

    def body():
        try:
            with siril.job_stage_launches():
                siril.run_script(str(executable), job, "stage", commands, check)
        except BaseException as exc:                     # noqa: B036 - asserted by the test
            stage.error = exc

    count = len(processes)
    stage.thread = threading.Thread(target=body, daemon=True)
    stage.thread.start()
    wait_for(lambda: len(processes) > count or not stage.thread.is_alive(), 60)
    stage.process = processes[count] if len(processes) > count else None
    return stage


def end_stage(stage):
    stage.cancel.set()
    stage.thread.join(timeout=120)
    assert not stage.thread.is_alive()


def stock_workload(job: Path, count=600):
    """A stock stage that lasts a few seconds (many tiny frames: Siril's per-frame cost dominates)."""
    from astropy.io import fits
    import numpy as np
    (job / "stage").mkdir(parents=True)
    (job / "work").mkdir()
    rng = np.random.default_rng(1)
    for number in range(1, count + 1):
        fits.PrimaryHDU((.2 + rng.random((3, 32, 48)) * .3).astype(np.float32)).writeto(
            job / "stage" / f"light{number:05d}.fit")
    return ["setcpu 1", "setext fit", "set32bits", "setcompress 0", "cd stage", "convert light -out=../work",
            "cd ../work", "stack light rej w 3 3 -norm=addscale -32b -out=stack.fit"]


@pytest.mark.skipif(os.name != "nt" or not (STOCK_SIRIL.is_file() and GDBUS.is_file()),
                    reason="needs the local Siril runtime")
def test_a_stage_end_leaves_a_users_glib_app_running(tmp_path, monkeypatch, machine):
    """SF-2 V1: before S-4d a stock stage started the bus daemon inside its Job Object, a GLib program opened
    meanwhile joined it, and the stage end killed both (exit code 3, 5/5). Now the stage starts no daemon, the app
    starts its own, and it survives the stage end."""
    no_session_bus_or_skip()
    net = safety_net(tmp_path, monkeypatch)
    processes = observe(monkeypatch)
    job = tmp_path / "stage-job"
    stage = start_stage(STOCK_SIRIL, job, stock_workload(job), processes)
    app, app_bus = None, []
    try:
        assert stage.process is not None, stage.error
        stage_bus = wait_for(lambda: gdbus_children([stage.process.pid]), 1.0)  # default: within about 0.2 s
        app = start_glib_app(tmp_path)
        app_bus = wait_for(lambda: gdbus_children([app.process.pid]), 3.0)
        time.sleep(1.0)
        assert stage.process.poll() is None and app.process.poll() is None, (
            "no overlap: the stage or the app ended early", app.path.read_text(errors="replace"))
        if not stage_bus and not app_bus:
            pytest.skip("the app joined a session bus that another program started meanwhile")
        end_stage(stage)
        time.sleep(3.0)
        assert app.process.poll() is None, (app.process.returncode, app.path.read_text(errors="replace")[-400:])
        assert stage_bus == [], "the stage started the session bus daemon (S-4d lost?)"
        assert isinstance(stage.error, siril.Cancelled), stage.error
        assert_quiet_launch(stage.process, job, net, "normal")
    finally:
        end_stage(stage)
        stop_glib_app(app, app_bus)


@pytest.mark.skipif(os.name != "nt" or not (STOCK_SIRIL.parent / "libgio-2.0-0.dll").is_file(),
                    reason="needs the local Siril runtime")
def test_the_bus_newcomer_check_sees_any_program_with_gio_loaded(tmp_path):
    """P1b review SIRIL-5: V3's guard flags a newcomer by the GLib library it has loaded, whatever its name (here a
    python.exe), and passes over a newcomer without it, the test's own processes and their children, and processes
    that have exited. Nothing here starts or joins a bus (the library is only loaded)."""
    present = {pid for pid, _, _ in process_table()}
    ready = tmp_path / "gio.ready"
    code = ("import ctypes, os, sys, time; os.add_dll_directory(sys.argv[1]); "
            "ctypes.CDLL(os.path.join(sys.argv[1], 'libgio-2.0-0.dll')); open(sys.argv[2], 'w').close(); "
            "time.sleep(120)")
    quiet = dict(os.environ, DBUS_SESSION_BUS_ADDRESS="disabled:")
    gio = subprocess.Popen([sys.executable, "-c", code, str(STOCK_SIRIL.parent), str(ready)],
                           creationflags=NO_WINDOW, env=quiet)
    plain = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], creationflags=NO_WINDOW)
    done = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=NO_WINDOW)
    done.wait(timeout=60)

    def tree(root):              # a venv python.exe may be a launcher that runs the interpreter as its child
        return {root} | {pid for pid, parent, _ in process_table() if parent == root}

    try:
        assert wait_for(ready.exists, 60), "the gio stand-in did not load the library"
        found = {pid: kind for pid, _, kind in possible_bus_clients(present, set())}
        assert "gio" in {found.get(pid) for pid in tree(gio.pid)}, found
        assert not tree(plain.pid) & {pid for pid, kind in found.items() if kind == "gio"}, found
        assert not tree(gio.pid) & {pid for pid, _, _ in possible_bus_clients(present, {gio.pid})}   # ours: passed
        assert loaded_modules(done.pid) == "gone"                     # exited (its handle is still open here)
        assert any(name.startswith("python") for name in loaded_modules(os.getpid()))
    finally:
        for process in (gio, plain):
            for pid in tree(process.pid) - {process.pid}:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
            process.kill()
            process.wait(timeout=30)


@pytest.mark.skipif(os.name != "nt" or not (FINITE_SIRIL.is_file() and GDBUS.is_file()),
                    reason="needs the local Siril runtimes")
@pytest.mark.skipif(os.environ.get(KILL_BUS_ENV) != "1",
                    reason=f"kills a real session-bus daemon in the user's logon session; set {KILL_BUS_ENV}=1 to run")
def test_losing_a_foreign_session_bus_does_not_fail_a_stage(tmp_path, monkeypatch, machine):
    """SF-2 V3: before S-4d a stage joined whatever session bus was running (the finite build cannot start one)
    and, when that bus died, exited with code 3 after its script had finished: run_script failed the job (5/5).
    Now the stage never joins, so losing the bus mid-stage changes nothing. Opt-in (KILL_BUS_ENV), and skipped
    before the kill when any process started meanwhile may be on the bus (possible_bus_clients)."""
    no_session_bus_or_skip()
    present = {pid for pid, _, _ in process_table()}
    net = safety_net(tmp_path, monkeypatch)
    processes = observe(monkeypatch)
    job = tmp_path / "finite-000000000000"
    job.mkdir()
    _finite_inputs(job, count=600, shape=(3, 32, 48))
    app = start_glib_app(tmp_path)
    stage, daemon = None, []
    try:
        daemon = wait_for(lambda: gdbus_children([app.process.pid]), 5.0)
        if not daemon:
            pytest.skip("the app joined a session bus that another program started meanwhile")
        stage = start_stage(FINITE_SIRIL, job, _finite_commands(cpu=1), processes)
        assert stage.process is not None, stage.error
        time.sleep(1.0)                                  # a stage would have joined the bus at its start-up
        assert stage.process.poll() is None, "no overlap: the stage ended before the bus was lost"
        ours = {app.process.pid, stage.process.pid, *daemon}
        newcomers = [row for row in process_table()
                     if row[2].startswith(BUS_PROGRAMS) and row[0] not in present and row[0] not in ours]
        newcomers += possible_bus_clients(present, ours)    # any GLib program, whatever its name (SIRIL-5)
        if newcomers:                                    # they may use this bus: never take it away from them
            stage.thread.join(timeout=120)
            pytest.skip(f"another possible bus program started meanwhile: {newcomers}")
        for pid in daemon:
            os.kill(pid, signal.SIGTERM)                 # TerminateProcess: the bus host dies mid-stage
        stage.thread.join(timeout=120)
        assert stage.error is None, stage.error          # rc 0
        assert all((job / name).is_file() for name in ("finite_master.fit", "finite_master_nused.fit",
                                                        "finite_master_nvalid.fit"))
        assert_quiet_launch(stage.process, job, net, "normal")
    finally:
        if stage is not None:
            end_stage(stage)
        stop_glib_app(app, daemon)


@pytest.mark.parametrize("commands, decodes", [
    (["setext fit", "set32bits", "setcompress 0", "cd stage", "convertraw light -debayer -out=../work"], True),
    (["cd stage", "convert light -out=../work"], True),
    (["cd stage", "CONVERTRAW light"], True),
    (["cd work", "register light -2pass"], False),
    (["cd work", "seqapplyreg light -filter-included"], False),
    (["cd work", "stack r_light rej w 3 3 -norm=addscale -out=sky_master"], False),
    ([], False), (None, False)])
def test_raw_decoding_runs_get_a_small_openmp_pool_and_others_one_thread(tmp_path, commands, decodes):
    # P1b review SIRIL-1: LibRaw's CR3 decoder follows OMP_NUM_THREADS, so RAW-decoding scripts keep a small pool
    expected = str(max(1, min(siril.SIRIL_DECODE_OMP_THREADS, os.cpu_count() or 1))) if decodes else "1"
    assert siril.siril_omp_threads(commands) == expected
    env = siril.siril_environment(tmp_path, None, base={"OMP_NUM_THREADS": "64"}, commands=commands)
    assert env["OMP_NUM_THREADS"] == expected
    pinned = siril.siril_environment(tmp_path, {"cpu": 3, "mem": None}, base={}, commands=commands)
    assert pinned["OMP_NUM_THREADS"] == "3"                    # a cpu pin still wins
