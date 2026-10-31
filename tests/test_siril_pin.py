"""Opt-in AESTER_SIRIL_PIN (efficiency P0 1.3): byte-identical when unset, exact lines and env when set.

No real Siril run: guarded_process is replaced by a recorder, except one test
that starts a real child through the real guard to prove the environment
reaches it. Since efficiency P1a S-4b every Siril run gets its own environment
block (siril.siril_environment) and, on Windows, a launch note in its log.
"""
import importlib.util
import os
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from starstack import siril
from test_siril import ControlledProcess

COMMANDS = [
    ["cd work", "convert light -out=../work"],
    ["offline", "setext fit", "set32bits", "setcompress 0", "setref q 1", "stack q rej w 3 3"],
    [],
    ["offline"],
    ["set gui.icc_autoconversion=0", "setcpu 1", "cd stage"],
]
BENCH = Path(__file__).resolve().parents[1] / "scripts" / "bench_workflow.py"


def legacy_script(commands):
    """The script text run_script wrote before the pin existed (commit 0eba79b)."""
    return "requires 1.4.4 1.5.0\n" + "\n".join(commands) + "\n"


def legacy_bytes(tmp_path, commands):
    """Those bytes as the old Path.write_text call put them on disk (platform newlines)."""
    probe = tmp_path / "legacy-probe.ssf"
    probe.write_text(legacy_script(commands), encoding="utf-8")
    return probe.read_bytes()


def launch_note(priority="normal"):
    """What a Windows Siril log starts with since S-4b (after a pin note); nothing elsewhere."""
    return siril.launch_note(priority).encode("utf-8") if os.name == "nt" else b""


def recorder(monkeypatch, target=siril):
    calls = []

    @contextmanager
    def launch(args, **kwargs):
        # The Siril process's OMP_NUM_THREADS: from its own environment block when one is passed
        # (S-4b), otherwise inherited from this process (older code, the bench's shim).
        environment = kwargs["env"] if kwargs.get("env") is not None else os.environ
        calls.append({"args": args, "kwargs": kwargs, "omp": environment.get("OMP_NUM_THREADS")})
        yield ControlledProcess(finished=0)

    monkeypatch.setattr(target, "guarded_process", launch)
    return calls


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("commands", COMMANDS)
def test_unset_pin_keeps_scripts_logs_and_launch_identical(tmp_path, monkeypatch, value, commands):
    if value is None:
        monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    else:
        monkeypatch.setenv(siril.SIRIL_PIN_ENV, value)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    calls = recorder(monkeypatch)
    before = dict(os.environ)
    assert siril.run_script("siril-cli.exe", tmp_path, "step", commands, lambda: None) is None
    assert (tmp_path / "step.ssf").read_bytes() == legacy_bytes(tmp_path, commands)
    assert (tmp_path / "step.log").read_bytes() == launch_note()
    # P1b U-G3: an unpinned Siril gets a one-thread OpenMP pool (its own threads unchanged); this process keeps none
    assert calls[0]["omp"] == siril.siril_omp_threads(commands) and dict(os.environ) == before
    assert set(calls[0]["kwargs"]) == {"stdin", "stdout", "stderr", "creationflags", "env"}
    assert siril.pin_diagnostics() is None and siril.pin_commands(None) == [] and siril.pin_environment(None) == {}


@pytest.mark.parametrize("commands", COMMANDS)
def test_script_text_without_pin_equals_the_old_formula(commands):
    assert siril.script_text(commands) == legacy_script(commands)
    assert siril.script_text(iter(commands), None) == legacy_script(commands)


@pytest.mark.parametrize("value, lines, omp", [
    ("cpu=16,mem=16", ["setcpu 16", "set core.mem_mode=1", "set core.mem_amount=16"], "16"),
    (" CPU = 8 ", ["setcpu 8"], "8"),
    ("mem=4", ["set core.mem_mode=1", "set core.mem_amount=4"], None),
    ("mem=12,cpu=4", ["setcpu 4", "set core.mem_mode=1", "set core.mem_amount=12"], "4"),
])
@pytest.mark.parametrize("commands", COMMANDS)
def test_pin_inserts_header_lines_and_siril_only_environment(tmp_path, monkeypatch, value, lines, omp, commands):
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, value)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")   # restored afterwards, not leaked
    calls = recorder(monkeypatch)
    siril.run_script("siril-cli.exe", tmp_path, "step", commands, lambda: None)
    head = 1 if commands and commands[0] == "offline" else 0
    expected = commands[:head] + lines + commands[head:]
    assert (tmp_path / "step.ssf").read_bytes() == legacy_bytes(tmp_path, expected)
    assert calls[0]["omp"] == (omp or siril.siril_omp_threads(commands))   # U-G3 replaces the inherited 3 for Siril only
    assert os.environ["OMP_NUM_THREADS"] == "3"
    note = (tmp_path / "step.log").read_text(encoding="utf-8")
    assert note.startswith("aester: AESTER_SIRIL_PIN applied: ") and note.count("\n") == (2 if os.name == "nt" else 1)
    assert note.encode("utf-8").endswith(launch_note())
    assert all(line in note for line in lines) and (f"OMP_NUM_THREADS={omp}" in note) == bool(omp)
    diagnostics = siril.pin_diagnostics()
    assert diagnostics["script_commands"] == lines and diagnostics["source"] == "AESTER_SIRIL_PIN"
    assert diagnostics["process_environment"] == ({"OMP_NUM_THREADS": omp} if omp else {})


def test_pin_environment_is_removed_again_when_it_was_unset(tmp_path, monkeypatch):
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=6")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    calls = recorder(monkeypatch)
    siril.run_script("siril-cli.exe", tmp_path, "step", ["close"], lambda: None)
    assert calls[0]["omp"] == "6" and "OMP_NUM_THREADS" not in os.environ


def test_explicit_later_setcpu_in_a_script_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=16")
    recorder(monkeypatch)
    siril.run_script("siril-cli.exe", tmp_path, "rcd", ["setcpu 1", "cd stage"], lambda: None)
    lines = (tmp_path / "rcd.ssf").read_text(encoding="utf-8").splitlines()
    assert lines == ["requires 1.4.4 1.5.0", "setcpu 16", "setcpu 1", "cd stage"]


@pytest.mark.parametrize("value", ["cpu=0", "cpu=-1", "cpu=16,", "threads=4", "cpu=4,cpu=8", "cpu=1.5",
                                   "mem=16GB", "cpu", "cpu=١٦", "cpu=123456"])
def test_malformed_pin_fails_before_any_script_is_written(tmp_path, monkeypatch, value):
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, value)
    calls = recorder(monkeypatch)
    with pytest.raises(ValueError, match="AESTER_SIRIL_PIN"):
        siril.run_script("siril-cli.exe", tmp_path, "step", ["close"], lambda: None)
    assert not calls and not (tmp_path / "step.ssf").exists()


def test_failed_start_still_restores_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(siril.SIRIL_PIN_ENV, "cpu=5")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    @contextmanager
    def failing(args, **kwargs):
        raise FileNotFoundError(args[0])
        yield  # pragma: no cover

    monkeypatch.setattr(siril, "guarded_process", failing)
    with pytest.raises(FileNotFoundError):
        siril.run_script("missing.exe", tmp_path, "step", ["close"], lambda: None)
    assert os.environ["OMP_NUM_THREADS"] == "2"


@pytest.mark.skipif(os.name != "nt", reason="the Job Object guard is Windows-only")
def test_real_guarded_child_receives_omp_and_parent_is_restored(tmp_path, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MSYSTEM", raising=False)
    before = dict(os.environ)
    out = tmp_path / "env.txt"
    code = ("import os,sys; open(sys.argv[1],'w',encoding='utf-8').write('|'.join(os.environ.get(k,'<unset>') "
            "for k in ('OMP_NUM_THREADS','XDG_DATA_HOME','MSYSTEM')))")
    with open(tmp_path / "x.log", "ab", buffering=0) as log, \
            siril._started([sys.executable, "-c", code, str(out)], log, {"cpu": 7}, tmp_path) as process:
        deadline = time.monotonic() + 60
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert process.poll() == 0
    assert out.read_text(encoding="utf-8").split("|") == ["7", os.path.abspath(tmp_path), "AESTER"]
    assert dict(os.environ) == before
    log = (tmp_path / "x.log").read_bytes()
    assert log.startswith(b"aester: AESTER_SIRIL_PIN applied: setcpu 7; OMP_NUM_THREADS=7\n")
    assert log.endswith(launch_note())


# --- the bench's copy of the rules and its shim for older code roots must match the product exactly

def _bench():
    spec = importlib.util.spec_from_file_location("bench_workflow_pin_tested", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hook(tmp_path):
    bench = _bench()
    path = bench.write_hook(tmp_path / "hook")
    spec = importlib.util.spec_from_file_location("aester_bench_hook_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return bench, module


@pytest.mark.parametrize("value", ["cpu=16,mem=16", "mem=3", "cpu=2", "", None, "cpu=0", "x=1", "cpu=4,cpu=4"])
def test_bench_and_hook_parsers_agree_with_the_product(tmp_path, value):
    bench, hook = _hook(tmp_path)
    try:
        expected = siril.siril_pin(value)
    except ValueError:
        with pytest.raises(bench.BenchError):
            bench.parse_pin(value)
        with pytest.raises(ValueError):
            hook._parse(value)
        return
    assert bench.parse_pin(value) == expected == hook._parse(value)
    assert bench.pin_lines(expected) == siril.pin_commands(expected) == hook._commands(expected)
    assert hook._environment(expected) == siril.pin_environment(expected)


def _old_siril(tmp_path):
    """A replica of the pre-pin starstack.siril.run_script launch path (commit 0eba79b)."""
    module = types.ModuleType("old_siril_replica")
    module.guarded_process = siril.guarded_process

    def run_script(executable, job_dir, name, commands, cancel_check, **_timeouts):
        script = job_dir / f"{name}.ssf"
        script.write_text("requires 1.4.4 1.5.0\n" + "\n".join(commands) + "\n", encoding="utf-8")
        log = job_dir / f"{name}.log"
        config = job_dir / "siril.ini"
        config.touch(exist_ok=True)
        args = [executable, "-i", str(config), "-d", str(job_dir), "-s", str(script)]
        with log.open("ab", buffering=0) as output, module.guarded_process(
                args, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            process.poll()

    module.run_script = run_script
    return module


@pytest.mark.parametrize("value", ["cpu=16,mem=16", "mem=8", ""])
@pytest.mark.parametrize("commands", COMMANDS[:2])
def test_hook_shim_on_old_code_matches_native_pin_byte_for_byte(tmp_path, monkeypatch, value, commands):
    _bench_module, hook = _hook(tmp_path)
    monkeypatch.setenv("AESTER_SIRIL_PIN", value)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    native_dir, old_dir = tmp_path / "native", tmp_path / "old"
    native_dir.mkdir(), old_dir.mkdir()
    native_calls = recorder(monkeypatch)
    siril.run_script("siril-cli.exe", native_dir, "step", commands, lambda: None)
    old = _old_siril(tmp_path)
    old_calls = recorder(monkeypatch, old)
    assert hook.patch_siril(old) is True and hook.patch_siril(old) is False
    old.run_script("siril-cli.exe", old_dir, "step", commands, lambda: None)
    for name in ("step.ssf", "siril.ini"):
        assert (native_dir / name).read_bytes() == (old_dir / name).read_bytes(), name
    # S-4b: the native log adds the launch note after the (identical) pin note
    assert (native_dir / "step.log").read_bytes() == (old_dir / "step.log").read_bytes() + launch_note()
    # U-G3: without a cpu pin the native run also gets a one-thread OpenMP pool; the older code never set one
    assert native_calls[0]["omp"] == (old_calls[0]["omp"] or siril.siril_omp_threads(commands))
    assert "OMP_NUM_THREADS" not in os.environ


def test_hook_leaves_a_natively_pinned_module_alone(tmp_path):
    _bench_module, hook = _hook(tmp_path)
    original = siril.run_script
    assert hook.patch_siril(siril) is False and siril.run_script is original
