"""Adapter process boundaries, using fake subprocesses; no live Siril claims."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import subprocess

import pytest

from starstack import siril


@pytest.fixture
def executable(tmp_path):
    p = tmp_path / "引擎 with spaces" / "siril-cli.exe"
    p.parent.mkdir()
    p.write_bytes(b"test placeholder, never executed")
    return p


@pytest.mark.parametrize("version", ["1.4.0", "1.4.3", "1.5.0"])
def test_doctor_rejects_versions_outside_script_contract(monkeypatch, executable, version):
    monkeypatch.setattr(siril.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=f"Siril {version}\n".encode(), stderr=b""))
    assert siril.doctor(str(executable))["available"] is False


def test_doctor_preserves_explicit_executable_path_as_one_argument(monkeypatch, executable):
    calls = []

    def version(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"Siril 1.4.4\n", stderr=b"")

    monkeypatch.setattr(siril.subprocess, "run", version)
    result = siril.doctor(str(executable))
    assert result["available"] is True
    assert result["version"] == "1.4.4"
    assert calls[0][0] == [str(executable.resolve()), "-v"]
    assert not calls[0][1].get("shell", False)
    assert calls[0][1]["stdin"] == subprocess.DEVNULL


def test_explicit_missing_executable_does_not_silently_choose_another(monkeypatch, executable):
    monkeypatch.setenv("STARSTACK_SIRIL", str(executable))

    def forbidden(*args, **kwargs):
        pytest.fail("An explicit invalid engine path must not fall back to another engine")

    monkeypatch.setattr(siril.subprocess, "run", forbidden)
    assert not siril.doctor(str(executable.parent / "missing.exe"))["available"]


def test_doctor_handles_timeout(monkeypatch, executable):
    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 20)

    monkeypatch.setattr(siril.subprocess, "run", timeout)
    result = siril.doctor(str(executable))
    assert result["available"] is False
    assert result["message"]


class ControlledProcess:
    def __init__(self, finished=None, ignore_terminate=False):
        self.returncode = finished
        self.ignore_terminate = ignore_terminate
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        if self.returncode is None:
            raise subprocess.TimeoutExpired("controlled child", timeout)
        return self.returncode


def attach_process(monkeypatch, process, payload=b""):
    calls = []

    @contextmanager
    def launch(args, **kwargs):
        calls.append((args, kwargs))
        kwargs["stdout"].write(payload)
        yield process

    monkeypatch.setattr(siril, "guarded_process", launch)
    return calls


def test_script_uses_job_config_and_paths_without_shell_interpretation(tmp_path, monkeypatch, executable):
    directory = tmp_path / "作业 & spaces"
    directory.mkdir()
    process = ControlledProcess(finished=0)
    calls = attach_process(monkeypatch, process, b"completed\n")
    siril.run_script(str(executable), directory, "operation", ["close"], lambda: None)
    args, kwargs = calls[0]
    assert args[0] == str(executable)
    assert args[args.index("-d") + 1] == str(directory)
    assert Path(args[args.index("-i") + 1]).parent == directory
    assert Path(args[args.index("-s") + 1]).parent == directory
    assert not kwargs.get("shell", False)
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert not process.terminated


def test_nonzero_exit_reports_engine_diagnostic(tmp_path, monkeypatch, executable):
    process = ControlledProcess(finished=2)
    attach_process(monkeypatch, process, b"Cannot register image: insufficient stars\n")
    with pytest.raises(RuntimeError, match="insufficient stars"):
        siril.run_script(str(executable), tmp_path, "operation", ["close"], lambda: None)


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_cancel_terminates_and_reaps_only_owned_child(tmp_path, monkeypatch, executable, ignore_terminate):
    process = ControlledProcess(ignore_terminate=ignore_terminate)
    attach_process(monkeypatch, process)

    def cancelled():
        raise siril.Cancelled("cancel this job")

    with pytest.raises(siril.Cancelled):
        siril.run_script(str(executable), tmp_path, "operation", ["close"], cancelled)
    assert process.terminated
    assert process.waited
    assert process.killed is ignore_terminate


def test_cancel_after_child_exit_still_wins_over_success(tmp_path, monkeypatch, executable):
    process = ControlledProcess(finished=0)
    attach_process(monkeypatch, process)

    def cancelled():
        raise siril.Cancelled("cancelled at completion")

    with pytest.raises(siril.Cancelled):
        siril.run_script(str(executable), tmp_path, "operation", ["close"], cancelled)
    assert not process.terminated
