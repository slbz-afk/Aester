"""Controlled-clock watchdog fault tests; no live-engine speed assumptions."""
import math
import subprocess

import pytest

from starstack import siril
from test_siril import ControlledProcess, attach_process


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(siril.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(siril.time, "sleep", clock.sleep)
    return clock


def test_silent_process_times_out_and_is_reaped(tmp_path, monkeypatch, clock):
    process = ControlledProcess(ignore_terminate=True)
    attach_process(monkeypatch, process)
    with pytest.raises(siril.SirilTimeoutError) as error:
        siril.run_script("dummy", tmp_path, "silent", ["close"], lambda: None,
                         total_timeout=8, progress_timeout=.5)
    assert error.value.kind == "progress"
    assert error.value.log_path == tmp_path / "silent.log"
    assert process.terminated and process.killed and process.waited
    assert clock.now < 1


def test_log_activity_resets_idle_but_cannot_extend_total_limit(tmp_path, monkeypatch, clock):
    process = ControlledProcess()
    attach_process(monkeypatch, process)

    def activity():
        with (tmp_path / "busy.log").open("ab") as stream:
            stream.write(b"progress\n")

    with pytest.raises(siril.SirilTimeoutError) as error:
        siril.run_script("dummy", tmp_path, "busy", [], activity,
                         total_timeout=1, progress_timeout=.3)
    assert error.value.kind == "total"
    assert process.terminated


def test_env_override_and_explicit_none_disable_watchdogs(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("STARSTACK_SIRIL_TOTAL_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("STARSTACK_SIRIL_PROGRESS_TIMEOUT_SECONDS", "0.1")
    process = ControlledProcess()
    attach_process(monkeypatch, process)

    def complete():
        if clock.now >= 1:
            process.returncode = 0

    siril.run_script("dummy", tmp_path, "disabled", [], complete,
                     total_timeout=None, progress_timeout=None)
    assert not process.terminated
    process = ControlledProcess()
    attach_process(monkeypatch, process)
    with pytest.raises(siril.SirilTimeoutError):
        siril.run_script("dummy", tmp_path, "env", [], lambda: None)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "garbage"])
def test_invalid_timeout_rejected_before_child_creation(tmp_path, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration must not start the engine")
    monkeypatch.setattr(siril, "guarded_process", forbidden)
    with pytest.raises(ValueError):
        siril.run_script("dummy", tmp_path, "invalid", [], lambda: None, total_timeout=value)
    assert not list(tmp_path.iterdir())


def test_cancel_remains_the_error_when_killed_child_wait_times_out(tmp_path, monkeypatch, clock):
    class StuckProcess(ControlledProcess):
        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            assert timeout is not None, "cleanup must never wait indefinitely"
            self.waited = True
            raise subprocess.TimeoutExpired("stuck", timeout)

    process = StuckProcess(ignore_terminate=True)
    attach_process(monkeypatch, process)
    def cancel():
        raise siril.Cancelled("stop")
    with pytest.raises(siril.Cancelled, match="stop"):
        siril.run_script("dummy", tmp_path, "stuck", [], cancel)
    assert process.terminated and process.killed and process.waited


def test_successful_silent_engine_can_finish_within_deadline(tmp_path, monkeypatch, clock):
    process = ControlledProcess()
    attach_process(monkeypatch, process)
    def complete():
        if clock.now >= .4:
            process.returncode = 0
    siril.run_script("dummy", tmp_path, "ok", [], complete,
                     total_timeout=1, progress_timeout=1)
    assert process.returncode == 0 and not process.terminated
