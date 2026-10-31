"""Job lifecycle regression tests using a controlled engine, never live Siril."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import threading

import numpy as np
import pytest
from astropy.io import fits

from starstack import core, imaging

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    source = tmp_path / "素材 with spaces"
    source.mkdir()
    for number in range(3):
        data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
        data[3:6, 5:8] += .2
        imaging.write_linear(source / f"原片 {number}.fit", data, "Test fixture")
    engine = {"available": True, "executable": str(tmp_path / "fake engine.exe"),
              "version": "1.4.4", "message": "Test engine"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    launched = []
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: launched.append(directory))
    return SimpleNamespace(source=source, output=tmp_path / "results with spaces", engine=engine,
                           launched=launched, original={p.name: p.read_bytes() for p in source.iterdir()})


def submit(env, key=None):
    return core.submit_job(str(env.source), str(env.output), idempotency_key=key)


class ControlledEngine:
    """Implement only the adapter's file contract, with injectable stage failures."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.before = None
        self.drop_frame = False
        self.integer_master = False
        self.bad_geometry = False
        monkeypatch.setattr(core, "run_script", self.run)

    def run(self, executable, directory, name, commands, cancel_check):
        phase = core.load_job(directory)["stage"]
        self.calls.append(phase)
        if self.before:
            self.before(phase, directory, cancel_check)
        cancel_check()
        work = directory / "work"
        if phase == "convert":
            sources = sorted((directory / "stage").glob("*"))
            for i, path in enumerate(sources[:-1] if self.drop_frame else sources, 1):
                shutil.copyfile(path, work / f"light_{i:05d}.fit")
        elif phase == "calibrate":
            pass  # transforms-only probe: the controlled engine writes no sequence file, so no distortion is fitted
        elif phase == "stack" and name == "02_export":
            pass  # P1b-3: L's export process; the controlled engine stacks the light frames directly
        elif phase == "stack" and name == "02_stack":
            inputs = [imaging.read_rgb(p) for p in sorted(work.glob("light_*.fit"))]
            data = np.mean(inputs, axis=0).astype(np.float32)
            if self.bad_geometry:
                data = data[:-1]
            if self.integer_master:
                fits.writeto(work / "sky_master.fit", np.moveaxis((data * 1000).astype(np.int16), -1, 0), overwrite=True)
            else:
                imaging.write_linear(work / "sky_master.fit", data, "Controlled engine, not Siril")
        else:
            raise AssertionError(f"Unexpected engine phase: {phase}")


def test_repeated_submission_returns_same_job_and_launches_once(environment):
    first = submit(environment, "client-request-1")
    second = submit(environment, "client-request-1")
    assert second["job_dir"] == first["job_dir"]
    assert len(environment.launched) == 1


def test_same_key_with_changed_input_is_rejected(environment):
    submit(environment, "request")
    p = next(environment.source.glob("*.fit"))
    with p.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        submit(environment, "request")
    assert len(environment.launched) == 1


def test_concurrent_idempotent_submissions_launch_once(environment):
    start = threading.Barrier(6)

    def call():
        start.wait(timeout=10)
        return submit(environment, "simultaneous-request")

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(call) for _ in range(6)]
        results = [future.result(timeout=20) for future in futures]
    assert len({job["job_dir"] for job in results}) == 1
    assert len(environment.launched) == 1


def test_source_directory_cannot_be_output_directory(environment):
    with pytest.raises(ValueError):
        core.submit_job(str(environment.source), str(environment.source))
    assert not environment.launched
    assert {p.name: p.read_bytes() for p in environment.source.iterdir()} == environment.original


def test_success_preserves_originals_and_publishes_readable_artifacts(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    job = submit(environment)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    assert fake.calls == ["convert", "calibrate", "stack", "stack"]   # P1b-3: export, then stack
    assert result["artifacts"]
    for artifact in result["artifacts"]:
        assert Path(artifact["path"]).is_file()
    assert {p.name: p.read_bytes() for p in environment.source.iterdir()} == environment.original


@pytest.mark.parametrize("stacked_count, skipped_count, text", [
    (3, 1, "天空堆栈实际使用 3／4 帧，未参与 1 帧。请检查配准日志。"),
    (4, 0, "天空堆栈实际使用 4／4 帧，未参与 0 帧。"),
    (None, None, "天空实际参与帧数未知（输入 4 帧），未参与数量未知。"),
])
def test_completion_reports_actual_sky_participation(environment, monkeypatch, stacked_count, skipped_count, text):
    shutil.copyfile(next(environment.source.glob("*.fit")), environment.source / "第四帧.fit")
    original = {p.name: p.read_bytes() for p in environment.source.iterdir()}
    engine = ControlledEngine(monkeypatch)

    def run(*args):
        engine.run(*args)
        if args[2] == "02_stack" and stacked_count is not None:
            with fits.open(args[1] / "work" / "sky_master.fit", mode="update") as hdus:
                hdus[0].header["STACKCNT"] = stacked_count

    monkeypatch.setattr(core, "run_script", run)
    job = submit(environment)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    assert text in result["message"]
    assert result["diagnostics"]["input_count"] == 4
    assert result["diagnostics"]["stacked_count"] == stacked_count
    assert result["diagnostics"]["skipped_count"] == skipped_count
    assert result["artifacts"] and engine.calls == ["convert", "calibrate", "stack", "stack"]   # P1b-3: export, then stack
    assert {p.name: p.read_bytes() for p in environment.source.iterdir()} == original


def test_cancel_before_worker_starts_does_not_invoke_engine(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    job = submit(environment)
    core.cancel_job(job["job_dir"])
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "cancelled"
    assert not result["artifacts"]
    assert not fake.calls


def test_cancelled_export_resumes_without_repeating_completed_engine_stages(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    export = imaging.export_products

    def interrupted_export(data, directory, check):
        (directory / "partial.jpg").write_bytes(b"incomplete")
        core.cancel_job(str(directory.parents[1]))
        check()

    monkeypatch.setattr(imaging, "export_products", interrupted_export)
    job = submit(environment)
    core.run_job(job["job_dir"])
    interrupted = core.load_job(job["job_dir"])
    assert interrupted["state"] == "cancelled"
    assert not interrupted["artifacts"]
    monkeypatch.setattr(imaging, "export_products", export)
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "succeeded", result.get("error")
    assert fake.calls == ["convert", "calibrate", "stack", "stack"]   # P1b-3: export, then stack
    assert all(Path(item["path"]).name != "partial.jpg" for item in result["artifacts"])


def test_copy_detects_source_changed_after_submission(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    job = submit(environment)
    source = next(environment.source.glob("*.fit"))
    changed = source.read_bytes() + b"changed after submit"
    source.write_bytes(changed)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed"
    assert not fake.calls
    assert not result["artifacts"]
    assert source.read_bytes() == changed


def test_incomplete_conversion_cannot_publish_success(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)
    fake.drop_frame = True
    job = submit(environment)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed"
    assert fake.calls == ["convert"]
    assert not result["artifacts"]


@pytest.mark.parametrize("invalid", ["integer_master", "bad_geometry"])
def test_master_must_be_float_and_match_reference_geometry(environment, monkeypatch, invalid):
    fake = ControlledEngine(monkeypatch)
    setattr(fake, invalid, True)
    job = submit(environment)
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed", "Invalid engine output was presented as a valid linear master"
    assert not result["artifacts"]


def cancel_during_stack(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)

    def interrupt(phase, directory, check):
        if phase == "stack":
            core.cancel_job(str(directory))
            check()

    fake.before = interrupt
    job = submit(environment)
    core.run_job(job["job_dir"])
    assert core.load_job(job["job_dir"])["state"] == "cancelled"
    fake.before = None
    return fake, job


@pytest.mark.parametrize("change", ["missing", "modified"])
def test_resume_rejects_broken_conversion_checkpoint(environment, monkeypatch, change):
    fake, job = cancel_during_stack(environment, monkeypatch)
    converted = next((Path(job["job_dir"]) / "work").glob("light_*.fit"))
    if change == "missing":
        converted.unlink()
    else:
        with fits.open(converted, mode="update") as hdus:
            hdus[0].data += .01
    try:
        core.resume_job(job["job_dir"])
    except (ValueError, RuntimeError):
        return
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed", "A missing/altered conversion checkpoint must not be trusted"
    assert not result["artifacts"]


def test_resume_rejects_changed_original(environment, monkeypatch):
    _, job = cancel_during_stack(environment, monkeypatch)
    p = next(environment.source.glob("*.fit"))
    p.write_bytes(p.read_bytes() + b"changed")
    launches = len(environment.launched)
    with pytest.raises(ValueError):
        core.resume_job(job["job_dir"])
    assert len(environment.launched) == launches


def test_failed_stack_output_cannot_make_later_no_output_attempt_succeed(environment, monkeypatch):
    fake = ControlledEngine(monkeypatch)

    def writes_then_fails(executable, directory, name, commands, check):
        fake.run(executable, directory, name, commands, check)
        if core.load_job(directory)["stage"] == "stack":
            raise RuntimeError("Engine failed after creating an uncommitted master")

    monkeypatch.setattr(core, "run_script", writes_then_fails)
    job = submit(environment)
    core.run_job(job["job_dir"])
    assert core.load_job(job["job_dir"])["state"] == "failed"
    # A return code alone is insufficient: the new attempt produces no master.
    monkeypatch.setattr(core, "run_script", lambda *args, **kwargs: None)
    core.resume_job(job["job_dir"])
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "failed", "An old failed-attempt master was incorrectly accepted"
    assert not result["artifacts"]


def test_abandoned_running_job_cannot_resume_into_potentially_live_engine(environment, monkeypatch):
    ControlledEngine(monkeypatch)
    job = submit(environment)
    # Simulate persisted state after a worker crash. No worker or OS lock is alive.
    file = Path(job["job_dir"]) / "job.json"
    stored = json.loads(file.read_text(encoding="utf-8"))
    stored.update(state="running", stage="stack", worker_pid=999999999)
    file.write_text(json.dumps(stored), encoding="utf-8")
    observed = core.load_job(job["job_dir"])
    assert observed["state"] == "failed", "A dead worker must not leave the job permanently running"
    with pytest.raises(ValueError, match="子进程"):
        core.resume_job(job["job_dir"])
    assert len(environment.launched) == 1


def test_cancel_completed_job_is_noop(environment, monkeypatch):
    ControlledEngine(monkeypatch)
    job = submit(environment)
    core.run_job(job["job_dir"])
    before = core.load_job(job["job_dir"])
    core.cancel_job(job["job_dir"])
    after = core.load_job(job["job_dir"])
    assert after["state"] == "succeeded"
    assert after["artifacts"] == before["artifacts"]
