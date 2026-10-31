"""Regression coverage for Siril reporting RAW failure with exit status zero.

Engine output and subprocesses are controlled; placeholder NEFs are never
decoded. These tests do not claim camera-format or image-quality validation.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from starstack import core, references, siril


UNSUPPORTED = (b"log: Error in libraw Unsupported file format or not RAW file\n"
               b"log: Conversion aborted, 0 file(s) created for 1 input file(s)\n"
               b"log: Script execution finished successfully.\n")


def engine_output(log):
    """The log without Aester's own notes (since S-4b every Windows Siril log starts with a launch note)."""
    return b"".join(line for line in log.read_bytes().splitlines(keepends=True) if not line.startswith(b"aester:"))


@pytest.fixture
def environment(tmp_path, monkeypatch):
    source = tmp_path / "素材 with spaces"
    source.mkdir()
    for index in range(3):
        (source / f"frame{index}.nef").write_bytes(b"controlled placeholder, never decoded")
    monkeypatch.setattr(core, "doctor", lambda *_: {
        "available": True, "executable": "controlled-not-executed.exe", "version": "1.4.4"})
    monkeypatch.setattr(core, "_spawn_worker", lambda *_: None)
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(references, "_optional_raw_metadata", lambda *_: {"available": False})
    return SimpleNamespace(source=source, output=tmp_path / "结果 with spaces",
                           originals={p.name: p.read_bytes() for p in source.iterdir()})


def attach_engine(monkeypatch, payload, exit_code=0, product_count=0):
    @contextmanager
    def launch(args, **kwargs):
        kwargs["stdout"].write(payload)
        directory = Path(args[args.index("-d") + 1])
        script = Path(args[args.index("-s") + 1]).read_text(encoding="utf-8")
        prefix = "ref" if "convertraw ref " in script else "light"
        for index in range(product_count):
            # Count mismatch must be diagnosed before trying to read pixels.
            (directory / "work" / f"{prefix}_{index + 1:05d}.fit").write_bytes(b"partial product")
        yield SimpleNamespace(returncode=exit_code, poll=lambda: exit_code)

    monkeypatch.setattr(siril, "guarded_process", launch)


def failed_result(mode, environment):
    if mode == "job":
        job = core.submit_job(str(environment.source), str(environment.output))
        core.run_job(job["job_dir"])
        result = core.load_job(job["job_dir"])
        assert result["checkpoints"] == [] and result["artifacts"] == []
        return result, Path(job["job_dir"])
    with pytest.raises(RuntimeError, match="参考图准备失败"):
        references.prepare_reference(str(environment.source), str(environment.output))
    manifest, = environment.output.glob("reference-*/reference.json")
    result = json.loads(manifest.read_text(encoding="utf-8"))
    assert "preview_path" not in result and "mask_template_path" not in result
    return result, manifest.parent


@pytest.mark.parametrize("mode", ["job", "reference"])
@pytest.mark.parametrize("exit_code", [0, 2])
def test_actual_libraw_error_survives_zero_or_nonzero_engine_exit(environment, monkeypatch, mode, exit_code):
    attach_engine(monkeypatch, UNSUPPORTED, exit_code)
    result, directory = failed_result(mode, environment)
    assert result["state"] == "failed" and result["stage"] == "convert"
    assert "无法解码此 RAW 文件的格式或压缩方式" in result["error"]
    assert "无额外有损压缩的 DNG 副本" in result["error"] and "请保留原片" in result["error"]
    assert str(directory / "01_convert.log") in result["error"]
    assert "HE" not in result["error"]
    if exit_code == 0:
        assert "实际找到 0" in result["error"]
    assert engine_output(directory / "01_convert.log") == UNSUPPORTED
    assert {p.name: p.read_bytes() for p in environment.source.iterdir()} == environment.originals


@pytest.mark.parametrize("mode", ["job", "reference"])
def test_unknown_conversion_failure_keeps_count_and_log_without_guessing_nef_compression(environment, monkeypatch, mode):
    attach_engine(monkeypatch, b"log: Conversion aborted for an unknown reason\n", product_count=2)
    result, directory = failed_result(mode, environment)
    assert result["state"] == "failed"
    assert "实际找到 2" in result["error"]
    assert str(directory / "01_convert.log") in result["error"]
    assert "DNG" not in result["error"] and "无法解码" not in result["error"]


@pytest.mark.parametrize("unreadable", [False, True])
def test_missing_or_unreadable_log_preserves_original_failure(tmp_path, unreadable):
    path = tmp_path / "01_convert.log"
    if unreadable:
        path.mkdir()  # Opening a directory as a file fails on all supported OSes.
    message = siril.conversion_failure_message(path, "预期 3 帧，实际找到 0 帧")
    assert "预期 3 帧，实际找到 0 帧" in message and str(path) in message
    assert "DNG" not in message


def test_appended_retry_diagnostics_exclude_old_unsupported_error(tmp_path):
    path = tmp_path / "01_convert.log"
    path.write_bytes(UNSUPPORTED)
    offset = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b"log: unexpected interruption\xff\n")
    message = siril.conversion_failure_message(path, "实际找到 0 帧", start_offset=offset)
    assert "实际找到 0 帧" in message and str(path) in message
    assert "DNG" not in message and "无法解码" not in message


def test_job_retry_uses_current_log_offset(environment, monkeypatch):
    attach_engine(monkeypatch, UNSUPPORTED)
    result, directory = failed_result("job", environment)
    assert "DNG" in result["error"]
    attach_engine(monkeypatch, b"log: another unknown conversion failure\n")
    core.resume_job(str(directory))
    core.run_job(str(directory))
    retried = core.load_job(directory)
    assert retried["state"] == "failed"
    assert "实际找到 0" in retried["error"] and "DNG" not in retried["error"]
    assert engine_output(directory / "01_convert.log").startswith(UNSUPPORTED)


def test_cancelled_conversion_remains_cancelled(environment, monkeypatch):
    def cancel(*args):
        raise siril.Cancelled("controlled cancellation")

    monkeypatch.setattr(core, "run_script", cancel)
    job = core.submit_job(str(environment.source), str(environment.output))
    core.run_job(job["job_dir"])
    result = core.load_job(job["job_dir"])
    assert result["state"] == "cancelled" and result["error"] is None
