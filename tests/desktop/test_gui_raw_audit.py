"""The GUI keeps RAW inspection separate from existing images and jobs."""

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from starstack import raw_audit
from aester_desktop import gui


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait_for_workers(app, window):
    deadline = time.monotonic() + 5
    while window._workers or window._pool.activeThreadCount():
        app.processEvents()
        if time.monotonic() >= deadline:
            pytest.fail("GUI inspection callback did not finish")
        time.sleep(.005)
    app.processEvents()


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    instance = gui.MainWindow(persist_preferences=False)
    instance.output_dir.setText(str(tmp_path / "outputs"))
    yield instance
    wait_for_workers(app, instance)
    instance.close()
    app.processEvents()


@pytest.mark.parametrize("has_derivative", [False, True])
def test_inspection_runs_async_without_replacing_job_or_preview(app, window, monkeypatch, tmp_path, has_derivative):
    source = str(tmp_path / "single.nef")
    out = tmp_path / "outputs"
    audit = out / "raw-audit-test"
    audit.mkdir(parents=True)
    record = {"state": "succeeded", "audit_dir": str(audit), "artifacts": {"raw_codes": {"path": "raw_codes.npy"}}}
    if has_derivative:
        record["artifacts"]["black_subtracted_cfa"] = {"path": "black_subtracted_cfa.npy"}
    (audit / "audit.json").write_text(json.dumps(record), encoding="utf-8")
    calls = []

    def inspect(selected, output):
        calls.append((selected, output))
        return record

    monkeypatch.setattr(raw_audit, "inspect_raw", inspect)
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: (source, ""))
    window.input_dir.setText(str(tmp_path / "original-input"))
    window.sky_mask.setText(str(tmp_path / "existing-mask.png"))
    window.job_dir = str(tmp_path / "previous-job")
    window.job_state = "succeeded"
    reference = {"retained": "reference"}
    preview = {"job_dir": window.job_dir, "retained": "preview"}
    window._reference_data = reference
    window._rendered_preview = preview
    window._view_mode = "reference"
    window._update_controls()
    before = (window.input_dir.text(), window.output_dir.text(), window.sky_mask.text(), window.job_dir)

    window.raw_audit_button.click()
    assert window._busy
    assert not window.raw_audit_button.isEnabled()
    wait_for_workers(app, window)

    assert calls == [(source, str(out))]
    assert not window._busy
    assert window.raw_audit_button.isEnabled()
    assert window.open_raw_audit_button.isEnabled()
    assert str(audit) in window.raw_audit_status.text()
    assert ("黑位衍生可用" if has_derivative else "仅保存解码采样") in window.raw_audit_status.text()
    assert before == (window.input_dir.text(), window.output_dir.text(), window.sky_mask.text(), window.job_dir)
    assert window._reference_data is reference
    assert window._rendered_preview is preview
    assert window._view_mode == "reference"


def test_failed_record_exposes_report_and_allows_retry(app, window, monkeypatch, tmp_path):
    audit = tmp_path / "failed-audit"
    audit.mkdir()
    report = audit / "audit.json"
    record = {"state": "failed", "audit_dir": str(audit), "error": "当前引擎无法读取此文件", "artifacts": {}}
    report.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(raw_audit, "inspect_raw", lambda *args: record)
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: (str(tmp_path / "bad.nef"), ""))
    window.inspect_single_raw()
    wait_for_workers(app, window)
    assert not window._busy
    assert window.raw_audit_button.isEnabled()
    assert window.open_raw_audit_button.isEnabled()
    assert "检查未完成" in window.raw_audit_status.text()
    assert str(report) in window.raw_audit_status.text()
    assert record["error"] in window.raw_audit_status.text()


def test_preflight_exception_releases_busy_without_inventing_report(app, window, monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError("RAW 检查结果与原片文件夹必须分开")

    monkeypatch.setattr(raw_audit, "inspect_raw", fail)
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: (str(tmp_path / "sample.dng"), ""))
    window.inspect_single_raw()
    wait_for_workers(app, window)
    assert not window._busy
    assert window.raw_audit_button.isEnabled()
    assert not window.open_raw_audit_button.isEnabled()
    assert "必须分开" in window.raw_audit_status.text()
    assert window._raw_audit_dir is None


def test_active_job_and_picker_cancel_do_not_start_inspection(window, monkeypatch):
    calls = []
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: (calls.append("picker") or "", ""))
    monkeypatch.setattr(raw_audit, "inspect_raw", lambda *args: pytest.fail("must not inspect"))
    window.job_state = "running"
    window._update_controls()
    assert not window.raw_audit_button.isEnabled()
    window.inspect_single_raw()
    assert not calls
    window.job_state = ""
    window.inspect_single_raw()
    assert calls == ["picker"]
    assert not window._busy
    assert not window._workers
