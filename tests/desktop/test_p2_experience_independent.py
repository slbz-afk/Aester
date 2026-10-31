"""Independent user-state checks; workflow admission remains the real service."""
from pathlib import Path
import sys

import pytest

from starstack import workflow as wf, regions
from test_gui_workflow import app, window
from test_p1_gui_routing_independent import scene


def capture_async(window, monkeypatch):
    queue = []
    monkeypatch.setattr(window, "_run_async", lambda *args: queue.append(args))
    return queue


def connection(window, tmp_path):
    source = tmp_path / "raw"
    source.mkdir(exist_ok=True)
    window.input_dir.setText(str(source))
    window.output_dir.setText(str(tmp_path / "works"))
    window.cache_dir.setText(str(tmp_path / "cache"))
    window.experience.show_connection()
    return window.experience


def test_invalid_new_scope_clears_previous_success(window, tmp_path):
    experience = connection(window, tmp_path)
    experience.connection_status.setText("本机服务已通过握手与工具调用")
    experience.mcp_input.setText("")
    experience._refresh_connection()
    assert experience._config is None
    assert "已通过" not in experience.connection_status.text()


@pytest.mark.parametrize("change", ["input", "output", "cache", "invalid"])
def test_pending_handshake_never_certifies_changed_configuration(window, monkeypatch, tmp_path, change):
    experience = connection(window, tmp_path)
    queue = capture_async(window, monkeypatch)
    experience._test_connection()
    assert len(queue) == 1 and experience._testing
    next_directory = tmp_path / "new"
    next_directory.mkdir()
    controls = {"input": experience.mcp_input, "output": window.output_dir,
                "cache": window.cache_dir, "invalid": experience.mcp_input}
    controls[change].setText("" if change == "invalid" else str(next_directory))
    queue[0][2]({"state": "ready", "message": "OLD DETECTION SUCCESS"})
    assert "OLD DETECTION SUCCESS" not in experience.connection_status.text()
    assert "改变" in experience.connection_status.text()
    assert not experience._testing and experience.test_connection.isEnabled()


def test_repeated_component_checks_have_one_outstanding_callback(window, monkeypatch):
    queue = capture_async(window, monkeypatch)
    experience = window.experience
    experience.show_components()
    for _ in range(4):
        experience._load_components()
        experience.show_components()
    assert len(queue) == 1 and queue[0][0] == "component-status"
    queue[0][2]({"ready": False, "components": [{"id": "base", "status": "missing"}]})
    assert not experience._checking_components
    experience._load_components()
    assert len(queue) == 2


def test_old_component_snapshot_cannot_overwrite_completed_install(window, monkeypatch):
    queue = capture_async(window, monkeypatch)
    experience = window.experience
    experience.show_components()
    old_callback = queue[0][2]
    missing = {"ready": False, "components": [{"id": "base", "status": "missing"}]}
    ready = {"ready": True, "components": [{"id": "base", "status": "ready"}]}
    experience._install_components()
    if len(queue) == 1:  # Serializing install behind the outstanding check is valid.
        old_callback(missing)
        experience._install_components()
        assert queue[-1][0] == "component-install"
        queue[-1][2](ready)
    else:  # Concurrent implementation must reject the obsolete snapshot.
        assert queue[-1][0] == "component-install"
        queue[-1][2](ready)
        old_callback(missing)
    assert experience._component_result["ready"] is True
    assert "未安装" not in experience.component_text.toPlainText()
    assert not window._busy


def test_gui_freezes_separate_roots_through_real_workflow_admission(window, monkeypatch, tmp_path):
    scene(window, monkeypatch, tmp_path, present=False)
    cache = tmp_path / "cache"
    works = Path(window.output_dir.text())
    window.cache_dir.setText(str(cache))
    queue = capture_async(window, monkeypatch)
    window.start_processing()
    assert len(queue) == 1
    window.cache_dir.setText(str(tmp_path / "later-cache"))
    window.output_dir.setText(str(tmp_path / "later-works"))
    try:
        admitted = queue[0][1]()  # Real workflow API; only engine and spawn are stubbed by scene().
        assert admitted["config"]["output_root"] == str(cache)
        assert admitted["config"]["delivery_root"] == str(works)
        assert Path(admitted["workflow_dir"]).parent == cache
        assert Path(admitted["request"]["regions_path"]).is_relative_to(cache)
        queue[0][2](admitted)
        window.timer.stop()
        assert window.cache_dir.text() == str(cache)
        assert window.output_dir.text() == str(works)
    finally:
        window._busy = False
        window.job_state = None
        window.workflow_dir = None


def test_invalid_cache_in_raw_does_not_write_region_before_rejection(window, monkeypatch, tmp_path):
    scene(window, monkeypatch, tmp_path, present=False)
    raw = Path(window.input_dir.text())
    window.cache_dir.setText(str(raw))
    before = {path.name for path in raw.iterdir()}
    queue = capture_async(window, monkeypatch)
    try:
        window.start_processing()
        if queue:
            with pytest.raises(ValueError):
                queue[0][1]()
        assert {path.name for path in raw.iterdir()} == before
    finally:
        window._busy = False


def test_reopening_legacy_workflow_retains_old_work_and_delivery_locations(window, monkeypatch, tmp_path):
    scene(window, monkeypatch, tmp_path, present=False)
    window.cache_dir.setText(str(tmp_path / "new-default-cache"))
    old_output = tmp_path / "legacy-output"
    admitted = wf.submit_workflow(window.input_dir.text(), old_output,
        siril_path=window.siril_path.text(), input_files=window.input_files)
    assert "delivery_root" not in admitted["request"]
    capture_async(window, monkeypatch)
    try:
        window._workflow_submitted(admitted)
        window.timer.stop()
        assert window.cache_dir.text() == str(tmp_path / "new-default-cache")
        assert window.output_dir.text() == str(old_output)
        assert window.workflow_dir == admitted["workflow_dir"]
        assert "delivery_root" not in wf._read(admitted["workflow_dir"])["config"]
    finally:
        window._busy = False
        window.job_state = None
        window.workflow_dir = None


def test_install_into_new_runtime_requires_restart_and_mcp_uses_new_python(window, monkeypatch, tmp_path):
    scene(window, monkeypatch, tmp_path, present=False)
    window.cache_dir.setText(str(tmp_path / "cache"))
    experience = window.experience
    queue = capture_async(window, monkeypatch)
    experience.show_components()
    queue.pop()[2]({"ready": False, "components": []})
    experience._install_components()
    runtime = tmp_path / "new-runtime"
    runtime.mkdir()
    python = runtime / "python.exe"
    python.write_bytes(b"synthetic executable identity, never executed")
    queue.pop()[2]({"ready": True, "python": str(python), "components": [{"id": "base", "status": "ready"}]})
    assert experience._restart_required
    assert "重启" in experience.component_message.text() or "重新启动" in experience.component_message.text()
    assert not window.start_button.isEnabled()
    assert not window.stars_enabled.isChecked()
    queue.clear()
    window.start_processing()
    assert not queue  # Direct slot invocation is guarded as well as the button.
    experience.show_connection()
    assert experience._config["mcpServers"]["aester"]["command"] == str(python)
    assert "已通过" not in experience.connection_status.text()


def test_existing_runtime_success_does_not_need_restart_or_enable_effects(window, monkeypatch):
    queue = capture_async(window, monkeypatch)
    experience = window.experience
    experience.show_components()
    queue.pop()[2]({"ready": False, "components": []})
    experience._install_components()
    queue.pop()[2]({"ready": True, "python": sys.executable, "components": [{"id": "base", "status": "ready"}]})
    assert not experience._restart_required
    assert not window.stars_enabled.isChecked() and not window.trails_enabled.isChecked()
    assert not window._busy


def test_welcome_can_be_skipped_and_reopened_without_forcing_mcp(app, window, tmp_path):
    from PySide6.QtCore import QSettings
    prefs = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    window._preferences = prefs
    experience = window.experience
    experience.maybe_welcome()
    app.processEvents()
    assert window._advanced_dialogs["welcome"].isVisible()
    experience._welcome_done()
    assert str(prefs.value("experience/welcome_seen")) == "1"
    experience.maybe_welcome()
    app.processEvents()
    assert not window._advanced_dialogs["welcome"].isVisible()
    assert "connect_ai" not in window._advanced_dialogs
    experience.show_welcome()
    assert window._advanced_dialogs["welcome"].isVisible()
