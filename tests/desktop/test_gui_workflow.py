"""Menu navigation and whole-workflow state, through real Qt callbacks."""
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from starstack import workflow
from aester_desktop import gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def drain(app, window):
    deadline = time.monotonic() + 5
    while window._workers or window._pool.activeThreadCount():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("GUI background callback timed out")
        time.sleep(.005)
    app.processEvents()


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    instance = gui.MainWindow(persist_preferences=False)
    instance.input_dir.setText(str(tmp_path))
    instance.output_dir.setText(str(tmp_path / "output"))
    instance.show()
    app.processEvents()
    yield instance
    instance.timer.stop()
    drain(app, instance)
    instance.close()
    app.processEvents()


@pytest.mark.parametrize("size", [(960, 680), (1200, 800)])
def test_essential_path_visible_without_scrolling(app, window, size):
    window.resize(*size)
    app.processEvents()
    area = window.centralWidget().rect()
    for widget in (window.add_files_button, window.file_strip, window.output_dir,
                   window.start_button, window.cancel_button, window.progress_bar, window.message_label):
        top = widget.mapTo(window.centralWidget(), QPoint(0, 0))
        bottom = widget.mapTo(window.centralWidget(), QPoint(widget.width() - 1, widget.height() - 1))
        assert widget.isVisible() and area.contains(top) and area.contains(bottom), widget.objectName()
    assert window.preview.width() >= 500 and window.preview.height() >= 220
    assert not window.siril_path.isVisible()
    assert not window.star_background_strength.isVisible()
    assert not window.red_gain.isVisible()
    assert not window.input_dir.isVisible()
    assert window.warmth_slider.isVisible() and window.tint_slider.isVisible()


@pytest.mark.parametrize("menu_key,dialog_key", [("settings", "settings"), ("colors", "colors"), ("raw", "raw"), ("stars", "stars")])
def test_menu_opens_advanced_and_close_preserves_work(app, window, menu_key, dialog_key):
    before = window.input_dir.text()
    window.menu_actions[menu_key].trigger()
    app.processEvents()
    dialog = window._advanced_dialogs[dialog_key]
    assert dialog.isVisible()
    dialog.close()
    app.processEvents()
    assert window.isVisible() and window.input_dir.text() == before


def test_f1_opens_shared_guide_and_contains_gui_and_mcp(app, window):
    window.activateWindow()
    window.setFocus()
    QTest.keyClick(window, Qt.Key.Key_F1)
    app.processEvents()
    assert window._advanced_dialogs["help"].isVisible()
    text = window.help_browser.toPlainText()
    assert "submit_workflow" in text and "GUI" in text and "MCP" in text
    window._advanced_dialogs["help"].close()


@pytest.mark.parametrize('offset', [0, -35, 40])
def test_model_adapter_and_strength_are_frozen_before_dispatch(window, monkeypatch, tmp_path, offset):
    files = []
    for name in ("model.onnx", "mask.png", "engine.exe"):
        path = tmp_path / name
        path.write_bytes(b"test")
        files.append(str(path))
    window.star_model_path.setText(files[0])
    window.stars_enabled.setChecked(True)
    window.sky_mask.setText(files[1])
    window.stack_engine_path.setText(files[2])
    window.model_channel_order.setCurrentIndex(1)
    window.model_range.setCurrentIndex(1)
    window.model_stitch.setCurrentIndex(1)
    window.workflow_auto_strength.setChecked(False)
    window.workflow_background_strength.setValue(74)
    if offset:
        from starstack import regions
        window._regions_project = {'operations': [{'kind': 'horizon'}],
                                   'milkyway': {'status': 'present', 'authority': 'human', 'confirmed': True, 'ellipse': [.5,.4,.2,.4]}}
        monkeypatch.setattr(regions, 'revise', lambda project, **kwargs: project)
        monkeypatch.setattr(regions, 'save_project', lambda project, output: {'project_path': str(tmp_path/'regions.json')})
        window.workflow_milkyway_offset.setValue(offset)
    pending, calls = [], []
    monkeypatch.setattr(window, "_run_async", lambda *args: pending.append(args))
    monkeypatch.setattr(workflow, "submit_workflow", lambda *a, **kw: calls.append((a, kw)))
    window.start_processing()
    assert window._busy and not window.workflow_background_strength.isEnabled()
    window.workflow_background_strength.setValue(1)
    window.model_channel_order.setCurrentIndex(0)
    window.workflow_milkyway_offset.setValue(0)
    pending[0][1]()
    options = calls[0][1]
    assert options["background_strength"] == .74
    assert options["backend_options"] == {"layout": "nhwc", "channel_order": "bgr", "value_range": "-1..1", "stitch": "crop"}
    assert options["stack_engine_path"] == files[2] and options["milkyway_offset"] == offset / 100
    window._busy = False


def test_model_without_explicit_sky_scope_is_rejected(window, monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test")
    window.star_model_path.setText(str(model))
    window.stars_enabled.setChecked(True)
    monkeypatch.setattr(window, "_run_async", lambda *a: pytest.fail("no dispatch"))
    window.start_processing()
    assert "需要明确的天空分区" in window.message_label.text()
    assert not window._busy


def test_stack_complete_does_not_finish_workflow_and_final_photo_is_first(app, window, monkeypatch, tmp_path):
    root = tmp_path / "workflow"
    root.mkdir()
    stack = root / "stack"
    stack.mkdir()
    result = {"workflow_dir": str(root), "stack_job_dir": str(stack), "state": "running",
              "stage": "separation", "progress": 80, "message": "生成星空图层", "artifacts": []}
    monkeypatch.setattr(workflow, "workflow_status", lambda p: result)
    monkeypatch.setattr(gui, "load_job", lambda p: {"state": "succeeded", "diagnostics": {"color_export": {"state": "succeeded"}}})
    window._workflow_submitted(result)
    assert window.job_state == "running" and window.timer.isActive()
    assert not window.start_button.isEnabled() and window.cancel_button.isEnabled()
    for name, color in (("before.jpg", (90, 90, 90)), ("final.jpg", (12, 25, 40))):
        Image.new("RGB", (64, 40), color).save(root / name)
    separation = root / "result.json"
    separation.write_text("{}", encoding="utf8")
    result.update(state="succeeded", progress=100, stage="complete", message="已保存",
                  separation_result=str(separation), finishing={"state": "succeeded"},
                  artifacts=[{"path": str(root / "before.jpg"), "label": "之前", "kind": "jpeg"},
                             {"path": str(root / "final.jpg"), "label": "最终", "kind": "jpeg"}])
    window.refresh_job()
    drain(app, window)
    assert window.job_state == "succeeded" and not window.timer.isActive()
    assert window.style_tabs.tabText(0) == "最终" and window.style_tabs.currentIndex() == 0
    assert window.star_separation.text() == str(separation)
    assert not window.preview._pixmap.isNull() and "星点整理已运行" in window.preview_note.text()


def test_workflow_failure_and_full_detail_are_visible(app, window, monkeypatch, tmp_path):
    root = tmp_path / "workflow"
    root.mkdir()
    error = "无法处理输入：" + "有可审查的详细失败信息。" * 80
    record = {"workflow_dir": str(root), "state": "failed", "progress": 20, "error": error, "artifacts": []}
    monkeypatch.setattr(workflow, "workflow_status", lambda p: record)
    window._workflow_submitted(record)
    assert error == window.message_label.text() and not window.timer.isActive()
    window.details_button.click()
    app.processEvents()
    assert window._advanced_dialogs["details"].isVisible() and error in window.job_details.toPlainText()


def test_cancel_dispatches_workflow_not_just_stack(app, window, monkeypatch, tmp_path):
    window.workflow_dir = str(tmp_path)
    window.job_dir = str(tmp_path / "stack")
    window.job_state = "running"
    calls = []
    monkeypatch.setattr(workflow, "cancel_workflow", lambda p: calls.append(p))
    monkeypatch.setattr(gui, "cancel_job", lambda p: pytest.fail("must stop whole workflow"))
    monkeypatch.setattr(window, "refresh_job", lambda: None)
    window.cancel_processing()
    drain(app, window)
    assert calls == [str(tmp_path)] and not window._busy


@pytest.mark.parametrize("state,note", [("skipped", "跳过缩星"), ("reduction_skipped", "密度建议不足")])
def test_skips_are_not_claimed_as_star_reduction(window, state, note):
    window._workflow = {"state": "succeeded", "finishing": {"state": state}, "config": {"star_model_path": "model.onnx"}}
    assert note in window._products_note()


def test_active_job_locks_file_menus_and_paths(window):
    window.job_state = "running"
    window._update_controls()
    assert not window.menu_actions["input"].isEnabled() and not window.menu_actions["load"].isEnabled()
    assert not window.input_dir.isEnabled() and not window.star_model_path.isEnabled()


def test_explicit_stack_engine_disables_old_positive_only_option(window):
    window.stack_engine_path.setText("candidate.exe")
    assert not window.stack_range_checkbox.isEnabled()
    assert "指定数值保真" in window.stack_range_checkbox.toolTip()


def test_preferences_reuse_tools_without_reusing_scene_inputs(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    original = gui.MainWindow(preferences=settings)
    original.siril_path.setText("siril.exe")
    original.stack_engine_path.setText("finite.exe")
    original.star_model_path.setText("my-model.onnx")
    original.output_dir.setText(str(tmp_path / "results"))
    original.input_dir.setText("private-scene")
    original.sky_mask.setText("scene-mask.png")
    original.star_milkyway_mask.setText("old-milkyway.png")
    original.model_range.setCurrentIndex(1)
    original.model_channel_order.setCurrentIndex(1)
    original.close()
    app.processEvents()
    restored = gui.MainWindow(preferences=settings)
    try:
        assert restored.siril_path.text() == "siril.exe"
        assert restored.stack_engine_path.text() == "finite.exe"
        assert restored.star_model_path.text() == "my-model.onnx"
        assert restored.model_range.currentData() == "-1..1" and restored.model_channel_order.currentData() == "bgr"
        assert restored.output_dir.text() == str(tmp_path / "results")
        assert restored.input_dir.text() == restored.sky_mask.text() == restored.star_milkyway_mask.text() == ""
    finally:
        restored.close()
        app.processEvents()


@pytest.mark.parametrize("key", ["background", "milkyway", "trails", "layers"])
def test_advanced_menu_exposes_existing_operations(app, window, key):
    window.menu_actions[key].trigger()
    app.processEvents()
    assert window._advanced_dialogs[key].isVisible()
    assert window._extra_tools.fields[key]
    window._advanced_dialogs[key].close()


def test_background_fields_dispatch_existing_backend_off_ui_thread(app, window, monkeypatch, tmp_path):
    from starstack import background_job
    window.menu_actions["background"].trigger()
    tools = window._extra_tools
    fields = tools.fields["background"]
    fields["job"].setText("completed-job")
    fields["samples"].setText("samples.png")
    fields["mode"].setCurrentIndex(1)
    fields["strength"].setValue(42)
    calls = []
    monkeypatch.setattr(background_job, "render_background", lambda *a, **kw: calls.append((a, kw)) or {"state": "succeeded", "directory": str(tmp_path)})
    tools.run_background()
    drain(app, window)
    assert calls == [(("completed-job", "samples.png"), {"mode": "gradient", "strength": .42})]
    assert not window._busy and tools.outputs["background"] == str(tmp_path)


def test_milkyway_requires_explicit_confirmation_and_binds_actual_files(app, window, monkeypatch, tmp_path):
    import json
    from starstack import core, photographic_job
    from starstack.milkyway_gate import assess_evidence
    window.menu_actions["milkyway"].trigger()
    tools = window._extra_tools
    fields = tools.fields["milkyway"]
    for key in ("source", "sky", "protect", "milkyway"):
        path = tmp_path / (key + ".tif")
        path.write_bytes(key.encode())
        fields[key].setText(str(path))
    assert not fields["confirm"].isChecked()
    calls = []
    def finish(*args, **kwargs):
        record = json.loads(Path(kwargs["milkyway_evidence"]).read_text(encoding="utf8"))
        gate = assess_evidence(record, source_sha256=core._sha(tmp_path / "source.tif"),
                               mask_sha256=core._sha(tmp_path / "milkyway.tif"))
        assert gate["enabled"] and record["method"] == "human-review"
        calls.append((args, kwargs))
        return {"state": "succeeded", "subject_gate": gate, "processing_directory": str(tmp_path)}
    monkeypatch.setattr(photographic_job, "finish_milkyway", finish)
    tools.run_milkyway()
    assert not calls and not window._busy
    fields["confirm"].setChecked(True)
    tools.run_milkyway()
    drain(app, window)
    assert len(calls) == 1 and "跳过" not in tools.statuses["milkyway"].text()
    fields["source"].setText("different.tif")
    assert not fields["confirm"].isChecked()


def test_layer_export_uses_selected_separation_and_current_output(app, window, monkeypatch, tmp_path):
    from starstack import star_layer_export
    window.star_separation.setText("chosen/result.json")
    window.menu_actions["layers"].trigger()
    tools = window._extra_tools
    calls = []
    monkeypatch.setattr(star_layer_export, "export_star_layers", lambda *a: calls.append(a) or {"state": "succeeded", "processing_directory": str(tmp_path)})
    tools.run_layers()
    drain(app, window)
    assert calls == [("chosen/result.json", str(tmp_path / "output"))]


def test_advanced_failure_keeps_input_and_restores_controls(app, window, monkeypatch):
    from starstack import star_layer_export
    window.menu_actions["layers"].trigger()
    tools = window._extra_tools
    tools.fields["layers"]["separation"].setText("broken.json")
    def fail(*args):
        raise ValueError("分层文件已改变")
    monkeypatch.setattr(star_layer_export, "export_star_layers", fail)
    tools.run_layers()
    drain(app, window)
    assert not window._busy and "分层文件已改变" in tools.statuses["layers"].text()
    assert tools.fields["layers"]["separation"].isEnabled()


def test_loaded_long_paths_keep_full_values_and_show_identifying_tails(window, monkeypatch, tmp_path):
    root = tmp_path / "workflow"
    root.mkdir()
    config = {"input_dir": str(tmp_path / "a-very-long-shared-prefix" / "raw_1" / "6"),
              "output_root": str(tmp_path / "a-very-long-shared-prefix" / "results"),
              "sky_mask_path": str(tmp_path / "a-very-long-shared-prefix" / "sky_mask.png")}
    record = {"workflow_dir": str(root), "state": "succeeded", "artifacts": [], "config": config}
    monkeypatch.setattr(workflow, "workflow_status", lambda p: record)
    window._workflow_submitted(record)
    for field, key in ((window.input_dir, "input_dir"), (window.output_dir, "output_root"), (window.sky_mask, "sky_mask_path")):
        assert field.text() == field.toolTip() == config[key]
        assert field.cursorPosition() == len(config[key])
        assert not field.hasSelectedText()
