"""Aester P0 selection, opt-in and stale-preview contracts through real Qt."""
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication
from starstack import references, workflow
from aester_desktop import gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    result = gui.MainWindow(persist_preferences=False)
    result.output_dir.setText(str(tmp_path / "output"))
    result.show()
    app.processEvents()
    yield result
    result._color_timer.stop()
    result.timer.stop()
    deadline = time.monotonic() + 5
    while result._workers or result._pool.activeThreadCount():
        app.processEvents()
        assert time.monotonic() < deadline
        time.sleep(.005)
    result.close()
    app.processEvents()


def sources(tmp_path, count=3):
    root = tmp_path / "source"
    root.mkdir(exist_ok=True)
    result = []
    for i in range(count):
        path = root / f"photo-{i}.dng"
        path.write_bytes(b"synthetic placeholder")
        result.append(str(path.resolve()))
    return result


def reference(tmp_path, name="one"):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    preview, mask = root / "preview.png", root / "mask.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(preview)
    Image.new("L", (64, 48), 255).save(mask)
    return {"reference_dir": str(root), "preview_path": str(preview),
            "mask_template_path": str(mask), "width": 64, "height": 48,
            "preview_width": 32, "preview_height": 24}


def capture_async(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_run_async", lambda *args: calls.append(args))
    return calls


def test_brand_menus_defaults_and_pictures(app, monkeypatch):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    window = gui.MainWindow(persist_preferences=False)
    try:
        assert "Aester" in window.windowTitle()
        assert [m.text() for m in window.menuBar().actions()] == ["文件", "设置", "帮助"]
        menu_texts = [a.text() for m in window.menuBar().actions() for a in m.menu().actions()]
        assert not any("RAW" in text or "背景协调" in text or "完整出图设置" in text for text in menu_texts)
        assert Path(window.output_dir.text()).name == "Aester"
        assert not window.trails_enabled.isChecked() and not window.stars_enabled.isChecked()
        assert not window.milkyway_enabled.isEnabled() and not window.milkyway_enabled.isChecked()
        assert window.color_adjustments() == {"warmth": 0., "tint": 0., "exposure_ev": 0.}
        assert (window.warmth_slider.minimum(), window.warmth_slider.maximum()) == (-100, 100)
        assert (window.exposure_slider.minimum(), window.exposure_slider.maximum()) == (-40, 40)
    finally:
        window.close()


def test_selection_preserves_explicit_order_and_deduplicates(window, tmp_path):
    paths = sources(tmp_path)
    window.add_input_files([paths[2], paths[0], paths[2]])
    window._color_timer.stop()
    assert window.input_files == [paths[2], paths[0]]
    assert window.file_strip.count() == 2
    assert "首帧" in window.file_strip.item(0).text()
    assert not window.start_button.isEnabled()
    window.add_input_files([paths[1]])
    window._color_timer.stop()
    assert window.input_files == [paths[2], paths[0], paths[1]]
    # The first preview freezes the automatic capture direction before submit.
    assert not window.start_button.isEnabled()
    window._reference_data = {**reference(tmp_path), 'image_rotation': 0}
    window._reference_scene = tuple(window.input_files)
    window._reference_pending = False
    window._update_controls()
    assert window.start_button.isEnabled()


def test_selection_never_sweeps_siblings_and_removing_first_invalidates_scope(window, tmp_path):
    paths = sources(tmp_path, 4)
    window.add_input_files(paths[:3])
    window._color_timer.stop()
    window._reference_data = reference(tmp_path)
    window._reference_scene = tuple(paths[:3])
    window.sky_mask.setText("old-scene-mask.png")
    window.file_strip.item(0).setSelected(True)
    window.remove_selected_files()
    window._color_timer.stop()
    assert window.input_files == paths[1:3] and paths[3] not in window.input_files
    assert window.sky_mask.text() == "" and window._reference_data is None
    assert window.scene_mode.currentData() == "sky"


def test_selection_ignores_directory_unsupported_and_missing_files(window, tmp_path):
    paths = sources(tmp_path)
    text = tmp_path / "note.txt"
    text.write_text("no image")
    window.add_input_files([str(tmp_path), str(text), str(tmp_path / "gone.nef")])
    assert not window.input_files and "未添加" in window.message_label.text()
    window.add_input_files(paths)
    window._color_timer.stop()
    assert window.input_files == paths


def test_progressive_stars_does_not_enable_from_installed_model(window, app):
    window.star_model_path.setText("installed.onnx")
    assert not window.stars_enabled.isChecked() and not window.workflow_star_panel.isVisible()
    window.stars_enabled.setChecked(True)
    app.processEvents()
    assert window.workflow_star_panel.isVisible() and window.workflow_auto_strength.isChecked()
    assert not window.workflow_background_strength.isEnabled()
    window.workflow_auto_strength.setChecked(False)
    window.workflow_background_strength.setValue(87)
    window.stars_enabled.setChecked(False)
    window.stars_enabled.setChecked(True)
    assert window.workflow_background_strength.value() == 87


def test_submission_freezes_selected_files_color_and_opt_in(window, monkeypatch, tmp_path):
    paths = sources(tmp_path)
    window.add_input_files(paths[::-1])
    window._color_timer.stop()
    record = reference(tmp_path)
    window._reference_data, window._reference_scene = record, tuple(paths[::-1])
    window.scene_mode.setCurrentIndex(1)
    window.warmth_slider.setValue(18)
    window.tint_slider.setValue(-7)
    window.exposure_slider.setValue(6)
    window._color_timer.stop()
    window.trails_enabled.setChecked(True)
    pending = capture_async(window, monkeypatch)
    received = []
    monkeypatch.setattr(workflow, "submit_workflow", lambda *a, **kw: received.append((a, kw)))
    window.start_processing()
    assert window._busy and len(pending) == 1
    window.warmth_slider.setValue(-30)
    pending[0][1]()
    kw = received[0][1]
    assert kw["input_files"] == paths[::-1]
    assert kw["color_adjustments"] == {"warmth": 18., "tint": -7., "exposure_ev": .6}
    assert kw["trails_enabled"] is True and kw["stars_enabled"] is False
    assert kw["sky_mask_path"] == record["mask_template_path"]
    window._busy = False


@pytest.mark.parametrize("mode", [0, 2])
def test_unmarked_scene_requires_reference_geometry_before_submit(window, monkeypatch, tmp_path, mode):
    window.add_input_files(sources(tmp_path))
    window._color_timer.stop()
    window.scene_mode.setCurrentIndex(mode)
    pending = capture_async(window, monkeypatch)
    window.start_processing()
    assert not pending and not window._busy and "预览" in window.message_label.text()


def test_unmarked_scene_defaults_to_sky_without_creating_external_mask(window, monkeypatch, tmp_path):
    paths = sources(tmp_path, 1)
    window.add_input_files(paths)
    window._color_timer.stop()
    pending = capture_async(window, monkeypatch)
    window._launch_reference_preview()
    record = reference(tmp_path)
    pending[0][2](record)
    assert window.sky_mask.text() == "" and window.scene_mode.currentData() == "sky"
    assert window._reference_data == record
    assert len(pending) == 2  # one backend preview + one QImage decode
    assert not window.start_button.isEnabled()


def test_preview_passes_color_and_cached_render_does_not_decode_again(window, monkeypatch, tmp_path):
    paths = sources(tmp_path)
    window.add_input_files(paths)
    window._color_timer.stop()
    pending = capture_async(window, monkeypatch)
    calls = []
    record = reference(tmp_path)
    monkeypatch.setattr(references, "prepare_reference", lambda *a, **kw: calls.append(("prepare", a, kw)) or record)
    monkeypatch.setattr(references, "render_reference", lambda *a, **kw: calls.append(("render", a, kw)) or record, raising=False)
    window._launch_reference_preview()
    first = pending[0]
    first[2](first[1]())
    assert calls[0][2]["input_files"] == paths
    window.warmth_slider.setValue(28)
    window._color_timer.stop()
    window._launch_reference_preview()
    pending[-1][2](pending[-1][1]())
    assert [c[0] for c in calls] == ["prepare", "render"]
    assert calls[1][2]["color_adjustments"]["warmth"] == 28


def test_stale_scene_completion_cannot_replace_new_scene(window, monkeypatch, tmp_path):
    paths = sources(tmp_path)
    window.add_input_files(paths[:1])
    window._color_timer.stop()
    pending = capture_async(window, monkeypatch)
    window._launch_reference_preview()
    window.file_strip.item(0).setSelected(True)
    window.remove_selected_files()
    window.add_input_files(paths[1:])
    window._color_timer.stop()
    window._launch_reference_preview()
    assert len(pending) == 1  # bound to one heavy preview in flight
    pending[0][2](reference(tmp_path, "old"))
    window._color_timer.stop()
    assert window._reference_data is None
    window._launch_reference_preview()
    assert len(pending) == 2
    pending[1][2](reference(tmp_path, "new"))
    assert window._reference_scene == tuple(paths[1:])
    assert "new" in window._reference_data["reference_dir"]


def test_stale_color_result_is_cached_but_not_displayed(window, monkeypatch, tmp_path):
    window.add_input_files(sources(tmp_path))
    window._color_timer.stop()
    pending = capture_async(window, monkeypatch)
    window._launch_reference_preview()
    window.tint_slider.setValue(22)
    window._color_timer.stop()
    pending[0][2](reference(tmp_path))
    window._color_timer.stop()
    assert len(pending) == 1 and window._reference_data is not None
    window._launch_reference_preview()
    assert len(pending) == 2


def test_old_preview_error_not_overwrite_new_scene_status(window, monkeypatch, tmp_path):
    window.add_input_files(sources(tmp_path))
    window._color_timer.stop()
    pending = capture_async(window, monkeypatch)
    window._launch_reference_preview()
    old = pending[0][0]
    window.warmth_slider.setValue(25)
    window._color_timer.stop()
    window.message_label.setText("new scene")
    window._handle_error(old, "old failed")
    window._color_timer.stop()
    assert window.message_label.text() == "new scene" and not window._reference_running


def test_review_completion_honestly_reports_base_image_only(window):
    window._workflow = {"state": "succeeded", "needs_review": True, "finishing": {"state": "skipped"}}
    assert "尚未加回" in window._products_note() and "待复核" in window._products_note()


def test_unavailable_trail_scan_is_not_reported_as_done(window):
    window._workflow = {"state": "succeeded", "trail_review": {"state": "unavailable"},
                        "finishing": {"state": "skipped"}}
    assert "扫描未完成" in window._products_note() and "尚未加回" in window._products_note()


def test_old_workflow_displays_its_effective_legacy_model_choice(window, monkeypatch, tmp_path):
    record = {"workflow_dir": str(tmp_path), "state": "succeeded", "artifacts": [],
              "config": {"star_model_path": "legacy-model.onnx"}}
    monkeypatch.setattr(workflow, "workflow_status", lambda p: record)
    window._workflow_submitted(record)
    assert window.stars_enabled.isChecked()
    record["config"]["stars_enabled"] = False
    window._workflow_submitted(record)
    assert not window.stars_enabled.isChecked()


def test_loaded_project_restores_effect_and_color_intent_without_new_compute(window, monkeypatch, tmp_path):
    paths = sources(tmp_path)
    record = {"workflow_dir": str(tmp_path), "state": "succeeded", "artifacts": [], "config": {
        "input_files": paths[::-1], "trails_enabled": True, "stars_enabled": False,
        "color_adjustments": {"warmth": 16., "tint": -3., "exposure_ev": -.8}}}
    monkeypatch.setattr(workflow, "workflow_status", lambda p: record)
    window._workflow_submitted(record)
    assert window.input_files == paths[::-1]
    assert window.trails_enabled.isChecked() and not window.stars_enabled.isChecked()
    assert window.color_adjustments() == record["config"]["color_adjustments"]
    assert not window._color_timer.isActive()
