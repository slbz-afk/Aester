"""Independent GUI state-transition probes; no engine, decoding or image claim."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication
from starstack import workflow
from aester_desktop import gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    widget = gui.MainWindow(persist_preferences=False)
    calls = []
    monkeypatch.setattr(widget, "_run_async", lambda *args: calls.append(args))
    monkeypatch.setattr(widget, "refresh_job", lambda: None)
    widget.output_dir.setText(str(tmp_path / "results"))
    paths = []
    for group in ("A", "B"):
        folder = tmp_path / group
        folder.mkdir()
        selected = []
        for i in range(3):
            p = folder / f"same{i}.DNG"
            p.write_bytes(f"{group}{i}".encode())
            selected.append(str(p))
        paths.append(selected)
    widget.add_input_files(paths[0])
    widget._color_timer.stop()
    yield widget, calls, paths, tmp_path
    widget._busy = False
    widget._reference_running = False
    widget.timer.stop()
    widget._color_timer.stop()
    widget.close()
    app.processEvents()


def reference(root, files, name="preview"):
    preview = root / (name + ".png")
    Image.new("RGB", (4, 3), (40, 30, 20)).save(preview)
    mask = root / (name + "-mask.png")
    Image.new("L", (4, 3), 255).save(mask)
    return {"reference_dir": str(root), "source_path": files[0], "preview_path": str(preview),
            "mask_template_path": str(mask), "width": 4, "height": 3}


def test_new_selected_source_invalidates_mask_and_reference(panel):
    w, _, paths, root = panel
    w._reference_data = reference(root, paths[0])
    w._reference_scene = tuple(paths[0])
    w.sky_mask.setText(w._reference_data["mask_template_path"])
    w.add_input_files([paths[1][0]])
    assert w._reference_data is None and not w._reference_scene
    assert not w.sky_mask.text()
    assert w.input_files == paths[0] + [paths[1][0]]


def test_first_frame_removal_preserves_remaining_order(panel):
    w, _, paths, _ = panel
    w.file_strip.item(0).setSelected(True)
    w.remove_selected_files()
    assert w.input_files == paths[0][1:]
    assert w.file_strip.item(0).text().startswith("首帧")
    assert w._reference_data is None


def test_submit_freezes_file_order_and_shared_adjustment_values(panel, monkeypatch):
    w, pending, paths, root = panel
    w._reference_data = reference(root, paths[0])
    w._reference_scene = tuple(paths[0])
    mask = root / "landscape.png"
    Image.new("L", (4, 3), 255).save(mask)
    w.sky_mask.setText(str(mask))
    w.warmth_slider.setValue(-17)
    w.tint_slider.setValue(23)
    w.exposure_slider.setValue(7)
    w.trails_enabled.setChecked(True)
    received = []
    monkeypatch.setattr(workflow, "submit_workflow", lambda *a, **k: received.append((a, k)))
    w.start_processing()
    assert len(pending) == 1
    w.input_files.reverse()
    w.warmth_slider.setValue(50)
    pending[0][1]()
    args = received[0][1]
    assert args["input_files"] == paths[0]
    assert args["color_adjustments"] == {"warmth": -17., "tint": 23., "exposure_ev": .7}
    assert args["trails_enabled"] is True and args["stars_enabled"] is False


def test_pure_sky_uses_own_reference_mask_not_imported_other_scene(panel, monkeypatch):
    w, pending, paths, root = panel
    ref = reference(root, paths[0])
    w._reference_data, w._reference_scene = ref, tuple(paths[0])
    other = root / "unrelated-mask.png"
    Image.new("L", (4, 3), 0).save(other)
    w.sky_mask.setText(str(other))
    w.sky_mask.clear()  # Clearing the foreground now replaces the removed pure-sky selector.
    w.scene_mode.setCurrentIndex(w.scene_mode.findData("sky"))
    received = []
    monkeypatch.setattr(workflow, "submit_workflow", lambda *a, **k: received.append(k))
    w.start_processing()
    pending[0][1]()
    assert received[0]["sky_mask_path"] == ref["mask_template_path"]


def test_selecting_siril_file_invalidates_previous_reference(panel, monkeypatch):
    w, _, paths, root = panel
    w._reference_data = reference(root, paths[0])
    w._reference_scene = tuple(paths[0])
    backend = root / "different.exe"
    backend.write_bytes(b"synthetic executable")
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *a, **k: (str(backend), ""))
    w._pick_siril()
    assert w._reference_data is None
    assert w._reference_pending


def test_old_backend_completion_cannot_repopulate_invalidated_cache(panel):
    w, pending, paths, root = panel
    w.siril_path.setText(str(root / "before.exe"))
    w._launch_reference_preview()
    assert len(pending) == 1
    w.siril_path.setText(str(root / "after.exe"))
    w._invalidate_reference_backend()
    pending[0][2](reference(root, paths[0], "old-backend"))
    assert w._reference_data is None, "Old backend result was accepted as current reusable cache"
    assert w._reference_pending


def test_loading_directory_workflow_never_reuses_previous_explicit_selection(panel):
    w, _, paths, root = panel
    old = {"workflow_dir": str(root / "workflow-old"), "state": "failed", "config": {
        "input_dir": str(Path(paths[1][0]).parent), "output_root": str(root / "results")},
        "request": {"input_files": paths[1]}}
    w._workflow_submitted(old)
    assert w.input_files in ([], paths[1]), "Unrelated prior scene remains the active file selection"
    assert w.input_files != paths[0]


def test_loading_different_workflow_removes_unrelated_reference(panel):
    w, _, paths, root = panel
    w._reference_data = reference(root, paths[0])
    w._reference_scene = tuple(paths[0])
    other = {"workflow_dir": str(root / "workflow-other"), "state": "succeeded", "config": {
        "input_dir": str(Path(paths[1][0]).parent), "input_files": paths[1],
        "output_root": str(root / "results"), "color_adjustments": {"exposure_ev": -.3}}}
    w._workflow_submitted(other)
    assert w.input_files == paths[1]
    assert w._reference_data is None and not w._reference_scene
    assert w.color_adjustments()["exposure_ev"] == -.3


def test_review_notice_does_not_claim_trails_added(panel):
    w, _, _, _ = panel
    w._workflow = {"state": "succeeded", "needs_review": True}
    assert "待复核" in w._products_note() and "尚未加回" in w._products_note()
    for control in (w.warmth_slider, w.tint_slider, w.exposure_slider):
        assert control.accessibleName()


def test_loading_legacy_job_clears_previous_selection_and_effect_intent(panel, monkeypatch):
    w, pending, paths, root = panel
    w._reference_data = reference(root, paths[0])
    w._reference_scene = tuple(paths[0])
    w.warmth_slider.setValue(33)
    w.trails_enabled.setChecked(True)
    w.stars_enabled.setChecked(True)
    w.scene_mode.setCurrentIndex(w.scene_mode.findData("sky"))
    directory = root / "job-legacy"
    directory.mkdir()
    (directory / "job.json").write_text("{}", encoding="utf-8")
    job = {"job_dir": str(directory), "state": "failed", "inputs": [{"path": p} for p in paths[1]],
           "mask": None, "engine": {"executable": ""}}
    monkeypatch.setattr(gui, "load_job", lambda *_: job)
    w.load_existing_job(str(directory))
    assert len(pending) == 1
    pending[0][2](pending[0][1]())
    assert not w.input_files and not w._reference_data and not w._reference_scene
    assert w.input_dir.text() == str(Path(paths[1][0]).parent)
    assert w.color_adjustments() == {"warmth": 0., "tint": 0., "exposure_ev": 0.}
    assert not w.trails_enabled.isChecked() and not w.stars_enabled.isChecked()
    assert w.scene_mode.currentData() != "sky"


def test_loading_workflow_resets_previous_pure_sky_scope(panel):
    w, _, paths, root = panel
    w.scene_mode.setCurrentIndex(w.scene_mode.findData("sky"))
    mask = root / "other-landscape.png"
    Image.new("L", (4, 3), 120).save(mask)
    result = {"workflow_dir": str(root / "workflow-different"), "state": "failed", "config": {
        "input_dir": str(Path(paths[1][0]).parent), "input_files": paths[1], "sky_mask_path": str(mask),
        "output_root": str(root / "results")}}
    w._workflow_submitted(result)
    assert w.scene_mode.currentData() != "sky"
    assert w.sky_mask.text() == str(mask)
