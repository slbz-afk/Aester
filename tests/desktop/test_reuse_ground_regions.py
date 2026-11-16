"""Fixed tripod, several segments: the previous job's ground marks are offered for the next segment's first frame.

Real Qt widgets offscreen, real region projects (starstack.regions) on synthetic references; the region editor is
stubbed to accept what it is given, as a person pressing 使用这些分区 would.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QDialog

from starstack import regions
from aester_desktop import gui
from test_meteor_shower_entry import app, make, settle  # noqa: F401  (fixtures)
from test_p1_gui_routing_independent import scene


def _first_segment(window, monkeypatch, tmp_path):
    """Segment 1 was submitted with drawn ground (and a Milky Way); the window shows that job."""
    (tmp_path / "one").mkdir()
    project, assets = scene(window, monkeypatch, tmp_path / "one", mode="landscape", present=True)
    record = {"workflow_dir": str(tmp_path / "one" / "workflow-1"), "state": "succeeded",
              "config": {"regions_path": assets["project_path"], "trails_enabled": True,
                         "input_files": list(window.input_files)}}
    window._workflow_submitted(record)
    window.timer.stop()
    return project


def _second_segment(window, monkeypatch, tmp_path, *, width=32, height=24, rotation=0):
    """New photos of the same set-up: a fresh selection with its own first-frame reference."""
    (tmp_path / "two").mkdir()
    scene(window, monkeypatch, tmp_path / "two", mode="sky", present=False)
    window._regions_project = None
    window._regions_path = None
    window.sky_mask.clear()
    window._reference_data = dict(window._reference_data, width=width, height=height, image_rotation=rotation)
    window._job_detached = True
    window._update_controls()


def test_the_previous_ground_is_offered_and_opens_the_editor_with_it(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    first = _first_segment(window, monkeypatch, tmp_path)
    button = window.reuse_ground_button
    assert not button.isVisible()                       # the job's own regions are on view
    _second_segment(window, monkeypatch, tmp_path)
    assert button.isVisible() and button.isEnabled() and "银河需要重新圈定" in button.toolTip()
    shown = []

    class Editor:
        def __init__(self, reference, project, parent):
            shown.append(project)
            self.project = project

        def exec(self):
            return QDialog.DialogCode.Accepted
    monkeypatch.setattr("aester_desktop.region_dialog.RegionDialog", Editor)
    button.click()
    settle(window, app)
    assert len(shown) == 1
    offered = shown[0]
    assert offered["operations"] == first["operations"] and offered["scene_mode"] == "landscape"
    assert offered["milkyway"]["status"] == "unknown"           # the Milky Way moves: never carried over
    bound = [entry["path"] for entry in offered["source_binding"]["input_files"]]
    assert bound == [str(Path(path).resolve()) for path in window.input_files]
    assert window._regions_project is offered and window._regions_saved_by_user is offered
    assert window._scene_from_current_marks() == "landscape" and window.sky_mask.text()
    assert regions.validate_project(window._regions_project, input_files=window.input_files)
    assert not button.isVisible()                       # regions drawn for this segment now
    window.workflow_dir = None
    window.job_state = ""


def test_a_different_size_or_orientation_is_explained_not_reused(make, app, tmp_path, monkeypatch):
    window, _ = make("en")
    _first_segment(window, monkeypatch, tmp_path)
    _second_segment(window, monkeypatch, tmp_path, rotation=90)
    button = window.reuse_ground_button
    assert button.isVisible() and not button.isEnabled()
    assert button.toolTip().startswith("The previous segment’s photos differ in size or orientation")
    opened = []
    monkeypatch.setattr(window, "edit_regions", lambda **kwargs: opened.append(kwargs))
    window.reuse_previous_ground()
    assert opened == []
    window.workflow_dir = None
    window.job_state = ""


def test_nothing_is_offered_without_previous_ground(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    (tmp_path / "sky").mkdir()
    scene(window, monkeypatch, tmp_path / "sky", mode="sky", present=False)
    window._workflow_submitted({"workflow_dir": str(tmp_path / "sky" / "workflow-1"), "state": "succeeded",
                                "config": {"regions_path": window._regions_path, "input_files": list(window.input_files)}})
    window.timer.stop()
    assert window._previous_ground is None                  # a pure-sky job has no ground to carry over
    _second_segment(window, monkeypatch, tmp_path)
    assert not window.reuse_ground_button.isVisible()
    window.workflow_dir = None
    window.job_state = ""


def test_reuse_button_reads_in_both_languages(make, app):
    for language, text in (("zh_CN", "沿用上一段的地景分区"), ("en", "Reuse the previous segment’s foreground")):
        window, _ = make(language)
        assert window.reuse_ground_button.text() == text
        assert window._tab_chain.index(window.reuse_ground_button) == window._tab_chain.index(window.regions_button) + 1
