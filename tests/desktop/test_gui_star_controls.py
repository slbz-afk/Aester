"""Real Qt event-loop checks for explicit star controls and density suggestions."""
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from starstack import star_reduction
from aester_desktop import gui


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def drain(app, window):
    deadline = time.monotonic() + 8
    while window._workers or window._pool.activeThreadCount():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("star GUI background operation did not finish")
        time.sleep(.005)
    app.processEvents()


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    result = tmp_path / "separation.json"
    result.write_text("{}", encoding="utf-8")
    instance = gui.MainWindow(persist_preferences=False)
    instance.star_separation.setText(str(result))
    instance.output_dir.setText(str(tmp_path / "out"))
    yield instance
    drain(app, instance)
    instance.close()
    app.processEvents()


def suggestion(available=True, **changes):
    value = {"available": available, "background_strength": .81 if available else None,
             "milkyway_offset": -.17 if available else None, "milkyway_strength": .64 if available else None,
             "reason": "背景星点较密；建议先整理中弱星。"}
    value.update(changes)
    return {"state": "succeeded", "recommendation": value,
            "milkyway_gate": {"enabled": True}, "metrics": {}}


def render_record(tmp_path, gate=False):
    output = tmp_path / "star-result"
    output.mkdir(exist_ok=True)
    image = output / "star_reduced.jpg"
    Image.new("RGB", (40, 24), (22, 32, 62)).save(image)
    return {"state": "succeeded", "processing_directory": str(output),
            "artifacts": {"star_reduced.jpg": {"path": str(image)}},
            "milkyway_gate": {"enabled": gate}, "separation_result": str(tmp_path / "separation.json")}


def test_controls_cover_full_range_and_document_bright_transition(window):
    bg, offset = window.star_background_strength, window.star_milkyway_offset
    assert (bg.minimum(), bg.maximum(), bg.value()) == (0., 100., 55.)
    assert (offset.minimum(), offset.maximum(), offset.value()) == (-100., 100., -20.)
    assert "银河中心 35.0%" in window.star_strength_note.text()
    bg.setValue(100)
    offset.setValue(100)
    assert "银河中心 100.0%" in window.star_strength_note.text()
    assert "亮星也逐步减弱" in window.star_strength_note.text()
    assert "残晕仍可能存在" in window.star_strength_note.text()
    bg.setValue(0)
    offset.setValue(-100)
    assert "银河中心 0.0%" in window.star_strength_note.text()


def test_menu_reaches_star_controls(app, window):
    window.show()
    app.processEvents()
    window.menu_actions["stars"].trigger()
    app.processEvents()
    assert window._advanced_dialogs["stars"].isVisible()
    assert window.star_separation.hasFocus()


def test_density_analysis_uses_thread_and_leaves_event_loop_and_values_alive(app, window, monkeypatch):
    started, release = threading.Event(), threading.Event()
    worker_threads = []
    ticks = []
    before = (window.star_background_strength.value(), window.star_milkyway_offset.value())

    def analyze(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        started.set()
        assert release.wait(3)
        return suggestion()

    monkeypatch.setattr(star_reduction, "suggest_star_reduction", analyze, raising=False)
    window.star_analyze_button.click()
    assert started.wait(2)
    assert window._busy and not window.star_analyze_button.isEnabled()
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(1)
    deadline = time.monotonic() + .1
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.002)
    timer.stop()
    release.set()
    drain(app, window)
    assert ticks
    assert worker_threads != [threading.get_ident()]
    assert before == (window.star_background_strength.value(), window.star_milkyway_offset.value())
    assert window.star_apply_button.isEnabled()
    assert "尚未采用" in window.star_status.text()
    window.star_apply_button.click()
    assert window.star_background_strength.value() == 81
    assert window.star_milkyway_offset.value() == -17
    window.star_background_strength.setValue(72)
    assert window.star_background_strength.value() == 72


def test_unavailable_recommendation_does_not_supply_defaults(app, window, monkeypatch):
    monkeypatch.setattr(star_reduction, "suggest_star_reduction", lambda *a, **k: suggestion(False), raising=False)
    window.analyze_star_density()
    drain(app, window)
    assert not window.star_apply_button.isEnabled()
    assert "未给出可靠强度建议" in window.star_status.text()
    assert window.star_background_strength.value() == 55


@pytest.mark.parametrize("name", ["star_separation", "star_milkyway_mask", "star_milkyway_evidence"])
def test_changing_any_input_invalidates_old_suggestion(app, window, monkeypatch, name):
    monkeypatch.setattr(star_reduction, "suggest_star_reduction", lambda *a, **k: suggestion(), raising=False)
    window.analyze_star_density()
    drain(app, window)
    assert window.star_apply_button.isEnabled()
    getattr(window, name).setText("changed-input")
    assert window._star_suggestion is None
    assert not window.star_apply_button.isEnabled()
    window.apply_star_suggestion()
    assert window.star_background_strength.value() == 55


@pytest.mark.parametrize("background,offset,quality", [(0, -100, 95), (100, 100, 60), (74.25, -18.5, 88)])
def test_render_captures_percentage_values_before_background_dispatch(window, monkeypatch, tmp_path, background,
                                                                       offset, quality):
    pending, calls = [], []
    monkeypatch.setattr(window, "_run_async", lambda *args: pending.append(args))
    monkeypatch.setattr(star_reduction, "reduce_stars", lambda *args, **kwargs: calls.append((args, kwargs)))
    window.star_background_strength.setValue(background)
    window.star_milkyway_offset.setValue(offset)
    window.jpeg_quality_slider.setValue(quality)
    window.render_star_reduction()
    assert window._busy
    window.star_background_strength.setValue(12)
    window.star_milkyway_offset.setValue(8)
    window.jpeg_quality_slider.setValue(70)
    pending[0][1]()
    # The GUI always sends its JPEG quality, read when the button was pressed.
    assert calls[0] == ((str(tmp_path / "separation.json"), str(tmp_path / "out")),
                        {"background_strength": background / 100, "milkyway_offset": offset / 100,
                         "milkyway_mask": None, "milkyway_evidence": None, "jpeg_quality": quality})
    window._busy = False


@pytest.mark.parametrize("gate", [False, True])
def test_render_displays_new_result_without_replacing_existing_job(app, window, monkeypatch, tmp_path, gate):
    record = render_record(tmp_path, gate)
    monkeypatch.setattr(star_reduction, "reduce_stars", lambda *a, **k: record)
    job = str(tmp_path / "previous-job")
    window.job_dir, window.job_state = job, "succeeded"
    old_preview = {"job_dir": job, "gains": (1, 1, 0), "directory": job, "jpgs": []}
    old_reference = {"unchanged": True}
    window._rendered_preview = old_preview
    window._reference_data = old_reference
    window._jpg_artifacts = []
    window._update_controls()
    window.star_render_button.click()
    drain(app, window)
    assert window._view_mode == "stars" and not window.preview._pixmap.isNull()
    assert window.job_dir == job and window.job_state == "succeeded"
    assert window._rendered_preview is old_preview and window._reference_data is old_reference
    assert "银河中心 35.0%" in window.preview_note.text() if gate else "银河未确认" in window.preview_note.text()
    assert window.stars_view_button.isChecked()
    assert not window.style_tabs.isEnabled()
    opened = []
    monkeypatch.setattr(gui.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    window.star_open_button.click()
    assert Path(opened[0]) == Path(record["processing_directory"])
    window.show_results()
    assert window._view_mode == "results"
    window.stars_view_button.click()
    drain(app, window)
    assert window._view_mode == "stars"


def test_full_endpoint_limit_is_visible_to_user(app, window, monkeypatch, tmp_path):
    record = render_record(tmp_path)
    record.update(full_removal_limited_pixels=12, full_removal_requested_interior_pixels=345)
    monkeypatch.setattr(star_reduction, "reduce_stars", lambda *a, **k: record)
    window.star_background_strength.setValue(100)
    window.render_star_reduction()
    drain(app, window)
    assert "12 / 345" in window.star_status.text()
    assert "未完全达到无星背景" in window.star_status.text()


@pytest.mark.parametrize("operation", ["analyze_star_density", "render_star_reduction"])
def test_operation_exception_releases_controls_and_keeps_prior_result(app, window, monkeypatch, operation):
    def fail(*args, **kwargs):
        raise ValueError("分层来源已改变")
    target = "suggest_star_reduction" if operation == "analyze_star_density" else "reduce_stars"
    monkeypatch.setattr(star_reduction, target, fail, raising=False)
    getattr(window, operation)()
    drain(app, window)
    assert not window._busy
    assert window.star_analyze_button.isEnabled() and window.star_render_button.isEnabled()
    assert "分层来源已改变" in window.star_status.text()
    assert window._star_result is None


@pytest.mark.parametrize("changes", [{"background_strength": float("nan")}, {"background_strength": True},
                                      {"background_strength": 1.01}, {"milkyway_offset": -1.01},
                                      {"milkyway_offset": None}])
def test_malformed_suggestion_fails_without_overwriting_controls(app, window, monkeypatch, changes):
    monkeypatch.setattr(star_reduction, "suggest_star_reduction", lambda *a, **k: suggestion(**changes), raising=False)
    window.analyze_star_density()
    drain(app, window)
    assert not window._busy and not window.star_apply_button.isEnabled()
    assert window.star_background_strength.value() == 55
    assert "参数超出" in window.star_status.text()


@pytest.mark.parametrize("state", ["queued", "running", "cancel_requested"])
def test_active_jobs_disable_new_operations_and_inputs(window, monkeypatch, state):
    window.job_state = state
    window._update_controls()
    monkeypatch.setattr(window, "_run_async", lambda *a: pytest.fail("cannot dispatch"))
    assert not window.star_render_button.isEnabled()
    assert not window.star_separation.isEnabled()
    assert not window.star_background_strength.isEnabled()
    assert all(not button.isEnabled() for button in window._star_path_buttons)
    window.analyze_star_density()
    window.render_star_reduction()


def test_evidence_without_mask_is_rejected_before_dispatch(window, monkeypatch, tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    window.star_milkyway_evidence.setText(str(evidence))
    monkeypatch.setattr(window, "_run_async", lambda *a: pytest.fail("cannot dispatch"))
    window.render_star_reduction()
    assert "对应的蒙版" in window.star_status.text()
    assert not window._busy


def test_programmatic_input_change_during_analysis_discards_stale_result(app, window, monkeypatch):
    pending = []
    monkeypatch.setattr(window, "_run_async", lambda *args: pending.append(args))
    window.analyze_star_density()
    window.star_separation.setText("different.json")
    pending[0][2](suggestion())
    assert window._star_suggestion is None
    assert not window.star_apply_button.isEnabled() and not window._busy
    assert "旧建议未采用" in window.star_status.text()


def test_closing_waits_for_local_operation_and_discards_callback(app, window, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def analyze(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return suggestion()
    monkeypatch.setattr(star_reduction, "suggest_star_reduction", analyze, raising=False)
    window.show()
    window.analyze_star_density()
    assert started.wait(2)
    window.close()
    assert window._closing_requested and window.isVisible()
    assert not window.centralWidget().isEnabled()
    release.set()
    drain(app, window)
    window._close_after_workers()
    assert not window.isVisible()
    assert window._star_suggestion is None


def test_picker_cancel_preserves_selected_input(window, monkeypatch):
    before = window.star_separation.text()
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *a: ("", ""))
    window._star_path_buttons[0].click()
    assert window.star_separation.text() == before
