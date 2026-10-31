"""「拓星」 main window (docs/UI_REDESIGN_IA_20260927.md §2–§4, acceptance U3/U4/U6/U8).

Real Qt widgets offscreen: layout at the two window sizes × language × theme/night, stage-bar
states derived from existing state, viewing conditions that never touch photo data, the
usability fixes (Enter in dialogs, tab order, regenerate reference, completion alert) and the
new preferences written only on change or close.
"""
import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, QSettings, Qt
from PySide6.QtGui import QFont, QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QPushButton, QWidget

from starstack import i18n, uninstall_options
from aester_desktop import fonts, gui, themes


@pytest.fixture(scope="module")
def app():
    value = QApplication.instance() or QApplication([])
    fonts.register_bundled_fonts()
    return value


@pytest.fixture
def make(app, tmp_path, monkeypatch):
    monkeypatch.delenv("AESTER_LANGUAGE", raising=False)
    monkeypatch.delenv("AESTER_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    windows = []

    def create(language="zh_CN", settings=None, values=None):
        prefs = settings or QSettings(str(tmp_path / f"prefs-{len(windows)}.ini"), QSettings.Format.IniFormat)
        if settings is None:
            prefs.setValue("appearance/language", language)
            for key, value in (values or {}).items():
                prefs.setValue(key, value)
            prefs.sync()
        window = gui.MainWindow(preferences=prefs)
        window.cache_dir.setText(str(tmp_path / "cache"))
        window.output_dir.setText(str(tmp_path / "photos"))
        windows.append(window)
        window.show()
        app.processEvents()
        return window, prefs

    yield create
    for window in windows:
        window._busy = False
        window.job_state = ""
        window._color_timer.stop()
        window._color_finish_timer.stop()
        window.timer.stop()
        for dialog in window._advanced_dialogs.values():
            dialog.close()
        window.close()
    app.processEvents()
    i18n.set_language("zh_CN")


def _rect(window, widget):
    top_left = widget.mapTo(window.centralWidget(), QPoint(0, 0))
    return QRect(top_left, widget.size())


def _image(width=90, height=60):
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[..., 0] = np.arange(width, dtype=np.uint8)[None, :] * 2
    pixels[..., 1] = 80
    pixels[20:40, 25:45, 2] = 240
    return QImage(pixels.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


def _rgba(image):
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.width(), 4).copy()


def _ready_reference(window, tmp_path, count=3, look=None):
    files = []
    for index in range(count):
        path = tmp_path / f"frame{index}.nef"
        path.write_bytes(b"synthetic, never decoded")
        files.append(str(path))
    window.input_files = files
    window._reference_data = {"reference_dir": str(tmp_path / "reference"), "image_rotation": 0,
                              "width": 90, "height": 60, "preview_path": str(tmp_path / "missing.png")}
    if look is not None:
        window._reference_data["color_cache"] = look
    window._reference_scene = tuple(files)
    window._image_rotation = 0
    window._reference_pending = False
    window.input_dir.setText(str(tmp_path))
    window._update_controls()
    return files


# ------------------------------------------------------------------------------------ U4 layout

@pytest.mark.parametrize("language", ["zh_CN", "en"])
@pytest.mark.parametrize("mode", ["dark", "light", "night"])
@pytest.mark.parametrize("size", [(960, 680), (1200, 800)])
def test_layout_required_widgets_visible_inside_and_not_overlapping(make, app, language, mode, size):
    window, _ = make(language)
    window.set_theme("light" if mode == "light" else "dark", persist=False)
    window.set_night(mode == "night", persist=False)
    window.resize(*size)
    app.processEvents()
    area = window.centralWidget().rect()
    required = [
        *window.stage_bar.nodes(), window.retouch_button,
        window.selection_label, window.add_files_button, window.remove_files_button, window.clear_files_button,
        window.view_switch, window.open_button, window.preview, window.preview_note,
        window.rotate_left_button, window.rotate_right_button, window.restore_orientation_button,
        window.surround_grey_button, window.night_button, window.file_strip,
        window.output_dir, window.jpeg_quality_slider, window.jpeg_quality_value,
        window.start_button, window.cancel_button,
        window.stage_label, window.progress_bar, window.percent_label, window.message_label, window.details_button,
    ]
    rects = []
    for widget in required:
        rect = _rect(window, widget)
        assert widget.isVisible(), widget.objectName()
        assert area.contains(rect.topLeft()) and area.contains(rect.bottomRight()), (widget.objectName(), rect, area)
        rects.append((widget, rect))
    for index, (first, a) in enumerate(rects):
        for second, b in rects[index + 1:]:
            assert not a.intersects(b), (first.objectName() or first.text(), second.objectName() or second.text())
    # Buttons are never clipped below their size hint.
    for button in (window.add_files_button, window.remove_files_button, window.clear_files_button, window.open_button,
                   window.rotate_left_button, window.rotate_right_button, window.restore_orientation_button,
                   window.surround_grey_button, window.night_button, window.retouch_button, window.details_button):
        assert button.width() >= button.sizeHint().width(), button.text()
    assert window.preview.width() >= 500 and window.preview.height() >= 220
    # The step column keeps its width budget; the first step is inside the scroll viewport.
    assert window.centralWidget().findChild(QWidget, "controlPanel").width() == (352 if language == "en" else 302)
    viewport = window._steps_scroll.viewport()
    top = window.regions_button.mapTo(viewport, QPoint(0, 0))
    assert viewport.rect().contains(top)
    assert window.warmth_slider.isVisible() and window.tint_slider.isVisible() and window.exposure_slider.isVisible()


def test_right_column_steps_are_in_working_order(make, app):
    window, _ = make("zh_CN")
    order = [window.regions_button, window.warmth_slider, window.tint_slider, window.exposure_slider,
             window.trails_enabled, window.stars_enabled, window.milkyway_enabled]
    content = window._steps_scroll.widget()
    tops = [widget.mapTo(content, QPoint(0, 0)).y() for widget in order]
    assert tops == sorted(tops)
    panel = window.centralWidget().findChild(QWidget, "controlPanel")
    fixed = [window.output_dir, window.start_button, window.trail_review_button]
    assert all(widget.parentWidget() is panel or panel.isAncestorOf(widget) for widget in fixed)
    assert not window._steps_scroll.widget().isAncestorOf(window.start_button)
    assert window.start_button.text() == "开始合成" and themes.role(window.start_button) == "primary"


# ------------------------------------------------------------------------------ stage states

def test_stage_states_idle_reference_running_succeeded(make, app, tmp_path):
    window, _ = make("zh_CN")
    bar = window.stage_bar
    assert bar.keys() == ["materials", "reference", "regions", "compose", "results"]
    assert bar.states() == {"materials": "current", "reference": "disabled", "regions": "disabled",
                            "compose": "todo", "results": "disabled"}
    assert bar.node("materials").isEnabled() and not bar.node("results").isEnabled()
    assert bar.node("reference").toolTip()

    _ready_reference(window, tmp_path)
    assert bar.states() == {"materials": "done", "reference": "done", "regions": "current",
                            "compose": "todo", "results": "disabled"}
    assert bar.connector_lit() == [True, True, False, False]
    assert bar.node("regions").isEnabled() and bar.node("reference").isEnabled()

    window.workflow_dir = str(tmp_path)
    window.job_state = "running"
    window._update_controls()
    assert bar.states()["compose"] == "current"
    assert [bar.state(key) for key in ("materials", "reference", "regions")] == ["done"] * 3
    # Reached stages keep their mark but cannot be used while the job runs.
    assert not bar.node("materials").isEnabled() and not bar.node("regions").isEnabled()
    assert bar.node("materials").toolTip() == "正在处理，完成后可用。"

    window.job_state = "succeeded"
    window._jpg_artifacts = [{"path": str(tmp_path / "final.jpg"), "label": "final"}]
    window._update_controls()
    assert bar.states()["results"] == "current" and bar.node("results").isEnabled()
    shown = []
    window.show_results = lambda: shown.append(True)
    bar.node("results").click()
    assert shown == [True]
    window.job_state = ""
    window.workflow_dir = None


def test_stage_02_regenerates_a_reference_cleared_after_a_run(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path)
    window._reference_data = None      # _workflow_submitted clears the reference
    window._reference_scene = ()
    window.workflow_dir = str(tmp_path)
    window.job_state = "succeeded"
    window._update_controls()
    node = window.stage_bar.node("reference")
    assert node.isEnabled() and "重新生成" in node.toolTip()
    assert window.reference_button.isVisible() and window.reference_button.isEnabled()
    calls = []
    monkeypatch.setattr(window, "prepare_reference", lambda: calls.append("stage"))
    node.click()
    assert calls == ["stage"]
    window.job_state = ""
    window.workflow_dir = None


def test_stage_click_routes_to_existing_handlers(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    calls = []
    monkeypatch.setattr(window, "_pick_input", lambda: calls.append("materials"))
    monkeypatch.setattr(window, "edit_regions", lambda: calls.append("regions"))
    window.stage_bar.node("materials").click()
    window.stage_bar.node("regions").click()        # disabled: no reference yet
    _ready_reference(window, tmp_path)
    window.stage_bar.node("regions").click()
    assert calls == ["materials", "regions"]
    window.stage_bar.node("compose").click()
    assert window.retouch_button.isEnabled()
    monkeypatch.setattr(window.experience, "show_retouch", lambda: calls.append("retouch"))
    window.retouch_button.click()
    assert calls[-1] == "retouch"
    assert window.retouch_button.objectName() == "retouchAgentButton"
    assert window.retouch_button.text() == "让 Agent 帮你修"
    assert window.stage_bar.trailing_widget() is window.retouch_button
    assert window.stage_bar.trailing_dot().kind() == "agent"


def test_recipe_tile_reads_the_real_camera_look(make, app, tmp_path):
    window, _ = make("zh_CN")
    assert not window.recipe_label.isVisible()
    look = {"base_params": {"look": "camera_hsm", "look_id": "sony-ilce7m4-rt-cc0-d50", "look_strength": 0.5},
            "master_metadata": {"colour_basis": {"reference": {"camera": {"make": "SONY", "model": "ILCE-7RM5"}}}}}
    _ready_reference(window, tmp_path, look=look)
    assert window.recipe_label.isVisible()
    assert window.recipe_label.text() == "相机观感 · α7R V · 借 α7 IV 表 · 0.50"
    window._reference_data = dict(window._reference_data, reference_dir=str(tmp_path / "other"),
                                  color_cache={"base_params": {"look": "none"}})
    window._update_controls()
    assert window.recipe_label.text() == "相机观感 · 无"


def test_regions_seal_only_for_a_human_save(make, app, tmp_path):
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path)
    assert not window.regions_seal.isVisible() and not window.regions_provenance.isVisible()
    project = {"operations": [{"kind": "paint", "target": "ground"}], "milkyway": {"status": "unknown"}}
    window._regions_project, window._regions_path = project, str(tmp_path / "regions.json")
    window._update_controls()
    assert not window.regions_seal.isVisible()          # origin unknown: no seal
    window._regions_saved_by_user = project
    window._update_controls()
    assert window.regions_seal.isVisible() and window.regions_provenance.kind() == "human"
    agent = {"operations": [], "milkyway": {"status": "present", "confirmed": True, "ellipse": [.5, .5, .2, .1, 0],
                                            "authority": "agent", "selection": "explicit_drawing"}}
    from starstack import regions
    agent["milkyway"]["selection"] = regions.DRAWING
    window._regions_project = agent
    window._update_controls()
    assert window.regions_provenance.isVisible() and window.regions_provenance.kind() == "agent"
    assert not window.regions_seal.isVisible()


# ---------------------------------------------------------------- U3 viewing conditions (display only)

def test_surround_and_night_never_change_stored_image_data(make, app):
    window, _ = make("zh_CN")
    preview = window.preview
    preview.resize(400, 300)
    preview.set_image(_image())
    app.processEvents()
    source = _rgba(preview._pixmap.toImage())
    before = _rgba(preview.grab().toImage())
    centre = before[150, 200].copy()
    assert tuple(before[2, 2, :3]) == (0x1f, 0x1f, 0x1f)

    window.set_surround("mid_grey", persist=False)
    app.processEvents()
    grey = _rgba(preview.grab().toImage())
    assert tuple(grey[2, 2, :3]) == (0x77, 0x77, 0x77)
    assert np.array_equal(grey[150, 200], centre)
    assert np.array_equal(_rgba(preview._pixmap.toImage()), source)
    window.set_surround("black", persist=False)
    app.processEvents()
    assert tuple(_rgba(preview.grab().toImage())[2, 2, :3]) == (0, 0, 0)
    window.set_surround("xuan", persist=False)

    window.set_night(True, persist=False)
    app.processEvents()
    night = _rgba(preview.grab().toImage())
    tag = window.preview.night_tag
    assert tag.isVisible() and tag.text() == "不可判色"
    mask = np.ones(night.shape[:2], bool)
    geometry = tag.geometry()
    mask[geometry.top():geometry.bottom() + 1, geometry.left():geometry.right() + 1] = False
    assert np.all(night[mask][:, 1] == 0) and np.all(night[mask][:, 2] == 0)   # red single channel
    assert night[150, 200, 0] > 0
    assert np.array_equal(_rgba(preview._pixmap.toImage()), source)             # stored pixels unchanged
    # Theme switches under night vision still never touch the preview.
    window.set_theme("light", persist=False)
    app.processEvents()
    assert np.array_equal(_rgba(preview.grab().toImage()), night)
    window.set_night(False, persist=False)
    window.set_theme("dark", persist=False)
    app.processEvents()
    assert np.array_equal(_rgba(preview.grab().toImage()), before)
    assert not tag.isVisible()
    assert "night" not in window._editing_state() and "surround" not in window._editing_state()


def test_night_mode_retints_window_and_dialogs_red_only(make, app):
    window, _ = make("zh_CN")
    window.menu_actions["settings"].trigger()
    window.menu_actions["night"].trigger()
    app.processEvents()
    assert window._night and window.night_button.isChecked() and window.menu_actions["night"].isChecked()
    assert window.property("aesterNight") is True
    dialog = window._advanced_dialogs["settings"]
    assert dialog.styleSheet() == ""
    grabbed = _rgba(window.grab().toImage())[..., :3].astype(int)
    assert np.all(grabbed[..., 1] == grabbed[..., 2]) and np.all(grabbed[..., 1] <= grabbed[..., 0])
    assert window.menu_actions["night"].shortcut().toString() == "Ctrl+Alt+N"
    window.night_button.click()
    assert not window._night and not window.menu_actions["night"].isChecked()


def test_appearance_menu_has_night_and_surround_without_a_fourth_menu(make, app):
    window, _ = make("zh_CN")
    assert [m.text() for m in window.menuBar().actions()] == ["文件", "设置", "帮助"]
    for key in ("night", "surround_xuan", "surround_grey", "surround_black", "theme_dark", "theme_light"):
        assert window.menu_actions[key].isCheckable()
    assert window.menu_actions["surround_xuan"].isChecked()
    window.menu_actions["surround_grey"].trigger()
    assert window._surround == "mid_grey" and window.surround_grey_button.isChecked()
    window.surround_grey_button.click()
    assert window._surround == "xuan" and window.menu_actions["surround_xuan"].isChecked()


# ----------------------------------------------------------------------- preferences (U7 cleanup)

def test_new_preferences_are_read_but_never_written_during_construction(make, app, tmp_path):
    def snapshot(settings):
        settings.sync()
        return {key: settings.value(key) for key in settings.allKeys()}

    prefs = QSettings(str(tmp_path / "viewing.ini"), QSettings.Format.IniFormat)
    for key, value in {"appearance/language": "zh_CN", "appearance/night": "true", "appearance/surround": "grey",
                       "output_directory": str(tmp_path / "works"), "storage/cache_root": str(tmp_path / "cache")}.items():
        prefs.setValue(key, value)
    before = snapshot(prefs)
    window, _ = make(settings=prefs)
    assert window._night and window.preview.is_night() and window.night_button.isChecked()
    assert window._surround == "mid_grey" and window.preview.surround() == "mid_grey"
    assert snapshot(prefs) == before
    window.set_night(False)
    assert str(prefs.value("appearance/night")).lower() == "false"
    window.menu_actions["surround_black"].trigger()
    assert prefs.value("appearance/surround") == "black"
    window.close()
    app.processEvents()
    reopened, _ = make(settings=prefs)
    assert not reopened._night and reopened._surround == "black"


def test_fresh_window_writes_no_viewing_preferences_until_close(make, app, tmp_path):
    prefs = QSettings(str(tmp_path / "fresh.ini"), QSettings.Format.IniFormat)
    prefs.setValue("appearance/language", "en")
    prefs.sync()
    window, _ = make(settings=prefs)
    prefs.sync()
    assert prefs.value("appearance/night") is None and prefs.value("appearance/surround") is None
    window.close()
    app.processEvents()
    prefs.sync()
    assert prefs.value("appearance/surround") == "xuan"
    assert str(prefs.value("appearance/night")).lower() == "false"


def test_uninstall_cleanup_owns_the_new_preference_keys():
    for key in ("appearance/surround", "appearance/night", "retouch/gimp_path", "output/jpeg_quality"):
        assert key in uninstall_options.SETTINGS_KEYS


# ------------------------------------------------------------------------------ U6 usability

def test_enter_in_siril_path_neither_opens_help_nor_clears_regions(make, app, tmp_path):
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path)
    project = {"operations": [{"kind": "paint", "target": "ground"}], "milkyway": {"status": "unknown"}}
    window._regions_project, window._regions_path = project, str(tmp_path / "regions.json")
    window.sky_mask.setText(str(tmp_path / "sky.png"))
    window.menu_actions["settings"].trigger()
    app.processEvents()
    dialog = window._advanced_dialogs["settings"]
    assert not [button.text() for button in dialog.findChildren(QPushButton) if button.isDefault() or button.autoDefault()]
    window.siril_path.setFocus()
    QTest.keyClick(window.siril_path, Qt.Key.Key_Return)
    app.processEvents()
    assert "help" not in window._advanced_dialogs or not window._advanced_dialogs["help"].isVisible()
    assert window._regions_project is project and window.sky_mask.text()
    assert dialog.isVisible()
    # A real edit still invalidates the reference exactly as before.
    QTest.keyClicks(window.siril_path, "x")
    QTest.keyClick(window.siril_path, Qt.Key.Key_Return)
    app.processEvents()
    assert window._regions_project is None and window.sky_mask.text() == ""
    assert window._reference_pending
    window._color_timer.stop()


def test_every_setup_dialog_has_no_enter_default_button(make, app):
    window, _ = make("zh_CN")
    window.experience.show_storage()
    window.experience.show_connection()
    window.experience.show_retouch()
    app.processEvents()
    for key in ("storage", "connect_ai", "retouch"):
        dialog = window._advanced_dialogs[key]
        assert not any(button.isDefault() or button.autoDefault() for button in dialog.findChildren(QPushButton)), key


def test_tab_order_follows_the_working_flow(make, app, tmp_path):
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path)
    window.trails_enabled.setChecked(False)
    expected = [window.add_files_button, window.clear_files_button, window.file_strip, window.results_view_button,
                window.rotate_left_button, window.restore_orientation_button, window.surround_grey_button,
                window.night_button, window.regions_button, window.reset_color_button, window.warmth_slider,
                window.tint_slider, window.exposure_slider, window.trails_enabled, window.stars_enabled,
                window.milkyway_strength, window.output_dir, window.jpeg_quality_slider, window.jpeg_quality_value,
                window.start_button, window.cancel_button,
                window.details_button, window.stage_bar.node("materials"), window.retouch_button]
    seen, widget = [], window.add_files_button
    for _ in range(400):
        if widget in expected and widget not in seen:
            seen.append(widget)
        widget = widget.nextInFocusChain()
        if widget is window.add_files_button:
            break
    assert seen == expected
    assert window._steps_scroll.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_completion_alerts_the_taskbar_only_when_in_background(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    alerts = []
    monkeypatch.setattr(gui.QApplication, "alert", staticmethod(lambda widget, duration=0: alerts.append(widget)))
    monkeypatch.setattr(window, "isActiveWindow", lambda: False)
    monkeypatch.setattr("starstack.workflow.workflow_status",
                        lambda *_: {"state": "succeeded", "progress": 100, "artifacts": []})
    window.workflow_dir = str(tmp_path)
    window.timer.start()
    window.refresh_workflow()
    assert alerts == [window] and not window.timer.isActive()
    window.refresh_workflow()           # already finished: no second alert
    assert alerts == [window]
    monkeypatch.setattr(window, "isActiveWindow", lambda: True)
    window.timer.start()
    window.refresh_workflow()
    assert alerts == [window]
    window.job_state = ""
    window.workflow_dir = None


def test_readout_error_keeps_full_text_and_adds_icon(make, app):
    window, _ = make("zh_CN")
    message = "后台故障：" + "很长的诊断 " * 30
    window._handle_error("submit", message)
    assert window.message_label.text() == "操作未完成：" + message
    assert window.message_label.is_error() and themes.role(window.message_label) == "error"
    assert window.error_icon.isVisible()
    window.message_label.setText("首帧预览已更新")
    assert not window.message_label.is_error() and not window.error_icon.isVisible()
    assert themes.role(window.percent_label) == "mono"


def test_colour_sliders_are_tick_sliders_with_reset_and_delta(make, app):
    # 2026-09-27: the read-out label became a typed value field (Δ mark + mono field); the
    # slider stays the source of the value and keeps its double-click reset.
    from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox
    from aester_desktop import ui_widgets
    window, _ = make("zh_CN")
    for slider in (window.warmth_slider, window.tint_slider, window.exposure_slider):
        assert isinstance(slider, ui_widgets.TickSlider) and slider.is_bipolar() and slider.default_value() == 0
        assert slider.accessibleName()
        assert slider.focusPolicy() == Qt.FocusPolicy.StrongFocus     # no wheel focus
    assert (window.warmth_slider.minimum(), window.warmth_slider.maximum()) == (-100, 100)
    assert (window.exposure_slider.minimum(), window.exposure_slider.maximum()) == (-40, 40)
    assert window.findChild(QLabel, "warmthReading") is None
    field = window.findChild(QDoubleSpinBox, "warmthValue")
    delta = window.findChild(QLabel, "warmthDelta")
    assert field is window.warmth_value and isinstance(field, ui_widgets.ValueField)
    assert field.text() == "0" and themes.role(field) == "value" and not delta.isVisible()
    assert delta.text() == "Δ" and themes.role(delta) == "mono"
    assert field.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons and not field.keyboardTracking()
    assert field.alignment() & Qt.AlignmentFlag.AlignRight
    assert (field.decimals(), field.minimum(), field.maximum(), field.suffix()) == (0, -100, 100, "")
    assert (window.tint_value.decimals(), window.tint_value.minimum(), window.tint_value.maximum()) == (0, -100, 100)
    exposure = window.exposure_value
    assert (exposure.decimals(), exposure.minimum(), exposure.maximum(), exposure.suffix()) == (1, -4.0, 4.0, " EV")
    assert exposure.singleStep() == pytest.approx(0.1)
    window._restoring_project = True
    window.warmth_slider.setValue(12)
    assert field.text() == "+12" and delta.isVisible() and field.property("changed") is True
    assert field.lineEdit().font().weight() == QFont.Weight.Medium      # what moved is set in 500
    QTest.mouseDClick(window.warmth_slider, Qt.MouseButton.LeftButton)
    assert window.warmth_slider.value() == 0 and field.text() == "0" and not delta.isVisible()
    assert field.property("changed") is False and field.lineEdit().font().weight() == QFont.Weight.Normal
    window.exposure_slider.setValue(-7)
    assert exposure.text() == "-0.7 EV"
    window.reset_color_adjustments()
    assert exposure.text() == "0.0 EV"
    window._restoring_project = False
    assert window._color_timer.interval() == 90 and window._color_finish_timer.interval() == 600


def test_welcome_closed_in_any_way_counts_as_seen(make, app, tmp_path):
    window, prefs = make("zh_CN")
    window.experience.show_welcome()
    app.processEvents()
    assert str(prefs.value("experience/welcome_seen", "0")) != "1"
    window._advanced_dialogs["welcome"].close()
    app.processEvents()
    assert str(prefs.value("experience/welcome_seen")) == "1"


def test_retouch_dialog_explains_the_agent_loop_and_marks_human_words(make, app):
    window, _ = make("zh_CN")
    window.retouch_button.click()
    app.processEvents()
    exp = window.experience
    assert "主 MCP" in exp.retouch_loop.text() and "作品目录" in exp.retouch_loop.text()
    assert exp.retouch_provenance.kind() == "agent"
    assert themes.role(exp.retouch_intent) == "human"
    assert themes.role(exp.retouch_prompt) == "mono"


# ------------------------------------------------------------------------------------- U8 i18n

def test_english_with_night_on_has_no_chinese_residue(make, app, tmp_path):
    window, _ = make("en")
    window.set_night(True, persist=False)
    _ready_reference(window, tmp_path, look={"base_params": {"look": "none"}})
    window._handle_error("submit", "backend said no")
    window.experience.show_retouch()
    app.processEvents()
    assert window.preview.night_tag.text() == "Not for colour judgement"
    leftovers = []
    for obj in window.findChildren(QLabel) + window.findChildren(QAbstractButton) + window.findChildren(QLineEdit):
        for text in (obj.text(), obj.toolTip(), obj.accessibleName(), obj.accessibleDescription()):
            if re.search("[一-龥]", text):
                leftovers.append((type(obj).__name__, obj.objectName(), text))
    for obj in window.findChildren(QComboBox):
        leftovers += [(obj.objectName(), obj.itemText(i)) for i in range(obj.count()) if re.search("[一-龥]", obj.itemText(i))]
    for key, action in window.menu_actions.items():
        if re.search("[一-龥]", action.text()) and key not in ("language_zh_CN",):
            leftovers.append((key, action.text()))
    assert leftovers == []
    assert window.stage_bar.node("reference").text() == "First frame"


# ------------------------------------------------------------------- 2026-09-27 review fixes

def test_stage_bar_follows_a_new_selection_after_a_finished_job(make, app, tmp_path):
    window, _ = make("zh_CN")
    files = _ready_reference(window, tmp_path)
    window.workflow_dir = str(tmp_path)
    window.job_state = "succeeded"
    window._jpg_artifacts = [{"path": str(tmp_path / "final.jpg"), "label": "final"}]
    window._update_controls()
    assert window.stage_bar.states()["results"] == "current"
    window.clear_input_files()
    assert window._current_stage() == "materials"
    assert window.stage_bar.states()["materials"] == "current"
    new = []
    for index in range(3):
        path = tmp_path / f"new{index}.nef"
        path.write_bytes(b"synthetic")
        new.append(str(path))
    window.input_files = new
    window._selection_changed()
    assert window._current_stage() == "reference"
    window._color_timer.stop()
    # A failed job of the *current* selection keeps 04 current and offers "Resume job".
    window.input_files = files
    window._job_detached = False
    window.job_state = "failed"
    window._update_controls()
    assert window._current_stage() == "compose" and window.resume_button.isVisible()
    window._job_detached = True
    window._update_controls()
    assert not window.resume_button.isVisible()
    window.job_state = ""
    window.workflow_dir = None


def test_start_button_and_compose_stage_say_why_start_is_unavailable(make, app, tmp_path):
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path, count=2)
    assert not window.start_button.isEnabled()
    assert window.start_button.toolTip() == "至少需要 3 张照片（当前 2 张）。"
    assert "至少需要 3 张照片" in window.stage_bar.node("compose").toolTip()
    _ready_reference(window, tmp_path, count=3)
    assert window.start_button.isEnabled() and window.start_button.toolTip() == ""
    window._reference_data = None
    window._reference_scene = ()
    window._update_controls()
    assert "首帧" in window.start_button.toolTip()
    window.workflow_dir = str(tmp_path)
    window.job_state = "running"
    window._update_controls()
    assert not window.stage_bar.node("compose").isEnabled()
    assert window.stage_bar.node("compose").toolTip() == "正在处理，完成后可用。"
    window.job_state = ""
    window.workflow_dir = None


def test_failed_reference_preview_tooltip_offers_a_retry(make, app, tmp_path):
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path)
    window._reference_data = None
    window._reference_scene = ()
    window._handle_error(f"reference-build:{window._reference_generation}", "decode failed")
    assert window._reference_failed
    assert window.stage_bar.node("reference").toolTip() == "首帧预览未完成，点此重试。"


def test_night_tag_never_covers_the_photograph(make, app):
    window, _ = make("zh_CN")
    preview = window.preview
    preview.resize(420, 280)
    preview.set_image(_image(900, 600))
    window.set_night(True, persist=False)
    app.processEvents()
    assert preview.night_tag.isVisible()
    assert not preview.night_tag.geometry().intersects(preview.photo_rect())
    window.set_night(False, persist=False)
    assert preview.photo_rect().top() >= 18


def test_preview_messages_readable_on_every_surround(make, app):
    from test_ui_foundation import _contrast
    window, _ = make("zh_CN")
    for surround, background in themes.SURROUNDS.items():
        window.set_surround(surround, persist=False)
        assert _contrast(window.preview._message_color().name(), background) >= 4.5, surround
    window.set_surround("xuan", persist=False)


def test_grey_toggle_returns_to_the_previous_surround(make, app):
    window, _ = make("zh_CN")
    window.set_surround("black", persist=False)
    window.surround_grey_button.click()
    assert window._surround == "mid_grey"
    window.surround_grey_button.click()
    assert window._surround == "black" and window.menu_actions["surround_black"].isChecked()
    window.set_surround("xuan", persist=False)


def test_camera_look_chip_keeps_body_and_table_in_narrow_english(make, app, tmp_path):
    window, _ = make("en")
    window.resize(960, 680)
    look = {"base_params": {"look": "camera_hsm", "look_id": "sony-ilce7m4-rt-cc0-d50", "look_strength": 0.5},
            "master_metadata": {"colour_basis": {"reference": {"camera": {"make": "SONY", "model": "ILCE-7RM5"}}}}}
    _ready_reference(window, tmp_path, look=look)
    app.processEvents()
    chip = window.recipe_label
    full = chip.text()
    assert full.startswith("Camera look · α7R V")
    assert chip.accessibleName() == full
    shown = chip.shown_text(chip.fontMetrics().horizontalAdvance("α7R V · borrowed α7 IV table · 0.50") + 2)
    assert shown == "α7R V · borrowed α7 IV table · 0.50"
    assert QLabel.text(chip).startswith("α7R V") or QLabel.text(chip) == full


def test_agent_ellipse_accepted_unchanged_never_gets_the_human_seal(make, app, tmp_path):
    from starstack import regions
    window, _ = make("zh_CN")
    _ready_reference(window, tmp_path)
    agent = {"operations": [], "milkyway": {"status": "present", "confirmed": True, "ellipse": [.5, .5, .2, .1],
                                            "authority": "agent", "selection": regions.DRAWING}}
    window._regions_project = agent
    window._regions_saved_by_user = agent       # the editor saved it without redrawing
    window._update_controls()
    assert window.regions_provenance.kind() == "agent" and not window.regions_seal.isVisible()


def test_stage_nodes_speak_their_progress_and_06_is_an_optional_node(make, app, tmp_path):
    window, _ = make("en")
    _ready_reference(window, tmp_path)
    bar = window.stage_bar
    assert bar.node("materials").accessibleDescription().startswith("Done · ")
    assert bar.node("regions").accessibleDescription().startswith("Current step · ")
    assert bar.trailing_number().text() == "06" and bar.trailing_connector_lit() is False
    assert window.frame_dots.accessibleName() == "No photos added yet"
    window._selection_changed(preview=False)
    assert window.frame_dots.accessibleName() == "3 photos; the first is the colour reference"
    assert themes.role(window.output_dir) == "mono"


def test_reopening_retouch_keeps_a_tiff_the_person_chose(make, app, tmp_path):
    window, _ = make("zh_CN")
    window.retouch_button.click()
    app.processEvents()
    exp = window.experience
    chosen = str(tmp_path / "mine.tif")
    exp.retouch_photo.setText(chosen)
    window._advanced_dialogs["retouch"].close()
    window.retouch_button.click()
    app.processEvents()
    assert exp.retouch_photo.text() == chosen


# ------------------------------------------------------------ 2026-09-27 small-issue batch (GUI)

def _wheel(widget, delta=-120):
    """One wheel notch over ``widget``, propagated to ancestors while ignored (as Qt does for a
    real, spontaneous wheel event); returns the widget that accepted it, or None."""
    centre = QPointF(widget.width() / 2, widget.height() / 2)
    target, position = widget, centre
    while target is not None:
        event = QWheelEvent(position, QPointF(widget.mapToGlobal(centre.toPoint())), QPoint(0, 0), QPoint(0, delta),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(target, event)
        if event.isAccepted() or target.isWindow():
            return target if event.isAccepted() else None
        position = position + QPointF(target.pos())
        target = target.parentWidget()
    return None


def _buddy_label(control):
    return next(label for label in control.window().findChildren(QLabel) if label.buddy() is control)


@pytest.mark.parametrize("language", ["zh_CN", "en"])
def test_number_rows_keep_single_line_labels_and_whole_values(make, app, language):
    """Item 4: 背景移星力度 / 银河相对偏移 read on one line beside a box sized to its widest value."""
    window, _ = make(language)
    window.resize(960, 680)
    window.stars_enabled.setChecked(True)
    window.workflow_auto_strength.setChecked(False)
    app.processEvents()
    window.show_star_controls()
    app.processEvents()
    controls = [window.workflow_background_strength, window.workflow_milkyway_offset, window.milkyway_strength,
                window.star_background_strength, window.star_milkyway_offset]
    for control in controls:
        label = _buddy_label(control)
        assert label.text() == control.accessibleName()
        assert not label.wordWrap(), label.text()
        # The whole single-line text fits (no wrap into a column of characters, no clipping).
        assert label.width() >= label.fontMetrics().horizontalAdvance(label.text()), (label.text(), label.width())
        assert label.geometry().right() < control.geometry().left()
        assert themes.role(control) == "mono" and control.decimals() == 2
        for value in (control.minimum(), control.maximum()):
            control.setValue(value)
            edit = control.lineEdit()
            # QLineEdit keeps a 2 px text margin on each side.
            assert edit.width() >= edit.fontMetrics().horizontalAdvance(control.text()) + 4, (control.text(), edit.width())
    assert window.star_milkyway_offset.suffix() == tr_suffix(language)
    window._advanced_dialogs["stars"].close()


def tr_suffix(language):
    return " percentage points" if language == "en" else " 个百分点"


def test_wheel_over_colour_and_number_controls_scrolls_the_column(make, app):
    """Item 6: the wheel never changes a value the person set; the steps column scrolls instead."""
    window, _ = make("zh_CN")
    window.resize(1200, 680)
    window.stars_enabled.setChecked(True)
    window.workflow_auto_strength.setChecked(False)
    app.processEvents()
    bar = window._steps_scroll.verticalScrollBar()
    assert bar.maximum() > 0
    controls = [window.warmth_slider, window.warmth_value, window.exposure_slider, window.exposure_value,
                window.workflow_background_strength]
    for control in controls:
        bar.setValue(0)
        app.processEvents()
        before_value, before_scroll = control.value(), bar.value()
        accepted_by = _wheel(control)
        app.processEvents()
        assert control.value() == before_value, control.objectName()
        assert accepted_by is not control and accepted_by is not None, control.objectName()
        assert bar.value() > before_scroll, control.objectName()
        assert control.focusPolicy() == Qt.FocusPolicy.StrongFocus
    window._color_timer.stop()
    window._color_finish_timer.stop()


def test_typing_a_colour_value_commits_once(make, app):
    window, _ = make("zh_CN")
    field, slider = window.exposure_value, window.exposure_slider
    generation = window._reference_generation
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "1.3")
    assert slider.value() == 0 and window._reference_generation == generation     # nothing until Enter
    QTest.keyClick(field, Qt.Key.Key_Return)
    window._color_timer.stop()
    window._color_finish_timer.stop()
    assert slider.value() == 13 and window.color_adjustments()["exposure_ev"] == pytest.approx(1.3)
    assert window._reference_generation == generation + 1                          # one preview update
    assert field.text() == "+1.3 EV" and window.findChild(QLabel, "displayExposureDelta").isVisible()
    warmth = window.warmth_value
    warmth.selectAll()
    QTest.keyClicks(warmth, "-25")
    QTest.keyClick(warmth, Qt.Key.Key_Return)
    window._color_timer.stop()
    window._color_finish_timer.stop()
    assert window.warmth_slider.value() == -25 and window._reference_generation == generation + 2
    # Re-committing the same value is not a change.
    warmth.selectAll()
    QTest.keyClicks(warmth, "-25")
    QTest.keyClick(warmth, Qt.Key.Key_Return)
    assert window._reference_generation == generation + 2
    window._color_timer.stop()
    window._color_finish_timer.stop()


def test_colour_fields_follow_restore_and_disabled_state(make, app, tmp_path):
    window, _ = make("zh_CN")
    window._restoring_project = True
    for slider, value in ((window.warmth_slider, 16), (window.tint_slider, -3), (window.exposure_slider, -8)):
        slider.setValue(value)
    window._restoring_project = False
    assert (window.warmth_value.text(), window.tint_value.text(), window.exposure_value.text()) == ("+16", "-3", "-0.8 EV")
    assert all(field.isEnabled() for field in (window.warmth_value, window.tint_value, window.exposure_value))
    window.workflow_dir = str(tmp_path)
    window.job_state = "running"
    window._update_controls()
    assert not any(field.isEnabled() for field in (window.warmth_value, window.tint_value, window.exposure_value))
    window.job_state = ""
    window.workflow_dir = None
    fits_file = tmp_path / "frame.fits"
    fits_file.write_bytes(b"synthetic")
    window.input_files = [str(fits_file)]
    window._update_controls()
    assert not window.warmth_slider.isEnabled() and not window.warmth_value.isEnabled()
    window.input_files = []
    window._update_controls()
    assert window.warmth_value.isEnabled()
    chain = window._tab_chain
    for slider in (window.warmth_slider, window.tint_slider, window.exposure_slider):
        assert chain.index(slider.value_field) == chain.index(slider) + 1


def test_running_job_marks_compose_busy_and_shows_the_spinner(make, app, tmp_path, monkeypatch):
    """Item 7: the 二十八宿 sweep runs only while work runs and the window is shown."""
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "0")
    window, _ = make("zh_CN")
    clock, spinner, bar = window._activity_clock, window.activity_spinner, window.stage_bar
    assert not spinner.isVisible() and not clock.is_running() and bar.busy_key() is None
    assert spinner.sizePolicy().retainSizeWhenHidden()
    window.workflow_dir = str(tmp_path)
    window.job_state = "running"
    window.preview.show_message("processing", art=True)
    window._update_controls()
    assert spinner.isVisible() and spinner.accessibleName() == "正在处理"
    assert bar.busy_key() == "compose" and bar.node("compose").is_busy() and bar.states()["compose"] == "current"
    assert clock.is_running() and window.preview.is_sweeping()
    step = clock.step()
    QTest.qWait(250)
    assert clock.step() != step                                   # about 12 whole steps a second
    window.hide()
    app.processEvents()
    assert not clock.is_running() and clock.is_active()            # paused while hidden
    window.show()
    app.processEvents()
    assert clock.is_running()
    window.showMinimized()
    app.processEvents()
    if window.isMinimized():
        assert not clock.is_running()
    window.showNormal()
    app.processEvents()
    window.job_state = "cancel_requested"
    window._update_controls()
    assert spinner.accessibleName() == "正在停止"
    window.job_state = "succeeded"
    window._update_controls()
    QTest.qWait(window.ACTIVITY_LINGER_MS + 150)
    assert not spinner.isVisible() and not clock.is_running() and bar.busy_key() is None
    assert not window.preview.is_sweeping()
    # Reduced motion (Windows "animation effects" off): the marks stay, drawn still.
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "1")
    window.job_state = "running"
    window._update_controls()
    assert spinner.isVisible() and bar.node("compose").is_busy()
    assert not clock.is_running() and clock.is_static() and not window.preview.is_sweeping()
    window.job_state = ""
    window.workflow_dir = None
    window._update_controls()


def test_first_frame_updates_show_the_mark_only_after_a_pause(make, app, tmp_path, monkeypatch):
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "1")
    window, _ = make("en")
    window._reference_running = True
    window._update_controls()
    assert not window.activity_spinner.isVisible()                 # short updates never flash
    QTest.qWait(window.ACTIVITY_SHOW_DELAY_MS + 150)
    assert window.activity_spinner.isVisible() and window.activity_spinner.accessibleName() == "Processing"
    assert window.stage_bar.busy_key() == "reference"
    window._reference_running = False
    window._update_controls()
    QTest.qWait(window.ACTIVITY_LINGER_MS + 150)
    assert not window.activity_spinner.isVisible() and window.stage_bar.busy_key() is None


def _submitted_options(window, monkeypatch, tmp_path, **checks):
    from starstack import workflow
    from test_p1_gui_routing_independent import scene
    scene(window, monkeypatch, tmp_path, present=True)
    for name, value in checks.items():
        getattr(window, name).setChecked(value)
    captured, requests = [], []
    monkeypatch.setattr(window, "_run_async", lambda *args: captured.append(args))
    monkeypatch.setattr(workflow, "submit_workflow", lambda *args, **kwargs: requests.append(kwargs) or {})
    try:
        window.start_processing()
        assert len(captured) == 1 and captured[0][0] == "submit"
        captured[0][1]()
    finally:
        window._busy = False
    return requests[0]


def test_trails_request_meteor_default_and_star_layers_are_opt_in(make, app, tmp_path, monkeypatch):
    """Items 9 and 14 (contract A): meteors added back by default; extra layers only on request."""
    window, _ = make("zh_CN")
    request = _submitted_options(window, monkeypatch, tmp_path, trails_enabled=True, stars_enabled=False)
    assert request["trails_enabled"] is True and request["trail_default"] == "meteor"
    assert "save_starless" not in request and "save_star_layer" not in request
    assert "流星自动加回" in window.trails_note.text() and "复核" in window.trails_enabled.toolTip()


def test_save_starless_and_star_layer_follow_star_tidying(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    starless, layer = window.workflow_save_starless, window.workflow_save_star_layer
    assert starless.objectName() == "workflowSaveStarless" and layer.objectName() == "workflowSaveStarLayer"
    assert not starless.isChecked() and not layer.isChecked()
    assert not starless.isVisible() and not layer.isVisible()
    window.stars_enabled.setChecked(True)
    app.processEvents()
    assert starless.isVisible() and layer.isVisible() and starless.isEnabled()
    assert "无星图＋星点层＝整理前的成片" in starless.toolTip() and "地景" in layer.toolTip()
    content = window._steps_scroll.widget()
    tops = [w.mapTo(content, QPoint(0, 0)).y() for w in
            (window.workflow_milkyway_offset, starless, layer, window.star_component_note)]
    assert tops == sorted(tops)
    chain = window._tab_chain
    assert chain.index(window.workflow_milkyway_offset) < chain.index(starless) < chain.index(layer) \
        < chain.index(window.star_component_button)
    window.workflow_dir = str(tmp_path)
    window.job_state = "running"
    window._update_controls()
    assert not starless.isEnabled() and not layer.isEnabled()
    window.job_state = ""
    window.workflow_dir = None
    window._update_controls()
    request = _submitted_options(window, monkeypatch, tmp_path, trails_enabled=False, stars_enabled=True,
                                 workflow_save_starless=True, workflow_save_star_layer=True)
    assert request["save_starless"] is True and request["save_star_layer"] is True
    assert "trail_default" not in request
    state = window._editing_state()
    assert state["save_starless"] is True and state["save_star_layer"] is True


def test_editing_project_keeps_layer_options_and_old_projects_load(make, app, tmp_path, monkeypatch):
    from starstack import editing_project
    window, _ = make("zh_CN")
    window.stars_enabled.setChecked(True)
    window.workflow_save_star_layer.setChecked(True)
    saved = {}

    def fake_save(path, state, reference, project):
        saved.update(state)
        return {"project_path": str(path)}
    _ready_reference(window, tmp_path)
    window._reference_data["preview_path"] = str(tmp_path / "missing.png")
    monkeypatch.setattr(editing_project, "save_project", fake_save)
    monkeypatch.setattr(window, "_run_async", lambda name, op, done: done(op()))
    monkeypatch.setattr("starstack.regions.create_project", lambda reference, files: {"scene_mode": "sky"})
    monkeypatch.setattr("starstack.regions.revise", lambda project, **kwargs: project)
    window.save_editing_project(str(tmp_path / "p.aester.json"))
    assert saved["save_star_layer"] is True and saved["save_starless"] is False
    # An older project without the keys loads with both options off.
    state = dict(saved)
    state.pop("save_starless")
    state.pop("save_star_layer")
    window.workflow_save_star_layer.setChecked(True)
    monkeypatch.setattr(window._color_timer, "start", lambda *a: None)
    window._restore_editing_project({"state": state, "regions": None, "regions_path": None,
                                     "reference": dict(window._reference_data)})
    assert not window.workflow_save_starless.isChecked() and not window.workflow_save_star_layer.isChecked()


def _trail_deliveries(tmp_path):
    base, trails = tmp_path / "delivery-base", tmp_path / "delivery-trails-2kept-abc"
    for folder in (base, trails):
        folder.mkdir()
        _image().save(str(folder / "final.jpg"))
        (folder / "final_srgb16.tif").write_bytes(b"synthetic display tiff")
        (folder / "master_scene_linear.tif").write_bytes(b"synthetic master")
    record = {"state": "succeeded", "request": {"trails_enabled": True},
              "delivery_dir": str(trails), "base_delivery_dir": str(base),
              "artifacts": [{"path": str(trails / name), "name": name, "kind": kind} for name, kind in
                            (("master_scene_linear.tif", "tiff"), ("final_srgb16.tif", "tiff"), ("final.jpg", "jpeg"))],
              "trail_review": {"automatic_addback": True, "default_policy": "meteor-default-keep-1", "kept_count": 2,
                               "kept_ids": ["a", "b"], "kept_by": {"algorithm": 1, "human": 1, "agent": 0},
                               "pending_count": 3},
              "extras": [{"role": "starless", "path": str(trails / "starless_srgb16.tif"), "sha256": "0" * 64}]}
    return base, trails, record


def test_review_publication_refreshes_results_without_resubmitting(make, app, tmp_path, monkeypatch):
    """Item 11 (contracts B/C): the with-trails photograph first; sliders and effects untouched."""
    window, _ = make("zh_CN")
    base, trails, record = _trail_deliveries(tmp_path)
    window.workflow_dir = str(tmp_path)
    window.job_state = "succeeded"
    window._restoring_project = True
    window.warmth_slider.setValue(15)
    window._restoring_project = False
    window._color_timer.stop()
    window._color_finish_timer.stop()
    window.stars_enabled.setChecked(True)
    monkeypatch.setattr(window, "_workflow_submitted", lambda *a: pytest.fail("a review never resubmits"))
    window._busy = True
    window._review_published({"workflow": record, "published": True, "kept_count": 2, "kept_ids": ["a", "b"],
                              "delivery_dir": str(trails), "identical_to_base": False, "trail_layer_path": None})
    assert not window._busy and window.warmth_slider.value() == 15 and window.stars_enabled.isChecked()
    assert [window.style_tabs.tabText(i) for i in range(window.style_tabs.count())] == ["含亮迹", "基础成片"]
    assert window.style_tabs.currentIndex() == 0 and window.style_tabs.isVisible()
    note = window._products_note()
    assert note.startswith("含亮迹版本 · 已加回 2 条（算法 1 / 人工 1 / Agent 0） · 另有 3 条待核实"), note
    assert "另存：无星图" in note and window.preview_note.text() == note
    assert "保留 2 条亮迹" in window.message_label.text() and str(trails) in window.message_label.text()
    assert window.retouch_photo_path() == str(trails / "final_srgb16.tif")
    window.style_tabs.setCurrentIndex(1)
    assert window.retouch_photo_path() == str(base / "final_srgb16.tif")
    same = dict(record, delivery_dir=str(base), artifacts=[dict(a, path=a["path"].replace(str(trails), str(base)))
                                                            for a in record["artifacts"]])
    window._review_published({"workflow": same, "published": False, "kept_count": 0, "kept_ids": [],
                              "delivery_dir": None, "identical_to_base": True, "trail_layer_path": None})
    assert window.message_label.text() == "没有保留任何亮迹；成片与基础版一致，身份改判已记录"
    assert window.style_tabs.count() == 1 and not window._products_note().startswith("含亮迹版本")
    window.job_state = ""
    window.workflow_dir = None


def test_finished_workflow_with_automatic_addback_shows_the_trails_version_first(make, app, tmp_path, monkeypatch):
    window, _ = make("en")
    base, trails, record = _trail_deliveries(tmp_path)
    monkeypatch.setattr("starstack.workflow.workflow_status", lambda *_: record)
    window.workflow_dir = str(tmp_path)
    window.refresh_workflow()
    assert [window.style_tabs.tabText(i) for i in range(window.style_tabs.count())] == ["With trails", "Base photograph"]
    assert window._jpg_artifacts[0]["path"] == str(trails / "final.jpg")
    assert window.preview_note.text().startswith("With-trails version · 2 added back (algorithm 1 / human 1 / Agent 0)")
    window.job_state = ""
    window.workflow_dir = None


# ------------------------------------------------------------------------------ 2026-10-09 JPEG quality

def _estimate_reference(window, tmp_path, *, size=(8280, 5520), preview=(1024, 683), quality="standard"):
    """A first-frame reference with a real preview PNG (night-sky-like noise) and the full frame size."""
    from test_jpeg_quality import scene
    files = _ready_reference(window, tmp_path)
    path = tmp_path / f"preview-{preview[0]}-{quality}.png"
    from PIL import Image
    Image.fromarray(scene(7, preview[1], preview[0])).save(path)
    window._reference_data.update(width=size[0], height=size[1], preview_path=str(path),
                                  preview_width=preview[0], preview_height=preview[1], preview_quality=quality)
    window._update_controls()
    return files, path


def _settle_estimate(window, app):
    window._jpeg_estimate_timer.stop()
    window._update_jpeg_estimate()
    app.processEvents()


def test_jpeg_quality_slider_is_unsigned_typed_wheel_proof_and_resets_to_95(make, app):
    from PySide6.QtWidgets import QAbstractSpinBox
    from aester_desktop import ui_widgets
    window, prefs = make("zh_CN")
    slider, field = window.jpeg_quality_slider, window.jpeg_quality_value
    assert isinstance(slider, ui_widgets.TickSlider) and not slider.is_bipolar() and slider.default_value() == 95
    assert (slider.minimum(), slider.maximum(), slider.singleStep(), slider.pageStep(), slider.value()) == (60, 100, 1, 5, 95)
    assert isinstance(field, ui_widgets.UnsignedValueField) and field.text() == "95" and window.jpeg_quality() == 95
    assert (field.decimals(), field.minimum(), field.maximum()) == (0, 60, 100)
    assert field.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons and not field.keyboardTracking()
    assert window.findChild(QLabel, "jpegQualityDelta") is None              # an absolute setting: no Δ from 0
    assert _buddy_label(field).text() == "JPEG 质量" and slider.accessibleName() == "JPEG 质量"
    assert "双击滑块恢复为 95" in slider.toolTip() and "亮迹版本沿用" in slider.toolTip()
    # The wheel never changes it.
    for control in (slider, field):
        accepted_by = _wheel(control)
        assert control.value() == 95 and accepted_by is not control, control.objectName()
    # Arrow keys step 1, Page Up / Down 5.
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Right)
    assert slider.value() == 96 and field.text() == "96"
    QTest.keyClick(slider, Qt.Key.Key_PageUp)
    assert slider.value() == 100
    QTest.keyClick(slider, Qt.Key.Key_PageDown)
    QTest.keyClick(slider, Qt.Key.Key_PageDown)
    assert slider.value() == 90 and field.text() == "90"                     # unsigned: never "+90"
    # A typed value is committed on Enter only.
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "72")
    assert slider.value() == 90
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert slider.value() == 72 and window.jpeg_quality() == 72 and field.text() == "72"
    # Double-click returns to 95.
    QTest.mouseDClick(slider, Qt.MouseButton.LeftButton)
    assert slider.value() == 95 and field.text() == "95"
    # Frozen into a job at its start: no change while one runs.
    window.workflow_dir = "running"
    window.job_state = "running"
    window._update_controls()
    assert not slider.isEnabled() and not field.isEnabled()
    window.job_state = ""
    window.workflow_dir = None
    window._update_controls()
    assert slider.isEnabled() and field.isEnabled()


def test_jpeg_quality_preference_is_restored_without_writing_and_saved_on_change(make, app, tmp_path):
    def snapshot(settings):
        settings.sync()
        return {key: settings.value(key) for key in settings.allKeys()}

    prefs = QSettings(str(tmp_path / "quality.ini"), QSettings.Format.IniFormat)
    prefs.setValue("appearance/language", "zh_CN")
    prefs.setValue("output/jpeg_quality", 83)
    before = snapshot(prefs)
    window, _ = make(settings=prefs)
    app.processEvents()
    assert window.jpeg_quality() == 83 and window.jpeg_quality_value.text() == "83"
    window._jpeg_estimate_timer.stop()
    window._update_jpeg_estimate()
    assert snapshot(prefs) == before                                        # read, never written while opening
    window.jpeg_quality_slider.setValue(77)
    window._update_jpeg_estimate()                                          # the debounced step also syncs
    reread = QSettings(str(tmp_path / "quality.ini"), QSettings.Format.IniFormat)
    assert int(reread.value("output/jpeg_quality")) == 77
    window.close()
    app.processEvents()
    reopened, _ = make(settings=QSettings(str(tmp_path / "quality.ini"), QSettings.Format.IniFormat))
    assert reopened.jpeg_quality() == 77
    for stored in ("abc", "120", "59", "95.5", ""):
        odd = QSettings(str(tmp_path / f"odd-{len(stored)}-{stored}.ini"), QSettings.Format.IniFormat)
        odd.setValue("appearance/language", "zh_CN")
        odd.setValue("output/jpeg_quality", stored)
        kept = snapshot(odd)
        other, _ = make(settings=odd)
        assert other.jpeg_quality() == 95, stored
        assert snapshot(odd) == kept, stored


def test_jpeg_estimate_follows_preview_quality_and_trails(make, app, tmp_path):
    from starstack import jpeg_estimate as je
    window, _ = make("zh_CN")
    label = window.jpeg_estimate_label
    _settle_estimate(window, app)
    assert not label.isVisible() and label.text() == ""                      # no photos: no size known
    window.trails_enabled.setChecked(False)
    _, preview = _estimate_reference(window, tmp_path)
    assert window._jpeg_estimate_timer.isActive()                            # a new preview schedules the estimate
    _settle_estimate(window, app)
    size95 = je.estimate_from_preview(preview, 8280, 5520, 95)
    assert label.isVisible() and label.text() == f"约 {je.megabytes(size95)} MB／张（粗略估计）"
    assert "首帧预览" in label.toolTip()
    window.jpeg_quality_slider.setValue(70)
    assert window._jpeg_estimate_timer.isActive()                            # debounced, not per value
    _settle_estimate(window, app)
    size70 = je.estimate_from_preview(preview, 8280, 5520, 70)
    assert size70 < size95 and label.text() == f"约 {je.megabytes(size70)} MB／张（粗略估计）"
    window.trails_enabled.setChecked(True)
    _settle_estimate(window, app)
    assert label.text() == f"约 {je.megabytes(size70)} MB／张（粗略估计）；含亮迹版本另存一张，大小相近"
    # A 640 px interactive preview while colours are dragged keeps the last standard preview as the basis.
    interactive = tmp_path / "interactive.png"
    from PIL import Image
    Image.new("RGB", (640, 427), (200, 10, 10)).save(interactive)
    window._reference_data.update(preview_path=str(interactive), preview_quality="interactive")
    window._update_controls()
    _settle_estimate(window, app)
    assert label.text().startswith(f"约 {je.megabytes(size70)} MB")
    # After a run the reference is cleared; the same selection keeps its estimate.
    window._reference_data = None
    window._update_controls()
    _settle_estimate(window, app)
    assert label.isVisible() and label.text().startswith(f"约 {je.megabytes(size70)} MB")
    # Without a preview image the rough table is used for the known frame size.
    window._jpeg_basis = None
    window._reference_data = {"width": 6000, "height": 4000, "preview_path": str(tmp_path / "missing.png")}
    window._update_controls()
    _settle_estimate(window, app)
    fallback = je.estimate_without_preview(6000, 4000, 70)
    assert label.text().startswith(f"约 {je.megabytes(fallback)} MB") and "尚无首帧预览" in label.toolTip()
    # No photos at all: the line is hidden again.
    window._reference_data = None
    window.input_files = []
    window._update_controls()
    _settle_estimate(window, app)
    assert not label.isVisible()


def test_jpeg_quality_reads_in_english(make, app, tmp_path):
    from starstack import jpeg_estimate as je
    window, _ = make("en")
    _, preview = _estimate_reference(window, tmp_path)
    window.trails_enabled.setChecked(True)
    window.jpeg_quality_slider.setValue(90)
    _settle_estimate(window, app)
    label = window.jpeg_estimate_label
    size = je.estimate_from_preview(preview, 8280, 5520, 90)
    assert _buddy_label(window.jpeg_quality_value).text() == "JPEG quality"
    assert label.text() == (f"About {je.megabytes(size)} MB per photo (rough estimate); "
                            "the trails version adds another copy of about the same size")
    for text in (label.text(), label.toolTip(), window.jpeg_quality_slider.toolTip(),
                 window.jpeg_quality_slider.accessibleName(), window.jpeg_quality_value.toolTip()):
        assert not re.search("[一-龥]", text), text


@pytest.mark.parametrize("note", [False, True])
@pytest.mark.parametrize("language", ["zh_CN", "en"])
@pytest.mark.parametrize("size", [(960, 680), (1200, 800)])
def test_jpeg_quality_row_and_estimate_fit_the_fixed_bottom(make, app, tmp_path, language, size, note):
    window, _ = make(language)
    _estimate_reference(window, tmp_path)
    window.trails_enabled.setChecked(True)                                 # the longest estimate text
    if note:                                    # ... and a shown workflow that keeps another quality
        window._jpeg_job = {"quality": 64}
    window.resize(*size)
    app.processEvents()
    _settle_estimate(window, app)
    app.processEvents()
    shown = window._jpeg_job_note()
    assert bool(shown) == note
    if note:
        assert "64" in shown and window.jpeg_estimate_label.text().endswith(shown)
    panel = window.centralWidget().findChild(QWidget, "controlPanel")
    area = window.centralWidget().rect()
    widgets = [window.output_dir, window.jpeg_quality_slider, window.jpeg_quality_value, window.jpeg_estimate_label,
               window.start_button, window.cancel_button]
    rects = []
    for widget in widgets:
        assert widget.isVisible() and panel.isAncestorOf(widget), widget.objectName()
        assert not window._steps_scroll.widget().isAncestorOf(widget)
        rect = _rect(window, widget)
        assert area.contains(rect.topLeft()) and area.contains(rect.bottomRight()), (widget.objectName(), rect)
        assert _rect(window, panel).contains(rect), widget.objectName()
        rects.append((widget, rect))
    for index, (first, a) in enumerate(rects):
        for second, b in rects[index + 1:]:
            assert not a.intersects(b), (first.objectName(), second.objectName())
    tops = [rect.top() for _, rect in rects]
    # Directly under 作品保存位置: the slider and its field on one row, then the estimate, then the buttons.
    assert tops[0] < tops[1] < tops[3] < tops[4] and abs(tops[1] - tops[2]) <= 12
    label = window.jpeg_estimate_label
    assert label.height() >= label.heightForWidth(label.width()) - 1       # the wrapped text is not clipped
    assert window.jpeg_quality_slider.width() >= 100
    viewport = window._steps_scroll.viewport()
    assert viewport.rect().contains(window.regions_button.mapTo(viewport, QPoint(0, 0)))
    assert window.preview.height() >= 220


def test_start_sends_the_slider_value_even_at_its_default(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    window.jpeg_quality_slider.setValue(77)
    request = _submitted_options(window, monkeypatch, tmp_path, trails_enabled=False, stars_enabled=False)
    assert request["jpeg_quality"] == 77
    window.jpeg_quality_slider.setValue(95)
    (tmp_path / "again").mkdir()
    request = _submitted_options(window, monkeypatch, tmp_path / "again", trails_enabled=False, stars_enabled=False)
    assert request["jpeg_quality"] == 95                                   # the default is sent, not left out


def _show_workflow(window, app, directory, config):
    """A loaded (or resumed) workflow record arriving in the window, then idle again."""
    window._workflow_submitted({"workflow_dir": str(directory), "state": "queued", "config": config})
    window.timer.stop()
    window.job_state = ""
    window._update_controls()
    _settle_estimate(window, app)


def test_a_loaded_workflow_shows_its_frozen_quality_and_keeps_the_preference(make, app, tmp_path, monkeypatch):
    def stored(settings):
        settings.sync()
        return int(settings.value("output/jpeg_quality"))

    window, prefs = make("zh_CN", values={"output/jpeg_quality": 83})
    monkeypatch.setattr(window, "_run_async", lambda *args: None)
    label = window.jpeg_estimate_label
    (tmp_path / "wf").mkdir()
    _show_workflow(window, app, tmp_path / "wf", {"input_files": [], "jpeg_quality": 70})
    assert window.jpeg_quality() == 70 and window.jpeg_quality_value.text() == "70"
    assert stored(prefs) == 83                                           # shown, not written as the preference
    assert not label.isVisible()                                         # the slider says it; no size known
    # A change while the job is shown is the preference for new jobs; the note says what the job keeps.
    window.jpeg_quality_slider.setValue(75)
    _settle_estimate(window, app)
    assert stored(prefs) == 75
    assert label.isVisible() and label.text() == "当前任务仍按 70 保存" and "新任务" in label.toolTip()
    # A workflow submitted without a quality: the preference, and what its later trails versions use.
    _show_workflow(window, app, tmp_path / "wf", {"input_files": []})
    assert window.jpeg_quality() == 75 and label.text() == "当前任务未设定 JPEG 质量，之后的亮迹版本按 97 保存"
    _show_workflow(window, app, tmp_path / "wf", {"input_files": [], "jpeg_quality": 64})
    assert window.jpeg_quality() == 64 and not label.isVisible()
    # With a size known, the note follows the estimate.
    window._jpeg_job = {"quality": 70}
    _estimate_reference(window, tmp_path)
    window.trails_enabled.setChecked(False)
    _settle_estimate(window, app)
    assert label.text().startswith("约 ") and label.text().endswith("（粗略估计）；当前任务仍按 70 保存")
    # Closing writes the preference, never the value a shown job froze.
    window.close()
    app.processEvents()
    assert stored(prefs) == 75


def test_a_new_selection_returns_the_slider_to_the_preference(make, app, tmp_path, monkeypatch):
    window, prefs = make("zh_CN", values={"output/jpeg_quality": 88})
    monkeypatch.setattr(window, "_run_async", lambda *args: None)
    (tmp_path / "wf").mkdir()
    _show_workflow(window, app, tmp_path / "wf", {"input_files": [], "jpeg_quality": 62})
    assert window.jpeg_quality() == 62
    files = []
    for index in range(2):
        path = tmp_path / f"new{index}.nef"
        path.write_bytes(b"synthetic, never decoded")
        files.append(str(path))
    window.input_files = files
    window._selection_changed()
    window._color_timer.stop()
    _settle_estimate(window, app)
    assert window.jpeg_quality() == 88 and window._jpeg_job is None
    assert "当前任务" not in window.jpeg_estimate_label.text()
    prefs.sync()
    assert int(prefs.value("output/jpeg_quality")) == 88


def test_the_open_job_notes_read_in_english(make, app, tmp_path, monkeypatch):
    window, _ = make("en")
    monkeypatch.setattr(window, "_run_async", lambda *args: None)
    label = window.jpeg_estimate_label
    (tmp_path / "wf").mkdir()
    _show_workflow(window, app, tmp_path / "wf", {"input_files": []})
    assert label.text() == "The open job has no JPEG quality set; its later trails versions are saved at 97"
    window.jpeg_quality_slider.setValue(80)
    _show_workflow(window, app, tmp_path / "wf", {"input_files": [], "jpeg_quality": 70})
    window.jpeg_quality_slider.setValue(81)
    _estimate_reference(window, tmp_path)
    window.trails_enabled.setChecked(False)
    _settle_estimate(window, app)
    assert label.text().startswith("About ") and label.text().endswith("(rough estimate). The open job keeps quality 70")
    for text in (label.text(), label.toolTip()):
        assert not re.search("[一-龥；]", text), text


def test_a_full_frame_reference_preview_is_reduced_before_the_estimate(make, app, tmp_path):
    """A FITS reference shows its full decoded frame; the estimate encodes at most a standard 1024 px preview."""
    from starstack import jpeg_estimate as je
    window, _ = make("zh_CN")
    window.trails_enabled.setChecked(False)
    _, preview = _estimate_reference(window, tmp_path, size=(2400, 1600), preview=(2400, 1600))
    _settle_estimate(window, app)
    image = window._jpeg_basis["image"]
    assert max(image.size) == je.PREVIEW_EDGE and image.mode == "RGB"
    reduced = je.load_preview(preview, max_edge=je.PREVIEW_EDGE)
    size = je.estimate_from_preview(reduced, 2400, 1600, 95)
    assert window.jpeg_estimate_label.text() == f"约 {je.megabytes(size)} MB／张（粗略估计）"
