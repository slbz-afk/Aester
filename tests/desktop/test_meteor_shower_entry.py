"""Meteor-shower entry, the cache estimate before start, and back to an old task.

Real Qt widgets offscreen. The engine is used as it is (admission on the controlled engine, synthetic DNGs from
tests/color_fixtures.py); only Siril, the workflow worker and the review dialog are stubbed.
"""
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QSettings
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

from starstack import core, disk_budget, i18n, stack_range
from aester_desktop import disk_space, gui, themes

import color_fixtures as cf

CJK = re.compile(r'[　-〿㐀-䶿一-鿿豈-﫿＀-￯]')


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make(app, tmp_path, monkeypatch):
    monkeypatch.delenv("AESTER_LANGUAGE", raising=False)
    monkeypatch.delenv("AESTER_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    windows = []

    def create(language="zh_CN", settings=None):
        prefs = settings or QSettings(str(tmp_path / f"prefs-{len(windows)}.ini"), QSettings.Format.IniFormat)
        if settings is None:
            prefs.setValue("appearance/language", language)
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
        window.workflow_dir = window.job_dir = None
        window._color_timer.stop()
        window.timer.stop()
        window._disk_estimate_timer.stop()
        window._pool.waitForDone(30000)
        for dialog in window._advanced_dialogs.values():
            dialog.close()
        window.close()
    app.processEvents()
    i18n.set_language("zh_CN")


def settle(window, app):
    for _ in range(4):
        app.processEvents()
        window._pool.waitForDone(30000)
        app.processEvents()


def reopen(window, prefs, make, app):
    """Close the window as a person would (preferences saved) and start a new one on the same settings file."""
    window.close()
    app.processEvents()
    prefs.sync()
    return make(settings=QSettings(prefs.fileName(), QSettings.Format.IniFormat))[0]


# ------------------------------------------------------------------------------------ 流星雨合成

def test_preset_turns_trails_on_shows_the_hints_and_survives_a_restart(make, app):
    window, prefs = make("zh_CN")
    assert not window.meteor_preset_button.isChecked() and not window.trails_enabled.isChecked()
    assert prefs.value(gui.METEOR_PRESET_KEY) is None                 # nothing written at start
    assert window.meteor_preset_button.text() == "流星雨合成"
    window.meteor_preset_button.click()
    assert window.meteor_preset_button.text() == "流星雨合成 · 已打开"
    assert window.trails_enabled.isChecked() and window._meteor_preset
    assert gui._preference_bool(prefs.value(gui.METEOR_PRESET_KEY))
    hints = window.meteor_hints
    assert hints.isVisible() and themes.role(hints) == "hint"
    assert "先在 ① 标记地景" in hints.text() and "30 分钟" in hints.text() and "150 张" in hints.text()
    assert "复核完亮迹再清理缓存" in hints.text()
    assert "下一段先点“清空素材”，再添加新照片" in hints.text()
    assert window.message_label.text() == "已打开流星雨合成：可信的流星自动加回，其他亮迹待你复核。下次启动仍然打开。"
    restarted = reopen(window, prefs, make, app)
    assert restarted.meteor_preset_button.isChecked() and restarted.trails_enabled.isChecked()
    assert restarted.meteor_hints.isVisible()
    # Unticking the trails box by hand ends the mode, and that is remembered too.
    restarted.trails_enabled.setChecked(False)
    assert not restarted.meteor_preset_button.isChecked() and not restarted.meteor_hints.isVisible()
    assert not gui._preference_bool(prefs.value(gui.METEOR_PRESET_KEY))
    again = reopen(restarted, prefs, make, app)
    assert not again.meteor_preset_button.isChecked() and not again.trails_enabled.isChecked()


def test_turning_the_preset_off_turns_trails_off(make, app):
    window, prefs = make("zh_CN")
    window.meteor_preset_button.click()
    window.meteor_preset_button.click()
    assert not window.trails_enabled.isChecked() and not window._meteor_preset
    assert not gui._preference_bool(prefs.value(gui.METEOR_PRESET_KEY))


def test_ground_hint_disappears_once_the_ground_is_marked(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    window.meteor_preset_button.click()
    assert "标记地景" in window.meteor_hints.text()
    monkeypatch.setattr(window, "_scene_from_current_marks", lambda: "landscape")
    window._update_controls()
    assert "标记地景" not in window.meteor_hints.text() and "150 张" in window.meteor_hints.text()


def test_welcome_entry_turns_the_preset_on_and_counts_as_seen(make, app):
    window, prefs = make("zh_CN")
    window.experience.show_welcome()
    app.processEvents()
    dialog = window._advanced_dialogs["welcome"]
    button = dialog.findChild(QPushButton, "welcomeMeteorShower")
    assert button is not None and button.text() == "流星雨合成" and not button.autoDefault()
    # Only meteors the app is confident about are added automatically; the text says so.
    assert any(label.text() == "拍流星雨时用：可信的流星自动加回，其他亮迹待你复核；每段照片出一张图。"
               for label in dialog.findChildren(QLabel))
    button.click()
    app.processEvents()
    assert not dialog.isVisible() and str(prefs.value("experience/welcome_seen")) == "1"
    assert window.meteor_preset_button.isChecked() and window.trails_enabled.isChecked()
    assert gui._preference_bool(prefs.value(gui.METEOR_PRESET_KEY))


def test_welcome_entry_waits_while_a_job_runs(make, app):
    window, prefs = make("zh_CN")
    window.experience.show_welcome()
    window.job_state = "running"
    window._advanced_dialogs["welcome"].findChild(QPushButton, "welcomeMeteorShower").click()
    assert not window._meteor_preset and prefs.value(gui.METEOR_PRESET_KEY) is None
    assert "处理完成后" in window.message_label.text()
    window.job_state = ""


def test_preset_request_is_the_hand_ticked_request_and_the_default_request_is_unchanged(make, app, tmp_path,
                                                                                      monkeypatch):
    """Contract A: the preset adds no option, so jobs that do not use it keep their request and digest."""
    from test_ui_mainwindow_tuoxing import _submitted_options
    for name in ("plain", "hand", "preset"):
        (tmp_path / name).mkdir()
    window, _ = make("zh_CN")
    plain = _submitted_options(window, monkeypatch, tmp_path / "plain", trails_enabled=False, stars_enabled=False)
    assert plain["trails_enabled"] is False and "trail_default" not in plain
    hand = _submitted_options(window, monkeypatch, tmp_path / "hand", trails_enabled=True, stars_enabled=False)
    window.trails_enabled.setChecked(False)
    window._update_controls()                          # the helper leaves the controls as submission set them
    window.meteor_preset_button.click()
    assert window.trails_enabled.isChecked()
    preset = _submitted_options(window, monkeypatch, tmp_path / "preset", stars_enabled=False)
    assert preset["trails_enabled"] is True and preset["trail_default"] == "meteor"

    def comparable(request, scene):
        """The request with this scene's own folder written as <scene> (each call builds its own synthetic scene)."""
        return {key: value.replace(str(tmp_path / scene), "<scene>") if isinstance(value, str) else
                [item.replace(str(tmp_path / scene), "<scene>") for item in value] if isinstance(value, list) else value
                for key, value in request.items() if key != "regions_path"}
    assert comparable(preset, "preset") == comparable(hand, "hand")
    assert set(plain) == set(hand) - {"trail_default"}


def test_a_viewed_job_keeps_its_own_trails_value_and_never_writes_the_preference(make, app, tmp_path):
    window, prefs = make("zh_CN")
    window.meteor_preset_button.click()
    window._workflow_submitted({"workflow_dir": str(tmp_path / "workflow-old"), "state": "succeeded",
                                "config": {"trails_enabled": False, "input_files": []}})
    window.timer.stop()
    assert not window.trails_enabled.isChecked()                     # the old job's own option
    assert window._meteor_preset and gui._preference_bool(prefs.value(gui.METEOR_PRESET_KEY))
    # The button shows what 「开始合成」 would send: trails are off for this job.
    assert not window.meteor_preset_button.isChecked() and window.meteor_preset_button.text() == "流星雨合成"
    assert not window.meteor_hints.isVisible()
    # A new selection is a new job: the mode applies again.
    window.input_files = [str(tmp_path / "a.nef")]
    window._selection_changed()
    window._color_timer.stop()
    assert window.trails_enabled.isChecked()
    assert window.meteor_preset_button.isChecked() and window.meteor_preset_button.text() == "流星雨合成 · 已打开"
    window.workflow_dir = None
    window.job_state = ""


def test_restored_project_without_trails_shows_the_mode_off_and_a_click_ticks_trails(make, app, tmp_path, monkeypatch):
    """A saved editing project is a new job: the button must say what will be sent, and a click must not end the
    saved mode the person asked for."""
    window, prefs = make("zh_CN")
    window.meteor_preset_button.click()
    state = window._editing_state()
    state.update(trails_enabled=False, input_files=[str(tmp_path / f"f{n}.dng") for n in range(3)])
    monkeypatch.setattr(window._color_timer, "start", lambda *args: None)
    window._restore_editing_project({"state": state, "reference": {}, "regions": None, "regions_path": None})
    assert not window.trails_enabled.isChecked() and window._meteor_preset
    assert not window.meteor_preset_button.isChecked() and window.meteor_preset_button.text() == "流星雨合成"
    assert not window.meteor_hints.isVisible()
    window.meteor_preset_button.click()
    assert window.trails_enabled.isChecked() and window._meteor_preset
    assert gui._preference_bool(prefs.value(gui.METEOR_PRESET_KEY))
    assert window.meteor_preset_button.isChecked() and window.meteor_preset_button.text() == "流星雨合成 · 已打开"
    assert window.meteor_hints.isVisible() and "150 张" in window.meteor_hints.text()


def test_review_cache_hint_shows_whenever_trails_are_kept(make, app):
    window, _ = make("zh_CN")
    window.trails_enabled.setChecked(True)                           # by hand, without the mode
    assert window.meteor_hints.isVisible() and window.meteor_hints.text() == "· 复核完亮迹再清理缓存。"
    assert not window.meteor_preset_button.isChecked()
    window.trails_enabled.setChecked(False)
    assert not window.meteor_hints.isVisible()


def test_preset_reads_in_english(make, app):
    window, _ = make("en")
    window.meteor_preset_button.click()
    assert window.meteor_preset_button.text() == "Meteor shower · on"
    assert "confident" in window.message_label.text() and "Clear all" in window.meteor_hints.text()
    for text in (window.meteor_hints.text(), window.meteor_preset_button.toolTip(), window.message_label.text()):
        assert text and not CJK.search(text), text


# ------------------------------------------------------------------------------------ cache estimate

@pytest.fixture
def dng_frames(tmp_path):
    (tmp_path / "frames").mkdir()
    first = cf.write_reference_dng(tmp_path / "frames" / "f00001.dng")
    files = [first]
    for number in range(2, 5):
        copy = tmp_path / "frames" / f"f{number:05d}.dng"
        copy.write_bytes(first.read_bytes())
        files.append(copy)
    return [str(path) for path in files]


@pytest.mark.parametrize("mode", ["legacy", "positive_power2_v1"])
def test_estimate_equals_the_admission_record(dng_frames, tmp_path, monkeypatch, mode):
    engine = {"available": True, "executable": str(tmp_path / "siril.exe"), "version": "1.4.4", "message": "test"}
    monkeypatch.setattr(core, "doctor", lambda *_: dict(engine))
    monkeypatch.setattr(core, "_spawn_worker", lambda directory: None)
    monkeypatch.setattr(stack_range, "require_engine", lambda info: None)
    monkeypatch.setattr("starstack.stack_settings.admission_field", lambda *args, **kwargs: None)
    cache = tmp_path / "cache"
    shown = disk_space.job_estimate(dng_frames, mode, cache)
    job = core.submit_job(None, str(cache / "stack"), input_files=dng_frames, stack_range_mode=mode,
                          idempotency_key="estimate-" + mode)
    admitted = dict(job["disk_estimate"])
    admitted.pop("free_bytes_at_admission")
    assert shown["record"] == admitted
    assert shown["record"]["required_bytes"] == job["disk_estimate"]["required_bytes"]
    assert shown["record"]["basis"] == "frames_x_frame_volume"
    assert shown["record"]["range_attempts"] == (stack_range.contract()["max_attempts"] if mode != "legacy" else 0)
    legacy = disk_budget.estimate(core.inspect_sources(None, input_files=dng_frames), None)
    assert shown["legacy_required"] == legacy["required_bytes"]
    assert shown["free_bytes"] > 0 and shown["fits"] is True


def test_stack_mode_follows_the_workflow_choice():
    assert disk_space.stack_mode(None, False) == "legacy"
    assert disk_space.stack_mode(None, True) == "positive_power2_v1"
    assert disk_space.stack_mode("C:/finite/siril-cli.exe", False) == "finite_validity_v1"
    assert disk_space.stack_mode("C:/finite/siril-cli.exe", True) == "finite_validity_v1"


def test_fitting_frames_is_the_largest_count_that_fits(dng_frames):
    sources = core.inspect_sources(None, input_files=dng_frames)
    record = disk_budget.estimate(sources, None)
    for frames in (3, 4, 57, 299):
        scaled = dict(sources, count=frames, bytes=sources["bytes"] // 4 * frames)
        need = disk_budget.estimate(scaled, None)["required_bytes"]
        assert disk_space.fitting_frames(record, need) == frames
        assert disk_space.fitting_frames(record, need - 1) == frames - 1
    assert disk_space.fitting_frames(record, 10) == 0
    assert disk_space.fitting_frames(record, 10 ** 15) == 300


def test_free_bytes_reads_the_nearest_existing_folder(tmp_path):
    assert disk_space.free_bytes(tmp_path / "not" / "made" / "yet") > 0


def _select(window, files, app):
    window.input_files = list(files)
    window._job_detached = True
    window._update_controls()
    window._disk_estimate_timer.stop()
    window._launch_disk_estimate()
    settle(window, app)


def test_estimate_is_one_short_line_beside_start(make, app, dng_frames):
    window, _ = make("zh_CN")
    window.trails_enabled.setChecked(True)
    _select(window, dng_frames, app)
    label = window.disk_estimate_label
    record = window._disk_estimate["record"]
    expected = disk_space.job_estimate(dng_frames, "legacy", window.cache_dir.text())["record"]
    assert record["required_bytes"] == expected["required_bytes"]
    assert label.isVisible() and themes.role(label) == "hint"
    # The admission requirement (what 「开始合成」 needs free), not a promise of what the job will use.
    assert label.text().startswith(f"开始需可用约 {record['required_bytes'] / 1e9:.1f} GB · 缓存盘可用 ")
    assert "\n" not in label.text()
    tip = label.toolTip()
    assert f"{record['required_bytes']:,}" in tip and "同一算法" in tip and "含安全余量" in tip
    # The review-cache note is a trails hint in step ③, not a line here.
    assert "复核" not in label.text() and window.meteor_hints.text() == "· 复核完亮迹再清理缓存。"
    # Directly above the start button, inside the fixed bottom of the step column.
    assert label.mapTo(window, label.rect().topLeft()).y() < window.start_button.mapTo(window, window.start_button.rect().topLeft()).y()
    assert not window._steps_scroll.widget().isAncestorOf(label)


def test_short_space_says_how_to_split(make, app, dng_frames, monkeypatch):
    window, _ = make("zh_CN")
    need = disk_space.job_estimate(dng_frames, "legacy", None)["record"]["required_bytes"]
    monkeypatch.setattr(disk_space, "free_bytes", lambda root: need - 1)
    _select(window, dng_frames, app)
    label = window.disk_estimate_label
    assert themes.role(label) == "error" and label.text().split("\n")[1] == "空间不够：每段最多 3 张"
    # The full advice: each segment's cache is reviewed and cleaned before the next one starts.
    assert label.toolTip().startswith("空间不够：每段不超过 3 张；复核并清理上一段的缓存后再做下一段，"
                                      "或在“设置 → 作品与缓存”换到更大的盘。\n")
    monkeypatch.setattr(disk_space, "free_bytes", lambda root: 1000)
    window._schedule_disk_estimate(force=True)
    window._disk_estimate_timer.stop()
    window._launch_disk_estimate()
    settle(window, app)
    assert label.text().split("\n")[1] == "空间不够：这块盘放不下一段"
    assert label.toolTip().startswith("空间不够，这块盘放不下一段。请在“设置 → 作品与缓存”换到更大的盘，或清理已复核的旧任务。")


@pytest.mark.parametrize("engine_path,checkbox,status,setting", [
    ("finite", False, "长序列建议改用 Siril 官方构建", "改用 Siril 官方构建：开始前要求的可用空间约为官方构建的"),
    ("", True, "长序列建议关闭“保护堆栈高光”", "关闭“保护堆栈高光”：开始前要求的可用空间约为关闭时的")])
def test_long_sequences_are_advised_against_the_range_modes(make, app, dng_frames, monkeypatch, tmp_path,
                                                            engine_path, checkbox, status, setting):
    monkeypatch.setattr(disk_space, "LONG_SEQUENCE", 3)
    window, _ = make("zh_CN")
    if engine_path:
        engine = tmp_path / "finite-siril.exe"
        engine.write_bytes(b"finite")
        window.stack_engine_path.setText(str(engine))
    window.stack_range_checkbox.setChecked(checkbox)
    _select(window, dng_frames, app)
    label = window.disk_estimate_label
    assert label.text().split("\n")[1] == status and themes.role(label) == "hint"
    ratio = window._disk_estimate["record"]["required_bytes"] / window._disk_estimate["legacy_required"]
    assert f"在“设置 → 处理后端”{setting} {ratio:.1f} 倍。" in label.toolTip() and ratio > 1
    # Short of space as well: the shortage is the one status line; the long-sequence advice stays in the tooltip.
    monkeypatch.setattr(disk_space, "free_bytes", lambda root: 1000)
    window._schedule_disk_estimate(force=True)
    window._disk_estimate_timer.stop()
    window._launch_disk_estimate()
    settle(window, app)
    assert label.text().split("\n")[1:] == ["空间不够：这块盘放不下一段"] and setting in label.toolTip()


@pytest.mark.parametrize("language", ["zh_CN", "en"])
def test_minimum_window_keeps_the_step_column_usable(make, app, dng_frames, monkeypatch, language):
    """Every estimate line at once (short space, long sequence, highlight protection, trails, wide figures) at the
    app's minimum size: the fixed bottom grows by at most two lines and the steps keep their room."""
    monkeypatch.setattr(disk_space, "LONG_SEQUENCE", 3)
    monkeypatch.setattr(gui.MainWindow, "_gigabytes", staticmethod(lambda value: "1234.5 GB"))
    window, _ = make(language)
    window.resize(960, 680)
    window.trails_enabled.setChecked(True)
    window.stack_range_checkbox.setChecked(True)
    need = disk_space.job_estimate(dng_frames, "positive_power2_v1", None)["record"]["required_bytes"]
    monkeypatch.setattr(disk_space, "free_bytes", lambda root: need - 1)
    _select(window, dng_frames, app)
    for _ in range(3):
        app.processEvents()
    label = window.disk_estimate_label
    assert (window.width(), window.height()) == (960, 680)
    assert label.isVisible() and themes.role(label) == "error" and label.text().count("\n") == 1
    metrics = label.fontMetrics()
    assert label.font().pixelSize() == 12                     # the same size in red as in grey
    assert all(metrics.horizontalAdvance(line) <= label.width() for line in label.text().split("\n")), label.text()
    assert label.height() <= 2 * metrics.lineSpacing() + 4
    shown = window._steps_scroll.viewport().height()
    assert shown >= 250, shown
    # Its whole cost, whatever else the fixed bottom shows (e.g. the JPEG size estimate): two lines and one gap.
    label.hide()
    for _ in range(3):
        app.processEvents()
    assert window._steps_scroll.viewport().height() - shown <= 2 * metrics.lineSpacing() + 4 + 10
    label.show()


def test_no_estimate_while_a_job_runs_or_with_fewer_than_three_photos(make, app, dng_frames, tmp_path):
    window, _ = make("zh_CN")
    _select(window, dng_frames[:2], app)                 # a preview is allowed; the start button names the limit
    assert not window.disk_estimate_label.isVisible() and window._disk_estimate is None
    _select(window, dng_frames, app)
    assert window.disk_estimate_label.isVisible()
    window.job_state = "running"
    window._update_controls()
    assert not window.disk_estimate_label.isVisible() and window._disk_estimate is None
    # Nothing is started while the job runs, whichever way the estimate is asked for.
    window._schedule_disk_estimate(force=True)
    window._launch_disk_estimate()
    window._refresh_disk_free()
    assert not window._disk_estimate_timer.isActive()
    assert not [name for name in window._workers if name.startswith("disk-estimate:")]
    window.job_state = ""


def test_a_finished_job_on_view_shows_what_a_new_run_needs(make, app, dng_frames, tmp_path):
    """「开始合成」 on a finished job starts a new job with its own cache (e.g. after ticking forgotten trails)."""
    window, _ = make("zh_CN")
    window.input_files = list(dng_frames)
    window.input_dir.setText(str(Path(dng_frames[0]).parent))
    window.workflow_dir = str(tmp_path)
    window._job_detached = False
    window.job_state = ""                                            # e.g. still loading
    window._update_controls()
    assert window._disk_estimate_inputs() is None and not window.disk_estimate_label.isVisible()
    window.job_state = "succeeded"
    window._update_controls()
    # 「开始合成」 stays off until the first-frame preview is refreshed; so does the estimate.
    assert not window.start_button.isEnabled() and window._disk_estimate_inputs() is None
    window._reference_data, window._reference_scene = {"image_rotation": 0}, tuple(dng_frames)
    window._update_controls()
    assert window.start_button.isEnabled()
    window._disk_estimate_timer.stop()
    window._launch_disk_estimate()
    settle(window, app)
    assert window._viewing_job() and window._disk_estimate_inputs() is not None
    assert window.disk_estimate_label.isVisible() and window.disk_estimate_label.text().startswith("开始需可用约 ")
    # An old job whose photos have moved away: nothing is said while it is only being viewed.
    window._disk_estimate = {"ok": False, "message": "gone"}
    window._show_disk_estimate()
    assert not window.disk_estimate_label.isVisible()
    window.workflow_dir = None
    window.job_state = ""


def test_free_space_is_read_again_when_the_window_is_activated(make, app, dng_frames, monkeypatch):
    window, _ = make("zh_CN")
    _select(window, dng_frames, app)
    label, record = window.disk_estimate_label, window._disk_estimate["record"]
    assert themes.role(label) == "hint"
    # Space filled elsewhere (e.g. in Explorer); the selection's own figure is not computed again.
    monkeypatch.setattr(disk_space, "free_bytes", lambda root: record["required_bytes"] - 1)
    monkeypatch.setattr(disk_space, "job_estimate", lambda *args: pytest.fail("only the free space is read"))
    monkeypatch.setattr(window, "isActiveWindow", lambda: True)
    window.changeEvent(QEvent(QEvent.Type.ActivationChange))
    settle(window, app)
    assert window._disk_estimate["record"] == record and window._disk_estimate["fits"] is False
    assert themes.role(label) == "error" and "空间不够" in label.text()


def test_another_selection_says_estimating_until_its_own_figure_arrives(make, app, dng_frames):
    window, _ = make("zh_CN")
    _select(window, dng_frames, app)
    assert window.disk_estimate_label.text().startswith("开始需可用约 ")
    window.input_files = list(dng_frames[:3])
    window._launch_disk_estimate()
    assert window._disk_estimate is None and window.disk_estimate_label.text() == "正在估算所需缓存空间…"
    settle(window, app)
    assert window.disk_estimate_label.text().startswith("开始需可用约 ")


def test_an_unreadable_selection_never_blocks_the_window(make, app, tmp_path):
    window, _ = make("en")
    missing = [str(tmp_path / ("a-long-folder-name-" * 6)) + f"gone{number}.dng" for number in range(3)]
    _select(window, missing, app)
    label = window.disk_estimate_label
    assert not window._busy and label.text().startswith("Cannot estimate the cache space needed: ")
    # One line at most (shortened in the middle); the whole reason is in the tooltip.
    assert "\n" not in label.text()
    assert label.fontMetrics().horizontalAdvance(label.text()) <= window._estimate_line_width()
    assert label.toolTip().startswith("Cannot estimate the cache space needed: ")
    assert len(label.toolTip()) >= len(label.text())
    assert window.message_label.text() == "" or "Cannot estimate" not in window.message_label.text()


def test_estimate_reads_in_english(make, app, dng_frames, monkeypatch):
    monkeypatch.setattr(disk_space, "LONG_SEQUENCE", 3)
    window, _ = make("en")
    window.trails_enabled.setChecked(True)
    window.stack_range_checkbox.setChecked(True)
    need = disk_space.job_estimate(dng_frames, "positive_power2_v1", None)["record"]["required_bytes"]
    monkeypatch.setattr(disk_space, "free_bytes", lambda root: need - 1)
    _select(window, dng_frames, app)
    for text in (window.disk_estimate_label.text(), window.disk_estimate_label.toolTip(), window.meteor_hints.text()):
        assert text and not CJK.search(text), text
    assert window.disk_estimate_label.text().startswith("Start needs ~")
    assert "Settings → Processing backends" in window.disk_estimate_label.toolTip()


def test_more_than_300_photos_explains_the_segments(make, app, tmp_path):
    window, _ = make("zh_CN")
    folder = tmp_path / "night"
    folder.mkdir()
    paths = []
    for number in range(301):
        path = folder / f"DSC_{number:04d}.dng"
        path.write_bytes(b"")
        paths.append(str(path))
    window.add_input_files(paths)
    text = window.message_label.text()
    # The 300 limit applies to every job; the 150 / 30-minute advice is named as the meteor-shower one.
    assert window.input_files == [] and text == ("一共 301 张，超过每个任务 300 张的上限，请分段处理"
                                                 "（拍流星雨建议每段约 150 张、约 30 分钟）；每段出一张图。")


# ------------------------------------------------------------------------------------ back to an old task

def _finished_record(directory, job_dir, *, trails=True, frames=144):
    return {"workflow_dir": str(directory), "state": "succeeded", "stage": "complete", "progress": 100,
            "created_at": "2026-11-15T09:01:30.671165+00:00", "stack_job_dir": str(job_dir),
            "request": {"trails_enabled": trails}, "needs_review": trails,
            "config": {"trails_enabled": trails, "input_files": [f"C:/raw/DSC_{n}.dng" for n in range(frames)]},
            "trail_review": {"state": "needs_review", "kept_count": 17, "pending_count": 1258,
                             "kept_by": {"algorithm": 17}}}


def test_reopening_a_recent_task_after_a_restart_reaches_trail_review(make, app, tmp_path, monkeypatch):
    directory = tmp_path / "cache" / "workflow-7fe7e1e10ba54266"
    job_dir = directory / "stack" / "job-d31ef0260e38b3456ebc42ba"
    job_dir.mkdir(parents=True)
    (directory / "workflow.json").write_text("{}", encoding="utf-8")
    record = _finished_record(directory, job_dir)
    monkeypatch.setattr("starstack.workflow.workflow_status", lambda path: dict(record))
    monkeypatch.setattr(gui, "load_job", lambda path: {"state": "succeeded", "stage": "complete"})
    window, prefs = make("zh_CN")
    window._workflow_submitted(dict(record))
    window.timer.stop()
    assert [entry["path"] for entry in window.recent_tasks()] == [str(directory.resolve())]
    restarted = reopen(window, prefs, make, app)
    assert restarted.workflow_dir is None
    restarted._fill_recent_menu()
    actions = [action for action in restarted.recent_menu.actions() if action.toolTip() == str(directory.resolve())]
    assert len(actions) == 1 and actions[0].isEnabled()
    assert actions[0].text() == datetime.fromisoformat(record["created_at"]).astimezone().strftime("%Y-%m-%d %H:%M") \
        + " · 144 张 · 含亮迹"
    actions[0].trigger()
    settle(restarted, app)
    restarted.timer.stop()
    assert restarted.workflow_dir == str(directory.resolve()) and restarted.job_state == "succeeded"
    assert restarted.trail_review_button.isVisible() and restarted.trail_review_button.isEnabled()
    opened, shown, given = [], [], []
    monkeypatch.setattr("starstack.trail_review_session.open_review",
                        lambda path: opened.append(path) or {"candidates": []})

    class Dialog:
        def __init__(self, session, parent, **options):
            # The triage dialog also takes the workflow record, the in-session original files and the cache
            # folder (where the composite list finds the import records of earlier sessions).
            shown.append(session)
            given.append(options)

        def exec(self):
            return QDialog.DialogCode.Rejected
    monkeypatch.setattr("aester_desktop.trail_review_dialog.TrailReviewDialog", Dialog)
    restarted.trail_review_button.click()
    settle(restarted, app)
    assert opened == [str(job_dir)] and shown == [{"candidates": []}]
    # After the restart there is no in-session conversion map; the dialog gets the window's cache folder.
    assert given[0]["originals"] == {} and given[0]["cache_root"] == str((tmp_path / "cache").absolute())
    restarted.workflow_dir = restarted.job_dir = None
    restarted.job_state = ""


def test_recent_list_is_newest_first_bounded_and_names_moved_tasks_when_chosen(make, app, tmp_path, monkeypatch):
    window, prefs = make("en")
    folders = []
    for number in range(gui.RECENT_TASKS_MAX + 2):
        folder = tmp_path / f"workflow-{number:02d}"
        folder.mkdir()
        (folder / "workflow.json").write_text("{}", encoding="utf-8")
        window._remember_task({"workflow_dir": str(folder), "created_at": "", "config": {"trails_enabled": False}})
        folders.append(str(folder.resolve()))
    window._remember_task({"workflow_dir": folders[-3], "created_at": "", "config": {}})
    paths = [entry["path"] for entry in window.recent_tasks()]
    assert len(paths) == gui.RECENT_TASKS_MAX and paths[0] == folders[-3] and len(set(paths)) == len(paths)
    (Path(folders[-1]) / "workflow.json").unlink()                    # moved, deleted or on a drive not connected
    # Opening the menu looks at no folder (a disconnected drive would freeze it): every entry is listed.
    looked = []
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: looked.append(self) or real_is_file(self))
    window._fill_recent_menu()
    assert looked == []
    tips = [action.toolTip() for action in window.recent_menu.actions()
            if action.toolTip() and action.toolTip() != action.text()]
    assert tips == paths and folders[-3] == tips[0]
    labels = [action.text() for action in window.recent_menu.actions()]
    assert labels[-1] == "Clear list" and not any(CJK.search(text) for text in labels)
    # Choosing the missing one says so and keeps the window as it was; it stays in the list.
    loaded = []
    monkeypatch.setattr(window, "load_existing_job", lambda path: loaded.append(path))
    [action for action in window.recent_menu.actions() if action.toolTip() == folders[-1]][0].trigger()
    assert loaded == [] and window.message_label.text() == (
        "This job cannot be found: it may have been moved or deleted, or its drive is not connected.")
    assert folders[-1] in [entry["path"] for entry in window.recent_tasks()]
    [action for action in window.recent_menu.actions() if action.toolTip() == folders[-3]][0].trigger()
    assert loaded == [folders[-3]]
    window.recent_menu.actions()[-1].trigger()
    assert window.recent_tasks() == [] and prefs.value(gui.RECENT_TASKS_KEY) is None
    window._fill_recent_menu()
    assert window.recent_menu.actions()[0].text() == "No recent jobs yet"


def test_open_job_dialog_starts_in_the_cache(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    (tmp_path / "cache").mkdir()
    seen = []
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda parent, title, start, filters: seen.append(start) or ("", ""))
    window._pick_job_file()
    assert seen == [str(tmp_path / "cache")]
