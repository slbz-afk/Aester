"""Desktop routing and restart-only localization through actual Qt controls."""
import ast
import re
from pathlib import Path
from string import Formatter

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QLabel, QAbstractButton, QComboBox, QLineEdit

from starstack import i18n
from aester_desktop import gui
from aester_desktop.translations_gui import EN


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(app, tmp_path, monkeypatch):
    monkeypatch.delenv('AESTER_LANGUAGE', raising=False)
    monkeypatch.delenv('AESTER_INSTALL_ROOT', raising=False)
    monkeypatch.setattr(gui.MainWindow, 'check_backend', lambda self: None)
    windows = []
    def make(language='zh_CN', settings=None):
        prefs = settings or QSettings(str(tmp_path / f'{len(windows)}.ini'), QSettings.Format.IniFormat)
        if settings is None:
            prefs.setValue('appearance/language', language)
        window = gui.MainWindow(preferences=prefs)
        window.cache_dir.setText(str(tmp_path / 'cache'))
        window.output_dir.setText(str(tmp_path / 'photos'))
        windows.append(window)
        window.show()
        app.processEvents()
        return window, prefs
    yield make
    for window in windows:
        window._busy = False
        window.job_state = ''
        window._color_timer.stop()
        window.timer.stop()
        for dialog in window._advanced_dialogs.values():
            dialog.close()
        window.close()
    app.processEvents()
    i18n.set_language('zh_CN')


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_settings_connection_opens_connection_and_help_opens_help(make_window, app, language):
    window, _ = make_window(language)
    settings = window.menu_actions['settings'].parent()
    connection = window.menu_actions['assistant']
    assert window.menu_actions['connect_ai'] is connection
    assert sum(action is connection for action in settings.actions()) == 1
    assert sum(action.text() == connection.text() for action in settings.actions()) == 1
    connection.trigger()
    app.processEvents()
    assert window._advanced_dialogs['connect_ai'].isVisible()
    assert 'help' not in window._advanced_dialogs
    window._advanced_dialogs['connect_ai'].hide()
    window.menu_actions['help_features'].trigger()
    app.processEvents()
    assert window._advanced_dialogs['help'].isVisible()
    assert not window._advanced_dialogs['connect_ai'].isVisible()


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_one_storage_entry_opens_storage_not_help(make_window, app, language):
    window, _ = make_window(language)
    settings = window.menu_actions['settings'].parent()
    entry = window.menu_actions['storage']
    assert sum(a.text() == entry.text() for a in settings.actions()) == 1
    entry.trigger()
    app.processEvents()
    assert window._advanced_dialogs['storage'].isVisible()
    assert 'help' not in window._advanced_dialogs


def test_language_selection_persists_without_mutating_active_edit_or_task(make_window, app):
    window, prefs = make_window()
    window.warmth_slider.setValue(27)
    window.workflow_milkyway_offset.setValue(-17)
    window._color_timer.stop()
    window.input_files = ['keep.nef']
    project = {'scene_mode': 'landscape', 'milkyway': {'status': 'unknown'}}
    window._regions_project = project
    window.job_state = 'running'
    window._busy = True
    before = (window.color_adjustments(), list(window.input_files), window._regions_project,
              window.workflow_milkyway_offset.value())
    window.menu_actions['language_en'].trigger()
    app.processEvents()
    assert prefs.value('appearance/language') == 'en'
    assert i18n.get_language() == 'zh_CN'
    assert window.start_button.text() == '开始合成'
    assert '下次启动' in window._language_notice.text()
    assert window.job_state == 'running' and window._busy
    assert before == (window.color_adjustments(), window.input_files, window._regions_project,
                      window.workflow_milkyway_offset.value())
    # No process restart is needed to model the next window's preference load.
    window.job_state = ''
    window._busy = False
    reopened, _ = make_window(settings=prefs)
    assert reopened.start_button.text() == 'Start stacking'
    assert reopened.menu_actions['language_en'].isChecked()
    reopened.menu_actions['language_zh_CN'].trigger()
    assert prefs.value('appearance/language') == 'zh_CN'
    assert 'next open Aester' in reopened._language_notice.text()


def test_english_has_stable_processing_values_and_translated_status(make_window, monkeypatch):
    window, _ = make_window('en')
    assert [a.text() for a in window.menuBar().actions()] == ['File', 'Settings', 'Help']
    assert [window.scene_mode.itemData(i) for i in range(window.scene_mode.count())] == ['unknown', 'sky', 'landscape']
    assert window.model_channel_order.currentData() == 'rgb'
    assert window.color_adjustments() == {'warmth': 0., 'tint': 0., 'exposure_ev': 0.}
    window.workflow_dir = 'synthetic-path-read-is-stubbed'
    monkeypatch.setattr('starstack.workflow.workflow_status', lambda *_: {'state': 'running', 'stage': 'stack', 'progress': 40})
    window.refresh_workflow()
    assert window.stage_label.text() == 'Aligning and stacking'
    window.job_state = ''
    window.workflow_dir = None


def test_default_english_widgets_have_no_untranslated_chinese(make_window):
    window, _ = make_window('en')
    leftovers = []
    for obj in window.findChildren(QLabel) + window.findChildren(QAbstractButton) + window.findChildren(QLineEdit):
        for text in [obj.text(), obj.toolTip(), obj.accessibleName()]:
            if re.search('[一-龥]', text):
                leftovers.append((obj.objectName(), text))
    for obj in window.findChildren(QComboBox):
        for index in range(obj.count()):
            if re.search('[一-龥]', obj.itemText(index)):
                leftovers.append((obj.objectName(), obj.itemText(index)))
    assert leftovers == []


def test_english_worker_status_uses_stage_and_keeps_original_diagnostics(make_window, monkeypatch):
    window, _ = make_window('en')
    record = {'state': 'running', 'stage': 'stack', 'progress': 40, 'message': '正在堆栈：3 / 6 帧'}
    window.workflow_dir = 'synthetic-path-read-is-stubbed'
    monkeypatch.setattr('starstack.workflow.workflow_status', lambda *_: record)
    window.refresh_workflow()
    assert window.message_label.text() == 'Aligning and stacking'
    assert window._workflow['message'] == record['message']
    window.show_job_details()
    assert record['message'] in window.job_details.toPlainText()
    record.update(error='具体后端诊断保持原文', state='failed')
    window.refresh_workflow()
    assert window.message_label.text() == 'Operation incomplete: 具体后端诊断保持原文'
    window.job_state = ''
    window.workflow_dir = None


def test_english_submission_preserves_workflow_values(make_window, monkeypatch, tmp_path):
    from test_p1_gui_routing_independent import scene
    window, _ = make_window('en')
    scene(window, monkeypatch, tmp_path, present=True)
    window.warmth_slider.setValue(23)
    window.tint_slider.setValue(-7)
    window.exposure_slider.setValue(4)
    window._color_timer.stop()
    window.stars_enabled.setChecked(True)
    window.workflow_milkyway_offset.setValue(-30)
    calls = []
    monkeypatch.setattr(window, '_run_async', lambda *args: calls.append(args))
    try:
        window.start_processing()
        assert len(calls) == 1
        admitted = calls[0][1]()  # Real admission, synthetic engine/spawn supplied by scene().
        assert admitted['request']['input_files'] == window.input_files
        assert admitted['request']['color_adjustments'] == {'warmth': 23., 'tint': -7., 'exposure_ev': .4}
        assert admitted['request']['milkyway_offset'] == -.3
        assert {key: admitted['request']['backend_options'][key]
                for key in ('channel_order', 'value_range', 'layout', 'stitch')} == {
                    'channel_order': 'rgb', 'value_range': '0..1', 'layout': 'nhwc', 'stitch': 'blend'}
        assert admitted['request']['trails_enabled'] is False
    finally:
        window._busy = False


def test_english_unknown_failure_keeps_diagnostic(make_window):
    make_window('en')
    message = gui.MainWindow._status_message({'message': '后端故障详细原因'}, 'Submission failed')
    assert message == 'Submission failed\n后端故障详细原因'


def test_catalog_preserves_format_fields_and_covers_static_translations():
    def fields(text):
        return [(field, spec, conversion) for _, field, spec, conversion in Formatter().parse(text) if field is not None]
    for source, english in EN.items():
        assert english.strip(), source
        assert fields(source) == fields(english), source
    for filename in ('gui.py', 'experience.py'):
        tree = ast.parse((Path(gui.__file__).parent / filename).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tr' and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and re.search('[一-龥]', arg.value):
                    assert arg.value in EN, (filename, node.lineno, arg.value)
