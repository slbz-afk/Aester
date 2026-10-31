"""End to end: backend Chinese messages reach an English window translated.

Records and exceptions stay Chinese (the core pins them); the window renders
them through ``i18n.translate_message``. The Chinese window stays unchanged.
"""
import json
import re

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from starstack import i18n
from aester_desktop import gui

CJK = re.compile(r'[　-〿㐀-䶿一-鿿豈-﫿＀-￯]')
PREFIX = 'Operation incomplete: '


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(app, tmp_path, monkeypatch):
    monkeypatch.delenv('AESTER_LANGUAGE', raising=False)
    monkeypatch.delenv('AESTER_INSTALL_ROOT', raising=False)
    monkeypatch.setattr(gui.QApplication, 'alert', staticmethod(lambda widget, duration=0: None))
    # The real check_backend runs; only the probes it calls are replaced.
    monkeypatch.setattr(gui, 'doctor', lambda selected=None: {
        'available': False, 'executable': None, 'version': None, 'message': '尚未找到 Siril 1.4.x，请选择 siril-cli.exe'})
    monkeypatch.setattr('aester_desktop.backend_preferences.discover_finite_backend', lambda *a, **k: {
        'available': False, 'reason': '有限值堆栈引擎或运行时清单未验证；请使用已核验的完整运行时目录'})
    windows = []

    def make(language):
        prefs = QSettings(str(tmp_path / f'{len(windows)}.ini'), QSettings.Format.IniFormat)
        prefs.setValue('appearance/language', language)
        window = gui.MainWindow(preferences=prefs)
        window.cache_dir.setText(str(tmp_path / 'cache'))
        window.output_dir.setText(str(tmp_path / 'photos'))
        windows.append(window)
        settle(window, app)
        return window

    yield make
    for window in windows:
        window._busy = False
        window.job_state = ''
        window.workflow_dir = window.job_dir = None
        window._color_timer.stop()
        window.timer.stop()
        for dialog in window._advanced_dialogs.values():
            dialog.close()
        window.close()
    app.processEvents()
    i18n.set_language('zh_CN')


def settle(window, app):
    for _ in range(3):              # deferred starts (QTimer 0) run first, then their workers finish
        app.processEvents()
        window._pool.waitForDone(30000)
        app.processEvents()


def workflow(window, app, monkeypatch, record):
    monkeypatch.setattr('starstack.workflow.workflow_status', lambda *_: {'artifacts': [], **record})
    window.workflow_dir = window.job_dir = 'unused'
    window.timer.start()
    window.refresh_workflow()
    return window.message_label.text()


def job(window, tmp_path, record):
    directory = tmp_path / f'job-{len(list(tmp_path.glob("job-*")))}'
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps({'artifacts': [], **record}, ensure_ascii=False), encoding='utf-8')
    window.workflow_dir, window.job_dir = None, str(directory)
    window.timer.start()
    window.refresh_job()
    return window.message_label.text()


def english(text):
    assert text and not CJK.search(text), text
    return text


FAILED_WORKFLOW = {'state': 'failed', 'stage': 'stack', 'progress': 40, 'message': '处理失败，可核验后恢复',
                   'error': 'RuntimeError: 堆栈作业未成功：实际参与堆栈不足 3 帧，请检查配准日志'}
CANCELLED_WORKFLOW = {'state': 'cancelled', 'stage': 'stack', 'progress': 40, 'message': '已停止，可从完整阶段恢复',
                      'error': '用户请求停止'}
RUNNING_WORKFLOW = {'state': 'running', 'stage': 'stack', 'progress': 40, 'message': 'Siril 正在对齐星空并堆栈'}
AUTOMATIC_TRAILS = {'state': 'running', 'stage': 'publish', 'progress': 98,
                    'message': '基础成片已保存；正在自动加回量测可靠的流星候选'}


def test_failed_cancelled_and_running_workflow_records(make_window, app, monkeypatch):
    window = make_window('en')
    text = english(workflow(window, app, monkeypatch, FAILED_WORKFLOW))
    assert text == (PREFIX + 'RuntimeError: Stacking did not finish: '
                    'Fewer than 3 frames actually took part in the stack. Please check the registration log.')
    text = english(workflow(window, app, monkeypatch, CANCELLED_WORKFLOW))
    assert text == 'Stop requested by user' and not text.startswith(PREFIX)
    assert english(workflow(window, app, monkeypatch, RUNNING_WORKFLOW)) == 'Siril is aligning the stars and stacking'
    assert english(workflow(window, app, monkeypatch, AUTOMATIC_TRAILS)) == (
        'Base photograph saved; adding back reliably measured meteor candidates')


def test_chinese_window_shows_records_unchanged(make_window, app, monkeypatch):
    window = make_window('zh_CN')
    assert workflow(window, app, monkeypatch, FAILED_WORKFLOW) == FAILED_WORKFLOW['error']
    assert workflow(window, app, monkeypatch, CANCELLED_WORKFLOW) == '用户请求停止'
    assert workflow(window, app, monkeypatch, RUNNING_WORKFLOW) == RUNNING_WORKFLOW['message']
    assert window.backend_status.text().endswith('\n尚未找到 Siril 1.4.x，请选择 siril-cli.exe')


def test_failed_and_cancelled_job_records(make_window, app, tmp_path):
    window = make_window('en')
    text = english(job(window, tmp_path, {'state': 'failed', 'stage': 'stack', 'progress': 40,
                                          'message': '处理失败，可查看日志或恢复',
                                          'error': '实际参与堆栈不足 3 帧，请检查配准日志'}))
    assert text.startswith(PREFIX)
    text = english(job(window, tmp_path, {'state': 'cancelled', 'stage': 'cancelled', 'progress': 40,
                                          'message': '已停止，可从完整阶段恢复', 'error': None}))
    assert text == 'Stopped. You can resume from the last complete stage.'


def test_status_read_failure_is_translated(make_window, app, monkeypatch):
    window = make_window('en')

    def unreadable(*_):
        raise ValueError('工作流参数或输入绑定已改变')
    monkeypatch.setattr('starstack.workflow.workflow_status', unreadable)
    window.workflow_dir = 'unused'
    window.refresh_workflow()
    assert english(window.message_label.text()).endswith('The workflow parameters or input binding have changed')


def test_failed_editing_project_open(make_window, app, tmp_path):
    window = make_window('en')
    wrong = tmp_path / 'other.aester.json'
    wrong.write_text('{"schema": "something-else"}', encoding='utf-8')
    window.open_editing_project(str(wrong))
    settle(window, app)
    assert english(window.message_label.text()) == PREFIX + 'Not an Aester editing project'
    window.open_editing_project(str(tmp_path / 'missing.aester.json'))     # localized OS error text
    settle(window, app)
    assert english(window.message_label.text()).startswith(PREFIX)


def test_siril_doctor_message_and_failures(make_window, app, monkeypatch):
    window = make_window('en')
    assert english(window.backend_status.text()).endswith('\nSiril 1.4.x not found yet. Please choose siril-cli.exe')
    assert english(window.finite_backend_note.toolTip()).startswith('The range-preserving stacking engine')

    def refused(selected=None):
        raise PermissionError(13, '拒绝访问。', None, 5)        # as a Chinese Windows reports it
    monkeypatch.setattr(gui, 'doctor', refused)
    window.check_backend()
    settle(window, app)
    assert english(window.backend_status.text()) == 'Siril check incomplete: [WinError 5] Access is denied'


def test_star_density_reasons(make_window, app, tmp_path, monkeypatch):
    window = make_window('en')
    separation = tmp_path / 'result.json'
    separation.write_text('{}', encoding='utf-8')
    window.star_separation.setText(str(separation))
    reasons = {True: '背景检测到120个紧致星点候选，按显眼程度加权约350单位/百万参考像素；建议背景55%。'
                     '结合银河密度建议低5至25个百分点，以渐变选区过渡；不保证两区星数相等。',
               False: '背景有效面积或可靠星点不足，暂不给自动力度；仍可手动调整。'}
    for available in (False, True):
        recommendation = {'available': available, 'reason': reasons[available],
                          'background_strength': .55 if available else None,
                          'milkyway_offset': -.1 if available else None,
                          'milkyway_strength': .45 if available else None}
        monkeypatch.setattr('starstack.star_reduction.suggest_star_reduction',
                            lambda *a, _r=recommendation, **k: {'state': 'succeeded', 'recommendation': _r})
        window.analyze_star_density()
        settle(window, app)
        text = english(window.star_status.text())
        assert ('Found 120 compact star candidates' in text) if available else ('Not enough valid background' in text)
    window.apply_star_suggestion()
    assert english(window.star_status.text()).startswith('Suggestion applied.')
