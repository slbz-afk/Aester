"""The 作品与缓存 page: the job list, idle-only cleaning, the confirmation and the cleaned-job review button.

Real Qt widgets and the real cleanup on a synthetic finished workflow (tests/test_cache_cleanup.py).
"""
from pathlib import Path
import json
import os
import re
import threading

import pytest
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from starstack import cache_cleanup, i18n, storage
from aester_desktop import experience as experience_module, gui
from aester_desktop.translations_gui import EN
from test_workflow import setup  # noqa: F401  (fixture)
from test_cache_cleanup import ALL_FRAMES, finished, frames_exist  # noqa: F401  (fixture)

CJK = re.compile(r'[　-〿㐀-䶿一-鿿豈-﫿＀-￯]')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_window(app, tmp_path, monkeypatch, finished):
    monkeypatch.delenv('AESTER_LANGUAGE', raising=False)
    monkeypatch.delenv('AESTER_INSTALL_ROOT', raising=False)
    monkeypatch.setattr(gui.MainWindow, 'check_backend', lambda self: None)
    monkeypatch.setattr(gui.QApplication, 'alert', staticmethod(lambda widget, duration=0: None))
    # The finished job sits in the default cache root; the window's preference names another folder.
    monkeypatch.setattr(storage, 'default_cache_root', lambda: finished['directory'].parent)
    windows = []

    def make(language='zh_CN'):
        prefs = QSettings(str(tmp_path / f'{len(windows)}.ini'), QSettings.Format.IniFormat)
        prefs.setValue('appearance/language', language)
        window = gui.MainWindow(preferences=prefs)
        window.cache_dir.setText(str(tmp_path / 'preferred-cache'))
        window.output_dir.setText(str(tmp_path / 'photos'))
        windows.append(window)
        window.show()
        settle(window, app)
        return window

    yield make
    for window in windows:
        window._busy = False
        window.job_state = ''
        window.workflow_dir = window.job_dir = None
        window._color_timer.stop()
        window.timer.stop()
        settle(window, app)
        for dialog in window._advanced_dialogs.values():
            dialog.close()
        window.close()
    app.processEvents()
    i18n.set_language('zh_CN')


def settle(window, app):
    for _ in range(4):
        app.processEvents()
        window._pool.waitForDone(30000)
        app.processEvents()


def open_page(window, app):
    window.experience.show_storage()
    settle(window, app)
    return window.experience


def test_page_lists_jobs_of_the_default_root_and_cleans_only_when_idle(make_window, app, finished):
    window = make_window()
    page = open_page(window, app)
    assert page.cache_jobs.topLevelItemCount() == 1
    item = page.cache_jobs.topLevelItem(0)
    plan = item.data(0, Qt.ItemDataRole.UserRole)
    assert plan['workflow_dir'] == str(finished['directory']) and plan['state'] == 'cleanable'
    assert item.text(0) == f"{plan['job']['name']} · 3 张" and item.text(5) == '可清理'
    assert item.text(2) == finished['done']['delivery_dir'] and item.text(4).endswith('GB')
    assert item.flags() & Qt.ItemFlag.ItemIsSelectable
    assert not page.clean_cache_button.isEnabled()                   # nothing selected yet
    item.setSelected(True)
    assert page.clean_cache_button.isEnabled()
    window._busy = True
    window._update_controls()
    assert not page.clean_cache_button.isEnabled()
    window._busy = False
    window.job_state = 'running'
    window._update_controls()
    assert not page.clean_cache_button.isEnabled() and page.clean_cache_button.toolTip()
    window.job_state = ''
    window._update_controls()
    assert page.clean_cache_button.isEnabled()
    assert '1 个任务可清理' in page.cache_status.text()


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_confirmation_states_space_what_is_kept_and_lost_review(make_window, app, finished, language):
    window = make_window(language)
    page = open_page(window, app)
    item = page.cache_jobs.topLevelItem(0)
    item.setSelected(True)
    plan = item.data(0, Qt.ItemDataRole.UserRole)
    seen = []

    def answer():
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox):
            QTimer.singleShot(10, answer)
            return
        seen.append({'title': box.windowTitle(), 'text': box.text(), 'buttons': [b.text() for b in box.buttons()],
                     'default': box.defaultButton().text()})
        box.defaultButton().click()                                  # the default keeps every file
    QTimer.singleShot(0, answer)
    page.clean_cache_button.click()
    settle(window, app)
    assert len(seen) == 1
    text = seen[0]['text']
    assert f"{plan['bytes_deletable'] / 1024**3:.1f} GB" in text
    if language == 'en':
        assert not CJK.search(text + seen[0]['title'] + ''.join(seen[0]['buttons']))
        for phrase in ('freeing about', 'photographs', 'originals', 'trail candidates can no longer be reviewed'):
            assert phrase in text
        assert sorted(seen[0]['buttons']) == ['Clean intermediate files', 'Keep files'] and seen[0]['default'] == 'Keep files'
    else:
        for phrase in ('约释放', '作品', '原片', '都会保留', '不能再复核亮迹候选'):
            assert phrase in text
        assert sorted(seen[0]['buttons']) == sorted(['清理中间文件', '暂不清理']) and seen[0]['default'] == '暂不清理'
    assert len(frames_exist(finished['child'])) == len(ALL_FRAMES) and not window._busy


def test_cleaning_runs_off_the_ui_thread_while_busy_then_refreshes(make_window, app, finished, monkeypatch):
    window = make_window()
    page = open_page(window, app)
    page.cache_jobs.topLevelItem(0).setSelected(True)
    asked = []
    monkeypatch.setattr(experience_module, 'ask_cleanup', lambda parent, text: asked.append(text) or True)
    real, seen = cache_cleanup.clean_workflow_cache, {}

    def spy(directory, plan_id):
        seen.update(main_thread=threading.current_thread() is threading.main_thread(), busy=window._busy)
        return real(directory, plan_id)
    monkeypatch.setattr(cache_cleanup, 'clean_workflow_cache', spy)
    refreshed = []
    window.workflow_dir = str(finished['directory'])
    monkeypatch.setattr(window, 'refresh_workflow', lambda: refreshed.append(window._busy))
    page.clean_cache_button.click()
    assert asked and window._busy and page._cleaning_cache
    assert not page.clean_cache_button.isEnabled() and not page.refresh_cache_button.isEnabled()
    settle(window, app)
    assert seen == {'main_thread': False, 'busy': True}
    assert not window._busy and not page._cleaning_cache and refreshed == [False]
    assert frames_exist(finished['child']) == []
    assert '已释放' in page.cache_result.text()
    item = page.cache_jobs.topLevelItem(0)
    assert item.text(5) == '已清理' and item.text(4) == '—'
    assert not item.flags() & Qt.ItemFlag.ItemIsSelectable and not page.clean_cache_button.isEnabled()


def test_refused_jobs_show_the_reason_and_cannot_be_selected(make_window, app, finished):
    (finished['delivery'] / 'final.jpg').write_bytes(b'edited elsewhere')
    window = make_window('en')
    page = open_page(window, app)
    item = page.cache_jobs.topLevelItem(0)
    assert item.text(5).startswith('Cannot clean: ') and 'did not verify' in item.text(5)
    assert not CJK.search(item.text(5)) and item.toolTip(5) == item.text(5)
    assert not item.flags() & Qt.ItemFlag.ItemIsSelectable


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_trail_review_is_disabled_and_explained_for_a_cleaned_job(make_window, app, tmp_path, language, monkeypatch):
    window = make_window(language)
    window.workflow_dir = window.job_dir = str(tmp_path)
    window.job_state = 'succeeded'
    window._workflow = {'state': 'succeeded', 'request': {'trails_enabled': True}}
    window._update_controls()
    button = window.trail_review_button
    assert not button.isHidden() and button.isEnabled() and button.toolTip() == ''
    window._workflow['cache_cleaned'] = True
    window._update_controls()
    assert not button.isHidden() and not button.isEnabled()
    explanation = button.toolTip()
    assert ('不能再复核亮迹候选' in explanation) if language == 'zh_CN' else (
        'can no longer be reviewed' in explanation and not CJK.search(explanation))
    started = []
    monkeypatch.setattr(window, '_run_async', lambda *args: started.append(args))
    window.open_trail_review()
    assert started == [] and window.message_label.text() == explanation
    note = window._products_note()
    assert ('亮迹候选不能再复核' in note) if language == 'zh_CN' else ('no longer be reviewed' in note)


def test_page_texts_have_english():
    for text in (*experience_module.CACHE_COLUMNS, *experience_module.CACHE_STATES.values()):
        assert text in EN and not CJK.search(EN[text]), text


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_confirmation_names_the_kept_copies_of_moved_originals(make_window, app, finished, language):
    job = json.loads((finished['child'] / 'job.json').read_text(encoding='utf-8'))
    for item in job['inputs']:
        Path(item['path']).unlink()
    window = make_window(language)
    page = open_page(window, app)
    plan = page.cache_jobs.topLevelItem(0).data(0, Qt.ItemDataRole.UserRole)
    assert plan['state'] == 'cleanable' and plan['stage_copies_kept'] == 3
    text, plain = page.cleanup_confirmation([plan]), page.cleanup_confirmation([dict(plan, stage_copies_kept=0)])
    if language == 'en':
        assert '3 originals are no longer at their recorded location' in text and 'kept, not deleted' in text
        assert not CJK.search(text) and 'kept, not deleted' not in plain
    else:
        assert '有 3 张原片已不在原来的位置' in text and '会保留，不会删除' in text and '不会删除' not in plain


def blocking_listing(monkeypatch):
    """list_workflow_caches that waits for ``gate`` on its first call; records each call's roots and cancel state."""
    real, gate, started, calls = cache_cleanup.list_workflow_caches, threading.Event(), threading.Event(), []

    def slow(roots, *, cancelled=None):
        calls.append({'roots': [os.path.normcase(str(root)) for root in roots]})
        if len(calls) == 1:
            started.set()
            assert gate.wait(30)
        calls[-1]['cancelled'] = bool(cancelled and cancelled())
        return real(roots, cancelled=cancelled)
    monkeypatch.setattr(cache_cleanup, 'list_workflow_caches', slow)
    return gate, started, calls


def test_a_refresh_during_a_listing_lists_the_new_cache_folder_afterwards(make_window, app, finished, monkeypatch, tmp_path):
    window = make_window()
    page = open_page(window, app)
    gate, started, calls = blocking_listing(monkeypatch)
    page.refresh_cache_jobs()
    assert started.wait(30) and page._listing_cache
    other = tmp_path / 'other-cache'
    other.mkdir()
    window.cache_dir.setText(str(other))
    page.refresh_cache_jobs()                     # e.g. the cache folder was changed while the list was built
    assert page._cache_refresh_pending
    gate.set()
    settle(window, app)
    assert len(calls) == 2 and not page._listing_cache and not page._cache_refresh_pending
    assert os.path.normcase(str(other)) not in calls[0]['roots'] and os.path.normcase(str(other)) in calls[1]['roots']
    assert page.cache_jobs.topLevelItemCount() == 1 and '1 个任务可清理' in page.cache_status.text()


@pytest.mark.parametrize('how', ['page', 'window'])
def test_closing_the_page_or_the_window_stops_the_listing(make_window, app, finished, monkeypatch, how):
    window = make_window()
    page = open_page(window, app)
    gate, started, calls = blocking_listing(monkeypatch)
    page.refresh_cache_jobs()
    assert started.wait(30)
    if how == 'page':
        window._advanced_dialogs['storage'].close()
    else:
        window.close()                            # waits for the pool, but the listing stops between jobs
    gate.set()
    settle(window, app)
    assert len(calls) == 1 and calls[0]['cancelled'] is True
    if how == 'page':
        assert not page._listing_cache and '列表检查已停止' in page.cache_status.text()
        assert page.cache_jobs.topLevelItemCount() == 1      # the earlier list stays
