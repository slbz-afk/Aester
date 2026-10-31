"""User regressions: input reset, numerical field hit areas and one final view."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox
from test_gui_language_routing import app, make_window


def test_clear_selection_removes_marks_and_stale_preview_not_files(make_window, tmp_path, monkeypatch):
    w, _ = make_window()
    files = [tmp_path/f'frame{i}.dng' for i in range(3)]
    for p in files:
        p.write_bytes(b'original sentinel')
    w.input_files = [str(p) for p in files]
    w._selection_changed(preview=False)
    w._regions_project = {'sentinel': True}
    w._regions_path = 'old-region.json'
    w._reference_data = {'preview_path': 'old.png'}
    w._reference_scene = tuple(w.input_files)
    generation = w._reference_generation
    w._reference_pending = True
    w._reference_running = True
    w.clear_files_button.click()
    assert not w.input_files and w.file_strip.count() == 0
    assert w._regions_project is None and w._regions_path is None
    assert w._reference_data is None and w._reference_scene == ()
    assert w._reference_generation > generation
    assert not w._reference_pending and not w._color_timer.isActive()
    assert not w.clear_files_button.isEnabled()
    assert all(p.read_bytes() == b'original sentinel' for p in files)
    w._reference_running = False


@pytest.mark.parametrize('state,busy,closing', [('running',False,False),('queued',False,False),('',True,False),('',False,True)])
def test_clear_cannot_change_active_or_closing_selection(make_window,state,busy,closing):
    w, _ = make_window()
    w.input_files = ['a.dng', 'b.dng', 'c.dng']
    w.job_state, w._busy, w._closing_requested = state, busy, closing
    w.clear_input_files()
    assert w.input_files == ['a.dng', 'b.dng', 'c.dng']
    w._closing_requested = False


@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_spin_selected_text_does_not_overlap_or_steal_step_buttons(make_window,app,theme,language):
    w, _ = make_window(language)
    w.set_theme(theme)
    box = w.workflow_milkyway_offset
    # Exercise the actual control independently of whether a Milky Way is marked.
    box.setParent(w)
    box.resize(230, 44)
    box.setEnabled(True)
    box.show()
    box.setValue(0)
    box.lineEdit().selectAll()
    app.processEvents()
    option = QStyleOptionSpinBox()
    box.initStyleOption(option)
    style = box.style()
    up = style.subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxUp, box)
    down = style.subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxDown, box)
    field = box.lineEdit().geometry()
    assert not field.intersects(up) and not field.intersects(down), (field, up, down)
    assert up.width() >= 20 and down.width() >= 20
    QTest.mouseClick(box, Qt.MouseButton.LeftButton, pos=up.center())
    assert box.value() == box.singleStep()
    box.lineEdit().selectAll()
    QTest.mouseClick(box, Qt.MouseButton.LeftButton, pos=down.center())
    assert box.value() == 0
    box.hide()
    box.setParent(w)


def test_workflow_does_not_show_second_star_result_entry(make_window):
    w, _ = make_window()
    w._star_result = {'state': 'succeeded'}
    w.workflow_dir = 'workflow'
    w._update_controls()
    assert w.stars_view_button.isHidden()
    w.workflow_dir = None
    w._update_controls()
    assert not w.stars_view_button.isHidden()


def test_backend_selection_sets_discovered_path_and_preserves_explicit_mode(make_window, tmp_path):
    w,prefs=make_window()
    exe=tmp_path/'finite.exe';exe.write_bytes(b'fixture')
    w._set_finite_backend({'available':True,'path':str(exe)})
    assert not w.stack_engine_path.text()  # Discovery itself is not selection.
    w.stack_backend_choice.setCurrentIndex(w.stack_backend_choice.findData('finite'))
    assert w.stack_engine_path.text()==str(exe)
    w._save_preferences()
    reopened,_=make_window(settings=prefs)
    assert reopened.stack_backend_choice.currentData()=='finite'
    assert reopened.stack_engine_path.text()==str(exe)
    reopened.stack_backend_choice.setCurrentIndex(reopened.stack_backend_choice.findData('official'))
    assert not reopened.stack_engine_path.text()
    reopened.stack_engine_path.setText(str(exe.with_name('custom.exe')))
    reopened.stack_engine_path.editingFinished.emit()
    assert reopened.stack_backend_choice.currentData()=='custom'


def test_missing_selected_backend_does_not_silently_start_official(make_window, monkeypatch):
    w,_=make_window();calls=[]
    monkeypatch.setattr(w,'_run_async',lambda *a:calls.append(a))
    w.stack_backend_choice.setCurrentIndex(w.stack_backend_choice.findData('finite'))
    w.start_processing()
    assert not calls and '尚未安装' in w.message_label.text()


def test_gui_compatibility_import_changes_selection_only_after_success(make_window, tmp_path, monkeypatch):
    from starstack import raw_import
    w,_=make_window();calls=[]
    sources=[tmp_path/f'photo{i}.nef' for i in range(3)]
    converted=[tmp_path/'cache'/f'photo{i}.dng' for i in range(3)]
    for p in sources:p.write_bytes(b'original')
    def prepare(files, cache, **kw):
        assert files==list(map(str,sources))
        assert callable(kw['cancelled'])
        return {'input_files':list(map(str,converted)),'originals':files,'converted':True}
    monkeypatch.setattr(raw_import,'prepare_import',prepare)
    monkeypatch.setattr(w,'_run_async',lambda *a:calls.append(a))
    w.add_input_files(list(map(str,sources)))
    assert w._busy and w.cancel_button.isEnabled() and not w.input_files
    calls[0][2](calls[0][1]())
    w._color_timer.stop()
    assert w.input_files==list(map(str,converted)) and not w._busy
    assert w._import_originals[str(converted[0])]==str(sources[0])
    assert str(sources[0]) in w.file_strip.item(0).toolTip()
    w.clear_input_files()
    assert not w._import_originals and all(p.read_bytes()==b'original' for p in sources)


def test_connection_config_pins_converter_from_host_not_agent(tmp_path):
    from starstack.onboarding import connection_config
    inp=tmp_path/'photos';inp.mkdir()
    converter=tmp_path/'converter.exe';converter.write_bytes(b'host fixture')
    config=connection_config(inp,tmp_path/'delivery',tmp_path/'cache',dng_converter_path=converter)
    assert config['mcpServers']['aester']['env']['AESTER_DNG_CONVERTER']==str(converter.resolve())
