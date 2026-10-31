"""Independent real-Qt scheduling probes with controlled expensive work."""
import os
from pathlib import Path
import time

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication
from starstack import references
from aester_desktop import gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app,predicate,timeout=2):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate():return
        time.sleep(.005)
    pytest.fail('Qt state transition did not complete in time')


def wait_ms(app,ms):
    deadline=time.monotonic()+ms/1000
    while time.monotonic()<deadline:
        app.processEvents();time.sleep(.005)


@pytest.fixture
def panel(app,monkeypatch,tmp_path):
    monkeypatch.setenv('AESTER_LANGUAGE','en')
    monkeypatch.setattr(gui.MainWindow,'check_backend',lambda self:None)
    w=gui.MainWindow(persist_preferences=False)
    w.output_dir.setText(str(tmp_path/'photographs'))
    w.cache_dir.setText(str(tmp_path/'cache'))
    files=[]
    for group in ('old','new'):
        directory=tmp_path/group;directory.mkdir()
        paths=[]
        for index in range(3):
            path=directory/f'{index}.dng';path.write_bytes(b'test only');paths.append(str(path))
        files.append(paths)
    w.input_files=list(files[0]);w._selection_changed();w._color_timer.stop()
    image=tmp_path/'display.png';Image.new('RGB',(12,8),(15,23,46)).save(image)
    def record(selected,quality='standard',adjustments=None):
        return {'reference_dir':str(tmp_path),'source_path':selected[0],'width':1200,'height':800,
                'preview_width':640 if quality=='interactive' else 1024,'preview_height':427 if quality=='interactive' else 683,
                'preview_path':str(image),'preview_quality':quality,'color_adjustments':adjustments or {},
                'input_count':len(selected)}
    w._reference_data=record(files[0]);w._reference_scene=tuple(files[0]);w._reference_pending=False
    pending=[];backend=[];displayed=[]
    monkeypatch.setattr(w,'_run_async',lambda name,fn,done:pending.append((name,fn,done)))
    monkeypatch.setattr(w,'show_reference',lambda:displayed.append(dict(w._reference_data)))
    def render(directory,*,color_adjustments=None,quality='standard'):
        backend.append((quality,dict(color_adjustments)))
        return record(files[0],quality,color_adjustments)
    def prepare(input_dir,output_dir,siril_path=None,*,input_files=None,color_adjustments=None):
        backend.append(('prepare',list(input_files)))
        return record(input_files,'standard',color_adjustments)
    monkeypatch.setattr(references,'render_reference',render)
    monkeypatch.setattr(references,'prepare_reference',prepare)
    yield w,pending,backend,displayed,files,record
    w._busy=False;w.job_state='';w._reference_running=False
    w._color_timer.stop();w._color_finish_timer.stop();w.timer.stop();w.close();app.processEvents()


def finish(item):
    item[2](item[1]())


def test_actual_timer_drag_updates_interactive_then_idle_standard(panel,app):
    w,pending,backend,shown,_,_=panel
    w.warmth_slider.setSliderDown(True)
    w.warmth_slider.setValue(10);w.warmth_slider.setValue(22);w.warmth_slider.setValue(31)
    wait_until(app,lambda:len(pending)==1)
    finish(pending[0])
    assert backend==[('interactive',{'warmth':31.,'tint':0.,'exposure_ev':0.})]
    assert shown[-1]['preview_quality']=='interactive'
    w.warmth_slider.setSliderDown(False)
    wait_until(app,lambda:len(pending)==2)
    finish(pending[1])
    assert backend[-1][0]=='standard' and backend[-1][1]['warmth']==31
    assert shown[-1]['preview_quality']=='standard'
    assert (shown[-1]['width'],shown[-1]['height'])==(1200,800)


def test_previous_colour_completion_is_not_shown_under_latest_controls(panel,app):
    w,pending,backend,shown,_,_=panel
    w.warmth_slider.setValue(12);wait_until(app,lambda:len(pending)==1)
    w.warmth_slider.setValue(29)
    wait_ms(app,140)
    assert len(pending)==1,'New expensive work overlapped in-flight work'
    finish(pending[0]);assert not shown
    wait_until(app,lambda:len(pending)==2)
    finish(pending[1])
    assert len(shown)==1 and shown[0]['color_adjustments']['warmth']==29


def test_standard_request_waits_for_inflight_interactive_and_wins(panel,app):
    w,pending,backend,shown,_,_=panel
    w.tint_slider.setValue(14);wait_until(app,lambda:len(pending)==1)
    wait_ms(app,650)
    assert len(pending)==1 and w._reference_quality=='standard'
    finish(pending[0]);assert not shown
    wait_until(app,lambda:len(pending)==2)
    finish(pending[1])
    assert [x[0] for x in backend]==['interactive','standard']
    assert len(shown)==1 and shown[-1]['preview_quality']=='standard'


def test_old_scene_completion_cannot_repopulate_or_display(panel,app):
    w,pending,backend,shown,files,_=panel
    w.warmth_slider.setValue(12);wait_until(app,lambda:len(pending)==1)
    w.input_files=list(files[1]);w._selection_changed()
    finish(pending[0])
    assert not shown and w._reference_data is None
    wait_until(app,lambda:len(pending)==2)
    finish(pending[1])
    assert shown[-1]['source_path']==files[1][0]


def test_clear_stops_pending_idle_and_rejects_inflight_result(panel,app):
    w,pending,_,shown,_,_=panel
    w.warmth_slider.setValue(12);wait_until(app,lambda:len(pending)==1)
    w.clear_input_files()
    assert not w._reference_pending and not w._color_timer.isActive() and not w._color_finish_timer.isActive()
    finish(pending[0]);wait_ms(app,720)
    assert len(pending)==1 and not shown and not w._reference_data and not w.input_files


@pytest.mark.parametrize('state',['queued','running','cancel_requested'])
def test_active_workflow_blocks_new_colour_work(panel,app,state):
    w,pending,_,_,_,_=panel
    w.job_state=state
    w.exposure_slider.setValue(5)
    wait_ms(app,750)
    assert not pending


def test_clear_before_debounce_launches_no_reference(panel,app):
    w,pending,_,_,_,_=panel
    w.warmth_slider.setValue(34);w.clear_input_files()
    wait_ms(app,750)
    assert not pending and not w._reference_pending
