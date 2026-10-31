"""Drawing contracts across Qt, persisted masks, MCP, and the real finishing gate."""
import asyncio
import copy
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
import tifffile
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from starstack import regions, milkyway_gate, workflow as wf
from aester_desktop.region_canvas import RegionCanvas
from aester_desktop.region_dialog import RegionDialog
from test_regions_p1 import sample, app
from test_p1_mcp_regions_independent import sample as mcp_sample
from test_workflow import setup, stub_model
from test_workflow_regions import region_project, bind_child_mask
from test_gui_workflow import window


def drag(canvas, first, last):
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=first.toPoint())
    QTest.mouseMove(canvas, last.toPoint())
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=last.toPoint())


def test_no_marks_no_ground_no_milkyway_without_second_confirmation(sample, app):
    reference, _, project, out = sample
    dialog = RegionDialog(reference, project)
    dialog.finish()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert regions.scene_from_marks(dialog.project) == 'sky'
    assets = regions.build_masks(dialog.project, out)
    assert assets['milkyway_mask_path'] is None
    assert np.all(np.asarray(Image.open(assets['sky_mask_path'])) == 255)


def test_drawing_across_edge_move_clear_undo_save_reopen(sample, app):
    reference, files, project, out = sample
    canvas = RegionCanvas(QImage(reference['preview_path']), project)
    canvas.resize(1000, 700); canvas.zoom = .6; canvas.show(); app.processEvents()
    canvas.set_mode('milkyway')
    drag(canvas, canvas.screen_point([.2, .1]), canvas.screen_point([1.2, .9]))
    assert canvas.project['milkyway']['ellipse'] == pytest.approx([.7, .5, .5, .4], abs=.004)
    assert regions.milkyway_active(canvas.project)
    first_revision = canvas.project['revision']
    first_center = canvas.ellipse_handles()['center']
    drag(canvas, first_center, first_center + QPointF(canvas.image_rect().width()*.4, 0))
    assert canvas.project['milkyway']['ellipse'][0] > 1.09
    saved = regions.save_project(canvas.project, out)
    loaded = regions.load_project(saved['project_path'], input_files=files)
    assert np.array_equal(regions.rasterize(canvas.project), regions.rasterize(loaded))
    assert np.any(regions.rasterize(loaded)[1][:, -1] > 0)
    moved_revision = canvas.project['revision']
    canvas.undo(); assert canvas.project['revision'] == first_revision
    canvas.redo(); assert canvas.project['revision'] == moved_revision
    canvas.clear_milkyway(); assert not regions.milkyway_active(canvas.project)
    assert not np.any(regions.rasterize(canvas.project)[1])
    canvas.undo(); assert canvas.project['revision'] == moved_revision
    canvas.close()


@pytest.mark.parametrize('operation', ['rotate', 'east', 'north', 'feather'])
def test_actual_transform_handle_drags_are_one_undo_step(sample, app, operation):
    reference, _, project, _ = sample
    project = regions.draw_milkyway(project, [.5, .5, .23, .28])
    canvas = RegionCanvas(QImage(reference['preview_path']), project)
    canvas.resize(1000, 700); canvas.zoom=.7; canvas.show(); app.processEvents(); canvas.set_mode('milkyway')
    handles = canvas.ellipse_handles(); center=handles['center']; rect=canvas.image_rect()
    if operation == 'rotate':
        a=math.radians(-20); point=center+QPointF(math.cos(a)*140, math.sin(a)*140)
    elif operation == 'east': point=center+QPointF(rect.width()*.42,0)
    elif operation == 'north': point=center-QPointF(0,rect.height()*.44)
    else: point=center+QPointF(rect.width()*.23*.35,0)
    drag(canvas, handles[operation], point)
    new=canvas.project['milkyway']
    if operation == 'rotate': assert new['angle_degrees'] == pytest.approx(70, abs=1)
    elif operation == 'east': assert new['ellipse'][2] == pytest.approx(.42, abs=.004)
    elif operation == 'north': assert new['ellipse'][3] == pytest.approx(.44, abs=.004)
    else: assert new['feather'] == pytest.approx(.65, abs=.01)
    assert len(canvas._undo)==1
    canvas.undo(); assert canvas.project==project
    canvas.close()


def test_brush_pointer_radius_tracks_zoom_and_masks(sample, app):
    reference, _, project, _ = sample
    canvas = RegionCanvas(QImage(reference['preview_path']), project)
    canvas.resize(900,700); canvas.show(); app.processEvents(); canvas.set_mode('ground'); canvas.radius=.05
    r1=canvas.brush_screen_radius(); canvas.zoom=2; r2=canvas.brush_screen_radius()
    assert r2 == pytest.approx(r1*2)
    point=canvas.screen_point([.5,.5])
    # QTest.mouseMove may skip dispatch if the process-global cursor already
    # occupies this position after another offscreen test. Send the real Qt
    # event explicitly so this checks our handler, independent of cursor state.
    move=QMouseEvent(QEvent.Type.MouseMove, point,
        QPointF(canvas.mapToGlobal(point.toPoint())), Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas,move); app.processEvents()
    assert canvas._hover == point
    QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=point.toPoint())
    assert regions.scene_from_marks(canvas.project)=='landscape'
    sky,_=regions.rasterize(canvas.project)
    ys,xs=np.where(sky==0)
    assert np.ptp(xs)==pytest.approx(.05*80*2,abs=1)
    canvas.clear_landscape(); assert regions.scene_from_marks(canvas.project)=='sky'
    canvas.undo(); assert regions.scene_from_marks(canvas.project)=='landscape'
    canvas.close()


def test_extended_ellipse_matches_independent_scalar_and_stays_off_ground(sample):
    project=regions.revise(sample[2],scene_mode='landscape',operations=[{'kind':'horizon','points':[[0,.8],[1,.8]]}])
    project=regions.draw_milkyway(project,[-.08,1.05,.74,1.1],angle_degrees=43,feather=.6)
    sky,mask=regions.rasterize(project)
    for x,y in [(0,0),(15,23),(59,33),(109,50),(0,79),(119,79)]:
        dx=(x/119+.08)*120;dy=(y/79-1.05)*80;t=math.radians(43)
        radius=math.hypot((math.cos(t)*dx+math.sin(t)*dy)/(.74*120),
                          (-math.sin(t)*dx+math.cos(t)*dy)/(1.1*80))
        f=min(1,max(0,(1-radius)/.6));expected=f*f*(3-2*f) if sky[y,x] else 0
        assert mask[y,x]==pytest.approx(expected,abs=4e-7)
    assert not np.any(mask[sky==0])


@pytest.mark.parametrize('ellipse', [[-4.1,.5,.3,.3],[.5,5.1,.3,.3],[.5,.5,8.1,.3],[.5,.5,float('nan'),.3]])
def test_extents_bounded_without_silent_clipping(sample,ellipse):
    with pytest.raises(ValueError): regions.draw_milkyway(sample[2],ellipse)


def prepare_mcp_preview(sample):
    service,refdir,files,reference=sample
    path=refdir/'preview.png';Image.new('RGB',(32,24),(40,50,60)).save(path)
    reference.update(preview_path=str(path),preview_sha256=regions._sha(path),preview_width=32,preview_height=24)
    (refdir/'reference.json').write_text(json.dumps(reference),encoding='utf8')
    return service,refdir,files


def test_high_level_agent_drawing_mask_preview_undo_and_no_impersonation(mcp_sample):
    service,refdir,files=prepare_mcp_preview(mcp_sample)
    first=service.draw_regions(str(refdir),input_files=files,horizon=[[0,.75],[1,.75]],
        ground_strokes=[[[.3,.3],[.4,.3]]],milkyway_ellipse=[1.1,.5,.85,.6],milkyway_angle=25)
    assert first['milkyway_enabled'] and first['scene_mode']=='landscape'
    assert first['project']['milkyway']['authority']=='agent'
    assert Path(first['preview']['preview_path']).is_file()
    assets=regions.build_masks(first['project'],service.output_root/'masks')
    assert np.any(tifffile.imread(assets['milkyway_mask_path']))
    second=service.draw_regions(str(refdir),regions_path=first['project_path'],clear_ground=True,clear_milkyway=True)
    assert not second['milkyway_enabled'] and second['scene_mode']=='sky'
    assert service.get_regions(first['project_path'])['revision']==first['revision']
    assert regions.milkyway_active(service.get_regions(first['project_path']))


def test_empty_update_does_not_enable_unknown_legacy_ellipse(mcp_sample):
    service,refdir,files=prepare_mcp_preview(mcp_sample)
    first=service.create_regions(str(refdir),files,milkyway={'ellipse':[.5,.5,.3,.3],'status':'unknown'})
    again=service.update_regions(first['project_path'],milkyway={})
    assert not regions.milkyway_active(again)


def test_new_mcp_tool_rejects_outside_reference_before_read(mcp_sample):
    service,refdir,files=prepare_mcp_preview(mcp_sample)
    from starstack.mcp_server import AccessDenied
    with pytest.raises(AccessDenied):
        service.draw_regions(str(service.output_root.parent/'private'),input_files=files)


def test_agent_drawn_regions_reach_actual_finishing_and_retain_method(setup,monkeypatch):
    bind_child_mask(setup,monkeypatch);stub_model(monkeypatch)
    path,files=region_project(setup)
    project=regions.load_project(path)
    project=regions.draw_milkyway(project,[.5,.35,.37,.45],authority='agent',reviewer='MCP agent')
    path=regions.save_project(project,setup['output']/'new-region')['project_path']
    job=wf.submit_workflow(setup['inputs'],setup['output'],input_files=files,regions_path=path,
        milkyway_enabled=True,stars_enabled=True,star_model_path=setup['model'],background_strength=.7,
        milkyway_offset=-.2,backend_options={'tile':64,'stride':32})
    done=wf.run_workflow(job['workflow_dir'])
    assert done['state']=='succeeded',done.get('error')
    gate=done['finishing']['milkyway']['subject_gate']
    assert gate['enabled'] and gate['evidence']['method']=='explicit-region-agent'
    assert done['finishing']['reduction']['milkyway_gate']['enabled']
    wf._verify(done)


def test_draw_regions_is_discoverable_and_callable_in_real_mcp_process(mcp_sample):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    service,refdir,files=prepare_mcp_preview(mcp_sample)
    env={k:v for k,v in os.environ.items() if not k.startswith('STARSTACK_')}
    env.update(PYTHONPATH=str(Path('src').resolve()),STARSTACK_INPUT_ROOT=str(service.input_root),
               STARSTACK_OUTPUT_ROOT=str(service.output_root))
    async def check():
        async with stdio_client(StdioServerParameters(command=sys.executable,args=['-X','utf8','-m','starstack.mcp_server'],env=env)) as (read,write):
            async with ClientSession(read,write) as client:
                await client.initialize()
                names=[t.name for t in (await client.list_tools()).tools]
                assert 'draw_regions' in names
                result=await client.call_tool('draw_regions',{'reference_dir':str(refdir),'input_files':files,
                    'milkyway_ellipse':[1.05,.5,.8,.6],'milkyway_feather':.65})
                assert not result.isError,result.content
                data=result.structuredContent
                assert data==json.loads(next(item.text for item in result.content if item.type=='text'))
                assert data['milkyway_enabled'] and data['project']['milkyway']['authority']=='agent'
                overlay=await client.call_tool('preview_regions',{'regions_path':data['project_path'],
                    'reference_dir':str(refdir),'include_image':True})
                assert not overlay.isError and any(item.type=='image' for item in overlay.content)
                assert overlay.structuredContent==json.loads(next(item.text for item in overlay.content if item.type=='text'))
    asyncio.run(check())


def test_actual_gui_dialog_saved_drawing_enables_real_workflow_admission(window,app,monkeypatch,tmp_path):
    from test_p1_gui_routing_independent import scene
    scene(window,monkeypatch,tmp_path,present=False)
    window.cache_dir.setText(str(tmp_path/'working-cache'))
    original=window._regions_project
    window._regions_project=regions.clear_milkyway(regions.clear_landscape(original))
    window._regions_path=None
    window._update_controls()
    assert not window.milkyway_enabled.isEnabled()
    def edit(dialog):
        dialog.show();app.processEvents()
        canvas=dialog.canvas;canvas.set_mode('milkyway');canvas.zoom=.65
        drag(canvas,canvas.screen_point([.15,.1]),canvas.screen_point([1.1,.9]))
        assert regions.milkyway_active(canvas.project)
        dialog.finish()
        return dialog.result()
    monkeypatch.setattr(RegionDialog,'exec',edit)
    calls=[]
    def synchronous(name,operation,callback):
        result=operation();calls.append((name,result));callback(result)
    monkeypatch.setattr(window,'_run_async',synchronous)
    window.edit_regions()
    assert [name for name,_ in calls]==['prepare-regions','save-regions']
    assert window.milkyway_enabled.isEnabled()
    assert regions.milkyway_active(regions.load_project(window._regions_path))
    window.milkyway_enabled.setChecked(True)
    # Keep the actual admission function, stopping only its detached worker spawn.
    submitted=[]
    monkeypatch.setattr(window,'_workflow_submitted',lambda result:submitted.append(result))
    window.start_processing()
    assert len(submitted)==1,window.message_label.text()
    request=submitted[0]['request']
    assert request['milkyway_enabled']
    assert request['region_assets']['milkyway_mask_path']
    assert regions.load_project(request['regions_path'])['scene_mode']=='sky'
    window._busy=False
