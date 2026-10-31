"""Orientation admission/binding checks; synthetic pixels, no imaging engine."""
import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from starstack import core, editing_project, orientation, references, regions, workflow
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server
from test_workflow import setup


@pytest.fixture
def scene(tmp_path):
    inputs, output = tmp_path / 'input', tmp_path / 'cache'
    inputs.mkdir(); output.mkdir()
    files = []
    for i in range(3):
        path = inputs / f'{i}.NEF'
        path.write_bytes(bytes([i + 1]) * 31)
        files.append(str(path))
    directory = output / 'reference-portrait'
    directory.mkdir()
    linear = directory / 'linear.fit'
    header = fits.Header({'ROWORDER': 'BOTTOM-UP'})
    fits.writeto(linear, np.arange(3*24*12, dtype=np.float32).reshape(3,24,12), header)
    preview = directory / 'preview.png'
    Image.new('RGB', (12,24), (12,24,48)).save(preview)
    mask = directory / 'mask.png'
    Image.new('L', (12,24), 255).save(mask)
    ref = {'state': 'succeeded', 'reference_dir': str(directory),
           'source_path': files[0], 'source_sha256': core._sha(Path(files[0])),
           'linear_path': str(linear), 'linear_sha256': core._sha(linear),
           'preview_path': str(preview), 'preview_sha256': core._sha(preview),
           'mask_template_path': str(mask), 'mask_template_sha256': core._sha(mask),
           'width': 12, 'height': 24, 'image_rotation': 90}
    (directory/'reference.json').write_text(json.dumps(ref), encoding='utf8')
    return WorkspaceTools(inputs, output), files, ref


def state_for(files, rotation):
    return {'input_files': files, 'image_rotation': rotation,
        'color_adjustments': {}, 'trails_enabled': False, 'stars_enabled': False,
        'milkyway_enabled': False, 'auto_strength': True, 'stack_range': False,
        'milkyway_strength': .6, 'background_strength': .5, 'milkyway_offset': 0,
        'scene_mode': 'sky', 'backend_options': workflow.BACKEND_DEFAULTS.copy()}


def test_portrait_regions_keep_display_coordinates_and_reopen(scene):
    service, files, ref = scene
    project = regions.create_project(ref, files)
    project = regions.revise(project, scene_mode='landscape',
        operations=[{'kind':'horizon', 'points':[[0,.7],[1,.7]]}])
    mask, _ = regions.rasterize(project)
    assert mask.shape == (24,12)
    assert np.all(mask[:15] == 255) and np.all(mask[18:] == 0)
    path = service.output_root/'editing.json'
    saved = editing_project.save_project(path, state_for(files,90), ref, project)
    readback = editing_project.load_project(path)
    assert readback['state']['image_rotation'] == 90
    assert readback['regions']['source_binding']['reference']['image_rotation'] == 90
    assert saved['signature'] == readback['signature']
    np.testing.assert_array_equal(regions.rasterize(readback['regions'])[0], mask)


@pytest.mark.parametrize('place', ['state', 'reference', 'project'])
def test_editing_rotation_mismatch_rejected_before_save(scene, place):
    service, files, ref = scene
    project = regions.create_project(ref, files)
    state = state_for(files,90)
    if place == 'project':
        project['source_binding']['reference']['image_rotation'] = 270
        project = regions._seal(project)
    elif place == 'state':
        state['image_rotation'] = 270
    else:
        ref['image_rotation'] = 270
    with pytest.raises(ValueError, match='方向'):
        editing_project.save_project(service.output_root/'bad.json', state, ref, project)
    assert not (service.output_root/'bad.json').exists()


def test_legacy_editing_without_any_rotation_remains_zero(scene):
    service, files, ref = scene
    ref.pop('image_rotation')
    project = regions.create_project(ref,files)
    state = state_for(files,0)
    state.pop('image_rotation')
    path = service.output_root/'legacy.json'
    editing_project.save_project(path,state,ref,project)
    assert 'image_rotation' not in editing_project.load_project(path)['reference']


@pytest.mark.parametrize('bad', [True, False, 90., '90', 1, -90, 360, None])
def test_region_rotation_rejects_non_integer_quarter_turn(scene,bad):
    _,files,ref = scene
    ref['image_rotation'] = bad
    with pytest.raises(ValueError):
        regions.create_project(ref,files)


def test_workflow_auto_and_explicit_orientation_in_signature(setup,monkeypatch):
    files = sorted(map(str,setup['inputs'].glob('*.NEF')))
    monkeypatch.setattr(orientation,'source_rotation',lambda path: 270)
    auto = workflow.submit_workflow(setup['inputs'],setup['output'],input_files=files,idempotency_key='angle')
    assert auto['request']['image_rotation'] == auto['config']['image_rotation'] == 270
    same = workflow.submit_workflow(setup['inputs'],setup['output'],input_files=files,
        image_rotation=270,idempotency_key='angle')
    assert same['signature'] == auto['signature']
    with pytest.raises(ValueError, match='不同'):
        workflow.submit_workflow(setup['inputs'],setup['output'],input_files=files,
            image_rotation=90,idempotency_key='angle')
    explicit = workflow.submit_workflow(setup['inputs'],setup['output'],input_files=files,image_rotation=0)
    assert explicit['request']['image_rotation'] == 0
    assert explicit['signature'] != auto['signature']


def test_directory_workflow_preserves_zero_without_metadata_read(setup,monkeypatch):
    monkeypatch.setattr(orientation,'source_rotation',lambda p: pytest.fail('legacy directory cannot auto-rotate'))
    record = workflow.submit_workflow(setup['inputs'],setup['output'])
    assert record['request']['image_rotation'] == 0


def test_standalone_legacy_mask_keeps_native_direction_unless_explicit(setup,monkeypatch,tmp_path):
    files = sorted(map(str,setup['inputs'].glob('*.NEF')))
    mask = tmp_path/'mask.png'
    Image.new('L',(24,12),255).save(mask)
    monkeypatch.setattr(orientation,'source_rotation',lambda p: pytest.fail('unbound PNG cannot auto-rotate'))
    record = workflow.submit_workflow(setup['inputs'],setup['output'],input_files=files,sky_mask_path=str(mask))
    assert record['request']['image_rotation'] == 0
    explicit = workflow.submit_workflow(setup['inputs'],setup['output'],input_files=files,
        sky_mask_path=str(mask),image_rotation=90)
    assert explicit['request']['image_rotation'] == 90


@pytest.mark.parametrize('angle',[0,90,180,270])
def test_rotation_forwarded_to_child_and_bound_on_resume(setup,monkeypatch,angle):
    original = setup['submit']
    def submit(*args,**kwargs):
        child = original(*args,**kwargs)
        child['image_rotation'] = kwargs['image_rotation']
        workflow.atomic_json(Path(child['job_dir'])/'job.json',child)
        return child
    monkeypatch.setattr(core,'submit_job',submit)
    submitted = workflow.submit_workflow(setup['inputs'],setup['output'],image_rotation=angle)
    done = workflow.run_workflow(submitted['workflow_dir'])
    assert done['state'] == 'succeeded', done.get('error')
    assert setup['calls']['kwargs']['image_rotation'] == angle
    child = core.load_job(done['stack_job_dir'])
    workflow.validate_stack_binding(done,child)
    child['image_rotation'] = (angle+90)%360
    with pytest.raises(ValueError,match='方向'):
        workflow.validate_stack_binding(done,child)


def test_bound_region_rotation_wins_over_metadata(scene,monkeypatch):
    service, files, ref = scene
    project = regions.save_project(regions.create_project(ref,files),service.output_root)
    engine = service.output_root/'fake-engine.exe'; engine.write_bytes(b'fake')
    monkeypatch.setattr(core,'doctor',lambda p: {'available':True,'executable':str(engine)})
    monkeypatch.setattr(workflow,'_spawn',lambda p: None)
    monkeypatch.setattr(orientation,'source_rotation',lambda p: pytest.fail('use bound canvas orientation'))
    submitted = workflow.submit_workflow(service.input_root,service.output_root,
        input_files=files,regions_path=project['project_path'])
    assert submitted['request']['image_rotation'] == 90
    with pytest.raises(ValueError,match='方向'):
        workflow.submit_workflow(service.input_root,service.output_root,
            input_files=files,regions_path=project['project_path'],image_rotation=270)


@pytest.mark.parametrize('bad',[True,90.,'90',45,-90,360])
def test_mcp_rotation_rejected_before_read(scene,bad):
    service,_,_ = scene
    with pytest.raises(ValueError):
        service.rotate_reference('does-not-exist',bad)
    with pytest.raises(ValueError):
        service.prepare_reference(input_files=['does-not-exist'],image_rotation=bad)
    with pytest.raises(ValueError):
        service.submit_workflow(input_files=['does-not-exist'],image_rotation=bad)


@pytest.mark.parametrize('field',['source_path','linear_path','preview_path','cache'])
def test_mcp_reference_external_paths_rejected_before_rotation(scene,monkeypatch,field):
    service,_,ref=scene
    outside=str(service.output_root.parent/'secret.bin')
    if field == 'cache':
        ref['color_cache']={'linear':outside,'saturated':ref['linear_path']}
    else:
        ref[field]=outside
    (Path(ref['reference_dir'])/'reference.json').write_text(json.dumps(ref),encoding='utf8')
    monkeypatch.setattr(references,'rotate_reference',lambda *a,**kw: pytest.fail('no engine before admission'))
    with pytest.raises(AccessDenied):
        service.rotate_reference(ref['reference_dir'],180)


def test_mcp_rotate_receives_exact_admitted_snapshot(scene,monkeypatch):
    service,_,ref=scene
    captured={}
    def rotate(directory,*,image_rotation,_record):
        captured.update(angle=image_rotation,record=copy.deepcopy(_record))
        return _record
    monkeypatch.setattr(references,'rotate_reference',rotate)
    result=service.rotate_reference(ref['reference_dir'],270)
    assert captured == {'angle':270,'record':ref}
    assert result['source_path'] == ref['source_path']


def test_actual_mcp_rotation_is_immutable_and_drawable(scene):
    service,files,ref=scene
    directory=Path(ref['reference_dir'])
    old_hashes={path:core._sha(path) for path in directory.iterdir() if path.is_file()}
    source_hashes={Path(path):core._sha(Path(path)) for path in files}
    old=fits.getdata(ref['linear_path'])
    result=service.rotate_reference(str(directory),180)
    assert (result['width'],result['height'],result['image_rotation']) == (24,12,180)
    new=fits.getdata(result['linear_path'])
    # Clockwise on displayed pixels is counterclockwise in bottom-up FITS.
    np.testing.assert_array_equal(new,old[:,:,::-1].transpose(0,2,1))
    project=service.create_regions(result['reference_dir'],files,
        operations=[{'kind':'paint','target':'ground','points':[[.5,.75]],'radius':.15}])
    assert project['source_binding']['reference']['image_rotation'] == 180
    mask,_=regions.rasterize(project)
    assert mask.shape == (12,24) and mask[8,12] == 0
    assert {path:core._sha(path) for path in old_hashes} == old_hashes
    assert {path:core._sha(path) for path in source_hashes} == source_hashes


def test_mcp_refuses_modified_linear_before_creating_copy(scene):
    service,_,ref=scene
    Path(ref['linear_path']).write_bytes(b'changed')
    before=set(service.output_root.iterdir())
    with pytest.raises(ValueError,match='改变'):
        service.rotate_reference(ref['reference_dir'],180)
    assert set(service.output_root.iterdir()) == before


def test_mcp_sdk_rotations_are_strict_integers(scene,monkeypatch):
    service,_,ref=scene
    seen=[]
    monkeypatch.setattr(service,'rotate_reference',lambda directory,image_rotation: seen.append(image_rotation) or {'angle':image_rotation})
    server=create_server(service)
    async def exercise():
        tools={tool.name:tool for tool in await server.list_tools()}
        assert 'rotate_reference' in tools
        for angle in [True,90.,'90']:
            with pytest.raises(Exception):
                await server.call_tool('rotate_reference',{'reference_dir':ref['reference_dir'],'image_rotation':angle})
        await server.call_tool('rotate_reference',{'reference_dir':ref['reference_dir'],'image_rotation':90})
    asyncio.run(exercise())
    assert seen == [90]
