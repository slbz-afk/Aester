"""MCP region entry tests with real FITS geometry, no decoder/network."""
import json
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from starstack import core, regions
from starstack.mcp_server import WorkspaceTools, AccessDenied


@pytest.fixture
def sample(tmp_path):
    inp, out = tmp_path/'input', tmp_path/'output'
    inp.mkdir(); out.mkdir()
    files=[]
    for i in range(3):
        raw=inp/f'{i}.NEF'; raw.write_bytes(bytes([i+1])*43); files.append(str(raw))
    refdir=out/'reference-small'; refdir.mkdir()
    linear=refdir/'linear.fit'; fits.writeto(linear,np.ones((3,24,32),np.float32))
    ref={'state':'succeeded','source_path':files[0],'source_sha256':core._sha(Path(files[0])),
         'linear_path':str(linear),'linear_sha256':core._sha(linear),'width':32,'height':24,
         'reference_dir':str(refdir)}
    (refdir/'reference.json').write_text(json.dumps(ref),encoding='utf8')
    return WorkspaceTools(inp,out), refdir, files, ref


def test_create_reopen_update_undo_immutable(sample):
    service, refdir, files, ref=sample
    first=service.create_regions(str(refdir), files, scene_mode='landscape',
        operations=[{'kind':'horizon','points':[[0,.75],[1,.75]]}])
    sky,_=regions.rasterize(first)
    assert np.all(sky[:17]==255) and np.all(sky[18:]==0)
    second=service.update_regions(first['project_path'],scene_mode='sky')
    assert second['revision']!=first['revision']
    assert service.get_regions(first['project_path'])['scene_mode']=='landscape'
    assert service.get_regions(second['project_path'])['scene_mode']=='sky'


def test_explicit_agent_drawing_enables_without_impersonating_human(sample):
    service, refdir, files, _=sample
    result=service.create_regions(str(refdir),files,scene_mode='sky',
        milkyway={'status':'present','ellipse':[.5,.5,.3,.4], 'authority':'human','confirmed':True})
    assert result['milkyway']['authority']=='agent'
    assert result['milkyway']['status']=='present'
    assert result['milkyway']['confirmed']
    assert regions.build_masks(result,service.output_root)['milkyway_mask_path']
    unknown=service.update_regions(result['project_path'],milkyway={'status':'unknown'})
    assert regions.build_masks(unknown,service.output_root)['milkyway_mask_path'] is None


def test_relayed_user_statement_retains_agent_editor_and_immutable_history(sample):
    service, refdir, files, _=sample
    old=service.create_regions(str(refdir),files,scene_mode='sky',
        milkyway={'status':'present','ellipse':[.5,.5,.3,.4]}, user_confirmation='我已看过，此区域是可见银河。')
    assert old['milkyway']['authority']=='agent'
    assert old['milkyway']['rationale']=='我已看过，此区域是可见银河。'
    assert regions.build_masks(old,service.output_root)['milkyway_mask_path']
    new=service.update_regions(old['project_path'],milkyway={'ellipse':[.6,.5,.3,.4]})
    assert new['milkyway']['status']=='present' and new['milkyway']['authority']=='agent'
    assert service.get_regions(old['project_path'])['milkyway']['status']=='present'


def test_modified_raw_refused(sample):
    service, refdir, files, _=sample
    project=service.create_regions(str(refdir),files,scene_mode='sky')
    Path(files[-1]).write_bytes(b'modified')
    with pytest.raises(ValueError): service.get_regions(project['project_path'])


@pytest.mark.parametrize('field',['source_path','linear_path'])
def test_reference_outside_host_refused_before_data_read(sample, field):
    service, refdir, files, ref=sample
    ref[field]=str(service.output_root.parent/'private.bin')
    (refdir/'reference.json').write_text(json.dumps(ref),encoding='utf8')
    with pytest.raises(AccessDenied): service.create_regions(str(refdir),files)


def test_region_record_external_source_cannot_be_opened(sample):
    service, refdir, files, _=sample
    project=service.create_regions(str(refdir),files,scene_mode='sky')
    path=Path(project['project_path'])
    project['source_binding']['input_files'][1]['path']=str(service.output_root.parent/'private.bin')
    path.write_text(json.dumps(project),encoding='utf8')
    with pytest.raises(AccessDenied): service.get_regions(str(path))
