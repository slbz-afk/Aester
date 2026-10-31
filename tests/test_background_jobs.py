"""Root acceptance: frozen background derivatives, source binding and publication."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from starstack import core, background_job as bj
from starstack.imaging import write_linear
from starstack.compose_evidence import write_weight


@pytest.fixture
def scene(tmp_path):
    jobdir=tmp_path/'job-scene'; jobdir.mkdir()
    yy,xx=np.mgrid[:256,:256]/255*2-1
    image=(.10+xx[...,None]*np.array([.02,.01,-.015])+yy[...,None]*np.array([.01,.02,.003])+np.array([0,.04,.015])).astype(np.float32)
    # A signed dark residual and a highlight must survive the float output.
    image[128,128]=[-.1,.2,1.3]
    source=jobdir/'master_linear.fit'; write_linear(source,np.flipud(image),'Controlled job fixture')
    job={'state':'succeeded','job_dir':str(jobdir),'mask':None,'diagnostics':{},
         'artifacts':[{'path':str(source),'kind':'fits','label':'base'}]}
    core.atomic_json(jobdir/'job.json',job)
    mask=tmp_path/'samples.png'; Image.fromarray(np.full((256,256),255,np.uint8)).save(mask)
    return jobdir,source,mask,image


def test_derivative_and_frozen_replay_are_identical_without_refit(scene,monkeypatch):
    jobdir,source,mask,image=scene; originals={p:p.read_bytes() for p in [source,jobdir/'job.json',mask]}
    first=bj.render_background(jobdir,mask)
    assert first['state']=='succeeded'
    a=tifffile.imread(first['artifacts'][0]['path']);assert a.dtype==np.float32 and a.min()<0 and a.max()>1
    monkeypatch.setattr(bj.bg,'fit_background',lambda *_:pytest.fail('must not refit on replay'))
    monkeypatch.setattr(bj.bg,'display_parameters',lambda *_:pytest.fail('must not re-estimate on replay'))
    second=bj.render_background(jobdir,recipe_path=first['recipe_path'])
    assert second['directory']!=first['directory']
    assert np.array_equal(a.view(np.uint32),tifffile.imread(second['artifacts'][0]['path']).view(np.uint32))
    for old,new in zip(first['artifacts'][1:],second['artifacts'][1:]):
        assert np.array_equal(np.asarray(Image.open(old['path'])),np.asarray(Image.open(new['path'])))
    assert {p:p.read_bytes() for p in originals}==originals


def test_actual_ground_weights_are_bound_and_ground_is_bitwise_unchanged(scene):
    jobdir,source,mask,image=scene
    w=np.ones(image.shape[:2],np.float32);w[192:]=0;w[160:192]=np.linspace(1,0,32)[:,None]
    descriptor=write_weight(jobdir,w,source)
    path=jobdir/'job.json';job=json.loads(path.read_text(encoding='utf-8'))
    job.update(mask={'path':'unused-in-this-branch'},diagnostics={'sky_validity':{'compose_weight':descriptor}})
    core.atomic_json(path,job)
    r=bj.render_background(jobdir,mask)
    a=tifffile.imread(r['artifacts'][0]['path'])
    assert np.array_equal(a[w==0].view(np.uint32),image[w==0].view(np.uint32))
    assert r['statistics']['ground_unchanged'] is True
    assert all(p[1]<.3 for p in r['payload']['model']['samples_xy'])


@pytest.mark.parametrize('case',['missing','wrong-base','wrong-hash','wrong-shape'])
def test_ground_without_valid_evidence_rejected(scene,case):
    jobdir,source,mask,image=scene; path=jobdir/'job.json';job=json.loads(path.read_text(encoding='utf-8'))
    descriptor=write_weight(jobdir,np.ones(image.shape[:2],np.float32),source)
    if case=='wrong-base':
        other=jobdir/'other.fit';other.write_bytes(source.read_bytes());descriptor['composite']=str(other)
    if case=='wrong-hash':descriptor['sha256']='0'*64
    if case=='wrong-shape':descriptor['shape']=[128,512]
    job.update(mask={'path':'unused'},diagnostics={'sky_validity':{'compose_weight':descriptor}} if case!='missing' else {})
    core.atomic_json(path,job)
    with pytest.raises(ValueError):bj.render_background(jobdir,mask)


@pytest.mark.parametrize('change',['source','job','mask','recipe'])
def test_replay_rejects_changed_evidence(scene,change):
    jobdir,source,mask,image=scene; r=bj.render_background(jobdir,mask)
    if change=='source':source.write_bytes(source.read_bytes()+b'changed')
    if change=='job':
        p=jobdir/'job.json';j=json.loads(p.read_text(encoding='utf-8'));j['changed']=True;core.atomic_json(p,j)
    if change=='mask':(Path(r['directory'])/'background_mask.png').write_bytes(b'changed')
    if change=='recipe':
        p=Path(r['recipe_path']);j=json.loads(p.read_text(encoding='utf-8'));j['payload']['strength']=.2;core.atomic_json(p,j)
    with pytest.raises(ValueError):bj.render_background(jobdir,recipe_path=r['recipe_path'])


@pytest.mark.parametrize('failure',['write','source-mutation','mask-mutation','display'])
def test_failed_derivative_not_published_and_job_is_preserved(scene,monkeypatch,failure):
    jobdir,source,mask,image=scene; oldjob=(jobdir/'job.json').read_bytes()
    original=bj.bg.render_display
    def fail(*args,**kwargs):
        if failure=='display':raise RuntimeError('injected display failure')
        if failure=='source-mutation':source.write_bytes(source.read_bytes()+b'changed')
        if failure=='mask-mutation':mask.write_bytes(mask.read_bytes()+b'changed')
        return original(*args,**kwargs)
    monkeypatch.setattr(bj.bg,'render_display',fail)
    if failure=='write':monkeypatch.setattr(bj.tifffile,'imwrite',lambda *a,**kw:(_ for _ in ()).throw(OSError('injected full disk')))
    with pytest.raises((ValueError,OSError,RuntimeError)):bj.render_background(jobdir,mask)
    records=list((jobdir/'background-variations').glob('*/recipe.json'));assert len(records)==1
    record=json.loads(records[0].read_text(encoding='utf-8'))
    assert record['state']=='failed' and record['artifacts']==[]
    assert (jobdir/'job.json').read_bytes()==oldjob


@pytest.mark.parametrize('problem',['nonbinary','wrong-size','RGB','absent','sparse'])
def test_invalid_masks_rejected(scene,problem):
    jobdir,source,mask,image=scene
    if problem=='nonbinary':Image.fromarray(np.full((256,256),127,np.uint8)).save(mask)
    if problem=='wrong-size':Image.new('L',(128,256),255).save(mask)
    if problem=='RGB':Image.new('RGB',(256,256),'white').save(mask)
    if problem=='sparse':Image.new('L',(256,256),0).save(mask)
    with pytest.raises(ValueError):bj.render_background(jobdir,None if problem=='absent' else mask)


def test_mcp_paths_bounded_before_mutation(scene,tmp_path):
    from starstack.mcp_server import WorkspaceTools,AccessDenied
    jobdir,source,mask,image=scene
    inp=tmp_path/'input';inp.mkdir();out=jobdir.parent
    service=WorkspaceTools(inp,out)
    with pytest.raises(AccessDenied):service.coordinate_background(str(jobdir),str(out.parent/'outside.png'))
    result=service.coordinate_background(str(jobdir),str(mask))
    assert result['state']=='succeeded'


def test_replay_with_controls_rejected(scene):
    jobdir,source,mask,image=scene;r=bj.render_background(jobdir,mask)
    with pytest.raises(ValueError):bj.render_background(jobdir,recipe_path=r['recipe_path'],strength=.5)
