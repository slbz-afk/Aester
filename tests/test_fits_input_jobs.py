"""Lifecycle and shared reference regressions for the new CFA job path."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import numpy as np
import pytest
from astropy.io import fits
from starstack import core, fits_input as fi, references, imaging

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


@pytest.fixture
def cfa_env(tmp_path, monkeypatch):
    source=tmp_path/'sources';source.mkdir()
    for i in range(3):
        h=fits.Header({'BAYERPAT':'GRBG','ROWORDER':'TOP-DOWN','EXPTIME':20.,'FILTER':'LP','BIAS':1107})
        fits.PrimaryHDU(np.full((16,18),1000+i,np.uint16),h).writeto(source/f'{i}.fit')
    engine={'available':True,'version':'1.4.4','executable':'mock-engine'}
    monkeypatch.setattr(core,'doctor',lambda *_:dict(engine))
    monkeypatch.setattr(core,'_spawn_worker',lambda *_:None)
    monkeypatch.setattr(fi,'require_engine',lambda *_:None)
    return SimpleNamespace(source=source,output=tmp_path/'jobs',engine=engine)


def fake_engine(executable,directory,name,commands,check):
    if name=='debayer':
        for p in sorted((directory/'float').glob('*.fit')):
            check();h=fits.getheader(p);a=fits.getdata(p)
            assert h['BITPIX']==-32 and a[0,0]==np.float32(1000+int(p.stem[-5:])-1)/np.float32(65535)
            assert 'BIAS' not in h and h['ROWORDER']=='TOP-DOWN'
            rgb=np.repeat(a[None],3,axis=0);rgb[:,0,0]=1.1
            prefix='ref' if 'convert ref' in commands[-1] else 'light'
            fits.PrimaryHDU(rgb,h).writeto(directory/'decoded'/f'{prefix}_{int(p.stem[-5:]):05d}.fit')
    elif name=='01_registration_probe':
        pass
    elif any(c.startswith('stack ') for c in commands):
        work=directory/'work'
        arrays=[imaging.read_rgb(p) for p in sorted(work.glob('light_*.fit'))]
        imaging.write_linear(work/'sky_master.fit',np.mean(arrays,axis=0),'Controlled stack',extra={'STACKCNT':3})
    else:
        # Registration probe name is defined by the production caller.
        assert 'register light -2pass -transf=homography' in commands


def test_direct_cfa_job_preserves_originals_and_publishes_float(cfa_env,monkeypatch):
    monkeypatch.setattr(core,'run_script',fake_engine)
    before={p.name:p.read_bytes() for p in cfa_env.source.iterdir()}
    j=core.submit_job(str(cfa_env.source),str(cfa_env.output));core.run_job(j['job_dir'])
    j=core.load_job(j['job_dir']);assert j['state']=='succeeded',j.get('error')
    work=Path(j['job_dir'])/'work'
    assert fits.getdata(work/'light_00001.fit').max()>1
    assert fits.getheader(work/'light_00001.fit')['ROWORDER']=='BOTTOM-UP'
    assert 'BAYERPAT' not in fits.getheader(work/'light_00001.fit')
    assert j['diagnostics']['exposure']['input_total_seconds']==60
    assert j['diagnostics']['exposure']['included_total_seconds'] is None  # no genuine engine sequence
    assert {p.name:p.read_bytes() for p in cfa_env.source.iterdir()}==before
    assert any('conversion.json' in r['path'] for r in j['checkpoint_files']['convert'])


@pytest.mark.parametrize('mutation',['recipe','fingerprint','legacy'])
def test_changed_or_missing_cfa_contract_rejected_before_engine_and_resume(cfa_env,monkeypatch,mutation):
    monkeypatch.setattr(core,'run_script',lambda *_:pytest.fail('must not invoke engine'))
    j=core.submit_job(str(cfa_env.source),str(cfa_env.output));path=Path(j['job_dir'])/'job.json'
    if mutation=='recipe':j['fits_input']['recipe']='other'
    elif mutation=='fingerprint':j['fits_input_sha256']='wrong'
    else:j.pop('fits_input');j.pop('fits_input_sha256')
    path.write_text(json.dumps(j),encoding='utf-8')
    core.run_job(j['job_dir']);result=core.load_job(j['job_dir'])
    assert result['state']=='failed' and not result['artifacts']
    with pytest.raises(ValueError):core.resume_job(j['job_dir'])


def test_cancelled_conversion_has_no_checkpoint_and_fresh_attempt_on_resume(cfa_env,monkeypatch):
    def interrupted(exe,directory,name,commands,check):
        fake_engine(exe,directory,name,commands,check)
        core.cancel_job(str(directory.parent))
        check()
    monkeypatch.setattr(core,'run_script',interrupted)
    j=core.submit_job(str(cfa_env.source),str(cfa_env.output));core.run_job(j['job_dir'])
    result=core.load_job(j['job_dir']);assert result['state']=='cancelled'
    assert 'convert' not in result['checkpoints'] and not result['artifacts']
    monkeypatch.setattr(core,'run_script',fake_engine)
    core.resume_job(j['job_dir']);core.run_job(j['job_dir'])
    result=core.load_job(j['job_dir']);assert result['state']=='succeeded',result.get('error')
    assert len(list(Path(j['job_dir']).glob('cfa-*')))==2


def test_reference_uses_same_floating_cfa_adapter(cfa_env,monkeypatch):
    monkeypatch.setattr(references.siril,'run_script',fake_engine)
    record=references.prepare_reference(str(cfa_env.source),str(cfa_env.output))
    assert record['state']=='succeeded'
    a=fits.getdata(record['linear_path']);assert a.dtype.kind=='f' and a.max()>1
    assert record['metadata']['fits_input']['layout']=='cfa'
    assert (record['width'],record['height'])==(18,16)


@pytest.mark.parametrize('entry',['job','reference'])
def test_cfa_engine_failure_points_to_its_actual_attempt(cfa_env,monkeypatch,entry):
    def broken(executable,directory,name,commands,check):
        (directory/'debayer.log').write_text('injected engine failure',encoding='utf-8')
        raise RuntimeError('injected engine failure')
    monkeypatch.setattr(core,'run_script',broken)
    monkeypatch.setattr(references.siril,'run_script',broken)
    if entry=='job':
        j=core.submit_job(str(cfa_env.source),str(cfa_env.output));core.run_job(j['job_dir'])
        result=core.load_job(j['job_dir']);assert result['state']=='failed'
        message=result['error']
    else:
        with pytest.raises(RuntimeError) as failure:
            references.prepare_reference(str(cfa_env.source),str(cfa_env.output))
        message=str(failure.value)
    attempts=list(cfa_env.output.rglob('cfa-*'));assert len(attempts)==1
    assert str(attempts[0]/'debayer.log') in message
    assert str(attempts[0]/'conversion.json') in message
    assert '01_convert.log' not in message
    assert json.loads((attempts[0]/'conversion.json').read_text(encoding='utf-8'))['state']=='failed'


def test_cancel_during_publication_is_cleared_before_resuming(cfa_env,monkeypatch):
    monkeypatch.setattr(core,'run_script',fake_engine)
    original_copy=shutil.copyfile
    cancelled=[]
    def interrupt_copy(source,dest,*args,**kwargs):
        result=original_copy(source,dest,*args,**kwargs)
        if Path(source).parent.name=='canonical' and not cancelled:
            cancelled.append(str(dest));core.cancel_job(str(Path(dest).parent.parent))
        return result
    monkeypatch.setattr(shutil,'copyfile',interrupt_copy)
    j=core.submit_job(str(cfa_env.source),str(cfa_env.output));core.run_job(j['job_dir'])
    result=core.load_job(j['job_dir'])
    assert result['state']=='cancelled' and cancelled
    assert Path(cancelled[0]).exists() and 'convert' not in result['checkpoints'] and not result['artifacts']
    monkeypatch.setattr(shutil,'copyfile',original_copy)
    core.resume_job(j['job_dir']);core.run_job(j['job_dir'])
    result=core.load_job(j['job_dir']);assert result['state']=='succeeded',result.get('error')
    assert len(list((Path(j['job_dir'])/'work').glob('light_*.fit')))==3


def test_legacy_rgb_conversion_path_stays_intact_and_unknown_exposure_is_not_zero(tmp_path,monkeypatch):
    paths=[]
    for i in range(3):
        p=tmp_path/f'light_{i+1:05d}.fit';imaging.write_linear(p,np.ones((8,10,3),np.float32),'fixture');paths.append(p)
    rec=core.exposure_diagnostics(tmp_path,paths,None,3)
    assert rec['input_total_seconds'] is None and rec['included_total_seconds'] is None
    assert rec['unknown_input_count']==3 and rec['membership_status']=='unknown'
