"""Independent lifecycle review of the integrated direction change.

The script stand-in produces tiny synthetic CFA/RGB values. Assertions concern
real production checkpoint, resume, GUI restoration and proxy contracts; they
are not a RAW/Siril photographic acceptance run.
"""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from starstack import core, orientation
from test_fits_input_jobs import cfa_env, fake_engine
from test_gui_aester_p0 import app, window

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


def assert_once_rotated(job):
    converted = sorted((Path(job['job_dir'])/'work').glob('light_*.fit'))
    assert len(converted) == 3
    for path in converted:
        actual = fits.getdata(path)
        assert actual.shape == (3,18,16)
        # Decoded fixture has its sole >1 marker at stored upper left. A 90°
        # displayed turn maps it to stored lower left under BOTTOM-UP.
        assert np.all(actual[:,-1,0] > 1)
        assert np.count_nonzero(actual > 1) == 3


def test_partial_rotation_restarts_conversion_before_resume(cfa_env,monkeypatch):
    engine_calls=[]
    def engine(*args):
        engine_calls.append(args[2]); return fake_engine(*args)
    monkeypatch.setattr(core,'run_script',engine)
    original_rotate=orientation.rotate_fits
    rotate_calls=[]
    def cancel_after_first(source,destination,angle,**kwargs):
        original_rotate(source,destination,angle,**kwargs)
        rotate_calls.append(str(source))
        if len(rotate_calls)==1:
            core.cancel_job(str(Path(source).parent.parent))
            kwargs['check']()
    monkeypatch.setattr(orientation,'rotate_fits',cancel_after_first)
    job=core.submit_job(str(cfa_env.source),str(cfa_env.output),image_rotation=90)
    core.run_job(job['job_dir'])
    failed=core.load_job(job['job_dir'])
    assert failed['state']=='cancelled'
    assert 'convert' not in failed['checkpoints']
    assert fits.getdata(Path(job['job_dir'])/'work/light_00001.fit').shape==(3,18,16)
    core.resume_job(job['job_dir']); core.run_job(job['job_dir'])
    done=core.load_job(job['job_dir'])
    assert done['state']=='succeeded',done.get('error')
    assert engine_calls.count('debayer')==2
    assert len(rotate_calls)==4
    assert_once_rotated(done)


def test_after_conversion_checkpoint_resume_never_rotates_again(cfa_env,monkeypatch):
    stopped=[]; engine_calls=[]
    def engine(*args):
        engine_calls.append(args[2])
        if args[2]=='02_probe' and not stopped:
            stopped.append(True)
            core.cancel_job(str(args[1])); args[4]()
        return fake_engine(*args)
    monkeypatch.setattr(core,'run_script',engine)
    original_rotate=orientation.rotate_fits
    rotate_calls=[]
    def counted(*args,**kwargs):
        rotate_calls.append(args[0]);return original_rotate(*args,**kwargs)
    monkeypatch.setattr(orientation,'rotate_fits',counted)
    job=core.submit_job(str(cfa_env.source),str(cfa_env.output),image_rotation=90)
    core.run_job(job['job_dir'])
    stopped_job=core.load_job(job['job_dir'])
    assert stopped_job['state']=='cancelled'
    assert 'convert' in stopped_job['checkpoints']
    before={path:core._sha(path) for path in (Path(job['job_dir'])/'work').glob('light_*.fit')}
    core.resume_job(job['job_dir']);core.run_job(job['job_dir'])
    done=core.load_job(job['job_dir'])
    assert done['state']=='succeeded',done.get('error')
    assert engine_calls.count('debayer')==1 and len(rotate_calls)==3
    assert {path:core._sha(path) for path in before}==before
    assert_once_rotated(done)


@pytest.mark.parametrize('saved_rotation',[None,90])
def test_open_core_job_replaces_previous_scene_orientation(window,tmp_path,monkeypatch,saved_rotation):
    window._image_rotation=270
    monkeypatch.setattr(window,'refresh_job',lambda:None)
    record={'job_dir':str(tmp_path/'job-review'), 'state':'failed'}
    if saved_rotation is not None:
        record['image_rotation']=saved_rotation
    window._submitted(record)
    window.timer.stop()
    assert window._image_rotation == (0 if saved_rotation is None else saved_rotation)
