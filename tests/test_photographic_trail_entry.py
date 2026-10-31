"""Optional photographic renderer must reach the same published revision machinery."""
import copy
from pathlib import Path
import numpy as np
import pytest
from starstack import transient_stage as ts,transient_trails as tt,trail_finish
from starstack.synthetic_starfield import write_fits_rgb32

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


def test_stage_optional_render_preserves_holes_and_duplicate_identity(tmp_path):
    shape=(96,192)
    master=np.full((3,*shape),.2,np.float32);frame=master.copy()
    frame[:,46:51,30:151]+=.08;frame[:,48,80]=0
    write_fits_rgb32(tmp_path/'sky_master.fit',master)
    write_fits_rgb32(tmp_path/'r_light_00001.fit',frame)
    c={'id':'one','stem':'light_00001','decision':'keep','measurement':{'status':'measured','x0':30.,'y0':48.,'x1':150.,'y1':48.,'half_width_px':2.,'noise':[.002]*3}}
    record={'candidates':[c,copy.deepcopy(c)],'params':{'layer':{'style':'photographic_v3'}},
        'reference_stem':'light_00001','frame_stats':{'light_00001':{'median':[.2]*3,'sigma':[.01]*3}}}
    # Identical statistics -> identity normalization; this test concerns layer support.
    record['reference_stem']=None
    boxes,kept=ts.native_boxes(tmp_path,record,own_registration=False,model=None,native_shape=shape)
    assert len(boxes)==1 and len(kept)==2
    b=boxes[0];x,y=b['x0'],b['y0']
    assert np.all(b['data'][:,48-y,78-x:83-x]==0)
    assert all(c['application']['photographic_render']['version']==trail_finish.VERSION for c in record['candidates'])
    # Two aliases from one source are not added twice.
    one=copy.deepcopy(record);one['candidates']=one['candidates'][:1]
    np.testing.assert_array_equal(ts.native_boxes(tmp_path,one,own_registration=False,model=None,native_shape=shape)[0][0]['data'],b['data'])


def test_style_validation_and_source_default(tmp_path):
    record={'candidates':[],'policy':tt.DEFAULT_POLICY.copy(),'params':{'layer':tt.DEFAULT_LAYER.copy()}}
    with pytest.raises(ValueError):ts.update_record(tmp_path,copy.deepcopy(record),layer_style='unknown',persist=False)
    updated=ts.update_record(tmp_path,copy.deepcopy(record),layer_style='photographic_v3',persist=False)
    assert updated['params']['layer']['style']=='photographic_v3'
    assert 'style' not in record['params']['layer']


def test_signed_union_is_weighted_and_order_independent():
    def box(key,estimate,weight):
        return {'candidate_id':key,'source_key':('a','b',(1,1,1),(0,0,0),(0,0,0)),
            'grid_shape':(20,20),'x0':3,'y0':4,'data':np.full((3,5,5),estimate*weight,np.float32),
            'weight':np.full((5,5),weight,np.float64),'validity':np.ones((5,5),bool)}
    a,b=box('a',-.4,1),box('b',.6,.5)
    actual=trail_finish.union_photographic_boxes([a,b])[0]['data']
    expected=(np.float64(np.float32(-.4))+np.float64(np.float32(.3)))/1.5
    np.testing.assert_array_equal(actual,np.full_like(actual,expected))
    np.testing.assert_array_equal(actual,trail_finish.union_photographic_boxes([b,a,a])[0]['data'])


def test_optional_renderer_through_real_revision_publication():
    # Controlled five-exposure product fixture, not an actual Siril or camera claim.
    from test_s4_accept_transients import AcceptanceScriptTest
    from starstack import core
    case=AcceptanceScriptTest('test_script_judges_a_synthetic_job')
    case.setUp()
    try:
        job=core.submit_job(str(case.source),str(case.output),sky_mask_path=str(case.mask_png))
        core.run_job(job['job_dir']);before=core.load_job(job['job_dir'])
        assert before['state']=='succeeded',before
        work=Path(job['job_dir'])/'work'
        master_hash=core._sha(work/'sky_master.fit')
        old_record=ts.read_record(work,job=before)
        new=core.recompose_transients(job['job_dir'],layer_style='photographic_v3')
        record=ts.read_record(work,job=new)
        assert record['params']['layer']['style']=='photographic_v3'
        assert new['transient_revision']['id']!=before['transient_revision']['id']
        assert core._sha(work/'sky_master.fit')==master_hash
        assert ts.read_record(work,job=before)==old_record
        assert any(c.get('application',{}).get('photographic_render') for c in record['candidates'])
        reset=core.recompose_transients(job['job_dir'],layer_style='source_residual')
        assert ts.read_record(work,job=reset)['params']['layer']['style']=='source_residual'
        assert core._sha(work/'sky_master.fit')==master_hash
    finally:
        case.tearDown();case.doCleanups()
