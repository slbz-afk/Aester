"""Independent preview/intent and bounded parallelism contracts."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image
from starstack import color_adjustments as ca, display_transform as dt, performance
from starstack import references, color_export as ce, master_export as me, photo_render as pr
sys.path.insert(0, str(Path(__file__).parent))
import color_fixtures as cf


@pytest.mark.parametrize('bad', [True, [], {'x': 1}, {'warmth': True}, {'warmth': 101},
                              {'tint': float('nan')}, {'exposure_ev': 4.01}, {'tint': '2'}])
def test_reject_invalid_intent(bad):
    with pytest.raises(ValueError):
        ca.normalize(bad)


def color_data(tmp_path):
    source=cf.write_reference_dng(tmp_path/'reference.dng')
    basis=ce.job_colour_basis(source)
    camera,*_=cf.synthetic_camera_master()
    data,sat,_=me.convert_camera_master(camera,basis,'rec2020')
    meta={'gamut':'rec2020','colour_basis':{**basis,'master_matrix':me.master_matrix(basis,'rec2020').tolist()}}
    return source,camera,data,sat,meta,pr.default_params(meta,data)


def test_zero_exact_and_gain_matrix_independent_camera_oracle(tmp_path):
    _,_,data,_,meta,base=color_data(tmp_path)
    assert ca.apply(base,meta,{}) is base
    intent={'warmth':27.,'tint':-18.,'exposure_ev':.7}
    new=ca.apply(base,meta,intent)
    matrix=np.asarray(meta['colour_basis']['master_matrix'])
    old=np.asarray(meta['colour_basis']['gains_rgb'])
    neutral=np.asarray(base.wb_gains_rgb)
    factors=np.array([2**(.27-.09),1,2**(-.27-.09)])
    # Decode master pixels to camera coordinates; reapply the altered camera gains.
    camera=np.asarray(data,dtype=np.float64) @ np.linalg.inv(matrix).T
    expected=(camera*(neutral/old)*factors) @ matrix.T * 2**.7
    actual=np.asarray(data,dtype=np.float64) @ np.asarray(new.wb_matrix).T
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    assert new.anchor==base.anchor
    assert new.black==base.black


def test_exposure_changes_photo_not_master(tmp_path):
    _,_,data,sat,meta,base=color_data(tmp_path)
    before=data.copy()
    values=[]
    for ev in (-1,0,1):
        params=ca.apply(base,meta,{'exposure_ev':ev})
        result,_=dt.render(data,params,camera_saturated=sat)
        values.append(result.astype(np.int32))
    # Restrict comparison to neutral unsaturated synthetic pixels; do not claim channel monotonicity through gamut mapping.
    neutral=np.ptp(data,axis=2)<1e-5
    if not neutral.any():
        neutral=np.ones(data.shape[:2],bool)
    assert np.mean(values[2])>np.mean(values[1])>np.mean(values[0])
    np.testing.assert_array_equal(data,before)


@pytest.mark.parametrize('height,rows',[(257,32),(300,128),(13,3)])
@pytest.mark.parametrize('radius',[0,3])
def test_parallel_is_bit_exact_including_halos_flags_and_statistics(height,rows,radius):
    rng=np.random.default_rng(193)
    data=rng.normal(.05,.08,(height,61,3)).astype(np.float32)
    data[2,7]=[1.3,.9,0.]
    sat=rng.random((height,61))<.01
    region=rng.integers(0,3,(height,61),dtype=np.uint8)
    params=dt.Params(anchor=.035,wb_matrix=((1.1,.1,-.1),(0,1,.02),(.03,0,.8)),
                     shadow_smooth_radius=radius,shadow_smooth_start=.02 if radius else 0.,
                     shadow_smooth_end=.1 if radius else 0.)
    a=np.zeros((height,61),np.uint8);b=a.copy()
    serial,s0=dt.render(data,params,camera_saturated=sat,region=region,flags=a,rows=rows)
    parallel,s1=dt.render(data,params,camera_saturated=sat,region=region,flags=b,rows=rows,workers=4)
    np.testing.assert_array_equal(parallel,serial)
    np.testing.assert_array_equal(a,b)
    assert s0==s1


def test_parallel_invalid_and_callback_safety():
    data=np.ones((270,10,3),np.float32)*.04
    for bad in (0,9,True,1.5):
        with pytest.raises(ValueError):dt.render(data,dt.Params(),workers=bad)
    seen=[]
    def look(x):
        import threading
        seen.append(threading.get_ident());return x
    dt.render(data,dt.Params(),look=look,workers=4)
    assert len(set(seen))==1
    data[200,1,0]=np.nan
    with pytest.raises(ValueError):dt.render(data,dt.Params(),workers=4)


def test_memory_budget_is_bounded():
    # ceiling raised from 4 to 8 (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md A); ≤8-CPU machines unchanged
    assert performance.render_workers(6000,8000,cpu_count=192,memory_bytes=128*1024**3)==8
    assert performance.render_workers(6000,8000,cpu_count=8,memory_bytes=128*1024**3)==4
    assert performance.render_workers(6000,8000,cpu_count=192,memory_bytes=1024**3)==1
    assert performance.render_workers(6000,8000,cpu_count=2,memory_bytes=128*1024**3)==1
    assert performance.render_workers(100,100,cpu_count=192,memory_bytes=128*1024**3)==1


def test_worker_rollback_and_invalid_config(monkeypatch):
    monkeypatch.setenv('AESTER_RENDER_WORKERS','1')
    assert performance.render_workers(6000,8000,cpu_count=192,memory_bytes=128*1024**3)==1
    monkeypatch.setenv('AESTER_RENDER_WORKERS','2')
    assert performance.render_workers(6000,8000,cpu_count=192,memory_bytes=128*1024**3)==2
    monkeypatch.setenv('AESTER_RENDER_WORKERS','192')
    with pytest.raises(ValueError):performance.render_workers(6000,8000)


def test_preview_matches_shared_full_size_color_at_small_size_and_is_immutable(tmp_path):
    source,camera,data,sat,meta,base=color_data(tmp_path)
    directory=tmp_path/'cache';directory.mkdir()
    intent={'warmth':20,'tint':8,'exposure_ev':.3}
    image,cache=references._prepare_color_preview(np.flipud(camera),source,directory,intent)
    expected,_=dt.render(data,ca.apply(base,meta,intent),camera_saturated=sat)
    np.testing.assert_array_equal(image,dt.to_8bit(expected))
    record={'reference_dir':str(directory),'state':'succeeded','color_cache':cache}
    (directory/'reference.json').write_text(json.dumps(record),encoding='utf-8')
    original=(directory/'reference.json').read_bytes()
    new=references.render_reference(directory,color_adjustments=intent)
    np.testing.assert_array_equal(np.asarray(Image.open(new['preview_path'])),image)
    assert (directory/'reference.json').read_bytes()==original
    references.render_reference(directory,color_adjustments={'exposure_ev':-1})
    assert len(list(directory.glob('adjusted-*.png')))==2
    cache['base_params']['anchor']*=2
    (directory/'reference.json').write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(ValueError,match='配方'):references.render_reference(directory)
