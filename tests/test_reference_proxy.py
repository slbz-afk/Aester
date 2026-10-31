"""Preview-only proxy contracts; scientific frame geometry stays unchanged."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image
from starstack import references as ref, master_export as me, color_export as ce, photo_render as pr
from starstack import core, display_transform as dt, color_adjustments as ca
from starstack.whole_color import resize_signed
sys.path.insert(0,str(Path(__file__).parent))
import color_fixtures as cf


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    source=cf.write_reference_dng(tmp_path/'source.dng')
    basis=ce.job_colour_basis(source)
    camera=np.random.default_rng(411).uniform(.018,.023,(256,512,3)).astype(np.float32)
    camera[:16]=0
    monkeypatch.setattr(ref,'PREVIEW_MAX_SIDE',128)
    monkeypatch.setattr(ref,'INTERACTIVE_MAX_SIDE',64)
    directory=tmp_path/'preview';directory.mkdir()
    return source,basis,camera,directory


def record_for(source,camera,directory):
    image,cache=ref._prepare_color_preview(np.flipud(camera),source,directory,{})
    record={'state':'succeeded','reference_dir':str(directory),'color_cache':cache,
            'width':camera.shape[1],'height':camera.shape[0],
            'preview_width':image.shape[1],'preview_height':image.shape[0]}
    core.atomic_json(directory/'reference.json',record)
    return image,record


def test_signed_proxy_affine_order_preserves_nodata_and_negative_values(fixture):
    _,basis,camera,_=fixture
    camera[40:80,40:90]=np.array([-.001,.002,-.004],np.float32)
    before=camera.copy()
    full,_,_=me.convert_camera_master(camera,basis,'rec2020')
    expected=resize_signed(full,128)
    proxy,_=ref._colour_proxy(camera,basis,128)
    np.testing.assert_allclose(proxy,expected,atol=1e-7,rtol=2e-6)
    assert np.any(proxy<0)
    assert (proxy[:4]==0).all()
    np.testing.assert_array_equal(camera,before)


def test_saturated_pinpoint_survives_strong_area_reduction(fixture):
    _,basis,camera,_=fixture
    camera.fill(.02);camera[23,211,1]=float(basis['white_DN'])/65535
    _,sat=ref._colour_proxy(camera,basis,8)
    assert sat.sum()==1


def test_cancellation_to_zero_is_not_misclassified_as_nodata(fixture):
    _,basis,_,_=fixture
    camera=np.full((32,64,3),.02,np.float32)
    camera[:,1::2]*=-1
    proxy,_=ref._colour_proxy(camera,basis,8)
    full,_,_=me.convert_camera_master(camera,basis,'rec2020')
    np.testing.assert_allclose(proxy,resize_signed(full,8),atol=1e-7,rtol=1e-5)


def test_statistics_equal_original_resolution_default(fixture):
    source,basis,camera,directory=fixture
    image,record=record_for(source,camera,directory)
    full,_,_=me.convert_camera_master(camera,basis,'rec2020')
    meta=record['color_cache']['master_metadata']
    expected=pr.default_params(meta,full)
    expected=replace(expected,shadow_smooth_radius=max(1,round(expected.shadow_smooth_radius*image.shape[0]/camera.shape[0])))
    assert record['color_cache']['base_params']==expected.to_json()
    assert record['width']==512 and record['height']==256


@pytest.mark.parametrize('quality,size',[('standard',128),('interactive',64)])
def test_warm_render_reads_only_proxy_and_keeps_original_geometry(fixture,monkeypatch,quality,size):
    source,_,camera,directory=fixture
    _,record=record_for(source,camera,directory)
    snapshot=(directory/'reference.json').read_bytes()
    source.unlink() # The preview must remain usable without re-opening RAW.
    monkeypatch.setattr(me,'convert_camera_master',lambda *_a,**_k:pytest.fail('must not re-convert camera pixels'))
    monkeypatch.setattr(ref.siril,'run_script',lambda *_a,**_k:pytest.fail('must not run Siril'))
    result=ref.render_reference(directory,color_adjustments={'warmth':18,'tint':-4,'exposure_ev':.3},quality=quality)
    assert result['width']==512 and result['height']==256
    assert result['preview_width']==size and result['preview_height']==size//2
    assert Image.open(result['preview_path']).size==(size,size//2)
    assert result['preview_scale']==size/512
    assert (directory/'reference.json').read_bytes()==snapshot


def test_parallel_previews_are_independent_immutable_files(fixture):
    source,_,camera,directory=fixture
    record_for(source,camera,directory)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:ref.render_reference(directory,quality='interactive'),range(4)))
    assert len({x['preview_path'] for x in results})==4
    images=[np.asarray(Image.open(x['preview_path'])) for x in results]
    assert all(np.array_equal(images[0],x) for x in images[1:])


@pytest.mark.parametrize('kind',['array','metadata','external'])
def test_interactive_cache_identity_is_checked(fixture,kind):
    source,_,camera,directory=fixture
    _,record=record_for(source,camera,directory)
    cache=record['color_cache']
    if kind=='array':
        Path(cache['interactive']['linear']).write_bytes(b'changed')
    elif kind=='metadata':cache['interactive']['width']+=1
    else:
        outside=directory.parent/'outside.npy'
        outside.write_bytes(Path(cache['interactive']['linear']).read_bytes())
        cache['interactive']['linear']=str(outside)
        cache['signature']=hashlib.sha256(json.dumps({k:v for k,v in cache.items() if k!='signature'},sort_keys=True,allow_nan=False).encode()).hexdigest()
    core.atomic_json(directory/'reference.json',record)
    with pytest.raises(ValueError):ref.render_reference(directory,quality='interactive')


def test_old_cache_remains_readable_for_both_qualities(fixture):
    source,_,camera,directory=fixture
    _,record=record_for(source,camera,directory)
    cache=record['color_cache'];cache.pop('interactive');cache['version']=1
    cache['signature']=hashlib.sha256(json.dumps({k:v for k,v in cache.items() if k!='signature'},sort_keys=True,allow_nan=False).encode()).hexdigest()
    core.atomic_json(directory/'reference.json',record)
    for quality,size in [('standard',128),('interactive',64)]:
        result=ref.render_reference(directory,quality=quality)
        assert result['preview_width']==size


def test_invalid_quality_rejected_before_any_source_access():
    with pytest.raises(ValueError,match='质量'):ref.render_reference('not-a-reference',quality='bogus')


def test_selected_48_dng_frames_decode_only_first_without_duplicate_raw_unpack(fixture,monkeypatch,tmp_path):
    source,_,camera,_=fixture
    source_dir=tmp_path/'sequence';source_dir.mkdir()
    sources=[]
    for i in range(48):
        path=source_dir/f'{i:03}.dng';path.write_bytes(source.read_bytes());sources.append(str(path))
    decoded=[]
    monkeypatch.setattr(core,'doctor',lambda *_:{'available':True,'executable':'fixture','version':'1.4.4'})
    monkeypatch.setattr(ref,'_optional_raw_metadata',lambda *_:pytest.fail('DNG preview must not unpack pixels twice'))
    def convert(executable,directory,name,commands,check):
        from starstack.imaging import write_linear
        decoded.extend((directory/'stage').glob('*.dng'))
        write_linear(directory/'work/ref_00001.fit',np.flipud(camera),'fixture')
    monkeypatch.setattr(ref.siril,'run_script',convert)
    result=ref.prepare_reference(None,str(tmp_path/'result'),input_files=sources,color_adjustments={})
    assert len(decoded)==1 and result['source_path']==sources[0]
    assert result['input_count']==48
    assert result['width']==512 and result['height']==256
    assert Image.open(result['mask_template_path']).size==(512,256)
