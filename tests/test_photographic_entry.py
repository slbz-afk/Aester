"""The new finishing boundary: explicit display domain, immutable inputs and isolated outputs."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from starstack import core
from starstack.photographic_job import finish_photo


@pytest.fixture
def scene(tmp_path):
    h, w = 160, 240
    yy, xx = np.mgrid[:h, :w]
    value = .15 + .06*np.sin(xx/31) + .04*np.exp(-((xx-120)**2+(yy-55)**2)/800)
    rgb = np.rint(value[..., None]*np.array([.88,1,.95])*65535).astype(np.uint16)
    sky = np.full((h,w),255,np.uint8);sky[120:]=0;sky[117:120]=128
    guard = np.zeros((h,w),np.uint8);guard[20:100,100:104]=255
    source = tmp_path/'source.tif'
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tifffile.imwrite(source,rgb,photometric='rgb',metadata={'scene_linear':False,'display_encoding_applied':True,
                     'roworder':'TOP-DOWN','color_space':'encoded_sRGB'},
                     extratags=[(34675,'B',len(icc),icc,False)])
    Image.fromarray(sky).save(tmp_path/'sky.png');Image.fromarray(guard).save(tmp_path/'guard.png')
    return source,tmp_path/'sky.png',tmp_path/'guard.png',rgb,sky,guard


def test_zero_strength_roundtrip_is_exact_and_repeat_is_new_directory(scene,tmp_path):
    source,sky,protect,rgb,_,_=scene
    before={p:core._sha(p) for p in (source,sky,protect)}
    records=[finish_photo(source,tmp_path/'out',sky,protect,detail_strength=0,star_strength=0) for _ in range(2)]
    assert all(r['state']=='succeeded' for r in records)
    assert records[0]['processing_directory'] != records[1]['processing_directory']
    for r in records:
        dst=Path(r['artifacts']['photographic_srgb16.tif']['path'])
        np.testing.assert_array_equal(tifffile.imread(dst),rgb)
        with tifffile.TiffFile(dst) as tf:
            assert tf.pages[0].dtype==np.uint16 and tf.pages[0].shape==rgb.shape
            assert 34675 in tf.pages[0].tags
        assert r['source_unchanged'] and not r['generative_model_used']
        assert json.loads((dst.parent/'result.json').read_text(encoding='utf-8'))['state']=='succeeded'
    assert before=={p:core._sha(p) for p in before}


def test_enhancement_holds_protection_mixed_and_ground(scene,tmp_path):
    source,sky,protect,rgb,mask,guard=scene
    r=finish_photo(source,tmp_path/'out',sky,protect)
    assert r['state']=='succeeded',r
    a=tifffile.imread(r['artifacts']['photographic_srgb16.tif']['path'])
    held=(mask<255)|(guard>0)
    np.testing.assert_array_equal(a[held],rgb[held])
    assert np.any(a[~held]!=rgb[~held])


def test_ground_entry_changes_only_explicit_unprotected_ground(scene,tmp_path):
    source,sky,protect,rgb,mask,guard=scene
    r=finish_photo(source,tmp_path/'out',sky,protect,detail_strength=0,star_strength=0,ground_strength=1)
    assert r['state']=='succeeded',r
    a=tifffile.imread(r['artifacts']['photographic_srgb16.tif']['path'])
    held=(mask>0)|(guard>0)
    np.testing.assert_array_equal(a[held],rgb[held])
    assert np.any(a[~held]!=rgb[~held])


def test_input_change_during_render_fails_without_publication(scene,tmp_path,monkeypatch):
    from starstack import photographic
    source,sky,protect,*_=scene
    render=photographic.render
    def change(*args,**kwargs):
        stats=render(*args,**kwargs)
        with protect.open('ab') as stream:stream.write(b'changed')
        return stats
    monkeypatch.setattr(photographic,'render',change)
    r=finish_photo(source,tmp_path/'out',sky,protect)
    assert r['state']=='failed' and not r['source_unchanged']
    assert Path(r['processing_directory']).name.startswith('.partial-')
    assert not list((tmp_path/'out').glob('photo-*'))


@pytest.mark.parametrize('bad',[True,-.1,1.01,float('nan')])
def test_ground_strength_rejected_before_output(scene,tmp_path,bad):
    with pytest.raises(ValueError):finish_photo(*[scene[0],tmp_path/'out',scene[1],scene[2]],ground_strength=bad)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['linear','missing_icc','wrong_mask','rgb_mask'])
def test_ambiguous_domains_and_masks_rejected(scene,tmp_path,kind):
    source,sky,protect,rgb,*_=scene
    if kind in ('linear','missing_icc'):
        icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        tifffile.imwrite(source,rgb,photometric='rgb',metadata={'scene_linear':kind=='linear'},
             extratags=[] if kind=='missing_icc' else [(34675,'B',len(icc),icc,False)])
    elif kind=='wrong_mask':Image.new('L',(239,160)).save(sky)
    else:Image.new('RGB',(240,160)).save(sky)
    with pytest.raises(ValueError):finish_photo(source,tmp_path/'out',sky,protect)
    assert not (tmp_path/'out').exists()


def test_mcp_paths_are_bounded_before_renderer(tmp_path,monkeypatch):
    from starstack.mcp_server import WorkspaceTools,AccessDenied
    from starstack import photographic_job
    inputs=tmp_path/'raw';outputs=tmp_path/'out'
    inputs.mkdir();outputs.mkdir()
    foreign=tmp_path/'foreign.tif';foreign.write_bytes(b'no image')
    service=WorkspaceTools(inputs,outputs)
    monkeypatch.setattr(photographic_job,'finish_photo',lambda *a,**kw:pytest.fail('renderer reached'))
    with pytest.raises(AccessDenied):service.finish_photo(str(foreign),'sky.png','guard.png')


def test_subnormal_float64_protect_survives_conversion(tmp_path):
    from starstack.photographic_job import _mask
    a=np.zeros((12,14),np.float64);a[4,5]=1e-50
    p=tmp_path/'mask.tif';tifffile.imwrite(p,a,metadata={'roworder':'TOP-DOWN'})
    result=_mask(p,a.shape,binary=True)
    assert result[4,5] and result.sum()==1


def test_wrong_mask_orientation_is_refused(tmp_path):
    from starstack.photographic_job import _mask
    p=tmp_path/'mask.tif';tifffile.imwrite(p,np.ones((12,14),np.uint8),extratags=[(274,'H',1,4,False)])
    with pytest.raises(ValueError):_mask(p,(12,14),binary=True)


def test_float64_mixed_weight_cannot_become_pure_after_conversion(tmp_path):
    from starstack.photographic_job import _mask
    a=np.zeros((12,14),np.float64);a[4,5]=1e-50;a[5,5]=np.nextafter(1.,0.)
    p=tmp_path/'weight.tif';tifffile.imwrite(p,a,metadata={'roworder':'TOP-DOWN'})
    result=_mask(p,a.shape)
    assert 0<result[4,5]<1 and 0<result[5,5]<1
