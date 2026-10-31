import json
from pathlib import Path
import numpy as np
import pytest
import tifffile
from PIL import Image,ImageCms
from starstack.star_separation import compact_estimate,separate_stars


def fixture_files(tmp_path):
    y,x=np.mgrid[:96,:120]
    a=np.stack([6000+x*20+y*2,8000+y*12+x*3,9000+x*5+y*7],-1).astype(np.uint16)
    a[25:28,60:63]+=22000
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    src=tmp_path/'source.tif'
    tifffile.imwrite(src,a,photometric='rgb',metadata={'scene_linear':False,'display_encoding_applied':True,
        'color_space':'encoded_sRGB','roworder':'TOP-DOWN'},extratags=[(34675,'B',len(icc),icc,False)])
    sky=np.ones(a.shape[:2],np.float32);sky[80:]=0;sky[79]=.999
    protect=np.zeros(a.shape[:2],np.uint8);protect[:,3]=255
    sp=tmp_path/'sky.tif';pp=tmp_path/'protect.png'
    tifffile.imwrite(sp,sky);Image.fromarray(protect).save(pp)
    return src,sp,pp,a,sky,protect.astype(bool)


def test_compact_exact_constant_and_tile_invariance():
    a=np.full((81,117,3),21000,np.uint16);out=np.empty_like(a)
    compact_estimate(a,out,rows=7);assert np.array_equal(a,out)
    rng=np.random.default_rng(5);a=rng.integers(6000,30000,(81,117,3),dtype=np.uint16)
    a.flags.writeable=False;before=a.copy();large=np.empty_like(a);small=np.empty_like(a)
    compact_estimate(a,large,radius=4,rows=128);compact_estimate(a,small,radius=4,rows=7)
    assert np.array_equal(large,small)
    assert np.array_equal(a,before)
    assert np.all(large<=a)


def test_signed_negative_roundtrip_and_locks(tmp_path):
    src,sky,protect,a,w,m=fixture_files(tmp_path)
    def estimator(s,d,stage):
        d[:]=np.clip(s.astype(np.int32)+np.where(np.indices(s.shape[:2])[0][...,None]%2,100,-500),0,65535)
        return {'synthetic':True}
    r=separate_stars(src,tmp_path/'jobs',sky,protect,provider=estimator,provider_identity={'backend':'independent-synthetic','uses_learned_model':True})
    d=tifffile.imread(r['artifacts']['diffuse_srgb16.tif']['path'])
    residual=tifffile.imread(r['artifacts']['stars_signed_codes.tif']['path'])
    assert residual.dtype==np.int32 and residual.min()==-100 and residual.max()==500
    assert np.array_equal(a.astype(np.int32),d.astype(np.int32)+residual)
    assert np.array_equal(d[(w!=1)|m],a[(w!=1)|m])
    assert np.array_equal(tifffile.imread(src),a)
    assert r['quality_status']=='not_accepted' and r['roundtrip_is_quality_test'] is False
    assert r['learned_star_separation'] is True
    with tifffile.TiffFile(r['artifacts']['diffuse_srgb16.tif']['path']) as t:
        assert json.loads(t.pages[0].description)['learned_star_separation'] is True


def test_provider_failure_never_publishes(tmp_path):
    src,sky,protect,*_=fixture_files(tmp_path)
    def fail(*_):raise RuntimeError('bad backend')
    with pytest.raises(RuntimeError,match='bad backend'):
        separate_stars(src,tmp_path/'jobs',sky,protect,provider=fail,provider_identity={'backend':'failure'})
    children=list((tmp_path/'jobs').iterdir())
    assert len(children)==1 and children[0].name.startswith('.partial-')
    assert json.loads((children[0]/'result.json').read_text('utf8'))['state']=='failed'


def test_provider_input_mutation_rejected(tmp_path):
    src,sky,protect,*_=fixture_files(tmp_path)
    def tamper(s,d,stage):
        d[:]=s
        with sky.open('ab') as f:f.write(b'changed')
        return {}
    with pytest.raises(ValueError,match='input changed'):
        separate_stars(src,tmp_path/'jobs',sky,protect,provider=tamper,provider_identity={'backend':'tamper'})
    assert not any(p.name.startswith('separation-') for p in (tmp_path/'jobs').iterdir())


@pytest.mark.parametrize('radius',[0,33,True,2.5,np.int32(2)])
def test_bad_radius_and_alias(radius):
    a=np.zeros((8,9,3),np.uint16)
    with pytest.raises(ValueError):compact_estimate(a,np.empty_like(a),radius=radius)
    with pytest.raises(ValueError):compact_estimate(a,a)
