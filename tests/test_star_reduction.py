"""Root acceptance against independent display arithmetic and failure cases."""
import json
import numpy as np
import pytest
import tifffile
from PIL import Image,ImageCms
from starstack.star_reduction_curve import reduction_delta
from starstack.star_reduction import reduce_stars, bright_control
from starstack.star_separation import separate_stars
from starstack import core


def scalar_delta(B,S,D,K,W,strength=.55,lo=.10,hi=.45):
    P=[max(int(b)-int(d),0) for b,d in zip(B,D)]
    room=[min(max(int(b)-1,0),max(int(s)-1,0)) for b,s in zip(B,S)]
    u=min(1,max(0,(int(K)/65535-lo)/(hi-lo)))
    a=strength*float(W)*(1-u*u*(3-2*u))
    a=min(a,*[min(1,c/p) if p else 1 for c,p in zip(room,P)])
    return [min(round(p*a),c,p) for p,c in zip(P,room)]


def test_kernel_independent_scalar_and_signed_headroom():
    rng=np.random.default_rng(481)
    B=rng.integers(0,65536,(13,21,3),dtype=np.uint16)
    S=rng.integers(0,65536,B.shape,dtype=np.uint16)
    D=rng.integers(0,65536,B.shape,dtype=np.uint16)
    B[0,:3]=[0,1,65535];S[0,:3]=[1,0,1];D[0,:3]=[100,0,0]
    P=np.maximum(B.astype(np.int32)-D,0);K=P.max(2).astype(np.uint16)
    W=rng.random(B.shape[:2]).astype(np.float32)
    saved=[a.copy() for a in (B,S,D,K,W)]
    for a in (B,S,D,K,W):a.flags.writeable=False
    delta=reduction_delta(B,S,D,K,W,strength=.9)
    ref=np.array([[scalar_delta(B[y,x],S[y,x],D[y,x],K[y,x],W[y,x],.9) for x in range(21)] for y in range(13)],np.int32)
    np.testing.assert_array_equal(delta,ref)
    assert delta.dtype==np.int32 and delta.flags.c_contiguous
    for a,old in zip((B,S,D,K,W),saved):
        np.testing.assert_array_equal(a,old);assert not np.shares_memory(delta,a)
    assert np.all(delta>=0) and np.all(delta<=P)
    assert np.all(delta[(B.astype(np.int32)-D)<0]==0)
    np.testing.assert_array_equal((S.astype(np.int32)-delta)-(B.astype(np.int32)-delta),S.astype(np.int32)-B)
    assert not np.any(((S.astype(np.int32)-delta)==0)&(S>0))


def test_curve_order_color_and_bright_identity():
    x=np.arange(0,65536,257,dtype=np.uint16)[None,:]
    B=np.repeat(x[...,None],3,axis=2);D=np.zeros_like(B);W=np.ones(x.shape,np.float64)
    delta=reduction_delta(B,B,D,x,W)
    F=B.astype(np.int32)-delta
    assert np.all(np.diff(F[0,:,0])>=0)
    assert np.all(delta[x>=.45*65535]==0)
    assert np.all(reduction_delta(B,B,D,x,W,strength=0)==0)
    B=np.array([[[2000,4000,8000]]],np.uint16);D=np.full_like(B,1000)
    K=(B.astype(np.int32)-D).max(2).astype(np.uint16)
    v=reduction_delta(B,B,D,K,np.ones((1,1),np.float32),bright_start=.2)
    gain=v.astype(float)/(B.astype(np.int32)-D)
    assert np.ptp(gain)<.001


@pytest.mark.parametrize('value',[True,np.float32(.5),np.int64(1),-1,2,float('nan'),float('inf'),10**400])
@pytest.mark.parametrize('param',['strength','bright_start','bright_end'])
def test_kernel_invalid_parameters(value,param):
    B=np.full((2,3,3),1000,np.uint16);D=np.zeros_like(B);K=B.max(2);W=np.ones((2,3),np.float32)
    with pytest.raises(ValueError):reduction_delta(B,B,D,K,W,**{param:value})


def test_kernel_views_validation_and_zero_validation():
    B=np.full((6,8,3),2000,np.uint16);D=np.full_like(B,1000);K=np.full((6,8),1000,np.uint16);W=np.ones((6,8),np.float64)
    ref=reduction_delta(B,B,D,K,W)
    np.testing.assert_array_equal(reduction_delta(B[::-1,::2].astype('>u2'),B[::-1,::2],D[::-1,::2],K[::-1,::2],W[::-1,::2]),ref[::-1,::2])
    for bad in [W.tolist(),np.zeros((6,8),np.int32),np.full_like(W,np.nan),W[:3],np.full_like(W,1.001)]:
        with pytest.raises(ValueError):reduction_delta(B,B,D,K,bad,strength=0)
    with pytest.raises(ValueError):reduction_delta(B,B,D,K-1,W)
    with pytest.raises(ValueError):reduction_delta(B,B,D,K,W,bright_start=.5,bright_end=.5)


def test_gaussian_bright_protection_streaming_and_radial_profile():
    y,x=np.mgrid[:101,:151];r2=(x-75)**2+(y-50)**2
    peak=np.rint(60000*np.exp(-r2/30)).astype(np.uint16)
    control=bright_control(peak,8)
    streamed=np.empty_like(control)
    for y0 in range(0,101,7):
        y1=min(101,y0+7);lo=max(0,y0-16);hi=min(101,y1+16)
        streamed[y0:y1]=bright_control(peak[lo:hi],8)[y0-lo:y1-lo]
    np.testing.assert_array_equal(streamed,control)
    assert np.all(control>=peak)
    B=np.repeat((peak+4000)[...,None],3,axis=2);D=np.full_like(B,4000)
    F=B.astype(np.int32)-reduction_delta(B,B,D,control,np.ones(peak.shape,np.float32),strength=.75)
    assert np.all(np.diff(F[50,75:,0])<=0) # no new radial ring maximum
    assert np.all(F>=D) and np.array_equal(F[50,75],B[50,75])


def make_separation(tmp,paired=False):
    shape=(96,120,3);B=np.full(shape,6000,np.uint16)
    B[20:24,30:34]=16000;B[40:44,50:54]=50000
    S=B.copy();S[:,35:38]=B[:,35:38]+1000
    if not paired:S=B.copy()
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    md={'scene_linear':False,'display_encoding_applied':True,'roworder':'TOP-DOWN','color_space':'encoded_sRGB'}
    def write(name,a,extra=None):
        path=tmp/name;tifffile.imwrite(path,a,photometric='rgb',metadata={**md,**(extra or {})},extratags=[(34675,'B',len(icc),icc,False)]);return path
    source=write('source.tif',S)
    base=write('base.tif',B,{'transient_background_for_sha256':core._sha(source)}) if paired else None
    sky=np.ones(shape[:2],np.float32);sky[80:]=0;sky[79]=.999
    protect=np.zeros(shape[:2],np.uint8);protect[:,35:38]=255
    sp=tmp/'sky.tif';pp=tmp/'protect.png';tifffile.imwrite(sp,sky);Image.fromarray(protect).save(pp)
    def provider(image,out,_):
        out[:]=np.maximum(image.astype(np.int32)-2000,0)
        out[10:15,60:65,2]=image[10:15,60:65,2]+400 # negative blue residual
        return {'synthetic':True}
    record=separate_stars(source,tmp/'separation',sp,pp,provider=provider,provider_identity={'backend':'test'},transient_background=base)
    path=Path(record['artifacts']['diffuse_srgb16.tif']['path']).parent/'result.json'
    return path,record,S,B,sky,protect.astype(bool)


from pathlib import Path

@pytest.mark.parametrize('paired',[False,True])
def test_job_invariants_zero_and_paired(tmp_path,paired):
    rp,sep,S,B,sky,protect=make_separation(tmp_path,paired)
    r=reduce_stars(rp,tmp_path/'runs',bright_radius=3)
    out=tifffile.imread(r['artifacts']['star_reduced_srgb16.tif']['path'])
    R=tifffile.imread(sep['artifacts']['stars_signed_codes.tif']['path'])
    newR=tifffile.imread(r['artifacts']['stars_reduced_signed_codes.tif']['path'])
    D=tifffile.imread(sep['artifacts']['diffuse_srgb16.tif']['path'])
    assert np.all(out<=S) and np.all(out[S>0]>0)
    np.testing.assert_array_equal(newR[R<0],R[R<0])
    np.testing.assert_array_equal(out[sky!=1],S[sky!=1])
    base=tifffile.imread(r['artifacts']['background_srgb16.tif']['path']) if paired else out
    np.testing.assert_array_equal(base.astype(np.int32),D.astype(np.int32)+newR)
    if paired:
        np.testing.assert_array_equal(out.astype(np.int32)-base,S.astype(np.int32)-B)
        with tifffile.TiffFile(r['artifacts']['background_srgb16.tif']['path']) as tf:
            assert json.loads(tf.pages[0].description)['transient_background_for_sha256']==r['artifacts']['star_reduced_srgb16.tif']['sha256']
    else:np.testing.assert_array_equal(out[protect],S[protect])
    z=reduce_stars(rp,tmp_path/'zero',strength=0)
    np.testing.assert_array_equal(tifffile.imread(z['artifacts']['star_reduced_srgb16.tif']['path']),S)


def test_mw_gated_relief_does_not_disable_general_reduction(tmp_path):
    rp,sep,S,*_=make_separation(tmp_path)
    mask=tmp_path/'mw.png';Image.fromarray(np.full(S.shape[:2],255,np.uint8)).save(mask)
    ep=tmp_path/'evidence.json'
    evidence={'version':'milkyway-subject-evidence-1','status':'present','method':'human-review','reviewer':'test','rationale':'synthetic gate test','source_sha256':core._sha(Path(sep['source'])),'milkyway_mask_sha256':core._sha(mask)}
    ep.write_text(json.dumps(evidence))
    u=reduce_stars(rp,tmp_path/'unknown',milkyway_mask=mask)
    p=reduce_stars(rp,tmp_path/'present',milkyway_mask=mask,milkyway_evidence=ep)
    U=tifffile.imread(u['artifacts']['star_reduced_srgb16.tif']['path']);P=tifffile.imread(p['artifacts']['star_reduced_srgb16.tif']['path'])
    assert np.any(U<S) and np.all(U<=P) and np.all(P<=S)
    evidence['status']='absent';ep.write_text(json.dumps(evidence))
    a=reduce_stars(rp,tmp_path/'absent',milkyway_mask=mask,milkyway_evidence=ep)
    np.testing.assert_array_equal(tifffile.imread(a['artifacts']['star_reduced_srgb16.tif']['path']),U)
    assert p['milkyway_relief_applied'] and not a['milkyway_relief_applied']


def test_tampering_and_nested_boundaries(tmp_path):
    rp,sep,*_=make_separation(tmp_path)
    with pytest.raises(ValueError):reduce_stars(rp,tmp_path/'bad',allowed_root=rp.parent)
    d=Path(sep['artifacts']['diffuse_srgb16.tif']['path'])
    with d.open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='changed'):reduce_stars(rp,tmp_path/'bad')


def test_during_publish_change_failed_record(tmp_path,monkeypatch):
    rp,sep,*_=make_separation(tmp_path)
    original=core.atomic_json
    def hook(path,record):
        original(path,record)
        if record.get('version')=='star-reduction-job-1' and record.get('state')=='succeeded':
            with Path(sep['source']).open('ab') as f:f.write(b'changed during publication')
    monkeypatch.setattr(core,'atomic_json',hook)
    with pytest.raises(ValueError,match='before publication'):reduce_stars(rp,tmp_path/'runs')
    dirs=list((tmp_path/'runs').iterdir());assert len(dirs)==1 and dirs[0].name.startswith('.partial-')
    r=json.loads((dirs[0]/'result.json').read_text());assert r['state']=='failed' and not r['artifacts']
