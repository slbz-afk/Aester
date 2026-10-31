"""Codex acceptance of the DeepSeek delivery, with a scalar Python oracle."""
import json
import math

import numpy as np
import pytest

from starstack.milkyway_metrics import compare_uint16


def oracle(a, b, mask):
    pixels=[]
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if mask[y,x]:
                aa=list(map(int,a[y,x]));bb=list(map(int,b[y,x]))
                pixels.append((aa,bb,[v-u for u,v in zip(aa,bb)]))
    d=[v for _,_,delta in pixels for v in delta]
    lum=[sum(v*w for v,w in zip(delta,(2126,7152,722))) for _,_,delta in pixels]
    return dict(pixel_count=len(pixels),changed_pixels=sum(any(delta) for _,_,delta in pixels),
        changed_channels=sum(v!=0 for v in d),decreased_channels=sum(v<0 for v in d),increased_channels=sum(v>0 for v in d),
        new_zero_channels=sum(u>0 and v==0 for aa,bb,_ in pixels for u,v in zip(aa,bb)),
        new_white_channels=sum(u<65535 and v==65535 for aa,bb,_ in pixels for u,v in zip(aa,bb)),
        min_delta_code=min(d) if d else None,max_delta_code=max(d) if d else None,
        mean_delta_code=sum(d)/len(d) if d else None,rms_delta_code=math.sqrt(sum(v*v for v in d)/len(d)) if d else None,
        luma_decreased_pixels=sum(v<0 for v in lum),luma_increased_pixels=sum(v>0 for v in lum),
        mean_luma_delta_code=sum(lum)/(10000*len(lum)) if lum else None)


@pytest.mark.parametrize('rows',[1,3,128,512])
@pytest.mark.parametrize('select',['mixed','none','all'])
def test_scalar_oracle_readonly_strided(rows,select):
    a=np.array([[[0,65535,5],[65535,0,12],[100,100,100]],[[12,99,0],[65000,2,37],[700,1200,30]]],np.uint16)[:,::-1]
    b=np.array([[[1,65535,0],[0,65535,13],[101,99,100]],[[12,99,0],[65535,1,38],[600,1300,20]]],np.uint16)[:,::-1]
    allowed=np.array([[True,False,True],[False,True,False]])
    if select!='mixed': allowed[:]=select=='all'
    before=(a.copy(),b.copy(),allowed.copy())
    for arr in (a,b,allowed): arr.flags.writeable=False
    result=compare_uint16(a,b,allowed,domain='display-sRGB',rows=rows)
    for name,mask in [('all',np.ones(allowed.shape,bool)),('allowed',allowed),('outside',~allowed)]:
        assert result['groups'][name]==oracle(a,b,mask)
    assert result['all_changes_within_allowed']==(result['groups']['outside']['changed_channels']==0)
    assert json.loads(json.dumps(result,allow_nan=False))==result
    for arr,old in zip((a,b,allowed),before): assert np.array_equal(arr,old)


def test_exact_luma_cancellation():
    a=np.full((1,1,3),40000,np.uint16);b=a.copy()
    b[0,0]=[47152,37874,40000]  # 2126*7152 + 7152*(-2126) = 0
    g=compare_uint16(a,b,np.ones((1,1),bool),domain='display-sRGB')['groups']['all']
    assert g['changed_channels']==2 and g['mean_luma_delta_code']==0
    assert g['luma_increased_pixels']==g['luma_decreased_pixels']==0


def test_max_difference_wide_column_tiling_and_reversal():
    a=np.zeros((1,1_000_007,3),np.uint16);b=np.full_like(a,65535)
    mask=np.ones(a.shape[:2],bool)
    for aa,bb,sign in [(a,b,1),(b,a,-1)]:
        r=compare_uint16(aa,bb,mask,domain='display-sRGB',rows=512)
        g=r['groups']['all']
        assert g['rms_delta_code']==65535 and g['mean_delta_code']==sign*65535
        assert g['mean_luma_delta_code']==sign*65535
        assert g['changed_channels']==a.size and g['changed_pixels']==mask.size
        assert g['new_white_channels']==(a.size if sign==1 else 0)
        assert g['new_zero_channels']==(a.size if sign==-1 else 0)
        assert r['groups']['outside']['rms_delta_code'] is None


def test_memmap_and_tile_invariance(tmp_path):
    p=tmp_path/'array.bin';a=np.memmap(p,mode='w+',shape=(515,43,3),dtype=np.uint16)
    a[:]=np.random.default_rng(871).integers(0,65536,size=a.shape,dtype=np.uint16);a.flush()
    ro=np.memmap(p,mode='r',shape=a.shape,dtype=np.uint16)
    b=65535-ro;mask=np.indices(a.shape[:2]).sum(axis=0)%3==0
    assert compare_uint16(ro,b,mask,domain='display-sRGB',rows=1)==compare_uint16(ro,b,mask,domain='display-sRGB',rows=512)
    assert np.array_equal(a,ro)


@pytest.mark.parametrize('bad',[True,False,0,513,1.,np.int64(128),'128',None])
def test_bad_rows(bad):
    a=np.zeros((2,2,3),np.uint16)
    with pytest.raises(ValueError):compare_uint16(a,a,np.ones((2,2),bool),domain='display-sRGB',rows=bad)


@pytest.mark.parametrize('bad',['sRGB','linear-sRGB',np.str_('display-sRGB'),None,1])
def test_bad_domain(bad):
    a=np.zeros((2,2,3),np.uint16)
    with pytest.raises(ValueError):compare_uint16(a,a,np.ones((2,2),bool),domain=bad)


@pytest.mark.parametrize('which,bad',[
    ('source',np.zeros((0,2,3),np.uint16)),('source',np.zeros((2,2,3),np.float32)),
    ('source',np.zeros((2,2,4),np.uint16)),('source',[[[1,2,3]]]),
    ('candidate',np.zeros((3,2,3),np.uint16)),('candidate',np.zeros((2,2,3),np.int16)),
    ('allowed',np.ones((2,2),np.uint8)),('allowed',np.ones((2,2,1),bool)),('allowed',[[True]*2]*2)])
def test_bad_arrays(which,bad):
    a=np.zeros((2,2,3),np.uint16)
    kwargs={'source':a,'candidate':a,'allowed':np.ones((2,2),bool),'domain':'display-sRGB'}
    kwargs[which]=bad
    with pytest.raises(ValueError):compare_uint16(**kwargs)
