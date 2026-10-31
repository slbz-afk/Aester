"""Independent contract checks for the experimental Siril range codec."""
from pathlib import Path
import math
import numpy as np
import pytest

from starstack.experimental_siril_range import encode_for_siril as ENC
from starstack.experimental_siril_range import decode_from_siril as DEC

def test_scalar_oracle_signed_range_support_and_input_unchanged():
    rng=np.random.default_rng(923)
    x=rng.uniform(-1008,25000,(41,37,3)).astype(np.float32)[::-1,::2]
    mask=(rng.random(x.shape[:2])>.2);x.setflags(write=False);mask.setflags(write=False)
    before=x.tobytes();mb=mask.tobytes()
    y=ENC(x,mask,offset=2048,scale=65536)
    expected=np.zeros(x.shape,np.float32)
    for i,j in np.ndindex(mask.shape):
        if mask[i,j]:
            for c in range(3):expected[i,j,c]=np.float32((float(x[i,j,c])+2048.)/65536.)
    np.testing.assert_array_equal(y.view(np.uint32),expected.view(np.uint32))
    z=DEC(y,mask,offset=2048,scale=65536)
    expected_dec=np.zeros(x.shape,np.float32)
    for i,j in np.ndindex(mask.shape):
        if mask[i,j]:
            for c in range(3):expected_dec[i,j,c]=np.float32(float(y[i,j,c])*65536.-2048.)
    np.testing.assert_array_equal(z.view(np.uint32),expected_dec.view(np.uint32))
    assert y.flags.c_contiguous and z.flags.c_contiguous
    assert not np.shares_memory(x,y) and not np.shares_memory(y,z)
    assert x.tobytes()==before and mask.tobytes()==mb
    assert np.max(np.abs(z[mask].astype(float)-x[mask]))<=.001953125

def test_true_zero_is_distinct_from_missing_zero_and_negative_survives():
    x=np.array([[[0.,-.5,2.],[3,4,5]]],np.float32)
    valid=np.array([[True,False]])
    y=ENC(x,valid,offset=2,scale=8)
    assert y[0,0,0]==.25 and np.all(y[0,1]==0)
    back=DEC(y,valid,offset=2,scale=8)
    np.testing.assert_array_equal(back[0,0],x[0,0]);assert np.all(back[0,1].view(np.uint32)==0)

@pytest.mark.parametrize('fn',[ENC,DEC])
@pytest.mark.parametrize('arg,value', [('offset',True),('offset',np.bool_(False)),('offset','2'),('offset',np.array(1.)),('offset',complex(1)),('offset',10**500),('offset',float('nan')),('scale',0),('scale',-1),('scale',False),('scale',float('inf'))])
def test_invalid_scalars(fn,arg,value):
    kw={'offset':0.,'scale':1.};kw[arg]=value
    with pytest.raises(ValueError):fn(np.full((2,3,3),.5,np.float32),np.ones((2,3),bool),**kw)

@pytest.mark.parametrize('fn',[ENC,DEC])
@pytest.mark.parametrize('shape',[(0,2,3),(2,0,3),(2,3),(2,3,2),(2,3,3,1)])
def test_shapes(fn,shape):
    with pytest.raises(ValueError):fn(np.zeros(shape,np.float32),np.ones((2,3),bool),offset=0,scale=1)

@pytest.mark.parametrize('fn',[ENC,DEC])
@pytest.mark.parametrize('kind',['double','int','list','masked','bad_mask_shape','mask_int','mask_list','masked_mask'])
def test_type_and_mask_contract(fn,kind):
    a=np.full((2,3,3),.5,np.float32);m=np.ones((2,3),bool)
    if kind=='double':a=a.astype(float)
    elif kind=='int':a=a.astype(int)
    elif kind=='list':a=a.tolist()
    elif kind=='masked':a=np.ma.array(a)
    elif kind=='bad_mask_shape':m=np.ones((2,3,1),bool)
    elif kind=='mask_int':m=m.astype(np.int8)
    elif kind=='mask_list':m=m.tolist()
    elif kind=='masked_mask':m=np.ma.array(m)
    with pytest.raises(ValueError):fn(a,m,offset=0,scale=1)

@pytest.mark.parametrize('fn',[ENC,DEC])
@pytest.mark.parametrize('v',[np.nan,np.inf,-np.inf])
def test_nonfinite_outside_support_still_rejected(fn,v):
    a=np.zeros((2,3,3),np.float32);a[0,0,0]=v
    with pytest.raises(ValueError):fn(a,np.zeros((2,3),bool),offset=0,scale=1)

def test_empty_support_ignored_finite_magnitudes():
    a=np.full((2,3,3),np.finfo(np.float32).max,np.float32);m=np.zeros((2,3),bool)
    y=ENC(a,m,offset=1e308,scale=1e-300)
    assert np.all(y.view(np.uint32)==0)
    z=DEC(y,m,offset=-1e308,scale=1e308);assert np.all(z.view(np.uint32)==0)

@pytest.mark.parametrize('v',[0,1,-.01,1.01])
@pytest.mark.parametrize('fn',[ENC,DEC])
def test_endpoints_never_silently_clipped(fn,v):
    with pytest.raises(ValueError):fn(np.full((1,1,3),v,np.float32),np.ones((1,1),bool),offset=0,scale=1)

def test_rounding_to_endpoint_rejected():
    a=np.full((1,1,3),np.nextafter(np.float32(0),np.float32(1)),np.float32)
    with pytest.raises(ValueError):ENC(a,np.ones((1,1),bool),offset=0,scale=4)
    with pytest.raises(ValueError):ENC(np.full((1,1,3),.5,np.float32),np.ones((1,1),bool),offset=.5-1e-10,scale=1)

def test_decode_inconsistent_invalid_nonzero_rejected():
    with pytest.raises(ValueError):DEC(np.ones((1,1,3),np.float32),np.zeros((1,1),bool),offset=0,scale=1)
    a=np.full((1,1,3),-0.,np.float32)
    out=DEC(a,np.zeros((1,1),bool),offset=0,scale=1);assert np.all(out.view(np.uint32)==0)

@pytest.mark.parametrize('fn,offset,scale',[(ENC,1e308,1e-300),(DEC,0,1e100),(DEC,-1.7e308,1.7e308)])
def test_finite_input_overflow_is_valueerror(fn,offset,scale):
    with pytest.raises(ValueError):fn(np.full((1,1,3),.9,np.float32),np.ones((1,1),bool),offset=offset,scale=scale)

def test_readonly_memmap(tmp_path):
    p=tmp_path/'data.bin';a=np.memmap(p,mode='w+',dtype=np.float32,shape=(2,3,3));a[:]=.5;a.flush();del a
    r=np.memmap(p,mode='r',dtype=np.float32,shape=(2,3,3));before=p.read_bytes()
    y=ENC(r,np.ones((2,3),bool),offset=np.int64(2),scale=np.float64(8))
    np.testing.assert_array_equal(DEC(y,np.ones((2,3),bool),offset=2,scale=8),r)
    assert p.read_bytes()==before
