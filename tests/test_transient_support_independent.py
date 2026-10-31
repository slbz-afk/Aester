"""Known missing exposures, signed layers, exact independent cubic footprints."""
import math
import numpy as np
import pytest
from starstack.transient_support import attach_source_support, dependency_support, mask_warped_box
from starstack.transient_union import union_boxes
from starstack.resample import warp_image


def scalar_cubic(t):
    x=abs(t)
    if x<1: return 1-2.5*x*x+1.5*x*x*x
    if x<2: return 2-4*x+2.5*x*x-.5*x*x*x
    return 0.


def scalar_support(v,p,origin,shape):
    x,y=p; h,w=v.shape; ox,oy=origin; gh,gw=shape
    if not math.isfinite(x+y) or not (-.5<=x<w-.5 and -.5<=y<h-.5): return False
    for j in range(math.floor(y)-1,math.floor(y)+3):
        for i in range(math.floor(x)-1,math.floor(x)+3):
            if scalar_cubic(x-i)*scalar_cubic(y-j)==0: continue
            if not (0<=i+ox<gw and 0<=j+oy<gh): return False
            if not v[min(h-1,max(0,j)),min(w-1,max(0,i))]: return False
    return True


@pytest.mark.parametrize('seed',range(15))
@pytest.mark.parametrize('origin',[(0,0),(5,6)])
def test_dependencies_match_independent_scalar(seed,origin):
    rng=np.random.default_rng(seed); v=rng.random((11,13))>.07
    # Exactly represented eighths avoid differences in zero polynomial taps.
    p=rng.integers(-8,110,(120,2))/8
    p=np.vstack([p,[[np.nan,2],[np.inf,0],[-1e300,0],[0,0],[12,10]]])
    shape=(25,26) if origin!=(0,0) else v.shape
    want=[scalar_support(v,q,origin,shape) for q in p]
    assert np.array_equal(dependency_support(v,p,origin=origin,grid_shape=shape),want)


def test_other_exposure_cannot_fill_a_trails_hole():
    source=np.ones((12,12),bool); source[5,5]=False
    points=np.array([[5.25,5.25],[8.25,8.25],[3.,3.]])
    aggregate=np.ones_like(source)
    assert dependency_support(aggregate,points,origin=(3,3),grid_shape=(20,20)).tolist()==[True,True,True]
    assert dependency_support(source,points,origin=(3,3),grid_shape=(20,20)).tolist()==[False,True,True]


def test_intentional_zero_outside_footprint_is_valid_and_signed_data_survives():
    w=np.zeros((9,9)); w[3:6,3:6]=1
    d=np.full((3,9,9),-.125,dtype=np.float32)
    observed=np.zeros((9,9),bool); observed[3:6,3:6]=True
    box=attach_source_support({'x0':0,'y0':0,'data':d,'weight':w},observed)
    assert box['validity'].all()
    assert np.all(box['data'][:,3:6,3:6]==-.125)
    assert np.all(box['data'][:,w==0]==0)
    assert np.all(d==-.125)


def test_warp_rejects_whole_invalid_dependency_and_preserves_good_values():
    class Shift:
        def apply(self,p): return p+np.array([.25,.25])
    v=np.ones((16,16),bool); v[7,7]=False
    w=np.ones((16,16)); w[[0,-1]]=0; w[:,[0,-1]]=0
    source=attach_source_support({'x0':2,'y0':2,'data':np.ones((3,16,16),np.float32)*-.25,'weight':w},v)
    raw=np.stack([warp_image(d,lambda p:p+.25,fill=0) for d in source['data']])
    warped={'x0':2,'y0':2,'data':raw}
    out=mask_warped_box(source,warped,Shift(),(20,20))
    expected=np.array([[scalar_support(source['validity'],(x+.25,y+.25),(2,2),(20,20))
                        for x in range(16)] for y in range(16)])
    assert np.array_equal(out['validity'],expected)
    assert np.array_equal(out['data'][:,expected],raw[:,expected])
    assert np.all(out['data'][:,~expected]==0)
    assert out['native_support']['nonzero_pixels_removed']>0
    assert np.any(raw[:,~expected]!=0)  # reproduces the original unfiltered interpolation.


def test_identity_copy_has_no_extra_edge_erosion():
    d=np.ones((3,3,3),np.float32); v=np.ones((3,3),bool); v[1,1]=False
    box=attach_source_support({'x0':0,'y0':0,'data':d,'weight':np.ones((3,3))},v)
    out=mask_warped_box(box,box,None,(3,3))
    assert out['validity'].sum()==8
    assert np.array_equal(out['data'],box['data'])


def test_same_exposure_union_preserves_hole_independent_of_order():
    a=attach_source_support({'x0':0,'y0':0,'data':np.ones((3,8,8),np.float32),
                             'weight':np.ones((8,8)),'grid_shape':(12,12)},np.ones((8,8),bool))
    v=np.ones((8,8),bool); v[2,2]=False
    b=attach_source_support({'x0':2,'y0':2,'data':np.ones((3,8,8),np.float32)*.5,
                             'weight':np.ones((8,8))*.5,'grid_shape':(12,12)},v)
    for boxes in ([a,b],[b,a]):
        out=union_boxes(boxes)[0]
        assert not out['validity'][4,4] and np.all(out['data'][:,4,4]==0)
        assert out['grid_shape']==(12,12)


def test_cancel_propagates_and_absent_support_never_certifies():
    d=np.ones((3,8,8),np.float32); box={'x0':0,'y0':0,'data':d,'weight':np.ones((8,8))}
    assert mask_warped_box(box,box,None,(8,8))['native_support']['status']=='unavailable'
    source=attach_source_support(box,np.ones((8,8),bool))
    class Stop(Exception): pass
    def stop(): raise Stop('cancel')
    with pytest.raises(Stop): mask_warped_box(source,box,None,(8,8),check=stop)
