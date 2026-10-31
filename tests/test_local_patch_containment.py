"""Generated content can enter only an explicit, small, unprotected defect mask."""
import numpy as np
import pytest
from starstack.local_patch import compose_local_patch


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16,np.float32])
def test_local_scalar_blend_and_every_outside_pixel_exact(dtype):
    rng=np.random.default_rng(155)
    if dtype==np.float32:
        source=rng.random((120,180,3),dtype=np.float32);source[0,0]=np.float32(-0.)
        patch=rng.random((10,12,3),dtype=np.float32)
    else:
        maximum=np.iinfo(dtype).max
        source=rng.integers(0,maximum,(120,180,3),dtype=dtype)
        patch=rng.integers(0,maximum,(10,12,3),dtype=dtype)
    before=source.copy();alpha=np.zeros((10,12),np.float32)
    alpha[3:7,4:8]=np.array([0,.25,.75,1.],np.float32)
    result,receipt=compose_local_patch(source,patch,(35,41,12,10),alpha)
    allowed=np.zeros(source.shape[:2],bool);allowed[41:51,35:47]=alpha>0
    assert result[~allowed].tobytes()==before[~allowed].tobytes()
    for y,x in zip(*np.nonzero(alpha)):
        f=float(alpha[y,x]);expected=(1-f)*before[y+41,x+35].astype(np.float64)+f*patch[y,x].astype(np.float64)
        if dtype!=np.float32:expected=np.rint(expected)
        np.testing.assert_array_equal(result[y+41,x+35],expected.astype(dtype))
    assert source.tobytes()==before.tobytes()


def test_protected_intersection_is_refused_not_silently_reduced():
    source=np.zeros((100,100,3),np.uint8);patch=np.ones((10,10,3),np.uint8)
    alpha=np.zeros((10,10),np.float32);alpha[5,5]=1
    guard=np.zeros((100,100),bool);guard[25,25]=True
    with pytest.raises(ValueError):compose_local_patch(source,patch,(20,20,10,10),alpha,protected=guard)


@pytest.mark.parametrize('size,active',[(21,1),(20,101)])
def test_area_limits_are_hard_gates(size,active):
    source=np.zeros((100,100,3),np.uint8);patch=np.ones((size,size,3),np.uint8)
    alpha=np.zeros((size,size),np.float32);alpha.flat[:active]=1
    with pytest.raises(ValueError):compose_local_patch(source,patch,(20,20,size,size),alpha)
