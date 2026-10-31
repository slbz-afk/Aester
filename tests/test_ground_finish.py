import numpy as np
import pytest
from starstack.ground_finish import finish_ground


def test_constant_lift_has_independent_analytic_value():
    a=np.full((90,100,3),6554,np.uint16);output=a.copy()
    sky=np.zeros(a.shape[:2],np.float32);protect=np.zeros(sky.shape,bool)
    report=finish_ground(a,sky,protect,output,strength=.6)
    l=6554/65535;gain=1+.18*.6*((.4-l)/.35)**2*l/(l+.025)
    assert np.all(output[35:55,35:65]==np.rint(6554*gain))
    assert np.all(a==6554) and np.all(output[:8]==a[:8])
    assert report['new_white_channels']==report['new_black_channels']==0


def test_zero_keeps_existing_output_and_all_non_ground_are_exact():
    a=np.full((100,110,3),8000,np.uint16);prior=np.full_like(a,9000)
    sky=np.zeros(a.shape[:2],np.float32);sky[:40]=1;sky[40:50]=.4
    guard=np.zeros(sky.shape,bool);guard[70:80,50:60]=True
    output=prior.copy();finish_ground(a,sky,guard,output,strength=0)
    np.testing.assert_array_equal(output,prior)
    finish_ground(a,sky,guard,output,strength=1)
    hold=(sky!=0)|guard
    np.testing.assert_array_equal(output[hold],prior[hold])
    assert np.any(output[~hold]!=prior[~hold])
    np.testing.assert_array_equal(output[64:70,51:59],prior[64:70,51:59])


def test_complete_mask_validation_precedes_first_write():
    a=np.full((100,110,3),8000,np.uint16);sky=np.zeros(a.shape[:2],np.float32)
    guard=np.zeros(sky.shape,bool);out=a.copy();sky[-1,-1]=np.nan
    with pytest.raises(ValueError):finish_ground(a,sky,guard,out,rows=1)
    np.testing.assert_array_equal(out,a)
    sky[-1,-1]=0
    with pytest.raises(ValueError):finish_ground(a,sky,guard,a)


@pytest.mark.parametrize('strength',[0.,.7])
@pytest.mark.parametrize('strides',[(0,6,2),(110*6,2,2),(6,6,2)])
def test_self_overlapping_output_is_rejected_before_mutation(strength,strides):
    a=np.full((100,110,3),8000,np.uint16);sky=np.zeros(a.shape[:2],np.float32);sky[:30]=1
    guard=np.zeros(sky.shape,bool)
    storage=np.full(100*110*3,9000,np.uint16);before=storage.copy()
    out=np.lib.stride_tricks.as_strided(storage,shape=a.shape,strides=strides,writeable=True)
    with pytest.raises(ValueError):finish_ground(a,sky,guard,out,strength=strength)
    np.testing.assert_array_equal(storage,before)


def test_nonoverlapping_transpose_has_same_values():
    a=np.full((100,110,3),8000,np.uint16);sky=np.zeros(a.shape[:2],np.float32)
    guard=np.zeros(sky.shape,bool);ordinary=a.copy()
    strided=np.empty((110,100,3),np.uint16).transpose(1,0,2);strided[:]=a
    finish_ground(a,sky,guard,ordinary);finish_ground(a,sky,guard,strided)
    np.testing.assert_array_equal(ordinary,strided)
