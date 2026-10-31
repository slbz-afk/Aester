"""Global colour contract and display isolation; independent analytic camera."""
from copy import deepcopy

import numpy as np
import pytest

from starstack import whole_color as wc
from starstack.raw_color import build_recipe, XYZ_FROM_LINEAR_SRGB


@pytest.fixture
def recipe():
    mixing = np.array([[1.,0.,0.],[.2,.8,0.],[0.,.25,.75]])
    cx = np.vstack((mixing @ np.linalg.inv(XYZ_FROM_LINEAR_SRGB), np.zeros(3)))
    fields = {'num_colors':3, 'color_desc':'RGBG', 'raw_pattern':[[0,1],[3,2]],
              'camera_whitebalance':[2.,1.,1.5,0.], 'rgb_xyz_matrix':cx.tolist()}
    metadata = {'available':True,'white_balance_applied':False,'color_correction_applied':False,
                'versions':{},'fields':{k:{'value':v} for k,v in fields.items()}}
    return build_recipe(metadata,'a'*64,1100,100)


def test_camera_matches_analytic_inverse_and_signed_range(recipe):
    # RAW camera codes include a 100 DN black pedestal. A green-heavy negative
    # input and a super-white input exercise the unclipped linear derivative.
    a = np.array([[[225,350,100+250/1.5],[150,360,500]],
                  [[50,50,50],[2100,4100,4100]]],dtype=np.float32)/np.float32(65535)
    before=a.copy(); original=deepcopy(recipe)
    actual=wc.camera_from_cache(a,recipe,black_dn=100,input_state=wc.CACHE_STATE)
    dn=a.astype(float)*65535-100
    r=dn[...,0]*2/1000; g=dn[...,1]/1000; b=dn[...,2]*1.5/1000
    expected=np.stack((r,-r/4+5*g/4,r/12-5*g/12+4*b/3),axis=-1)
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-8)
    assert actual.min()<0 and actual.max()>1
    np.testing.assert_array_equal(a.view(np.uint32),before.view(np.uint32))
    assert recipe==original


def test_affine_color_commutes_with_sky_ground_blend(recipe):
    rng=np.random.default_rng(302)
    sky=rng.uniform(.001,.05,(130,7,3)).astype(np.float32)
    ground=rng.uniform(.001,.05,(130,7,3)).astype(np.float32)
    weight=rng.uniform(0,1,(130,7,1)).astype(np.float32)
    apply=lambda x:wc.camera_from_cache(x,recipe,black_dn=100,input_state=wc.CACHE_STATE)
    # Floating rounding is expected; no mask-dependent colour discontinuity.
    np.testing.assert_allclose(apply(sky*weight+ground*(1-weight)),
                               apply(sky)*weight+apply(ground)*(1-weight),rtol=2e-5,atol=3e-6)


@pytest.mark.parametrize('state', ['linear_srgb','camera_RGB',None,True])
def test_cache_state_must_be_explicit(recipe,state):
    with pytest.raises(ValueError):
        wc.camera_from_cache(np.ones((2,2,3),np.float32),recipe,black_dn=100,input_state=state)


@pytest.mark.parametrize('black',[0,99,True,'100',np.float32(100)])
def test_black_must_match_reference_and_type(recipe,black):
    with pytest.raises(ValueError):
        wc.camera_from_cache(np.ones((2,2,3),np.float32),recipe,black_dn=black,input_state=wc.CACHE_STATE)


def test_overflow_is_value_error_without_changing_source(recipe):
    a=np.full((2,2,3),np.finfo(np.float32).max,np.float32)
    before=a.copy()
    with pytest.raises(ValueError):
        wc.camera_from_cache(a,recipe,black_dn=100,input_state=wc.CACHE_STATE)
    np.testing.assert_array_equal(a,before)


def test_signed_reduction_precedes_display_clip():
    a=np.tile(np.array([-3.,1.,1.,1.],np.float32)[None,:,None],(4,1,3))
    small=wc.resize_signed(a,2)
    np.testing.assert_array_equal(small[:,0],-np.ones((2,3)))
    np.testing.assert_array_equal(small[:,1],np.ones((2,3)))


@pytest.mark.parametrize('k',[0,1,12,100])
def test_gray_anchor_and_one_srgb_encoding(k):
    r=wc.make_display_recipe(black=.1,white=1.1,anchor=.3,target_srgb=.22,asinh_strength=k)
    pixels=wc.render_display(np.full((2,2,3),.3,np.float32),r)
    np.testing.assert_array_equal(pixels,np.full((2,2,3),56,np.uint8))


def test_fixed_curve_and_input_do_not_change_during_replay():
    r=wc.make_display_recipe(black=0,white=1,anchor=.1,target_srgb=.15)
    a=np.random.default_rng(44).uniform(-.2,2,(8,7,3)).astype(np.float32)
    before=a.copy(); old=deepcopy(r)
    x=wc.render_display(a,r); y=wc.render_display(a,r)
    np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a,before)
    assert r==old


def test_reject_tampered_midtone_and_spatial_scope():
    a=np.ones((2,2,3),np.float32)
    r=wc.make_display_recipe(black=0,white=1,anchor=.1,target_srgb=.15)
    r['midtone']*=.9
    with pytest.raises(ValueError):wc.render_display(a,r)
    r['spatial_scope']='sky_only'
    with pytest.raises(ValueError):wc.render_display(a,r)


@pytest.mark.parametrize('anchor',[0,1])
def test_anchor_cannot_be_endpoint(anchor):
    with pytest.raises(ValueError):
        wc.make_display_recipe(black=0,white=1,anchor=anchor,target_srgb=.2)


def test_nonfinite_input_cannot_hide_during_resize():
    a=np.ones((5,5,3),np.float32);a[0,0,0]=np.nan
    with pytest.raises(ValueError):wc.resize_signed(a,2)
