"""Actual boundary counterexamples, independent geometry, and default routing."""
import inspect
import numpy as np
import pytest
from starstack import milkyway as mw
from starstack import photographic_job, mcp_server


@pytest.mark.parametrize('kind',['hole','diagonal','island','all','none'])
def test_distance_feather_against_brute_geometry(kind):
    y,x=np.indices((23,29))
    unsafe={'hole':(x==12)&(y==9),'diagonal':x+y>29,
            'island':~((x>8)&(x<17)&(y>8)&(y<17)),
            'all':np.ones_like(x,bool),'none':np.zeros_like(x,bool)}[kind]
    old=unsafe.copy();width=7
    points=np.argwhere(unsafe)
    expected=np.ones(unsafe.shape,np.float32)
    for yy in range(len(unsafe)):
        for xx in range(unsafe.shape[1]):
            if len(points):
                d=min(abs(int(py)-yy)+abs(int(px)-xx) for py,px in points)
                t=min(1.,max(0.,(d-1)/width));expected[yy,xx]=t*t*(3-2*t)
    result=mw._inward_feather(unsafe,width)
    np.testing.assert_allclose(result,expected,atol=1e-7,rtol=0)
    np.testing.assert_array_equal(unsafe,old)
    assert result.dtype==np.float32
    assert np.all(result[unsafe]==0)


def flat_boundary():
    a=np.full((512,768,3),.25,np.float32)
    sky=np.ones(a.shape[:2],np.float32);sky[400:]=0
    protect=np.zeros_like(sky,bool);mask=np.ones_like(sky)
    return a,sky,protect,mask


def process(inputs,recipe):
    prep=mw.prepare(*inputs,recipe,domain='display-sRGB')
    out=np.empty_like(inputs[0]);mw.render(*inputs,prep,out,domain='display-sRGB')
    return prep,out


def test_sky_boundary_lift_fades_broadly_without_a_new_step():
    arrays=flat_boundary()
    _,out=process(arrays,mw.Recipe(brightness=.02,contrast=0,structure=0,color=0,strength=1.,work_long_edge=768))
    line=(out-arrays[0])[:,384,0]
    # Old narrow box shoulder produced a ~.004 change in one native pixel.
    # A positive lift must now retire over a broad monotonic region.
    assert line.max()==pytest.approx(.02,abs=1e-7)
    assert np.max(np.abs(np.diff(line))) < .0006
    partial=(line>.002)&(line<.018)
    assert partial.sum()>=35
    assert np.all(np.diff(line[300:])<=1e-7)
    np.testing.assert_array_equal(out[400:],arrays[0][400:])


def test_default_reduces_lift_to_sixty_percent_with_same_support():
    arrays=flat_boundary()
    p1,a=process(arrays,mw.Recipe(strength=1.,work_long_edge=768))
    p2,b=process(arrays,mw.Recipe(work_long_edge=768))
    np.testing.assert_array_equal(p1['delta_rgb'],p2['delta_rgb'])
    np.testing.assert_array_equal(p1['safe'],p2['safe'])
    np.testing.assert_allclose(b-arrays[0],.6*(a-arrays[0]),atol=4e-8,rtol=0)
    assert mw.Recipe().strength==.6
    assert inspect.signature(photographic_job.finish_milkyway).parameters['strength'].default==.6
    assert inspect.signature(mcp_server.WorkspaceTools.finish_milkyway).parameters['strength'].default==.6


@pytest.mark.parametrize('width',[True,0,4097,2.5,np.int64(7)])
def test_bad_feather_width(width):
    with pytest.raises(ValueError):mw._inward_feather(np.zeros((4,4),bool),width)
