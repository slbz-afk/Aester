"""Explicit canvas-feather admission and actual base/trail composition."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from starstack import core, imaging, sky_validity as sv, region_blend
from test_core_selected_inputs import admitted


def test_feather_requires_mask_and_signature_freezes_recipe(admitted,tmp_path):
    _,files,out=admitted
    with pytest.raises(ValueError,match="显式天空蒙版"):
        core.submit_job(None,str(out),input_files=files[:3],compose_feather_mode=region_blend.RECIPE)
    mask=tmp_path/"sky.png";Image.new("L",(24,16),255).save(mask)
    old=core.submit_job(None,str(out),input_files=files[:3],sky_mask_path=str(mask))
    assert "compose_feather_mode" not in old and "compose_feather_mode" not in old["admission_options"]
    new=core.submit_job(None,str(out),input_files=files[:3],sky_mask_path=str(mask),compose_feather_mode=region_blend.RECIPE)
    assert old["signature"]!=new["signature"] and new["admission_options"]["compose_feather_mode"]==region_blend.RECIPE
    core._verify_admission_options(new)
    new["compose_feather_mode"]=None
    with pytest.raises(ValueError,match="选项已改变"):
        core._verify_admission_options(new)


@pytest.mark.parametrize("mode",[True,"", "any-auto",1,{}])
def test_unknown_recipe_rejected_before_engine(admitted,mode):
    with pytest.raises(ValueError,match="未知"):
        core.submit_job(None,str(admitted[2]),input_files=admitted[1][:3],compose_feather_mode=mode)


def test_actual_weight_feather_zero_preservation_and_recompose_equivalence(tmp_path,monkeypatch):
    work=tmp_path/"work";work.mkdir()
    shape=(48,60)
    weight=np.ones(shape,np.float32);weight[35:]=0;weight[9:15,16:24]=0;weight[4,5]=.3
    Image.fromarray((weight>0).astype(np.uint8)*255).save(tmp_path/"sky_mask.png")
    Image.new("L",(10,10)).save(work/sv.COVERAGE_PNG)
    sky=np.full((*shape,3),.15,np.float32);ground=np.full_like(sky,.05)
    imaging.write_linear(work/"foreground_master.fit",ground,"synthetic ground")
    src=SimpleNamespace(mask=weight,for_frame=lambda _:weight)
    monkeypatch.setattr(core,"_validity_source",lambda *a:src)
    monkeypatch.setattr(core,"registration_transforms",lambda *a:({},None))
    monkeypatch.setattr(core,"sequence_reference",lambda *a:{})
    monkeypatch.setattr(sv,"effective_sky_weight",lambda *a,**k:(weight.copy(),{"fixture":True}))
    kwargs=dict(directory=tmp_path,job={"mask":{},"inputs":[],"compose_feather_mode":region_blend.RECIPE},work=work,
                calibration={},used=False,validity_record={"dilation_px":3},sky=sky,converted=[],
                header={"NAXIS2":shape[0],"NAXIS1":shape[1]},check=lambda:None)
    kwargs["job"]["mask"]={"path":str(tmp_path/"sky_mask.png")}
    base=core._compose_sky(**kwargs)
    expected,receipt=region_blend.inward_weight(weight)
    np.testing.assert_array_equal(base["weight"],expected)
    assert np.all(expected<=weight) and np.array_equal(expected[weight==0],weight[weight==0])
    assert base["record"]["compose"]["feather"]["recipe"]==region_blend.RECIPE
    fresh_replay=core._compose_sky(**kwargs)
    np.testing.assert_array_equal(fresh_replay["output"],base["output"])
    cached_replay=core._compose_sky(**kwargs,weight=base["weight"])
    np.testing.assert_array_equal(cached_replay["output"],base["output"])
    # _export_transients uses this same call with modified sky. The common
    # weight scales only T; it cannot re-feather the base a second time.
    sky2=sky.copy();sky2[18:24,27:32]+=.02
    added=core._compose_sky(**{**kwargs,"sky":sky2},weight=base["weight"])
    assert np.array_equal(added["output"][:15],base["output"][:15])
    np.testing.assert_allclose(added["output"]-base["output"],(sky2-sky)*expected[::-1,:,None],atol=1e-8,rtol=0)
