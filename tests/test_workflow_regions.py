"""P1 workflow boundaries with synthetic pixels and the real finishing code."""
from pathlib import Path
import json

import numpy as np
import pytest
import tifffile

from starstack import workflow as wf, regions
from test_workflow import setup, stub_model


def region_project(s, *, present=True, authority="human"):
    files=sorted(map(str,s["inputs"].glob("*.NEF")))
    linear=s["output"].parent/"first.fit"
    from astropy.io import fits
    fits.writeto(linear,np.ones((3,144,160),np.float32))
    p=regions.create_project({"state":"succeeded","source_path":files[0],"source_sha256":wf.core._sha(Path(files[0])),
        "linear_path":str(linear),"linear_sha256":wf.core._sha(linear),"width":160,"height":144}, files)
    mw={"status":"present" if present else "absent","ellipse":[.52,.45,.38,.43],"feather":.5,
        "authority":authority,"confirmed":present,"reviewer":"synthetic reviewer","rationale":"synthetic positive fixture"}
    p=regions.revise(p,scene_mode="landscape",operations=[{"kind":"horizon","points":[[0,120/143],[1,120/143]]}],milkyway=mw)
    return regions.save_project(p,s["output"]/"canvas")["project_path"],files


def bind_child_mask(s,monkeypatch):
    original=s["submit"]
    def submit(*a,**kw):
        result=original(*a,**kw)
        path=Path(kw["sky_mask_path"])
        result["mask"]={"path":str(path),"sha256":wf.core._sha(path)}
        if kw.get("compose_feather_mode"):
            result["compose_feather_mode"]=kw["compose_feather_mode"]
        result["diagnostics"]["first_frame_geometry"]=wf.core.first_frame_geometry(
            Path(result["job_dir"])/"work",None,False,(144,160),result["inputs"])
        wf.atomic_json(Path(result["job_dir"])/"job.json",result)
        return result
    monkeypatch.setattr(wf.core,"submit_job",submit)
    monkeypatch.setattr(wf.core,"registration_transforms",lambda *a:({"light_00001":np.eye(3)},None))


def test_regions_admit_deterministic_idempotency_and_source_identity(setup):
    path,files=region_project(setup)
    a=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,idempotency_key="canvas")
    b=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,idempotency_key="canvas")
    assert a["signature"]==b["signature"]
    assert a["request"]["region_assets"]["source_binding"]["input_files"][0]["path"]==files[0]
    assert a["request"]["assets"]["sky_mask_path"]["path"].endswith("sky.png")
    with pytest.raises(ValueError,match="顺序"):
        wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=list(reversed(files)))
    Path(files[0]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="素材"):
        wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files)


@pytest.mark.parametrize("present,authority",[(False,"human"),(True,"agent")])
def test_nonhuman_or_absent_does_not_enable_milkyway(setup,present,authority):
    path,files=region_project(setup,present=present,authority=authority)
    with pytest.raises(ValueError,match="画布区域"):
        wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,milkyway_enabled=True)


def test_milkyway_independent_of_star_model_and_ground_unchanged(setup,monkeypatch):
    bind_child_mask(setup,monkeypatch)
    path,files=region_project(setup)
    r=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,milkyway_enabled=True)
    done=wf.run_workflow(r["workflow_dir"])
    assert done["state"]=="succeeded",done.get("error")
    assert done["finishing"]["milkyway"]["subject_gate"]["enabled"]
    assert done["separation_result"] is None
    original=tifffile.imread(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["tiff"])
    final=tifffile.imread(Path(done["delivery_dir"])/"final_srgb16.tif")
    assert np.array_equal(final[120:],original[120:])
    assert np.any(final[:110]!=original[:110])
    wf._verify(done)


def test_both_effects_use_milkyway_finished_source_and_valid_gate(setup,monkeypatch):
    bind_child_mask(setup,monkeypatch);stub_model(monkeypatch)
    path,files=region_project(setup)
    r=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,milkyway_enabled=True,
        stars_enabled=True,star_model_path=setup["model"],background_strength=.7,milkyway_offset=-.25,
        backend_options={"tile":64,"stride":32})
    done=wf.run_workflow(r["workflow_dir"])
    assert done["state"]=="succeeded",done.get("error")
    mw=done["finishing"]["milkyway"]["artifacts"]["milkyway_srgb16.tif"]["path"]
    separation=json.loads(Path(done["separation_result"]).read_text())
    assert separation["source"]==mw
    assert done["finishing"]["reduction"]["milkyway_gate"]["enabled"]
    wf._verify(done)


def test_missing_first_transform_fails_without_identity_guess(setup,monkeypatch):
    bind_child_mask(setup,monkeypatch)
    monkeypatch.setattr(wf.core,"registration_transforms",lambda *a:({},None))
    path,files=region_project(setup)
    r=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,milkyway_enabled=True)
    done=wf.run_workflow(r["workflow_dir"])
    assert done["state"]=="failed" and "首帧" in done["error"]
    assert done["checkpoints"]["stack"] and not done["artifacts"]


def test_milkyway_first_frame_shift_uses_inverse_sampling(setup,monkeypatch):
    bind_child_mask(setup,monkeypatch)
    monkeypatch.setattr(wf.core,"registration_transforms",lambda *a:({"light_00001":np.array([[1,0,7],[0,1,0],[0,0,1]])},None))
    path,files=region_project(setup)
    r=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,milkyway_enabled=True)
    done=wf.run_workflow(r["workflow_dir"])
    assert done["state"]=="succeeded",done.get("error")
    original=tifffile.imread(r["request"]["region_assets"]["milkyway_mask_path"])
    moved=tifffile.imread(done["checkpoints"]["milkyway_region"]["data"]["path"])
    assert np.array_equal(moved[:120,7:],original[:120,:-7])
    assert not np.any(moved[:,:7])


def test_color_intent_changes_signature_but_reuses_geometry(setup):
    path,files=region_project(setup)
    a=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files)
    b=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,color_adjustments={"warmth":12})
    assert a["signature"]!=b["signature"]
    assert a["request"]["region_assets"]==b["request"]["region_assets"]


def test_mutable_registration_is_not_read_after_stack_snapshot(setup,monkeypatch):
    bind_child_mask(setup,monkeypatch)
    path,files=region_project(setup)
    r=wf.submit_workflow(setup["inputs"],setup["output"],regions_path=path,input_files=files,milkyway_enabled=True)
    directory=Path(r["workflow_dir"])
    stack=wf._stack(directory,r["request"])
    # Tampered live work state must have no influence on the frozen geometry.
    work=Path(stack["job"]["job_dir"])/"work";work.mkdir()
    (work/"light_.seq").write_text("tampered sequence")
    (work/"distortion.json").write_text('{"parameters": {"untrusted": true}}')
    monkeypatch.setattr(wf.core,"registration_transforms",lambda *a:pytest.fail("read mutable geometry"))
    result=wf._mapped_milkyway(directory,r["request"],stack)
    assert result["first_frame_to_reference"]==np.eye(3).tolist()
