"""Workflow publication is separate from the core's atomic candidate revision.

Core detection/apply is stubbed here; real master, colour and star processing
exercise the shared product boundary and retry after interrupted publication.
"""
from pathlib import Path
import json
import numpy as np
import pytest
import tifffile
from PIL import Image
from starstack import workflow as wf, workflow_edit as edit, trail_review_session as review
from starstack import color_export as ce, master_export as me, imaging
from starstack.compose_evidence import write_weight
from test_workflow import setup, stub_model
import color_fixtures as cf


def calibrated_workflow(s,monkeypatch):
    original=s["submit"]
    ref=cf.write_reference_dng(s["output"].parent/"reference.dng")
    basis=me.colour_basis_from_dng(ref)
    camera,*_=cf.synthetic_camera_master(144,160,ground_from=120,nodata_columns=0)
    def submit(*a,**kw):
        job=original(*a,**kw)
        directory=Path(job["job_dir"]);exp=directory/"calibrated";exp.mkdir()
        import shutil
        shutil.copyfile(s["mask"],directory/"sky_mask.png")
        fit=exp/"master_linear.fit";imaging.write_linear(fit,np.flipud(camera),"synthetic input")
        _,colour=ce.export_job_products(np.flipud(camera),exp,lambda:None,basis=basis,source_fits=fit,
                                        sky_mask_path=s["mask"],color_adjustments=kw.get("color_adjustments"))
        desc=write_weight(exp,np.asarray(Image.open(s["mask"]),np.float32)/255,fit)
        candidate=exp/"candidates.json";candidate.write_text('{"candidates": []}')
        job.update(artifacts=[{"path":str(fit),"kind":"fits"}],transient_revision={"id":"original","record":str(candidate),"sha256":wf.core._sha(candidate)})
        job["diagnostics"].update(color_export=colour,sky_validity={"compose_weight":desc},transients={"status":"needs_review"})
        wf.atomic_json(directory/"job.json",job)
        return job
    monkeypatch.setattr(wf.core,"submit_job",submit)
    stub_model(monkeypatch)
    r=wf.submit_workflow(s["inputs"],s["output"],sky_mask_path=s["mask"],trails_enabled=True,
        color_adjustments={"warmth":12,"tint":5,"exposure_ev":.3},stars_enabled=True,star_model_path=s["model"],
        background_strength=.7,backend_options={"tile":64,"stride":32})
    done=wf.run_workflow(r["workflow_dir"])
    assert done["state"]=="succeeded",done.get("error")
    jobdir=Path(done["stack_job_dir"]);session=jobdir/"review"/"draft.json";session.parent.mkdir();session.write_text('{}')
    state={"calls":0,"applies":0,"session":{"state":"editing","reviewer":"human"}}
    def apply(path,**kwargs):
        assert path==str(session)
        assert kwargs["expected_draft_revision"]=="token"
        state["calls"]+=1
        job=wf.core.load_job(jobdir)
        if state["session"]["state"]=="applied":
            return {"job":job,"session":state["session"],"changed":False}
        state["applies"]+=1
        exp=jobdir/"reviewed-colour";exp.mkdir()
        changed=camera.copy();changed[30:36,42:75]+=.07
        fit=exp/"transients.fit";imaging.write_linear(fit,np.flipud(changed),"synthetic trail")
        _,colour=ce.export_job_products(np.flipud(changed),exp,lambda:None,basis=basis,source_fits=fit,
                                        sky_mask_path=s["mask"],color_adjustments=job["color_adjustments"])
        revision=exp/"candidates.json";revision.write_text('{"candidates": [], "reviewed": true}')
        job["transient_revision"]={"id":"reviewed","record":str(revision),"sha256":wf.core._sha(revision)}
        job["diagnostics"]["transients"]={"composite":str(fit),"color_export":colour,"kept":[{"id":"light_00001-1"}],
                                          "summary":{"recall_review":{"candidate_ids":[]}}}
        wf.atomic_json(jobdir/"job.json",job)
        state["session"]={"state":"applied","reviewer":"human","applied_revision":job["transient_revision"]}
        return {"job":job,"session":state["session"],"changed":True}
    monkeypatch.setattr(review,"apply_review",apply)
    return done,session,state


def test_review_delivery_preserves_stars_base_master_and_repeat_is_idempotent(setup,monkeypatch):
    done,session,state=calibrated_workflow(setup,monkeypatch)
    original={a["name"]:a for a in done["artifacts"]}
    result=edit.apply_review(done["workflow_dir"],session,expected_draft_revision="token")
    current=result["workflow"]
    assert current["delivery_dir"]!=done["delivery_dir"]
    # 2026-09-27: a trails version is named by its kept count, beside the base.
    assert Path(current["delivery_dir"]).name.startswith("delivery-trails-1kept-")
    assert result["published"] and result["delivery_dir"]==current["delivery_dir"] and result["kept_ids"]==["light_00001-1"]
    assert current["base_delivery_dir"]==done["delivery_dir"] and not result["identical_to_base"]
    assert current["finishing"]==done["finishing"]
    assert not current["needs_review"]
    assert all(wf.core._sha(Path(a["path"]))==a["sha256"] for a in original.values())
    assert wf.core._sha(Path(current["delivery_dir"])/"master_scene_linear.tif")==original["master_scene_linear.tif"]["sha256"]
    assert (Path(current["delivery_dir"])/"camera_saturation_mask.png").exists()
    final=tifffile.imread(Path(current["delivery_dir"])/"final_srgb16.tif")
    prior=tifffile.imread(original["final_srgb16.tif"]["path"])
    assert np.array_equal(final[48:],prior[48:])
    assert np.any(final[27:40,38:79]!=prior[27:40,38:79])
    replay=json.loads((Path(current["delivery_dir"])/"fixed-colour"/"photo_render.json").read_text())
    assert Path(replay["tiff"]).exists()
    wf._verify(current)
    twice=edit.apply_review(done["workflow_dir"],session,expected_draft_revision="token")
    assert twice["reused"] and twice["workflow"]["delivery_dir"]==current["delivery_dir"]
    assert state["applies"]==1


def test_render_failure_after_apply_retries_same_revision_once(setup,monkeypatch):
    done,session,state=calibrated_workflow(setup,monkeypatch)
    render=edit._render_fixed
    monkeypatch.setattr(edit,"_render_fixed",lambda *a:(_ for _ in ()).throw(RuntimeError("synthetic interrupted render")))
    with pytest.raises(RuntimeError,match="interrupted"):
        edit.apply_review(done["workflow_dir"],session,expected_draft_revision="token")
    unchanged=wf.workflow_status(done["workflow_dir"])
    assert unchanged["delivery_dir"]==done["delivery_dir"]
    assert state["session"]["state"]=="applied"
    monkeypatch.setattr(edit,"_render_fixed",render)
    result=edit.apply_review(done["workflow_dir"],session,expected_draft_revision="token")
    assert result["workflow"]["delivery_dir"]!=done["delivery_dir"] and state["applies"]==1


def test_empty_draft_does_not_claim_reviewed_delivery(setup,monkeypatch):
    done,session,_=calibrated_workflow(setup,monkeypatch)
    monkeypatch.setattr(review,"apply_review",lambda *a,**k:{"job":wf.core.load_job(done["stack_job_dir"]),"session":{"state":"editing"},"changed":False})
    result=edit.apply_review(done["workflow_dir"],session,expected_draft_revision="token")
    assert result["delivery"] is None and result["workflow"]["trail_review"]["state"]=="needs_review"


def test_new_child_revision_at_publish_lock_refuses_stale_delivery(setup,monkeypatch):
    done,session,_=calibrated_workflow(setup,monkeypatch)
    load=wf.core.load_job
    seen={"count":0}
    def racing_load(path):
        job=load(path);seen["count"]+=1
        if seen["count"]==3:
            job["transient_revision"]={**job["transient_revision"],"id":"another-concurrent-review"}
        return job
    monkeypatch.setattr(wf.core,"load_job",racing_load)
    with pytest.raises(ValueError,match="发布锁内"):
        edit.apply_review(done["workflow_dir"],session,expected_draft_revision="token")
    assert wf.workflow_status(done["workflow_dir"])["delivery_dir"]==done["delivery_dir"]
