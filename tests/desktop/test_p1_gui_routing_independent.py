"""Independent Qt-to-real-workflow admission tests; no RAW decoder is stubbed in this boundary."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from astropy.io import fits
from starstack import regions, core, workflow
from test_gui_workflow import app, window, drain


def scene(window,monkeypatch,tmp_path,*,mode="landscape",present=True):
    inp=tmp_path/"raw";inp.mkdir()
    files=[]
    for i in range(3):
        path=inp/f"{i}.NEF";path.write_bytes(bytes([i+1])*20);files.append(str(path))
    out=tmp_path/"output";out.mkdir(exist_ok=True)
    linear=out/"first.fit";fits.writeto(linear,np.ones((3,24,32),np.float32))
    template=out/"template.png";Image.new("L",(32,24),255).save(template)
    preview=out/"preview.png";Image.new("RGB",(32,24),(20,25,30)).save(preview)
    reference={"state":"succeeded","source_path":files[0],"source_sha256":core._sha(Path(files[0])),
               "linear_path":str(linear),"linear_sha256":core._sha(linear),"width":32,"height":24,
               "mask_template_path":str(template),"preview_path":str(preview),"preview_sha256":core._sha(preview)}
    project=regions.create_project(reference,files)
    project=regions.revise(project,scene_mode=mode,operations=[{"kind":"horizon","points":[[0,.8],[1,.8]]}] if mode=="landscape" else [],
        milkyway={"status":"present" if present else "unknown","authority":"human","confirmed":present,
                  "reviewer":"synthetic test human","rationale":"synthetic boundary test",
                  "ellipse":[.5,.35,.3,.4],"feather":.5})
    assets=regions.build_masks(project,out)
    engine=tmp_path/"siril.exe";engine.write_bytes(b"testengine")
    model=tmp_path/"model.onnx";model.write_bytes(b"testmodel")
    monkeypatch.setattr(workflow.core,"doctor",lambda *a:{"available":True,"executable":str(engine)})
    monkeypatch.setattr(workflow,"_spawn",lambda *a:None)
    window.input_files=files
    window.input_dir.setText(str(inp));window.output_dir.setText(str(out))
    window.siril_path.setText(str(engine));window.star_model_path.setText(str(model))
    window._reference_data=reference;window._reference_scene=tuple(files)
    window._regions_project=project;window._regions_path=assets["project_path"]
    window.sky_mask.setText(assets["sky_mask_path"])
    window.scene_mode.setCurrentIndex(window.scene_mode.findData(mode))
    return project,assets


@pytest.mark.parametrize("mode,stars,offset",[("landscape",False,0),("landscape",True,-35),("sky",True,40)])
def test_canvas_gui_dispatches_real_workflow_without_dual_mask(window,monkeypatch,tmp_path,mode,stars,offset):
    scene(window,monkeypatch,tmp_path,mode=mode)
    window.stars_enabled.setChecked(stars)
    window.workflow_milkyway_offset.setValue(offset)
    captured=[];monkeypatch.setattr(window,"_run_async",lambda *a:captured.append(a))
    window.start_processing()
    try:
        assert len(captured)==1
        admitted=captured[0][1]()  # Real workflow.submit_workflow: do not replace this boundary.
        assert admitted["request"]["milkyway_offset"]==offset/100
        assert admitted["request"]["stars_enabled"] is stars
        project=regions.load_project(admitted["request"]["regions_path"])
        assert project["scene_mode"]==mode
        assert admitted["request"]["compose_feather_mode"]=="canvas-inward-feather-1"
    finally:window._busy=False


def test_unknown_mw_gate_suppresses_stale_offset_without_enabling_edit(window,monkeypatch,tmp_path):
    scene(window,monkeypatch,tmp_path,present=False)
    window.stars_enabled.setChecked(True)
    window.workflow_milkyway_offset.setValue(-50)
    window._update_controls()
    assert not window.milkyway_enabled.isEnabled() and not window.workflow_milkyway_offset.isEnabled()
    captured=[];monkeypatch.setattr(window,"_run_async",lambda *a:captured.append(a))
    window.start_processing()
    try:
        admitted=captured[0][1]()
        assert admitted["request"]["milkyway_offset"]==0
        assert admitted["request"]["milkyway_enabled"] is False
    finally:window._busy=False


def test_stars_off_ignores_saved_offset(window,monkeypatch,tmp_path):
    scene(window,monkeypatch,tmp_path)
    window.stars_enabled.setChecked(False);window.workflow_milkyway_offset.setValue(-45)
    captured=[];monkeypatch.setattr(window,"_run_async",lambda *a:captured.append(a))
    window.start_processing()
    try:
        admitted=captured[0][1]()
        assert admitted["request"]["milkyway_offset"]==0
    finally:window._busy=False


def test_external_mask_editing_project_reopen_keeps_usable_mask_route(window,monkeypatch,tmp_path):
    from starstack import editing_project
    scene(window,monkeypatch,tmp_path,present=False)
    window._regions_project=None;window._regions_path=None  # Advanced: imported existing PNG only.
    captured=[];monkeypatch.setattr(window,"_run_async",lambda *a:captured.append(a))
    project_path=tmp_path/"external-mask.aester.json"
    window.save_editing_project(str(project_path))
    captured[-1][1]()
    saved=editing_project.load_project(project_path)
    window._restore_editing_project(saved)
    window._color_timer.stop();window._reference_pending=False
    captured.clear()
    window.start_processing()
    try:
        assert len(captured)==1
        result=captured[0][1]()
        assert result["state"]=="queued"
        assert result["request"]["assets"]["sky_mask_path"] is not None
    finally:window._busy=False


def test_completed_star_result_keeps_actual_milkyway_offset(window,monkeypatch,tmp_path):
    record={"workflow_dir":str(tmp_path),"state":"succeeded","stage":"complete","artifacts":[],
            "request":{"milkyway_offset":-.35},"config":{"milkyway_offset":-.35},
            "finishing":{"state":"succeeded","background_strength_applied":.74,
                         "reduction":{"state":"succeeded","recipe":{"milkyway_offset":-.35}}}}
    monkeypatch.setattr(workflow,"workflow_status",lambda _:record)
    window.workflow_dir=str(tmp_path)
    window.refresh_workflow()
    assert window._star_result["gui_milkyway_offset"]==-.35
