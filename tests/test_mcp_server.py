"""MCP boundary checks and a real SDK stdio handshake, without running Siril."""
import asyncio
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

from PIL import Image
from astropy.io import fits
import numpy as np
import pytest


def test_configured_engine_reaches_submission(workspace, monkeypatch):
    configured=str(workspace.outputs/'configured-siril.exe')
    workspace.service.siril_path=configured
    seen=[]
    def submit(*args,**kwargs):
        seen.append(kwargs)
        return {'job_dir':str(workspace.job_dir)}
    monkeypatch.setattr(core,'submit_job',submit)
    workspace.service.submit_job('夜景')
    assert seen[0]['siril_path']==configured


def test_configured_engine_is_checked_on_resume(workspace, monkeypatch):
    configured=str(workspace.outputs/'configured-siril.exe')
    workspace.service.siril_path=configured
    workspace.job['state']='cancelled';workspace.job['engine']['executable']=configured
    save_job(workspace)
    seen=[]
    def doctor(path):
        seen.append(path);return {'available':True,'executable':configured}
    monkeypatch.setattr(core,'doctor',doctor)
    monkeypatch.setattr(core,'resume_job',lambda *_:None)
    workspace.service.resume_job(str(workspace.job_dir))
    assert seen==[configured]

from starstack import core, references, preview as preview_module
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


@pytest.fixture
def workspace(tmp_path):
    inputs, outputs = tmp_path / "素材", tmp_path / "结果"
    source = inputs / "夜景"
    source.mkdir(parents=True)
    for number in range(3):
        fits.writeto(source / f"frame{number}.fit", np.zeros((3, 12, 20), dtype=np.float32))
    job_dir = outputs / "job-example"
    products = job_dir / "products"
    products.mkdir(parents=True)
    image = products / "natural.jpg"
    Image.new("RGB", (20, 12), "black").save(image)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.fit").write_bytes(b"must not read")
    job = {"job_dir": str(job_dir), "state": "succeeded", "stage": "complete", "progress": 100,
           "message": "测试夹具", "error": None,
           "inputs": [{"path": str(source / "frame0.fit"), "sha256": "unused"}],
           "mask": None, "engine": {"executable": str(tmp_path / "engine.exe"), "version": "test"},
           "checkpoint_files": {}, "checkpoints": [],
           "artifacts": [{"path": str(image), "kind": "jpeg", "label": "自然"}]}
    record = job_dir / "job.json"
    record.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    return SimpleNamespace(inputs=inputs, outputs=outputs, source=source, job_dir=job_dir,
                           record=record, job=job, outside=outside, image=image,
                           service=WorkspaceTools(inputs, outputs))


def save_job(workspace):
    workspace.record.write_text(json.dumps(workspace.job), encoding="utf-8")


@pytest.mark.parametrize("diagnostics, expected", [
    ({"input_count": 144, "stacked_count": 120},
     {"input_count": 144, "stacked_count": 120, "skipped_count": 24}),
    ({"input_count": 144, "stacked_count": 144, "skipped_count": 0},
     {"input_count": 144, "stacked_count": 144, "skipped_count": 0}),
    ({"input_count": 144, "stacked_count": None},
     {"input_count": 144, "stacked_count": None, "skipped_count": None}),
    (None, {"input_count": None, "stacked_count": None, "skipped_count": None}),
])
def test_status_exposes_sky_counts_without_rewriting_older_jobs(workspace, diagnostics, expected):
    if diagnostics is not None:
        workspace.job["diagnostics"] = diagnostics
    save_job(workspace)
    before = workspace.record.read_bytes()
    result = workspace.service.job_status(str(workspace.job_dir))
    assert result["state"] == "succeeded"
    assert result["diagnostics"] == expected
    assert workspace.record.read_bytes() == before


def symlink_or_skip(link, target, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if directory and os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0:
                assert link.is_junction()
                return
        pytest.skip(f"This OS account cannot create symlinks: {error}")


@pytest.mark.parametrize("relative", ["../outside", "夜景/../../outside"])
def test_input_traversal_rejected_before_core(workspace, monkeypatch, relative):
    monkeypatch.setattr(core, "inspect_sources", lambda *_: pytest.fail("core must not run"))
    with pytest.raises(AccessDenied):
        workspace.service.inspect_sources(relative)


def test_absolute_input_outside_rejected(workspace):
    with pytest.raises(AccessDenied):
        workspace.service.inspect_sources(str(workspace.outside))


def test_sources_are_real_owned_files(workspace):
    result = workspace.service.inspect_sources("夜景")
    assert result["count"] == 3
    assert all(Path(item).is_relative_to(workspace.inputs) for item in result["files"])


def test_symlink_source_directory_rejected(workspace):
    symlink_or_skip(workspace.inputs / "shortcut", workspace.outside, True)
    with pytest.raises(AccessDenied):
        workspace.service.inspect_sources("shortcut")


def test_symlink_raw_child_rejected_before_core(workspace, monkeypatch):
    symlink_or_skip(workspace.source / "leak.fit", workspace.outside / "secret.fit")
    monkeypatch.setattr(core, "inspect_sources", lambda *_: pytest.fail("core must not stat escaped child"))
    with pytest.raises(AccessDenied):
        workspace.service.inspect_sources("夜景")


def test_mask_outside_rejected_before_submit(workspace, monkeypatch):
    monkeypatch.setattr(core, "submit_job", lambda *a, **k: pytest.fail("must not submit"))
    with pytest.raises(AccessDenied):
        workspace.service.submit_job("夜景", str(workspace.outside / "mask.png"))


def test_submission_output_is_host_root_and_result_is_owned(workspace, monkeypatch):
    def submit(input_dir, output_root, **kwargs):
        assert input_dir == str(workspace.source)
        assert output_root == str(workspace.outputs)
        assert kwargs["idempotency_key"] == "repeat-request"
        return workspace.job

    monkeypatch.setattr(core, "submit_job", submit)
    result = workspace.service.submit_job("夜景", idempotency_key="repeat-request")
    assert result["job_dir"] == str(workspace.job_dir)


def test_submission_cannot_return_unowned_job(workspace, monkeypatch):
    monkeypatch.setattr(core, "submit_job", lambda *a, **k: {"job_dir": str(workspace.outside)})
    with pytest.raises(AccessDenied):
        workspace.service.submit_job("夜景")


def test_output_root_symlink_rejected(workspace):
    link = workspace.outputs.parent / "linked-output"
    symlink_or_skip(link, workspace.outside, True)
    with pytest.raises(AccessDenied):
        WorkspaceTools(workspace.inputs, link)


@pytest.mark.parametrize("operation", ["job_status", "cancel_job", "resume_job", "preview", "render_job_preview", "list_transients", "set_transients"])
def test_unowned_job_rejected(workspace, operation):
    with pytest.raises(AccessDenied):
        getattr(workspace.service, operation)(str(workspace.outside))


def test_job_symlink_rejected(workspace):
    symlink_or_skip(workspace.outputs / "job-link", workspace.outside, True)
    with pytest.raises(AccessDenied):
        workspace.service.job_status("job-link")


def test_internal_job_link_rejected_before_cancel(workspace, monkeypatch):
    symlink_or_skip(workspace.job_dir / "cancel.request", workspace.outside / "secret.fit")
    monkeypatch.setattr(core, "cancel_job", lambda *_: pytest.fail("must not touch link"))
    with pytest.raises(AccessDenied):
        workspace.service.cancel_job("job-example")


def test_metadata_job_owner_must_match(workspace):
    workspace.job["job_dir"] = str(workspace.outside)
    save_job(workspace)
    with pytest.raises(AccessDenied):
        workspace.service.job_status("job-example")


@pytest.mark.parametrize("escaped", ["../secret.jpg", "absolute"])
def test_preview_artifact_cannot_escape_job(workspace, escaped):
    workspace.job["artifacts"][0]["path"] = (
        str(workspace.outside / "secret.jpg") if escaped == "absolute" else escaped)
    save_job(workspace)
    with pytest.raises(AccessDenied):
        workspace.service.preview("job-example")


def test_preview_requires_registered_jpeg_and_never_returns_pixels(workspace):
    before = workspace.record.read_bytes()
    result = workspace.service.preview("job-example")
    assert (result["width"], result["height"]) == (20, 12)
    assert result["transport"] == "local_path_only"
    assert not any(key in result for key in ("data", "base64", "image"))
    assert workspace.record.read_bytes() == before
    with pytest.raises(ValueError):
        workspace.service.preview("job-example", -1)
    workspace.job["artifacts"][0]["kind"] = "fits"
    save_job(workspace)
    with pytest.raises(ValueError):
        workspace.service.preview("job-example")


def _transients_record(workspace):
    """A finished job's bright-trail record (RUN_LOG §28) with one measured candidate."""
    from starstack import transient_stage, transient_trails
    work = workspace.job_dir / "work"
    (work / "transients" / "crops").mkdir(parents=True)
    png = work / "transients" / "crops" / "light_00001-1.png"
    Image.new("L", (8, 8), 0).save(png)
    record = {"version": transient_trails.TRANSIENT_VERSION, "stage_version": transient_stage.STAGE_VERSION, "grid": "native",
              "frames_scanned": 1, "frame_order": ["light_00001"], "reference_stem": "light_00001",
              "params": {"detection": dict(transient_trails.DEFAULT_DETECTION), "evidence": dict(transient_trails.DEFAULT_EVIDENCE),
                         "layer": dict(transient_trails.DEFAULT_LAYER)},
              "policy": dict(transient_trails.DEFAULT_POLICY), "marks": [], "per_frame": {}, "frame_stats": {},
              "candidates": [{"id": "light_00001-1", "stem": "light_00001", "source": "auto", "detection": {"status": "detected"},
                              "measurement": {"status": "measured", "x0": 1.0, "y0": 1.0, "x1": 9.0, "y1": 9.0, "length_px": 11.3,
                                              "half_width_px": 2, "contrast_median": 0.5, "total_snr": 80.0, "colour": {"RG": 1.0, "BG": 1.0},
                                              # a meteor-like light curve (rise, peak, fade; ends ~0.1 of the peak) with data
                                              # beyond both ends -> suggested meteor under the 1.1.0 light-curve rule
                                              "series": [round(30.0 * (0.1 + np.exp(-((i - 16) / 7.0) ** 2)), 3) for i in range(30)],
                                              "profile_step_px": 4.0,
                                              "ends": {"start": {"data_beyond": True, "at_border": False}, "stop": {"data_beyond": True, "at_border": False}}},
                              "provenance": {"stem": "light_00001", "frame_index": 1, "input": workspace.job["inputs"][0], "date_obs": None},
                              "geometry": {"native_source": [[1.0, 1.0], [9.0, 9.0]]},
                              "label": None, "override": None, "note": "", "review_png": str(png)}]}
    transient_trails.add_evidence(record["candidates"], record["frame_order"])
    for c in record["candidates"]:
        c["suggestion"] = transient_trails.suggest(c)
    transient_trails.resolve_decisions(record)
    record["summary"] = transient_stage.summary(record)
    transient_trails.write_candidates(work / "transients", record)
    return record


def test_list_transients_reports_candidates_and_bounded_review_paths(workspace):
    assert workspace.service.list_transients("job-example")["transients"] is None     # no record: says so, no error
    _transients_record(workspace)
    result = workspace.service.list_transients("job-example")
    assert result["summary"]["candidates"] == 1
    assert result["candidates"][0]["suggestion"]["class"] == "meteor"
    assert result["candidates"][0]["decision"] == "keep"
    assert Path(result["candidates"][0]["review_png"]).is_file()
    assert set(result["classes"]) == {"meteor", "satellite", "aircraft", "artifact", "unknown"}
    # a review path forged outside the job is not returned
    from starstack import transient_trails
    record = transient_trails.read_candidates(workspace.job_dir / "work" / "transients")
    record["candidates"][0]["review_png"] = str(workspace.outside / "secret.fit")
    transient_trails.write_candidates(workspace.job_dir / "work" / "transients", record)
    with pytest.raises(AccessDenied):
        workspace.service.list_transients("job-example")


def test_set_transients_validates_before_backend_and_forwards(workspace, monkeypatch):
    _transients_record(workspace)
    calls = []
    monkeypatch.setattr(core, "recompose_transients", lambda job_dir, **kwargs: calls.append((job_dir, kwargs)))
    for bad in ({"labels": "meteor"}, {"overrides": {"x": 3}}, {"marks": "nope"}, {"marks": [{"x0": 1}]}, {"policy": {1: "keep"}}):
        with pytest.raises(ValueError):
            workspace.service.set_transients("job-example", **bad)
    assert calls == []
    result = workspace.service.set_transients("job-example", labels={"light_00001-1": "satellite"}, policy={"satellite": "keep"},
                                              marks=[{"input": "frame0.fit", "x0": 1, "y0": 2, "x1": 3, "y1": 4}])
    assert calls and calls[0][0] == str(workspace.job_dir)
    assert calls[0][1]["labels"] == {"light_00001-1": "satellite"}
    assert result["summary"]["candidates"] == 1


@pytest.mark.parametrize("field", ["source", "checkpoint", "engine"])
def test_resume_cannot_follow_forged_paths(workspace, monkeypatch, field):
    workspace.job["state"] = "cancelled"
    if field == "source":
        workspace.job["inputs"][0]["path"] = str(workspace.outside / "secret.fit")
    elif field == "checkpoint":
        workspace.job["checkpoint_files"] = {"stack": [{"path": "../secret.fit"}]}
    else:
        workspace.job["engine"]["executable"] = str(workspace.outside / "execute.exe")
    save_job(workspace)
    monkeypatch.setattr(core, "doctor", lambda: {"available": True, "executable": str(workspace.outputs / "siril.exe")})
    monkeypatch.setattr(core, "resume_job", lambda *_: pytest.fail("must not resume"))
    with pytest.raises(AccessDenied):
        workspace.service.resume_job("job-example")


def test_official_sdk_stdio_handshake_tools_and_read_only_calls(workspace):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise():
        env = dict(os.environ)
        env.update(STARSTACK_INPUT_ROOT=str(workspace.inputs), STARSTACK_OUTPUT_ROOT=str(workspace.outputs))
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "starstack.mcp_server"], env=env)
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listing = await session.list_tools()
                names = {tool.name for tool in listing.tools}
                assert names == {"rotate_reference", "draw_regions", "import_photos", "create_regions", "update_regions", "get_regions", "preview_regions", "open_trail_review",
                                 "save_trail_review", "trail_candidate_context", "apply_trail_review",
                                 "coordinate_background", "inspect_sources", "submit_job", "job_status", "cancel_job", "resume_job", "preview",
                                 "prepare_reference", "render_job_preview", "inspect_raw", "protect_trace",
                                 "list_transients", "set_transients", "finish_photo", "finish_milkyway", "export_star_layers", "reduce_stars", "suggest_star_reduction",
                                 "export_master", "render_photo", "get_help", "submit_workflow", "workflow_status", "cancel_workflow", "resume_workflow",
                                 "inspect_workflow_cache",
                                 "retouch_setup", "finishing_capabilities", "finishing_create", "finishing_get", "finishing_add_reference",
                                 "finishing_add_mask", "finishing_add_layers", "finishing_draw_mask", "finishing_tonal_mask", "finishing_revise",
                                 "finishing_submit_render", "finishing_render_status", "finishing_cancel_render", "finishing_preview",
                                 "finishing_feedback", "finishing_export", "finishing_recover"}
                complete = next(tool for tool in listing.tools if tool.name == "submit_workflow")
                assert not {"siril_path", "output_root", "stack_engine_path", "star_model_path"} & complete.inputSchema['properties'].keys()
                help_result = await session.call_tool("get_help", {})
                assert not help_result.isError and "GUI" in str(help_result) and "MCP" in str(help_result)
                layers = next(tool for tool in listing.tools if tool.name == "export_star_layers")
                assert set(layers.inputSchema['properties']) == {'separation_result'}
                invalid_layer = await session.call_tool('export_star_layers', {'separation_result': '../outside/result.json'})
                assert invalid_layer.isError
                submit = next(tool for tool in listing.tools if tool.name == "submit_job")
                assert "output_root" not in submit.inputSchema["properties"]
                assert "siril_path" not in submit.inputSchema["properties"]
                restart = submit.inputSchema["properties"]["force_restart"]          # H2: strict bool, default off
                assert restart["type"] == "boolean" and restart["default"] is False
                for value in ["true", 1, "yes"]:
                    invalid = await session.call_tool("submit_job", {"input_dir": "夜景", "idempotency_key": "k", "force_restart": value})
                    assert invalid.isError and "force_restart" in str(invalid)
                reference = next(tool for tool in listing.tools if tool.name == "prepare_reference")
                assert set(reference.inputSchema["properties"]) == {"input_dir", "input_files", "color_adjustments", "image_rotation"}
                render = next(tool for tool in listing.tools if tool.name == "render_job_preview")
                assert set(render.inputSchema["properties"]) == {"job_dir", "red_gain", "blue_gain", "exposure_ev"}
                assert render.inputSchema["properties"]["red_gain"]["minimum"] == .25
                assert render.inputSchema["properties"]["exposure_ev"]["maximum"] == 3
                for name, arguments in [("inspect_sources", {"input_dir": "夜景"}),
                                        ("job_status", {"job_dir": "job-example"}),
                                        ("preview", {"job_dir": "job-example"})]:
                    result = await session.call_tool(name, arguments)
                    assert not result.isError, result
                    assert result.structuredContent
                invalid = await session.call_tool("inspect_sources", {"input_dir": "../outside"})
                assert invalid.isError
                # Strict SDK validation must preserve the backend's rejection of
                # booleans and numeric strings instead of coercing them to gains.
                for gain in [True, "1.25", .1, 5.]:
                    invalid = await session.call_tool("render_job_preview", {
                        "job_dir": "job-example", "red_gain": gain})
                    assert invalid.isError
                    assert "red_gain" in str(invalid)
                    assert "线性母版" not in str(invalid)

    before = workspace.record.read_bytes()
    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
    assert workspace.record.read_bytes() == before


def fake_reference(workspace):
    directory = workspace.outputs / "reference-controlled"
    directory.mkdir()
    result = {"reference_dir": str(directory), "state": "succeeded", "width": 20, "height": 12,
              "source_path": str(workspace.source / "frame0.fit"), "metadata": {"mask": {"is_segmentation": False}}}
    for key, name in [("preview_path", "reference_preview.png"),
                      ("mask_template_path", "sky_mask_template.png"), ("linear_path", "ref_00001.fit")]:
        path = directory / name
        path.write_bytes(b"controlled file; not a real image")
        result[key] = str(path)
    return result


def fake_render(workspace):
    directory = workspace.job_dir / "preview-variations" / "controlled"
    directory.mkdir(parents=True)
    recipe = directory / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    artifacts = []
    for name in ["natural.jpg", "cool.jpg", "warm.jpg", "natural_display_16bit.tif"]:
        path = directory / name
        path.write_bytes(b"controlled artifact")
        artifacts.append({"path": str(path), "kind": "jpeg" if path.suffix == ".jpg" else "tiff",
                          "label": name, "sha256": "controlled-checksum"})
    return {"directory": str(directory), "recipe_path": str(recipe), "artifacts": artifacts,
            "state": "succeeded", "job_dir": str(workspace.job_dir)}


def test_prepare_reference_uses_host_roots_and_engine_without_modifying_existing_job(workspace, monkeypatch):
    result = fake_reference(workspace)
    service = WorkspaceTools(workspace.inputs, workspace.outputs, siril_path="host-selected-siril.exe")
    before = workspace.record.read_bytes()

    def prepare(input_dir, output_root, siril_path):
        assert input_dir == str(workspace.source)
        assert output_root == str(workspace.outputs)
        assert siril_path == "host-selected-siril.exe"
        return result

    monkeypatch.setattr(references, "prepare_reference", prepare)
    actual = service.prepare_reference("夜景")
    assert actual == result
    assert workspace.record.read_bytes() == before


@pytest.mark.parametrize("value", ["../outside", "absolute"])
def test_prepare_reference_rejects_escaped_inputs_before_backend(workspace, monkeypatch, value):
    monkeypatch.setattr(references, "prepare_reference", lambda *a, **k: pytest.fail("must not prepare"))
    with pytest.raises(AccessDenied):
        workspace.service.prepare_reference(str(workspace.outside) if value == "absolute" else value)


def test_prepare_reference_rejects_linked_input_before_backend(workspace, monkeypatch):
    symlink_or_skip(workspace.inputs / "shortcut", workspace.source, True)
    monkeypatch.setattr(references, "prepare_reference", lambda *a, **k: pytest.fail("must not prepare"))
    with pytest.raises(AccessDenied):
        workspace.service.prepare_reference("shortcut")


@pytest.mark.parametrize("key", ["reference_dir", "preview_path", "mask_template_path", "linear_path", "source_path"])
def test_prepare_reference_rejects_unowned_returned_paths(workspace, monkeypatch, key):
    result = fake_reference(workspace)
    result[key] = str(workspace.outside if key == "reference_dir" else workspace.outside / "secret.fit")
    monkeypatch.setattr(references, "prepare_reference", lambda *a, **k: result)
    with pytest.raises(AccessDenied):
        workspace.service.prepare_reference("夜景")


def test_render_job_preview_forwards_valid_parameters_and_keeps_artifact_hashes(workspace, monkeypatch):
    result = fake_render(workspace)
    before = workspace.record.read_bytes()

    def render(job_dir, **parameters):
        assert job_dir == str(workspace.job_dir)
        assert parameters == {"red_gain": 1.4, "blue_gain": .8, "exposure_ev": -.5}
        return result

    monkeypatch.setattr(preview_module, "render_job_preview", render)
    actual = workspace.service.render_job_preview("job-example", 1.4, .8, -.5)
    assert actual == result
    assert len(actual["artifacts"]) == 4
    assert all(item["sha256"] == "controlled-checksum" for item in actual["artifacts"])
    assert workspace.record.read_bytes() == before


@pytest.mark.parametrize("name,value", [("red_gain", True), ("red_gain", "1"), ("red_gain", float("nan")),
    ("red_gain", float("inf")), ("red_gain", .249), ("blue_gain", 4.01),
    ("blue_gain", None), ("exposure_ev", -3.01), ("exposure_ev", 3.01)])
def test_render_job_preview_rejects_invalid_parameters_before_backend(workspace, monkeypatch, name, value):
    monkeypatch.setattr(preview_module, "render_job_preview", lambda *a, **k: pytest.fail("must not render"))
    with pytest.raises(ValueError, match="有限数值"):
        workspace.service.render_job_preview("job-example", **{name: value})


@pytest.mark.parametrize("operation", ["prepare_reference", "render_job_preview"])
def test_new_mutations_reject_linked_output_entries_before_backend(workspace, monkeypatch, operation):
    symlink_or_skip(workspace.outputs / "linked-entry", workspace.outside, True)
    monkeypatch.setattr(references, "prepare_reference", lambda *a, **k: pytest.fail("must not prepare"))
    monkeypatch.setattr(preview_module, "render_job_preview", lambda *a, **k: pytest.fail("must not render"))
    with pytest.raises(AccessDenied):
        getattr(workspace.service, operation)("夜景" if operation == "prepare_reference" else "job-example")


def test_render_rejects_forged_master_path_before_backend(workspace, monkeypatch):
    workspace.job["artifacts"].append({"path": str(workspace.outside / "secret.fit"),
                                        "kind": "fits", "label": "forged master"})
    save_job(workspace)
    monkeypatch.setattr(preview_module, "render_job_preview", lambda *a, **k: pytest.fail("must not read master"))
    with pytest.raises(AccessDenied):
        workspace.service.render_job_preview("job-example")


def test_render_rejects_link_inside_job_before_backend(workspace, monkeypatch):
    symlink_or_skip(workspace.job_dir / "preview-variations", workspace.outside, True)
    monkeypatch.setattr(preview_module, "render_job_preview", lambda *a, **k: pytest.fail("must not render"))
    with pytest.raises(AccessDenied):
        workspace.service.render_job_preview("job-example")


@pytest.mark.parametrize("field", ["directory", "recipe", "artifact"])
def test_render_rejects_backend_result_outside_variation(workspace, monkeypatch, field):
    result = fake_render(workspace)
    if field == "directory":
        result["directory"] = str(workspace.outside)
    elif field == "recipe":
        result["recipe_path"] = str(workspace.outside / "secret.fit")
    else:
        result["artifacts"][0]["path"] = str(workspace.image)
    monkeypatch.setattr(preview_module, "render_job_preview", lambda *a, **k: result)
    with pytest.raises(AccessDenied):
        workspace.service.render_job_preview("job-example")
