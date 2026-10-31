"""P0 project intent regression: synthetic inputs, no real decode/engine."""
from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest

from starstack import workflow as wf
from starstack.mcp_server import WorkspaceTools, AccessDenied
from test_workflow import setup  # Existing synthetic engine/product fixture.


def files(scene):
    return [str(scene["inputs"] / f"{i}.NEF") for i in (2, 0, 1)]


def test_configured_model_alone_is_inert_and_need_not_exist(setup):
    record = wf.submit_workflow(setup["inputs"], setup["output"],
                                star_model_path=setup["output"] / "not-installed.onnx")
    assert record["request"]["assets"]["star_model_path"] is None
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert setup["calls"]["kwargs"]["transient_mode"] == "off"
    assert done["finishing"]["reason"] == "not-requested"
    assert done["trail_review"]["state"] == "not_requested"
    assert done["needs_review"] is False
    assert not (Path(done["workflow_dir"]) / "finishing").exists()
    assert {a["name"] for a in done["artifacts"]}.isdisjoint(
        {"stack_before_reduction_srgb16.tif", "stack_before_reduction.jpg"})


def test_explicit_list_preserves_order_and_ignores_new_sibling(setup):
    ordered = files(setup)
    record = wf.submit_workflow(None, setup["output"], input_files=ordered,
                                color_adjustments={"warmth": 12, "exposure_ev": .5})
    assert record["request"]["input_files"] == ordered
    (setup["inputs"] / "unselected.NEF").write_bytes(b"new exposure")
    wf._verify(record)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert setup["calls"]["kwargs"]["input_files"] == ordered
    assert setup["calls"]["kwargs"]["color_adjustments"] == {
        "warmth": 12., "tint": 0., "exposure_ev": .5}
    assert [x["path"] for x in done["checkpoints"]["stack"]["data"]["job"]["inputs"]] == ordered


def test_explicit_selected_source_change_rejected(setup):
    record = wf.submit_workflow(None, setup["output"], input_files=files(setup))
    (setup["inputs"] / "0.NEF").write_bytes(b"changed")
    result = wf.run_workflow(record["workflow_dir"])
    assert result["state"] == "failed"
    assert setup["calls"]["submits"] == 0


def test_explicit_output_excludes_each_source_parent(setup):
    another = setup["output"] / "originals"
    another.mkdir(parents=True)
    extra = another / "more.NEF"
    extra.write_bytes(b"another raw")
    with pytest.raises(ValueError, match="之外"):
        wf.submit_workflow(None, another / "results", input_files=[*files(setup)[:2], str(extra)])


@pytest.mark.parametrize("intent,value", [("trails_enabled", 1), ("trails_enabled", "false"),
                                           ("stars_enabled", None), ("stars_enabled", "true")])
def test_effect_intents_are_strict_booleans(setup, intent, value):
    with pytest.raises(ValueError, match="bool"):
        wf.submit_workflow(setup["inputs"], setup["output"], **{intent: value})


def test_fits_controls_fail_honestly_before_spawn(setup, monkeypatch):
    inspect = wf.core.inspect_sources
    monkeypatch.setattr(wf.core, "inspect_sources", lambda *a, **k: {**inspect(*a, **k), "kind": "fits"})
    with pytest.raises(ValueError, match="FITS"):
        wf.submit_workflow(setup["inputs"], setup["output"], color_adjustments={"exposure_ev": 1})
    assert setup["calls"]["submits"] == 0


@pytest.mark.parametrize("key,value", [("trails_enabled", "false"), ("stars_enabled", 1),
                                       ("star_reduction", "true")])
def test_mcp_does_not_coerce_effect_intents(setup, key, value):
    service = WorkspaceTools(setup["inputs"], setup["output"], setup["engine"])
    with pytest.raises(ValueError, match="bool"):
        service.submit_workflow(input_files=files(setup), **{key: value})


@pytest.mark.parametrize("changed", ["transient_mode", "color_adjustments", "input_selection"])
def test_child_intent_tampering_rejected(setup, changed):
    r = wf.submit_workflow(None, setup["output"], input_files=files(setup))
    done = wf.run_workflow(r["workflow_dir"])
    child = deepcopy(done["checkpoints"]["stack"]["data"]["job"])
    if changed == "transient_mode":
        child[changed] = "legacy"
    elif changed == "color_adjustments":
        child[changed]["warmth"] = 10.
    else:
        child[changed]["files"].reverse()
    with pytest.raises(ValueError):
        wf.validate_stack_binding(done, child)


def test_review_publishes_base_and_freezes_candidate_revision(setup, monkeypatch):
    real_submit = setup["submit"]
    def reviewed(*args, **kwargs):
        job = real_submit(*args, **kwargs)
        p = Path(job["job_dir"]) / "candidates.json"
        p.write_text('{"candidates":[]}', encoding="utf-8")
        revision = {"id": "test", "record": str(p), "sha256": wf.core._sha(p)}
        job["transient_revision"] = revision
        job["diagnostics"]["transients"] = {"status": "needs_review", "mode": "review",
                                            "record": str(p), "composite": None}
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", reviewed)
    r = wf.submit_workflow(setup["inputs"], setup["output"], trails_enabled=True)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["needs_review"] is True
    assert done["trail_review"]["automatic_addback"] is False
    assert setup["calls"]["kwargs"]["transient_mode"] == "review"
    old = done["checkpoints"]["stack"]["data"]["colour"]["photo"]["tiff_sha256"]
    assert next(a for a in done["artifacts"] if a["name"] == "final_srgb16.tif")["sha256"] == old
    Path(done["trail_review"]["scan"]["record"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        wf._cached(Path(done["workflow_dir"]), "stack")


def test_unavailable_trail_scan_is_visible_in_successful_base_delivery(setup, monkeypatch):
    real_submit = setup["submit"]
    def unavailable(*args, **kwargs):
        job = real_submit(*args, **kwargs)
        job["diagnostics"]["transients"] = {"status": "failed", "error": "synthetic scan failure"}
        wf.atomic_json(Path(job["job_dir"]) / "job.json", job)
        return job
    monkeypatch.setattr(wf.core, "submit_job", unavailable)
    r = wf.submit_workflow(setup["inputs"], setup["output"], trails_enabled=True)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["trail_review"]["state"] == "unavailable"
    assert "扫描未完成" in done["message"] and "尚未加回" in done["message"]


def test_old_request_retains_legacy_call_and_delivery(setup):
    r = wf.submit_workflow(setup["inputs"], setup["output"])
    d = Path(r["workflow_dir"])
    for mapping in (r["request"], r["config"]):
        for key in ("selection_mode", "trails_enabled", "stars_enabled", "color_adjustments"):
            mapping.pop(key)
    r["request"]["config"] = deepcopy(r["config"])
    r["signature"] = wf._digest(r["request"])
    wf.atomic_json(d / "workflow.json", r)
    result = wf.run_workflow(d)
    assert result["state"] == "succeeded", result.get("error")
    assert "transient_mode" not in setup["calls"]["kwargs"]
    assert "color_adjustments" not in setup["calls"]["kwargs"]
    assert result["finishing"]["reason"] == "no-star-model"
    assert "stack_before_reduction_srgb16.tif" in {a["name"] for a in result["artifacts"]}


def test_mcp_bounds_every_selected_file_before_submission(setup, monkeypatch):
    service = WorkspaceTools(setup["inputs"], setup["output"], setup["engine"])
    outside = setup["inputs"].parent / "outside.NEF"
    outside.write_bytes(b"private")
    with pytest.raises(AccessDenied):
        service.submit_workflow(input_files=[*files(setup)[:2], str(outside)])
    assert setup["calls"]["submits"] == 0
    result = service.submit_workflow(input_files=files(setup), trails_enabled=True,
                                     color_adjustments={"tint": -5})
    assert result["request"]["input_files"] == files(setup)
    assert result["request"]["trails_enabled"] is True
    assert result["request"]["color_adjustments"]["tint"] == -5


def test_cli_selects_files_and_shared_intents(monkeypatch, capsys):
    from starstack import cli
    called = {}
    monkeypatch.setattr(wf, "submit_workflow", lambda *a, **k: called.update(args=a, kwargs=k) or {"state": "queued"})
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--files", "one.NEF", "two.NEF", "three.NEF",
        "--retain-trails", "--reduce-stars", "--warmth", "-8", "--tint", "4", "--exposure-ev", ".3"])
    assert cli.main() == 0
    assert called["args"][0] is None
    assert called["kwargs"]["input_files"] == ["one.NEF", "two.NEF", "three.NEF"]
    assert called["kwargs"]["trails_enabled"] and called["kwargs"]["stars_enabled"]
    assert called["kwargs"]["color_adjustments"] == {"warmth": -8., "tint": 4., "exposure_ev": .3}


def test_cli_file_and_folder_are_exclusive(monkeypatch):
    from starstack import cli
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--input", "raw", "--files", "one.NEF"])
    with pytest.raises(SystemExit):
        cli.main()
