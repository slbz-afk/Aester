"""MCP side of the per-job cache cleanup: one read-only inspection tool, no deletion tool."""
from pathlib import Path
import asyncio
import json

import pytest

from starstack import cache_cleanup as cc, workflow as wf
from starstack.mcp_server import AccessDenied, create_server
from test_mcp_delivery_v2 import setup, service_for  # noqa: F401  (fixture: the mask lives in the input root)
from test_workflow import setup as base_setup  # noqa: F401  (fixture used by the setup above)
from test_delivery_images import completed  # noqa: F401  (fixture)
from test_cache_cleanup import ALL_FRAMES, frames_exist, populate, snapshot


def test_inspect_tool_is_read_only_and_uses_the_shared_entry_point(setup, completed, tmp_path):
    done, _, _ = completed
    directory, child = Path(done["workflow_dir"]), Path(done["stack_job_dir"])
    job = json.loads((child / "job.json").read_text(encoding="utf-8"))
    for artifact in job["artifacts"]:
        artifact.setdefault("label", "合成线性母版")      # as a real stack job records it (MCP lists labels)
    wf.atomic_json(child / "job.json", job)
    populate(child)
    service = service_for(setup, done)
    before = snapshot(directory)
    plan = service.inspect_workflow_cache(done["workflow_dir"])
    assert plan == cc.inspect_workflow_cache(directory)
    assert plan["state"] == "cleanable" and plan["files_deletable"] == len(ALL_FRAMES)
    assert snapshot(directory) == before and len(frames_exist(child)) == len(ALL_FRAMES)
    assert not any("clean" in name and name != "inspect_workflow_cache" for name in dir(service))
    stray = tmp_path / "elsewhere" / directory.name
    stray.mkdir(parents=True)
    with pytest.raises((AccessDenied, ValueError)):
        service.inspect_workflow_cache(str(stray))
    # After the user cleaned it on the desktop: status says so, the plan is "cleaned", recompose is refused.
    cc.clean_workflow_cache(directory, plan["plan_id"])
    status = service.workflow_status(done["workflow_dir"])
    assert status["cache_cleaned"] is True and status["cache_cleanup"]["state"] == "completed"
    assert service.inspect_workflow_cache(done["workflow_dir"])["state"] == "cleaned"
    with pytest.raises(ValueError, match="缓存已清理"):
        service.set_transients(str(child), labels={"light_00001-1": "meteor"})
    with pytest.raises(ValueError, match="缓存已清理"):
        service.open_trail_review(done["workflow_dir"])


def test_sdk_lists_one_cache_tool_taking_only_the_job_folder(setup, completed):
    done, _, _ = completed
    tools = asyncio.run(create_server(service_for(setup, done)).list_tools())
    cache_tools = {tool.name: tool for tool in tools if "cache" in tool.name or "clean" in tool.name}
    assert set(cache_tools) == {"inspect_workflow_cache"}
    assert set(cache_tools["inspect_workflow_cache"].inputSchema["properties"]) == {"workflow_dir"}


def test_inspect_tool_does_not_need_the_originals_in_place(setup, completed, tmp_path):
    """The desktop cleans a job whose RAWs were moved; the read-only tool inspects it too (no original is admitted)."""
    done, _, _ = completed
    directory, child = Path(done["workflow_dir"]), Path(done["stack_job_dir"])
    populate(child)
    service = service_for(setup, done)
    for original in json.loads((child / "job.json").read_text(encoding="utf-8"))["inputs"]:
        Path(original["path"]).unlink()
    with pytest.raises((AccessDenied, OSError, ValueError)):     # the full admission needs the originals
        service.workflow_status(done["workflow_dir"])
    plan = service.inspect_workflow_cache(done["workflow_dir"])
    assert plan["state"] == "cleanable" and plan == cc.inspect_workflow_cache(directory)
    assert plan["stage_copies_kept"] == len([p for p in ALL_FRAMES if p.startswith("stage/")])
    # Still bounded: only a workflow- folder directly in the host's cache root, whose record names it.
    for value in (str(child), str(directory.parent), str(tmp_path)):
        with pytest.raises((AccessDenied, ValueError)):
            service.inspect_workflow_cache(value)
    record = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    record["config"]["delivery_root"] = str(tmp_path / "other-photos")
    (directory / "workflow.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AccessDenied):
        service.inspect_workflow_cache(done["workflow_dir"])
