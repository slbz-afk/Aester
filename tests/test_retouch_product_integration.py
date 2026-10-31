"""Product-level scope, discovery, resources and lifecycle; no GIMP launches."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import ImageCms
import pytest
import tifffile

from starstack import finishing_components, retouch_integration as guide
from starstack import finishing_service, finishing_candidates
from starstack.finishing_backend import RenderCancelled
from starstack.mcp_server import WorkspaceTools, create_server
from starstack.onboarding import connection_config

ROOT = Path(__file__).resolve().parents[1]
FINISHING = {"finishing_" + name for name in (
    "capabilities", "create", "get", "add_reference", "add_mask", "add_layers", "draw_mask", "tonal_mask",
    "revise", "submit_render", "render_status", "cancel_render", "preview", "feedback", "export", "recover")}


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("AESTER_APP_ROOT", str(ROOT))
    monkeypatch.setenv("AESTER_COMPONENT_ROOT", str(tmp_path / "components"))
    monkeypatch.delenv("AESTER_GIMP", raising=False)
    monkeypatch.setattr(finishing_components, "_automatic_candidates", lambda: ([], []))
    inp = tmp_path / "input"
    inp.mkdir()
    source = inp / "photo.tif"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tifffile.imwrite(source, np.full((24, 32, 3), 12345, np.uint16), photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    return WorkspaceTools(inp, tmp_path / "photos", cache_root=tmp_path / "work"), source


def test_discovery_and_prompt_read_no_install_or_runtime(host, tmp_path, monkeypatch):
    monkeypatch.setattr(finishing_components, "_probe_version", lambda *a: pytest.fail("must not execute"))
    before = set(tmp_path.rglob("*"))
    info = guide.get_retouch_info()
    assert not info["gimp"]["available"]
    assert info["skill"]["available"] and info["skill"]["installed"] is None
    assert info["skill"]["required"] is False
    prompt = guide.build_retouch_prompt(photo_path=host[1], intent='保留气辉 "green"', language="en")
    data = json.loads(prompt.split("\n\n")[1])
    assert data["photo_path"] == str(host[1]) and data["desired_look"] == '保留气辉 "green"'
    assert data["skill_read_path"] == info["skill"]["read_path"]
    assert data["reference_images"] is None and "ask once whether I have a reference" in prompt
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("language", ["zh_CN", "en"])
def test_prompt_carries_reference_images_by_kind_as_aesthetic_goals_only(host, language):
    references = [{"path": str(host[1].parent / "look.jpg"), "kind": "generated"},
                  {"path": str(host[1].parent / "mine.png"), "kind": "original"}]
    prompt = guide.build_retouch_prompt(photo_path=host[1], language=language, reference_images=references)
    data = json.loads(prompt.split("\n\n")[1])
    assert data["reference_images"] == references
    assert "finishing_add_reference" in prompt
    if language == "en":
        assert "never copy their stars, clouds, ridgelines" in prompt and "ask once" not in prompt
        assert "Components & prompts" in prompt
    else:
        assert "绝不能把参考图里的星点、云、山脊线" in prompt and "组件与提示词管理" in prompt
    plain = guide.build_retouch_prompt(language=language)
    assert json.loads(plain.split("\n\n")[1])["reference_images"] is None
    assert ("是否有喜欢的效果参考图" in plain) if language == "zh_CN" else ("ask once whether I have a reference" in plain)
    with pytest.raises(ValueError):
        guide.build_retouch_prompt(language=language, reference_images=[{"path": "x.jpg", "kind": "stolen"}])


def test_explicit_missing_selection_never_uses_other_install(host, tmp_path, monkeypatch):
    alternative = tmp_path / "other/gimp-console-3.2.exe"
    alternative.parent.mkdir()
    alternative.write_bytes(b"not executed")
    monkeypatch.setenv("AESTER_GIMP", str(alternative))
    missing = tmp_path / "missing/gimp-console-3.2.exe"
    info = guide.get_retouch_info(gimp_path=missing)
    assert info["gimp"]["status"] == "selected_missing"
    service = finishing_service.FinishingTools(host[0], gimp_path=missing)
    try:
        assert service.gimp_path == str(missing)
        assert service.capabilities()["skill_required"] is False
    finally:
        service.close()


def test_connected_service_discovers_component_prepared_after_start(host, tmp_path, monkeypatch):
    service = finishing_service.FinishingTools(host[0])
    try:
        assert service.capabilities()["gimp_configured"] is False
        executable = tmp_path / "gimp-console-3.2.exe"
        executable.write_bytes(b"not executed")
        monkeypatch.setenv("AESTER_GIMP", str(executable))
        assert service.capabilities()["gimp_configured"] is True
        assert service.gimp_path == str(executable)
    finally:
        service.close()


def test_removed_optional_gimp_keeps_base_connection_config_usable(host, tmp_path):
    workspace, source = host
    missing = tmp_path / "removed/gimp-console-3.2.exe"
    config = connection_config(source.parent, workspace.delivery_root, workspace.output_root, gimp_path=missing)
    spec = config["mcpServers"]["aester"]
    assert spec["env"]["AESTER_GIMP"] == str(missing)
    assert spec["args"][-1] == "starstack.mcp_server"
    assert guide.get_retouch_info(gimp_path=spec["env"]["AESTER_GIMP"])["gimp"]["status"] == "selected_missing"


@pytest.mark.parametrize("section", ["../README.md", "references/../../README.md", "C:/secret", "other.md"])
def test_skill_resource_cannot_read_arbitrary_host_file(host, section):
    with pytest.raises(ValueError, match="Unknown"):
        guide.read_retouch_skill(section)


def test_main_protocol_without_gimp_loads_skill_and_accepts_host_product(host):
    from mcp.shared.memory import create_connected_server_and_client_session
    workspace, source = host
    workspace.delivery_root.mkdir()
    product = workspace.delivery_root / "previous-photo.tif"
    product.write_bytes(source.read_bytes())
    original = product.read_bytes()
    server = create_server(workspace)

    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            tools = (await client.list_tools()).tools
            names = {t.name for t in tools}
            assert FINISHING <= names and "submit_workflow" in names and "retouch_setup" in names
            for tool in tools:
                if tool.name in FINISHING or tool.name == "retouch_setup":
                    assert not {"gimp_path", "script", "executable", "origin", "output_root"} & tool.inputSchema.get("properties", {}).keys()
            setup = await client.call_tool("retouch_setup", {})
            assert not setup.isError and setup.structuredContent["skill"]["available"]
            assert not setup.structuredContent["gimp"]["available"]
            for section in ("SKILL.md", "references/backend-interface.md"):
                resource = await client.read_resource("aester://retouch/skill/" + section)
                assert resource.contents[0].text == guide.read_retouch_skill(section)
            created = await client.call_tool("finishing_create", {"source": str(product), "intent": "brighten dunes"})
            assert not created.isError
            value = created.structuredContent
            assert Path(value["directory"]).is_relative_to(workspace.output_root)
            preview = await client.call_tool("finishing_preview", {"session_dir": value["directory"]})
            assert not preview.isError
            escaped = await client.call_tool("finishing_create", {"source": str(ROOT / "README.md"), "intent": "invalid"})
            assert escaped.isError
            revised = deepcopy(value["revisions"][0]["recipe"])
            revised["operations"] = [{"id": "raise", "type": "exposure", "enabled": True,
                                      "mask_id": None, "params": {"ev": .25}}]
            result = await client.call_tool("finishing_revise", {"session_dir": value["directory"],
                "recipe": revised, "expected_revision": 0})
            assert not result.isError
            stale = await client.call_tool("finishing_revise", {"session_dir": value["directory"],
                "recipe": revised, "expected_revision": 0})
            assert stale.isError
            assert not (await client.call_tool("get_help", {})).isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=15))
    assert product.read_bytes() == original


def test_missing_skill_does_not_hide_finishing_tools(host, tmp_path, monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session
    monkeypatch.setenv("AESTER_APP_ROOT", str(tmp_path / "no-skill-app"))
    server = create_server(host[0])

    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            setup = (await client.call_tool("retouch_setup", {})).structuredContent
            assert not setup["skill"]["available"] and setup["skill"]["required"] is False
            assert FINISHING <= {t.name for t in (await client.list_tools()).tools}
            created = await client.call_tool("finishing_create", {"source": str(host[1]), "intent": "manual recipe"})
            assert not created.isError
    asyncio.run(asyncio.wait_for(exercise(), timeout=15))


def test_main_disconnect_cancels_and_joins_only_its_render(host, monkeypatch, tmp_path):
    from mcp.shared.memory import create_connected_server_and_client_session
    executable = tmp_path / "gimp-console-3.2.exe"
    executable.write_bytes(b"not executed")
    monkeypatch.setenv("AESTER_GIMP", str(executable))
    started, stopped = threading.Event(), threading.Event()

    def render(*args, **kwargs):
        started.set()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if kwargs["cancel"]():
                    raise RenderCancelled("owned job cancelled")
                time.sleep(.01)
            pytest.fail("disconnect did not cancel")
        finally:
            stopped.set()

    monkeypatch.setattr(finishing_candidates, "render_recipe", render)
    server = create_server(host[0])

    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            record = (await client.call_tool("finishing_create", {"source": str(host[1]), "intent": "test cleanup"})).structuredContent
            request = await client.call_tool("finishing_submit_render", {"session_dir": record["directory"], "expected_revision": 0})
            assert not request.isError
            assert await asyncio.to_thread(started.wait, 3)
    asyncio.run(asyncio.wait_for(exercise(), timeout=10))
    assert stopped.is_set()
