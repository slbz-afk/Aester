"""CLI routing and real MCP protocol checks; no render backend is launched."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from starstack import finishing_cli as cli, finishing_mcp as adapter


class RecordingService:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.failure = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def invoke(**kwargs):
            self.calls.append((name, kwargs, threading.get_ident()))
            if self.failure is not None:
                raise self.failure
            return {"method": name, "arguments": kwargs, "fixture": True}
        return invoke

    def close(self):
        self.closed = True


_CLI_CASES = [
    (["capabilities"], "capabilities", {}),
    (["create", "--source", "原图.tif", "--intent", "保留气辉"], "create",
     {"source": "原图.tif", "intent": "保留气辉"}),
    (["get", "--session", "s"], "get", {"session_dir": "s"}),
    (["add-reference", "--session", "s", "--path", "ref.png", "--kind", "generated"],
     "add_reference", {"session_dir": "s", "path": "ref.png", "kind": "generated", "note": ""}),
    (["add-mask", "--session", "s", "--path", "mask.png", "--mask-id", "sky",
      "--source-sha256", "a" * 64, "--note", "依据源图绘制"], "add_mask",
     {"session_dir": "s", "path": "mask.png", "mask_id": "sky", "source_sha256": "a" * 64,
      "note": "依据源图绘制"}),
    (["render", "--session", "s", "--expected-revision", "2", "--timeout", "8.5"], "render",
     {"session_dir": "s", "expected_revision": 2, "timeout": 8.5}),
    (["preview", "--session", "s"], "preview", {"session_dir": "s", "candidate_id": None, "crop": None}),
    (["preview", "--session", "s", "--candidate-id", "c", "--crop", "7", "8", "12", "9"],
     "preview", {"session_dir": "s", "candidate_id": "c", "crop": [7, 8, 12, 9]}),
    (["feedback", "--session", "s", "--candidate-id", "c", "--text", "用户说偏亮", "--decision", "revise"],
     "feedback", {"session_dir": "s", "candidate_id": "c", "text": "用户说偏亮", "decision": "revise"}),
    (["export", "--session", "s", "--candidate-id", "c"], "export", {"session_dir": "s", "candidate_id": "c"}),
    (["recover", "--session", "s"], "recover", {"session_dir": "s"}),
]


@pytest.mark.parametrize("arguments,method,expected", _CLI_CASES)
def test_cli_command_arguments(arguments, method, expected, capsys):
    service = RecordingService()
    assert cli.main(arguments, service=service) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {"method": method, "arguments": expected, "fixture": True}
    assert service.calls[0][:2] == (method, expected)
    assert not service.closed  # The injected caller still owns its service.


@pytest.mark.parametrize("command,key,value", [
    ("revise", "recipe", {"intent": "__import__('os').system('must remain data')", "operations": []}),
    ("draw-mask", "polygons", [[[0, 0], [8, 0], [8, 6]]]),
])
def test_cli_json_files_are_data_and_remain_unchanged(tmp_path, command, key, value, capsys):
    path = tmp_path / "参数 空格.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")
    before = path.read_bytes()
    arguments = [command, "--session", "s", f"--{key}-file", str(path)]
    if command == "revise":
        arguments += ["--expected-revision", "4"]
    else:
        arguments += ["--mask-id", "sky", "--feather-px", "1.5", "--invert"]
    service = RecordingService()
    assert cli.main(arguments, service=service) == 0
    assert service.calls[0][1][key] == value
    assert path.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["arguments"][key] == value


@pytest.mark.parametrize("payload", ["__import__('os')", "[]", '{"value":NaN}', '{"value":Infinity}', "{broken"])
def test_cli_bad_json_returns_failure_without_service_call(tmp_path, payload, capsys):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    service = RecordingService()
    assert cli.main(["revise", "--session", "s", "--expected-revision", "0",
                     "--recipe-file", str(path)], service=service) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Finishing failed" in output.err
    assert service.calls == []


def test_cli_service_failure_is_stderr_and_nonzero(capsys):
    service = RecordingService()
    service.failure = ValueError("Source changed; choose the current revision")
    assert cli.main(["get", "--session", "s"], service=service) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Source changed" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("arguments", [
    ["feedback", "--session", "s", "--candidate-id", "c", "--text", "accept", "--decision", "accept"],
    ["render", "--session", "s", "--expected-revision", "True"],
    ["render", "--session", "s", "--expected-revision", "0", "--timeout", "nan"],
    ["export", "--session", "s", "--candidate-id", "c", "--gimp", "caller.exe"],
])
def test_cli_invalid_arguments_fail_before_service(arguments, capsys):
    service = RecordingService()
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments, service=service)
    assert exc.value.code == 2
    assert service.calls == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("fail", [False, True])
def test_cli_owned_service_is_closed_and_host_config_is_not_forwarded(monkeypatch, capsys, fail):
    service = RecordingService()
    if fail:
        service.failure = ValueError("fixture failure")
    configurations = []

    def configured(args):
        configurations.append(vars(args).copy())
        return service

    monkeypatch.setattr(cli, "configured_service", configured)
    assert cli.main(["--input-root", "in", "--output-root", "out", "--cache-root", "cache",
                     "--gimp", "host.exe", "capabilities"]) == (1 if fail else 0)
    assert configurations[0]["gimp"] == "host.exe"
    assert service.calls[0][1] == {}
    assert service.closed
    capsys.readouterr()


_MCP_CASES = [
    ("capabilities", {}), ("create", {"source": "s.tif", "intent": "保留气辉"}),
    ("get", {"session_dir": "s"}),
    ("add_reference", {"session_dir": "s", "path": "ref.png", "kind": "generated"}),
    ("add_mask", {"session_dir": "s", "path": "mask.png", "mask_id": "sky", "source_sha256": "a" * 64}),
    ("add_layers", {"session_dir": "s", "bundle_id": "paired", "background_path": "b.tif",
        "diffuse_path": "d.tif", "pure_sky_mask_id": "pure", "transient_support_mask_id": "trails",
        "source_sha256": "a" * 64, "provenance": {"kind": "caller_reported_layer_chain"}}),
    ("draw_mask", {"session_dir": "s", "mask_id": "sky", "polygons": [[[0, 0], [6, 0], [6, 4]]]}),
    ("tonal_mask", {"session_dir": "s", "mask_id": "shadows", "knots": [0., 0., .25, .5]}),
    ("revise", {"session_dir": "s", "recipe": {"operations": []}, "expected_revision": 0}),
    ("submit_render", {"session_dir": "s", "expected_revision": 1}),
    ("render_status", {"job_id": "job"}), ("cancel_render", {"job_id": "job"}),
    ("preview", {"session_dir": "s", "crop": [7, 8, 12, 9]}),
    ("feedback", {"session_dir": "s", "candidate_id": "c", "text": "用户说偏亮"}),
    ("export", {"session_dir": "s", "candidate_id": "c"}), ("recover", {"session_dir": "s"}),
]


def test_mcp_actual_protocol_lists_and_calls_all_frozen_tools(monkeypatch):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    # Supplying a service must never import the default implementation.
    monkeypatch.setitem(sys.modules, "starstack.finishing_service", None)
    service = RecordingService()
    server = adapter.create_server(service=service)
    owner_thread = threading.get_ident()

    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            listing = await client.list_tools()
            assert {tool.name for tool in listing.tools} == {"finishing_" + name for name, _ in _MCP_CASES}
            for tool in listing.tools:
                properties = tool.inputSchema.get("properties", {})
                assert not {"gimp", "gimp_path", "executable", "output_root", "input_root", "script", "origin"} & properties.keys()
            for method, arguments in _MCP_CASES:
                result = await client.call_tool("finishing_" + method, arguments)
                assert not result.isError
                assert result.structuredContent["method"] == method
                for key, value in arguments.items():
                    assert result.structuredContent["arguments"][key] == value
            invalid = await client.call_tool("finishing_get", {"session_dir": 123})
            assert invalid.isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=15))
    assert service.closed
    for method, _, worker_thread in service.calls:
        if method in {"submit_render", "render_status", "cancel_render"}:
            assert worker_thread == owner_thread
        else:
            assert worker_thread != owner_thread


@pytest.mark.parametrize("tool,arguments", [
    ("finishing_revise", {"session_dir": "s", "recipe": {}, "expected_revision": True}),
    ("finishing_revise", {"session_dir": "s", "recipe": {}, "expected_revision": "0"}),
    ("finishing_submit_render", {"session_dir": "s", "expected_revision": 0, "timeout": "600"}),
    ("finishing_draw_mask", {"session_dir": "s", "mask_id": "sky", "polygons": [], "invert": 1}),
    ("finishing_feedback", {"session_dir": "s", "candidate_id": "c", "text": "用户接受", "decision": "accept"}),
])
def test_mcp_invalid_types_and_agent_acceptance_are_rejected(tool, arguments):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    service = RecordingService()
    server = adapter.create_server(service=service)

    async def exercise():
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool(tool, arguments)
            assert result.isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))
    assert service.calls == []


def test_mcp_service_error_is_a_protocol_error():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    service = RecordingService()
    service.failure = ValueError("Fixture path outside configured roots")

    async def exercise():
        async with create_connected_server_and_client_session(adapter.create_server(service)) as client:
            result = await client.call_tool("finishing_get", {"session_dir": "../outside"})
            assert result.isError
            assert "outside configured roots" in str(result)

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


def test_real_cli_module_capabilities_without_render(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-m", "starstack.finishing_cli",
            "--input-root", str(tmp_path / "inputs"), "--output-root", str(tmp_path / "outputs"),
            "capabilities"], capture_output=True, text=True, encoding="utf-8", timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema"] == "aester-finishing-capabilities-1"


def test_real_mcp_module_stdio_handshake_without_render(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    parameters = StdioServerParameters(command=sys.executable,
        args=["-m", "starstack.finishing_mcp", "--input-root", str(tmp_path / "inputs"),
              "--output-root", str(tmp_path / "outputs")], env=env)

    async def exercise():
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                identity = await client.initialize()
                assert identity.serverInfo.name == "AesterRetouch"
                assert len((await client.list_tools()).tools) == 16
                result = await client.call_tool("finishing_capabilities", {})
                assert not result.isError
                assert result.structuredContent["schema"] == "aester-finishing-capabilities-1"
                failure = await client.call_tool("finishing_render_status", {"job_id": "unknown"})
                assert failure.isError

    asyncio.run(asyncio.wait_for(exercise(), timeout=25))
