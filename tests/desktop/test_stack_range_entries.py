"""Independent public-entry checks; no image processing or engine execution."""
import asyncio
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from starstack import cli, core
from starstack.mcp_server import WorkspaceTools, create_server


@pytest.mark.parametrize("extra, expected", [
    ([], "legacy"),
    (["--stack-range-mode", "legacy"], "legacy"),
    (["--stack-range-mode", "positive_power2_v1"], "positive_power2_v1"),
])
def test_cli_passes_explicit_range_independently_of_support(monkeypatch, capsys, extra, expected):
    calls = []
    monkeypatch.setattr(cli, "submit_job", lambda *args, **kwargs: calls.append((args, kwargs)) or {"state": "queued"})
    monkeypatch.setattr(sys, "argv", ["starstack", "run", "--input", "input",
                                    "--support-mode", "strict_siril_1_4_4", *extra])
    assert cli.main() == 0
    assert len(calls) == 1
    assert calls[0][1]["stack_range_mode"] == expected
    assert calls[0][1]["support_mode"] == "strict_siril_1_4_4"
    assert "dcp" not in calls[0][1]
    assert json.loads(capsys.readouterr().out)["state"] == "queued"


def test_cli_rejects_unknown_range_before_submission(monkeypatch):
    monkeypatch.setattr(cli, "submit_job", lambda *args, **kwargs: pytest.fail("must not submit"))
    monkeypatch.setattr(sys, "argv", ["starstack", "run", "--input", "input", "--stack-range-mode", "automatic"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.fixture
def service_record(tmp_path, monkeypatch):
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    job_dir = outputs / "job-entry-test"
    job_dir.mkdir(parents=True)
    job = {"job_dir": str(job_dir), "state": "succeeded", "artifacts": [], "support_mode": "strict_siril_1_4_4"}
    path = job_dir / "job.json"
    path.write_text(json.dumps(job), encoding="utf8")
    service = WorkspaceTools(inputs, outputs)
    monkeypatch.setattr(service, "inspect_sources", lambda _: {"input_dir": str(inputs)})
    return service, job_dir, path, job


@pytest.mark.parametrize("mode", [None, "legacy", "positive_power2_v1"])
def test_mcp_service_forwards_range_keyword_to_core(service_record, monkeypatch, mode):
    service, job_dir, _, _ = service_record
    calls = []
    monkeypatch.setattr(core, "submit_job", lambda *a, **kw: calls.append((a, kw)) or {"job_dir": str(job_dir)})
    arguments = {} if mode is None else {"stack_range_mode": mode}
    result = service.submit_job(".", support_mode="strict_siril_1_4_4", **arguments)
    assert result["job_dir"] == str(job_dir)
    assert len(calls) == 1
    assert calls[0][1]["stack_range_mode"] == (mode or "legacy")
    assert calls[0][1]["support_mode"] == "strict_siril_1_4_4"


@pytest.mark.parametrize("summary", [None, {"mode": "positive_power2_v1", "encoded_frames": 15, "scale": .5}])
def test_mcp_status_only_exposes_recorded_range_summary(service_record, summary):
    service, job_dir, path, job = service_record
    job["diagnostics"] = {"input_count": 16, "stacked_count": 15}
    if summary is not None:
        job["diagnostics"]["stack_range"] = summary
    path.write_text(json.dumps(job), encoding="utf8")
    before = path.read_bytes()
    result = service.job_status(str(job_dir))
    assert result["diagnostics"]["skipped_count"] == 1
    if summary is None:
        assert "stack_range" not in result["diagnostics"]
    else:
        assert result["diagnostics"]["stack_range"] == summary
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", [None, "positive_power2_v1"])
def test_async_mcp_tool_schema_and_forwarding(service_record, monkeypatch, mode):
    pytest.importorskip("mcp")
    service, _, _, _ = service_record
    calls = []
    monkeypatch.setattr(service, "submit_job", lambda *a, **kw: calls.append((a, kw)) or {"state": "queued"})
    server = create_server(service)

    async def exercise():
        tools = await server.list_tools()
        schema = next(t.inputSchema for t in tools if t.name == "submit_job")
        option = schema["properties"]["stack_range_mode"]
        assert option["default"] == "legacy"
        assert option["enum"] == ["legacy", "positive_power2_v1", "finite_validity_v1"]
        arguments = {"input_dir": ".", "support_mode": "strict_siril_1_4_4"}
        if mode is not None:
            arguments["stack_range_mode"] = mode
        await server.call_tool("submit_job", arguments)

    asyncio.run(exercise())
    assert calls == [((".", None, None, "strict_siril_1_4_4"), {"stack_range_mode": mode or "legacy"})]


def test_async_mcp_tool_rejects_unknown_mode(service_record, monkeypatch):
    pytest.importorskip("mcp")
    from mcp.server.fastmcp.exceptions import ToolError
    service, _, _, _ = service_record
    monkeypatch.setattr(service, "submit_job", lambda *a, **kw: pytest.fail("must not submit"))
    server = create_server(service)
    with pytest.raises(ToolError, match="stack_range_mode"):
        asyncio.run(server.call_tool("submit_job", {"input_dir": ".", "stack_range_mode": "automatic"}))


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    from aester_desktop import gui
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    monkeypatch.setattr(gui.MainWindow, "refresh_job", lambda self: None)
    instance = gui.MainWindow(persist_preferences=False)
    instance.input_dir.setText(str(tmp_path))
    instance.output_dir.setText(str(tmp_path / "output"))
    yield instance
    instance.timer.stop()
    instance._busy = False
    instance._closing_requested = False
    instance.job_state = ""
    instance.close()
    app.processEvents()


def test_gui_option_default_text_and_tooltip(window):
    checkbox = window.stack_range_checkbox
    assert checkbox.text() == "保护堆栈高光"
    assert not checkbox.isChecked()
    assert checkbox.isEnabled()
    assert checkbox.toolTip() == "保留额外高光范围；单通道零值和负值照常堆栈，暗端与普通堆栈相同在0处截断；遇非有限值会停止并提示。"


@pytest.mark.parametrize("busy,state,closing,enabled", [
    (True, "", False, False), (False, "queued", False, False),
    (False, "running", False, False), (False, "cancel_requested", False, False),
    (False, "succeeded", False, True), (False, "failed", False, True),
    (False, "cancelled", False, True), (False, "", True, False),
])
def test_gui_range_disabled_during_busy_or_active_jobs(window, busy, state, closing, enabled):
    window._busy, window.job_state, window._closing_requested = busy, state, closing
    window._update_controls()
    assert window.stack_range_checkbox.isEnabled() is enabled


@pytest.mark.parametrize("checked,mode", [(False, "legacy"), (True, "positive_power2_v1")])
def test_gui_captures_range_before_async_submission(window, monkeypatch, checked, mode):
    from starstack import workflow
    pending, calls = [], []
    monkeypatch.setattr(window, "_run_async", lambda name, operation, callback: pending.append((name, operation)))
    monkeypatch.setattr(workflow, "submit_workflow", lambda *a, **kw: calls.append((a, kw)) or {"state": "queued"})
    window.stack_range_checkbox.setChecked(checked)
    window.start_processing()
    assert window._busy and not window.stack_range_checkbox.isEnabled()
    assert len(pending) == 1 and pending[0][0] == "submit"
    # A later programmatic UI change cannot alter the already submitted choice.
    window.stack_range_checkbox.setChecked(not checked)
    pending[0][1]()
    assert calls[0][1]["stack_range_mode"] == mode
    assert "support_mode" not in calls[0][1]
    assert "dcp" not in calls[0][1]


@pytest.mark.parametrize("mode,checked", [(None, False), ("legacy", False), ("positive_power2_v1", True), ("unknown", False)])
def test_gui_open_job_uses_only_its_recorded_mode(window, monkeypatch, tmp_path, mode, checked):
    from aester_desktop import gui
    job_dir = tmp_path / "old-job"
    job_dir.mkdir()
    job = {"state": "succeeded", "support_mode": "strict_siril_1_4_4", "artifacts": []}
    if mode is not None:
        job["stack_range_mode"] = mode
    path = job_dir / "job.json"
    path.write_text(json.dumps(job), encoding="utf8")
    before = path.read_bytes()
    window.stack_range_checkbox.setChecked(True)
    monkeypatch.setattr(gui, "load_job", lambda p: json.loads(Path(p, "job.json").read_text(encoding="utf8")))
    monkeypatch.setattr(window, "_run_async", lambda name, operation, callback: callback(operation()))
    window.load_existing_job(str(job_dir))
    assert window.stack_range_checkbox.isChecked() is checked
    assert not window._busy
    assert path.read_bytes() == before
