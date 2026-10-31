"""Workspace boundary checks; no real RAW decoding, crop writing, or MCP IO."""

import json
import sys
from types import SimpleNamespace

import pytest

from starstack import raw_audit
from starstack.mcp_server import AccessDenied, WorkspaceTools


@pytest.fixture
def raw_workspace(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    outside = tmp_path / "inputs-other"
    for path in (inputs, outputs, outside):
        path.mkdir()
    source = inputs / "银河 001.DNG"
    source.write_bytes(b"controlled source, do not decode")
    secret = outside / "outside.dng"
    secret.write_bytes(b"must not reach a backend")
    directory = outputs / "raw-audit-controlled"
    directory.mkdir()
    artifact = directory / "raw_codes.npy"
    artifact.write_bytes(b"controlled artifact, do not load")
    audit = {
        "schema_version": 1, "state": "succeeded", "audit_dir": str(directory),
        "source": {"path": str(source), "sha256": "controlled-source-sha"},
        "artifacts": {"raw_codes": {"path": str(artifact), "sha256": "controlled-artifact-sha"}},
    }
    manifest = directory / "audit.json"
    manifest.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    return SimpleNamespace(inputs=inputs, outputs=outputs, outside=outside,
                           source=source, secret=secret, directory=directory,
                           artifact=artifact, manifest=manifest, audit=audit,
                           service=WorkspaceTools(inputs, outputs))


def save_audit(workspace):
    workspace.manifest.write_text(json.dumps(workspace.audit), encoding="utf-8")


@pytest.mark.parametrize('field',['owner','source','artifact'])
def test_relative_persisted_paths_cannot_change_meaning_in_backend(raw_workspace,backend_calls,field):
    ws=raw_workspace
    if field=='owner': ws.audit['audit_dir']=ws.directory.name
    elif field=='source': ws.audit['source']['path']=ws.source.name
    else: ws.audit['artifacts']['raw_codes']['path']=ws.artifact.name
    save_audit(ws)
    with pytest.raises(AccessDenied):
        ws.service.protect_trace(str(ws.directory),[0,0,2,2])
    assert not backend_calls


@pytest.fixture
def backend_calls(monkeypatch):
    calls = []

    def inspect(source, output):
        calls.append(("inspect", source, output))
        return {"state": "controlled", "backend": "inspect"}

    def candidate(directory, output, **parameters):
        calls.append(("candidate", directory, output, parameters))
        return {"state": "controlled", "backend": "candidate"}

    monkeypatch.setattr(raw_audit, "inspect_raw", inspect)
    # The optional module is imported by the adapter at call time. Use a scoped
    # test double so no product crop-writing code can run in boundary tests.
    monkeypatch.setitem(sys.modules, "starstack.transients", SimpleNamespace(record_candidate=candidate))
    return calls


@pytest.mark.parametrize("relative", ["../inputs-other/outside.dng", "nested/../../outside.dng", "", "bad\x00.dng"])
def test_inspect_raw_rejects_invalid_input_before_backend(raw_workspace, backend_calls, relative):
    with pytest.raises(AccessDenied):
        raw_workspace.service.inspect_raw(relative)
    assert backend_calls == []


def test_inspect_raw_rejects_absolute_sibling_with_same_path_prefix(raw_workspace, backend_calls):
    with pytest.raises(AccessDenied):
        raw_workspace.service.inspect_raw(str(raw_workspace.secret))
    assert backend_calls == []


@pytest.mark.parametrize("kind", ["directory", "unsupported_suffix", "missing"])
def test_inspect_raw_requires_existing_raw_file(raw_workspace, backend_calls, kind):
    path = raw_workspace.inputs / ("not_raw.jpg" if kind == "unsupported_suffix" else "test.dng")
    if kind == "directory":
        path.mkdir()
    elif kind == "unsupported_suffix":
        path.write_bytes(b"not RAW")
    exception = FileNotFoundError if kind == "missing" else ValueError
    with pytest.raises(exception):
        raw_workspace.service.inspect_raw(str(path))
    assert backend_calls == []


@pytest.mark.parametrize("absolute", [False, True])
def test_inspect_raw_forwards_resolved_source_and_host_output(raw_workspace, backend_calls, absolute):
    before = raw_workspace.source.read_bytes()
    value = str(raw_workspace.source) if absolute else raw_workspace.source.name
    result = raw_workspace.service.inspect_raw(value)
    assert result == {"state": "controlled", "backend": "inspect"}
    assert backend_calls == [("inspect", raw_workspace.source, raw_workspace.outputs)]
    assert raw_workspace.source.read_bytes() == before


@pytest.mark.parametrize("absolute", [False, True])
def test_protect_trace_forwards_parameters_and_host_output_without_rewriting_audit(raw_workspace, backend_calls, absolute):
    before = raw_workspace.manifest.read_bytes()
    value = str(raw_workspace.directory) if absolute else raw_workspace.directory.name
    result = raw_workspace.service.protect_trace(value, [5, 6, 20, 30], "unknown", "亮线候选，未分类")
    assert result == {"state": "controlled", "backend": "candidate"}
    assert backend_calls == [("candidate", raw_workspace.directory, raw_workspace.outputs,
                             {"rectangle": [5, 6, 20, 30], "classification": "unknown",
                              "note": "亮线候选，未分类"})]
    assert raw_workspace.manifest.read_bytes() == before


@pytest.mark.parametrize("kind", ["outside", "traversal", "wrong_prefix", "nested"])
def test_protect_trace_requires_top_level_owned_audit(raw_workspace, backend_calls, kind):
    if kind == "outside":
        value = str(raw_workspace.outside)
    elif kind == "traversal":
        value = "../inputs-other"
    elif kind == "wrong_prefix":
        directory = raw_workspace.outputs / "job-controlled"
        directory.mkdir()
        value = str(directory)
    else:
        directory = raw_workspace.outputs / "nested" / "raw-audit-controlled"
        directory.mkdir(parents=True)
        value = str(directory)
    with pytest.raises(AccessDenied):
        raw_workspace.service.protect_trace(value, [0, 0, 1, 1])
    assert backend_calls == []


@pytest.mark.parametrize("kind", ["absolute_outside", "traversal", "missing_path"])
def test_protect_trace_rejects_forged_source_outside_input_root(raw_workspace, backend_calls, kind):
    if kind == "absolute_outside":
        raw_workspace.audit["source"]["path"] = str(raw_workspace.secret)
    elif kind == "traversal":
        raw_workspace.audit["source"]["path"] = "../inputs-other/outside.dng"
    else:
        raw_workspace.audit["source"] = {}
    save_audit(raw_workspace)
    with pytest.raises(AccessDenied):
        raw_workspace.service.protect_trace(raw_workspace.directory.name, [0, 0, 1, 1])
    assert backend_calls == []


@pytest.mark.parametrize("kind", ["outside", "other_audit", "traversal", "source_file", "missing_path"])
def test_protect_trace_rejects_artifact_paths_outside_this_audit(raw_workspace, backend_calls, kind):
    artifact = raw_workspace.audit["artifacts"]["raw_codes"]
    if kind == "outside":
        artifact["path"] = str(raw_workspace.secret)
    elif kind == "other_audit":
        other = raw_workspace.outputs / "raw-audit-other"
        other.mkdir()
        path = other / "raw_codes.npy"
        path.write_bytes(b"different audit")
        artifact["path"] = str(path)
    elif kind == "traversal":
        artifact["path"] = "../raw_codes.npy"
    elif kind == "source_file":
        artifact["path"] = str(raw_workspace.source)
    else:
        del artifact["path"]
    save_audit(raw_workspace)
    with pytest.raises(AccessDenied):
        raw_workspace.service.protect_trace(raw_workspace.directory.name, [0, 0, 1, 1])
    assert backend_calls == []


@pytest.mark.parametrize("kind", ["outside", "other_owned_audit", "missing_owner"])
def test_protect_trace_rejects_forged_manifest_owner(raw_workspace, backend_calls, kind):
    if kind == "outside":
        raw_workspace.audit["audit_dir"] = str(raw_workspace.outside)
    elif kind == "other_owned_audit":
        other = raw_workspace.outputs / "raw-audit-other"
        other.mkdir()
        raw_workspace.audit["audit_dir"] = str(other)
    else:
        del raw_workspace.audit["audit_dir"]
    save_audit(raw_workspace)
    with pytest.raises(AccessDenied):
        raw_workspace.service.protect_trace(raw_workspace.directory.name, [0, 0, 1, 1])
    assert backend_calls == []


def test_protect_trace_checks_every_artifact_not_only_raw_codes(raw_workspace, backend_calls):
    raw_workspace.audit["artifacts"]["black_subtracted_cfa"] = {"path": str(raw_workspace.secret)}
    save_audit(raw_workspace)
    with pytest.raises(AccessDenied):
        raw_workspace.service.protect_trace(raw_workspace.directory.name, [0, 0, 1, 1])
    assert backend_calls == []


def test_protect_trace_rejects_oversized_manifest_before_parsing_or_backend(raw_workspace, backend_calls):
    raw_workspace.manifest.write_bytes(b" " * (2 * 1024 ** 2 + 1))
    with pytest.raises(ValueError, match="过大"):
        raw_workspace.service.protect_trace(raw_workspace.directory.name, [0, 0, 1, 1])
    assert backend_calls == []


def test_protect_trace_rejects_missing_artifact_before_backend(raw_workspace, backend_calls):
    raw_workspace.artifact.unlink()
    with pytest.raises(FileNotFoundError):
        raw_workspace.service.protect_trace(raw_workspace.directory.name, [0, 0, 1, 1])
    assert backend_calls == []
