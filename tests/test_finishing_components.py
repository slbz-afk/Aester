"""Component discovery/probe simulation. Never execute a GIMP or fixture binary."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from starstack import finishing_components as components

_READ_REGISTRY = components._registry_locations


@pytest.fixture(autouse=True)
def prevent_real_discovery_and_execution(monkeypatch):
    monkeypatch.delenv("AESTER_GIMP", raising=False)
    monkeypatch.setattr(components, "_registry_locations", lambda: ([], []))
    monkeypatch.setattr(components, "_standard_locations", lambda: [])

    def forbidden(*args, **kwargs):
        pytest.fail("This test must not start a real executable")
    monkeypatch.setattr(components, "guarded_process", forbidden)


@pytest.fixture
def executable(tmp_path):
    directory = tmp_path / "用户选定 GIMP"
    directory.mkdir()
    path = directory / "gimp-console-3.2.exe"
    path.write_bytes(b"Not a PE executable; test fixture must never run")
    return path


def _install_candidate(tmp_path, name="gimp-console-3.2.exe"):
    root = tmp_path / "external-install"
    (root / "bin").mkdir(parents=True)
    path = root / "bin" / name
    path.write_bytes(b"Non-executable fixture")
    return root, path


def test_default_discovery_does_not_execute_or_certify_operators(executable):
    before = executable.read_bytes()
    result = components.discover_gimp(executable)
    assert result["status"] == "available"
    assert result["path_exists"] is True
    assert result["version_checked"] is False
    assert result["version_compatible"] is None
    assert result["operator_verified"] is False
    assert result["selected_path"] == str(executable)
    assert result["ownership"] == "external_unmanaged"
    assert result["managed_by_aester"] is False
    assert executable.read_bytes() == before
    json.dumps(result, allow_nan=False)


def test_explicit_path_preserves_choice_over_environment_and_discovery(executable, tmp_path, monkeypatch):
    monkeypatch.setenv("AESTER_GIMP", str(tmp_path / "different.exe"))
    monkeypatch.setattr(components, "_automatic_candidates", lambda: pytest.fail("Pinned selection must not be replaced"))
    before = os.environ.copy()
    result = components.discover_gimp(executable)
    assert result["selection_source"] == "explicit"
    assert result["selection_locked"] is True
    assert result["selected_path"] == str(executable)
    assert os.environ == before


def test_missing_explicit_choice_is_not_silently_repaired(executable, tmp_path, monkeypatch):
    missing = tmp_path / "missing.exe"
    monkeypatch.setenv("AESTER_GIMP", str(executable))
    monkeypatch.setattr(components, "_automatic_candidates", lambda: pytest.fail("Must not auto-repair preference"))
    result = components.discover_gimp(missing, probe=True)
    assert result["status"] == "selected_missing"
    assert result["selected_path"] == str(missing)
    assert result["version_checked"] is False


@pytest.mark.parametrize("exists", [True, False])
def test_environment_choice_is_pinned(executable, tmp_path, monkeypatch, exists):
    choice = executable if exists else tmp_path / "missing.exe"
    monkeypatch.setenv("AESTER_GIMP", str(choice))
    monkeypatch.setattr(components, "_automatic_candidates", lambda: pytest.fail("Environment choice is authoritative"))
    result = components.discover_gimp()
    assert result["selected_path"] == str(choice)
    assert result["selection_source"] == "environment"
    assert result["selection_locked"] is True
    assert result["status"] == ("available" if exists else "selected_missing")


@pytest.mark.parametrize("choice", ["relative.exe", "", True, 3, b"binary.exe", "C:\\bad\x00.exe"])
def test_invalid_selection_is_a_json_diagnostic_not_execution(choice):
    result = components.discover_gimp(choice, probe=True)
    assert result["status"] == "check_failed"
    assert result["version_checked"] is False
    assert result["selection_locked"] is True
    json.dumps(result, allow_nan=False)


def test_parent_traversal_and_scripts_are_rejected(tmp_path):
    for choice in (tmp_path / "child" / ".." / "gimp.exe", tmp_path / "gimp.cmd"):
        result = components.discover_gimp(choice, probe=True)
        assert result["status"] == "check_failed"


def test_no_installation_reports_not_found_without_writes(tmp_path):
    before = list(tmp_path.iterdir())
    result = components.discover_gimp()
    assert result["status"] == "not_found"
    assert result["selected_path"] is None
    assert result["install_url"] == "https://www.gimp.org/downloads/"
    assert list(tmp_path.iterdir()) == before


def test_registry_hint_does_not_certify_version_or_claim_ownership(tmp_path, monkeypatch):
    root, path = _install_candidate(tmp_path)
    monkeypatch.setattr(components, "_registry_locations", lambda: ([(str(root), "HKCU official-install key", "3.2.6")], []))
    result = components.discover_gimp()
    assert result["status"] == "available"
    assert result["selection_source"] == "registry"
    assert result["selected_path"] == str(path)
    assert result["candidates"][0]["registry_version_hint"] == "3.2.6"
    assert result["version"] is None and result["version_compatible"] is None
    assert result["managed_by_aester"] is False


def test_standard_paths_are_bounded_and_deduplicated(tmp_path, monkeypatch):
    root, path = _install_candidate(tmp_path)
    monkeypatch.setattr(components, "_standard_locations", lambda: [str(root), str(root)])
    monkeypatch.setattr(Path, "rglob", lambda *a: pytest.fail("No recursive scan"))
    monkeypatch.setattr(Path, "glob", lambda *a: pytest.fail("No directory glob scan"))
    monkeypatch.setattr(os, "walk", lambda *a: pytest.fail("No disk walk"))
    result = components.discover_gimp()
    assert result["selected_path"] == str(path)
    assert result["selection_source"] == "standard"
    assert len(result["candidates"]) == 3


def test_unreadable_discovery_is_not_reported_as_definite_absence(monkeypatch):
    monkeypatch.setattr(components, "_registry_locations", lambda: ([], [
        {"code": "registry_check_failed", "message": "Access denied"}]))
    result = components.discover_gimp()
    assert result["status"] == "check_failed"
    assert result["selected_path"] is None


@pytest.mark.skipif(os.name != "nt", reason="Windows registry adapter")
def test_registry_reads_only_fixed_keys_and_never_commands(tmp_path, monkeypatch):
    calls = []
    registry = SimpleNamespace(HKEY_CURRENT_USER=1, HKEY_LOCAL_MACHINE=2,
        KEY_WOW64_64KEY=256, KEY_WOW64_32KEY=512, KEY_READ=1)

    @contextmanager
    def open_key(hive, key, reserved, flags):
        calls.append((hive, key, flags))
        yield key

    def value(key, name):
        assert name in {"DisplayName", "DisplayVersion", "InstallLocation"}
        return {"DisplayName": "GIMP 3.2.6", "DisplayVersion": "3.2.6",
                "InstallLocation": str(tmp_path / "registered")} [name], 1

    registry.OpenKey, registry.QueryValueEx = open_key, value
    monkeypatch.setitem(sys.modules, "winreg", registry)
    locations, diagnostics = _READ_REGISTRY()
    assert len(calls) == 8 and len(locations) == 8
    assert diagnostics == []
    assert {item[1] for item in calls} == set(components._UNINSTALL_KEYS)


def _fake_probe_process(monkeypatch, *, stdout=b"GNU Image Manipulation Program version 3.2.6\n",
                        stderr=b"", code=0, running=False):
    captured = {"killed": False, "closed": False}

    class Process:
        returncode = code

        def poll(self):
            return None if running else code

        def kill(self):
            captured["killed"] = True

    @contextmanager
    def guarded(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        kwargs["stdout"].write(stdout)
        kwargs["stdout"].flush()
        kwargs["stderr"].write(stderr)
        kwargs["stderr"].flush()
        try:
            yield Process()
        finally:
            captured["closed"] = True

    monkeypatch.setattr(components, "guard_kind", lambda: components.GUARD_JOB_OBJECT)
    monkeypatch.setattr(components, "guarded_process", guarded)
    return captured


def test_opt_in_probe_uses_only_absolute_version_command(executable, monkeypatch):
    captured = _fake_probe_process(monkeypatch)
    result = components.discover_gimp(executable, probe=True)
    assert result["status"] == "available"
    assert result["version_checked"] is True
    assert result["version"] == "3.2.6" and result["version_compatible"] is True
    assert result["operator_verified"] is False
    assert captured["command"] == [str(executable), "--version"]
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert captured["kwargs"]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert captured["closed"]


@pytest.mark.parametrize("version", ["2.10.38", "3.0.8", "3.2.4", "3.2.8", "3.2.6-RC1", "4.0.0"])
def test_other_versions_are_not_labeled_compatible(executable, monkeypatch, version):
    _fake_probe_process(monkeypatch, stdout=f"GIMP version {version}\n".encode())
    result = components.discover_gimp(executable, probe=True)
    assert result["status"] == "incompatible_version"
    assert result["version"] == version and result["version_compatible"] is False
    assert result["operator_verified"] is False


@pytest.mark.parametrize("output,version", [
    ("GNU 图像处理程序 版本 3.2.6\r\n".encode("utf-8"), "3.2.6"),   # Chinese Windows without LANG (2026-09-27)
    ("GNU 画像処理プログラム バージョン 3.0.4\n".encode("utf-8"), "3.0.4"),
    (b"GNU Bildbearbeitungsprogramm Version 3.2.6\n", "3.2.6")])
def test_localized_single_line_version_output_is_read(executable, monkeypatch, output, version):
    _fake_probe_process(monkeypatch, stdout=output)
    result = components.discover_gimp(executable, probe=True)
    assert result["version"] == version
    assert result["status"] == ("available" if version == "3.2.6" else "incompatible_version")


@pytest.mark.parametrize("output", [b"Python 3.2.6\n", b"3.2.6\n", b"GIMP 3.2.6\nGIMP 3.2.8\n", b"",
                                    "GNU 图像处理程序 版本 3.2.6 3.2.7\n".encode("utf-8"),
                                    "GNU 图像处理程序\n版本 3.2.6\n".encode("utf-8")])
def test_unknown_or_ambiguous_version_output_fails(executable, monkeypatch, output):
    _fake_probe_process(monkeypatch, stdout=output)
    result = components.discover_gimp(executable, probe=True)
    assert result["status"] == "check_failed"
    assert result["version_checked"] is False and result["version_compatible"] is None


def test_nonzero_version_process_is_not_success(executable, monkeypatch):
    captured = _fake_probe_process(monkeypatch, code=7, stderr=b"broken runtime")
    result = components.discover_gimp(executable, probe=True)
    assert result["status"] == "check_failed"
    assert "status 7" in result["diagnostics"][-1]["message"]
    assert captured["closed"]


def test_timeout_requests_owned_tree_kill(executable, monkeypatch):
    captured = _fake_probe_process(monkeypatch, running=True)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(components.time, "monotonic", lambda: next(clock))
    result = components.discover_gimp(executable, probe=True, timeout=0.1)
    assert result["status"] == "check_failed"
    assert "timed out" in result["diagnostics"][-1]["message"]
    assert captured["killed"] and captured["closed"]


def test_oversized_version_output_fails_closed(executable, monkeypatch):
    _fake_probe_process(monkeypatch, stdout=b"x" * (components._OUTPUT_LIMIT + 1))
    result = components.discover_gimp(executable, probe=True)
    assert result["status"] == "check_failed"
    assert "size budget" in result["diagnostics"][-1]["message"]


def test_probe_without_windows_job_never_spawns(executable, monkeypatch):
    monkeypatch.setattr(components, "guard_kind", lambda: "none")
    result = components.discover_gimp(executable, probe=True)
    assert result["status"] == "check_failed"
    assert result["path_exists"] is True


@pytest.mark.parametrize("options", [{"probe": 1}, {"timeout": True}, {"timeout": 0},
                                    {"timeout": 16}, {"timeout": float("nan")}, {"timeout": float("inf")}])
def test_invalid_probe_options_do_not_execute(executable, options):
    result = components.discover_gimp(executable, **options)
    assert result["status"] == "check_failed"
    assert result["diagnostics"][0]["code"] == "invalid_check_options"
    json.dumps(result, allow_nan=False)
