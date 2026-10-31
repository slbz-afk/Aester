"""Siril version probes with a private environment (efficiency P1b W4), using fake subprocesses.

siril.doctor and components._validate run ``siril-cli -v``; it exits during option parsing, but the executable may
be any Siril 1.4.x found, a desktop one included. Both now pass the same Windows keys as a script run
(XDG_DATA_HOME in a throw-away folder, MSYSTEM, DBUS_SESSION_BUS_ADDRESS); components keeps a stdlib twin of
siril.siril_environment because the installer runs it in a bare Python. The real-Siril probe test is
tests/test_siril_launch.py::test_version_probe_never_reaches_the_venv.
"""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import siril_guard
from starstack import components, siril

KEYS = ("XDG_DATA_HOME", "MSYSTEM", "DBUS_SESSION_BUS_ADDRESS")


@pytest.fixture
def executable(tmp_path):
    path = tmp_path / "引擎 with spaces" / "siril-cli.exe"
    path.parent.mkdir()
    path.write_bytes(b"test placeholder, never executed")
    return path


@pytest.mark.parametrize("windows", [True, False])
def test_the_components_twin_matches_siril_environment(tmp_path, windows):
    base = {"PATH": "C:\\x", "XDG_DATA_HOME": "C:\\user", "MSYSTEM": "MINGW64",
            "DBUS_SESSION_BUS_ADDRESS": "autolaunch:", "OMP_NUM_THREADS": "3"}
    original = dict(base)
    twin = components._siril_probe_environment(tmp_path, base, windows=windows)
    product = siril.siril_environment(tmp_path, None, base, windows=windows)
    assert base == original
    assert {k: twin.get(k) for k in KEYS} == {k: product.get(k) for k in KEYS}
    if windows:
        assert twin == {**base, "XDG_DATA_HOME": os.path.abspath(tmp_path), "MSYSTEM": "AESTER",
                        "DBUS_SESSION_BUS_ADDRESS": "disabled:"}
    else:
        assert twin == base


def test_doctor_probes_in_a_private_folder_and_keeps_its_answer(monkeypatch, executable):
    monkeypatch.setenv("XDG_DATA_HOME", str(executable.parent / "desktop"))
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    calls = []

    def version(args, **kwargs):
        env = kwargs["env"]
        calls.append({"args": args, "env": dict(env), "probe_existed": Path(env["XDG_DATA_HOME"]).is_dir()})
        return SimpleNamespace(returncode=0, stdout=b"siril-cli 1.4.4\n", stderr=b"")

    before = dict(os.environ)
    monkeypatch.setattr(siril.subprocess, "run", version)
    assert siril.doctor(str(executable)) == {"available": True, "executable": str(executable.resolve()),
                                             "version": "1.4.4", "message": "Siril 已就绪"}
    assert dict(os.environ) == before
    (call,) = calls
    assert call["args"] == [str(executable.resolve()), "-v"]
    probe = Path(call["env"]["XDG_DATA_HOME"])
    if os.name == "nt":
        assert call["probe_existed"] and not probe.exists() and probe.name.startswith("aester-siril-probe-")
        assert os.path.normcase(str(probe)) != siril_guard.REAL_DATA
        assert call["env"]["MSYSTEM"] == "AESTER" and call["env"]["DBUS_SESSION_BUS_ADDRESS"] == "disabled:"
    assert call["env"] == {**siril.siril_environment(probe), "XDG_DATA_HOME": call["env"]["XDG_DATA_HOME"]}


def test_doctor_reports_a_failed_probe_folder_like_any_start_failure(monkeypatch, executable):
    def unavailable(*args, **kwargs):
        raise OSError("no temporary folder")

    monkeypatch.setattr(siril.tempfile, "TemporaryDirectory", unavailable)
    result = siril.doctor(str(executable))
    assert result == {"available": False, "message": "no temporary folder", "executable": str(executable),
                      "version": None}


def test_component_validation_probes_siril_in_a_private_folder(tmp_path, monkeypatch, executable):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "desktop"))
    calls = []

    def run(args, **kwargs):
        env = kwargs.get("env")
        calls.append({"args": args, "env": None if env is None else dict(env),
                      "probe_existed": env is not None and Path(env["XDG_DATA_HOME"]).is_dir()})
        output = b"siril-cli 1.4.4\n" if args[-1] == "-v" else b""
        return subprocess.CompletedProcess(args, 0, output, b"")

    monkeypatch.setattr(components.subprocess, "run", run)
    runtime = tmp_path / "runtime"
    components._validate(runtime, executable, None)
    python, probe = calls
    assert python["args"][0] == str(runtime / "python.exe") and python["env"] is None      # unchanged
    assert probe["args"] == [str(executable), "-v"]
    folder = Path(probe["env"]["XDG_DATA_HOME"])
    assert probe["env"] == components._siril_probe_environment(folder)
    if os.name == "nt":
        assert probe["probe_existed"] and not folder.exists() and folder.name.startswith("aester-siril-probe-")
        assert {k: probe["env"][k] for k in KEYS[1:]} == {"MSYSTEM": "AESTER", "DBUS_SESSION_BUS_ADDRESS": "disabled:"}


def test_component_validation_still_rejects_another_version(tmp_path, monkeypatch, executable):
    monkeypatch.setattr(components.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(
        args, 0, b"siril-cli 1.4.3\n" if args[-1] == "-v" else b"", b""))
    with pytest.raises(ValueError, match="Unexpected Siril version"):
        components._validate(tmp_path / "runtime", executable, None)
