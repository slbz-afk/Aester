"""Independent host lifecycle checks; no test launches GIMP.

The real-process cases substitute a fixed local Python launcher and exercise
the production Windows Job path. They do not certify GIMP or XCF replay.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import ImageCms
import pytest
import tifffile

from starstack import finishing_assets as assets
from starstack import finishing_backend as backend
from starstack import finishing_launch as launch


@pytest.fixture
def request_case(tmp_path):
    path = tmp_path / "原图 with spaces.tif"
    pixels = np.arange(12 * 8 * 3, dtype=np.uint16).reshape(8, 12, 3) * 173
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    source = assets.inspect_source(path, encoding="srgb16")
    recipe = {"schema": "aester-finishing-recipe-1",
              "source": assets.recipe_source(source), "intent": "保留气辉", "operations": []}
    work = tmp_path / "private run 空格"
    work.mkdir()
    return source, recipe, work


def _install_helper(tmp_path, monkeypatch, body):
    """Substitute the fixed launcher path only, retaining the real process guard."""
    directory = tmp_path / "fixed helper 空格"
    directory.mkdir()
    (directory / "finishing_gimp_script.py").write_text("# No GIMP import or execution.\n")
    (directory / "finishing_launch.py").write_text(body, encoding="utf-8")
    monkeypatch.setattr(backend, "__file__", str(directory / "finishing_backend.py"))


def _alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetExitCodeProcess.restype = ctypes.c_int32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        assert error == 87, f"Unexpected process-query error {error}"
        return False
    try:
        status = ctypes.c_uint32()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(status))
        return status.value == 259
    finally:
        kernel.CloseHandle(handle)


_CHILD = """
import os, pathlib, sys, time
work = pathlib.Path(sys.argv[1])
(work / 'child.pid').write_text(str(os.getpid()))
deadline = time.monotonic() + 10
with (work / 'heartbeat').open('ab', buffering=0) as stream:
    while time.monotonic() < deadline:
        stream.write(b'x')
        time.sleep(0.03)
"""
_TREE = """
import os, pathlib, subprocess, sys, time
work = pathlib.Path(sys.argv[2]).parent
(work / 'launcher.pid').write_text(str(os.getpid()))
child = subprocess.Popen([sys._base_executable, '-I', '-c', CHILD, str(work)])
try:
    child.wait(timeout=12)
finally:
    if child.poll() is None:
        child.kill()
    child.wait(timeout=2)
""".replace("CHILD", repr(_CHILD))


@pytest.mark.skipif(os.name != "nt", reason="NL-1 currently requires Windows Job ownership")
@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_real_launcher_descendants_stop_on_cancel_or_timeout(request_case, tmp_path, monkeypatch, reason):
    source, recipe, work = request_case
    _install_helper(tmp_path, monkeypatch, _TREE)
    callback = (lambda: (work / "heartbeat").is_file()) if reason == "cancel" else None
    expected = backend.RenderCancelled if reason == "cancel" else TimeoutError
    with pytest.raises(expected):
        backend.render_recipe(source, recipe, {}, work, gimp_path=sys.executable,
                              timeout=1.5, cancel=callback)
    # A live descendant's PID comes from the descendant itself, avoiding Windows
    # virtualenv redirector PID ambiguity. Bounded helpers are an extra failsafe.
    pids = [int((work / name).read_text()) for name in ("launcher.pid", "child.pid")]
    deadline = time.monotonic() + 3
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.03)
    assert not any(_alive(pid) for pid in pids)
    size = (work / "heartbeat").stat().st_size
    time.sleep(0.1)
    assert (work / "heartbeat").stat().st_size == size
    record = json.loads((work / "execution.json").read_text(encoding="utf-8"))
    assert record["status"] == ("cancelled" if reason == "cancel" else "failed")
    assert "products" not in record


@pytest.mark.skipif(os.name != "nt", reason="NL-1 currently requires Windows Job ownership")
def test_real_nonzero_launcher_never_reaches_product_validation(request_case, tmp_path, monkeypatch):
    source, recipe, work = request_case
    _install_helper(tmp_path, monkeypatch, "raise SystemExit(23)\n")
    monkeypatch.setattr(backend, "verify_products", lambda *a: pytest.fail("validated a failed process"))
    with pytest.raises(RuntimeError, match="status 23"):
        backend.render_recipe(source, recipe, {}, work, gimp_path=sys.executable)
    assert json.loads((work / "execution.json").read_text())["status"] == "failed"


@pytest.mark.skipif(os.name != "nt", reason="NL-1 currently requires Windows Job ownership")
def test_source_changed_during_launcher_is_not_published(request_case, tmp_path, monkeypatch):
    source, recipe, work = request_case
    _install_helper(tmp_path, monkeypatch, """
import json, pathlib, sys
request = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8'))
with open(request['source']['path'], 'ab') as stream:
    stream.write(b'changed while processing')
""")
    monkeypatch.setattr(backend, "verify_products", lambda *a: pytest.fail("validated a changed source"))
    with pytest.raises(ValueError):
        backend.render_recipe(source, recipe, {}, work, gimp_path=sys.executable)
    record = json.loads((work / "execution.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "products" not in record


@pytest.mark.skipif(os.name != "nt", reason="NL-1 currently requires Windows Job ownership")
def test_cancel_during_product_validation_records_cancelled(request_case, tmp_path, monkeypatch):
    source, recipe, work = request_case
    _install_helper(tmp_path, monkeypatch, "raise SystemExit(0)\n")
    state = {"cancelled": False}

    def finish_validation(*args):
        state["cancelled"] = True
        return {"test_stub": "Products decoded while cancellation arrived"}

    monkeypatch.setattr(backend, "verify_products", finish_validation)
    with pytest.raises(backend.RenderCancelled):
        backend.render_recipe(source, recipe, {}, work, gimp_path=sys.executable,
                              cancel=lambda: state["cancelled"])
    record = json.loads((work / "execution.json").read_text(encoding="utf-8"))
    assert record["status"] == "cancelled"
    assert "products" not in record


def test_environment_keeps_parent_unchanged_and_uses_private_paths(tmp_path, monkeypatch):
    contaminated = {"PYTHONPATH": "foreign-import", "PYTHONHOME": "foreign-python",
                    "GIMP3_DIRECTORY": "foreign-profile", "GI_TYPELIB_PATH": "foreign-gi",
                    "GEGL_PATH": "foreign-gegl", "BABL_PATH": "foreign-babl",
                    "GTK_PATH": "foreign-gtk", "GEGL_SWAP": "foreign-swap",
                    "GIO_MODULE_DIR": "foreign-modules", "GIO_EXTRA_MODULES": "extra-modules",
                    "GTK_MODULES": "foreign-gtk-module", "BABL_TOLERANCE": "0.9",
                    "GSETTINGS_SCHEMA_DIR": "foreign-settings", "XDG_CONFIG_HOME": "foreign-config",
                    "FONTCONFIG_FILE": "foreign-fonts", "AESTER_FINISHING_DRIVER": "foreign-driver"}
    for key, value in contaminated.items():
        monkeypatch.setenv(key, value)
    before = os.environ.copy()
    request = tmp_path / "request.json"
    env = launch.isolated_environment(tmp_path, request)
    assert os.environ == before
    for key in ("PYTHONPATH", "PYTHONHOME", "GI_TYPELIB_PATH", "GEGL_PATH", "BABL_PATH", "GTK_PATH",
                "GEGL_SWAP", "GIO_MODULE_DIR", "GIO_EXTRA_MODULES", "GTK_MODULES", "BABL_TOLERANCE",
                "GSETTINGS_SCHEMA_DIR"):
        assert key not in env
    for key in ("GIMP3_DIRECTORY", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "GIMP3_CACHEDIR",
                "GIMP3_TEMPDIR", "TEMP", "TMP"):
        assert Path(env[key]).is_relative_to(tmp_path)
        assert Path(env[key]).is_dir()
    assert env["AESTER_FINISHING_REQUEST"] == str(request)
    assert env["AESTER_FINISHING_DRIVER"] == str(Path(launch.__file__).with_name("finishing_gimp_script.py"))
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize("schema", ["aester-finishing-render-1", "aester-finishing-render-2"])
def test_launcher_uses_fixed_batch_expression_and_explicit_environment(tmp_path, monkeypatch, schema):
    request = tmp_path / "request 空格.json"
    request.write_text(json.dumps({"schema": schema,
                                  "intent": "'); malicious_text_is_data #"}), encoding="utf-8")
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self):
            return 17

    monkeypatch.setattr(launch.subprocess, "Popen", FakeProcess)
    assert launch.main([sys.executable, str(request)]) == 17
    assert isinstance(captured["command"], list)
    assert captured["command"][0] == str(Path(sys.executable).resolve())
    assert not any("malicious_text" in arg for arg in captured["command"])
    assert "--new-instance" in captured["command"]
    assert captured["env"]["AESTER_FINISHING_REQUEST"] == str(request.resolve())
    assert captured["stdin"] == launch.subprocess.DEVNULL
