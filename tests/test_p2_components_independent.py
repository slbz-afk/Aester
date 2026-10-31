"""Independent network/archive/publication probes; no real downloads or executables."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import urllib.request
import zipfile

import pytest

from starstack import components as c


def identity(body, filename="component.zip", url="https://www.python.org/component.zip"):
    return {"filename": filename, "size": len(body), "sha256": hashlib.sha256(body).hexdigest(), "url": url}


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None, url="https://www.python.org/component.zip"):
        super().__init__(body)
        self.status, self.headers, self.url = status, headers or {}, url


def responses(monkeypatch, factory):
    requests = []
    class Opener:
        def open(self, request, **kwargs):
            requests.append(request)
            return factory(len(requests), request)
    monkeypatch.setattr(c.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(c.time, "sleep", lambda *args: None)
    return requests


def test_exact_range_resume_needs_matching_size_and_digest(tmp_path, monkeypatch):
    body = b"0123456789abcdef"
    spec = identity(body)
    (tmp_path / "component.zip.part").write_bytes(body[:5])
    calls = responses(monkeypatch, lambda *args: Response(body[5:], 206, {"Content-Range": "bytes 5-15/16"}))
    result = c._download(spec, tmp_path)
    assert calls[0].get_header("Range") == "bytes=5-"
    assert result.read_bytes() == body
    assert not (tmp_path / "component.zip.part").exists()
    c._download(spec, tmp_path)
    assert len(calls) == 1, "cached body must be validated before avoiding a network request"


def test_ignored_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    body = b"abcdefghij"
    (tmp_path / "component.zip.part").write_bytes(b"xxxxx")
    calls = responses(monkeypatch, lambda *args: Response(body))
    assert c._download(identity(body), tmp_path).read_bytes() == body
    assert len(calls) == 1


@pytest.mark.parametrize("content_range", ["bytes 0-15/16", "bytes 5-14/16", "bytes 5-15/18", "bytes 5-15/*"])
def test_mismatched_ranges_never_modify_existing_partial(tmp_path, monkeypatch, content_range):
    body = b"0123456789abcdef"
    partial = tmp_path / "component.zip.part"
    partial.write_bytes(body[:5])
    responses(monkeypatch, lambda *args: Response(body[5:], 206, {"Content-Range": content_range}))
    with pytest.raises(ValueError, match="range"):
        c._download(identity(body), tmp_path)
    assert partial.read_bytes() == body[:5]
    assert not (tmp_path / "component.zip").exists()


def test_short_body_is_retained_then_resumed_and_verified(tmp_path, monkeypatch):
    body = b"0123456789abcdef"
    calls = responses(monkeypatch, lambda i, request: Response(body[:7]) if i == 1 else Response(body[7:], 206, {"Content-Range": "bytes 7-15/16"}))
    assert c._download(identity(body), tmp_path).read_bytes() == body
    assert calls[1].get_header("Range") == "bytes=7-"


def test_wrong_hash_preserves_old_final_and_never_publishes_bad_partial(tmp_path, monkeypatch):
    final = tmp_path / "component.zip"
    final.write_bytes(b"old data, kept while repairing")
    original = final.read_bytes()
    responses(monkeypatch, lambda *args: Response(b"bad-bytes"))
    with pytest.raises(ValueError, match="SHA256"):
        c._download(identity(b"correct!!"), tmp_path)
    assert final.read_bytes() == original
    assert not (tmp_path / "component.zip.part").exists()


def test_server_cannot_fill_disk_beyond_pinned_download_size(tmp_path, monkeypatch):
    responses(monkeypatch, lambda *args: Response(b"x" * 2048))
    with pytest.raises(ValueError, match="exceeds pinned size"):
        c._download(identity(b"bounded"), tmp_path)
    assert (tmp_path / "component.zip.part").stat().st_size <= len(b"bounded")
    assert not (tmp_path / "component.zip").exists()


@pytest.mark.parametrize("url", ["http://www.python.org/file", "https://unapproved.invalid/file", "https://www.python.org.evil.invalid/file",
                                 "https://files.pythonhosted.org/file", "http://cdn.npmmirror.com/binaries/python/file"])
def test_redirect_downgrade_or_host_escape_is_rejected(url):
    request = urllib.request.Request("https://www.python.org/file")
    with pytest.raises(ValueError):
        c._HTTPSRedirect(c.family_hosts("python")).redirect_request(request, None, 302, "", {}, url)


@pytest.mark.parametrize("url", ["https://user:password@www.python.org/file", "https://@www.python.org/file",
                                  "https://www.python.org:444/file", "https://www.python.org:invalid/file",
                                  "https://user@cdn.npmmirror.com/binaries/python/file"])
def test_redirect_userinfo_and_nonstandard_origin_port_are_rejected(url):
    request = urllib.request.Request("https://www.python.org/file")
    with pytest.raises(ValueError):
        c._HTTPSRedirect(c.family_hosts("python")).redirect_request(request, None, 302, "", {}, url)


def test_redirect_within_the_family_hosts_is_followed():
    request = urllib.request.Request("https://www.python.org/ftp/python/3.13.15/file.zip")
    for url in ("https://cdn.npmmirror.com/binaries/python/3.13.15/file.zip", "https://www.python.org:443/x"):
        follow = c._HTTPSRedirect(c.family_hosts("python")).redirect_request(request, None, 302, "", {}, url)
        assert follow.full_url == url


@pytest.mark.parametrize("url", ["https://user:password@www.python.org/file", "https://@www.python.org/file",
                                  "https://www.python.org:444/file", "https://www.python.org:invalid/file"])
def test_initial_url_uses_the_same_strict_origin_gate(tmp_path, monkeypatch, url):
    def network_must_not_run(*args):
        pytest.fail("invalid origin reached network opener")
    responses(monkeypatch, network_must_not_run)
    with pytest.raises(ValueError):
        c._download(identity(b"bounded", url=url), tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def archive(path, members):
    with zipfile.ZipFile(path, "w") as stream:
        for name, body in members:
            # ZipInfo.__init__ normalises backslashes on Windows; set the stored
            # name afterwards to exercise the actual adversarial archive bytes.
            if isinstance(name, str):
                raw_name = name
                name = zipfile.ZipInfo("placeholder")
                name.filename = raw_name
            stream.writestr(name, body)
    return path


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "C:/outside.txt", "a/../../outside.txt", "folder/NUL.txt", "folder/file. "])
def test_archive_paths_cannot_escape_or_use_windows_devices(tmp_path, name):
    source = archive(tmp_path / "bad.zip", [(name, b"bad")])
    destination = tmp_path / "private"
    with pytest.raises(ValueError):
        c._extract(source, destination)
    assert not (tmp_path / "outside.txt").exists()
    assert not any(p.is_file() for p in destination.rglob("*"))


def test_raw_backslash_archive_name_is_rejected_by_path_parser():
    # On Windows ZipFile normalises member backslashes while reading; that
    # canonical form remains inside the destination. Test our stricter parser directly.
    with pytest.raises(ValueError):
        c._archive_parts("a\\outside.txt")


def test_archive_symlink_rejected_before_link_payload_written(tmp_path):
    link = zipfile.ZipInfo("linked")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    source = archive(tmp_path / "bad.zip", [(link, b"../outside")])
    with pytest.raises(ValueError, match="links"):
        c._extract(source, tmp_path / "private")
    assert not (tmp_path / "private/linked").exists()


def test_rejected_destination_is_checked_before_any_directory_creation(tmp_path, monkeypatch):
    source = archive(tmp_path / "archive.zip", [("file.txt", b"content")])
    destination = tmp_path / "rejected"
    original = c._safe_root
    def reject(path):
        path = Path(path)
        if path == destination or path.is_relative_to(destination):
            raise ValueError("rejected destination fixture")
        return original(path)
    monkeypatch.setattr(c, "_safe_root", reject)
    with pytest.raises(ValueError, match="rejected destination"):
        c._extract(source, destination)
    assert not destination.exists()


def test_wheel_data_targets_are_bounded_to_the_private_runtime(tmp_path):
    source = archive(tmp_path / "sample.whl", [("sample/__init__.py", b"x=1"),
                     ("sample-1.data/purelib/pure/__init__.py", b"x=2"),
                     ("sample-1.data/scripts/tool.py", b"x=3")])
    runtime = tmp_path / "runtime"
    c._extract(source, runtime, wheel=True)
    assert (runtime / "Lib/site-packages/sample/__init__.py").read_bytes() == b"x=1"
    assert (runtime / "Lib/site-packages/pure/__init__.py").read_bytes() == b"x=2"
    assert (runtime / "Scripts/tool.py").read_bytes() == b"x=3"


@pytest.fixture
def installation(tmp_path, monkeypatch):
    app = tmp_path / "application"
    (app / "src/starstack").mkdir(parents=True)
    python = archive(tmp_path / "python.zip", [("python.exe", b"FAKE; DO NOT EXECUTE"),
             ("pythonw.exe", b"FAKE; DO NOT EXECUTE"), ("python313._pth", b"python313.zip\n.\n"), ("python313.zip", b"")])
    siril = archive(tmp_path / "siril.zip", [("bin/siril-cli.exe", b"FAKE; DO NOT EXECUTE")])
    files = {"python.zip": python, "siril.zip": siril}
    manifest = {"python": identity(python.read_bytes(), "python.zip"), "siril": identity(siril.read_bytes(), "siril.zip"),
                "wheels": [], "optional_wheels": [], "nox": {"size": 4, "sha256": hashlib.sha256(b"nox!").hexdigest(),
                "adapter": {"value_range": "-1..1", "channel_order": "rgb", "layout": "nhwc", "stitch": "crop"}}}
    monkeypatch.setattr(c, "_download", lambda item, *args: files[item["filename"]])
    validations = []
    monkeypatch.setattr(c, "_validate", lambda runtime, siril, model, *args: validations.append((runtime, siril, model)))
    root = tmp_path / "components"
    result = c.install_components(root, app_root=app, manifest=manifest)
    assert result["ready"]
    return root, app, manifest, result, validations


def test_default_install_is_base_only_and_pth_uses_current_application(installation):
    root, app, manifest, result, validations = installation
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert result["model_path"] is None and not result["star_backend_options"]
    assert receipt["app_root"] == str(app.absolute())
    pth = Path(result["python"]).with_name("python313._pth").read_text()
    assert str(app.absolute() / "src") in pth.splitlines()
    assert not any(item[2] for item in validations)
    assert not (root / "models").exists()


def test_offline_validation_failure_does_not_replace_active_version_or_user_files(installation, monkeypatch):
    root, app, manifest, result, validations = installation
    old_pointer = (root / "current.json").read_bytes()
    user_file = root.parent / "user-photo.nef"
    user_file.write_bytes(b"original")
    newer = {**manifest, "audit_revision": 2}
    def fail(*args):
        raise RuntimeError("synthetic validation failure")
    monkeypatch.setattr(c, "_validate", fail)
    with pytest.raises(RuntimeError, match="synthetic validation"):
        c.install_components(root, app_root=app, manifest=newer)
    assert (root / "current.json").read_bytes() == old_pointer
    assert Path(result["python"]).is_file()
    assert c.component_status(root)["ready"]
    assert user_file.read_bytes() == b"original"
    assert list((root / "staging").glob("*/failure.json"))


def test_pointer_publication_failure_preserves_previous_working_version(installation, monkeypatch):
    root, app, manifest, result, validations = installation
    before = (root / "current.json").read_bytes()
    original = c._atomic
    def fail_pointer(path, record):
        if path == root / "current.json":
            raise OSError("synthetic pointer publication failure")
        return original(path, record)
    monkeypatch.setattr(c, "_atomic", fail_pointer)
    with pytest.raises(OSError, match="pointer publication"):
        c.install_components(root, app_root=app, manifest={**manifest, "new_revision": 1})
    assert (root / "current.json").read_bytes() == before
    assert c.component_status(root)["ready"]
    assert Path(result["python"]).is_file()


def test_cached_receipt_cannot_send_root_external_executable_to_validator(installation, monkeypatch):
    root, app, manifest, result, validations = installation
    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    outside = root.parent / "unexpected"
    outside.mkdir()
    (outside / "python.exe").write_bytes(b"never execute")
    receipt["python"] = str(outside / "python.exe")
    receipt_path.write_text(json.dumps(receipt))
    before = (root / "current.json").read_bytes()
    def guard(runtime, *args):
        assert runtime.resolve().is_relative_to(root.resolve()), "unsafe executable path reached _validate"
    monkeypatch.setattr(c, "_validate", guard)
    with pytest.raises(ValueError):
        c.install_components(root, app_root=app, manifest=manifest)
    assert (root / "current.json").read_bytes() == before


def test_cached_nox_receipt_cannot_silently_change_the_pinned_adapter(installation, monkeypatch):
    root, app, manifest, base, validations = installation
    model = root.parent / "nox.onnx"
    model.write_bytes(b"nox!")
    result = c.install_components(root, app_root=app, manifest=manifest, model_source=model, include_star_support=True)
    receipt_path = Path(result["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["star_backend_options"]["value_range"] = "0..1"
    receipt_path.write_text(json.dumps(receipt))
    before = (root / "current.json").read_bytes()
    def guard(*args):
        pytest.fail("changed adapter reached runtime validator and could be republished")
    monkeypatch.setattr(c, "_validate", guard)
    with pytest.raises(ValueError):
        c.install_components(root, app_root=app, manifest=manifest, model_source=model, include_star_support=True)
    assert (root / "current.json").read_bytes() == before


def snapshot_tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["import", "timeout", "bad_executable"])
def test_same_version_repair_keeps_open_old_runtime_and_switches_only_after_validation(installation, monkeypatch, failure):
    root, app, manifest, old, validations = installation
    old_dir = Path(old["receipt_path"]).parent
    old_runtime = Path(old["python"]).parent
    before = snapshot_tree(old_dir)
    pointer_before = (root / "current.json").read_bytes()
    calls = []
    def validate(runtime, *args):
        calls.append(runtime)
        if runtime == old_runtime:
            if failure == "import":
                raise subprocess.CalledProcessError(1, [str(runtime / "python.exe")])
            if failure == "timeout":
                raise subprocess.TimeoutExpired([str(runtime / "python.exe")], 1)
            raise OSError("synthetic invalid executable format")
        assert (root / "current.json").read_bytes() == pointer_before
        assert snapshot_tree(old_dir) == before
    monkeypatch.setattr(c, "_validate", validate)
    with Path(old["python"]).open("rb") as running_file:
        result = c.install_components(root, app_root=app, manifest=manifest)
        assert running_file.read() == before[str(Path("runtime") / "python.exe")]
    assert result["ready"] and result["version"] != old["version"]
    assert "-repair" in result["version"]
    assert snapshot_tree(old_dir) == before
    assert json.loads((root / "current.json").read_text())["version"] == result["version"]
    assert Path(result["python"]).parent != old_runtime
    assert len(calls) == 2
    monkeypatch.setattr(c, "_validate", lambda *args: None)
    monkeypatch.setattr(c, "_download", lambda *args: pytest.fail("validated repair should be reusable"))
    reused = c.install_components(root, app_root=app, manifest=manifest)
    assert reused["version"] == result["version"]


def test_missing_file_repair_does_not_restore_or_replace_files_inside_old_directory(installation):
    root, app, manifest, old, validations = installation
    old_dir = Path(old["receipt_path"]).parent
    Path(old["pythonw"]).unlink()
    damaged_tree = snapshot_tree(old_dir)
    result = c.install_components(root, app_root=app, manifest=manifest)
    assert result["ready"] and result["version"] != old["version"]
    assert not Path(old["pythonw"]).exists()
    assert snapshot_tree(old_dir) == damaged_tree
    assert Path(result["pythonw"]).is_file()


def test_failed_repair_leaves_old_pointer_and_open_runtime_intact(installation, monkeypatch):
    root, app, manifest, old, validations = installation
    old_dir = Path(old["receipt_path"]).parent
    old_runtime = Path(old["python"]).parent
    before = snapshot_tree(old_dir)
    pointer_before = (root / "current.json").read_bytes()
    def validate(runtime, *args):
        if runtime == old_runtime:
            raise subprocess.CalledProcessError(1, [str(runtime / "python.exe")])
        raise RuntimeError("new runtime cannot validate")
    monkeypatch.setattr(c, "_validate", validate)
    with Path(old["python"]).open("rb") as running_file:
        with pytest.raises(RuntimeError, match="new runtime cannot validate"):
            c.install_components(root, app_root=app, manifest=manifest)
        assert running_file.read() == before[str(Path("runtime") / "python.exe")]
    assert (root / "current.json").read_bytes() == pointer_before
    assert snapshot_tree(old_dir) == before


def test_prepared_repair_can_be_validated_without_switching_active_pointer(installation):
    root, app, manifest, old, validations = installation
    Path(old["pythonw"]).unlink()
    before = (root / "current.json").read_bytes()
    prepared = c.install_components(root, app_root=app, manifest=manifest, activate=False)
    assert prepared["prepared"] and "-repair" in prepared["version"]
    assert Path(prepared["pythonw"]).is_file()
    assert (root / "current.json").read_bytes() == before


def test_truncated_current_json_repair_reuses_verified_owned_version(installation):
    root, app, manifest, old, validations = installation
    old_tree = snapshot_tree(Path(old["receipt_path"]).parent)
    (root / "current.json").write_text('{"version":')
    result = c.install_components(root, app_root=app, manifest=manifest)
    assert result["ready"] and result["version"] == old["version"]
    assert snapshot_tree(Path(old["receipt_path"]).parent) == old_tree


def worker_activation(installation, monkeypatch):
    root, app, manifest, old, validations = installation
    next_app = root.parent / "installation" / "versions" / "new-application"
    (next_app / "src").mkdir(parents=True)
    prepared = c.install_components(root, app_root=next_app, manifest=manifest, activate=False)
    prepared_path = root.parent / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))
    state_path = next_app.parent.parent / "install-state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "app_root": str(app), "component_root": str(root),
             "receipt_path": old["receipt_path"], "receipt_sha256": c._digest(Path(old["receipt_path"]))}))
    monkeypatch.setitem(sys.modules, "components", c)
    from starstack import install_ownership, install_lifecycle
    monkeypatch.setitem(sys.modules, "install_ownership", install_ownership)
    monkeypatch.setitem(sys.modules, "install_lifecycle", install_lifecycle)
    spec = importlib.util.spec_from_file_location("aester_installer_worker_audit", Path(__file__).resolve().parents[1] / "packaging/aester_p2/installer_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    monkeypatch.setattr(sys, "argv", ["installer_worker", "--component-root", str(root), "--app-root", str(next_app),
                                   "--result", str(prepared_path), "--activate", str(state_path)])
    return worker, root, old, prepared, state_path


def test_installer_activation_can_replace_a_truncated_old_current_json(installation, monkeypatch):
    worker, root, old, prepared, state_path = worker_activation(installation, monkeypatch)
    (root / "current.json").write_text('{"version":')
    assert worker.main() == 0
    assert json.loads((root / "current.json").read_text())["version"] == prepared["version"]
    assert json.loads(state_path.read_text())["receipt_path"] == prepared["receipt_path"]
    assert Path(old["python"]).is_file()


def test_installer_activation_second_pointer_failure_rolls_back_both_original_records(installation, monkeypatch):
    worker, root, old, prepared, state_path = worker_activation(installation, monkeypatch)
    before_state = state_path.read_bytes()
    before_pointer = (root / "current.json").read_bytes()
    original = c._atomic
    def fail(path, record):
        if path == root / "current.json" and record.get("version") == prepared["version"]:
            raise OSError("synthetic second pointer write failure")
        original(path, record)
    monkeypatch.setattr(c, "_atomic", fail)
    assert worker.main() == 1
    assert state_path.read_bytes() == before_state
    assert (root / "current.json").read_bytes() == before_pointer
    assert c.component_status(root)["version"] == old["version"]


def test_installer_failed_app_import_does_not_change_any_active_record(installation, monkeypatch):
    worker, root, old, prepared, state_path = worker_activation(installation, monkeypatch)
    before_state = state_path.read_bytes()
    before_pointer = (root / "current.json").read_bytes()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output=b"", stderr=b"synthetic import failure")
    monkeypatch.setattr(worker.subprocess, "run", fail)
    assert worker.main() == 1
    assert state_path.read_bytes() == before_state
    assert (root / "current.json").read_bytes() == before_pointer
