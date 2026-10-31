"""Installer boundary checks; network streams are synthetic, archives are real."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
from unittest.mock import Mock

import pytest
from starstack import components as c


def item(body=b"component-bytes"):
    return {"filename": "component.zip", "url": "https://files.pythonhosted.org/component.zip",
            "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers, self.url = status, headers or {}, "https://files.pythonhosted.org/component.zip"


def opener(monkeypatch, responses):
    mock = Mock()
    mock.open.side_effect = responses
    monkeypatch.setattr(c.urllib.request, "build_opener", lambda *a: mock)
    monkeypatch.setattr(c.time, "sleep", lambda *a: None)
    return mock


def test_verified_download_cache_never_requests_network(tmp_path, monkeypatch):
    data = item(); (tmp_path / data["filename"]).write_bytes(b"component-bytes")
    call = opener(monkeypatch, [AssertionError("offline")])
    assert c._download(data, tmp_path).read_bytes() == b"component-bytes"
    call.open.assert_not_called()


def test_resume_range_matches_existing_prefix(tmp_path, monkeypatch):
    body = b"1234567890"; data = item(body)
    (tmp_path / "component.zip.part").write_bytes(body[:4])
    call = opener(monkeypatch, [Response(body[4:], 206, {"Content-Range": "bytes 4-9/10"})])
    assert c._download(data, tmp_path).read_bytes() == body
    assert call.open.call_args.args[0].get_header("Range") == "bytes=4-"


def test_ignored_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    body = b"1234567890"
    (tmp_path / "component.zip.part").write_bytes(body[:4])
    opener(monkeypatch, [Response(body)])
    assert c._download(item(body), tmp_path).read_bytes() == body


@pytest.mark.parametrize("header", ["bytes 0-9/10", "bytes 4-8/10", "bytes 4-9/*", "bytes 4-9/11", ""])
def test_wrong_range_never_publishes(tmp_path, monkeypatch, header):
    (tmp_path / "component.zip.part").write_bytes(b"1234")
    opener(monkeypatch, [Response(b"567890", 206, {"Content-Range": header}) for _ in range(3)])
    with pytest.raises(ValueError): c._download(item(b"1234567890"), tmp_path)
    assert not (tmp_path / "component.zip").exists()
    assert (tmp_path / "component.zip.part").read_bytes() == b"1234"


def test_complete_corrupt_body_never_claims_success(tmp_path, monkeypatch):
    data = item(b"1234567890")
    opener(monkeypatch, [Response(b"xxxxxxxxxx") for _ in range(3)])
    with pytest.raises(ValueError): c._download(data, tmp_path)
    assert not (tmp_path / "component.zip").exists()


def test_partial_disconnect_retains_bytes_for_retry(tmp_path, monkeypatch):
    opener(monkeypatch, [Response(b"1234"), Response(b"567890", 206, {"Content-Range": "bytes 4-9/10"})])
    assert c._download(item(b"1234567890"), tmp_path).read_bytes() == b"1234567890"


def test_complete_partial_reuses_without_network(tmp_path, monkeypatch):
    body = b"1234567890"; (tmp_path / "component.zip.part").write_bytes(body)
    call = opener(monkeypatch, [])
    assert c._download(item(body), tmp_path).read_bytes() == body
    call.open.assert_not_called()


@pytest.mark.parametrize("filename", ["../outside", "C:evil", "a/b", "a\\b", "", ".hidden", "file. "])
def test_download_names_cannot_escape(tmp_path, filename):
    with pytest.raises(ValueError): c._download({**item(), "filename": filename}, tmp_path)


@pytest.mark.parametrize("url", ["http://files.pythonhosted.org/a", "https://evil.test/a", "https://user:pw@files.pythonhosted.org/a"])
def test_unapproved_origins_fail_before_network(tmp_path, url):
    with pytest.raises(ValueError): c._download({**item(), "url": url}, tmp_path)


def test_redirect_off_allowlist_is_rejected():
    with pytest.raises(ValueError):
        c._HTTPSRedirect().redirect_request(None,None,302,"",{},"https://evil.test/payload")
    with pytest.raises(ValueError):  # a host of another family is off this family's list (2026-10-09)
        c._HTTPSRedirect(c.family_hosts("pypi")).redirect_request(None,None,302,"",{},"https://www.python.org/ftp/python/x.zip")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "CON.txt", "a./x", "b /x"])
def test_archive_traversal_refused(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive,"w") as z:
        entry=zipfile.ZipInfo("placeholder")
        entry.filename=name  # Windows ZipInfo normalizes slashes at construction.
        z.writestr(entry,b"bad")
    with pytest.raises(ValueError): c._extract(archive,tmp_path / "destination")


def test_raw_archive_backslash_parser_rejects():
    # Windows zipfile itself canonicalizes a backslash to / when reading ZIPs.
    with pytest.raises(ValueError): c._archive_parts("a\\b")


def test_real_wheel_extract_keeps_license_and_data(tmp_path):
    archive=tmp_path / "wheel.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("example/__init__.py",b"x=1")
        z.writestr("example-1.dist-info/licenses/LICENSE",b"notice")
        z.writestr("example-1.data/scripts/example",b"entry")
    dest=tmp_path / "runtime";c._extract(archive,dest,wheel=True)
    assert (dest / "Lib/site-packages/example-1.dist-info/licenses/LICENSE").read_bytes()==b"notice"
    assert (dest / "Scripts/example").read_bytes()==b"entry"


def install_fixture(tmp_path, monkeypatch):
    py=tmp_path / "python.zip";engine=tmp_path / "siril.zip"
    with zipfile.ZipFile(py,"w") as z:
        for name in ["python.exe","pythonw.exe","python313._pth"]: z.writestr(name,b"test")
    with zipfile.ZipFile(engine,"w") as z: z.writestr("Siril/bin/siril-cli.exe",b"test")
    data={"python":{"kind":"python"},"wheels":[],"optional_wheels":[],"siril":{"kind":"siril"},
          "nox":{"filename":"nox.onnx","sha256":hashlib.sha256(b"model").hexdigest(),"size":5,"adapter":{"layout":"nhwc"}}}
    monkeypatch.setattr(c,"_download",lambda item,*a: py if item["kind"]=="python" else engine)
    monkeypatch.setattr(c,"_validate",lambda *a:None)
    return data


def test_install_activates_only_complete_version_and_preserves_old(tmp_path, monkeypatch):
    data=install_fixture(tmp_path,monkeypatch);root=tmp_path / "components"
    old=c.install_components(root,app_root=tmp_path / "app1",manifest=data)
    marker=(root / "current.json").read_bytes()
    def fail(*args): raise RuntimeError("validation failed")
    monkeypatch.setattr(c,"_validate",fail)
    with pytest.raises(RuntimeError): c.install_components(root,app_root=tmp_path / "app2",manifest=data)
    assert (root / "current.json").read_bytes()==marker
    assert Path(old["python"]).exists()
    assert list((root / "staging").glob("*/failure.json"))


def test_installer_preparation_does_not_activate(tmp_path, monkeypatch):
    data=install_fixture(tmp_path,monkeypatch);root=tmp_path / "components"
    result=c.install_components(root,app_root=tmp_path / "app",manifest=data,activate=False)
    assert result["prepared"] and Path(result["python"]).is_file()
    assert not (root / "current.json").exists()
    assert not c.component_status(root)["ready"]


def test_missing_model_fails_before_base_download(tmp_path, monkeypatch):
    data=install_fixture(tmp_path,monkeypatch)
    call=Mock(side_effect=AssertionError("should not download"));monkeypatch.setattr(c,"_download",call)
    with pytest.raises(ValueError,match="nox"): c.install_components(tmp_path / "components",app_root=tmp_path / "app",include_star_support=True,manifest=data)
    call.assert_not_called()


def test_receipt_tampering_detected(tmp_path, monkeypatch):
    data=install_fixture(tmp_path,monkeypatch);root=tmp_path / "components"
    r=c.install_components(root,app_root=tmp_path / "app",manifest=data)
    p=Path(r["receipt_path"]);p.write_text(p.read_text()+" ")
    assert c.component_status(root)["components"][0]["status"]=="broken"


def test_reuse_receipt_cannot_execute_external_python(tmp_path, monkeypatch):
    data=install_fixture(tmp_path,monkeypatch);root=tmp_path / "components"
    r=c.install_components(root,app_root=tmp_path / "app",manifest=data)
    path=Path(r["receipt_path"]);record=json.loads(path.read_text());record["python"]=str(tmp_path / "evil.exe")
    path.write_text(json.dumps(record));call=Mock();monkeypatch.setattr(c,"_validate",call)
    with pytest.raises(ValueError):c.install_components(root,app_root=tmp_path / "app",manifest=data)
    call.assert_not_called()


def test_missing_runtime_repair_creates_new_version_without_moving_old(tmp_path, monkeypatch):
    data=install_fixture(tmp_path,monkeypatch);root=tmp_path / "components"
    old=c.install_components(root,app_root=tmp_path / "app",manifest=data)
    old_receipt=Path(old["receipt_path"]);unchanged=old_receipt.read_bytes()
    Path(old["python"]).unlink()
    new=c.install_components(root,app_root=tmp_path / "app",manifest=data)
    assert new["ready"] and "-repair" in new["version"] and new["version"]!=old["version"]
    assert old_receipt.read_bytes()==unchanged and old_receipt.parent.exists()
    same=c.install_components(root,app_root=tmp_path / "app",manifest=data)
    assert same["version"]==new["version"]


def test_lock_excludes_second_writer_and_releases(tmp_path):
    root=tmp_path / "components"
    with c._lock(root):
        with pytest.raises(RuntimeError):
            with c._lock(root): pass
    with c._lock(root): pass


def test_catalog_pins_nox_cpu_without_activating_features():
    catalog=c.component_catalog()
    assert [x["id"] for x in catalog["components"]]==["base","nox"]
    assert catalog["components"][1]["required"] is False
    assert c._manifest()["nox"]["adapter"]["channel_order"]=="bgr"


def test_component_root_override_does_not_touch_global_environment(monkeypatch,tmp_path):
    monkeypatch.setenv("AESTER_COMPONENT_ROOT",str(tmp_path / "components"))
    assert c.default_component_root()==tmp_path / "components"
