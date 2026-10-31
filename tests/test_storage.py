"""Durable deliverables vs cached computation; RAW/Siril itself is stubbed."""
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest
import numpy as np

from starstack import workflow as wf, workflow_edit as edit, storage
from starstack import master_export as me, photo_render as pr, display_transform as dt
from starstack.locking import file_lock
from test_workflow import setup, admit as base_admit
from test_workflow_review_lifecycle import calibrated_workflow


def admit(s, **kwargs):
    if kwargs.get("delivery_root") is not None:
        kwargs.setdefault("delivery_format", "aester-delivery-1")
    return base_admit(s, **kwargs)


def run_external(s, **kwargs):
    works = s["output"].parent / "Pictures" / "Aester"
    record = wf.submit_workflow(s["inputs"], s["output"], delivery_root=works, delivery_format="aester-delivery-1", **kwargs)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    return done, works


def external_calibrated(s, monkeypatch):
    submit = wf.submit_workflow
    works = s["output"].parent / "Pictures" / "Aester"
    monkeypatch.setattr(wf, "submit_workflow", lambda *a, **kw: submit(*a, delivery_root=works, delivery_format="aester-delivery-1", **kw))
    return calibrated_workflow(s, monkeypatch)


def test_legacy_signature_layout_remains_unchanged(setup):
    record = admit(setup, delivery_root=None)
    assert "delivery_root" not in record["request"]
    assert "delivery_root" not in record["config"]
    done = wf.run_workflow(record["workflow_dir"])
    assert Path(done["delivery_dir"]).parent == Path(done["workflow_dir"])
    assert not (Path(done["delivery_dir"]) / "delivery.json").exists()


def test_external_publish_freezes_path_and_sources_unchanged(setup):
    before = {str(p): wf.core._sha(p) for p in setup["inputs"].iterdir()}
    done, works = run_external(setup, idempotency_key="public")
    root = Path(done["workflow_dir"])
    assert done["config"]["delivery_root"] == str(works.resolve())
    assert done["request"]["delivery_root"] == str(works.resolve())
    assert Path(done["delivery_dir"]).parent == storage.publication_parent(done)
    assert Path(done["delivery_dir"]).is_relative_to(works)
    assert not list(root.glob("delivery-*"))
    assert all(Path(item["path"]).is_relative_to(root) for cp in done["checkpoints"].values() for item in cp["files"])
    assert {str(p): wf.core._sha(p) for p in setup["inputs"].iterdir()} == before
    assert storage.verify_delivery(done["delivery_dir"])["workflow_signature"] == done["signature"]
    wf._verify(done)
    repeated = wf.submit_workflow(setup["inputs"], setup["output"], delivery_root=works, delivery_format="aester-delivery-1", idempotency_key="public")
    assert repeated["delivery_dir"] == done["delivery_dir"]
    assert setup["calls"]["submits"] == 1
    with pytest.raises(ValueError, match="不同"):
        wf.submit_workflow(setup["inputs"], setup["output"], delivery_root=works.parent / "other", idempotency_key="public")


@pytest.mark.parametrize("where", ["same", "below", "above", "raw", "raw-below"])
def test_delivery_rejects_cache_overlap_and_raw_paths(setup, where):
    path = {"same": setup["output"], "below": setup["output"] / "photos", "above": setup["output"].parent,
            "raw": setup["inputs"], "raw-below": setup["inputs"] / "photos"}[where]
    with pytest.raises(ValueError):
        admit(setup, delivery_root=path)
    assert setup["calls"]["submits"] == 0


def test_two_workflows_do_not_share_delivery_ownership(setup):
    first, works = run_external(setup, idempotency_key="one")
    second, _ = run_external(setup, idempotency_key="two")
    assert storage.publication_parent(first) != storage.publication_parent(second)
    with pytest.raises(ValueError, match="不属于"):
        storage.delivery_file(first, second["artifacts"][0]["path"])
    forged = dict(first, artifacts=second["artifacts"])
    with pytest.raises(ValueError, match="不属于"):
        wf._verify(forged)


def test_copy_failure_has_no_published_delivery_and_resume_reuses_stack(setup, monkeypatch):
    copy = storage._copy_verified
    count = 0
    def broken(source, destination, check):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated full drive")
        return copy(source, destination, check)
    monkeypatch.setattr(storage, "_copy_verified", broken)
    works = setup["output"].parent / "works"
    record = admit(setup, delivery_root=works)
    failed = wf.run_workflow(record["workflow_dir"])
    assert failed["state"] == "failed" and "full drive" in failed["error"]
    assert not list(works.rglob("delivery-*"))
    assert list(works.rglob("failure.json"))
    monkeypatch.setattr(storage, "_copy_verified", copy)
    wf.resume_workflow(record["workflow_dir"])
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert setup["calls"]["submits"] == 1
    assert len(list(works.rglob("delivery-*"))) == 1


def test_cancel_during_external_copy_never_publishes_then_resumes(setup, monkeypatch):
    copy = storage._copy_verified
    works = setup["output"].parent / "works"
    record = admit(setup, delivery_root=works)
    def stop(source, destination, check):
        value = copy(source, destination, check)
        (Path(record["workflow_dir"]) / "cancel.request").touch()
        return value
    monkeypatch.setattr(storage, "_copy_verified", stop)
    stopped = wf.run_workflow(record["workflow_dir"])
    assert stopped["state"] == "cancelled"
    assert not list(works.rglob("delivery-*"))
    monkeypatch.setattr(storage, "_copy_verified", copy)
    wf.resume_workflow(record["workflow_dir"])
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded"
    assert setup["calls"]["submits"] == 1


def test_real_copy_detects_written_byte_corruption(setup, monkeypatch):
    source = setup["inputs"] / "1.NEF"
    target = setup["output"].parent / "copied.bin"
    sync = storage.os.fsync
    def corrupt(fd):
        sync(fd)
        with target.open("r+b") as handle:
            handle.write(b"broken")
    monkeypatch.setattr(storage.os, "fsync", corrupt)
    with pytest.raises(ValueError, match="复制校验"):
        storage._copy_verified(source, target, lambda: None)


def test_delivered_pixel_tampering_is_rejected(setup):
    done, _ = run_external(setup)
    (Path(done["delivery_dir"]) / "final.jpg").write_bytes(b"modified")
    with pytest.raises(ValueError, match="改变"):
        wf._verify(done)
    with pytest.raises(ValueError, match="改变"):
        storage.plan_cache_cleanup(done["workflow_dir"], cache_root=setup["output"])


def test_external_review_keeps_previous_files_and_reuses_same_namespace(setup, monkeypatch):
    done, session, state = external_calibrated(setup, monkeypatch)
    original = {item["path"]: item["sha256"] for item in done["artifacts"]}
    reviewed = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    current = reviewed["workflow"]
    assert Path(current["delivery_dir"]).parent == Path(done["delivery_dir"]).parent
    assert current["delivery_dir"] != done["delivery_dir"]
    assert current["base_delivery_dir"] == done["delivery_dir"]
    assert all(wf.core._sha(Path(path)) == sha for path, sha in original.items())
    base_master = Path(done["delivery_dir"]) / "master_scene_linear.tif"
    assert wf.core._sha(base_master) == wf.core._sha(Path(current["delivery_dir"]) / base_master.name)
    storage.verify_delivery(current["delivery_dir"])
    wf._verify(current)
    again = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    assert again["reused"] and state["applies"] == 1


def test_external_review_publication_failure_retries_without_second_apply(setup, monkeypatch):
    done, session, state = external_calibrated(setup, monkeypatch)
    original = storage.stage_delivery
    monkeypatch.setattr(storage, "stage_delivery", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline works")))
    with pytest.raises(OSError, match="offline works"):
        edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    assert wf.workflow_status(done["workflow_dir"])["delivery_dir"] == done["delivery_dir"]
    monkeypatch.setattr(storage, "stage_delivery", original)
    result = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    assert result["workflow"]["delivery_dir"] != done["delivery_dir"]
    assert state["applies"] == 1


def test_movable_master_recipe_and_render_do_not_need_cache(setup, monkeypatch):
    done, _, _ = external_calibrated(setup, monkeypatch)
    delivery = Path(done["delivery_dir"])
    copy = delivery.parent.parent / "moved-photographs"
    shutil.copytree(delivery, copy)
    cache = Path(done["workflow_dir"])
    cache.rename(cache.with_name(cache.name + "-temporarily-unavailable"))
    storage.verify_delivery(copy)
    recipe = json.loads((copy / "master_recipe.json").read_text(encoding="utf-8"))
    assert recipe["master"] == "master_scene_linear.tif"
    assert recipe["mask"] == "camera_saturation_mask.png"
    assert wf.core._sha(copy / recipe["master"]) == recipe["master_sha256"]
    assert wf.core._sha(copy / recipe["mask"]) == recipe["mask_sha256"]
    pixels, meta, _ = me.read_master(copy / recipe["master"])
    assert np.isfinite(pixels).all()
    del pixels
    frozen = json.loads((copy / "photo_render.json").read_text(encoding="utf-8"))
    params = dt.Params.from_json(frozen["metadata"]["display_transform"])
    result = pr.render_photo(copy / recipe["master"], copy.parent / "independent-render", params)
    assert Path(result["jpeg"]).is_file()


def test_cleanup_only_proposes_one_job_and_protects_works_originals(setup):
    done, _ = run_external(setup)
    root = Path(done["workflow_dir"])
    before = set(root.rglob("*"))
    plan = storage.plan_cache_cleanup(root, cache_root=setup["output"])
    assert not plan["executed"] and plan["requires_explicit_confirmation"]
    assert plan["delete_root"] == str(root)
    assert plan["preserved_deliveries"] == [done["delivery_dir"]]
    assert plan["preserved_originals"] == done["request"]["input_files"]
    assert set(root.rglob("*")) == before
    assert plan["bytes"] > 0 and plan["files"] > 0
    with file_lock(root / "edit.lock"):
        with pytest.raises(ValueError, match="正在使用"):
            storage.plan_cache_cleanup(root, cache_root=setup["output"])
    with pytest.raises(ValueError, match="单个工作流"):
        storage.plan_cache_cleanup(root, cache_root=root)


def test_symlink_directory_cannot_redirect_delivery(setup):
    alias = setup["output"].parent / "alias"
    real = setup["output"].parent / "target"
    real.mkdir()
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip("OS does not allow creating test directory symlink: " + str(exc))
    with pytest.raises(ValueError, match="软链接"):
        admit(setup, delivery_root=alias)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction safeguard")
def test_windows_junction_after_admission_cannot_redirect_delivery(setup):
    works = setup["output"].parent / "works"
    record = admit(setup, delivery_root=works)
    target = setup["output"].parent / "redirect-target"
    target.mkdir()
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(works), str(target)], capture_output=True)
    assert result.returncode == 0, result.stderr
    failed = wf.run_workflow(record["workflow_dir"])
    assert failed["state"] == "failed" and "软链接" in failed["error"]
    assert not list(target.iterdir())


def test_delivery_root_tampering_fails_before_resume(setup):
    record = admit(setup, delivery_root=setup["output"].parent / "works")
    directory = Path(record["workflow_dir"])
    record["state"] = "failed"
    record["request"]["delivery_root"] = str(setup["output"].parent / "changed")
    wf.atomic_json(directory / "workflow.json", record)
    with pytest.raises(ValueError, match="改变"):
        wf.resume_workflow(directory)
    assert setup["calls"]["submits"] == 0


def test_file_changed_after_copy_cannot_be_self_certified(setup, monkeypatch):
    copy = storage._copy_verified
    def corrupt_after_copy(source, destination, check):
        frozen = copy(source, destination, check)
        if destination.suffix == ".tif":
            with destination.open("r+b") as handle:
                handle.write(b"broken")
        return frozen
    monkeypatch.setattr(storage, "_copy_verified", corrupt_after_copy)
    record = admit(setup, delivery_root=setup["output"].parent / "works")
    failed = wf.run_workflow(record["workflow_dir"])
    assert failed["state"] == "failed" and "改变" in failed["error"]
    assert not list((setup["output"].parent / "works").rglob("delivery-*"))


def test_cloud_placeholder_attribute_is_not_mistaken_for_junction(tmp_path, monkeypatch):
    from types import SimpleNamespace
    original = Path.lstat
    def cloud(path):
        actual = original(path)
        return SimpleNamespace(st_mode=actual.st_mode, st_file_attributes=0x400,
                               st_reparse_tag=0x9000001A)
    monkeypatch.setattr(Path, "lstat", cloud)
    assert storage._no_links(tmp_path) == tmp_path
