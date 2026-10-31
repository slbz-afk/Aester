"""Deletion decisions use real temp files; native settings use an isolated adapter."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import subprocess

import pytest

from starstack import uninstall_options as u
from starstack.locking import file_lock


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def entry(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


@pytest.fixture
def job(tmp_path):
    cache, works, inputs = [tmp_path / name for name in ("cache", "works", "raw")]
    for p in (cache, works, inputs):
        p.mkdir()
    source = inputs / "one.NEF"
    source.write_bytes(b"untouched original")
    directory = cache / "workflow-a123"
    child = directory / "stack/job-012345"
    products = child / "products/abcdef"
    products.mkdir(parents=True)
    generated = []
    for name in ("master_scene_linear.tif", "natural_display_16bit.tif", "natural.jpg"):
        p = products / name
        p.write_bytes((name + " test pixels").encode())
        generated.append(p)
    write_json(child / "job.json", {"state": "succeeded", "job_dir": str(child)})
    delivery = works / "aester-123/delivery-abcdef"
    delivery.mkdir(parents=True)
    for name in ("master_scene_linear.tif", "final_srgb16.tif", "final.jpg"):
        (delivery / name).write_bytes(("publication " + name).encode())
    request = {"input_files": [str(source)], "delivery_root": str(works)}
    data = {"job": {"job_dir": str(child)}, "colour": {"master": {"path": str(generated[0])},
            "photo": {"tiff": str(generated[1]), "jpeg": str(generated[2])}}}
    record = {"version": "workflow-1", "workflow_dir": str(directory), "state": "succeeded",
              "request": request, "signature": digest(request), "stack_job_dir": str(child),
              "delivery_dir": str(delivery), "checkpoints": {"stack": {"data": data,
                    "data_sha256": digest(data), "files": [entry(p) for p in generated]}}}
    receipt = {"version": "aester-delivery-1", "workflow_dir": str(directory),
               "workflow_signature": record["signature"], "files": [
                   {"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in delivery.iterdir()]}
    write_json(delivery / "delivery.json", receipt)
    write_json(directory / "workflow.json", record)
    unknown = directory / "my_notes.txt"
    unknown.write_bytes(b"user must keep this")
    return {"cache": cache, "works": works, "source": source, "directory": directory,
            "child": child, "products": products, "generated": generated,
            "delivery": delivery, "record": record, "receipt": receipt, "unknown": unknown}


def plan(j):
    return u.prepare_cache_cleanup(j["directory"], cache_root=j["cache"])


def update(j):
    write_json(j["directory"] / "workflow.json", j["record"])


def refresh_checkpoint(j):
    checkpoint = j["record"]["checkpoints"]["stack"]
    checkpoint["data_sha256"] = digest(checkpoint["data"])
    update(j)


def test_partial_cleanup_deletes_only_proven_images_and_keeps_all_control_and_user_files(job):
    original = job["source"].read_bytes()
    photos = {p: p.read_bytes() for p in job["delivery"].iterdir()}
    before_record = (job["directory"] / "workflow.json").read_bytes()
    proposed = plan(job)
    assert len(proposed["files"]) == 3 and proposed["bytes"] > 0
    assert "my_notes.txt" in proposed["retained_cache_files"]
    result = u.execute_cache_cleanup(proposed, exclusive=True)
    assert result["state"] == "completed" and not result["errors"]
    assert result["removed_bytes"] == proposed["bytes"]
    assert all(not p.exists() for p in job["generated"])
    assert job["unknown"].read_bytes() == b"user must keep this"
    assert (job["directory"] / "workflow.json").read_bytes() == before_record
    assert job["source"].read_bytes() == original
    assert {p: p.read_bytes() for p in job["delivery"].iterdir()} == photos
    marker = json.loads((job["directory"] / u.CLEANUP_MARKER).read_bytes())
    assert marker == result and marker["loses_workflow_resume_and_review"]
    with pytest.raises(ValueError, match="already"):
        plan(job)


def test_no_execution_without_lifecycle_exclusion(job):
    with pytest.raises(ValueError, match="exclusion"):
        u.execute_cache_cleanup(plan(job))
    assert all(p.exists() for p in job["generated"])


@pytest.mark.parametrize("mutation", ["photo", "generated", "record", "add_unknown", "failed", "busy"])
def test_changed_or_active_job_after_planning_preserves_all_cache_before_first_delete(job, mutation):
    proposed = plan(job)
    if mutation == "photo":
        (job["delivery"] / "final.jpg").write_bytes(b"edited photograph")
    elif mutation == "generated":
        job["generated"][0].write_bytes(b"changed cache")
    elif mutation == "record":
        job["record"]["message"] = "new record"
        update(job)
    elif mutation == "add_unknown":
        (job["directory"] / "extra.txt").write_text("keep")
    elif mutation == "failed":
        job["record"]["state"] = "failed"
        update(job)
    if mutation == "busy":
        with file_lock(job["directory"] / "worker.lock"):
            with pytest.raises(ValueError, match="active"):
                u.execute_cache_cleanup(proposed, exclusive=True)
    else:
        with pytest.raises(ValueError):
            u.execute_cache_cleanup(proposed, exclusive=True)
    assert all(p.exists() for p in job["generated"])
    assert not (job["directory"] / u.CLEANUP_MARKER).exists()


@pytest.mark.parametrize("state", ["failed", "running", "queued", "cancelled"])
def test_incomplete_jobs_are_never_cleaned(job, state):
    job["record"]["state"] = state
    update(job)
    with pytest.raises(ValueError):
        plan(job)


@pytest.mark.parametrize("kind", ["same_cache", "embedded_works", "original_inside", "missing_receipt", "missing_output_hash", "checkpoint_changed", "output_original", "output_outside", "untracked_field"])
def test_fail_closed_boundaries(job, kind):
    checkpoint = job["record"]["checkpoints"]["stack"]
    if kind == "same_cache":
        with pytest.raises(ValueError):
            u.prepare_cache_cleanup(job["directory"], cache_root=job["directory"])
        return
    if kind == "embedded_works":
        job["record"]["delivery_dir"] = str(job["products"])
    elif kind == "original_inside":
        job["record"]["request"]["input_files"] = [str(job["generated"][0])]
        job["record"]["signature"] = digest(job["record"]["request"])
    elif kind == "missing_receipt":
        (job["delivery"] / "delivery.json").unlink()
    elif kind == "missing_output_hash":
        checkpoint["files"] = []
    elif kind == "checkpoint_changed":
        checkpoint["data"]["injected"] = "changed"
    elif kind in {"output_original", "output_outside"}:
        target = job["source"] if kind == "output_original" else job["delivery"] / "final.jpg"
        checkpoint["data"]["colour"]["master"]["path"] = str(target)
        checkpoint["files"].append(entry(target))
        checkpoint["data_sha256"] = digest(checkpoint["data"])
    elif kind == "untracked_field":
        checkpoint["data"] = {"random": {"path": str(job["generated"][0])}}
        checkpoint["data_sha256"] = digest(checkpoint["data"])
    update(job)
    with pytest.raises(ValueError):
        plan(job)


def test_copied_raw_and_unrecognized_files_are_preserved(job):
    copied_raw = job["directory"] / "saved.NEF"
    copied_raw.write_bytes(b"not selected for deletion")
    proposed = plan(job)
    assert "saved.NEF" in proposed["retained_cache_files"]
    u.execute_cache_cleanup(proposed, exclusive=True)
    assert copied_raw.read_bytes() == b"not selected for deletion"


def test_publication_root_cannot_hide_inside_a_sibling_cache(job):
    job["record"]["request"]["delivery_root"] = str(job["cache"] / "sibling")
    (job["cache"] / "sibling").mkdir()
    job["record"]["signature"] = digest(job["record"]["request"])
    update(job)
    with pytest.raises(ValueError, match="independent"):
        plan(job)


def test_child_snapshot_must_match_its_bound_stack(job):
    job["record"]["checkpoints"]["stack"]["data"]["job"]["job_dir"] = str(job["works"])
    refresh_checkpoint(job)
    with pytest.raises(ValueError, match="another child"):
        plan(job)


def test_hardlink_alias_of_a_candidate_is_rejected(job):
    duplicate = job["works"] / "aliased.tif"
    os.link(job["generated"][0], duplicate)
    with pytest.raises(ValueError, match="unlinked"):
        plan(job)
    assert duplicate.exists()


@pytest.mark.skipif(os.name != "nt", reason="actual Windows directory junction")
def test_unknown_link_anywhere_in_cache_rejects_whole_cleanup(job):
    link = job["directory"] / "user_link"
    run = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(job["works"])], capture_output=True)
    assert run.returncode == 0
    try:
        with pytest.raises(ValueError, match="reparse"):
            plan(job)
    finally:
        link.rmdir()


def test_image_only_receipts_live_in_cache_and_survive(job):
    receipt_dir = job["directory"] / "publication-receipts"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "delivery-abcdef.json"
    job["receipt"]["version"] = "aester-delivery-2"
    job["receipt"]["delivery_dir"] = str(job["delivery"])
    job["receipt"]["master_file"] = "master_scene_linear.tif"
    job["record"]["request"]["delivery_format"] = "aester-delivery-2"
    job["record"]["signature"] = digest(job["record"]["request"])
    job["receipt"]["workflow_signature"] = job["record"]["signature"]
    write_json(receipt_path, job["receipt"])
    (job["delivery"] / "delivery.json").unlink()
    job["record"]["publication_receipts"] = {str(job["delivery"]): entry(receipt_path)}
    update(job)
    proposed = plan(job)
    u.execute_cache_cleanup(proposed, exclusive=True)
    assert receipt_path.exists()
    assert len(list(job["delivery"].iterdir())) == 3


@pytest.mark.parametrize("extra", [None, "unknown_role", "undeclared"])
def test_image_only_receipt_accepts_only_declared_allowlisted_extras(job, extra):
    """2026-09-27: v2 deliveries may declare starless/star/trail layers beside the three images."""
    receipt_dir = job["directory"] / "publication-receipts"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "delivery-abcdef.json"
    (job["delivery"] / "delivery.json").unlink()
    layer = job["delivery"] / "starless_srgb16.tif"
    layer.write_bytes(b"publication starless")
    job["record"]["request"]["delivery_format"] = "aester-delivery-2"
    job["record"]["signature"] = digest(job["record"]["request"])
    files = [{"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in job["delivery"].iterdir()]
    receipt = dict(job["receipt"], version="aester-delivery-2", delivery_dir=str(job["delivery"]), files=files,
                   master_file="master_scene_linear.tif", workflow_signature=job["record"]["signature"],
                   extras=[{"file": "starless_srgb16.tif", "role": "trails_layer" if extra == "unknown_role" else "starless"}])
    if extra == "undeclared":
        receipt.pop("extras")
    write_json(receipt_path, receipt)
    job["record"]["publication_receipts"] = {str(job["delivery"]): entry(receipt_path)}
    update(job)
    if extra is None:
        u.execute_cache_cleanup(plan(job), exclusive=True)
        assert layer.exists() and len(list(job["delivery"].iterdir())) == 4
    else:
        with pytest.raises(ValueError):
            plan(job)
        assert all(p.exists() for p in job["generated"])


@pytest.mark.parametrize("kind", ["missing", "outside_receipt", "wrong_delivery", "two_images", "fake_legacy_fallback"])
def test_image_only_invalid_receipt_cannot_enable_deletion(job, kind):
    receipt_dir = job["directory"] / "publication-receipts"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "delivery-abcdef.json"
    job["record"]["request"]["delivery_format"] = "aester-delivery-2"
    job["record"]["signature"] = digest(job["record"]["request"])
    receipt = dict(job["receipt"], version="aester-delivery-2", delivery_dir=str(job["delivery"]),
                   master_file="master_scene_linear.tif", workflow_signature=job["record"]["signature"])
    if kind == "outside_receipt": receipt_path = job["works"] / "forged.json"
    if kind == "wrong_delivery": receipt["delivery_dir"] = str(job["works"])
    if kind == "two_images": receipt["files"] = receipt["files"][:2]
    write_json(receipt_path, receipt)
    if kind not in {"missing", "fake_legacy_fallback"}:
        job["record"]["publication_receipts"] = {str(job["delivery"]): entry(receipt_path)}
    update(job)
    with pytest.raises(ValueError):
        plan(job)
    assert all(p.exists() for p in job["generated"])


def test_all_base_and_reviewed_photographs_are_verified(job):
    job["record"]["base_delivery_dir"] = str(job["works"] / "missing-base")
    update(job)
    with pytest.raises(ValueError):
        plan(job)


def test_marker_precedes_first_delete_and_failure_is_recorded(job, monkeypatch):
    proposed = plan(job)
    original = Path.unlink
    calls = []
    def fail(path, *args, **kwargs):
        if path in job["generated"]:
            state = json.loads((job["directory"] / u.CLEANUP_MARKER).read_bytes())
            assert state["state"] == "cleanup_started"
            calls.append(path)
            if len(calls) == 2:
                raise PermissionError("synthetic locked file")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail)
    result = u.execute_cache_cleanup(proposed, exclusive=True)
    assert result["state"] == "partial" and len(result["removed_files"]) == 1
    assert "synthetic locked" in result["errors"][0]
    assert job["unknown"].exists()


class Key:
    def __init__(self, name): self.name = name
    def __enter__(self): return self
    def __exit__(self, *args): return False


class Registry:
    HKEY_CURRENT_USER, KEY_READ, KEY_SET_VALUE = "HKCU", 1, 2
    def __init__(self): self.values = {}; self.deleted = []
    def OpenKey(self, root, path, reserved, access):
        if not any(key[0] == path for key in self.values): raise FileNotFoundError(path)
        return Key(path)
    def QueryValueEx(self, handle, name):
        try: return self.values[(handle.name, name)]
        except KeyError: raise FileNotFoundError(name)
    def DeleteValue(self, handle, name):
        self.deleted.append((handle.name, name)); del self.values[(handle.name, name)]


@pytest.fixture
def registry(monkeypatch):
    reg = Registry()
    monkeypatch.setattr(u, "_registry", lambda: reg)
    for key in u.SETTINGS_KEYS:
        reg.values[u._setting_location(key)] = ("saved " + key, 1)
    reg.values[(u.SETTINGS_ROOT, "other_product")] = ("must keep", 1)
    return reg


def test_settings_exact_values_only_changes_preserved_no_tree_delete(registry):
    proposed = u.plan_settings_cleanup()
    assert "saved" not in json.dumps(proposed)
    changed = u._setting_location("appearance/theme")
    registry.values[changed] = ("new choice", 1)
    result = u.execute_settings_cleanup(proposed, exclusive=True)
    assert result["preserved_changed"] == ["appearance/theme"]
    assert len(result["removed"]) == len(u.SETTINGS_KEYS) - 1
    assert registry.values[changed] == ("new choice", 1)
    assert registry.values[(u.SETTINGS_ROOT, "other_product")] == ("must keep", 1)


@pytest.mark.parametrize("case", ["wrong_root", "unknown_key", "duplicate", "bad_hash", "no_exclusion"])
def test_settings_malformed_plan_never_deletes(registry, case):
    proposed = u.plan_settings_cleanup()
    if case == "wrong_root": proposed["root"] = "Software\\SomeoneElse"
    elif case == "unknown_key": proposed["values"][0]["key"] = "other_product"
    elif case == "duplicate": proposed["values"].append(dict(proposed["values"][0]))
    elif case == "bad_hash": proposed["values"][0]["sha256"] = "x"
    with pytest.raises(ValueError):
        u.execute_settings_cleanup(proposed, exclusive=case != "no_exclusion")
    assert not registry.deleted


def test_scan_defaults_off_and_real_verifiable_cache_is_nonzero(job, registry, monkeypatch):
    monkeypatch.setattr(u, "_known_cache_roots", lambda: [job["cache"]])
    result = u.scan_options(job["works"])
    assert len(result["cache_plans"]) == 1 and result["cache_plans"][0]["bytes"] > 0
    assert result["default_clear_safe_cache"] is False and result["default_clear_settings"] is False
    assert "不保证工作目录清空" in result["summary_zh"]


def test_scan_failed_and_unknown_jobs_are_reported_not_deleted(job, registry, monkeypatch):
    unknown = job["cache"] / "workflow-1234"
    unknown.mkdir(); (unknown / "private.txt").write_text("user file")
    job["record"]["state"] = "failed"; update(job)
    monkeypatch.setattr(u, "_known_cache_roots", lambda: [job["cache"]])
    result = u.scan_options(job["works"])
    assert not result["cache_plans"] and len(result["preserved"]) == 2
    assert (unknown / "private.txt").read_text() == "user file"


def test_module_can_import_as_loose_bootstrap_without_starstack_package():
    path = Path(u.__file__)
    spec = importlib.util.spec_from_file_location("uninstall_options_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PLAN_VERSION == u.PLAN_VERSION
