"""Candidate publication/feedback tests; fake renders are not GIMP validation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from starstack import finishing_candidates as candidates
from starstack import finishing_session as sessions
from starstack.finishing_assets import inspect_source
from starstack.finishing_backend import RenderCancelled, verify_products
from starstack.finishing_recipe import recipe_digest


@pytest.fixture
def case(tmp_path):
    pixels = np.arange(8 * 12 * 3, dtype=np.uint16).reshape(8, 12, 3) * 149
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "source.tif"
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    return sessions.create_session(path, tmp_path, intent="保留气辉"), pixels, icc


def _products(source, recipe, directory, pixels, icc):
    directory.mkdir()
    tifffile.imwrite(directory / "result.tif", pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    shutil.copyfile(directory / "result.tif", directory / "replay.tif")
    Image.fromarray((pixels // 257).astype(np.uint8)).save(directory / "result.jpg", icc_profile=icc)
    # Only a magic-byte fixture. These tests deliberately do not assert XCF replay.
    (directory / "result.xcf").write_bytes(b"gimp xcf v013\x00fixture")
    receipt = {"schema": "aester-finishing-gimp-receipt-1", "ok": True,
               "source_sha256": source["sha256"], "recipe_sha256": recipe_digest(recipe),
               "width": source["width"], "height": source["height"], "gimp_version": "test fixture",
               "applied_operations": [op["id"] for op in recipe["operations"] if op["enabled"]]}
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return verify_products(directory, source, recipe)


def _fake_render(case, monkeypatch):
    record, pixels, icc = case
    def render(source, recipe, masks, work, **kwargs):
        # Lifecycle lock must remain held until publication.
        with pytest.raises(sessions.RevisionConflict):
            sessions.revise_recipe(record["directory"], recipe, expected_revision=record["revision"])
        assert sessions.read_session(record["directory"])["candidates"][-1]["status"] == "running"
        return {"status": "validated", "products": _products(source, recipe, work / "artifacts", pixels, icc)}
    monkeypatch.setattr(candidates, "render_recipe", render)


def test_candidates_bind_recipe_and_keep_previous(case, monkeypatch):
    _fake_render(case, monkeypatch)
    record = case[0]
    first = candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="host only")
    second = candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="host only")
    assert first["id"] != second["id"]
    assert first["status"] == second["status"] == "validated"
    assert first["recipe_sha256"] == record["revisions"][0]["recipe_sha256"]
    assert sessions.read_session(record["directory"])["candidates"] == [first, second]
    assert candidates.verified_candidate(record["directory"], first["id"]) == first


@pytest.mark.parametrize("error,status", [(TimeoutError("deadline"), "failed"),
    (RenderCancelled("cancel"), "cancelled"), (RuntimeError("plugin failed"), "failed")])
def test_failed_render_is_recorded_and_never_exportable(case, monkeypatch, error, status):
    record = case[0]
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(candidates, "render_recipe", fail)
    with pytest.raises(type(error)):
        candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="unused")
    current = sessions.read_session(record["directory"])
    item = current["candidates"][0]
    assert item["status"] == status and "execution" not in item
    assert current["revisions"] == record["revisions"]
    with pytest.raises(ValueError):
        candidates.export_candidate(record["directory"], item["id"], Path(record["directory"]).parent)


def test_cancel_after_render_cannot_publish(case, monkeypatch):
    _fake_render(case, monkeypatch)
    record = case[0]
    with pytest.raises(RenderCancelled):
        candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="unused", cancel=lambda: True)
    assert sessions.read_session(record["directory"])["candidates"][0]["status"] == "cancelled"


def test_feedback_targets_real_candidate_and_does_not_relabel_engineering(case, monkeypatch):
    _fake_render(case, monkeypatch)
    record = case[0]
    result = candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="unused")
    with pytest.raises(ValueError):
        candidates.candidate_feedback(record["directory"], candidate_id=result["id"],
            text="my approval", origin="agent", decision="accept")
    value = candidates.candidate_feedback(record["directory"], candidate_id=result["id"],
        text="仍太黄", origin="user", decision="reject")
    assert value["feedback"][-1]["scope"] == "candidate"
    assert value["feedback"][-1]["candidate_id"] == result["id"]
    assert value["candidates"][0]["status"] == "validated"


@pytest.mark.parametrize("kind", ["tiff", "jpeg", "xcf"])
def test_modified_product_cannot_be_accepted_or_exported(case, monkeypatch, kind):
    _fake_render(case, monkeypatch)
    record = case[0]
    result = candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="unused")
    path = Path(result["execution"]["products"][kind]["path"])
    with path.open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError):
        candidates.candidate_feedback(record["directory"], candidate_id=result["id"],
            text="accept", origin="user", decision="accept")
    with pytest.raises(ValueError):
        candidates.export_candidate(record["directory"], result["id"], path.parent)


def test_export_complete_set_has_own_name_and_matching_manifest(case, monkeypatch, tmp_path):
    _fake_render(case, monkeypatch)
    record = case[0]
    result = candidates.render_candidate(record["directory"], expected_revision=0, gimp_path="unused")
    one = candidates.export_candidate(record["directory"], result["id"], tmp_path)
    two = candidates.export_candidate(record["directory"], result["id"], tmp_path)
    assert one["directory"] != two["directory"]
    for exported in (one, two):
        root = Path(exported["directory"])
        assert {item.name for item in root.iterdir()} == {"photo.tif", "photo.jpg", "photo.xcf", "manifest.json"}
        assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == exported["manifest"]
        assert exported["manifest"]["recipe"] == record["revisions"][0]["recipe"]


def test_product_validator_rejects_identity_pixel_change(case, tmp_path):
    record, pixels, icc = case
    changed = pixels.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="Empty recipe"):
        _products(record["source"], record["revisions"][0]["recipe"], tmp_path / "wrong", changed, icc)


def test_receipt_not_accepted_for_different_recipe(case, tmp_path):
    record, pixels, icc = case
    directory = tmp_path / "artifacts"
    recipe = record["revisions"][0]["recipe"]
    _products(record["source"], recipe, directory, pixels, icc)
    changed = deepcopy(recipe)
    changed["intent"] = "different intention"
    with pytest.raises(ValueError, match="receipt"):
        verify_products(directory, record["source"], changed)


def test_corrupt_jpeg_does_not_pass_on_exit_and_tiff_alone(case, tmp_path):
    record, pixels, icc = case
    directory = tmp_path / "artifacts"
    recipe = record["revisions"][0]["recipe"]
    _products(record["source"], recipe, directory, pixels, icc)
    (directory / "result.jpg").write_bytes(b"not an image")
    with pytest.raises(Exception):
        verify_products(directory, record["source"], recipe)


def test_xcf_replay_pixel_change_prevents_validation(case, tmp_path):
    record, pixels, icc = case
    directory = tmp_path / "artifacts"
    recipe = record["revisions"][0]["recipe"]
    _products(record["source"], recipe, directory, pixels, icc)
    changed = pixels.copy()
    changed[-1, -1, -1] ^= 1
    tifffile.imwrite(directory / "replay.tif", changed, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    with pytest.raises(ValueError, match="XCF replay"):
        verify_products(directory, record["source"], recipe)


def test_stale_revision_does_not_start_render(case, monkeypatch):
    record = case[0]
    def forbidden(*args, **kwargs):
        pytest.fail("Should not render stale revision")
    monkeypatch.setattr(candidates, "render_recipe", forbidden)
    with pytest.raises(sessions.RevisionConflict):
        candidates.render_candidate(record["directory"], expected_revision=1, gimp_path="unused")
    assert sessions.read_session(record["directory"])["candidates"] == []


def test_recovery_marks_abandoned_only_after_owner_lock(case):
    record = case[0]
    path = Path(record["directory"]) / "session.json"
    record["candidates"] = [{"id": "candidate-" + "a" * 32, "status": "running"}]
    sessions._save(path, record)
    with sessions._locked(record["directory"]):
        with pytest.raises(sessions.RevisionConflict):
            candidates.recover_interrupted(record["directory"])
    recovered = candidates.recover_interrupted(record["directory"])
    assert recovered["candidates"][0]["status"] == "interrupted"
    assert candidates.recover_interrupted(record["directory"]) == recovered
