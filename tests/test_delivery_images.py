"""Image-only delivery: real TIFF contracts, fake RAW/Siril orchestration."""
from pathlib import Path
import json
import shutil

import numpy as np
import pytest
import tifffile

from starstack import workflow as wf, workflow_edit as edit, storage
from starstack import master_export as me, photo_render as pr, display_transform as dt
from test_workflow import setup
from test_workflow_review_lifecycle import calibrated_workflow
import color_fixtures as cf


@pytest.fixture
def completed(setup, monkeypatch):
    original = wf.submit_workflow
    monkeypatch.setattr(wf, "submit_workflow", lambda *a, **kw: original(
        *a, delivery_root=setup["output"].parent/"works", **kw))
    return calibrated_workflow(setup, monkeypatch)


def test_new_workflow_only_three_images_and_receipt_stays_in_cache(completed):
    done, _, _ = completed
    root = Path(done["delivery_dir"])
    assert done["request"]["delivery_format"] == done["config"]["delivery_format"] == storage.IMAGE_VERSION
    expected = {"master_scene_linear.tif", "final_srgb16.tif", "final.jpg"}
    assert {p.name for p in root.iterdir()} == expected
    assert {a["name"] for a in done["artifacts"]} == expected
    receipt = storage.delivery_receipt(done, root)
    assert storage.verify_delivery(root, receipt=receipt)["version"] == storage.IMAGE_VERSION
    assert Path(done["publication_receipts"][str(root)]["path"]).parent == Path(done["workflow_dir"])/"publication-receipts"
    wf._verify(done)
    with pytest.raises(FileNotFoundError):
        storage.verify_delivery(root)


def test_moving_only_three_images_renders_with_cache_gone(completed):
    done, _, _ = completed
    root = Path(done["delivery_dir"])
    moved = root.parent.parent/"moved"
    shutil.copytree(root, moved)
    original, _, profile = me.read_master(done["checkpoints"]["stack"]["data"]["colour"]["master"]["path"])
    original = original.copy()
    cache = Path(done["workflow_dir"])
    away = cache.with_name(cache.name+"-offline")
    cache.rename(away)
    try:
        pixels, meta, actual_icc = me.read_master(moved/"master_scene_linear.tif")
        assert np.array_equal(pixels.view(np.uint32), original.view(np.uint32))
        assert profile == actual_icc
        assert meta["camera_saturation"]["storage"] == "tiff-page-1"
        assert "mask_file" not in meta["camera_saturation"]
        params = dt.Params.from_json(meta["portable_master"]["display_transform"])
        rendered = pr.render_photo(moved/"master_scene_linear.tif", moved.parent/"rerender", params)
        assert Path(rendered["jpeg"]).exists()
    finally:
        away.rename(cache)


def test_review_versions_use_same_three_images_and_leave_all_pair_data_cached(completed):
    done, session, state = completed
    before = {a["path"]: a["sha256"] for a in done["artifacts"]}
    result = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    current = result["workflow"]
    root = Path(current["delivery_dir"])
    assert {p.suffix for p in root.iterdir()} == {".jpg", ".tif"}
    assert len(list(root.iterdir())) == 3
    assert Path(result["delivery"]["receipt"]).is_relative_to(Path(done["workflow_dir"]))
    review_record = json.loads(Path(result["delivery"]["receipt"]).read_text(encoding="utf-8"))
    assert review_record["master_pixel_bits_unchanged"] and review_record["master_container_repackaged"]
    assert len(current["publication_receipts"]) == 2
    assert all(wf.core._sha(Path(p)) == sha for p, sha in before.items())
    a, _, _ = me.read_master(Path(done["delivery_dir"])/"master_scene_linear.tif")
    b, _, _ = me.read_master(root/"master_scene_linear.tif")
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    wf._verify(current)
    assert edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")["reused"]
    assert state["applies"] == 1
    plan = storage.plan_cache_cleanup(done["workflow_dir"], cache_root=Path(done["workflow_dir"]).parent)
    assert set(plan["preserved_deliveries"]) == {done["delivery_dir"], current["delivery_dir"]}


@pytest.mark.parametrize("tamper", ["missing", "changed", "foreign"])
def test_v2_manifest_cannot_be_missing_changed_or_foreign(completed, tamper):
    done, _, _ = completed
    desc = done["publication_receipts"][done["delivery_dir"]]
    if tamper == "missing":
        done["publication_receipts"] = {}
    elif tamper == "changed":
        Path(desc["path"]).write_text("{}", encoding="utf-8")
    else:
        outside = Path(done["delivery_dir"])/"forged.json"
        shutil.copyfile(desc["path"], outside)
        desc.update(wf._file(outside))
    with pytest.raises(ValueError):
        wf._verify(done)


def test_cleanup_marker_prevents_review_before_candidate_apply(completed):
    done, session, state = completed
    (Path(done["workflow_dir"])/".aester-cache-cleaned.json").write_text("broken-marker")
    with pytest.raises(ValueError, match="缓存已清理"):
        edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    assert state["applies"] == 0
    assert wf.workflow_status(done["workflow_dir"])["state"] == "succeeded"


@pytest.fixture
def master(tmp_path):
    ref = cf.write_reference_dng(tmp_path/"ref.dng")
    basis = me.colour_basis_from_dng(ref)
    camera, *_ = cf.synthetic_camera_master(144, 160)
    record = me.export_master_array(camera, basis, tmp_path/"master", source_path=ref, source_sha256=me.sha256_file(ref))
    path = Path(record["master"])
    data = tifffile.memmap(path, mode="r+")
    data[0, :3] = [[-0.0, -0.01, 5.0], [np.finfo(np.float32).tiny, .123456789, 2000.], [0, 1, -15]]
    data.flush()
    del data
    return path


def test_portable_float_primary_bit_identity_and_icc(master, tmp_path):
    target = tmp_path/"portable.tif"
    result = me.package_portable_master(master, target)
    a, old, old_icc = me.read_master(master)
    b, meta, profile = me.read_master(target)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert old_icc == profile and meta["colour_basis"] == old["colour_basis"]
    assert result["primary_pixels_bit_identical"]
    assert np.array_equal(me.read_saturation_mask(master, old, a.shape[:2]), me.read_saturation_mask(target, meta, b.shape[:2]))


def test_embedded_mask_tamper_rejected(master, tmp_path):
    target = tmp_path/"portable.tif"
    me.package_portable_master(master, target)
    rgb, meta, profile = me.read_master(target)
    rgb = np.array(rgb)
    mask = tifffile.imread(target, key=1)
    mask[0, 0] = 255-mask[0, 0]
    with tifffile.TiffWriter(target) as writer:
        writer.write(rgb, photometric="rgb", iccprofile=profile, metadata=meta)
        writer.write(mask, photometric="minisblack", compression="deflate")
    pixels, meta, _ = me.read_master(target)
    with pytest.raises(ValueError, match="哈希"):
        me.read_saturation_mask(target, meta, pixels.shape[:2])


def test_embedded_mask_is_losslessly_compressed_without_imagecodecs(master, tmp_path, monkeypatch):
    from types import SimpleNamespace
    # Force the same stdlib fallback available in a minimal installed runtime.
    monkeypatch.setattr(tifffile.tifffile, "imagecodecs", SimpleNamespace())
    encode = tifffile.TIFF.COMPRESSORS
    decode = tifffile.TIFF.DECOMPRESSORS
    for codecs in (encode, decode):
        for key in (8, 32946):
            monkeypatch.delitem(codecs._codecs, key, raising=False)
    target = tmp_path/"compressed.tif"
    me.package_portable_master(master, target)
    with tifffile.TiffFile(target) as tf:
        assert int(tf.pages[0].compression) == 1
        assert int(tf.pages[1].compression) in (8, 32946)
        assert sum(tf.pages[1].databytecounts) < np.prod(tf.pages[1].shape)/4
    a, old, _ = me.read_master(master)
    b, current, _ = me.read_master(target)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert np.array_equal(me.read_saturation_mask(master, old, a.shape[:2]), me.read_saturation_mask(target, current, b.shape[:2]))


def test_unannounced_extra_page_is_rejected(master):
    with tifffile.TiffWriter(master, append=True) as writer:
        writer.write(np.zeros((144, 160), np.uint8))
    with pytest.raises(ValueError, match="单页"):
        me.read_master(master)


@pytest.mark.parametrize("failure", ["copy", "cancel"])
def test_v2_failed_publication_keeps_receipts_cached_and_can_retry(completed, monkeypatch, failure):
    done, _, _ = completed
    directory = Path(done["workflow_dir"])
    stack = done["checkpoints"]["stack"]["data"]
    before = done["delivery_dir"]
    original = storage._copy_verified
    def broken(source, target, check):
        if failure == "cancel":
            raise wf.WorkflowCancelled("test cancellation")
        raise OSError("test full disk")
    monkeypatch.setattr(storage, "_copy_verified", broken)
    with pytest.raises((OSError, wf.WorkflowCancelled)):
        wf._publish(directory, stack, done["finishing"])
    assert wf._read(directory)["delivery_dir"] == before
    assert not list(storage.publication_parent(done).rglob("*.json"))
    assert list((directory/"publication-receipts").glob("*.failure.json"))
    monkeypatch.setattr(storage, "_copy_verified", original)
    result = wf._publish(directory, stack, done["finishing"])
    assert result["delivery_dir"] != before
    assert len(list(Path(result["delivery_dir"]).iterdir())) == 3
    wf._verify(result)


def test_receipt_changed_between_stage_and_publication_is_rejected(completed):
    done, _, _ = completed
    root = Path(done["delivery_dir"])
    sources = {p.name: {"path": str(p), "kind": "tiff", "label": p.name} for p in root.iterdir()}
    package = storage.stage_delivery(done, sources, "delivery-test-tamper")
    Path(package["receipt"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="改变"):
        storage.verify_staged(package)
    assert not package["target"].exists()
