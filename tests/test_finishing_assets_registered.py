"""Independent small-image tests for registered NL-1 execution assets."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from starstack import finishing_assets as assets, finishing_session as sessions


@pytest.fixture
def registered_session(tmp_path):
    pixels = np.arange(8 * 12 * 3, dtype=np.uint16).reshape(8, 12, 3) * 137
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "source.tif"
    tifffile.imwrite(path, pixels, photometric="rgb", rowsperstrip=2, metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    return sessions.create_session(path, tmp_path, intent="small asset contract test")


def _mask(path, bits=8, *, fill=None):
    dtype = np.uint8 if bits == 8 else np.uint16
    data = np.arange(8 * 12, dtype=dtype).reshape(8, 12)
    if bits == 16:
        data = data * 601  # Preserve weights that cannot come from uint8 expansion.
    if fill is not None:
        data.fill(fill)
    if path.suffix == ".png":
        Image.fromarray(data).save(path)
    else:
        tifffile.imwrite(path, data, photometric="minisblack", metadata=None)
    return data


def _register(record, path, mask_id="ground"):
    return sessions.register_mask(record["directory"], path, mask_id=mask_id,
                                  source_sha256=record["source"]["sha256"],
                                  origin="agent", note="synthetic coverage")


def _recipe(record, *, mask_id="ground", enabled=True):
    recipe = deepcopy(record["revisions"][-1]["recipe"])
    recipe["operations"] = [{"id": "exposure", "type": "exposure", "enabled": enabled,
                              "mask_id": mask_id, "params": {"ev": 0.25}}]
    return recipe


def _pixels(path):
    if path.suffix == ".png":
        with Image.open(path) as im:
            return np.asarray(im).copy()
    return tifffile.imread(path)


@pytest.mark.parametrize("suffix,bits", [(".png", 8), (".png", 16), (".tif", 8), (".tif", 16)])
def test_literal_gray_weights_are_snapshotted_without_changing_inputs(registered_session, tmp_path, suffix, bits):
    external = tmp_path / ("coverage" + suffix)
    pixels = _mask(external, bits)
    before = external.read_bytes()
    source_before = Path(registered_session["source"]["path"]).read_bytes()
    record = _register(registered_session, external)
    stored = record["masks"]["ground"]
    snapshot = Path(stored["path"])
    assert snapshot.parent == Path(record["directory"]) / "assets"
    assert not os.path.samefile(snapshot, external)
    assert snapshot.read_bytes() == before == external.read_bytes()
    assert np.array_equal(_pixels(snapshot), pixels)
    assert stored["bits"] == bits
    assert stored["sha256"] == hashlib.sha256(before).hexdigest()
    assert stored["source_sha256"] == record["source"]["sha256"]
    assert stored["origin"] == "agent"
    updated = sessions.revise_recipe(record["directory"], _recipe(record), expected_revision=0)
    assert sessions.renderer_masks(updated, updated["revisions"][-1]["recipe"])["ground"]["path"] == str(snapshot)
    assert Path(registered_session["source"]["path"]).read_bytes() == source_before


def test_external_mask_replacement_does_not_change_registered_execution(registered_session, tmp_path):
    external = tmp_path / "replaceable.png"
    pixels = _mask(external, 16)
    record = _register(registered_session, external)
    snapshot = Path(record["masks"]["ground"]["path"])
    original_hash = record["masks"]["ground"]["sha256"]
    _mask(external, 16, fill=0)
    result = sessions.renderer_masks(record, _recipe(record))
    assert result["ground"]["sha256"] == original_hash
    assert np.array_equal(_pixels(snapshot), pixels)
    assert not np.array_equal(_pixels(external), pixels)


@pytest.mark.parametrize("enabled", [True, False])
def test_changed_internal_mask_rejected_even_if_operation_disabled(registered_session, tmp_path, enabled):
    external = tmp_path / "coverage.tif"
    _mask(external, 16)
    record = _register(registered_session, external)
    _mask(Path(record["masks"]["ground"]["path"]), 16, fill=65535)
    journal = Path(record["directory"]) / "session.json"
    before = journal.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        sessions.revise_recipe(record["directory"], _recipe(record, enabled=enabled), expected_revision=0)
    assert journal.read_bytes() == before


def test_mask_wrong_source_rejected_before_snapshot_or_journal_change(registered_session, tmp_path):
    external = tmp_path / "coverage.png"
    _mask(external)
    with pytest.raises(ValueError, match="different source"):
        sessions.register_mask(registered_session["directory"], external, mask_id="ground",
                               source_sha256="0" * 64, origin="user")
    assert sessions.read_session(registered_session["directory"]) == registered_session
    assert not (Path(registered_session["directory"]) / "assets").exists()


@pytest.mark.parametrize("kind", ["wrong_dimensions", "rgb_png", "rgba_png", "la_png", "palette_png",
                                  "rgb_tif", "rgba_tif", "float_tif", "signed_tif", "whiteiszero",
                                  "multipage_tif", "icc_png", "icc_tif", "tiff_orientation",
                                  "png_transparency", "png_orientation", "truncated_tif"])
def test_incompatible_masks_are_not_silently_converted(registered_session, tmp_path, kind):
    is_png = kind.endswith("png") or kind in {"wrong_dimensions", "png_transparency", "png_orientation"}
    path = tmp_path / (kind + (".png" if is_png else ".tif"))
    gray = np.arange(96, dtype=np.uint8).reshape(8, 12)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    if kind == "wrong_dimensions":
        Image.fromarray(gray[:-1]).save(path)
    elif kind in {"rgb_png", "rgba_png", "la_png", "palette_png"}:
        mode = {"rgb_png": "RGB", "rgba_png": "RGBA", "la_png": "LA", "palette_png": "P"}[kind]
        Image.fromarray(gray).convert(mode).save(path)
    elif kind == "icc_png":
        Image.fromarray(gray).save(path, icc_profile=icc)
    elif kind == "png_transparency":
        Image.fromarray(gray).save(path, transparency=0)  # tRNS, although IHDR color type is gray.
    elif kind == "png_orientation":
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(gray).save(path, exif=exif)
    elif kind in {"rgb_tif", "rgba_tif"}:
        tifffile.imwrite(path, np.repeat(gray[:, :, None], 3 if kind == "rgb_tif" else 4, axis=2), photometric="rgb")
    elif kind == "multipage_tif":
        with tifffile.TiffWriter(path) as writer:
            writer.write(gray, photometric="minisblack")
            writer.write(gray, photometric="minisblack")
    else:
        data = gray.astype(np.float32) if kind == "float_tif" else gray.astype(np.int16) if kind == "signed_tif" else gray
        tags = [(34675, "B", len(icc), icc, False)] if kind == "icc_tif" else [(274, "H", 1, 6, False)] if kind == "tiff_orientation" else []
        tifffile.imwrite(path, data, photometric="miniswhite" if kind == "whiteiszero" else "minisblack", extratags=tags)
        if kind == "truncated_tif":
            path.write_bytes(path.read_bytes()[:-7])
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _register(registered_session, path)
    assert path.read_bytes() == before
    assert sessions.read_session(registered_session["directory"]) == registered_session


def test_disabled_unknown_mask_never_becomes_global(registered_session):
    with pytest.raises(ValueError, match="Unregistered mask"):
        sessions.revise_recipe(registered_session["directory"], _recipe(registered_session, mask_id="missing", enabled=False), expected_revision=0)
    assert sessions.read_session(registered_session["directory"])["revision"] == 0
    explicit_global = sessions.revise_recipe(registered_session["directory"], _recipe(registered_session, mask_id=None, enabled=False), expected_revision=0)
    assert sessions.renderer_masks(explicit_global, explicit_global["revisions"][-1]["recipe"]) == {}


@pytest.mark.parametrize("kind", ["original", "generated", "external"])
def test_reference_provenance_is_preserved_but_never_becomes_renderer_input(registered_session, tmp_path, kind):
    reference = tmp_path / "aesthetic-reference.png"
    Image.new("RGB", (5, 3), "red").save(reference)
    original = reference.read_bytes()
    record = sessions.register_reference(registered_session["directory"], reference, kind=kind, note="aesthetic guide only")
    ref = record["references"][0]
    assert ref["kind"] == kind and ref["note"] == "aesthetic guide only"
    assert Path(ref["path"]).read_bytes() == original
    Image.new("RGB", (5, 3), "blue").save(reference)
    assert Path(ref["path"]).read_bytes() == original
    assert sessions.renderer_masks(record, _recipe(record, mask_id=None)) == {}
    with pytest.raises(ValueError, match="Unregistered mask"):
        sessions.renderer_masks(record, _recipe(record, mask_id=ref["id"]))
    mask_path = tmp_path / "real-coverage.png"
    _mask(mask_path)
    record = _register(record, mask_path)
    assert set(sessions.renderer_masks(record, _recipe(record))) == {"ground"}
    with pytest.raises(ValueError, match="provenance"):
        sessions.register_reference(record["directory"], reference, kind="unclassified")


def test_duplicate_mask_id_does_not_replace_existing_snapshot(registered_session, tmp_path):
    mask_path = tmp_path / "coverage.png"
    _mask(mask_path)
    record = _register(registered_session, mask_path)
    _mask(mask_path, fill=0)
    with pytest.raises(ValueError, match="already exists"):
        _register(record, mask_path)
    assert sessions.read_session(record["directory"]) == record


@pytest.mark.parametrize("asset_type", ["mask", "reference"])
def test_failed_asset_journal_write_removes_only_uncommitted_copy(registered_session, tmp_path, monkeypatch, asset_type):
    external = tmp_path / "coverage.png"
    _mask(external)
    before = external.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("synthetic journal failure")
    monkeypatch.setattr(sessions, "atomic_json", fail)
    with pytest.raises(OSError, match="synthetic journal failure"):
        if asset_type == "mask":
            _register(registered_session, external)
        else:
            sessions.register_reference(registered_session["directory"], external, kind="original")
    assert external.read_bytes() == before
    assert sessions.read_session(registered_session["directory"]) == registered_session
    assert list((Path(registered_session["directory"]) / "assets").iterdir()) == []


def test_truncated_source_strip_is_rejected_before_session_creation(registered_session, tmp_path):
    source = Path(registered_session["source"]["path"])
    truncated = tmp_path / "truncated-source.tif"
    truncated.write_bytes(source.read_bytes()[:-7])
    with tifffile.TiffFile(truncated) as tf:
        assert tf.pages[0].shape == (8, 12, 3)  # Tags alone still look valid.
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="TIFF pixel segment"):
        sessions.create_session(truncated, tmp_path, intent="must reject truncation")
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("origin", ["user", "agent", "imported"])
def test_mask_origin_uses_contract_provenance_without_human_promotion(registered_session, tmp_path, origin):
    external = tmp_path / "coverage.png"
    _mask(external)
    record = sessions.register_mask(registered_session["directory"], external, mask_id="ground",
                                    source_sha256=registered_session["source"]["sha256"], origin=origin)
    assert record["masks"]["ground"]["origin"] == origin
    assert "human_confirmed" not in record["masks"]["ground"]


@pytest.mark.parametrize("kind", ["user_reference", "generated_reference", "unknown", "", None])
def test_reference_noncontract_provenance_is_not_accepted(registered_session, tmp_path, kind):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (5, 3), "red").save(reference)
    with pytest.raises(ValueError):
        sessions.register_reference(registered_session["directory"], reference, kind=kind)
    assert sessions.read_session(registered_session["directory"]) == registered_session


@pytest.mark.parametrize("mask_id", ["../ground", "ground.png", "C:\\ground", "", "human/confirmed"])
def test_mask_identifier_never_resolves_as_a_file_path(registered_session, tmp_path, mask_id):
    external = tmp_path / "coverage.png"
    _mask(external)
    with pytest.raises(ValueError, match="asset ID"):
        _register(registered_session, external, mask_id=mask_id)
    assert sessions.read_session(registered_session["directory"]) == registered_session
