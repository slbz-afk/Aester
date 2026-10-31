"""Small real files for layer binding, signed supports and publication failures.

Implementation tests, distinct from the preceding independent design review.
No renderer, GIMP or large image is invoked.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from starstack import finishing_layers as layers, finishing_session as sessions
from starstack.finishing_assets import sha256_file, recipe_source


def rgb(path, pixels, icc, **kwargs):
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)], **kwargs)


@pytest.fixture
def case(tmp_path):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    d = np.full((16, 20, 3), 10000, np.uint16)
    b = d.copy()
    b[5, 5] = [12000, 9000, 10000]  # Both signs of the star residual.
    s = b.copy()
    s[6, 7] = [10400, 9750, 10000]  # Both signs of the transient residual.
    files = {key: tmp_path / f"{key}.tif" for key in ("source", "background", "diffuse")}
    for key, pixels in zip(files, (s, b, d)):
        rgb(files[key], pixels, icc, rowsperstrip=2)
    value = sessions.create_session(files["source"], tmp_path, intent="bound layer implementation tests")
    pure = np.zeros((16, 20), np.uint8)
    pure[2:-2, 1:-1] = 255
    support = np.zeros_like(pure)
    support[6:8, 7:9] = 255
    for identity, pixels in (("pure", pure), ("support", support)):
        path = tmp_path / f"{identity}.png"
        Image.fromarray(pixels).save(path)
        value = sessions.register_mask(value["directory"], path, mask_id=identity,
            source_sha256=value["source"]["sha256"], origin="agent")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"claim":"caller supplied chain, not certified"}', encoding="utf-8")
    provenance = {"kind": "caller_reported_layer_chain", "description": "Reported derivation evidence",
                  "evidence": [{"path": str(evidence), "sha256": sha256_file(evidence)}]}
    return {"value": value, "files": files, "pixels": (s, b, d), "icc": icc,
            "provenance": provenance, "evidence": evidence}


def register(case, **changes):
    value = case["value"]
    args = dict(bundle_id="pair", background_path=case["files"]["background"],
                diffuse_path=case["files"]["diffuse"], pure_sky_mask_id="pure",
                transient_support_mask_id="support", source_sha256=value["source"]["sha256"],
                provenance=case["provenance"])
    args.update(changes)
    return layers.register_layers(value["directory"], **args)


def asset_names(value):
    return set((Path(value["directory"]) / "assets").iterdir())


def assert_unchanged_failed_registration(case, call):
    value = case["value"]
    journal = Path(value["directory"]) / "session.json"
    before, files = journal.read_bytes(), asset_names(value)
    with pytest.raises((ValueError, OSError)):
        call()
    assert journal.read_bytes() == before
    assert asset_names(value) == files


def test_snapshots_signed_structure_and_truthful_provenance(case):
    original = {key: path.read_bytes() for key, path in case["files"].items()}
    value = register(case)
    bundle = layers.verify_bundle(value, "pair")
    assert bundle["lineage_verified"] is False
    assert bundle["structural_checks"]["lineage_or_separation_quality_proven"] is False
    assert bundle["structural_checks"]["R_range"] == [-1000, 2000]
    assert bundle["structural_checks"]["T_range"] == [-250, 400]
    assert bundle["structural_checks"]["checked_pixels"] == 320
    assert bundle["structural_checks"]["pure_sky_pixels"] == 216
    for name, pixels in zip(("source", "background", "diffuse"), case["pixels"]):
        assert bundle["structural_checks"]["pixel_sha256"][name] == hashlib.sha256(pixels.astype("<u2").tobytes()).hexdigest()
        assert case["files"][name].read_bytes() == original[name]
    assert bundle["source"] == value["source"]
    for name in ("background", "diffuse"):
        assert Path(bundle[name]["path"]).parent == Path(value["directory"]) / "assets"
        assert Path(bundle[name]["path"]).read_bytes() == original[name]
        assert Path(bundle[name]["path"]) != case["files"][name]
    copied_evidence = Path(bundle["provenance"]["evidence"][0]["path"])
    assert copied_evidence.read_bytes() == case["evidence"].read_bytes()
    assert copied_evidence.parent == Path(value["directory"]) / "assets"
    assert value["revisions"] == case["value"]["revisions"]


def test_external_changes_do_not_affect_owned_bundle(case):
    value = register(case)
    expected = deepcopy(value["layer_bundles"]["pair"])
    for name in ("background", "diffuse"):
        case["files"][name].write_bytes(b"external replacement")
    case["evidence"].write_text("new external evidence", encoding="utf-8")
    assert layers.verify_bundle(value, "pair", expected["bundle_sha256"]) == expected
    result = layers.verify_bundle(value, "pair")
    result["structural_checks"]["checked_pixels"] = 1
    assert value["layer_bundles"]["pair"] == expected  # Returned record is independent.


@pytest.mark.parametrize("which", ["source", "background", "diffuse", "pure_sky", "transient_support", "evidence"])
def test_internal_or_source_changes_are_rejected(case, which):
    value = register(case)
    bundle = value["layer_bundles"]["pair"]
    if which in {"pure_sky", "transient_support"}:
        path = Path(bundle[which]["asset"]["path"])
    elif which == "evidence":
        path = Path(bundle["provenance"]["evidence"][0]["path"])
    else:
        path = Path(bundle[which]["path"])
    path.write_bytes(path.read_bytes() + b"identity changed")
    with pytest.raises(ValueError):
        layers.verify_bundle(value, "pair")


def test_wrong_source_and_duplicate_id_do_not_publish(case):
    assert_unchanged_failed_registration(case, lambda: register(case, source_sha256="0" * 64))
    value = register(case)
    before = (Path(value["directory"]) / "session.json").read_bytes()
    files = asset_names(value)
    with pytest.raises(ValueError, match="ID exists"):
        register(case)
    assert (Path(value["directory"]) / "session.json").read_bytes() == before
    assert asset_names(value) == files


@pytest.mark.parametrize("kind", ["geometry", "icc", "signed", "alpha", "compression", "multipage", "orientation", "truncated"])
def test_incompatible_photograph_domains_rejected(case, kind):
    path = case["files"]["background"]
    pixels, icc = case["pixels"][1].copy(), case["icc"]
    if kind == "geometry":
        rgb(path, pixels[:-1], icc)
    elif kind == "icc":
        other = bytearray(icc)
        other[24:26] = (2001).to_bytes(2, "big")  # Valid sRGB, different profile bytes.
        rgb(path, pixels, bytes(other))
    elif kind == "signed":
        rgb(path, pixels.astype(np.int32), icc)
    elif kind == "alpha":
        rgb(path, np.dstack([pixels, np.full(pixels.shape[:2], 65535, np.uint16)]), icc)
    elif kind == "compression":
        rgb(path, pixels, icc, compression="deflate")
    elif kind == "multipage":
        with tifffile.TiffWriter(path) as writer:
            for _ in range(2):
                writer.write(pixels, photometric="rgb", metadata=None,
                    extratags=[(34675, "B", len(icc), icc, False)])
    elif kind == "orientation":
        tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
            extratags=[(34675, "B", len(icc), icc, False), (274, "H", 1, 6, False)])
    elif kind == "truncated":
        path.write_bytes(path.read_bytes()[:-2])
    assert_unchanged_failed_registration(case, lambda: register(case))


@pytest.mark.parametrize("which", ["pure", "support"])
def test_fractional_registered_support_is_rejected(case, tmp_path, which):
    data = np.full((16, 20), 255, np.uint8)
    data[0, 0] = 254
    path = tmp_path / "fractional.png"
    Image.fromarray(data).save(path)
    value = sessions.register_mask(case["value"]["directory"], path, mask_id="fractional",
        source_sha256=case["value"]["source"]["sha256"], origin="agent")
    case["value"] = value
    keyword = "pure_sky_mask_id" if which == "pure" else "transient_support_mask_id"
    assert_unchanged_failed_registration(case, lambda: register(case, **{keyword: "fractional"}))


def test_binary_16bit_tiff_support_is_accepted(case, tmp_path):
    value = case["value"]
    for identity, source_id in (("pure16", "pure"), ("support16", "support")):
        with Image.open(value["masks"][source_id]["path"]) as im:
            data = np.asarray(im).astype(np.uint16) * 257
        path = tmp_path / f"{identity}.tif"
        tifffile.imwrite(path, data, photometric="minisblack", metadata=None)
        value = sessions.register_mask(value["directory"], path, mask_id=identity,
            source_sha256=value["source"]["sha256"], origin="agent")
    case["value"] = value
    registered = register(case, pure_sky_mask_id="pure16", transient_support_mask_id="support16")
    assert layers.verify_bundle(registered, "pair")["pure_sky"]["asset"]["bits"] == 16


@pytest.mark.parametrize("kind", ["T_outside_support", "T_outside_pure", "R_outside_pure"])
def test_full_raster_support_relationships_rejected(case, tmp_path, kind):
    if kind == "T_outside_support":
        support = np.zeros((16, 20), np.uint8)
        path = tmp_path / "empty.png"
        Image.fromarray(support).save(path)
        case["value"] = sessions.register_mask(case["value"]["directory"], path, mask_id="empty",
            source_sha256=case["value"]["source"]["sha256"], origin="agent")
        call = lambda: register(case, transient_support_mask_id="empty")
    elif kind == "T_outside_pure":
        # Source/B transient at a supported pixel, then exclude that pixel from pure sky.
        pure = np.full((16, 20), 255, np.uint8)
        pure[6, 7] = 0
        path = tmp_path / "hole.png"
        Image.fromarray(pure).save(path)
        case["value"] = sessions.register_mask(case["value"]["directory"], path, mask_id="hole",
            source_sha256=case["value"]["source"]["sha256"], origin="agent")
        call = lambda: register(case, pure_sky_mask_id="hole")
    else:
        data = case["pixels"][2].copy()
        data[15, 19, 2] += 1  # Last pixel proves checks are not just a top-left sample.
        rgb(case["files"]["diffuse"], data, case["icc"])
        call = lambda: register(case)
    assert_unchanged_failed_registration(case, call)


@pytest.mark.parametrize("kind", ["generated", "external"])
@pytest.mark.parametrize("which", ["background", "diffuse"])
def test_reference_identity_not_allowed_as_layer_even_at_other_path(case, kind, which, tmp_path):
    sessions.register_reference(case["value"]["directory"], case["files"][which], kind=kind)
    path = tmp_path / "renamed.tif"
    path.write_bytes(case["files"][which].read_bytes())
    assert_unchanged_failed_registration(case, lambda: register(case, **{which + "_path": path}))


@pytest.mark.parametrize("change", ["unknown", "kind", "empty_description", "too_long", "empty_evidence", "too_many", "hash", "item_unknown"])
def test_provenance_is_bounded_data_and_not_self_certification(case, change):
    p = deepcopy(case["provenance"])
    if change == "unknown": p["verified"] = True
    elif change == "kind": p["kind"] = "verified_layer_chain"
    elif change == "empty_description": p["description"] = "  "
    elif change == "too_long": p["description"] = "x" * 4001
    elif change == "empty_evidence": p["evidence"] = []
    elif change == "too_many": p["evidence"] *= 17
    elif change == "hash": p["evidence"][0]["sha256"] = "0" * 64
    elif change == "item_unknown": p["evidence"][0]["execute"] = True
    assert_unchanged_failed_registration(case, lambda: register(case, provenance=p))


@pytest.mark.parametrize("target", ["background", "source", "mask", "evidence", "copied_background"])
def test_input_mutation_after_structural_scan_is_rejected_and_copies_removed(case, monkeypatch, target):
    real = layers._structure
    def mutate(*args, **kwargs):
        result = real(*args, **kwargs)
        path = (Path(args[1]["path"]) if target == "copied_background" else
                Path(case["value"]["masks"]["pure"]["path"]) if target == "mask" else
                case["evidence"] if target == "evidence" else case["files"][target])
        path.write_bytes(path.read_bytes() + b"changed after scan")
        return result
    monkeypatch.setattr(layers, "_structure", mutate)
    assert_unchanged_failed_registration(case, lambda: register(case))


def test_copy_failure_cleans_only_new_files(case, monkeypatch):
    real = sessions._copy_asset
    calls = 0
    def copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-copy failure")
        return real(*args, **kwargs)
    monkeypatch.setattr(sessions, "_copy_asset", copy)
    assert_unchanged_failed_registration(case, lambda: register(case))
    assert all(path.is_file() for path in case["files"].values())


def test_source_change_during_copy_rejected_even_when_snapshot_is_good(case, monkeypatch):
    real = sessions._copy_asset
    def copy(directory, record):
        result = real(directory, record)
        if Path(record["path"]) == case["files"]["background"]:
            case["files"]["background"].write_bytes(case["files"]["background"].read_bytes() + b"external race")
        return result
    monkeypatch.setattr(sessions, "_copy_asset", copy)
    assert_unchanged_failed_registration(case, lambda: register(case))


def test_save_failure_rolls_back_new_copies(case, monkeypatch):
    def fail(*args):
        raise ValueError("session budget exceeded")
    monkeypatch.setattr(sessions, "_save", fail)
    assert_unchanged_failed_registration(case, lambda: register(case))


def test_bundle_digest_source_mask_and_owned_path_bindings(case):
    value = register(case)
    with pytest.raises(ValueError, match="digest"):
        layers.verify_bundle(value, "pair", "0" * 64)
    for mutation in ("source", "mask", "lineage", "owned_path"):
        altered = deepcopy(value)
        bundle = altered["layer_bundles"]["pair"]
        if mutation == "source": bundle["source"]["sha256"] = "0" * 64
        elif mutation == "mask": bundle["pure_sky"]["mask_id"] = "support"
        elif mutation == "lineage": bundle["lineage_verified"] = True
        else: bundle["background"]["path"] = str(case["files"]["background"])
        # Even a recomputed content digest cannot grant source/path/lineage authority.
        bundle["bundle_sha256"] = layers._digest(bundle)
        with pytest.raises(ValueError):
            layers.verify_bundle(altered, "pair")


def test_recipe_only_collects_referenced_bundle_and_checks_disabled(case):
    value = register(case)
    bundle = value["layer_bundles"]["pair"]
    value["layer_bundles"]["unused"] = {"not": "validated or read"}
    recipe = {"source": recipe_source(value["source"]), "operations": []}
    assert layers.recipe_bundles(value, recipe) == {}
    op = {"type": "astrophotographic_layers", "enabled": False,
          "params": {"bundle_id": "pair", "bundle_sha256": bundle["bundle_sha256"]}}
    recipe["operations"] = [op]
    assert set(layers.recipe_bundles(value, recipe)) == {"pair"}
    op["params"]["bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        layers.recipe_bundles(value, recipe)
    op["params"]["bundle_id"] = "missing"
    with pytest.raises(ValueError):
        layers.recipe_bundles(value, recipe)


@pytest.mark.parametrize("digest", [None, True, "A" * 64, "bad"])
def test_evidence_requires_an_explicit_real_hash(case, digest):
    p = deepcopy(case["provenance"])
    p["evidence"][0]["sha256"] = digest
    assert_unchanged_failed_registration(case, lambda: register(case, provenance=p))


@pytest.mark.parametrize("rows", [1, 3, 17])
def test_structural_results_do_not_depend_on_scan_blocks(case, monkeypatch, rows):
    value = register(case)
    expected = deepcopy(value["layer_bundles"]["pair"])
    monkeypatch.setattr(layers, "_ROWS", rows)
    assert layers.verify_bundle(value, "pair") == expected


def test_frozen_render_verification_needs_no_session_read(case, monkeypatch):
    value = register(case)
    bundle = value["layer_bundles"]["pair"]
    def forbidden(*args):
        raise AssertionError("render verification must use the frozen inputs")
    monkeypatch.setattr(sessions, "read_session", forbidden)
    assert layers.verify_render_bundle(value["source"], bundle) == bundle
    wrong_source = deepcopy(value["source"])
    wrong_source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        layers.verify_render_bundle(wrong_source, bundle)


@pytest.mark.parametrize("changed", ["diffuse", "support", "evidence", "digest", "assets_parent"])
def test_frozen_render_bundle_cannot_trust_old_json(case, changed, tmp_path):
    value = register(case)
    bundle = deepcopy(value["layer_bundles"]["pair"])
    if changed == "digest":
        bundle["bundle_sha256"] = "0" * 64
    elif changed == "assets_parent":
        # A matching file/hash outside the bundle assets is still rejected.
        other = tmp_path / "other.tif"
        other.write_bytes(Path(bundle["diffuse"]["path"]).read_bytes())
        bundle["diffuse"]["path"] = str(other)
        bundle["bundle_sha256"] = layers._digest(bundle)
    else:
        record = (bundle["diffuse"] if changed == "diffuse" else
                  bundle["transient_support"]["asset"] if changed == "support" else
                  bundle["provenance"]["evidence"][0])
        path = Path(record["path"])
        path.write_bytes(path.read_bytes() + b"modified after freeze")
    with pytest.raises(ValueError):
        layers.verify_render_bundle(value["source"], bundle)
