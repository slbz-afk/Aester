"""Tonal selection contracts: exact zeros, encoded domain, binding and API."""
import numpy as np
from PIL import ImageCms
import pytest
import tifffile

from starstack.finishing_masks import tonal_coverage
from starstack.finishing_assets import inspect_source
from starstack.finishing_service import FinishingTools
from starstack.mcp_server import WorkspaceTools


@pytest.fixture
def scene(tmp_path):
    pixels = np.repeat(np.arange(0, 65536, 257, dtype=np.uint16)[None, :, None], 3, axis=2)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "original.tif"
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    return tmp_path, inspect_source(path, encoding="srgb16")


def test_full_range_and_inversion_include_both_endpoints(scene):
    folder, source = scene
    for invert in (False, True):
        made = tonal_coverage(folder, source, source, [0, 0, 1, 1], invert=invert)
        actual = tifffile.imread(made["path"])
        assert np.all(actual == (0 if invert else 65535))


def test_midpoint_rolloff_and_complement(scene):
    folder, source = scene
    # 85/255 is exactly midway along [0,170/255].
    first = tifffile.imread(tonal_coverage(folder, source, source, [0, 170/255, 1, 1])["path"])
    second = tifffile.imread(tonal_coverage(folder, source, source, [0, 170/255, 1, 1], invert=True)["path"])
    assert abs(int(first[0, 85]) - 32768) <= 1
    assert first[0, 0] == 0 and first[0, -1] == 65535
    assert np.max(abs(first.astype(int) + second.astype(int) - 65535)) <= 1
    assert np.all(np.diff(first.astype(int), axis=1) >= 0)


@pytest.mark.parametrize("knots", [[0, 1, 0, 1], [0, 0, 0, 0], [True, 0, 1, 1],
    [0, 0, 1, float("nan")], [0, 0, 1], [-.1, 0, 1, 1], [0, 0, 1, 2], (0, 0, 1, 1)])
def test_invalid_selection_rejected_before_output(scene, knots):
    folder, source = scene
    with pytest.raises(ValueError):
        tonal_coverage(folder, source, source, knots)
    assert not (folder / "previews").exists()


def test_public_region_zero_and_uint16_intermediate_weight(scene):
    folder, source = scene
    cache, delivery = folder / "cache", folder / "exports"
    cache.mkdir(); delivery.mkdir()
    service = FinishingTools(WorkspaceTools(input_root=folder, output_root=delivery, cache_root=cache))
    try:
        session = service.create(source["path"], "Tonal zoning")
        path = folder / "region.tif"
        region = np.zeros((1, 256), np.uint16)
        region[:, 20:40] = 32768
        region[:, 40:60] = 65535
        tifffile.imwrite(path, region, photometric="minisblack", metadata=None)
        service.add_mask(session["directory"], str(path), "ground", source["sha256"])
        result = service.tonal_mask(session["directory"], "shadows", [.5, .6, 1, 1],
                                    region_mask_id="ground", invert=True)
        mask = result["session"]["masks"]["shadows"]
        np.testing.assert_array_equal(tifffile.imread(mask["path"]), region)
        assert mask["origin"] == "agent" and mask["source_sha256"] == source["sha256"]
        assert result["geometry"]["image_sha256"] == source["sha256"]
        assert result["mask_preview"]["kind"] == "mask_overlay"
        with pytest.raises(ValueError, match="replaced"):
            service.tonal_mask(session["directory"], "shadows", [0, 0, 1, 1])
        with pytest.raises(ValueError, match="Unknown region"):
            service.tonal_mask(session["directory"], "new", [0, 0, 1, 1], region_mask_id="absent")
        assert service.get(session["directory"])["masks"]["shadows"] == mask
    finally:
        service.close()


def test_source_changed_is_rejected(scene):
    folder, source = scene
    other = {**source, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="changed"):
        tonal_coverage(folder, source, other, [0, 0, 1, 1])


@pytest.mark.parametrize("code,knots", [(1, [1/65535, 1/65535, 1, 1]),
                                      (101, [0, 0, 101/65535, 101/65535])])
def test_equal_gray_inclusive_hard_threshold_is_exact(scene, code, knots):
    folder, source = scene
    with tifffile.TiffFile(source["path"]) as tf:
        icc = bytes(tf.pages[0].tags[34675].value)
    tifffile.imwrite(source["path"], np.full((1, 256, 3), code, dtype=np.uint16),
        photometric="rgb", metadata=None, extratags=[(34675, "B", len(icc), icc, False)])
    source = inspect_source(source["path"], encoding="srgb16")
    made = tonal_coverage(folder, source, source, knots)
    assert np.all(tifffile.imread(made["path"]) == 65535)


def test_cli_accepts_explicit_knots_and_maps_service():
    from starstack.finishing_cli import build_parser, _arguments
    args = build_parser().parse_args(["tonal-mask", "--session", "s", "--mask-id", "ground-shadows",
        "--knots", "0", "0", ".3", ".6", "--region-mask-id", "ground", "--invert"])
    assert _arguments(args) == {"session_dir": "s", "mask_id": "ground-shadows",
        "knots": [0., 0., .3, .6], "candidate_id": None, "region_mask_id": "ground", "invert": True}


@pytest.mark.parametrize("changed", ["image", "region"])
def test_dependency_change_during_publication_leaves_no_registered_asset(scene, monkeypatch, changed):
    from pathlib import Path
    from starstack import finishing_session as sessions

    folder, source = scene
    cache, delivery = folder / "cache", folder / "exports"
    cache.mkdir(); delivery.mkdir()
    service = FinishingTools(WorkspaceTools(input_root=folder, output_root=delivery, cache_root=cache))
    try:
        session = service.create(source["path"], "Publication race regression")
        region_path = folder / "region.tif"
        tifffile.imwrite(region_path, np.full((1, 256), 32768, np.uint16),
                         photometric="minisblack", metadata=None)
        session = service.add_mask(session["directory"], str(region_path), "region", source["sha256"])
        assets = Path(session["directory"]) / "assets"
        before = set(assets.iterdir())
        copy_asset = sessions._copy_asset

        def mutate_after_copy(directory, record):
            copied = copy_asset(directory, record)
            path = Path(source["path"] if changed == "image" else session["masks"]["region"]["path"])
            # Change content after selection and after the new asset copy, at the
            # actual publication boundary. No timing-dependent concurrent test.
            with path.open("ab") as handle:
                handle.write(b"changed")
            return copied

        monkeypatch.setattr(sessions, "_copy_asset", mutate_after_copy)
        with pytest.raises(ValueError, match="changed"):
            service.tonal_mask(session["directory"], "derived", [0, 0, 1, 1], region_mask_id="region")
        assert "derived" not in sessions.read_session(session["directory"])["masks"]
        assert set(assets.iterdir()) == before
    finally:
        service.close()
