"""Fixed color math, safety contracts, and an optional live LibRaw DNG oracle."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from starstack.raw_color import (XYZ_FROM_LINEAR_SRGB, apply_color, build_recipe,
                                cam_xyz_to_rgb_cam, validate_recipe)


@pytest.fixture
def metadata():
    # An independently specified triangular color-mixing camera. Its analytical
    # inverse is checked below; this is not an assertion against our own output.
    mixing = np.array([[1., 0., 0.], [.2, .8, 0.], [0., .25, .75]])
    cam_xyz = np.vstack((mixing @ np.linalg.inv(XYZ_FROM_LINEAR_SRGB), np.zeros(3)))
    values = {"num_colors": 3, "color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]],
              "rgb_xyz_matrix": cam_xyz.tolist(), "camera_whitebalance": [2., 1., 1.5, 1.]}
    return {"available": True, "source": "libraw_reported", "white_balance_applied": False,
            "color_correction_applied": False, "versions": {"rawpy": "0.27.1", "libraw": [0, 22, 1]},
            "fields": {name: {"value": value} for name, value in values.items()}}


@pytest.fixture
def recipe(metadata):
    return build_recipe(metadata, "a" * 64, 1100, 100)


def test_camxyz_construction_matches_known_inverse_not_matrix_name(metadata):
    converted = cam_xyz_to_rgb_cam(metadata["fields"]["rgb_xyz_matrix"]["value"])
    expected = [[1., 0., 0.], [-.25, 1.25, 0.], [1 / 12, -5 / 12, 4 / 3]]
    np.testing.assert_allclose(converted["values"], expected, atol=1e-14)
    assert converted["daylight_normalization_applied_separately"] is False
    assert converted["direction"].startswith("white_balanced_camera_RGB_to_")


def test_matrix_normalizes_response_rows_instead_of_directly_inverting_camxyz(metadata):
    cam_xyz = np.asarray(metadata["fields"]["rgb_xyz_matrix"]["value"])
    cam_xyz[:3] *= np.array([2., 3., 4.])[:, None]
    result = cam_xyz_to_rgb_cam(cam_xyz)
    np.testing.assert_allclose(result["row_sums"], [2, 3, 4], atol=1e-14)
    np.testing.assert_allclose(np.asarray(result["values"]) @ np.ones(3), np.ones(3), atol=1e-14)
    np.testing.assert_allclose(result["values"], [[1, 0, 0], [-.25, 1.25, 0], [1/12, -5/12, 4/3]], atol=1e-14)


def test_fixed_recipe_neutral_and_known_color_patches(recipe):
    # DN/scale -> WB -> known triangular inverse. Includes a neutral patch.
    camera = np.array([[[125., 250., 250/1.5], [50., 260., 600/1.5]]], dtype=np.float32)
    result = apply_color(camera, recipe)
    # Second balanced camera vector is [.1,.26,.6]; analytical inverse -> [.1,.3,.7].
    np.testing.assert_allclose(result, [[[.25, .25, .25], [.1, .3, .7]]], atol=8e-8)
    assert result.dtype == np.float32
    assert recipe["reference"]["common_black_for_scale_DN"] == 100
    assert recipe["scale"]["common_denominator_DN"] == 1000
    assert json.loads(json.dumps(recipe, allow_nan=False)) == recipe


def test_zero_g2_is_explicit_reference_mapping_without_altering_metadata(metadata):
    metadata["fields"]["camera_whitebalance"]["value"][3] = 0.
    before = deepcopy(metadata)
    recipe = build_recipe(metadata, reference_sha256="b"*64, common_scale=1000)
    assert recipe["white_balance"]["reported_four_slots"] == [2., 1., 1.5, 0.]
    assert recipe["white_balance"]["green_mapping"] == "explicit_LibRaw_zero_G2_uses_G1"
    assert recipe["white_balance"]["coefficient_slot_mapping"] == [0, 1, 2, 1]
    assert recipe["white_balance"]["gains_rgb"] == [2., 1., 1.5]
    assert metadata == before


def test_four_reported_wb_normalizes_by_green_not_by_channel_extrema(metadata):
    metadata["fields"]["camera_whitebalance"]["value"] = [1771., 1024., 1920., 1024.]
    result = build_recipe(metadata, "a"*64, common_scale=1000)
    assert result["white_balance"]["gains_rgb"] == [1771/1024, 1., 1920/1024]


def test_frame_wb_changes_do_not_change_or_invalidate_fixed_reference(metadata, recipe):
    frame = deepcopy(metadata)
    frame["fields"]["camera_whitebalance"]["value"] = [9., 2., 3., 2.]
    before = deepcopy(recipe)
    assert validate_recipe(recipe, frame) == before
    assert recipe == before
    changed = build_recipe(frame, "b"*64, common_scale=1000)
    assert changed["white_balance"]["gains_rgb"] != recipe["white_balance"]["gains_rgb"]


@pytest.mark.parametrize("field,value", [
    ("num_colors", 4), ("num_colors", True), ("color_desc", "RGBE"),
    ("raw_pattern", [[0, 2], [1, 1]]),
    ("rgb_xyz_matrix", np.zeros((4, 3)).tolist()),
    ("camera_whitebalance", [2., 1., 1.5, 1.00000001]),
    ("camera_whitebalance", [2., 0., 1.5, 0.]),
    ("camera_whitebalance", [2., 1., 1.5, float("nan")]),
    ("camera_whitebalance", [2., 1., 1.5]),
])
def test_unsupported_model_or_wb_cannot_be_silently_fixed(metadata, field, value):
    metadata["fields"][field]["value"] = value
    with pytest.raises(ValueError):
        build_recipe(metadata, "a"*64, common_scale=1000)


def test_fourth_matrix_row_cannot_be_discarded(metadata):
    metadata["fields"]["rgb_xyz_matrix"]["value"][3] = [.1, .1, .1]
    with pytest.raises(ValueError, match="第四"):
        build_recipe(metadata, "a"*64, common_scale=1000)


def test_nearly_singular_matrix_is_rejected():
    mixing = np.array([[1., 0., 0.], [1., 1e-9, 0.], [1., 0., 1e-9]])
    c = np.vstack((mixing @ np.linalg.inv(XYZ_FROM_LINEAR_SRGB), np.zeros(3)))
    with pytest.raises(ValueError, match="奇异|条件"):
        cam_xyz_to_rgb_cam(c)


@pytest.mark.parametrize("kwargs", [
    {"common_scale": 0}, {"common_scale": -1}, {"common_scale": True},
    {"common_scale": float("nan")}, {"reference_white": 100, "reference_black_green": 100},
    {"reference_white": 100, "reference_black_green": -1},
    {"reference_white": 100, "reference_black_green": 10, "common_scale": 80},
])
def test_invalid_scale_is_not_inferred_from_image(metadata, kwargs):
    with pytest.raises(ValueError):
        build_recipe(metadata, "a"*64, **kwargs)


@pytest.mark.parametrize("sha", [None, "not-a-hash", "a"*63])
def test_recipe_requires_real_sha256_shape(metadata, sha):
    with pytest.raises(ValueError):
        build_recipe(metadata, sha, common_scale=1000)


def test_validate_recipe_allows_pipeline_extension_but_checks_matrix_and_frame(metadata, recipe):
    recipe["frame_compatibility"] = {"geometry": "caller-owned-check"}
    assert validate_recipe(recipe, metadata) == recipe
    frame = deepcopy(metadata)
    frame["fields"]["rgb_xyz_matrix"]["value"][0][0] += .01
    with pytest.raises(ValueError, match="不兼容"):
        validate_recipe(recipe, frame)
    recipe["matrix"]["values"][0][0] += .001
    with pytest.raises(ValueError, match="matrix"):
        validate_recipe(recipe)


def test_serialized_direction_or_state_changes_are_rejected(recipe):
    recipe["matrix"]["direction"] = "XYZ_to_RGB"
    with pytest.raises(ValueError):
        validate_recipe(recipe)
    with pytest.raises(ValueError):
        validate_recipe(None)


def test_signed_and_overrange_values_survive_without_mutating_source(recipe):
    camera = np.array([[[-100, 2000, 300], [400, -100, 500]],
                       [[2000, 100, -200], [100, 200, 300]]], dtype=np.float32)
    original = camera.copy()
    out = np.empty_like(camera)
    result = apply_color(camera, recipe, out=out, block_rows=1)
    analytical = np.array([[1, 0, 0], [-.25, 1.25, 0], [1/12, -5/12, 4/3]])
    expected = (camera.astype(float) * np.array([2, 1, 1.5]) / 1000) @ analytical.T
    np.testing.assert_allclose(result, expected, atol=4e-7)
    assert result is out and np.any(out < 0) and np.any(out > 1)
    np.testing.assert_array_equal(camera, original)
    np.testing.assert_allclose(apply_color(camera*2, recipe), result*2, atol=8e-7)


def test_chunk_sizes_do_not_change_result_or_row_order(recipe):
    camera = np.random.default_rng(23).normal(100, 500, (19, 13, 3)).astype(np.float32)
    np.testing.assert_array_equal(apply_color(camera, recipe, block_rows=1),
                                  apply_color(camera, recipe, block_rows=17))


@pytest.mark.parametrize("kwargs", [{"row_order": "BOTTOM-UP"}, {"input_state": "linear_sRGB"},
                                    {"block_rows": True}, {"block_rows": 0}])
def test_wrong_processing_contract_is_rejected(recipe, kwargs):
    with pytest.raises(ValueError):
        apply_color(np.ones((2, 2, 3), np.float32), recipe, **kwargs)


@pytest.mark.parametrize("array", [np.ones((2, 2, 3), np.uint16), np.ones((2, 2, 3), np.float64),
                                   np.ones((2, 2), np.float32), np.full((2, 2, 3), np.nan, np.float32)])
def test_wrong_shape_dtype_or_nonfinite_data_is_rejected(recipe, array):
    with pytest.raises(ValueError):
        apply_color(array, recipe)


def test_out_cannot_alias_original_memory_or_backing_file(recipe, tmp_path):
    original = np.ones((2, 2, 3), np.float32)
    with pytest.raises(ValueError):
        apply_color(original, recipe, out=original)
    file = tmp_path / "camera.dat"
    a = np.memmap(file, dtype=np.float32, mode="w+", shape=(2, 2, 3))
    a[:] = 100
    a.flush()
    b = np.memmap(file, dtype=np.float32, mode="r+", shape=a.shape)
    with pytest.raises(ValueError, match="映射文件"):
        apply_color(a, recipe, out=b)
    del a, b


@pytest.mark.parametrize("override_saturation", [False, True])
def test_live_libraw_unsaturated_color_patches(tmp_path, override_saturation):
    """Actual rawpy/LibRaw processing of a tiny, explicitly synthetic Nikon DNG.

    Uniform interiors isolate WB/scale/matrix from interpolation and clipping.
    This does not validate all DNG profiles or photographic rendering quality.
    """
    rawpy = pytest.importorskip("rawpy")
    import tifffile
    from starstack.raw_metadata import read_raw_metadata

    path = tmp_path / "synthetic_nikon_color_patches.dng"
    signal = np.array([[600, 800, 1000], [1000, 800, 600], [800, 800, 800]], dtype=np.uint16)
    pattern = np.array([[0, 1], [1, 2]])
    cfa = np.empty((64, 96), dtype=np.uint16)
    for y in range(64):
        for x in range(96):
            cfa[y, x] = 64 + signal[x // 32, pattern[y % 2, x % 2]]
    cm = [11423, -4564, -1123, -4816, 12895, 2119, -210, 1061, 7282]
    tags = [(271, "s", 0, "NIKON CORPORATION", False), (272, "s", 0, "NIKON Z 8", False),
            (50708, "s", 0, "Nikon Z 8", False), (50706, "B", 4, (1, 4, 0, 0), False),
            (50707, "B", 4, (1, 3, 0, 0), False), (33421, "H", 2, (2, 2), False),
            (33422, "B", 4, (0, 1, 1, 2), False), (50710, "B", 3, (0, 1, 2), False),
            (50713, "H", 2, (1, 1), False), (50714, "2I", 1, (64, 1), False),
            (50717, "I", 1, 4095, False), (50829, "I", 4, (0, 0, 64, 96), False),
            (50721, "2i", 9, tuple(v for c in cm for v in (c, 10000)), False),
            (50778, "H", 1, 21, False), (50728, "2I", 3, (1, 2, 1, 1, 2, 3), False)]
    tifffile.imwrite(path, cfa, photometric=32803, metadata=None, extratags=tags)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = read_raw_metadata(path)
    recipe = build_recipe(metadata, digest, 4095, 64)
    # On this inline-black Bayer path, dcraw_process applies user_sat after
    # raw2image_ex has subtracted black and reset C.black to zero. Thus the
    # explicit override is the remaining signal range, not the file white DN.
    # https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/dcraw_process.cpp#L78-L88
    # https://github.com/LibRaw/LibRaw/blob/0.22.1/src/preprocessing/raw2image.cpp#L527-L533
    saturation = {"user_black": 64, "user_cblack": [0, 0, 0, 0], "user_sat": 4095-64} if override_saturation else {}
    # Z8's populated cmatrix is an independent decoder-produced matrix check.
    with rawpy.imread(str(path)) as raw:
        np.testing.assert_allclose(recipe["matrix"]["values"], raw.color_matrix[:, :3], atol=2e-7)
        reference = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, user_wb=[2, 1, 1.5, 1],
            use_camera_wb=False, use_auto_wb=False,
            output_color=rawpy.ColorSpace.sRGB, output_bps=16, gamma=(1, 1),
            no_auto_bright=True, no_auto_scale=False, adjust_maximum_thr=0,
            bright=1, user_flip=0, highlight_mode=rawpy.HighlightMode.Clip,
            median_filter_passes=0,
            **saturation,
        ).astype(np.float32) / 65535
    assert reference.shape == (64, 96, 3)
    camera = np.broadcast_to(np.repeat(signal, 32, axis=0)[None], (64, 96, 3)).astype(np.float32)
    ours = apply_color(camera, recipe)
    assert np.all((ours > .01) & (ours < .95))  # No earlier/later clipping involved.
    for x in (8, 40, 72):
        np.testing.assert_allclose(ours[8:56, x:x+16], reference[8:56, x:x+16], atol=5/65535, rtol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
