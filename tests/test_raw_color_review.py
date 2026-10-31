"""Regression coverage for reviewed recipe portability and arithmetic errors.

ULP perturbations emulate a different math backend; they are not a claim that
cross-platform BLAS execution has been tested on this machine.
"""
from copy import deepcopy

import numpy as np
import pytest

from starstack import raw_color


@pytest.fixture
def metadata():
    mixing = np.array([[1., 0., 0.], [.2, .8, 0.], [0., .25, .75]])
    cam_xyz = np.vstack((mixing @ np.linalg.inv(raw_color.XYZ_FROM_LINEAR_SRGB), np.zeros(3)))
    fields = {"num_colors": 3, "color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]],
              "rgb_xyz_matrix": cam_xyz.tolist(), "camera_whitebalance": [2., 1., 1.5, 0.]}
    return {"available": True, "white_balance_applied": False,
            "color_correction_applied": False, "versions": {"libraw": [0, 22, 1]},
            "fields": {key: {"value": value} for key, value in fields.items()}}


@pytest.fixture
def recipe(metadata):
    return raw_color.build_recipe(metadata, "a" * 64, common_scale=1000)


@pytest.mark.parametrize("field", ["values", "row_sums", "daylight_normalization_diagnostic",
                                   "condition_number"])
def test_last_bit_derived_changes_are_accepted_and_preserved(recipe, field):
    changed = np.nextafter(np.asarray(recipe["matrix"][field]), np.inf).tolist()
    recipe["matrix"][field] = changed
    original = deepcopy(recipe)
    accepted = raw_color.validate_recipe(recipe)
    assert accepted == original and recipe == original
    assert accepted is not recipe


def test_near_zero_matrix_entry_is_preserved_and_used_in_render(metadata):
    recipe = raw_color.build_recipe(metadata, "a" * 64, common_scale=1)
    pristine = deepcopy(recipe)
    assert abs(recipe["matrix"]["values"][0][1]) < 1e-15
    recipe["matrix"]["values"][0][1] = 1e-14
    recipe["frame_compatibility"] = {"caller_checked": True}
    assert raw_color.validate_recipe(recipe) == recipe
    camera = np.array([[[0., 1e10, 0.]]], dtype=np.float32)
    factors = np.asarray(recipe["white_balance"]["gains_rgb"], dtype=np.float32)
    expected = (camera * factors) @ np.asarray(recipe["matrix"]["values"], np.float32).T
    result = raw_color.apply_color(camera, recipe)
    np.testing.assert_array_equal(result, expected)
    assert abs(float(result[0, 0, 0] - raw_color.apply_color(camera, pristine)[0, 0, 0])) > 9e-5


@pytest.mark.parametrize("field", ["values", "row_sums", "daylight_normalization_diagnostic",
                                   "condition_number"])
def test_material_changes_still_rejected(recipe, field):
    recipe["matrix"][field] = (np.asarray(recipe["matrix"][field]) * (1 + 1e-6)).tolist()
    with pytest.raises(ValueError, match="matrix"):
        raw_color.validate_recipe(recipe)


def test_near_zero_tolerance_is_small_not_blanket_acceptance(recipe):
    recipe["matrix"]["values"][0][1] = 1e-10
    with pytest.raises(ValueError, match="matrix"):
        raw_color.validate_recipe(recipe)


@pytest.mark.parametrize("field,invalid", [
    ("values", [[True, 0, 0], [0, 1., 0], [0, 0, 1.]]),
    ("values", [["1.0", 0, 0], [0, 1., 0], [0, 0, 1.]]),
    ("values", [1., 0., 0.]),
    ("values", [[1., 2.], [3.]]),
    ("values", [[float("nan")] * 3] * 3),
    ("values", [[float("inf")] * 3] * 3),
    ("row_sums", [True, 1., 1.]),
    ("row_sums", ["1", 1., 1.]),
    ("row_sums", [[1., 1., 1.]]),
    ("row_sums", [0., 0., 0.]),
    ("daylight_normalization_diagnostic", [float("nan"), 1., 1.]),
    ("condition_number", True),
    ("condition_number", "1.5"),
    ("condition_number", [1.5]),
    ("condition_number", float("inf")),
    ("condition_number", float("nan")),
])
def test_derived_fields_reject_impostors_shapes_and_nonfinite(recipe, field, invalid):
    recipe["matrix"][field] = invalid
    with pytest.raises(ValueError):
        raw_color.validate_recipe(recipe)


@pytest.mark.parametrize("field", ["values", "row_sums", "daylight_normalization_diagnostic",
                                   "condition_number"])
def test_numeric_string_with_identical_value_is_not_a_coefficient(recipe, field):
    value = recipe["matrix"][field]
    if field == "condition_number":
        recipe["matrix"][field] = repr(value)
    elif field == "values":
        value[0][0] = repr(value[0][0])
    else:
        value[0] = repr(value[0])
    with pytest.raises(ValueError, match="matrix"):
        raw_color.validate_recipe(recipe)


@pytest.mark.parametrize("field", ["values", "row_sums", "daylight_normalization_diagnostic"])
def test_true_cannot_impersonate_near_one_derived_value(recipe, field):
    value = recipe["matrix"][field]
    target = value[0] if field == "values" else value
    assert abs(target[0] - 1) < 1e-14
    target[0] = True
    with pytest.raises(ValueError, match="matrix"):
        raw_color.validate_recipe(recipe)


@pytest.mark.parametrize("mutation", [
    lambda r: r["matrix"].pop("direction"),
    lambda r: r["matrix"].update(unrecognized_coefficient=1.),
    lambda r: r["matrix"].update(direction="XYZ_to_camera"),
    lambda r: r["matrix"].update(daylight_normalization_applied_separately=0),
    lambda r: r["conditions"].update(input_black_subtracted_required=1),
    lambda r: r["white_balance"]["gains_rgb"].__setitem__(1, True),
    lambda r: r["matrix"]["cam_xyz_reported"][0].__setitem__(0, np.nextafter(r["matrix"]["cam_xyz_reported"][0][0], np.inf)),
    lambda r: r["camera"]["model"]["cam_xyz"][0].__setitem__(0, np.nextafter(r["camera"]["model"]["cam_xyz"][0][0], np.inf)),
])
def test_input_labels_structure_and_boolean_types_remain_strict(recipe, mutation):
    mutation(recipe)
    with pytest.raises(ValueError):
        raw_color.validate_recipe(recipe)


@pytest.mark.parametrize("field,indices", [("rgb_xyz_matrix", (0, 0)),
                                          ("raw_pattern", (0, 1)),
                                          ("camera_whitebalance", (1,))])
def test_build_does_not_coerce_mixed_boolean_metadata_to_numbers(metadata, field, indices):
    target = metadata["fields"][field]["value"]
    for index in indices[:-1]:
        target = target[index]
    target[indices[-1]] = True
    with pytest.raises(ValueError, match="布尔"):
        raw_color.build_recipe(metadata, "a" * 64, common_scale=1)


@pytest.mark.parametrize("condition,accepted", [(9e5, True), (1.1e6, False)])
def test_condition_cutoff_is_not_loosened(condition, accepted):
    # Symmetric row-stochastic mixing has eigenvalues 1, e, e, hence cond=1/e.
    e = 1 / condition
    mixing = (1-e) * np.ones((3, 3)) / 3 + e * np.eye(3)
    cam_xyz = np.vstack((mixing @ np.linalg.inv(raw_color.XYZ_FROM_LINEAR_SRGB), np.zeros(3)))
    if accepted:
        result = raw_color.cam_xyz_to_rgb_cam(cam_xyz)
        assert result["condition_number"] < 1e6
    else:
        with pytest.raises(ValueError, match="条件"):
            raw_color.cam_xyz_to_rgb_cam(cam_xyz)


def test_recorded_condition_above_hard_limit_rejected_even_within_tolerance(metadata, monkeypatch):
    # A controlled backend result isolates the boundary from SVD rounding.
    monkeypatch.setattr(raw_color.np.linalg, "cond", lambda _: 1e6)
    recipe = raw_color.build_recipe(metadata, "a" * 64, common_scale=1)
    recipe["matrix"]["condition_number"] = float(np.nextafter(1e6, np.inf))
    with pytest.raises(ValueError, match="matrix"):
        raw_color.validate_recipe(recipe)


@pytest.mark.parametrize("scale", [1e-320, 1e-40])
def test_factor_division_and_float32_cast_overflow_are_value_errors(metadata, scale):
    recipe = raw_color.build_recipe(metadata, "a" * 64, common_scale=scale)
    camera = np.ones((1, 1, 3), np.float32)
    original = camera.copy()
    with pytest.raises(ValueError, match="float32") as raised:
        raw_color.apply_color(camera, recipe)
    assert isinstance(raised.value.__cause__, FloatingPointError)
    np.testing.assert_array_equal(camera, original)


def test_wb_multiplication_overflow_is_value_error(metadata):
    recipe = raw_color.build_recipe(metadata, "a" * 64, common_scale=1)
    camera = np.array([[[np.finfo(np.float32).max, 0, 0]]], np.float32)
    original = camera.copy()
    with pytest.raises(ValueError, match="float32"):
        raw_color.apply_color(camera, recipe)
    np.testing.assert_array_equal(camera, original)


def test_matrix_multiplication_overflow_is_value_error(metadata):
    metadata["fields"]["camera_whitebalance"]["value"] = [1., 1., 1., 1.]
    recipe = raw_color.build_recipe(metadata, "a" * 64, common_scale=1)
    camera = np.array([[[0, 0, np.finfo(np.float32).max * .9]]], np.float32)
    original = camera.copy()
    # Factors are one, so only the final analytical B gain (4/3) can overflow.
    assert np.isfinite(camera).all()
    with pytest.raises(ValueError):
        raw_color.apply_color(camera, recipe)
    np.testing.assert_array_equal(camera, original)
