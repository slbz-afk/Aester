import copy
import hashlib
import json
import math

import pytest

from starstack.finishing_recipe import normalize_recipe, recipe_digest

SCHEMA = "aester-finishing-recipe-1"
SHA = "a" * 64


def base_recipe(**overrides):
    recipe = {
        "schema": SCHEMA,
        "source": {
            "sha256": SHA,
            "width": 1920,
            "height": 1080,
            "encoding": "srgb16",
        },
        "intent": "",
        "operations": [],
    }
    recipe.update(overrides)
    return recipe


def make_op(**overrides):
    operation = {
        "id": "op1",
        "type": "exposure",
        "enabled": True,
        "mask_id": None,
        "params": {"ev": 0},
    }
    operation.update(overrides)
    return operation


def test_empty_operations_and_intent_are_valid():
    normalized = normalize_recipe(base_recipe())
    assert normalized["operations"] == []
    assert normalized["intent"] == ""


def test_normalize_is_independent_deep_copy_and_keeps_order():
    recipe = base_recipe(
        operations=[
            make_op(id="z_last", params={"ev": 1}),
            make_op(id="a_first", type="saturation", params={"scale": 1}),
        ]
    )
    original = copy.deepcopy(recipe)
    normalized = normalize_recipe(recipe)
    assert recipe == original
    assert normalized is not recipe
    assert normalized["source"] is not recipe["source"]
    assert normalized["operations"] is not recipe["operations"]
    assert [op["id"] for op in normalized["operations"]] == ["z_last", "a_first"]
    normalized["operations"][0]["params"]["ev"] = 99
    normalized["operations"].append({})
    normalized["source"]["width"] = 1
    assert recipe == original


def test_normalized_output_is_json_compatible():
    normalized = normalize_recipe(base_recipe(operations=[make_op(params={"ev": -0.0})]))
    json.dumps(normalized, ensure_ascii=False, allow_nan=False)


def test_unknown_and_missing_fields_are_rejected():
    recipe = base_recipe()
    recipe["extra"] = 1
    with pytest.raises(ValueError):
        normalize_recipe(recipe)

    for key in ("schema", "source", "intent", "operations"):
        recipe = base_recipe()
        del recipe[key]
        with pytest.raises(ValueError):
            normalize_recipe(recipe)

    recipe = base_recipe()
    recipe["source"]["extra"] = 1
    with pytest.raises(ValueError):
        normalize_recipe(recipe)

    recipe = base_recipe()
    del recipe["source"]["encoding"]
    with pytest.raises(ValueError):
        normalize_recipe(recipe)

    operation = make_op()
    operation["extra"] = 1
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[operation]))

    operation = make_op()
    del operation["enabled"]
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[operation]))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[make_op(params={"ev": 0, "extra": 0})]))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[make_op(params={})]))


@pytest.mark.parametrize("bad_schema", ["", "aester-finishing-recipe-4", None, 1, True])
def test_schema_must_be_exact_string(bad_schema):
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(schema=bad_schema))


def test_v2_unsharp_is_explicit_versioned_and_legacy_rejects():
    operation = make_op(type="local_contrast_unsharp", params={"sigma_px": 4, "amount": 0})
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[operation]))
    result = normalize_recipe(base_recipe(schema="aester-finishing-recipe-2", operations=[operation]))
    assert result["schema"] == "aester-finishing-recipe-2"
    assert result["operations"][0]["params"] == {"sigma_px": 4.0, "amount": 0.0}
    operation["params"] = {"sigma_px": 80, "amount": .6}
    normalize_recipe(base_recipe(schema="aester-finishing-recipe-2", operations=[operation]))


@pytest.mark.parametrize("params", [{"sigma_px": 3.9, "amount": 0}, {"sigma_px": 81, "amount": .3},
    {"sigma_px": True, "amount": .3}, {"sigma_px": 18, "amount": float("nan")},
    {"sigma_px": 18, "amount": -.1}, {"sigma_px": 18, "amount": .61},
    {"sigma_px": 18, "amount": False}, {"sigma_px": 18, "amount": .3, "threshold": 0}])
def test_v2_unsharp_bounds_even_when_disabled(params):
    operation = make_op(type="local_contrast_unsharp", enabled=False, params=params)
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(schema="aester-finishing-recipe-2", operations=[operation]))


def test_intent_boundaries_and_type():
    assert normalize_recipe(base_recipe(intent="x" * 4000))["intent"] == "x" * 4000
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(intent="x" * 4001))
    for bad in [None, 0, True, 1.0, ["x"]]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(intent=bad))


@pytest.mark.parametrize("bad_sha", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, "", None, 1, True])
def test_source_sha256_format(bad_sha):
    recipe = base_recipe()
    recipe["source"]["sha256"] = bad_sha
    with pytest.raises(ValueError):
        normalize_recipe(recipe)


def test_source_dimensions_and_product_boundaries():
    recipe = base_recipe()
    recipe["source"].update({"width": 1, "height": 1})
    assert normalize_recipe(recipe)["source"]["width"] == 1

    recipe = base_recipe()
    recipe["source"].update({"width": 100000, "height": 3000})
    assert normalize_recipe(recipe)["source"]["height"] == 3000

    for bad_width in [0, 100001, 1.0, True, None, 10**10000]:
        recipe = base_recipe()
        recipe["source"]["width"] = bad_width
        with pytest.raises(ValueError):
            normalize_recipe(recipe)

    recipe = base_recipe()
    recipe["source"].update({"width": 20000, "height": 15000})
    assert normalize_recipe(recipe)["source"]["height"] == 15000

    recipe = base_recipe()
    recipe["source"].update({"width": 20000, "height": 15001})
    with pytest.raises(ValueError):
        normalize_recipe(recipe)


@pytest.mark.parametrize("bad_encoding", ["SRGB16", "srgb8", "", None, 1, True])
def test_source_encoding_fixed(bad_encoding):
    recipe = base_recipe()
    recipe["source"]["encoding"] = bad_encoding
    with pytest.raises(ValueError):
        normalize_recipe(recipe)


def test_operations_count_and_type_boundaries():
    ops = [make_op(id=f"op{i}") for i in range(32)]
    assert len(normalize_recipe(base_recipe(operations=ops))["operations"]) == 32

    too_many = [make_op(id=f"op{i}") for i in range(33)]
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=too_many))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=(make_op(),)))


@pytest.mark.parametrize("bad_id", ["Op", "1op", "_op", "a" * 49, "", None, 1, True])
def test_operation_id_format(bad_id):
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[make_op(id=bad_id)]))


def test_operation_id_length_boundary():
    good = "a" + "b" * 47
    assert normalize_recipe(base_recipe(operations=[make_op(id=good)]))["operations"][0]["id"] == good
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[make_op(id="a" + "b" * 48)]))


def test_duplicate_ids_rejected_across_disabled_operations():
    operations = [make_op(id="same"), make_op(id="same", enabled=False)]
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=operations))


@pytest.mark.parametrize("bad_type", ["gegl:exposure", "blur", "", None, 1, True])
def test_operation_type_is_closed_set(bad_type):
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[make_op(type=bad_type)]))


@pytest.mark.parametrize("bad_enabled", [1, 0, "true", None, []])
def test_enabled_must_be_builtin_bool(bad_enabled):
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[make_op(enabled=bad_enabled)]))


def test_disabled_operation_still_validates_params():
    operation = make_op(enabled=False, params={"ev": 3})
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[operation]))


def test_mask_id_null_or_identifier():
    normalized = normalize_recipe(base_recipe(operations=[make_op(mask_id=None)]))
    assert normalized["operations"][0]["mask_id"] is None

    normalized = normalize_recipe(base_recipe(operations=[make_op(mask_id="mask_1")]))
    assert normalized["operations"][0]["mask_id"] == "mask_1"

    for bad in ["Mask", "1mask", "", 1, True, [], 0]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[make_op(mask_id=bad)]))


def test_exposure_ev_boundaries_and_float_canonicalization():
    normalized = normalize_recipe(base_recipe(operations=[make_op(params={"ev": -2})]))
    ev = normalized["operations"][0]["params"]["ev"]
    assert ev == -2.0 and type(ev) is float

    normalized = normalize_recipe(base_recipe(operations=[make_op(params={"ev": 2.0})]))
    assert normalized["operations"][0]["params"]["ev"] == 2.0

    normalized = normalize_recipe(base_recipe(operations=[make_op(params={"ev": -0.0})]))
    ev = normalized["operations"][0]["params"]["ev"]
    assert ev == 0.0
    assert math.copysign(1.0, ev) == 1.0

    for bad in [-2.0001, 2.0001, True, "0", None, float("nan"), float("inf"), -float("inf"), 10**10000]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[make_op(params={"ev": bad})]))


def test_exposure_params_must_contain_only_ev():
    for params in [{}, {"ev": 0, "amount": 0}, {"amount": 0}]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[make_op(params=params)]))


def test_saturation_scale_boundaries_and_canonicalization():
    normalized = normalize_recipe(base_recipe(operations=[make_op(type="saturation", params={"scale": 0})]))
    scale = normalized["operations"][0]["params"]["scale"]
    assert scale == 0.0 and math.copysign(1.0, scale) == 1.0

    normalized = normalize_recipe(base_recipe(operations=[make_op(type="saturation", params={"scale": 2})]))
    assert normalized["operations"][0]["params"]["scale"] == 2.0

    for bad in [-0.0001, 2.0001, True, "1", None, float("nan"), float("inf"), 10**10000]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[make_op(type="saturation", params={"scale": bad})]))


def test_saturation_params_must_contain_only_scale():
    for params in [{}, {"scale": 1, "extra": 1}, {"extra": 1}]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[make_op(type="saturation", params=params)]))


@pytest.mark.parametrize("channel", ["value", "red", "green", "blue"])
def test_curves_channels(channel):
    normalized = normalize_recipe(base_recipe(operations=[
        make_op(type="curves", params={"channel": channel, "points": [[0, 0], [1, 1]]})
    ]))
    assert normalized["operations"][0]["params"]["channel"] == channel


@pytest.mark.parametrize("bad_channel", ["luma", "alpha", "", None, 1, True])
def test_curves_channel_closed_set(bad_channel):
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": bad_channel, "points": [[0, 0], [1, 1]]})
        ]))


def test_curves_points_count_boundaries():
    normalized = normalize_recipe(base_recipe(operations=[
        make_op(type="curves", params={"channel": "value", "points": [[0, 0], [1, 1]]})
    ]))
    assert normalized["operations"][0]["params"]["points"] == [[0.0, 0.0], [1.0, 1.0]]

    thirty_two = [[i / 31, 0.5] for i in range(32)]
    normalized = normalize_recipe(base_recipe(operations=[
        make_op(type="curves", params={"channel": "value", "points": thirty_two})
    ]))
    assert len(normalized["operations"][0]["params"]["points"]) == 32

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": [[0, 0]]})
        ]))

    thirty_three = [[i / 32, 0.5] for i in range(33)]
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": thirty_three})
        ]))


@pytest.mark.parametrize("bad_point", [(0, 0), [0], [0, 0, 0], "00", None, 0])
def test_curves_point_shape(bad_point):
    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": [bad_point, [1, 1]]})
        ]))


def test_curves_point_values_must_be_finite_unit_numbers():
    for bad in [-0.0001, 1.0001, True, None, float("nan"), float("inf"), 10**10000]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[
                make_op(type="curves", params={"channel": "value", "points": [[0, 0], [1, bad]]})
            ]))
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[
                make_op(type="curves", params={"channel": "value", "points": [[0, 0], [bad, 0]]})
            ]))


def test_curves_endpoint_and_monotonicity_rules():
    for points in [
        [[0, 0.2], [1, 0.2]],
        [[0, 0.8], [1, 1]],
        [[0, 1], [1, 1]],
        [[0, 0], [0.5, 0.5], [1, 0.5]],
    ]:
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": points})
        ]))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": [[0.1, 0], [1, 1]]})
        ]))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": [[0, 0], [0.9, 1]]})
        ]))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": [[0, 0], [0.5, 0.5], [0.5, 0.6], [1, 1]]})
        ]))

    with pytest.raises(ValueError):
        normalize_recipe(base_recipe(operations=[
            make_op(type="curves", params={"channel": "value", "points": [[0, 0.8], [1, 0.2]]})
        ]))


def test_curves_numeric_canonicalization_including_negative_zero():
    normalized = normalize_recipe(base_recipe(operations=[
        make_op(type="curves", params={"channel": "red", "points": [[-0.0, -0.0], [0.5, 0.0], [1, -0.0]]})
    ]))
    points = normalized["operations"][0]["params"]["points"]
    assert points == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]
    for x, y in points:
        assert type(x) is float
        assert type(y) is float
        assert math.copysign(1.0, x) == 1.0
        assert math.copysign(1.0, y) == 1.0


def test_curves_params_must_contain_only_channel_and_points():
    valid_points = [[0, 0], [1, 1]]
    for params in [
        {},
        {"channel": "value"},
        {"points": valid_points},
        {"channel": "value", "points": valid_points, "extra": 1},
    ]:
        with pytest.raises(ValueError):
            normalize_recipe(base_recipe(operations=[make_op(type="curves", params=params)]))


def test_recipe_digest_matches_canonical_json_definition():
    recipe = base_recipe(intent="café", operations=[make_op(params={"ev": 1})])
    normalized_literal = {
        "schema": SCHEMA,
        "source": {"sha256": SHA, "width": 1920, "height": 1080, "encoding": "srgb16"},
        "intent": "café",
        "operations": [{
            "id": "op1",
            "type": "exposure",
            "enabled": True,
            "mask_id": None,
            "params": {"ev": 1.0},
        }],
    }
    expected = hashlib.sha256(
        json.dumps(
            normalized_literal,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    digest = recipe_digest(recipe)
    assert digest == expected
    assert len(digest) == 64
    assert digest == recipe_digest(recipe)


def test_recipe_digest_key_order_and_numeric_canonicalization():
    first = base_recipe(operations=[make_op(params={"ev": -0.0})])
    second = {
        "operations": [{
            "params": {"ev": 0},
            "mask_id": None,
            "enabled": True,
            "type": "exposure",
            "id": "op1",
        }],
        "intent": "",
        "source": {
            "encoding": "srgb16",
            "height": 1080,
            "width": 1920,
            "sha256": SHA,
        },
        "schema": SCHEMA,
    }
    assert recipe_digest(first) == recipe_digest(second)


def test_recipe_digest_preserves_list_order():
    a = base_recipe(operations=[make_op(id="a"), make_op(id="b")])
    b = base_recipe(operations=[make_op(id="b"), make_op(id="a")])
    assert recipe_digest(a) != recipe_digest(b)


def test_recipe_digest_changes_with_operations():
    a = base_recipe(operations=[make_op(params={"ev": 0})])
    b = base_recipe(operations=[make_op(params={"ev": 0.5})])
    assert recipe_digest(a) != recipe_digest(b)


def test_recipe_digest_rejects_invalid_recipe():
    with pytest.raises(ValueError):
        recipe_digest(None)
    with pytest.raises(ValueError):
        recipe_digest(base_recipe(intent=1))
