"""Versioned layer operation grammar and shared nested dependency resolution."""
from copy import deepcopy

import pytest

from starstack.finishing_recipe import normalize_recipe, recipe_digest, recipe_mask_ids


def layer_recipe():
    return {"schema": "aester-finishing-recipe-3",
        "source": {"sha256": "a" * 64, "width": 41, "height": 33, "encoding": "srgb16"},
        "intent": "Structure with accepted color preserved",
        "operations": [{"id": "layers", "type": "astrophotographic_layers", "enabled": True,
            "mask_id": None, "params": {"bundle_id": "paired", "bundle_sha256": "b" * 64,
            "star_strength": 0, "milkyway_amount": 0, "milkyway_mask_id": None,
            "meteor_core_extra": 0, "meteor_glow_amount": 0}}]}


def test_zero_layer_recipe_is_explicit_and_stable():
    original = layer_recipe()
    result = normalize_recipe(original)
    assert result == original and result is not original
    assert result["operations"][0]["params"]["star_strength"] == 0.0
    assert recipe_digest(result) == recipe_digest(original)
    assert recipe_mask_ids(result) == set()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("field,bad", [("star_strength", -0.01), ("star_strength", 1.01),
    ("milkyway_amount", 3.01), ("meteor_core_extra", 2.01), ("meteor_glow_amount", 4.01),
    ("star_strength", True), ("meteor_core_extra", float("inf")), ("milkyway_amount", float("nan")),
    ("bundle_id", "../image.tif"), ("bundle_sha256", "A"*64), ("milkyway_mask_id", "../mask")])
def test_invalid_layer_parameters_rejected_even_disabled(field, bad, enabled):
    value = layer_recipe()
    value["operations"][0]["enabled"] = enabled
    value["operations"][0]["params"][field] = bad
    with pytest.raises(ValueError):
        normalize_recipe(value)


@pytest.mark.parametrize("schema", ["aester-finishing-recipe-1", "aester-finishing-recipe-2"])
def test_layers_not_silently_added_to_old_recipe_versions(schema):
    value = layer_recipe()
    value["schema"] = schema
    with pytest.raises(ValueError, match="unsupported"):
        normalize_recipe(value)


@pytest.mark.parametrize("mutation", ["after", "duplicate", "outer-mask", "missing-mw", "script"])
def test_layer_order_regions_and_no_arbitrary_execution(mutation):
    value = layer_recipe()
    op = value["operations"][0]
    if mutation == "after":
        value["operations"].insert(0, {"id": "exposure", "type": "exposure", "enabled": True,
                                      "mask_id": None, "params": {"ev": 0}})
    elif mutation == "duplicate":
        second = deepcopy(op); second["id"] = "layers-two"
        value["operations"].append(second)
    elif mutation == "outer-mask":
        op["mask_id"] = "sky"
    elif mutation == "missing-mw":
        op["params"]["milkyway_amount"] = 1
    else:
        op["params"]["script"] = "arbitrary.py"
    with pytest.raises(ValueError):
        normalize_recipe(value)


def test_nested_mask_collector_includes_disabled_dependencies():
    value = layer_recipe()
    value["operations"][0]["enabled"] = False
    value["operations"][0]["params"].update(milkyway_amount=1, milkyway_mask_id="galaxy")
    value["operations"].append({"id": "land", "type": "exposure", "enabled": True,
                                  "mask_id": "ground", "params": {"ev": .5}})
    assert recipe_mask_ids(value) == {"galaxy", "ground"}


def test_all_control_upper_endpoints_are_valid_explicit_experiments():
    value = layer_recipe()
    value["operations"][0]["params"].update(star_strength=1, milkyway_amount=3,
        milkyway_mask_id="galaxy", meteor_core_extra=2, meteor_glow_amount=4)
    assert normalize_recipe(value) == value
