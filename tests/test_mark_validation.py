"""Independent boundary checks against the Worker task's input contract."""
from copy import deepcopy
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
import math
from types import MappingProxyType

import numpy as np
import pytest

from starstack.mark_validation import MAX_COORD, validate_mark_values


BASE = {"x0": 0, "y0": 1, "x1": 7, "y1": 9}


class IntegerSubclass(int):
    def __float__(self):
        raise RuntimeError("must not run subclass conversion")


class FloatSubclass(float):
    def __float__(self):
        raise RuntimeError("must not run subclass conversion")


class NumberEnum(IntEnum):
    ONE = 1


def rejected(mark, **kwargs):
    with pytest.raises(ValueError) as failure:
        validate_mark_values(mark, **kwargs)
    assert str(failure.value), "failure needs a readable reason"


def test_new_five_float_values_and_no_mutation():
    original = {**BASE, "frame": "light_00001", "extra": {"a": [1, 2]}}
    before = deepcopy(original)
    result = validate_mark_values(original)
    assert result == {"x0": 0.0, "y0": 1.0, "x1": 7.0, "y1": 9.0, "half_width_px": 30.0}
    assert all(type(value) is float for value in result.values())
    assert original == before and result is not original


def test_integer_and_float_equivalents_have_the_same_result():
    assert validate_mark_values(BASE) == validate_mark_values({k: float(v) for k, v in BASE.items()})


def test_negative_zero_is_positive_in_output():
    result = validate_mark_values({**BASE, "x0": -0.0, "y0": -0.0})
    assert math.copysign(1, result["x0"]) == 1
    assert math.copysign(1, result["y0"]) == 1


@pytest.mark.parametrize("mark", [None, [], (), "", 1, True, MappingProxyType(BASE)])
def test_only_dict_inputs_are_accepted(mark):
    rejected(mark)


@pytest.mark.parametrize("key", ["x0", "y0", "x1", "y1"])
def test_each_required_coordinate_is_required(key):
    mark = dict(BASE)
    del mark[key]
    rejected(mark)


@pytest.mark.parametrize("key", ["x0", "y0", "x1", "y1", "half_width_px"])
@pytest.mark.parametrize("value", [True, False, None, "5", math.nan, math.inf, -math.inf,
                                  10**400, Decimal(5), Fraction(5), np.float64(5), np.int64(5),
                                  IntegerSubclass(5), FloatSubclass(5), NumberEnum.ONE])
def test_invalid_numeric_values_fail_uniformly_without_coercion(key, value):
    rejected({**BASE, key: value})


@pytest.mark.parametrize("key", ["x0", "y0", "x1", "y1"])
@pytest.mark.parametrize("value", [-1, math.nextafter(0.0, -math.inf), MAX_COORD + 1,
                                  math.nextafter(float(MAX_COORD), math.inf)])
def test_coordinate_address_boundaries_do_not_clip(key, value):
    mark = {**BASE, key: value}
    rejected(mark)
    assert mark[key] == value


def test_largest_allowed_coordinate_can_be_checked_structurally():
    result = validate_mark_values({"x0": 0, "y0": 0, "x1": MAX_COORD, "y1": MAX_COORD})
    assert result["x1"] == result["y1"] == MAX_COORD
    rejected({"x0": 0, "y0": 0, "x1": MAX_COORD, "y1": MAX_COORD}, width=100, height=100)


@pytest.mark.parametrize("value", [4, 4.0, 200, 200.0])
def test_half_width_endpoints_are_inclusive(value):
    assert validate_mark_values({**BASE, "half_width_px": value})["half_width_px"] == value


@pytest.mark.parametrize("value", [0, -1, 3.999, 200.001, math.nextafter(4.0, -math.inf),
                                  math.nextafter(200.0, math.inf)])
def test_half_width_outside_contract_is_rejected(value):
    rejected({**BASE, "half_width_px": value})


@pytest.mark.parametrize("point", [(0, 0, -0.0, 0.0), (1, 2, 1.0, 2.0), (MAX_COORD,) * 4])
def test_identical_endpoints_have_no_line(point):
    rejected(dict(zip(("x0", "y0", "x1", "y1"), point)))


@pytest.mark.parametrize("dimensions", [{"width": 10}, {"height": 10},
                                        {"width": None, "height": 10}, {"width": 10, "height": None}])
def test_partial_frame_dimensions_are_rejected(dimensions):
    rejected(BASE, **dimensions)


@pytest.mark.parametrize("side", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1, True, False, 10.0, "10", math.nan, math.inf,
                                  2**31 + 1, 10**400, np.int64(10), IntegerSubclass(10), NumberEnum.ONE])
def test_frame_dimensions_require_bounded_builtin_integers(side, value):
    dimensions = {"width": 20, "height": 20, side: value}
    rejected(BASE, **dimensions)


def test_frame_address_convention_includes_last_pixel_center():
    assert validate_mark_values(BASE, width=8, height=10)["x1"] == 7.0
    rejected(BASE, width=7, height=10)
    rejected(BASE, width=8, height=9)
    validate_mark_values({"x0": 0, "y0": 0, "x1": 0, "y1": 1}, width=1, height=2)
    validate_mark_values({"x0": 0, "y0": 0, "x1": MAX_COORD, "y1": MAX_COORD}, width=2**31, height=2**31)


def test_invalid_frame_bounds_leave_input_intact():
    mark = {**BASE, "half_width_px": 20, "notes": ["keep this"]}
    before = deepcopy(mark)
    rejected(mark, width=2, height=2)
    assert mark == before
