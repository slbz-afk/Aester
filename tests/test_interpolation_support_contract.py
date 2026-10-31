"""Independent input, ownership and boundary contract checks for support sampling.

Written before reviewing the worker implementation. The separate interpolation
audit owns the scalar interpolation oracle and multistage numerical experiments.
"""
import warnings

import numpy as np
import pytest

import starstack.interpolation_support as support_module
from starstack.interpolation_support import SUPPORT_VERSION, sample_support


@pytest.mark.parametrize("mask", [
    None, True, [], [[]], np.zeros((0, 3), bool), np.zeros((3, 0), bool),
    np.ones(3, bool), np.ones((2, 3, 1), bool), [[1, 0]], [[1.0, 0.0]],
    np.ones((2, 2), dtype=object), np.ones((2, 2), dtype=complex),
    np.array([["yes", "no"]]), [[True], [True, False]],
])
def test_invalid_masks_raise_value_error_even_for_no_points(mask):
    with pytest.raises(ValueError):
        sample_support(mask, np.empty((0, 2)))


@pytest.mark.parametrize("points", [
    None, 0, [], [0, 1], np.empty((0,)), np.zeros((1, 1)), np.zeros((1, 3)),
    np.zeros((2, 1, 2)), [[True, False]], [["1", "2"]],
    np.array([[1, 2]], dtype=object), np.array([[1, 2]], dtype=complex),
    [[0, np.nan]], [[np.inf, 0]], [[0, -np.inf]],
    [[0, 0], [np.nan, 0]], [[0, 0], [1]],
    np.empty((0, 2), dtype=object), np.empty((0, 2), dtype=bool),
    np.empty((0, 2), dtype=complex), np.empty((0, 2), dtype=str),
    np.array([["2026-01-01", "2026-01-02"]], dtype="datetime64[D]"),
])
def test_invalid_points_raise_value_error_without_partial_results(points):
    with pytest.raises(ValueError):
        sample_support(np.ones((3, 4), bool), points)


@pytest.mark.parametrize("keyword", ["kernel", "boundary"])
@pytest.mark.parametrize("bad", [None, 0, True, "", "unknown", [], {}, np.array([1, 2])])
def test_invalid_modes_raise_value_error_even_for_no_points(keyword, bad):
    with pytest.raises(ValueError):
        sample_support(np.ones((3, 4), bool), np.empty((0, 2)), **{keyword: bad})


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
@pytest.mark.parametrize("boundary", ["clamp", "strict"])
def test_zero_points_still_return_fresh_bool_vector(kernel, boundary):
    points = np.empty((0, 2))
    first = sample_support([[True]], points, kernel=kernel, boundary=boundary)
    second = sample_support([[True]], points, kernel=kernel, boundary=boundary)
    assert first.shape == (0,) and first.dtype == np.bool_
    assert first is not second and not np.shares_memory(first, points)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64, np.float16, np.float32, np.float64])
def test_real_numeric_point_dtypes_are_accepted(dtype):
    points = np.array([[0, 0], [3, 2]], dtype=dtype)
    result = sample_support(np.ones((3, 4), bool), points)
    np.testing.assert_array_equal(result, [True, True])


def test_top_down_xy_convention_and_boolean_lists():
    mask = [[True, False, True], [False, True, False]]
    points = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    np.testing.assert_array_equal(sample_support(mask, points), [True, False, True, False, True, False])


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
def test_integer_border_zero_taps_do_not_require_outside_pixels(kernel):
    points = [[0, 0], [3, 0], [0, 2], [3, 2]]
    np.testing.assert_array_equal(
        sample_support(np.ones((3, 4), bool), points, kernel=kernel, boundary="strict"),
        [True, True, True, True],
    )


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
def test_half_pixel_domain_is_lower_closed_upper_open(kernel):
    low_outside = np.nextafter(-0.5, -np.inf)
    right_inside = np.nextafter(3.5, -np.inf)
    bottom_inside = np.nextafter(2.5, -np.inf)
    points = [[-0.5, -0.5], [right_inside, bottom_inside],
              [low_outside, 0], [0, low_outside], [3.5, 0], [0, 2.5]]
    np.testing.assert_array_equal(
        sample_support(np.ones((3, 4), bool), points, kernel=kernel),
        [True, True, False, False, False, False],
    )


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
def test_finite_giant_coordinates_are_false_without_cast_warnings(kernel):
    giant = np.finfo(np.float64).max
    points = [[giant, 0], [-giant, 0], [0, giant], [0, -giant], [0, 0]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = sample_support(np.ones((3, 4), bool), points, kernel=kernel)
    np.testing.assert_array_equal(result, [False, False, False, False, True])


def test_uint64_extreme_is_false_without_overflow_or_wraparound():
    points = np.array([[np.iinfo(np.uint64).max, 0], [0, 0]], dtype=np.uint64)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(sample_support([[True]], points), [False, True])


def test_extended_float_domain_filter_precedes_narrowing():
    # On this Windows host longdouble has float64 width. The same check also
    # exercises the wider type on platforms that provide an extended dtype.
    scalar = np.longdouble
    giant = np.finfo(scalar).max
    right_inside = np.nextafter(scalar(3.5), scalar(-np.inf))
    left_outside = np.nextafter(scalar(-0.5), scalar(-np.inf))
    points = np.array([[giant, 0], [-giant, 0], [right_inside, 1], [left_outside, 1]], dtype=scalar)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(sample_support(np.ones((3, 4), bool), points),
                                          [False, False, True, False])


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
@pytest.mark.parametrize("boundary", ["clamp", "strict"])
def test_single_pixel_image_has_only_integer_sample_under_strict(kernel, boundary):
    points = [[0, 0], [-0.5, 0], [0.25, 0.25], [0.5, 0], [-0.500001, 0]]
    expected = [True, boundary == "clamp", boundary == "clamp", False, False]
    np.testing.assert_array_equal(sample_support([[True]], points, kernel=kernel, boundary=boundary), expected)


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
def test_mask_holes_are_dependencies_and_cannot_be_averaged_away(kernel):
    mask = np.ones((7, 7), bool)
    mask[3, 3] = False
    np.testing.assert_array_equal(sample_support(mask, [[3, 3], [2.5, 2.5], [0, 0]], kernel=kernel),
                                  [False, False, True])


def test_negative_cubic_tap_is_a_real_dependency():
    mask = np.ones((7, 7), bool)
    mask[3, 1] = False  # x=2.5 has a nonzero, negative coefficient at x=1.
    assert not sample_support(mask, [[2.5, 3]], kernel="cubic")[0]
    assert sample_support(mask, [[2.5, 3]], kernel="bilinear")[0]


def test_read_only_strided_inputs_are_accepted_and_never_mutated():
    mask = np.ones((8, 10), bool)[::2, ::2]
    mask[1, 1] = False
    points = np.array([[0.0, 0], [99, 99], [1, 1], [99, 99], [4, 3]])[::2]
    before_mask = mask.copy()
    before_points = points.copy()
    mask.flags.writeable = False
    points.flags.writeable = False
    result = sample_support(mask, points)
    np.testing.assert_array_equal(result, [True, False, True])
    np.testing.assert_array_equal(mask, before_mask)
    np.testing.assert_array_equal(points, before_points)
    assert result.flags.writeable
    assert not np.shares_memory(result, mask) and not np.shares_memory(result, points)
    result[:] = False
    np.testing.assert_array_equal(sample_support(mask, points), [True, False, True])


def test_defined_version_and_explicit_kernel_scope():
    assert SUPPORT_VERSION == "1.0.0"
    documentation = (support_module.__doc__ or "") + (sample_support.__doc__ or "")
    assert "Siril" in documentation
    assert "stage" in documentation.lower()
