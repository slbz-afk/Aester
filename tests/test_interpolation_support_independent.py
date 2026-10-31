"""Independent scalar acceptance of explicitly defined interpolation dependencies."""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from starstack.interpolation_support import sample_support


def scalar_kernel(distance, kind):
    z = abs(float(distance))
    if kind == "bilinear":
        return max(0., 1.-z)
    if z < 1.:
        return 1.-2.5*z*z+1.5*z*z*z
    if z < 2.:
        return 2.-4.*z+2.5*z*z-.5*z*z*z
    return 0.


def oracle(mask, x, y, kernel, boundary):
    h, w = mask.shape
    if not (-.5 <= x < w-.5 and -.5 <= y < h-.5):
        return False
    offsets = (0, 1) if kernel == "bilinear" else (-1, 0, 1, 2)
    ix, iy = math.floor(x), math.floor(y)
    for dy in offsets:
        for dx in offsets:
            coefficient = scalar_kernel(x-ix-dx, kernel)*scalar_kernel(y-iy-dy, kernel)
            if coefficient == 0.:
                continue
            xx, yy = ix+dx, iy+dy
            if boundary == "strict" and (xx < 0 or xx >= w or yy < 0 or yy >= h):
                return False
            if not mask[min(max(yy, 0), h-1), min(max(xx, 0), w-1)]:
                return False
    return True


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
@pytest.mark.parametrize("boundary", ["clamp", "strict"])
def test_dense_independent_scalar_oracle(kernel, boundary):
    mask = np.ones((13, 17), dtype=bool)
    mask[:, 9] = False
    mask[3, 4] = False
    xy = np.array([(x, y) for x in np.arange(-.5, 17., .25) for y in np.arange(-.5, 13., .25)])
    expected = np.array([oracle(mask, x, y, kernel, boundary) for x, y in xy])
    np.testing.assert_array_equal(sample_support(mask, xy, kernel=kernel, boundary=boundary), expected)


@pytest.mark.parametrize("kernel", ["bilinear", "cubic"])
@pytest.mark.parametrize("boundary", ["clamp", "strict"])
def test_integer_zero_taps_do_not_reject_borders(kernel, boundary):
    mask = np.eye(4, dtype=bool)
    points = np.array([[0., 0.], [1., 1.], [3., 3.], [2., 1.]])
    np.testing.assert_array_equal(sample_support(mask, points, kernel=kernel, boundary=boundary), [True, True, True, False])


def test_negative_tap_reaches_invalid_even_when_nearest_valid():
    mask = np.ones((8, 9), dtype=bool)
    mask[4, 5] = False
    # x=3.25 has a negative contribution from x=5, outside bilinear support.
    assert sample_support(mask, [[3.25, 4.]], kernel="bilinear").tolist() == [True]
    assert sample_support(mask, [[3.25, 4.]], kernel="cubic").tolist() == [False]


@pytest.mark.parametrize("boundary", ["clamp", "strict"])
def test_rotation_and_two_stage_propagation(boundary):
    y, x = np.indices((21, 25))
    mask = x+y < 27
    points = np.column_stack((x.ravel(), y.ravel())).astype(float)
    t = math.radians(3.)
    rotation = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    first = (points-[12., 10.]) @ rotation.T+[12.25, 9.75]
    second = (points-[12.25, 9.75]) @ rotation+[12., 10.]
    truth1 = np.array([oracle(mask, a, b, "cubic", boundary) for a, b in first]).reshape(mask.shape)
    actual1 = sample_support(mask, first, boundary=boundary).reshape(mask.shape)
    np.testing.assert_array_equal(actual1, truth1)
    truth2 = np.array([oracle(truth1, a, b, "cubic", boundary) for a, b in second])
    actual2 = sample_support(actual1, second, boundary=boundary)
    np.testing.assert_array_equal(actual2, truth2)
    assert (mask.ravel() & ~actual2).any()


def test_two_stage_net_identity_is_not_dependency_identity():
    y, x = np.indices((9, 12))
    mask = x < 5
    p = np.column_stack((x.ravel(), y.ravel())).astype(float)
    stage1 = sample_support(mask, p+[.25, 0.]).reshape(mask.shape)
    final = sample_support(stage1, p+[-.25, 0.]).reshape(mask.shape)
    assert mask[4, 4] and not final[4, 4]
    assert final[4, 1]


def test_giant_coordinates_filtered_before_integer_conversion():
    with np.errstate(all="raise"):
        answer = sample_support(np.ones((2, 2), dtype=bool), [[1e308, 0], [-1e308, 0], [0, 1e308], [0, -1e308]])
    np.testing.assert_array_equal(answer, [False]*4)


@pytest.mark.parametrize("point,clamp,strict", [([-.5, 0], True, False), ([1.5, 0], False, False),
                                                    ([0, -.5], True, False), ([0, 1.5], False, False),
                                                    ([0, 0], True, True), ([1, 1], True, True)])
def test_half_open_position_and_border_policies(point, clamp, strict):
    mask = np.ones((2, 2), dtype=bool)
    assert sample_support(mask, [point]).tolist() == [clamp]
    assert sample_support(mask, [point], boundary="strict").tolist() == [strict]


def test_inputs_not_mutated_and_empty_is_fresh_boolean():
    mask = np.ones((4, 4), dtype=bool)
    points = np.array([[1.25, 1.25]])
    before_mask, before_points = mask.copy(), points.copy()
    sample_support(mask, points)
    np.testing.assert_array_equal(mask, before_mask)
    np.testing.assert_array_equal(points, before_points)
    empty = sample_support(mask, np.empty((0, 2)))
    assert empty.shape == (0,) and empty.dtype == bool


@pytest.mark.parametrize("mask", [np.ones((2, 2), int), np.ones((2, 2), float), np.ones((2, 2), object),
                                 np.ones((0, 2), bool), np.ones(4, bool), True, None])
def test_invalid_validity(mask):
    with pytest.raises(ValueError):
        sample_support(mask, [[0., 0.]])


@pytest.mark.parametrize("points", [[], [0, 0], [[0, 0, 0]], [[True, False]], [["0", "0"]],
                                    [[0j, 0j]], [[np.nan, 0]], [[0, np.inf]], [[0, -np.inf]], None])
def test_invalid_points(points):
    with pytest.raises(ValueError):
        sample_support(np.ones((3, 3), dtype=bool), points)


@pytest.mark.parametrize("kwargs", [{"kernel": "lanczos"}, {"boundary": "wrap"}, {"kernel": None}, {"boundary": None},
                                    {"kernel": []}, {"boundary": []}])
def test_empty_points_still_validate_options(kwargs):
    with pytest.raises(ValueError):
        sample_support(np.ones((3, 3), dtype=bool), np.empty((0, 2)), **kwargs)
