"""Independent scalar acceptance of the DeepSeek rectangle-query delivery.

Expected values use literal per-pixel rectangles; they do not call either
implementation function to generate the oracle.
"""
import math
import warnings

import numpy as np
import pytest

from starstack.support_bounds import invalid_integral, rectangle_support


def _oracle(mask, points, before, after):
    height, width = mask.shape
    result = []
    for x, y in points:
        if x < before or y < before or x >= width - after or y >= height - after:
            result.append(False)
            continue
        left, top = math.floor(x) - before, math.floor(y) - before
        right, bottom = math.floor(x) + after, math.floor(y) + after
        result.append(all(bool(mask[yy, xx]) for yy in range(top, bottom + 1) for xx in range(left, right + 1)))
    return np.asarray(result, dtype=bool)


@pytest.mark.parametrize("shape", [(1, 1), (2, 5), (9, 11), (32, 25)])
@pytest.mark.parametrize("layout", ["plain", "transpose", "reverse", "strided"])
def test_integral_literal_counts_nonmutating(shape, layout):
    rng = np.random.default_rng(440)
    base = rng.random(shape) > .31
    mask = {"plain": base, "transpose": base.T, "reverse": base[::-1, ::-1], "strided": base[::2, ::2]}[layout]
    before = mask.copy()
    table = invalid_integral(mask)
    expected = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.uint64)
    for y in range(expected.shape[0]):
        for x in range(expected.shape[1]):
            expected[y, x] = sum(not bool(mask[yy, xx]) for yy in range(y) for xx in range(x))
    np.testing.assert_array_equal(table, expected)
    np.testing.assert_array_equal(mask, before)
    assert table.dtype == np.uint64 and table.flags.c_contiguous


@pytest.mark.parametrize("before,after", [(0, 0), (0, 1), (2, 0), (1, 2), (4, 5), (5, 4), (20, 100)])
@pytest.mark.parametrize("kind", ["valid", "invalid", "hole", "thin", "random"])
def test_support_every_point_scalar_oracle(before, after, kind):
    rng = np.random.default_rng(719)
    mask = np.ones((20, 23), dtype=bool)
    if kind == "invalid":
        mask[:] = False
    elif kind == "hole":
        mask[7:9, 12:14] = False
    elif kind == "thin":
        mask[10, :] = False
    elif kind == "random":
        mask = rng.random(mask.shape) > .12
    y, x = np.meshgrid(np.arange(-1, 22, .49), np.arange(-1, 25, .51))
    points = np.column_stack((x.ravel(), y.ravel()))
    np.testing.assert_array_equal(rectangle_support(invalid_integral(mask), points, before=before, after=after),
                                  _oracle(mask, points, before, after))


def test_noncontiguous_table_and_points_and_unsigned_no_underflow():
    mask = np.zeros((10, 12), dtype=bool)
    mask[7:9, 8:11] = True
    tab = invalid_integral(mask)
    padded = np.zeros((tab.shape[0] * 2, tab.shape[1] * 2), dtype=np.uint64)
    padded[::2, ::2] = tab
    table_view = padded[::2, ::2]
    points = np.array([[9, 8, 0], [10, 8, 0], [7, 6, 0], [8, 7, 0]], dtype=np.uint64)[:, :2]
    old_table, old_points = table_view.copy(), points.copy()
    with np.errstate(all="raise"):
        actual = rectangle_support(table_view, points, before=0, after=0)
    np.testing.assert_array_equal(actual, _oracle(mask, points, 0, 0))
    np.testing.assert_array_equal(table_view, old_table)
    np.testing.assert_array_equal(points, old_points)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64, np.longdouble])
def test_extreme_coordinates_filtered_before_cast(dtype):
    mask = np.ones((20, 30), dtype=bool)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        points = np.array([[limits.max, 5], [5, limits.max], [limits.min, limits.max], [5, 5]], dtype=dtype)
    else:
        limits = np.finfo(dtype)
        points = np.array([[limits.max, 5], [5, limits.max], [-limits.max, -limits.max], [5, 5]], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            got = rectangle_support(invalid_integral(mask), points)
    np.testing.assert_array_equal(got, [False, False, False, True])


def test_integer_phase_and_nextafter_bounds():
    mask = np.ones((20, 20), dtype=bool)
    points = np.array([[4, 4], [np.nextafter(4.0, -np.inf), 4], [15, 4],
                       [np.nextafter(15.0, -np.inf), 4], [np.nextafter(15.0, np.inf), 4], [14.999, 14.999]])
    np.testing.assert_array_equal(rectangle_support(invalid_integral(mask), points), [True, False, False, True, False, True])


def test_empty_points_validated_and_huge_footprints_safe():
    table = invalid_integral(np.ones((10, 10), dtype=bool))
    assert rectangle_support(table, np.empty((0, 2))).shape == (0,)
    assert rectangle_support(table, np.empty((0, 2))).dtype == bool
    for kwargs in ({"before": 10**10000}, {"after": 10**10000}, {"before": 10**10000, "after": 10**10000}):
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(rectangle_support(table, np.array([[1, 2], [3, 4]]), **kwargs), [False, False])
        with pytest.raises(ValueError):
            rectangle_support(table, np.array([[np.nan, 1]]), **kwargs)


@pytest.mark.parametrize("bad", [None, [[True]], np.ones((2, 2)), np.ones((2,), dtype=bool),
                                   np.ones((2, 2, 2), dtype=bool), np.empty((0, 2), dtype=bool), np.empty((2, 0), dtype=bool)])
def test_reject_invalid_masks(bad):
    with pytest.raises(ValueError):
        invalid_integral(bad)


@pytest.mark.parametrize("bad", [None, [[1, 2]], np.ones((3,)), np.ones((1, 3)), np.ones((1, 2, 1)),
                                   np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=complex),
                                   np.array([[1, 2]], dtype=object), np.array([["1", "2"]]),
                                   np.array([[np.nan, 2]]), np.array([[np.inf, 2]]), np.array([[-np.inf, 2]])])
def test_reject_invalid_points(bad):
    with pytest.raises(ValueError):
        rectangle_support(invalid_integral(np.ones((5, 5), dtype=bool)), bad)


@pytest.mark.parametrize("bad", [None, [[0, 0], [0, 0]], np.zeros((3, 3)), np.zeros((3, 3), dtype=np.int64),
                                   np.zeros((1, 4), dtype=np.uint64), np.zeros((4, 1), dtype=np.uint64),
                                   np.zeros((3,), dtype=np.uint64), np.ones((3, 3), dtype=np.uint64)])
def test_reject_invalid_integrals(bad):
    with pytest.raises(ValueError):
        rectangle_support(bad, np.empty((0, 2)))


@pytest.mark.parametrize("value", [True, False, np.int64(1), np.uint64(1), 1.0, -1, None])
@pytest.mark.parametrize("name", ["before", "after"])
def test_reject_invalid_footprint_parameters(name, value):
    with pytest.raises(ValueError):
        rectangle_support(invalid_integral(np.ones((5, 5), dtype=bool)), np.empty((0, 2)), **{name: value})


def test_reject_ndarray_subclasses_as_specified():
    class Array(np.ndarray):
        pass
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        invalid_integral(mask.view(Array))
    with pytest.raises(ValueError):
        rectangle_support(invalid_integral(mask).view(Array), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        rectangle_support(invalid_integral(mask), np.zeros((1, 2)).view(Array))


def test_large_rectangular_table_without_image_per_query_allocation():
    # About 11.5 MB table, rather than allocating several full Z8-sized tables
    # while the production pipeline is being tested in another process.
    mask = np.ones((900, 1600), dtype=bool)
    mask[449, 799] = False
    table = invalid_integral(mask)
    points = np.tile([[100.1, 100.2], [799, 449], [0, 0], [1500.8, 800.9]], (25000, 1))
    actual = rectangle_support(table, points)
    np.testing.assert_array_equal(actual, np.tile([True, False, False, True], 25000))
    assert table.nbytes == 901 * 1601 * 8
