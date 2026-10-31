"""Independent acceptance of the support-anchor optimization contract.

The primary oracle enumerates source pixels using Python integers.  It shares
neither integral construction nor rectangle-query arithmetic with production.
The retained rectangle_support helper is a secondary cross-check only.  These
are small correctness tests, not performance measurements or Siril validation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from starstack.support_anchor import anchor_support, build_anchor_grid
from starstack.support_bounds import rectangle_support


def _scalar_integral(validity):
    height, width = validity.shape
    result = np.zeros((height + 1, width + 1), dtype=np.uint64)
    for y in range(1, height + 1):
        running = 0
        for x in range(1, width + 1):
            running += int(not bool(validity[y - 1, x - 1]))
            result[y, x] = int(result[y - 1, x]) + running
    return result


def _scalar_grid(validity, before=4, after=5):
    height, width = validity.shape
    result = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if x < before or y < before or x + after >= width or y + after >= height:
                continue
            result[y, x] = all(
                bool(validity[yy, xx])
                for yy in range(y - before, y + after + 1)
                for xx in range(x - before, x + after + 1)
            )
    return result


def _scalar_lookup(anchors, points):
    height, width = anchors.shape
    answer = []
    for pair in points:
        # In particular, uint64.item() becomes a Python int without precision
        # loss.  No floor or narrowing is attempted for out-of-image values.
        x, y = pair[0].item(), pair[1].item()
        answer.append(
            bool(anchors[math.floor(y), math.floor(x)])
            if 0 <= x < width and 0 <= y < height
            else False
        )
    return np.array(answer, dtype=bool)


def _layout(array, layout):
    if layout == "c":
        return np.array(array, order="C", copy=True)
    if layout == "fortran":
        return np.array(array, order="F", copy=True)
    if layout == "transpose":
        return np.array(array.T, order="C", copy=True).T
    if layout == "stepped":
        owner = np.empty(tuple(2 * n + 1 for n in array.shape), dtype=array.dtype)
        result = owner[1::2, 1::2]
    elif layout == "reversed":
        result = np.empty_like(array)[::-1, ::-1]
    else:
        raise AssertionError(layout)
    result[...] = array
    return result


@pytest.mark.parametrize("shape", [(1, 1), (1, 12), (12, 1), (8, 13), (9, 9),
                                   (10, 10), (10, 13), (11, 11), (17, 23)])
@pytest.mark.parametrize("before,after", [(0, 0), (0, 3), (3, 0), (1, 2), (4, 5)])
def test_grid_matches_scalar_pixels_for_small_and_asymmetric_footprints(shape, before, after):
    validity = np.ones(shape, dtype=bool)
    if min(shape) > 3:
        validity[shape[0] // 3, shape[1] // 3] = False
    table = _scalar_integral(validity)
    before_bytes = table.tobytes()
    expected = _scalar_grid(validity, before, after)
    for rows in (1, 3, 128):
        actual = build_anchor_grid(table, before=before, after=after, rows=rows)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.bool_
        assert actual.shape == shape
        assert actual.flags.c_contiguous
        assert not actual.flags.writeable
        assert not np.shares_memory(actual, table)
    assert table.tobytes() == before_bytes


@pytest.mark.parametrize("pattern", ["all_true", "all_false", "one_hole", "horizontal",
                                     "vertical", "diagonal", "random_sparse"])
def test_grid_sparse_and_thin_structures_against_scalar_truth(pattern):
    validity = np.ones((21, 27), dtype=bool)
    if pattern == "all_false":
        validity[:] = False
    elif pattern == "one_hole":
        validity[9, 11] = False
    elif pattern == "horizontal":
        validity[8, 4:23] = False
    elif pattern == "vertical":
        validity[2:19, 12:14] = False
    elif pattern == "diagonal":
        validity[np.arange(18), np.arange(18) + 3] = False
    elif pattern == "random_sparse":
        validity = np.random.default_rng(20260919).random(validity.shape) > 0.025
    np.testing.assert_array_equal(
        build_anchor_grid(_scalar_integral(validity), rows=4), _scalar_grid(validity)
    )


def test_default_ten_by_ten_anchor_is_offset_four_not_symmetric():
    actual = build_anchor_grid(_scalar_integral(np.ones((12, 15), dtype=bool)))
    expected = np.zeros((12, 15), dtype=bool)
    expected[4:7, 4:10] = True
    np.testing.assert_array_equal(actual, expected)
    assert actual[4, 4] and actual[6, 9]
    assert not actual[3, 4] and not actual[7, 4]
    assert not actual[4, 3] and not actual[4, 10]


@pytest.mark.parametrize("layout", ["c", "fortran", "transpose", "stepped", "reversed"])
def test_integral_arbitrary_strides_and_readonly_are_supported_without_mutation(layout):
    validity = np.ones((18, 24), dtype=bool)
    validity[7, 8] = False
    table = _layout(_scalar_integral(validity), layout)
    saved = table.copy()
    table.flags.writeable = False
    actual = build_anchor_grid(table, rows=2)
    np.testing.assert_array_equal(actual, _scalar_grid(validity))
    np.testing.assert_array_equal(table, saved)
    assert actual.flags.c_contiguous and not actual.flags.writeable


def test_grid_output_is_new_readonly_storage_on_every_call():
    table = _scalar_integral(np.ones((13, 17), dtype=bool))
    first = build_anchor_grid(table)
    second = build_anchor_grid(table)
    assert first is not second
    assert not np.shares_memory(first, second)
    with pytest.raises(ValueError):
        first[4, 4] = False
    table[1:, 1:] = 1
    assert first[4, 4] and second[4, 4]


def test_grid_construction_does_not_use_floor(monkeypatch):
    validity = np.ones((13, 17), dtype=bool)
    validity[7, 8] = False
    table = _scalar_integral(validity)
    expected = _scalar_grid(validity)

    def forbidden(*args, **kwargs):
        raise AssertionError("integer anchor construction must not use np.floor")

    monkeypatch.setattr(np, "floor", forbidden)
    np.testing.assert_array_equal(build_anchor_grid(table, rows=2), expected)


@pytest.mark.parametrize("parameter", ["before", "after"])
def test_huge_footprint_is_all_false_without_integer_narrowing(parameter):
    table = _scalar_integral(np.ones((12, 16), dtype=bool))
    actual = build_anchor_grid(table, **{parameter: 10 ** 10000})
    assert actual.shape == (12, 16)
    assert not actual.any()
    assert actual.flags.c_contiguous and not actual.flags.writeable


def test_huge_positive_row_count_is_valid():
    validity = np.ones((13, 17), dtype=bool)
    validity[5, 6] = False
    np.testing.assert_array_equal(
        build_anchor_grid(_scalar_integral(validity), rows=10 ** 10000), _scalar_grid(validity)
    )


def test_check_runs_for_bounded_row_tiles_and_output_is_unchanged():
    validity = np.ones((20, 24), dtype=bool)
    calls = []
    actual = build_anchor_grid(_scalar_integral(validity), rows=2, check=lambda: calls.append(1))
    # Eleven eligible anchor rows cannot be computed in fewer than six
    # two-row tiles.  Do not prescribe whether border-only tiles are skipped.
    assert len(calls) >= math.ceil((20 - 4 - 5) / 2)
    np.testing.assert_array_equal(actual, _scalar_grid(validity))


def test_cancellation_propagates_original_exception_without_mutating_input():
    class Cancelled(RuntimeError):
        pass

    exception = Cancelled("independent cancellation marker")
    table = _scalar_integral(np.ones((20, 24), dtype=bool))
    saved = table.copy()
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception

    with pytest.raises(Cancelled) as caught:
        build_anchor_grid(table, rows=1, check=check)
    assert caught.value is exception
    assert calls == 2
    np.testing.assert_array_equal(table, saved)


def test_impossible_footprint_may_check_zero_or_one_times():
    calls = []
    result = build_anchor_grid(
        _scalar_integral(np.ones((3, 4), dtype=bool)), check=lambda: calls.append(1)
    )
    assert not result.any()
    assert len(calls) <= 1


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("bad", [
    None, [[0, 0], [0, 0]], np.zeros((3, 4), dtype=np.int64),
    np.zeros((3, 4), dtype=np.uint32), np.zeros((3, 4), dtype=np.float64),
    np.zeros((3, 4), dtype=bool), np.zeros((3, 4), dtype=object),
    np.zeros((3, 4), dtype=">u8"), np.zeros(4, dtype=np.uint64),
    np.zeros((2, 2, 2), dtype=np.uint64), np.zeros((0, 2), dtype=np.uint64),
    np.zeros((2, 0), dtype=np.uint64), np.zeros((1, 4), dtype=np.uint64),
    np.zeros((4, 1), dtype=np.uint64), np.zeros((3, 4), dtype=np.uint64).view(_ArraySubclass),
], ids=["none", "list", "signed", "uint32", "float", "bool", "object", "big_endian",
        "rank1", "rank3", "empty_y", "empty_x", "short_y", "short_x", "subclass"])
def test_invalid_integral_structure_is_rejected(bad):
    with pytest.raises(ValueError):
        build_anchor_grid(bad)


@pytest.mark.parametrize("position", [(0, 0), (0, 7), (5, 0)])
def test_nonzero_integral_border_is_rejected(position):
    table = np.zeros((13, 17), dtype=np.uint64)
    table[position] = 1
    with pytest.raises(ValueError):
        build_anchor_grid(table)


@pytest.mark.parametrize("name", ["before", "after", "rows"])
@pytest.mark.parametrize("value", [True, False, np.int64(1), np.uint64(1), 1.0, None, -1, "1"])
def test_invalid_scalar_parameters_are_rejected_even_when_footprint_cannot_fit(name, value):
    with pytest.raises(ValueError):
        build_anchor_grid(np.zeros((2, 2), dtype=np.uint64), **{name: value})


def test_zero_rows_is_rejected_before_early_return():
    with pytest.raises(ValueError):
        build_anchor_grid(np.zeros((2, 2), dtype=np.uint64), rows=0, before=10 ** 10000)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.uint64,
                                   np.float16, np.float32, np.float64, np.longdouble])
def test_lookup_accepts_real_integer_and_floating_dtypes(dtype):
    anchors = np.array([[True, False, True], [False, True, False]], dtype=bool)
    points = np.array([[0, 0], [1, 0], [2, 0], [1, 1], [3, 0], [0, 2]], dtype=dtype)
    actual = anchor_support(anchors, points)
    np.testing.assert_array_equal(actual, _scalar_lookup(anchors, points))
    assert actual.shape == (len(points),) and actual.dtype == np.bool_


def test_lookup_does_not_reapply_default_footprint_or_clip_to_edges():
    # This is a lookup primitive for an arbitrary internally prepared grid;
    # true values at the grid edge must be respected, including the 1x1 case.
    anchors = np.ones((1, 1), dtype=bool)
    points = np.array([[0, 0], [.75, .75], [-.01, 0], [1, 0], [0, 1], [-0., -0.]])
    np.testing.assert_array_equal(anchor_support(anchors, points), [True, True, False, False, False, True])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nextafter_on_integer_and_image_boundaries_uses_floor(dtype):
    anchors = (np.indices((4, 6)).sum(axis=0) % 2 == 0)
    xs, ys = [], []
    for k in range(-1, 8):
        v = dtype(k)
        xs.extend([np.nextafter(v, dtype(-np.inf)), v, np.nextafter(v, dtype(np.inf))])
    for k in range(-1, 6):
        v = dtype(k)
        ys.extend([np.nextafter(v, dtype(-np.inf)), v, np.nextafter(v, dtype(np.inf))])
    points = np.array([(x, y) for x in xs for y in ys], dtype=dtype)
    np.testing.assert_array_equal(anchor_support(anchors, points), _scalar_lookup(anchors, points))


def test_uint64_max_and_int64_extremes_are_rejected_before_narrowing():
    anchors = np.ones((4, 6), dtype=bool)
    unsigned = np.array([[0, 0], [5, 3], [2 ** 63, 1], [2 ** 64 - 1, 0],
                         [0, 2 ** 64 - 1], [2 ** 64 - 1, 2 ** 64 - 1]], dtype=np.uint64)
    signed = np.array([[0, 0], [5, 3], [-(2 ** 63), 0], [2 ** 63 - 1, 0],
                       [0, -(2 ** 63)], [0, 2 ** 63 - 1]], dtype=np.int64)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(anchor_support(anchors, unsigned), _scalar_lookup(anchors, unsigned))
        np.testing.assert_array_equal(anchor_support(anchors, signed), _scalar_lookup(anchors, signed))


def test_huge_finite_float_coordinates_never_become_valid_indices():
    anchors = np.ones((4, 6), dtype=bool)
    large = np.finfo(np.float64).max
    points = np.array([[large, 1], [-large, 1], [1, large], [1, -large], [0, 0]])
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(anchor_support(anchors, points), [False, False, False, False, True])


@pytest.mark.parametrize("layout", ["c", "fortran", "transpose", "stepped", "reversed"])
@pytest.mark.parametrize("readonly", [False, True])
def test_lookup_arbitrary_strides_and_readonly_inputs_are_not_mutated(layout, readonly):
    anchors = _layout(np.indices((7, 11)).sum(axis=0) % 3 != 0, layout)
    points = _layout(np.array([[0., 0.], [10.8, 6.9], [4.1, 3.9], [-1., 2.], [11., 6.]]), layout)
    saved_anchors, saved_points = anchors.copy(), points.copy()
    if readonly:
        anchors.flags.writeable = False
        points.flags.writeable = False
    np.testing.assert_array_equal(anchor_support(anchors, points), _scalar_lookup(anchors, points))
    np.testing.assert_array_equal(anchors, saved_anchors)
    np.testing.assert_array_equal(points, saved_points)
    assert anchors.flags.writeable is not readonly
    assert points.flags.writeable is not readonly


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_empty_real_points_are_accepted_and_return_bool_vector(dtype):
    result = anchor_support(np.ones((2, 3), dtype=bool), np.empty((0, 2), dtype=dtype))
    assert result.shape == (0,) and result.dtype == np.bool_


@pytest.mark.parametrize("bad", [
    None, [[True]], np.ones((2, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.float32),
    np.ones((2, 3), dtype=object), np.ones(3, dtype=bool), np.ones((2, 2, 2), dtype=bool),
    np.ones((0, 3), dtype=bool), np.ones((2, 0), dtype=bool),
    np.ones((2, 3), dtype=bool).view(_ArraySubclass),
], ids=["none", "list", "uint8", "float", "object", "rank1", "rank3", "empty_y", "empty_x", "subclass"])
def test_invalid_anchors_are_rejected_even_for_empty_points(bad):
    with pytest.raises(ValueError):
        anchor_support(bad, np.empty((0, 2)))


@pytest.mark.parametrize("bad", [
    None, [[0., 0.]], np.zeros(2), np.zeros((1, 3)), np.zeros((1, 1, 2)),
    np.empty((0, 1)), np.zeros((1, 2), dtype=bool), np.empty((0, 2), dtype=bool),
    np.zeros((1, 2), dtype=complex), np.empty((0, 2), dtype=complex),
    np.zeros((1, 2), dtype=object), np.empty((0, 2), dtype=object),
    np.full((1, 2), "0"), np.empty((0, 2), dtype="U1"),
    np.zeros((1, 2), dtype="timedelta64[s]"), np.zeros((1, 2), dtype="datetime64[s]"),
    np.zeros((1, 2)).view(_ArraySubclass),
], ids=["none", "list", "rank1", "wrong_width", "rank3", "empty_wrong_width", "bool", "empty_bool",
        "complex", "empty_complex", "object", "empty_object", "string", "empty_string", "timedelta", "datetime", "subclass"])
def test_invalid_points_are_rejected_including_empty_wrong_types(bad):
    with pytest.raises(ValueError):
        anchor_support(np.ones((4, 6), dtype=bool), bad)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis", [0, 1])
def test_nonfinite_points_are_rejected_even_if_other_coordinate_is_outside(nonfinite, axis):
    points = np.array([[-100., -100.], [0., 0.]])
    points[0, axis] = nonfinite
    with pytest.raises(ValueError):
        anchor_support(np.ones((4, 6), dtype=bool), points)


@pytest.mark.parametrize("before,after", [(0, 0), (0, 2), (2, 0), (1, 3), (4, 5)])
def test_built_grid_queries_match_scalar_footprints_and_retained_rectangle_reference(before, after):
    validity = np.ones((19, 25), dtype=bool)
    validity[8, 10] = False
    validity[14:, 20] = False
    table = _scalar_integral(validity)
    grid = build_anchor_grid(table, before=before, after=after, rows=3)
    scalar = _scalar_grid(validity, before, after)
    points = np.array([(x + dx, y + dy)
                       for y in range(-1, 21) for x in range(-1, 27)
                       for dx, dy in [(0., 0.), (.25, .75), (.999, .001)]])
    actual = anchor_support(grid, points)
    np.testing.assert_array_equal(actual, _scalar_lookup(scalar, points))
    np.testing.assert_array_equal(actual, rectangle_support(table, points, before=before, after=after))


def test_no_global_cache_confuses_same_shape_different_content_or_changed_input():
    validity_a = np.ones((17, 23), dtype=bool)
    validity_b = validity_a.copy()
    validity_b[8, 11] = False
    table = _scalar_integral(validity_a)
    first = build_anchor_grid(table)
    table[:] = _scalar_integral(validity_b)
    second = build_anchor_grid(table)
    third = build_anchor_grid(_scalar_integral(validity_a))
    np.testing.assert_array_equal(first, _scalar_grid(validity_a))
    np.testing.assert_array_equal(second, _scalar_grid(validity_b))
    np.testing.assert_array_equal(third, first)
    assert np.any(first != second)
