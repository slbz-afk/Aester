"""Independent scalar truth and finite-radius missingness for test-only identities."""
import math
import numpy as np
import pytest

from starstack.holdout_association import associate_holdout


def truth(ref, src, radius):
    neighbours = [[j for j, q in enumerate(src)
                   if math.dist(tuple(p), tuple(q)) <= radius] for p in ref]
    uses = [sum(j in indices for indices in neighbours) for j in range(len(src))]
    result = []
    for i, indices in enumerate(neighbours):
        matched = len(indices) == 1 and uses[indices[0]] == 1
        j = indices[0] if matched else None
        result.append(dict(reference_index=i, source_index=j,
                           status='matched' if matched else ('ambiguous' if indices else 'missing'),
                           candidate_count=len(indices),
                           distance_px=math.dist(tuple(ref[i]), tuple(src[j])) if matched else None))
    return result


@pytest.mark.parametrize('seed', range(35))
@pytest.mark.parametrize('radius', [0.5, 1.0, 3.0])
def test_all_reference_entries_match_independent_scalar_truth(seed, radius):
    rng = np.random.default_rng(seed)
    # Half-integer coordinates make inclusive boundaries exact.
    ref = rng.integers(-8, 9, (21, 2)).astype(float) / 2
    src = rng.integers(-8, 9, (25, 2)).astype(float) / 2
    assert associate_holdout(ref, src, radius=radius) == truth(ref, src, radius)


def test_more_than_one_chunk_and_reverse_views():
    ref = np.column_stack([np.arange(300)*10., np.zeros(300)])[::-1]
    src = np.column_stack([np.arange(300)*10.+.25, np.zeros(300)])[::-1]
    ref.flags.writeable = src.flags.writeable = False
    assert associate_holdout(ref, src, radius=1) == truth(ref, src, 1)


def test_far_error_is_missing_not_dropped_and_no_greedy_choice():
    ref = np.array([[0., 0.], [3., 0.], [20., 0.], [100., 100.]])
    src = np.array([[1.5, 0.], [20., 0.], [20.1, 0.]])
    rows = associate_holdout(ref, src, radius=2)
    assert [r['status'] for r in rows] == ['ambiguous', 'ambiguous', 'ambiguous', 'missing']
    assert len(rows) == 4


def test_inclusive_pythagorean_boundary_and_no_mutation():
    ref = np.array([[0, 0], [100, 100]], dtype=np.int32)
    src = np.array([[3, 4], [103, 104]], dtype=np.int64)
    a, b = ref.copy(), src.copy()
    assert associate_holdout(ref, src, radius=5) == truth(ref, src, 5)
    np.testing.assert_array_equal(ref, a)
    np.testing.assert_array_equal(src, b)


def test_empty_sides_keep_denominator():
    empty = np.empty((0, 2))
    assert associate_holdout(empty, empty, radius=1) == []
    assert associate_holdout(empty, np.ones((1, 2)), radius=1) == []
    assert associate_holdout(np.zeros((2, 2)), empty, radius=1) == truth(np.zeros((2, 2)), empty, 1)


@pytest.mark.parametrize('points', [[], [1, 2], [[True, False]], [[1+1j, 2]], [['1', '2']],
                                   np.array([[1, 2]], dtype=object), [[float('nan'), 0]],
                                   [[float('inf'), 0]], [[1e9+1, 0]], [[[1, 2]]]])
def test_reject_invalid_coordinates(points):
    for args in ((points, np.zeros((1, 2))), (np.zeros((1, 2)), points)):
        with pytest.raises(ValueError):
            associate_holdout(*args, radius=1)


@pytest.mark.parametrize('radius', [True, np.float64(1), np.int64(1), '1', None, 0, -1,
                                   float('nan'), float('inf'), 1e9+1, 10**1000])
def test_reject_invalid_radius(radius):
    with pytest.raises(ValueError):
        associate_holdout(np.zeros((1, 2)), np.zeros((1, 2)), radius=radius)


def test_extreme_supported_values_and_fresh_builtin_json_output():
    import json
    ref = np.array([[-1e9, -1e9], [1e9, 1e9]])
    src = np.array([[-1e9+.5, -1e9], [0, 1e9]])
    a = associate_holdout(ref, src, radius=1e9)
    assert a == truth(ref, src, 1e9)
    assert json.loads(json.dumps(a, allow_nan=False)) == a
    a[0]['status'] = 'changed'
    assert associate_holdout(ref, src, radius=1e9) == truth(ref, src, 1e9)
