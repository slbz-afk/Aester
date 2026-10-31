"""Independent H5 contracts: explicit scalar truth, without product blending.

The expected pixels below are constructed using Python scalar arithmetic.
Neither the product's blend functions nor its visibility/weight calculation
participates in the oracle. These are numerical checks, not a real-scene
verification of which sky pixels should have contributed.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from starstack import horizon_acceptance as ha


def scalar_composite(sky, ground, weight):
    """Apply the stated same-position contract independently of production."""
    height, width, _ = sky.shape
    result = np.empty(sky.shape, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            w = float(weight[y, x])
            for channel in range(3):
                result[y, x, channel] = w * float(sky[y, x, channel]) + (1.0 - w) * float(ground[y, x, channel])
    return result


def mixed_fixture():
    mask = np.array([[1, 1, .75, 0], [1, .25, .5, 0], [.5, 1, .5, 0]], dtype=np.float64)
    weight = np.array([[1, 0, .75, 0], [1, 0, .5, 0], [.5, 1, .5, 0]], dtype=np.float64)
    sky = np.empty((3, 4, 3), dtype=np.float64)
    ground = np.empty_like(sky)
    for y in range(3):
        for x in range(4):
            sky[y, x] = [5 - .5 * x, 1 - y, 2 + .25 * x]
            ground[y, x] = [2 + y - x, -3 + x, .1 * y]
    return scalar_composite(sky, ground, weight), sky, ground, mask, weight


def measure_and_judge(composite, sky, ground, mask, weight, **kwargs):
    metrics = ha.seam(composite, sky, ground, mask, effective_weight=weight, **kwargs)
    return metrics, ha.judge_seam(metrics, ha.CRITERIA["H5"])


def check_named(judgement, name):
    matches = [item for item in judgement["checks"] if item["name"] == name]
    assert len(matches) == 1, judgement
    return matches[0]


def test_same_position_signed_hdr_truth_and_explicit_region_counts():
    fixture = mixed_fixture()
    before = [a.copy() for a in fixture]
    metrics, judged = measure_and_judge(*fixture)
    assert judged["status"] == "passed"
    assert judged["passed"] is True
    for name, count in {"all_pixels": 12, "sky_only": 3, "ground_only": 3, "unobserved_sky": 2, "feather": 4}.items():
        region = metrics["regions"][name]
        assert region["eligible_samples"] == count
        assert region["samples"] == count
        assert region["finite_samples"] == count
        assert region["max_abs_diff"] <= 1e-12
        assert check_named(judged, name)["status"] == "passed"
    for actual, original in zip(fixture, before):
        np.testing.assert_array_equal(actual, original)


def test_using_original_mask_instead_of_actual_weight_is_detected():
    correct, sky, ground, mask, weight = mixed_fixture()
    wrong = scalar_composite(sky, ground, mask)
    assert not np.array_equal(correct, wrong)
    _, judged = measure_and_judge(wrong, sky, ground, mask, weight)
    assert judged["status"] == "failed"
    assert check_named(judged, "unobserved_sky")["status"] == "failed"


def test_coverage_is_an_observation_and_cannot_override_actual_weight():
    fixture = mixed_fixture()
    # Coarse coverage cannot dictate full-resolution compose eligibility.
    for coverage in (np.zeros((1, 1), dtype=np.int32), np.full((1, 1), 9, dtype=np.int32)):
        _, judged = measure_and_judge(*fixture, coverage=coverage, step=4)
        assert judged["status"] == "passed"


def test_missing_actual_weight_is_insufficient_even_with_coverage_and_zero_error():
    composite, sky, ground, mask, _ = mixed_fixture()
    metrics = ha.seam(composite, sky, ground, mask, np.ones((1, 1), np.int32), 4)
    judged = ha.judge_seam(metrics, ha.CRITERIA["H5"])
    assert judged["status"] == "insufficient_data"
    assert judged["passed"] is False


def test_empty_metrics_do_not_certify_h5():
    judged = ha.judge_seam({}, ha.CRITERIA["H5"])
    assert judged["status"] == "insufficient_data"
    assert judged["passed"] is False


@pytest.mark.parametrize("shape", [(0, 7), (5, 0), (0, 0)])
def test_empty_dimensions_are_insufficient_not_a_vacuous_pass(shape):
    sky = np.zeros((*shape, 3), dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)
    _, judged = measure_and_judge(sky.copy(), sky, sky.copy(), weight, weight.copy())
    assert judged["status"] == "insufficient_data"
    assert check_named(judged, "all_pixels")["status"] == "insufficient_data"


@pytest.mark.parametrize("constant_mask", [0., 1.])
def test_constant_hard_mask_has_not_applicable_feather_but_real_pixels_tested(constant_mask):
    mask = np.full((5, 7), constant_mask, dtype=np.float64)
    sky = np.full((5, 7, 3), 3., dtype=np.float64)
    ground = np.full_like(sky, -.5)
    composite = scalar_composite(sky, ground, mask)
    metrics, judged = measure_and_judge(composite, sky, ground, mask, mask.copy())
    assert judged["status"] == "passed"
    assert metrics["regions"]["feather"]["eligible_samples"] == 0
    assert check_named(judged, "feather")["status"] == "not_applicable"
    assert check_named(judged, "all_pixels")["status"] == "passed"


def narrow_feather_fixture():
    mask = np.zeros((9, 9), dtype=np.float64)
    mask[:3] = 1
    mask[3] = .5  # Entire feather lies between step-4 grid rows.
    sky = np.full((9, 9, 3), 2., dtype=np.float64)
    ground = np.full_like(sky, .25)
    return scalar_composite(sky, ground, mask), sky, ground, mask, mask.copy()


def test_default_full_resolution_sees_one_pixel_feather_between_coverage_rows():
    metrics, judged = measure_and_judge(*narrow_feather_fixture(), coverage=np.ones((3, 3)), step=4)
    assert judged["status"] == "passed"
    assert metrics["regions"]["feather"]["samples"] == 9
    assert check_named(judged, "feather")["status"] == "passed"


def test_explicit_subsampling_does_not_certify_an_unseen_thin_feather():
    metrics, judged = measure_and_judge(*narrow_feather_fixture(), sample_step=4)
    assert metrics["regions"]["feather"]["eligible_samples"] == 9
    assert metrics["regions"]["feather"]["samples"] == 0
    assert check_named(judged, "feather")["status"] == "insufficient_data"
    assert judged["status"] == "insufficient_data"


def test_single_pixel_error_between_old_grid_rows_and_columns_is_detected():
    composite, sky, ground, mask, weight = narrow_feather_fixture()
    composite[3, 3, 2] -= .1
    metrics, judged = measure_and_judge(composite, sky, ground, mask, weight, step=4)
    assert metrics["regions"]["feather"]["max_abs_diff"] > .09
    assert check_named(judged, "feather")["status"] == "failed"


def test_seen_error_wins_over_incomplete_subsampling():
    composite, sky, ground, mask, weight = narrow_feather_fixture()
    composite[0, 0, 0] += .2
    _, judged = measure_and_judge(composite, sky, ground, mask, weight, sample_step=4)
    assert judged["status"] == "failed"
    assert check_named(judged, "all_pixels")["status"] == "failed"
    assert check_named(judged, "feather")["status"] == "insufficient_data"


def test_natural_dark_structure_does_not_fail_against_brighter_neighbours():
    shape = (96, 24)
    mask = np.zeros(shape, np.float64)
    mask[:48] = 1.
    mask[48:52] = .5
    sky = np.full((*shape, 3), .4, np.float64)
    ground = np.full_like(sky, .3)
    # Both masters genuinely contain darkness at this same location.
    sky[48:52] = .006
    ground[48:52] = .004
    composite = scalar_composite(sky, ground, mask)
    _, judged = measure_and_judge(composite, sky, ground, mask, mask.copy())
    assert judged["status"] == "passed"
    composite[49:51] -= .001
    _, wrong = measure_and_judge(composite, sky, ground, mask, mask.copy())
    assert wrong["status"] == "failed"
    assert check_named(wrong, "feather")["status"] == "failed"


@pytest.mark.parametrize("cancel_to_zero", [False, True])
def test_true_zero_pixels_and_signed_cancellation_are_legal(cancel_to_zero):
    sky = np.zeros((4, 5, 3), np.float64)
    ground = np.zeros_like(sky)
    mask = np.full((4, 5), .5, np.float64)
    if cancel_to_zero:
        sky[:] = [-2, -4, -6]
        ground[:] = [2, 4, 6]
    composite = np.zeros_like(sky)  # Hand-computed exact same-position truth.
    _, judged = measure_and_judge(composite, sky, ground, mask, mask.copy())
    assert judged["status"] == "passed"


@pytest.mark.parametrize("field", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pixels_or_weights_never_become_a_pass(field, value):
    fixture = list(mixed_fixture())
    fixture[field].flat[0] = value
    with pytest.raises(ValueError):
        measure_and_judge(*fixture)


@pytest.mark.parametrize("field", [3, 4])
@pytest.mark.parametrize("value", [-.01, 1.01])
def test_mask_and_effective_weight_are_not_silently_clipped(field, value):
    fixture = list(mixed_fixture())
    fixture[field].flat[0] = value
    with pytest.raises(ValueError):
        measure_and_judge(*fixture)


@pytest.mark.parametrize("field", [0, 1, 2, 3, 4])
def test_mismatched_shapes_do_not_broadcast_or_truncate(field):
    fixture = list(mixed_fixture())
    fixture[field] = fixture[field][:1]
    with pytest.raises(ValueError):
        measure_and_judge(*fixture)


@pytest.mark.parametrize("name", ["step", "sample_step"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 2., np.int64(2), "2", None])
def test_sampling_steps_are_positive_builtin_integers(name, value):
    with pytest.raises(ValueError):
        measure_and_judge(*mixed_fixture(), **{name: value})


def test_coverage_shape_uses_ceiling_and_no_partial_truncation():
    sky = np.ones((5, 7, 3), np.float64)
    ground = np.full_like(sky, .25)
    mask = np.full((5, 7), .5, np.float64)
    composite = np.full_like(sky, .625)  # Hand-calculated 0.5*1 + 0.5*0.25.
    _, judged = measure_and_judge(composite, sky, ground, mask, mask.copy(), coverage=np.ones((2, 3)), step=3)
    assert judged["status"] == "passed"
    with pytest.raises(ValueError):
        measure_and_judge(composite, sky, ground, mask, mask.copy(), coverage=np.ones((1, 2)), step=3)


@pytest.mark.parametrize("value", [-1., .5, float("nan"), float("inf")])
def test_coverage_counts_are_finite_nonnegative_integers(value):
    with pytest.raises(ValueError):
        measure_and_judge(*mixed_fixture(), coverage=np.array([[value]]), step=4)


def test_unchanged_absolute_compose_tolerance_distinguishes_small_and_large_error():
    assert ha.CRITERIA["H5"]["compose_tolerance"] == 1e-5
    composite, sky, ground, mask, weight = mixed_fixture()
    small = composite.copy(); small[0, 0, 0] += 5e-6
    large = composite.copy(); large[0, 0, 0] += 2e-5
    assert measure_and_judge(small, sky, ground, mask, weight)[1]["status"] == "passed"
    assert measure_and_judge(large, sky, ground, mask, weight)[1]["status"] == "failed"


def test_missing_region_metric_is_not_silently_assumed_to_be_zero_error():
    metrics, _ = measure_and_judge(*mixed_fixture())
    damaged = copy.deepcopy(metrics)
    del damaged["regions"]["feather"]["max_abs_diff"]
    judged = ha.judge_seam(damaged, ha.CRITERIA["H5"])
    assert judged["status"] != "passed"
    assert check_named(judged, "feather")["status"] == "insufficient_data"


def test_missing_all_pixel_check_cannot_be_hidden_by_passing_regional_checks():
    metrics, _ = measure_and_judge(*mixed_fixture())
    del metrics["regions"]["all_pixels"]
    judged = ha.judge_seam(metrics, ha.CRITERIA["H5"])
    assert judged["status"] == "insufficient_data"


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_reported_difference_cannot_certify_a_region(value):
    metrics, _ = measure_and_judge(*mixed_fixture())
    metrics["regions"]["feather"]["max_abs_diff"] = value
    judged = ha.judge_seam(metrics, ha.CRITERIA["H5"])
    assert judged["status"] != "passed"


def test_nonfinite_outside_explicit_sample_grid_is_still_invalid_input():
    composite, sky, ground, mask, weight = narrow_feather_fixture()
    composite[3, 3, 1] = np.nan
    with pytest.raises(ValueError):
        measure_and_judge(composite, sky, ground, mask, weight, sample_step=4)


def test_subsampling_uses_one_global_grid_across_processing_blocks():
    sky = np.full((260, 7, 3), 4., np.float64)
    ground = np.full_like(sky, 2.)
    mask = np.full((260, 7), .5, np.float64)
    composite = np.full_like(sky, 3.)  # Hand-calculated scalar truth.
    composite[129, 3, 1] = 3.01  # On the global stride-3 grid, in block 2.
    metrics, judged = measure_and_judge(composite, sky, ground, mask, mask.copy(), sample_step=3)
    assert metrics["regions"]["all_pixels"]["eligible_samples"] == 260 * 7
    assert metrics["regions"]["all_pixels"]["samples"] == 87 * 3
    assert metrics["regions"]["all_pixels"]["max_abs_diff"] > .009
    assert judged["status"] == "failed"


def test_actual_failure_wins_over_another_regions_missing_evidence():
    metrics, _ = measure_and_judge(*mixed_fixture())
    metrics["regions"]["sky_only"]["max_abs_diff"] = .1
    del metrics["regions"]["feather"]["max_abs_diff"]
    judged = ha.judge_seam(metrics, ha.CRITERIA["H5"])
    assert judged["status"] == "failed"
    assert check_named(judged, "sky_only")["status"] == "failed"
    assert check_named(judged, "feather")["status"] == "insufficient_data"
    assert judged["failing"] and judged["incomplete"]


@pytest.mark.parametrize("counts", [
    {"eligible_samples": 0, "samples": 1, "finite_samples": 1},
    {"eligible_samples": 4, "samples": 4, "finite_samples": 3},
    {"eligible_samples": 4, "samples": True, "finite_samples": 1},
])
def test_invalid_counts_are_failure_even_when_other_regions_are_incomplete(counts):
    metrics, _ = measure_and_judge(*mixed_fixture(), sample_step=2)
    metrics["regions"]["feather"].update(counts)
    judged = ha.judge_seam(metrics, ha.CRITERIA["H5"])
    assert judged["status"] == "failed"
    assert check_named(judged, "feather")["status"] == "failed"


def test_non_object_regions_are_explicitly_rejected_as_invalid_reports():
    metrics, _ = measure_and_judge(*mixed_fixture())
    for invalid in ([], [{}], True, 0, "invalid"):
        damaged = {**metrics, "regions": invalid}
        with pytest.raises(ValueError, match="regions"):
            ha.judge_seam(damaged, ha.CRITERIA["H5"])
