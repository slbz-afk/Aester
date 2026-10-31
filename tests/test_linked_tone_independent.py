"""Independent contract tests for the lossy, display-linear linked tone core.

The oracle uses scalar Python arithmetic; rational anchors are hand-derived.
No product helper or product-generated output supplies expected values. These
tests validate the numerical contract, not photographic or aesthetic quality.
"""

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from starstack.linked_tone import linked_tone


BASE = {"black": 0.0, "white": 1.0, "asinh_strength": 0.0, "midtone": 0.5}


def render(rgb, **changes):
    return linked_tone(rgb, **(BASE | changes))


def scalar_expected(rgb, **changes):
    """Unblocked, scalar evaluation of the externally supplied contract."""
    recipe = BASE | changes
    black, white = recipe["black"], recipe["white"]
    k, m = recipe["asinh_strength"], recipe["midtone"]
    expected = np.empty(rgb.shape, dtype=np.float32)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            channels = [max(0.0, (float(v) - black) / (white - black)) for v in rgb[y, x]]
            peak = max(channels)
            if peak == 0.0:
                expected[y, x] = 0.0
                continue
            a = min(peak, 1.0)
            if k != 0.0:
                a = math.asinh(k * a) / math.asinh(k)
            # Algebraically equivalent positive form avoids copying the
            # specified numerator/denominator signs into the oracle.
            mapped = (1.0 - m) * a / (m + (1.0 - 2.0 * m) * a)
            factor = mapped / peak
            expected[y, x] = [v * factor for v in channels]
    return expected


def close_to_scalar(rgb, **changes):
    actual = render(rgb, **changes)
    assert type(actual) is np.ndarray
    assert actual.dtype == np.dtype(np.float32)
    assert actual.shape == rgb.shape
    assert np.isfinite(actual).all()
    assert (actual >= 0).all() and (actual <= 1).all()
    np.testing.assert_allclose(actual, scalar_expected(rgb, **changes), rtol=3e-7, atol=2e-8)
    return actual


@pytest.mark.parametrize(
    ("midtone", "expected_peaks"),
    [(0.25, [0.0, 0.5, 0.75, 1.0]),
     (0.5, [0.0, 0.25, 0.5, 1.0]),
     (0.75, [0.0, 0.1, 0.25, 1.0])],
)
def test_hand_computed_rational_midtone_anchors(midtone, expected_peaks):
    peaks = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    rgb = peaks[..., None] * np.array([1.0, 0.5, 0.25], dtype=np.float32)
    expected = np.array(expected_peaks, dtype=np.float32).reshape(2, 2, 1)
    expected = expected * np.array([1.0, 0.5, 0.25], dtype=np.float32)
    np.testing.assert_allclose(render(rgb, midtone=midtone), expected, rtol=0, atol=2e-8)


def test_hand_computed_black_normalization_and_over_white_ratio():
    rgb = np.array([[[0.125, 0.125, 0.125], [0.375, 0.625, 1.125]],
                    [[2.125, 1.125, 0.625], [-2.0, 0.375, 0.625]]], np.float32)
    expected = np.array([[[0.0, 0.0, 0.0], [0.25, 0.5, 1.0]],
                         [[1.0, 0.5, 0.25], [0.0, 0.25, 0.5]]], np.float32)
    np.testing.assert_array_equal(render(rgb, black=0.125, white=1.125), expected)


def test_channels_share_one_multiplier_without_per_channel_gamma_or_midtone():
    rgb = np.tile(np.array([0.25, 0.5, 1.0], np.float32), (2, 2, 1))
    # Peak=white: the entire colour is unchanged even with a strong stretch.
    np.testing.assert_array_equal(render(rgb, asinh_strength=100, midtone=0.01), rgb)
    dim = rgb * np.float32(0.125)
    output = render(dim, asinh_strength=12.0, midtone=0.2)
    np.testing.assert_allclose(output[..., 0] * 4, output[..., 2], rtol=2e-7, atol=0)
    np.testing.assert_allclose(output[..., 1] * 2, output[..., 2], rtol=2e-7, atol=0)
    assert not np.allclose(output, dim)


@pytest.mark.parametrize("k", [0, -0.0, 1e-300, 0.01, 1.0, 12.0, 100])
@pytest.mark.parametrize("m", [1e-4, 0.25, 0.5, 0.75, 1.0 - 1e-4])
def test_scalar_oracle_across_asinh_and_midtone_range(k, m):
    rgb = np.array([[[-0.3, 0.0, 0.002], [0.015, 0.1, 0.9], [0.5, 0.25, 0.0]],
                    [[4.0, 2.0, 0.25], [0.77, 0.81, 0.63], [-1.0, -2.0, -3.0]]], np.float32)
    close_to_scalar(rgb, black=-0.01, white=0.9, asinh_strength=k, midtone=m)


def test_peak_mapping_is_monotone_with_black_and_white_anchors():
    peaks = np.linspace(-0.1, 2.0, 1024, dtype=np.float32)
    rgb = np.tile(peaks[:, None, None], (1, 2, 3))
    result = render(rgb, asinh_strength=35, midtone=0.18)
    assert (np.diff(result[:, 0, 0]) >= 0).all()
    assert (result[peaks <= 0] == 0).all()
    assert (result[peaks >= 1] == 1).all()
    np.testing.assert_array_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_equal(result[..., 1], result[..., 2])


def test_display_remains_linear_at_identity_no_hidden_srgb_encoding():
    rgb = np.full((2, 2, 3), np.float32(0.18))
    np.testing.assert_array_equal(render(rgb), rgb)


def test_identity_does_not_erase_subnormal_or_small_positive_samples():
    minimum = np.nextafter(np.float32(0), np.float32(1))
    tiny = np.finfo(np.float32).tiny
    rgb = np.array([[[minimum, 2 * minimum, 3 * minimum], [tiny, tiny / 2, tiny / 4]],
                    [[1e-20, 1e-30, 1e-40], [0.0, minimum, 0.0]]], np.float32)
    np.testing.assert_array_equal(render(rgb), rgb)


def test_fixed_recipe_does_not_depend_on_other_pixels_or_histogram():
    probe = np.array([[[0.007, 0.011, 0.017], [0.1, 0.2, 0.3]],
                      [[0.001, -0.02, 0.05], [1.0, 2.0, 4.0]]], np.float32)
    larger = np.full((130, 9, 3), 9e10, dtype=np.float32)
    larger[127:129, 3:5] = probe
    recipe = {"asinh_strength": 50.0, "midtone": 0.08}
    np.testing.assert_array_equal(render(larger, **recipe)[127:129, 3:5], render(probe, **recipe))


def test_crosses_two_block_boundaries_and_preserves_pixel_coordinates():
    rng = np.random.default_rng(83629)
    rgb = rng.uniform(-0.4, 3.0, size=(259, 5, 3)).astype(np.float32)
    rgb[[0, 127, 128, 255, 256, 258], 2] = [0.0, 0.2, 0.4]
    changes = {"black": -0.125, "white": 1.5, "asinh_strength": 17, "midtone": 0.23}
    whole = close_to_scalar(rgb, **changes)
    for start, end in [(0, 127), (127, 129), (129, 255), (255, 259)]:
        np.testing.assert_array_equal(whole[start:end], render(rgb[start:end], **changes))
    np.testing.assert_array_equal(whole[::-1, ::-1], render(rgb[::-1, ::-1], **changes))
    np.testing.assert_array_equal(whole[..., ::-1], render(rgb[..., ::-1], **changes))


def test_common_affine_input_unit_change_has_identical_result():
    rgb = np.array([[[-0.25, 0.0, 0.125], [0.25, 0.5, 0.75]],
                    [[1.0, 2.0, 0.5], [0.125, 0.125, 0.125]]], np.float32)
    recipe = {"black": -0.25, "white": 1.75, "asinh_strength": 9, "midtone": 0.3}
    shifted_recipe = recipe | {"black": 4 * recipe["black"] + 2, "white": 4 * recipe["white"] + 2}
    np.testing.assert_array_equal(render(rgb, **recipe), render(4 * rgb + 2, **shifted_recipe))


@pytest.mark.parametrize("mode", ["contiguous", "readonly", "strided", "fortran", "negative_strides"])
def test_input_is_not_mutated_or_aliased_for_supported_array_layouts(mode):
    source = np.arange(180, dtype=np.float32).reshape(10, 6, 3) / 32 - 0.2
    if mode == "strided":
        rgb = source[::2, ::2]
    elif mode == "fortran":
        rgb = np.asfortranarray(source)
    elif mode == "negative_strides":
        rgb = source[::-1, :, ::-1]
    else:
        rgb = source
    if mode == "readonly":
        rgb.flags.writeable = False
    before = rgb.tobytes()
    source_before = source.tobytes()
    out = close_to_scalar(rgb, asinh_strength=20, midtone=0.12)
    assert rgb.tobytes() == before
    assert source.tobytes() == source_before
    assert not np.shares_memory(out, rgb)
    out[0, 0] = 0
    assert rgb.tobytes() == before


@pytest.mark.parametrize(
    "levels",
    [{"black": 0.0, "white": 1e-20},
     {"black": -1e20, "white": 1e20},
     {"black": -1e-20, "white": 0.0},
     {"black": 0, "white": 1}],
)
def test_extreme_finite_float32_values_use_safe_float64_intermediates(levels):
    biggest = np.finfo(np.float32).max
    smallest = np.nextafter(np.float32(0), np.float32(1))
    rgb = np.array([[[biggest, biggest / 2, biggest / 4], [-biggest, 0, smallest]],
                    [[1, 1e-30, -1], [smallest, smallest, smallest]]], np.float32)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        close_to_scalar(rgb, **levels, asinh_strength=100, midtone=1e-4)


def test_signed_zero_and_all_below_black_have_no_division_singularity():
    rgb = np.array([[[0.0, -0.0, 0.0], [-1.0, -2.0, -3.0]],
                    [[-0.0, -0.0, -0.0], [-1e20, 0.0, -1e-30]]], np.float32)
    with np.errstate(all="raise"):
        out = render(rgb, asinh_strength=100, midtone=1e-4)
    np.testing.assert_array_equal(out, np.zeros_like(rgb))


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (1, 2, 3), (2, 1, 3),
                                  (2, 2), (2, 2, 1), (2, 2, 4), (2, 2, 3, 1), ()])
def test_rejects_unsupported_shapes(shape):
    with pytest.raises(ValueError):
        render(np.zeros(shape, np.float32))


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int16, np.uint16, np.bool_, np.complex64, object])
def test_rejects_implicit_input_dtype_conversion(dtype):
    with pytest.raises(ValueError):
        render(np.zeros((2, 2, 3), dtype=dtype))


class ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize("bad", [None, 0, [[[0.0] * 3] * 2] * 2,
                                 np.zeros((2, 2, 3), np.float32).view(ArraySubclass),
                                 np.ma.array(np.zeros((2, 2, 3), np.float32))])
def test_rejects_non_plain_ndarray_and_subclasses(bad):
    with pytest.raises(ValueError):
        render(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("row", [0, 127, 128, 256])
def test_rejects_nonfinite_pixels_in_every_block_without_mutating_input(bad, row):
    rgb = np.ones((258, 2, 3), np.float32)
    rgb[row, 1, 2] = bad
    before = rgb.tobytes()
    with pytest.raises(ValueError):
        render(rgb)
    assert rgb.tobytes() == before


class FloatSubclass(float):
    pass


@pytest.mark.parametrize("name", ["black", "white", "asinh_strength", "midtone"])
@pytest.mark.parametrize("bad", [True, False, None, "0.5", np.float32(0.5), np.float64(0.5),
                                 np.int64(1), Decimal("0.5"), Fraction(1, 2), FloatSubclass(0.5),
                                 float("nan"), float("inf"), -float("inf")])
def test_rejects_non_builtin_or_nonfinite_scalar_parameters(name, bad):
    rgb = np.zeros((2, 2, 3), np.float32)
    with pytest.raises(ValueError):
        render(rgb, **{name: bad})


@pytest.mark.parametrize(
    "changes",
    [{"black": -1.0000000001e20}, {"black": 1.0000000001e20},
     {"white": -1.0000000001e20}, {"white": 1.0000000001e20},
     {"black": 1.0, "white": 1.0}, {"black": 2, "white": 1},
     {"white": float(np.nextafter(1e-20, 0.0))},
     {"asinh_strength": -1e-300}, {"asinh_strength": 100.00000001},
     {"midtone": float(np.nextafter(1e-4, 0.0))},
     {"midtone": float(np.nextafter(1 - 1e-4, 1.0))},
     {"midtone": 0}, {"midtone": 1}, {"white": 10 ** 1000},
     {"white": 10 ** 20 + 1}, {"black": -(10 ** 20) - 1}],
)
def test_rejects_out_of_range_or_unrepresentable_parameters(changes):
    with pytest.raises(ValueError):
        render(np.zeros((2, 2, 3), np.float32), **changes)


def test_repeated_calls_reproduce_bits_and_do_not_retain_mutable_output_state():
    rng = np.random.default_rng(3737)
    rgb = rng.uniform(-0.1, 1.4, size=(130, 7, 3)).astype(np.float32)
    recipe = {"black": -0.02, "white": 1.2, "asinh_strength": 28, "midtone": 0.24}
    first = render(rgb, **recipe)
    original_bytes = first.tobytes()
    render(rgb * 2, asinh_strength=1, midtone=0.8)
    second = render(rgb, **recipe)
    assert second.tobytes() == original_bytes
    first[...] = 1.0
    assert second.tobytes() == original_bytes
