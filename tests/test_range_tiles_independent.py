"""Independent scalar/IEEE-754 checks against the frozen tile contract.

Expected arithmetic uses math.ldexp and struct's binary32 rounding, never the
production function or NumPy's vectorized ldexp. This tests a tile adapter, not
Siril normalization, rejection, registration, or scientific pixel validity.
"""
from __future__ import annotations

import math
import random
import struct
import sys
import warnings

import numpy as np
import pytest

from starstack.range_tiles import choose_divisor, rescale_positive_tile


def _from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _scalar_oracle(bits, divisor, restore):
    """Return exact expected binary32 bits or reject per scalar contract."""
    value = _from_bits(bits)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid sample")
    if restore and value > 0 and value >= 1:
        raise ValueError("encoded sample must be below one")
    exponent = divisor.bit_length() - 1
    if not restore:
        exponent = -exponent
    try:
        result_bits = _bits(math.ldexp(value, exponent))
        result = _from_bits(result_bits)
        inverse_bits = _bits(math.ldexp(result, -exponent))
    except (OverflowError, ValueError) as exc:
        raise ValueError("binary32 overflow") from exc
    if (not math.isfinite(result)
            or (value > 0 and result <= 0)
            or (not restore and value > 0 and result >= 1)
            or inverse_bits != bits):
        raise ValueError("not an exactly reversible admissible sample")
    return result_bits


@pytest.fixture(autouse=True)
def no_runtime_warning_leak():
    """Adversarial inputs must produce ValueError, not floating warnings."""
    with warnings.catch_warnings(record=True) as recorded, np.errstate(all="warn"):
        warnings.simplefilter("always")
        yield
    runtime = [item for item in recorded if issubclass(item.category, RuntimeWarning)]
    assert not runtime, [str(item.message) for item in runtime]


@pytest.mark.parametrize("value, expected", [
    (0, 1), (-0.0, 1), (0.5, 1), (.75, 1),
    (math.nextafter(.75, math.inf), 2),
    (math.nextafter(.75, 0), 1), (1, 2), (1.167, 2),
    (1.5, 2), (math.nextafter(1.5, math.inf), 4),
    (3, 4), (math.nextafter(3, math.inf), 8),
    (6, 8), (math.nextafter(6, math.inf), 16), (8, 16),
    (np.float16(.75), 1), (np.float32(.75), 1),
    (np.float32(_from_bits(0x3F400001)), 2),
    (np.float64(1.167), 2), (np.int8(8), 16), (np.uint64(8), 16),
])
def test_choose_smallest_power_and_builtin_return(value, expected):
    result = choose_divisor(value)
    assert type(result) is int
    assert result == expected
    assert float(value) / result <= .75
    assert result == 1 or float(value) / (result // 2) > .75


class _MalformedFloat(float):
    def __float__(self):
        raise TypeError("malformed float conversion")


@pytest.mark.parametrize("value", [
    True, False, np.bool_(True), None, "0.5", [], {}, .5 + 0j,
    np.complex64(.5), np.array(.5), np.array([.5]),
    -.1, math.nextafter(0, -math.inf), 9,
    math.nextafter(8, math.inf), math.inf, -math.inf, math.nan,
    np.float32(math.nan), np.uint64(2**64 - 1), 10**10000,
    _MalformedFloat(.5),
], ids=lambda value: type(value).__name__)
def test_choose_rejects_malformed_scalar_with_valueerror(value):
    with pytest.raises(ValueError):
        choose_divisor(value)


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("divisor", [1, 2, 4, 16, 256, 65536])
def test_scalar_ieee_oracle_boundary_and_random_bits(divisor, restore):
    boundaries = [
        0, 0x80000000, 1, 2, 3, 4, 5, 0x7F, 0x80, 0xFF, 0x100,
        0xFFFF, 0x10000, 0x10001, 0x7FFFFE, 0x7FFFFF, 0x800000,
        0x800001, 0x800002, 0xFFFFFE, 0xFFFFFF, 0x1000000,
        0x3E800000, 0x3F000000, 0x3F400000, 0x3F7FFFFF,
        0x3F800000, 0x3F800001, 0x40000000, 0x41000000,
        0x477FFF00, 0x47800000, 0x7F7FFFFE, 0x7F7FFFFF,
    ]
    rng = random.Random(0x20260920)
    patterns = boundaries + [rng.randrange(1, 0x7F800000) for _ in range(160)]
    accepted = rejected = 0
    for bits in patterns:
        source = np.full((3, 1, 1), _from_bits(bits), dtype=np.float32)
        original = source.tobytes()
        try:
            expected = _scalar_oracle(bits, divisor, restore)
        except ValueError:
            with pytest.raises(ValueError):
                rescale_positive_tile(source, divisor, restore=restore)
            rejected += 1
        else:
            result = rescale_positive_tile(source, divisor, restore=restore)
            assert type(result) is np.ndarray
            assert result.dtype == np.dtype("float32")
            assert result.shape == source.shape
            assert result.flags.c_contiguous
            assert not np.shares_memory(source, result)
            assert np.all(result.view(np.uint32) == expected), (hex(bits), divisor, restore)
            accepted += 1
        assert source.tobytes() == original
    assert accepted > 10
    assert rejected > 10


@pytest.mark.parametrize("bits, divisor, succeeds", [
    (1, 2, False),       # underflows to zero
    (2, 2, True),        # smallest exactly encodable nonzero subnormal
    (3, 2, False),       # rounds up but loses an input bit
    (4, 4, True),
    (5, 2, False),       # rounds down but remains nonzero
    (0x10000, 65536, True),
    (0x10001, 65536, False),
    (0x800000, 2, True), # normal -> exact subnormal
    (0x800001, 2, False),
    (0x800002, 2, True),
])
def test_subnormal_reversibility_is_stricter_than_nonzero(bits, divisor, succeeds):
    source = np.full((3, 2, 2), _from_bits(bits), dtype=np.float32)
    source.flags.writeable = False
    if not succeeds:
        with pytest.raises(ValueError):
            rescale_positive_tile(source, divisor)
        return
    encoded = rescale_positive_tile(source, divisor)
    expected = _scalar_oracle(bits, divisor, False)
    assert np.all(encoded.view(np.uint32) == expected)
    restored = rescale_positive_tile(encoded, divisor, restore=True)
    np.testing.assert_array_equal(restored.view(np.uint32), source.view(np.uint32))


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("divisor", [1, 2, 65536])
def test_zero_signs_and_identity_still_allocate(divisor, restore):
    signs = np.array([
        [[0, 0x80000000], [0x80000000, 0]],
        [[0x80000000, 0], [0x80000000, 0]],
        [[0, 0], [0x80000000, 0x80000000]],
    ], dtype=np.uint32)
    source = signs.view(np.float32)
    source.flags.writeable = False
    result = rescale_positive_tile(source, divisor, restore=restore)
    np.testing.assert_array_equal(result.view(np.uint32), signs)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)
    result[0, 0, 0] = .5
    np.testing.assert_array_equal(source.view(np.uint32), signs)


@pytest.mark.parametrize("restore", [False, True])
def test_strided_readonly_channels_use_same_exact_factor(restore):
    storage = np.arange(1, 1 + 3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8) / np.float32(256)
    source = storage[::-1, ::-2, 1::2]
    source[:, 1, 1] = np.array([-0.0, 0.0, -0.0], dtype=np.float32)
    source.flags.writeable = False
    original = storage.tobytes()
    assert not source.flags.c_contiguous
    result = rescale_positive_tile(source, 8, restore=restore)
    expected = np.empty(source.shape, dtype=np.uint32)
    for index in np.ndindex(source.shape):
        expected[index] = _scalar_oracle(_bits(float(source[index])), 8, restore)
    np.testing.assert_array_equal(result.view(np.uint32), expected)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)
    assert not np.shares_memory(result, storage)
    assert storage.tobytes() == original


def test_restore_preserves_highlights_above_one():
    source = np.array([.25, .5, .75], dtype=np.float32).reshape(3, 1, 1)
    result = rescale_positive_tile(source, 65536, restore=True)
    expected = np.array([16384, 32768, 49152], dtype=np.float32).reshape(3, 1, 1)
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    assert np.all(result > 1)


class _ArraySubclass(np.ndarray):
    pass


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("data", [
    None, [[[.5]]], np.float32(.5),
    np.ones((3, 2, 2), dtype=np.float64),
    np.ones((3, 2, 2), dtype=np.float16),
    np.ones((3, 2, 2), dtype=np.int32),
    np.ones((3, 2, 2), dtype=bool),
    np.ones((3, 2, 2), dtype=object),
    np.ones((3, 2, 2), dtype=np.complex64),
    np.ones((3, 2, 2), dtype=np.dtype(">f4" if sys.byteorder == "little" else "<f4")),
    np.ones((3, 2, 2), dtype=np.float32).view(_ArraySubclass),
    np.ma.array(np.ones((3, 2, 2), dtype=np.float32)),
])
def test_rejects_wrong_array_type_or_dtype(data):
    with pytest.raises(ValueError):
        rescale_positive_tile(data, 2)


@pytest.mark.parametrize("shape", [(), (3,), (3, 2), (2, 2, 2), (4, 2, 2), (3, 0, 2), (3, 2, 0), (3, 2, 2, 1)])
def test_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        rescale_positive_tile(np.full(shape, .5, dtype=np.float32), 2)


@pytest.mark.parametrize("divisor", [None, True, False, np.bool_(True), 2.0, np.int32(2), np.uint64(2), _IntSubclass(2), "2", 0, -2, 3, 65537, 131072, 10**10000], ids=lambda value: type(value).__name__)
def test_rejects_wrong_divisor(divisor):
    with pytest.raises(ValueError):
        rescale_positive_tile(np.full((3, 2, 2), .5, dtype=np.float32), divisor)


@pytest.mark.parametrize("restore", [None, 0, 1, 0.0, "False", np.bool_(True)])
def test_restore_requires_builtin_bool(restore):
    with pytest.raises(ValueError):
        rescale_positive_tile(np.full((3, 2, 2), .5, dtype=np.float32), 2, restore=restore)


@pytest.mark.parametrize("triplet", [
    [0, .5, .5], [.5, -0.0, .5], [0, 0, .5],
    [-.5, .5, .5], [-_from_bits(1), .5, .5],
    [math.inf, .5, .5], [-math.inf, .5, .5], [math.nan, .5, .5],
    [_from_bits(0x7F7FFFFF)] * 3,
])
@pytest.mark.parametrize("restore", [False, True])
def test_invalid_pixels_do_not_mutate_input_or_emit_warning(triplet, restore):
    data = np.full((3, 2, 3), .5, dtype=np.float32)
    data[:, 1, 2] = triplet
    before = data.tobytes()
    data.flags.writeable = False
    with pytest.raises(ValueError):
        rescale_positive_tile(data, 65536, restore=restore)
    assert data.tobytes() == before


@pytest.mark.parametrize("value, divisor, restore", [(1, 1, False), (2, 2, False), (1, 2, True), (1.01, 2, True)])
def test_open_upper_bound_for_encoded_positive_data(value, divisor, restore):
    with pytest.raises(ValueError):
        rescale_positive_tile(np.full((3, 1, 1), value, dtype=np.float32), divisor, restore=restore)
