"""Numerical contracts for the float Bayer baseline, with a scalar oracle."""
import numpy as np
import pytest

from starstack.float_demosaic import demosaic_bilinear, rgb_pattern


PATTERNS = [
    pytest.param([[0, 1], [1, 2]], id="RGGB"),
    pytest.param([[1, 0], [2, 1]], id="GRBG"),
    pytest.param([[1, 2], [0, 1]], id="GBRG"),
    pytest.param([[2, 1], [1, 0]], id="BGGR"),
]


def color_at(pattern, y, x):
    color = int(pattern[y % 2][x % 2])
    return 1 if color == 3 else color


def scalar_reference(cfa, pattern):
    """At missing Bayer sites all available same-color neighbours weigh equally.

    This enumerates sensor samples with scalar arithmetic; it does not reuse
    the production convolution, its mask builder, or its row-tile traversal.
    Measured samples bypass interpolation, including signed zero.
    """
    h, w = cfa.shape
    rgb = np.empty((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            for channel in range(3):
                if color_at(pattern, y, x) == channel:
                    rgb[y, x, channel] = cfa[y, x]
                    continue
                values = [float(cfa[sy, sx])
                          for sy in range(max(y - 1, 0), min(y + 2, h))
                          for sx in range(max(x - 1, 0), min(x + 2, w))
                          if color_at(pattern, sy, sx) == channel]
                assert values
                rgb[y, x, channel] = sum(values) / len(values)
    return rgb


def mosaic(channels, pattern):
    h, w = channels.shape[:2]
    return np.asarray([[channels[y, x, color_at(pattern, y, x)]
                        for x in range(w)] for y in range(h)], dtype=np.float32)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("shape", [(2, 2), (3, 5), (8, 9), (11, 6)])
def test_matches_independent_scalar_reference_at_edges_and_interior(pattern, shape):
    rng = np.random.default_rng(20260916)
    cfa = rng.integers(-32, 80, size=shape).astype(np.float32) / 8
    original = cfa.copy()
    actual = demosaic_bilinear(cfa, pattern, block_rows=3)
    np.testing.assert_array_equal(actual, scalar_reference(cfa, pattern))
    np.testing.assert_array_equal(cfa.view(np.uint32), original.view(np.uint32))
    assert actual.dtype == np.float32 and actual.shape == (*shape, 3)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_sensor_samples_kept_bitwise_including_negative_zero_and_extremes(pattern):
    values = np.array([-0., 0., -2.5, .125, 1., 9., np.finfo(np.float32).max,
                       -np.finfo(np.float32).max,
                       np.nextafter(np.float32(0), np.float32(1)),
                       -np.nextafter(np.float32(0), np.float32(1))], dtype=np.float32)
    cfa = np.resize(values, (9, 13))
    actual = demosaic_bilinear(cfa, pattern, block_rows=2)
    assert np.isfinite(actual).all()
    for y, x in np.ndindex(cfa.shape):
        ch = color_at(pattern, y, x)
        assert actual.view(np.uint32)[y, x, ch] == cfa.view(np.uint32)[y, x]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_each_channel_constant_is_preserved_including_borders(pattern):
    rgb = np.broadcast_to(np.array([-.125, .5, 7.75], dtype=np.float32), (7, 9, 3)).copy()
    actual = demosaic_bilinear(mosaic(rgb, pattern), pattern, block_rows=1)
    np.testing.assert_array_equal(actual, rgb)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_asymmetric_top_down_affine_planes_are_recovered_interior(pattern):
    y, x = np.mgrid[:9, :13]
    # All coefficients are dyadic, so the expected interior values are exact.
    rgb = np.stack((2 + y / 4 + x / 8, -4 + y / 2 - x / 4,
                    10 - y / 8 + x / 2), axis=-1).astype(np.float32)
    actual = demosaic_bilinear(mosaic(rgb, pattern), pattern, block_rows=3)
    np.testing.assert_array_equal(actual[1:-1, 1:-1], rgb[1:-1, 1:-1])
    assert actual[2, 3, 0] < actual[6, 3, 0]
    assert actual[2, 3, 2] > actual[6, 3, 2]
    assert actual[2, 3, 0] != actual[2, 9, 0]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_linearity_and_affine_intensity_response_without_global_normalization(pattern):
    rng = np.random.default_rng(72)
    a = rng.integers(-50, 90, size=(9, 11)).astype(np.float32) / 8
    b = rng.integers(-30, 40, size=a.shape).astype(np.float32) / 16
    da, db = demosaic_bilinear(a, pattern), demosaic_bilinear(b, pattern)
    actual = demosaic_bilinear(2 * a - .5 * b, pattern)
    np.testing.assert_allclose(actual, 2 * da - .5 * db, rtol=3e-7, atol=2e-6)
    shifted = demosaic_bilinear(.5 * a - 2, pattern)
    np.testing.assert_allclose(shifted, .5 * da - 2, rtol=3e-7, atol=1e-6)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_distant_outlier_cannot_change_an_unrelated_roi(pattern):
    cfa = np.arange(143, dtype=np.float32).reshape(11, 13) / 16 - 3
    changed = cfa.copy()
    changed[0, 0] = np.finfo(np.float32).max
    baseline = demosaic_bilinear(cfa, pattern, block_rows=3)
    actual = demosaic_bilinear(changed, pattern, block_rows=3)
    np.testing.assert_array_equal(actual[3:9, 3:11], baseline[3:9, 3:11])
    assert not np.array_equal(actual[:2, :2], baseline[:2, :2])


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("block_rows", [1, 2, 3, 5, 128, 4096])
def test_tiles_and_odd_sizes_match_whole_frame_without_seams(pattern, block_rows):
    cfa = np.random.default_rng(17).normal(size=(13, 17)).astype(np.float32)
    expected = scalar_reference(cfa, pattern)
    actual = demosaic_bilinear(cfa, pattern, block_rows=block_rows)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_odd_origin_crop_matches_whole_frame_when_phase_and_halo_follow_source(pattern):
    cfa = np.random.default_rng(6).normal(size=(15, 17)).astype(np.float32)
    full = demosaic_bilinear(cfa, pattern)
    y, x = 3, 5
    shifted_pattern = [[pattern[(y + dy) % 2][(x + dx) % 2] for dx in range(2)] for dy in range(2)]
    crop = demosaic_bilinear(cfa[y:12, x:14], shifted_pattern, block_rows=3)
    np.testing.assert_array_equal(crop[1:-1, 1:-1], full[y + 1:11, x + 1:13])


@pytest.mark.parametrize("pattern", PATTERNS)
def test_four_slot_green_encoding_is_equivalent_and_pattern_not_mutated(pattern):
    four = np.array(pattern, dtype=np.uint8)
    green = np.argwhere(four == 1)[1]
    four[tuple(green)] = 3
    original = four.copy()
    cfa = np.arange(35, dtype=np.float32).reshape(5, 7) - 18
    np.testing.assert_array_equal(demosaic_bilinear(cfa, four), demosaic_bilinear(cfa, pattern))
    np.testing.assert_array_equal(four, original)


def test_noncontiguous_readonly_input_and_caller_supplied_output(tmp_path):
    source = np.arange(140, dtype=np.float32).reshape(10, 14)[::2, ::2]
    source.flags.writeable = False
    out = np.lib.format.open_memmap(tmp_path / "rgb.npy", mode="w+", dtype=np.float32, shape=(5, 7, 3))
    returned = demosaic_bilinear(source, [[0, 1], [1, 2]], out=out, block_rows=2)
    assert returned is out
    np.testing.assert_array_equal(out, scalar_reference(source, [[0, 1], [1, 2]]))
    out.flush()
    del returned, out


@pytest.mark.parametrize("bad", [None, [], [[0, 1, 2]], [[0., 1.], [1., 2.]],
                                [[True, False], [False, True]], [[0, 1], [1, 4]],
                                [[-1, 1], [1, 2]], [[0, 0], [2, 2]], [[0, 2], [1, 1]],
                                [[0, 1], [2]], [["0", "1"], ["1", "2"]]])
def test_rejects_unverified_bayer_pattern(bad):
    with pytest.raises(ValueError):
        rgb_pattern(bad)


@pytest.mark.parametrize("bad", [None, [[1., 2.], [3., 4.]],
                                np.zeros((2, 2), np.float64), np.zeros((2, 2), np.uint16),
                                np.zeros((2, 2, 1), np.float32), np.zeros((1, 4), np.float32),
                                np.zeros((0, 2), np.float32)])
def test_rejects_invalid_cfa(bad):
    with pytest.raises(ValueError):
        demosaic_bilinear(bad, [[0, 1], [1, 2]])


@pytest.mark.parametrize("bad", [True, 0, -1, 4097, 2., "2", None, np.int64(2)])
def test_rejects_invalid_block_rows(bad):
    with pytest.raises(ValueError):
        demosaic_bilinear(np.zeros((4, 4), np.float32), [[0, 1], [1, 2]], block_rows=bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", [(0, 0), (3, 4), (8, 10)])
def test_rejects_nonfinite_input_even_in_late_tile(value, location):
    cfa = np.zeros((9, 11), dtype=np.float32)
    cfa[location] = value
    original = cfa.view(np.uint32).copy()
    with pytest.raises(ValueError):
        demosaic_bilinear(cfa, [[0, 1], [1, 2]], block_rows=2)
    np.testing.assert_array_equal(cfa.view(np.uint32), original)


@pytest.mark.parametrize("bad_out", [np.zeros((4, 4, 3), np.float64), np.zeros((4, 4), np.float32),
                                    np.zeros((4, 4, 4), np.float32), []])
def test_rejects_invalid_output(bad_out):
    with pytest.raises(ValueError):
        demosaic_bilinear(np.zeros((4, 4), np.float32), [[0, 1], [1, 2]], out=bad_out)


def test_rejects_readonly_output():
    out = np.zeros((4, 4, 3), np.float32)
    out.flags.writeable = False
    with pytest.raises(ValueError):
        demosaic_bilinear(np.zeros((4, 4), np.float32), [[0, 1], [1, 2]], out=out)


def test_rejects_in_memory_input_output_alias_before_writing():
    out = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
    source = out[:, :, 0]
    original = out.copy()
    with pytest.raises(ValueError):
        demosaic_bilinear(source, [[0, 1], [1, 2]], out=out)
    np.testing.assert_array_equal(out, original)


def test_rejects_separately_mapped_same_file_before_writing(tmp_path):
    path = tmp_path / "backing.bin"
    np.arange(48, dtype=np.float32).tofile(path)
    original = path.read_bytes()
    source = np.memmap(path, dtype=np.float32, mode="r+", shape=(4, 4))
    out = np.memmap(path, dtype=np.float32, mode="r+", shape=(4, 4, 3))
    try:
        assert not np.shares_memory(source, out)
        with pytest.raises(ValueError):
            demosaic_bilinear(source, [[0, 1], [1, 2]], out=out)
        assert path.read_bytes() == original
    finally:
        del out, source
