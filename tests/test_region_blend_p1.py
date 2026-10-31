import numpy as np
import pytest
from starstack.region_blend import inward_weight, RECIPE


def test_half_plane_has_smooth_inward_ramp_and_unchanged_ground():
    weight = np.ones((40, 30), np.float32)
    weight[20:] = 0
    before = weight.copy()
    out, receipt = inward_weight(weight, width_px=8)
    expected = np.clip((np.arange(20, 0, -1, dtype=float) - 1) / 8, 0, 1)
    expected = expected**2 * (3 - 2 * expected)
    assert out[:20, 0] == pytest.approx(expected, abs=1e-7)
    assert np.array_equal(out[20:], before[20:])
    assert np.array_equal(weight, before)
    assert np.all(out <= weight)
    assert receipt['recipe'] == RECIPE and receipt['width_px'] == 8


def test_small_holes_use_actual_distance_and_do_not_expand_support():
    source = np.ones((11, 13), np.float32)
    source[2, 4] = source[8, 9] = 0
    source[4, 4] = .2
    result, _ = inward_weight(source, width_px=3)
    for y in range(11):
        for x in range(13):
            distance = min(abs(y - 2) + abs(x - 4), abs(y - 8) + abs(x - 9))
            t = min(1., max(0., (distance - 1) / 3))
            expected = min(float(source[y, x]), t*t*(3-2*t))
            assert float(result[y, x]) == pytest.approx(expected, abs=1e-7)


def test_tile_boundaries_are_identical_and_border_not_exclusion():
    source = np.ones((810, 320), np.float32)
    source[250:265, 100:140] = 0
    source[508:517, 210:215] = 0
    result, _ = inward_weight(source, width_px=64)
    from starstack.milkyway import _inward_feather
    assert np.array_equal(result, _inward_feather(source == 0, 64))
    assert np.all(result[:, 0] == 1) and np.all(result[:, -1] == 1)
    full = np.ones((12, 20), np.float32)
    assert np.array_equal(inward_weight(full)[0], full)


def test_convex_blend_avoids_hard_step_without_altering_ground():
    old = np.ones((100, 200), np.float32)
    old[50:] = 0
    weight, _ = inward_weight(old, width_px=20)
    sky, ground = np.float32(.25), np.float32(.1)
    blend = sky * weight + ground * (1 - weight)
    assert np.all(blend[50:] == ground)
    assert np.max(np.abs(np.diff(blend[:, 100]))) < .012
    assert np.all(blend >= ground) and np.all(blend <= sky)


@pytest.mark.parametrize('bad', [True, 0, 513, 2.5, float('nan')])
def test_invalid_width_rejected(bad):
    with pytest.raises(ValueError):
        inward_weight(np.ones((4, 4), np.float32), width_px=bad)


def test_invalid_weight_and_cancellation():
    for bad in [np.zeros((4, 4), np.float64), np.full((4, 4), np.nan, np.float32), np.full((4, 4), 1.1, np.float32)]:
        with pytest.raises(ValueError):
            inward_weight(bad)
    def cancel():
        raise RuntimeError('cancel')
    with pytest.raises(RuntimeError, match='cancel'):
        inward_weight(np.ones((4, 4), np.float32), check=cancel)
