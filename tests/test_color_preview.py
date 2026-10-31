import numpy as np
import pytest
from PIL import Image

from starstack.color_preview import preview_recipe, render_preview, save_preview


def test_fixed_preview_does_not_fit_each_frame_or_mutate_master(tmp_path):
    rgb = np.full((32, 48, 3), .04, np.float32)
    rgb[10, 15] = [-.2, .3, 2.0]
    original = rgb.copy()
    recipe = preview_recipe(0)
    first = render_preview(rgb, recipe)
    other = rgb.copy()
    other[-3:, -3:] = 100
    second = render_preview(other, recipe)
    np.testing.assert_array_equal(first[:-3, :-3], second[:-3, :-3])
    np.testing.assert_array_equal(rgb, original)
    assert first[10, 15, 0] == 0 and first[10, 15, 2] == 255
    result = save_preview(tmp_path/'test.jpg', rgb, recipe)
    assert result['dimensions'] == [48, 32]
    with Image.open(tmp_path/'test.jpg') as im:
        assert im.info['icc_profile']


def test_signed_average_before_clipping_and_single_encoding():
    rgb = np.tile(np.array([-.1, .1], np.float32)[None, :, None], (2, 1, 3))
    assert not render_preview(rgb, preview_recipe(), max_side=1).any()
    # The small linear value goes through luminance tone and sRGB exactly once.
    value = .01
    rgb = np.full((2, 2, 3), value, np.float32)
    tone = np.arcsinh(8*value)/np.arcsinh(8)
    expected = round((1.055*tone**(1/2.4)-.055)*255)
    assert np.all(render_preview(rgb, preview_recipe(0)) == expected)


@pytest.mark.parametrize('ev', [True, float('nan'), float('inf'), 13, -13, '2'])
def test_invalid_display_exposure(ev):
    with pytest.raises(ValueError):
        preview_recipe(ev)


def test_invalid_array_and_recipe():
    rgb = np.ones((3, 5, 3), np.float32)
    recipe = preview_recipe()
    recipe['automatic_brightness'] = True
    with pytest.raises(ValueError):
        render_preview(rgb, recipe)
    rgb[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        render_preview(rgb, preview_recipe())
