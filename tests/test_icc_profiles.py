"""icc_profiles: gamut colorimetry and the minimal ICC writer/reader, cross-checked against lcms (Pillow ImageCms)."""
import io

import numpy as np
import pytest
from PIL import Image, ImageCms

from starstack import icc_profiles as icc


def test_luminance_weights_match_published_values():
    assert np.allclose(icc.Gamut("srgb").luma, [0.2126, 0.7152, 0.0722], atol=2e-4)
    assert np.allclose(icc.Gamut("rec2020").luma, [0.2627, 0.6780, 0.0593], atol=2e-4)
    for name in icc.GAMUTS:
        g = icc.Gamut(name)
        assert abs(g.luma.sum() - 1.0) < 1e-9
        assert np.allclose(g.rgb_to_xyz @ np.ones(3), icc.xy_to_xyz(g.white_xy), atol=1e-12)


def test_d50_colorants_match_adobe_and_lcms_srgb():
    # Adobe SDK named-space sRGB (D50) rows, as used in experimental_dcp
    adobe = np.array([[.4361, .3851, .1431], [.2225, .7169, .0606], [.0139, .0971, .7141]])
    assert np.abs(icc.Gamut("srgb").colorants_d50 - adobe).max() < 1e-3
    lcms = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).profile
    assert np.allclose(icc.Gamut("srgb").colorants_d50[:, 0], lcms.red_colorant[0], atol=1e-3)
    prophoto = np.array([[.7977, .1352, .0313], [.2880, .7119, .0001], [0., 0., .8249]])
    assert np.abs(icc.Gamut("prophoto").colorants_d50 - prophoto).max() < 1e-3


def test_gamut_conversion_round_trips_and_keeps_white():
    a, b = icc.Gamut("rec2020"), icc.Gamut("srgb")
    m = a.matrix_to(b)
    assert np.allclose(m @ np.linalg.inv(m), np.eye(3), atol=1e-12)
    assert np.allclose(m @ np.ones(3), np.ones(3), atol=1e-9)             # D65 white → D65 white
    p = icc.Gamut("prophoto")
    assert np.allclose(b.matrix_to(p) @ np.ones(3), np.ones(3), atol=1e-6)  # Bradford keeps white on white


@pytest.mark.parametrize("name", sorted(icc.GAMUTS))
def test_profile_round_trip_and_identification(name):
    data = icc.write_linear_profile(name)
    assert data == icc.write_linear_profile(name)          # deterministic bytes
    info = icc.read_profile(data)
    assert info["class"] == "mntr" and info["color_space"] == "RGB " and info["pcs"] == "XYZ "
    assert info["description"] == icc.GAMUTS[name]["description"]
    assert info["trc"] == {"r": "linear", "g": "linear", "b": "linear"}
    assert np.abs(info["colorants_d50"] - icc.Gamut(name).colorants_d50).max() < 2e-5
    assert np.allclose(info["wtpt"], icc.xy_to_xyz(icc.D50_XY), atol=2e-5)
    assert icc.identify_linear_profile(data) == name
    tampered = bytearray(data)
    tampered[-40:] = bytes(40)      # damage the shared TRC element / a colorant
    with pytest.raises(ValueError):
        icc.identify_linear_profile(bytes(tampered))


def test_lcms_reads_our_profile_and_agrees_with_the_matrix_maths():
    data = icc.write_linear_profile("rec2020")
    profile = ImageCms.ImageCmsProfile(io.BytesIO(data))
    assert profile.profile.profile_description == icc.GAMUTS["rec2020"]["description"]
    assert profile.profile.xcolor_space.strip() == "RGB"
    assert np.allclose(profile.profile.red_colorant[0], icc.Gamut("rec2020").colorants_d50[:, 0], atol=1e-4)
    transform = ImageCms.buildTransform(profile, ImageCms.createProfile("sRGB"), "RGB", "RGB",
                                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    colours = np.array([[128, 128, 128], [128, 64, 32], [40, 120, 60], [30, 30, 200], [200, 180, 20], [10, 10, 10], [250, 250, 250]], np.uint8)
    out = np.asarray(ImageCms.applyTransform(Image.fromarray(colours.reshape(1, -1, 3)), transform)).reshape(-1, 3).astype(float)
    linear = colours / 255.0 @ icc.Gamut("rec2020").matrix_to(icc.SRGB).T
    expected = icc.srgb_encode(np.clip(linear, 0, 1)) * 255
    assert np.abs(out - expected).max() <= 1.0


def test_srgb_transfer_functions_invert():
    x = np.linspace(0, 1, 1001)
    assert np.allclose(icc.srgb_decode(icc.srgb_encode(x)), x, atol=1e-12)
    assert icc.srgb_encode(np.array(0.0)) == 0 and abs(float(icc.srgb_encode(np.array(1.0))) - 1) < 1e-12


def test_read_profile_rejects_garbage():
    with pytest.raises(ValueError):
        icc.read_profile(b"not a profile")
    with pytest.raises(ValueError):
        icc.identify_linear_profile(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())   # gamma curves, not linear
