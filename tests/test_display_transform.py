"""display_transform: the pre-registered synthetic checks M2 (hue), M3 (neutral axis), M4 (highlight desaturation) of
docs/COLOR_CONTRACT_PACKAGE_20260922.md, plus parameter and flag behaviour."""
import colorsys
import sys
from pathlib import Path

import numpy as np
import pytest

from starstack import display_transform as dt, icc_profiles as icc, imaging, master_export as me, raw_color

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402


def _params(**kw):
    fields = {"gamut": "rec2020", "anchor": 0.02, "target": 0.22}
    fields.update(kw)
    return dt.Params(**fields).validate()


def _decode(rgb16):
    return icc.srgb_decode(rgb16.astype(np.float64) / 65535.0)


# ---- parameters ----------------------------------------------------------------------------------------------------

def test_params_validate_and_round_trip():
    p = _params()
    j = p.to_json()
    assert j["version"] == dt.VERSION and 0 < j["midtone"] < 1
    assert dt.Params.from_json(j) == p
    bad = dict(j, midtone=j["midtone"] * 1.01)
    with pytest.raises(ValueError):
        dt.Params.from_json(bad)
    for kw in ({"anchor": 0.0}, {"target": 0.999}, {"asinh_strength": -1}, {"saturation": 3}, {"white": 0.0},
               {"highlight_desat_end": 0.5}, {"look": "magic"}, {"gamut": "adobe"}):
        with pytest.raises(ValueError):
            _params(**kw)


def test_tone_curve_is_monotone_and_hits_anchor_target():
    p = _params()
    x = np.linspace(0, 1, 2001)
    t = dt.tone_curve(x, p)
    assert t[0] == 0 and abs(t[-1] - 1) < 1e-12 and np.all(np.diff(t) >= 0)
    at_anchor = float(dt.tone_curve(np.array([p.anchor]), p)[0])
    assert abs(float(icc.srgb_encode(np.array(at_anchor))) - p.target) < 1e-9


# ---- M3 neutral axis ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("black", [0.0, 0.004])
def test_m3_neutral_ramp_stays_neutral_monotone_and_reaches_white(black):
    ramp = np.linspace(-0.05, 2.0, 8192).astype(np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], 3, axis=2), 2, axis=0)
    out, stats = dt.render(img, _params(black=black, anchor=0.03))
    row = out[0].astype(int)
    assert (row.max(axis=1) - row.min(axis=1)).max() <= 1                                  # ≤ 1 code between channels
    assert np.all(np.diff(row[:, 1]) >= 0)                                                  # monotone
    assert row[ramp >= 1.0].min() == 65535                                                  # ≥ white → white
    partial = (row.min(axis=1) == 0) & (row.max(axis=1) > 0)
    assert int(partial.sum()) == 0                                                          # no partial zeros
    assert np.all(row[ramp <= black].max(axis=1) == 0) and np.all(row[ramp > black + 1e-3].min(axis=1) > 0)
    assert stats["luminance_at_or_below_black"] == 2 * int((ramp <= black).sum())


# ---- M2 hue preservation --------------------------------------------------------------------------------------------

def _hue_grid():
    """12 hues × 6 luminance levels × 2 saturations as linear sRGB, all inside the sRGB gamut, converted to Rec.2020."""
    rows = []
    for lum in (0.002, 0.005, 0.02, 0.05, 0.15, 0.4):
        for sat in (0.5, 0.9):
            for hue in range(0, 360, 30):
                rgb = np.array(colorsys.hsv_to_rgb(hue / 360, sat, 1.0))
                rows.append(rgb * (lum / float(rgb @ icc.SRGB.luma)))
    srgb = np.array(rows)
    return srgb, (srgb @ icc.SRGB.matrix_to(icc.Gamut("rec2020")).T)


def test_m2_hue_preserved_within_one_degree_and_control_mapping_is_not():
    srgb_in, lin2020 = _hue_grid()
    img = np.repeat(lin2020.reshape(1, -1, 3).astype(np.float32), 2, axis=0)
    out, stats = dt.render(img, _params())
    o = _decode(out[0])
    y_enc = icc.srgb_encode(o @ icc.SRGB.luma)
    flags_free = (o.max(axis=1) <= 0.9) & (y_enc >= 0.02)
    assert flags_free.sum() > 100
    dh = dt.hue_difference_degrees(dt.oklab_hue_degrees(srgb_in), dt.oklab_hue_degrees(o))
    assert dh[flags_free].max() <= 1.0
    # control: the legacy per-channel asinh mapping must fail the same criterion (the criterion is falsifiable)
    legacy = imaging.display_base(np.flipud(np.repeat(srgb_in.reshape(1, -1, 3).astype(np.float32), 2, axis=0)))
    lo = icc.srgb_decode(legacy[0].astype(np.float64))
    keep = (lo.max(axis=1) <= 0.9) & (icc.srgb_encode(lo @ icc.SRGB.luma) >= 0.02) & (lo.min(axis=1) > 0)
    dh_legacy = dt.hue_difference_degrees(dt.oklab_hue_degrees(srgb_in), dt.oklab_hue_degrees(lo))
    assert dh_legacy[keep].max() > 1.0


def test_saturation_parameter_scales_chroma_about_luminance():
    srgb_in, lin2020 = _hue_grid()
    img = np.repeat(lin2020.reshape(1, -1, 3).astype(np.float32), 2, axis=0)
    full, _ = dt.render(img, _params(saturation=1.0))
    zero, _ = dt.render(img, _params(saturation=0.0))
    z = zero[0].astype(int)
    assert (z.max(axis=1) - z.min(axis=1)).max() <= 1          # s = 0 → neutral
    # luminance unchanged by the saturation setting (ratio-preserving tone acts on Y only)
    yf = _decode(full[0]) @ icc.SRGB.luma
    yz = _decode(zero[0]) @ icc.SRGB.luma
    ok = (_decode(full[0]).max(axis=1) <= 0.9)
    assert np.abs(yf - yz)[ok].max() < 2e-3


# ---- M4 highlight desaturation ---------------------------------------------------------------------------------------

def _basis():
    recipe = raw_color.build_recipe(cf.libraw_like_metadata(), "a" * 64, cf.Z8_WHITE, cf.Z8_BLACK)
    return me.colour_basis_from_recipe(recipe, wb_mode="daylight")


def test_m4_camera_saturated_and_near_white_pixels_are_neutral():
    cam, sky, saturated_expected, nodata = cf.synthetic_camera_master()
    basis = _basis()
    linear, saturated, nd = me.convert_camera_master(cam, basis, "rec2020")
    assert np.array_equal(saturated, saturated_expected) and np.array_equal(nd, nodata)
    assert saturated.sum() > 0
    p = _params(anchor=dt.measure_anchor(linear, "rec2020", mask=sky))
    flags = np.zeros(cam.shape[:2], np.uint8)
    out, stats = dt.render(linear, p, camera_saturated=saturated, flags=flags)
    o = out.astype(int)
    spread = o.max(axis=2) - o.min(axis=2)
    # (a) camera-saturated pixels are neutral
    assert spread[saturated].max() <= 2
    # (b) near-white pixels are neutral
    y = _decode(out) @ icc.SRGB.luma
    near_white = y >= 0.95
    assert near_white.sum() > 0 and spread[near_white].max() <= 2
    # the star cores really were the G-first clipping case: unbalanced R/B below white, G at white
    core = saturated & (cam[..., 1] * 65535 >= 0.97 * cf.Z8_WHITE)
    assert core.sum() > 0
    # flags agree with the stats
    assert int((flags & dt.FLAG_CAMERA_SATURATED > 0).sum()) == stats["camera_saturated"] == int(saturated.sum())
    assert int((flags & dt.FLAG_AT_OR_BELOW_BLACK > 0).sum()) == stats["luminance_at_or_below_black"]
    assert int((flags & dt.FLAG_GAMUT_NEGATIVE > 0).sum()) == stats["gamut_negative_mapped"]
    # (c) report: gamut-edge saturated colours may exist (the red light) but are neither (a) nor (b) violations
    edge = (o.max(axis=2) == 65535) & (o.min(axis=2) < 0.97 * 65535)
    assert not np.any(edge & saturated) and not np.any(edge & near_white)


def test_m4_control_per_channel_mapping_produces_magenta_cores():
    """The legacy mapping leaves R/B above G on the same star cores (documents what the new transform removes)."""
    cam, sky, saturated, nodata = cf.synthetic_camera_master()
    linear, _, _ = me.convert_camera_master(cam, _basis(), "srgb")
    legacy = imaging.display_base(np.flipud(linear))
    l16 = np.flipud(np.rint(legacy * 65535)).astype(int)
    spread = l16.max(axis=2) - l16.min(axis=2)
    assert spread[saturated].max() > 2


def test_gamut_mapping_keeps_luminance_and_flags_pixels():
    g = icc.Gamut("rec2020")
    green = np.array([[[0.0, 0.05, 0.0]]], np.float32)            # pure Rec.2020 green: negative sRGB R
    p = _params(anchor=0.01)
    flags = np.zeros((1, 1), np.uint8)
    out, stats = dt.render(np.repeat(np.repeat(green, 2, 0), 2, 1), p, flags=np.zeros((2, 2), np.uint8))
    px = out[0, 0].astype(int)
    assert px[0] == 0 and px[1] > 0 and stats["gamut_negative_mapped"] == 4
    t = float(dt.tone_curve(np.array([float(green[0, 0] @ g.luma)]), p)[0])
    assert abs(float(_decode(out[0, 0]) @ icc.SRGB.luma) - t) < 2e-3   # luminance kept through the mapping


def test_render_rejects_bad_inputs():
    p = _params()
    with pytest.raises(ValueError):
        dt.render(np.zeros((4, 4, 3), np.uint16), p)
    with pytest.raises(ValueError):
        dt.render(np.full((4, 4, 3), np.nan, np.float32), p)
    with pytest.raises(ValueError):
        dt.render(np.zeros((4, 4, 3), np.float32), p, camera_saturated=np.zeros((3, 3), bool))
    with pytest.raises(ValueError):
        dt.render(np.zeros((4, 4, 3), np.float32), p, flags=np.zeros((4, 4), np.uint16))


def test_to_8bit_is_the_rounded_16bit_array():
    a = np.array([[[0, 257, 65535], [128, 129, 65278]]], np.uint16)
    assert np.array_equal(dt.to_8bit(a), np.array([[[0, 1, 255], [0, 1, 254]]], np.uint8))


# ---- stage 1b: shadow chroma smoothing (package doc §9) ---------------------------------------------------------------

def test_box_mean_matches_naive_edge_replicated_mean():
    rng = np.random.default_rng(0)
    a = rng.random((13, 17, 3))
    r = 2
    padded = np.pad(a, [(r, r), (r, r), (0, 0)], mode="edge")
    naive = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            naive[i, j] = padded[i:i + 2 * r + 1, j:j + 2 * r + 1].mean(axis=(0, 1))
    assert np.abs(dt.box_mean(a, r) - naive).max() < 1e-12
    assert np.array_equal(dt.box_mean(a, 0), a)


def _shadow_scene():
    cam, sky, saturated, nodata = cf.synthetic_camera_master(h=256, w=320)
    linear, sat, nd = me.convert_camera_master(cam, _basis(), "rec2020")
    sigma = dt.measure_noise_sigma(linear, "rec2020", stride=2)
    anchor = dt.measure_anchor(linear, "rec2020", mask=sky)
    off = dt.Params(gamut="rec2020", anchor=anchor, noise_sigma=sigma)
    on = dt.Params(gamut="rec2020", anchor=anchor, noise_sigma=sigma, shadow_smooth_start=10 * sigma, shadow_smooth_end=40 * sigma)
    return linear, sat, sky, off, on


def _chroma_noise(rgb16, sl):
    from starstack import color_metrics as cm
    lab = cm.lab_from_encoded_srgb8(dt.to_8bit(rgb16[sl]))
    res = lab - dt.box_mean(lab, 4)
    return res.reshape(-1, 3).std(axis=0)


def test_shadow_smoothing_reduces_chroma_noise_keeps_luminance_and_leaves_the_sky_alone():
    linear, sat, sky, off, on = _shadow_scene()
    o_off, s_off = dt.render(linear, off, camera_saturated=sat)
    o_on, s_on = dt.render(linear, on, camera_saturated=sat)
    band = (slice(185, 240), slice(20, 270))                       # ground band, away from the deep-shadow patch and no-data
    n_off, n_on = _chroma_noise(o_off, band), _chroma_noise(o_on, band)
    assert n_on[1] < n_off[1] / 3 and n_on[2] < n_off[2] / 3       # a*, b* noise cut by more than 3×
    assert abs(n_on[0] - n_off[0]) < 0.05                          # L* noise (detail) untouched
    y_off = np.rint(icc.srgb_encode(_decode(o_off) @ icc.SRGB.luma) * 65535)
    y_on = np.rint(icc.srgb_encode(_decode(o_on) @ icc.SRGB.luma) * 65535)
    assert np.abs(y_off - y_on).max() <= 2                          # M9e: luminance unchanged pixel by pixel
    assert np.array_equal(o_off[sky[:, 0]][:, :].reshape(-1, 3)[:1000], o_on[sky[:, 0]][:, :].reshape(-1, 3)[:1000])  # bright sky rows identical
    assert s_on["shadow_chroma_smoothed"] > 0 and s_off["shadow_chroma_smoothed"] == 0
    # mean colour of the band preserved (M9c on synthetic data)
    from starstack import color_metrics as cm
    la, lb = (cm.lab_from_encoded_srgb8(dt.to_8bit(o[band])) for o in (o_off, o_on))
    assert abs(la[..., 1].mean() - lb[..., 1].mean()) <= 0.5 and abs(la[..., 2].mean() - lb[..., 2].mean()) <= 0.5


def test_shadow_smoothing_is_independent_of_the_block_size_and_replays():
    linear, sat, sky, off, on = _shadow_scene()
    a, _ = dt.render(linear, on, camera_saturated=sat, rows=16)
    b, _ = dt.render(linear, on, camera_saturated=sat, rows=1000)
    assert np.array_equal(a, b)
    assert dt.Params.from_json(on.to_json()) == on and on.to_json()["version"] == dt.VERSION


def test_shadow_chroma_fade_and_parameter_validation():
    linear, sat, sky, off, on = _shadow_scene()
    fade = dt.Params(gamut="rec2020", anchor=on.anchor, shadow_chroma_end=1e6)     # everything below → neutral
    o, st = dt.render(linear, fade, camera_saturated=sat)
    spread = o.astype(int).max(axis=2) - o.astype(int).min(axis=2)
    assert spread.max() <= 1 and st["shadow_chroma_faded"] == o.shape[0] * o.shape[1]
    for kw in ({"shadow_smooth_start": 0.01, "shadow_smooth_end": 0.005}, {"shadow_smooth_end": 0.01, "shadow_smooth_radius": 0},
               {"wb_mode": "auto"}, {"wb_matrix": ((1, 0), (0, 1))}, {"shadow_smooth_radius": 2.5}):
        with pytest.raises(ValueError):
            _params(**kw)


def test_wb_matrix_is_applied_before_everything_else():
    linear, sat, sky, off, on = _shadow_scene()
    swap = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))          # swap R and G before rendering
    p = dt.Params(gamut="rec2020", anchor=off.anchor, wb_mode="as_shot", wb_matrix=swap, wb_gains_rgb=(1.0, 1.0, 1.0))
    swapped_input = linear[..., [1, 0, 2]].copy()
    a, _ = dt.render(linear, p)
    b, _ = dt.render(swapped_input, dt.Params(gamut="rec2020", anchor=off.anchor))
    assert np.array_equal(a, b)
