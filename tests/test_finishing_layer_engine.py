"""Numerical contracts for the teacher port, without GIMP or photo fixtures."""
from copy import deepcopy
import hashlib
import time

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from starstack.finishing_assets import inspect_source, inspect_mask
from starstack.finishing_layer_engine import render_layer_stage, StageCancelled, _bounded_joint_delta


@pytest.fixture
def make_scene(tmp_path):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    sequence = 0
    def make(s, b=None, d=None, *, pure=None, support=None, galaxy=None, png=False):
        nonlocal sequence
        sequence += 1
        folder = tmp_path / str(sequence)
        folder.mkdir()
        b = s.copy() if b is None else b
        d = b.copy() if d is None else d
        images = {}
        for name, pixels in (("s", s), ("b", b), ("d", d)):
            path = folder / (name + ".tif")
            tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                             extratags=[(34675, "B", len(icc), icc, False)])
            images[name] = inspect_source(path, encoding="srgb16")
        masks = {}
        for name, pixels in (("pure", pure), ("support", support), ("galaxy", galaxy)):
            if pixels is None:
                pixels = np.full(s.shape[:2], 255, np.uint8)
            path = folder / (name + (".png" if png else ".tif"))
            if png:
                Image.fromarray(pixels).save(path)
            else:
                tifffile.imwrite(path, pixels, photometric="minisblack", metadata=None)
            masks[name] = inspect_mask(path, images["s"], source_sha256=images["s"]["sha256"])
        bundle = {"schema": "aester-finishing-layer-bundle-1", "id": "test_layers",
                  "bundle_sha256": "a" * 64, "source": images["s"],
                  "background": images["b"], "diffuse": images["d"],
                  "pure_sky": {"mask_id": "pure", "asset": masks["pure"]},
                  "transient_support": {"mask_id": "support", "asset": masks["support"]}}
        params = {"bundle_id": bundle["id"], "bundle_sha256": bundle["bundle_sha256"],
                  "star_strength": 0., "milkyway_amount": 0., "milkyway_mask_id": None,
                  "meteor_core_extra": 0., "meteor_glow_amount": 0.}
        return images["s"], bundle, params, masks, folder
    return make


def render(scene, name="result", *, tile_rows=64, **changes):
    source, bundle, params, masks, folder = scene
    params = {**params, **changes}
    target = folder / (name + ".tif")
    result = render_layer_stage(source, bundle, params, masks, target, tile_rows=tile_rows)
    pixels = tifffile.imread(target)
    assert result["pixel_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    assert inspect_source(target, encoding="srgb16")["icc_sha256"] == source["icc_sha256"]
    return pixels, result


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (65, 67)])
def test_zero_is_pixel_exact_even_with_endpoints_and_odd_sizes(make_scene, shape):
    rng = np.random.default_rng(43)
    source = rng.integers(0, 65536, (*shape, 3), dtype=np.uint16)
    source.flat[0] = 0
    source.flat[-1] = 65535
    actual, report = render(make_scene(source), tile_rows=3)
    np.testing.assert_array_equal(actual, source)
    assert report["metrics"]["changed_pixels"] == 0


def test_constant_diffuse_has_no_structure_to_invent(make_scene):
    d = np.full((35, 37, 3), (12000, 18000, 24000), np.uint16)
    b = d.copy(); b[15, 15] += 2000
    s = b.copy(); s[20, 20] += 3000
    actual, _ = render(make_scene(s, b, d), milkyway_amount=3, milkyway_mask_id="galaxy")
    np.testing.assert_array_equal(actual, s)


def test_star_positive_only_and_same_signed_transient(make_scene):
    d = np.full((65, 65, 3), 20000, np.uint16)
    b = d.copy(); b[32, 32] = (24096, 18976, 21024)
    s = b.copy(); s[32, 32] = (25096, 18476, 21024)
    actual, report = render(make_scene(s, b, d), star_strength=.5)
    expected = s.copy(); expected[32, 32] -= np.array([2048, 0, 512], np.uint16)
    np.testing.assert_array_equal(actual, expected)
    assert report["metrics"]["star_T_mismatch_components"] == 0


def test_core_known_signal_and_negative_channel_not_multiplied(make_scene):
    b = np.full((65, 65, 3), 20000, np.uint16)
    s = b.copy(); s[32, 32] = (24096, 18976, 21024)
    support = np.zeros((65, 65), np.uint8); support[32, 32] = 255
    actual, _ = render(make_scene(s, b, b, support=support), meteor_core_extra=.5)
    expected = s.copy(); expected[32, 32] += np.array([2048, 0, 512], np.uint16)
    np.testing.assert_array_equal(actual, expected)


def test_glow_is_local_and_can_reach_an_original_negative_transient(make_scene):
    b = np.full((101, 101, 3), 15000, np.uint16)
    s = b.copy(); s[50, 50] += 12000; s[50, 51] -= 100
    support = np.zeros((101, 101), np.uint8); support[50, 50:52] = 255
    actual, _ = render(make_scene(s, b, b, support=support), meteor_glow_amount=3)
    assert np.all(actual[50, 51] > s[50, 51])  # Not "all negative T codes unchanged".
    assert np.all(actual[50, 54] > s[50, 54])  # Beyond the original support is allowed glow.
    assert np.all(actual[50, 50] > actual[50, 54])
    outside = np.ones(s.shape[:2], bool); outside[23:78, 23:78] = False
    np.testing.assert_array_equal(actual[outside], s[outside])


def test_highlight_and_signed_room_use_common_rgb_cap(make_scene):
    b = np.full((65, 65, 3), 30000, np.uint16)
    s = b.copy(); b[30, 30] = (60000, 29000, 0); s[30, 30] = (65534, 30000, 0)
    s[32, 32] = (65535, 32000, 33000)
    actual, report = render(make_scene(s, b, b), meteor_core_extra=2)
    # A full channel makes the shared color-preserving positive cap zero.
    np.testing.assert_array_equal(actual[30, 30], s[30, 30])
    np.testing.assert_array_equal(actual[32, 32], s[32, 32])
    assert report["metrics"]["new_zero_components"] == report["metrics"]["new_max_components"] == 0
    arr = np.array([[[1, 30000, 65535], [40000, 40000, 40000]]], np.uint16)
    request = np.array([[[-10, -20, -30], [10000, 20000, 30000]]], np.float32)
    delta = _bounded_joint_delta(arr, arr, arr, request)
    np.testing.assert_array_equal(delta[0, 0], [0, 0, 0])
    assert delta[0, 1, 0] < delta[0, 1, 1] < delta[0, 1, 2]
    assert np.all(arr.astype(np.int32) + delta <= 65535)


def _spatial_scene():
    h, w = 97, 83
    yy, xx = np.mgrid[:h, :w]
    signal = 12000 + 4000 * np.exp(-((xx - 35)**2 + (yy - 34)**2) / 130) + xx * 20
    d = np.repeat(np.rint(signal)[..., None], 3, 2).astype(np.uint16)
    pure = np.full((h, w), 255, np.uint8); pure[80:] = 0; pure[37:40, 47:50] = 0
    b = d.copy(); b[29:32, 21:24] += 3500; b[62:65, 59:62] += 1000
    b[43, 45, 1] -= 250  # Signed negative star residual.
    s = b.copy(); s[31, 30:55] += 4000; s[32, 30:55, 1] -= 500
    support = np.zeros((h, w), np.uint8); support[31:33, 30:55] = 255
    galaxy = np.full((h, w), 32768, np.uint16); galaxy[:, :8] = 0
    return s, b, d, pure, support, galaxy


def test_true_halos_and_odd_padding_are_tile_size_invariant(make_scene):
    s, b, d, pure, support, galaxy = _spatial_scene()
    scene = make_scene(s, b, d, pure=pure, support=support, galaxy=galaxy)
    params = dict(star_strength=.75, milkyway_amount=1.6, milkyway_mask_id="galaxy",
                  meteor_core_extra=.8, meteor_glow_amount=3)
    whole, info = render(scene, "whole", tile_rows=97, **params)
    assert np.any(whole != s)
    for rows in (1, 17, 64):
        tiled, _ = render(scene, "tile" + str(rows), tile_rows=rows, **params)
        np.testing.assert_array_equal(tiled, whole)
    np.testing.assert_array_equal(whole[pure == 0], s[pure == 0])
    assert info["processing"]["structure_sigma_source_proxies"] == [8, 256]


def test_structure_and_stars_preserve_transient_by_an_independent_pair(make_scene):
    s, b, d, pure, support, galaxy = _spatial_scene()
    first = make_scene(b, b, d, pure=pure, support=support, galaxy=galaxy)
    second = make_scene(s, b, d, pure=pure, support=support, galaxy=galaxy)
    args = dict(star_strength=.4, milkyway_amount=1.6, milkyway_mask_id="galaxy")
    a, _ = render(first, **args)
    c, _ = render(second, **args)
    np.testing.assert_array_equal(c.astype(np.int32) - a.astype(np.int32), s.astype(np.int32) - b.astype(np.int32))


def test_literal_png_and_uint16_weights_are_not_auto_stretched(make_scene):
    d = np.full((35, 37, 3), 10000, np.uint16); d[15:20, 15:20] = 20000
    mask = np.full(d.shape[:2], 32768, np.uint16)
    tiff = make_scene(d, galaxy=mask)
    png = make_scene(d, galaxy=mask, png=True)
    a, _ = render(tiff, milkyway_amount=1.6, milkyway_mask_id="galaxy")
    b, _ = render(png, milkyway_amount=1.6, milkyway_mask_id="galaxy")
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("change", [{"star_strength": True}, {"milkyway_amount": float("nan")},
    {"meteor_glow_amount": 4.01}, {"meteor_core_extra": -1}, {"milkyway_amount": 1},
    {"milkyway_mask_id": "missing"}, {"bundle_sha256": "b" * 64}, {"star_strength": 10**500}])
def test_invalid_parameters_fail_before_output(make_scene, change):
    scene = make_scene(np.zeros((8, 8, 3), np.uint16))
    with pytest.raises(ValueError):
        render(scene, **change)
    assert not (scene[-1] / "result.tif").exists()


def test_binary_and_spatial_contracts_are_enforced(make_scene):
    s = np.full((9, 11, 3), 10000, np.uint16)
    bad = np.full(s.shape[:2], 254, np.uint8)
    with pytest.raises(ValueError, match="binary"):
        render(make_scene(s, pure=bad))
    sky = np.zeros(s.shape[:2], np.uint8)
    b = s.copy(); b[3, 3] -= 1
    with pytest.raises(ValueError, match="Transient"):
        render(make_scene(s, b, b, pure=sky))
    with pytest.raises(ValueError, match="residual"):
        render(make_scene(s, s, b, pure=sky))


def test_failure_cleanup_cancel_deadline_and_existing_target(make_scene):
    scene = make_scene(np.full((70, 80, 3), 10000, np.uint16))
    source, bundle, params, masks, folder = scene
    target = folder / "cancelled.tif"
    before = {p: p.read_bytes() for p in folder.glob("*.tif")}
    with pytest.raises(StageCancelled):
        render_layer_stage(source, bundle, params, masks, target, cancel=lambda: target.exists())
    assert not target.exists()
    with pytest.raises(TimeoutError):
        render_layer_stage(source, bundle, params, masks, target, deadline=time.monotonic() - 1)
    assert not target.exists()
    target.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        render_layer_stage(source, bundle, params, masks, target)
    assert target.read_bytes() == b"existing user data"
    assert all(p.read_bytes() == data for p, data in before.items())


def test_actual_layer_shape_and_dtype_are_rechecked(make_scene):
    scene = make_scene(np.full((8, 9, 3), 12000, np.uint16))
    source, bundle, params, masks, folder = scene
    wrong = folder / "float.tif"
    tifffile.imwrite(wrong, np.zeros((8, 9, 3), np.float32), photometric="rgb")
    changed = deepcopy(bundle); changed["diffuse"]["path"] = str(wrong)
    with pytest.raises(ValueError, match="RGB16"):
        render_layer_stage(source, changed, params, masks, folder / "failed.tif")
    assert not (folder / "failed.tif").exists()
