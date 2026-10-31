"""Independent analytic whole-image colour/display and replay identity checks.

Synthetic colour truth uses an explicitly invertible cyclic mixing camera.
Display truth is scalar Python arithmetic and the standard sRGB encoding.
No actual RAW photograph or existing output is required by these tests.
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from starstack import whole_color as wc
from starstack.raw_color import build_recipe


@pytest.fixture
def cyclic_camera():
    # Public D65 sRGB basis, copied as constants rather than obtained from a
    # product helper. A averages two channels: A^-1 has only +1/-1 entries.
    xyz = np.array([[.4124564, .3575761, .1804375],
                    [.2126729, .7151522, .0721750],
                    [.0193339, .1191920, .9503041]])
    mixing = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
    cam_xyz = np.vstack((mixing @ np.linalg.inv(xyz), np.zeros(3)))
    fields = {"num_colors": 3, "color_desc": "RGBG", "raw_pattern": [[0, 1], [3, 2]],
              "camera_whitebalance": [4, 2, 6, 2], "rgb_xyz_matrix": cam_xyz.tolist()}
    metadata = {"available": True, "white_balance_applied": False,
                "color_correction_applied": False, "versions": {},
                "fields": {key: {"value": value} for key, value in fields.items()}}
    return build_recipe(metadata, "6" * 64, 4224, 128)


def scalar_camera(data):
    result = np.empty_like(data)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            r, g, b = [float(v) * 65535 - 128 for v in data[y, x]]
            result[y, x] = [(2 * r - g + 3 * b) / 4096,
                            (2 * r + g - 3 * b) / 4096,
                            (-2 * r + g + 3 * b) / 4096]
    return result


def scalar_display(data, recipe):
    # Derive the midtone from the anchor using the odds-ratio form, then
    # evaluate each pixel without production helpers or vectorized output.
    target = recipe["target_srgb"]
    target = target / 12.92 if target <= .04045 else ((target + .055) / 1.055) ** 2.4
    k = recipe["asinh_strength"]
    a0 = (recipe["anchor"] - recipe["black"]) / (recipe["white"] - recipe["black"])
    a0 = math.asinh(k * a0) / math.asinh(k) if k else a0
    m = a0 * (1 - target) / (a0 * (1 - target) + target * (1 - a0))
    result = np.empty(data.shape, np.uint8)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            # The balanced buffer is explicitly float32 at the core boundary.
            rgb = [float(np.float32(float(v) * gain)) for v, gain in zip(data[y, x], recipe["gains_rgb"])]
            u = [max(0, (v - recipe["black"]) / (recipe["white"] - recipe["black"])) for v in rgb]
            peak = max(u)
            a = min(peak, 1)
            a = math.asinh(k * a) / math.asinh(k) if k else a
            value = (1 - m) * a / (m + (1 - 2 * m) * a)
            for c, channel in enumerate(u):
                v = float(np.float32(channel * (value / peak))) if peak else 0.0
                encoded = 12.92 * v if v <= .0031308 else 1.055 * v ** (1 / 2.4) - .055
                result[y, x, c] = round(max(0, min(encoded, 1)) * 255)
    return result


def test_independent_cyclic_camera_black_wb_matrix_order_and_signed_values(cyclic_camera):
    rgb = np.array([[[100, 300, 800], [128, 128, 128]],
                    [[-500, 8000, 400], [65535, 65535, 65535]]], np.float32) / np.float32(65535)
    expected = scalar_camera(rgb)
    actual = wc.camera_from_cache(rgb, cyclic_camera, black_dn=128, input_state=wc.CACHE_STATE)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)
    assert actual.min() < 0 and actual.max() > 1
    assert actual.dtype == np.float32


def test_camera_is_position_invariant_through_block_edges_and_has_no_mask(cyclic_camera):
    rgb = np.full((258, 4, 3), .05, np.float32)
    rgb[0, 1] = [.01, .015, .02]
    for row in [127, 128, 255, 256, 257]:
        rgb[row, 1] = rgb[0, 1]
    rgb.flags.writeable = False
    before = rgb.tobytes()
    saved = deepcopy(cyclic_camera)
    actual = wc.camera_from_cache(rgb, cyclic_camera, black_dn=128, input_state=wc.CACHE_STATE)
    np.testing.assert_allclose(actual, scalar_camera(rgb), rtol=2e-7, atol=2e-8)
    for row in [127, 128, 255, 256, 257]:
        np.testing.assert_array_equal(actual[row, 1], actual[0, 1])
    assert rgb.tobytes() == before and cyclic_camera == saved
    assert not np.shares_memory(rgb, actual)


@pytest.mark.parametrize("changes", [
    {"black": 0., "white": 1., "anchor": .1, "target_srgb": .22, "asinh_strength": 0},
    {"black": -.1, "white": 1.4, "anchor": .15, "target_srgb": .12, "asinh_strength": 12},
    {"black": .01, "white": .8, "anchor": .05, "target_srgb": .3, "asinh_strength": 100,
     "gains_rgb": (2., .5, 1.25)},
])
def test_display_matches_independent_scalar_encoding_and_fixed_global_gains(changes):
    rgb = np.array([[[-.2, 0., .01], [.05, .08, .1], [.2, .4, .8]],
                    [[3., .5, 1.], [.025, .013, .008], [-1., -1., -1.]]], np.float32)
    recipe = wc.make_display_recipe(**changes)
    before = rgb.tobytes()
    saved = deepcopy(recipe)
    actual = wc.render_display(rgb, recipe)
    np.testing.assert_array_equal(actual, scalar_display(rgb, recipe))
    assert rgb.tobytes() == before and recipe == saved


def test_signed_box_average_is_done_before_display_gain_black_and_clipping():
    rgb = np.array([[[3, -3, 0], [-1, 1, 0], [2, 6, -2], [2, -2, 2]]] * 4, np.float32)
    # Each output cell averages 2x2 input pixels, independently of colour.
    expected_small = np.array([[[1, -1, 0], [2, 2, 0]]] * 2, np.float32)
    np.testing.assert_array_equal(wc.resize_signed(rgb, 2), expected_small)
    recipe = wc.make_display_recipe(black=-.2, white=3., anchor=.5, target_srgb=.3)
    np.testing.assert_array_equal(wc.render_display(rgb, recipe, max_side=2), scalar_display(expected_small, recipe))


def test_lossy_preview_cannot_mutate_signed_camera_derivative(cyclic_camera):
    data = np.array([[[-.1, 0., 0.3], [2, 1, 3]], [[.01, .02, .03], [0, 0, 0]]], np.float32)
    linear = wc.camera_from_cache(data, cyclic_camera, black_dn=128, input_state=wc.CACHE_STATE)
    before = linear.tobytes()
    recipe = wc.make_display_recipe(black=0, white=2, anchor=.1, target_srgb=.15)
    preview = wc.render_display(linear, recipe)
    assert preview.dtype == np.uint8
    assert linear.tobytes() == before
    assert (linear < 0).any() and (linear > 1).any()
    assert not np.shares_memory(preview, linear)


@pytest.mark.parametrize("field,value", [("version", "unknown"), ("method", "per_channel"),
                                         ("spatial_scope", "sky"), ("display_only", False),
                                         ("midtone", .9999)])
def test_rejects_display_recipe_semantic_or_derived_field_tampering(field, value):
    recipe = wc.make_display_recipe(black=0, white=1, anchor=.1, target_srgb=.2)
    recipe[field] = value
    with pytest.raises(ValueError):
        wc.render_display(np.ones((2, 2, 3), np.float32), recipe)


def test_json_roundtrip_uses_frozen_recipe_without_context_refit():
    recipe = wc.make_display_recipe(black=-.02, white=2, anchor=.06, target_srgb=.15)
    frozen = json.loads(json.dumps(recipe, allow_nan=False))
    probe = np.array([[[.05, .1, .12], [1., 2., 3.]], [[-.1, 0., .1], [.07, .08, .09]]], np.float32)
    large = np.full((128, 4, 3), 1e8, np.float32)
    large[100:102, 1:3] = probe
    np.testing.assert_array_equal(wc.render_display(probe, recipe), wc.render_display(large, frozen)[100:102, 1:3])


def test_srgb_low_segment_and_curved_segment_decode_known_points():
    np.testing.assert_allclose(wc.srgb_decode([0., .04045, .5, 1.]),
                               [0., .04045 / 12.92, .21404114048223255, 1.], rtol=1e-14, atol=0)


def load_replay_module():
    path = Path(__file__).resolve().parents[1] / "scripts/replay_whole_color.py"
    spec = importlib.util.spec_from_file_location("independent_whole_color_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_record_digest_binds_recipe_and_ignores_only_digest_field():
    script = load_replay_module()
    record = {"state": "succeeded", "version": wc.VERSION, "cases": [{"display": {"black": .1}}]}
    expected = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert script.record_digest(record) == expected
    record["record_sha256"] = "anything"
    assert script.record_digest(record) == expected
    record["cases"][0]["display"]["black"] = .2
    assert script.record_digest(record) != expected


def test_json_snapshot_hashes_the_read_buffer_even_if_path_is_replaced(tmp_path):
    script = load_replay_module()
    path = tmp_path / "state.json"
    original = b'{"state":"old"}'
    replacement = b'{"state":"new"}'
    path.write_bytes(original)

    class ReplaceAfterRead:
        def read_bytes(self):
            data = path.read_bytes()
            path.write_bytes(replacement)
            return data

    parsed, digest = script.read_json_snapshot(ReplaceAfterRead())
    assert parsed == {"state": "old"}
    assert digest == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == replacement
    assert digest != hashlib.sha256(replacement).hexdigest()


def test_replay_identity_hash_matches_the_json_bytes_actually_parsed(tmp_path, monkeypatch):
    """A writer replaces an audit after parsing, at the first path-hash call.

    This is an identity/race test, not a forged-hash adversarial security claim.
    The deliberately incomplete case stops execution before real image work.
    """
    script = load_replay_module()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    audit_path = tmp_path / "outputs/whole-color-20260920/input-audit/RAW_STATE_RESULT.json"
    b1_path = tmp_path / "outputs/background-color-20260920/real-v1/RESULTS.json"
    audit_path.parent.mkdir(parents=True)
    b1_path.parent.mkdir(parents=True)
    original_bytes = b'{"cases":[{"key":"parsed_old_case"}]}'
    audit_path.write_bytes(original_bytes)
    b1_path.write_text("[]", encoding="utf-8")
    real_sha = script.sha
    switched = False

    def racing_sha(path):
        nonlocal switched
        path = Path(path)
        if path == audit_path and not switched:
            audit_path.write_bytes(b'{"cases":[]}')
            switched = True
        # No production files exist in the intentionally isolated fake root.
        return real_sha(path) if path.exists() else "0" * 64

    monkeypatch.setattr(script, "sha", racing_sha)
    out = tmp_path / "out"
    with pytest.raises((KeyError, ValueError)):
        script.run(out)
    record = json.loads((out / "RESULT.json").read_text(encoding="utf-8"))
    assert record["sources"][str(audit_path)] == hashlib.sha256(original_bytes).hexdigest(), (
        "The recorded digest identifies a replacement audit, while execution used the previous parsed content"
    )
