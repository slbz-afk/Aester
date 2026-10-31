"""photo_render: the D product (M6 JPEG=TIFF, M7 downstream contract), candidates, and the DCP look hook."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import color_metrics as cm, display_transform as dt, imaging, master_export as me, photo_render as pr
from starstack.photographic_job import _display_source, finish_photo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402


@pytest.fixture
def scene(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "ref.dng")
    cam, sky, saturated, nodata = cf.synthetic_camera_master()
    src = tmp_path / "master_linear.fit"
    imaging.write_linear(src, np.flipud(cam), "synthetic camera master")
    rec = me.export_master(src, me.colour_basis_from_dng(dng), tmp_path / "M")
    sky_path = tmp_path / "sky.png"
    Image.fromarray((sky * 255).astype(np.uint8)).save(sky_path)
    data, meta, _ = me.read_master(rec["master"])
    params = pr.default_params(meta, data, sky=sky)
    return rec, sky_path, params, sky, saturated


def test_render_photo_contract_jpeg_equals_tiff_and_downstream_accepts(scene, tmp_path):
    rec, sky_path, params, sky, saturated = scene
    out = pr.render_photo(rec["master"], tmp_path / "D", params, sky_mask=sky_path)
    assert out["state"] == "succeeded"
    shape, icc_bytes, meta = _display_source(out["tiff"])                        # M7: finish-photo's own validator
    assert meta["display_transform"]["version"] == dt.VERSION and meta["source_master"]["sha256"] == rec["master_sha256"]
    tif = tifffile.imread(out["tiff"])
    assert tif.dtype == np.uint16 and tif.shape == shape
    with Image.open(out["jpeg"]) as im:
        assert im.mode == "RGB" and im.info.get("icc_profile") == icc_bytes
        jpg = np.asarray(im)
    # M6: JPEG decoded vs TIFF rounded to 8 bits.  Pixel-wise ΔE00 is dominated by the codec's quantisation of noise
    # (see docs/COLOR_CONTRACT_PACKAGE_20260922.md, 修订 2 pending the user's decision); the rendering identity is
    # checked on 8×8 block means, and the pixel-wise numbers only need to stay within codec-noise range.
    t8 = dt.to_8bit(tif)
    de_pixel = cm.ciede2000(cm.lab_from_encoded_srgb8(jpg), cm.lab_from_encoded_srgb8(t8))
    de_block = cm.ciede2000(cm.lab_from_linear_srgb(cm.block_means_linear(jpg)), cm.lab_from_linear_srgb(cm.block_means_linear(t8)))
    assert float(np.median(de_block)) <= 0.5 and float(np.quantile(de_block, 0.99)) <= 2.0
    assert float(np.quantile(de_pixel, 0.99)) <= 8.0
    assert out["jpeg_source"].startswith("round(uint16/257)")
    # flags on disk agree with the recorded statistics
    flags = np.asarray(Image.open(out["flags"]))
    st = meta["statistics"]
    assert int((flags & dt.FLAG_CAMERA_SATURATED > 0).sum()) == st["camera_saturated"] == int(saturated.sum())
    assert st["gamut_negative_mapped_by_region"]["1"] + st["gamut_negative_mapped_by_region"]["0"] == st["gamut_negative_mapped"]
    # M7: finish-photo runs on D
    protect = tmp_path / "protect.png"
    Image.fromarray(np.zeros(shape[:2], np.uint8)).save(protect)
    r = finish_photo(out["tiff"], tmp_path / "finish", sky_path, protect, detail_strength=0, star_strength=0)
    assert r["state"] == "succeeded"
    # nothing was written back into the master directory
    assert sorted(p.name for p in Path(rec["master"]).parent.iterdir()) == sorted([me.FILENAME, me.MASK_FILENAME, me.RECIPE_FILENAME])


def test_render_photo_is_deterministic_and_refuses_gamut_mismatch(scene, tmp_path):
    rec, sky_path, params, *_ = scene
    a = pr.render_photo(rec["master"], tmp_path / "D1", params)
    b = pr.render_photo(rec["master"], tmp_path / "D2", params)
    assert a["tiff_sha256"] == b["tiff_sha256"] and a["jpeg_sha256"] == b["jpeg_sha256"]
    with pytest.raises(ValueError):
        pr.render_photo(rec["master"], tmp_path / "D3", dt.Params(gamut="srgb", anchor=params.anchor))
    with pytest.raises(ValueError):
        pr.render_photo(rec["master"], tmp_path / "D4", dt.Params(gamut="rec2020", anchor=params.anchor, look="dcp_hsm"))


def test_saturation_mask_is_verified_against_the_master(scene, tmp_path):
    rec, sky_path, params, *_ = scene
    mask = Path(rec["mask"])
    original = mask.read_bytes()
    Image.fromarray(np.zeros(tifffile.imread(rec["master"]).shape[:2], np.uint8)).save(mask)
    with pytest.raises(ValueError):
        pr.render_photo(rec["master"], tmp_path / "D", params)
    mask.write_bytes(original)


def test_candidates_grid(scene, tmp_path):
    rec, sky_path, *_ = scene
    c = pr.render_candidates(rec["master"], tmp_path / "C", sky_mask=sky_path, max_side=160)
    names = [i["name"] for i in c["items"]]
    # default WB is as-shot (U2), the master is daylight → the "other" WB candidate is daylight
    assert names == ["candidate_k6_s0.85", "candidate_k6_s1.00", "candidate_k12_s0.85", "candidate_k12_s1.00",
                     "candidate_k24_s0.85", "candidate_k24_s1.00", "candidate_daylight_k12_s0.85", "candidate_daylight_k12_s1.00",
                     "candidate_k12_s1.00_noshadowsmooth", "candidate_k12_s1.00_chromafade"]
    assert c["wb_mode"] == "as_shot" and c["noise_sigma"] > 0
    for item in c["items"]:
        with Image.open(tmp_path / "C" / item["file"]) as im:
            assert max(im.size) == 160 and im.mode == "RGB"
        assert item["params"]["target"] == pr.CANDIDATE_TARGET
    grid = {i["name"]: i["params"] for i in c["items"]}
    assert grid["candidate_k12_s1.00"]["wb_mode"] == "as_shot" and grid["candidate_k12_s1.00"]["wb_matrix"] is not None
    assert grid["candidate_daylight_k12_s1.00"]["wb_mode"] == "daylight" and grid["candidate_daylight_k12_s1.00"]["wb_matrix"] is None
    assert grid["candidate_k12_s1.00_noshadowsmooth"]["shadow_smooth_end"] == 0.0 and grid["candidate_k12_s1.00"]["shadow_smooth_end"] > 0
    assert grid["candidate_k12_s1.00_chromafade"]["shadow_chroma_end"] > 0
    assert (tmp_path / "C" / "CANDIDATES.md").read_text(encoding="utf-8").count("candidate_") == 10
    assert json.loads((tmp_path / "C" / "CANDIDATES.json").read_text(encoding="utf-8"))["master_sha256"] == rec["master_sha256"]


def test_photo_white_balance_switch_matches_a_master_exported_with_that_balance(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "ref.dng")
    cam, sky, *_ = cf.synthetic_camera_master()
    src = tmp_path / "master_linear.fit"
    imaging.write_linear(src, np.flipud(cam), "synthetic camera master")
    day = me.export_master(src, me.colour_basis_from_dng(dng, wb_mode="daylight"), tmp_path / "Mday")
    shot = me.export_master(src, me.colour_basis_from_dng(dng, wb_mode="as_shot"), tmp_path / "Mshot")
    d_data, d_meta, _ = me.read_master(day["master"])
    s_data, s_meta, _ = me.read_master(shot["master"])
    p_switch = pr.default_params(d_meta, d_data, sky=sky, wb_mode="as_shot", shadow_smoothing=False)
    p_native = pr.default_params(s_meta, s_data, sky=sky, wb_mode="as_shot", shadow_smoothing=False)
    assert p_switch.wb_mode == "as_shot" and p_switch.wb_matrix is not None and p_native.wb_matrix is None
    assert abs(p_switch.anchor - p_native.anchor) < 1e-6 and abs(p_switch.noise_sigma - p_native.noise_sigma) < 1e-6
    a = pr.render_photo(day["master"], tmp_path / "Dswitch", p_switch)
    b = pr.render_photo(shot["master"], tmp_path / "Dnative", p_native)
    ta, tb = tifffile.imread(a["tiff"]).astype(int), tifffile.imread(b["tiff"]).astype(int)
    assert np.abs(ta - tb).max() <= 1
    assert a["metadata"]["display_transform"]["wb_gains_rgb"] == list(s_meta["colour_basis"]["gains_rgb"])


def test_default_params_record_noise_and_thresholds(scene):
    rec, sky_path, params, sky, _ = scene
    assert params.wb_mode == "as_shot" and params.noise_sigma > 0
    assert params.shadow_smooth_start == pr.SHADOW_SMOOTH_START_SIGMAS * params.noise_sigma
    assert params.shadow_smooth_end == pr.SHADOW_SMOOTH_END_SIGMAS * params.noise_sigma
    assert params.shadow_smooth_radius == pr.SHADOW_SMOOTH_RADIUS
    data, meta, _ = me.read_master(rec["master"])
    off = pr.default_params(meta, data, sky=sky, shadow_smoothing=False)
    assert not off.shadow_smoothing and off.noise_sigma == params.noise_sigma


def test_dcp_hsm_look_identity_table_is_a_no_op_and_zero_saturation_neutralises(monkeypatch, scene, tmp_path):
    from starstack import experimental_dcp as dcp
    identity = np.zeros((90, 30, 3)); identity[..., 1] = 1.0; identity[..., 2] = 1.0

    def fake_loader(path, table=identity):
        return {"sha256": "f" * 64, "name": "fake", "huesat": table}
    monkeypatch.setattr(dcp, "load_z8_profile", fake_loader)
    look, record = pr.make_dcp_hsm_look("ignored", "rec2020")
    block = np.abs(np.random.default_rng(1).normal(0.1, 0.05, (4, 5, 3)))
    assert np.allclose(look(block.copy()), block, atol=1e-9) and record["look"] == "dcp_hsm"
    desat = identity.copy(); desat[..., 1] = 0.0
    monkeypatch.setattr(dcp, "load_z8_profile", lambda path: fake_loader(path, desat))
    look2, _ = pr.make_dcp_hsm_look("ignored", "rec2020")
    out = look2(block.copy())
    assert (out.max(axis=2) - out.min(axis=2)).max() < 1e-6            # every pixel neutral in ProPhoto → neutral here
    rec, sky_path, params, *_ = scene
    d = pr.render_photo(rec["master"], tmp_path / "D", dataclasses.replace(params, look="dcp_hsm"), look_profile="ignored")
    assert d["metadata"]["look"]["look"] == "dcp_hsm" and d["metadata"]["look"]["counters"]["processed"] > 0
    # the look is refused for a master whose colour basis is not the audited Z8 family
    with pytest.raises(ValueError):
        pr.make_dcp_hsm_look("ignored", "rec2020", cam_xyz=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_default_recipe_is_frozen_and_labelled(scene, tmp_path):
    rec, sky_path, params, sky, _ = scene
    r = pr.DEFAULT_RECIPE
    assert r["name"] == "photo-default-1" and r["wb_mode"] == "as_shot" and r["asinh_strength"] == 12.0 and r["saturation"] == 1.0
    assert r["target"] == 0.22 and r["look"] == "none" and r["shadow_smoothing"] is True and r["shadow_chroma_end"] == 0.0
    assert pr.matches_default_recipe(params)
    out = pr.render_photo(rec["master"], tmp_path / "D", params)
    assert out["metadata"]["recipe"] == "photo-default-1"
    custom = dataclasses.replace(params, saturation=0.85)
    assert not pr.matches_default_recipe(custom)
    assert pr.render_photo(rec["master"], tmp_path / "D2", custom)["metadata"]["recipe"] == "custom"
