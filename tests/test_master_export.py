"""master_export: colour basis from DNG tags / recipes and the M product contract (M1 synthetic part)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import icc_profiles as icc, imaging, master_export as me, raw_color

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402


@pytest.fixture
def dng(tmp_path):
    return cf.write_reference_dng(tmp_path / "ref.dng")


@pytest.fixture
def master(tmp_path):
    cam, sky, saturated, nodata = cf.synthetic_camera_master()
    path = tmp_path / "master_linear.fit"
    imaging.write_linear(path, np.flipud(cam), "synthetic camera master")   # FITS stores bottom-up
    return path, cam, sky, saturated, nodata


def test_basis_from_dng_uses_d65_matrix_levels_and_both_white_balances(dng):
    b = me.colour_basis_from_dng(dng)
    assert b["black_DN"] == [cf.Z8_BLACK] * 3 and b["white_DN"] == cf.Z8_WHITE and b["scale_DN"] == cf.Z8_WHITE - cf.Z8_BLACK
    assert b["source"]["color_matrix_tag"] == "ColorMatrix2" and b["source"]["calibration_illuminant"] == 21
    assert np.allclose(b["cam_xyz"][:3], np.array(cf.Z8_COLOR_MATRIX2).reshape(3, 3), atol=1e-9)
    n = cf.Z8_AS_SHOT_NEUTRAL
    assert np.allclose(b["as_shot_gains_rgb"], [n[1] / n[0], 1.0, n[1] / n[2]], rtol=1e-9)
    # daylight gains: camera response to D65 white = row sums of cam_xyz @ XYZ_FROM_SRGB, gains = 1/row sums (G = 1)
    rows = (np.array(cf.Z8_COLOR_MATRIX2).reshape(3, 3) @ raw_color.XYZ_FROM_LINEAR_SRGB).sum(axis=1)
    assert np.allclose(b["daylight_gains_rgb"], (1 / rows) / (1 / rows[1]), rtol=1e-9)
    assert b["wb_mode"] == "daylight" and b["gains_rgb"] == b["daylight_gains_rgb"]
    assert me.colour_basis_from_dng(dng, wb_mode="as_shot")["gains_rgb"] == b["as_shot_gains_rgb"]
    assert b["source"]["baseline_exposure_ev"] == pytest.approx(0.2) and b["source"]["baseline_exposure_applied"] is False
    assert b["source"]["camera"]["unique_camera_model"] == "Nikon Z 8"
    raw_color.validate_recipe(b["raw_color_recipe"])
    assert np.allclose(b["camera_to_linear_srgb"], b["raw_color_recipe"]["matrix"]["values"])


def test_basis_from_dng_falls_back_and_refuses_unsupported_calibration(tmp_path):
    only1 = cf.write_reference_dng(tmp_path / "m1.dng", include_matrix2=False)
    b = me.colour_basis_from_dng(only1)
    assert b["source"]["color_matrix_tag"] == "ColorMatrix1" and b["source"]["calibration_illuminant"] == 17
    with pytest.raises(ValueError):
        me.colour_basis_from_dng(cf.write_reference_dng(tmp_path / "ab.dng", analog_balance=(1.1, 1, 1)))
    with pytest.raises(ValueError):
        me.colour_basis_from_dng(cf.write_reference_dng(tmp_path / "cc.dng", calibration_identity=False))
    with pytest.raises(ValueError):
        me.colour_basis_from_dng(only1, wb_mode="auto")


def test_basis_from_recipe_matches_dng_route(dng):
    b = me.colour_basis_from_dng(dng)
    r = me.colour_basis_from_recipe(b["raw_color_recipe"], wb_mode="as_shot")
    assert r["gains_rgb"] == b["as_shot_gains_rgb"] and r["camera_to_linear_srgb"] == b["camera_to_linear_srgb"]
    assert r["recipe_sha256"] == b["recipe_sha256"]


def test_export_master_contract_and_bit_exact_replay(dng, master, tmp_path):
    path, cam, sky, saturated, nodata = master
    basis = me.colour_basis_from_dng(dng)
    rec = me.export_master(path, basis, tmp_path / "M")
    assert rec["state"] == "succeeded"
    with tifffile.TiffFile(rec["master"]) as tf:
        page = tf.pages[0]
        assert page.dtype == np.float32 and page.shape == cam.shape and int(page.photometric) == 2
        profile = page.tags[34675].value
        meta = json.loads(page.description)
    assert icc.identify_linear_profile(profile) == "rec2020"
    for key in ("schema", "scene_linear", "display_encoding_applied", "roworder", "color_space", "white_point", "units",
                "clipping_applied", "nodata", "camera_saturation", "colour_basis", "source", "stack_note", "icc"):
        assert key in meta
    assert meta["scene_linear"] is True and meta["display_encoding_applied"] is False and meta["roworder"] == "TOP-DOWN"
    assert meta["color_space"] == "linear_rec2020_D65" and meta["clipping_applied"] is False
    assert meta["nodata"]["count"] == int(nodata.sum()) and meta["camera_saturation"]["count"] == int(saturated.sum())
    assert meta["colour_basis"]["wb_mode"] == "daylight" and meta["colour_basis"]["source_recipe_sha256"] == basis["recipe_sha256"]
    assert meta["source"]["sha256"] == me.sha256_file(path)
    data, meta2, _ = me.read_master(rec["master"])
    assert np.isfinite(np.asarray(data)).all() and meta2 == meta
    # no clipping: values above 1 survive, negatives (noise below black) survive
    assert float(np.asarray(data).max()) > 1.0 and meta["statistics"]["negative_channels"] > 0
    # no-data pixels stay exactly zero, and no other pixel is exactly zero in all channels
    zero = (np.asarray(data) == 0).all(axis=2)
    assert np.array_equal(zero, nodata)
    # saturation mask on disk matches
    mask = np.asarray(Image.open(rec["mask"])) > 0
    assert np.array_equal(mask, saturated) and me.sha256_file(rec["mask"]) == meta["camera_saturation"]["mask_sha256"]
    # TOP-DOWN: the ground band (bottom of the scene) is at the bottom of the master
    assert np.asarray(data)[-1, 100, 1] < np.asarray(data)[10, 100, 1]
    # replay: same source, same basis → identical bytes
    rec2 = me.export_master(path, basis, tmp_path / "M2")
    assert rec2["master_sha256"] == rec["master_sha256"]
    # the recipe on disk records what is needed to replay
    recipe = json.loads((tmp_path / "M" / me.RECIPE_FILENAME).read_text(encoding="utf-8"))
    assert recipe["master_sha256"] == rec["master_sha256"] and recipe["basis"]["gains_rgb"] == basis["gains_rgb"]
    with pytest.raises(FileExistsError):
        me.export_master(path, basis, tmp_path / "M")


def test_master_units_put_sensor_white_at_one_and_neutral_stays_neutral(dng, tmp_path):
    basis = me.colour_basis_from_dng(dng)
    n = np.array(cf.Z8_AS_SHOT_NEUTRAL)
    # a neutral (as-shot) element at sensor white on G, and at half signal
    cam = np.zeros((2, 2, 3), np.float32)
    cam[0, 0] = (cf.Z8_BLACK + n * (cf.Z8_WHITE - cf.Z8_BLACK)) / 65535
    cam[0, 1] = (cf.Z8_BLACK + n * 0.5 * (cf.Z8_WHITE - cf.Z8_BLACK)) / 65535
    cam[1, :] = cf.Z8_BLACK / 65535
    as_shot = me.colour_basis_from_dng(dng, wb_mode="as_shot")
    lin, sat, nd = me.convert_camera_master(cam, as_shot, "srgb")
    assert np.allclose(lin[0, 0], [1, 1, 1], atol=1e-5) and np.allclose(lin[0, 1], [0.5] * 3, atol=1e-5)
    assert np.allclose(lin[1], 0, atol=1e-6) and not nd.any() and sat[0, 0] and not sat[0, 1]
    # daylight gains differ from as-shot, so the same element is not neutral under daylight WB (recorded, not hidden)
    lin_d, _, _ = me.convert_camera_master(cam, basis, "srgb")
    assert not np.allclose(lin_d[0, 0], [1, 1, 1], atol=1e-3)


def test_switch_white_balance_matrix_matches_direct_export(dng, master, tmp_path):
    path, cam, *_ = master
    day = me.colour_basis_from_dng(dng, wb_mode="daylight")
    shot = me.colour_basis_from_dng(dng, wb_mode="as_shot")
    lin_day, _, _ = me.convert_camera_master(cam, day, "rec2020")
    lin_shot, _, _ = me.convert_camera_master(cam, shot, "rec2020")
    m = me.switch_white_balance_matrix(day, shot["gains_rgb"], "rec2020")
    assert np.allclose(lin_day.astype(np.float64) @ m.T, lin_shot, atol=1e-5)


def test_read_master_rejects_foreign_tiffs(tmp_path, dng, master):
    path, cam, *_ = master
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, cam, photometric="rgb")
    with pytest.raises(ValueError):
        me.read_master(plain)
    rec = me.export_master(path, me.colour_basis_from_dng(dng), tmp_path / "M")
    data = tifffile.imread(rec["master"])
    with tifffile.TiffFile(rec["master"]) as tf:
        meta = json.loads(tf.pages[0].description)
    meta["gamut"] = "srgb"                                     # metadata no longer matches the embedded profile
    forged = tmp_path / "forged.tif"
    tifffile.imwrite(forged, data, photometric="rgb", iccprofile=icc.write_linear_profile("rec2020"), metadata=meta)
    with pytest.raises(ValueError):
        me.read_master(forged)
