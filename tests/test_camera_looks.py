"""α7R V default look (docs/SONY_A7RV_LOOK_PREREG_20260926.md L1–L3, L5 synthetic): the borrowed α7 IV HueSatMap at
half strength is the default look of α7R V masters only, on every path that renders default parameters (job D,
standalone render, CLI/MCP defaults, c1, reference preview), and nothing changes for other bodies."""
import dataclasses
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starstack import camera_looks as cl, color_export as ce, color_products as cp, devices, devices_registry as reg
from starstack import display_transform as dt, imaging, master_export as me, photo_render as pr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import c1_accept_color_contract as c1  # noqa: E402

LOOK_ID = "sony-ilce7m4-rt-cc0-d50"


# ---- L1 data -----------------------------------------------------------------------------------------------------
def test_table_is_the_pinned_source_table():
    info = cl.LOOKS[LOOK_ID]
    assert info["source_sha256"] == "8b209fdcf76b996cd4a663a9a8d784895c7daf4a78415adc37141e8205ef7c4d"
    assert info["table_sha256"] == "4acfd9674101dcb948b760787750082f1df6154ab8e39c321f2acd203a823de2"
    assert info["profile_copyright"] == "RawTherapee CC0" and info["tag"] == 50939 and info["dims"] == [90, 30, 1]
    t = cl.table(LOOK_ID)
    assert t.shape == (90, 30, 3) and t.dtype == np.float64 and not t.flags.writeable
    # float32 source values, widened exactly
    assert np.array_equal(t, t.astype(np.float32).astype(np.float64))
    assert -14 < t[..., 0].min() < t[..., 0].max() < 7 and 0.94 < t[..., 1].min() and t[..., 2].max() < 1.28


def test_tampered_table_is_refused(monkeypatch):
    data = dict(cl._DATA)
    raw = bytearray(__import__("base64").b64decode(data[LOOK_ID]))
    raw[0] ^= 1
    data[LOOK_ID] = __import__("base64").b64encode(bytes(raw)).decode()
    monkeypatch.setattr(cl, "_DATA", data)
    with pytest.raises(ValueError):
        cl.table(LOOK_ID)
    with pytest.raises(ValueError):
        cl.table("no-such-look")


def test_a7rv_look_is_recorded_and_other_early_bodies_have_none():
    # since the look batch of 2026-09-26 many bodies carry a look (tests/test_camera_look_batch.py); these must not
    entry = next(e for e in reg.REGISTRY if "ILCE-7RM5" in e["model_ids"])
    assert entry["colour_look"]["id"] == LOOK_ID and entry["colour_look"]["strength"] == 0.5
    assert entry["colour_look"]["borrowed"] is True and entry["tier"] == "B" and "接受" in entry["colour_look"]["status"]
    for model in ("NIKON Z 8", "ZWO Seestar S30 Pro", "Canon EOS 5D Mark III", "DC-S5", "ILCE-7CR", "ILCE-7RM3"):
        assert "colour_look" not in next(e for e in reg.REGISTRY if model in e["model_ids"]), model
    assert devices.registry_problems() == []


@pytest.mark.parametrize("look", [{"id": "nope", "strength": 0.5, "decided": "x"}, {"id": LOOK_ID, "strength": 1.5, "decided": "x"},
                                  {"id": LOOK_ID, "strength": 0.5}, "not-a-dict"])
def test_registry_self_check_reports_bad_colour_looks(monkeypatch, look):
    entry = dict(next(e for e in reg.REGISTRY if "ILCE-7RM5" in e["model_ids"]))
    entry["colour_look"] = look
    monkeypatch.setattr(reg, "REGISTRY", [entry])
    assert devices.registry_problems()


# ---- L2 mathematics ----------------------------------------------------------------------------------------------
def _block(seed=5):
    rng = np.random.default_rng(seed)
    block = np.abs(rng.normal(0.1, 0.08, (16, 24, 3)))
    block[0, :4] = [[-0.02, 0.1, 0.1], [0.3, -0.01, 0.2], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]     # negative, black, grey
    return block


def test_full_strength_equals_the_dcp_hsm_mathematics_bit_for_bit():
    look_full, _ = cl.make_look(LOOK_ID, 1.0, "rec2020")
    reference, _ = pr.hsm_look(cl.table(LOOK_ID), "rec2020")
    assert np.array_equal(look_full(_block()), reference(_block()))
    assert np.array_equal(cl.scaled_table(LOOK_ID, 1.0), cl.table(LOOK_ID))


@pytest.mark.parametrize("s", [0.25, 0.5, 0.75])
def test_strength_scales_hue_shift_and_multipliers(s):
    t, st = cl.table(LOOK_ID), cl.scaled_table(LOOK_ID, s)
    assert np.array_equal(st[..., 0], t[..., 0] * s)
    assert np.array_equal(st[..., 1:], 1.0 + s * (t[..., 1:] - 1.0))
    # through the lookup: at grid nodes the interpolated modifiers are the scaled entries
    from starstack.experimental_huesat import lookup_huesat
    hues = np.array([0.0, 4.0 * 7, 4.0 * 45]); sats = np.array([0.0, 10 / 29, 20 / 29])
    got = lookup_huesat(st, hues, sats)
    want = st[[0, 7, 45], [0, 10, 20]]
    assert np.allclose(got, want, atol=1e-12)


def test_negative_prophoto_pixels_pass_unchanged():
    look, record = cl.make_look(LOOK_ID, 0.5, "rec2020")
    block = _block()
    out = look(block.copy())
    from starstack import experimental_dcp as dcp, icc_profiles as icc
    g = icc.Gamut("rec2020")
    pro = block @ (dcp.XYZ_TO_PROPHOTO @ np.linalg.inv(dcp.XYZ_TO_SRGB) @ g.matrix_to(icc.SRGB)).T
    negative = ~(pro >= 0).all(axis=-1)
    assert negative.any() and np.array_equal(out[negative], block[negative])
    assert record["counters"]["bypassed_negative"] == int(negative.sum())
    assert record["look"] == "camera_hsm" and record["strength"] == 0.5 and record["table_sha256"] == cl.LOOKS[LOOK_ID]["table_sha256"]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, float("nan"), True, "0.5"])
def test_strength_outside_the_range_is_refused(bad):
    with pytest.raises(ValueError):
        cl.scaled_table(LOOK_ID, bad)


# ---- Params --------------------------------------------------------------------------------------------------------
def test_params_validation_and_json_round_trip():
    p = dt.Params(look="camera_hsm", look_id=LOOK_ID, look_strength=0.5).validate()
    j = p.to_json()
    assert j["look_id"] == LOOK_ID and j["look_strength"] == 0.5
    assert dt.Params.from_json(j) == p
    plain = dt.Params().to_json()
    assert "look_id" not in plain and "look_strength" not in plain            # other looks keep their record form
    assert dt.Params.from_json(plain) == dt.Params()
    for bad in ({"look": "camera_hsm"}, {"look": "camera_hsm", "look_id": "nope"},
                {"look": "camera_hsm", "look_id": LOOK_ID, "look_strength": 0.0},
                {"look": "none", "look_id": LOOK_ID}, {"look": "none", "look_strength": 0.5}):
        with pytest.raises(ValueError):
            dt.Params(**bad).validate()


def test_render_resolves_the_camera_look_from_the_parameters():
    data = np.abs(np.random.default_rng(2).normal(0.05, 0.03, (300, 40, 3))).astype(np.float32)
    p = dt.Params(anchor=0.05, look="camera_hsm", look_id=LOOK_ID, look_strength=0.5)
    explicit, _ = cl.make_look(LOOK_ID, 0.5, "rec2020")
    a, _ = dt.render(data, p, look=explicit)
    b, _ = dt.render(data, p, workers=4)                    # no callable given: same look, still serial
    c, _ = dt.render(data, dataclasses.replace(p, look="none", look_id=None, look_strength=1.0))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ---- L3 default selection ------------------------------------------------------------------------------------------
def _meta(camera):
    return {"gamut": "rec2020", "colour_basis": {"reference": {"kind": "libraw", "camera": camera}}}


@pytest.mark.parametrize("camera,expected", [
    ({"make": "SONY", "model": "ILCE-7RM5", "software": "ILCE-7RM5 v2.02", "source": "container_tags"}, LOOK_ID),
    ({"make": "SONY", "model": "ILCE-7RM5", "software": "ILCE-7RM5 v1.00", "source": "container_tags"}, LOOK_ID),
    ({"unique_camera_model": "Nikon Z 8", "make": "NIKON CORPORATION", "model": "NIKON Z 8"}, None),
    ({"make": "Canon", "model": "Canon EOS 5D Mark III"}, None),
    ({"make": "SONY", "model": "ILCE-7M4"}, (LOOK_ID, 0.75)),         # α7 IV: own table since the look batch, 0.75 (user 2026-09-27)
    ({"make": "SONY", "model": "ILCE-7CR"}, None),
    ({"make": "FUJIFILM", "model": "X-T5"}, None),
    (None, None),
])
def test_default_look_follows_the_registered_body(camera, expected):
    fields = pr.default_look_fields(_meta(camera))
    if expected is None:
        assert fields == {"look": "none", "look_id": None, "look_strength": 1.0}
    else:
        look_id, strength = expected if isinstance(expected, tuple) else (expected, 0.5)
        assert fields == {"look": "camera_hsm", "look_id": look_id, "look_strength": strength}
    assert pr.default_look_fields({"gamut": "rec2020", "colour_basis": {}}) == {"look": "none", "look_id": None, "look_strength": 1.0}


def _sony_arw(path, model="ILCE-7RM5"):
    tags = [(271, 2, None, "SONY", True), (272, 2, None, model, True), (0x7000, 3, 1, 4, True)]
    tifffile.imwrite(path, np.zeros((6, 10), np.uint16), photometric="cfa", software=f"{model} v2.02", extratags=tags)
    return path


def _libraw_basis(reference):
    meta = cf.libraw_like_metadata()
    meta["fields"]["black_level_per_channel"] = {"value": [cf.Z8_BLACK] * 4}
    meta["fields"]["white_level"] = {"value": cf.Z8_WHITE}
    return me.colour_basis_for_reference(reference, metadata=meta)


def test_libraw_route_records_the_body_from_the_container(tmp_path):
    basis = _libraw_basis(_sony_arw(tmp_path / "DSC00001.ARW"))
    assert basis["source"]["camera"] == {"make": "SONY", "model": "ILCE-7RM5", "software": "ILCE-7RM5 v2.02", "source": "container_tags"}
    fake = tmp_path / "ref.nef"
    fake.write_bytes(b"not a TIFF container")
    assert _libraw_basis(fake)["source"]["camera"] is None


@pytest.fixture
def sony_job(tmp_path):
    """A synthetic composite whose colour basis came from an α7R V ARW (LibRaw route), exported the way the job does."""
    reference = _sony_arw(tmp_path / "DSC00001.ARW")
    basis = _libraw_basis(reference)
    cam, sky, _, _ = cf.synthetic_camera_master()
    export_dir = tmp_path / "products" / "p"
    export_dir.mkdir(parents=True)
    bottom_up = np.flipud(cam).copy()
    fits_path = export_dir / "master_linear.fit"
    imaging.write_linear(fits_path, bottom_up, "synthetic composite")
    artifacts, record = ce.export_job_products(bottom_up, export_dir, lambda: None, basis=basis, source_fits=fits_path)
    return basis, cam, bottom_up, export_dir, fits_path, record, reference


def test_job_d_standalone_render_and_cli_default_agree(sony_job, tmp_path):
    basis, cam, bottom_up, export_dir, fits_path, record, _ = sony_job
    photo_meta = json.loads(tifffile.TiffFile(export_dir / ce.D_TIFF_NAME).pages[0].description)
    assert record["photo"]["recipe"] == "photo-default-1" and photo_meta["recipe"] == "photo-default-1"
    assert photo_meta["look"]["look"] == "camera_hsm" and photo_meta["look"]["look_id"] == LOOK_ID
    assert photo_meta["display_transform"]["look_strength"] == 0.5
    # standalone default render (the c1 S2b path) and the CLI/MCP default (look="auto") give the same pixels and params
    standalone = me.export_master(fits_path, basis, tmp_path / "M")
    assert standalone["master_sha256"] == record["master"]["sha256"]
    via_cli = cp.render_photo(standalone["master"], tmp_path / "cli")
    assert via_cli["metadata"]["display_transform"] == photo_meta["display_transform"]
    assert np.array_equal(tifffile.imread(via_cli["tiff"]), tifffile.imread(export_dir / ce.D_TIFF_NAME))
    # explicit none is honoured and differs
    plain = cp.render_photo(standalone["master"], tmp_path / "none", look="none")
    assert plain["metadata"]["look"] == {"look": "none"} and plain["metadata"]["recipe"] == "custom"
    assert not np.array_equal(tifffile.imread(plain["tiff"]), tifffile.imread(via_cli["tiff"]))
    # a non-default saturation is labelled custom even with the camera look
    data, meta, _ = me.read_master(standalone["master"])
    custom = pr.default_params(meta, data, saturation=0.85)
    assert custom.look == "camera_hsm" and not pr.matches_default_recipe(custom, meta)


def test_candidates_carry_the_camera_look(sony_job, tmp_path):
    _, _, _, export_dir, *_ = sony_job
    rec = pr.render_candidates(export_dir / me.FILENAME, tmp_path / "cand", max_side=120)
    assert rec["items"] and all(i["params"]["look"] == "camera_hsm" and i["params"]["look_id"] == LOOK_ID for i in rec["items"])


def test_z8_dng_route_is_unchanged(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "frame00001.dng")
    basis = ce.job_colour_basis(dng)
    cam, *_ = cf.synthetic_camera_master()
    export_dir = tmp_path / "p"
    export_dir.mkdir()
    fits_path = export_dir / "master_linear.fit"
    imaging.write_linear(fits_path, np.flipud(cam).copy(), "synthetic composite")
    _, record = ce.export_job_products(np.flipud(cam).copy(), export_dir, lambda: None, basis=basis, source_fits=fits_path)
    photo_meta = json.loads(tifffile.TiffFile(export_dir / ce.D_TIFF_NAME).pages[0].description)
    assert photo_meta["look"] == {"look": "none"} and "look_id" not in photo_meta["display_transform"]
    assert photo_meta["recipe"] == "photo-default-1"


def test_reference_preview_uses_the_same_default_look(sony_job, tmp_path, monkeypatch):
    basis, cam, bottom_up, *_ , reference = sony_job
    from starstack import references
    monkeypatch.setattr(ce, "job_colour_basis", lambda source, **k: basis)
    directory = tmp_path / "preview"
    directory.mkdir()
    _, cache = references._prepare_color_preview(bottom_up, reference, directory, None)
    base = cache["base_params"]
    assert base["look"] == "camera_hsm" and base["look_id"] == LOOK_ID and base["look_strength"] == 0.5


def test_entry_points_default_to_auto():
    assert inspect.signature(cp.render_photo).parameters["look"].default == "auto"
    from starstack.mcp_server import WorkspaceTools
    assert inspect.signature(WorkspaceTools.render_photo).parameters["look"].default == "auto"
    from starstack import cli
    assert 'render.add_argument("--look", choices=("auto", "none", "dcp_hsm"), default="auto"' in inspect.getsource(cli.main)
    assert c1.build_parser().parse_args(["--job", "j", "--output", "o"]).look == "auto"


def test_inspect_device_reports_the_default_look(tmp_path):
    report = devices.inspect_device(_sony_arw(tmp_path / "DSC00004.ARW"))
    assert report["tier"] == "B" and "默认观感" in report["statement"] and "借用" in report["statement"]
