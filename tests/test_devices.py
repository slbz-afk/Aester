"""Device support tiers (docs/PRE_RELEASE_CLOSURE_20260922.md D): registry integrity, container-only identification,
the inspect report on a synthetic Z8 DNG, an unreadable container and a FITS header; no camera-specific code path."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starstack import cli, devices, devices_registry as reg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402


def test_registry_is_consistent_and_states_the_agreed_tiers():
    assert devices.registry_problems() == []
    by_model = {m: e for e in reg.REGISTRY for m in e["model_ids"]}
    for model in ("ILCE-7M3", "ILCE-7M4", "ILCE-7CM2", "ILCE-7CR", "ILCE-7RM3", "ILCE-7RM3A", "ILCE-7RM5"):
        assert by_model[model]["tier"] == "B" and by_model[model]["verified_frames"] == []        # 用户决定：索尼停留在 B 级
        assert set(by_model[model]["raw_variants"]) == {"uncompressed", "lossy_compressed", "lossless_compressed"}
        assert all(v["decoder"] == "declared" for v in by_model[model]["raw_variants"].values())
        assert "star_eater" in by_model[model]["quirks"]
    assert by_model["NIKON Z 8"]["tier"] == "A" and by_model["ZWO Seestar S30 Pro"]["tier"] == "A"
    assert by_model["Canon EOS 5D Mark III"]["tier"] == "B+" and by_model["DC-S5"]["tier"] == "B+"
    listing = devices.registry_listing()
    assert listing["tiers"] == reg.TIERS and len(listing["devices"]) == len(reg.REGISTRY)
    json.dumps(listing, allow_nan=False)


def _sony_like_tiff(path, model="ILCE-7M4", raw_file_type=3):
    tags = [(271, 2, None, "SONY", True), (272, 2, None, model, True), (0x7000, 3, 1, raw_file_type, True)]
    tifffile.imwrite(path, np.zeros((8, 8), np.uint16), extratags=tags)
    return path


def test_identify_reads_make_model_and_sony_raw_file_type_from_the_container(tmp_path):
    arw = _sony_like_tiff(tmp_path / "DSC00001.ARW")
    ident = devices.identify_camera(arw)
    assert ident["identified"] and ident["container"] == "tiff" and ident["make"] == "SONY" and ident["model"] == "ILCE-7M4"
    assert ident["sony_raw_file_type"] == 3 and ident["sony_raw_variant"] == "lossless_compressed"
    assert devices.lookup("SONY", "ILCE-7M4")["name"] == "Sony α7 IV"
    assert devices.lookup("SONY", " ilce-7rm3a ")["model_ids"] == ["ILCE-7RM3", "ILCE-7RM3A"]
    assert devices.lookup("SONY", "ILCE-7M5") is None
    report = devices.inspect_device(arw)                       # not a real ARW: metadata route fails, no exception
    assert report["tier"] == "B" and report["registered"] and report["colour_basis"]["state"] in ("unavailable", "error")
    assert "B 级" in report["statement"] and "B+" in report["next_tier"]
    json.dumps(report, allow_nan=False)


def test_inspect_builds_the_colour_basis_on_a_z8_dng_and_reports_tier_a(tmp_path):
    dng = cf.write_reference_dng(tmp_path / "frame00001.dng")
    ident = devices.identify_camera(dng)
    assert ident["make"] == "NIKON CORPORATION" and ident["model"] == "NIKON Z 8" and ident["unique_model"] == "Nikon Z 8"
    assert ident["sony_raw_file_type"] is None
    report = devices.inspect_device(dng)
    assert report["tier"] == "A" and report["registry"]["name"] == "Nikon Z 8"
    assert report["metadata"]["route"] == "dng_tags" and report["metadata"]["available"]
    assert report["colour_basis"]["state"] == "ok" and report["colour_basis"]["black_DN"] == [cf.Z8_BLACK] * 3
    assert report["next_tier"] == reg.NEXT["A"]


def test_unreadable_container_and_fits_header_are_reported_not_guessed(tmp_path):
    rw2 = tmp_path / "P1000001.RW2"
    rw2.write_bytes(b"IIU\x00not a tiff container at all")
    ident = devices.identify_camera(rw2)
    assert not ident["identified"] and ident["container"] is None and "未识别" in ident["note"]
    report = devices.inspect_device(rw2)
    assert report["tier"] == "unregistered" and not report["registered"] and "未登记" in report["statement"]
    from astropy.io import fits
    header = fits.Header()
    header["CREATOR"] = "ZWO Seestar S30 Pro"
    header["EXPTIME"] = 10.0
    header["BAYERPAT"] = "GRBG"
    header["PROGRAM"] = "8.46"
    header["WIDECAM"] = 0
    path = tmp_path / "light.fit"
    fits.writeto(path, np.zeros((3840, 2160), np.uint16), header)
    ident = devices.identify_camera(path)
    assert ident["container"] == "fits" and ident["model"] == "ZWO Seestar S30 Pro" and ident["identified"]
    report = devices.inspect_device(path)
    assert report["tier"] == "A" and report["metadata"]["route"] == "fits_header" and report["metadata"]["layout"] == "cfa"
    assert report["colour_basis"]["state"] == "not_applicable"
    with pytest.raises(ValueError):
        devices.inspect_device(tmp_path / "missing.arw")


def test_cli_devices_and_inspect_device_print_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["starstack", "devices"])
    assert cli.main() == 0
    listing = json.loads(capsys.readouterr().out)
    assert {d["name"] for d in listing["devices"]} >= {"Sony α7 IV", "Nikon Z 8"}
    monkeypatch.setattr(sys, "argv", ["starstack", "devices", "--check"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out) == {"problems": [], "ok": True}
    arw = _sony_like_tiff(tmp_path / "DSC00002.ARW", model="ILCE-7CR", raw_file_type=2)
    monkeypatch.setattr(sys, "argv", ["starstack", "inspect-device", str(arw)])
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["registry"]["name"] == "Sony α7CR" and report["sony_raw_variant"] == "lossy_compressed" and report["tier"] == "B"
