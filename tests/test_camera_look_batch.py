"""Look batch of 2026-09-26 (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md B): pinned open tables, the registry
mapping (own DCP vs borrowed, strength 0.5, provisional), CR3 identification and exact model matching."""
import base64
import hashlib
import io
import struct

import numpy as np
import pytest
import tifffile

from starstack import camera_look_data as data, camera_looks as cl, devices, devices_registry as reg, photo_render as pr

OWN = {"ILCE-7M3": "sony-ilce7m3-art-pd-d65", "ILCE-7M4": "sony-ilce7m4-rt-cc0-d50", "ILCE-7SM3": "sony-ilce7sm3-rt-pd-d65",
       "ILCE-6400": "sony-ilce6400-rt-pd-d50", "NIKON Z 6": "nikon-z6-rt-pd-d65", "NIKON Z 6_2": "nikon-z6ii-rt-pd-d65",
       "NIKON Z 7": "nikon-z7-rt-pd-d65", "NIKON D850": "nikon-d850-rt-pd-d65", "NIKON D750": "nikon-d750-art-pd-d65",
       "Canon EOS 6D": "canon-eos6d-art-pd-d65", "Canon EOS 5D Mark IV": "canon-eos5dmk4-rt-pd-d65",
       "Canon EOS R": "canon-eosr-rt-pd-d65", "Canon EOS RP": "canon-eosrp-rt-cc0-d50", "Canon EOS R5": "canon-eosr5-rt-pd-d65",
       "Canon EOS R6": "canon-eosr6-rt-pd-d65", "Canon EOS R8": "canon-eosr8-rt-cc0-d50"}
BORROWED = {"ILCE-7CM2": "sony-ilce7m4-rt-cc0-d50", "NIKON Z 7_2": "nikon-z7-rt-pd-d65", "Canon EOS R6m2": "canon-eosr8-rt-cc0-d50",
            "ILCE-7RM5": "sony-ilce7m4-rt-cc0-d50"}


# strengths the user set on 2026-09-27 after the comparison page; unlisted bodies keep 0.5 (α7R V: accepted at 0.5)
STRENGTH = {"ILCE-7M3": 0.75, "ILCE-7M4": 0.75, "Canon EOS R": 0.75, "Canon EOS R6": 0.75, "Canon EOS R5": 0.4,
            "ILCE-7SM3": 0.75, "NIKON D850": 0.65, "Canon EOS 5D Mark IV": 0.75, "ILCE-6400": 0.75,
            "ILCE-7CM2": 0.75, "NIKON Z 7_2": 0.75}


def by_model():
    return {m: e for e in reg.REGISTRY for m in e["model_ids"]}


def test_every_table_is_open_pinned_and_well_formed():
    assert set(data.LOOKS) == set(data.TABLES) == set(OWN.values())
    for look_id, info in data.LOOKS.items():
        raw = base64.b64decode(data.TABLES[look_id])
        assert hashlib.sha256(raw).hexdigest() == info["table_sha256"] and len(raw) == 90 * 30 * 3 * 4
        assert info["licence"] in ("CC0", "public domain")
        assert info["profile_copyright"] in ("RawTherapee CC0", "public domain")
        assert info["calibration_illuminant"] in ("D65 (21)", "D50 (23)") and info["dims"] == [90, 30, 1]
        assert len(info["source_sha256"]) == 64
        t = cl.table(look_id)
        assert np.isfinite(t).all() and (t[..., 1:] > 0).all()
    # the α7R V table is unchanged by the batch
    assert data.LOOKS["sony-ilce7m4-rt-cc0-d50"]["table_sha256"] == "4acfd9674101dcb948b760787750082f1df6154ab8e39c321f2acd203a823de2"


def test_registry_maps_each_body_to_its_look():
    models = by_model()
    for model, look_id in {**OWN, **BORROWED}.items():
        entry = models[model]
        look = entry["colour_look"]
        assert entry["tier"] == "B" and entry["verified_frames"] == []
        assert look["id"] == look_id and look["strength"] == STRENGTH.get(model, 0.5)
        assert look["borrowed"] is (model in BORROWED)
        if model != "ILCE-7RM5":
            assert look["status"] == (reg.SET_BY_USER.format(strength=STRENGTH[model]) if model in STRENGTH else reg.KEPT_BY_USER)
        if look["borrowed"] and model != "ILCE-7RM5":
            assert look["donor"] and look["donor_reason"]
    assert devices.registry_problems() == []
    with_look = {m for m, e in models.items() if e.get("colour_look")}
    assert with_look == set(OWN) | set(BORROWED)


@pytest.mark.parametrize("model", sorted({**OWN, **BORROWED}))
def test_default_look_for_each_registered_model(model):
    meta = {"gamut": "rec2020", "colour_basis": {"reference": {"camera": {"make": "x", "model": model}}}}
    assert pr.default_look_fields(meta) == {"look": "camera_hsm", "look_id": {**OWN, **BORROWED}[model],
                                            "look_strength": STRENGTH.get(model, 0.5)}


@pytest.mark.parametrize("model,expected", [
    ("Canon EOS R6m3", None), ("Canon EOS R6 Mark III", None), ("Canon EOS R7", None), ("Canon EOS Ra", None),
    ("NIKON Z6_3", None), ("NIKON Z 6_3", None), ("ILCE-7M5", None), ("NIKON D810A", None),
    ("canon eos r6", "Canon EOS R6"), ("  NIKON  Z 6  ", "Nikon Z 6"), ("ILCE-7RM3A", "Sony α7R III"),
])
def test_lookup_is_exact_after_folding(model, expected):
    entry = devices.lookup("x", model)
    assert (entry["name"] if entry else None) == expected


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _cmt1(make="Canon", model="Canon EOS R6", software="Firmware Version 1.8.1"):
    stream = io.BytesIO()
    tifffile.imwrite(stream, np.zeros((1, 1), np.uint8), software=software,
                     extratags=[(271, 2, None, make, True), (272, 2, None, model, True)])
    return stream.getvalue()


def _cr3(path, cmt1=None, *, canon_uuid=True):
    inner = b"" if cmt1 is None else _box(b"CMT1", cmt1)
    uuid = (devices.CR3_CANON_UUID if canon_uuid else b"\x00" * 16) + _box(b"CNCV", b"CanonCR3_001/00.09.00/00.00.00") + inner
    data = _box(b"ftyp", b"crx \x00\x00\x00\x01crx isom") + _box(b"moov", _box(b"uuid", uuid)) + _box(b"mdat", b"\x00" * 64)
    path.write_bytes(data)
    return path


def test_cr3_identity_from_cmt1(tmp_path):
    ident = devices.identify_camera(_cr3(tmp_path / "a.CR3", _cmt1()))
    assert ident["container"] == "cr3" and ident["identified"]
    assert (ident["make"], ident["model"], ident["software"]) == ("Canon", "Canon EOS R6", "Firmware Version 1.8.1")
    report = devices.inspect_device(tmp_path / "a.CR3")
    assert report["registered"] and report["registry"]["name"] == "Canon EOS R6" and "默认观感" in report["statement"]


def test_cr3_without_cmt1_or_broken_is_unidentified(tmp_path):
    for name, path in (("no-cmt1", _cr3(tmp_path / "b.CR3", None)), ("other-uuid", _cr3(tmp_path / "c.CR3", _cmt1(), canon_uuid=False))):
        ident = devices.identify_camera(path)
        assert not ident["identified"] and "CMT1" in ident["note"], name
    broken = tmp_path / "d.CR3"
    broken.write_bytes(struct.pack(">I4s", 4, b"ftyp") + b"\x00" * 32)
    ident = devices.identify_camera(broken)
    assert not ident["identified"] and "CR3" in ident["note"]
    huge = tmp_path / "e.CR3"
    huge.write_bytes(struct.pack(">I4s", 10**9, b"moov") + b"\x00" * 32)
    assert not devices.identify_camera(huge)["identified"]
