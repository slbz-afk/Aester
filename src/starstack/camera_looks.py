"""Borrowed camera colour looks (docs/SONY_A7RV_LOOK_PREREG_20260926.md).

A camera look is the HueSatMap of a public DCP profile, applied on top of the product's own colour basis (the body's
LibRaw/DNG matrix and white balance stay the product's) at a strength the user decided.  The tables live in
``camera_look_data`` (generated from pinned DCP files by ``scripts/build_camera_look_data.py``) as the exact float32
little-endian bytes of the source tag, so they travel with the source code; ``table`` checks them against the pinned
hash on every load.  Which body gets which look is recorded per body in ``devices_registry``
(``colour_look``) by user decision; nothing here guesses a look for a body without one.

Strength ``s`` scales the table entries: hue shift ``s·Δh``, saturation and value multipliers ``1 + s·(m − 1)``.  At
``s = 1`` the scaled table equals the source table bit for bit.
"""
from __future__ import annotations

import base64
import hashlib

import numpy as np

VERSION = "camera-looks-2"          # 2: tables in camera_look_data (16 bodies)
LOOK_NAME = "camera_hsm"
# Provenance and table bytes are generated from pinned DCP files by scripts/build_camera_look_data.py.
from .camera_look_data import LOOKS, TABLES as _DATA  # noqa: E402


def table(look_id: str) -> np.ndarray:
    """The source table as float64 (hue, sat, val) × 90 hues × 30 saturations, read-only; the bytes must match the pin."""
    if look_id not in LOOKS:
        raise ValueError(f"未知机型观感 {look_id!r}")
    raw = base64.b64decode(_DATA[look_id])
    if hashlib.sha256(raw).hexdigest() != LOOKS[look_id]["table_sha256"]:
        raise ValueError(f"机型观感 {look_id} 的表数据与固定哈希不符")
    hues, sats, vals = LOOKS[look_id]["dims"]
    if vals != 1:
        raise ValueError("只支持单一明度分区的色相/饱和度表")
    out = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(hues, sats, 3)
    out.setflags(write=False)
    return out


def check_strength(strength) -> float:
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, (int, float, np.integer, np.floating)):
        raise ValueError("观感强度必须为数值")
    s = float(strength)
    if not 0.0 < s <= 1.0:
        raise ValueError("观感强度须在 (0, 1] 内")
    return s


def scaled_table(look_id: str, strength) -> np.ndarray:
    """Table entries moved towards the identity by ``strength``: Δh·s, 1 + s·(m − 1) for saturation and value."""
    s = check_strength(strength)
    source = table(look_id)
    out = np.empty_like(source)
    out[..., 0] = source[..., 0] * s
    out[..., 1:] = 1.0 + s * (source[..., 1:] - 1.0)
    out.setflags(write=False)
    return out


def make_look(look_id: str, strength, gamut: str):
    """(callable, record) applying the scaled table with the same mathematics as the Z8 ``dcp_hsm`` look."""
    from .photo_render import hsm_look
    look, counters = hsm_look(scaled_table(look_id, strength), gamut)
    info = LOOKS[look_id]
    record = {"look": LOOK_NAME, "look_id": look_id, "strength": check_strength(strength), "version": VERSION,
              "donor": info["donor"], "source": info["source"], "source_sha256": info["source_sha256"],
              "profile_copyright": info["profile_copyright"], "table": info["tag_name"], "table_sha256": info["table_sha256"],
              "space": "ProPhoto HSV via XYZ D50 (experimental_dcp constants)", "counters": counters,
              "range_policy": "negative_ProPhoto_bypasses_HSM; no_S_or_V_clipping"}
    return look, record


def camera_of(master_meta: dict) -> dict | None:
    """The body recorded in a master's colour basis (``colour_basis.reference.camera``: DNG tags or, since
    camera-looks-1, the RAW container tags)."""
    basis = master_meta.get("colour_basis") if isinstance(master_meta, dict) else None
    reference = basis.get("reference") if isinstance(basis, dict) else None
    camera = reference.get("camera") if isinstance(reference, dict) else None
    return camera if isinstance(camera, dict) else None


def default_for(master_meta: dict) -> dict | None:
    """{look_id, strength} registered for the master's body, or None (no body recorded, unregistered, or no look)."""
    from . import devices
    camera = camera_of(master_meta)
    if camera is None:
        return None
    entry = devices.lookup(camera.get("make"), camera.get("model"))
    if entry is None and camera.get("unique_camera_model"):
        entry = devices.lookup(None, camera.get("unique_camera_model"))
    chosen = (entry or {}).get("colour_look")
    if not chosen:
        return None
    return {"look_id": chosen["id"], "strength": check_strength(chosen["strength"])}
