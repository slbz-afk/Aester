"""Working-gamut definitions and a minimal ICC v4 matrix/TRC profile writer and reader.

Everything here is plain colorimetry from published chromaticities: RGB→XYZ matrices are derived from the
primaries and the native white, chromatic adaptation is Bradford, the ICC colorants are the D50-adapted
primaries as every matrix/TRC profile stores them.  The profiles written here describe *linear* scene-referred
masters (gamma 1.0 TRC).  The display product keeps using the lcms sRGB profile (``ImageCms.createProfile``)
because ``photographic_job._validate_srgb`` compares against that profile tag by tag.

No colour appearance modelling and no camera characterisation live here.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import struct

import numpy as np

VERSION = "icc-profiles-1"

# CIE 1931 chromaticities; whites as (x, y).
D65_XY = (0.3127, 0.3290)
D50_XY = (0.3457, 0.3585)

GAMUTS = {
    "srgb": {"label": "linear sRGB / Rec.709", "primaries": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), "white": D65_XY,
             "description": "StarStack linear sRGB (D65)"},
    "rec2020": {"label": "linear Rec.2020", "primaries": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), "white": D65_XY,
                "description": "StarStack linear Rec.2020 (D65)"},
    "prophoto": {"label": "linear ProPhoto RGB", "primaries": ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)),
                 "white": D50_XY, "description": "StarStack linear ProPhoto RGB (D50)"},
}

# Bradford cone response matrix (Lam 1985), as used by the ICC specification annex E.
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]], dtype=np.float64)


def xy_to_xyz(xy) -> np.ndarray:
    """Chromaticity (x, y) → XYZ with Y = 1."""
    x, y = float(xy[0]), float(xy[1])
    if not 0 < y <= 1 or not 0 <= x <= 1:
        raise ValueError("invalid chromaticity")
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def rgb_to_xyz_matrix(primaries, white_xy) -> np.ndarray:
    """RGB→XYZ for the given primaries, scaled so that RGB=(1,1,1) maps to the white with Y=1."""
    p = np.array([xy_to_xyz(c) for c in primaries], dtype=np.float64).T   # columns = XYZ of each primary at Y=1
    w = xy_to_xyz(white_xy)
    s = np.linalg.solve(p, w)
    m = p * s[None, :]
    if not np.isfinite(m).all() or abs(m[1].sum() - 1.0) > 1e-9:
        raise ValueError("degenerate primaries")
    return m


def bradford(from_white_xy, to_white_xy) -> np.ndarray:
    """Bradford chromatic adaptation matrix XYZ(from) → XYZ(to)."""
    src, dst = xy_to_xyz(from_white_xy), xy_to_xyz(to_white_xy)
    ratio = (_BRADFORD @ dst) / (_BRADFORD @ src)
    return np.linalg.inv(_BRADFORD) @ np.diag(ratio) @ _BRADFORD


class Gamut:
    """Derived colorimetry of one working gamut (all matrices float64, read-only)."""

    def __init__(self, name: str):
        if name not in GAMUTS:
            raise ValueError(f"unknown gamut {name!r}; expected one of {sorted(GAMUTS)}")
        spec = GAMUTS[name]
        self.name = name
        self.label = spec["label"]
        self.description = spec["description"]
        self.white_xy = tuple(spec["white"])
        self.primaries = tuple(tuple(c) for c in spec["primaries"])
        self.rgb_to_xyz = rgb_to_xyz_matrix(self.primaries, self.white_xy)          # native white
        self.xyz_to_rgb = np.linalg.inv(self.rgb_to_xyz)
        self.luma = self.rgb_to_xyz[1].copy()                                         # Y row: luminance weights
        self.to_d50 = bradford(self.white_xy, D50_XY)
        self.colorants_d50 = self.to_d50 @ self.rgb_to_xyz                              # what an ICC profile stores
        for a in (self.rgb_to_xyz, self.xyz_to_rgb, self.luma, self.to_d50, self.colorants_d50):
            a.setflags(write=False)

    def matrix_to(self, other: "Gamut") -> np.ndarray:
        """Linear RGB (this gamut, native white) → linear RGB (other gamut, its native white), Bradford-adapted."""
        adapt = np.eye(3) if self.white_xy == other.white_xy else bradford(self.white_xy, other.white_xy)
        return other.xyz_to_rgb @ adapt @ self.rgb_to_xyz

    def matrix_from_xyz_d50(self) -> np.ndarray:
        """XYZ (D50 PCS) → linear RGB of this gamut, i.e. the inverse of the ICC colorants."""
        return np.linalg.inv(self.colorants_d50)


SRGB = Gamut("srgb")


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    a = np.asarray(linear, dtype=np.float64)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(np.maximum(a, 0.0), 1 / 2.4) - 0.055)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    a = np.asarray(encoded, dtype=np.float64)
    return np.where(a <= 0.04045, a / 12.92, np.power((np.maximum(a, 0.0) + 0.055) / 1.055, 2.4))


# ----------------------------------------------------------------------------------------------------------------
# ICC v4 matrix/TRC profile (display class, RGB, XYZ PCS)

def _s15(values) -> bytes:
    out = b""
    for v in np.asarray(values, dtype=np.float64).reshape(-1):
        out += struct.pack(">i", int(round(float(v) * 65536)))
    return out


def _xyz_tag(xyz) -> bytes:
    return b"XYZ " + b"\0" * 4 + _s15(xyz)


def _mluc(text: str) -> bytes:
    data = text.encode("utf-16-be")
    return b"mluc" + b"\0" * 4 + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(data), 28) + data


def _para_gamma(gamma: float) -> bytes:
    return b"para" + b"\0" * 4 + struct.pack(">HH", 0, 0) + _s15([gamma])


def _pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


def write_linear_profile(gamut: str | Gamut, *, copyright_text: str = "StarStack; no rights reserved on this profile") -> bytes:
    """Bytes of a v4.3 RGB matrix/TRC profile with gamma 1.0 curves for the given working gamut.

    Tags: desc, cprt, wtpt (D50 PCS white), chad (Bradford native→D50), rXYZ/gXYZ/bXYZ (D50-adapted colorants),
    rTRC/gTRC/bTRC (parametric gamma 1.0).  The profile ID is the MD5 defined by the specification (header with
    flags, rendering intent and ID zeroed).  The date is fixed so that identical inputs give identical bytes.
    """
    g = gamut if isinstance(gamut, Gamut) else Gamut(gamut)
    d50 = xy_to_xyz(D50_XY)
    tags = [
        (b"desc", _mluc(g.description)),
        (b"cprt", _mluc(copyright_text)),
        (b"wtpt", _xyz_tag(d50)),
        (b"chad", b"sf32" + b"\0" * 4 + _s15(g.to_d50)),
        (b"rXYZ", _xyz_tag(g.colorants_d50[:, 0])),
        (b"gXYZ", _xyz_tag(g.colorants_d50[:, 1])),
        (b"bXYZ", _xyz_tag(g.colorants_d50[:, 2])),
    ]
    trc = _para_gamma(1.0)
    table_size = 4 + 12 * (len(tags) + 3)
    offset = 128 + table_size
    entries, body = [], b""
    for key, data in tags:
        padded = _pad4(data)
        entries.append((key, offset + len(body), len(data)))
        body += padded
    trc_offset = offset + len(body)          # the three curves share one element, as the specification allows
    body += _pad4(trc)
    for key in (b"rTRC", b"gTRC", b"bTRC"):
        entries.append((key, trc_offset, len(trc)))
    table = struct.pack(">I", len(entries)) + b"".join(struct.pack(">4sII", k, o, n) for k, o, n in entries)
    size = 128 + len(table) + len(body)
    when = _dt.datetime(2026, 9, 22, 0, 0, 0)
    header = struct.pack(">I4sI4s4s4s", size, b"\0" * 4, 0x04300000, b"mntr", b"RGB ", b"XYZ ")
    header += struct.pack(">6H", when.year, when.month, when.day, when.hour, when.minute, when.second)
    header += b"acsp" + b"\0" * 4                 # signature, primary platform (unspecified)
    header += struct.pack(">I", 0)                # flags
    header += b"\0" * 8                            # manufacturer, model
    header += b"\0" * 8                            # attributes
    header += struct.pack(">I", 1)                # rendering intent: relative colorimetric
    header += _s15(d50)                           # PCS illuminant
    header += b"\0" * 4                            # creator
    header += b"\0" * 16                           # profile ID (computed below)
    header += b"\0" * 28
    assert len(header) == 128
    profile = bytearray(header + table + body)
    hashed = bytearray(profile)
    hashed[44:48] = b"\0" * 4; hashed[64:68] = b"\0" * 4; hashed[84:100] = b"\0" * 16
    profile[84:100] = hashlib.md5(bytes(hashed)).digest()
    return bytes(profile)


def read_profile(data: bytes) -> dict:
    """Bounded parse of the colorimetric tags of a matrix/TRC RGB profile (ours or anyone's).

    Returns version, class, colour space, PCS, description, wtpt, chad (or None), colorants (3×3, columns R G B, D50),
    TRC kind per channel ('linear' for gamma-1 parametric/identity curves, otherwise 'other'), and the tag table.
    """
    if len(data) < 132 or data[36:40] != b"acsp":
        raise ValueError("not an ICC profile")
    size = struct.unpack_from(">I", data)[0]
    count = struct.unpack_from(">I", data, 128)[0]
    if size > len(data) or size < 132 or count > 256 or 132 + 12 * count > size:
        raise ValueError("invalid ICC size or tag table")
    tags = {}
    for i in range(count):
        key, offset, length = struct.unpack_from(">4sII", data, 132 + 12 * i)
        if key in tags or offset < 132 + 12 * count or length < 8 or offset + length > size:
            raise ValueError("invalid ICC tag")
        tags[key] = data[offset:offset + length]

    def xyz(key):
        block = tags.get(key)
        if block is None or block[:4] != b"XYZ " or len(block) < 20:
            raise ValueError(f"missing XYZ tag {key!r}")
        return np.array([struct.unpack_from(">i", block, 8 + 4 * k)[0] / 65536 for k in range(3)], dtype=np.float64)

    def trc_kind(key):
        block = tags.get(key)
        if block is None:
            return "missing"
        if block[:4] == b"para" and len(block) >= 16:
            fn = struct.unpack_from(">H", block, 8)[0]
            g = struct.unpack_from(">i", block, 12)[0] / 65536
            return "linear" if fn == 0 and abs(g - 1.0) < 1e-6 else "other"
        if block[:4] == b"curv" and len(block) >= 12:
            n = struct.unpack_from(">I", block, 8)[0]
            if n == 0:
                return "linear"
            if n == 1 and len(block) >= 14:
                return "linear" if struct.unpack_from(">H", block, 12)[0] == 0x0100 else "other"
            return "other"
        return "other"

    def text(key):
        block = tags.get(key)
        if block is None:
            return None
        if block[:4] == b"mluc" and len(block) >= 28:
            n_rec, rec_size = struct.unpack_from(">II", block, 8)
            if n_rec >= 1 and rec_size >= 12:
                length, offset = struct.unpack_from(">II", block, 20)
                return block[offset:offset + length].decode("utf-16-be", "replace")
        if block[:4] == b"desc" and len(block) >= 12:
            n = struct.unpack_from(">I", block, 8)[0]
            return block[12:12 + n].rstrip(b"\0").decode("ascii", "replace")
        return None

    chad = None
    if b"chad" in tags and tags[b"chad"][:4] == b"sf32" and len(tags[b"chad"]) >= 44:
        chad = np.array([struct.unpack_from(">i", tags[b"chad"], 8 + 4 * k)[0] / 65536 for k in range(9)]).reshape(3, 3)
    return {
        "version": tuple(data[8:10]), "class": data[12:16].decode("ascii", "replace"),
        "color_space": data[16:20].decode("ascii", "replace"), "pcs": data[20:24].decode("ascii", "replace"),
        "description": text(b"desc"), "wtpt": xyz(b"wtpt"), "chad": chad,
        "colorants_d50": np.stack([xyz(b"rXYZ"), xyz(b"gXYZ"), xyz(b"bXYZ")], axis=1),
        "trc": {c: trc_kind(k) for c, k in (("r", b"rTRC"), ("g", b"gTRC"), ("b", b"bTRC"))},
        "tags": sorted(k.decode("ascii", "replace") for k in tags),
        "sha256": hashlib.sha256(data[:size]).hexdigest(),
    }


def identify_linear_profile(data: bytes, *, tolerance: float = 2e-4) -> str:
    """Name of the working gamut whose D50 colorants match the profile within ``tolerance`` (or raise)."""
    info = read_profile(data)
    if info["color_space"] != "RGB " or info["pcs"] != "XYZ ":
        raise ValueError("expected an RGB profile with XYZ PCS")
    if any(kind != "linear" for kind in info["trc"].values()):
        raise ValueError("profile curves are not linear (gamma 1.0)")
    for name in GAMUTS:
        g = Gamut(name)
        if np.abs(info["colorants_d50"] - g.colorants_d50).max() <= tolerance:
            return name
    raise ValueError("profile colorants match no supported working gamut")
