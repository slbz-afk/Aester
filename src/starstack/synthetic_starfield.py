"""Asymmetric synthetic star fields with analytic ground truth.

Used to prove Siril's coordinate conventions (row order, pixel origin,
homography direction) and to exercise the registration adapter without any
photograph.  Truth positions are defined in the diagnostics convention:
TOP-DOWN rows, zero-based pixel centres, ``(x, y)`` = (column, row).  Frames
are written as standard bottom-up FITS (``ROWORDER = 'BOTTOM-UP'``), three
identical RGB planes, 16-bit unsigned via BZERO, using only NumPy so that the
generator runs anywhere; Siril reads such files natively.

Transforms are stored as source(frame)->reference matrices acting on column
vectors ``[x, y, 1]`` in truth coordinates; a frame is rendered at
``F_k^{-1}(p_ref)`` for every reference star that lands inside the frame.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

__all__ = ["GENERATOR_VERSION", "default_layout", "rotation_scale_about", "translation",
           "render_frame", "write_fits_rgb16", "write_fits_rgb32", "read_fits_minimal", "generate_dataset"]

GENERATOR_VERSION = "1.1.0"   # 1.1.0: write_fits_rgb32 (float cubes with distinct planes, for the transient tests)


def translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def rotation_scale_about(angle_deg: float, scale: float, cx: float, cy: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """p' = s R(p - c) + c + t; positive angle rotates +x towards +y (image rows grow downwards)."""
    t = math.radians(angle_deg)
    c, s = math.cos(t) * scale, math.sin(t) * scale
    return np.array([[c, -s, cx - c * cx + s * cy + tx], [s, c, cy - s * cx - c * cy + ty], [0.0, 0.0, 1.0]])


def default_layout(width: int = 1200, height: int = 901, seed: int = 20260917, count: int = 170) -> dict:
    """Reference star list: anchors at the centre and four corners, deliberately
    asymmetric asterisms, and a random cloud denser in the upper-left half.
    Minimum separation 12 px so nearest-neighbour matching is unambiguous."""
    rng = np.random.default_rng(seed)
    anchors = [
        ("center", 603.25, 447.75, 30000.0),
        ("corner_top_left", 28.5, 31.25, 26000.0),
        ("corner_top_right", 1163.75, 24.5, 24000.0),
        ("corner_bottom_left", 36.25, 866.5, 25000.0),
        ("corner_bottom_right", 1171.5, 872.25, 27000.0),
        ("asterism_L_1", 150.0, 120.0, 20000.0), ("asterism_L_2", 150.0, 170.0, 18000.0),
        ("asterism_L_3", 150.0, 220.0, 16000.0), ("asterism_L_4", 200.0, 220.0, 14000.0),
        ("asterism_L_5", 250.0, 220.0, 12000.0),
        ("chain_1", 900.0, 700.0, 15000.0), ("chain_2", 930.0, 725.0, 13000.0), ("chain_3", 960.0, 750.0, 11000.0),
    ]
    stars = [{"star_id": name, "x": x, "y": y, "amplitude": a, "sigma": 1.6} for name, x, y, a in anchors]
    positions = np.array([[s["x"], s["y"]] for s in stars])
    attempts = 0
    while len(stars) < count and attempts < 20000:
        attempts += 1
        # asymmetric density: 65 % of the random stars in the upper-left region
        if rng.random() < 0.65:
            x = rng.uniform(12, width * 0.55)
            y = rng.uniform(12, height * 0.6)
        else:
            x = rng.uniform(12, width - 13)
            y = rng.uniform(12, height - 13)
        if np.min(np.hypot(positions[:, 0] - x, positions[:, 1] - y)) < 12.0:
            continue
        amplitude = float(10 ** rng.uniform(math.log10(2000), math.log10(40000)))
        stars.append({"star_id": f"r{len(stars):03d}", "x": round(float(x), 3), "y": round(float(y), 3),
                      "amplitude": round(amplitude, 1), "sigma": round(float(rng.uniform(1.4, 1.9)), 3)})
        positions = np.vstack([positions, [x, y]])
    return {"width": width, "height": height, "seed": seed, "stars": stars}


def default_transforms(width: int = 1200, height: int = 901) -> list[dict]:
    """Four frames: identity reference, translation, rotation about an explicit
    centre with translation, rotation plus scale about another centre."""
    return [
        {"frame": 1, "kind": "identity", "matrix": np.eye(3).tolist(), "description": "reference"},
        {"frame": 2, "kind": "translation", "matrix": translation(7.25, -3.5).tolist(),
         "description": "translation (+7.25, -3.5)"},
        {"frame": 3, "kind": "rotation", "matrix": rotation_scale_about(2.0, 1.0, 600.0, 450.0, -4.0, 6.25).tolist(),
         "description": "rotation +2.0 deg about (600, 450) then translation (-4.0, +6.25)"},
        {"frame": 4, "kind": "rotation_scale", "matrix": rotation_scale_about(-1.3, 1.004, 300.0, 200.0, 2.5, 1.75).tolist(),
         "description": "rotation -1.3 deg, scale 1.004 about (300, 200), translation (+2.5, +1.75)"},
    ]


def render_frame(layout: dict, matrix, *, background: float = 800.0, noise_sigma: float = 12.0,
                 seed: int = 1) -> tuple[np.ndarray, list[dict]]:
    """Render the stars of ``layout`` seen through frame->reference ``matrix``.

    Returns the TOP-DOWN uint16 image (H, W) and the list of rendered stars
    with their frame coordinates (truth for this frame).
    """
    width, height = layout["width"], layout["height"]
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    rng = np.random.default_rng(seed)
    image = np.full((height, width), background, dtype=np.float64)
    image += rng.normal(0.0, noise_sigma, size=image.shape)
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    rendered = []
    for star in layout["stars"]:
        v = inverse @ np.array([star["x"], star["y"], 1.0])
        x, y = v[0] / v[2], v[1] / v[2]
        if not (2.0 <= x <= width - 3.0 and 2.0 <= y <= height - 3.0):
            continue
        s = star["sigma"]
        r0, r1 = max(0, int(y - 5 * s)), min(height, int(y + 5 * s) + 2)
        c0, c1 = max(0, int(x - 5 * s)), min(width, int(x + 5 * s) + 2)
        yy, xx = rows[r0:r1], cols[:, c0:c1]
        image[r0:r1, c0:c1] += star["amplitude"] * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
        rendered.append({"star_id": star["star_id"], "x": float(x), "y": float(y), "amplitude": star["amplitude"]})
    return np.clip(np.round(image), 0, 65535).astype(np.uint16), rendered


def _card(key: str, value, comment: str = "") -> str:
    if isinstance(value, bool):
        text = "T" if value else "F"
        card = f"{key:<8}= {text:>20}"
    elif isinstance(value, int):
        card = f"{key:<8}= {value:>20d}"
    elif isinstance(value, float):
        card = f"{key:<8}= {value:>20.10G}"
    else:
        card = f"{key:<8}= '{str(value):<8}'"
    if comment:
        card = f"{card} / {comment}"
    return card[:80].ljust(80)


def write_fits_rgb16(path: Path, image_top_down: np.ndarray, *, comments: list[str] = (), extra_cards: list[str] = ()) -> str:
    """Write a TOP-DOWN uint16 image as a bottom-up 16-bit RGB FITS (three equal planes).

    ``extra_cards`` are complete 80-character header cards (see ``_card``)
    inserted before the comments, e.g. a WCS/SIP solution.
    """
    if image_top_down.dtype != np.uint16 or image_top_down.ndim != 2:
        raise ValueError("expected a 2-D uint16 image")
    height, width = image_top_down.shape
    cards = [_card("SIMPLE", True, "conforms to FITS standard"), _card("BITPIX", 16), _card("NAXIS", 3),
             _card("NAXIS1", width), _card("NAXIS2", height), _card("NAXIS3", 3), _card("EXTEND", True),
             _card("BZERO", 32768, "unsigned 16-bit"), _card("BSCALE", 1),
             _card("ROWORDER", "BOTTOM-UP", "standard FITS row order"), _card("CTYPE3", "RGB")]
    for card in extra_cards:
        if len(card) != 80:
            raise ValueError("extra cards must be exactly 80 characters")
        cards.append(card)
    for comment in comments:
        cards.append(("COMMENT " + comment)[:80].ljust(80))
    cards.append("END".ljust(80))
    header = "".join(cards)
    header += " " * ((2880 - len(header) % 2880) % 2880)
    plane = image_top_down[::-1, :].astype(np.int32) - 32768   # FITS row 0 is the bottom row
    data = np.repeat(plane.astype(">i2")[None, :, :], 3, axis=0).tobytes()
    data += b"\0" * ((2880 - len(data) % 2880) % 2880)
    path = Path(path)
    path.write_bytes(header.encode("ascii") + data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fits_rgb32(path: Path, cube_top_down: np.ndarray, *, comments: list[str] = (), extra_cards: list[str] = ()) -> str:
    """Write a TOP-DOWN float32 (3, H, W) cube as a bottom-up 32-bit RGB FITS (three distinct planes, BITPIX -32)."""
    cube = np.asarray(cube_top_down, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[0] != 3:
        raise ValueError("expected a (3, H, W) float cube")
    height, width = cube.shape[1:]
    cards = [_card("SIMPLE", True, "conforms to FITS standard"), _card("BITPIX", -32), _card("NAXIS", 3),
             _card("NAXIS1", width), _card("NAXIS2", height), _card("NAXIS3", 3), _card("EXTEND", True),
             _card("ROWORDER", "BOTTOM-UP", "standard FITS row order"), _card("CTYPE3", "RGB")]
    for card in extra_cards:
        if len(card) != 80:
            raise ValueError("extra cards must be exactly 80 characters")
        cards.append(card)
    for comment in comments:
        cards.append(("COMMENT " + comment)[:80].ljust(80))
    cards.append("END".ljust(80))
    header = "".join(cards)
    header += " " * ((2880 - len(header) % 2880) % 2880)
    data = np.ascontiguousarray(cube[:, ::-1, :]).astype(">f4").tobytes()
    data += b"\0" * ((2880 - len(data) % 2880) % 2880)
    path = Path(path)
    path.write_bytes(header.encode("ascii") + data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fits_minimal(path: Path) -> tuple[dict, np.ndarray]:
    """Read back a simple 16-bit RGB FITS written above; returns (header, TOP-DOWN uint16 plane 0)."""
    raw = Path(path).read_bytes()
    header = {}
    offset = 0
    while True:
        block = raw[offset:offset + 2880].decode("ascii")
        offset += 2880
        for i in range(0, 2880, 80):
            card = block[i:i + 80]
            key = card[:8].strip()
            if key == "END":
                break
            if card[8:10] == "= ":
                value = card[10:].split("/", 1)[0].strip()
                header[key] = value.strip("'").strip() if value.startswith("'") else value
        else:
            continue
        break
    width, height, planes = int(header["NAXIS1"]), int(header["NAXIS2"]), int(header.get("NAXIS3", 1))
    count = width * height * planes
    data = np.frombuffer(raw[offset:offset + count * 2], dtype=">i2").astype(np.int32) + int(header.get("BZERO", 0))
    plane = data.reshape(planes, height, width)[0]
    top_down = plane[::-1, :] if header.get("ROWORDER", "BOTTOM-UP") == "BOTTOM-UP" else plane
    return header, top_down.astype(np.uint16)


def generate_dataset(output_dir: Path, *, layout: dict | None = None, transforms: list[dict] | None = None,
                     stem: str = "syn") -> dict:
    """Write frames ``<stem>_00001.fit`` … and ``truth.json`` into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    layout = layout or default_layout()
    transforms = transforms or default_transforms(layout["width"], layout["height"])
    frames = []
    for spec in transforms:
        image, rendered = render_frame(layout, spec["matrix"], seed=spec["frame"])
        filename = f"{stem}_{spec['frame']:05d}.fit"
        digest = write_fits_rgb16(output_dir / filename, image,
                                  comments=[f"synthetic frame {spec['frame']}: {spec['description']}",
                                            "truth coordinates: TOP-DOWN zero-based pixel centres"])
        frames.append({"frame": spec["frame"], "filename": filename, "sha256": digest, "kind": spec["kind"],
                       "description": spec["description"], "frame_to_reference": spec["matrix"],
                       "rendered_stars": rendered})
    truth = {"generator_version": GENERATOR_VERSION, "coordinate_convention":
             {"row_order": "TOP-DOWN", "origin": "top_left_pixel_center", "units": "pixel"},
             "width": layout["width"], "height": layout["height"], "layout_seed": layout["seed"],
             "reference_stars": layout["stars"], "frames": frames,
             "fits": {"bitpix": 16, "bzero": 32768, "planes": 3, "roworder_keyword": "BOTTOM-UP"}}
    (output_dir / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return truth
