"""Synthetic FITS frames for the P1c trail sites (T-3m / T-3 / T-2 tests): a small FITS writer for every format the
product reads (BITPIX 16 with BZERO 32768 or -32; ROWORDER BOTTOM-UP or TOP-DOWN), written plane by plane and row
block by row block so that a 45.7 MP frame never needs a whole cube in memory, and the T-2 worst case: a bright trail
across the whole frame's diagonal plus a manual mark's corridor along the other diagonal (prereg §3.4: the child's
code-derived worst peak).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

Z8_SHAPE = (5520, 8280)                  # 45.7 MP (height, width): the planner's reference frame


def _card(key: str, value) -> str:
    if isinstance(value, bool):
        text = "T" if value else "F"
    elif isinstance(value, str):
        text = f"'{value}'"
    else:
        text = str(value)
    return f"{key:<8}= {text:>20}".ljust(80)[:80]


def header_bytes(height: int, width: int, *, bitpix: int, roworder: str) -> bytes:
    cards = [_card("SIMPLE", True), _card("BITPIX", bitpix), _card("NAXIS", 3), _card("NAXIS1", width),
             _card("NAXIS2", height), _card("NAXIS3", 3)]
    if bitpix == 16:
        cards += [_card("BZERO", 32768), _card("BSCALE", 1)]
    cards += [_card("ROWORDER", roworder), "END".ljust(80)]
    text = "".join(cards)
    text += " " * ((2880 - len(text) % 2880) % 2880)
    return text.encode("ascii")


def write_fits(path, height: int, width: int, rows_of, *, bitpix: int = -32, roworder: str = "BOTTOM-UP",
               block: int = 512) -> Path:
    """Write a 3-plane FITS whose TOP-DOWN rows ``y0:y1`` of plane ``c`` are ``rows_of(c, y0, y1)`` (float values in
    the [0, 1] scale of the float frames; 16-bit files store round(v * 65535) - 32768, so 0 stays 0)."""
    path = Path(path)
    dtype = ">i2" if bitpix == 16 else ">f4"
    itemsize = 2 if bitpix == 16 else 4
    with path.open("wb") as stream:
        stream.write(header_bytes(height, width, bitpix=bitpix, roworder=roworder))
        for plane in range(3):
            starts = list(range(0, height, block))
            if roworder == "BOTTOM-UP":                 # storage row 0 is the bottom (top-down row height - 1)
                starts = starts[::-1]
            for y0 in starts:
                y1 = min(height, y0 + block)
                values = np.asarray(rows_of(plane, y0, y1), dtype=np.float64)
                if bitpix == 16:
                    data = (np.clip(np.rint(values * 65535.0), 0, 65535) - 32768).astype(dtype)
                else:
                    data = values.astype(dtype)
                if roworder == "BOTTOM-UP":
                    data = data[::-1]
                stream.write(np.ascontiguousarray(data).tobytes())
        size = 3 * height * width * itemsize
        stream.write(b"\0" * ((2880 - size % 2880) % 2880))
    return path


def _line_distance(xx, yy, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    return np.abs((xx - x0) * dy - (yy - y0) * dx) / math.hypot(dx, dy)


def worst_case_pair(folder, shape=Z8_SHAPE, *, seed: int = 20261006, name: str = "light_00001") -> dict:
    """The T-2 peak test's frame and master (prereg §3.4): a bright 3 px trail along the main diagonal across the whole
    frame, found by the detection (an automatic segment) and also marked by hand (a manual mark's corridor, half width
    30 px, along the same trail): both measurements read the whole frame (the measurement box spans the trail), the
    child's largest arrays. (A second trail crossing it would join both into one crooked component that the detection
    rejects.) Returns the ``_scan_frame`` keyword arguments (the mark in grid coordinates)."""
    folder = Path(folder)
    height, width = shape
    background = (0.020, 0.024, 0.018)

    def master_rows(plane, y0, y1):
        rng = np.random.default_rng([seed, 1, plane, y0])
        return background[plane] + np.float32(0.0004) * rng.standard_normal((y1 - y0, width), dtype=np.float32)

    profiles = {}                     # the trail's cross-profile per row block, shared by the three planes

    def frame_rows(plane, y0, y1):
        if y0 not in profiles:
            yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
            main = _line_distance(xx, yy, 0.0, 0.0, width - 1.0, height - 1.0)
            profiles[y0] = np.exp(-main ** 2 / np.float32(2 * 1.5 ** 2))
        rng = np.random.default_rng([seed, 2, plane, y0])
        return (master_rows(plane, y0, y1) + np.float32(0.0004) * rng.standard_normal((y1 - y0, width), dtype=np.float32)
                + np.float32(0.6 * background[plane]) * profiles[y0])

    master = write_fits(folder / "master.fit", height, width, master_rows)
    frame = write_fits(folder / f"r_{name}.fit", height, width, frame_rows)
    mark = {"stem": name, "x0": 0.0, "y0": 0.0, "x1": width - 1.0, "y1": height - 1.0, "half_width_px": 30}
    from starstack import transient_trails as tt
    return {"frame_path": str(frame), "master_path": str(master), "detection": dict(tt.DEFAULT_DETECTION),
            "grid_marks": [mark], "stats_step": 16}
