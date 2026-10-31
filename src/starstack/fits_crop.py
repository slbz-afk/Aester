"""Windowed access to Siril's 16-bit RGB FITS frames and pixel-level checks.

Reads only the rows a crop needs (memory-mapped), returns crops in the
diagnostics convention (TOP-DOWN rows, zero-based pixel centres), warps a
frame crop into reference coordinates through a source->reference matrix
given in that same convention, and measures star centroid offsets between
two crops.  Pure NumPy; no astropy dependency, so it runs wherever the
sequence files are (the Windows product venv or a Linux VM).

Only simple primary-HDU files are supported: BITPIX 16 with BZERO 32768
(unsigned) or BITPIX -32, NAXIS 2 or 3, ROWORDER BOTTOM-UP (default) or
TOP-DOWN.  Anything else raises ``ValueError`` rather than guessing.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

__all__ = ["FITS_CROP_VERSION", "read_header", "plane_memmap", "crop_top_down", "warp_to_reference",
           "measure_star_offsets", "measure_star_shapes", "asinh_stretch", "write_png_gray", "stack_two"]

FITS_CROP_VERSION = "1.0.0"


def read_header(path) -> tuple[dict, int]:
    """Parse the primary header; returns (cards as dict, data offset in bytes)."""
    header = {}
    offset = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(2880)
            if len(block) < 2880:
                raise ValueError("truncated FITS header")
            offset += 2880
            done = False
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    done = True
                    break
                if card[8:10] == "= ":
                    value = card[10:].split("/", 1)[0].strip()
                    if value.startswith("'"):
                        value = value.strip().strip("'").strip()
                    header[key] = value
            if done:
                break
    return header, offset


def _geometry(header: dict) -> tuple[int, int, int, int, str]:
    bitpix = int(header["BITPIX"])
    naxis = int(header["NAXIS"])
    if naxis not in (2, 3):
        raise ValueError(f"unsupported NAXIS {naxis}")
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    planes = int(header.get("NAXIS3", 1)) if naxis == 3 else 1
    roworder = header.get("ROWORDER", "BOTTOM-UP").upper()
    if roworder not in ("BOTTOM-UP", "TOP-DOWN"):
        raise ValueError(f"unsupported ROWORDER {roworder!r}")
    if bitpix == 16:
        if float(header.get("BZERO", 0)) != 32768.0 or float(header.get("BSCALE", 1)) != 1.0:
            raise ValueError("BITPIX 16 files are only supported as unsigned (BZERO 32768, BSCALE 1)")
    elif bitpix != -32:
        raise ValueError(f"unsupported BITPIX {bitpix}")
    return width, height, planes, bitpix, roworder


def plane_memmap(path, plane: int = 1):
    """Memory-mapped view of one plane in *storage* row order plus the header."""
    header, offset = read_header(path)
    width, height, planes, bitpix, roworder = _geometry(header)
    if not 0 <= plane < planes:
        raise ValueError(f"plane {plane} outside 0..{planes - 1}")
    dtype = ">i2" if bitpix == 16 else ">f4"
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(planes, height, width))
    return mm[plane], header, roworder


def crop_top_down(path, plane: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """Crop ``[y0:y1, x0:x1]`` of a plane in TOP-DOWN display coordinates as float32."""
    view, header, roworder = plane_memmap(path, plane)
    height, width = view.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop box outside the image")
    if roworder == "BOTTOM-UP":
        rows = view[height - y1:height - y0, x0:x1][::-1, :]
    else:
        rows = view[y0:y1, x0:x1]
    data = np.asarray(rows, dtype=np.float32)
    if int(header["BITPIX"]) == 16:
        data = data + 32768.0
    return np.ascontiguousarray(data)


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample ``image`` at zero-based pixel-centre coordinates; NaN outside."""
    h, w = image.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    xi, yi = x0[inside], y0[inside]
    fxi, fyi = fx[inside], fy[inside]
    out[inside] = ((1 - fxi) * (1 - fyi) * image[yi, xi] + fxi * (1 - fyi) * image[yi, xi + 1]
                   + (1 - fxi) * fyi * image[yi + 1, xi] + fxi * fyi * image[yi + 1, xi + 1])
    return out


def warp_to_reference(frame_path, plane: int, h_frame_to_reference_td, box, *, margin: int = 16) -> dict:
    """Resample a frame into reference coordinates over ``box = (x0, x1, y0, y1)``.

    ``h_frame_to_reference_td`` is the source->reference matrix in the
    diagnostics convention.  Only the frame rows the box maps to are read.
    Returns the warped crop (NaN where the frame does not cover) and the
    source window used.
    """
    x0, x1, y0, y1 = box
    h = np.asarray(h_frame_to_reference_td, dtype=np.float64)
    h_inv = np.linalg.inv(h)
    view, header, roworder = plane_memmap(frame_path, plane)
    height, width = view.shape
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    src = h_inv @ pts
    sx = (src[0] / src[2]).reshape(xs.shape)
    sy = (src[1] / src[2]).reshape(xs.shape)
    wx0 = max(0, int(np.floor(np.nanmin(sx))) - margin)
    wx1 = min(width, int(np.ceil(np.nanmax(sx))) + margin + 1)
    wy0 = max(0, int(np.floor(np.nanmin(sy))) - margin)
    wy1 = min(height, int(np.ceil(np.nanmax(sy))) + margin + 1)
    if wx1 <= wx0 or wy1 <= wy0:
        raise ValueError("the box maps entirely outside the frame")
    window = crop_top_down(frame_path, plane, wx0, wx1, wy0, wy1)
    warped = _bilinear(window, sx - wx0, sy - wy0)
    return {"crop": warped.astype(np.float32), "source_window": [wx0, wx1, wy0, wy1],
            "coverage_fraction": float(np.isfinite(warped).mean())}


def _centroid(crop: np.ndarray, x: float, y: float, radius: int, iterations: int = 4):
    """Iterative background-subtracted centroid in a (2r+1)^2 window; None if it leaves the crop."""
    h, w = crop.shape
    cx, cy = float(x), float(y)
    for _ in range(iterations):
        ix, iy = int(round(cx)), int(round(cy))
        if not (radius + 2 <= ix < w - radius - 2 and radius + 2 <= iy < h - radius - 2):
            return None
        win = crop[iy - radius - 2:iy + radius + 3, ix - radius - 2:ix + radius + 3]
        if not np.isfinite(win).all():
            return None
        border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
        bkg = float(np.median(border))
        inner = crop[iy - radius:iy + radius + 1, ix - radius:ix + radius + 1] - bkg
        inner = np.clip(inner, 0, None)
        total = float(inner.sum())
        if total <= 0:
            return None
        yy, xx = np.mgrid[iy - radius:iy + radius + 1, ix - radius:ix + radius + 1]
        nx, ny = float((inner * xx).sum() / total), float((inner * yy).sum() / total)
        if abs(nx - cx) < 1e-3 and abs(ny - cy) < 1e-3:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny
    peak = float(np.nanmax(crop[max(0, int(cy) - 1):int(cy) + 2, max(0, int(cx) - 1):int(cx) + 2]))
    return cx, cy, bkg, peak


def measure_star_offsets(reference_crop: np.ndarray, other_crop: np.ndarray, star_xy, *, radius: int = 4,
                         min_peak_over_background: float = 50.0) -> dict:
    """Centroid each star in both crops (same crop coordinates) and report other - reference.

    ``star_xy`` are crop coordinates (TOP-DOWN, zero-based centres) of stars
    to measure, typically the reference star list inside the box.  Stars whose
    windows leave either crop, contain NaN, or are too faint are listed as
    skipped rather than silently dropped.
    """
    offsets = []
    skipped = []
    for k, (x, y) in enumerate(star_xy):
        ref = _centroid(reference_crop, x, y, radius)
        if ref is None or ref[3] - ref[2] < min_peak_over_background:
            skipped.append({"index": k, "reason": "reference window invalid or too faint"})
            continue
        other = _centroid(other_crop, ref[0], ref[1], radius)
        if other is None or other[3] - other[2] < min_peak_over_background:
            skipped.append({"index": k, "reason": "other window invalid or too faint"})
            continue
        offsets.append({"index": k, "reference_xy": [ref[0], ref[1]], "other_xy": [other[0], other[1]],
                        "dx_px": other[0] - ref[0], "dy_px": other[1] - ref[1],
                        "distance_px": float(np.hypot(other[0] - ref[0], other[1] - ref[1]))})
    d = np.array([o["distance_px"] for o in offsets]) if offsets else np.zeros(0)
    summary = {"count": int(len(offsets)), "skipped": len(skipped),
               "rms_px": float(np.sqrt(np.mean(d ** 2))) if len(d) else None,
               "median_px": float(np.median(d)) if len(d) else None,
               "max_px": float(d.max()) if len(d) else None,
               "mean_dx_px": float(np.mean([o["dx_px"] for o in offsets])) if offsets else None,
               "mean_dy_px": float(np.mean([o["dy_px"] for o in offsets])) if offsets else None}
    return {"summary": summary, "offsets": offsets, "skipped": skipped, "centroid_radius_px": radius}


def measure_star_shapes(crop: np.ndarray, star_xy, *, radius: int = 6, min_peak_over_background: float = 50.0) -> dict:
    """Background-subtracted second moments of each star: major/minor sigma, elongation, angle.

    The same measurement on a single frame and on a two-frame stack separates
    optical star shape (present in the single frame) from registration smear
    (extra elongation in the stack).  Windows are (2r+1)^2 around the centroid.
    """
    shapes = []
    for k, (x, y) in enumerate(star_xy):
        c = _centroid(crop, x, y, min(radius, 4))
        if c is None or c[3] - c[2] < min_peak_over_background:
            continue
        cx, cy, bkg, _ = c
        ix, iy = int(round(cx)), int(round(cy))
        h, w = crop.shape
        if not (radius <= ix < w - radius and radius <= iy < h - radius):
            continue
        win = crop[iy - radius:iy + radius + 1, ix - radius:ix + radius + 1] - bkg
        if not np.isfinite(win).all():
            continue
        win = np.clip(win, 0, None)
        total = float(win.sum())
        if total <= 0:
            continue
        yy, xx = np.mgrid[iy - radius:iy + radius + 1, ix - radius:ix + radius + 1]
        mx, my = float((win * xx).sum() / total), float((win * yy).sum() / total)
        cxx = float((win * (xx - mx) ** 2).sum() / total)
        cyy = float((win * (yy - my) ** 2).sum() / total)
        cxy = float((win * (xx - mx) * (yy - my)).sum() / total)
        tr, det = cxx + cyy, cxx * cyy - cxy * cxy
        disc = max(0.0, tr * tr / 4 - det)
        l1, l2 = tr / 2 + np.sqrt(disc), max(1e-9, tr / 2 - np.sqrt(disc))
        shapes.append({"index": k, "xy": [mx, my], "sigma_major_px": float(np.sqrt(l1)), "sigma_minor_px": float(np.sqrt(l2)),
                       "elongation": float(np.sqrt(l1 / l2)), "angle_deg": float(np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy))),
                       "flux": total})
    if not shapes:
        return {"count": 0}
    maj = np.array([s["sigma_major_px"] for s in shapes])
    mn = np.array([s["sigma_minor_px"] for s in shapes])
    el = np.array([s["elongation"] for s in shapes])
    return {"count": len(shapes), "median_sigma_major_px": float(np.median(maj)), "median_sigma_minor_px": float(np.median(mn)),
            "median_elongation": float(np.median(el)), "mean_elongation": float(el.mean()),
            "median_fwhm_major_px": float(2.3548 * np.median(maj)), "stars": shapes, "moment_radius_px": radius}


def asinh_stretch(crop: np.ndarray, *, low_percentile: float = 5.0, high_percentile: float = 99.7,
                  softness: float = 20.0) -> np.ndarray:
    """Fixed, documented display stretch to 8-bit for crops (NaN -> 0)."""
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        return np.zeros(crop.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low_percentile, high_percentile])
    hi = max(hi, lo + 1.0)
    x = np.clip((np.nan_to_num(crop, nan=lo) - lo) / (hi - lo), 0, None)
    y = np.arcsinh(softness * x) / np.arcsinh(softness)
    return np.clip(np.round(255 * y), 0, 255).astype(np.uint8)


def write_png_gray(path, image8: np.ndarray) -> None:
    """Minimal 8-bit grayscale PNG writer (no PIL dependency)."""
    image8 = np.ascontiguousarray(image8, dtype=np.uint8)
    h, w = image8.shape
    raw = b"".join(b"\x00" + image8[r].tobytes() for r in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (len(payload).to_bytes(4, "big") + tag + payload
                + (zlib.crc32(tag + payload) & 0xFFFFFFFF).to_bytes(4, "big"))

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9))
                           + chunk(b"IEND", b""))


def stack_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean of two crops where both are finite (a two-frame stack shows misregistration as elongation)."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b)
    out[ok] = 0.5 * (a[ok] + b[ok])
    return out
