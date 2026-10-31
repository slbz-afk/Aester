"""Bounded display-sRGB layer stage, before the fixed GIMP recipe.

This is the numerical port of the frozen NL-1 teacher: stars, then diffuse
structure, then real transient core/glow. It is neither a separation model
nor a scene-linear flux operation. The host owns registry/hash verification,
publication and GIMP. This module only creates a new RGB16 TIFF.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import time

import numpy as np
import tifffile
from PIL import Image

from . import hashing
from .finishing_assets import ordinary_path, MAX_PIXELS
from .milkyway import _inward_feather
from .photographic import box_mean, luminance
from .star_reduction import bright_control
from .star_reduction_curve import reduction_delta

IMPLEMENTATION = "aester-layer-stage-teacher-1"
_SCALE = 4
_HALO = 32
_SHA = re.compile(r"[0-9a-f]{64}")
_PARAMS = {"bundle_id", "bundle_sha256", "star_strength", "milkyway_amount",
           "milkyway_mask_id", "meteor_core_extra", "meteor_glow_amount"}


class StageCancelled(RuntimeError):
    """The caller cancelled before the stage could be completed."""


def _check(cancel, deadline):
    if cancel is not None and cancel():
        raise StageCancelled("Layer stage cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Layer stage exceeded its deadline")


def _identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def _dimensions(record):
    if type(record) is not dict:
        raise ValueError("Expected a registered image or mask record")
    w, h = record.get("width"), record.get("height")
    if (type(w) is not int or type(h) is not int or not 1 <= w <= 100000
            or not 1 <= h <= 100000 or w * h > MAX_PIXELS):
        raise ValueError("Invalid layer dimensions")
    return h, w


class _RGB:
    """Open a fresh short-lived mapping for each copied strip."""
    def __init__(self, record, shape):
        if _dimensions(record) != shape or record.get("encoding") != "srgb16":
            raise ValueError("Layer shape or encoding differs from source")
        self.path = ordinary_path(record["path"])
        self.identity = _identity(self.path)
        with tifffile.TiffFile(self.path) as tf:
            p = tf.pages[0]
            if (len(tf.pages) != 1 or p.shape != (*shape, 3)
                    or p.dtype.kind != "u" or p.dtype.itemsize != 2
                    or int(p.photometric) != 2 or int(p.planarconfig) != 1
                    or int(p.compression) != 1 or not p.is_memmappable):
                raise ValueError("Layer engine needs mappable uncompressed RGB16 TIFF")
            if p.tags.get("Orientation") is not None and p.tags["Orientation"].value != 1:
                raise ValueError("Layer orientation must be TOP-DOWN")
            tag = p.tags.get(34675)
            if tag is None:
                raise ValueError("Layer ICC is missing")
            self.icc = bytes(tag.value)
        if hashlib.sha256(self.icc).hexdigest() != record.get("icc_sha256"):
            raise ValueError("Layer ICC differs from its record")

    def read(self, lo, hi):
        a = tifffile.memmap(self.path, mode="r")
        try:
            return np.array(a[lo:hi], dtype=np.uint16, copy=True)
        finally:
            a._mmap.close()

    def verify_stat(self):
        if _identity(self.path) != self.identity:
            raise ValueError("Layer input changed during execution")


class _Mask:
    """TIFF strips stream; PNG is decoded once at its admitted gray depth."""
    def __init__(self, record, source, *, binary=False):
        self.shape = _dimensions(source)
        if (_dimensions(record) != self.shape
                or record.get("source_sha256") != source["sha256"]
                or type(record.get("bits")) is not int or record["bits"] not in (8, 16)):
            raise ValueError("Mask shape, depth or source binding differs")
        self.path = ordinary_path(record["path"])
        self.identity = _identity(self.path)
        self.bits = record["bits"]
        self.maximum = (1 << self.bits) - 1
        self.binary = binary
        self.array = None
        if self.path.suffix.lower() in (".tif", ".tiff"):
            with tifffile.TiffFile(self.path) as tf:
                p = tf.pages[0]
                if (len(tf.pages) != 1 or p.shape != self.shape
                        or p.dtype.kind != "u" or p.dtype.itemsize * 8 != self.bits
                        or int(p.photometric) != 1 or not p.is_memmappable):
                    raise ValueError("Layer mask TIFF must be mappable literal grayscale")
                if p.tags.get(34675) is not None:
                    raise ValueError("Coverage mask must not have an ICC profile")
                if p.tags.get("Orientation") is not None and p.tags["Orientation"].value != 1:
                    raise ValueError("Coverage mask must be TOP-DOWN")
        elif self.path.suffix.lower() == ".png":
            with Image.open(self.path) as im:
                permitted = ("L",) if self.bits == 8 else ("I", "I;16", "I;16B", "I;16L")
                if (im.format != "PNG" or im.mode not in permitted or im.info.get("icc_profile")
                        or "transparency" in im.info or getattr(im, "n_frames", 1) != 1
                        or im.getexif().get(274, 1) != 1):
                    raise ValueError("Expected a single literal gray PNG mask")
                self.array = np.array(im)
            if (self.array.shape != self.shape or self.array.dtype.kind not in "ui"
                    or self.array.min() < 0 or self.array.max() > self.maximum):
                raise ValueError("Decoded PNG mask shape or range differs")
            self.array = self.array.astype(np.uint8 if self.bits == 8 else np.uint16)
        else:
            raise ValueError("Unsupported layer mask format")

    def read(self, lo, hi):
        if self.array is None:
            a = tifffile.memmap(self.path, mode="r")
            try:
                data = np.array(a[lo:hi], copy=True)
            finally:
                a._mmap.close()
        else:
            data = self.array[lo:hi]
        if self.binary:
            if np.any((data != 0) & (data != self.maximum)):
                raise ValueError("Pure sky and transient support must be exactly binary")
            return data == self.maximum
        return data.astype(np.float32) / self.maximum

    def verify_stat(self):
        if _identity(self.path) != self.identity:
            raise ValueError("Layer mask changed during execution")


def _parameters(params, source, bundle, masks):
    if type(params) is not dict or set(params) != _PARAMS:
        raise ValueError("Unexpected or missing layer parameters")
    if (type(bundle) is not dict or bundle.get("schema") != "aester-finishing-layer-bundle-1"
            or bundle.get("source") != source
            or params["bundle_id"] != bundle.get("id")
            or params["bundle_sha256"] != bundle.get("bundle_sha256")):
        raise ValueError("Layer bundle binding differs")
    if (type(params["bundle_id"]) is not str or not params["bundle_id"]
            or type(params["bundle_sha256"]) is not str
            or not _SHA.fullmatch(params["bundle_sha256"])):
        raise ValueError("Invalid layer bundle identity")
    for key, maximum in (("star_strength", 1), ("milkyway_amount", 3),
                         ("meteor_core_extra", 2), ("meteor_glow_amount", 4)):
        v = params[key]
        if type(v) not in (int, float) or not 0 <= v <= maximum or not math.isfinite(v):
            raise ValueError("Invalid " + key)
    if type(masks) is not dict:
        raise ValueError("Expected registered mask records")
    mid = params["milkyway_mask_id"]
    if mid is not None and (type(mid) is not str or not mid or mid not in masks):
        raise ValueError("Unknown Milky Way mask")
    if params["milkyway_amount"] > 0 and mid is None:
        raise ValueError("Milky Way enhancement needs its bound mask")


def _gaussian(a, sigma, check):
    # Exact frozen teacher convention, not GEGL's Gaussian.
    radius = max(1, int(round((np.sqrt(1 + 4 * sigma * sigma) - 1) / 2)))
    for _ in range(3):
        check()
        a = box_mean(a, radius)
    return a


def _prepare_field(diffuse, pure, shape, amount, check):
    h, w = shape
    wh, ww = (h + 3) // 4, (w + 3) // 4
    work_y = np.empty((wh, ww), np.float32)
    valid = np.empty_like(work_y)
    # Fixed preparation strips make the global field independent of tile_rows.
    for y in range(0, h, 64):
        check()
        end = min(h, y + 64)
        d = diffuse.read(y, end).astype(np.float32) / 65535
        p = pure.read(y, end).astype(np.float32)
        ly = luminance(d) * p
        pad = ((0, (-(end - y)) % 4), (0, (-w) % 4))
        # Odd dimensions replicate the last physical row/column, for both
        # weighted signal and validity; they do not invent excluded samples.
        ly = np.pad(ly, pad, mode="edge")
        p = np.pad(p, pad, mode="edge")
        rows = ly.shape[0] // 4
        work_y[y // 4:y // 4 + rows] = ly.reshape(rows, 4, ww, 4).mean((1, 3), dtype=np.float32)
        valid[y // 4:y // 4 + rows] = p.reshape(rows, 4, ww, 4).mean((1, 3), dtype=np.float32)
    def normalized(sigma):
        den = _gaussian(valid, sigma, check)
        num = _gaussian(work_y, sigma, check)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    small, large = normalized(2), normalized(64)
    base = np.divide(work_y, valid, out=np.zeros_like(work_y), where=valid > 0)
    field = amount * (small - large) / np.maximum(base, 1 / 65535)
    return np.clip(field, -.65, .85).astype(np.float32)


def _sample_field(a, lo, hi, width):
    xx = np.clip((np.arange(width, dtype=np.float32) + .5) / _SCALE - .5, 0, a.shape[1] - 1)
    yy = np.clip((np.arange(lo, hi, dtype=np.float32) + .5) / _SCALE - .5, 0, a.shape[0] - 1)
    ix, iy = xx.astype(np.int32), yy.astype(np.int32)
    jx, jy = np.minimum(ix + 1, a.shape[1] - 1), np.minimum(iy + 1, a.shape[0] - 1)
    tx, ty = xx - ix, yy - iy
    top = a[iy[:, None], ix[None, :]] * (1 - tx) + a[iy[:, None], jx[None, :]] * tx
    bot = a[jy[:, None], ix[None, :]] * (1 - tx) + a[jy[:, None], jx[None, :]] * tx
    return (top * (1 - ty[:, None]) + bot * ty[:, None]).astype(np.float32)


def _stellar_delta(base, source, diffuse, weight, strength, check):
    peak = np.maximum(base.astype(np.int32) - diffuse.astype(np.int32), 0).max(2).astype(np.uint16)
    control = bright_control(peak, 8)
    delta = np.zeros(base.shape, np.int32)
    for y in range(0, base.shape[0], 32):
        for x in range(0, base.shape[1], 1024):
            check()
            sl = slice(y, y + 32), slice(x, x + 1024)
            delta[sl] = reduction_delta(base[sl], source[sl], diffuse[sl], control[sl], weight[sl],
                                        strength=strength, bright_start=.10, bright_end=.45)
    return delta


def _bounded_joint_delta(source, base, diffuse, request):
    arrays = [a.astype(np.int32) for a in (source, base, diffuse)]
    up = np.minimum.reduce([np.maximum(65534 - a, 0) for a in arrays])
    down = np.minimum.reduce([np.maximum(a - 1, 0) for a in arrays])
    room = np.where(request >= 0, up, down)
    ratio = np.ones(request.shape, np.float32)
    np.divide(room, np.abs(request), out=ratio, where=request != 0)
    factor = np.minimum(ratio.min(2), 1).clip(0)
    delta = np.rint(request * factor[..., None]).astype(np.int32)
    return np.minimum(np.maximum(delta, -down), up)


def _meteor_addition(source, base, weight, core_extra, glow_amount, check):
    transient = source.astype(np.int32) - base.astype(np.int32)
    positive = np.maximum(transient, 0).astype(np.float32)
    u = ((positive.max(2) / 65535 - .002) / .018).clip(0, 1)
    signal = positive * (u * u * (3 - 2 * u))[..., None]
    if glow_amount:
        glow = np.stack([.65 * _gaussian(signal[..., c], 3, check)
                         + .35 * _gaussian(signal[..., c], 9, check) for c in range(3)], 2)
    else:
        glow = np.zeros_like(signal)
    addition = (core_extra * signal + glow_amount * glow) * weight[..., None]
    original = source.astype(np.int32)
    headroom = np.where(original < 65535, 65534 - original, 0).clip(0)
    ratio = np.ones(addition.shape, np.float32)
    np.divide(headroom, addition, out=ratio, where=addition > 0)
    cap = np.minimum(ratio.min(2), 1).clip(0)
    delta = np.minimum(np.rint(addition * cap[..., None]).astype(np.int32), headroom)
    return (original + delta).astype(np.uint16)


def render_layer_stage(source, bundle, params, masks, target, *, cancel=None, deadline=None, tile_rows=64):
    """Create *target* once; remove our incomplete target on any failure.

    ``deadline`` is an absolute ``time.monotonic()`` value. Pure sky is the
    registered exact binary product of the float-sky support decision; this
    engine does not infer it from a quantized soft sky mask. TIFF inputs must
    be mappable/uncompressed. Hash authenticity is the host registry's job.
    """
    started = time.monotonic()
    if type(tile_rows) is not int or not 1 <= tile_rows <= 512:
        raise ValueError("tile_rows must be an integer from 1 to 512")
    if cancel is not None and not callable(cancel):
        raise ValueError("cancel must be callable")
    if deadline is not None:
        try:
            valid_deadline = type(deadline) in (int, float) and math.isfinite(deadline)
        except OverflowError:
            valid_deadline = False
        if not valid_deadline:
            raise ValueError("deadline must be a finite monotonic time")
    check = lambda: _check(cancel, deadline)
    check()
    _parameters(params, source, bundle, masks)
    shape = _dimensions(source)
    h, w = shape
    readers = {"s": _RGB(source, shape), "b": _RGB(bundle["background"], shape),
               "d": _RGB(bundle["diffuse"], shape)}
    if any(r.icc != readers["s"].icc for r in readers.values()):
        raise ValueError("All paired layers must share the source ICC bytes")
    pure = _Mask(bundle["pure_sky"]["asset"], source, binary=True)
    support = _Mask(bundle["transient_support"]["asset"], source, binary=True)
    mid = params["milkyway_mask_id"]
    galaxy = _Mask(masks[mid], source) if mid is not None else None
    tracked = [*readers.values(), pure, support] + ([] if galaxy is None else [galaxy])
    for key in ("pure_sky", "transient_support"):
        entry = bundle[key]
        if entry["mask_id"] in masks and masks[entry["mask_id"]] != entry["asset"]:
            raise ValueError("Bundle mask differs from current registered mask")
    check()
    requested = Path(target).expanduser().absolute()
    parent = ordinary_path(requested.parent, directory=True)
    target = parent / requested.name
    if target.exists() or target.is_symlink():
        raise FileExistsError("Layer target must be a new file")
    if target.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("Layer target must be TIFF")
    field = (_prepare_field(readers["d"], pure, shape, float(params["milkyway_amount"]), check)
             if params["milkyway_amount"] else None)
    metrics = {key: 0 for key in ("changed_pixels", "non_pure_sky_changed_components",
               "new_zero_components", "new_max_components", "near_max_ge_65000_components",
               "star_T_mismatch_components", "galaxy_R_mismatch_components", "galaxy_T_mismatch_components",
               "max_abs_delta")}
    pixel_hash = hashlib.sha256()
    created = False
    try:
        check()
        with target.open("xb"):
            created = True
        icc = readers["s"].icc
        with hashing.writing(target):
            out = tifffile.memmap(target, shape=(h, w, 3), dtype=np.uint16, photometric="rgb",
                                  metadata={"implementation": IMPLEMENTATION, "source_sha256": source["sha256"],
                                            "bundle_sha256": params["bundle_sha256"]},
                                  extratags=[(34675, "B", len(icc), icc, False)])
            out._mmap.close()
        for y in range(0, h, tile_rows):
            check()
            end = min(h, y + tile_rows)
            lo, hi = max(0, y - _HALO), min(h, end + _HALO)
            cut = slice(y - lo, end - lo)
            a = {k: r.read(lo, hi) for k, r in readers.items()}
            sky = pure.read(lo, hi)
            transient = a["s"].astype(np.int32) - a["b"].astype(np.int32)
            residual = a["b"].astype(np.int32) - a["d"].astype(np.int32)
            if np.any(transient[~support.read(lo, hi)]) or np.any(transient[~sky]):
                raise ValueError("Transient difference exceeds its sky/support contract")
            if np.any(residual[~sky]):
                raise ValueError("Star residual exceeds pure sky")
            weight = _inward_feather(~sky, 24)
            removed = (_stellar_delta(a["b"], a["s"], a["d"], weight, float(params["star_strength"]), check)
                       if params["star_strength"] else np.zeros(a["s"].shape, np.int32))
            s = (a["s"].astype(np.int32) - removed).astype(np.uint16)
            b = (a["b"].astype(np.int32) - removed).astype(np.uint16)
            metrics["star_T_mismatch_components"] += int(np.count_nonzero(
                (s.astype(np.int32) - b.astype(np.int32) - transient)[cut]))
            if field is not None:
                gain = _sample_field(field, lo, hi, w) * galaxy.read(lo, hi) * weight
                delta = _bounded_joint_delta(s, b, a["d"], a["d"].astype(np.float32) * gain[..., None])
                sn = (s.astype(np.int32) + delta).astype(np.uint16)
                bn = (b.astype(np.int32) + delta).astype(np.uint16)
                dn = (a["d"].astype(np.int32) + delta).astype(np.uint16)
                metrics["galaxy_R_mismatch_components"] += int(np.count_nonzero(
                    (bn.astype(np.int32) - dn.astype(np.int32) - (b.astype(np.int32) - a["d"].astype(np.int32)))[cut]))
                metrics["galaxy_T_mismatch_components"] += int(np.count_nonzero(
                    (sn.astype(np.int32) - bn.astype(np.int32) - transient)[cut]))
                s, b = sn, bn
            if params["meteor_core_extra"] or params["meteor_glow_amount"]:
                s = _meteor_addition(s, b, weight, float(params["meteor_core_extra"]),
                                     float(params["meteor_glow_amount"]), check)
            final = np.ascontiguousarray(s[cut])
            original = a["s"][cut]
            diff = final.astype(np.int32) - original.astype(np.int32)
            metrics["changed_pixels"] += int(np.any(diff != 0, 2).sum())
            metrics["non_pure_sky_changed_components"] += int(np.count_nonzero(diff[~sky[cut]]))
            metrics["new_zero_components"] += int(((final == 0) & (original > 0)).sum())
            metrics["new_max_components"] += int(((final == 65535) & (original < 65535)).sum())
            metrics["near_max_ge_65000_components"] += int((final >= 65000).sum())
            metrics["max_abs_delta"] = max(metrics["max_abs_delta"], int(np.abs(diff).max()))
            pixel_hash.update(final.tobytes())
            check()
            # A write through a memory map keeps the file's modification time.
            with hashing.writing(target):
                output = tifffile.memmap(target, mode="r+")
                try:
                    output[y:end] = final
                    output.flush()
                finally:
                    output._mmap.close()
        check()
        for reader in tracked:
            reader.verify_stat()
        for key in ("non_pure_sky_changed_components", "star_T_mismatch_components",
                    "galaxy_R_mismatch_components", "galaxy_T_mismatch_components",
                    "new_zero_components", "new_max_components"):
            if metrics[key]:
                raise RuntimeError("Layer stage violated " + key)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return {"path": str(target), "pixel_sha256": pixel_hash.hexdigest(), "metrics": metrics,
            "seconds": time.monotonic() - started, "implementation": IMPLEMENTATION,
            "parameters": dict(params), "processing": {
                "domain": "encoded-sRGB16-display-differences",
                "order": ["stars", "milkyway", "meteor"],
                "work_scale": 4, "odd_size_padding": "edge-repeat-bottom-right-before-area4",
                "structure_sigma_source_proxies": [8, 256], "structure_gain_limits": [-.65, .85],
                "blur": "three-equal-clamped-boxes", "field_sampling": "bilinear-pixel-centers",
                "halo_source_pixels": 32, "sky_inward_source_pixels": 24,
                "bright_radius_source_pixels": 8, "bright_ramp": [.10, .45],
                "meteor_confidence_ramp": [.002, .02], "glow_sigma_source_proxies": [3, 9],
                "glow_weights": [.65, .35], "glow_max_support_source_pixels": 27,
                "tile_rows": tile_rows, "input_size_mtime_identity_unchanged": True}}
