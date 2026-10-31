"""Per-frame sky validity: keep the ground out of the sky stack (S3, RUN_LOG §24–25).

The ground is static on the sensor while the sky is registered, so after
registration every frame's ground drifts into the sky region of the
reference grid; where it lands it tints the stack (the sand of the Z8 night
is 70–83 % as bright as the sky in green with R/G 1.10–1.22× the sky's, so
the band is a colour shift the winsorized rejection cannot see).  Siril
1.4.4 treats pixels that are exactly zero as "no data" in stacking — proven
on synthetic frames (``siril_nodata_proof``: a zero majority leaves the sky
value intact, a single valid frame survives, ``-norm=addscale`` still
targets the reference, and ``seqapplyreg`` keeps exact zeros exact with a
one-pixel mixed rim).  So the transferable mechanism is:

1. ``static_validity``: the job's sky mask (any pixel with ground weight is
   invalid) dilated by an empirical margin, the same for every frame
   of a fixed-tripod sequence — the first *validity source*; per-frame
   sources (moving foregrounds, automatic sky detection) plug in here later.
2. ``apply_validity_to_fits``: the invalid pixels of every converted frame
   are set to zero in place before the probe registration and the export.
3. ``coverage``: how many frames observe each reference pixel, from the
   job's own registration (model + per-frame matrices), so the compose step
   can tell "sky observed by fewer frames" (keep) from "sky observed by no
   frame" (fall back to the ground master) and the job can report it.

Everything is in TOP-DOWN pixel coordinates unless a name says otherwise;
FITS storage is bottom-up and is flipped at the file boundary only.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import fits_crop, frame_pool, hashing
from .registration_models import apply_matrix

SKY_VALIDITY_VERSION = "1.1.1"
VALIDITY_RECORD = "sky_validity.json"
COVERAGE_PNG = "sky_coverage.png"
DEFAULT_DILATION_PX = 3          # seqapplyreg's clamped interpolation mixes one pixel at a zero edge (proof: 0.508 then 0); margin for rotation/undistortion
DEFAULT_COVERAGE_STEP = 4
# coverage_at on the frame pool (efficiency P1b-1p, G-6 pilot): a unit's peak is 218 MB plus a 10.9 MB result at
# 45.7 MP with step 4 (p1b design pool exp6), and the real F48 calibration gains little past 8 threads (exp5).
# core._write_coverage hands both to frame_pool.default_workers, the unit scaled by the frame's pixels over
# COVERAGE_UNIT_PIXELS (never below COVERAGE_UNIT_BYTES).
COVERAGE_WORKER_CAP = 8
COVERAGE_UNIT_BYTES = 230 << 20
COVERAGE_UNIT_PIXELS = 5520 * 8280   # the 45.7 MP frame COVERAGE_UNIT_BYTES was measured on


def load_mask_top_down(path: str | Path, width: int, height: int) -> np.ndarray:
    """The job's sky mask as float32 0..1 (1 = sky), top-down, checked against the frame size."""
    from PIL import Image
    with Image.open(path) as img:
        if img.size != (width, height):
            raise ValueError(f"sky mask is {img.size[0]}x{img.size[1]}, frames are {width}x{height}")
        mask = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return mask


def dilate(invalid: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1)² square (Chebyshev radius r); pure NumPy."""
    out = invalid.copy()
    if radius <= 0:
        return out
    for axis in (1, 0):
        src = out.copy()
        for s in range(1, radius + 1):
            if axis == 1:
                out[:, s:] |= src[:, :-s]
                out[:, :-s] |= src[:, s:]
            else:
                out[s:, :] |= src[:-s, :]
                out[:-s, :] |= src[s:, :]
    return out


def static_validity(mask_top_down: np.ndarray, *, dilation_px: int = DEFAULT_DILATION_PX) -> np.ndarray:
    """True where a frame's pixel is valid sky: full sky weight in the mask, farther than ``dilation_px`` from any ground weight."""
    invalid = mask_top_down < 1.0
    return ~dilate(invalid, dilation_px)


class StaticMaskSource:
    """One validity for every frame of a fixed-tripod sequence: the job's sky mask, dilated."""

    name = "job_mask_static"

    def __init__(self, mask_top_down: np.ndarray, *, dilation_px: int = DEFAULT_DILATION_PX):
        self.mask = mask_top_down
        self.dilation_px = int(dilation_px)
        self.validity = static_validity(mask_top_down, dilation_px=dilation_px)

    def for_frame(self, stem: str) -> np.ndarray:
        return self.validity

    def by_frame(self, stems) -> dict:
        return {stem: self.validity for stem in stems}

    def describe(self) -> dict:
        return {"source": self.name, "dilation_px": self.dilation_px, "invalid_fraction": float((~self.validity).mean())}


class PerFrameMaskSource:
    """A mask per frame (moving foreground, clouds, automatic sky detection later): ``files`` maps a frame stem to a PNG
    in the frame's native geometry; frames without their own file use the static mask.  Each mask is dilated like the
    static one.  This is the entry point later validity sources plug into — the rest of the flow only asks
    ``for_frame``/``by_frame``."""

    name = "job_mask_per_frame"

    def __init__(self, static: StaticMaskSource, files: dict, width: int, height: int):
        self.static = static
        self.files = {stem: Path(path) for stem, path in files.items()}
        self.width, self.height = int(width), int(height)
        self.dilation_px = static.dilation_px
        self._cache = {}

    def for_frame(self, stem: str) -> np.ndarray:
        if stem not in self.files:
            return self.static.validity
        if stem not in self._cache:
            mask = load_mask_top_down(self.files[stem], self.width, self.height)
            self._cache[stem] = static_validity(np.minimum(mask, self.static.mask), dilation_px=self.dilation_px)
        return self._cache[stem]

    def by_frame(self, stems) -> dict:
        return {stem: self.for_frame(stem) for stem in stems}

    def describe(self) -> dict:
        return {"source": self.name, "dilation_px": self.dilation_px, "invalid_fraction": float((~self.static.validity).mean()),
                "frames_with_own_mask": sorted(self.files), "own_mask_count": len(self.files)}


FUSED_SOURCE = "sky_validity.apply_validity_to_fits"
# Chunks queued to a digest thread before the pass waits for it (bounds the memory held for hashing).
_DIGEST_QUEUE = 4


def apply_validity_to_fits(path: str | Path, validity_top_down: np.ndarray, *, check=None, rows: int = 256) -> dict:
    """Set every invalid pixel of a Siril FITS frame (16-bit RGB or 32-bit float RGB, bottom-up) to zero, in place.

    Inside a hash run scope that is not "off" this is one streaming pass over the file
    (docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md): the resulting bytes are exactly those of
    the former memory-mapped write, and the SHA-256 of the result, computed from the bytes
    written, is recorded with the ledger as written by this process (so the next hash of the
    frame is a ledger hit, or a verified one in shadow mode).  Outside a run scope, or "off",
    it is the former memory-mapped write.  ``check`` runs before every row block (of every
    plane, in the streaming pass).
    """
    return _apply_validity(path, validity_top_down, check=check, rows=rows, before=False)[0]


def apply_validity_with_digests(path: str | Path, validity_top_down: np.ndarray, *, check=None,
                                rows: int = 256) -> tuple[dict, str, str | None]:
    """:func:`apply_validity_to_fits` plus the file's SHA-256 before and after, from the same single pass.

    Returns ``(stats, before_sha256, after_sha256)``.  The file is read once, in order
    (header, data, padding): the digest before is taken of the bytes read, the digest after
    (recorded as written, as in :func:`apply_validity_to_fits`) of the same bytes with the
    invalid samples zeroed, which are the bytes written back.  Outside a run scope, or "off",
    and for a frame whose data region runs past the end of the file (which the memory-mapped
    write extended with zeros), the file is hashed for real and then masked by the former
    memory-mapped write, and ``after_sha256`` is None.
    """
    return _apply_validity(path, validity_top_down, check=check, rows=rows, before=True)


def _apply_validity(path, validity_top_down, *, check, rows, before):
    path = Path(path)
    header, offset = fits_crop.read_header(path)
    width, height, planes, bitpix, roworder = fits_crop._geometry(header)
    if validity_top_down.shape != (height, width):
        raise ValueError("validity mask does not match the frame")
    valid_storage = validity_top_down[::-1, :] if roworder == "BOTTOM-UP" else validity_top_down
    if bitpix == 16:
        dtype, zero = ">i2", np.int16(-int(float(header.get("BZERO", 0))))   # raw value that reads back as 0 after BZERO
    elif bitpix == -32:
        dtype, zero = ">f4", np.float32(0.0)
    else:
        raise ValueError(f"unsupported BITPIX {bitpix}")
    data_bytes = planes * height * width * np.dtype(dtype).itemsize
    with hashing.writing(path) as block:
        streamable = False
        if hashing.digest_writes():     # the fused pass only where a ledger records the digest of the result
            # The same open as numpy's memmap (so a missing or locked file fails exactly as before).
            with open(os.fspath(path), "r+b") as handle:
                streamable = data_bytes > 0 and os.fstat(handle.fileno()).st_size >= offset + data_bytes
                if streamable:
                    stats, before_sha, after_sha = _apply_streaming(
                        handle, valid_storage, dtype, zero, offset, planes, height, width, rows, check, before)
        if not streamable:
            before_sha = hashing.sha256_file(path, source=FUSED_SOURCE, trust=hashing.TRUST_REAL) if before else None
            stats = _apply_memmap(path, valid_storage, dtype, zero, offset, planes, height, width, rows, check)
            return {"file": path.name, **stats}, before_sha, None
        block.written(after_sha, FUSED_SOURCE)
    return {"file": path.name, **stats}, before_sha, after_sha


def _apply_memmap(path, valid_storage, dtype, zero, offset, planes, height, width, rows, check):
    """The former memory-mapped write (outside a ledger, and for a data region past the end of the file)."""
    mm = np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=(planes, height, width))
    zeroed = 0
    try:
        for y in range(0, height, rows):
            if check is not None:
                check()
            block = ~valid_storage[y:y + rows, :]
            n = int(block.sum())
            if n:
                sub = mm[:, y:y + rows, :]
                sub[:, block] = zero
                zeroed += n
        mm.flush()
    finally:
        del mm
    return {"zeroed_pixels": zeroed, "invalid_fraction": float(zeroed / (width * height))}


class _Digest:
    """SHA-256 fed in order; for large files on a helper thread, so hashing overlaps the pass's reads and writes."""

    def __init__(self, threaded):
        self._hash = hashlib.sha256()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aester-validity-hash") if threaded else None
        self._pending = deque()
        self.nbytes = 0

    def update(self, data):
        self.nbytes += len(data)
        if self._pool is None:
            self._hash.update(data)
            return
        while len(self._pending) >= _DIGEST_QUEUE:
            self._pending.popleft().result()
        self._pending.append(self._pool.submit(self._hash.update, data))

    def hexdigest(self):
        while self._pending:
            self._pending.popleft().result()
        return self._hash.hexdigest()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)


def _read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise EOFError(f"FITS file ended inside its data: {getattr(handle, 'name', '')}")
    return data


def _apply_streaming(handle, valid_storage, dtype, zero, offset, planes, height, width, rows, check, before):
    """One pass in file order (plane-major rows); a row block holding an invalid pixel is written back.

    Every byte of the file is read once.  The digest after is of the bytes as they are
    written back; the digest before (when asked) of the bytes as read.
    """
    row_bytes = width * np.dtype(dtype).itemsize
    threaded = offset + planes * height * row_bytes >= hashing.PARALLEL_MIN_BYTES
    after = _Digest(threaded)
    first = _Digest(threaded) if before else None
    zeroed = 0
    try:
        head = _read_exact(handle, offset)
        if first is not None:
            first.update(head)
        after.update(head)
        position = offset
        for plane in range(planes):
            for y in range(0, height, rows):
                if check is not None:
                    check()
                block = ~valid_storage[y:y + rows, :]
                n = int(block.sum())
                count = block.shape[0] * row_bytes
                data = _read_exact(handle, count)
                if first is not None:
                    first.update(data)
                if n:
                    out = bytearray(data)
                    np.frombuffer(out, dtype=dtype).reshape(block.shape)[block] = zero
                    handle.seek(position)
                    handle.write(out)
                    after.update(out)
                else:
                    after.update(data)
                position += count
                if plane == 0:
                    zeroed += n
        tail = handle.read()
        if first is not None:
            first.update(tail)
        after.update(tail)
        handle.flush()
        before_sha = first.hexdigest() if first is not None else None
        after_sha = after.hexdigest()
    finally:
        after.close()
        if first is not None:
            first.close()
    # The whole file was read from disk once: a real read feeding a digest (as a sha256_file of it
    # would count).  The digest of the result is then of bytes in memory: in-memory hashing when the
    # read already counts for the digest before, else the read itself is what fed it.
    if first is not None:
        hashing.note_read(first.nbytes, source=FUSED_SOURCE)
        hashing.note_memory(after.nbytes, source=FUSED_SOURCE)
    else:
        hashing.note_read(after.nbytes, source=FUSED_SOURCE)
    return {"zeroed_pixels": zeroed, "invalid_fraction": float(zeroed / (width * height))}, before_sha, after_sha


def frame_to_reference_native(points_ref: np.ndarray, h_frame_to_ref_td: np.ndarray, model, *, undistorted_ref=None) -> np.ndarray:
    """Reference-grid (native, top-down) points -> the frame's native pixel positions that the registration maps onto them.

    ``model`` is the job's ``RadialDistortion`` (with magnification; ``apply``
    = native -> magnified undistorted grid, the grid the matrices act in) or
    ``None`` for a plain homography in native coordinates.  ``undistorted_ref``
    may carry ``model.apply(points_ref)`` computed once for many frames (the
    forward map is the iterative, expensive half).
    """
    h_inv = np.linalg.inv(np.asarray(h_frame_to_ref_td, dtype=np.float64))
    if model is None:
        return apply_matrix(h_inv, points_ref)
    u = model.apply(points_ref) if undistorted_ref is None else undistorted_ref
    return model.inverse(apply_matrix(h_inv, u))


def _validity_lookup(validity, stem: str) -> np.ndarray:
    if isinstance(validity, np.ndarray):
        return validity
    if isinstance(validity, dict):
        return validity[stem]
    return validity.for_frame(stem)


# The pre-P1b serial loop, kept unchanged as the test oracle of coverage_at (efficiency P1b-1p).
def _coverage_at_reference(points_ref: np.ndarray, validity, transforms: dict, model, *, check=None) -> np.ndarray:
    """Count valid nearest-neighbour source centres, not full interpolation support.

    ``transforms`` maps stem -> 3x3 frame->reference matrix (top-down, in the
    model's grid); the reference frame itself is the identity.  ``validity``
    is one array for every frame (fixed tripod, static mask), a dict stem ->
    array, or a source with ``for_frame``.  The forward distortion of the
    reference points is computed once for all frames (RUN_LOG §26).
    """
    count = np.zeros(len(points_ref), dtype=np.int32)
    u_ref = model.apply(points_ref) if model is not None else None
    for stem, h in transforms.items():
        if check is not None:
            check()
        valid = _validity_lookup(validity, stem)
        height, width = valid.shape
        p = frame_to_reference_native(points_ref, h, model, undistorted_ref=u_ref)
        inside = (p[:, 0] >= -0.5) & (p[:, 0] < width - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] < height - 0.5)
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, height - 1)
        count += (valid[yi, xi] & inside).astype(np.int32)
    return count


def coverage_at(points_ref: np.ndarray, validity, transforms: dict, model, *, check=None, workers: int = 1,
                stats: dict | None = None) -> np.ndarray:
    """Count valid nearest-neighbour source centres, not full interpolation support.

    ``transforms`` maps stem -> 3x3 frame->reference matrix (top-down, in the
    model's grid); the reference frame itself is the identity.  ``validity``
    is one array for every frame (fixed tripod, static mask), a dict stem ->
    array, or a source with ``for_frame``.  The forward distortion of the
    reference points is computed once for all frames (RUN_LOG §26).

    Efficiency P1b-1p (G-6 pilot): the frames run on ``frame_pool`` with
    ``workers`` threads.  The validity lookup stays in the calling thread in
    frame order (a per-frame source loads and caches its masks as before), a
    unit computes one frame's int32 term with the expressions of
    ``_coverage_at_reference``, and the terms are added in frame order, so
    the counts are identical for any ``workers``.  ``workers=1`` (the default)
    is the pool's serial path, the reference's order of calls.  ``stats``
    receives the pool's diagnostics (timings notes only).
    """
    count = np.zeros(len(points_ref), dtype=np.int32)
    u_ref = model.apply(points_ref) if model is not None else None

    def lookup(i, item, _check):
        stem, h = item
        return _validity_lookup(validity, stem), h

    def term(payload, unit):
        valid, h = payload
        height, width = valid.shape
        p = frame_to_reference_native(points_ref, h, model, undistorted_ref=u_ref)
        inside = (p[:, 0] >= -0.5) & (p[:, 0] < width - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] < height - 0.5)
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, height - 1)
        return (valid[yi, xi] & inside).astype(np.int32)

    def fold(i, value):
        np.add(count, value, out=count)        # the ufunc that ``count += value`` calls

    frame_pool.run_ordered(term, list(transforms.items()), workers=workers, name="coverage", check=check,
                           prepare=lookup, on_result=fold, stats=stats)
    return count


def coverage_map(validity, transforms: dict, model, *, step: int = DEFAULT_COVERAGE_STEP, check=None, shape=None,
                 workers: int = 1, stats: dict | None = None) -> np.ndarray:
    """Coverage sampled every ``step`` pixels (top-down grid of shape (ceil(H/step), ceil(W/step)))."""
    if shape is None:
        if isinstance(validity, np.ndarray):
            shape = validity.shape
        elif isinstance(validity, dict):
            shape = next(iter(validity.values())).shape
        else:
            shape = validity.for_frame(next(iter(transforms))).shape
    height, width = shape
    gy, gx = np.mgrid[0:height:step, 0:width:step]
    points = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    return coverage_at(points, validity, transforms, model, check=check, workers=workers,
                       stats=stats).reshape(gy.shape)


def coverage_summary(cov: np.ndarray, mask_top_down: np.ndarray, *, step: int, frames: int) -> dict:
    """What the coverage map says about the user's sky region (mask == 1, sampled on the same grid)."""
    sky = mask_top_down[::step, ::step] >= 1.0
    sky = sky[:cov.shape[0], :cov.shape[1]]
    values = cov[sky]
    out = {"frames": int(frames), "step": int(step), "sky_samples": int(sky.sum()),
           "semantics": "nearest_neighbour_source_centres", "interpolation_support": "not_certified"}
    if values.size:
        out.update({"min": int(values.min()), "median": float(np.median(values)),
                    "fraction_full": float((values >= frames).mean()), "fraction_zero": float((values == 0).mean()),
                    "fraction_below_half": float((values < frames / 2).mean())})
        # height of the reduced-coverage band directly above the mask edge, per column (columns that have a ground
        # edge): the contiguous run of samples with fewer than all frames, counted upwards from the edge — the frames'
        # own top borders also reduce coverage near the top of the image, so only the run touching the edge counts
        heights = []
        for cx in range(cov.shape[1]):
            col_sky = sky[:, cx]
            if not col_sky.any() or col_sky.all():
                continue
            edge = int(np.argmax(~col_sky))                    # first ground sample from the top
            if edge == 0:
                continue
            col = cov[:edge, cx][::-1]                         # index 0 = the sample just above the edge
            full = np.where(col >= frames)[0]
            heights.append(int((full.min() if full.size else len(col)) * step))
        if heights:
            out["reduced_coverage_band_px"] = {"median": float(np.median(heights)), "p90": float(np.percentile(heights, 90)), "max": int(max(heights))}
    return out


def write_coverage_png(cov: np.ndarray, path: str | Path) -> None:
    """16-bit grey PNG of the sampled coverage counts (value = number of frames)."""
    from PIL import Image
    Image.fromarray(cov.astype(np.uint16)).save(path)


def read_coverage_png(path: str | Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img, dtype=np.int32)


def effective_sky_weight(mask_top_down: np.ndarray, validity_ref_top_down: np.ndarray, transforms: dict, model, *, check=None,
                         validity=None) -> tuple[np.ndarray, dict]:
    """The compose weight of the sky master: the user's mask, zeroed where no frame observed the sky.

    Only pixels the reference frame itself does not observe (its own invalid
    zone) can have zero coverage, and only those with a positive mask weight
    matter for the composite, so coverage is evaluated at full resolution on
    that band alone.  ``validity`` (per-frame source/dict) defaults to the
    reference's own validity for every frame (static mask).
    """
    band = (~validity_ref_top_down) & (mask_top_down > 0.0)
    ys, xs = np.nonzero(band)
    weight = mask_top_down.astype(np.float32).copy()
    info = {"band_pixels": int(len(ys)), "no_coverage_pixels": 0,
            "coverage_semantics": "nearest_neighbour_source_centres", "interpolation_support": "not_certified"}
    if len(ys):
        points = np.column_stack([xs, ys]).astype(np.float64)
        cov = coverage_at(points, validity_ref_top_down if validity is None else validity, transforms, model, check=check)
        zero = cov == 0
        weight[ys[zero], xs[zero]] = 0.0
        info["no_coverage_pixels"] = int(zero.sum())
    return weight, info


def write_record(work: str | Path, record: dict) -> Path:
    path = Path(work) / VALIDITY_RECORD
    from .records import atomic_json
    atomic_json(path, record)
    return path


def read_record(work: str | Path) -> dict | None:
    path = Path(work) / VALIDITY_RECORD
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("expected object")
        return record
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("天空有效区记录损坏；停止使用该检查点，请新建作业") from exc


def sha256_file(path: str | Path) -> str:
    return hashing.sha256_file(path, source="sky_validity.sha256_file")
