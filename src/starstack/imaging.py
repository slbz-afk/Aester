"""Numerical image I/O and deliberately limited display rendering.

Siril RAW data is camera RGB. Display rendering here is an uncalibrated
visualization, not a camera-to-working-space characterization.
"""

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
import contextlib
import json
import math
import os
import struct
import time

import numpy as np
from astropy.io import fits
from PIL import Image, ImageCms
import tifffile

from . import frame_pool, hashing


def read_rgb(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdus:
        integer = hdus[0].header["BITPIX"] == 16
        a = np.asarray(hdus[0].data, dtype=np.float32)
        top_down = hdus[0].header.get("ROWORDER", "BOTTOM-UP") == "TOP-DOWN"
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[0] == 3:
        a = np.moveaxis(a, 0, -1)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"Unsupported or non-finite FITS pixels: {path}")
    if integer:
        a = a / np.float32(65535)
    if top_down:
        a = np.flipud(a).copy()
    return a


def write_linear(path: Path, data: np.ndarray, note: str, extra: dict | None = None) -> None:
    """Write ``data`` ((H, W, 3), bottom-up rows) as a float32 FITS master; the bytes are exactly the former
    ``fits.writeto`` of ``np.moveaxis(data.astype(np.float32), -1, 0)`` (kept as ``_write_linear_former``).

    Efficiency P1c U-M14 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.1): a native float32 (H, W, 3) array is streamed
    to the file -- astropy's own header bytes, then each plane in row blocks of at most 8 MiB copied into one reused
    buffer and byte-swapped, then astropy's padding -- instead of the float32 copy and the big-endian ``flatten()``
    copy the former write made (two copies of the whole image). Inside a hash run scope (the job's ledger) the stream
    goes through ``hashing.write_with_digest`` as the former write did; outside one, or with AESTER_HASH_LEDGER=off,
    into the plain file the former ``fits.writeto(path, ...)`` created (_stream_linear_plain). Anything else (another
    dtype or shape, a compressing extension, an existing path, a failed plan or a failed per-call self-test) runs the
    former write.

    The stream reads the caller's array during the whole write: no thread may write ``data`` before this returns.
    """
    if os.path.exists(path):
        # astropy's own refusal to overwrite (and its overwrite of an empty file): exactly the former write.
        _write_linear_former(path, data, note, extra)
        return
    header = _linear_header(note, extra)        # the former's first step: a header error is raised as before
    plan = _linear_plan(path, data, header)
    why = "plan" if plan is None else _linear_self_test(header)
    if why is not None:
        _note_linear_fallback(why)
        _write_linear_former(path, data, note, extra)
        return
    # astropy validates before it creates the file and so does the plan: a failure before any byte is written
    # leaves no file, as before; a write failure leaves the part written and raises astropy's text.
    if not hashing.digest_writes():
        _stream_linear_plain(path, data, plan)
        return
    hashing.write_with_digest(path, lambda stream: _stream_linear(stream, data, plan),
                              source="imaging.write_linear", mode="xb", discard=hashing.DISCARD_UNWRITTEN)


def _linear_header(note: str, extra: dict | None = None) -> fits.Header:
    h = fits.Header()
    h["ORIGIN"] = "Starstack prototype"
    h["ROWORDER"] = "BOTTOM-UP"
    for key, value in (extra or {}).items():   # e.g. STACKCNT carried over from a Siril stack that was re-projected
        h[key] = value
    h["HISTORY"] = note
    return h


def _write_linear_former(path: Path, data: np.ndarray, note: str, extra: dict | None = None) -> None:
    # The pre-U-M14 write_linear, verbatim apart from the header lines moved into _linear_header: the reference of
    # the tests and the fallback of write_linear.
    h = _linear_header(note, extra)
    pixels = np.moveaxis(data.astype(np.float32), -1, 0)
    if not hashing.digest_writes() or os.path.exists(path):
        # Outside a hash run scope, with AESTER_HASH_LEDGER=off, or onto an existing path (astropy's own
        # refusal to overwrite): exactly the former write.
        fits.writeto(path, pixels, h, overwrite=False)
        return
    # Same bytes as fits.writeto(path, ...): astropy writes the file sequentially, so the digest of
    # the bytes written is recorded (the next hash of this file is a ledger hit).  astropy validates
    # before it creates the file: a failure before any byte is written leaves no file, as before.
    hashing.write_with_digest(path, lambda stream: fits.writeto(stream, pixels, h, overwrite=False),
                              source="imaging.write_linear", mode="xb", discard=hashing.DISCARD_UNWRITTEN)


# ---- U-M14: streamed linear masters (efficiency P1c) ----------------------------------------------------------------
LINEAR_CHUNK_BYTES = 8 << 20      # at most this many data bytes per streamed block (one reused buffer)
# Extensions astropy compresses when it opens a path itself (case-sensitive, as astropy's own os.path.splitext test)
_COMPRESSING_SUFFIXES = (".gz", ".zip", ".bz2", ".xz", ".Z")


class _LinearPlan(NamedTuple):
    hdus: object          # the HDUList astropy's writeto would write (its outer free-space check)
    hdu: object           # its PrimaryHDU (the inner free-space check): header and sizes of the real data
    head: bytes           # the header bytes astropy writes
    padding: bytes        # the data padding astropy writes


def _linear_plan(path, data, header) -> _LinearPlan | None:
    """astropy's own header and padding for ``data``, through the steps fits.writeto takes (``_makehdu``, the
    PrimaryHDU it then makes, ``verify('exception')``, ``update_extend``, ``_prewriteto``) on a read-only broadcast
    stand-in of the same dtype and shape (no pixel memory); None (the former write runs) for anything but a native
    float32, three-dimensional ndarray with every axis >= 1 written to a path without a compressing extension, and
    on any exception."""
    try:
        if (type(data) is not np.ndarray or data.dtype != np.float32 or not data.dtype.isnative or data.ndim != 3
                or min(data.shape) < 1):
            return None
        if os.path.splitext(os.fspath(path))[1] in _COMPRESSING_SUFFIXES:
            return None
        from astropy.io.fits import convenience
        from astropy.io.fits.header import _pad_length
        stand_in = np.broadcast_to(np.zeros((), dtype=np.float32), (data.shape[2], data.shape[0], data.shape[1]))
        hdu = convenience._makehdu(stand_in, header)                  # fits.writeto, verbatim
        if hdu.is_image and not isinstance(hdu, fits.PrimaryHDU):
            hdu = fits.PrimaryHDU(stand_in, header=header)
        hdus = fits.HDUList([hdu])                                     # _BaseHDU.writeto
        hdus.verify(option="exception")                                # HDUList.writeto
        hdus.update_extend()
        hdu._output_checksum = False
        hdu._prewriteto()
        text = hdu._header.tostring(sep="", endcard=True, padding=True)   # Header.tofile's defaults
        size = hdu.size
        if len(text) % 2880 or size != data.size * 4:
            return None
        padding = _pad_length(size) * hdu._padding_byte if size > 0 and _pad_length(size) > 0 else ""
        return _LinearPlan(hdus, hdu, text.encode("ascii"), padding.encode("ascii"))
    except Exception:
        return None


def _stream_linear(stream, data, plan: _LinearPlan) -> None:
    """Header, data (plane by plane, row blocks byte-swapped in one reused buffer: astropy's own byteswap of a
    native array) and padding, inside astropy's two free-space checks (HDUList.writeto, then _BaseHDU._writeto), so
    a write error is raised with astropy's text."""
    from astropy.io.fits.util import _free_space_check, fileobj_name
    try:
        dirname = os.path.dirname(fileobj_name(stream))
    except (AttributeError, TypeError):
        dirname = None
    height, width, planes = data.shape
    rows = max(1, min(height, LINEAR_CHUNK_BYTES // (width * 4)))
    with _free_space_check(plan.hdus, dirname=dirname):
        with _free_space_check(plan.hdu, dirname):
            stream.write(plan.head)
            buffer = np.empty((rows, width), dtype=np.float32)
            for plane in range(planes):
                for y in range(0, height, rows):
                    block = buffer[:min(rows, height - y)]
                    np.copyto(block, data[y:y + rows, :, plane])
                    block.byteswap(inplace=True)
                    stream.write(memoryview(block).cast("B"))
            if plan.padding:
                stream.write(plan.padding)


class _WriteAttempts:
    """A plain file for _stream_linear that notes whether a write was attempted (hashing.HashingWriter's
    ``attempted``, without its digest). ``name`` is the file's, so astropy's free-space checks look at the file's own
    directory, as they do when astropy opens the path itself."""

    def __init__(self, raw):
        self._raw, self.name, self.attempted = raw, raw.name, False

    def write(self, data):
        self.attempted = True
        return self._raw.write(data)


def _stream_linear_plain(path, data, plan: _LinearPlan) -> None:
    """The stream outside a hash run scope (or with AESTER_HASH_LEDGER=off), into the file the former
    ``fits.writeto(path, ...)`` created: astropy opens ``os.path.expanduser(os.fspath(path))`` 'wb' after its own
    existence check, this opens the same name 'xb' (the path was just found absent). As inside a ledger
    (write_with_digest's DISCARD_UNWRITTEN): a failure before any write is attempted deletes the file this call
    created (astropy validates before it creates the file, and so did the plan: no file, as before); a later one
    leaves the part written and raises astropy's text (_stream_linear's free-space checks)."""
    name = os.path.expanduser(os.fspath(path))
    with open(name, "xb") as raw:
        stream = _WriteAttempts(raw)
        try:
            _stream_linear(stream, data, plan)
        except BaseException:
            if not stream.attempted:
                raw.close()
                with contextlib.suppress(PermissionError):
                    os.remove(name)
            raise


def _linear_sentinel() -> np.ndarray:
    """(3, 4, 3) float32, rows reversed (a negative stride, like np.flipud): quiet and signalling NaNs with payloads,
    +-0, +-inf, subnormals, the extremes and ordinary values."""
    bits = np.array([0x7FC00000, 0x7FC12345, 0xFFC00001, 0x7F800001, 0x7FA00005, 0xFF80ABCD, 0x00000000, 0x80000000,
                     0x7F800000, 0xFF800000, 0x00000001, 0x807FFFFF, 0x00400000, 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000,
                     0x3F800000, 0xBF800000, 0x3EAAAAAB, 0x47800000, 0x33D6BF95, 0x80000001, 0x7F7FFFFE, 0x3F7FFFFF,
                     0x40490FDB, 0xC0490FDB, 0x00000002, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800002, 0x00012345, 0x3DCCCCCD,
                     0x42F6E979, 0x7E967699, 0x0000FFFF, 0x80800000], dtype=np.uint32)
    return bits.view(np.float32).reshape(3, 4, 3)[::-1]


def _linear_self_test(header) -> str | None:
    """None when the stream writes exactly the bytes of the former write (``fits.writeto`` of the moved float32 copy)
    for the sentinel with this call's header; otherwise why not. Any exception counts as a failure: the former write
    then runs and keeps its own outcome and error text."""
    import io
    try:
        sentinel = _linear_sentinel()
        plan = _linear_plan("self-test.fit", sentinel, header)
        if plan is None:
            return "self-test: no plan"
        streamed, expected = io.BytesIO(), io.BytesIO()
        _stream_linear(streamed, sentinel, plan)
        fits.writeto(expected, np.moveaxis(sentinel.astype(np.float32), -1, 0), header, overwrite=False)
        return None if streamed.getvalue() == expected.getvalue() else "self-test: bytes differ"
    except Exception as exc:
        return f"self-test: {type(exc).__name__}: {str(exc)[:160]}"


def _note_linear_fallback(why: str) -> None:
    try:   # diagnostics only (timings notes; never a record, signature or digest)
        from . import profiling
        profiling.note_append("write_linear", {"route": "former", "reason": why})
    except Exception:
        pass


# ---- foreground median (efficiency P1a C-1) ---------------------------------------------------------------------
MEDIAN_BLOCK_ROWS = 64            # rows per cancel_check(), exactly as before
MEDIAN_LANE_BUDGET = 8 << 20      # bytes of integer lanes (all frames) per tile: fits a desktop L3; speed only
MEDIAN_TILE_ROWS = (4, 64)        # clamp for the rows per tile; speed only -- the result does not depend on it
MEDIAN_KERNEL = "u16-selection-network-1"   # diagnostics note only
# Efficiency P1c C-1 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.1): the integer-network route on frame_pool, one
# 64-row block per unit, at most MEDIAN_WORKER_CAP threads and MEDIAN_OPEN_FILES frame handles (n per thread).
MEDIAN_WORKER_CAP = 4
MEDIAN_OPEN_FILES = 1024
MEDIAN_THREAD_SLACK = 12 << 20    # per median thread beyond its buffers (median_unit_bytes): the measured excess
                                  # (state 0's network, built before the pool, included) with a margin; 8 MiB left
                                  # n = 16 at 1 thread 33.0-36.3 MiB against 34.7 declared (P1c integration, R17)
_FP_ENV_LIMIT = np.float32(2 * np.finfo(np.float32).tiny)   # table values below it (and not 0): serial, 'fp-env'
_SIGN_BIT = np.uint16(0x8000)
_PLAIN_FITS = b"SIMPLE  ="


class _MedianFrame(NamedTuple):
    path: str
    offset: int          # first data byte: astropy's hdu._data_offset (HDU.fileinfo()["datLoc"])
    shape: tuple         # (3, H, W): astropy's own hdu.shape (hdu._axes), the former .data.shape
    bitpix: int          # 16 or -32: the full header's BITPIX, required equal to astropy's hdu._orig_bitpix
    bscale: float        # float(header.get("BSCALE", 1)): the value the former loop multiplied by
    bzero: float         # float(header.get("BZERO", 0))
    top_down: bool       # header.get("ROWORDER", "BOTTOM-UP") == "TOP-DOWN"


def median_foreground(paths: list[Path], cancel_check, *, workers: int | None = 1, timer=None) -> np.ndarray:
    """Per-pixel temporal median of converted RGB FITS frames: float32, bottom-up, (H, W, 3).

    Bit-identical to the former memory-mapped np.median path (kept below as _median_foreground_memmap and still used,
    unchanged, for any input the explicit reader does not cover).  Frames are read explicitly, 64 rows at a time, so no
    file is memory-mapped (the former copy-on-write maps committed the whole burst: ~14 GB for 48 frames).  Unsigned
    16-bit frames sharing one BSCALE/BZERO take the exact middle order statistics in the integer domain (a pruned
    Batcher selection network) and map them through a table of the former float32 conversion; np.mean of the middle
    element(s) is then the very call np.median makes (efficiency P1a, design C-1).

    Efficiency P1c C-1: the integer-network route runs its 64-row blocks on ``frame_pool`` -- ``workers`` threads
    (None: the site's own decision, see _median_workers; 1, the default: the pool's serial path, today's loop), at
    most MEDIAN_OPEN_FILES // n -- inside the timer operation ``foreground_median`` of ``timer`` (the job's
    StageTimer) when one is given. Every output element comes from the same comparators, table and float32 mean, and
    units write disjoint rows, so the result does not depend on the thread count. The float and former routes stay
    serial (their note: reason 'route'), as does a table with values near the subnormal range (reason 'fp-env').
    """
    frames = _median_frames(paths)
    if frames is None:
        _note_median("memmap", len(paths))
        with _serial_route_note(workers, None, "route"):
            return _median_foreground_memmap(paths, cancel_check)
    return _median_explicit(frames, cancel_check, workers=workers, timer=timer)


def _median_foreground_memmap(paths, cancel_check) -> np.ndarray:
    # The former implementation, verbatim (the oracle of the tests, and the reader for anything unusual).
    handles = []
    try:
        for path in paths:
            handles.append(fits.open(path, memmap=True, do_not_scale_image_data=True))
        shape = handles[0][0].data.shape
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError("Foreground fusion currently requires RGB FITS")
        if any(h[0].data.shape != shape for h in handles):
            raise ValueError("Frame dimensions differ; use one camera and framing")
        output = np.empty((shape[1], shape[2], 3), dtype=np.float32)
        for y in range(0, shape[1], 64):
            cancel_check()
            rows = []
            for h in handles:
                end = min(y + 64, shape[1])
                top_down = h[0].header.get("ROWORDER", "BOTTOM-UP") == "TOP-DOWN"
                block = h[0].data[:, shape[1]-end:shape[1]-y, :] if top_down else h[0].data[:, y:end, :]
                raw = np.array(block[:, ::-1, :] if top_down else block, dtype=np.float32)
                raw *= float(h[0].header.get("BSCALE", 1))
                raw += float(h[0].header.get("BZERO", 0))
                if h[0].header["BITPIX"] == 16:
                    raw /= np.float32(65535)
                rows.append(raw)
            output[y:y + 64] = np.moveaxis(np.median(np.stack(rows), axis=0), 0, -1)
        return output
    finally:
        for h in handles:
            h.close()


def fits_structure_unique(header, naxis: int) -> bool:
    """True when no structural keyword repeats in the full header (a second guard: the layout itself is always taken
    from astropy's own fast-parsed values, see _median_frame / complete_unsigned16_shape)."""
    keys = ("SIMPLE", "BITPIX", "NAXIS", "BZERO", "BSCALE", "ROWORDER", "GROUPS", "BLANK") + tuple(
        f"NAXIS{i}" for i in range(1, naxis + 1))
    return all(key not in header or header.count(key) == 1 for key in keys)   # count() raises KeyError if absent


def plain_fits_stream(path) -> bool:
    """True when the file's own bytes are the stream astropy parses.  astropy decompresses gzip, zip, bzip2, xz and
    compress (.Z) files by their first bytes whatever their name, and its data offsets are then offsets into the
    decompressed stream; a plain FITS file starts with the SIMPLE card."""
    with open(path, "rb") as raw:
        return raw.read(len(_PLAIN_FITS)) == _PLAIN_FITS


def _median_frames(paths):
    """Header facts of every frame from the same astropy open as the former reader, or None when the former reader must
    run: no frames, a frame that is not a plain (uncompressed) primary 3-plane image with BITPIX 16 or -32 and a
    complete data unit, a BITPIX the full header and astropy's own parse disagree on, a repeated structural keyword,
    frames of different shapes, or scaling cards float() cannot read.  Every file is opened, in order, so an open error
    (missing file, broken header) is the one the former first loop raised."""
    frames, covered = [], True
    for path in paths:
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:   # header only: .data never touched
            frame = _median_frame(path, hdus) if covered else None
        if frame is None:
            covered = False
        else:
            frames.append(frame)
    if not covered or not frames or any(f.shape != frames[0].shape for f in frames):
        return None
    return frames


def _median_frame(path, hdus):
    """The former loop read .data with astropy's OWN layout (dtype from hdu._orig_bitpix, shape from hdu._axes, offset
    hdu._data_offset -- all from astropy's fast header pass) and scaled it with the FULL header's BSCALE/BZERO/BITPIX/
    ROWORDER.  The explicit reader takes the layout from the same astropy values and the scaling from the same full-header
    calls; the full-header BITPIX must equal astropy's, so the /65535 decision and the read dtype cannot diverge."""
    try:
        hdu = hdus[0]
        if type(hdu) is not fits.PrimaryHDU:          # not GroupsHDU / non-standard
            return None
        header = hdu.header
        bitpix = header["BITPIX"]                     # the former `h[0].header["BITPIX"] == 16` test reads this
        shape = hdu.shape                             # astropy's _axes: the former .data.shape when the data is complete
        if type(bitpix) is not int or bitpix not in (16, -32) or type(hdu._orig_bitpix) is not int \
                or hdu._orig_bitpix != bitpix:        # astropy's read dtype (BITPIX2DTYPE[_orig_bitpix]) agrees
            return None
        if len(shape) != 3 or shape[0] != 3 or any(type(n) is not int or n <= 0 for n in shape):
            return None
        if not fits_structure_unique(header, len(shape)):
            return None
        bscale = float(header.get("BSCALE", 1))
        bzero = float(header.get("BZERO", 0))
        top_down = header.get("ROWORDER", "BOTTOM-UP") == "TOP-DOWN"
        offset = hdu.fileinfo()["datLoc"]             # the HDU's own method: hdu._data_offset, no header side effects
        size = os.stat(path).st_size
        if not plain_fits_stream(path):               # not a compressed stream astropy would decompress
            return None
    except Exception:                                 # anything unusual: the former reader reproduces the former outcome
        return None
    if type(offset) is not int or size < offset + abs(bitpix) // 8 * math.prod(shape):
        return None
    return _MedianFrame(os.fspath(path), offset, tuple(shape), bitpix, bscale, bzero, top_down)


def _read_rows(handle, frame, plane, start, stop, out):
    """Storage rows [start, stop) of one plane into ``out`` (C-contiguous, big-endian dtype of the file)."""
    _, height, width = frame.shape
    handle.seek(frame.offset + (plane * height + start) * width * out.itemsize)
    view = memoryview(out).cast("B")
    done = 0
    while done < view.nbytes:
        got = handle.readinto(view[done:])
        if not got:
            raise OSError(f"FITS data ended early (the file changed while it was read): {frame.path}")
        done += got


def _median_explicit(frames, cancel_check, *, workers: int | None = 1, timer=None, inflight=None) -> np.ndarray:
    """The explicit reader (efficiency P1c C-1): the integer-network route on frame_pool, the float route and a
    table near the subnormal range serially (_median_serial, today's loop)."""
    n = len(frames)
    _, height, width = frames[0].shape
    network = _median_network(frames)
    _note_median(MEDIAN_KERNEL if network is not None else "float-np-median", n, network)
    blocks = list(range(0, height, MEDIAN_BLOCK_ROWS))
    unit_bytes = median_unit_bytes(n, width)
    serial = "route" if network is None else "fp-env" if _table_fp_sensitive(network.table) else None
    if serial is not None:
        with _serial_route_note(workers, len(blocks), serial, unit_bytes):
            return _median_serial(frames, network, cancel_check)
    bound = "caller"
    if workers is None:
        workers, bound = _median_workers(n, width, len(blocks))
    elif workers > MEDIAN_OPEN_FILES // n:
        workers, bound = max(1, MEDIAN_OPEN_FILES // n), "handles"
    output = np.empty((height, width, 3), dtype=np.float32)
    spare = [network]                      # the first state takes the network built above (its buffers), the rest
                                           # fresh ones; init() runs on the calling thread only

    def init():
        handles = []
        try:
            for f in frames:               # in frame order; a failed open closes the ones already open
                handles.append(open(f.path, "rb", buffering=0))
        except BaseException:
            for handle in handles:
                handle.close()
            raise
        return handles, np.empty((n, MEDIAN_BLOCK_ROWS, width), dtype=">u2"), spare.pop() if spare else network.fresh()

    def finalize(state):
        for handle in state[0]:
            handle.close()

    def block_unit(y, unit):
        handles, raw, net = unit.state
        end = min(y + MEDIAN_BLOCK_ROWS, height)
        block = raw[:, :end - y]
        probe = frame_pool.in_pool()       # R18: a probe between planes on the parallel path only (the serial path
        for plane in range(3):             # keeps today's one cancel_check per block)
            if plane and probe:
                unit.check()
            for h, f, rows in zip(handles, frames, block):
                start, stop = (height - end, height - y) if f.top_down else (y, end)
                _read_rows(h, f, plane, start, stop, rows)
            net.plane(block, output[y:end, :, plane])

    from . import performance, resources
    pool_stats = {"bound": bound}
    operation = timer.operation if timer is not None else contextlib.nullcontext
    with operation("foreground_median"):
        frame_pool.run_ordered(block_unit, blocks, workers=workers, name="median", check=cancel_check, init=init,
                               finalize=finalize, inflight=inflight, stats=pool_stats, unit_bytes=unit_bytes,
                               blas=False)
    performance.note_pool("median", pool_stats)
    resources.shadow_pool("py.median", actual=pool_stats.get("workers"), items=len(blocks), width=width, frames=n)
    return output


def median_unit_bytes(frames: int, width: int) -> int:
    """One median thread's memory (P1c R10/R17): its raw block (n, 64, W) '>u2', its network's n + 1 lanes and two
    gathered middle rows of a tile, np.mean's tile result, and MEDIAN_THREAD_SLACK (the thread itself, numpy's
    iteration buffers, and the call's fixed part when one thread carries it alone) -- 72.6 MB (69.2 MiB) at n = 48,
    W = 8280. Measured on the re-converted F48 frames, private bytes sampled around the whole _median_explicit call
    (so state 0's network, built before the pool, counts) less the output array, per granted thread: n = 48 59.2-62.4
    MiB at 1 thread, 57.9-58.4 at 4, 57.4 at 16; n = 16 30.3-32.0, 28.4-28.8, 27.3-28.3 MiB -- against 69.2 / 38.7 MiB
    here (34.7 MiB with 8 MiB slack failed once at n = 16, 1 thread: 33.0-36.3 MiB on the integrated tree) (the buffers alone, 57.2 / 26.7 MiB, match the per-thread slope: about 57.2 / 27.4 MiB per thread plus a
    fixed 3.6-4.8 MiB per call). A unit's payload is a block start and its result None; the output rows it writes are
    the caller's array."""
    rows = _median_tile_rows(frames, width)
    return (frames * MEDIAN_BLOCK_ROWS * width * 2 + (frames + 1) * rows * width * 2 + 3 * rows * width * 4
            + MEDIAN_THREAD_SLACK)


def _median_tile_rows(n, width):
    return min(max(MEDIAN_LANE_BUDGET // ((n + 1) * width * 2), MEDIAN_TILE_ROWS[0]), MEDIAN_TILE_ROWS[1])


def _median_workers(n, width, units):
    """C-1's thread count and what bounded it: performance.frame_workers (AESTER_FRAME_WORKERS, else cap 4, half the
    usable CPUs and the memory budget at median_unit_bytes per thread; no BLAS in a unit), then at most
    MEDIAN_OPEN_FILES // n threads (each holds n open frames)."""
    from . import performance
    decision = {}
    workers = performance.frame_workers(units, median_unit_bytes(n, width), MEDIAN_WORKER_CAP, blas=False,
                                        decision=decision)
    if workers > MEDIAN_OPEN_FILES // n:
        return max(1, MEDIAN_OPEN_FILES // n), "handles"
    return workers, decision.get("bound")


def _table_fp_sensitive(table) -> bool:
    """True when a table value is not 0 but below 2 x float32's smallest normal: the float32 mean of two such values
    could depend on a thread's denormal control bits, so C-1 keeps those frames serial (reason 'fp-env')."""
    magnitude = np.abs(table)
    return bool(((magnitude > 0) & (magnitude < _FP_ENV_LIMIT)).any())


@contextlib.contextmanager
def _serial_route_note(requested, units, reason, unit_bytes=None):
    """After a serial route returns: its ``frame_pool.median`` timings note in the frame pool's serial format
    (requested / workers 1 / units / serial / reason, the CPU fields; no per-block clock). Diagnostics only."""
    started, thread_cpu, process = time.perf_counter(), time.thread_time(), os.times()
    yield
    try:
        from . import performance
        end, fp_env = os.times(), frame_pool.fp_environment()
        performance.note_pool("median", {
            "requested": requested, "workers": 1, "units": units, "serial": True, "reason": reason,
            "wall_seconds": round(time.perf_counter() - started, 4), "unit_bytes": unit_bytes, "reserved_bytes": 0,
            "unit_cpu_seconds": round(time.thread_time() - thread_cpu, 4), "max_unit_seconds": None,
            "cpu_user_seconds": round(end.user - process.user, 4),
            "cpu_kernel_seconds": round(end.system - process.system, 4),
            "fp_env": None if fp_env is None else hex(fp_env), "fp_env_applied": 0})
    except Exception:
        pass


def _median_serial(frames, network, cancel_check) -> np.ndarray:
    """The P1a loop for a given network (None: the float route), verbatim: one cancel_check per 64-row block."""
    n = len(frames)
    _, height, width = frames[0].shape
    output = np.empty((height, width, 3), dtype=np.float32)
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(open(f.path, "rb", buffering=0)) for f in frames]
        raw = np.empty((n, MEDIAN_BLOCK_ROWS, width), dtype=">u2") if network is not None else None
        for y in range(0, height, MEDIAN_BLOCK_ROWS):
            cancel_check()
            end = min(y + MEDIAN_BLOCK_ROWS, height)
            if network is None:
                output[y:end] = np.moveaxis(np.median(np.stack(
                    [_float_block(h, f, y, end) for h, f in zip(handles, frames)]), axis=0), 0, -1)
                continue
            block = raw[:, :end - y]
            for plane in range(3):
                for h, f, rows in zip(handles, frames, block):
                    start, stop = (height - end, height - y) if f.top_down else (y, end)
                    _read_rows(h, f, plane, start, stop, rows)
                network.plane(block, output[y:end, :, plane])
    return output


def _median_foreground_reference(paths, cancel_check) -> np.ndarray:
    # The pre-P1c median_foreground, verbatim but for the reader's name: the serial reference of the C-1 tests.
    frames = _median_frames(paths)
    if frames is None:
        _note_median("memmap", len(paths))
        return _median_foreground_memmap(paths, cancel_check)
    return _median_reference(frames, cancel_check)


def _median_reference(frames, cancel_check) -> np.ndarray:
    # The pre-P1c _median_explicit, verbatim: the serial reference of the C-1 tests (prereg §6.2 (a)).
    n = len(frames)
    _, height, width = frames[0].shape
    network = _median_network(frames)
    _note_median(MEDIAN_KERNEL if network is not None else "float-np-median", n, network)
    output = np.empty((height, width, 3), dtype=np.float32)
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(open(f.path, "rb", buffering=0)) for f in frames]
        raw = np.empty((n, MEDIAN_BLOCK_ROWS, width), dtype=">u2") if network is not None else None
        for y in range(0, height, MEDIAN_BLOCK_ROWS):
            cancel_check()
            end = min(y + MEDIAN_BLOCK_ROWS, height)
            if network is None:
                output[y:end] = np.moveaxis(np.median(np.stack(
                    [_float_block(h, f, y, end) for h, f in zip(handles, frames)]), axis=0), 0, -1)
                continue
            block = raw[:, :end - y]
            for plane in range(3):
                for h, f, rows in zip(handles, frames, block):
                    start, stop = (height - end, height - y) if f.top_down else (y, end)
                    _read_rows(h, f, plane, start, stop, rows)
                network.plane(block, output[y:end, :, plane])
    return output


def _float_block(handle, frame, y, end):
    """The former per-frame conversion of output rows [y, end), on an explicit read: same dtype, order and scalars."""
    _, height, width = frame.shape
    storage = np.empty((3, end - y, width), dtype=">i2" if frame.bitpix == 16 else ">f4")
    start, stop = (height - end, height - y) if frame.top_down else (y, end)
    for plane in range(3):
        _read_rows(handle, frame, plane, start, stop, storage[plane])
    raw = np.array(storage[:, ::-1, :] if frame.top_down else storage, dtype=np.float32)
    raw *= frame.bscale
    raw += frame.bzero
    if frame.bitpix == 16:
        raw /= np.float32(65535)
    return raw


def median_conversion_table(bscale: float, bzero: float) -> np.ndarray:
    """float32 value of every stored 16-bit code by the former op sequence; index = code XOR 0x8000 (= code + 32768)."""
    codes = np.arange(-32768, 32768, dtype=np.int32).astype(">i2")    # stored big-endian int16 codes, as in the file
    table = np.array(codes, dtype=np.float32)
    with np.errstate(all="ignore"):   # a scaling that overflows float32 leaves the table unusable (float path), and the
        table *= bscale               # float path then warns exactly as the former loop did; the values are unaffected
        table += bzero
        table /= np.float32(65535)
    return table


def _table_usable(table) -> bool:
    # Order statistics commute with a non-decreasing map; finite values without -0.0 make equal values equal bits.
    return bool(np.isfinite(table).all() and (table[1:] >= table[:-1]).all()
                and not (np.signbit(table) & (table == 0)).any())


@lru_cache(maxsize=None)
def median_pairs(n: int) -> tuple:
    """Comparators (a, b), a < b, of Batcher's odd-even merge sort on n inputs padded to a power of two (padding wires
    would hold +inf, so a comparator touching one is a no-op and is left out), pruned backwards to those that can
    reach the middle outputs n//2-1 and n//2 (n even) or n//2 (n odd)."""
    size = 1
    while size < n:
        size *= 2
    pairs, p = [], 1
    while p < size:
        k = p
        while k >= 1:
            for j in range(k % p, size - k, 2 * k):
                for i in range(min(k, size - j - k)):
                    a, b = i + j, i + j + k
                    if a // (2 * p) == b // (2 * p) and b < n:
                        pairs.append((a, b))
            k //= 2
        p *= 2
    need = {n // 2 - 1, n // 2} if n % 2 == 0 else {n // 2}
    kept = []
    for a, b in reversed(pairs):
        if a in need or b in need:
            kept.append((a, b))
            need.update((a, b))
    return tuple(reversed(kept))


class _MedianNetwork:
    """Middle order statistics of n uint16 lanes by min/max; then the table and np.mean exactly as np.median."""

    def __init__(self, n, table, width, top_down, tile_rows=None):
        self.n, self.table, self.top_down, self.width = n, table, list(top_down), width
        self.pairs = median_pairs(n)
        self.middle = (n // 2 - 1, n // 2) if n % 2 == 0 else (n // 2,)
        self.rows = tile_rows or _median_tile_rows(n, width)
        self.buffers = [np.empty((self.rows, width), dtype=np.uint16) for _ in range(n + 1)]
        self.gathered = np.empty((len(self.middle), self.rows, width), dtype=np.float32)

    def fresh(self):
        """The same network (table shared read-only, comparators, tile rows) with buffers of its own: one per pool
        state (P1c C-1). Every buffer is written before it is read within a tile, so a unit has no history (R12)."""
        return _MedianNetwork(self.n, self.table, self.width, self.top_down, tile_rows=self.rows)

    def plane(self, block, out):
        """block: (n, h, W) '>u2', each frame's rows as stored (a TOP-DOWN frame's are the mirror image of the output
        rows); out: the (h, W) float32 output rows of one plane."""
        rows = block.shape[1]
        for s in range(0, rows, self.rows):
            e = min(s + self.rows, rows)
            out[s:e] = self.tile([block[i, rows - e:rows - s][::-1] if td else block[i, s:e]
                                  for i, td in enumerate(self.top_down)])

    def tile(self, codes):
        """codes: n arrays (m, W) of stored big-endian codes ('>u2' views) -> (m, W) float32 medians."""
        m = codes[0].shape[0]
        lanes = [b[:m] for b in self.buffers[:self.n]]
        spare = self.buffers[self.n][:m]
        for lane, frame_codes in zip(lanes, codes):
            np.bitwise_xor(frame_codes, _SIGN_BIT, out=lane)    # stored int16 bits -> order-preserving uint16
        for a, b in self.pairs:
            x, y = lanes[a], lanes[b]
            np.minimum(x, y, out=spare)
            np.maximum(x, y, out=y)
            lanes[a], spare = spare, x
        part = self.gathered[:, :m]
        for slot, index in enumerate(self.middle):
            np.take(self.table, lanes[index], out=part[slot], mode="clip")   # every uint16 index is in range
        return np.mean(part, axis=0)                                          # the call np.median makes


def _median_network(frames):
    """The integer kernel when it is exactly the former math for these frames, else None (float path)."""
    if any(f.bitpix != 16 for f in frames):
        return None
    first = frames[0]
    key = struct.pack("<dd", first.bscale, first.bzero)                    # bit-equal scalings only (-0.0 != 0.0)
    if any(struct.pack("<dd", f.bscale, f.bzero) != key for f in frames):
        return None
    table = median_conversion_table(first.bscale, first.bzero)
    if not _table_usable(table) or not _median_self_test(len(frames), table, first.bscale, first.bzero):
        return None
    return _MedianNetwork(len(frames), table, first.shape[2], [f.top_down for f in frames])


def _self_test_columns(n):
    """(n, C) physical uint16 columns: random, heavy ties, extremes, sorted both ways, 0/1 near the middle, splits."""
    rng = np.random.default_rng(0x5EEDC1)
    cols = [rng.integers(0, 65536, size=(n, 2048)), rng.integers(0, 4, size=(n, 512)) * 21845,
            rng.integers(30000, 30004, size=(n, 512)), np.repeat(rng.integers(0, 65536, size=(1, 64)), n, 0),
            np.tile(np.array([[0], [65535]]), (n // 2 + 1, 64))[:n],
            np.sort(rng.integers(0, 65536, size=(n, 64)), axis=0),
            np.sort(rng.integers(0, 65536, size=(n, 64)), axis=0)[::-1], np.full((n, 8), 0), np.full((n, 8), 65535)]
    k = n // 2
    for count in range(max(0, k - 3), min(n, k + 3) + 1):
        order = np.argsort(rng.random((n, 256)), axis=0)
        cols.append(np.where(order < count, 40000, 39999))
    split = np.zeros((n, 2 * (n + 1)), dtype=np.int64)
    for j in range(n + 1):
        split[:j, j] = 65535
        split[:j, n + 1 + j] = 1
    cols.append(split)
    return np.concatenate(cols, axis=1).astype(np.uint16)


def _median_self_test(n, table, bscale, bzero) -> bool:
    """The integer kernel against the former float math and np.median, for this n and table (8 ms at n = 48)."""
    u = _self_test_columns(n)
    stored = (u ^ 0x8000).view(np.int16).astype(">i2")
    rows = []
    for i in range(n):
        raw = np.array(stored[i][None, None, :], dtype=np.float32)
        raw *= bscale
        raw += bzero
        raw /= np.float32(65535)
        rows.append(raw)
    expected = np.median(np.stack(rows), axis=0)[0, 0]
    got = _MedianNetwork(n, table, u.shape[1], [False] * n, tile_rows=1).tile(
        [s.view(">u2")[None, :] for s in stored])[0]
    return bool(np.array_equal(got.view(np.uint32), expected.view(np.uint32)))


def _note_median(kernel, frames, network=None):
    try:   # diagnostics only (the timings receipt's notes; never a record, signature or digest)
        from . import profiling
        profiling.note("foreground_median", {"kernel": kernel, "frames": frames,
                                             **({"tile_rows": network.rows, "comparators": len(network.pairs)}
                                                if network is not None else {})})
    except Exception:
        pass


def complete_unsigned16_shape(path, hdus, *, read_through=True):
    """The shape astropy's ``hdus[0].data`` would certainly have as a complete uint16 array -- a plain primary image
    that astropy itself reads as pseudo-unsigned 16-bit (its own _orig_bitpix/_orig_bscale/_orig_bzero with the uint
    option and scaling on; BLANK does not apply), the full header agreeing, each structural keyword once, positive
    integer axes and the whole data unit in the file -- else None (read the pixels).  With ``read_through`` (the
    default) every data byte is also read once, unconverted; if that fails the caller's pixel read raises today's
    error."""
    try:
        hdu = hdus[0]
        if type(hdu) is not fits.PrimaryHDU:
            return None
        # astropy's own rule (ImageHDU._dtype_for_bitpix, _scale_data, _convert_pseudo_integer) on the values it reads
        # the data with: the fast-parsed _orig_* of this very open, with scaling on and the uint option on
        if getattr(hdu, "_uint", None) is not True or getattr(hdu, "_do_not_scale_image_data", None) is not False:
            return None
        if not (hdu._orig_bitpix == 16 and hdu._orig_bscale == 1 and hdu._orig_bzero == 32768):
            return None
        header = hdu.header                           # second guard: the full header must say the same
        if not (header.get("BITPIX") == 16 and header.get("BSCALE", 1) == 1 and header.get("BZERO", 0) == 32768):
            return None
        shape = hdu.shape                             # astropy's _axes: the shape .data would have
        if not shape or any(type(n) is not int or n <= 0 for n in shape) or not fits_structure_unique(header, len(shape)):
            return None
        offset = hdu.fileinfo()["datLoc"]             # the HDU's own method: hdu._data_offset, no header side effects
        size = os.stat(path).st_size
        if not plain_fits_stream(path):               # st_size is the size of the stream astropy reads
            return None
    except Exception:
        return None
    if type(offset) is not int or size < offset + 2 * math.prod(shape):
        return None
    if read_through and not data_unit_readable(path, offset, 2 * math.prod(shape)):
        return None                                   # unreadable now: the pixel read raises today's error
    return tuple(shape)


READ_THROUGH_CHUNK = 8 << 20


def data_unit_readable(path, offset: int, nbytes: int) -> bool:
    """Read bytes [offset, offset + nbytes) of the file once, without converting them (the bytes astropy's pixel read
    would fetch).  False on any read problem, so the caller can let the former pixel read raise its own error."""
    try:
        buffer = bytearray(min(nbytes, READ_THROUGH_CHUNK))
        view = memoryview(buffer)
        with open(path, "rb", buffering=0) as raw:
            raw.seek(offset)
            while nbytes:
                got = raw.readinto(view[:min(nbytes, len(buffer))])
                if not got:
                    return False
                nbytes -= got
        return True
    except Exception:
        return False


def blend_with_mask(sky: np.ndarray, ground: np.ndarray, mask_path: Path) -> np.ndarray:
    if sky.shape != ground.shape:
        raise ValueError("Sky and foreground geometry differ; cannot apply source mask")
    with Image.open(mask_path) as img:
        if img.size != (sky.shape[1], sky.shape[0]):
            raise ValueError("Sky mask must exactly match reference image width and height")
        mask = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    # PNG uses normal top-to-bottom display coordinates; FITS is bottom-up.
    mask = np.flipud(mask)[..., None]
    return sky * mask + ground * (1 - mask)


def blend_with_weight(sky: np.ndarray, ground: np.ndarray, weight_top_down: np.ndarray) -> np.ndarray:
    """Like ``blend_with_mask`` with an explicit sky weight (0..1, top-down, e.g. the mask zeroed where no frame observed the sky)."""
    if sky.shape != ground.shape:
        raise ValueError("Sky and foreground geometry differ; cannot compose")
    if weight_top_down.shape != sky.shape[:2]:
        raise ValueError("Sky weight must exactly match the reference image width and height")
    w = np.flipud(np.asarray(weight_top_down, dtype=np.float32))[..., None]   # arrays here are in FITS (bottom-up) row order
    return sky * w + ground * (1 - w)


def display_base(data: np.ndarray, *, red_gain: float = 1.0, blue_gain: float = 1.0,
                 exposure_ev: float = 0.0) -> np.ndarray:
    sample = data[::max(1, data.shape[0] // 512), ::max(1, data.shape[1] // 512)]
    black = float(np.quantile(sample, 0.003))
    white = float(np.quantile(sample, 0.998))
    if white <= black:
        white = black + max(abs(black) * 0.01, 1e-6)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x = np.maximum((data - black) / (white - black), 0)
            # Display-only controls after common percentile normalization, so
            # exposure is not cancelled by re-estimating the white point.
            if red_gain != 1.0 or blue_gain != 1.0 or exposure_ev != 0.0:
                x *= np.array([red_gain, 1.0, blue_gain], dtype=np.float32) * np.float32(2 ** exposure_ev)
            x = np.arcsinh(6 * x) / np.arcsinh(6)
    except FloatingPointError as exc:
        raise ValueError("预览数值范围超出当前渲染器能力；线性母版未修改") from exc
    x = np.clip(x, 0, 1)
    # Map the visualization to an sRGB display; no claim of camera color accuracy.
    x = np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055)
    if not np.isfinite(x).all():
        raise ValueError("预览包含无效数值，无法导出；线性母版未修改")
    return np.flipud(x).copy()


def export_products(data: np.ndarray, directory: Path, cancel_check) -> list[dict]:
    directory.mkdir(exist_ok=True)
    artifacts = []
    linear_path = directory / "master_linear_32bit.tif"
    tifffile.imwrite(linear_path, np.flipud(data), photometric="rgb",
                     metadata={"color_space": "uncalibrated_camera_or_input_RGB", "linear": True})
    artifacts.append({"path": str(linear_path), "kind": "tiff", "label": "32位线性母版（未标定RGB）"})
    artifacts.extend(export_display_products(data, directory, cancel_check))
    return artifacts


def export_display_products(data: np.ndarray, directory: Path, cancel_check, *,
                            red_gain: float = 1.0, blue_gain: float = 1.0,
                            exposure_ev: float = 0.0) -> list[dict]:
    """Render only display TIFF/JPEG files; never create or alter a master."""
    directory.mkdir(exist_ok=True)
    artifacts = []
    cancel_check()
    base = display_base(data, red_gain=red_gain, blue_gain=blue_gain, exposure_ev=exposure_ev)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tiff_path = directory / "natural_display_16bit.tif"
    tifffile.imwrite(tiff_path, np.round(base * 65535).astype(np.uint16), photometric="rgb",
                     iccprofile=icc, metadata={"linear": False, "purpose": "uncalibrated display rendering"})
    artifacts.append({"path": str(tiff_path), "kind": "tiff", "label": "16位预览渲染（非线性）"})
    for key, label, gains in [("natural", "自然", (1, 1, 1)), ("cool", "清冷", (.94, 1.0, 1.07)),
                              ("warm", "暖调", (1.08, 1.015, .93))]:
        cancel_check()
        rendered = np.clip(base * np.array(gains, dtype=np.float32), 0, 1)
        p = directory / f"{key}.jpg"
        Image.fromarray(np.round(rendered * 255).astype(np.uint8)).save(p, quality=94, icc_profile=icc)
        artifacts.append({"path": str(p), "kind": "jpeg", "label": label})
    return artifacts


def create_demo(output_root: str) -> str:
    """Produce explicit numerical test fixtures, not an AI-generated photograph."""
    import uuid
    root = Path(output_root).expanduser().resolve() / f"synthetic-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True)
    rng = np.random.default_rng(41029)
    h, w = 512, 768
    yy, xx = np.mgrid[:h, :w]
    stars = [(rng.uniform(20, w-20), rng.uniform(185, h-20), rng.uniform(.2, .9),
              rng.uniform(.8, 1.6), rng.uniform(.8, 1.15, 3)) for _ in range(110)]
    horizon = 130 + 30 * np.sin(np.arange(w) / 95) + 12 * np.cos(np.arange(w) / 31)
    visible = yy > horizon[None, :]
    for i in range(8):
        rgb = np.zeros((h, w, 3), np.float32) + (.015, .018, .027)
        haze = .035 * np.exp(-((xx - 380 - .25 * yy) / 140) ** 2)
        rgb += haze[..., None] * np.array([1.1, 1.0, .9])
        for x, y, flux, radius, color in stars:
            shape = flux * np.exp(-((xx - (x + i * 1.35)) ** 2 + (yy - (y + i * .4)) ** 2) / (2 * radius ** 2))
            rgb += shape[..., None] * color
        ground = .009 + .005 * np.sin(xx / 38) ** 2 + .003 * np.cos(yy / 29)
        rgb = np.where(visible[..., None], rgb, ground[..., None] * np.array([.85, 1, .83]))
        rgb += rng.normal(0, .004, rgb.shape)
        write_linear(root / f"frame_{i + 1:03d}.fit", rgb, "Synthetic fixture; no real RAW or photographic validation")
    Image.fromarray((np.flipud(visible) * 255).astype(np.uint8)).save(root / "sky_mask.png")
    (root / "SYNTHETIC.json").write_text(json.dumps({"synthetic": True, "frames": 8, "seed": 41029,
                                                   "mask": "sky_mask.png", "reference": 1}, indent=2), encoding="utf-8")
    return str(root)
