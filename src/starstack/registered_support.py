"""Conservative, explicitly selected Siril 1.4.4 export/stack support.

This certifies dependency support, not PSF quality or radiometric accuracy.
Lanczos4 with the default anti-ringing clamp depends on the centre and four
neighbouring output locations. Each uses a bounded source rectangle, including
one anchor of phase/float rounding slack. The image edge is rejected, rather
than certified through OpenCV's synthetic REFLECT_101 extension.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
import re
from pathlib import Path
import time

import numpy as np

from . import fits_crop, frame_pool, hashing, support_kernel
from .interpolation_support import sample_support
from .records import atomic_json
from .siril_support_map import make_export_mapping, release_distortion_grid
from .support_bounds import invalid_integral, rectangle_support
from .support_anchor import build_anchor_grid, anchor_grid_from_validity, anchor_support

SUPPORT_VERSION = "1.0.0"
# Code-path labels (outputs are identical). The buffered label is recorded only when every frame used the
# P1a kernel; finish_stack then also takes the quick native path. Anything else runs the reference paths.
SUPPORT_IMPLEMENTATION = "numpy_buffered_kernel_2"
REFERENCE_IMPLEMENTATION = "numpy_anchor_axis_tiles_1"
MODE = "strict_siril_1_4_4"
MODES = ("legacy", MODE)
RECORD = "interpolation_support.json"
COUNTS = "sky_support_eligible.npy"
STACK_VALID = "sky_support_stack.npy"
NATIVE_VALID = "sky_support_native.npy"
EXPORT_SEQUENCE = "light_.support-export.seq"
EXPORT_COMMAND = "seqapplyreg light -filter-included -interp=lanczos4 -layer=1"
SIRIL_SHA256 = "132c735ecb623559a945c43a617811da84c19c43988c8a234861f903dd362e0b"
INTERPOLATION_DLLS = {
    "libopencv_imgproc-411.dll": "e6551a2db8b4c3b8a720e49bcc6ae8501182f0d03d54de0f1239b5f6a7faaac4",
    "libopencv_core-411.dll": "3e018791a9f0babc5b550b70acb83c94de65fdb708db2994c29bd623331a5e44",
}
CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
# Efficiency P1c (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.2): F-2 runs prepare_exports' frames, G-4t native_mask's
# 64-row blocks on frame_pool. The worker counts are decided in core and handed in (rule R22: this module is in the
# frozen holdout closure and imports nothing new); these are the figures core gives performance.frame_workers.
# F-2: one export per unit. STRICT_UNIT_BYTES is a unit's measured peak at STRICT_UNIT_PIXELS plus the one result a
# worker holds in flight (R17), rounded up: on the F48 exports one unit (the kernel's map and cross, the streamed zeroing
# with its digest queues, the .npy write, the digest after; the kernel's own buffers included) rose 175-182 MiB, its
# result (the supported map, held until delivered) is 43.6 MiB: 226 MiB -> 256 MiB. Core scales it by the frame's
# pixels (never below) and charges the OpenBLAS slot of the unit's np.linalg.det (R8: a 3x3 LU, one thread, the same
# value on any thread; it only gates the matrix). A pool of 8 rose 815-902 MiB on 12 frames (P1c validation report).
# The cap is PROVISIONAL. §3.2 fixes it by sweeping 4/6/8 on an idle machine for each profile (the development PC, the
# ordinary-PC profile, the 6900HX proxy) and taking the smallest value within 5 % of the best. The implementation's
# sweeps (the 48 F48 exports, the whole site and the kernel alone through replay_support --workers, fastest of 4-6 runs)
# ran on a host shared with the other P1c benchmarks: one setting varied up to 2.4x between runs (4 workers on the
# development PC: 61.5-148.4 s) and the series were not monotonic in the worker count, so the rule was not applied:
# 4 was taken because it was within 5 % of the fastest on both consumer profiles, where 8 was 8-13 % slower, and in
# the kernel alone everywhere; the rule alone would give 8 on the development PC. The idle-window resweep, the rule per
# profile and how the three results make one constant are open for the acceptance window (P1c validation report).
STRICT_WORKER_CAP = 4
STRICT_VOLUME_CAP = 2                    # the work folder is on a removable or network volume (GetDriveType)
STRICT_UNIT_BYTES = 256 << 20
STRICT_UNIT_PIXELS = 5520 * 8280
# A per-frame mask source decodes the next masks in prepare, ahead of the workers (mask PNG to float32, the minimum
# with the static mask, the dilation): the transient peak per frame pixel (measured: an 11.0-12.0 B/px rise at 45.7 MP,
# of which 1 B/px is the validity the source then keeps in its cache, charged separately per own mask).
# R18: the decode is one source.for_frame call that cannot be probed, since sky_validity is frozen (prereg §6.7 A16):
# 0.22-0.27 s per own mask at 45.7 MP on the idle development PC, 0.26-0.39 s while the pool's units run. It joins the
# pool's unprobeable segments of §5 item 5 (P1c validation report); a cancel during it returned in 0.29-0.43 s at
# 4/8/16 workers (tests/test_p1c_site_strict_support.py, real size).
STRICT_MASK_DECODE_BYTES_PER_PIXEL = 11
# G-4t: one 64-row block per unit; the worst case (every footprint mixed, so every point goes through sample_support)
# at NATIVE_UNIT_WIDTH, scaled by the width: measured 183-196 MiB per block on a checkerboard map (about 360 B per
# point), so 200 MiB, measured on the serial path's single sample_support call (the parallel path's probed chunks of
# support_kernel.PROBE_POINTS only lower it). No BLAS call in a block (RadialDistortion.apply and sample_support are
# elementwise).
# The cap is PROVISIONAL: §3.2 asks for a 2/4/8 rescan on an idle machine first. The implementation swept 2/3/4 (and 8)
# on the F48 stack map (87 blocks; serial 9-13 s) on the three profiles on the same shared host: 4 was the fastest of
# 2/3/4 on each, 3 more than 5 % slower on the development PC and the 6900HX proxy, and 8 gained another 0.4-0.9 s.
# The idle-window 2/4/8 rescan is open for the acceptance window (P1c validation report).
NATIVE_WORKER_CAP = 4
NATIVE_UNIT_BYTES = 200 << 20
NATIVE_UNIT_WIDTH = 8280


def _kernel_enabled() -> bool:
    """AESTER_STRICT_KERNEL=reference runs the reference paths (A/B runs and a field fallback)."""
    return os.environ.get("AESTER_STRICT_KERNEL", "buffered") != "reference"


def sha256(path: Path, check=None) -> str:
    """``check`` (P1c F-2, the parallel path only): called before every 8 MiB read, once on a ledger hit."""
    path.open  # a value without Path.open fails as before (AttributeError), before any hashing
    return hashing.sha256_file(path, source="registered_support.sha256", check=check)


def require_engine(engine: dict) -> None:
    executable = Path(engine.get("executable") or "")
    if engine.get("version") != "1.4.4" or not executable.is_file() or sha256(executable) != SIRIL_SHA256:
        raise ValueError("严格插值支持仅验过本项目固定的 Siril 1.4.4 构建；请改用 legacy 或先验证新引擎")
    for name, digest in INTERPOLATION_DLLS.items():
        dependency = executable.parent / name
        if not dependency.is_file() or sha256(dependency) != digest:
            raise ValueError(f"严格插值支持的引擎依赖未验证或已改变：{name}")


def save_array(path: Path, array: np.ndarray, check=None) -> dict:
    """Atomic sidecar publication; the owner controls these job-local names. ``check`` probes the read-back digest
    (P1c F-2, the parallel path only); np.save and its fsync are not interruptible."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not hashing.digest_writes():   # outside a hash run scope or with the ledger off: the former write
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        with hashing.writing(path):
            os.replace(temporary, path)
        return {"file": path.name, "sha256": sha256(path, check), "shape": list(array.shape),
                "dtype": str(array.dtype)}
    # Same bytes as np.save into temporary.open("wb") (flushed and fsynced); the digest of the
    # bytes written is recorded for the final name, so the read-back below is a ledger hit.
    digest = hashing.write_with_digest(temporary, lambda stream: np.save(stream, array, allow_pickle=False),
                                       source="registered_support.save_array", fsync=True)
    with hashing.writing(path) as block:
        os.replace(temporary, path)
        if digest is not None:
            block.written(digest, "registered_support.save_array")
    return {"file": path.name, "sha256": sha256(path, check), "shape": list(array.shape), "dtype": str(array.dtype)}


def load_array(work: Path, descriptor: dict) -> np.ndarray:
    name = descriptor["file"]
    if not isinstance(name, str) or Path(name).name != name or name not in (COUNTS, STACK_VALID, NATIVE_VALID):
        raise ValueError("invalid support sidecar name")
    path = work / name
    raw = path.read_bytes()
    # The parsed bytes are the hashed bytes: kept in memory on purpose (no ledger).
    if hashing.sha256_bytes(raw, source="registered_support.load_array") != descriptor["sha256"]:
        raise ValueError("插值支持侧车已改变；请新建作业")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if list(array.shape) != descriptor["shape"] or str(array.dtype) != descriptor["dtype"]:
        raise ValueError("support sidecar geometry or dtype mismatch")
    return array


def load_frame_support(work: Path, descriptor: dict) -> np.ndarray:
    """Read a retained independent Boolean support map, never infer it from RGB."""
    if not isinstance(descriptor, dict):
        raise ValueError("逐帧独立有效性缺失，请新建有限值作业")
    name = descriptor.get("file")
    if not isinstance(name, str) or not re.fullmatch(r"frame_support_light_\d+\.npy", name):
        raise ValueError("invalid frame support sidecar name")
    raw = (work / name).read_bytes()
    if hashing.sha256_bytes(raw, source="registered_support.load_frame_support") != descriptor.get("sha256"):
        raise ValueError("逐帧独立有效性已改变")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if (array.dtype != np.bool_ or array.ndim != 2 or not all(array.shape)
            or list(array.shape) != descriptor.get("shape") or str(array.dtype) != descriptor.get("dtype")):
        raise ValueError("frame support must be a nonempty matching bool image")
    return array


def support_at(table: np.ndarray, points: np.ndarray, mapping, *, shape: tuple[int, int]) -> np.ndarray:
    """Conservative full dependency support, including the clamp cross."""
    height, width = shape
    result = np.ones(len(points), dtype=bool)
    for dx, dy in CROSS:
        target = points + (dx, dy)
        # Output-boundary clamp decisions use padded neighbours. Reject those
        # dependencies rather than making a claim about their extension.
        inside = (target[:, 0] >= 0) & (target[:, 0] < width) & (target[:, 1] >= 0) & (target[:, 1] < height)
        mapped = mapping(target)
        finite = np.isfinite(mapped).all(axis=1)
        valid = np.zeros(len(points), dtype=bool)
        keep = inside & finite
        valid[keep] = rectangle_support(table, mapped[keep], before=4, after=5)
        result &= valid
    return result


def registered_mask(validity: np.ndarray, h_stored: np.ndarray, model=None, *, mapping=None, table=None, anchors=None, rows=None, check=None) -> np.ndarray:
    if type(validity) is not np.ndarray or validity.ndim != 2 or validity.dtype != np.bool_ or not all(validity.shape):
        raise ValueError("validity must be a nonempty bool image")
    h = np.asarray(h_stored, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("registration matrix must be finite and invertible")
    height, width = validity.shape
    if rows is None:
        # Keep the coordinate and gather working set small. Full-width Z8 rows
        # benchmark faster than large tiles without changing any arithmetic.
        rows = max(1, min(64, 8192 // width))
    if type(rows) is not int or rows < 1:
        raise ValueError("rows must be positive")
    if anchors is not None and table is not None:
        raise ValueError("provide anchors or integral table, not both")
    if anchors is None:
        table = invalid_integral(validity) if table is None else table
        anchors = build_anchor_grid(table, check=check)
    if type(anchors) is not np.ndarray or anchors.dtype != np.bool_ or anchors.shape != validity.shape:
        raise ValueError("anchor support must be bool and match source geometry")
    if mapping is None:
        if model is not None:
            raise ValueError("SIP support requires an actual-WCS export mapping, not an ideal radial model")
        mapping = make_export_mapping(h, np.eye(3), width, height)
    output = np.zeros_like(validity)
    for y0 in range(0, height, rows):
        if check:
            check()
        yy, xx = np.mgrid[y0:min(y0 + rows, height), :width]
        points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
        mapped = mapping(points)
        finite = np.isfinite(mapped).all(axis=1)
        accepted = np.zeros(len(points), dtype=bool)
        accepted[finite] = anchor_support(anchors, mapped[finite])
        output[y0:y0 + rows] = accepted.reshape(yy.shape)
    # Compute each source map once; the clamp decision uses a cross in OUTPUT
    # coordinates. This is exactly the same intersection as five queries.
    centre = output.copy()
    output[0, :] = output[-1, :] = False
    output[:, 0] = output[:, -1] = False
    output[1:-1, 1:-1] &= (centre[:-2, 1:-1] & centre[2:, 1:-1]
                           & centre[1:-1, :-2] & centre[1:-1, 2:])
    return output


def native_mask(stack_valid: np.ndarray, model, *, rows=64, check=None, quick=None, workers: int = 1,
                pool_stats: dict | None = None, unit_bytes: int | None = None) -> np.ndarray:
    """Propagate support after stack aggregation; contributors may differ.

    Zero cubic weights are excluded and negative weights are included. Edge
    clamping matches the production native cubic, after first-stage strict
    boundary rejection. No attempt is made to equate this with an exposure.

    quick=None follows AESTER_STRICT_KERNEL. With quick decisions, footprints whose 16 clamped taps are all
    valid (or all invalid) are decided without evaluating weights; every other point goes through the
    unchanged sample_support. The result is identical either way.

    Efficiency P1c G-4t: the quick route's row blocks run on frame_pool with ``workers`` threads (only core passes
    more than one; ``unit_bytes`` is its memory figure, ``pool_stats`` the pool's diagnostics for the timings note).
    The tap map is built in prepare(0), still after the first check(); a unit is the former loop body, unchanged; the
    coordinator places each block's rows in block order. ``workers=1`` is the former loop's order of calls. On the
    parallel path a unit also probes after the model and before each chunk of PROBE_POINTS mixed footprints (R18: a
    block whose every footprint is mixed takes 0.42 s at W=8280 on the development PC, 0.32 s of it in sample_support).
    The reference route stays serial (it needs the 366 MB integral table) and runs no pool.
    """
    if type(stack_valid) is not np.ndarray or stack_valid.ndim != 2 or stack_valid.dtype != np.bool_ or not all(stack_valid.shape):
        raise ValueError("stack support must be bool HxW")
    if type(rows) is not int or rows < 1:
        raise ValueError("rows must be positive")
    if model is None:
        return stack_valid.copy()
    if not (_kernel_enabled() if quick is None else quick):
        return _native_mask_reference(stack_valid, model, rows=rows, check=check)
    height, width = stack_valid.shape
    result = np.zeros_like(stack_valid)
    starts = list(range(0, height, rows))
    code = [None]                               # prepare's own state (R11): the tap map, built once

    def prepare(i, y0, _check):
        if code[0] is None:                     # after the first check(), like the reference's first work
            code[0] = support_kernel.tap_state(stack_valid).reshape(-1)
        return y0, code[0]

    def block(payload, unit):
        y0, flat = payload
        probe = unit.check if frame_pool.in_pool() else None     # R18: probes the serial path never had
        yy, xx = np.mgrid[y0:min(y0 + rows, height), :width]
        points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
        mapped = model.apply(points)            # unchanged inputs, block shape and call
        if probe is not None:
            probe()
        accepted = support_kernel.native_block(stack_valid, mapped, flat, height, width, len(points), check=probe)
        if accepted is None:                    # unusual model output: the reference decision, same errors
            finite = np.isfinite(mapped).all(axis=1)
            accepted = np.zeros(len(points), dtype=bool)
            accepted[finite] = support_kernel.sample_support_probed(stack_valid, mapped[finite], probe)
        return accepted.reshape(yy.shape)

    def place(i, rows_accepted):                # by block number, in the coordinator (R1: disjoint rows)
        result[starts[i]:starts[i] + rows] = rows_accepted

    frame_pool.run_ordered(block, starts, workers=workers, name="native_mask", check=check, prepare=prepare,
                           on_result=place, unit_bytes=unit_bytes, blas=False, stats=pool_stats)
    return result


def _native_mask_reference(stack_valid: np.ndarray, model, *, rows=64, check=None) -> np.ndarray:
    """The pre-P1a native_mask body after its validations (test oracle and AESTER_STRICT_KERNEL=reference)."""
    height, width = stack_valid.shape
    result = np.zeros_like(stack_valid)
    for y0 in range(0, height, rows):
        if check:
            check()
        yy, xx = np.mgrid[y0:min(y0 + rows, height), :width]
        points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
        mapped = model.apply(points)
        finite = np.isfinite(mapped).all(axis=1)
        accepted = np.zeros(len(points), dtype=bool)
        accepted[finite] = sample_support(stack_valid, mapped[finite], kernel="cubic", boundary="clamp")
        result[y0:y0 + rows] = accepted.reshape(yy.shape)
    return result


def export_transforms(sequence_path: Path) -> dict:
    from .siril_sequence import read_seq
    sequence = read_seq(sequence_path)
    images = sequence["images"]
    matrices = sequence["regdata"].get(1, [])
    if len(matrices) != len(images):
        raise ValueError("support requires all sequence registration entries")
    out = {}
    for image, reg in zip(images, matrices):
        if not image.get("selected", True):
            continue
        h = np.asarray(reg.get("H"), dtype=np.float64)
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("selected frame has no valid export matrix")
        out[image["stem"]] = h
    if not out:
        raise ValueError("no selected export frames")
    return out


def _kernel_slot() -> dict:
    """A frame-pool state of prepare_exports: one worker's ExportMaskKernel (its buffers are per instance)."""
    return {"kernel": None}


def _drop_kernel(state) -> None:
    if state is not None:
        state["kernel"] = None              # its SIP grid views go with it before release_distortion_grid()


def prepare_exports(work: Path, source, model, *, check=None, progress=None, retain_frame_support=False,
                    workers: int = 1, pool_stats: dict | None = None, unit_bytes: int | None = None) -> dict:
    """Clear unsupported exported pixels before stacking, and count eligibility.

    The snapshot is made BEFORE Siril export, since Siril rewrites matrices
    at reduced text precision. Counts describe pre-rejection geometry only.
    Source files are job-local derivatives, never original RAWs.

    Efficiency P1c F-2 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.2): the frames run on frame_pool with ``workers``
    threads (core decides them, rule R22; ``unit_bytes`` is its memory figure and ``pool_stats`` receives the pool's
    diagnostics for the timings note only).
    - prepare (the coordinator, in frame order): the frame's validity, its anchor grid (one per distinct validity
      array, as before), the export mapping and its parameters; prepare(0) also notes the first frame's geometry
      and reports progress 1.
    - a unit (a worker; one ExportMaskKernel per worker, from init): the support map, the geometry check against the
      first frame (same error, same place: after the map, before the export is touched), the zeroing of the export
      (the fused pass, or the former digest then in-place write), the retained map's .npy and the digest after.
    - on_result (the coordinator, in frame order): ``counts += supported`` (uint16), the source-validity digest by
      the former rule, the frame's record, the kernel count, the last frame's mapping contract, and progress i + 2.
    ``workers=1`` is the serial loop of _prepare_exports_reference with the same calls per frame; on the parallel path
    the export, map and read-back digests and the SIP grid wait also probe the unit's check (R18). A per-frame mask
    source's decode in prepare is one call without a probe (STRICT_MASK_DECODE_BYTES_PER_PIXEL). The record is the
    same for any ``workers`` and is published only after every frame succeeded. A failure is the one of the lowest
    frame; frames after it may already be zeroed in place and their .npy written (prereg §5 item 2): every attempt
    without a stack checkpoint deletes and re-exports them.
    """
    from .sky_validity import apply_validity_to_fits, apply_validity_with_digests
    start = time.monotonic()
    transforms = export_transforms(work / EXPORT_SEQUENCE)
    from .siril_sequence import read_seq
    sequence = read_seq(work / EXPORT_SEQUENCE)
    reference_h = sequence["regdata"][1][sequence["header"]["reference_image"]]["H"]
    wcs_header = None
    distortion_lines = [line.strip() for line in (work / EXPORT_SEQUENCE).read_text(encoding="utf-8").splitlines()
                        if line.strip().startswith("D")]
    if distortion_lines != (["D1 2 distortion.wcs"] if model is not None else []):
        raise ValueError("support requires the stated no-distortion or D1 2 distortion.wcs export contract")
    if model is not None:
        from astropy.io import fits
        wcs_header = fits.getheader(work / "distortion.wcs")
    expected = {"r_" + stem for stem in transforms}
    actual = {p.stem for p in work.glob("r_light_*.fit")}
    if actual != expected:
        raise ValueError("导出帧与支持序列选择不一致")
    if len(transforms) > 65535:
        raise ValueError("too many support frames")
    items = list(transforms.items())
    total = len(items)
    buffered = _kernel_enabled()                         # P1a: one read per call (A/B switch)
    plan = {"validity": None, "anchors": None, "shape": None}           # prepare's own state (R11)
    totals = {"counts": None, "digested": None, "valid_sha256": None,   # on_result's, in frame order
              "kernel_frames": 0, "contract": None}
    frames = []

    def prepare(i, item, own_check):
        stem, h = item
        if i == 0 and progress:
            progress(1, total, stem)
        valid = source.for_frame(stem)
        if valid is not plan["validity"]:
            plan["anchors"] = (anchor_grid_from_validity(valid, check=own_check) if buffered
                               else build_anchor_grid(invalid_integral(valid), check=own_check))
            plan["validity"] = valid
        mapping = make_export_mapping(h, reference_h, valid.shape[1], valid.shape[0], wcs_header)
        params = getattr(mapping, "export_parameters", None) if buffered else None
        if plan["shape"] is None:
            plan["shape"] = valid.shape                  # the first frame's geometry (R4)
        return stem, h, valid, plan["anchors"], mapping, params, plan["shape"]

    def export(payload, unit):
        stem, h, valid, anchors, mapping, params, first_shape = payload
        probe = unit.check if frame_pool.in_pool() else None     # R18: probes the serial path never had
        if params is not None:
            kernel = unit.state["kernel"] = support_kernel.kernel_for(unit.state["kernel"], params)
            supported = support_kernel.registered_mask_kernel(valid, h, params=params, anchors=anchors,
                                                              kernel=kernel, check=unit.check, grid_check=probe)
        else:   # AESTER_STRICT_KERNEL=reference, or a caller-supplied mapping without export_parameters
            supported = registered_mask(valid, h, model, mapping=mapping, anchors=anchors, check=unit.check)
        if valid.shape != first_shape:
            raise ValueError("source validity geometry changed")
        path = work / f"r_{stem}.fit"
        fused = hashing.digest_writes()
        if fused:
            # One pass reads the export (digest before), zeroes the unsupported samples and writes them
            # back (digest after, recorded as written: the sha256(path) below is a ledger hit).
            info, before_sha, _ = apply_validity_with_digests(path, supported, check=unit.check)
        else:   # outside a hash run scope or with the ledger off: the former digest, then the in-place write
            before_sha = sha256(path, probe)
            info = apply_validity_to_fits(path, supported, check=unit.check)
        explicit = ({"explicit_validity": save_array(work / f"frame_support_{stem}.npy", supported, probe)}
                    if retain_frame_support else {})
        entry = {"supported_pixels": int(supported.sum()), "export_before_sha256": before_sha,
                 "export_after_sha256": sha256(path, probe), **info, **explicit}
        return stem, h, valid, supported, fused, params is not None, mapping.contract, entry

    def deliver(i, done):
        stem, h, valid, supported, fused, kernelled, contract, entry = done
        if totals["counts"] is None:
            totals["counts"] = np.zeros(valid.shape, dtype=np.uint16)
        totals["counts"] += supported
        if valid is not totals["digested"] or not fused:   # the source maps are never modified here: one digest per array
            totals["valid_sha256"] = hashing.sha256_bytes(valid.tobytes(), source="registered_support.source_valid")
            totals["digested"] = valid
        frames.append({"stem": stem, "source_valid_sha256": totals["valid_sha256"], "H_stored": h.tolist(), **entry})
        totals["kernel_frames"] += kernelled
        totals["contract"] = contract
        if progress and i + 1 < total:
            progress(i + 2, total, items[i + 1][0])

    try:
        # inflight=1: the pool raises it to the workers it grants, i.e. one frame in flight per worker (§3.2)
        frame_pool.run_ordered(export, items, workers=workers, name="strict_support", check=check, prepare=prepare,
                               on_result=deliver, init=_kernel_slot, finalize=_drop_kernel, inflight=1,
                               unit_bytes=unit_bytes, stats=pool_stats)
    finally:
        # F-5a: nothing later in the job reads the SIP grid (366 MB at 45.7 MP); success, error or Cancelled. The
        # pool has finalized every worker's kernel before it returns or raises.
        release_distortion_grid()
    counts = totals["counts"]
    implementation = SUPPORT_IMPLEMENTATION if totals["kernel_frames"] == len(frames) else REFERENCE_IMPLEMENTATION
    record = {"version": SUPPORT_VERSION, "implementation": implementation,
              "mode": MODE, "state": "exports_filtered",
              "export_command": EXPORT_COMMAND, "engine_sha256": SIRIL_SHA256,
              "interpolation_dll_sha256": dict(INTERPOLATION_DLLS),
              "sequence_sha256": sha256(work / EXPORT_SEQUENCE), "shape": list(counts.shape),
              "wcs_sha256": sha256(work / "distortion.wcs") if model is not None else None,
              "mapping_contract": totals["contract"],
              "model": None if model is None else {key: getattr(model, key) for key in ("k1", "k2", "cx", "cy", "scale", "magnification", "direction")},
              "support_rule": "five_output_centres_each_strict_10x10_source_rectangle",
              "boundary": "strict (no reflected source samples certified)",
              "counts_semantics": "fully_supported_input_frame_eligibility_before_normalization_and_rejection_not_effective_exposure",
              "eligible": save_array(work / COUNTS, counts), "frames": frames,
              "filter_seconds": time.monotonic() - start}
    atomic_json(work / RECORD, record)
    return record


def _prepare_exports_reference(work: Path, source, model, *, check=None, progress=None,
                               retain_frame_support=False) -> dict:
    """The pre-P1c serial prepare_exports (P1a/P1b body, unchanged): the test oracle of F-2."""
    from .sky_validity import apply_validity_to_fits, apply_validity_with_digests
    start = time.monotonic()
    transforms = export_transforms(work / EXPORT_SEQUENCE)
    from .siril_sequence import read_seq
    sequence = read_seq(work / EXPORT_SEQUENCE)
    reference_h = sequence["regdata"][1][sequence["header"]["reference_image"]]["H"]
    wcs_header = None
    distortion_lines = [line.strip() for line in (work / EXPORT_SEQUENCE).read_text(encoding="utf-8").splitlines()
                        if line.strip().startswith("D")]
    if distortion_lines != (["D1 2 distortion.wcs"] if model is not None else []):
        raise ValueError("support requires the stated no-distortion or D1 2 distortion.wcs export contract")
    if model is not None:
        from astropy.io import fits
        wcs_header = fits.getheader(work / "distortion.wcs")
    expected = {"r_" + stem for stem in transforms}
    actual = {p.stem for p in work.glob("r_light_*.fit")}
    if actual != expected:
        raise ValueError("导出帧与支持序列选择不一致")
    if len(transforms) > 65535:
        raise ValueError("too many support frames")
    counts, old_validity, anchors = None, None, None
    digested, valid_sha256 = None, None     # source_valid_sha256 of the last validity array (one array for a static mask)
    frames = []
    buffered = _kernel_enabled()                         # P1a: one read per call (A/B switch)
    kernel, kernel_frames = None, 0
    try:
        for number, (stem, h) in enumerate(transforms.items(), 1):
            if check:
                check()
            if progress:
                progress(number, len(transforms), stem)
            valid = source.for_frame(stem)
            if valid is not old_validity:
                anchors = (anchor_grid_from_validity(valid, check=check) if buffered
                           else build_anchor_grid(invalid_integral(valid), check=check))
                old_validity = valid
            mapping = make_export_mapping(h, reference_h, valid.shape[1], valid.shape[0], wcs_header)
            params = getattr(mapping, "export_parameters", None) if buffered else None
            if params is not None:
                kernel = support_kernel.kernel_for(kernel, params)
                supported = support_kernel.registered_mask_kernel(valid, h, params=params, anchors=anchors,
                                                                  kernel=kernel, check=check)
                kernel_frames += 1
            else:   # AESTER_STRICT_KERNEL=reference, or a caller-supplied mapping without export_parameters
                supported = registered_mask(valid, h, model, mapping=mapping, anchors=anchors, check=check)
            if counts is None:
                counts = np.zeros(valid.shape, dtype=np.uint16)
            elif counts.shape != valid.shape:
                raise ValueError("source validity geometry changed")
            path = work / f"r_{stem}.fit"
            fused = hashing.digest_writes()
            if fused:
                # One pass reads the export (digest before), zeroes the unsupported samples and writes them
                # back (digest after, recorded as written: the sha256(path) below is a ledger hit).
                info, before_sha, _ = apply_validity_with_digests(path, supported, check=check)
            else:   # outside a hash run scope or with the ledger off: the former digest, then the in-place write
                before_sha = sha256(path)
                info = apply_validity_to_fits(path, supported, check=check)
            counts += supported
            explicit = ({"explicit_validity": save_array(work / f"frame_support_{stem}.npy", supported)}
                        if retain_frame_support else {})
            if valid is not digested or not fused:   # the source maps are never modified here: one digest per array
                valid_sha256 = hashing.sha256_bytes(valid.tobytes(), source="registered_support.source_valid")
                digested = valid
            frames.append({"stem": stem, "source_valid_sha256": valid_sha256,
                           "H_stored": h.tolist(), "supported_pixels": int(supported.sum()),
                           "export_before_sha256": before_sha, "export_after_sha256": sha256(path), **info, **explicit})
    finally:
        # F-5a: nothing later in the job reads the SIP grid (366 MB at 45.7 MP); success, error or Cancelled.
        kernel = None
        release_distortion_grid()
    implementation = SUPPORT_IMPLEMENTATION if kernel_frames == len(frames) else REFERENCE_IMPLEMENTATION
    record = {"version": SUPPORT_VERSION, "implementation": implementation,
              "mode": MODE, "state": "exports_filtered",
              "export_command": EXPORT_COMMAND, "engine_sha256": SIRIL_SHA256,
              "interpolation_dll_sha256": dict(INTERPOLATION_DLLS),
              "sequence_sha256": sha256(work / EXPORT_SEQUENCE), "shape": list(counts.shape),
              "wcs_sha256": sha256(work / "distortion.wcs") if model is not None else None,
              "mapping_contract": mapping.contract,
              "model": None if model is None else {key: getattr(model, key) for key in ("k1", "k2", "cx", "cy", "scale", "magnification", "direction")},
              "support_rule": "five_output_centres_each_strict_10x10_source_rectangle",
              "boundary": "strict (no reflected source samples certified)",
              "counts_semantics": "fully_supported_input_frame_eligibility_before_normalization_and_rejection_not_effective_exposure",
              "eligible": save_array(work / COUNTS, counts), "frames": frames,
              "filter_seconds": time.monotonic() - start}
    atomic_json(work / RECORD, record)
    return record


@contextmanager
def _untimed(name):
    """finish_stack's ``operation`` when the caller has no timer."""
    yield


def finish_stack(work: Path, stack_path: Path, model, *, check=None, explicit_counts=False, native_workers: int = 1,
                 native_unit_bytes: int | None = None, pool_stats: dict | None = None, operation=None) -> dict:
    """Conservative available-value mask; exact zero stays explicitly ambiguous.

    Siril uses exact zero as nodata and clips its default float stack output.
    A geometric input count therefore is not a rejection-survivor count. We
    certify only finite nonzero RGB outputs where eligible inputs existed.
    Natural zeros and negative/clipped results are marked unavailable for this
    conservative composite, never asserted to be absent source observations.

    Efficiency P1c G-4t: native_mask runs inside ``operation("native_mask")`` (core passes its timer's ``operation``,
    so that the timer's sampler sees the pool) with ``native_workers`` threads, ``native_unit_bytes`` each, its pool
    diagnostics in ``pool_stats`` (left empty on the reference route, which runs no pool).
    """
    operation = operation or _untimed
    record = json.loads((work / RECORD).read_text(encoding="utf-8"))
    counts = load_array(work, record["eligible"])
    # The native stage follows the label prepare_exports recorded, so the record names both stages' path.
    quick = record.get("implementation") == SUPPORT_IMPLEMENTATION
    if explicit_counts:
        from . import finite_stack
        supported = finite_stack.counted_support(work, stack_path, counts, check or (lambda: None))
        with operation("native_mask"):
            native = native_mask(supported, model, check=check, quick=quick, workers=native_workers,
                                 pool_stats=pool_stats, unit_bytes=native_unit_bytes)
        record.update(state="complete", stack_file=stack_path.name, stack_sha256=sha256(stack_path),
                      stack_valid=save_array(work / STACK_VALID, supported), native_valid=save_array(work / NATIVE_VALID, native),
                      ambiguous_zero_or_nonfinite_pixels=0,
                      stack_supported_pixels=int(supported.sum()), native_supported_pixels=int(native.sum()),
                      effective_exposure="not_measured; per-channel survivor counts are not inverse variance or exposure seconds",
                      count_files={name: sha256(work / name) for name in (finite_stack.NVALID, finite_stack.NUSED)},
                      zero_semantics="finite zero/negative output remains available wherever every channel has nused > 0")
        atomic_json(work / RECORD, record)
        return record
    header, offset = fits_crop.read_header(stack_path)
    width, height, planes, bitpix, order = fits_crop._geometry(header)
    if counts.shape != (height, width) or planes != 3 or bitpix != -32:
        raise ValueError("stack support geometry/type mismatch")
    mm = np.memmap(stack_path, dtype=">f4", mode="r", offset=offset, shape=(3, height, width))
    supported = counts > 0
    ambiguous = 0
    try:
        view = mm[:, ::-1, :] if order == "BOTTOM-UP" else mm
        for y0 in range(0, height, 64):
            if check:
                check()
            tile = view[:, y0:y0 + 64]
            present = np.isfinite(tile).all(axis=0) & (tile != 0).all(axis=0)
            ambiguous += int(np.count_nonzero(supported[y0:y0 + 64] & ~present))
            supported[y0:y0 + 64] &= present
    finally:
        del view, mm
    with operation("native_mask"):
        native = native_mask(supported, model, check=check, quick=quick, workers=native_workers,
                             pool_stats=pool_stats, unit_bytes=native_unit_bytes)
    record.update(state="complete", stack_file=stack_path.name, stack_sha256=sha256(stack_path),
                  stack_valid=save_array(work / STACK_VALID, supported), native_valid=save_array(work / NATIVE_VALID, native),
                  ambiguous_zero_or_nonfinite_pixels=ambiguous,
                  stack_supported_pixels=int(supported.sum()), native_supported_pixels=int(native.sum()),
                  effective_exposure="not_measured; eligibility is not an inverse variance or rejection-survivor count",
                  zero_semantics="zero output with geometric eligibility is conservatively unavailable; natural zero, normalization-zero and clipping are not disambiguated")
    atomic_json(work / RECORD, record)
    return record


def read_native(work: Path, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    record = json.loads((work / RECORD).read_text(encoding="utf-8"))
    if record.get("version") != SUPPORT_VERSION or record.get("mode") != MODE or record.get("state") != "complete":
        raise ValueError("插值支持记录缺失或未完成；请新建作业")
    if record.get("native_master_sha256") != sha256(work / "sky_master.fit"):
        raise ValueError("天空母版与插值支持记录不一致")
    array = load_array(work, record["native_valid"])
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError("native support must match full image geometry")
    return array, record
