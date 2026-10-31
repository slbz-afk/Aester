"""Opt-in fixed-pedestal Siril RCD experiment; never a default RAW backend.

The caller supplies one fixed DN interval for the whole comparison/sequence.
Encoding to [0, 1] avoids Siril's FITS >10 unit conversion. Even, reflected
32-pixel padding preserves Bayer phase; two outer-corner anchors force the
wrapper's extrema to 0 and 1 even for constant frames. No frame extrema are
used to choose the interval. Anchor locality must be verified experimentally.

This is NOT RCD applied equivalently to signed signals: adding a pedestal
changes RCD's ratios and its fixed epsilon. Original measured CFA samples are
explicitly restored after decoding; the pre-restoration error and reconstructed
limit counts are retained so that restoration cannot conceal numerical changes.
The interval constrains inputs only: finite reconstruction overshoots survive
inverse mapping and are counted; nonfinite values or float32 overflow fail.
This module neither applies WB nor promises that WB commutes with RCD.

Input arrays are TOP-DOWN, already black-subtracted float32 DN. This processing
history is a caller contract, not something an array can prove. All file output
is retained in unique experiment directories, including failed engine output.
The timeout governs the guarded Siril run; preprocessing and file I/O are not
interruptible native jobs. A caller callback may raise siril.Cancelled.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
import uuid

import numpy as np
from astropy.io import fits

from . import core, siril
from .float_demosaic import rgb_pattern


ALGORITHM = "experimental-fixed-pedestal-siril-rcd-v1"
PAD = 32
COMMANDS = ["setext fit", "set32bits", "setcompress 0", "set debayer.interpolation=8",
            "set debayer.use_bayer_header=true", "set debayer.orientation=0",
            "set debayer.offset_x=0", "set debayer.offset_y=0", "set gui.icc_autoassignment=0",
            "set gui.icc_autoconversion=0", "setcpu 1", "cd stage", "convert rgb -debayer -out=../rgb"]


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} 必须为有限实数")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须为有限实数") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须为有限实数")
    return number


def _bounds(lower_dn, upper_dn):
    lower = _finite_number(lower_dn, "lower_dn")
    upper = _finite_number(upper_dn, "upper_dn")
    span = upper - lower
    if not lower < upper or not math.isfinite(span) or span <= 0:
        raise ValueError("固定 DN 区间必须 lower < upper 且跨度有限")
    return lower, upper, span


def _validate_cfa(cfa, pattern):
    if (not isinstance(cfa, np.ndarray) or cfa.dtype != np.float32 or cfa.ndim != 2
            or min(cfa.shape) < 2 or cfa.size > 120_000_000):
        raise ValueError("需要 TOP-DOWN 二维 float32 CFA，边长至少 2，最多 1.2 亿采样")
    # Reject mixed boolean lists before numpy can turn True into Bayer index 1.
    flat = np.asarray(pattern, dtype=object)
    if flat.shape != (2, 2) or any(isinstance(v, (bool, np.bool_)) for v in flat.flat):
        raise ValueError("需要真实整数索引的 2×2 Bayer 阵列")
    return rgb_pattern(pattern)


def _array_sha(array):
    digest = hashlib.sha256()
    for y in range(0, array.shape[0], 128):
        digest.update(np.asarray(array[y:y+128], dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def _summary(array):
    finite = np.isfinite(array)
    if not finite.all():
        return {"shape": list(array.shape), "nonfinite": int(array.size-np.count_nonzero(finite))}
    return {"shape": list(array.shape), "nonfinite": 0, "minimum": float(array.min()),
            "maximum": float(array.max()), "below_zero": int(np.count_nonzero(array < 0)),
            "above_one": int(np.count_nonzero(array > 1)),
            "at_zero": int(np.count_nonzero(array == 0)), "at_one": int(np.count_nonzero(array == 1))}


def prepare_fixed_range(cfa, pattern, *, lower_dn, upper_dn):
    """Pure preparation helper: return padded encoded CFA and its fixed plan."""
    p = _validate_cfa(cfa, pattern)
    lower, upper, span = _bounds(lower_dn, upper_dn)
    before = _array_sha(cfa)
    encoded = np.empty(cfa.shape, np.float32)
    maximum_roundtrip_error = 0.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for y in range(0, cfa.shape[0], 128):
                tile = cfa[y:y+128].astype(np.float64)
                if not np.isfinite(tile).all() or np.any(tile < lower) or np.any(tile > upper):
                    raise ValueError("CFA 包含非有限值或超出调用者固定 DN 区间；不会自动改范围")
                encoded[y:y+128] = ((tile-lower)/span).astype(np.float32)
                error = np.abs(encoded[y:y+128].astype(np.float64)*span+lower-tile)
                maximum_roundtrip_error = max(maximum_roundtrip_error, float(error.max()))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("固定 DN 编码发生数值溢出") from exc
    if not np.isfinite(encoded).all() or np.any(encoded < 0) or np.any(encoded > 1):
        raise ValueError("固定 DN 编码无法安全表示为 [0,1] float32")
    if _array_sha(cfa) != before:
        raise ValueError("准备期间输入 CFA 已变化")
    padded = np.pad(encoded, PAD, mode="reflect")
    padded[0, 0], padded[0, 1] = np.float32(0), np.float32(1)
    plan = {"algorithm": ALGORITHM, "row_order": "TOP-DOWN", "channel_order": "RGB",
            "input_state": "caller_asserted_black_subtracted_camera_CFA_DN",
            "input_shape": list(cfa.shape), "input_sha256_float32_le": before,
            "pattern_reported": np.asarray(pattern).tolist(), "pattern_rgb": p.tolist(),
            "green_mapping": "CFA_slot_3_explicitly_maps_to_RGB_green_1",
            "lower_dn": lower, "upper_dn": upper, "span_dn": span,
            "per_frame_range_estimation": False, "padding": {"pixels": PAD, "mode": "reflect", "even": True},
            "anchors": [{"y": 0, "x": 0, "encoded_value": 0}, {"y": 0, "x": 1, "encoded_value": 1}],
            "maximum_encoding_roundtrip_error_DN": maximum_roundtrip_error,
            "signed_RCD_equivalence_claimed": False, "WB_applied": False,
            "measured_CFA_restoration_required": True}
    return padded, plan


def unpack_fixed_range(encoded_rgb, cfa, pattern, *, lower_dn, upper_dn):
    """Pure decode helper; HWC padded RGB must already be TOP-DOWN float32."""
    p = _validate_cfa(cfa, pattern)
    lower, upper, span = _bounds(lower_dn, upper_dn)
    expected = (cfa.shape[0]+2*PAD, cfa.shape[1]+2*PAD, 3)
    if not isinstance(encoded_rgb, np.ndarray) or encoded_rgb.dtype != np.float32 or encoded_rgb.shape != expected:
        raise ValueError("RCD 输出必须匹配已填充画幅的 float32 HWC RGB")
    if not np.isfinite(encoded_rgb).all():
        raise ValueError("RCD 输出包含非有限值；保留文件供检查")
    crop = encoded_rgb[PAD:-PAD, PAD:-PAD]
    result = np.empty(crop.shape, np.float32)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for y in range(0, crop.shape[0], 128):
                result[y:y+128] = (crop[y:y+128].astype(np.float64)*span+lower).astype(np.float32)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("RCD 逆映射无法安全表达为 float32 DN") from exc
    if not np.isfinite(result).all():
        raise ValueError("RCD 逆映射含非有限值")
    if not np.isfinite(cfa).all() or np.any(cfa < lower) or np.any(cfa > upper):
        raise ValueError("待还原的 CFA 无效或已超固定区间")
    errors, changed, limit_low, limit_high, outside_low, outside_high = 0., 0, 0, 0, 0, 0
    for yy in range(2):
        for xx in range(2):
            channel = int(p[yy, xx])
            original = cfa[yy::2, xx::2]
            measured = result[yy::2, xx::2, channel]
            errors = max(errors, float(np.max(np.abs(measured.astype(np.float64)-original.astype(np.float64)))))
            changed += int(np.count_nonzero(measured.view(np.uint32) != original.view(np.uint32)))
            for other in range(3):
                if other != channel:
                    reconstructed = crop[yy::2, xx::2, other]
                    limit_low += int(np.count_nonzero(reconstructed == 0))
                    limit_high += int(np.count_nonzero(reconstructed == 1))
                    outside_low += int(np.count_nonzero(reconstructed < 0))
                    outside_high += int(np.count_nonzero(reconstructed > 1))
            measured[...] = original  # Intentional, explicitly recorded sample restoration.
    diagnostics = {"measured_sites_before_restoration": {"count": int(cfa.size),
                       "maximum_absolute_error_DN": errors, "bitwise_different_count": changed},
                   "measured_sites_restored_bitwise": True,
                   "reconstructed_limits_before_restoration": {"count": int(2*cfa.size),
                       "at_lower_dn": limit_low, "at_upper_dn": limit_high,
                       "below_lower_dn": outside_low, "above_upper_dn": outside_high,
                       "at_or_below_lower_dn": limit_low+outside_low,
                       "at_or_above_upper_dn": limit_high+outside_high,
                       "output_clipping_applied": False},
                   "cropped_encoded_before_restoration": _summary(crop),
                   "returned_sha256_float32_le": _array_sha(result)}
    return result, diagnostics


def _retained_artifacts(directory):
    artifacts = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "experiment.json":
            artifacts[path.relative_to(directory).as_posix()] = {
                "path": str(path), "size_bytes": path.stat().st_size, "sha256": core._sha(path)}
    return artifacts


def rcd_fixed_range(cfa, pattern, output_root, *, lower_dn, upper_dn, engine=None,
                    timeout_seconds=120, cancel_check=None) -> tuple[np.ndarray, dict]:
    """Run an isolated fixed-pedestal RCD experiment, retaining success/failure evidence.

    engine may be an executable path or a doctor result (its path is rechecked).
    No output array parameter is supported: input storage is never reused.
    """
    timeout = _finite_number(timeout_seconds, "timeout_seconds")
    if timeout <= 0:
        raise ValueError("timeout_seconds 必须为正")
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check 必须为回调函数")
    if cancel_check:
        cancel_check()
    encoded, plan = prepare_fixed_range(cfa, pattern, lower_dn=lower_dn, upper_dn=upper_dn)
    if isinstance(engine, dict):
        engine = engine.get("executable")
        if not engine:
            raise ValueError("引擎记录缺少可执行文件")
    if engine is not None and not isinstance(engine, (str, Path)):
        raise ValueError("engine 必须为 Siril 路径或 doctor 记录")
    status = siril.doctor(str(engine) if engine is not None else None)
    if status.get("available") is not True:
        raise RuntimeError(status.get("message", "Siril 引擎不可用"))
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root/f"rcd-fixed-{uuid.uuid4().hex}"
    directory.mkdir()
    stage = directory/"stage"
    stage.mkdir()
    record = {"schema_version": 1, "state": "running", "directory": str(directory),
              "created_at": core.now(), "recipe": plan, "engine": status,
              "commands": list(COMMANDS), "timeout_seconds": timeout, "artifacts": {},
              "limitations": ["固定 pedestal 改变 RCD 比值，非 signed RCD 等价实现",
                              "锚点的局部独立性和真实星点画质需独立验收",
                              "恢复已测 CFA 点不证明未测通道正确，必须检查恢复前误差",
                              "TOP-DOWN 与已扣黑 DN 由调用者保证；此函数不自动 WB"]}
    manifest = directory/"experiment.json"
    core.atomic_json(manifest, record)
    started = time.monotonic()
    try:
        source = stage/"input.fit"
        bayer = "".join("RGB"[v] for row in plan["pattern_rgb"] for v in row)
        header = fits.Header({"BAYERPAT": bayer, "ROWORDER": "TOP-DOWN", "XBAYROFF": 0,
                              "YBAYROFF": 0, "DATAMIN": 0., "DATAMAX": 1.})
        fits.PrimaryHDU(encoded, header).writeto(source)
        source_hash = core._sha(source)
        record["encoded_input"] = {**_summary(encoded), "sha256": source_hash, "path": str(source)}
        record["engine_sha256"] = core._sha(Path(status["executable"]))
        deadline = time.monotonic()+timeout

        def check():
            if cancel_check:
                cancel_check()
            if time.monotonic() >= deadline:
                raise TimeoutError("固定范围 RCD 实验超过 Siril 执行时限")

        siril.run_script(status["executable"], directory, "rcd", list(COMMANDS), check)
        check()
        candidates = list((directory/"rgb").glob("rgb_*.fit"))
        if len(candidates) != 1:
            raise RuntimeError(f"RCD 应产生一个 RGB FITS，实际 {len(candidates)}；见保留的 rcd.log")
        output_path = candidates[0]
        with fits.open(output_path, memmap=False, do_not_scale_image_data=True) as hdul:
            h = hdul[0].header
            a = hdul[0].data
            if (h.get("BITPIX") != -32 or h.get("BSCALE", 1) != 1 or h.get("BZERO", 0) != 0
                    or a is None or a.shape != (3, *encoded.shape)):
                raise ValueError("RCD FITS 的浮点类型、单位或画幅不符合固定编码约定")
            row_order = h.get("ROWORDER")
            if row_order not in {"TOP-DOWN", "BOTTOM-UP"}:
                raise ValueError("RCD FITS 缺少明确行方向")
            data = np.asarray(a[:, ::-1] if row_order == "BOTTOM-UP" else a, dtype=np.float32)
            rgb = np.moveaxis(data, 0, -1)
            record["encoded_output"] = {**_summary(rgb), "path": str(output_path),
                                        "sha256": core._sha(output_path), "file_row_order": row_order}
            result, diagnostics = unpack_fixed_range(rgb, cfa, pattern, lower_dn=lower_dn, upper_dn=upper_dn)
        record["diagnostics"] = diagnostics
        record["input_unchanged"] = _array_sha(cfa) == plan["input_sha256_float32_le"]
        record["encoded_input_unchanged"] = core._sha(source) == source_hash
        if not record["input_unchanged"] or not record["encoded_input_unchanged"]:
            raise ValueError("实验期间输入 CFA 或编码 FITS 已变化")
        record["state"] = "succeeded"
        return result, record
    except BaseException as exc:
        record["state"] = "cancelled" if isinstance(exc, siril.Cancelled) else "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        record["artifacts"] = _retained_artifacts(directory)
        core.atomic_json(manifest, record)
