"""Read-only decoded-CFA audit, with an explicitly bounded float derivative.

Decoded codes are not claimed to be untouched sensor ADC measurements. DNG
file tags and LibRaw-reported metadata remain separate evidence sources.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
import tifffile

from . import core, __version__
from .raw_metadata import read_raw_metadata


_SMALL_TAGS = (
    "DNGVersion", "DNGBackwardVersion", "ImageWidth", "ImageLength", "BitsPerSample",
    "Compression", "PhotometricInterpretation", "SamplesPerPixel", "CFARepeatPatternDim",
    "CFAPattern", "CFAPlaneColor", "BlackLevelRepeatDim", "BlackLevel", "WhiteLevel",
    "ActiveArea", "DefaultCropOrigin", "DefaultCropSize", "Orientation", "AsShotNeutral",
    "ColorMatrix1", "ColorMatrix2", "ForwardMatrix1", "ForwardMatrix2",
    "CameraCalibration1", "CameraCalibration2", "AnalogBalance",
    "CalibrationIlluminant1", "CalibrationIlluminant2", "BaselineExposure",
    "BaselineNoise", "NoiseReductionApplied", "LossyImageCompression",
)
_LARGE_TAGS = ("LinearizationTable", "BlackLevelDeltaH", "BlackLevelDeltaV",
               "OpcodeList1", "OpcodeList2", "OpcodeList3", "DNGPrivateData")
_IDENTITY_TAGS = {"DNGVersion", "DNGBackwardVersion"}
_RAW_IFD_TAGS = {
    "ImageWidth", "ImageLength", "BitsPerSample", "Compression",
    "PhotometricInterpretation", "SamplesPerPixel", "CFARepeatPatternDim",
    "CFAPattern", "CFAPlaneColor", "BlackLevelRepeatDim", "BlackLevel",
    "WhiteLevel", "ActiveArea", "DefaultCropOrigin", "DefaultCropSize",
}


def _plain(value):
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite file tag")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Unsupported file tag representation")


def _tag_record(tag, ifd_offset: int) -> dict:
    result = {"source": "file_tag", "ifd_offset": ifd_offset,
              "type": int(tag.dtype), "count": int(tag.count), "value": _plain(tag.value)}
    if int(tag.dtype) in (5, 10):
        values = np.asarray(tag.value, dtype=np.float64).reshape(-1)
        if len(values) != 2 * tag.count or np.any(values[1::2] == 0):
            raise ValueError("Invalid rational file tag")
        rational = values[::2] / values[1::2]
        if not np.isfinite(rational).all():
            raise ValueError("Non-finite rational file tag")
        result["numeric"] = rational.tolist()
    return result


def read_dng_tags(path: Path) -> dict:
    """Read selected public tags, preserving rational numerator/denominator pairs.

    Opaque/private payloads are not interpreted or copied into reports. EXIF
    capture time is kept as reported, with an unknown timezone unless provided.
    """
    result = {"status": "not_dng", "tags": {}, "errors": [],
              "capture_time": {"value": None, "timezone": None, "status": "unknown"}}
    if path.suffix.lower() != ".dng":
        return result
    try:
        with tifffile.TiffFile(path) as tf:
            pages = list(tf.pages)
            for page in pages:
                if len(pages) > 64:
                    raise ValueError("Too many DNG IFDs")
                if page.pages is not None:
                    pages.extend(page.pages)
            mosaics = [p for p in pages if int(p.photometric) == 32803]
            if not mosaics:
                raise ValueError("DNG contains no mosaic CFA IFD")
            page = max(mosaics, key=lambda p: p.imagewidth * p.imagelength)
            result.update(status="ok", raw_ifd_offset=int(page.offset),
                          raw_shape=[int(page.imagelength), int(page.imagewidth)])
            # Mosaic numerical tags must belong to its IFD. Color/profile tags
            # can be in IFD 0; do not silently source black levels from a preview.
            for name in _SMALL_TAGS:
                owner = tf.pages[0] if name in _IDENTITY_TAGS else page
                tag = owner.tags.get(name)
                if tag is None and name not in _RAW_IFD_TAGS | _IDENTITY_TAGS:
                    owner = tf.pages[0]
                    tag = owner.tags.get(name)
                if tag is None:
                    continue
                if tag.count > 256:
                    result["errors"].append({"tag": name, "error": "tag_too_large"})
                    continue
                try:
                    result["tags"][name] = _tag_record(tag, int(owner.offset))
                except (TypeError, ValueError, OverflowError) as exc:
                    result["errors"].append({"tag": name, "error": str(exc)})
            for name in _IDENTITY_TAGS:
                item = result["tags"].get(name)
                if item is None:
                    if name == "DNGVersion":
                        result["errors"].append({"tag": name, "error": "required_in_ifd0"})
                    continue
                value = item["value"]
                if (item["type"] != 1 or item["count"] != 4 or not isinstance(value, list)
                        or len(value) != 4 or value[0] != 1
                        or any(type(v) is not int or not 0 <= v <= 255 for v in value)):
                    result["errors"].append({"tag": name, "error": "unsupported_dng_version"})
            for name in _LARGE_TAGS:
                found = [(int(p.offset), p.tags.get(name)) for p in pages if p.tags.get(name) is not None]
                result["tags"][name] = {"source": "file_tag", "present": bool(found),
                    "payload_read": False, "locations": [
                        {"ifd_offset": offset, "type": int(tag.dtype), "count": int(tag.count)}
                        for offset, tag in found]}
            exif = tf.pages[0].tags.get("ExifTag")
            if exif is not None and isinstance(exif.value, dict):
                value = exif.value.get("DateTimeOriginal")
                offset = exif.value.get("OffsetTimeOriginal")
                if isinstance(value, str) and len(value) <= 80:
                    result["capture_time"] = {"value": value, "timezone": offset if isinstance(offset, str) else None,
                        "status": "file_reported", "source": "EXIF.DateTimeOriginal",
                        "clock_accuracy": "unknown"}
            if result["errors"]:
                result["status"] = "partial"
    except Exception as exc:
        result["status"] = "read_error"
        result["errors"].append({"error": str(exc), "type": type(exc).__name__})
    return result


def _numeric(tags: dict, name: str):
    item = tags.get(name)
    if not item or "value" not in item:
        return None
    return np.asarray(item.get("numeric", item["value"]), dtype=np.float64).reshape(-1)


def black_model(dng: dict, shape: tuple[int, int]) -> dict:
    """Select only explicit, full-frame, simple DNG calibration metadata.

    Complex DNG linearization/opcodes remain inspectable without guessing how
    LibRaw applied them. No camera-name/ISO lookup or sky-based black estimate.
    """
    result = {"status": "unsupported", "source": "file_tag", "reasons": []}
    if dng.get("status") != "ok":
        result["reasons"].append("complete_dng_metadata_required")
        if dng.get("errors"):
            result["metadata_errors"] = dng["errors"]
        return result
    tags = dng["tags"]
    for name in ("LinearizationTable", "BlackLevelDeltaH", "BlackLevelDeltaV", "OpcodeList1", "OpcodeList2"):
        if tags.get(name, {}).get("present"):
            result["reasons"].append(f"unverified_{name}")
    black, repeat = _numeric(tags, "BlackLevel"), _numeric(tags, "BlackLevelRepeatDim")
    white, active = _numeric(tags, "WhiteLevel"), _numeric(tags, "ActiveArea")
    if dng.get("raw_shape") != list(shape):
        result["reasons"].append("decoder_and_ifd_shape_mismatch")
    if active is None or not np.array_equal(active, [0, 0, *shape]):
        result["reasons"].append("explicit_full_frame_active_area_required")
    if repeat is None or len(repeat) != 2 or not np.isfinite(repeat).all() or not np.equal(repeat, np.floor(repeat)).all() or np.any(repeat < 1) or np.any(repeat > 2):
        result["reasons"].append("unsupported_black_repeat")
    elif black is None or len(black) != int(np.prod(repeat)) or not np.isfinite(black).all() or np.any(black < 0):
        result["reasons"].append("invalid_black_levels")
    if white is None or len(white) != 1 or not np.isfinite(white).all() or white[0] <= 0:
        result["reasons"].append("scalar_white_level_required")
    elif black is not None and (not np.isfinite(black).all() or np.any(black >= white[0])):
        result["reasons"].append("white_not_above_black")
    if result["reasons"]:
        return result
    return {"status": "available", "source": "file_tag", "black_values": black.tolist(),
            "repeat_shape": repeat.astype(int).tolist(), "white_level": float(white[0]),
            "origin": "full_raw_image_top_left", "interpretation": "DNG_declared_black_not_independently_measured",
            "reasons": []}


def subtract_declared_black(codes: np.ndarray, model: dict) -> np.ndarray:
    """Subtract once from integer decoded codes; keep negative/overwhite samples.

    Deliberately accepts integer codes only. A prior float derivative cannot
    accidentally be passed back for a second black subtraction.
    """
    if codes.ndim != 2 or codes.dtype.kind not in "ui" or codes.dtype.itemsize > 2:
        raise ValueError("Expected a 2D integer decoded CFA, at most 16 bits")
    if model.get("status") != "available":
        raise ValueError("No supported declared black model")
    repeat = tuple(model["repeat_shape"])
    levels = np.asarray(model["black_values"], dtype=np.float64).reshape(repeat)
    if not np.isfinite(levels).all() or np.any(levels < 0):
        raise ValueError("Invalid black levels")
    signal = codes.astype(np.float32)
    for y in range(repeat[0]):
        for x in range(repeat[1]):
            signal[y::repeat[0], x::repeat[1]] -= levels[y, x]
    return signal


def _artifact(path: Path, array: np.ndarray) -> dict:
    return {"path": str(path), "sha256": core._sha(path), "dtype": str(array.dtype),
            "shape": list(array.shape), "size_bytes": path.stat().st_size}


def inspect_raw(source_path: str | Path, output_root: str | Path) -> dict:
    """Create an immutable numerical inspection folder for one RAW.

    No dcraw_process, WB, demosaic, matrix, geometric opcode, or tone mapping
    is requested. Unsupported RAW keeps a failed report without claiming CFA
    extraction. Successful audit may legitimately have no float derivative.
    """
    source = Path(source_path).expanduser().resolve(strict=True)
    root = Path(output_root).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in core.RAW:
        raise ValueError("请选择单张 RAW／CFA DNG 文件")
    if root.is_relative_to(source.parent):
        raise ValueError("RAW 检查结果须保存在原片文件夹（含子目录）之外")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"raw-audit-{uuid.uuid4().hex}"
    directory.mkdir()
    record = {"schema_version": 1, "state": "preparing", "audit_dir": str(directory),
              "implementation": {"starstack_version": __version__, "algorithm": "decoded-cfa-audit-v1",
                  "module_sha256": core._sha(Path(__file__))},
              "created_at": core.now(), "source": {"path": str(source), "sha256": core._sha(source),
                  "size_bytes": source.stat().st_size}, "artifacts": {}, "error": None,
              "camera_pipeline": {"status": "unknown", "inverse_correction_applied": False,
                  "message": "解码采样不等同于未经相机处理的 ADC 数据；未推测或逆转机内处理"},
              "operations_requested": {"unpack": True, "dcraw_process": False, "demosaic": False,
                  "white_balance": False, "color_matrix": False, "auto_brightness": False,
                  "crop": False, "rotation": False, "tone_curve": False},
              "limitations": ["只核查解码后的采样，无法证明相机端信息完整性",
                  "本模块不改变现有 Siril 堆栈路径；原型完整显色仍待完成"]}
    destination = directory / "audit.json"
    core.atomic_json(destination, record)
    try:
        dng = read_dng_tags(source)
        record["file_metadata"] = dng
        record["capture_time"] = dng["capture_time"]
        record["libraw_metadata"] = read_raw_metadata(source)
        rawpy = importlib.import_module("rawpy")
        record["versions"] = {"rawpy": rawpy.__version__, "libraw": list(rawpy.libraw_version),
                              "tifffile": tifffile.__version__, "numpy": np.__version__}
        with rawpy.imread(str(source)) as raw:
            data = raw.raw_image
            if data is None or data.ndim != 2 or data.dtype.kind != "u" or data.dtype.itemsize > 2:
                raise ValueError("此检查入口仅支持二维整数 CFA，未执行显影替代")
            if data.size > 120_000_000:
                raise ValueError("单张检查目前最多支持 1.2 亿 CFA 采样")
            if shutil.disk_usage(root).free < data.nbytes * 4 + 64 * 1024**2:
                raise ValueError("空间不足以保存独立 CFA 证据和浮点衍生")
            codes = np.array(data, copy=True)
            sizes = raw.sizes
            pattern = raw.raw_pattern
            geometry = {"width": int(codes.shape[1]), "height": int(codes.shape[0]),
                        "row_order": "TOP-DOWN", "coordinate_basis": "decoder_full_raw_mosaic",
                        "active_area_libraw": [sizes.top_margin, sizes.left_margin,
                            sizes.top_margin + sizes.height, sizes.left_margin + sizes.width],
                        "orientation_applied": False, "visible_pattern": None,
                        "full_pattern": None, "pattern_source": "libraw_reported"}
            if pattern is not None:
                pattern = np.asarray(pattern)
                if pattern.ndim == 2 and 0 < pattern.size <= 256:
                    # rawpy.raw_pattern is in full raw_image coordinates.
                    # A visible crop advances its origin by the sensor margins.
                    geometry["full_pattern"] = pattern.tolist()
                    visible = np.roll(pattern, (-sizes.top_margin % pattern.shape[0], -sizes.left_margin % pattern.shape[1]), axis=(0, 1))
                    geometry["visible_pattern"] = visible.tolist()
            record["geometry"] = geometry
        path = directory / "raw_codes.npy"
        np.save(path, codes, allow_pickle=False)
        record["artifacts"]["raw_codes"] = _artifact(path, codes)
        record["numeric_summary"] = {"units": "decoder_output_code_values", "sample_count": int(codes.size),
            "minimum": int(codes.min()), "maximum": int(codes.max()), "zeros": int(np.count_nonzero(codes == 0)),
            "unpack_format_specific_processing": "not_independently_verified"}
        model = black_model(dng, codes.shape)
        record["black_model"] = model
        if model["status"] == "available":
            signal = subtract_declared_black(codes, model)
            float_path = directory / "black_subtracted_cfa.npy"
            np.save(float_path, signal, allow_pickle=False)
            record["artifacts"]["black_subtracted_cfa"] = {**_artifact(float_path, signal),
                "units": "decoder_code_values_minus_DNG_declared_black", "black_subtractions": 1,
                "normalization_applied": False, "clipping_applied": False, "white_balance_applied": False,
                "color_space": "CFA_sensor_channels", "demosaic_applied": False}
            record["numeric_summary"].update(negative_after_declared_black=int(np.count_nonzero(signal < 0)),
                at_or_above_declared_white=int(np.count_nonzero(codes >= model["white_level"])),
                minimum_after_declared_black=float(signal.min()), maximum_after_declared_black=float(signal.max()))
        else:
            record["limitations"].append("此文件的浮点扣黑模型未验证，仅保留解码采样")
        if core._sha(source) != record["source"]["sha256"]:
            raise ValueError("检查过程中原片发生变化，未发布成功记录")
        record.update(state="succeeded", completed_at=core.now(), source_unchanged=True)
        core.atomic_json(destination, record)
        return record
    except Exception as exc:
        record.update(state="failed", error=str(exc), completed_at=core.now(), error_type=type(exc).__name__)
        core.atomic_json(destination, record)
        return record
