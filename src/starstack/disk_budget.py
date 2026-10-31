"""Disk-space gate for ``submit_job`` by frame count and frame geometry (docs/HARDENING_PACKAGE_20260922.md H3.3).

The previous gate multiplied the *source bytes* by a fixed factor, which under-estimates the float intermediates
Siril writes for compressed RAW (the real 8-frame Z8 job: 0.42 GB of sources, 8.2 GB of ``work/``).  This module
estimates from what the job will actually write, measured on that job (RUN_LOG §34):

* converted frames ``work/light_*.fit``: RAW input → 16-bit RGB (8280 × 5520 × 3 × 2 B + 2880 = 274 236 480 B measured);
  FITS input → float32 RGB (existing float frames are kept as float; the CFA adaptation writes float32);
* registered frames ``work/r_light_*.fit``: float32 RGB (548 470 080 B measured);
* a fixed number of full-frame float volumes for the masters and products (sky / foreground / undistorted masters,
  composite, M, D, the transient composite and its M/D);
* the staged copies of the originals (``stage/``);
* stack-range mode: up to ``max_attempts`` scaled copies of the registered frames (``range-*/trial*/``).

When the first frame's geometry cannot be read (no rawpy, unreadable file) the legacy source-bytes estimate is
used and the record says so.  The estimate is deliberately on the safe side; the refusal message carries the numbers.
"""
from __future__ import annotations

from pathlib import Path

VERSION = "disk-estimate-2"
BYTES_PER_PIXEL_FLOAT_RGB = 12       # 3 channels × float32
BYTES_PER_PIXEL_RAW_CONVERTED = 6    # 3 channels × 16-bit (Siril convertraw -debayer of a 16-bit RAW, measured)
FIXED_FRAME_VOLUMES = 8              # masters + products, in full-frame float volumes (measured 4.4 GB on the Z8 job)
SAFETY_FACTOR = 1.1
MARGIN_BYTES = 512 * 1024 ** 2
LEGACY_FACTOR = {"raw": 20, "fits": 20, "cfa": 35}
LEGACY_RANGE_FACTOR = 40


def frame_shape_hw(path, kind: str, fits_contract: dict | None = None) -> tuple[int, int] | None:
    """(height, width) of one input frame from its metadata only, or None when it cannot be read.

    FITS: the inspected contract's header geometry.  DNG: the mosaic IFD geometry (tifffile).  Other RAW: LibRaw's
    reported visible size through ``raw_metadata`` (needs rawpy); reading LibRaw metadata unpacks the RAW data (no
    demosaic, the file is never modified).
    """
    path = Path(path)
    if kind == "fits":
        try:
            height, width = fits_contract["frames"][0]["shape_hw"]
            return int(height), int(width)
        except (TypeError, KeyError, IndexError, ValueError):
            return None
    if path.suffix.lower() == ".dng":
        try:
            from .raw_audit import read_dng_tags
            tags = read_dng_tags(path)
            if tags.get("status") == "ok":
                height, width = tags["raw_shape"]
                return int(height), int(width)
        except Exception:      # any unreadable DNG: fall back to the legacy estimate rather than refusing the job
            return None
        return None
    try:
        from .raw_metadata import read_raw_metadata
        meta = read_raw_metadata(path)
        sizes = meta["fields"]["sizes"]
        if meta.get("available") and sizes.get("valid") and not sizes.get("issues"):
            return int(sizes["value"]["height"]), int(sizes["value"]["width"])
    except Exception:
        return None
    return None


def estimate(sources: dict, fits_contract: dict | None, *, stack_range_mode: str = "legacy", range_attempts: int = 3) -> dict:
    """Bytes the job is expected to write under its directory, with the basis recorded (JSON-ready)."""
    kind = sources["kind"]
    frames = int(sources["count"])
    source_bytes = int(sources["bytes"])
    cfa = bool(fits_contract and fits_contract.get("layout") == "cfa")
    shape = frame_shape_hw(sources["files"][0], kind, fits_contract) if sources.get("files") else None
    range_mode = stack_range_mode != "legacy"
    record = {"version": VERSION, "frames": frames, "source_bytes": source_bytes, "input_kind": kind, "cfa": cfa,
              "stack_range_mode": stack_range_mode, "safety_factor": SAFETY_FACTOR, "margin_bytes": MARGIN_BYTES}
    if shape is None:
        factor = LEGACY_FACTOR["cfa" if cfa else kind] + (LEGACY_RANGE_FACTOR if range_mode else 0)
        required = source_bytes * factor + MARGIN_BYTES
        record.update(basis="legacy_source_bytes", reason="first frame geometry unreadable from metadata",
                      legacy_factor=factor, frame_shape_hw=None, required_bytes=int(required))
        return record
    height, width = shape
    pixels = height * width
    float_volume = pixels * BYTES_PER_PIXEL_FLOAT_RGB
    converted = pixels * (BYTES_PER_PIXEL_RAW_CONVERTED if kind == "raw" else BYTES_PER_PIXEL_FLOAT_RGB)
    per_frame = converted + float_volume
    intermediates = frames * per_frame
    fixed = FIXED_FRAME_VOLUMES * float_volume
    range_copies = frames * float_volume * range_attempts if range_mode else 0
    raw_total = intermediates + fixed + source_bytes + range_copies
    required = int(raw_total * SAFETY_FACTOR + MARGIN_BYTES)
    record.update(basis="frames_x_frame_volume", frame_shape_hw=[height, width], frame_pixels=pixels,
                  converted_frame_bytes=converted, registered_frame_bytes=float_volume, per_frame_bytes=per_frame,
                  intermediate_bytes=intermediates, fixed_frame_volumes=FIXED_FRAME_VOLUMES, fixed_bytes=fixed,
                  range_trial_bytes=range_copies, range_attempts=range_attempts if range_mode else 0,
                  unfactored_bytes=int(raw_total), required_bytes=required)
    return record


def _gb(value: int) -> str:
    return f"{value / 1e9:.2f} GB"


def shortfall_message(record: dict, free_bytes: int) -> str:
    """The refusal text: the numbers of the estimate, so the user can decide what to free or where to run."""
    need = record["required_bytes"]
    if record["basis"] == "legacy_source_bytes":
        detail = (f"原片 {_gb(record['source_bytes'])} × {record['legacy_factor']} + 余量 {_gb(record['margin_bytes'])}"
                  f"（首帧尺寸无法从元数据读出，按旧的原片字节估算）")
    else:
        h, w = record["frame_shape_hw"]
        detail = (f"{record['frames']} 帧 × 单帧 {_gb(record['per_frame_bytes'])}（{w}×{h}：转换帧 {_gb(record['converted_frame_bytes'])}"
                  f" + 配准帧 {_gb(record['registered_frame_bytes'])}）= {_gb(record['intermediate_bytes'])}，"
                  f"母版与产物约 {record['fixed_frame_volumes']} 帧浮点体积 {_gb(record['fixed_bytes'])}，原片副本 {_gb(record['source_bytes'])}"
                  + (f"，范围保护试验副本 {_gb(record['range_trial_bytes'])}" if record["range_trial_bytes"] else "")
                  + f"；× 系数 {record['safety_factor']} + 余量 {_gb(record['margin_bytes'])}")
    return (f"可用磁盘空间不足：结果文件夹所在磁盘可用 {_gb(free_bytes)}，本作业估算需要 {_gb(need)}（{detail}）。"
            f"请清理空间或换一个结果文件夹；已完成的旧任务可在“设置 → 作品与缓存”中清理中间文件（命令行：starstack gc）")
