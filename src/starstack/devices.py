"""Device identification, support tiers and the ``inspect-device`` report (docs/PRE_RELEASE_CLOSURE_20260922.md D).

Nothing here applies camera-specific handling: the product's RAW path is the same for every camera, and this module
only says what has been verified for a given body.  ``identify_camera`` reads container tags only (EXIF Make/Model and
Software of TIFF-container RAWs — ARW / NEF / DNG / CR2 / PEF — plus the Sony ``SonyRawFileType`` tag and the tags of the
page carrying it) and of CR3 (the TIFF IFD0 in the Canon ``CMT1`` box); containers it cannot parse (RW2, ORF, RAF)
are reported as unidentified rather than guessed.
``inspect_device``'s metadata probe goes through LibRaw for non-DNG files, which unpacks the RAW data (no demosaic, the
file is never modified).
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import struct

from . import devices_registry as reg

VERSION = "devices-1"
SONY_RAW_FILE_TYPE_TAG = 0x7000
SONY_RAW_FILE_TYPES = {0: "uncompressed_14bit", 1: "uncompressed_12bit", 2: "lossy_compressed",
                       3: "lossless_compressed", 4: "lossless_compressed_2", 6: "lossy_compressed_2"}
# Registry variant family of each SonyRawFileType.  A family name is a label, not a verification: what was actually
# checked is recorded per body in ``checked_subvariants`` and matched exactly by ``subvariant_check``.
REGISTRY_VARIANT_OF_TYPE = {0: "uncompressed", 1: "uncompressed", 2: "lossy_compressed", 6: "lossy_compressed",
                            3: "lossless_compressed", 4: "lossless_compressed"}
PHOTOMETRIC_CFA = 32803
UNIQUE_CAMERA_MODEL_TAG = 50708
FITS_SUFFIXES = {".fit", ".fits", ".fts"}


def _clean(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value).strip().strip("\x00").strip()
    return text or None


def _norm(text: str | None) -> str:
    return " ".join((text or "").upper().split())


def identify_camera(path) -> dict:
    """EXIF make/model from the file container only (no decoding).  Never raises for an unreadable file."""
    path = Path(path)
    info = {"path": str(path), "container": None, "make": None, "model": None, "unique_model": None, "software": None,
            "sony_raw_file_type": None, "sony_raw_variant": None, "registry_variant": None, "raw_ifd": None,
            "compression": None, "identified": False, "note": None}
    suffix = path.suffix.lower()
    if suffix in FITS_SUFFIXES:
        try:
            from astropy.io import fits
            header = fits.getheader(path)
        except Exception as exc:      # unreadable FITS: report, don't guess
            info["note"] = f"FITS 头无法读取：{type(exc).__name__}"
            return info
        info["container"] = "fits"
        creator = _clean(header.get("CREATOR")) or _clean(header.get("INSTRUME")) or _clean(header.get("TELESCOP"))
        info["model"] = creator
        info["make"] = creator.split(" ")[0] if creator else None
        info["identified"] = bool(creator)
        info["note"] = None if creator else "FITS 头没有 CREATOR / INSTRUME / TELESCOP"
        return info
    if suffix == ".cr3":
        try:
            tiff = _cr3_cmt1(path)
        except (OSError, ValueError, struct.error) as exc:
            info["note"] = f"CR3 容器无法解析（{type(exc).__name__}）：机型未识别，不猜测"
            return info
        if tiff is None:
            info["note"] = "CR3 容器里没有 CMT1（IFD0）：机型未识别，不猜测"
            return info
        try:
            import io
            import logging
            import tifffile
            log = logging.getLogger("tifffile")
            level = log.level
            log.setLevel(logging.CRITICAL)       # CMT1 is a bare IFD0 without image data; tifffile logs errors about that
            try:
                with tifffile.TiffFile(io.BytesIO(tiff)) as tf:
                    tags = tf.pages[0].tags
                    values = {key: _clean(tags.get(name).value if tags.get(name) else None)
                              for key, name in (("make", "Make"), ("model", "Model"), ("software", "Software"))}
            finally:
                log.setLevel(level)
            info["container"] = "cr3"
            info.update(values)
        except Exception as exc:
            info["note"] = f"CR3 的 CMT1 无法解析（{type(exc).__name__}）：机型未识别，不猜测"
            return info
        info["identified"] = bool(info["make"] or info["model"])
        if not info["identified"]:
            info["note"] = "CR3 的 CMT1 里没有 Make / Model 标签"
        return info
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            first = tf.pages[0]
            info["container"] = "tiff"
            info["make"] = _clean(first.tags.get("Make").value if first.tags.get("Make") else None)
            info["model"] = _clean(first.tags.get("Model").value if first.tags.get("Model") else None)
            info["software"] = _clean(first.tags.get("Software").value if first.tags.get("Software") else None)
            unique = first.tags.get(UNIQUE_CAMERA_MODEL_TAG)
            info["unique_model"] = _clean(unique.value) if unique is not None else None
            compression = first.tags.get("Compression")
            info["compression"] = int(compression.value) if compression is not None else None
            pages = list(tf.pages)
            for page in pages:
                if page.pages is not None:
                    pages.extend(page.pages)
            for page in pages:
                tag = page.tags.get(SONY_RAW_FILE_TYPE_TAG)
                if tag is not None:
                    value = tag.value
                    value = int(value[0]) if isinstance(value, (tuple, list)) else int(value)
                    info["sony_raw_file_type"] = value
                    info["sony_raw_variant"] = SONY_RAW_FILE_TYPES.get(value, f"unknown_{value}")
                    info["registry_variant"] = REGISTRY_VARIANT_OF_TYPE.get(value)
                    photometric = page.tags.get("PhotometricInterpretation")
                    width, length = page.tags.get("ImageWidth"), page.tags.get("ImageLength")
                    info["raw_ifd"] = {"photometric": int(photometric.value) if photometric is not None else None,
                                       "width": int(width.value) if width is not None else None,
                                       "height": int(length.value) if length is not None else None}
                    break
    except Exception as exc:
        info["note"] = f"非 TIFF 容器或无法解析（{type(exc).__name__}）：机型未识别，不猜测"
        return info
    info["identified"] = bool(info["make"] or info["model"])
    if not info["identified"]:
        info["note"] = "TIFF 容器里没有 Make / Model 标签"
    return info


CR3_CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")


def _boxes(stream, start: int, end: int):
    """ISO BMFF boxes (kind, payload start, payload end) between ``start`` and ``end``; headers only are read."""
    pos = start
    while pos + 8 <= end:
        stream.seek(pos)
        header = stream.read(16)
        size, kind = struct.unpack(">I4s", header[:8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", header[8:16])[0], 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            raise ValueError("CR3 box 长度无效")
        yield kind, pos + head, pos + size
        pos += size


def _cr3_cmt1(path) -> bytes | None:
    """The TIFF IFD0 (``CMT1``) inside a CR3's Canon ``uuid`` box in ``moov``; None when absent."""
    with open(path, "rb") as stream:
        stream.seek(0, 2)
        total = stream.tell()
        for kind, start, end in _boxes(stream, 0, total):
            if kind != b"moov":
                continue
            for inner, s2, e2 in _boxes(stream, start, end):
                if inner != b"uuid":
                    continue
                stream.seek(s2)
                if stream.read(16) != CR3_CANON_UUID:
                    continue
                for leaf, s3, e3 in _boxes(stream, s2 + 16, e2):
                    if leaf == b"CMT1":
                        if e3 - s3 > (1 << 20):
                            raise ValueError("CMT1 过大")
                        stream.seek(s3)
                        return stream.read(e3 - s3)
            return None
    return None


def lookup(make: str | None, model: str | None) -> dict | None:
    """The registry entry whose EXIF model id equals ``model`` after whitespace/case folding.  Exact only: a substring
    rule would send e.g. an unregistered "Canon EOS R6m3" to "Canon EOS R" (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md B)."""
    wanted = _norm(model)
    if not wanted:
        return None
    for entry in reg.REGISTRY:
        if wanted in {_norm(m) for m in entry["model_ids"]}:
            return dict(entry)
    return None


def tier_of(entry: dict | None) -> str:
    return entry["tier"] if entry else "unregistered"


def _metadata_report(path: Path, suffix: str) -> tuple[dict, dict]:
    """(metadata availability, colour-basis attempt) without ever raising."""
    from . import master_export as me
    metadata = {"route": None, "available": False, "status": None}
    basis = {"state": "not_attempted", "error": None}
    try:
        if suffix == ".dng":
            from .raw_audit import read_dng_tags
            tags = read_dng_tags(path)
            metadata.update(route="dng_tags", status=tags.get("status"), available=tags.get("status") == "ok",
                            errors=tags.get("errors", []), raw_shape=tags.get("raw_shape"))
            if metadata["available"]:
                b = me.colour_basis_from_dng(path)
                basis = {"state": "ok", "error": None, "black_DN": b["black_DN"], "white_DN": b["white_DN"],
                         "as_shot_gains_rgb": b["as_shot_gains_rgb"], "daylight_gains_rgb": b["daylight_gains_rgb"]}
        elif suffix in FITS_SUFFIXES:
            from .fits_input import describe_header
            from astropy.io import fits
            described = describe_header(dict(fits.getheader(path)))
            metadata.update(route="fits_header", status="ok", available=True, layout=described["layout"],
                            shape_hw=described["shape_hw"], bayer=described["bayer"], roworder=described["interpreted_roworder"])
            basis = {"state": "not_applicable", "error": None, "note": "FITS 输入没有相机颜色底子（作业保持旧产物）"}
        else:
            from .raw_metadata import read_raw_metadata
            meta = read_raw_metadata(path)
            metadata.update(route="libraw", status=meta.get("status"), available=bool(meta.get("available")),
                            versions=meta.get("versions"))
            if metadata["available"]:
                sizes = meta["fields"]["sizes"]
                metadata["shape_hw"] = ([sizes["value"]["height"], sizes["value"]["width"]] if sizes.get("valid") else None)
                b = me.colour_basis_from_raw(path, metadata=meta)
                basis = {"state": "ok", "error": None, "black_DN": b["black_DN"], "white_DN": b["white_DN"],
                         "as_shot_gains_rgb": b["as_shot_gains_rgb"], "daylight_gains_rgb": b["daylight_gains_rgb"]}
            else:
                basis = {"state": "unavailable", "error": f"metadata {meta.get('status')}"}
    except Exception as exc:
        basis = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
    return metadata, basis


def subvariant_check(entry: dict | None, ident: dict) -> dict:
    """Does this file belong to a sub-variant recorded as checked for its body?  Exact match on container tags only:
    SonyRawFileType, CFA photometric on the page carrying it, RAW IFD size and firmware (IFD0 Software)."""
    if not entry or not entry.get("checked_subvariants"):
        return {"state": "none_registered", "reasons": []}
    raw_ifd = ident.get("raw_ifd") or {}
    attempts = []
    for sub in entry["checked_subvariants"]:
        reasons = []
        if ident.get("sony_raw_file_type") != sub["raw_file_type"]:
            reasons.append(f"压缩类型 {ident.get('sony_raw_file_type')} ≠ 已核对的 {sub['raw_file_type']}")
        if sub.get("cfa") and raw_ifd.get("photometric") != PHOTOMETRIC_CFA:
            reasons.append("RAW 页不是 CFA（可能是无损 M/S 等非 Bayer 变体）")
        if [raw_ifd.get("width"), raw_ifd.get("height")] != list(sub["raw_ifd_wh"]):
            reasons.append(f"RAW IFD 尺寸 {raw_ifd.get('width')}×{raw_ifd.get('height')} ≠ 已核对的 {sub['raw_ifd_wh'][0]}×{sub['raw_ifd_wh'][1]}")
        if ident.get("software") != sub["firmware"]:
            reasons.append(f"固件 {ident.get('software')} ≠ 已核对的 {sub['firmware']}")
        if not reasons:
            return {"state": "checked", "subvariant": {k: sub[k] for k in ("name", "raw_file_type", "firmware", "evidence") if k in sub},
                    "reasons": []}
        attempts.append({"subvariant": sub.get("name"), "reasons": reasons})
    return {"state": "unchecked", "reasons": [r for a in attempts for r in a["reasons"]], "attempts": attempts}


def inspect_device(path) -> dict:
    """Identify the body, look it up, probe the metadata/colour-basis route and say which tier applies and why.

    For non-DNG RAWs the metadata probe reads LibRaw metadata, which unpacks the RAW data (no demosaic, file unchanged)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("需要一个存在的 RAW / DNG / FITS 文件")
    ident = identify_camera(path)
    entry = lookup(ident["make"], ident["model"]) if ident["identified"] else None
    if entry is None and ident.get("unique_model"):
        entry = lookup(None, ident["unique_model"])
    suffix = path.suffix.lower()
    metadata, basis = _metadata_report(path, suffix)
    tier = tier_of(entry)
    sub = subvariant_check(entry, ident)
    report = {"version": VERSION, "registry_version": reg.REGISTRY_VERSION, **ident,
              "registered": entry is not None, "tier": tier, "tier_meaning": reg.TIERS.get(tier, "未登记：按解码器支持列表处理，本产品未核对"),
              "registry": entry, "metadata": metadata, "colour_basis": basis, "subvariant_check": sub,
              "next_tier": reg.NEXT.get(tier, reg.NEXT["B"]) if entry else "登记 + " + reg.NEXT["B"],
              "statement": None}
    if entry and tier == "B" and entry.get("checked_subvariants"):
        report["tier_meaning"] = "B 级：已有真实文件核对过的子变体，但按用户决定分级保持 B（见 next_note）"
        if entry.get("next_note"):          # the body-specific upgrade rule replaces the generic NEXT["B"] text
            report["next_tier"] = entry["next_note"]
        report["statement"] = (f"{entry['name']}：B 级——已用真实文件核对过部分子变体，按用户决定分级保持 B。"
                               + (" " + entry["next_note"] if entry.get("next_note") else ""))
    elif entry:
        report["statement"] = (f"{entry['name']}：{tier} 级——{reg.TIERS[tier]}。" + (" " + entry["next_note"] if entry.get("next_note") else ""))
    else:
        report["statement"] = "机型未登记：产品按解码器（LibRaw / Siril）自己的支持列表处理，本产品没有为它核对过任何东西。"
    if sub["state"] == "checked":
        report["statement"] += (f" 本文件属于已核对子变体（{sub['subvariant'].get('name')}，证据：{sub['subvariant'].get('evidence')}）；"
                                f"分级不因此改变。")
    elif sub["state"] == "unchecked":
        report["statement"] += " 本文件的变体未核对：" + "；".join(sub["reasons"]) + "。"
    if basis["state"] == "ok" and tier == "B":
        report["statement"] += " 本文件的颜色底子可建立（这不是 B+：B+ 需要与厂商规格逐项核对并记录）。"
    look = (entry or {}).get("colour_look")
    if look:
        report["statement"] += (f" 成片默认观感：{look['id']}，强度 {look['strength']:g}"
                                + ("（借用其他机型的颜色表，不是本机专用配置）" if look.get("borrowed") else "") + "。")
    return report


def registry_listing() -> dict:
    return {"version": VERSION, "registry_version": reg.REGISTRY_VERSION, "tiers": reg.TIERS, "next": reg.NEXT,
            "devices": [{k: e[k] for k in ("make", "model_ids", "name", "container", "tier", "cfa", "bits", "megapixels", "shape_hw",
                                            "shape_source", "raw_variants", "quirks", "evidence", "verified_frames",
                                            "checked_subvariants", "public_samples", "next_note", "colour_look")
                         if k in e} for e in reg.REGISTRY]}


def registry_problems() -> list[str]:
    """Self-check used by the tests and the CLI: required fields, tier values, unique model ids."""
    problems, seen = [], set()
    required = ("make", "model_ids", "name", "container", "cfa", "bits", "megapixels", "shape_hw", "shape_source",
                "raw_variants", "tier", "evidence", "verified_frames", "quirks", "public_samples")
    for entry in reg.REGISTRY:
        for key in required:
            if key not in entry:
                problems.append(f"{entry.get('name')}: 缺 {key}")
        if entry.get("tier") not in reg.TIERS:
            problems.append(f"{entry.get('name')}: 分级 {entry.get('tier')!r} 无效")
        for model in entry.get("model_ids", []):
            if _norm(model) in seen:
                problems.append(f"{model}: 重复登记")
            seen.add(_norm(model))
        if entry.get("tier") == "A" and not entry.get("verified_frames"):
            problems.append(f"{entry.get('name')}: A 级必须有 verified_frames")
        if entry.get("tier") == "B" and entry.get("verified_frames"):
            problems.append(f"{entry.get('name')}: B 级不应有 verified_frames（核对过就是 B+）")
        if entry.get("tier") == "B+" and not entry.get("verified_frames"):
            problems.append(f"{entry.get('name')}: B+ 级必须有 verified_frames")
        for sub in entry.get("checked_subvariants", []):
            if not isinstance(sub, dict):
                problems.append(f"{entry.get('name')}: checked_subvariants 项不是对象")
                continue
            missing = [k for k in ("name", "raw_file_type", "cfa", "raw_ifd_wh", "firmware", "evidence", "frames") if k not in sub]
            if missing:
                problems.append(f"{entry.get('name')}: checked_subvariants 缺 {missing}")
                continue
            frames = sub["frames"]
            if (not isinstance(frames, list) or not frames
                    or any(not isinstance(f, dict) or not isinstance(f.get("sha256"), str)
                           or not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in frames)):
                problems.append(f"{entry.get('name')}: checked_subvariants 需要带 SHA-256 的核对帧")
        for name, variant in entry.get("raw_variants", {}).items():
            if variant.get("decoder") not in ("declared", "verified", "workaround"):
                problems.append(f"{entry.get('name')}/{name}: decoder 状态无效")
        if "colour_look" in entry:
            from . import camera_looks
            look = entry["colour_look"]
            try:
                if not isinstance(look, dict) or look.get("id") not in camera_looks.LOOKS or not look.get("decided"):
                    raise ValueError("colour_look 需要已登记的观感 id 与用户决定记录")
                camera_looks.check_strength(look.get("strength"))
            except ValueError as exc:
                problems.append(f"{entry.get('name')}: {exc}")
    return problems


def to_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
