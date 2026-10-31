"""Optional, read-only RAW metadata inspection; no color processing is applied.

``valid`` describes structural/numerical checks, not camera characterization.
Values are reported by LibRaw and can include its estimates or database values.
Missing or invalid metadata must never be treated as usable color coefficients.
Channel applicability is only a check against the reported CFA indices; it is
not proof of physical calibration, and never supplies missing coefficients.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np


_FIELDS = (
    "sizes", "raw_pattern", "color_desc", "num_colors",
    "black_level_per_channel", "white_level", "camera_white_level_per_channel",
    "camera_whitebalance", "daylight_whitebalance", "color_matrix", "rgb_xyz_matrix",
)
_SHAPES = {
    "num_colors": [], "white_level": [],
    "black_level_per_channel": [4], "camera_white_level_per_channel": [4],
    "camera_whitebalance": [4], "daylight_whitebalance": [4],
    "color_matrix": [3, 4], "rgb_xyz_matrix": [4, 3],
}
_CHANNEL_FIELDS = (
    "black_level_per_channel", "camera_white_level_per_channel",
    "camera_whitebalance", "daylight_whitebalance",
)
_WB_FIELDS = ("camera_whitebalance", "daylight_whitebalance")
_LIBRAW_GREEN_REFERENCE = (
    "https://github.com/LibRaw/LibRaw/blob/0.22.1/"
    "src/postprocessing/postprocessing_utils_dcrdefs.cpp#L170-L187"
)


def _unknown_applicability(reason="channel_usage_not_checked"):
    return {"status": "unknown", "reason": reason,
            "used_slots": None, "unused_slots": None}


def _json_value(value):
    """Copy metadata into strict JSON values, retaining invalid positions as null."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if hasattr(value, "_asdict"):
        return _json_value(value._asdict())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _record(name):
    result = {
        "value": None, "status": "missing", "valid": False,
        "finite": None, "shape": None, "shape_valid": None,
        "expected_shape": _SHAPES.get(name), "source": "libraw_reported",
        "issues": [],
    }
    if name in _CHANNEL_FIELDS or name == "white_level":
        result["applicability"] = _unknown_applicability()
    if name in _WB_FIELDS:
        result["white_balance_semantics"] = {
            "status": "unknown", "reason": "metadata_unavailable",
            "message": "白平衡通道信息尚未确认；尚未应用",
            "rgb_coefficients": None, "rgb_coefficients_valid": None,
            "cfa_to_coefficient_slots": None, "mapping_applied": False,
            "file_white_balance_tags_verified": False,
        }
    return result


def _inspect_field(name, value):
    result = _record(name)
    if value is None:
        result["issues"] = ["not_reported"]
        return result
    issues = result["issues"]
    if name == "color_desc":
        try:
            text = value.decode("ascii") if isinstance(value, bytes) else value
            if not isinstance(text, str):
                raise TypeError("color_desc must be ASCII text or bytes")
            result["value"] = text
            if not text or len(text) > 4 or not text.isascii():
                issues.append("invalid_color_description")
        except (UnicodeError, TypeError) as exc:
            issues.append(type(exc).__name__)
    elif name == "sizes":
        values = value._asdict() if hasattr(value, "_asdict") else value
        if not isinstance(values, dict):
            raise TypeError("sizes must be a named tuple or mapping")
        result["value"] = _json_value(values)
        numeric = [v for v in values.values() if isinstance(v, (int, float, np.number))]
        result["finite"] = all(math.isfinite(float(v)) for v in numeric)
        if not result["finite"]:
            issues.append("non_finite")
        for key in ("raw_width", "raw_height", "width", "height"):
            v = values.get(key)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0:
                issues.append(f"invalid_{key}")
        for key in ("top_margin", "left_margin"):
            v = values.get(key)
            if v is not None and (isinstance(v, (bool, np.bool_)) or
                                  not isinstance(v, (int, np.integer)) or v < 0):
                issues.append(f"invalid_{key}")
        for key in ("iwidth", "iheight"):
            v = values.get(key)
            if v is not None and (isinstance(v, (bool, np.bool_)) or
                                  not isinstance(v, (int, np.integer)) or v <= 0):
                issues.append(f"invalid_{key}")
        aspect = values.get("pixel_aspect")
        if aspect is not None and (isinstance(aspect, (bool, np.bool_)) or
                                   not isinstance(aspect, (int, float, np.number)) or
                                   not math.isfinite(float(aspect)) or aspect <= 0):
            issues.append("invalid_pixel_aspect")
        if not issues:
            for margin, side, full in (("left_margin", "width", "raw_width"),
                                       ("top_margin", "height", "raw_height")):
                if values.get(margin, 0) + values[side] > values[full]:
                    issues.append(f"{side}_outside_raw_area")
    else:
        array = np.asarray(value)
        result["shape"] = list(array.shape)
        if name == "raw_pattern":
            result["expected_shape"] = "nonempty 2D pattern, each side <=16"
            result["shape_valid"] = array.ndim == 2 and all(0 < n <= 16 for n in array.shape)
        else:
            result["shape_valid"] = list(array.shape) == _SHAPES[name]
        if not result["shape_valid"]:
            issues.append("unexpected_shape")
        if array.size > 256:
            result["shape_valid"] = False
            issues.append("metadata_array_too_large")
        elif array.dtype.kind not in "iuf":
            result["value"] = _json_value(array)
            issues.append("not_numeric")
        else:
            result["value"] = _json_value(array)
            result["finite"] = bool(np.isfinite(array).all())
            if not result["finite"]:
                issues.append("non_finite")
            if result["finite"] and result["shape_valid"]:
                if name in ("raw_pattern", "black_level_per_channel", "white_level",
                            "camera_white_level_per_channel", "num_colors"):
                    if not np.equal(array, np.floor(array)).all():
                        issues.append("not_integer")
                if name == "raw_pattern" and (np.any(array < 0) or np.any(array > 3)):
                    issues.append("invalid_color_index")
                elif name == "num_colors" and not 1 <= array.item() <= 4:
                    issues.append("invalid_color_count")
                elif name == "black_level_per_channel" and np.any(array < 0):
                    issues.append("negative_black_level")
                elif name in ("white_level", "camera_white_level_per_channel",
                              "camera_whitebalance", "daylight_whitebalance") and np.any(array <= 0):
                    # Preserve all four entries; do not fill a missing G2 coefficient.
                    issues.append("non_positive_coefficient")
                elif name in ("color_matrix", "rgb_xyz_matrix"):
                    if np.linalg.matrix_rank(array.astype(np.float64)) < 3:
                        issues.append("rank_deficient_matrix")
    result["valid"] = not issues
    result["status"] = "ok" if result["valid"] else "invalid"
    return result


def _invalidate(field, issue):
    field["issues"].append(issue)
    field["valid"] = False
    field["status"] = "invalid"


def _channel_usage(fields):
    """Do not confuse a count of colors with the CFA's actual slot indices.

    RGBG with num_colors=3 may still use slot 3 for its second green. Conversely
    a three-index pattern may leave that slot unused. This reports pixel slots,
    not necessarily the number of independent WB coefficients.
    """
    pattern, description, count = (fields[name] for name in
                                   ("raw_pattern", "color_desc", "num_colors"))
    usage = {"status": "unknown", "basis": "raw_pattern_indices",
             "used_slots": None, "unused_slots": None, "observed_slots": None,
             "reason": "raw_pattern_unavailable_or_invalid"}
    if not pattern["valid"]:
        return usage
    slots = sorted({int(value) for row in pattern["value"] for value in row})
    usage["observed_slots"] = slots
    if not description["valid"]:
        usage["reason"] = "color_description_unavailable_or_invalid"
        return usage
    if not count["valid"]:
        usage["reason"] = "color_count_unavailable_or_invalid"
        return usage
    # A duplicated letter can describe the same color or two distinct colors
    # (rawpy's documented RGBG=3/4 case). Neither implies an unused CFA index.
    labels = {description["value"][slot] for slot in slots}
    if not len(labels) <= count["value"] <= len(slots):
        usage["reason"] = "color_count_inconsistent_with_pattern"
        return usage
    usage.update(status="known", used_slots=slots,
                 unused_slots=[slot for slot in range(4) if slot not in slots],
                 reason="indices_confirmed_by_pattern_and_color_metadata")
    return usage


def _apply_channel_semantics(fields, usage):
    """Keep all reported values, with checks scoped to known used channels.

    An unused zero is a permitted placeholder for WB/per-channel saturation.
    Non-finite values and negative values remain invalid even in unused slots.
    Unknown layouts keep the conservative all-four-slots numeric checks.
    """
    known = usage["status"] == "known"
    slots = usage["used_slots"] if known else list(range(4))
    for name in _CHANNEL_FIELDS:
        field = fields[name]
        if (known and name != "black_level_per_channel" and
                field["shape_valid"] and field["finite"] and
                "non_positive_coefficient" in field["issues"]):
            values = np.asarray(field["value"])
            if np.all(values[slots] > 0) and np.all(values >= 0):
                field["issues"].remove("non_positive_coefficient")
                field["valid"] = not field["issues"]
                field["status"] = "ok" if field["valid"] else "invalid"

    black = fields["black_level_per_channel"]
    for name in ("white_level", "camera_white_level_per_channel"):
        white = fields[name]
        if black["valid"] and white["valid"]:
            black_values = np.asarray(black["value"])[slots]
            white_values = np.asarray(white["value"])
            if name == "camera_white_level_per_channel":
                white_values = white_values[slots]
            if np.any(white_values <= black_values):
                _invalidate(white, "white_not_above_black")

    for name in (*_CHANNEL_FIELDS, "white_level"):
        field = fields[name]
        applicability = _unknown_applicability(usage["reason"])
        if known:
            applicability.update(used_slots=list(slots),
                                 unused_slots=list(usage["unused_slots"]))
            if field["status"] in ("missing", "error"):
                applicability["reason"] = "metadata_unavailable"
            elif not field["valid"]:
                applicability.update(status="not_applicable", reason="invalid_metadata")
            elif name in ("white_level", "camera_white_level_per_channel") and not black["valid"]:
                applicability["reason"] = "black_level_comparison_unavailable"
            else:
                applicability.update(status="applicable", reason="reported_used_slots_valid")
        if name in _CHANNEL_FIELDS:
            values = field["value"] if field["shape_valid"] else None
            description = fields["color_desc"]
            label = description["value"] if description["valid"] else ""
            applicability["slot_checks"] = []
            for slot in range(4):
                value = values[slot] if values is not None else None
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                try:
                    finite = math.isfinite(value) if numeric else None
                except OverflowError:
                    finite = False
                slot_usage = ("used" if slot in slots else "unused") if known else "unknown"
                # Range checks are diagnostics, separate from whole-field validity.
                minimum_inclusive = name == "black_level_per_channel" or slot_usage == "unused"
                range_valid = ((value >= 0 if minimum_inclusive else value > 0)
                               if numeric and finite else None)
                applicability["slot_checks"].append({
                    "slot": slot, "color": label[slot] if slot < len(label) else None,
                    "usage": slot_usage, "value": value, "finite": finite,
                    "range_valid": range_valid,
                })
        field["applicability"] = applicability

    for name in _WB_FIELDS:
        field = fields[name]
        semantics = field["white_balance_semantics"]
        if not known:
            semantics["reason"] = usage["reason"]
            continue
        if field["valid"]:
            semantics.update(status="direct_reported_slots",
                             reason="reported_coefficients_cover_cfa_slots",
                             message="已报告的白平衡系数覆盖所用通道；尚未应用",
                             cfa_to_coefficient_slots=[i if i in slots else None for i in range(4)])
        elif field["status"] not in ("missing", "error"):
            semantics.update(status="not_applicable", reason="invalid_reported_coefficients",
                             message="已报告的白平衡系数未通过通道检查；尚未应用")

        # LibRaw scale_colors() explicitly substitutes green for a zero fourth
        # pre_mul when colors < 4. Record that convention as an unapplied mapping
        # candidate, not an invented camera G2 coefficient or verified DNG tag.
        if (fields["num_colors"]["value"] == 3 and
                fields["color_desc"]["value"] == "RGBG" and
                slots == [0, 1, 2, 3] and field["shape_valid"] and field["finite"]):
            values = field["value"]
            if all(value > 0 for value in values[:3]) and values[3] == 0:
                semantics.update(
                    status="requires_explicit_green_mapping",
                    reason="three_rgb_coefficients_present_fourth_pixel_slot_uses_green",
                    message="RGB白平衡可用，第二绿色需显式映射；尚未应用",
                    rgb_coefficients=list(values[:3]), rgb_coefficients_valid=True,
                    cfa_to_coefficient_slots=[0, 1, 2, 1],
                    mapping_basis="libraw_scale_colors_zero_fourth_coefficient_convention",
                    mapping_reference=_LIBRAW_GREEN_REFERENCE,
                )


def read_raw_metadata(path: str | Path) -> dict:
    """Inspect one RAW without applying WB/matrices or changing the input.

    Failure is data: ``available=False`` with a specific status. If RAW reading
    succeeds, individual missing/invalid fields produce ``status='partial'``.
    All return values support ``json.dumps(result, allow_nan=False)``.
    ``channel_usage`` identifies CFA slots only when pattern/count/description
    agree. WB/black/white ``applicability`` is additive, never an applied transform.
    """
    result = {
        "available": False, "status": "unavailable", "source": "libraw_reported",
        "versions": {"rawpy": None, "libraw": None},
        "white_balance_applied": False, "color_correction_applied": False,
        "fields": {name: _record(name) for name in _FIELDS}, "errors": [],
        "channel_usage": {"status": "unknown", "basis": "raw_pattern_indices",
                          "used_slots": None, "unused_slots": None,
                          "observed_slots": None, "reason": "metadata_unavailable"},
    }
    try:
        source = Path(path).expanduser()
        if not source.is_file():
            result["status"] = "input_not_found"
            return result
    except (TypeError, ValueError, OSError) as exc:
        result["status"] = "invalid_path"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        return result
    try:
        rawpy = importlib.import_module("rawpy")
    except Exception as exc:
        result["status"] = "rawpy_unavailable" if isinstance(exc, ModuleNotFoundError) and exc.name == "rawpy" else "rawpy_import_error"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        return result
    for public, attribute in (("rawpy", "__version__"), ("libraw", "libraw_version")):
        try:
            result["versions"][public] = _json_value(getattr(rawpy, attribute, None))
        except (TypeError, ValueError):
            pass  # Version unknown; never invent one from the installed DLL name.
    try:
        # imread/close owns the decoder's lifetime. No postprocess or pixel edits.
        with rawpy.imread(str(source)) as raw:
            for name in _FIELDS:
                try:
                    result["fields"][name] = _inspect_field(name, getattr(raw, name, None))
                except Exception as exc:
                    field = _record(name)
                    field["status"] = "error"
                    field["issues"] = [type(exc).__name__]
                    result["fields"][name] = field
                    result["errors"].append({"field": name, "type": type(exc).__name__, "message": str(exc)})
    except Exception as exc:
        unsupported = getattr(rawpy, "LibRawFileUnsupportedError", None)
        result["status"] = "unsupported_format" if isinstance(unsupported, type) and isinstance(exc, unsupported) else "read_error"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        return result

    fields = result["fields"]
    pattern, description = fields["raw_pattern"], fields["color_desc"]
    if pattern["valid"] and description["valid"] and np.max(pattern["value"]) >= len(description["value"]):
        _invalidate(pattern, "color_index_not_in_description")
    result["channel_usage"] = _channel_usage(fields)
    _apply_channel_semantics(fields, result["channel_usage"])
    result["available"] = True
    result["status"] = ("ok" if all(field["valid"] for field in fields.values()) and
                        result["channel_usage"]["status"] == "known" else "partial")
    return result
