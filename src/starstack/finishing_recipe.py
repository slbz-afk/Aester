"""Pure validation and canonical digest for versioned Aester finishing recipes."""

from __future__ import annotations

import hashlib
import json
import math
import re

__all__ = ["normalize_recipe", "recipe_digest", "recipe_mask_ids"]

_SCHEMA = "aester-finishing-recipe-1"
_SCHEMA_V2 = "aester-finishing-recipe-2"
_SCHEMA_V3 = "aester-finishing-recipe-3"
_ENCODING = "srgb16"
_MAX_INTENT = 4000
_MAX_OPS = 32
_MIN_POINTS = 2
_MAX_POINTS = 32
_MAX_DIMENSION = 100_000
_MAX_PIXELS = 300_000_000
_TYPES = frozenset({"curves", "exposure", "saturation"})
_CHANNELS = frozenset({"value", "red", "green", "blue"})
_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,47}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = ("schema", "source", "intent", "operations")
_SOURCE_KEYS = ("sha256", "width", "height", "encoding")
_OPERATION_KEYS = ("id", "type", "enabled", "mask_id", "params")
_PARAM_KEYS = {
    "exposure": ("ev",),
    "saturation": ("scale",),
    "curves": ("channel", "points"),
    "local_contrast_unsharp": ("sigma_px", "amount"),
    "astrophotographic_layers": ("bundle_id", "bundle_sha256", "star_strength", "milkyway_amount",
                                 "milkyway_mask_id", "meteor_core_extra", "meteor_glow_amount"),
}


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _exact_dict(value, path: str, keys: tuple[str, ...]) -> dict:
    if type(value) is not dict:
        _fail(path, "expected a builtin dict")
    unknown = [key for key in value if type(key) is not str or key not in keys]
    if unknown:
        _fail(path, f"unknown field(s): {unknown!r}")
    missing = [key for key in keys if key not in value]
    if missing:
        _fail(path, f"missing field(s): {missing!r}")
    return value


def _text(value, path: str) -> str:
    if type(value) is not str:
        _fail(path, "expected a builtin str")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(path, "must be valid UTF-8 text")
    return value


def _number(value, path: str):
    if type(value) is bool or (type(value) is not int and type(value) is not float):
        _fail(path, "expected a builtin int or float (not bool)")
    if type(value) is float and not math.isfinite(value):
        _fail(path, "expected a finite number")
    return value


def _number_in_range(value, path: str, low, high) -> float:
    _number(value, path)
    if value < low or value > high:
        _fail(path, f"must be between {low} and {high}")
    result = float(value)
    if result == 0.0:
        return 0.0
    return result


def _dimension(value, path: str) -> int:
    if type(value) is not int:
        _fail(path, "expected a builtin int")
    if value < 1 or value > _MAX_DIMENSION:
        _fail(path, "must be between 1 and 100000")
    return value


def _normalize_source(value) -> dict:
    source = _exact_dict(value, "source", _SOURCE_KEYS)
    sha256 = source["sha256"]
    if type(sha256) is not str or not _SHA256_PATTERN.fullmatch(sha256):
        _fail("source.sha256", "must be 64 lowercase hexadecimal characters")
    width = _dimension(source["width"], "source.width")
    height = _dimension(source["height"], "source.height")
    if width * height > _MAX_PIXELS:
        _fail("source", "width * height must be at most 300000000")
    encoding = source["encoding"]
    if type(encoding) is not str or encoding != _ENCODING:
        _fail("source.encoding", "must be srgb16")
    return {
        "sha256": sha256,
        "width": width,
        "height": height,
        "encoding": _ENCODING,
    }


def _normalize_channel(value, path: str) -> str:
    if type(value) is not str or value not in _CHANNELS:
        _fail(path, "must be one of value, red, green, blue")
    return value


def _normalize_points(value, path: str) -> list:
    if type(value) is not list:
        _fail(path, "expected a builtin list")
    count = len(value)
    if count < _MIN_POINTS or count > _MAX_POINTS:
        _fail(path, "must contain between 2 and 32 points")
    points = []
    for index, raw in enumerate(value):
        point_path = f"{path}[{index}]"
        if type(raw) is not list or len(raw) != 2:
            _fail(point_path, "must be a list of exactly two numbers")
        x = _number_in_range(raw[0], f"{point_path}[0]", 0.0, 1.0)
        y = _number_in_range(raw[1], f"{point_path}[1]", 0.0, 1.0)
        points.append([x, y])
    if points[0][0] != 0.0:
        _fail(f"{path}[0][0]", "first x must be 0")
    if points[-1][0] != 1.0:
        _fail(f"{path}[-1][0]", "last x must be 1")
    for index in range(1, count):
        previous = points[index - 1]
        current = points[index]
        if current[0] <= previous[0]:
            _fail(f"{path}[{index}][0]", "x must be strictly increasing")
        if current[1] < previous[1]:
            _fail(f"{path}[{index}][1]", "y must be non-decreasing")
    return points


def _normalize_params(op_type: str, value, path: str) -> dict:
    params = _exact_dict(value, path, _PARAM_KEYS[op_type])
    if op_type == "astrophotographic_layers":
        bundle = params["bundle_id"]
        digest = params["bundle_sha256"]
        mask = params["milkyway_mask_id"]
        if type(bundle) is not str or not _ID_PATTERN.fullmatch(bundle):
            _fail(f"{path}.bundle_id", "must be a registered layer bundle id")
        if type(digest) is not str or not _SHA256_PATTERN.fullmatch(digest):
            _fail(f"{path}.bundle_sha256", "must be 64 lowercase hexadecimal characters")
        if mask is not None and (type(mask) is not str or not _ID_PATTERN.fullmatch(mask)):
            _fail(f"{path}.milkyway_mask_id", "must be null or a registered mask id")
        result = {"bundle_id": bundle, "bundle_sha256": digest,
                  "star_strength": _number_in_range(params["star_strength"], f"{path}.star_strength", 0, 1),
                  "milkyway_amount": _number_in_range(params["milkyway_amount"], f"{path}.milkyway_amount", 0, 3),
                  "milkyway_mask_id": mask,
                  "meteor_core_extra": _number_in_range(params["meteor_core_extra"], f"{path}.meteor_core_extra", 0, 2),
                  "meteor_glow_amount": _number_in_range(params["meteor_glow_amount"], f"{path}.meteor_glow_amount", 0, 4)}
        if result["milkyway_amount"] != 0 and mask is None:
            _fail(f"{path}.milkyway_mask_id", "nonzero Milky Way adjustment requires its region")
        return result
    if op_type == "exposure":
        return {"ev": _number_in_range(params["ev"], f"{path}.ev", -2, 2)}
    if op_type == "saturation":
        return {"scale": _number_in_range(params["scale"], f"{path}.scale", 0, 2)}
    if op_type == "local_contrast_unsharp":
        return {"sigma_px": _number_in_range(params["sigma_px"], f"{path}.sigma_px", 4, 80),
                "amount": _number_in_range(params["amount"], f"{path}.amount", 0, .6)}
    return {
        "channel": _normalize_channel(params["channel"], f"{path}.channel"),
        "points": _normalize_points(params["points"], f"{path}.points"),
    }


def _normalize_operation(value, path: str, schema: str) -> dict:
    operation = _exact_dict(value, path, _OPERATION_KEYS)
    op_id = operation["id"]
    if type(op_id) is not str or not _ID_PATTERN.fullmatch(op_id):
        _fail(f"{path}.id", "must match [a-z][a-z0-9_-]{0,47}")
    op_type = operation["type"]
    allowed = _TYPES if schema == _SCHEMA else _TYPES | {"local_contrast_unsharp"}
    if schema == _SCHEMA_V3:
        allowed = allowed | {"astrophotographic_layers"}
    if type(op_type) is not str or op_type not in allowed:
        _fail(f"{path}.type", "unsupported operation for this recipe schema")
    enabled = operation["enabled"]
    if type(enabled) is not bool:
        _fail(f"{path}.enabled", "must be a builtin bool")
    mask_id = operation["mask_id"]
    if mask_id is not None:
        if type(mask_id) is not str or not _ID_PATTERN.fullmatch(mask_id):
            _fail(f"{path}.mask_id", "must be null or a valid mask id")
    if op_type == "astrophotographic_layers" and mask_id is not None:
        _fail(f"{path}.mask_id", "layer operation uses explicit regions; outer mask must be null")
    params = _normalize_params(op_type, operation["params"], f"{path}.params")
    return {
        "id": op_id,
        "type": op_type,
        "enabled": enabled,
        "mask_id": mask_id,
        "params": params,
    }


def _normalize_operations(value, schema: str) -> list:
    if type(value) is not list:
        _fail("operations", "expected a builtin list")
    if len(value) > _MAX_OPS:
        _fail("operations", "must contain at most 32 items")
    operations = []
    seen = set()
    for index, raw in enumerate(value):
        operation = _normalize_operation(raw, f"operations[{index}]", schema)
        if operation["type"] == "astrophotographic_layers" and index != 0:
            _fail(f"operations[{index}]", "layer operation must be first and unique")
        if operation["id"] in seen:
            _fail(f"operations[{index}].id", "duplicate operation id")
        seen.add(operation["id"])
        operations.append(operation)
    return operations


def normalize_recipe(value: dict) -> dict:
    """Validate *value* and return an independent JSON-compatible copy."""
    recipe = _exact_dict(value, "recipe", _TOP_KEYS)
    schema = recipe["schema"]
    if type(schema) is not str or schema not in {_SCHEMA, _SCHEMA_V2, _SCHEMA_V3}:
        _fail("schema", "must be aester-finishing-recipe-1, -2 or -3")
    intent = _text(recipe["intent"], "intent")
    if len(intent) > _MAX_INTENT:
        _fail("intent", "must be at most 4000 characters")
    return {
        "schema": schema,
        "source": _normalize_source(recipe["source"]),
        "intent": intent,
        "operations": _normalize_operations(recipe["operations"], schema),
    }


def recipe_digest(value: dict) -> str:
    """Return the SHA-256 digest of the canonical normalized recipe JSON."""
    normalized = normalize_recipe(value)
    try:
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as exc:
        raise ValueError("recipe is not encodable as canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def recipe_mask_ids(value: dict) -> set[str]:
    """One dependency collector for ordinary and nested operation regions."""
    result = set()
    for op in normalize_recipe(value)["operations"]:
        if op["mask_id"] is not None:
            result.add(op["mask_id"])
        if op["type"] == "astrophotographic_layers" and op["params"]["milkyway_mask_id"] is not None:
            result.add(op["params"]["milkyway_mask_id"])
    return result
