"""Registration quality diagnostics for a sequence of star matches.

Contract: CONTRACT.md v1 (FABLE-REGISTRATION-DIAGNOSTICS-20260916).

Given, per frame, an explicit 3x3 source->reference homography, the source and
reference frame sizes and a list of star correspondences whose *role* (fit,
holdout, unknown) is declared by the caller, this module reports

* per-star signed residuals ``predicted_reference - measured_reference``,
* per-role summaries (count, RMS, median, P95, max) overall and per region
  (center, four corners, other) of the reference frame,
* the continuous geometric intersection of the projected source frame with
  the reference frame (polygon, area, fraction of the reference area).

It does not detect or match stars, fit or refine transforms, resample pixels,
build sky masks, stack, or reject frames.  Geometric overlap is a frame
footprint only: it says nothing about sky visibility, resampling support,
exposure weight or registration accuracy.  Roles, the independence of holdout
points and the coordinate spaces are caller declarations; the module can only
check data constraints.  No pass/fail thresholds are applied.

Only the standard library and NumPy are used.  ``analyze_frame`` and
``analyze_sequence`` never write files or exit the process; the CLI does.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np

__all__ = ["SCHEMA_VERSION", "MODULE_VERSION", "COORDINATE_CONVENTION", "REGIONS", "ROLES",
           "FrameError", "analyze_frame", "analyze_sequence", "parse_document", "load_document", "main"]

SCHEMA_VERSION = 1
MODULE_VERSION = "1.0.1"
COORDINATE_CONVENTION = {"row_order": "TOP-DOWN", "origin": "top_left_pixel_center", "units": "pixel"}
ROLES = ("fit", "holdout", "unknown")
REGIONS = ("center", "top_left", "top_right", "bottom_left", "bottom_right", "other")
SUMMARY_KEYS = ("count", "rms_px", "median_px", "p95_px", "max_px")

# Engineering limits from CONTRACT.md; not photographic thresholds.
CONDITION_LIMIT = 1e12
CANCELLATION_RELATIVE = 1e-12

LIMITATIONS = [
    "geometric_overlap 只描述投影后源画幅与参考画幅的连续矩形交集；不等于天空可见性、重采样核有效支持、有效曝光权重或配准精度。",
    "role（fit/holdout/unknown）、留出点未参与拟合的独立性、以及各帧 coordinate_space_id 与实际坐标/画幅/矩阵的一致性均为调用方声明；模块只核验数据约束，不能证明这些物理前提。",
    "所有匹配点都进入其角色的统计；不按残差过滤离群点，不剔帧，不给出摄影合格/不合格阈值。",
    "没有留出点时 holdout 统计为空（count=0），拟合点残差不能替代独立精度；unknown 角色的点单独统计。",
    "elapsed_seconds 由调用方提供，缺失为 null；序列按输入顺序报告，不按时间重排，不由文件名推断时间。",
    "少样本统计（count=1 时四个统计均等于该距离）只是记录，不构成可靠性保证。",
    "投影角点用 float64 计算；交集多边形与面积在这些角点上用有理数精确裁剪，仅触边/触点时面积恰为 0。",
    "矩阵条件数上限 1e12 与分母相对抵消阈值 1e-12 是本模块的工程边界，不是配准精度阈值。",
]


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #

class FrameError(ValueError):
    """A frame failed validation; ``code`` is a stable machine-readable string."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value):
    """Return ``float(value)`` for a finite JSON number, else None."""
    if isinstance(value, np.generic):  # programmatic callers may pass NumPy scalars
        value = value.item()
    if not _is_number(value):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonempty_str(value) -> bool:
    return isinstance(value, str) and value != ""


def _validate_reference(reference) -> dict:
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object")
    if not _nonempty_str(reference.get("frame_id")):
        raise ValueError("reference.frame_id must be a non-empty string")
    if not _positive_int(reference.get("width")) or not _positive_int(reference.get("height")):
        raise ValueError("reference.width and reference.height must be positive integers")
    if _finite(reference["width"] * reference["height"]) is None:
        raise ValueError("reference frame area must be representable as finite float64")
    if not _nonempty_str(reference.get("coordinate_space_id")):
        raise ValueError("reference.coordinate_space_id must be a non-empty string")
    return {"frame_id": reference["frame_id"], "width": reference["width"],
            "height": reference["height"], "coordinate_space_id": reference["coordinate_space_id"]}


def _validate_convention(convention) -> dict:
    if not isinstance(convention, dict) or set(convention) != set(COORDINATE_CONVENTION):
        raise ValueError("coordinate_convention must contain exactly row_order, origin and units")
    for key, expected in COORDINATE_CONVENTION.items():
        if convention[key] != expected:
            raise ValueError(f"unsupported coordinate_convention.{key}={convention[key]!r}; "
                             f"this module only accepts {expected!r} and does not convert")
    return dict(COORDINATE_CONVENTION)


def _validate_matrix(value) -> tuple[list, np.ndarray]:
    """Return (echo copy of the input rows, float64 matrix).

    Rank and condition number use the copy divided by its largest absolute
    element (CONTRACT.md). Projections use an equivalent power-of-two rescaling
    with largest coefficient in [0.5, 1). Binary scaling avoids introducing
    extra rounding into exact integer/affine inputs while preventing overflow
    from an arbitrary common magnitude. Only the echo keeps the original scale.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
            not isinstance(row, (list, tuple)) or len(row) != 3 for row in value):
        raise FrameError("matrix_shape_invalid", "source_to_reference must be a 3x3 array of numbers")
    echo = []
    matrix = np.empty((3, 3), dtype=np.float64)
    for i, row in enumerate(value):
        echo_row = []
        for j, element in enumerate(row):
            number = _finite(element)
            if number is None:
                raise FrameError("matrix_not_finite", f"source_to_reference[{i}][{j}] is not a finite number")
            matrix[i, j] = number
            echo_row.append(element.item() if isinstance(element, np.generic) else element)
        echo.append(echo_row)
    largest = float(np.max(np.abs(matrix)))
    if largest == 0.0:
        raise FrameError("matrix_zero", "source_to_reference is the zero matrix")
    normalised = matrix / largest
    if int(np.linalg.matrix_rank(normalised)) != 3:
        raise FrameError("matrix_singular_or_ill_conditioned", "source_to_reference does not have rank 3")
    condition = float(np.linalg.cond(normalised))
    if not math.isfinite(condition) or condition > CONDITION_LIMIT:
        raise FrameError("matrix_singular_or_ill_conditioned",
                         f"condition number {condition:.3e} exceeds the module limit {CONDITION_LIMIT:.0e}")
    exponent = math.frexp(largest)[1]
    projection_matrix = np.ldexp(matrix, -exponent)
    return echo, projection_matrix


def _denominator(matrix: np.ndarray, x: float, y: float) -> tuple[float, bool]:
    """Homogeneous denominator for (x, y) and whether it is numerically usable.

    The relative-cancellation test is invariant under scaling of the matrix.
    """
    h20, h21, h22 = (float(v) for v in matrix[2])
    den = h20 * x + h21 * y + h22
    scale = abs(h20 * x) + abs(h21 * y) + abs(h22)
    usable = math.isfinite(den) and den != 0.0 and abs(den) > CANCELLATION_RELATIVE * scale
    return den, usable


def _project(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    den, usable = _denominator(matrix, x, y)
    if not usable:
        raise FrameError("point_at_projective_infinity",
                         f"source point ({x}, {y}) has a vanishing homogeneous denominator")
    px = (float(matrix[0, 0]) * x + float(matrix[0, 1]) * y + float(matrix[0, 2])) / den
    py = (float(matrix[1, 0]) * x + float(matrix[1, 1]) * y + float(matrix[1, 2])) / den
    if not (math.isfinite(px) and math.isfinite(py)):
        raise FrameError("projection_not_finite",
                         f"source point ({x}, {y}) projects to a non-finite value (matrix magnitude too large)")
    return px, py


def _corners(width: int, height: int) -> list[tuple[float, float]]:
    """Continuous frame rectangle corners, clockwise from top-left."""
    x0, y0, x1, y1 = -0.5, -0.5, width - 0.5, height - 0.5
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _check_frame_denominators(matrix: np.ndarray, width: int, height: int) -> None:
    signs = []
    for x, y in _corners(width, height):
        den, usable = _denominator(matrix, x, y)
        if not usable:
            raise FrameError("frame_crosses_projective_infinity",
                             f"source corner ({x}, {y}) maps to projective infinity (denominator {den:.3e})")
        signs.append(den > 0)
    if any(signs) and not all(signs):
        raise FrameError("frame_crosses_projective_infinity",
                         "source frame corners have homogeneous denominators of opposite sign")


def _validate_xy(value, name: str, star_id) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise FrameError("match_xy_invalid", f"{name} of star {star_id!r} must be [x, y]")
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        raise FrameError("match_xy_invalid", f"{name} of star {star_id!r} must contain two finite numbers")
    return x, y


def _inside(x: float, y: float, width: int, height: int) -> bool:
    return -0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5


def _validate_matches(matches, width: int, height: int, ref_width: int, ref_height: int) -> list[dict]:
    if not isinstance(matches, list):
        raise FrameError("matches_invalid", "matches must be a list")
    seen = set()
    validated = []
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise FrameError("matches_invalid", f"matches[{index}] must be an object")
        star_id = match.get("star_id")
        if not _nonempty_str(star_id):
            raise FrameError("star_id_invalid", f"matches[{index}].star_id must be a non-empty string")
        if star_id in seen:
            raise FrameError("star_id_duplicate", f"star_id {star_id!r} occurs more than once in this frame")
        seen.add(star_id)
        role = match.get("role")
        if role not in ROLES:
            raise FrameError("role_invalid", f"star {star_id!r} role must be one of {ROLES}, got {role!r}")
        sx, sy = _validate_xy(match.get("source_xy"), "source_xy", star_id)
        rx, ry = _validate_xy(match.get("reference_xy"), "reference_xy", star_id)
        if not _inside(sx, sy, width, height):
            raise FrameError("match_out_of_bounds",
                             f"star {star_id!r} source_xy ({sx}, {sy}) lies outside the source frame "
                             f"[-0.5, {width - 0.5}] x [-0.5, {height - 0.5}]")
        if not _inside(rx, ry, ref_width, ref_height):
            raise FrameError("match_out_of_bounds",
                             f"star {star_id!r} reference_xy ({rx}, {ry}) lies outside the reference frame "
                             f"[-0.5, {ref_width - 0.5}] x [-0.5, {ref_height - 0.5}]")
        validated.append({"star_id": star_id, "role": role, "source_xy": (sx, sy), "reference_xy": (rx, ry)})
    return validated


# --------------------------------------------------------------------------- #
# Regions and statistics                                                      #
# --------------------------------------------------------------------------- #

def _region(x: float, y: float, width: int, height: int) -> str:
    u, v = (x + 0.5) / width, (y + 0.5) / height
    u_low, u_high = u < 0.3, u >= 0.7
    v_low, v_high = v < 0.3, v >= 0.7
    if not (u_low or u_high) and not (v_low or v_high):
        return "center"
    if u_low and v_low:
        return "top_left"
    if u_high and v_low:
        return "top_right"
    if u_low and v_high:
        return "bottom_left"
    if u_high and v_high:
        return "bottom_right"
    return "other"


def _summary(distances: list[float]) -> dict:
    count = len(distances)
    if count == 0:
        return {"count": 0, "rms_px": None, "median_px": None, "p95_px": None, "max_px": None}
    d = np.asarray(distances, dtype=np.float64)
    if not np.all(np.isfinite(d)):
        raise FrameError("derived_numeric_overflow", "residual distance is not finite")
    # Normalize before squaring or averaging. A finite distance can exceed
    # sqrt(float64.max); even a median of two large equal values can overflow.
    largest = float(np.max(d))
    scaled = d / largest if largest else d
    return {"count": count,
            "rms_px": float(np.sqrt(np.mean(scaled * scaled))) * largest,
            "median_px": float(np.median(scaled)) * largest,
            "p95_px": float(np.percentile(scaled, 95, method="linear")) * largest,
            "max_px": largest}


def _statistics(residuals: list[dict]) -> dict:
    statistics = {}
    for role in ROLES:
        of_role = [r for r in residuals if r["role"] == role]
        statistics[role] = {
            "overall": _summary([r["distance_px"] for r in of_role]),
            "regions": {region: _summary([r["distance_px"] for r in of_role if r["region"] == region])
                        for region in REGIONS},
        }
    return statistics


# --------------------------------------------------------------------------- #
# Exact polygon clipping (rational arithmetic on the float64 corner values)   #
# --------------------------------------------------------------------------- #

def _clip_half_plane(polygon, axis: int, bound: Fraction, keep_below: bool):
    """Sutherland-Hodgman step against x/y <= bound (keep_below) or >= bound."""
    def inside(point):
        return point[axis] <= bound if keep_below else point[axis] >= bound

    def intersect(a, b):
        t = (bound - a[axis]) / (b[axis] - a[axis])
        other = 1 - axis
        crossing = [None, None]
        crossing[axis] = bound
        crossing[other] = a[other] + t * (b[other] - a[other])
        return (crossing[0], crossing[1])

    result = []
    n = len(polygon)
    for i in range(n):
        current, previous = polygon[i], polygon[i - 1]
        current_in, previous_in = inside(current), inside(previous)
        if current_in:
            if not previous_in:
                result.append(intersect(previous, current))
            result.append(current)
        elif previous_in:
            result.append(intersect(previous, current))
    return result


def _dedupe(polygon):
    cleaned = []
    for point in polygon:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    while len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _shoelace(polygon) -> Fraction:
    if len(polygon) < 3:
        return Fraction(0)
    total = Fraction(0)
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def _geometric_overlap(matrix: np.ndarray, width: int, height: int, ref_width: int, ref_height: int) -> dict:
    projected = [_project(matrix, x, y) for x, y in _corners(width, height)]
    polygon = [(Fraction(px), Fraction(py)) for px, py in projected]
    x0, y0 = Fraction(-1, 2), Fraction(-1, 2)
    x1, y1 = Fraction(2 * ref_width - 1, 2), Fraction(2 * ref_height - 1, 2)
    for axis, bound, keep_below in ((0, x0, False), (0, x1, True), (1, y0, False), (1, y1, True)):
        if not polygon:
            break
        polygon = _clip_half_plane(polygon, axis, bound, keep_below)
    polygon = _dedupe(polygon)
    area = _shoelace(polygon)
    return {"projected_source_polygon_xy": [[float(px), float(py)] for px, py in projected],
            "intersection_polygon_xy": [[float(px), float(py)] for px, py in polygon],
            "intersection_area_px2": float(area),
            "reference_area_fraction": float(area / (ref_width * ref_height)),
            "coverage_kind": "continuous_frame_footprint_only"}


# --------------------------------------------------------------------------- #
# Frame and sequence analysis                                                 #
# --------------------------------------------------------------------------- #

def _frame_record(frame_id, elapsed=None, geometry=None, matrix=None, error=None) -> dict:
    return {"frame_id": frame_id, "elapsed_seconds": elapsed, "state": "invalid",
            "source_geometry": geometry, "source_to_reference": matrix,
            "holdout_evidence": "unavailable", "match_counts": None, "residuals": None,
            "statistics": None, "geometric_overlap": None, "error": error}


def _analyze_validated_frame(reference: dict, frame: dict, frame_id) -> dict:
    """Validate ``frame`` and compute the success record; raise FrameError otherwise.

    Validation order: source geometry, elapsed_seconds, matrix, matches,
    projections.  The first failure is reported.  Partial echoes of already
    validated fields are attached to the raised FrameError as ``record``.
    """
    record = _frame_record(frame_id)
    try:
        width, height, space = frame.get("width"), frame.get("height"), frame.get("coordinate_space_id")
        if not _positive_int(width) or not _positive_int(height) or not _nonempty_str(space):
            raise FrameError("frame_geometry_invalid",
                             "width and height must be positive integers and coordinate_space_id a non-empty string")
        if _finite(width) is None or _finite(height) is None:
            raise FrameError("frame_geometry_invalid", "width and height must be representable as finite float64")
        geometry = {"width": width, "height": height, "coordinate_space_id": space}
        record["source_geometry"] = geometry
        elapsed = frame.get("elapsed_seconds")
        if elapsed is not None and _finite(elapsed) is None:
            raise FrameError("elapsed_seconds_invalid", "elapsed_seconds must be a finite number or null")
        if isinstance(elapsed, np.generic):
            elapsed = elapsed.item()
        record["elapsed_seconds"] = elapsed
        echo, matrix = _validate_matrix(frame.get("source_to_reference"))
        _check_frame_denominators(matrix, width, height)
        record["source_to_reference"] = copy.deepcopy(echo)
        matches = _validate_matches(frame.get("matches"), width, height,
                                    reference["width"], reference["height"])
        residuals = []
        for match in matches:
            sx, sy = match["source_xy"]
            rx, ry = match["reference_xy"]
            px, py = _project(matrix, sx, sy)
            dx, dy = px - rx, py - ry
            distance = math.hypot(dx, dy)
            if not all(math.isfinite(v) for v in (dx, dy, distance)):
                raise FrameError("derived_numeric_overflow", "residual exceeds finite float64")
            residuals.append({"star_id": match["star_id"], "role": match["role"],
                              "source_xy": [sx, sy], "reference_xy": [rx, ry],
                              "predicted_reference_xy": [px, py],
                              "dx_px": dx, "dy_px": dy, "distance_px": distance,
                              "region": _region(rx, ry, reference["width"], reference["height"])})
        overlap = _geometric_overlap(matrix, width, height, reference["width"], reference["height"])
        statistics = _statistics(residuals)
    except FrameError as exc:
        exc.record = record
        raise
    counts = {role: sum(1 for r in residuals if r["role"] == role) for role in ROLES}
    return {"frame_id": frame_id, "elapsed_seconds": elapsed, "state": "succeeded",
            "source_geometry": geometry, "source_to_reference": copy.deepcopy(echo),
            "holdout_evidence": "caller_declared" if counts["holdout"] > 0 else "unavailable",
            "match_counts": counts, "residuals": residuals, "statistics": statistics,
            "geometric_overlap": overlap, "error": None}


def analyze_frame(reference, frame) -> dict:
    """Analyse one frame against ``reference``; raise ValueError on any input error."""
    validated = _validate_reference(reference)
    if not isinstance(frame, dict):
        raise ValueError("frame_not_object: frame must be an object")
    frame_id = frame.get("frame_id")
    if not _nonempty_str(frame_id):
        raise ValueError("frame_id_invalid: frame_id must be a non-empty string")
    return _analyze_validated_frame(validated, frame, frame_id)


def analyze_sequence(document) -> dict:
    """Analyse every frame of a document; top-level errors raise ValueError.

    Frame-level validation failures are kept as ``state: invalid`` records so
    that the sequence keeps its input order and no frame is silently dropped.
    """
    if not isinstance(document, dict):
        raise ValueError("document must be an object")
    if document.get("schema_version") != SCHEMA_VERSION or isinstance(document.get("schema_version"), bool):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    convention = _validate_convention(document.get("coordinate_convention"))
    reference = _validate_reference(document.get("reference"))
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list")
    seen = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        frame_id = frame.get("frame_id")
        if not _nonempty_str(frame_id):
            raise ValueError(f"frames[{index}].frame_id must be a non-empty string")
        if frame_id in seen:
            raise ValueError(f"duplicate frame_id {frame_id!r}")
        seen.add(frame_id)
    results = []
    for frame in frames:
        try:
            results.append(_analyze_validated_frame(reference, frame, frame["frame_id"]))
        except FrameError as exc:
            record = exc.record
            record["error"] = {"code": exc.code, "message": exc.message}
            results.append(record)
    succeeded = sum(1 for r in results if r["state"] == "succeeded")
    state = "succeeded" if succeeded == len(results) else ("failed" if succeeded == 0 else "partial")
    return {"schema_version": SCHEMA_VERSION, "state": state, "coordinate_convention": convention,
            "reference": reference, "frames": results, "limitations": list(LIMITATIONS)}


# --------------------------------------------------------------------------- #
# Command line                                                                #
# --------------------------------------------------------------------------- #

def _reject_constant(name: str):
    raise ValueError(f"JSON constant {name} is not allowed (NaN/Infinity are rejected)")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _finite_json_float(token: str) -> float:
    number = float(token)
    if not math.isfinite(number):
        raise ValueError("JSON number is outside finite float64 range")
    return number


def parse_document(raw: bytes) -> dict:
    """Parse JSON bytes strictly: UTF-8 (BOM tolerated), no NaN/Infinity, no duplicate keys."""
    return json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant,
                      parse_float=_finite_json_float,
                      object_pairs_hook=_reject_duplicate_keys)


def load_document(path) -> dict:
    """Read and strictly parse a JSON document from ``path``."""
    return parse_document(Path(path).read_bytes())


def _write_new_file(path: Path, text: str, overwrite: bool) -> None:
    """Publish a complete temporary file atomically.

    Default publication uses an exclusive hard link, so a concurrent writer
    cannot be replaced between a path check and publication. Filesystems that
    do not support hard links fail closed; no unsafe copy/replace fallback.
    Explicit --overwrite retains os.replace semantics.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Registration quality diagnostics report (contract v1).")
    parser.add_argument("--input", required=True, help="input JSON document (schema_version 1)")
    parser.add_argument("--output", required=True, help="path of the new JSON report to write")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)
    input_path, output_path = Path(args.input), Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"error: output already exists, refusing to overwrite: {output_path}", file=sys.stderr)
        return 1
    if output_path.exists() and input_path.exists() and output_path.resolve() == input_path.resolve():
        print("error: --output must not be the --input file", file=sys.stderr)
        return 1
    try:
        raw = input_path.read_bytes()
        document = parse_document(raw)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        print(f"error: cannot read input JSON: {exc}", file=sys.stderr)
        return 1
    try:
        report = analyze_sequence(document)
    except ValueError as exc:
        print(f"error: invalid input document: {exc}", file=sys.stderr)
        return 1
    report["provenance"] = {
        "module": "registration_diagnostics", "module_version": MODULE_VERSION,
        "input_argument": str(args.input), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    try:
        text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        _write_new_file(output_path, text + "\n", args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"error: cannot write report: {exc}", file=sys.stderr)
        return 1
    print(f"{report['state']}: {output_path}")
    return {"succeeded": 0, "partial": 2, "failed": 3}[report["state"]]


if __name__ == "__main__":
    sys.exit(main())
