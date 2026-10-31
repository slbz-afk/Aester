"""Fixed-reference, approximate camera RGB -> linear sRGB in floating point.

This is an independently written matrix construction following LibRaw 0.22.1
cam_xyz_coeff's mathematical convention, not a port of its C++ implementation.
rawpy.rgb_xyz_matrix exposes LibRaw cam_xyz, despite its ambiguous public name.
For the supported three-color model, construct A = cam_xyz @ XYZ_FROM_SRGB,
normalize each row by its sum, then invert A. Do not multiply RGB by cam_xyz.

This module never subtracts black, demosaics, estimates per-frame WB, clips,
encodes sRGB gamma, applies a DNG profile/LUT, or claims physical calibration.
Inputs must already be unbalanced, black-subtracted, TOP-DOWN camera RGB DN.
The recipe records a caller-asserted input contract; array values cannot prove
their processing history. Nonlinear demosaicing need not commute with WB.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ALGORITHM = "libraw-0.22.1-three-color-camxyz-v1"
INPUT_STATE = "black_subtracted_camera_rgb_DN"
MATRIX_DIRECTION = "white_balanced_camera_RGB_to_approximate_linear_sRGB"
SOURCES = {
    "cam_xyz_coeff": "https://github.com/LibRaw/LibRaw/blob/0.22.1/src/utils/utils_dcraw.cpp#L266-L295",
    "rgb_basis": "https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/colorconst.cpp#L15-L18",
    "wb_green": "https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/postprocessing_utils_dcrdefs.cpp#L170-L187",
    "rawpy_fields": "https://github.com/letmaik/rawpy/blob/v0.27.1/rawpy/_rawpy.pyx",
}
# Standard linear sRGB (D65) -> XYZ constants, as used by the pinned LibRaw.
XYZ_FROM_LINEAR_SRGB = np.array([
    [.4124564, .3575761, .1804375],
    [.2126729, .7151522, .0721750],
    [.0193339, .1191920, .9503041],
], dtype=np.float64)
XYZ_FROM_LINEAR_SRGB.setflags(write=False)

# These are validation tolerances, not rendering adjustments. Permit small
# float64 reconstruction differences without changing saved coefficients.
# An absolute floor is needed for inverse entries whose true value is zero;
# 64 eps is still far below float32 precision at the neutral-axis unit scale.
# Positive row sums/reciprocals use only relative tolerance to avoid accepting
# zero for a very small, but meaningful, diagnostic. The condition-number
# cutoff remains hard: tolerance is not permission to admit a singular model.
_MATRIX_RTOL = 1e-12
_MATRIX_ATOL = 64 * np.finfo(np.float64).eps
_CONDITION_RTOL = 1e-9
_MAX_CONDITION = 1e6


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} 必须为有限数值")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} 必须为有限{'正' if positive else ''}数值")
    return value


def _array(value, shape, name):
    def numeric_leaves(item):
        if isinstance(item, np.ndarray):
            return item.dtype.kind in "iuf"
        if isinstance(item, (list, tuple)):
            return all(numeric_leaves(child) for child in item)
        return (isinstance(item, (int, float, np.integer, np.floating))
                and not isinstance(item, (bool, np.bool_)))

    # Check before numpy can coerce a mixed [True, 1.0] list to float64.
    if not numeric_leaves(value):
        raise ValueError(f"{name} 必须由真实数值组成，不能含布尔值或数字字符串")
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} 形状或数值类型不受支持")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限值")
    return array


def _field(metadata, name):
    fields = metadata.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("缺少 RAW 元数据字段")
    field = fields.get(name)
    if not isinstance(field, dict) or field.get("value") is None:
        raise ValueError(f"缺少 {name} 元数据")
    return field["value"]


def _camera_model(metadata):
    if not isinstance(metadata, dict) or metadata.get("available") is not True:
        raise ValueError("需要可读取的 LibRaw 元数据")
    if metadata.get("white_balance_applied") is not False or metadata.get("color_correction_applied") is not False:
        raise ValueError("输入元数据必须明确尚未应用白平衡和颜色变换")
    count = _field(metadata, "num_colors")
    if type(count) is not int or count != 3 or _field(metadata, "color_desc") != "RGBG":
        raise ValueError("此颜色核心仅支持三色 RGBG Bayer")
    pattern = _array(_field(metadata, "raw_pattern"), (2, 2), "CFA pattern")
    if not np.equal(pattern, np.floor(pattern)).all() or np.any(pattern < 0) or np.any(pattern > 3):
        raise ValueError("CFA 索引不合法")
    rgb = np.where(pattern == 3, 1, pattern).astype(int)
    if (sorted(rgb.reshape(-1)) != [0, 1, 1, 2] or
            int(np.argwhere(rgb == 0)[0].sum()) % 2 != int(np.argwhere(rgb == 2)[0].sum()) % 2):
        raise ValueError("CFA 必须是 R/B 对角的 Bayer 排列")
    cam_xyz = _array(_field(metadata, "rgb_xyz_matrix"), (4, 3), "LibRaw cam_xyz")
    if np.any(cam_xyz[3] != 0):
        raise ValueError("第四矩阵行非零，不能丢弃或混同第四种颜色")
    return {"num_colors": 3, "color_desc": "RGBG", "raw_pattern": pattern.astype(int).tolist(),
            "cam_xyz": cam_xyz.tolist(), "decoder_versions": metadata.get("versions", {})}


def cam_xyz_to_rgb_cam(cam_xyz) -> dict:
    """Construct the normalized three-color matrix; reject degenerate models."""
    cam_xyz = _array(cam_xyz, (4, 3), "LibRaw cam_xyz")
    if np.any(cam_xyz[3] != 0):
        raise ValueError("此构造只接受第四行全零的三色矩阵")
    combined = cam_xyz[:3] @ XYZ_FROM_LINEAR_SRGB
    row_sums = combined.sum(axis=1)
    if np.any(row_sums <= 1e-5):
        raise ValueError("相机响应的行归一化不成立")
    normalized = combined / row_sums[:, None]
    condition = float(np.linalg.cond(normalized))
    if not math.isfinite(condition) or condition > _MAX_CONDITION or np.linalg.matrix_rank(normalized) != 3:
        raise ValueError("颜色矩阵奇异或数值条件过差")
    matrix = np.linalg.inv(normalized)
    if not np.isfinite(matrix).all():
        raise ValueError("颜色矩阵构造产生非有限值")
    return {"values": matrix.tolist(), "direction": MATRIX_DIRECTION,
            "construction": "inverse(row_normalize(cam_xyz_RGB @ XYZ_FROM_LINEAR_SRGB))",
            "cam_xyz_reported": cam_xyz.tolist(), "row_sums": row_sums.tolist(),
            "daylight_normalization_diagnostic": (1 / row_sums).tolist(),
            "condition_number": condition,
            "daylight_normalization_applied_separately": False}


def _white_balance(metadata):
    reported = _array(_field(metadata, "camera_whitebalance"), (4,), "参考帧白平衡")
    if np.any(reported[:3] <= 0) or reported[3] < 0:
        raise ValueError("参考帧 RGB 白平衡缺失或无效")
    if reported[3] == 0:
        mapping = "explicit_LibRaw_zero_G2_uses_G1"
    elif reported[3] == reported[1]:
        mapping = "reported_G1_equals_G2"
    else:
        raise ValueError("两个绿色白平衡不相等，不能在三通道合并后应用")
    gains = reported[:3] / reported[1]
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("白平衡归一化无效")
    return {"source": "reference_frame_libraw_camera_whitebalance", "reported_four_slots": reported.tolist(),
            "gains_rgb": gains.tolist(), "normalization": "G1_equals_one",
            "green_mapping": mapping, "coefficient_slot_mapping": [0, 1, 2, 1],
            "per_frame_estimation": False,
            "file_as_shot_neutral_independently_verified": False}


def _fingerprint(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_recipe(metadata: dict, reference_sha256: str, reference_white=None,
                 reference_black_green=None, *, common_scale=None, camera_identity=None) -> dict:
    """Fix WB, matrix and one common DN denominator from a reference frame.

    Either supply common_scale, or reference_white/reference_black_green. The
    latter black is only the caller-selected common scale baseline (historical
    parameter name); it is never subtracted here or claimed to equal all CFA
    black levels. Caller may select max(black_repeat) and record its own policy.
    Metadata fingerprint excludes frame WB; it is not proof of camera identity.
    """
    if not isinstance(reference_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", reference_sha256) is None:
        raise ValueError("需要参考原片 SHA256")
    model = _camera_model(metadata)
    baseline, white = None, None
    if reference_white is not None or reference_black_green is not None:
        white = _number(reference_white, "参考白位", positive=True)
        baseline = _number(reference_black_green, "公共尺度黑位")
        if baseline < 0 or baseline >= white:
            raise ValueError("公共尺度黑位必须非负且小于白位")
        derived = white - baseline
        if common_scale is not None and _number(common_scale, "公共尺度", positive=True) != derived:
            raise ValueError("公共尺度与白黑位差不一致")
        common_scale = derived
    scale = _number(common_scale, "公共尺度", positive=True)
    if camera_identity is not None and not isinstance(camera_identity, dict):
        raise ValueError("相机身份须为来源明确的对象")
    recipe = {
        "schema_version": 1, "algorithm": ALGORITHM,
        "reference": {"source_sha256": reference_sha256.lower(), "white_DN": white,
                      "common_black_for_scale_DN": baseline},
        "camera": {"metadata_fingerprint": _fingerprint(model), "model": model,
                   "caller_identity": camera_identity,
                   "identity_scope": "matching_reported_matrix_CFA_versions_not_camera_body_or_spectral_calibration"},
        "white_balance": _white_balance(metadata), "matrix": cam_xyz_to_rgb_cam(model["cam_xyz"]),
        "scale": {"common_denominator_DN": scale, "policy": "fixed_reference_common_DN_denominator",
                  "per_frame_estimation": False},
        "conditions": {"input_state": INPUT_STATE, "row_order": "TOP-DOWN", "channel_order": "RGB",
                       "input_dtype": "float32", "black_subtraction_performed_here": False,
                       "input_black_subtracted_required": True, "input_WB_applied_required": False,
                       "input_color_matrix_applied_required": False,
                       "WB_after_demosaic_exactness": "linear_interpolation_with_equal_green_gains_only"},
        "output": {"color_space": "approximate_linear_sRGB", "dtype": "float32",
                   "clipping_applied": False, "gamma_applied": False},
        "limitations": ["近似矩阵显色，不是完整 Adobe/DNG profile 或物理色彩标定",
                        "没有双光源插值、色相饱和度表、观感曲线、相机滤镜/改机校准",
                        "配方不按帧估计白平衡或尺度；调用者须核对解码/黑位/坐标及曝光兼容性",
                        "数组无法自证处理历史；此函数不检测伪装为cameraRGB的已显色数据"],
        "sources": dict(SOURCES),
    }
    # Copy caller metadata and ensure every public value is strict JSON.
    return json.loads(json.dumps(recipe, allow_nan=False))


def validate_recipe(recipe: dict, metadata: dict | None = None) -> dict:
    """Validate inputs exactly and matrix derivation within tight tolerances.

    Extra root fields remain allowed. Accepted stored coefficients are returned
    unchanged, not replaced with this machine's reconstruction. This handles
    last-bit differences, not arbitrary BLAS differences in ill-conditioned
    models; recomputed and recorded condition numbers must both stay <= 1e6.
    """
    if not isinstance(recipe, dict):
        raise ValueError("颜色配方必须为对象")
    try:
        if type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1 or recipe.get("algorithm") != ALGORITHM:
            raise ValueError("颜色配方版本不受支持")
        camera = recipe["camera"]
        if _fingerprint(camera["model"]) != camera["metadata_fingerprint"]:
            raise ValueError("配方相机元数据指纹不一致")
        if metadata is not None and _fingerprint(_camera_model(metadata)) != camera["metadata_fingerprint"]:
            raise ValueError("此帧的矩阵/CFA/解码版本与参考配方不兼容")
        reference = recipe["reference"]
        # Rebuild the core from recorded inputs: reject altered coefficients or
        # changed matrix direction rather than trusting serialized labels.
        model = camera["model"]
        recorded_metadata = {"available": True, "white_balance_applied": False,
                             "color_correction_applied": False, "versions": model["decoder_versions"],
                             "fields": {name: {"value": model[name]} for name in ("num_colors", "color_desc", "raw_pattern")}}
        recorded_metadata["fields"]["rgb_xyz_matrix"] = {"value": model["cam_xyz"]}
        recorded_metadata["fields"]["camera_whitebalance"] = {"value": recipe["white_balance"]["reported_four_slots"]}
        expected = build_recipe(recorded_metadata, reference["source_sha256"], reference["white_DN"],
                                reference["common_black_for_scale_DN"], common_scale=recipe["scale"]["common_denominator_DN"],
                                camera_identity=camera.get("caller_identity"))
        for name in ("camera", "reference", "white_balance", "scale", "conditions", "output"):
            if not _same_exact_json(recipe[name], expected[name]):
                raise ValueError(f"配方 {name} 与记录的输入/数学约定不一致")
        if not _same_matrix_section(recipe["matrix"], expected["matrix"]):
            raise ValueError("配方 matrix 与记录的输入/数学约定不一致")
        return json.loads(json.dumps(recipe, allow_nan=False))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("颜色配方缺失字段或结构无效") from exc


def _same_exact_json(recorded, expected):
    """Exact values/structure, without Python's True == 1/False == 0 alias."""
    if isinstance(expected, dict):
        return (isinstance(recorded, dict) and recorded.keys() == expected.keys()
                and all(_same_exact_json(recorded[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(recorded, list) and len(recorded) == len(expected)
                and all(_same_exact_json(a, b) for a, b in zip(recorded, expected)))
    if type(expected) in (int, float):
        return type(recorded) in (int, float) and recorded == expected
    return type(recorded) is type(expected) and recorded == expected


def _same_matrix_section(recorded, expected):
    derived = {"values": (3, 3), "row_sums": (3,),
               "daylight_normalization_diagnostic": (3,), "condition_number": ()}
    if not isinstance(recorded, dict) or recorded.keys() != expected.keys():
        return False
    for key, value in expected.items():
        if key not in derived:
            if not _same_exact_json(recorded[key], value):
                return False
            continue
        try:
            a = _array(recorded[key], derived[key], f"matrix.{key}")
            b = _array(value, derived[key], f"matrix.{key}")
        except (TypeError, ValueError, OverflowError):
            return False
        if key == "condition_number":
            if not 1 <= float(a) <= _MAX_CONDITION:
                return False
            rtol, atol = _CONDITION_RTOL, 0.
        else:
            if key != "values" and np.any(a <= 0):
                return False
            rtol = _MATRIX_RTOL
            atol = _MATRIX_ATOL if key == "values" else 0.
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            return False
    return True


def _backing_file(array):
    while array is not None:
        filename = getattr(array, "filename", None)
        if filename is not None:
            return Path(filename).resolve()
        array = getattr(array, "base", None)
    return None


def apply_color(camera_rgb: np.ndarray, recipe: dict, *, out=None,
                input_state=INPUT_STATE, row_order="TOP-DOWN", block_rows=256) -> np.ndarray:
    """Apply fixed WB/common-scale/matrix; out may be an independent memmap.

    Input is never modified. On failure an explicitly supplied out may contain
    incomplete rows; the caller must not publish it as a successful artifact.
    """
    recipe = validate_recipe(recipe)
    if input_state != INPUT_STATE or row_order != "TOP-DOWN":
        raise ValueError("需要已扣黑、未WB/校色的 TOP-DOWN camera RGB DN")
    if (not isinstance(camera_rgb, np.ndarray) or camera_rgb.dtype != np.float32 or camera_rgb.ndim != 3
            or camera_rgb.shape[2] != 3 or min(camera_rgb.shape[:2]) < 1):
        raise ValueError("需要 float32 HWC RGB 数组")
    if type(block_rows) is not int or not 1 <= block_rows <= 65536:
        raise ValueError("块行数必须为 1–65536 的整数")
    if out is None:
        out = np.empty(camera_rgb.shape, dtype=np.float32)
    if (not isinstance(out, np.ndarray) or out.shape != camera_rgb.shape or out.dtype != np.float32
            or not out.flags.writeable or np.shares_memory(out, camera_rgb)):
        raise ValueError("输出须为形状匹配、可写且不覆盖输入的 float32 数组")
    backing = _backing_file(camera_rgb)
    output_backing = _backing_file(out)
    if backing is not None and output_backing is not None:
        if backing == output_backing or backing.samefile(output_backing):
            raise ValueError("输出不能覆盖输入映射文件")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            factors = (np.asarray(recipe["white_balance"]["gains_rgb"], dtype=np.float64) /
                       recipe["scale"]["common_denominator_DN"]).astype(np.float32)
            matrix = np.asarray(recipe["matrix"]["values"], dtype=np.float32)
            if not np.isfinite(factors).all() or np.any(factors <= 0):
                raise ValueError("颜色系数不能安全表达为 float32")
            for y in range(0, camera_rgb.shape[0], block_rows):
                tile = camera_rgb[y:y + block_rows]
                if not np.isfinite(tile).all():
                    raise ValueError("cameraRGB 包含非有限值")
                balanced = tile * factors
                np.matmul(balanced, matrix.T, out=out[y:y + block_rows])
                if not np.isfinite(out[y:y + block_rows]).all():
                    raise ValueError("颜色输出发生非有限溢出")
    except (FloatingPointError, OverflowError) as exc:
        # Cover division, float32 conversions, WB multiplication and the final
        # matrix multiply alike. As documented above, out may be incomplete.
        raise ValueError("颜色运算无法在 float32 中安全表示（溢出或无效数值）") from exc
    return out
