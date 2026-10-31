"""亮迹（mark）几何参数的手工校验。

本模块只做结构/数值检查，不处理帧名、图像解码、坐标变换或输出状态。
注意：未提供 width/height 时只能做结构校验；真实画幅边界检查必须由调用者
在坐标变换或写状态之前补足。本模块不裁切线段，也不猜测缺失的画幅。
"""

import math

# 坐标上限是像素地址范围防护，不是图像质量阈值。
MAX_COORD = 2**31 - 1

_MAX_FRAME_SIDE = 2**31
_DEFAULT_HALF_WIDTH_PX = 30.0
_MIN_HALF_WIDTH_PX = 4.0
_MAX_HALF_WIDTH_PX = 200.0
_REQUIRED_KEYS = ("x0", "y0", "x1", "y1")


def _as_float(value, label: str) -> float:
    """只接受 Python int/float（拒绝 bool），并把 -0.0 归一为 +0.0。"""
    if type(value) not in (int, float):
        raise ValueError(f"{label} 必须是 int 或 float")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} 数值过大，无法转换为 float") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 无法转换为 float") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} 必须是有限数值")
    return number + 0.0


def _coordinate(value, label: str) -> float:
    number = _as_float(value, label)
    if not 0.0 <= number <= MAX_COORD:
        raise ValueError(f"{label} 必须在 0 到 {MAX_COORD} 之间")
    return number


def _half_width(value) -> float:
    number = _as_float(value, "half_width_px")
    if not _MIN_HALF_WIDTH_PX <= number <= _MAX_HALF_WIDTH_PX:
        raise ValueError("half_width_px 必须在 4 到 200 之间")
    return number


def _frame_side(value, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} 必须是正整数")
    if not 1 <= value <= _MAX_FRAME_SIDE:
        raise ValueError(f"{label} 必须在 1 到 {_MAX_FRAME_SIDE} 之间")
    return value


def validate_mark_values(mark, *, width=None, height=None) -> dict:
    """校验几何；显式 protect 只接受 bool，缺省仍为搜索走廊。"""
    if not isinstance(mark, dict):
        raise ValueError("mark 必须是 dict")
    if "protect" in mark and type(mark["protect"]) is not bool:
        raise ValueError("protect 必须是 bool；true 请求保护，省略或 false 仅搜索")
    for key in _REQUIRED_KEYS:
        if key not in mark:
            raise ValueError(f"mark 缺少 {key}")

    x0 = _coordinate(mark["x0"], "x0")
    y0 = _coordinate(mark["y0"], "y0")
    x1 = _coordinate(mark["x1"], "x1")
    y1 = _coordinate(mark["y1"], "y1")
    if x0 == x1 and y0 == y1:
        raise ValueError("亮迹两个端点不能完全相同")

    half_width_px = _half_width(mark.get("half_width_px", _DEFAULT_HALF_WIDTH_PX))

    if (width is None) != (height is None):
        raise ValueError("width 与 height 必须同时提供或同时省略")

    # width/height 同时为 None：只做结构校验，画幅边界由调用者在坐标变换
    # 或写状态之前补足；同时提供：检查端点是否越界，不裁切、不猜画幅。
    if width is not None:
        width = _frame_side(width, "width")
        height = _frame_side(height, "height")
        if x0 > width - 1 or y0 > height - 1 or x1 > width - 1 or y1 > height - 1:
            raise ValueError("端点坐标超出画幅范围")

    result = {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "half_width_px": half_width_px,
    }
    if "protect" in mark:
        result["protect"] = mark["protect"]
    return result
