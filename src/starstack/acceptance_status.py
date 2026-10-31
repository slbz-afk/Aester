"""Acceptance-status summarisation helpers.

This module is intentionally side-effect free: it only validates, copies and
renders caller-supplied data.  It does not read files, access the network or
depend on third-party packages.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["summarize_checks", "render_checks"]

_STATUS_PASSED = "passed"
_STATUS_FAILED = "failed"
_STATUS_NOT_APPLICABLE = "not_applicable"
_STATUS_INSUFFICIENT_DATA = "insufficient_data"

_STATUS_ORDER = (
    _STATUS_PASSED,
    _STATUS_FAILED,
    _STATUS_NOT_APPLICABLE,
    _STATUS_INSUFFICIENT_DATA,
)
_STATUS_LABELS = {
    _STATUS_PASSED: "通过",
    _STATUS_FAILED: "失败",
    _STATUS_NOT_APPLICABLE: "不适用",
    _STATUS_INSUFFICIENT_DATA: "样本不足",
}
_SUMMARY_KEYS = (
    "status",
    "passed",
    "checks",
    "counts",
    "failing",
    "incomplete",
    "not_applicable",
)
_NO_CHECKS_NOTE = "no checks supplied"
_TABLE_HEADER = "| 子项 | 状态 | 样本数 | 说明 |"
_TABLE_SEPARATOR = "| --- | --- | --- | --- |"


def _error(message: str) -> None:
    raise ValueError(message)


def _copy_json_value(value: Any, path: str, active: set[int]) -> Any:
    """Return a deep, strict-JSON copy of *value* or raise ValueError."""
    if value is None:
        return None
    if value is True or value is False:
        return value

    value_type = type(value)

    if value_type is int:
        return value

    if value_type is float:
        if math.isnan(value) or math.isinf(value):
            _error(f"{path}: NaN and Infinity are not valid strict JSON numbers")
        return value

    if value_type is str:
        return value

    if value_type is list:
        identity = id(value)
        if identity in active:
            _error(f"{path}: circular references are not strict JSON")
        active.add(identity)
        try:
            return [
                _copy_json_value(item, f"{path}[{index}]", active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)

    if value_type is dict:
        identity = id(value)
        if identity in active:
            _error(f"{path}: circular references are not strict JSON")
        active.add(identity)
        try:
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    _error(f"{path}: strict JSON object keys must be strings")
                copied[key] = _copy_json_value(item, f"{path}.{key}", active)
            return copied
        finally:
            active.remove(identity)

    raise ValueError(f"{path}: unsupported type for strict JSON: {value_type.__name__}")


def _normalise_check(
    check: dict[str, Any],
    index: int,
    seen_names: set[str],
) -> dict[str, Any]:
    if "name" not in check:
        _error(f"checks[{index}] is missing required field 'name'")
    name = check["name"]
    if type(name) is not str or name == "":
        _error(f"checks[{index}]['name'] must be a non-empty string")
    if name in seen_names:
        _error(f"checks[{index}]['name'] is duplicated: {name!r}")
    seen_names.add(name)

    if "status" not in check:
        _error(f"checks[{index}] is missing required field 'status'")
    status = check["status"]
    if type(status) is not str or status not in _STATUS_ORDER:
        _error(f"checks[{index}]['status'] is not an allowed status")

    if "samples" not in check:
        _error(f"checks[{index}] is missing required field 'samples'")
    samples = check["samples"]
    if type(samples) is not int:
        _error(f"checks[{index}]['samples'] must be a strict built-in int")
    if samples < 0:
        _error(f"checks[{index}]['samples'] must be non-negative")

    if "reason" not in check:
        _error(f"checks[{index}] is missing required field 'reason'")
    reason = check["reason"]
    if type(reason) is not str or reason == "":
        _error(f"checks[{index}]['reason'] must be a non-empty string")

    if status == _STATUS_PASSED and samples <= 0:
        _error(f"checks[{index}] cannot be passed with zero samples")
    if status == _STATUS_NOT_APPLICABLE and samples != 0:
        _error(f"checks[{index}] is not_applicable but has samples != 0")

    if "metrics" not in check:
        check["metrics"] = {}
    elif type(check["metrics"]) is not dict:
        _error(f"checks[{index}]['metrics'] must be a dict")

    return check


def summarize_checks(checks: Any) -> dict[str, Any]:
    """Validate and summarise acceptance checks."""
    if type(checks) is not list:
        _error("checks must be a list")

    try:
        copied_checks = _copy_json_value(checks, "checks", set())
    except RecursionError:
        _error("checks contains a circular or too deeply nested structure")

    # The copy enforces JSON types; the standard encoder also enforces this
    # runtime's finite recursion / integer-to-decimal serialization limits.
    try:
        json.dumps(copied_checks, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("checks cannot be serialized as strict JSON") from exc

    if type(copied_checks) is not list:
        _error("checks must be a list")

    seen_names: set[str] = set()
    normalised_checks: list[dict[str, Any]] = []

    for index, check in enumerate(copied_checks):
        if type(check) is not dict:
            _error(f"checks[{index}] must be a dict")
        normalised_checks.append(_normalise_check(check, index, seen_names))

    counts = {status: 0 for status in _STATUS_ORDER}
    for check in normalised_checks:
        counts[check["status"]] += 1

    failing: list[str] = []
    incomplete: list[str] = []
    not_applicable: list[str] = []

    for check in normalised_checks:
        label = f"{check['name']}: {check['reason']}"
        status = check["status"]
        if status == _STATUS_FAILED:
            failing.append(label)
        elif status == _STATUS_INSUFFICIENT_DATA:
            incomplete.append(label)
        elif status == _STATUS_NOT_APPLICABLE:
            not_applicable.append(label)

    if not normalised_checks:
        incomplete = [_NO_CHECKS_NOTE]

    total = len(normalised_checks)
    if counts[_STATUS_FAILED] > 0:
        overall = _STATUS_FAILED
    elif counts[_STATUS_INSUFFICIENT_DATA] > 0:
        overall = _STATUS_INSUFFICIENT_DATA
    elif counts[_STATUS_PASSED] > 0:
        overall = _STATUS_PASSED
    elif total > 0:
        overall = _STATUS_NOT_APPLICABLE
    else:
        overall = _STATUS_INSUFFICIENT_DATA

    return {
        "status": overall,
        "passed": overall == _STATUS_PASSED,
        "checks": normalised_checks,
        "counts": counts,
        "failing": failing,
        "incomplete": incomplete,
        "not_applicable": not_applicable,
    }


def _escape_table_cell(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("|", "\\|")
    escaped = escaped.replace("&", "&amp;")
    escaped = escaped.replace("<", "&lt;")
    escaped = escaped.replace(">", "&gt;")
    escaped = escaped.replace("\r\n", "\n")
    escaped = escaped.replace("\r", "\n")
    escaped = escaped.replace("\n", "<br>")
    return escaped


def _validate_summary(summary: Any) -> dict[str, Any]:
    if type(summary) is not dict:
        _error("summary must be a dict")

    if set(summary) != set(_SUMMARY_KEYS):
        _error("summary must contain exactly the fixed summary fields")

    for key in _SUMMARY_KEYS:
        if key not in summary:
            _error(f"summary is missing required field: {key}")

    checks = summary["checks"]
    if type(checks) is not list:
        _error("summary['checks'] must be a list")

    recomputed = summarize_checks(checks)

    status = summary["status"]
    if type(status) is not str:
        _error("summary['status'] must be a string")
    if status != recomputed["status"]:
        _error("summary['status'] is inconsistent with summary['checks']")

    passed = summary["passed"]
    if type(passed) is not bool:
        _error("summary['passed'] must be a bool")
    if passed != recomputed["passed"]:
        _error("summary['passed'] is inconsistent with summary['checks']")

    counts = summary["counts"]
    if type(counts) is not dict:
        _error("summary['counts'] must be a dict")

    if set(counts) != set(_STATUS_ORDER):
        _error("summary['counts'] must contain exactly the four status fields")

    for status_key in _STATUS_ORDER:
        if status_key not in counts:
            _error(f"summary['counts'] is missing {status_key!r}")
        value = counts[status_key]
        if type(value) is not int:
            _error(f"summary['counts'][{status_key!r}] must be an int")
        if value < 0:
            _error(f"summary['counts'][{status_key!r}] must be non-negative")
        if value != recomputed["counts"][status_key]:
            _error(
                f"summary['counts'][{status_key!r}] is inconsistent "
                "with summary['checks']"
            )

    for key in ("failing", "incomplete", "not_applicable"):
        value = summary[key]
        if type(value) is not list:
            _error(f"summary[{key!r}] must be a list")
        for index, item in enumerate(value):
            if type(item) is not str:
                _error(f"summary[{key!r}][{index}] must be a string")
        if value != recomputed[key]:
            _error(f"summary[{key!r}] is inconsistent with summary['checks']")

    return recomputed


def render_checks(summary: Any) -> str:
    """Render a validated summary as a Markdown table in Chinese."""
    validated = _validate_summary(summary)

    lines: list[str] = [
        f"总体状态：{_STATUS_LABELS[validated['status']]}",
        "",
        _TABLE_HEADER,
        _TABLE_SEPARATOR,
    ]

    for check in validated["checks"]:
        name = _escape_table_cell(check["name"])
        status_label = _STATUS_LABELS[check["status"]]
        samples = str(check["samples"])
        reason = _escape_table_cell(check["reason"])
        lines.append(f"| {name} | {status_label} | {samples} | {reason} |")

    if not validated["checks"]:
        lines.append("")
        lines.append(_NO_CHECKS_NOTE)

    return "\n".join(lines)
