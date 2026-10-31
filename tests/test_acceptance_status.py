"""Independent contract and false-pass counterexamples for acceptance summaries."""
from copy import deepcopy
import json
import math
from types import MappingProxyType

import numpy as np
import pytest

from starstack.acceptance_status import render_checks, summarize_checks


STATES = ("passed", "failed", "not_applicable", "insufficient_data")


def check(name="one", status="passed", samples=None, reason="reason", **extra):
    return {"name": name, "status": status, "samples": (1 if status == "passed" else 0) if samples is None else samples,
            "reason": reason, **extra}


class IntSubclass(int):
    pass


def test_empty_is_insufficient_and_never_success():
    result = summarize_checks([])
    assert result == {"status": "insufficient_data", "passed": False, "checks": [],
                      "counts": {state: 0 for state in STATES}, "failing": [],
                      "incomplete": ["no checks supplied"], "not_applicable": []}
    text = render_checks(result)
    assert text.startswith("总体状态：样本不足") and "no checks supplied" in text


def test_all_not_applicable_does_not_become_passed():
    result = summarize_checks([check("first", "not_applicable"), check("second", "not_applicable")])
    assert result["status"] == "not_applicable" and result["passed"] is False
    assert result["counts"]["not_applicable"] == 2
    assert result["not_applicable"] == ["first: reason", "second: reason"]


@pytest.mark.parametrize("states,expected", [
    (["passed"], "passed"),
    (["not_applicable", "passed"], "passed"),
    (["passed", "insufficient_data"], "insufficient_data"),
    (["insufficient_data", "not_applicable"], "insufficient_data"),
    (["passed", "insufficient_data", "failed", "not_applicable"], "failed"),
    (["failed", "insufficient_data"], "failed"),
])
def test_priority_preserves_all_categories(states, expected):
    result = summarize_checks([check(str(i), state) for i, state in enumerate(states)])
    assert result["status"] == expected
    assert result["passed"] is (expected == "passed")
    assert result["counts"] == {state: states.count(state) for state in STATES}
    assert result["failing"] == [f"{i}: reason" for i, state in enumerate(states) if state == "failed"]
    assert result["incomplete"] == [f"{i}: reason" for i, state in enumerate(states) if state == "insufficient_data"]


def test_deep_copy_and_default_metrics_do_not_mutate_or_alias_input():
    shared = {"nested": [{"x": 2}]}
    original = [check("first", metrics=shared, extra=[1, {"ok": True}]), check("second", metrics=shared), check("third")]
    before = deepcopy(original)
    result = summarize_checks(original)
    assert original == before
    assert "metrics" not in original[2] and result["checks"][2]["metrics"] == {}
    result["checks"][0]["metrics"]["nested"][0]["x"] = 9
    result["checks"][0]["extra"][1]["ok"] = False
    assert original == before
    assert result["checks"][1]["metrics"]["nested"][0]["x"] == 2


@pytest.mark.parametrize("value", [None, {}, (), 1, True, "checks", MappingProxyType({})])
def test_checks_requires_builtin_list(value):
    with pytest.raises(ValueError):
        summarize_checks(value)


@pytest.mark.parametrize("value", [None, [], "check", True, 1])
def test_each_check_requires_dict(value):
    with pytest.raises(ValueError):
        summarize_checks([value])


@pytest.mark.parametrize("field", ["name", "status", "samples", "reason"])
def test_each_required_field_is_required(field):
    value = check()
    del value[field]
    with pytest.raises(ValueError):
        summarize_checks([value])


@pytest.mark.parametrize("field", ["name", "reason"])
@pytest.mark.parametrize("value", ["", None, False, 1, [], {}])
def test_names_and_reasons_require_nonempty_strings(field, value):
    with pytest.raises(ValueError):
        summarize_checks([{**check(), field: value}])


@pytest.mark.parametrize("status", ["pass", "PASSED", "unknown", "", 0, None, True])
def test_only_explicit_statuses_are_accepted(status):
    with pytest.raises(ValueError):
        summarize_checks([check(status=status)])


@pytest.mark.parametrize("samples", [-1, True, False, 1.0, "1", None, math.nan, math.inf, IntSubclass(1), np.int64(1)])
def test_samples_are_strict_nonnegative_builtin_integers(samples):
    with pytest.raises(ValueError):
        summarize_checks([{**check(), "samples": samples}])


@pytest.mark.parametrize("status,samples", [("passed", 0), ("not_applicable", 1)])
def test_sample_status_contract_cannot_claim_no_data_passed(status, samples):
    with pytest.raises(ValueError):
        summarize_checks([check(status=status, samples=samples)])


@pytest.mark.parametrize("status,samples", [("failed", 0), ("insufficient_data", 0), ("insufficient_data", 3)])
def test_failure_and_insufficiency_may_report_zero_or_available_samples(status, samples):
    assert summarize_checks([check(status=status, samples=samples)])["status"] == status


def test_duplicate_names_fail_without_mutating_original():
    original = [check(), check(status="failed")]
    before = deepcopy(original)
    with pytest.raises(ValueError):
        summarize_checks(original)
    assert original == before


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, {1, 2}, (1, 2), b"bytes", object(), np.float64(1)])
def test_nested_metadata_requires_strict_json(bad):
    with pytest.raises(ValueError):
        summarize_checks([check(extra={"deep": [bad]})])


@pytest.mark.parametrize("metrics", [None, [], (), 1, True, "metrics"])
def test_metrics_must_be_dict(metrics):
    with pytest.raises(ValueError):
        summarize_checks([check(metrics=metrics)])


def test_object_keys_and_cycles_rejected():
    with pytest.raises(ValueError):
        summarize_checks([check(metrics={1: "not a JSON string key"})])
    value = check()
    value["cycle"] = value
    with pytest.raises(ValueError):
        summarize_checks([value])
    nested = []
    nested.append(nested)
    with pytest.raises(ValueError):
        summarize_checks([check(extra=nested)])


def test_serialization_limit_cannot_leak_nonserializable_summary():
    # The default Python guard rejects decimal conversion beyond 4300 digits.
    # No module-specific scientific threshold is introduced by this check.
    with pytest.raises(ValueError):
        summarize_checks([check(samples=10**5000)])


def test_valid_metadata_and_large_safe_integer_remain_serializable():
    result = summarize_checks([check(samples=10**100, metrics={"nullable": None, "finite": -0.2, "enabled": True})])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("field", ["status", "passed", "checks", "counts", "failing", "incomplete", "not_applicable"])
def test_render_requires_complete_summary(field):
    value = summarize_checks([check()])
    del value[field]
    with pytest.raises(ValueError):
        render_checks(value)


@pytest.mark.parametrize("field,replacement", [("status", "passed"), ("passed", True), ("passed", 0),
                                               ("failing", []), ("incomplete", ["fake"]), ("not_applicable", ["fake"])])
def test_render_rejects_forged_summary_even_if_passed_looks_true(field, replacement):
    value = summarize_checks([check(status="failed")])
    value[field] = replacement
    with pytest.raises(ValueError):
        render_checks(value)


@pytest.mark.parametrize("replacement", [True, 1.0, -1, 999, np.int64(1), IntSubclass(1)])
def test_render_rejects_invalid_or_forged_counts(replacement):
    value = summarize_checks([check()])
    value["counts"]["passed"] = replacement
    with pytest.raises(ValueError):
        render_checks(value)


@pytest.mark.parametrize("where", ["summary", "counts"])
@pytest.mark.parametrize("extra", [math.nan, math.inf, {1}, "otherwise JSON-safe but outside fixed keys"])
def test_render_rejects_extra_unvalidated_summary_fields(where, extra):
    value = summarize_checks([check()])
    target = value if where == "summary" else value["counts"]
    target["unexpected"] = extra
    with pytest.raises(ValueError):
        render_checks(value)


def test_render_revalidates_check_data_and_leaves_input_unchanged():
    value = summarize_checks([check()])
    before = deepcopy(value)
    render_checks(value)
    assert value == before
    value["checks"][0]["metrics"]["bad"] = math.nan
    with pytest.raises(ValueError):
        render_checks(value)


def test_markdown_table_escapes_cells_and_omits_metrics():
    name = "A|B\\C\n<script>"
    reason = "R\r\nX\rY & <img src=x>"
    value = summarize_checks([check(name, reason=reason, metrics={"private": "DO_NOT_RENDER"}),
                              check("second", "not_applicable", reason="out of scope")])
    text = render_checks(value)
    assert text.startswith("总体状态：通过")
    assert "| 子项 | 状态 | 样本数 | 说明 |" in text
    assert "A\\|B\\\\C<br>&lt;script&gt;" in text
    assert "R<br>X<br>Y &amp; &lt;img src=x&gt;" in text
    assert "<script>" not in text and "<img" not in text and "DO_NOT_RENDER" not in text
    assert "| second | 不适用 | 0 | out of scope |" in text
    assert len(text.splitlines()) == 6
