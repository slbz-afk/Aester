"""Pure bright-trail candidate intent resolver.

This module performs no image processing, no identity grouping, no geometry,
and no classification inference. It resolves caller-supplied metadata into
review intents. A protection status of "eligible" means only that the desired
keep is measurable; it is never a claim that protection was applied.
"""

from copy import deepcopy

_ALLOWED_CLASSES = ("meteor", "satellite", "aircraft", "artifact", "unknown")
_ALLOWED_CLASS_SET = frozenset(_ALLOWED_CLASSES)
_DECISIONS = frozenset(("keep", "omit"))
_DEFAULT_POLICY = {
    "meteor": "keep",
    "satellite": "omit",
    "aircraft": "omit",
    "artifact": "omit",
    "unknown": "omit",
}


def _merge_policy(policy):
    merged = dict(_DEFAULT_POLICY)
    if policy is None:
        return merged
    if type(policy) is not dict:
        raise ValueError("policy must be None or a builtin dict")
    for class_name, decision in policy.items():
        if not isinstance(class_name, str) or class_name not in _ALLOWED_CLASS_SET:
            raise ValueError("policy contains an unsupported class key")
        if not isinstance(decision, str) or decision not in _DECISIONS:
            raise ValueError("policy values must be 'keep' or 'omit'")
        merged[class_name] = decision
    return merged


def _resolve_one(candidate, index, merged_policy):
    if type(candidate) is not dict:
        raise ValueError(f"candidates[{index}] must be a builtin dict")

    measurement = candidate.get("measurement")
    if type(measurement) is not dict:
        raise ValueError(f"candidates[{index}]['measurement'] must be a builtin dict")
    if "status" in measurement:
        status = measurement["status"]
        if not isinstance(status, str):
            raise ValueError(
                f"candidates[{index}]['measurement']['status'] must be a string"
            )
    else:
        status = None

    manual_label = candidate.get("label")
    if manual_label is not None:
        if not isinstance(manual_label, str) or manual_label not in _ALLOWED_CLASS_SET:
            raise ValueError(
                f"candidates[{index}]['label'] must be an allowed class string"
            )

    suggestion_label = None
    suggestion = candidate.get("suggestion")
    if suggestion is not None:
        if type(suggestion) is not dict:
            raise ValueError(f"candidates[{index}]['suggestion'] must be a builtin dict")
        if "class" in suggestion:
            suggested_class = suggestion["class"]
            if suggested_class is not None:
                if (
                    not isinstance(suggested_class, str)
                    or suggested_class not in _ALLOWED_CLASS_SET
                ):
                    raise ValueError(
                        f"candidates[{index}]['suggestion']['class'] must be an "
                        "allowed class string"
                    )
                suggestion_label = suggested_class

    override = candidate.get("override")
    if override is not None:
        if not isinstance(override, str) or override not in _DECISIONS:
            raise ValueError(
                f"candidates[{index}]['override'] must be 'keep' or 'omit'"
            )

    mark_protect = False
    mark = candidate.get("mark")
    if mark is not None:
        if type(mark) is not dict:
            raise ValueError(f"candidates[{index}]['mark'] must be a builtin dict")
        if "protect" in mark:
            protect = mark["protect"]
            if type(protect) is not bool:
                raise ValueError(
                    f"candidates[{index}]['mark']['protect'] must be exactly bool"
                )
            mark_protect = protect is True

    if manual_label is not None:
        effective_label = manual_label
    elif suggestion_label is not None:
        effective_label = suggestion_label
    else:
        effective_label = "unknown"

    if override is not None:
        requested_decision = override
        requested_reason = "override"
    elif mark_protect:
        requested_decision = "keep"
        requested_reason = "mark_protect"
    else:
        requested_decision = merged_policy[effective_label]
        requested_reason = "policy"

    protection_explicit = (
        requested_decision == "keep"
        and requested_reason in ("override", "mark_protect")
    )

    measured = status == "measured"
    decision = requested_decision if measured else "omit"

    if not measured:
        decision_reason = f"not measured ({status})"
    elif requested_reason == "override":
        decision_reason = f"override {requested_decision}"
    elif requested_reason == "mark_protect":
        decision_reason = "explicit mark protection"
    else:
        label_origin = (
            "manual label" if manual_label is not None else "suggested label"
        )
        decision_reason = (
            f"policy {effective_label}: {requested_decision} ({label_origin})"
        )

    if requested_decision == "omit":
        protection_status = "not_requested"
        protection_reason = None
    elif measured:
        protection_status = "eligible"
        protection_reason = None
    else:
        protection_status = "needs_review"
        protection_reason = f"not_measured:{status}"

    resolved = deepcopy(candidate)
    resolved["effective_label"] = effective_label
    resolved["requested_decision"] = requested_decision
    resolved["requested_reason"] = requested_reason
    resolved["protection_explicit"] = protection_explicit
    resolved["decision"] = decision
    resolved["decision_reason"] = decision_reason
    resolved["protection_status"] = protection_status
    resolved["protection_reason"] = protection_reason
    return resolved


def resolve_candidate_intents(candidates, policy=None):
    """Resolve candidate intents without mutating caller input."""
    if type(candidates) is not list:
        raise ValueError("candidates must be a builtin list")

    merged_policy = _merge_policy(policy)

    summary = {
        "requested_keep": 0,
        "explicit_requested_keep": 0,
        "eligible_keep": 0,
        "needs_review": 0,
        "not_requested": 0,
    }

    resolved_candidates = []
    for index, candidate in enumerate(candidates):
        resolved = _resolve_one(candidate, index, merged_policy)
        resolved_candidates.append(resolved)

        if resolved["requested_decision"] == "keep":
            summary["requested_keep"] += 1
        if resolved["protection_explicit"] is True:
            summary["explicit_requested_keep"] += 1

        protection_status = resolved["protection_status"]
        if protection_status == "eligible":
            summary["eligible_keep"] += 1
        elif protection_status == "needs_review":
            summary["needs_review"] += 1
        elif protection_status == "not_requested":
            summary["not_requested"] += 1

    return {
        "candidates": resolved_candidates,
        "policy": merged_policy,
        "summary": summary,
    }
