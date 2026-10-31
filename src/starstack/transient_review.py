"""Read-only recall review of recorded candidate dispositions.

This module does not classify objects, select trails, modify policy, or process
pixels. A review priority orders attention; it is never an identity probability.
Legacy records need only id, decision, and an optional measurement dictionary.
"""
from __future__ import annotations

VERSION = "transient-recall-review-1"
_CLASSES = frozenset(("meteor", "satellite", "aircraft", "artifact", "unknown"))
_KNOWN_MANUAL = _CLASSES - {"unknown"}
_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def review_candidate(candidate: dict) -> dict:
    """Return new review metadata without changing any part of the candidate.

    A manual unknown label is unresolved. A manual search mark (source=manual)
    is not a reviewed disposition. Known manual labels and override=omit record
    a deliberate decision, but do not certify physical identity. Explicit keep
    requests which could not be measured take precedence over every other state.
    """
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a dictionary")
    cid = candidate.get("id")
    if not isinstance(cid, str) or not cid:
        raise ValueError("candidate must have a nonempty string id")
    decision = candidate.get("decision")
    if decision not in ("keep", "omit"):
        raise ValueError("candidate must have a recorded keep/omit decision")
    measurement = _mapping(candidate.get("measurement"))
    measured = measurement.get("status") == "measured"
    manual_label = candidate.get("label")
    for field in ("label", "effective_label"):
        value = candidate.get(field)
        if value is not None and (not isinstance(value, str) or value not in _CLASSES):
            raise ValueError(f"candidate {field} must be a known class string or None")
    suggestion = candidate.get("suggestion")
    if suggestion is not None:
        if not isinstance(suggestion, dict):
            raise ValueError("candidate suggestion must be a dictionary or None")
        value = suggestion.get("class")
        if value is not None and (not isinstance(value, str) or value not in _CLASSES):
            raise ValueError("candidate suggestion.class must be a known class string or None")
    suggested = _mapping(candidate.get("suggestion")).get("class")
    effective = manual_label if manual_label in _CLASSES else candidate.get("effective_label")
    if effective not in _CLASSES:
        effective = suggested if suggested in _CLASSES else "unknown"
    override = candidate.get("override")
    mark = _mapping(candidate.get("mark"))
    # Intent in newer records is authoritative, with explicit fields providing
    # the compatible fallback for older records. An omit override cancels a mark.
    requested_keep = (override == "keep" or (override != "omit" and (
        candidate.get("requested_decision") == "keep" or mark.get("protect") is True
        or decision == "keep")))
    explicit_omit = override == "omit"
    known_manual = manual_label in _KNOWN_MANUAL
    reasons = []
    priority = "low"
    if requested_keep and not measured:
        status = "requested_keep_unmeasured"
        priority = "high"
        reasons.append("requested_keep_not_measured")
        message = "已请求保留，但未能测量并兑现；请优先检查原片与标记范围。"
    elif decision == "keep":
        status = "retained"
        reasons.append("recorded_keep_decision")
        message = "记录为保留；是否实际加回及完整范围仍以应用证据为准。"
    elif explicit_omit or known_manual:
        status = "explicitly_reviewed_decision"
        reasons.append("explicit_omit_override" if explicit_omit else "known_manual_label")
        message = "已有明确人工标签或省略处置；这记录了取舍，不认证物理身份。"
    else:
        status = "omitted_unverified"
        evidence = _mapping(candidate.get("evidence"))
        ends = _mapping(_mapping(evidence.get("light_curve")).get("ends"))
        raw_ends = _mapping(measurement.get("ends"))
        quality = _mapping(measurement.get("quality"))
        if any(ends.get(end) == "faded" for end in ("start", "stop")):
            reasons.append("faded_endpoint_omitted")
        if any(_mapping(raw_ends.get(end)).get("at_border") is True for end in ("start", "stop")):
            reasons.append("frame_border_endpoint")
        if quality.get("profile_boundary_hit") is True:
            reasons.append("measurement_profile_boundary")
        if reasons:
            priority = "high"
        elif measured and effective == "unknown":
            priority = "normal"
        reasons.append("automatic_omission_is_not_nonmeteor_confirmation")
        if not measured:
            reasons.append("candidate_not_measured")
        if manual_label == "unknown":
            reasons.append("manual_unknown_remains_unresolved")
        message = ("已自动省略，但身份尚未核实，不能视为已排除流星。"
                   if measured else "候选未能测量而被省略，不能据此确认原片没有流星。")
    return {
        "version": VERSION, "candidate_id": cid, "status": status,
        "measurement": "measured" if measured else "unmeasured",
        "measurement_status": measurement.get("status"), "priority": priority,
        "effective_label": effective, "reason_codes": reasons, "message": message,
        "physical_identity_certified": False, "recall_not_certified": True,
    }


def summarize_review(record: dict) -> dict:
    """Summarize unresolved omissions and unfulfilled keep requests.

    candidate_ids contains only these attention items, ordered high/normal/low,
    then lexicographically by the original stable ID. The policy and candidate
    list are never rewritten. Counting an item does not request automatic keep.
    """
    if not isinstance(record, dict) or not isinstance(record.get("candidates"), list):
        raise ValueError("record must contain a candidates list")
    reviews = [review_candidate(c) for c in record["candidates"]]
    ids = [r["candidate_id"] for r in reviews]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    auto = [r for r in reviews if r["status"] == "omitted_unverified"]
    requested = [r for r in reviews if r["status"] == "requested_keep_unmeasured"]
    attention = sorted(auto + requested, key=lambda r: (_PRIORITY_ORDER[r["priority"]], r["candidate_id"]))
    counts = {
        "total": len(auto),
        "measured_unknown": sum(r["measurement"] == "measured" and r["effective_label"] == "unknown" for r in auto),
        "unmeasured": sum(r["measurement"] == "unmeasured" for r in auto),
        "high_priority": sum(r["priority"] == "high" for r in auto),
    }
    high = sum(r["priority"] == "high" for r in attention)
    message = (f"有 {counts['total']} 条候选被自动省略且尚未核实，其中 {counts['measured_unknown']} 条已测量但身份未知、"
               f"{counts['unmeasured']} 条未能测量；另有 {len(requested)} 条保留请求未兑现。"
               f"建议优先复核 {high} 条。未知或未测量不等于非流星，当前不能保证没有遗漏。")
    if not attention:
        message = "没有记录内待复核的自动省略或未兑现保留请求；这不证明检测已覆盖全部流星。"
    return {
        "version": VERSION, "total_candidates": len(reviews),
        "automatic_omitted": counts, "requested_keep_unmeasured": len(requested),
        "retained": sum(r["status"] == "retained" for r in reviews),
        "explicitly_reviewed_decision": sum(r["status"] == "explicitly_reviewed_decision" for r in reviews),
        "high_priority": high, "candidate_ids": [r["candidate_id"] for r in attention],
        "candidate_ids_by_priority": {p: [r["candidate_id"] for r in attention if r["priority"] == p]
                                      for p in ("high", "normal", "low")},
        "recall_not_certified": True, "physical_identity_certified": False, "message": message,
    }


def candidate_view(c: dict) -> dict:
    """The compact, review-oriented view of one candidate."""
    m = c.get("measurement") or {}
    ev = c.get("evidence") or {}
    lc = ev.get("light_curve") or {}
    return {"id": c["id"], "frame": c["stem"], "input": ((c.get("provenance") or {}).get("input") or {}).get("path"),
            "date_obs": (c.get("provenance") or {}).get("date_obs"), "source": c.get("source"), "status": m.get("status"),
            "length_px": m.get("length_px"), "contrast_median": m.get("contrast_median"), "total_snr": m.get("total_snr"),
            "colour": m.get("colour"), "endpoints_native_source": (c.get("geometry") or {}).get("native_source"),
            "native_source_status": (c.get("geometry") or {}).get("native_source_status", "unknown"),
            "native_source_reason": (c.get("geometry") or {}).get("native_source_reason"),
            "evidence": {"continues": [link.get("candidate") for link in ev.get("continues") or []], "persistent": ev.get("persistent"),
                         "periodic": (ev.get("periodic") or {}).get("periodic"), "ends": lc.get("ends"),
                         "end_levels": [lc.get("start_level"), lc.get("stop_level")]},
            "suggestion": c.get("suggestion"), "label": c.get("label"), "override": c.get("override"),
            "label_source": c.get("label_source"), "override_source": c.get("override_source"), "keep_source": c.get("keep_source"),
            "effective_label": c.get("effective_label"), "decision": c.get("decision"), "decision_reason": c.get("decision_reason"),
            "requested_decision": c.get("requested_decision"), "requested_reason": c.get("requested_reason"),
            "protection_status": c.get("protection_status"), "protection_reason": c.get("protection_reason"),
            "protection_explicit": c.get("protection_explicit"), "application": c.get("application"),
            "related_candidate_ids": c.get("related_candidate_ids"), "relation_conflicts": c.get("relation_conflicts"),
            "measurement_quality": m.get("quality"), "clipping_evidence": m.get("clipping_evidence", "unknown"),
            "mark_association": (c.get("detection") or {}).get("mark_association"),
            "review_png": c.get("review_png"), "note": c.get("note"), "recall_review": review_candidate(c)}
