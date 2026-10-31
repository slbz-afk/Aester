"""Strict batch review intent validation. See DeepSeek task receipt 20260924."""

_MAX_ENTRIES = 5000
_MAX_ID_LEN = 256
_MAX_NOTE_LEN = 2000
_LABELS = frozenset(("meteor", "satellite", "aircraft", "artifact", "unknown"))
_OVERRIDES = frozenset(("keep", "omit"))
_FIELDS = frozenset(("label", "override", "note"))


def normalize_decisions(decisions, candidate_ids):
    if type(candidate_ids) is not list:
        raise ValueError("candidate_ids must be an exact built-in list")
    if len(candidate_ids) > _MAX_ENTRIES:
        raise ValueError("candidate_ids must contain at most 5000 entries")

    known_ids = set()
    for cid in candidate_ids:
        if type(cid) is not str:
            raise ValueError("each candidate id must be an exact built-in str")
        if len(cid) > _MAX_ID_LEN:
            raise ValueError("candidate id must be at most 256 characters")
        if any(ord(ch) <= 0x1F or ord(ch) == 0x7F for ch in cid):
            raise ValueError("candidate id must not contain control characters")
        if not cid.strip():
            raise ValueError("candidate ids must be nonempty after stripping")
        if cid in known_ids:
            raise ValueError("candidate ids must be unique")
        known_ids.add(cid)

    if type(decisions) is not dict:
        raise ValueError("decisions must be an exact built-in dict")
    if len(decisions) > _MAX_ENTRIES:
        raise ValueError("decisions must contain at most 5000 entries")

    result = {}
    for key, value in decisions.items():
        if type(key) is not str:
            raise ValueError("each decision key must be an exact built-in str")
        if key not in known_ids:
            raise ValueError("decision key must be an exact candidate id")
        if type(value) is not dict:
            raise ValueError("each decision value must be an exact built-in dict")
        if not 1 <= len(value) <= 3:
            raise ValueError("each decision value must have 1..3 fields")

        for field, field_value in value.items():
            if type(field) is not str or field not in _FIELDS:
                raise ValueError("decision fields must be label, override, or note")

            if field == "label":
                if field_value is not None and (
                    type(field_value) is not str or field_value not in _LABELS
                ):
                    raise ValueError("label must be None or a known label")
            elif field == "override":
                if field_value is not None and (
                    type(field_value) is not str or field_value not in _OVERRIDES
                ):
                    raise ValueError("override must be None, keep, or omit")
            else:
                if type(field_value) is not str:
                    raise ValueError("note must be an exact built-in str")
                if len(field_value) > _MAX_NOTE_LEN:
                    raise ValueError("note must be at most 2000 characters")
                try:
                    field_value.encode("utf-8")
                except UnicodeEncodeError:
                    raise ValueError("note must be valid UTF-8") from None

        result[key] = dict(value)

    return result
