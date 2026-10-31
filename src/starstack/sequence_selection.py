"""Apply quality exclusions without reserializing Siril registration matrices.

This helper operates on the original registered RGB sequence (Siril 1.4,
``R1`` layer). The caller must serialize access and wait for Siril to exit
before editing it. It never selects an image that Siril already deselected.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from pathlib import Path

from .siril_sequence import SequenceError, parse_seq_text


_HEADER_SELECTION = re.compile(
    rb"^[ \t]*S[ \t]+'[^']*'[ \t]+\S+[ \t]+\S+[ \t]+(?P<value>\S+)"
)
_IMAGE_SELECTION = re.compile(rb"^[ \t]*I[ \t]+(?P<index>\S+)[ \t]+(?P<value>\S+)")


def _replace_value(line: bytes, match: re.Match, value: int) -> bytes:
    start, end = match.span("value")
    return line[:start] + str(value).encode("ascii") + line[end:]


def apply_exclusions(path: Path, excluded_stems: list[str], *, minimum: int = 3) -> dict:
    """Deselect named frames atomically, preserving every registration byte.

    Unknown names, malformed image selection flags, duplicate identities, or
    fewer than ``minimum`` usable selected frames leave the file unchanged.
    A selected frame must carry a finite R1 matrix with ``abs(H[2, 2]) >=
    1e-12``, matching the current product registration adapter's eligibility
    check. This is not a new image quality threshold or a geometric fit test.

    A stale S-line selected count is repaired from the I lines and reported.
    Diagnostic lists follow sequence order; repeated exclusions are idempotent.
    """
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum must be a positive integer")
    if not isinstance(excluded_stems, list) or any(type(stem) is not str or not stem for stem in excluded_stems):
        raise ValueError("excluded_stems must be a list of non-empty frame stems")
    path = Path(path)
    original = path.read_bytes()
    try:
        parsed = parse_seq_text(original.decode("utf-8", errors="strict"))
    except (OverflowError, UnicodeError) as exc:
        raise SequenceError("invalid sequence encoding or numeric fields") from exc
    images = parsed["images"]
    all_stems = [image["stem"] for image in images]
    if len(set(all_stems)) != len(all_stems):
        raise SequenceError("sequence contains duplicate image identities")
    unknown = sorted(set(excluded_stems) - set(all_stems))
    if unknown:
        raise SequenceError("excluded frames are absent from sequence: " + ", ".join(unknown))

    requested = set(excluded_stems)
    selected_before = [image["stem"] for image in images if image["selected"]]
    selected = [stem for stem in selected_before if stem not in requested]
    registration = parsed["regdata"].get(1, [])
    invalid_selected = []
    for image in images:
        if image["stem"] not in selected:
            continue
        position = image["position"]
        h = registration[position]["H"] if position < len(registration) else None
        if h is None or not all(math.isfinite(v) for row in h for v in row) or abs(h[2][2]) < 1e-12:
            invalid_selected.append(image["stem"])
    if invalid_selected:
        raise SequenceError("selected frames have no usable R1 registration: " + ", ".join(invalid_selected))
    if len(selected) < minimum:
        raise SequenceError(f"quality exclusions leave {len(selected)} usable selected frames; at least {minimum} required")

    output = []
    image_position = 0
    header_count = 0
    for line in original.splitlines(keepends=True):
        stripped = line.lstrip()
        tag = stripped.split(None, 1)[0] if stripped.strip() else b""
        if tag == b"S":
            match = _HEADER_SELECTION.match(line)
            if match is None:
                raise SequenceError("unsupported S-line layout")
            header_count += 1
            # Preserve a count's original spelling too when already correct.
            if parsed["header"]["nb_selected"] != len(selected):
                line = _replace_value(line, match, len(selected))
        elif tag == b"I":
            match = _IMAGE_SELECTION.match(line)
            if match is None or re.fullmatch(rb"[0-9]+", match.group("index")) is None:
                raise SequenceError("image indices must be non-negative integer tokens")
            if match.group("value") not in (b"0", b"1"):
                raise SequenceError("image selection flags must be 0 or 1")
            image = images[image_position]
            if image["selected"] and image["stem"] in requested:
                line = _replace_value(line, match, 0)
            image_position += 1
        output.append(line)
    if header_count != 1:
        raise SequenceError("sequence must contain exactly one S line")
    updated = b"".join(output)
    validated = parse_seq_text(updated.decode("utf-8", errors="strict"))
    if validated["header"]["nb_selected"] != len(selected) or [im["stem"] for im in validated["images"] if im["selected"]] != selected:
        raise SequenceError("selection update failed consistency validation")

    if updated != original:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256_before": hashlib.sha256(original).hexdigest(),
        "sha256_after": hashlib.sha256(updated).hexdigest(),
        "changed": updated != original,
        "header_selected_before": parsed["header"]["nb_selected"],
        "header_count_repaired": parsed["header"]["nb_selected"] != len(selected_before),
        "excluded": [stem for stem in all_stems if stem in requested],
        "newly_excluded": [stem for stem in selected_before if stem in requested],
        "previously_deselected": [image["stem"] for image in images if not image["selected"]],
        "selected_before": selected_before,
        "selected": selected,
        "selected_count": len(selected),
        "minimum": minimum,
    }
