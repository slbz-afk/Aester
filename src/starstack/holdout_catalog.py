"""Identity-separated registration data for star-stacking catalogs.

This module validates the catalog schema, slices it into train/holdout views,
and produces canonical JSON SHA-256 fingerprints.  The ``identity_basis``
declaration is provenance metadata only: it does not prove real-star
identities or detection independence, and old Siril match caches cannot be
relabelled as independent by this module.
"""

import hashlib
import json
import math

CATALOG_VERSION = "1.0.0"

__all__ = [
    "CATALOG_VERSION",
    "validate_catalog",
    "training_view",
    "holdout_view",
    "canonical_sha256",
]

_ROOT_KEYS = frozenset((
    "version",
    "width",
    "height",
    "reference",
    "identity_basis",
    "stars",
    "frames",
))
_STAR_KEYS = frozenset(("id", "role"))
_FRAME_KEYS = frozenset(("id", "observations"))
_OBSERVATION_KEYS = frozenset(("star_id", "x", "y"))
_ROLES = frozenset(("train", "holdout"))
_IDENTITY_BASES = frozenset(("synthetic_truth", "external_verified"))


def _raise(message):
    raise ValueError(message)


def _require_exact_mapping(value, allowed, label):
    if type(value) is not dict:
        _raise(f"{label} must be a built-in dict")
    seen = set()
    for key in value:
        if type(key) is not str:
            _raise(f"{label} keys must be strings")
        if key not in allowed:
            _raise(f"{label} has an unknown key: {key!r}")
        seen.add(key)
    if seen != allowed:
        _raise(f"{label} is missing a required key")


def _require_identifier(value, label):
    if type(value) is not str:
        _raise(f"{label} must be a string")
    if not value or not value.strip():
        _raise(f"{label} must be nonempty and not whitespace-only")
    return value


def _normalize_coordinate(value, limit, label):
    value_type = type(value)
    if value_type is bool or value_type not in (int, float):
        _raise(f"{label} must be a built-in int or float")
    if value_type is float and not math.isfinite(value):
        _raise(f"{label} must be finite")
    if not (0 <= value <= limit - 1):
        _raise(f"{label} is outside the allowed range")
    try:
        coerced = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"{label} cannot be coerced safely to float") from exc
    if not math.isfinite(coerced):
        _raise(f"{label} must be finite")
    return coerced


def validate_catalog(document):
    """Validate and normalize one catalog document.

    Returns an independent deep copy with stars sorted by id, frames sorted by
    id, and observations sorted by star_id.  Coordinates are coerced to finite
    floats.  Raises ValueError for malformed input.

    The ``identity_basis`` declaration is provenance metadata only: it does
    not prove real-star identities or detection independence, and old Siril
    match caches cannot be relabelled as independent.
    """
    _require_exact_mapping(document, _ROOT_KEYS, "catalog")

    version = document["version"]
    if type(version) is not str or version != CATALOG_VERSION:
        _raise(f"version must be {CATALOG_VERSION!r}")

    width = document["width"]
    if type(width) is not int or width < 2:
        _raise("width must be a built-in int >= 2")

    height = document["height"]
    if type(height) is not int or height < 2:
        _raise("height must be a built-in int >= 2")

    reference = _require_identifier(document["reference"], "reference")

    identity_basis = document["identity_basis"]
    if type(identity_basis) is not str or identity_basis not in _IDENTITY_BASES:
        _raise("identity_basis must be 'synthetic_truth' or 'external_verified'")

    raw_stars = document["stars"]
    if type(raw_stars) is not list:
        _raise("stars must be a list")
    if not raw_stars:
        _raise("stars must be nonempty")

    stars_out = []
    star_ids = set()
    role_counts = {"train": 0, "holdout": 0}

    for item in raw_stars:
        _require_exact_mapping(item, _STAR_KEYS, "star")
        star_id = _require_identifier(item["id"], "star id")
        if star_id in star_ids:
            _raise("star ids must be unique")
        role = item["role"]
        if type(role) is not str or role not in _ROLES:
            _raise("star role must be 'train' or 'holdout'")
        star_ids.add(star_id)
        role_counts[role] += 1
        stars_out.append({"id": star_id, "role": role})

    if role_counts["train"] == 0 or role_counts["holdout"] == 0:
        _raise("stars must include at least one train and at least one holdout star")

    stars_out.sort(key=lambda entry: entry["id"])

    raw_frames = document["frames"]
    if type(raw_frames) is not list:
        _raise("frames must be a list")
    if len(raw_frames) < 2:
        _raise("frames must contain at least two entries")

    frames_out = []
    frame_ids = set()
    reference_count = 0
    reference_observations = None

    for item in raw_frames:
        _require_exact_mapping(item, _FRAME_KEYS, "frame")
        frame_id = _require_identifier(item["id"], "frame id")
        if frame_id in frame_ids:
            _raise("frame ids must be unique")
        frame_ids.add(frame_id)

        raw_observations = item["observations"]
        if type(raw_observations) is not list:
            _raise("frame observations must be a list")

        observations_out = []
        seen_star_ids = set()

        for observation in raw_observations:
            _require_exact_mapping(observation, _OBSERVATION_KEYS, "observation")
            star_id = observation["star_id"]
            if type(star_id) is not str:
                _raise("observation star_id must be a string")
            if star_id not in star_ids:
                _raise("observation star_id must refer to a declared star")
            if star_id in seen_star_ids:
                _raise("a frame cannot repeat a star identity")
            seen_star_ids.add(star_id)

            x = _normalize_coordinate(observation["x"], width, "x")
            y = _normalize_coordinate(observation["y"], height, "y")
            observations_out.append({"star_id": star_id, "x": x, "y": y})

        observations_out.sort(key=lambda entry: entry["star_id"])

        if frame_id == reference:
            reference_count += 1
            reference_observations = observations_out

        frames_out.append({"id": frame_id, "observations": observations_out})

    if reference_count != 1:
        _raise("frames must contain the reference frame exactly once")
    if reference_observations is None:
        _raise("reference frame is missing")
    if len(reference_observations) != len(stars_out):
        _raise("every declared star must have exactly one reference-frame observation")

    frames_out.sort(key=lambda entry: entry["id"])

    normalized = {
        "version": version,
        "width": width,
        "height": height,
        "reference": reference,
        "identity_basis": identity_basis,
        "stars": stars_out,
        "frames": frames_out,
    }
    # A valid catalog must also fit the actual strict-JSON encoder contract.
    # In particular, a dimension can be a builtin int yet exceed its digit limit.
    canonical_sha256(normalized)
    return normalized


def _role_view(document, role):
    catalog = validate_catalog(document)

    selected_stars = [star["id"] for star in catalog["stars"] if star["role"] == role]
    selected_ids = set(selected_stars)

    frames_out = []
    for frame in catalog["frames"]:
        observations_out = []
        for observation in frame["observations"]:
            if observation["star_id"] in selected_ids:
                observations_out.append({
                    "star_id": observation["star_id"],
                    "x": observation["x"],
                    "y": observation["y"],
                })
        frames_out.append({"id": frame["id"], "observations": observations_out})

    return {
        "version": catalog["version"],
        "width": catalog["width"],
        "height": catalog["height"],
        "reference": catalog["reference"],
        "identity_basis": catalog["identity_basis"],
        "role": role,
        "stars": selected_stars,
        "frames": frames_out,
    }


def training_view(document):
    """Return an independent train-only view of a catalog.

    The input is validated first.  The view contains only train star identities
    and only train observations; frames are preserved even when they have no
    train observations.  The ``identity_basis`` declaration is provenance
    metadata only: it does not prove real-star identities or detection
    independence, and old Siril match caches cannot be relabelled as
    independent.
    """
    return _role_view(document, "train")


def holdout_view(document):
    """Return an independent holdout-only view of a catalog.

    The input is validated first.  The view contains only holdout star
    identities and only holdout observations; frames are preserved even when
    they have no holdout observations.  The ``identity_basis`` declaration is
    provenance metadata only: it does not prove real-star identities or
    detection independence, and old Siril match caches cannot be relabelled as
    independent.
    """
    return _role_view(document, "holdout")


def _validate_json_tree(value):
    stack = [(value, False)]
    active = set()

    while stack:
        current, leaving = stack.pop()
        if leaving:
            active.discard(id(current))
            continue

        current_type = type(current)

        if current is None:
            continue
        if current_type is bool:
            continue
        if current_type is int:
            continue
        if current_type is float:
            if not math.isfinite(current):
                _raise("canonical JSON floats must be finite")
            continue
        if current_type is str:
            continue
        if current_type is list:
            marker = id(current)
            if marker in active:
                _raise("canonical JSON data must not contain cycles")
            active.add(marker)
            stack.append((current, True))
            for item in reversed(current):
                stack.append((item, False))
            continue
        if current_type is dict:
            marker = id(current)
            if marker in active:
                _raise("canonical JSON data must not contain cycles")
            active.add(marker)
            stack.append((current, True))
            for key, item in reversed(list(current.items())):
                if type(key) is not str:
                    _raise("canonical JSON dict keys must be strings")
                stack.append((item, False))
            continue

        _raise("unsupported type for canonical JSON")


def canonical_sha256(document):
    """Return the canonical SHA-256 hex digest of JSON-compatible data.

    Only ordinary built-in JSON values are accepted: dictionaries with string
    keys, lists, strings, integers, floats, booleans, and None.  Cycles,
    non-finite floats, tuples, subclasses, and unsupported objects raise
    ValueError.  Dictionaries are encoded with sorted keys; list order is
    preserved.
    """
    _validate_json_tree(document)

    try:
        payload = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception as exc:
        raise ValueError("document cannot be encoded as strict canonical JSON") from exc
