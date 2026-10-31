"""Independent, revisioned finishing sessions; no renderer or model is invoked."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .finishing_assets import (inspect_source, ordinary_path, recipe_source, verify_source,
                               inspect_mask, inspect_reference, sha256_file)
from .finishing_recipe import normalize_recipe, recipe_digest, recipe_mask_ids
from .records import atomic_json

SCHEMA = "aester-finishing-session-1"
MAX_RECORD_BYTES = 32 * 1024 * 1024


class RevisionConflict(ValueError):
    """Another writer changed the session after the caller read it."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, field, limit=4000):
    if type(value) is not str or len(value) > limit:
        raise ValueError(f"{field} must be text of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be valid UTF-8 text") from exc
    return value


def _save(path, value):
    # A successful write must remain readable under the same resource limit.
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    # records.atomic_json uses a text stream with newline=None. Its Windows
    # CRLF translation is part of the file's byte budget, not just JSON data.
    payload = text.replace("\n", os.linesep).encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("Finishing session size limit reached; previous revision is unchanged")
    atomic_json(path, value)


@contextmanager
def _locked(directory):
    """OS lock releases on owner exit, including abnormal process termination."""
    path = ordinary_path(directory, directory=True) / ".session.lock"
    if path.exists():
        ordinary_path(path)
    with path.open("a+b") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RevisionConflict("Finishing session is being updated") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_session(directory) -> dict:
    path = ordinary_path(directory, directory=True) / "session.json"
    ordinary_path(path)
    if path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("Finishing session record is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Unsupported finishing session")
    if value.get("directory") != str(path.parent):
        raise ValueError("Session directory binding changed")
    revisions = value.get("revisions")
    if not isinstance(revisions, list) or not revisions or len(revisions) > 1000:
        raise ValueError("Invalid finishing revision history")
    if type(value.get("revision")) is not int or value["revision"] != len(revisions) - 1:
        raise ValueError("Invalid finishing revision counter")
    for index, revision in enumerate(revisions):
        if (not isinstance(revision, dict) or revision.get("revision") != index
                or recipe_digest(revision.get("recipe")) != revision.get("recipe_sha256")
                or revision["recipe"]["source"] != recipe_source(value["source"])):
            raise ValueError("Finishing revision binding changed")
    return value


def create_session(source, output_root, *, intent: str, encoding: str = "srgb16") -> dict:
    _text(intent, "intent")
    # Require an explicit existing output root, keeping host path policy external.
    root = ordinary_path(output_root, directory=True)
    asset = inspect_source(source, encoding=encoding)
    recipe = normalize_recipe({"schema": "aester-finishing-recipe-1", "source": recipe_source(asset),
                               "intent": intent, "operations": []})
    directory = root / ("finishing-" + uuid.uuid4().hex)
    directory.mkdir()
    value = {"schema": SCHEMA, "directory": str(directory), "created_at": _now(),
             "source": asset, "intent": intent, "revision": 0, "masks": {}, "references": [],
             "revisions": [{"revision": 0, "created_at": _now(), "recipe": recipe,
                            "recipe_sha256": recipe_digest(recipe)}], "feedback": [], "candidates": []}
    _save(directory / "session.json", value)
    return value


def _expect(value, expected_revision):
    if type(expected_revision) is not int or expected_revision != value["revision"]:
        raise RevisionConflict("Expected revision does not match the current session")


def update_intent(directory, intent, *, expected_intent_revision):
    """Append a local user's intent without rewriting rendered recipes or history."""
    _text(intent, "intent")
    with _locked(directory):
        value = read_session(directory)
        current = value.get("intent_revision", 0)
        if type(expected_intent_revision) is not int or expected_intent_revision != current:
            raise RevisionConflict("Expected intent revision does not match the current session")
        history = value.get("intent_history", [])
        if type(history) is not list or len(history) >= 1000:
            raise ValueError("Session intent history limit reached")
        if "initial_intent" not in value:
            value["initial_intent"] = value["intent"]
        history.append({"revision": current + 1, "previous_intent": value["intent"],
                        "intent": intent, "origin": "user", "created_at": _now()})
        value.update(intent=intent, intent_revision=current + 1, intent_history=history)
        _save(Path(value["directory"]) / "session.json", value)
        return value


def revise_recipe(directory, recipe: dict, *, expected_revision: int) -> dict:
    normalized = normalize_recipe(recipe)
    with _locked(directory):
        value = read_session(directory)
        _expect(value, expected_revision)
        verify_source(value["source"])
        if normalized["source"] != recipe_source(value["source"]):
            raise ValueError("Recipe belongs to a different source")
        renderer_masks(value, normalized)
        if any(op["type"] == "astrophotographic_layers" for op in normalized["operations"]):
            from .finishing_layers import recipe_bundles
            recipe_bundles(value, normalized)
        if len(value["revisions"]) >= 1000:
            raise ValueError("Session revision limit reached")
        revision = value["revision"] + 1
        value["revisions"].append({"revision": revision, "created_at": _now(),
                                   "recipe": normalized, "recipe_sha256": recipe_digest(normalized)})
        value["revision"] = revision
        _save(Path(value["directory"]) / "session.json", value)
        return value


def _asset_id(value):
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", value) is None:
        raise ValueError("Invalid asset ID")


def _copy_asset(directory, record):
    asset_dir = Path(directory) / "assets"
    asset_dir.mkdir(exist_ok=True)
    ordinary_path(asset_dir, directory=True)
    destination = asset_dir / (uuid.uuid4().hex + Path(record["path"]).suffix.lower())
    try:
        with ordinary_path(record["path"]).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if sha256_file(destination) != record["sha256"]:
            raise ValueError("Asset changed while being copied")
    except Exception:
        # Only this function's unique new file; no user source or recursive cleanup.
        destination.unlink(missing_ok=True)
        raise
    return {**record, "path": str(destination)}


def register_mask(directory, path, *, mask_id: str, source_sha256: str,
                  origin: str, note: str = "", geometry: dict | None = None,
                  dependencies: list | None = None) -> dict:
    """Copy and bind a coverage mask. Existing IDs cannot be redefined."""
    _asset_id(mask_id)
    _text(note, "mask note")
    if origin not in {"user", "agent", "imported"}:
        raise ValueError("Invalid mask origin")
    if geometry is not None:
        if type(geometry) is not dict:
            raise ValueError("Mask creation geometry must be a JSON object")
        geometry = json.loads(json.dumps(geometry, allow_nan=False))
    with _locked(directory):
        value = read_session(directory)
        verify_source(value["source"])
        if mask_id in value["masks"] or len(value["masks"]) >= 64:
            raise ValueError("Mask ID already exists or mask limit reached")
        asset = inspect_mask(path, value["source"], source_sha256=source_sha256)
        copied = _copy_asset(value["directory"], asset)
        value["masks"][mask_id] = {**copied, "origin": origin, "note": note, "created_at": _now()}
        if geometry is not None:
            value["masks"][mask_id]["creation_geometry"] = geometry
        try:
            # Tonal masks derive from an actual candidate and possibly a region.
            # Recheck those identities at publication, after the copy; a stale
            # dependency must not be silently published as a current selection.
            for dependency in dependencies or []:
                if dependency["kind"] == "image":
                    verify_source(dependency["asset"])
                elif dependency["kind"] == "region":
                    record = dependency["asset"]
                    current = inspect_mask(record["path"], value["source"], source_sha256=source_sha256)
                    if any(record.get(key) != item for key, item in current.items()):
                        raise ValueError("Tonal region dependency changed before publication")
                else:
                    raise ValueError("Unknown mask dependency")
            _save(Path(value["directory"]) / "session.json", value)
        except Exception:
            Path(copied["path"]).unlink(missing_ok=True)
            raise
        return value


def register_reference(directory, path, *, kind: str, note: str = "") -> dict:
    """Keep a stable, explicitly labeled reference; never feed it to rendering."""
    if kind not in {"original", "generated", "external"}:
        raise ValueError("Reference kind must identify its provenance")
    _text(note, "reference note")
    with _locked(directory):
        value = read_session(directory)
        if len(value["references"]) >= 16:
            raise ValueError("Reference limit reached")
        copied = _copy_asset(value["directory"], inspect_reference(path))
        value["references"].append({**copied, "id": "reference-" + uuid.uuid4().hex,
                                    "kind": kind, "note": note, "created_at": _now()})
        try:
            _save(Path(value["directory"]) / "session.json", value)
        except Exception:
            Path(copied["path"]).unlink(missing_ok=True)
            raise
        return value


def renderer_masks(value: dict, recipe: dict) -> dict:
    """Return only validated coverage files used by this recipe, no references."""
    result = {}
    for key in sorted(recipe_mask_ids(recipe)):
        record = value["masks"].get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Unregistered mask: {key}")
        path = ordinary_path(record["path"])
        if path.parent != Path(value["directory"]) / "assets":
            raise ValueError("Registered mask must remain in this session's assets")
        actual = inspect_mask(path, value["source"], source_sha256=record["source_sha256"])
        if any(record.get(field) != item for field, item in actual.items()):
            raise ValueError("Registered mask changed; register it under a new ID")
        result[key] = actual
    return result


def record_feedback(directory, *, revision: int, text: str, origin: str, decision: str) -> dict:
    _text(text, "feedback")
    if origin not in {"user", "agent"} or decision not in {"revise", "reject", "accept", "observe"}:
        raise ValueError("Invalid feedback origin or decision")
    if origin == "agent" and decision == "accept":
        raise ValueError("Agent feedback cannot stand in for user acceptance")
    with _locked(directory):
        value = read_session(directory)
        if type(revision) is not int or not 0 <= revision <= value["revision"]:
            raise ValueError("Unknown feedback revision")
        if len(value["feedback"]) >= 1000:
            raise ValueError("Session feedback limit reached")
        value["feedback"].append({"revision": revision, "text": text, "origin": origin,
                                   "decision": decision, "created_at": _now()})
        _save(Path(value["directory"]) / "session.json", value)
        return value
