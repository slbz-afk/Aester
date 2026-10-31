"""Immutable render identities, explicit candidate feedback and safe exports."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import uuid

from .finishing_assets import ordinary_path, sha256_file, verify_source
from .finishing_backend import render_recipe, RenderCancelled
from .finishing_session import _locked, _expect, _now, _save, _text, read_session, renderer_masks
from .records import atomic_json


def render_candidate(directory, *, expected_revision, gimp_path, timeout=600, cancel=None):
    """Serialize rendering per session; failures never become usable candidates."""
    with _locked(directory):
        value = read_session(directory)
        _expect(value, expected_revision)
        verify_source(value["source"])
        if len(value["candidates"]) >= 128:
            raise ValueError("Candidate limit reached")
        revision = value["revisions"][-1]
        masks = renderer_masks(value, revision["recipe"])
        bundles = {}
        if any(op["type"] == "astrophotographic_layers" for op in revision["recipe"]["operations"]):
            from .finishing_layers import recipe_bundles
            bundles = recipe_bundles(value, revision["recipe"])
        root = Path(value["directory"])
        renders = root / "renders"
        renders.mkdir(exist_ok=True)
        ordinary_path(renders, directory=True)
        candidate_id = "candidate-" + uuid.uuid4().hex
        work = renders / candidate_id
        work.mkdir()
        candidate = {"id": candidate_id, "revision": value["revision"],
                     "recipe_sha256": revision["recipe_sha256"], "directory": str(work),
                     "status": "running", "created_at": _now()}
        value["candidates"].append(candidate)
        _save(root / "session.json", value)
        try:
            execution = render_recipe(value["source"], revision["recipe"], masks, work,
                                      gimp_path=gimp_path, timeout=timeout, cancel=cancel,
                                      **({"bundles": bundles} if bundles else {}))
            if cancel is not None and cancel():
                raise RenderCancelled("Candidate cancelled before publication")
            verify_source(value["source"])
            renderer_masks(value, revision["recipe"])
            if bundles:
                recipe_bundles(value, revision["recipe"])
            if execution.get("status") != "validated":
                raise ValueError("Renderer did not validate the candidate")
            if cancel is not None and cancel():
                raise RenderCancelled("Candidate cancelled during final dependency verification")
            candidate.update({"status": "validated", "finished_at": _now(),
                              "execution": execution})
            _save(root / "session.json", value)
        except Exception as exc:
            candidate.pop("execution", None)
            candidate.update({"status": "cancelled" if isinstance(exc, RenderCancelled) else "failed",
                              "finished_at": _now(), "error": str(exc)})
            _save(root / "session.json", value)
            raise
        return candidate


def recover_interrupted(directory):
    """Mark abandoned records only after obtaining the render owner's OS lock.

    Partial artifacts are retained for diagnosis, never resumed or published.
    A later render always receives a new directory and identity.
    """
    with _locked(directory):
        value = read_session(directory)
        changed = False
        for candidate in value["candidates"]:
            if candidate["status"] == "running":
                candidate.pop("execution", None)
                candidate.update({"status": "interrupted", "finished_at": _now(),
                                  "error": "Render owner ended before candidate publication"})
                changed = True
        if changed:
            _save(Path(value["directory"]) / "session.json", value)
        return value


def _candidate(value, candidate_id):
    if type(candidate_id) is not str or re.fullmatch(r"candidate-[0-9a-f]{32}", candidate_id) is None:
        raise ValueError("Invalid candidate ID")
    matches = [item for item in value["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Unknown candidate")
    candidate = matches[0]
    if candidate["status"] != "validated":
        raise ValueError("Candidate is not a validated render")
    index = candidate["revision"]
    if type(index) is not int or not 0 <= index <= value["revision"]:
        raise ValueError("Candidate revision binding changed")
    if candidate["recipe_sha256"] != value["revisions"][index]["recipe_sha256"]:
        raise ValueError("Candidate recipe binding changed")
    expected = Path(value["directory"]) / "renders" / candidate_id
    if Path(candidate["directory"]) != expected:
        raise ValueError("Candidate directory binding changed")
    return candidate


def verified_candidate(directory, candidate_id):
    value = read_session(directory)
    candidate = _candidate(value, candidate_id)
    products = candidate["execution"]["products"]
    for key, filename in (("tiff", "result.tif"), ("jpeg", "result.jpg"), ("xcf", "result.xcf")):
        record = products[key]
        expected = Path(candidate["directory"]) / "artifacts" / filename
        path = ordinary_path(record["path"])
        if path != expected or sha256_file(path) != record["sha256"]:
            raise ValueError("Candidate product changed after validation")
    return candidate


def candidate_feedback(directory, *, candidate_id, text, origin, decision):
    """An aesthetic decision must identify an actual, unchanged rendered image."""
    _text(text, "candidate feedback")
    if origin not in {"user", "agent"} or decision not in {"accept", "reject", "revise", "observe"}:
        raise ValueError("Invalid candidate feedback")
    if origin == "agent" and decision == "accept":
        raise ValueError("Agent feedback cannot stand in for user acceptance")
    with _locked(directory):
        value = read_session(directory)
        candidate = verified_candidate(directory, candidate_id)
        if len(value["feedback"]) >= 1000:
            raise ValueError("Session feedback limit reached")
        value["feedback"].append({"scope": "candidate", "candidate_id": candidate_id,
                                  "revision": candidate["revision"], "text": text,
                                  "origin": origin, "decision": decision, "created_at": _now()})
        _save(Path(value["directory"]) / "session.json", value)
        return value


def export_candidate(directory, candidate_id, output_root):
    """Publish a complete, uniquely named copy. Originals and older exports stay."""
    with _locked(directory):
        candidate = verified_candidate(directory, candidate_id)
        value = read_session(directory)
        root = ordinary_path(output_root, directory=True)
        final = root / ("aester-photo-" + uuid.uuid4().hex)
        staging = root / (".finishing-export-" + uuid.uuid4().hex)
        staging.mkdir()
        products = candidate["execution"]["products"]
        manifest = {"schema": "aester-finishing-export-1", "candidate_id": candidate_id,
                    "source": value["source"], "recipe_sha256": candidate["recipe_sha256"],
                    "recipe": value["revisions"][candidate["revision"]]["recipe"], "products": {}}
        if "layer_bundles" in candidate["execution"]:
            manifest.update(layer_bundles=candidate["execution"]["layer_bundles"],
                host_stage=candidate["execution"].get("host_stage"),
                parameter_rerun="Requires retained local source/bundle/masks; this export does not copy B/D assets",
                xcf_editability="Cumulative snapshots; saved appearance reopens without external B/D files")
        for key, name in (("tiff", "photo.tif"), ("jpeg", "photo.jpg"), ("xcf", "photo.xcf")):
            target = staging / name
            with ordinary_path(products[key]["path"]).open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            digest = sha256_file(target)
            if digest != products[key]["sha256"]:
                raise ValueError("Product changed while exporting; incomplete staging retained")
            manifest["products"][key] = {"file": name, "sha256": digest, "bytes": target.stat().st_size}
        atomic_json(staging / "manifest.json", manifest)
        # Rename only our unique child after verifying both absolute locations.
        if staging.resolve().parent != root or final.absolute().parent != root or final.exists():
            raise ValueError("Export destination binding changed")
        staging.rename(final)
        return {"directory": str(final), "manifest": manifest}
