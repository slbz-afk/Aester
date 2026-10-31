"""Coordinator tests for source binding, revision conflicts and durable feedback."""
from copy import deepcopy
import hashlib
import json

import numpy as np
from PIL import ImageCms
import pytest
import tifffile

from starstack import finishing_assets as assets, finishing_session as sessions


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "原图 with spaces.tif"
    pixels = np.arange(12 * 8 * 3, dtype=np.uint16).reshape(8, 12, 3) * 173
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None,
                     extratags=[(34675, "B", len(icc), icc, False)])
    return path


def _session(source, tmp_path):
    return sessions.create_session(source, tmp_path, intent="银河更清晰，保留绿色气辉")


def _exposure(record, ev=0.25):
    recipe = deepcopy(record["revisions"][-1]["recipe"])
    recipe["operations"].append({"id": "sky_lift", "type": "exposure", "enabled": True,
                                  "mask_id": None, "params": {"ev": ev}})
    return recipe


def test_inspection_is_read_only_and_high_precision(source):
    before = source.read_bytes()
    record = assets.inspect_source(source, encoding="srgb16")
    assert record["sha256"] == hashlib.sha256(before).hexdigest()
    assert (record["width"], record["height"]) == (12, 8)
    assert source.read_bytes() == before
    assert np.any(tifffile.imread(source) % 257 != 0)


@pytest.mark.parametrize("encoding", ["linear", "srgb", None, True])
def test_wrong_encoding_refused(source, encoding):
    with pytest.raises(ValueError):
        assets.inspect_source(source, encoding=encoding)


@pytest.mark.parametrize("kind", ["float", "gray", "rgba", "missing_icc", "linear_claim", "orientation"])
def test_incompatible_source_is_not_silently_converted(source, tmp_path, kind):
    with tifffile.TiffFile(source) as tf:
        icc = bytes(tf.pages[0].tags[34675].value)
    pixels = tifffile.imread(source)
    if kind == "float":
        pixels = pixels.astype(np.float32) / 65535
    elif kind == "gray":
        pixels = pixels[:, :, 0]
    elif kind == "rgba":
        pixels = np.dstack([pixels, np.full((8, 12), 65535, dtype=np.uint16)])
    tags = [] if kind == "missing_icc" else [(34675, "B", len(icc), icc, False)]
    if kind == "orientation":
        tags.append((274, "H", 1, 6, False))
    path = tmp_path / (kind + ".tif")
    tifffile.imwrite(path, pixels, photometric="minisblack" if kind == "gray" else "rgb",
                     metadata=None, extratags=tags,
                     description=json.dumps({"scene_linear": True}) if kind == "linear_claim" else None)
    with pytest.raises(ValueError):
        assets.inspect_source(path, encoding="srgb16")


def test_multipage_rejected(source, tmp_path):
    path = tmp_path / "multiple.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((8, 12, 3), np.uint16), photometric="rgb")
        writer.write(np.zeros((8, 12, 3), np.uint16), photometric="rgb")
    with pytest.raises(ValueError):
        assets.inspect_source(path, encoding="srgb16")


def test_revision_retains_original_and_rejects_stale_writer(source, tmp_path):
    record = _session(source, tmp_path)
    first = deepcopy(record["revisions"][0])
    updated = sessions.revise_recipe(record["directory"], _exposure(record), expected_revision=0)
    assert updated["revision"] == 1
    assert updated["revisions"][0] == first
    assert updated["intent"] == record["intent"]
    with pytest.raises(sessions.RevisionConflict):
        sessions.revise_recipe(record["directory"], _exposure(record), expected_revision=0)
    assert sessions.read_session(record["directory"]) == updated


def test_wrong_source_recipe_refused(source, tmp_path):
    record = _session(source, tmp_path)
    recipe = _exposure(record)
    recipe["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        sessions.revise_recipe(record["directory"], recipe, expected_revision=0)
    assert sessions.read_session(record["directory"])["revision"] == 0


def test_changed_source_refused_without_losing_history(source, tmp_path):
    record = _session(source, tmp_path)
    with source.open("ab") as handle:
        handle.write(b"changed after registration")
    with pytest.raises(ValueError):
        sessions.revise_recipe(record["directory"], _exposure(record), expected_revision=0)
    assert sessions.read_session(record["directory"])["revision"] == 0


def test_missing_mask_never_falls_back_to_global(source, tmp_path):
    record = _session(source, tmp_path)
    recipe = _exposure(record)
    recipe["operations"][0]["mask_id"] = "missing_sky"
    with pytest.raises(ValueError):
        sessions.revise_recipe(record["directory"], recipe, expected_revision=0)


def test_failed_write_leaves_previous_revision(source, tmp_path, monkeypatch):
    record = _session(source, tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")
    monkeypatch.setattr(sessions, "atomic_json", fail)
    with pytest.raises(OSError):
        sessions.revise_recipe(record["directory"], _exposure(record), expected_revision=0)
    assert sessions.read_session(record["directory"]) == record


def test_size_limit_cannot_create_unreadable_session(source, tmp_path, monkeypatch):
    from pathlib import Path
    record = _session(source, tmp_path)
    path = Path(record["directory"]) / "session.json"
    before = path.read_bytes()
    monkeypatch.setattr(sessions, "MAX_RECORD_BYTES", len(before) + 10)
    with pytest.raises(ValueError):
        sessions.revise_recipe(record["directory"], _exposure(record), expected_revision=0)
    assert path.read_bytes() == before
    assert sessions.read_session(record["directory"]) == record


def test_exact_size_boundary_uses_actual_platform_text_bytes(source, tmp_path, monkeypatch):
    from pathlib import Path
    from starstack.finishing_recipe import normalize_recipe, recipe_digest
    record = _session(source, tmp_path)
    monkeypatch.setattr(sessions, "_now", lambda: "fixed-time")
    recipe = normalize_recipe(_exposure(record))
    expected = deepcopy(record)
    expected["revision"] = 1
    expected["revisions"].append({"revision": 1, "created_at": "fixed-time", "recipe": recipe,
                                   "recipe_sha256": recipe_digest(recipe)})
    # Measure through an independent text writer, including Windows CRLF.
    probe = tmp_path / "expected-serialization.json"
    with probe.open("w", encoding="utf-8") as handle:
        json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
    boundary = probe.stat().st_size
    path = Path(record["directory"]) / "session.json"
    before = path.read_bytes()
    monkeypatch.setattr(sessions, "MAX_RECORD_BYTES", boundary - 1)
    with pytest.raises(ValueError):
        sessions.revise_recipe(record["directory"], recipe, expected_revision=0)
    assert path.read_bytes() == before
    monkeypatch.setattr(sessions, "MAX_RECORD_BYTES", boundary)
    updated = sessions.revise_recipe(record["directory"], recipe, expected_revision=0)
    assert path.stat().st_size == boundary
    assert sessions.read_session(record["directory"]) == updated == expected
    with pytest.raises(ValueError):
        sessions.record_feedback(record["directory"], revision=0, text="more notes", origin="user", decision="revise")
    assert sessions.read_session(record["directory"]) == updated


def test_busy_session_rejects_other_writer(source, tmp_path):
    record = _session(source, tmp_path)
    with sessions._locked(record["directory"]):
        with pytest.raises(sessions.RevisionConflict):
            sessions.revise_recipe(record["directory"], _exposure(record), expected_revision=0)


def test_feedback_remembers_failure_and_cannot_claim_agent_acceptance(source, tmp_path):
    record = _session(source, tmp_path)
    failed = sessions.record_feedback(record["directory"], revision=0, text="成图质量非常糟糕",
                                       origin="user", decision="reject")
    assert failed["feedback"][0]["decision"] == "reject"
    with pytest.raises(ValueError):
        sessions.record_feedback(record["directory"], revision=0, text="looks good to me",
                                  origin="agent", decision="accept")
    reread = sessions.read_session(record["directory"])
    assert reread["feedback"] == failed["feedback"]
    assert reread["revisions"] == record["revisions"]


def test_recipe_tampering_detected(source, tmp_path):
    record = _session(source, tmp_path)
    from pathlib import Path
    path = Path(record["directory"]) / "session.json"
    record["revisions"][0]["recipe"]["intent"] = "changed behind digest"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        sessions.read_session(path.parent)
