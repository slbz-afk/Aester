import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from starstack import trail_review_session as review
from starstack.records import atomic_json
from starstack.transient_revision import prepare


@pytest.fixture
def job_directory(tmp_path):
    directory = tmp_path / "job"
    (directory / "work").mkdir(parents=True)
    inputs = []
    for i in range(1, 4):
        source = tmp_path / f"frame-{i}.nef"
        source.write_bytes(f"source {i}".encode())
        inputs.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size": source.stat().st_size})
    rgb = np.full((3, 80, 100), .03, np.float32)
    header = fits.Header({"ROWORDER": "BOTTOM-UP"})
    fits.writeto(directory / "work" / "sky_master.fit", rgb, header)
    master_path = directory / "work" / "sky_master.fit"
    master = {"path": str(master_path.relative_to(directory)), "sha256": hashlib.sha256(master_path.read_bytes()).hexdigest()}
    checkpoints = []
    for i in (1, 2, 3):
        frame = rgb.copy()
        frame[1, 80-1-24:80-1-19, 30:61] += i*.02
        frame_path = directory / "work" / f"r_light_{i:05d}.fit"
        fits.writeto(frame_path, frame, header)
        checkpoints.append({"path": str(frame_path.relative_to(directory)), "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest()})
    candidates = []
    for i in (1, 2):
        candidates.append({"id": f"light_{i:05d}-1", "stem": f"light_{i:05d}", "source": "auto",
                           "provenance": {"frame_index": i, "input": inputs[i-1]},
                           "geometry": {"grid": [[30., 21.], [60., 21.]], "native_reference": [[30., 21.], [60., 21.]],
                                        "native_source": [[30., 21.], [60., 21.]]},
                           "measurement": {"status": "measured", "x0": 30., "x1": 60., "y0": 21., "y1": 21.},
                           "suggestion": {"class": "unknown"}, "effective_label": "unknown", "label": None,
                           "override": None, "decision": "omit", "note": ""})
    record = {"candidates": candidates}
    export = directory / "products" / "revision1"
    export.mkdir(parents=True)
    revision = prepare(export, record)
    atomic_json(directory / "job.json", {"job_dir": str(directory), "state": "succeeded", "signature": "job-signature",
                                        "inputs": inputs, "transient_revision": revision, "artifacts": [],
                                        "checkpoint_files": {"registered": checkpoints, "stack": [master]}})
    return directory


def save(session, decisions, **kwargs):
    return review.save_review(session["session_path"], decisions,
                              expected_draft_revision=session["draft_revision"], **kwargs)


def legacy(session):
    """A draft saved before the 2026-09-27 user decision (meteors kept by default).

    Such drafts carry no application_policy_version and keep the original
    explicit-keep-only-1 contract; new sessions are version-gated instead.
    """
    path = Path(session["session_path"])
    value = json.loads(path.read_text())
    value.pop("application_policy_version", None)
    value.pop("options", None)
    atomic_json(path, value)
    return review.read_review(str(path))


def change_job(directory, change):
    path = directory / "job.json"
    value = json.loads(path.read_text())
    change(value)
    atomic_json(path, value)


def new_revision(directory):
    job = json.loads((directory / "job.json").read_text())
    record = json.loads(Path(job["transient_revision"]["record"]).read_text())
    target = directory / "products" / "revision2"
    target.mkdir()
    revision = prepare(target, record)
    change_job(directory, lambda value: value.update(transient_revision=revision))
    return revision


def test_open_reopen_draft_and_pixel_coordinates(job_directory):
    first = review.open_review(str(job_directory))
    assert first["geometry"] == {"width": 100, "height": 80, "row_order": "TOP-DOWN", "coordinates": "zero-based pixel centres"}
    assert first["candidates"][0]["endpoints_native_reference"] == [[30., 21.], [60., 21.]]
    saved = save(first, {"light_00001-1": {"label": "unknown", "override": "keep"}}, reviewer="agent")
    reopened = review.open_review(str(job_directory))
    assert reopened["decisions"] == saved["decisions"]
    assert reopened["reviewer"] == "agent"
    assert reopened["candidates"][0]["effective_label"] == "unknown"
    assert reopened["candidates"][0]["decision"] == "omit"  # draft does not mutate candidates
    assert first["base_revision"] == reopened["base_revision"]


def test_concurrent_draft_writer_rejected(job_directory):
    first = review.open_review(str(job_directory))
    second = review.open_review(str(job_directory))
    save(first, {"light_00001-1": {"override": "keep"}})
    with pytest.raises(ValueError, match="另一窗口"):
        save(second, {"light_00002-1": {"override": "omit"}})


def test_apply_cannot_consume_another_windows_changed_draft(job_directory, monkeypatch):
    from starstack import core
    first = save(review.open_review(str(job_directory)), {"light_00001-1": {"override": "keep"}})
    save(first, {"light_00001-1": {"override": "omit"}})
    monkeypatch.setattr(core, "recompose_transients", lambda *a, **k: pytest.fail("applied changed intent"))
    with pytest.raises(ValueError, match="应用前草稿"):
        review.apply_review(first["session_path"], expected_draft_revision=first["draft_revision"])


@pytest.mark.parametrize("change", [lambda job: job.update(signature="other"),
                                    lambda job: job["inputs"][0].update(sha256="changed"),
                                    lambda job: job["inputs"][0].update(path="changed")])
def test_source_binding_rejected(job_directory, change):
    session = review.open_review(str(job_directory))
    change_job(job_directory, change)
    with pytest.raises(ValueError):
        save(session, {})


def test_revision_change_does_not_reuse_candidate_ids(job_directory):
    session = save(review.open_review(str(job_directory)), {"light_00001-1": {"override": "keep"}})
    new_revision(job_directory)
    with pytest.raises(ValueError, match="另一操作"):
        review.apply_review(session["session_path"])
    reopened = review.open_review(str(job_directory))
    assert reopened["session_path"] != session["session_path"]
    assert reopened["decisions"] == {}


def test_candidate_record_tamper_rejected(job_directory):
    session = review.open_review(str(job_directory))
    Path(session["base_revision"]["record"]).write_text("{}")
    with pytest.raises(ValueError, match="校验失败"):
        review.read_review(session["session_path"])


def test_one_batch_and_idempotent_reapply(job_directory, monkeypatch):
    # Version-gated (2026-09-27): the all-omit policy below is the legacy
    # explicit-keep-only-1 contract; new sessions use meteor-default-keep-1.
    from starstack import core
    session = save(legacy(review.open_review(str(job_directory))), {"light_00001-1": {"label": "unknown", "override": "keep"},
                                                           "light_00002-1": {"label": None, "override": "omit", "note": "manual decision"}})
    calls = []
    def recompose(directory, **kwargs):
        calls.append(kwargs)
        new_revision(Path(directory))
        return core.load_job(directory)
    monkeypatch.setattr(core, "recompose_transients", recompose)
    applied = review.apply_review(session["session_path"], expected_revision=session["base_revision"])
    repeat = review.apply_review(session["session_path"])
    assert len(calls) == 1
    assert calls[0]["expected_revision"] == session["base_revision"]
    assert calls[0]["labels"] == {"light_00001-1": "unknown", "light_00002-1": None}
    assert calls[0]["overrides"] == {"light_00001-1": "keep", "light_00002-1": "omit"}
    assert calls[0]["policy"] == {name: "omit" for name in ("meteor", "satellite", "aircraft", "artifact", "unknown")}
    assert applied["changed"] and not repeat["changed"]
    assert applied["session"]["reviewer"] == "human"
    with pytest.raises(ValueError, match="已应用"):
        save(session, {})


def test_failed_apply_keeps_draft_and_cannot_replay_stale(job_directory, monkeypatch):
    from starstack import core
    session = save(review.open_review(str(job_directory)), {"light_00001-1": {"override": "keep"}})
    def recompose(directory, **kwargs):
        new_revision(Path(directory))
        raise OSError("post-publication failure")
    monkeypatch.setattr(core, "recompose_transients", recompose)
    with pytest.raises(OSError):
        review.apply_review(session["session_path"])
    with pytest.raises(ValueError, match="另一操作"):
        review.apply_review(session["session_path"])


def test_empty_draft_no_recompose(job_directory, monkeypatch):
    # Version-gated (2026-09-27): a legacy explicit-keep-only draft never
    # recomposes when empty. Under meteor-default-keep-1 an empty draft applies
    # the default once (test_meteor_default_empty_draft_applies_once).
    from starstack import core
    monkeypatch.setattr(core, "recompose_transients", lambda *a, **k: pytest.fail("empty draft recomposed"))
    session = legacy(review.open_review(str(job_directory)))
    assert not review.apply_review(session["session_path"])["changed"]


def test_actual_adjacent_exposure_not_candidate_neighbor(job_directory):
    from PIL import Image
    session = review.open_review(str(job_directory))
    context = review.candidate_context(session["session_path"], "light_00002-1")
    assert [p["frame_index"] for p in context["panels"]] == [1, 2, 3]
    assert all(p["available"] for p in context["panels"])
    # Frame 3 has no candidate but is still the true following exposure.
    data = [np.asarray(Image.open(p["path"])) for p in context["panels"]]
    assert data[0][21, 40] < data[1][21, 40] < data[2][21, 40]
    assert all(a[21, 40] > a[58, 40] for a in data)  # bottom-up FITS correctly mapped to top-down


def test_missing_and_sequence_edge_are_not_fabricated(job_directory):
    session = review.open_review(str(job_directory))
    (job_directory / "work" / "r_light_00003.fit").unlink()
    context = review.candidate_context(session["session_path"], "light_00002-1")
    assert context["panels"][2]["available"] is False
    assert "未保留" in context["panels"][2]["reason"]
    first = review.candidate_context(session["session_path"], "light_00001-1")
    assert first["panels"][0]["available"] is False
    assert "序列边界" in first["panels"][0]["reason"]


def test_bad_reviewer_and_candidate_and_geometry_binding(job_directory):
    session = review.open_review(str(job_directory))
    with pytest.raises(ValueError):
        save(session, {}, reviewer="human-guessed-by-model")
    with pytest.raises(ValueError):
        save(session, {"other-1": {"override": "keep"}})
    path = Path(session["session_path"])
    value = json.loads(path.read_text())
    value["binding"]["candidate_identity_sha256"] = "forged"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="位置绑定"):
        review.read_review(str(path))


def test_legacy_workflow_freeze_refuses_apply(job_directory):
    atomic_json(job_directory.parent / "workflow.json", {"state": "succeeded", "checkpoints": {}})
    session = save(review.open_review(str(job_directory)), {"light_00001-1": {"override": "keep"}})
    with pytest.raises(ValueError, match="旧任务"):
        review.apply_review(session["session_path"])


def test_snapshot_tamper_refuses_apply(job_directory):
    snapshot = job_directory.parent / "stack_snapshot-old.json"
    snapshot.write_text("{}")
    atomic_json(job_directory.parent / "workflow.json", {"state": "succeeded", "checkpoints": {"stack": {
        "files": [{"path": str(snapshot), "sha256": hashlib.sha256(b"different").hexdigest()}]}}})
    session = save(review.open_review(str(job_directory)), {"light_00001-1": {"override": "keep"}})
    with pytest.raises(ValueError, match="快照"):
        review.apply_review(session["session_path"])


def test_registered_checkpoint_tamper_refuses_evidence(job_directory):
    session = review.open_review(str(job_directory))
    path = job_directory / "work" / "r_light_00002.fit"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    result = review.candidate_context(session["session_path"], "light_00002-1")
    assert result["panels"][1]["available"] is False
    assert "已改变" in result["panels"][1]["reason"]


def test_original_tamper_prevents_service_apply(job_directory, monkeypatch):
    from starstack import core
    session = save(review.open_review(str(job_directory)), {"light_00001-1": {"override": "keep"}})
    source = Path(session["binding"]["inputs"][0]["path"])
    source.write_bytes(b"changed source")
    monkeypatch.setattr(core, "recompose_transients", lambda *a, **k: pytest.fail("changed RAW recomposed"))
    with pytest.raises(ValueError, match="原始曝光"):
        review.apply_review(session["session_path"])


def test_verified_native_fallback_is_explicit(job_directory):
    native = []
    for i in (1, 2, 3):
        path = job_directory / "work" / f"light_{i:05d}.fit"
        path.write_bytes((job_directory / "work" / f"r_light_{i:05d}.fit").read_bytes())
        native.append({"path": str(path.relative_to(job_directory)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    def converted_only(job):
        job["checkpoint_files"].pop("registered")
        job["checkpoint_files"]["convert"] = native
    change_job(job_directory, converted_only)
    session = review.open_review(str(job_directory))
    result = review.candidate_context(session["session_path"], "light_00002-1")
    assert all(p["available"] for p in result["panels"])
    assert all("unaligned" in p["grid"] for p in result["panels"])
    assert "未对齐" in result["notice"]


def test_changed_full_preview_not_shown_with_old_geometry(job_directory):
    path = job_directory / "products" / "preview.jpg"
    path.write_bytes(b"old preview")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    change_job(job_directory, lambda job: job.update(diagnostics={"color_export": {"photo": {"jpeg": str(path), "jpeg_sha256": digest}}}))
    session = review.open_review(str(job_directory))
    assert session["preview_path"] == str(path)
    path.write_bytes(b"changed preview")
    with pytest.raises(ValueError, match="整图预览"):
        review.open_review(str(job_directory))


def test_changed_geometry_master_rejected(job_directory):
    review.open_review(str(job_directory))
    path = job_directory / "work" / "sky_master.fit"
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="坐标参考"):
        review.open_review(str(job_directory))


def test_review_never_adds_unselected_automatic_meteor(job_directory, monkeypatch):
    """Real resolver: unknown explicit keep survives, unrelated automatic meteor does not.

    Version-gated (2026-09-27 user decision): this pins the legacy
    explicit-keep-only-1 draft. New meteor-default-keep-1 drafts keep a
    trustworthy meteor by default (test_meteor_default_policy_resolution).
    """
    from starstack import core, transient_stage
    job = core.load_job(job_directory)
    record = json.loads(Path(job["transient_revision"]["record"]).read_text())
    record["policy"] = {"meteor": "keep", "satellite": "omit", "aircraft": "omit", "artifact": "omit", "unknown": "omit"}
    record["candidates"][1].update(suggestion={"class": "meteor"}, effective_label="meteor", decision="keep")
    revision = prepare(job_directory/"products"/"revision1", record)
    change_job(job_directory, lambda value: value.update(transient_revision=revision))
    session = legacy(review.open_review(str(job_directory)))
    assert session["candidates"][1]["review_disposition"] == "建议保留（未采用）"
    session = save(session, {"light_00001-1": {"label": "unknown", "override": "keep"}})
    observed = []
    def recompose(directory, **kwargs):
        updated = transient_stage.update_record(Path(directory)/"work", copy.deepcopy(record),
                    labels=kwargs["labels"], overrides=kwargs["overrides"], notes=kwargs["notes"], policy=kwargs["policy"], persist=False)
        observed.extend(updated["candidates"])
        return core.load_job(directory)
    monkeypatch.setattr(core, "recompose_transients", recompose)
    review.apply_review(session["session_path"], expected_draft_revision=session["draft_revision"])
    assert observed[0]["label"] == "unknown" and observed[0]["decision"] == "keep"
    assert observed[1]["label"] is None and observed[1]["override"] is None and observed[1]["decision"] == "omit"


def test_classification_and_reset_do_not_authorize_addback(job_directory, monkeypatch):
    # Version-gated (2026-09-27): legacy explicit-keep-only-1 drafts only. In
    # meteor-default-keep-1 a meteor label does add back a trustworthy trail.
    from starstack import core, transient_stage
    job = core.load_job(job_directory)
    record = json.loads(Path(job["transient_revision"]["record"]).read_text())
    record["policy"] = dict(review.REVIEW_POLICY)
    record["candidates"][0].update(override="keep", decision="keep")
    session = save(legacy(review.open_review(str(job_directory))), {"light_00001-1": {"override": None}, "light_00002-1": {"label": "meteor"}})
    observed = []
    def recompose(directory, **kwargs):
        updated = transient_stage.update_record(Path(directory)/"work", copy.deepcopy(record),
                    labels=kwargs["labels"], overrides=kwargs["overrides"], notes=kwargs["notes"], policy=kwargs["policy"], persist=False)
        observed.extend(updated["candidates"])
        return core.load_job(directory)
    monkeypatch.setattr(core, "recompose_transients", recompose)
    review.apply_review(session["session_path"])
    assert all(c["decision"] == "omit" for c in observed)
    assert observed[1]["label"] == "meteor" and observed[1]["override"] is None


# --- 2026-09-27 user decision: meteors are the only class kept by default ---

def meteor_record(directory, **quality):
    """Current revision with a trustworthy suggested meteor (1) and a satellite (2)."""
    job = json.loads((directory / "job.json").read_text())
    record = json.loads(Path(job["transient_revision"]["record"]).read_text())
    record["policy"] = {"meteor": "keep", "satellite": "omit", "aircraft": "omit", "artifact": "omit", "unknown": "omit"}
    record["candidates"][0].update(suggestion={"class": "meteor"}, effective_label="meteor", decision="keep")
    record["candidates"][0]["measurement"]["quality"] = dict(quality)
    record["candidates"][1].update(suggestion={"class": "satellite"}, effective_label="satellite")
    target = directory / "products" / "revision-meteor"
    target.mkdir()
    revision = prepare(target, record)
    change_job(directory, lambda value: value.update(transient_revision=revision))
    return record


def resolving(directory, record, observed):
    from starstack import core, transient_stage
    def recompose(job_dir, **kwargs):
        observed.append(kwargs)
        updated = transient_stage.update_record(Path(job_dir) / "work", copy.deepcopy(record), labels=kwargs["labels"],
                    overrides=kwargs["overrides"], notes=kwargs["notes"], policy=kwargs["policy"], persist=False,
                    default_keep_rule=kwargs.get("default_keep_rule"), reviewer=kwargs.get("reviewer"))
        observed.append(updated)
        new_revision(Path(job_dir))
        return core.load_job(job_dir)
    return recompose


def test_new_session_uses_meteor_default_and_layers_draft(job_directory):
    meteor_record(job_directory)
    session = review.open_review(str(job_directory))
    assert session["application_policy_version"] == "meteor-default-keep-1"
    assert session["options"] == {"export_trail_layer": False}
    first, second = session["candidates"]
    assert (first["identity"], first["identity_source"], first["default_keep"], first["effective_keep"],
            first["keep_source"]) == ("meteor", "algorithm", True, True, "algorithm")
    assert first["review_disposition"] == "默认保留（流星）"
    assert (second["identity"], second["default_keep"], second["effective_keep"], second["keep_source"]) == \
        ("satellite", False, False, None)
    saved = save(session, {"light_00001-1": {"label": "satellite"}, "light_00002-1": {"label": "meteor"}},
                 reviewer="agent", options={"export_trail_layer": True})
    first, second = saved["candidates"]
    assert (first["identity"], first["identity_source"], first["effective_keep"]) == ("satellite", "agent", False)
    assert (second["identity"], second["effective_keep"], second["keep_source"]) == ("meteor", True, "agent")
    assert saved["options"] == {"export_trail_layer": True}
    # the record-level disposition (before this draft) is unchanged
    assert first["review_disposition"] == "默认保留（流星）"
    saved = save(saved, {"light_00001-1": {"override": "omit"}, "light_00002-1": {"override": "keep"}}, reviewer="human")
    first, second = saved["candidates"]
    assert first["effective_keep"] is False and second["effective_keep"] is True
    assert second["keep_source"] == "human"
    assert saved["options"] == {"export_trail_layer": True}  # None keeps the stored option


@pytest.mark.parametrize("options", [{"export_trail_layer": 1}, {"other": True}, ["export_trail_layer"]])
def test_review_options_are_strict(job_directory, options):
    session = review.open_review(str(job_directory))
    with pytest.raises(ValueError, match="export_trail_layer"):
        save(session, {}, options=options)


@pytest.mark.parametrize("flag", ["width_limit_reached", "profile_boundary_hit", "sideband_insufficient"])
def test_untrustworthy_meteor_is_not_a_default_keep(job_directory, flag):
    meteor_record(job_directory, **{flag: True})
    candidate = review.open_review(str(job_directory))["candidates"][0]
    assert candidate["identity"] == "meteor" and candidate["measurement_trustworthy"] is False
    assert candidate["untrustworthy_reasons"] == [flag]
    assert not candidate["default_keep"] and not candidate["effective_keep"]
    assert candidate["review_disposition"] == "流星量测不可靠，默认不加回"


def test_meteor_default_policy_resolution(job_directory, monkeypatch):
    """Real resolver: meteor kept by algorithm, relabels move the default, omit removes, keep wins."""
    from starstack import core
    record = meteor_record(job_directory)
    observed = []
    monkeypatch.setattr(core, "recompose_transients", resolving(job_directory, record, observed))
    session = save(review.open_review(str(job_directory)), {"light_00002-1": {"label": "meteor"}}, reviewer="human")
    applied = review.apply_review(session["session_path"], expected_draft_revision=session["draft_revision"])
    kwargs, updated = observed
    assert kwargs["policy"] == {"meteor": "keep", "satellite": "omit", "aircraft": "omit", "artifact": "omit", "unknown": "omit"}
    assert kwargs["default_keep_rule"] == "meteor-default-keep-1" and kwargs["reviewer"] == "human"
    first, second = updated["candidates"]
    assert first["decision"] == "keep" and first["keep_source"] == "algorithm"
    assert second["decision"] == "keep" and second["label_source"] == "human" and second["keep_source"] == "human"
    assert updated["default_keep_rule"]["version"] == "meteor-default-keep-1"
    assert applied["session"]["application_policy"]["version"] == "meteor-default-keep-1"
    assert applied["session"]["application_policy"]["applied_by"] == "human"


def test_meteor_default_relabel_and_omit_drop_keep(job_directory, monkeypatch):
    from starstack import core
    record = meteor_record(job_directory)
    observed = []
    monkeypatch.setattr(core, "recompose_transients", resolving(job_directory, record, observed))
    session = save(review.open_review(str(job_directory)), {"light_00001-1": {"label": "satellite"},
                                                           "light_00002-1": {"override": "keep"}}, reviewer="agent")
    review.apply_review(session["session_path"])
    first, second = observed[1]["candidates"]
    assert first["decision"] == "omit" and "keep_source" not in first
    assert second["decision"] == "keep" and second["keep_source"] == "agent" and second["override_source"] == "agent"


def test_meteor_default_withholds_untrustworthy_measurement(job_directory, monkeypatch):
    from starstack import core
    record = meteor_record(job_directory, profile_boundary_hit=True)
    observed = []
    monkeypatch.setattr(core, "recompose_transients", resolving(job_directory, record, observed))
    session = review.open_review(str(job_directory))
    review.apply_review(session["session_path"])
    first = observed[1]["candidates"][0]
    assert first["decision"] == "omit" and first["requested_decision"] == "keep"
    assert "untrustworthy" in first["decision_reason"] and first["protection_status"] == "needs_review"
    # the withheld meteor stays in the attention list; it is not "not a meteor"
    assert "light_00001-1" in observed[1]["summary"]["recall_review"]["candidate_ids"]


def test_meteor_default_empty_draft_applies_once(job_directory, monkeypatch):
    from starstack import core
    record = meteor_record(job_directory)
    observed = []
    monkeypatch.setattr(core, "recompose_transients", resolving(job_directory, record, observed))
    session = review.open_review(str(job_directory))
    applied = review.apply_review(session["session_path"], expected_draft_revision=session["draft_revision"])
    assert applied["changed"] and observed[0]["reviewer"] is None
    assert applied["session"]["application_policy"]["applied_by"] == "algorithm"
    assert observed[1]["candidates"][0]["keep_source"] == "algorithm"
    # The new revision now carries the rule: an empty draft on it changes nothing.
    job = json.loads((job_directory / "job.json").read_text())
    ruled = dict(observed[1], revision_id=None)
    target = job_directory / "products" / "revision-ruled"
    target.mkdir()
    change_job(job_directory, lambda value: value.update(transient_revision=prepare(target, ruled)))
    monkeypatch.setattr(core, "recompose_transients", lambda *a, **k: pytest.fail("default applied twice"))
    again = review.open_review(str(job_directory))
    assert again["candidates"][0]["effective_keep"] and again["candidates"][0]["keep_source"] == "algorithm"
    assert not review.apply_review(again["session_path"])["changed"]


def test_workflow_without_meteor_default_keeps_explicit_contract(job_directory):
    atomic_json(job_directory.parent / "workflow.json", {"state": "succeeded", "request": {"trails_enabled": True}})
    meteor_record(job_directory)
    session = review.open_review(str(job_directory))
    assert session["application_policy_version"] == "explicit-keep-only-1"
    first = session["candidates"][0]
    assert first["identity"] == "meteor" and not first["default_keep"] and not first["effective_keep"]
    assert first["review_disposition"] == "建议保留（未采用）"


def test_meteor_workflow_session_and_trails_preview(job_directory):
    from PIL import Image
    works = job_directory.parent / "works"
    works.mkdir()
    previews = {}
    for name in ("base", "trails"):
        (works / name).mkdir()
        path = works / name / "final.jpg"
        Image.new("RGB", (100, 80), (10, 20, 30) if name == "base" else (40, 50, 60)).save(path)
        previews[name] = {"name": "final.jpg", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(job_directory.parent / "workflow.json", {
        "state": "succeeded", "request": {"trails_enabled": True, "trail_default": "meteor"},
        "stack_job_dir": str(job_directory), "delivery_dir": str(works / "trails"),
        "base_delivery_dir": str(works / "base"), "artifacts": [previews["trails"]], "base_artifacts": [previews["base"]]})
    session = review.open_review(str(job_directory))
    assert session["application_policy_version"] == "meteor-default-keep-1"
    assert session["preview_path"] == str((works / "trails" / "final.jpg").resolve())
    assert session["base_preview_path"] == str((works / "base" / "final.jpg").resolve())
    (works / "trails" / "final.jpg").write_bytes(b"changed after publication")
    assert review.read_review(session["session_path"])["preview_path"] != str((works / "trails" / "final.jpg").resolve())
