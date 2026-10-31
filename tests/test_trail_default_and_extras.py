"""2026-09-27 user batch: meteor default add-back, trails deliveries, trail layer, star extras.

Only RAW stacking, Siril and the star model are stubbed (as in test_workflow);
camera colour, fixed-colour rendering, signed pairing, publication, receipts
and verification run for real. None of this certifies a physical meteor.
"""
from pathlib import Path
import asyncio
import json
import shutil
import sys

import numpy as np
import pytest
import tifffile
from PIL import Image

from starstack import workflow as wf, workflow_edit as edit, trail_review_session as review, storage
from starstack import color_export as ce, master_export as me, imaging
from starstack.compose_evidence import write_weight
from test_workflow import setup, stub_model
import color_fixtures as cf

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")

LAYER = "trails_layer_rgba16.tif"


def build(s, monkeypatch, *, works=None, layer=False, kept=True, fail=None, trails=True, stars=False,
          save_starless=False, save_star_layer=False, observe=None, jpeg_quality=None, delivery_format=None):
    """Calibrated synthetic workflow; the child trail review is stubbed at the session boundary."""
    original = s["submit"]
    ref = cf.write_reference_dng(s["output"].parent / "reference.dng")
    basis = me.colour_basis_from_dng(ref)
    camera, *_ = cf.synthetic_camera_master(144, 160, ground_from=120, nodata_columns=0)

    def submit(*a, **kw):
        job = original(*a, **kw)
        directory = Path(job["job_dir"])
        exp = directory / "calibrated"
        exp.mkdir()
        shutil.copyfile(s["mask"], directory / "sky_mask.png")
        fit = exp / "master_linear.fit"
        imaging.write_linear(fit, np.flipud(camera), "synthetic input")
        _, colour = ce.export_job_products(np.flipud(camera), exp, lambda: None, basis=basis, source_fits=fit,
                                           sky_mask_path=s["mask"], color_adjustments=kw.get("color_adjustments"))
        desc = write_weight(exp, np.asarray(Image.open(s["mask"]), np.float32) / 255, fit)
        candidate = exp / "candidates.json"
        candidate.write_text('{"candidates": []}')
        job.update(artifacts=[{"path": str(fit), "kind": "fits"}],
                   transient_revision={"id": "original", "record": str(candidate), "sha256": wf.core._sha(candidate)})
        job["diagnostics"].update(color_export=colour, sky_validity={"compose_weight": desc},
                                  transients={"status": "needs_review", "summary": {"candidates": 2}})
        wf.atomic_json(directory / "job.json", job)
        return job

    monkeypatch.setattr(wf.core, "submit_job", submit)
    state = {"opens": 0, "applies": 0}

    def open_review(job_dir):
        state["opens"] += 1
        session = Path(job_dir) / "review" / "draft-auto.json"
        session.parent.mkdir(exist_ok=True)
        session.write_text("{}")
        state["session"] = session
        return {"session_path": str(session), "draft_revision": "auto-token",
                "application_policy_version": review.METEOR_POLICY_VERSION}

    def apply(path, **kwargs):
        if observe:
            observe(Path(path))
        if fail:
            raise fail
        state["applies"] += 1
        jobdir = Path(path).parent.parent
        job = wf.core.load_job(jobdir)
        exp = jobdir / "products" / ("reviewed-%d" % state["applies"])
        (exp / "transients").mkdir(parents=True)
        candidates = [{"id": "light_00001-1", "decision": "keep" if kept else "omit", "requested_reason": "policy",
                       "label": None, "suggestion": {"class": "meteor"}},
                      {"id": "light_00002-1", "decision": "omit", "requested_reason": "policy", "label": None,
                       "suggestion": {"class": "unknown"}}]
        record = exp / "transients" / "candidates.json"
        record.write_text(json.dumps({"revision_id": exp.name, "candidates": candidates}))
        diag = {"composite": None, "kept": [], "summary": {"recall_review": {"candidate_ids": ["light_00002-1"]}}}
        if kept:
            changed = camera.copy()
            changed[30:36, 42:75] += .07
            fit = exp / "transients.fit"
            imaging.write_linear(fit, np.flipud(changed), "synthetic trail")
            _, colour = ce.export_job_products(np.flipud(changed), exp, lambda: None, basis=basis, source_fits=fit,
                                               sky_mask_path=s["mask"], color_adjustments=job["color_adjustments"])
            diag.update(composite=str(fit), color_export=colour, kept=[{"id": "light_00001-1"}])
        job["transient_revision"] = {"id": exp.name, "record": str(record), "sha256": wf.core._sha(record)}
        job["diagnostics"]["transients"] = diag
        wf.atomic_json(jobdir / "job.json", job)
        session = {"state": "applied", "reviewer": kwargs.get("reviewer"),
                   "application_policy_version": review.METEOR_POLICY_VERSION,
                   "application_policy": {"applied_by": "algorithm"}, "options": {"export_trail_layer": layer},
                   "applied_revision": job["transient_revision"]}
        return {"job": job, "session": session, "changed": True}

    monkeypatch.setattr(review, "open_review", open_review)
    monkeypatch.setattr(review, "apply_review", apply)
    stub_model(monkeypatch)
    options = {"delivery_root": works} if works else {}
    if delivery_format:
        options["delivery_format"] = delivery_format
    if stars:
        options.update(stars_enabled=True, star_model_path=s["model"], background_strength=.7,
                       backend_options={"tile": 64, "stride": 32}, save_starless=save_starless,
                       save_star_layer=save_star_layer)
    if trails:
        options.update(trails_enabled=True, trail_default="meteor")
    if jpeg_quality is not None:
        options["jpeg_quality"] = jpeg_quality
    record = wf.submit_workflow(s["inputs"], s["output"], sky_mask_path=s["mask"],
                                color_adjustments={"warmth": 12, "tint": 5, "exposure_ev": .3}, **options)
    return record, state


def test_new_options_are_strict_and_absent_options_keep_the_digest(setup):
    first = wf.submit_workflow(setup["inputs"], setup["output"], idempotency_key="a")
    same = wf.submit_workflow(setup["inputs"], setup["output"], idempotency_key="b", trail_default="review_only",
                              save_starless=False, save_star_layer=False, jpeg_quality=None)
    assert first["signature"] == same["signature"]
    assert not {"trail_default", "save_starless", "save_star_layer", "jpeg_quality"} & set(same["request"])
    assert "jpeg_quality" not in same["config"]
    # Unset, the child stack job's folder (derived from the signature) is today's as well.
    assert wf._stack_job_dir(Path(first["workflow_dir"]), first["signature"]).name == \
        wf._stack_job_dir(Path(same["workflow_dir"]), same["signature"]).name
    quality = wf.submit_workflow(setup["inputs"], setup["output"], idempotency_key="q", jpeg_quality=80)
    assert quality["request"]["jpeg_quality"] == quality["config"]["jpeg_quality"] == 80
    assert quality["signature"] != first["signature"]
    for bad in (59, 101, 95.0, True, False, "95", [95]):
        with pytest.raises(ValueError, match="jpeg_quality"):
            wf.submit_workflow(setup["inputs"], setup["output"], jpeg_quality=bad)
    tampered = wf._read(Path(quality["workflow_dir"]))
    tampered["config"] = {**tampered["config"], "jpeg_quality": 90}
    tampered["request"] = {**tampered["request"], "config": tampered["config"]}
    tampered["signature"] = wf._digest(tampered["request"])
    with pytest.raises(ValueError, match="JPEG 质量"):
        wf._verify(tampered)
    meteor = wf.submit_workflow(setup["inputs"], setup["output"], trails_enabled=True, trail_default="meteor")
    assert meteor["request"]["trail_default"] == meteor["config"]["trail_default"] == "meteor"
    assert "自动加回" in meteor["limitations"][0]
    for kwargs, match in (({"trail_default": "always"}, "trail_default"), ({"trail_default": ["meteor"]}, "trail_default"),
                          ({"trail_default": "meteor"}, "亮迹"), ({"save_starless": 1}, "bool"),
                          ({"save_star_layer": "true"}, "bool"), ({"save_starless": True}, "整理星点"),
                          ({"save_star_layer": True}, "整理星点")):
        with pytest.raises(ValueError, match=match):
            wf.submit_workflow(setup["inputs"], setup["output"], **kwargs)
    assert setup["calls"]["submits"] == 0


def test_meteor_default_is_added_back_after_the_base_and_shown_first(setup, monkeypatch):
    seen = []
    record, state = build(setup, monkeypatch, observe=lambda path: seen.append(wf._read(Path(record["workflow_dir"]))))
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    # The add-back runs while the workflow is still non-terminal, after the base publication.
    assert seen[0]["state"] == "running" and seen[0]["stage"] == "trail_addback" and seen[0]["trail_addback_pending"]
    assert seen[0]["delivery_dir"] == seen[0]["base_delivery_dir"]
    base, current = Path(done["base_delivery_dir"]), Path(done["delivery_dir"])
    assert base.name.startswith("delivery-") and not base.name.startswith("delivery-trails-")
    assert current.name.startswith("delivery-trails-1kept-") and current.parent == base.parent
    assert "trail_addback_pending" not in done and done["stage"] == "complete" and done["progress"] == 100
    review_state = done["trail_review"]
    assert review_state["automatic_addback"] is True and review_state["applied_by"] == "algorithm"
    assert review_state["default_policy"] == "meteor-default-keep-1"
    assert review_state["kept_ids"] == ["light_00001-1"] and review_state["kept_count"] == 1
    assert review_state["kept_by"] == {"algorithm": 1, "human": 0, "agent": 0}
    assert review_state["pending_count"] == 1 and review_state["state"] == "needs_review" and done["needs_review"]
    assert "自动加回 1 条" in done["message"] and "1 条候选待复核" in review_state["message"]
    trails = tifffile.imread(current / "final_srgb16.tif")
    plain = tifffile.imread(base / "final_srgb16.tif")
    assert np.any(trails[27:40, 38:79] != plain[27:40, 38:79]) and np.array_equal(trails[48:], plain[48:])
    assert {a["name"] for a in done["base_artifacts"]} >= {"final.jpg", "final_srgb16.tif"}
    assert state["opens"] == 1 and state["applies"] == 1
    receipt = json.loads((current / "review_delivery.json").read_text(encoding="utf-8"))
    assert receipt["applied_by"] == "algorithm" and receipt["kept_ids"] == ["light_00001-1"]
    assert receipt["identity_certified"] is False
    wf._verify(done)
    assert wf.run_workflow(record["workflow_dir"])["delivery_dir"] == done["delivery_dir"]


def test_no_trustworthy_meteor_publishes_no_second_folder(setup, monkeypatch):
    record, _ = build(setup, monkeypatch, kept=False)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["delivery_dir"] == done["base_delivery_dir"]
    assert not list(Path(done["workflow_dir"]).glob("delivery-trails-*"))
    assert not list(Path(done["workflow_dir"]).glob(".partial-reviewed-*"))
    assert done["trail_review"]["kept_count"] == 0 and done["trail_review"]["automatic_addback"] is False
    assert "未发现量测可靠的流星候选" in done["message"]
    wf._verify(done)


def test_failed_or_stopped_addback_keeps_the_base_photograph(setup, monkeypatch):
    record, _ = build(setup, monkeypatch, fail=RuntimeError("synthetic add-back failure"))
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded" and done["artifacts"]
    assert done["delivery_dir"] == done["base_delivery_dir"]
    assert "synthetic add-back failure" in done["trail_review"]["automatic_addback_error"]
    assert "重试" in done["message"] and "trail_addback_pending" not in done
    wf._verify(done)


def test_cancel_during_addback_keeps_the_base_and_clears_the_request(setup, monkeypatch):
    def stop(path):
        wf.cancel_workflow(record["workflow_dir"])
    record, _ = build(setup, monkeypatch, observe=stop, fail=wf.WorkflowCancelled("stop"))
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded" and done["delivery_dir"] == done["base_delivery_dir"]
    assert not (Path(record["workflow_dir"]) / "cancel.request").exists()
    assert "已停止自动加回" in done["message"]


def test_interrupted_addback_resumes_without_republishing_the_base(setup, monkeypatch):
    record, state = build(setup, monkeypatch)
    publish = wf._publish
    calls = []
    monkeypatch.setattr(wf, "_publish", lambda *a: calls.append(1) or publish(*a))
    automatic = wf._automatic_trails
    monkeypatch.setattr(wf, "_automatic_trails", lambda directory: wf._read(directory))  # worker stopped
    stopped = wf.run_workflow(record["workflow_dir"])
    assert stopped["state"] == "running" and stopped["trail_addback_pending"] and len(calls) == 1
    monkeypatch.setattr(wf, "_automatic_trails", automatic)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded" and len(calls) == 1
    assert done["base_delivery_dir"] == stopped["delivery_dir"] != done["delivery_dir"]
    assert state["applies"] == 1


def test_later_review_keeping_nothing_reverts_to_base_without_new_folder(setup, monkeypatch):
    record, _ = build(setup, monkeypatch)
    done = wf.run_workflow(record["workflow_dir"])
    folders = set(Path(done["workflow_dir"]).glob("delivery-*"))
    jobdir = Path(done["stack_job_dir"])
    session = jobdir / "review" / "draft-manual.json"
    session.write_text("{}")

    def omit_all(path, **kwargs):
        job = wf.core.load_job(jobdir)
        exp = jobdir / "products" / "omitted"
        (exp / "transients").mkdir(parents=True)
        record_path = exp / "transients" / "candidates.json"
        record_path.write_text(json.dumps({"revision_id": "omitted", "candidates": []}))
        job["transient_revision"] = {"id": "omitted", "record": str(record_path), "sha256": wf.core._sha(record_path)}
        job["diagnostics"]["transients"] = {"composite": None, "kept": [], "summary": {"recall_review": {"candidate_ids": []}}}
        wf.atomic_json(jobdir / "job.json", job)
        return {"job": job, "session": {"state": "applied", "reviewer": "human", "application_policy":
                {"applied_by": "human"}, "application_policy_version": review.METEOR_POLICY_VERSION}, "changed": True}

    monkeypatch.setattr(review, "apply_review", omit_all)
    result = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="manual")
    current = result["workflow"]
    assert result["published"] is False and result["delivery_dir"] is None and result["identical_to_base"]
    assert result["kept_count"] == 0 and result["delivery"] is None
    assert current["delivery_dir"] == done["base_delivery_dir"] and current["artifacts"] == done["base_artifacts"]
    assert set(Path(done["workflow_dir"]).glob("delivery-*")) == folders
    assert "未另存新文件夹" in current["message"] and not current["needs_review"]
    assert all(item["role"] != "trails_layer" for item in current["extras"])
    wf._verify(current)


def test_external_trail_layer_is_bit_exact_and_cache_stage_is_slim(setup, monkeypatch):
    works = setup["output"].parent / "works"
    record, _ = build(setup, monkeypatch, works=works, layer=True)
    copies = []
    original = edit._copy_verified
    monkeypatch.setattr(edit, "_copy_verified", lambda *a: copies.append(a) or original(*a))
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert not copies  # the base master is packaged from the frozen stack, never duplicated in the cache stage
    base, current = Path(done["base_delivery_dir"]), Path(done["delivery_dir"])
    assert current.is_relative_to(works) and current.name.startswith("delivery-trails-1kept-")
    assert {p.name for p in current.iterdir()} == {"master_scene_linear.tif", "final_srgb16.tif", "final.jpg", LAYER}
    receipt = storage.delivery_receipt(done, current)
    assert receipt["extras"] == [{"file": LAYER, "role": "trails_layer"}]
    storage.verify_delivery(current, receipt=receipt)
    assert "extras" not in storage.delivery_receipt(done, base)
    with tifffile.TiffFile(current / LAYER) as tf:
        page = tf.pages[0]
        assert int(page.compression) == 8 and page.shape[-1] == 4 and page.dtype == np.uint16
        rgba = page.asarray()
    alpha = rgba[..., 3:].astype(np.uint64)
    assert set(np.unique(alpha)) <= {0, 65535} and alpha.any()
    under = tifffile.imread(base / "final_srgb16.tif").astype(np.uint64)
    blended = (rgba[..., :3].astype(np.uint64) * alpha + under * (65535 - alpha)) // 65535
    assert np.array_equal(blended, tifffile.imread(current / "final_srgb16.tif"))
    extra = next(item for item in done["extras"] if item["role"] == "trails_layer")
    assert extra["path"] == str(current / LAYER) and extra["sha256"] == wf.core._sha(current / LAYER)
    assert done["trail_review"]["trail_layer"] == str(current / LAYER)
    receipt_path = Path(done["review_deliveries"][-1]["receipt"])
    assert receipt_path.parent.name.startswith("reviewed-") and receipt_path.parent.parent == Path(done["workflow_dir"])
    review_record = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert review_record["trail_layer"]["blend_verified"] and "not scene-linear" in review_record["trail_layer"]["encoding"]
    assert all(p.suffix == ".json" for p in receipt_path.parent.rglob("*") if p.is_file())
    assert not list(Path(done["workflow_dir"]).glob(".partial-reviewed-*"))
    wf._verify(done)
    plan = storage.plan_cache_cleanup(done["workflow_dir"], cache_root=Path(done["workflow_dir"]).parent)
    assert set(plan["preserved_deliveries"]) == {str(base), str(current)}


def test_star_extras_are_copied_from_the_verified_layer_pack(setup, monkeypatch):
    stub_model(monkeypatch)
    r = wf.submit_workflow(setup["inputs"], setup["output"], sky_mask_path=setup["mask"], stars_enabled=True,
                           star_model_path=setup["model"], background_strength=.7, backend_options={"tile": 64, "stride": 32},
                           save_starless=True, save_star_layer=True)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    root = Path(done["delivery_dir"])
    pack = done["finishing"]["layers"]["artifacts"]
    for name, layer in (("starless_srgb16.tif", "00_background.tif"), ("stars_add_srgb16.tif", "02_stars_add.tif"),
                        ("stars_subtract_srgb16.tif", "01_stars_subtract.tif")):
        assert wf.core._sha(root / name) == pack[layer]["sha256"]
    starless, add, subtract = (tifffile.imread(root / n).astype(np.int64) for n in
                               ("starless_srgb16.tif", "stars_add_srgb16.tif", "stars_subtract_srgb16.tif"))
    before = tifffile.imread(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["tiff"]).astype(np.int64)
    assert np.array_equal(starless - subtract + add, before)
    assert {item["role"] for item in done["extras"]} == {"starless", "stars_add", "stars_subtract"}
    assert all(item["sha256"] == wf.core._sha(Path(item["path"])) for item in done["extras"])
    wf._verify(done)
    stack = done["checkpoints"]["stack"]["data"]
    with pytest.raises(ValueError, match="星点图层包"):
        wf._publish(Path(done["workflow_dir"]), stack, {**done["finishing"], "layers": None})


def test_external_v2_star_extras_are_declared_in_the_receipt(setup, monkeypatch):
    works = setup["output"].parent / "works"
    record, _ = build(setup, monkeypatch, works=works, trails=False, stars=True, save_starless=True)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    root = Path(done["delivery_dir"])
    assert {p.name for p in root.iterdir()} == {"master_scene_linear.tif", "final_srgb16.tif", "final.jpg",
                                                "starless_srgb16.tif"}
    receipt = storage.delivery_receipt(done, root)
    assert receipt["extras"] == [{"file": "starless_srgb16.tif", "role": "starless"}]
    storage.verify_delivery(root, receipt=receipt)
    assert [item["role"] for item in done["extras"]] == ["starless"]
    wf._verify(done)


@pytest.mark.parametrize("change", ["unknown", "role", "undeclared", "duplicate", "missing_file"])
def test_v2_receipt_accepts_only_declared_allowlisted_extras(tmp_path, change):
    root = tmp_path / "delivery-x"
    root.mkdir()
    names = ["master_scene_linear.tif", "final_srgb16.tif", "final.jpg", "starless_srgb16.tif"]
    for name in names:
        (root / name).write_bytes(name.encode())
    files = [{"file": n, "sha256": wf.core._sha(root / n), "bytes": (root / n).stat().st_size} for n in names]
    receipt = {"version": storage.IMAGE_VERSION, "master_file": "master_scene_linear.tif", "files": files,
               "extras": [{"file": "starless_srgb16.tif", "role": "starless"}]}
    assert storage.verify_delivery(root, receipt=receipt)
    three = dict(receipt, files=files[:3])
    three.pop("extras")
    assert storage.verify_delivery(root, receipt=three)
    if change == "unknown":
        (root / "notes.tif").write_bytes(b"x")
        receipt["files"].append({"file": "notes.tif", "sha256": wf.core._sha(root / "notes.tif"), "bytes": 1})
        receipt["extras"].append({"file": "notes.tif", "role": "starless"})
    elif change == "role":
        receipt["extras"] = [{"file": "starless_srgb16.tif", "role": "trails_layer"}]
    elif change == "undeclared":
        receipt.pop("extras")
    elif change == "duplicate":
        receipt["extras"] = receipt["extras"] * 2
    else:
        receipt["extras"].append({"file": LAYER, "role": "trails_layer"})
    with pytest.raises(ValueError):
        storage.verify_delivery(root, receipt=receipt)


def test_mcp_and_cli_forward_the_new_intents(setup, monkeypatch):
    from starstack.mcp_server import WorkspaceTools, create_server
    from starstack import cli
    service = WorkspaceTools(setup["inputs"], setup["output"], setup["engine"])
    files = [str(setup["inputs"] / f"{i}.NEF") for i in range(3)]
    for kwargs in ({"trail_default": "always"}, {"save_starless": 1}, {"save_star_layer": "true"}):
        with pytest.raises(ValueError):
            service.submit_workflow(input_files=files, trails_enabled=True, **kwargs)
    result = service.submit_workflow(input_files=files, trails_enabled=True, trail_default="meteor")
    assert result["request"]["trail_default"] == "meteor"
    with pytest.raises(ValueError, match="bool"):
        service.save_trail_review(result["workflow_dir"], "draft.json", {}, "token", export_trail_layer="yes")

    async def schemas():
        tools = {tool.name: tool for tool in await create_server(service).list_tools()}
        submit = tools["submit_workflow"].inputSchema["properties"]
        assert submit["save_starless"]["default"] is False and submit["save_star_layer"]["type"] == "boolean"
        assert "meteor" in json.dumps(submit["trail_default"]) and "review_only" in json.dumps(submit["trail_default"])
        assert "export_trail_layer" in tools["save_trail_review"].inputSchema["properties"]
    asyncio.run(schemas())
    called = {}
    monkeypatch.setattr(wf, "submit_workflow", lambda *a, **k: called.update(kwargs=k) or {"state": "queued"})
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--files", "one.NEF", "two.NEF", "three.NEF",
                                      "--retain-trails", "--trail-default", "meteor", "--reduce-stars",
                                      "--save-starless", "--save-star-layer"])
    assert cli.main() == 0
    assert called["kwargs"]["trail_default"] == "meteor"
    assert called["kwargs"]["save_starless"] is True and called["kwargs"]["save_star_layer"] is True
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--files", "one.NEF", "two.NEF", "three.NEF"])
    assert cli.main() == 0
    assert not {"trail_default", "save_starless", "save_star_layer"} & set(called["kwargs"])


from test_s4_protection_integration import synthetic_job  # noqa: E402  real core flow with a controlled engine


def test_real_recompose_records_the_meteor_rule_and_keep_provenance(synthetic_job):
    from starstack import core, transient_stage as ts
    _, job = synthetic_job
    work = Path(job["job_dir"]) / "work"
    measured = [c for c in ts.read_record(work)["candidates"] if c["measurement"].get("status") == "measured"]
    assert measured
    labelled = measured[0]["id"]
    updated = core.recompose_transients(job["job_dir"], labels={labelled: "meteor"}, policy=dict(ts.METEOR_DEFAULT_POLICY),
                                        default_keep_rule=ts.METEOR_DEFAULT_RULE, reviewer="human")
    after = ts.read_record(work, job=updated)
    assert after["default_keep_rule"]["version"] == "meteor-default-keep-1"
    target = next(c for c in after["candidates"] if c["id"] == labelled)
    assert target["label_source"] == "human" and target["effective_label"] == "meteor"
    if ts.untrustworthy_reasons(target):
        assert target["decision"] == "omit" and target["protection_reason"].startswith("untrustworthy_measurement:")
    else:
        assert target["decision"] == "keep" and target["keep_source"] == "human"
    for c in after["candidates"]:
        if c.get("requested_reason") == "policy" and c["effective_label"] == "meteor":
            assert (c["decision"] == "keep") == (not ts.untrustworthy_reasons(c))
        if c["decision"] == "keep":
            assert c["keep_source"] in ("human", "algorithm")
        else:
            assert "keep_source" not in c
    kept = [item["id"] for item in updated["diagnostics"]["transients"]["kept"]]
    assert sorted(kept) == sorted(c["id"] for c in after["candidates"] if c["decision"] == "keep")
    again = core.recompose_transients(job["job_dir"], notes={labelled: "checked"})
    kept_again = ts.read_record(work, job=again)
    assert kept_again["default_keep_rule"]["version"] == "meteor-default-keep-1"
    assert next(c for c in kept_again["candidates"] if c["id"] == labelled)["label_source"] == "human"


def test_star_extras_are_saved_when_the_density_advice_skips_reduction(setup, monkeypatch):
    stub_model(monkeypatch)
    mask = np.zeros((144, 160), np.uint8)
    mask[:70] = 255
    Image.fromarray(mask).save(setup["mask"])
    r = wf.submit_workflow(setup["inputs"], setup["output"], sky_mask_path=setup["mask"], stars_enabled=True,
                           star_model_path=setup["model"], backend_options={"tile": 64, "stride": 32}, save_starless=True)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert done["finishing"]["state"] == "reduction_skipped"
    assert (Path(done["delivery_dir"]) / "starless_srgb16.tif").is_file()
    assert not (Path(done["delivery_dir"]) / "stars_add_srgb16.tif").exists()
    assert [item["role"] for item in done["extras"]] == ["starless"]


def test_a_saved_reviewer_draft_is_never_applied_as_the_algorithm_default(setup, monkeypatch):
    record, state = build(setup, monkeypatch)
    opened = review.open_review
    monkeypatch.setattr(review, "open_review", lambda job_dir: {**opened(job_dir), "decisions": {"light_00001-1": {"override": "omit"}}})
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded" and done["delivery_dir"] == done["base_delivery_dir"]
    assert state["applies"] == 0 and "自动加回让位" in done["trail_review"]["automatic_addback_error"]


# ---------------------------------------------------------------------------------------- 2026-10-09 JPEG quality
# The quality is read from a JPEG's quantization tables and its chroma subsampling from Pillow (0 = 4:4:4).

from test_jpeg_quality import encoding, expected_jpeg, tables  # noqa: E402


@pytest.mark.parametrize("external", [False, True])
def test_jpeg_quality_is_frozen_and_applied_to_the_base_and_trails_versions(setup, monkeypatch, external):
    works = setup["output"].parent / "works" if external else None
    record, state = build(setup, monkeypatch, works=works, jpeg_quality=80)
    assert record["request"]["jpeg_quality"] == record["config"]["jpeg_quality"] == 80
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    base, current = Path(done["base_delivery_dir"]), Path(done["delivery_dir"])
    assert current.name.startswith("delivery-trails-1kept-") and state["applies"] == 1
    for folder in (base, current):
        quantization, sampling, icc = encoding(folder / "final.jpg")
        assert quantization == tables(80) and sampling == 0 and icc
        # round(code / 257) of the delivered 16-bit TIFF with its own ICC profile, 4:4:4, at the frozen quality
        assert (folder / "final.jpg").read_bytes() == expected_jpeg(folder / "final_srgb16.tif", 80)
    # The frozen stack JPEG is not rewritten: only the delivered copy is encoded afresh.
    frozen = Path(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["jpeg"])
    assert wf.core._sha(frozen) == done["checkpoints"]["stack"]["data"]["colour"]["photo"]["jpeg_sha256"]
    assert encoding(frozen)[0] == tables(95)
    # The encoded files were copied into the delivery; their work folder does not stay in the cache.
    assert not list(Path(done["workflow_dir"]).glob("jpeg-*"))
    if not external:            # the trails version's record states the encoding it used
        receipt = json.loads((current / "review_delivery.json").read_text(encoding="utf-8"))
        assert (receipt["jpeg_quality"], receipt["jpeg_subsampling"]) == (80, "4:4:4")
    wf._verify(done)


def test_without_jpeg_quality_every_delivered_jpeg_keeps_its_encoding(setup, monkeypatch):
    record, _ = build(setup, monkeypatch)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    base, current = Path(done["base_delivery_dir"]), Path(done["delivery_dir"])
    frozen = Path(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["jpeg"])
    assert (base / "final.jpg").read_bytes() == frozen.read_bytes()            # the base: today's 95 / 4:4:4 bytes
    assert encoding(base / "final.jpg")[:2] == (tables(95), 0)
    assert encoding(current / "final.jpg")[:2] == (tables(97), 0)                # a trails version: still 97 / 4:4:4
    assert (current / "final.jpg").read_bytes() == expected_jpeg(current / "final_srgb16.tif", 97)
    assert not list(Path(done["workflow_dir"]).glob("jpeg-*"))
    assert "jpeg_quality" not in done["request"]
    assert not {"jpeg_quality", "jpeg_subsampling"} & set(json.loads((current / "review_delivery.json").read_text(
        encoding="utf-8")))
    frozen_record = Path(done["checkpoints"]["stack"]["data"]["colour"]["photo"]["record"])
    assert (base / "photo_render.json").read_bytes() == frozen_record.read_bytes()


def test_a_later_review_uses_the_quality_frozen_in_its_workflow(setup, monkeypatch):
    """A trails version applied after the run takes the workflow's quality, whatever the GUI shows by then."""
    record, _ = build(setup, monkeypatch, jpeg_quality=63)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    session = Path(done["stack_job_dir"]) / "review" / "draft-manual.json"
    session.write_text("{}")
    apply = review.apply_review

    def keep_again(path, **kwargs):
        result = apply(path, **kwargs)
        return {**result, "session": {**result["session"], "reviewer": "human",
                                       "application_policy": {"applied_by": "human"}}}
    monkeypatch.setattr(review, "apply_review", keep_again)
    result = edit.apply_review(done["workflow_dir"], session, expected_draft_revision="manual")
    assert result["published"] is True
    latest = Path(result["delivery_dir"])
    assert latest != Path(done["delivery_dir"]) and encoding(latest / "final.jpg")[:2] == (tables(63), 0)


def test_star_reduced_final_and_the_local_before_reduction_jpeg_follow_the_quality(setup, monkeypatch):
    stub_model(monkeypatch)
    r = wf.submit_workflow(setup["inputs"], setup["output"], sky_mask_path=setup["mask"], stars_enabled=True,
                           star_model_path=setup["model"], background_strength=.7,
                           backend_options={"tile": 64, "stride": 32}, jpeg_quality=70)
    done = wf.run_workflow(r["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    root = Path(done["delivery_dir"])
    assert {"final.jpg", "stack_before_reduction.jpg"} <= {a["name"] for a in done["artifacts"]}
    for jpeg, tiff in (("final.jpg", "final_srgb16.tif"),
                       ("stack_before_reduction.jpg", "stack_before_reduction_srgb16.tif")):
        assert encoding(root / jpeg)[:2] == (tables(70), 0), jpeg
        assert (root / jpeg).read_bytes() == expected_jpeg(root / tiff, 70)
    # The frozen star-reduction stage keeps its own JPEG (95 / 4:2:0); the delivery is encoded from its TIFF.
    staged = Path(done["finishing"]["reduction"]["processing_directory"]) / "star_reduced.jpg"
    assert encoding(staged)[:2] == (tables(95), 2)
    wf._verify(done)


def _record_encodes(monkeypatch):
    """The delivered names workflow._encode_jpeg is called for, in order."""
    calls = []
    encode = wf._encode_jpeg

    def spy(source, target, quality):
        calls.append(Path(target).name)
        return encode(source, target, quality)
    monkeypatch.setattr(wf, "_encode_jpeg", spy)
    return calls


def test_an_images_only_delivery_encodes_only_the_jpeg_it_carries(setup, monkeypatch):
    """aester-delivery-2 (the GUI's) carries final.jpg alone: the stack photograph's JPEG is not encoded for it."""
    calls = _record_encodes(monkeypatch)
    record, _ = build(setup, monkeypatch, works=setup["output"].parent / "works", stars=True, trails=False,
                      jpeg_quality=70)
    directory = Path(record["workflow_dir"])
    stale = directory / "jpeg-0123456789ab"            # left by an earlier, cancelled publication attempt
    stale.mkdir()
    (stale / "final.jpg").write_bytes(b"stale")
    (directory / "jpeg-notes").mkdir()                 # not an attempt's folder: left alone
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    assert calls == ["final.jpg"]
    root = Path(done["delivery_dir"])
    assert {a["name"] for a in done["artifacts"]} == {"master_scene_linear.tif", "final_srgb16.tif", "final.jpg"}
    assert encoding(root / "final.jpg")[:2] == (tables(70), 0)
    assert (root / "final.jpg").read_bytes() == expected_jpeg(root / "final_srgb16.tif", 70)
    assert sorted(path.name for path in directory.glob("jpeg-*")) == ["jpeg-notes"]
    wf._verify(done)


def test_at_the_frozen_photo_encoding_the_base_jpeg_is_delivered_as_it_is(setup, monkeypatch):
    """q95 equals the stack photograph's own JPEG (round(uint16/257), its ICC, 4:4:4): no encode, the same bytes."""
    calls = _record_encodes(monkeypatch)
    record, _ = build(setup, monkeypatch, trails=False, jpeg_quality=95)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    photo = done["checkpoints"]["stack"]["data"]["colour"]["photo"]
    root = Path(done["delivery_dir"])
    assert calls == []
    assert (root / "final.jpg").read_bytes() == Path(photo["jpeg"]).read_bytes() \
        == expected_jpeg(root / "final_srgb16.tif", 95)
    assert (root / "photo_render.json").read_bytes() == Path(photo["record"]).read_bytes()
    assert not list(Path(done["workflow_dir"]).glob("jpeg-*"))
    # Another quality is encoded afresh, from the same TIFF.
    other = wf.submit_workflow(setup["inputs"], setup["output"], sky_mask_path=setup["mask"], idempotency_key="q96",
                               color_adjustments={"warmth": 12, "tint": 5, "exposure_ev": .3}, jpeg_quality=96)
    again = wf.run_workflow(other["workflow_dir"])
    assert again["state"] == "succeeded", again.get("error")
    assert calls == ["final.jpg"]
    assert (Path(again["delivery_dir"]) / "final.jpg").read_bytes() == \
        expected_jpeg(Path(again["delivery_dir"]) / "final_srgb16.tif", 96)


@pytest.mark.parametrize("delivery", ["local", "aester-delivery-1"])
def test_a_delivered_photo_record_describes_the_delivered_jpeg(setup, monkeypatch, delivery):
    """Local and v1 layouts deliver photo_render.json: it names the delivered (re-encoded) JPEG and its encoding."""
    local = delivery == "local"
    record, _ = build(setup, monkeypatch, works=None if local else setup["output"].parent / "works",
                      delivery_format=None if local else delivery, stars=True, trails=False, jpeg_quality=80)
    done = wf.run_workflow(record["workflow_dir"])
    assert done["state"] == "succeeded", done.get("error")
    photo = done["checkpoints"]["stack"]["data"]["colour"]["photo"]
    root = Path(done["delivery_dir"])
    jpeg = root / "stack_before_reduction.jpg"
    assert encoding(jpeg)[:2] == (tables(80), 0)
    rendered = json.loads((root / "photo_render.json").read_text(encoding="utf-8"))
    assert rendered["jpeg"] == (str(jpeg) if local else "stack_before_reduction.jpg")   # v1: relative, as before
    assert (rendered["jpeg_sha256"], rendered["jpeg_quality"], rendered["jpeg_subsampling"]) == \
        (wf.core._sha(jpeg), 80, "4:4:4")
    frozen = rendered["frozen_jpeg"]
    assert (frozen["jpeg_sha256"], frozen["jpeg_quality"], frozen["jpeg_subsampling"]) == \
        (photo["jpeg_sha256"], 95, "4:4:4")
    assert Path(frozen["jpeg"]) == Path(photo["jpeg"])
    if local:                   # everything else is the frozen record's (v1 makes the delivered paths relative)
        original = json.loads(Path(photo["record"]).read_text(encoding="utf-8"))
        assert {k: v for k, v in rendered.items() if not k.startswith("jpeg") and k != "frozen_jpeg"} == \
            {k: v for k, v in original.items() if not k.startswith("jpeg")}
    else:
        provenance = json.loads((root / "photo_render.original.json").read_text(encoding="utf-8"))
        assert provenance["jpeg_sha256"] == wf.core._sha(jpeg) and provenance["frozen_jpeg"] == frozen
        storage.verify_delivery(root)
    assert not list(Path(done["workflow_dir"]).glob("jpeg-*"))
    wf._verify(done)


def test_publication_encoder_reads_any_single_page_16_bit_rgb_tiff(tmp_path):
    """A legacy FITS display TIFF carries no display metadata; the encoder needs only 16-bit RGB (and its ICC)."""
    from PIL import ImageCms
    from starstack import display_transform as dt
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    codes = np.random.default_rng(4).integers(0, 65536, (70, 50, 3), dtype=np.uint16)
    codes[0, :3] = [[0, 0, 0], [128, 129, 32767], [65407, 65408, 65535]]       # rounding edges of code / 257
    source = tmp_path / "natural_display_16bit.tif"
    tifffile.imwrite(source, codes, photometric="rgb", iccprofile=icc)
    target = wf._encode_jpeg(source, tmp_path / "jpeg-x" / "final.jpg", 88)
    assert target.read_bytes() == expected_jpeg(source, 88)
    assert encoding(target) == (tables(88), 0, icc)
    with Image.open(target) as image:
        assert image.size == (50, 70)
    assert np.array_equal(((codes.astype(np.uint32) + 128) // 257).astype(np.uint8), dt.to_8bit(codes))
    tifffile.imwrite(tmp_path / "bytes.tif", codes.astype(np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="16 位 RGB"):
        wf._encode_jpeg(tmp_path / "bytes.tif", tmp_path / "jpeg-y" / "final.jpg", 88)


def test_cli_and_service_forward_jpeg_quality_only_when_given(setup, monkeypatch, capsys):
    from starstack.mcp_server import WorkspaceTools
    from starstack import cli
    service = WorkspaceTools(setup["inputs"], setup["output"], setup["engine"])
    files = [str(setup["inputs"] / f"{i}.NEF") for i in range(3)]
    for bad in (59, 101, 95.5, True, "95"):
        with pytest.raises(ValueError, match="jpeg_quality"):
            service.submit_workflow(input_files=files, jpeg_quality=bad)
    assert service.submit_workflow(input_files=files, jpeg_quality=85)["request"]["jpeg_quality"] == 85
    assert "jpeg_quality" not in service.submit_workflow(input_files=files)["request"]
    called = {}
    monkeypatch.setattr(wf, "submit_workflow", lambda *a, **k: called.update(kwargs=k) or {"state": "queued"})
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--files", "one.NEF", "two.NEF", "three.NEF",
                                      "--jpeg-quality", "92"])
    assert cli.main() == 0
    assert called["kwargs"]["jpeg_quality"] == 92
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--files", "one.NEF", "two.NEF", "three.NEF"])
    assert cli.main() == 0
    assert "jpeg_quality" not in called["kwargs"]
    monkeypatch.setattr(sys, "argv", ["aester", "workflow", "--files", "one.NEF", "--jpeg-quality", "59"])
    with pytest.raises(SystemExit):
        cli.main()
    capsys.readouterr()
