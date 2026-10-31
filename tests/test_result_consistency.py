"""Independent signed-pixel, publication and fallback counterexamples.

Small controlled FITS scenes test product integration, not astronomical
classification accuracy. No RAW data or historical outputs are touched.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from starstack import core, imaging, transient_stage as ts, transient_trails as tt
from starstack import transient_revision as revisions, sky_validity as sv
from starstack.transient_union import union_boxes
from test_core_two_pass_reference import TransientFlowTest as _Fixture, _seq_text

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


@pytest.fixture
def scene():
    fixture = _Fixture()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()
        fixture.doCleanups()


def _finish(scene):
    job = core.submit_job(str(scene.source), str(scene.output))
    core.run_job(job["job_dir"])
    job = core.load_job(job["job_dir"])
    assert job["state"] == "succeeded", job.get("error")
    assert job["transient_revision"], {
        'diagnostics': job.get('diagnostics', {}).get('transients'),
        'worker_log': (Path(job['job_dir']) / 'worker.log').read_text(encoding='utf-8', errors='replace')
        if (Path(job['job_dir']) / 'worker.log').is_file() else None,
    }
    return job, Path(job["job_dir"]) / "work"


def _published_bytes(job):
    paths = [Path(job["job_dir"]) / "job.json", Path(job["transient_revision"]["record"])]
    paths += [Path(a["path"]) for a in job["artifacts"]]
    record = ts.read_record(Path(job["job_dir"]) / "work", job=job)
    paths += [Path(c["review_png"]) for c in record["candidates"] if c.get("review_png")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_union_chooses_weight_not_brightness_and_keeps_separate_exposures():
    residual = np.array([[-3, -1, 2, 5], [4, -2, 1, -6]], dtype=np.float32)
    wa = np.array([[1, .25, 0, 0], [1, .5, 0, 0]])
    wb = np.array([[0, .5, 1, 1], [0, 1, .25, .5]])
    def box(w):
        return {"x0": 1, "y0": 2, "weight": w, "data": np.stack([residual*w]*3).astype(np.float32)}
    boxes = union_boxes([box(wa), box(wb), box(wa)])
    expected = np.stack([residual*np.maximum(wa, wb)]*3).astype(np.float32)
    assert len(boxes) == 1
    np.testing.assert_array_equal(boxes[0]["data"], expected)
    target = np.zeros((6, 8, 3), dtype=np.float32)
    # Separate exposures remain separate, even at identical coordinates.
    tt.apply_boxes(target, [boxes[0], boxes[0]])
    np.testing.assert_array_equal(target[::-1][2:4, 1:5], np.moveaxis(expected*2, 0, -1))


def test_union_keeps_distant_boxes_sparse_and_merges_chain():
    def box(x):
        return {"x0": x, "y0": 0, "weight": np.ones((2, 4)), "data": np.ones((3, 2, 4), np.float32)}
    assert len(union_boxes([box(0), box(100000)])) == 2
    result = union_boxes([box(0), box(6), box(3)])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0]["data"], np.ones((3, 2, 10), np.float32))


def test_duplicate_candidate_has_exactly_one_source_contribution(scene):
    job, work = _finish(scene)
    record = ts.read_record(work)
    shape = (scene.HEIGHT, scene.WIDTH)
    single, _ = ts.native_boxes(work, record, own_registration=False, model=None, native_shape=shape)
    duplicate = copy.deepcopy(record["candidates"][0])
    duplicate["id"] += "-manual-copy"
    duplicate["source"] = "manual"
    record["candidates"].append(duplicate)
    multiple, _ = ts.native_boxes(work, record, own_registration=False, model=None, native_shape=shape)
    a, b = (np.zeros((*shape, 3), dtype=np.float32) for _ in range(2))
    np.testing.assert_array_equal(tt.apply_boxes(a, single), tt.apply_boxes(b, multiple))


def test_auto_then_manual_and_repeated_reversed_mark_do_not_double_trail(scene):
    job, work = _finish(scene)
    base, original, _ = scene._composites(job)
    mark = {"input": "原片 3.fit", "x0": 40, "y0": 150, "x1": 260, "y1": 40}
    updated = core.recompose_transients(job["job_dir"], marks=[mark], policy={"unknown": "keep"})
    record = ts.read_record(work)
    assert len(record["marks"]) == 1
    # A corridor can alter the feather boundary; bright-core pixels already
    # contributed by auto detection must be unchanged, not doubled.
    _, result, _ = scene._composites(updated)
    core_mask = scene.trail > .01
    np.testing.assert_array_equal((result-base)[::-1][core_mask], (original-base)[::-1][core_mask])
    reverse = {"stem": "light_00003", "x0": 260, "y0": 40, "x1": 40, "y1": 150}
    again = core.recompose_transients(job["job_dir"], marks=[reverse])
    assert again["transient_revision"] == updated["transient_revision"]
    again_record = ts.read_record(work)
    assert len(again_record["candidates"]) == len(record["candidates"])
    assert len(again_record["marks"]) == 1
    np.testing.assert_array_equal(scene._composites(again)[1], result)


@pytest.mark.parametrize("failure", ["export", "record", "commit"])
def test_failure_keeps_entire_old_publication(scene, monkeypatch, failure):
    job, work = _finish(scene)
    old = _published_bytes(job)
    candidate = ts.read_record(work)["candidates"][0]["id"]
    def fail(*args, **kwargs):
        raise OSError("injected disk/publication failure")
    if failure == "export":
        monkeypatch.setattr(imaging, "export_products", fail)
    elif failure == "record":
        monkeypatch.setattr(revisions, "prepare", fail)
    else:
        monkeypatch.setattr(core, "atomic_json", fail)
    with pytest.raises(OSError, match="injected"):
        core.recompose_transients(job["job_dir"], labels={candidate: "satellite"}, policy={"satellite": "keep"},
                                  marks=[{"input": "原片 2.fit", "x0": 50, "y0": 100, "x1": 150, "y1": 120}])
    assert _published_bytes(core.load_job(job["job_dir"])) == old
    assert ts.read_record(work)["candidates"][0]["label"] is None


def test_reader_pins_old_revision_and_omit_is_a_real_publication(scene):
    job, work = _finish(scene)
    old = ts.read_record(work, job=job)
    cid = old["candidates"][0]["id"]
    updated = core.recompose_transients(job["job_dir"], overrides={cid: "omit"})
    assert updated["transient_revision"]["id"] != job["transient_revision"]["id"]
    assert ts.read_record(work, job=job) == old
    assert ts.read_record(work)["candidates"][0]["decision"] == "omit"
    assert not [a for a in updated["artifacts"] if a.get("group") == "transients"]
    assert updated["diagnostics"]["transients"]["record"] == updated["transient_revision"]["record"]


def test_unpublished_checkpoint_is_hidden_and_legacy_is_readable(tmp_path):
    work = tmp_path / "work"
    tt.write_candidates(work / "transients", {"candidates": []})
    assert ts.read_record(work, job={}) == {"candidates": []}
    assert ts.read_record(work, job={"transient_revision": None}) is None


def test_first_export_hides_checkpoint_until_commit(scene, monkeypatch):
    original = imaging.export_products
    seen = []
    def export(data, directory, check):
        job_dir = Path(directory)
        while not (job_dir / "job.json").is_file():
            job_dir = job_dir.parent
        job = core.load_job(job_dir)
        assert job["transient_revision"] is None
        assert (job_dir / "work" / "transients" / tt.CANDIDATES_FILE).is_file()
        assert ts.read_record(job_dir / "work", job=job) is None
        seen.append(str(directory))
        return original(data, directory, check)
    monkeypatch.setattr(imaging, "export_products", export)
    job, work = _finish(scene)
    assert len(seen) >= 2
    assert ts.read_record(work)["revision_id"] == job["transient_revision"]["id"]


def test_detection_checkpoint_failure_does_not_publish_leftover_record(scene, monkeypatch):
    original = core._update
    def update(directory, **values):
        if "transients" in values.get("checkpoints", []):
            raise OSError("injected checkpoint failure")
        return original(directory, **values)
    monkeypatch.setattr(core, "_update", update)
    job = core.submit_job(str(scene.source), str(scene.output))
    core.run_job(job["job_dir"])
    job = core.load_job(job["job_dir"])
    work = Path(job["job_dir"]) / "work"
    assert job["state"] == "succeeded"  # base products survive a candidate-stage failure
    assert job["diagnostics"]["transients"]["status"] == "failed"
    assert (work / "transients" / tt.CANDIDATES_FILE).is_file()
    assert ts.read_record(work) is None


def test_missing_validity_does_not_enable_legacy_composition(tmp_path):
    for job in ({"sky_validity_required": True}, {"diagnostics": {"sky_validity": {"applied": True}}}):
        with pytest.raises(ValueError, match="记录缺失"):
            core._read_sky_validity(tmp_path, job)
    assert core._read_sky_validity(tmp_path, {}) is None


def test_corrupt_revision_or_escaped_pointer_never_falls_back(tmp_path):
    work = tmp_path / "work"
    tt.write_candidates(work / "transients", {"candidates": ["stale"]})
    export = tmp_path / "products" / "r1"
    pointer = revisions.prepare(export, {"candidates": []})
    job = {"transient_revision": pointer}
    assert ts.read_record(work, job=job)["candidates"] == []
    Path(pointer["record"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="校验"):
        ts.read_record(work, job=job)
    pointer["record"] = str(work / "transients" / tt.CANDIDATES_FILE)
    with pytest.raises(ValueError, match="超出"):
        ts.read_record(work, job=job)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 10**1000, -1, 320])
def test_bad_marks_fail_before_revision_or_geometry(scene, value, monkeypatch):
    job, work = _finish(scene)
    old = _published_bytes(job)
    monkeypatch.setattr(ts, "mark_to_grid", lambda *args: pytest.fail("invalid source bounds reached mapping"))
    with pytest.raises(ValueError):
        core.recompose_transients(job["job_dir"], marks=[{"stem": "light_00003", "x0": value, "y0": 10, "x1": 20, "y1": 15}])
    assert _published_bytes(core.load_job(job["job_dir"])) == old


@pytest.mark.parametrize("h,own", [(None, False), (np.zeros((3, 3)), False), (np.eye(3), True)])
def test_missing_or_invalid_transform_does_not_invent_source_coordinates(h, own):
    result = ts._native_endpoints({"x0": 10, "y0": 20, "x1": 30, "y1": 40}, h, None, own_registration=own)
    assert result["native_source"] is None
    assert result["native_source_status"] == "unavailable"
    assert result["grid"] == [[10., 20.], [30., 40.]]
    if own:
        assert result["native_reference"] is None


def test_validity_atomic_failure_and_corruption_are_explicit(tmp_path, monkeypatch):
    from starstack import records
    path = sv.write_record(tmp_path, {"dilation_px": 3})
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(records.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            sv.write_record(tmp_path, {"dilation_px": 5})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    path.write_text('{"dilation_px":', encoding="utf-8")
    with pytest.raises(ValueError, match="天空有效区记录损坏"):
        sv.read_record(tmp_path)


@pytest.mark.parametrize("status", ["no_improvement", "unavailable", "fitted"])
def test_fallback_exclusion_survives_register_and_stale_exports(scene, monkeypatch, status):
    from starstack import registration_calibration as rc
    old_engine = core.run_script
    seen = {}
    def calibration(work, **kwargs):
        record = {"status": status, "reason": "controlled fallback", "quality_gate": {
            "applied": True, "excluded_frames": ["light_00005"],
            "excluded": [{"stem": "light_00005", "reasons": ["controlled blurred frame"]}]}}
        core.atomic_json(work / rc.DISTORTION_JSON, record)
        return record
    def engine(exe, directory, name, commands, check):
        work = directory / "work"
        if name == "02_probe":
            # A prior partial attempt left an export which must be removed.
            shutil.copyfile(work / "light_00005.fit", work / "r_light_00005.fit")
        elif name == "02_register_fallback":
            (work / core.SEQUENCE_FILE).write_text(_seq_text(5, 1), encoding="utf-8")
        elif name == "02_export":   # P1b-3: the export process applies the product's selection
            assert commands == ["cd work", "seqapplyreg light -filter-included"]
            seq = core.read_seq(work / core.SEQUENCE_FILE)
            selected = [f["stem"] for f in seq["images"] if f["selected"]]
            seen["selected"] = selected
            assert selected == [f"light_{i:05d}" for i in range(1, 5)]
            assert not (work / "r_light_00005.fit").exists()
            for stem in selected:
                shutil.copyfile(work / f"{stem}.fit", work / f"r_{stem}.fit")
        elif name == "02_stack":   # P1b-3: the stack process
            assert commands == list(core.LEGACY_STACK_SCRIPT)
            selected = seen["selected"]
            master = np.median([imaging.read_rgb(work / f"{s}.fit") for s in selected], axis=0)
            imaging.write_linear(work / "sky_master.fit", master, "Controlled selected stack", extra={"STACKCNT": 4})
        else:
            old_engine(exe, directory, name, commands, check)
    # Undone before the scene fixture's tearDown restores the real functions: a plain monkeypatch.setattr would be
    # undone after it and reinstall the scene's fake core.run_script for later test modules.
    with monkeypatch.context() as patch:
        patch.setattr(rc, "calibrate", calibration)
        patch.setattr(core, "run_script", engine)
        job, work = _finish(scene)
    assert job["diagnostics"]["quality_gate"]["in_effect"]
    assert job["diagnostics"]["stacked_count"] == 4
    assert ts.read_record(work)["frame_order"] == seen["selected"]
    transforms, _ = core.registration_transforms(work, calibration(work), False, scene.HEIGHT)
    assert "light_00005" not in transforms
