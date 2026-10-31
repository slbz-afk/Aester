"""Bounded protection-intent integration; synthetic arrays and a controlled engine.

No meteor class truth, real RAW or external engine is used here. Geometry rules
were frozen in the S4 design review before these test results.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from starstack import core, cli, transient_stage as ts, transient_trails as tt
from starstack.mark_validation import validate_mark_values
import test_core_two_pass_reference as flow_fixture

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


MARK = {"x0": 30.0, "y0": 80.0, "x1": 230.0, "y1": 80.0, "half_width_px": 20.0}


def write_planes(tmp_path, name, data):
    from starstack.imaging import write_linear
    path = tmp_path / name
    write_linear(path, np.moveaxis(data[:, ::-1], 0, -1).copy(), "S4 bounded synthetic truth")
    return path


def segment(**changes):
    return {"status": "detected", **MARK, **changes}


def candidate(cid="auto", *, source="auto", **changes):
    result = {"id": cid, "stem": "light_00001", "source": source,
              "measurement": {"status": "measured", **MARK},
              "provenance": {"input": {"sha256": "same-exposure"}},
              "suggestion": {"class": "unknown"}, "label": None, "override": None}
    result.update(changes)
    return result


@pytest.mark.parametrize("value", [1, 0, "true", None, [], np.bool_(True)])
def test_protect_rejects_non_builtin_bool(value):
    with pytest.raises(ValueError, match="protect"):
        validate_mark_values({**MARK, "protect": value})


def test_old_mark_shape_and_new_flag_are_not_coerced():
    assert "protect" not in validate_mark_values(MARK)
    assert validate_mark_values({**MARK, "protect": False})["protect"] is False
    assert validate_mark_values({**MARK, "protect": True})["protect"] is True


def test_batch_reversed_mark_upgrades_without_duplicate_and_does_not_downgrade():
    inputs = [{"path": "a.fit"}]
    first = {"input": "a.fit", **MARK}
    reverse = {**first, "x0": MARK["x1"], "x1": MARK["x0"], "protect": True}
    result = core._normalise_marks([first, reverse, {**first, "protect": False}], inputs)
    assert len(result) == 1 and result[0]["protect"] is True


def test_source_mark_survives_nonidentity_grid_mapping(tmp_path):
    mark = {"stem": "light_00001", **MARK, "protect": True}
    h = np.array([[1.2, 0, 7], [0, 1.2, 11], [0, 0, 1]], float)
    grid = ts.mark_to_grid(mark, h, None)
    got = ts._candidate("light_00001", 1, {"source": "manual", "mark": grid,
                        "measurement": {"status": "no_signal"}, "detection": {}},
                        inputs=[{"path": "a.fit", "sha256": "source"}], work=tmp_path,
                        h=h, model=None, own_registration=False)
    assert got["mark"] == mark
    assert got["mark_grid"]["x0"] == 43 and got["mark_grid"]["y0"] == 107
    assert got["mark_mapping"] == "endpoints_only_width_not_transformed"


def test_legacy_grid_mark_upgrade_requires_exact_frozen_transform(monkeypatch, tmp_path):
    original = {"stem": "light_00001", **MARK}
    h = np.array([[1.2, 0, 7], [0, 1.2, 11], [0, 0, 1]], float)
    old_grid = ts.mark_to_grid(original, h, None)
    old_grid.pop("_source_mark")
    old_grid.pop("corridor_mapping")
    c = candidate(source="manual", mark=old_grid)
    record = {"marks": [dict(original)], "candidates": [c], "params": {"detection": {}}}
    monkeypatch.setattr(tt, "Planes", lambda *a: object())
    monkeypatch.setattr(ts, "_finish", lambda record, *a, **k: record)
    monkeypatch.setattr(tt, "detection_image", lambda *a, **k: pytest.fail("legacy intent upgrade remeasured"))
    ts.detect_marks(tmp_path, record, [{**original, "protect": True}], own_registration=False,
                    transforms={"light_00001": h}, model=None, inputs=[], persist=False)
    assert c["mark"] == {**original, "protect": True}
    assert c["mark_grid"]["x0"] == 43
    assert record["marks"][0]["protect"] is True


def test_unmatched_old_request_never_claims_successful_upgrade(monkeypatch, tmp_path):
    mark = {"stem": "light_00001", **MARK}
    record = {"marks": [dict(mark)], "candidates": [], "params": {"detection": {}}}
    monkeypatch.setattr(tt, "Planes", lambda *a: object())
    with pytest.raises(ValueError, match="对应候选"):
        ts.detect_marks(tmp_path, record, [{**mark, "protect": True}], own_registration=False,
                        transforms={"light_00001": np.eye(3)}, model=None, inputs=[], persist=False)
    assert "protect" not in record["marks"][0]


def test_short_requested_target_beats_longer_parallel_interference(monkeypatch):
    calls = []
    monkeypatch.setattr(tt, "measure_segment", lambda *args, **kw: calls.append(args[2:]) or {"status": "measured"})
    long_wrong = segment(x0=0, x1=300, y0=95, y1=95)
    target = segment(x0=35, x1=225)
    selected, _ = tt.measure_mark(None, None, MARK, [long_wrong, target], params={})
    assert selected["x0"] == 35 and calls == [(35, 80.0, 225, 80.0)]


@pytest.mark.parametrize("segments,reason", [
    ([segment(y0=78, y1=78), segment(y0=82, y1=82)], "multiple_geometric_matches"),
    ([segment(y0=90, y1=90)], "no_geometric_match"),
    ([segment(x0=120, y0=0, x1=140, y1=160)], "no_geometric_match"),
])
def test_ambiguous_or_wrong_target_never_measured(monkeypatch, segments, reason):
    monkeypatch.setattr(tt, "measure_segment", lambda *a, **k: pytest.fail("ambiguous target reached measurement"))
    selected, measured = tt.measure_mark(None, None, MARK, segments, params={})
    assert selected["status"] == "mark_unresolved"
    assert measured == {"status": "ambiguous_target", "reason": reason, "clipping_evidence": "unknown"}


def test_no_detection_uses_user_centerline_with_no_extension(monkeypatch):
    seen = {}
    def fake(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "no_signal"}
    monkeypatch.setattr(tt, "measure_segment", fake)
    seg, _ = tt.measure_mark(None, None, MARK, [], params={})
    assert seg["status"] == "mark_only" and seen["search_beyond_px"] == 0
    assert seen["x0"] == MARK["x0"] and seen["y1"] == MARK["y1"]


def test_relation_reversal_and_boundary_are_fixed():
    reverse = segment(x0=230, x1=30)
    assert tt.marked_line_relation(MARK, reverse)["matches"]
    assert tt.marked_line_relation(MARK, segment(y0=83, y1=83))["matches"]
    assert not tt.marked_line_relation(MARK, segment(y0=83.00001, y1=83.00001))["matches"]
    assert not tt.marked_line_relation(MARK, segment(x0=231, x1=431))["matches"]


def test_related_pairs_are_not_transitive_identity_or_decision_merges():
    a = candidate(requested_decision="keep", label="meteor")
    b = candidate("manual", source="manual", requested_decision="omit", label="satellite")
    c = candidate("other-exposure", stem="light_00002", requested_decision="keep")
    d = candidate("different-source-hash", source="manual", requested_decision="keep", provenance={"input": {"sha256": "other"}})
    record = {"candidates": [d, b, a, c], "grid": "native"}
    ts.related_candidates(record)
    assert len(record["related_candidate_groups"]) == 1
    group = record["related_candidate_groups"][0]
    assert group["members"] == ["auto", "manual"] and group["identity_status"] == "unconfirmed_no_merge"
    assert len(group["conflicts"]) == 2
    assert a["label"] == "meteor" and b["label"] == "satellite" and b["requested_decision"] == "omit"
    assert c["related_candidate_ids"] == [] and d["related_candidate_ids"] == []


def test_unmeasured_explicit_request_remains_in_compact_view():
    c = candidate("manual", source="manual", mark={**MARK, "protect": True}, measurement={"status": "no_signal"})
    record = {"candidates": [c]}
    tt.resolve_decisions(record)
    ts.reset_application(record)
    view = cli.candidate_view(c)
    assert view["requested_decision"] == "keep" and view["decision"] == "omit"
    assert view["protection_status"] == "needs_review" and view["application"]["status"] == "not_applied"
    assert c["label"] is None and c["effective_label"] == "unknown"


def test_layer_support_evidence_and_final_weight_do_not_claim_full_recovery(tmp_path):
    frame = np.full((3, 180, 280), .3, np.float32)
    master = np.full_like(frame, .2)
    frame[:, 79:82, 90:95] = 0
    c = candidate("manual", source="manual", mark={**MARK, "protect": True})
    record = {"candidates": [c]}
    tt.resolve_decisions(record)
    ts.reset_application(record)
    box = tt.layer_box(write_planes(tmp_path, "frame.fit", frame), write_planes(tmp_path, "master.fit", master), c, [1]*3, [0]*3, [0]*3)
    c["application"].update(box["application_evidence"])
    assert c["application"]["grid_zero_or_nonfinite_pixels"] > 0
    box["stem"] = c["stem"]
    weight = np.ones((180, 280), np.float32)
    weight[:, 150:] = 0
    ts.finish_application(record, [box], weight=weight, native_warped=True, composite_path="separate.fit")
    assert c["application"]["status"] == "partial"
    assert c["application"]["contribution_observed"] is True
    assert "per_exposure_native_kernel_support_unverified" in c["application"]["reasons"]
    assert c["application"]["complete_trail_recovery"] == "unverified"
    assert record["application_by_source"][c["stem"]]["fully_suppressed_box_samples"] > 0


def test_zero_final_weight_reports_no_application():
    c = candidate(mark={**MARK, "protect": True})
    record = {"candidates": [c]}
    tt.resolve_decisions(record)
    ts.reset_application(record)
    c["application"].update(grid_nonzero_residual_pixels=4)
    box = {"stem": c["stem"], "x0": 0, "y0": 0, "data": np.ones((3, 2, 2), np.float32)}
    ts.finish_application(record, [box], weight=np.zeros((2, 2), np.float32))
    assert c["application"]["status"] == "not_applied" and not c["application"]["contribution_observed"]


def test_same_source_union_never_claims_individual_alias_contribution():
    a, b = candidate("a", mark={**MARK, "protect": True}), candidate("b", mark={**MARK, "protect": True})
    record = {"candidates": [a, b]}
    tt.resolve_decisions(record)
    ts.reset_application(record)
    for c in (a, b):
        c["application"].update(grid_nonzero_residual_pixels=4)
    box = {"stem": a["stem"], "x0": 0, "y0": 0, "data": np.ones((3, 2, 2), np.float32)}
    ts.finish_application(record, [box])
    assert all(c["application"]["contribution_observed"] is None for c in (a, b))
    assert all(c["application"]["source_union_contribution_observed"] for c in (a, b))


def test_warp_support_propagates_cancel_callback():
    box = {"x0": 0, "y0": 0, "data": np.ones((3, 2, 2), np.float32),
           "validity": np.ones((2, 2), bool), "grid_shape": (2, 2)}
    def cancel():
        raise RuntimeError("cancelled support")
    with pytest.raises(RuntimeError, match="cancelled support"):
        tt.warp_box_to_native(box, None, (2, 2), check=cancel)


def test_detector_overload_is_not_sensor_clipping():
    record = {"candidates": [], "per_frame": {"f": {"saturated": True}}}
    summary = ts.summary(record)
    assert summary["detector_budget_exceeded_frames"] == ["f"]
    assert "not_sensor_clipping" in summary["saturated_frames_semantics"]
    detector = {"bin": 1, "snr": np.full((12, 15), 10.0), "valid": np.ones((12, 15), bool)}
    result = tt.find_segments(detector, params={"max_foreground_pixels": 100})
    assert result == [{"status": "saturated", "reason": "detector_budget_exceeded", "clipping_evidence": "unknown", "foreground_pixels": 180}]


@pytest.mark.parametrize("sigma,expect_limited", [(2, False), (22, True)])
def test_width_limit_flag_on_controlled_injection(tmp_path, sigma, expect_limited):
    yy, xx = np.mgrid[:220, :320]
    rng = np.random.default_rng(90210)
    master = np.full((3, 220, 320), .1, np.float32)
    signal = .08 * np.exp(-((yy - 110) / sigma)**2 / 2) * ((xx >= 40) & (xx <= 280))
    frame = (master + signal[None] + rng.normal(0, .00015, master.shape)).astype(np.float32)
    measured = tt.measure_segment(write_planes(tmp_path, "frame.fit", frame), write_planes(tmp_path, "master.fit", master),
                                  40, 110, 280, 110, search_beyond_px=0)
    assert measured["status"] == "measured"
    assert measured["quality"]["width_limit_reached"] is expect_limited
    assert measured["clipping_evidence"] == "unknown"


@pytest.fixture
def synthetic_job():
    fixture = flow_fixture.TransientFlowTest()
    fixture.setUp()
    # The older flow fixture deliberately supplies nonidentity diagnostic H /
    # distortion but its fake engine copies pixels unchanged. This protection
    # test needs actual geometry truth, so both declarations must be identity.
    from starstack import registration_calibration as rc
    old_calibrate, old_script = rc.calibrate, core.run_script
    def identity_calibration(work, **kwargs):
        result = old_calibrate(work, **kwargs)
        result["parameters"]["k1"] = result["parameters"]["k2"] = 0.0
        for frame in result["registration"]["frames"].values():
            frame["H_top_down_magnified"] = np.eye(3).tolist()
            frame["H_siril"] = np.eye(3).tolist()
        (work / rc.DISTORTION_JSON).write_text(json.dumps(result), encoding="utf-8")
        return result
    def identity_script(executable, directory, name, commands, check):
        result = old_script(executable, directory, name, commands, check)
        if name == "02_export":   # P1b-3: the export process writes the applied sequence
            sequence = directory / "work" / core.SEQUENCE_FILE
            lines = sequence.read_text(encoding="utf-8").splitlines()
            sequence.write_text("\n".join(line.split(" H ")[0] + " H 1 0 0 0 1 0 0 0 1" if " H " in line else line for line in lines) + "\n", encoding="utf-8")
        return result
    rc.calibrate, core.run_script = identity_calibration, identity_script
    try:
        job = core.submit_job(str(fixture.source), str(fixture.output))
        core.run_job(job["job_dir"])
        job = core.load_job(job["job_dir"])
        assert job["state"] == "succeeded", job.get("error")
        yield fixture, job
    finally:
        fixture.tearDown()
        fixture.doCleanups()


def test_upgrade_old_mark_protect_without_remeasurement_and_override_retraction(synthetic_job, monkeypatch):
    fixture, job = synthetic_job
    work = Path(job["job_dir"]) / "work"
    base = fixture._composites(job)[0].copy()
    mark = {"input": "原片 3.fit", "x0": 40, "y0": 150, "x1": 260, "y1": 40}
    initial = core.recompose_transients(job["job_dir"], marks=[mark], policy={key: "omit" for key in tt.CLASSES})
    before = ts.read_record(work)
    manual = next(c for c in before["candidates"] if c["source"] == "manual")
    assert manual["decision"] == "omit"
    assert manual["measurement"]["status"] == "measured"
    monkeypatch.setattr(tt, "detection_image", lambda *a, **k: pytest.fail("protection upgrade repeated detection"))
    monkeypatch.setattr(tt, "measure_segment", lambda *a, **k: pytest.fail("protection upgrade repeated measurement"))
    reverse = {**mark, "x0": 260, "y0": 40, "x1": 40, "y1": 150, "protect": True}
    updated = core.recompose_transients(job["job_dir"], marks=[reverse], labels={manual["id"]: "unknown"})
    record = ts.read_record(work)
    protected = next(c for c in record["candidates"] if c["id"] == manual["id"])
    assert protected["requested_reason"] == "mark_protect" and protected["decision"] == "keep"
    assert protected["effective_label"] == "unknown" and protected["application"]["contribution_observed"]
    assert protected["application"]["status"] in {"partial", "unknown"}
    assert len(record["marks"]) == 1 and len(record["candidates"]) == len(before["candidates"])
    assert record["related_candidate_groups"] and protected["relation_conflicts"]
    repeated = core.recompose_transients(job["job_dir"], marks=[reverse])
    assert repeated["transient_revision"] == updated["transient_revision"]
    false_search = core.recompose_transients(job["job_dir"], marks=[{**mark, "protect": False}])
    assert false_search["transient_revision"] == updated["transient_revision"]
    omitted = core.recompose_transients(job["job_dir"], overrides={manual["id"]: "omit"})
    last = ts.read_record(work)
    requested = next(c for c in last["candidates"] if c["id"] == manual["id"])
    assert requested["requested_decision"] == "omit" and requested["application"]["status"] == "not_requested"
    assert not any(a.get("group") == "transients" for a in omitted["artifacts"])
    np.testing.assert_array_equal(fixture._composites(omitted)[0], base)


def test_cli_protect_requires_a_mark():
    args = SimpleNamespace(label=None, override=None, policy=None, mark=None, note=None, protect=True)
    with pytest.raises(ValueError, match="--protect"):
        cli._transients(args)


def test_cli_protect_passes_explicit_flag_without_labelling(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "recompose_transients", lambda *a, **k: calls.append(k) or {"job_dir": str(tmp_path)})
    monkeypatch.setattr(ts, "read_record", lambda *a, **k: None)
    args = SimpleNamespace(job=str(tmp_path), label=None, override=None, policy=None,
                           mark=[["source.fit", "1", "2", "30", "40"]], note=None, protect=True)
    cli._transients(args)
    assert calls[0]["marks"] == [{"input": "source.fit", "x0": 1., "y0": 2., "x1": 30., "y1": 40., "protect": True}]
    assert calls[0]["labels"] is None


@pytest.mark.parametrize("protect", [True, False, 1, "true", None])
def test_mcp_mark_protection_boundary(monkeypatch, tmp_path, protect):
    from starstack.mcp_server import WorkspaceTools
    service = object.__new__(WorkspaceTools)
    service.output_root = tmp_path.parent
    monkeypatch.setattr(service, "_job", lambda *a: (tmp_path, {}))
    monkeypatch.setattr(service, "_artifacts", lambda *a: [])
    monkeypatch.setattr(service, "list_transients", lambda *a: {"ok": True})
    calls = []
    monkeypatch.setattr(core, "recompose_transients", lambda *a, **k: calls.append(k))
    mark = {"input": "source.fit", **MARK, "protect": protect}
    if type(protect) is bool:
        assert service.set_transients(str(tmp_path), marks=[mark]) == {"ok": True}
        assert calls[0]["marks"][0]["protect"] is protect
    else:
        with pytest.raises(ValueError, match="protect"):
            service.set_transients(str(tmp_path), marks=[mark])
        assert not calls
