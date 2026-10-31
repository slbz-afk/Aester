"""Engine helpers the desktop app uses instead of private symbols or record parsing (open/closed split, S1)."""
from pathlib import Path
import json

import pytest

from starstack import cli, onboarding, photographic_job as pj, trail_review_session as trs, transient_review, workflow_edit


def _old_gui_record(source, mask, version):
    """The record the desktop panel wrote itself before the split (advanced_tools.run_milkyway at 4afcd3f)."""
    from starstack import core
    return {"version": version, "status": "present", "method": "human-review",
            "reviewer": "GUI user", "rationale": "用户在银河增强窗口显式确认所选源图的银河选区。",
            "source_sha256": core._sha(Path(source)), "milkyway_mask_sha256": core._sha(Path(mask))}


def test_milkyway_review_record_is_byte_identical_to_the_old_gui_record(tmp_path):
    from starstack.milkyway_gate import VERSION
    source, mask = tmp_path / "s.tif", tmp_path / "m.png"
    source.write_bytes(b"source"), mask.write_bytes(b"mask")
    confirmation = pj.milkyway_confirmation(str(source), str(mask))
    assert confirmation == {"source_path": str(source.resolve()), "mask_path": str(mask.resolve()),
                            "source_sha256": _old_gui_record(source, mask, VERSION)["source_sha256"],
                            "milkyway_mask_sha256": _old_gui_record(source, mask, VERSION)["milkyway_mask_sha256"]}
    directory = tmp_path / "out" / "gui-milkyway-review-0123456789ab"
    path = pj.write_milkyway_review(confirmation, str(source), str(mask), directory,
                                    reviewer="GUI user", rationale="用户在银河增强窗口显式确认所选源图的银河选区。")
    assert Path(path) == directory / "evidence.json"
    old = tmp_path / "old-evidence.json"   # written exactly as the pre-split panel did (platform newlines included)
    old.write_text(json.dumps(_old_gui_record(source, mask, VERSION), ensure_ascii=False, indent=2), encoding="utf8")
    assert Path(path).read_bytes() == old.read_bytes()


@pytest.mark.parametrize("change", ["source", "mask", "move"])
def test_milkyway_review_refuses_a_changed_or_moved_file(tmp_path, change):
    source, mask = tmp_path / "s.tif", tmp_path / "m.png"
    source.write_bytes(b"source"), mask.write_bytes(b"mask")
    confirmation = pj.milkyway_confirmation(source, mask)
    if change == "source":
        source.write_bytes(b"edited")
    elif change == "mask":
        mask.write_bytes(b"edited")
    else:
        moved = tmp_path / "other.tif"
        moved.write_bytes(b"source")
        source = moved
    with pytest.raises(pj.MilkywayConfirmationChanged):
        pj.write_milkyway_review(confirmation, source, mask, tmp_path / "review", reviewer="r", rationale="x")
    assert not (tmp_path / "review").exists()
    assert issubclass(pj.MilkywayConfirmationChanged, ValueError)


def test_list_trail_candidates_reads_through_patchable_engine_functions(tmp_path, monkeypatch):
    from starstack import core, transient_stage
    job = {"job_dir": str(tmp_path), "artifacts": [{"group": "transients", "path": "a"}, {"label": "含亮迹成片", "path": "b"},
                                                    {"group": "other", "path": "c"}]}
    candidate = {"id": "c1", "stem": "frame00001", "measurement": {"status": "measured"}, "decision": "omit"}
    monkeypatch.setattr(core, "load_job", lambda directory: job)
    monkeypatch.setattr(transient_stage, "read_record", lambda work, job=None: {"candidates": [candidate]})
    result = trs.list_trail_candidates(tmp_path)
    assert result["state"] == "succeeded" and result["job_dir"] == str(tmp_path.resolve())
    assert [a["path"] for a in result["artifacts"]] == ["a", "b"]
    assert result["candidates"] == [transient_review.candidate_view(candidate)]


def _workflow(tmp_path, state="succeeded", *, snapshot=True, exists=True, frozen_index=False):
    directory = tmp_path / "workflow-1"
    child = directory / "stack" / "job-1"
    child.mkdir(parents=True)
    snap = directory / "stack_snapshot-1.json"
    if exists:
        snap.write_text("{}", encoding="utf8")
    files = [{"path": str(snap)}] if snapshot else [{"path": str(directory / "job.json")}]
    data = {"job_snapshot": str(directory / "elsewhere.json") if frozen_index else str(snap)}
    (directory / "workflow.json").write_text(json.dumps({"state": state, "checkpoints": {"stack": {"files": files, "data": data}}}),
                                             encoding="utf8")
    return child


def test_trail_rederive_scope_reason_codes(tmp_path):
    assert workflow_edit.trail_rederive_scope(tmp_path / "loose") is None
    assert workflow_edit.trail_rederive_scope(_workflow(tmp_path / "a")) is None
    assert workflow_edit.trail_rederive_scope(_workflow(tmp_path / "b", "running")) == "workflow_not_succeeded"
    assert workflow_edit.trail_rederive_scope(_workflow(tmp_path / "c", snapshot=False)) == "frozen_job_index"
    assert workflow_edit.trail_rederive_scope(_workflow(tmp_path / "d", frozen_index=True)) == "frozen_job_index"
    assert workflow_edit.trail_rederive_scope(_workflow(tmp_path / "e", exists=False)) == "snapshot_missing"


def test_candidate_view_lives_in_transient_review_and_cli_reexports_it():
    assert cli.candidate_view is transient_review.candidate_view
    assert onboarding.checked_directory is onboarding._directory
