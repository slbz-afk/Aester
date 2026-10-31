"""In-app per-job cache cleanup (REL-CACHE option A): synthetic workflows, real files and real locks.

The stack job and Siril are the synthetic ones of tests/test_workflow.py; the per-frame intermediates are
written by the test with the names the stack job uses. No photo job runs.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from starstack import cache_cleanup as cc, core, hashing, i18n, storage, uninstall_options as u, workflow as wf
from starstack import trail_review_session as review, workflow_edit as edit
from starstack.i18n import translate_message
from starstack.locking import file_lock, is_locked
from test_workflow import setup  # noqa: F401  (fixture)
from test_storage import run_external, admit
from test_delivery_images import completed  # noqa: F401  (fixture)
from test_uninstall_options import registry  # noqa: F401  (fixture)

CJK = re.compile(r'[　-〿㐀-䶿一-鿿豈-﫿＀-￯]')
# Files of the stack job that the cleanup removes, by plan group (paths relative to the stack job folder).
FRAMES = {
    "converted_frames": ["work/light_00001.fit", "work/light_00002.fit", "work/light_00003.fit"],
    "registered_frames": ["work/r_light_00001.fit", "work/r_light_00002.fit", "work/r_light_00003.fit"],
    "sequences": ["work/light_.seq", "work/light_.initial.seq", "work/r_light_.seq", "work/light_.support-export.seq"],
    "stage_copies": ["stage/frame00001.nef", "stage/frame00002.nef", "stage/frame00003.nef"],
    "range_trials": ["range-0123456789ab/trial1/r_light_00001.fit", "range-0123456789ab/trial1/encoded_master.fit"],
    "finite_frames": ["finite-0123456789ab/q_00001.fit", "finite-0123456789ab/q_00002.fit"],
    "frame_support": ["work/frame_support_light_00001.npy", "work/frame_support_light_00002.npy"],
    "cfa_frames": ["cfa-0123456789ab/float/frame00001.fit", "cfa-0123456789ab/decoded/light_00001.fit",
                   "cfa-0123456789ab/canonical/light_00001.fit"],
}
# Files beside them that are kept: masters, records, review evidence, logs, previews and unknown files.
KEPT = ["work/sky_master.fit", "work/sky_master_undistorted.fit", "work/foreground_master.fit", "work/sky_coverage.png",
        "work/sky_validity.json", "work/distortion.json", "work/cache/light_00001.lst", "work/light_conversion.txt",
        "work/transients/candidates.json", "work/transients/crops/light_00001-1.png", "work/light_00001.fit.tmp",
        "work/finite_master_nvalid.fit", "work/sky_support_stack.npy", "02_stack.log", "worker.log", "my_notes.txt",
        "range-0123456789ab/decoded.fit", "range-0123456789ab/receipt.json", "range-0123456789ab/trial1/r_light_.seq",
        "finite-0123456789ab/finite_master.fit", "finite-0123456789ab/q_.seq", "finite-0123456789ab/receipt.json",
        "preview-variations/0a1b/recipe.json", "frame_masks/light_00001.png", "stage-notes/frame00001.nef",
        "cfa-0123456789ab/conversion.json", "cfa-0123456789ab/debayer.log", "cfa-0123456789ab/decoded/light_.seq",
        "products/0123456789ab/master_linear.fit"]
ALL_FRAMES = [path for paths in FRAMES.values() for path in paths]


def populate(child):
    for relative in ALL_FRAMES + KEPT:
        path = child / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode() * 37)


def snapshot(root, *, skip_locks=False):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file() and not (skip_locks and p.suffix == ".lock")}


def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


@pytest.fixture
def finished(setup):
    done, works = run_external(setup, idempotency_key="cache-clean")
    directory, child = Path(done["workflow_dir"]), Path(done["stack_job_dir"])
    populate(child)
    layers = directory / "finishing" / "layers"
    layers.mkdir(parents=True, exist_ok=True)
    (layers / "00_background.tif").write_bytes(b"workflow-level layer kept for star re-runs")
    return {"done": done, "works": works, "directory": directory, "child": child, "setup": setup,
            "delivery": Path(done["delivery_dir"])}


def frames_exist(child):
    return [relative for relative in ALL_FRAMES if (child / relative).exists()]


def child_gc(child):
    return json.loads((child / "job.json").read_text(encoding="utf-8")).get("gc")


# ---------------------------------------------------------------------------------------------------- planning

def test_inspect_reports_groups_and_bytes_and_changes_nothing(finished):
    directory, child, works = finished["directory"], finished["child"], finished["works"]
    before, works_before = snapshot(directory), snapshot(works)
    plan = cc.inspect_workflow_cache(directory)
    assert plan["state"] == "cleanable" and plan["refusal"] is None and re.fullmatch("[0-9a-f]{64}", plan["plan_id"])
    assert plan["cache_root"] == str(directory.parent) == str(finished["setup"]["output"].resolve())
    groups = {group["category"]: group for group in plan["groups"]}
    assert list(groups) == [category for category, *_ in cc.GROUPS]
    for category, paths in FRAMES.items():
        assert groups[category]["files"] == len(paths)
        assert groups[category]["bytes"] == sum((child / path).stat().st_size for path in paths)
    deletable = sum((child / path).stat().st_size for path in ALL_FRAMES)
    total = size(directory)
    assert (plan["files_deletable"], plan["bytes_deletable"]) == (len(ALL_FRAMES), deletable)
    assert (plan["bytes_total"], plan["bytes_kept"]) == (total, total - deletable)
    assert plan["files_kept"] == plan["files_total"] - len(ALL_FRAMES)
    assert plan["delivery_dirs"] == [str(finished["delivery"].resolve())]
    assert plan["job"]["frames"] == 3 and plan["job"]["delivery_dir"] == finished["done"]["delivery_dir"]
    # Read-only: nothing written, no lock or marker files created, and the plan is stable.
    assert snapshot(directory) == before and snapshot(works) == works_before
    assert cc.inspect_workflow_cache(directory)["plan_id"] == plan["plan_id"]


def test_cache_root_is_the_jobs_recorded_root_not_the_current_preference(finished, tmp_path):
    directory = finished["directory"]
    preference = tmp_path / "new-preference"
    preference.mkdir()
    listing = cc.list_workflow_caches([preference, directory.parent, directory.parent, tmp_path / "missing"])
    assert [plan["workflow_dir"] for plan in listing["jobs"]] == [str(directory)]
    assert listing["jobs"][0]["cache_root"] == str(directory.parent) and listing["jobs"][0]["state"] == "cleanable"
    assert len(listing["cache_roots"]) == 2 and not listing["problems"]
    # A record whose cache root is not the folder holding it is refused.
    record = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    record["config"]["output_root"] = str(preference)
    (directory / "workflow.json").write_text(json.dumps(record), encoding="utf-8")
    plan = cc.inspect_workflow_cache(directory)
    assert plan["plan_id"] is None and "缓存位置" in plan["refusal"]


def originals(finished):
    job = json.loads((finished["child"] / "job.json").read_text(encoding="utf-8"))
    return [Path(item["path"]) for item in job["inputs"]]


def test_moved_originals_do_not_block_cleanup_but_their_cached_copies_are_kept(finished):
    child = finished["child"]
    stage = FRAMES["stage_copies"]
    copies = {relative: (child / relative).read_bytes() for relative in stage}
    for original in originals(finished):
        original.unlink()
    plan = cc.inspect_workflow_cache(finished["directory"])
    assert plan["state"] == "cleanable"
    assert "stage_copies" not in {group["category"] for group in plan["groups"]}
    assert (plan["stage_copies_kept"], plan["stage_copies_kept_bytes"]) == (3, sum(len(data) for data in copies.values()))
    assert plan["files_deletable"] == len(ALL_FRAMES) - len(stage)
    result = cc.clean_workflow_cache(finished["directory"], plan["plan_id"])
    assert result["state"] == "completed" and result["stage_copies_kept"] == 3
    # The work frames are gone; the copies, possibly the last ones of these photos, are byte-identical.
    assert frames_exist(child) == stage
    assert {relative: (child / relative).read_bytes() for relative in stage} == copies
    again = cc.inspect_workflow_cache(finished["directory"])
    assert again["state"] == "cleaned" and again["plan_id"] is None and "原片副本" in again["refusal"]
    assert again["stage_copies_kept"] == 3


def test_only_copies_of_originals_still_in_place_with_their_size_are_deleted(finished):
    child = finished["child"]
    first, second, _ = originals(finished)
    first.unlink()                                          # moved away
    second.write_bytes(second.read_bytes() + b"edited")     # changed in place
    plan = cc.inspect_workflow_cache(finished["directory"])
    stage = {group["category"]: group for group in plan["groups"]}["stage_copies"]
    assert stage["files"] == 1 and plan["stage_copies_kept"] == 2
    cc.clean_workflow_cache(finished["directory"], plan["plan_id"])
    assert frames_exist(child) == ["stage/frame00001.nef", "stage/frame00002.nef"]


def test_an_original_moved_between_inspect_and_clean_changes_the_plan(finished):
    plan = cc.inspect_workflow_cache(finished["directory"])
    originals(finished)[2].unlink()
    with pytest.raises(cc.CleanupRefused, match="清理计划已改变"):
        cc.clean_workflow_cache(finished["directory"], plan["plan_id"])
    assert len(frames_exist(finished["child"])) == len(ALL_FRAMES)


# ---------------------------------------------------------------------------------------------------- deletion

def test_clean_deletes_only_planned_frames_and_keeps_works_and_records(finished):
    directory, child, works = finished["directory"], finished["child"], finished["works"]
    plan = cc.inspect_workflow_cache(directory)
    kept_before = {path: data for path, data in snapshot(directory).items()
                   if path not in {f"{child.relative_to(directory).as_posix()}/{frame}" for frame in ALL_FRAMES}}
    works_before = snapshot(works)
    record_bytes = (directory / "workflow.json").read_bytes()
    result = cc.clean_workflow_cache(directory, plan["plan_id"])
    assert result["state"] == "completed" and not result["errors"]
    assert (result["removed_files"], result["removed_bytes"]) == (plan["files_deletable"], plan["bytes_deletable"])
    # Kept bytes are re-measured inside the locks: lock files this cleanup created (1 byte each) are kept files too.
    assert result["remaining_files"] == 0 and 0 <= result["kept_bytes"] - plan["bytes_kept"] < 8
    assert frames_exist(child) == []
    after = snapshot(directory)
    job_json = child.relative_to(directory).as_posix() + "/job.json"
    assert {path: data for path, data in after.items() if path in kept_before and path != job_json} == \
        {path: data for path, data in kept_before.items() if path != job_json}
    created = set(after) - set(kept_before)
    assert cc.MARKER in created and all(path == cc.MARKER or path.endswith(".lock") for path in created)
    assert (directory / "workflow.json").read_bytes() == record_bytes
    assert snapshot(works) == works_before
    storage.verify_delivery(finished["delivery"])
    gc = child_gc(child)
    assert gc["source"] == "app" and gc["state"] == "done" and gc["work_cache_removed"] is True
    assert (gc["bytes"], gc["files"]) == (plan["bytes_deletable"], len(ALL_FRAMES)) and gc["history"] == []
    marker = json.loads((directory / cc.MARKER).read_text(encoding="utf-8"))
    assert marker["state"] == "completed" and marker["cleaned_by"] == "aester-app" and marker["attempt"] == 1
    assert sorted(marker["removed_files"]) == sorted(f"{child.relative_to(directory).as_posix()}/{frame}" for frame in ALL_FRAMES)
    status = wf.workflow_status(directory)
    assert status["cache_cleaned"] is True
    assert status["cache_cleanup"]["state"] == "completed" and status["cache_cleanup"]["cleaned_by"] == "aester-app"
    again = cc.inspect_workflow_cache(directory)
    assert again["state"] == "cleaned" and again["plan_id"] is None
    with pytest.raises(ValueError, match="已经清理"):
        cc.clean_workflow_cache(directory, plan["plan_id"])
    # Resubmitting with the same submission ID is refused rather than reusing the cleaned cache.
    with pytest.raises(ValueError, match="缓存已清理"):
        wf.submit_workflow(finished["setup"]["inputs"], finished["setup"]["output"], delivery_root=works,
                           delivery_format="aester-delivery-1", idempotency_key="cache-clean")


def test_uncleaned_workflow_status_is_unchanged(finished):
    status = wf.workflow_status(finished["directory"])
    assert "cache_cleaned" not in status and "cache_cleanup" not in status
    assert status == wf._read(finished["directory"])


def test_image_delivery_keeps_receipts_checkpoints_and_rerendering(setup, completed, monkeypatch):
    done, session, state = completed
    directory, child = Path(done["workflow_dir"]), Path(done["stack_job_dir"])
    populate(child)
    receipts = snapshot(directory / "publication-receipts")
    plan = cc.inspect_workflow_cache(directory)
    assert plan["state"] == "cleanable" and plan["files_deletable"] == len(ALL_FRAMES)
    assert cc.clean_workflow_cache(directory, plan["plan_id"])["state"] == "completed"
    assert snapshot(directory / "publication-receipts") == receipts
    record = wf._read(directory)
    for delivered in plan["delivery_dirs"]:
        storage.verify_delivery(delivered, receipt=storage.delivery_receipt(record, delivered))
    # Every frozen workflow stage (checkpoint images, snapshot, masks, finishing layers) still verifies.
    for name in record["checkpoints"]:
        assert wf._cached(directory, name) is not None
    # "另存新预览" re-renders from the kept master: it needs no deleted frame.
    from starstack.preview import render_job_preview
    rendered = render_job_preview(str(child), red_gain=1.1)
    assert rendered["state"] == "succeeded" and all(Path(a["path"]).is_file() for a in rendered["artifacts"])
    # Review, its apply and a trail recompose need the frames: refused with the cleanup message.
    with pytest.raises(ValueError, match="缓存已清理.*作品与缓存.*复核亮迹"):
        review.open_review(str(child))
    assert not (child / "review" / "draft-x").exists()
    with pytest.raises(ValueError, match="缓存已清理"):
        edit.apply_review(done["workflow_dir"], session, expected_draft_revision="token")
    assert state["applies"] == 0
    with pytest.raises(ValueError, match="缓存已清理.*重做亮迹合成"):
        core.recompose_transients(str(child))


def test_preview_after_an_uninstaller_cleanup_explains_the_missing_master(setup, completed):
    done, _, _ = completed
    child = Path(done["stack_job_dir"])
    job = json.loads((child / "job.json").read_text(encoding="utf-8"))
    master = Path(next(a["path"] for a in job["artifacts"] if Path(a["path"]).name == "master_linear.fit"))
    master.unlink()                                       # what the uninstaller's whitelist removes
    from starstack.preview import render_job_preview
    with pytest.raises(FileNotFoundError):                # without a cleanup marker: unchanged behaviour
        render_job_preview(str(child))
    (Path(done["workflow_dir"]) / cc.UNINSTALL_MARKER).write_text(
        '{"version": "aester-cache-cleaned-1", "state": "completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="缓存已清理.*重新生成预览"):
        render_job_preview(str(child))


def test_saved_review_draft_cannot_be_saved_after_cleanup(finished):
    directory, child = finished["directory"], finished["child"]
    job = json.loads((child / "job.json").read_text(encoding="utf-8"))
    job["transient_revision"] = {"id": "base", "record": str(child / "work/transients/candidates.json"),
                                 "sha256": core._sha(child / "work/transients/candidates.json")}
    wf.atomic_json(child / "job.json", job)
    path = review._session_path(child, job["transient_revision"])
    path.parent.mkdir(exist_ok=True)
    wf.atomic_json(path, {"version": review.VERSION, "job_dir": str(child), "session_path": str(path),
                          "base_revision": job["transient_revision"], "draft_revision": "draft-1",
                          "state": "editing", "decisions": {}, "binding": {}})
    plan = cc.inspect_workflow_cache(directory)
    cc.clean_workflow_cache(directory, plan["plan_id"])
    for call in (lambda: review.save_review(str(path), {}, expected_draft_revision="draft-1"),
                 lambda: review.read_review(str(path)),
                 lambda: review.candidate_context(str(path), "light_00001-1")):
        with pytest.raises(ValueError, match="缓存已清理.*复核亮迹"):
            call()
    assert json.loads(path.read_text(encoding="utf-8"))["draft_revision"] == "draft-1"


def test_gc_record_alone_refuses_with_the_app_message(tmp_path):
    job = {"gc": {"source": "app", "work_cache_removed": True, "at": "2026-10-09T00:00:00+00:00"}}
    assert "缓存已清理" in core.gc_refusal(job, "恢复") and "作品与缓存" in core.gc_refusal(job, "恢复")
    # The uninstaller's marker alone (no gc record) refuses with the generic message.
    workflow = tmp_path / "workflow-0abc"
    (workflow / "stack" / "job-0def").mkdir(parents=True)
    (workflow / "workflow.json").write_text("{}", encoding="utf-8")
    assert core.cache_cleanup_refusal(workflow / "stack" / "job-0def", "恢复") is None
    for name in core.CACHE_CLEANUP_MARKERS:              # the uninstaller's marker, then the app's
        (workflow / name).write_text("{}", encoding="utf-8")
        generic = core.cache_cleanup_refusal(workflow / "stack" / "job-0def", "恢复")
        assert "缓存已清理" in generic and "作品与缓存" not in generic
        (workflow / name).unlink()
    assert core.cache_cleanup_refusal(workflow / "stack" / "job-0def", "恢复") is None
    assert core.cache_cleanup_refusal(tmp_path / "job-standalone", "恢复") is None
    cli = core.gc_refusal({"gc": {"work_cache_removed": True, "at": "t"}}, "恢复")
    assert "starstack gc --work-cache" in cli
    assert core.gc_refusal({}, "恢复") is None


# ---------------------------------------------------------------------------------------------------- safety

LOCKS = [("root", ".workflow-admission.lock", "提交或恢复"),
         ("workflow", "worker.lock", "正在运行或被其他窗口使用"),
         ("workflow", "edit.lock", "正在运行或被其他窗口使用"),
         ("workflow", "metadata.lock", "正在运行或被其他窗口使用"),
         ("child", "worker.lock", "正在运行或被其他窗口使用"),
         ("child", "metadata.lock", "正在运行或被其他窗口使用"),
         ("child", "trail_review.lock", "亮迹复核正在进行")]


@pytest.mark.parametrize("where,name,message", LOCKS)
def test_every_held_lock_refuses_before_any_change(finished, where, name, message):
    directory, child = finished["directory"], finished["child"]
    plan = cc.inspect_workflow_cache(directory)
    holder = {"root": directory.parent, "workflow": directory, "child": child}[where] / name
    with file_lock(holder):
        if where != "root":
            busy = cc.inspect_workflow_cache(directory)
            assert busy["state"] == "running" and busy["plan_id"] is None
        before = snapshot(directory, skip_locks=True)
        with pytest.raises(cc.CleanupRefused, match=message) as refused:
            cc.clean_workflow_cache(directory, plan["plan_id"])
        assert refused.value.state == "running"
        assert snapshot(directory, skip_locks=True) == before
    assert len(frames_exist(child)) == len(ALL_FRAMES)
    assert not (directory / cc.MARKER).exists() and child_gc(child) is None
    # Lock files created by the refused attempt are not part of the plan: the same plan now cleans.
    assert cc.clean_workflow_cache(directory, plan["plan_id"])["state"] == "completed"


@pytest.mark.parametrize("change", ["new_frame", "grown_frame", "workflow_record", "stack_record"])
def test_plan_changed_between_inspect_and_clean_deletes_nothing(finished, change):
    directory, child = finished["directory"], finished["child"]
    plan = cc.inspect_workflow_cache(directory)
    if change == "new_frame":
        (child / "work/r_light_00004.fit").write_bytes(b"late registered frame")
    elif change == "grown_frame":
        (child / "work/light_00002.fit").write_bytes(b"x" * 4096)
    elif change == "workflow_record":
        wf._update(directory, message="edited after inspection")
    else:
        core._update(child, note="edited after inspection")
    before = snapshot(directory, skip_locks=True)
    with pytest.raises(cc.CleanupRefused, match="清理计划已改变"):
        cc.clean_workflow_cache(directory, plan["plan_id"])
    assert snapshot(directory, skip_locks=True) == before
    assert not (directory / cc.MARKER).exists() and child_gc(child) is None
    fresh = cc.inspect_workflow_cache(directory)
    assert fresh["plan_id"] != plan["plan_id"]
    assert cc.clean_workflow_cache(directory, fresh["plan_id"])["state"] == "completed"


def test_malformed_plan_id_is_refused(finished):
    with pytest.raises(ValueError, match="编号无效"):
        cc.clean_workflow_cache(finished["directory"], "not-a-plan")
    assert len(frames_exist(finished["child"])) == len(ALL_FRAMES)


@pytest.mark.skipif(os.name != "nt", reason="Windows refuses to delete a file that another handle holds open")
def test_file_held_open_gives_partial_then_the_same_function_finishes(finished):
    directory, child = finished["directory"], finished["child"]
    plan = cc.inspect_workflow_cache(directory)
    held = child / "work/r_light_00002.fit"
    relative = held.relative_to(directory).as_posix()
    with held.open("rb"):
        result = cc.clean_workflow_cache(directory, plan["plan_id"])
        assert result["state"] == "partial" and [item["path"] for item in result["errors"]] == [relative]
        assert result["removed_files"] == len(ALL_FRAMES) - 1 and result["remaining_files"] == 1
        assert frames_exist(child) == ["work/r_light_00002.fit"]
        marker = json.loads((directory / cc.MARKER).read_text(encoding="utf-8"))
        assert marker["state"] == "partial" and child_gc(child)["state"] == "partial"
        assert child_gc(child)["work_cache_removed"] is True
        assert wf.workflow_status(directory)["cache_cleanup"]["state"] == "partial"
        with pytest.raises(ValueError, match="缓存已清理"):
            core.recompose_transients(str(child))
        retry = cc.inspect_workflow_cache(directory)
        assert retry["state"] == "partial" and retry["files_deletable"] == 1
        assert retry["previous_cleanup"]["state"] == "partial"
    done = cc.clean_workflow_cache(directory, retry["plan_id"])
    assert done["state"] == "completed" and done["removed_files"] == 1 and not held.exists()
    marker = json.loads((directory / cc.MARKER).read_text(encoding="utf-8"))
    assert marker["state"] == "completed" and marker["attempt"] == 2
    assert len(marker["removed_files"]) == len(ALL_FRAMES) and marker["removed_bytes"] == plan["bytes_deletable"]
    gc = child_gc(child)
    assert gc["state"] == "done" and [entry["state"] for entry in gc["history"]] == ["partial"]
    assert cc.inspect_workflow_cache(directory)["state"] == "cleaned"


def test_unfinished_foreign_marked_and_legacy_jobs_are_not_offered(finished, setup):
    directory = finished["directory"]
    wf._update(directory, state="failed")
    assert cc.inspect_workflow_cache(directory)["refusal"].startswith("任务未成功完成")
    wf._update(directory, state="running")
    with file_lock(directory / "worker.lock"):
        running = cc.inspect_workflow_cache(directory)
    assert running["state"] == "running" and running["plan_id"] is None and running["bytes_total"] > 0
    wf._update(directory, state="succeeded")
    # A marker that does not name this job (or cannot be read) is not trusted: the job is shown as cleaned.
    for name in (cc.MARKER, cc.UNINSTALL_MARKER):
        for marker in ('{"version": "aester-cache-cleaned-1", "state": "completed"}', "broken-marker"):
            (directory / name).write_text(marker, encoding="utf-8")
            plan = cc.inspect_workflow_cache(directory)
            assert plan["state"] == "cleaned" and plan["plan_id"] is None and "无法识别" in plan["refusal"]
            assert wf.workflow_status(directory)["cache_cleaned"] is True
        (directory / name).unlink()
    legacy = wf.run_workflow(admit(setup, delivery_root=None)["workflow_dir"])
    assert legacy["state"] == "succeeded"
    assert "旧版任务" in cc.inspect_workflow_cache(legacy["workflow_dir"])["refusal"]


@pytest.mark.parametrize("problem", ["edited_photo", "moved_works", "original_inside", "hard_link", "junction"])
def test_unsafe_jobs_are_refused_and_nothing_is_deleted(finished, tmp_path, problem):
    directory, child = finished["directory"], finished["child"]
    if problem == "edited_photo":
        (finished["delivery"] / "final.jpg").write_bytes(b"edited elsewhere")
        expected = "作品校验未通过"
    elif problem == "moved_works":
        finished["delivery"].rename(finished["delivery"].with_name("moved-by-user"))
        expected = "作品文件夹缺失"
    elif problem == "original_inside":
        job = json.loads((child / "job.json").read_text(encoding="utf-8"))
        job["inputs"][0]["path"] = str(child / "stage" / "frame00001.nef")
        wf.atomic_json(child / "job.json", job)
        expected = "缓存内包含原片"
    elif problem == "hard_link":
        os.link(child / "work/light_00001.fit", tmp_path / "elsewhere.fit")
        expected = "硬链接"
    else:
        if os.name != "nt":
            pytest.skip("Windows junction safeguard")
        target = tmp_path / "junction-target"
        target.mkdir()
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(child / "work" / "linked"), str(target)], capture_output=True)
        assert made.returncode == 0, made.stderr
        expected = "目录联接"
    plan = cc.inspect_workflow_cache(directory)
    assert plan["plan_id"] is None and expected in plan["refusal"], plan["refusal"]
    with pytest.raises(ValueError, match=expected):
        cc.clean_workflow_cache(directory, "0" * 64)
    assert len(frames_exist(child)) == len(ALL_FRAMES) and not (directory / cc.MARKER).exists()


def test_only_workflow_folders_inside_their_root_are_planned(finished, tmp_path):
    plan = cc.inspect_workflow_cache(finished["child"])
    assert plan["plan_id"] is None and "workflow-" in plan["refusal"]
    copy = tmp_path / "elsewhere" / finished["directory"].name
    copy.mkdir(parents=True)
    (copy / "workflow.json").write_bytes((finished["directory"] / "workflow.json").read_bytes())
    assert cc.inspect_workflow_cache(copy)["plan_id"] is None


def test_recorded_files_matching_a_pattern_are_kept(finished):
    """Checkpoint files, stack-job artifacts and the transient record are never deleted, whatever their name."""
    directory, child = finished["directory"], finished["child"]
    checkpointed, artifact, transient = (child / "work/light_00001.fit", child / "range-0123456789ab/trial1/r_light_00001.fit",
                                         child / "finite-0123456789ab/q_00001.fit")
    record = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    name = next(iter(record["checkpoints"]))
    record["checkpoints"][name]["files"].append(wf._file(checkpointed))
    wf.atomic_json(directory / "workflow.json", record)
    job = json.loads((child / "job.json").read_text(encoding="utf-8"))
    job["artifacts"].append({"path": str(artifact), "sha256": core._sha(artifact)})
    job["transient_revision"] = {"id": "base", "record": str(transient), "sha256": core._sha(transient)}
    wf.atomic_json(child / "job.json", job)
    kept = {path: path.read_bytes() for path in (checkpointed, artifact, transient)}
    plan = cc.inspect_workflow_cache(directory)
    assert plan["files_deletable"] == len(ALL_FRAMES) - 3
    assert plan["bytes_deletable"] == sum((child / f).stat().st_size for f in ALL_FRAMES) - sum(map(len, kept.values()))
    cc.clean_workflow_cache(directory, plan["plan_id"])
    assert {path: path.read_bytes() for path in kept} == kept
    assert sorted(frames_exist(child)) == sorted(path.relative_to(child).as_posix() for path in kept)


def test_a_damaged_job_folder_does_not_hide_the_other_jobs(finished):
    root = finished["directory"].parent
    for name, content in (("workflow-00ff", "[]"), ("workflow-00fe", '{"config": "x"}'), ("workflow-00fd", "{not json")):
        (root / name).mkdir()
        (root / name / "workflow.json").write_text(content, encoding="utf-8")
    listing = cc.list_workflow_caches([root])
    plans = {Path(plan["workflow_dir"]).name: plan for plan in listing["jobs"]}
    assert plans[finished["directory"].name]["state"] == "cleanable"
    for name in ("workflow-00ff", "workflow-00fe", "workflow-00fd"):
        assert plans[name]["state"] == "refused" and plans[name]["plan_id"] is None and plans[name]["refusal"]
    assert "无法读取" in plans["workflow-00ff"]["refusal"]
    english = translate_message(plans["workflow-00ff"]["refusal"], "en")
    assert english.startswith("The job record is invalid or unreadable") and not CJK.search(english.split(":", 1)[0])


def test_listing_reads_no_image_and_cleaning_hashes_the_works(finished, monkeypatch):
    directory, works = finished["directory"], finished["works"].resolve()
    read = []
    real = hashing.verify_entries

    def spy(entries, **kwargs):
        read.extend(Path(kwargs["path"](entry) if "path" in kwargs else entry["path"]) for entry in list(entries))
        return real(entries, **kwargs)
    monkeypatch.setattr(hashing, "verify_entries", spy)
    for _ in range(3):
        assert cc.list_workflow_caches([directory.parent])["jobs"][0]["state"] == "cleanable"
    assert not [path for path in read if path.is_relative_to(works)]
    # Same size, other bytes: only the hashed check inside clean notices, and nothing is deleted.
    photo = finished["delivery"] / "final.jpg"
    photo.write_bytes(bytes(255 - b for b in photo.read_bytes()))
    plan = cc.inspect_workflow_cache(directory)
    assert plan["state"] == "cleanable"
    with pytest.raises(cc.CleanupRefused, match="作品校验未通过"):
        cc.clean_workflow_cache(directory, plan["plan_id"])
    assert any(path.is_relative_to(works) for path in read)
    assert len(frames_exist(finished["child"])) == len(ALL_FRAMES) and not (directory / cc.MARKER).exists()


def test_works_hashed_without_the_admission_lock_which_is_released_before_deleting(finished, monkeypatch):
    directory, child = finished["directory"], finished["child"]
    admission = directory.parent / ".workflow-admission.lock"
    seen = {"hashed": [], "deleting": []}
    verify, admit = storage.verify_delivery, cc._admit_deletion

    def hashed(*args, **kwargs):
        if kwargs.get("hashes", True):
            seen["hashed"].append(admission.exists() and is_locked(admission))
        return verify(*args, **kwargs)

    def deleting(*args):
        seen["deleting"].append((is_locked(admission), is_locked(directory / "worker.lock"),
                                 (directory / cc.MARKER).exists(), child_gc(child)["state"]))
        return admit(*args)
    monkeypatch.setattr(storage, "verify_delivery", hashed)
    monkeypatch.setattr(cc, "_admit_deletion", deleting)
    plan = cc.inspect_workflow_cache(directory)
    assert cc.clean_workflow_cache(directory, plan["plan_id"])["state"] == "completed"
    assert seen["hashed"] == [False]
    assert seen["deleting"] and all(state == (False, True, True, "in_progress") for state in seen["deleting"])


def test_works_changed_after_hashing_refuses(finished, monkeypatch):
    directory = finished["directory"]
    verify = storage.verify_delivery

    def touched(*args, **kwargs):
        receipt = verify(*args, **kwargs)
        if kwargs.get("hashes", True):
            photo = finished["delivery"] / "final.jpg"
            os.utime(photo, ns=(photo.stat().st_atime_ns, photo.stat().st_mtime_ns + 10**9))
        return receipt
    monkeypatch.setattr(storage, "verify_delivery", touched)
    plan = cc.inspect_workflow_cache(directory)
    with pytest.raises(cc.CleanupRefused, match="清理计划已改变"):
        cc.clean_workflow_cache(directory, plan["plan_id"])
    assert len(frames_exist(finished["child"])) == len(ALL_FRAMES) and not (directory / cc.MARKER).exists()


def test_listing_stops_when_cancelled(finished):
    root = finished["directory"].parent
    calls = []
    listing = cc.list_workflow_caches([root], cancelled=lambda: calls.append(1) or True)
    assert listing["cancelled"] is True and listing["jobs"] == [] and calls == [1]
    assert cc.list_workflow_caches([root])["cancelled"] is False


# ---------------------------------------------------------------------------------------------------- uninstaller

def uninstaller_clean(directory):
    plan = u.prepare_cache_cleanup(directory, cache_root=directory.parent)
    result = u.execute_cache_cleanup(plan, exclusive=True)
    assert result["state"] == "completed", result
    return plan, result


def test_app_then_uninstaller_each_frees_its_own_files(finished, registry, monkeypatch):
    directory, child = finished["directory"], finished["child"]
    assert u.CLEANUP_MARKER == cc.UNINSTALL_MARKER == core.CACHE_CLEANUP_MARKER != cc.MARKER
    plan = cc.inspect_workflow_cache(directory)
    cc.clean_workflow_cache(directory, plan["plan_id"])
    assert frames_exist(child) == [] and not (directory / cc.UNINSTALL_MARKER).exists()
    # The uninstaller's scan still offers the masters and products the in-app cleanup kept.
    monkeypatch.setattr(u, "_known_cache_roots", lambda: [directory.parent])
    scan = u.scan_options(finished["works"])
    offered = [item for item in scan["cache_plans"] if item["workflow_dir"] == str(directory)]
    assert len(offered) == 1 and offered[0]["bytes"] > 0
    assert str(directory) not in {item["path"] for item in scan["preserved"]}
    works = snapshot(finished["works"])
    _, result = uninstaller_clean(directory)
    assert result["removed_bytes"] == offered[0]["bytes"] and snapshot(finished["works"]) == works
    status = wf.workflow_status(directory)
    assert status["cache_cleaned"] is True and status["cache_cleanup"]["cleaned_by"] == "aester-app"
    assert status["cache_cleanup"]["cleaned_by_all"] == ["aester-app", "uninstaller"]
    assert cc.inspect_workflow_cache(directory)["state"] == "cleaned"


def test_uninstaller_then_app_frees_the_frames(finished, registry):
    directory, child = finished["directory"], finished["child"]
    _, result = uninstaller_clean(directory)
    assert len(frames_exist(child)) == len(ALL_FRAMES)       # the uninstaller never removes per-frame files
    uninstaller_marker = (directory / cc.UNINSTALL_MARKER).read_bytes()
    assert wf.workflow_status(directory)["cache_cleanup"]["cleaned_by"] == "uninstaller"
    plan = cc.inspect_workflow_cache(directory)
    assert plan["state"] == "cleanable" and plan["files_deletable"] == len(ALL_FRAMES)
    assert plan["uninstaller_cleanup"] == {"state": "completed", "removed_bytes": result["removed_bytes"]}
    assert plan["previous_cleanup"] is None
    done = cc.clean_workflow_cache(directory, plan["plan_id"])
    assert done["state"] == "completed" and frames_exist(child) == []
    assert (directory / cc.UNINSTALL_MARKER).read_bytes() == uninstaller_marker
    marker = json.loads((directory / cc.MARKER).read_text(encoding="utf-8"))
    assert marker["after_uninstaller_cleanup"]["state"] == "completed" and marker["attempt"] == 1
    assert cc.inspect_workflow_cache(directory)["state"] == "cleaned"
    with pytest.raises(ValueError, match="already"):
        u.prepare_cache_cleanup(directory, cache_root=directory.parent)


def test_an_unbound_uninstaller_marker_is_not_trusted(finished, registry):
    directory = finished["directory"]
    uninstaller_clean(directory)
    marker = json.loads((directory / cc.UNINSTALL_MARKER).read_text(encoding="utf-8"))
    for change in ({"workflow_signature": "0" * 64}, {"workflow_dir": str(directory.parent)}, {"state": "odd"}):
        (directory / cc.UNINSTALL_MARKER).write_text(json.dumps({**marker, **change}), encoding="utf-8")
        plan = cc.inspect_workflow_cache(directory)
        assert plan["state"] == "cleaned" and plan["plan_id"] is None and "无法识别" in plan["refusal"]


def test_uninstall_options_still_imports_with_the_standard_library_only():
    code = ("import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('bootstrap_uninstall_options', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            "loaded = sorted(n for n in sys.modules if n.split('.')[0] in "
            "('starstack', 'numpy', 'PySide6', 'tifffile', 'astropy', 'PIL'))\n"
            "assert not loaded, loaded\n"
            "print(module.CLEANUP_MARKER)\n")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    finished = subprocess.run([sys.executable, "-I", "-c", code, u.__file__], capture_output=True, text=True,
                              env=environment, timeout=120)
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == cc.UNINSTALL_MARKER


# ---------------------------------------------------------------------------------------------------- language

def test_cleanup_messages_and_labels_have_english():
    previous = i18n.get_language()
    try:
        messages = [cc.BUSY, cc.ADMISSION_BUSY, cc.REVIEW_BUSY, cc.CHANGED, cc.FOREIGN, cc.CLEANED, cc.NOTHING,
                    cc.COPIES_KEPT, *(label for *_, label in cc.GROUPS)]
        messages += [core._cache_cleaned_message(action, in_app=in_app) for action in ("恢复", "重做亮迹合成", "复核亮迹")
                     for in_app in (False, True)] + [core._cache_cleaned_message("重新生成预览")]
        messages += [core.gc_refusal({"gc": {"work_cache_removed": True, "at": "2026-10-09"}}, "复核亮迹")]
        messages += ["作品校验未通过，不能清理：输入或已完成阶段的文件已改变／缺失：C:\\works\\final.jpg",
                     "缓存中有软链接或目录联接，不能清理：C:\\cache\\x", "文件在检查后改变，未删除：C:\\cache\\y"]
        for message in messages:
            english = translate_message(message, "en")
            assert not CJK.search(english), (message, english)
            assert translate_message(message, "zh_CN") == message
    finally:
        i18n.set_language(previous)


def test_uninstaller_marker_version_is_the_literal_the_uninstaller_writes():
    """cache_cleanup recognises the uninstaller's marker by version; both sides must keep the same literal."""
    import inspect
    assert cc.UNINSTALL_MARKER == u.CLEANUP_MARKER
    assert f'"version": "{cc.UNINSTALL_MARKER_VERSION}"' in inspect.getsource(u.execute_cache_cleanup)
