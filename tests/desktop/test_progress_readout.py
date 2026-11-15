"""The progress read-out: the stage name (including the trail scan and add-back) and the elapsed time.

Real Qt widgets offscreen; the workflow record is given directly and the stack job is read through a stub of the
window's core.load_job.
"""
import os
import re
from datetime import datetime, timedelta, timezone

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from aester_desktop import gui
from test_meteor_shower_entry import app, make  # noqa: F401  (fixtures)


def _running(tmp_path, stage, *, seconds=75, attempt=1):
    started = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return {"workflow_dir": str(tmp_path), "state": "running", "stage": stage, "progress": 40, "attempt": attempt,
            "created_at": started.isoformat(), "stack_job_dir": str(tmp_path / "stack" / "job-1"),
            "request": {"trails_enabled": True}, "config": {}}


@pytest.mark.parametrize("language,child,expected", [
    ("zh_CN", "transients", "寻找流星等亮迹"), ("zh_CN", "convert", "读取图像"), ("zh_CN", "calibrate", "校准镜头畸变"),
    ("en", "transients", "Finding meteors and trails"), ("en", "compose", "Combining sky and foreground")])
def test_stage_name_comes_from_the_stack_job_with_elapsed_time(make, app, tmp_path, monkeypatch, language, child,
                                                               expected):
    window, _ = make(language)
    monkeypatch.setattr(gui, "load_job", lambda path: {"state": "running", "stage": child})
    window.workflow_dir = str(tmp_path)
    window._apply_workflow_record(_running(tmp_path, "stack"))
    window.timer.stop()
    text = window.stage_label.text()
    separator = " · 已用 " if language == "zh_CN" else " · elapsed "
    name, _, elapsed = text.partition(separator)
    assert name == expected and re.fullmatch(r"1:1[5-7]", elapsed), text
    window.workflow_dir = None
    window.job_state = ""


def test_trail_addback_and_long_runs_read_clearly(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    monkeypatch.setattr(gui, "load_job", lambda path: {"state": "succeeded", "stage": "complete"})
    window.workflow_dir = str(tmp_path)
    window._apply_workflow_record(_running(tmp_path, "trail_addback", seconds=3725))
    window.timer.stop()
    assert re.fullmatch(r"加回流星 · 已用 1:02:0[5-7]", window.stage_label.text()), window.stage_label.text()
    window.workflow_dir = None
    window.job_state = ""


def test_resumed_attempt_is_timed_only_from_this_windows_resume(make, app, tmp_path, monkeypatch):
    window, _ = make("zh_CN")
    monkeypatch.setattr(gui, "load_job", lambda path: {"state": "running", "stage": "stack"})
    window.workflow_dir = str(tmp_path)
    window._apply_workflow_record(_running(tmp_path, "publish", seconds=4000, attempt=2))
    window.timer.stop()
    assert window.stage_label.text() == "保存结果"                     # resumed elsewhere: no elapsed time shown
    window._run_started_for = None
    window._resume_requested = (os.path.normcase(str(tmp_path)), datetime.now(timezone.utc) - timedelta(seconds=20))
    window._apply_workflow_record(_running(tmp_path, "publish", seconds=4000, attempt=2))
    window.timer.stop()
    assert re.fullmatch(r"保存结果 · 已用 0:2[0-2]", window.stage_label.text()), window.stage_label.text()
    finished = dict(_running(tmp_path, "complete"), state="succeeded")
    window._apply_workflow_record(finished)
    assert window.stage_label.text() == "处理完成"
    window.workflow_dir = None
    window.job_state = ""
