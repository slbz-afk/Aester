# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See src/aester_desktop/LICENSE.txt.
"""Trail review triage: priority and add-back filters, sortable columns, review keys, the next-to-review
button, a screen-sized dialog with a collapsible side panel, the scan thumbnail and the composite list."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import csv
import io
import json
import re
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from aester_desktop import trail_manifest
from aester_desktop import trail_review_dialog as dialog_module
from aester_desktop.trail_review_dialog import (COL_ID, COL_LENGTH, COL_PRIORITY, COL_SNR, COL_TIME,
                                                TrailReviewDialog)
from starstack import i18n

from test_trail_review_session import job_directory  # noqa: F401 - fixture


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def chinese():
    old = i18n.get_language()
    i18n.set_language("zh_CN")
    yield
    i18n.set_language(old)


def _candidate(cid, frame, *, status="omitted_unverified", priority="low", label="unknown", snr=None,
               length=None, date=None, measured=True):
    return {"id": cid, "frame_index": frame, "input": f"C:\\Users\\someone\\shots\\DSC_{frame:04d}.NEF",
            "input_sha256": f"{frame:064x}", "status": "measured" if measured else "too_short",
            "suggestion": {"class": label}, "effective_label": label, "decision": "omit",
            "total_snr": snr, "length_px": length, "date_obs": date,
            "endpoints_native_reference": [[100., frame*10.], [200., frame*10.]],
            "endpoints_native_source": [[101., frame*10.], [201., frame*10.]],
            "recall_review": {"status": status, "priority": priority}}


@pytest.fixture
def session():
    candidates = [
        _candidate("light_00001-1", 1, priority="low", snr=40., length=300., date="2026-12-14T22:00:10"),
        _candidate("light_00002-1", 2, priority="high", snr=900., length=120., date="2026-12-14T22:00:30"),
        _candidate("light_00002-2", 2, priority="normal", snr=None, length=80., date="2026-12-14T22:00:30"),
        _candidate("light_00003-1", 3, status="retained", label="meteor", snr=1500., length=600.,
                   date="2026-12-14T22:00:50"),
        _candidate("light_00004-1", 4, priority="high", snr=12., length=None, date=None),
    ]
    candidates[3]["decision"] = "keep"
    return {"session_path": "/job/review/draft.json", "draft_revision": "draft-1", "base_revision": {"id": "rev1"},
            "job_dir": "/job", "geometry": {"width": 1000, "height": 600}, "decisions": {}, "preview_path": None,
            "application_policy_version": "meteor-default-keep-1", "candidates": candidates}


@pytest.fixture
def make(app, monkeypatch):
    monkeypatch.setattr(dialog_module.service, "candidate_context", lambda *a: {"panels": [], "notice": "test"})
    created = []

    def build(value, **kwargs):
        view = TrailReviewDialog(value, **kwargs)
        created.append(view)
        return view
    yield build
    for view in created:
        view._dirty = False
        view._finish_workers()
        view.reject()
    app.processEvents()


def order(view):
    return [view.list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) for i in range(view.list.topLevelItemCount())]


def test_opens_by_review_priority_with_counts_and_dash_for_kept(make, session):
    view = make(session)
    # high, normal, low, then candidates the recall review does not list (kept), each in record order.
    assert order(view) == ["light_00002-1", "light_00004-1", "light_00002-2", "light_00001-1", "light_00003-1"]
    assert view.list.currentItem() is view._items["light_00002-1"]
    item = view._items["light_00003-1"]
    assert item.text(COL_PRIORITY) == "—" and view._items["light_00002-1"].text(COL_PRIORITY) == "高"
    assert [view.priority_buttons[k].text() for k in (None, "high", "normal", "low")] == ["全部", "高 2", "一般 1", "低 1"]
    assert view.keep_buttons["keep"].text() == "会加回 1" and view.keep_buttons["omit"].text() == "不加回 4"
    assert "不是可信度" in view.list.headerItem().toolTip(COL_PRIORITY)
    # The priority column is shown second; logical column numbers (and the decision columns) are unchanged.
    assert view.list.header().visualIndex(COL_PRIORITY) == 1 and view.list.headerItem().text(1) == "身份"


def test_priority_and_keep_filters_combine_and_follow_the_draft(make, session):
    view = make(session)
    view.set_priority_filter("high")
    assert sorted(view.visible_ids()) == ["light_00002-1", "light_00004-1"]
    assert view.filter_count.text() == "显示 2 / 5 条"
    view.set_priority_filter(None)
    view.set_keep_filter("keep")
    assert view.visible_ids() == ["light_00003-1"]
    view.set_keep_filter("omit")
    view.select_candidate("light_00002-1")
    QTest.keyClick(view.list, Qt.Key.Key_K)
    # Kept now, so outside 不加回, but it stays shown (pinned) until the filter changes.
    assert view.decisions == {"light_00002-1": {"override": "keep"}}
    assert "light_00002-1" in view.visible_ids() and view.keep_buttons["keep"].text() == "会加回 2"
    view.set_keep_filter("omit")
    assert "light_00002-1" not in view.visible_ids()


def test_sortable_columns_keep_ids_and_put_missing_values_last(make, session):
    view = make(session)
    view.sort_by(COL_SNR)              # first click: brightest first
    assert order(view) == ["light_00003-1", "light_00002-1", "light_00001-1", "light_00004-1", "light_00002-2"]
    view.sort_by(COL_SNR)              # again: reversed, the missing value still last
    assert order(view) == ["light_00004-1", "light_00001-1", "light_00002-1", "light_00003-1", "light_00002-2"]
    view.sort_by(COL_LENGTH)
    assert order(view)[:2] == ["light_00003-1", "light_00001-1"] and order(view)[-1] == "light_00004-1"
    view.sort_by(COL_TIME)             # ascending, equal times in record order, no time last
    assert order(view) == ["light_00001-1", "light_00002-1", "light_00002-2", "light_00003-1", "light_00004-1"]
    view.sort_by(COL_ID)
    assert order(view) == [c["id"] for c in session["candidates"]]
    for cid, item in view._items.items():
        assert item.data(0, Qt.ItemDataRole.UserRole) == cid
    assert view._items["light_00001-1"].text(COL_TIME) == "22:00:10"
    assert "时区未知" in view._items["light_00001-1"].toolTip(COL_TIME)
    # The identity column is not sortable (it changes while reviewing): the sort stays where it was.
    view._header_clicked(1)
    assert order(view) == [c["id"] for c in session["candidates"]] and view.list.sort_state == (COL_ID, False)


def test_header_click_sorts(make, session):
    view = make(session)
    view.show()
    QApplication.processEvents()
    header = view.list.header()
    x = header.sectionViewportPosition(COL_TIME) + 5
    QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x, 5))
    assert view.list.sort_state == (COL_TIME, False)
    assert order(view)[0] == "light_00001-1"


def test_review_keys_set_keep_omit_reset_with_undo_and_enter_goes_to_the_next(make, session):
    view = make(session)
    view.show()
    QApplication.processEvents()
    view.select_candidate("light_00002-1")
    QTest.keyClick(view.list, Qt.Key.Key_O)
    assert view.decisions == {"light_00002-1": {"override": "omit"}}
    QTest.keyClick(view.list, Qt.Key.Key_K)
    assert view.decisions == {"light_00002-1": {"override": "keep"}}
    QTest.keyClick(view.list, Qt.Key.Key_U)
    assert view.decisions == {"light_00002-1": {"override": None}}
    assert view._items["light_00002-1"].text(3) == "恢复默认"
    view.undo()
    view.undo()
    assert view.decisions == {"light_00002-1": {"override": "omit"}}
    # Enter: the next shown row without a choice, in list order; a row with a choice is skipped.
    QTest.keyClick(view.list, Qt.Key.Key_Return)
    assert view.list.currentItem() is view._items["light_00004-1"]
    view.decisions["light_00002-2"] = {"override": "omit"}
    view._refresh_decisions()
    QTest.keyClick(view, Qt.Key.Key_Enter)                         # the dialog takes Enter too
    assert view.list.currentItem() is view._items["light_00001-1"]
    # Modified letters are not review keys; the list's type-to-search does not see review letters.
    QTest.keyClick(view.list, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    assert "light_00001-1" not in view.decisions


def test_next_to_review_wraps_and_reports_when_all_are_chosen(make, session):
    view = make(session)
    view.set_priority_filter("high")
    view.select_candidate("light_00004-1")
    assert view.next_to_review() and view.list.currentItem() is view._items["light_00002-1"]   # wrapped
    for cid in ("light_00002-1", "light_00004-1"):
        view.decisions[cid] = {"override": "omit"}
    view._refresh_decisions()
    assert not view.next_to_review()
    assert view.status.text() == "当前筛选下的候选都已有选择。"
    view.next_button.click()
    assert view.status.text() == "当前筛选下的候选都已有选择。"


def test_bulk_choices_still_use_the_comboboxes_and_pin_rows(make, session):
    view = make(session)
    view.set_priority_filter("high")
    for cid in ("light_00002-1", "light_00004-1"):
        view._items[cid].setSelected(True)
    view.action_choice.setCurrentIndex(view.action_choice.findData("keep"))
    view.set_selected_decisions()
    assert view.decisions == {"light_00002-1": {"override": "keep"}, "light_00004-1": {"override": "keep"}}
    view.undo()
    assert view.decisions == {}


def test_dialog_fits_the_screen_and_the_side_panel_collapses(make, session, app):
    view = make(session)
    area = app.primaryScreen().availableGeometry()
    assert view.width() <= area.width() and view.height() + 48 <= area.height()
    view.show()
    QApplication.processEvents()
    assert view.split.sizes()[2] > 0 or view.width() < 1100
    view.show_details(False)
    assert view.split.sizes()[2] == 0 and not view.details_button.isChecked()
    view.show_details(True)
    assert view.split.sizes()[2] > 0 and view.details_button.isChecked()


def test_scan_thumbnail_and_its_note_are_shown(make, session, tmp_path):
    image = QImage(200, 100, QImage.Format.Format_RGB32)
    image.fill(QColor("#808080"))
    path = tmp_path / "light_00002-1.png"
    image.save(str(path))
    session["candidates"][1].update(review_png=str(path), preview_note="扫描时生成的缩略图，未核对文件此后是否被改动，仅供浏览；判断请以经过校验的实际曝光对照为准")
    view = make(session)
    view.select_candidate("light_00002-1")
    assert view.thumbnail.pixmap() is not None and not view.thumbnail.pixmap().isNull()
    assert view.thumbnail_note.text().startswith("扫描时生成的缩略图") and not view.thumbnail_note.isHidden()
    view.select_candidate("light_00004-1")
    assert view.thumbnail.text() == "此候选没有扫描缩略图。" and view.thumbnail_note.isHidden()
    i18n.set_language("en")
    english = make(session)
    english.select_candidate("light_00002-1")
    assert english.thumbnail_note.text().startswith("Thumbnail made during the scan")


def test_summary_names_the_file_only_and_keeps_the_path_in_its_tooltip(make, session):
    view = make(session)
    view.select_candidate("light_00002-1")
    assert "DSC_0002.NEF" in view.summary.text() and "someone" not in view.summary.text()
    assert view.summary.toolTip().endswith("DSC_0002.NEF") and "someone" in view.summary.toolTip()


# ---- composite list ------------------------------------------------------------------------------

DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\")


def _job(tmp_path):
    inputs = [{"path": str(tmp_path / "shots" / f"DSC_{n:04d}.NEF"), "sha256": f"{n:064x}", "size": 1}
              for n in range(1, 6)]
    return {"job_dir": str(tmp_path / "job"), "inputs": inputs,
            "diagnostics": {"input_count": 5, "stacked_count": 4, "reference_stem": "light_00003",
                            "reference_input": inputs[2]["path"],
                            "exposure": {"input_total_seconds": 50.0, "included_total_seconds": 40.0},
                            "registration_detail": {"unregistered_frames": ["light_00005"],
                                                    "quality_excluded_frames": ["light_00005"]},
                            "quality_gate": {"excluded": [{"stem": "light_00005", "reasons": ["stars 0.34x median"]}]},
                            "transients": {"frames_scanned": 4}}}


def _workflow(tmp_path, session, kept):
    works = tmp_path / "works" / "aester-0123456789abcdef0123"
    return {"request": {"delivery_root": str(tmp_path / "works"), "image_rotation": 0},
            "delivery_dir": str(works / "delivery-trails-1kept-0123456789ab"),
            "base_delivery_dir": str(works / "delivery-abcdef012345"),
            "trail_review": {"kept_ids": kept, "kept_count": len(kept), "revision": dict(session["base_revision"]),
                             "kept_by": {"algorithm": len(kept), "human": 0, "agent": 0}}}


def test_manifest_rows_are_the_published_keeps_with_header_facts(session, tmp_path):
    workflow = _workflow(tmp_path, session, ["light_00003-1"])
    manifest = trail_manifest.build(session, _job(tmp_path), workflow)
    assert len(manifest["rows"]) == workflow["trail_review"]["kept_count"] == 1
    row = manifest["rows"][0]
    assert row["source_file"] == "DSC_0003.NEF" and row["frame_index"] == 3 and row["input_sha256"] == f"{3:064x}"
    assert row["capture_time_camera_clock"] == "2026-12-14T22:00:50" and row["exposure_s"] == ""
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (100., 30., 200., 30.)
    assert (row["identity"], row["identity_source"], row["keep_source"]) == ("meteor", "algorithm", "algorithm")
    header = manifest["header"]
    assert header["time_zone"] == "unknown" and header["added_back_by"] == {"algorithm": 1, "human": 0, "agent": 0}
    assert header["reference_frame"] == {"file": "DSC_0003.NEF", "frame_index": 3,
                                         "capture_time_camera_clock": "2026-12-14T22:00:50",
                                         "capture_time_basis": "candidates_on_reference_frame"}
    assert header["segment"]["first_capture_time_camera_clock"] == "2026-12-14T22:00:10"
    assert header["segment"]["last_capture_time_camera_clock"] == "2026-12-14T22:00:50"
    assert header["not_scanned"] == {"count": 1, "frames": [{"frame_index": 5, "file": "DSC_0005.NEF",
                                                             "reason": "quality_gate",
                                                             "details": ["stars 0.34x median"]}]}
    assert header["trails_folder"] == "delivery-trails-1kept-0123456789ab"
    assert any("线性母版" in note and "不含流星" in note for note in header["notes"])
    assert any("时区未知" in note for note in header["notes"])


def test_manifest_keep_source_is_the_published_one_not_the_draft(session, tmp_path):
    session["candidates"][1].update(override="keep", override_source="human", decision="keep",
                                    keep_source="human")
    session["candidates"][3]["keep_source"] = None          # a saved draft omits it: the view shows the draft
    session["decisions"] = {"light_00003-1": {"override": "omit"}}
    workflow = _workflow(tmp_path, session, ["light_00002-1", "light_00003-1"])
    rows = trail_manifest.build(session, _job(tmp_path), workflow)["rows"]
    assert [(r["id"], r["keep_source"]) for r in rows] == [("light_00002-1", "human"), ("light_00003-1", "algorithm")]


def test_manifest_refuses_without_a_matching_trails_version(session, tmp_path):
    workflow = _workflow(tmp_path, session, ["light_00003-1"])
    for broken in ({**workflow, "trail_review": {**workflow["trail_review"], "revision": {"id": "older"}}},
                   {**workflow, "delivery_dir": workflow["base_delivery_dir"]}, None):
        assert trail_manifest.published_keeps(session, broken) is None
        with pytest.raises(ValueError):
            trail_manifest.build(session, _job(tmp_path), broken)


def test_written_list_has_no_drive_or_user_folder_and_names_files_only(session, tmp_path):
    workflow = _workflow(tmp_path, session, ["light_00003-1"])
    manifest = trail_manifest.build(session, _job(tmp_path), workflow)
    (tmp_path / "out").mkdir()
    csv_path, json_path = trail_manifest.write(manifest, tmp_path / "out" / "list")
    csv_text, json_text = csv_path.read_text(encoding="utf-8-sig"), json_path.read_text(encoding="utf-8")
    assert csv_path.suffix == ".csv" and json_path.suffix == ".json"
    home = str(Path.home())
    for text in (csv_text, json_text):
        assert not DRIVE.search(text), DRIVE.search(text)
        for part in (home, Path.home().name + "\\", str(tmp_path), "someone", "shots"):
            assert part not in text, part
    rows = list(csv.DictReader(io.StringIO("\n".join(l for l in csv_text.splitlines() if not l.startswith("#")))))
    assert [r["id"] for r in rows] == ["light_00003-1"] and rows[0]["source_file"] == "DSC_0003.NEF"
    assert json.loads(json_text)["rows"][0]["source_file"] == "DSC_0003.NEF"
    assert "# time_zone: unknown" in csv_text


def test_export_refuses_the_works_folder_and_writes_elsewhere(make, session, tmp_path, monkeypatch):
    from starstack import core
    workflow = _workflow(tmp_path, session, ["light_00003-1"])
    monkeypatch.setattr(core, "load_job", lambda directory: _job(tmp_path))
    view = make(session, workflow=workflow)
    assert view.export_button.isEnabled()
    trails = Path(workflow["delivery_dir"])
    trails.mkdir(parents=True)
    for target in (trails / "list.csv", trails.parent / "list.csv", tmp_path / "works" / "list.csv"):
        assert view.export_manifest(target) is None and not target.exists()
        assert "作品文件夹" in view.status.text()
    assert trail_manifest.inside_works(tmp_path / "other" / "aester-0123456789abcdef0123" / "x.csv", None)
    out = tmp_path / "lists"
    out.mkdir()
    written = view.export_manifest(out / "geminids.csv")
    assert written == (out / "geminids.csv", out / "geminids.json")
    assert "加回 1 条" in view.status.text() and all(path.is_file() for path in written)
    assert not any(trails.iterdir())


def test_export_is_disabled_without_a_trails_version(make, session):
    view = make(session)
    assert not view.export_button.isEnabled() and "暂时不能导出" in view.export_button.toolTip()


def test_manifest_from_a_real_review_session(job_directory, tmp_path):  # noqa: F811 - fixture
    """open_review's real view (two measured candidates, both unknown) with one explicit keep published."""
    from starstack import core, trail_review_session as service
    session = service.open_review(str(job_directory))
    keep = session["candidates"][0]["id"]
    session["candidates"][0].update(override="keep", override_source="human", decision="keep")
    job = core.load_job(job_directory)
    workflow = {"delivery_dir": str(tmp_path / "w" / "delivery-trails-1kept-0123456789ab"),
                "base_delivery_dir": str(tmp_path / "w" / "delivery-abcdef012345"),
                "trail_review": {"kept_ids": [keep], "revision": dict(session["base_revision"])}}
    manifest = trail_manifest.build(session, job, workflow)
    assert [row["id"] for row in manifest["rows"]] == [keep]
    assert manifest["rows"][0]["source_file"] == "frame-1.nef" and manifest["rows"][0]["keep_source"] == "human"
    assert not DRIVE.search(trail_manifest.csv_text(manifest))
    assert not DRIVE.search(json.dumps(manifest, ensure_ascii=False))
