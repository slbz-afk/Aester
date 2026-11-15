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
import subprocess
import sys
from datetime import timedelta, timezone
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


BEIJING = timezone(timedelta(hours=8))


@pytest.fixture(autouse=True)
def beijing(monkeypatch):
    """The list shows UT capture times in this computer's zone; the tests fix it at UTC+8."""
    monkeypatch.setattr(dialog_module, "LOCAL_ZONE", BEIJING)


def _candidate(cid, frame, *, status="omitted_unverified", priority="low", label="unknown", snr=None,
               length=None, date=None, measured=True, exposure=10.0):
    return {"id": cid, "frame_index": frame, "input": f"C:\\Users\\someone\\shots\\DSC_{frame:04d}.NEF",
            "input_sha256": f"{frame:064x}", "status": "measured" if measured else "too_short",
            "suggestion": {"class": label}, "effective_label": label, "decision": "omit",
            "total_snr": snr, "length_px": length, "date_obs": date, "exptime_s": exposure,
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
    # 22:00:10 UT is 06:00:10 the next morning in UTC+8.
    assert view._items["light_00001-1"].text(COL_TIME) == "06:00:10"
    tip = view._items["light_00001-1"].toolTip(COL_TIME)
    assert "2026-12-15 06:00:10" in tip and "UTC+08:00" in tip and "世界时：2026-12-14T22:00:10" in tip
    assert "这台电脑的时区" in view.list.headerItem().toolTip(COL_TIME)
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


def test_capture_time_is_ut_shown_in_this_computers_zone(make, session):
    """DATE-OBS is UT (Siril converts the camera clock with the processing computer's zone): a camera in
    Beijing that wrote 04:22:06 on 08-13 is recorded as 2026-08-12T20:22:06 and shown as 04:22:06 again."""
    assert dialog_module.capture_clock("2026-08-12T20:22:06") == "04:22:06"
    local = trail_manifest.capture_time("2026-08-12T20:22:06", BEIJING)
    assert local.isoformat() == "2026-08-13T04:22:06+08:00"
    assert trail_manifest.capture_time("2026-08-12T20:22:06Z", BEIJING) == local
    assert trail_manifest.capture_time("2026-08-12T21:22:06+01:00", BEIJING) == local
    assert trail_manifest.capture_time("not a time") is None and dialog_module.capture_clock("not a time") == "not a time"
    session["candidates"][0]["date_obs"] = "2026-08-12T20:22:06"
    view = make(session)
    item = view._items["light_00001-1"]
    assert item.text(COL_TIME) == "04:22:06" and "2026-08-13 04:22:06" in item.toolTip(COL_TIME)
    view.sort_by(COL_TIME)
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
    view.set_priority_filter(None)
    view.set_filter("aircraft")                     # nothing shown: no claim that everything has a choice
    assert not view.next_to_review() and view.status.text() == "当前筛选下没有候选。"


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


_FIT_SCRIPT = r'''
import json, sys
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication
app = QApplication([])
from aester_desktop import fonts, themes
fonts.register_bundled_fonts()
from starstack import i18n
from aester_desktop import trail_review_dialog as trd
trd.service.candidate_context = lambda *a: {"panels": [], "notice": ""}
session = json.load(open(sys.argv[1], encoding="utf-8"))
out = []
for language in ("zh_CN", "en"):
    i18n.set_language(language)
    view = trd.TrailReviewDialog(session)
    themes.apply_theme(view, "dark")
    view.show()
    app.processEvents()
    frame, area = view.frameGeometry(), view.screen().availableGeometry()
    widgets = {name: getattr(view, name) for name in ("next_button", "bulk_button", "undo_button", "export_button",
                                                       "apply_button", "close_button", "details_button")}
    rects = {name: QRect(w.mapTo(view, QPoint(0, 0)), w.size()) for name, w in widgets.items()}
    out.append({"language": language, "frame": [frame.x(), frame.y(), frame.width(), frame.height()],
                "area": [area.x(), area.y(), area.width(), area.height()],
                "outside": [name for name, rect in rects.items()
                            if not (widgets[name].isVisible() and view.rect().contains(rect))],
                "controls_bottom": max(rects[n].bottom() for n in ("next_button", "bulk_button", "undo_button")),
                "footer_top": min(rects[n].top() for n in ("export_button", "apply_button", "close_button"))})
    view._dirty = False
    view._finish_workers()
    view.reject()
print(json.dumps(out))
'''


@pytest.mark.parametrize("width,height", [(1280, 720), (1093, 614)])   # 1366 x 768 at 125 % is 1093 x 614
def test_dialog_fits_laptop_screens_in_both_languages(session, tmp_path, width, height):
    """With the bundled fonts and the theme, the shown dialog lies inside the screen and nothing is clipped."""
    session["candidates"] = [dict(copy.deepcopy(session["candidates"][n % 5]), id=f"light_{n:05d}-{n % 13 + 1}",
                                  frame_index=n) for n in range(1, 145)]
    (tmp_path / "session.json").write_text(json.dumps(session), encoding="utf-8")
    (tmp_path / "screen.json").write_text(json.dumps({"windowFrameMargins": True, "screens": [
        {"name": "laptop", "x": 0, "y": 0, "width": width, "height": height, "logicalDpi": 96,
         "logicalBaseDpi": 96, "dpr": 1}]}), encoding="utf-8")
    (tmp_path / "fit.py").write_text(_FIT_SCRIPT, encoding="utf-8")
    src = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen:configfile=screen.json",   # relative: Qt splits at ':'
           "PYTHONPATH": str(src), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "fit.py", "session.json"], cwd=tmp_path, env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr
    results = json.loads(done.stdout.strip().splitlines()[-1])
    assert [r["language"] for r in results] == ["zh_CN", "en"]
    for result in results:
        x, y, w, h = result["frame"]
        ax, ay, aw, ah = result["area"]
        assert [aw, ah] == [width, height]
        assert ax <= x and ay <= y and x + w <= ax + aw and y + h <= ay + ah, result
        assert result["outside"] == [], result
        assert result["controls_bottom"] < result["footer_top"], result


def test_side_panel_collapses(make, session, app):
    view = make(session)
    view.show()
    QApplication.processEvents()
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


def test_summary_shows_the_exposure_time_when_the_view_has_it(make, session):
    session["candidates"][1]["exptime_s"] = 2.5
    session["candidates"][4]["exptime_s"] = None
    view = make(session)
    view.select_candidate("light_00002-1")
    assert view.summary.text().startswith("light_00002-1 · DSC_0002.NEF · 曝光 2.5 秒\n")
    view.select_candidate("light_00003-1")
    assert "DSC_0003.NEF · 曝光 10 秒\n" in view.summary.text()
    view.select_candidate("light_00004-1")
    assert "曝光" not in view.summary.text() and "DSC_0004.NEF\n" in view.summary.text()
    i18n.set_language("en")
    english = make(session)
    english.select_candidate("light_00003-1")
    assert "DSC_0003.NEF · exposure 10 s\n" in english.summary.text()


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
    manifest = trail_manifest.build(session, _job(tmp_path), workflow, zone=BEIJING)
    assert len(manifest["rows"]) == workflow["trail_review"]["kept_count"] == 1
    row = manifest["rows"][0]
    assert row["source_file"] == row["stacked_input"] == "DSC_0003.NEF" and row["frame_index"] == 3
    assert row["source_sha256"] == row["stacked_input_sha256"] == f"{3:064x}"
    assert row["capture_time_utc"] == "2026-12-14T22:00:50Z"
    assert row["capture_time_local"] == "2026-12-15T06:00:50+08:00"
    assert row["exposure_s"] == 10.0 and list(row) == list(trail_manifest.COLUMNS)
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (100., 30., 200., 30.)
    assert (row["identity"], row["identity_source"], row["keep_source"]) == ("meteor", "algorithm", "algorithm")
    header = manifest["header"]
    assert "time_zone" not in header and header["capture_time_basis"].startswith("FITS DATE-OBS (UT)")
    assert header["capture_time_local_zone"] == "UTC+08:00"
    assert header["added_back_by"] == {"algorithm": 1, "human": 0, "agent": 0}
    assert header["source_file_basis"] == {"import_record_of_this_session": 0, "stacked_input": 1}
    assert header["reference_frame"] == {"file": "DSC_0003.NEF", "stacked_input": "DSC_0003.NEF", "frame_index": 3,
                                         "capture_time_local": "2026-12-15T06:00:50+08:00",
                                         "capture_time_utc": "2026-12-14T22:00:50Z",
                                         "capture_time_from": "candidates_on_reference_frame"}
    assert header["segment"]["first_capture_time_utc"] == "2026-12-14T22:00:10Z"
    assert header["segment"]["first_capture_time_local"] == "2026-12-15T06:00:10+08:00"
    assert header["segment"]["last_capture_time_utc"] == "2026-12-14T22:00:50Z"
    assert header["segment"]["frames_with_capture_time"] == 3
    assert header["exposure"]["input_total_s"] == 50.0 and header["exposure"]["stacked_total_s"] == 40.0
    assert header["exposure"]["per_row"].startswith("exposure_s: the row's exposure time in seconds")
    assert header["format"] == "aester-composite-list-3"
    assert any(note.startswith("exposure_s 是每行那张曝光的时长") for note in header["notes"])
    assert header["not_scanned"] == {"count": 1, "frames": [{"frame_index": 5, "file": "DSC_0005.NEF",
                                                             "reason": "quality_gate",
                                                             "details": ["stars 0.34x median"]}]}
    assert header["trails_folder"] == "delivery-trails-1kept-0123456789ab"
    assert any("线性母版" in note and "不含流星" in note for note in header["notes"])
    assert any("世界时" in note and "差整小时" in note for note in header["notes"])
    assert not any("工作副本" in note for note in header["notes"])        # NEF inputs: no working copies


def test_manifest_names_the_originals_behind_dng_working_copies(session, tmp_path):
    """A conversion of this session (what raw_import.prepare_import returned) names the NEF the user shot."""
    job = _job(tmp_path)
    copies = tmp_path / "cache" / "raw-imports" / "key"
    for number, entry in enumerate(job["inputs"], 1):
        entry["path"] = str(copies / f"{number:04d}" / f"DSC_{number:04d}.dng")
        entry["sha256"] = f"d{number:063x}"
    job["diagnostics"]["reference_input"] = job["inputs"][2]["path"]
    for candidate in session["candidates"]:
        number = candidate["frame_index"]
        candidate.update(input=job["inputs"][number - 1]["path"], input_sha256=job["inputs"][number - 1]["sha256"])
    result = {"converted": True, "input_files": [entry["path"] for entry in job["inputs"]],
              "sources": [{"path": str(tmp_path / "shots" / f"DSC_{n:04d}.NEF"), "sha256": f"{n:064x}"}
                          for n in range(1, 6)],
              "outputs": [{"relative_path": f"{n:04d}\\DSC_{n:04d}.dng", "sha256": f"d{n:063x}"} for n in range(1, 6)]}
    originals = trail_manifest.import_sources(result)
    assert trail_manifest.import_sources({**result, "converted": False}) == {}
    workflow = _workflow(tmp_path, session, ["light_00003-1"])
    manifest = trail_manifest.build(session, job, workflow, originals=originals)
    row = manifest["rows"][0]
    assert (row["source_file"], row["source_sha256"]) == ("DSC_0003.NEF", f"{3:064x}")
    assert (row["stacked_input"], row["stacked_input_sha256"]) == ("DSC_0003.dng", f"d{3:063x}")
    header = manifest["header"]
    assert header["reference_frame"]["file"] == "DSC_0003.NEF" and header["reference_frame"]["stacked_input"] == "DSC_0003.dng"
    assert header["not_scanned"]["frames"][0]["file"] == "DSC_0005.NEF"
    assert header["source_file_basis"] == {"import_record_of_this_session": 1, "stacked_input": 0}
    assert not any("工作副本" in note for note in header["notes"])
    # Without that record (a restart), or for a DNG that changed since: the stacked DNG, said plainly.
    for known in ({}, {key: {**value, "input_sha256": "0" * 64} for key, value in originals.items()}):
        manifest = trail_manifest.build(session, job, workflow, originals=known)
        row = manifest["rows"][0]
        assert (row["source_file"], row["source_sha256"]) == ("DSC_0003.dng", f"d{3:063x}")
        assert any("工作副本" in note and "主干相同" in note for note in manifest["header"]["notes"])
    text = trail_manifest.csv_text(trail_manifest.build(session, job, workflow, originals=originals))
    assert not DRIVE.search(text) and "raw-imports" not in text and "shots" not in text


def test_manifest_exposure_column_is_empty_when_the_view_has_no_usable_time(session, tmp_path):
    def exported(value):
        session["candidates"][3]["exptime_s"] = value
        manifest = trail_manifest.build(session, _job(tmp_path), _workflow(tmp_path, session, ["light_00003-1"]))
        table = [line for line in trail_manifest.csv_text(manifest).splitlines() if not line.startswith("#")]
        return manifest["rows"][0]["exposure_s"], list(csv.DictReader(table))[0]["exposure_s"]

    for value in (None, "10.", 0.0, -1.0, float("nan"), float("inf"), True):
        assert exported(value) == (None, ""), value
    assert exported(30) == (30, "30") and exported(2.5) == (2.5, "2.5")


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
    nothing_kept = {**workflow, "delivery_dir": workflow["base_delivery_dir"],      # add-back found nothing
                    "trail_review": {**workflow["trail_review"], "kept_ids": [], "kept_count": 0}}
    for broken in ({**workflow, "trail_review": {**workflow["trail_review"], "revision": {"id": "older"}}},
                   {**workflow, "delivery_dir": workflow["base_delivery_dir"]}, None, nothing_kept,
                   {**nothing_kept, "delivery_dir": workflow["delivery_dir"]}):
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
    # Read as a spreadsheet does: each header entry is two cells (its JSON is not split at the commas),
    # then the table header and one row.
    records = list(csv.reader(io.StringIO(csv_text)))
    head = [record for record in records if record[0].startswith("# ")]
    assert len(head) == len(manifest["header"]) and all(len(record) == 2 for record in head)
    assert dict(head)["# capture_time_basis"].startswith("FITS DATE-OBS (UT)")
    assert json.loads(dict(head)["# reference_frame"])["file"] == "DSC_0003.NEF"
    assert records[len(head)] == list(trail_manifest.COLUMNS) and len(records) == len(head) + 2


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
    # The file dialog asks about the CSV; the JSON beside it is replaced only after asking too.
    asked, before = [], written[1].read_bytes()
    assert view.export_manifest(out / "geminids.csv", confirm_overwrite=lambda name: asked.append(name) or False) is None
    assert asked == ["geminids.json"] and written[1].read_bytes() == before
    assert view.export_manifest(out / "geminids.csv", confirm_overwrite=lambda name: True) == written


def test_export_passes_this_sessions_originals_and_time_zone(make, session, tmp_path, monkeypatch):
    from starstack import core
    workflow = _workflow(tmp_path, session, ["light_00003-1"])
    monkeypatch.setattr(core, "load_job", lambda directory: _job(tmp_path))
    seen = []
    real = trail_manifest.build
    monkeypatch.setattr(trail_manifest, "build", lambda *a, **k: seen.append(k) or real(*a, **k))
    originals = {"key": {"file": "DSC_0003.NEF", "sha256": "1" * 64, "input_sha256": "2" * 64}}
    view = make(session, workflow=workflow, originals=originals)
    (tmp_path / "lists").mkdir()
    assert view.export_manifest(tmp_path / "lists" / "x.csv")
    assert seen == [{"originals": originals, "zone": BEIJING}]


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


# ---- everything the current filter shows ---------------------------------------------------------

def test_bulk_choice_needs_a_filter_asks_first_and_undoes_in_one_step(make, session):
    view = make(session)
    asked = []
    assert not view.bulk_button.isEnabled()
    assert not view.bulk_choice("keep", confirm=lambda *a: asked.append(a) or True) and not asked
    view.set_priority_filter("high")
    assert view.bulk_button.isEnabled()
    assert not view.bulk_choice("keep", confirm=lambda *a: asked.append(a) or False)       # declined
    assert asked == [("keep", 2, 0)] and view.decisions == {}
    assert view.bulk_choice("keep", confirm=lambda *a: True)
    assert view.decisions == {"light_00002-1": {"override": "keep"}, "light_00004-1": {"override": "keep"}}
    assert "2 条" in view.status.text()
    # Everything not added back, set in one step; one undo restores it.
    view.set_priority_filter(None)
    view.set_keep_filter("omit")
    assert sorted(view.visible_ids()) == ["light_00001-1", "light_00002-2"]
    assert view.bulk_choice("omit", confirm=lambda *a: True)
    assert {cid for cid, d in view.decisions.items() if d.get("override") == "omit"} == set(view.visible_ids())
    view.undo()
    assert view.decisions == {"light_00002-1": {"override": "keep"}, "light_00004-1": {"override": "keep"}}
    view.undo()
    assert view.decisions == {}


def test_bulk_choice_leaves_rows_you_just_moved_out_of_the_filter(make, session):
    """A row changed a moment ago stays shown (pinned) but is not part of the filter any more."""
    view = make(session)
    view.set_keep_filter("omit")
    view.select_candidate("light_00002-1")
    QTest.keyClick(view.list, Qt.Key.Key_K)                  # kept now: shown, but no longer 不加回
    assert "light_00002-1" in view.visible_ids()
    asked = []
    assert view.bulk_choice("omit", confirm=lambda *a: asked.append(a) or True)
    assert asked == [("omit", 3, 1)]
    assert view.decisions["light_00002-1"] == {"override": "keep"}
    assert {cid for cid, d in view.decisions.items() if d.get("override") == "omit"} == {
        "light_00001-1", "light_00002-2", "light_00004-1"}
    # Relabelled as a satellite inside the 未知 filter: keeping everything unknown leaves it out.
    view.undo()
    view.undo()
    view.set_keep_filter(None)
    view.set_filter("unknown")
    view.select_candidate("light_00004-1")
    view.label_choice.setCurrentIndex(view.label_choice.findData("satellite"))
    view.set_selected_decisions()
    assert view.bulk_choice("keep", confirm=lambda *a: True)
    assert "override" not in view.decisions["light_00004-1"]
    assert view.decisions["light_00001-1"] == {"override": "keep"}
    # Only pinned rows left in view: nothing to set in one step.
    view.set_keep_filter("keep")
    view.set_filter(None)
    for cid in list(view.visible_ids()):
        view.select_candidate(cid)
        QTest.keyClick(view.list, Qt.Key.Key_O)
    assert view.visible_ids() and not view.bulk_button.isEnabled()


def test_bulk_confirmation_mentions_rows_left_out(make, session, app):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox
    view = make(session)
    seen = []

    def answer():
        box = next(w for w in QApplication.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible())
        seen.append(box.text())
        box.defaultButton().click()
    QTimer.singleShot(0, answer)
    assert not view._confirm_bulk("omit", 3, 1)
    assert "另有 1 条刚改过、已不在此筛选，不会改动" in seen[0]


def test_bulk_confirmation_defaults_to_cancel(make, session, monkeypatch, app):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox
    view = make(session)
    view.set_priority_filter("high")
    seen = []

    def answer():
        box = next(w for w in QApplication.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible())
        seen.append((box.defaultButton().text(), [b.text() for b in box.buttons()], box.text()))
        box.defaultButton().click()
    QTimer.singleShot(0, answer)
    assert not view.bulk_choice("keep")
    assert seen and seen[0][0] == "取消" and "全部保留" in seen[0][1] and "飞机或卫星" in seen[0][2]
    assert view.decisions == {}


# ---- the scan's suggestion reasons in Chinese --------------------------------------------------

REASONS = [
    ("geometrically compatible segment in adjacent frame light_00044 (angle 0.6°, lateral 0.8 px, gap 113 px); "
     "same-object identity unverified",
     "相邻曝光第44张里有一段方向和位置都接得上（夹角 0.6°，横向偏 0.8 像素，相隔 113 像素）；是否同一物体未核实"),
    ("light curve: start faded (0.20 of the peak), stop between (0.38 of the peak)",
     "亮度变化：起点渐隐（峰值的 0.20），终点介于渐隐和中断之间（峰值的 0.38）"),
    ("light curve: start unjudged, stop unjudged", "亮度变化：起点无法判断，终点无法判断"),
    ("single frame candidate; no compatible continuation was linked in the scanned adjacent-frame catalog "
     "(light_00002, light_00004); original-exposure absence is not established; no periodic modulation, total S/N 60.7",
     "只在本张出现：已扫描的相邻曝光（第2张、第4张）里没有接得上的一段，但不能断定原片里没有；没有周期性明暗，总信噪比 60.7"),
    ("single frame candidate; no compatible continuation was linked in the scanned adjacent-frame catalog "
     "(none available); original-exposure absence is not established; no periodic modulation, total S/N 9.0",
     "只在本张出现：已扫描的相邻曝光（无）里没有接得上的一段，但不能断定原片里没有；没有周期性明暗，总信噪比 9.0"),
    ("same place in 3 frames: light_00063, light_00070, light_00090", "在 3 张曝光的同一位置都有：第63张、第70张、第90张"),
    ("not measured (too_short)", "未能测量（太短）"),
    ("total S/N 6.2 below 8", "总信噪比 6.2，低于 8，太弱，不判断身份"),
]


@pytest.mark.parametrize("english,expected", REASONS)
def test_suggestion_reasons_are_shown_in_chinese_and_kept_in_english(english, expected):
    assert i18n.get_language() == "zh_CN"
    assert dialog_module.suggestion_text(english) == expected
    i18n.set_language("en")
    assert dialog_module.suggestion_text(english) == english


def test_unknown_reason_wording_is_shown_as_written_and_the_tooltip_uses_chinese(make, session):
    assert dialog_module.suggestion_text("a future reason (x 1)") == "a future reason (x 1)"
    session["candidates"][1]["suggestion"]["reasons"] = [REASONS[0][0], "a future reason"]
    view = make(session)
    tip = view._items["light_00002-1"].toolTip(1)
    assert REASONS[0][1] in tip and "a future reason" in tip and "geometrically" not in tip
    view.select_candidate("light_00002-1")           # the side panel lists the same reasons
    assert view.reasons.text().startswith("算法建议：未知") and REASONS[0][1] in view.reasons.text()
    assert session["candidates"][1]["suggestion"]["reasons"][0] == REASONS[0][0]      # the record is unchanged


def test_plain_words_for_omit_in_the_dialog(make, session):
    session["candidates"][0]["recall_review"]["message"] = "已自动省略，但身份尚未核实，不能视为已排除流星。"
    view = make(session)
    assert view._items["light_00001-1"].text(2) == "默认不加回"
    assert view.action_choice.itemText(view.action_choice.findData("omit")) == "不加回"
    assert view.list.headerItem().text(2) == "是否加回"
    view.select_candidate("light_00001-1")
    assert "已自动不加回，但身份尚未核实" in view.summary.text() and "省略" not in view.summary.text()
    assert view.keys_hint.text() == "K\u00a0保留 · O\u00a0不加回\nU\u00a0恢复默认 · 回车\u00a0下一条"


def test_older_records_say_omit_plainly_in_chinese_and_omit_in_english(make, session):
    """Explicit-keep-only sessions carry the engine's 默认省略 / 明确省略; the record keeps them."""
    session["application_policy_version"] = None
    session["candidates"][0]["review_disposition"] = "默认省略"
    session["candidates"][1].update(review_disposition="明确省略", override="omit")
    view = make(session)
    assert view._items["light_00001-1"].text(2) == "默认不加回"
    assert view._items["light_00002-1"].text(2) == "明确不加回"
    view.select_candidate("light_00002-1")
    assert "明确不加回" in view.summary.text() and "省略" not in view.summary.text()
    assert session["candidates"][0]["review_disposition"] == "默认省略"
    i18n.set_language("en")
    english = make(session)
    assert english._items["light_00001-1"].text(2) == "Omit by default"
    assert english.action_choice.itemText(english.action_choice.findData("omit")) == "Omit"
    assert english.keep_buttons["omit"].text().startswith("Not added back")
    english.select_candidate("light_00001-1")
    QTest.keyClick(english.list, Qt.Key.Key_O)
    assert english._items["light_00001-1"].text(3) == "Omit"
    assert english.keys_hint.text().replace("\u00a0", " ") == "K keep · O omit\nU reset · Enter next"
    english.set_filter("unknown")
    assert english.bulk_choice("omit", confirm=lambda *a: True) and "“Omit”" in english.status.text()
