import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from aester_desktop import trail_review_dialog as dialog_module
from aester_desktop.trail_review_dialog import TrailReviewDialog, TrailLocationView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session():
    return {"session_path": "/job/review/draft.json", "draft_revision": "draft-1", "base_revision": {"id": "rev1"},
            "geometry": {"width": 1000, "height": 600}, "decisions": {}, "preview_path": None,
            "candidates": [{"id": f"trail-{i}", "frame_index": i, "suggestion": {"class": "unknown"},
                            "decision": "omit", "effective_label": "unknown", "input": f"frame{i}.nef",
                            "endpoints_native_reference": [[100., i*100.], [200., i*100.]]} for i in (1, 2)]}


@pytest.fixture
def dialog(app, session, monkeypatch):
    monkeypatch.setattr(dialog_module.service, "candidate_context", lambda *a: {"panels": [], "notice": "test context"})
    value = TrailReviewDialog(session)
    yield value
    value._dirty = False
    value._finish_workers()
    value.reject()
    app.processEvents()


def test_batch_unknown_keep_undo_and_clear(dialog):
    for item in dialog._items.values():
        item.setSelected(True)
    dialog.label_choice.setCurrentIndex(dialog.label_choice.findData("unknown"))
    dialog.action_choice.setCurrentIndex(dialog.action_choice.findData("keep"))
    dialog.set_selected_decisions()
    assert dialog.decisions == {"trail-1": {"label": "unknown", "override": "keep"},
                                "trail-2": {"label": "unknown", "override": "keep"}}
    dialog.undo()
    assert dialog.decisions == {}
    dialog.set_selected_decisions()
    dialog.select_candidate("trail-1")
    dialog.clear_selected_decisions()
    assert set(dialog.decisions) == {"trail-2"}
    dialog.undo()
    assert set(dialog.decisions) == {"trail-1", "trail-2"}


def test_classification_can_change_without_action(dialog):
    dialog.select_candidate("trail-1")
    dialog.label_choice.setCurrentIndex(dialog.label_choice.findData("meteor"))
    dialog.set_selected_decisions()
    assert dialog.decisions == {"trail-1": {"label": "meteor"}}
    assert "override" not in dialog.decisions["trail-1"]


def test_explicit_reset_none_is_preserved(dialog):
    dialog.select_candidate("trail-1")
    dialog.label_choice.setCurrentIndex(dialog.label_choice.count()-1)
    dialog.action_choice.setCurrentIndex(dialog.action_choice.count()-1)
    dialog.set_selected_decisions()
    assert dialog.decisions == {"trail-1": {"label": None, "override": None}}


def test_save_human_origin_and_optimistic_draft_then_apply_request_only(dialog, monkeypatch):
    calls = []
    def save(path, decisions, **kwargs):
        calls.append((path, copy.deepcopy(decisions), kwargs))
        result = copy.deepcopy(dialog.session)
        result.update(draft_revision="draft-2", decisions=copy.deepcopy(decisions))
        return result
    monkeypatch.setattr(dialog_module.service, "save_review", save)
    monkeypatch.setattr(dialog_module.service, "apply_review", lambda *a, **k: pytest.fail("dialog must not publish/recompose"))
    dialog.select_candidate("trail-1")
    dialog.action_choice.setCurrentIndex(dialog.action_choice.findData("keep"))
    dialog.set_selected_decisions()
    emitted = []
    dialog.applyRequested.connect(emitted.append)
    dialog.request_apply()
    assert len(calls) == 1
    assert calls[0][2] == {"reviewer": "human", "expected_draft_revision": "draft-1"}
    assert dialog.apply_session_path == "/job/review/draft.json"
    assert dialog.apply_draft_revision == "draft-2"
    assert emitted == [dialog.apply_session_path]
    assert dialog.result() == dialog.DialogCode.Accepted


def test_stale_save_does_not_close_or_request_apply(dialog, monkeypatch):
    def stale(*args, **kwargs):
        raise ValueError("草稿已在另一窗口更改")
    monkeypatch.setattr(dialog_module.service, "save_review", stale)
    emitted = []
    dialog.applyRequested.connect(emitted.append)
    dialog.request_apply()
    assert not emitted and dialog.apply_session_path is None
    assert "另一窗口" in dialog.status.text()


def test_open_existing_pending_intent_is_restored(app, session, monkeypatch):
    monkeypatch.setattr(dialog_module.service, "candidate_context", lambda *a: {"panels": []})
    session["decisions"] = {"trail-2": {"label": "unknown", "override": "keep"}}
    value = TrailReviewDialog(session)
    try:
        assert value.decisions == session["decisions"]
        assert "未知" in value._items["trail-2"].text(3)
        assert "保留" in value._items["trail-2"].text(3)
        assert not value._dirty
    finally:
        value._finish_workers()
        value.reject()


def test_canvas_coordinate_mapping_and_hit_test(app, session):
    canvas = TrailLocationView(session)
    canvas.resize(760, 520)
    point = canvas._point([150., 100.])
    assert canvas.hit_test(point) == "trail-1"
    assert canvas.hit_test(canvas._point([150., 200.])) == "trail-2"
    assert canvas.hit_test(QPointF(0, 0)) is None
    rect = canvas._rect()
    first = canvas._point([0., 0.])
    assert first.x() == pytest.approx(rect.x() + rect.width()*.5/1000)
    assert first.y() == pytest.approx(rect.y() + rect.height()*.5/600)
    # Zoomed about a trail: the same pixel-centre mapping, and hit testing follows the view.
    canvas.zoom_to(4., point)
    assert canvas._point([150., 100.]).x() == pytest.approx(point.x())
    rect = canvas._rect()
    assert rect.width() == pytest.approx(canvas._bounds().width() * 4, rel=.01) or \
        rect.height() == pytest.approx(canvas._bounds().height() * 4, rel=.01)
    first = canvas._point([0., 0.])
    assert first.x() == pytest.approx(rect.x() + rect.width()*.5/1000)
    assert canvas.hit_test(canvas._point([199., 100.])) == "trail-1"
    assert canvas.hit_test(canvas._point([150., 200.])) == "trail-2"
    assert canvas.hit_test(canvas._point([150., 105.])) is None      # 5 image px: about 15 screen px at 4x
    canvas.fit_view()
    assert canvas._point([150., 100.]).x() == pytest.approx(point.x()) and canvas.zoom == 1.


def test_stale_context_does_not_replace_current_selection(dialog):
    dialog.context_notice.setText("current")
    dialog._context_ready(dialog._context_token-1, {"notice": "stale", "panels": []})
    assert dialog.context_notice.text() == "current"
