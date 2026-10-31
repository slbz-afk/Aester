"""Trail review fixes (2026-09-27, items 8, 9 display, 10, 13 UI).

Identity column with provenance dots, filter by identity linked with the preview, identity
colours with kept/omitted line styles, a zoomable preview whose selection is a yellow ring
(hover never moves the selection), and the trail-layer export option passed to save_review.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import itertools
import re

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from starstack import i18n, overlay_colors as oc
from aester_desktop import themes, trail_review_dialog as trd
from aester_desktop.trail_review_dialog import DOT_ROLE, TrailLocationView, TrailReviewDialog
from test_trail_review_dialog import session  # noqa: F401 - shared fixture
from test_trail_review_session import job_directory  # noqa: F401 - shared fixture
from test_ui_foundation import _contrast, _de2000, _simulated_lab

CJK = re.compile(r'[㐀-鿿＀-￯　-〿]')
SURROUND = (0x1f, 0x1f, 0x1f)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_language():
    previous = i18n.get_language()
    i18n.set_language('zh_CN')
    yield
    i18n.set_language(previous)


@pytest.fixture
def policy_session(session):
    """A session from the new backend (contract D): meteor default keep, per-candidate fields."""
    value = copy.deepcopy(session)
    value.update(application_policy_version="meteor-default-keep-1", options={"export_trail_layer": False})
    first, second = value["candidates"]
    first.update(suggestion={"class": "meteor", "reasons": ["both measured ends fade inside the frame"]},
                 effective_label="meteor", identity="meteor", identity_source="algorithm", status="measured",
                 measurement_quality={}, default_keep=True, effective_keep=True, keep_source="algorithm")
    second.update(status="measured", measurement_quality={}, identity="unknown", identity_source="algorithm",
                  default_keep=False, effective_keep=False, keep_source=None)
    return value


@pytest.fixture
def make(app, monkeypatch):
    monkeypatch.setattr(trd.service, "candidate_context", lambda *a: {"panels": [], "notice": "context"})
    created = []

    def build(value):
        view = TrailReviewDialog(value)
        created.append(view)
        return view
    yield build
    for view in created:
        view._dirty = False
        view.hide()
        view._finish_workers()
        view.reject()
    app.processEvents()


def _rgb(image):
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    stride = image.bytesPerLine()
    data = np.frombuffer(image.constBits(), np.uint8, count=stride * image.height()).reshape(image.height(), stride)
    return data[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


def _mask(pixels, color, tolerance=3):
    return np.all(np.abs(pixels.astype(int) - np.array(color)) <= tolerance, axis=-1)


def _relabel(view, cid, label=None, action="unchanged"):
    view.select_candidate(cid)
    view.label_choice.setCurrentIndex(view.label_choice.findData(label) if label != "unchanged" else 0)
    view.action_choice.setCurrentIndex(view.action_choice.findData(action))
    view.set_selected_decisions()


def _move(widget, point):
    widget.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(point), QPointF(point),
                                      Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))


def _mouse(widget, kind, point, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    event = QMouseEvent(kind, QPointF(point), QPointF(point), button,
                        button if kind != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton, modifiers)
    {QEvent.Type.MouseButtonPress: widget.mousePressEvent, QEvent.Type.MouseMove: widget.mouseMoveEvent,
     QEvent.Type.MouseButtonRelease: widget.mouseReleaseEvent,
     QEvent.Type.MouseButtonDblClick: widget.mouseDoubleClickEvent}[kind](event)


def _wheel(widget, point, steps=1):
    widget.wheelEvent(QWheelEvent(QPointF(point), QPointF(point), QPoint(), QPoint(0, 120 * steps),
                                  Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                  Qt.ScrollPhase.NoScrollPhase, False))


# ---- item 8: identity column, provenance dot, filter -------------------------------------------

def test_identity_column_follows_the_draft_with_a_zhu_dot_and_undo(app, make, session):
    view = make(session)
    view.show(); app.processEvents()
    assert view.list.headerItem().text(1) == '身份'
    item = view._items['trail-1']
    assert item.text(1) == '未知' and item.data(1, DOT_ROLE) == 'algorithm'
    zhu = themes.COLORS['dark']['zhu']
    zhu_rgb = tuple(int(zhu[i:i + 2], 16) for i in (1, 3, 5))
    assert not _mask(_rgb(view.list.grab().toImage()), zhu_rgb, 2).any()
    _relabel(view, 'trail-1', 'meteor')
    app.processEvents()
    assert item.text(1) == '流星' and item.data(1, DOT_ROLE) == 'human'
    assert '人工确定' in item.toolTip(1) and '算法建议：未知' in item.toolTip(1)
    assert view.location.styles['trail-1'] == ('meteor', False)       # legacy session: no default keep
    assert _mask(_rgb(view.list.grab().toImage()), zhu_rgb, 2).sum() >= 12   # the filled 朱 dot
    view.undo()
    app.processEvents()
    assert item.text(1) == '未知' and item.data(1, DOT_ROLE) == 'algorithm'
    assert view.location.styles['trail-1'] == ('unknown', False)
    assert not _mask(_rgb(view.list.grab().toImage()), zhu_rgb, 2).any()


def test_agent_draft_identity_shows_xiang_and_becomes_human_when_changed(app, make, session):
    draft = copy.deepcopy(session)
    draft.update(reviewer='agent', decisions={'trail-2': {'label': 'satellite'}})
    view = make(draft)
    item = view._items['trail-2']
    assert item.text(1) == '卫星' and item.data(1, DOT_ROLE) == 'agent'
    _relabel(view, 'trail-2', 'unchanged', 'omit')
    assert item.data(1, DOT_ROLE) == 'agent'           # the label is still the Agent's
    assert item.data(2, DOT_ROLE) == 'human' and item.text(2) == '明确省略'
    _relabel(view, 'trail-2', 'aircraft')
    assert item.text(1) == '飞机' and item.data(1, DOT_ROLE) == 'human'


def test_meteor_default_keep_is_shown_and_follows_identity(app, make, policy_session):
    view = make(policy_session)
    first, second = view._items['trail-1'], view._items['trail-2']
    assert first.text(2) == '默认保留（算法）' and first.data(2, DOT_ROLE) == 'algorithm'
    assert second.text(2) == '默认省略' and second.data(2, DOT_ROLE) == 'algorithm'
    assert view.location.styles == {'trail-1': ('meteor', True), 'trail-2': ('unknown', False)}
    assert '默认保留' in view.findChildren(trd.QLabel)[0].text()
    assert view.action_choice.itemText(view.action_choice.count() - 1) == '恢复默认（按身份）'
    _relabel(view, 'trail-1', 'satellite')                  # relabelling drops the default keep
    assert first.text(2) == '默认省略' and view.location.styles['trail-1'] == ('satellite', False)
    assert first.data(2, DOT_ROLE) == 'human'               # the omission follows your identity
    _relabel(view, 'trail-2', 'meteor')                     # relabelling as meteor adds it
    assert second.text(2) == '默认保留（按改判身份）' and second.data(2, DOT_ROLE) == 'human'
    assert view.location.styles['trail-2'] == ('meteor', True)
    _relabel(view, 'trail-1', 'unchanged', 'keep')          # an explicit keep wins
    assert first.text(2) == '明确保留' and first.data(2, DOT_ROLE) == 'human'
    _relabel(view, 'trail-1', 'unchanged', None)            # reset: the default for a satellite
    assert first.text(2) == '默认省略' and view.location.styles['trail-1'] == ('satellite', False)
    assert view.decisions['trail-1'] == {'label': 'satellite', 'override': None}
    _relabel(view, 'trail-1', None)                          # clearing the label: the suggestion again
    assert first.text(1) == '流星' and first.data(1, DOT_ROLE) == 'algorithm'
    assert first.text(2) == '默认保留（算法）' and first.data(2, DOT_ROLE) == 'algorithm'


def test_untrustworthy_meteor_is_omitted_by_default_without_backend_fields(app, make, session):
    value = copy.deepcopy(session)
    value['application_policy_version'] = 'meteor-default-keep-1'
    value['candidates'][0].update(suggestion={'class': 'meteor'}, effective_label='meteor', status='measured',
                                  measurement_quality={'profile_boundary_hit': True})
    value['candidates'][1].update(suggestion={'class': 'meteor'}, effective_label='meteor', status='measured',
                                  measurement_quality={})
    view = make(value)
    assert view._items['trail-1'].text(2) == '默认省略（测量不可靠）'
    assert view._items['trail-2'].text(2) == '默认保留（算法）'
    assert view.location.styles == {'trail-1': ('meteor', False), 'trail-2': ('meteor', True)}


def test_candidate_state_rules():
    candidate = {'id': 'a', 'suggestion': {'class': 'unknown'}, 'status': 'measured', 'label': 'meteor'}
    state = trd.candidate_state(candidate, {}, True)
    assert state['identity'] == 'meteor' and state['identity_source'] == 'human' and state['keep']
    state = trd.candidate_state(candidate, {'label': None}, True)          # clear the manual label
    assert state['identity'] == 'unknown' and state['identity_source'] == 'algorithm' and not state['keep']
    state = trd.candidate_state({**candidate, 'override': 'omit'}, {'label': 'meteor'}, True)
    assert not state['keep'] and state['keep_source'] == 'human'          # a recorded omit still wins
    state = trd.candidate_state({**candidate, 'status': 'failed'}, {}, True)
    assert not state['keep']                                                # unmeasured is never added
    assert not trd.candidate_state(candidate, {}, False)['keep']            # old sessions: explicit only
    assert trd.candidate_state(candidate, {'override': 'keep'}, False, {'override': 'agent'})['keep_source'] == 'agent'
    state = trd.candidate_state({**candidate, 'status': 'failed'}, {'override': 'keep'}, True)
    assert not state['keep'] and trd.disposition_text(state, True) == '明确保留（未测量，不会加回）'
    # Every disposition the display can produce has English text.
    catalog = i18n.english_catalog()
    shown = {trd.disposition_text(trd.candidate_state({**candidate, 'status': status, 'label': label}, decision, policy), policy)
             for status in ('measured', 'failed') for label in (None, 'meteor', 'satellite') for policy in (True, False)
             for decision in ({}, {'override': 'keep'}, {'override': 'omit'}, {'label': 'meteor'}, {'label': None})}
    assert len(shown) >= 6 and all(text in catalog for text in shown), shown - set(catalog)


def test_filter_hides_rows_preview_lines_and_hit_test(app, make, session):
    view = make(session)
    view.show(); app.processEvents()
    _relabel(view, 'trail-1', 'meteor')
    assert view.filter_buttons['meteor'].text() == '流星 1' and view.filter_buttons[None].text() == '全部 2'
    assert view.revised_button.text() == '已改判 1'
    for item in view._items.values():
        item.setSelected(True)
    location = view.location
    b2 = location._point([150., 200.])
    x0, x1 = int(location._point([100., 200.]).x()), int(location._point([200., 200.]).x())
    # The line and its selection ring (padding >= 8 px) both lie in this box.
    region = lambda image: _rgb(image)[int(b2.y()) - 14:int(b2.y()) + 15, x0 - 14:x1 + 15]
    assert not np.all(region(location.grab().toImage()) == SURROUND)
    view.filter_buttons['meteor'].click()
    app.processEvents()
    assert view._items['trail-2'].isHidden() and not view._items['trail-2'].isSelected()
    assert view.selected_ids() == ['trail-1'] and location.selected_ids == {'trail-1'}
    assert location.visible_ids == {'trail-1'} and location.hit_test(b2) is None
    assert location.hit_test(location._point([150., 100.])) == 'trail-1'
    assert np.all(region(location.grab().toImage()) == SURROUND)          # the hidden line is not drawn
    assert view.filter_count.text() == '显示 1 / 2 条'
    view.set_filter(None)
    assert not view._items['trail-2'].isHidden() and location.visible_ids is None
    view.revised_button.setChecked(True)
    assert view.visible_ids() == ['trail-1']
    view.revised_button.setChecked(False)
    assert view.visible_ids() == ['trail-1', 'trail-2']


def test_revised_rows_stay_visible_until_the_filter_changes(app, make, session):
    view = make(session)
    view.set_filter('unknown')
    _relabel(view, 'trail-2', 'satellite')
    assert not view._items['trail-2'].isHidden() and view._items['trail-2'].text(1) == '卫星'
    assert view.filter_buttons['unknown'].text() == '未知 1'
    view.set_filter('unknown')
    assert view._items['trail-2'].isHidden() and view.visible_ids() == ['trail-1']
    view.set_filter('artifact')
    assert view.visible_ids() == [] and view.list.currentItem() is None
    assert view.summary.text() == '当前筛选下没有候选。'
    view.set_selected_decisions()                           # nothing visible: nothing changes
    assert view.decisions == {'trail-2': {'label': 'satellite'}}


def test_real_review_view_layers_saved_drafts_and_round_trips_the_option(app, make, job_directory):
    # The backend's identity/keep fields include a saved draft (contract D); clearing that
    # draft choice here must return to the recorded state, not stay on the draft's value.
    from starstack import trail_review_session as service
    opened = service.open_review(str(job_directory))
    assert opened['application_policy_version'] == 'meteor-default-keep-1'
    first, second = (c['id'] for c in opened['candidates'])
    drafted = service.save_review(opened['session_path'], {first: {'label': 'meteor'}}, reviewer='agent',
                                  expected_draft_revision=opened['draft_revision'])
    view = make(drafted)
    item = view._items[first]
    assert item.text(1) == '流星' and item.data(1, DOT_ROLE) == 'agent'
    assert item.text(2) == '默认保留（按改判身份）' and item.data(2, DOT_ROLE) == 'agent'
    assert view.location.styles[first] == ('meteor', True)
    view.select_candidate(first)
    caption, old, arrow, new = view.revision_rows['label']
    assert old.text() == '未知' and new.text() == '流星' and view.revision_dot.kind() == 'agent'
    view.clear_selected_decisions()
    assert item.text(1) == '未知' and item.data(1, DOT_ROLE) == 'algorithm' and item.text(2) == '默认省略'
    view.undo()
    view.export_layer.setChecked(True)
    assert view.save()
    stored = service.read_review(view.session['session_path'])
    assert stored['options'] == {'export_trail_layer': True} and stored['reviewer'] == 'human'
    assert stored['decisions'] == {first: {'label': 'meteor'}}
    assert item.data(1, DOT_ROLE) == 'human' and view.export_layer.isChecked() and not view._dirty
    for candidate in stored['candidates']:     # the display agrees with the backend's own view
        state = view._states[candidate['id']]
        assert (state['identity'], state['identity_source'], state['keep']) ==             (candidate['identity'], candidate['identity_source'], candidate['effective_keep'])


# ---- item 8 colours / item 9 display: identity colours, kept solid, omitted dotted -------------

def test_trail_identity_colours_stay_distinct_under_colour_vision_deficiency():
    colours = [spec['hex'] for spec in oc.TRAIL_IDENTITY.values()] + [oc.TRAIL_SELECTED]
    assert set(oc.TRAIL_IDENTITY) == set(trd.LABELS) and len(set(colours)) == 6
    for vision in ('normal', 'protan', 'deutan', 'tritan'):
        worst = min(_de2000(_simulated_lab(a, vision), _simulated_lab(b, vision))
                    for a, b in itertools.combinations(colours, 2))
        assert worst >= 10, (vision, round(worst, 1))
    for spec in oc.TRAIL_IDENTITY.values():
        assert spec['rgb'] == tuple(int(spec['hex'][i:i + 2], 16) for i in (1, 3, 5))
        assert _contrast(spec['hex'], '#0d1118') >= 4.5          # readable on a night sky
    assert (oc.TRAIL_KEPT_LINE, oc.TRAIL_OMITTED_LINE) == (oc.SOLID, oc.DOT)


def test_kept_trails_are_solid_and_omitted_trails_dotted(app, session):
    view = TrailLocationView(session)
    view.resize(760, 520)
    view.set_styles({'trail-1': ('meteor', True), 'trail-2': ('satellite', False)})
    shown = _rgb(view.grab().toImage())
    runs = {}
    for cid, y in (('trail-1', 100.), ('trail-2', 200.)):
        a, b = view._point([100., y]), view._point([200., y])
        colour = oc.TRAIL_IDENTITY['meteor' if cid == 'trail-1' else 'satellite']['rgb']
        rows = shown[int(a.y()) - 2:int(a.y()) + 3, int(a.x()) + 4:int(b.x()) - 4]
        hits = _mask(rows, colour, 40).any(axis=0)
        assert hits.any(), cid
        runs[cid] = hits.mean()
    assert runs['trail-1'] > .95                  # solid: the colour runs along the whole line
    assert .15 < runs['trail-2'] < .8             # dotted: dots and dark gaps alternate


# ---- item 10: selection ring, hover, zoom ------------------------------------------------------

def _ring_around(view, pixels, y, x0, x1):
    a, b = view._point([x0, y]), view._point([x1, y])
    pad = trd.ring_padding(b.x() - a.x())
    handle = _mask(pixels, oc.HANDLE_RGB, 3)
    mid = int((a.x() + b.x()) / 2)
    column, row = handle[:, mid], handle[int(round(a.y())), :]
    above = column[int(a.y() - pad - 3):int(a.y() - 2)].any()
    below = column[int(a.y() + 2):int(a.y() + pad + 4)].any()
    left = row[int(a.x() - pad - 3):int(a.x() - 1)].any()
    right = row[int(b.x() + 1):int(b.x() + pad + 4)].any()
    inside = handle[int(a.y() - pad + 3):int(a.y() + pad - 2), int(a.x()):int(b.x())].any()
    return above and below and left and right and not inside


def test_selected_trail_becomes_a_ring_that_encloses_it(app, session):
    view = TrailLocationView(session)
    view.resize(760, 520)
    view.set_styles({'trail-1': ('meteor', True), 'trail-2': ('meteor', True)})
    view.set_candidate('trail-1')
    shown = _rgb(view.grab().toImage())
    assert _ring_around(view, shown, 100., 100., 200.)
    assert not _ring_around(view, shown, 200., 100., 200.)
    a, b = view._point([100., 100.]), view._point([200., 100.])
    middle = shown[int(round(a.y())), int(a.x()) + 5:int(b.x()) - 5]
    # The selected line itself is replaced by the ring, leaving the trail's pixels visible.
    assert not _mask(middle[None], oc.TRAIL_RGB, 40).any()
    view.set_selection({'trail-1', 'trail-2'}, 'trail-1')
    shown = _rgb(view.grab().toImage())
    assert _ring_around(view, shown, 100., 100., 200.) and _ring_around(view, shown, 200., 100., 200.)


def test_hover_thickens_without_ring_and_never_moves_the_selection(app, make, session):
    view = make(session)
    view.show(); app.processEvents()
    view.select_candidate('trail-1')
    location = view.location
    before = _rgb(location.grab().toImage())
    _move(location, location._point([150., 200.]))
    assert location.hover_id == 'trail-2' and '候选2' in location.toolTip()
    assert view.list.currentItem() is view._items['trail-1'] and location.selected_ids == {'trail-1'}
    hovered = _rgb(location.grab().toImage())
    assert not _ring_around(location, hovered, 200., 100., 200.)
    b = location._point([150., 200.])
    band = lambda pixels: (pixels[int(b.y()) - 4:int(b.y()) + 5] != SURROUND).any(axis=-1).sum()
    assert band(hovered) > band(before)                        # the hovered line is thicker
    view.list.itemEntered.emit(view._items['trail-2'], 0)       # list hover: highlight only
    assert view.list.currentItem() is view._items['trail-1'] and location.hover_id == 'trail-2'
    location.leaveEvent(QEvent(QEvent.Type.Leave))
    assert location.hover_id is None and location.toolTip() == ''


def test_preview_click_selects_and_ctrl_click_extends(app, make, session):
    view = make(session)
    location = view.location
    location.resize(760, 520)
    _mouse(location, QEvent.Type.MouseButtonPress, location._point([150., 200.]))
    assert view.selected_ids() == ['trail-2'] and view.list.currentItem() is view._items['trail-2']
    _mouse(location, QEvent.Type.MouseButtonPress, location._point([150., 100.]),
           modifiers=Qt.KeyboardModifier.ControlModifier)
    assert sorted(view.selected_ids()) == ['trail-1', 'trail-2'] and location.selected_ids == {'trail-1', 'trail-2'}


def test_wheel_zoom_about_cursor_pan_and_reset_keep_hit_test_mapping(app, session):
    view = TrailLocationView(session)
    view.resize(760, 520)
    anchor = view._point([150., 100.])
    _wheel(view, anchor)
    assert view.zoom == pytest.approx(1.25)
    assert view._point([150., 100.]).x() == pytest.approx(anchor.x()) and view._point([150., 100.]).y() == pytest.approx(anchor.y())
    _wheel(view, anchor, 8)
    assert view.zoom == pytest.approx(1.25 ** 9)
    rect = view._rect()
    point = view._point([150., 200.])
    assert point.x() == pytest.approx(rect.x() + rect.width() * 150.5 / 1000)
    assert view.hit_test(view._point([150., 100.])) == 'trail-1'
    assert view.hit_test(view._point([150., 100.]) + QPointF(0, 15)) is None   # 12 px is screen space
    # Dragging empty space pans; the mapping follows.
    start = view._point([500., 400.])
    assert view.hit_test(start) is None
    before = view._point([150., 100.])
    _mouse(view, QEvent.Type.MouseButtonPress, start)
    _mouse(view, QEvent.Type.MouseMove, start + QPointF(-30, 20))
    _mouse(view, QEvent.Type.MouseButtonRelease, start + QPointF(-30, 20))
    moved = view._point([150., 100.])
    assert (moved.x() - before.x(), moved.y() - before.y()) == (pytest.approx(-30), pytest.approx(20))
    assert view.hit_test(moved) == 'trail-1'
    _wheel(view, anchor, 40)
    assert view.zoom == view.MAX_ZOOM
    _wheel(view, anchor, -40)
    assert view.zoom == view.MIN_ZOOM
    _mouse(view, QEvent.Type.MouseButtonDblClick, view._point([500., 400.]))
    assert view.zoom == 1. and view.pan == QPointF()


def test_focus_selection_zooms_to_the_selected_trails(app, make, session):
    view = make(session)
    view.location.resize(760, 520)
    view.select_candidate('trail-2')
    view.focus_button.click()
    location = view.location
    assert location.zoom > 1.
    middle = (location._point([100., 200.]) + location._point([200., 200.])) / 2
    centre = location._bounds().center()
    assert abs(middle.x() - centre.x()) < 1 and abs(middle.y() - centre.y()) < 1
    assert view.zoom_label.text() == f"{round(location.zoom * 100)}%"
    view.fit_button.click()
    assert location.zoom == 1. and view.zoom_label.text() == '100%'


def test_zoomed_view_decodes_a_sharper_visible_region(app, session, tmp_path):
    path = tmp_path / "preview.png"
    x = np.linspace(0, 255, 2400, dtype=np.float32)
    pixels = np.stack([np.tile(x, (1500, 1)), np.tile(x[::-1], (1500, 1)), np.full((1500, 2400), 60, np.float32)], -1)
    Image.fromarray(pixels.astype(np.uint8)).save(path)
    value = copy.deepcopy(session)
    value.update(preview_path=str(path), geometry={'width': 2400, 'height': 1500})
    view = TrailLocationView(value)
    view.resize(600, 400)
    assert view.pixmap.width() == 1600
    view._load_detail()
    assert view._detail is None                    # the preview is sharp enough when fitted
    view.zoom_to(8.)
    view._load_detail()
    image, part, scale, clip = view._detail
    assert scale == 1. and 0 < clip.width() < 2400 / 4 and image.width() == clip.width()
    shown = _rgb(view.grab().toImage())
    assert shown.shape == (400, 600, 3)
    view.fit_view()
    view._load_detail()
    assert view._detail is None


# ---- item 13 UI: trail layer export option ----------------------------------------------------

def test_export_trail_layer_option_is_passed_to_save_review(app, make, session, monkeypatch):
    calls = []

    def save(path, decisions, **kwargs):
        calls.append(kwargs)
        return {**copy.deepcopy(session), 'decisions': copy.deepcopy(decisions), 'draft_revision': 'next',
                'options': kwargs.get('options', {})}
    monkeypatch.setattr(trd.service, 'save_review', save)
    view = make(session)
    assert not view.export_layer.isChecked() and not view._dirty
    assert view.save() and calls[-1] == {'reviewer': 'human', 'expected_draft_revision': 'draft-1'}
    view.export_layer.setChecked(True)
    assert view._dirty and '尚未保存' in view.status.text()
    assert view.save()
    assert calls[-1] == {'reviewer': 'human', 'expected_draft_revision': 'next', 'options': {'export_trail_layer': True}}
    view.export_layer.setChecked(False)          # the session now carries options: send False too
    assert view.save() and calls[-1]['options'] == {'export_trail_layer': False}


def test_saved_option_is_restored(app, make, policy_session):
    policy_session['options'] = {'export_trail_layer': True}
    view = make(policy_session)
    assert view.export_layer.isChecked() and not view._dirty


# ---- English and night ---------------------------------------------------------------------------

def test_english_review_with_filter_zoom_and_option_has_no_chinese(app, make, policy_session):
    from test_ui_dialogs_tuoxing import _chinese_texts
    i18n.set_language('en')
    policy_session['candidates'][0]['recall_review'] = {'message': '已自动省略，但身份尚未核实，不能视为已排除流星。'}
    view = make(policy_session)
    view.show(); app.processEvents()
    _relabel(view, 'trail-2', 'meteor', 'keep')
    view.set_filter('meteor')
    view.focus_selection()
    _move(view.location, view.location._point([150., 200.]))
    app.processEvents()
    assert _chinese_texts(view) == []
    texts = [view._items[c].text(k) for c in ('trail-1', 'trail-2') for k in range(4)]
    texts += [view._items[c].toolTip(k) for c in ('trail-1', 'trail-2') for k in (1, 2)]
    assert not any(CJK.search(t) for t in texts), texts
    assert view._items['trail-1'].text(2) == 'Keep by default (algorithm)'
    assert view.filter_buttons['meteor'].text() == 'Meteor 2' and view.export_layer.text() == 'Export trail layer after adding back'
    assert view.list.headerItem().text(1) == 'Identity'
    view.select_candidate('trail-1')
    assert not CJK.search(view.summary.text()), view.summary.text()


def test_night_preview_legend_and_dots_are_red_only(app, make, policy_session):
    view = make(policy_session)
    themes.apply_theme(view, 'dark', night=True)
    view.show(); app.processEvents()
    _relabel(view, 'trail-2', 'aircraft')
    view._items['trail-1'].setSelected(True)
    app.processEvents()
    for widget in (view.location, view.list, *view.legend_swatches.values()):
        pixels = _rgb(widget.grab().toImage())
        assert np.all(pixels[..., 1] == pixels[..., 2]) and np.all(pixels[..., 1] <= pixels[..., 0]), type(widget)
        assert pixels[..., 0].max() > 0
    themes.apply_theme(view, 'dark')
    app.processEvents()
    day = _rgb(view.location.grab().toImage())
    assert _mask(day, oc.HANDLE_RGB, 3).any() and _mask(day, oc.GROUND_RGB, 40).any()
