"""拓星 secondary editors: overlay colours, discard/save prompts, Enter, 底片改判, English text."""
import copy
import os
import re

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QWidget)

from starstack import i18n, overlay_colors as oc, regions, region_preview
from aester_desktop import themes, trail_review_dialog
from aester_desktop.region_canvas import RegionCanvas, night_display
from aester_desktop.region_dialog import RegionDialog
from test_region_preview_p1 import scene
from test_trail_review_dialog import session

CJK = re.compile(r'[㐀-鿿＀-￯　-〿]')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_language():
    previous = i18n.get_language()
    yield
    i18n.set_language(previous)


def _rgb(image):
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    stride = image.bytesPerLine()
    data = np.frombuffer(image.constBits(), np.uint8, count=stride * image.height()).reshape(image.height(), stride)
    return data[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


def _has(pixels, color, tolerance=0):
    return bool(np.any(np.all(np.abs(pixels.astype(int) - np.array(color)) <= tolerance, axis=-1)))


def _answer_modal(app, button_text, seen):
    """Click ``button_text`` on the next modal message box and record every button text."""
    def poll():
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox):
            QTimer.singleShot(10, poll)
            return
        seen.append({'title': box.windowTitle(), 'text': box.text(),
                     'buttons': [b.text() for b in box.buttons()],
                     'default': box.defaultButton().text() if box.defaultButton() else None})
        target = next(b for b in box.buttons() if b.text() == button_text)
        target.click()
    QTimer.singleShot(0, poll)


def _drawn(project):
    return regions.revise(project, scene_mode='landscape', operations=[{'kind': 'horizon', 'points': [[0, .6], [1, .6]]}],
                          milkyway={**project['milkyway'], 'ellipse': [.5, .3, .25, .2]})


# ---- overlay colours -------------------------------------------------------------------------

def test_shared_overlay_uses_colour_blind_safe_set_with_cased_boundary(scene):
    project, reference, original, _ = scene
    sky, mw = regions.rasterize(_drawn(project), 100, 60)
    rgba = region_preview.overlay_rgba(sky, mw)
    assert tuple(rgba[55, 20]) == (0xD5, 0x5E, 0x00, 95)                  # ground fill, old alpha
    inside = np.argwhere((mw > 0) & (sky > 0))[0]
    assert tuple(rgba[tuple(inside)][:3]) == (0xCC, 0x79, 0xA7)            # Milky Way fill
    column = rgba[:, 20]
    line_rows = [y for y in range(60) if tuple(column[y]) == (*oc.GROUND_RGB, 255)]
    casing_rows = [y for y in range(60) if tuple(column[y]) == (*oc.OUTLINE_RGB, 255)]
    assert line_rows and casing_rows, 'foreground edge must be a cased solid line'
    assert min(casing_rows) < min(line_rows) and max(casing_rows) > max(line_rows)
    for old in ((250, 133, 67), (221, 217, 127)):
        assert not _has(rgba[..., :3], old)


def test_mcp_preview_draws_milkyway_double_outline_and_new_legend(scene):
    project, reference, original, out = scene
    result = region_preview.render_preview(_drawn(project), reference, out)
    image = np.asarray(Image.open(result['preview_path']))
    assert _has(image, oc.MILKYWAY_RGB) and _has(image, oc.OUTLINE_RGB)
    # Two strokes cross the vertical line through the centre above the ellipse.
    column = image[:, 50]
    runs = np.flatnonzero(np.all(column == oc.MILKYWAY_RGB, axis=-1))
    assert len(runs) and np.any(np.diff(runs[runs < 18]) > 1), 'double outline expected at the top of the ellipse'
    assert '粉紫' in result['legend']['milkyway'] and '不代表' in result['legend']['milkyway']
    assert '淡黄' not in str(result['legend'])


def test_canvas_draws_handles_suggestion_and_milkyway_in_overlay_colours(app, scene):
    project, reference, _, _ = scene
    canvas = RegionCanvas(QImage(reference['preview_path']), _drawn(project))
    canvas.resize(700, 460)
    canvas.set_mode('milkyway')
    shown = _rgb(canvas.grab().toImage())
    assert _has(shown, oc.HANDLE_RGB) and _has(shown, oc.MILKYWAY_RGB) and _has(shown, oc.OUTLINE_RGB)
    for old in ((0xe8, 0xdc, 0x81), (0x62, 0xd3, 0xda), (0x11, 0x17, 0x17)):
        assert not _has(shown, old)
    assert tuple(shown[2, 2]) == (0x1f, 0x1f, 0x1f)  # neutral 玄 surround
    canvas.set_mode('horizon')
    canvas.suggestion = [[0., .5], [1., .5]]
    shown = _rgb(canvas.grab().toImage())
    assert _has(shown, oc.SKY_RGB)
    canvas.close()


def test_night_is_display_only_and_red(app, scene):
    project, reference, _, _ = scene
    host = QWidget()
    themes.apply_theme(host, 'dark')
    canvas = RegionCanvas(QImage(reference['preview_path']), _drawn(project), host)
    canvas.resize(500, 320)
    source, overlay = _rgb(canvas.image), _rgb(canvas._overlay)
    day = _rgb(canvas.grab().toImage())
    themes.apply_theme(host, 'dark', night=True)
    night = _rgb(canvas.grab().toImage())
    assert np.all(night[..., 1] == 0) and np.all(night[..., 2] == 0) and night[..., 0].max() > 0
    assert np.array_equal(_rgb(canvas.image), source) and np.array_equal(_rgb(canvas._overlay), overlay)
    themes.apply_theme(host, 'light')
    assert np.array_equal(_rgb(canvas.grab().toImage()), day)
    filtered = _rgb(night_display(QImage(reference['preview_path'])))
    assert np.all(filtered[..., 1:] == 0)
    host.deleteLater()


# ---- RegionDialog ----------------------------------------------------------------------------

def test_region_dialog_buttons_roles_and_enter_does_nothing(app, scene):
    project, reference, _, _ = scene
    view = RegionDialog(reference, project)
    try:
        view.show(); app.processEvents()
        buttons = view.findChildren(QPushButton)
        assert buttons and not any(b.isDefault() or b.autoDefault() for b in buttons)
        assert themes.role(view.accept_button) == 'primary'
        assert [b for b in buttons if themes.role(b) == 'primary'] == [view.accept_button]
        canvas = view.canvas
        canvas.pending = [[.1, .5], [.5, .55]]
        before = copy.deepcopy(canvas.project)
        for target in (view, view.tools, view.canvas):
            QTest.keyClick(target, Qt.Key.Key_Return)
            QTest.keyClick(target, Qt.Key.Key_Enter)
        app.processEvents()
        assert view.isVisible() and canvas.pending == [[.1, .5], [.5, .55]] and canvas.project == before
        assert not any(w.styleSheet() for w in view.findChildren(QWidget))
    finally:
        view.canvas.pending = []
        view.reject()


def test_unchanged_region_dialog_cancels_without_prompt(app, scene):
    project, reference, _, _ = scene
    view = RegionDialog(reference, project)
    view.show(); app.processEvents()
    view.cancel_button.click()
    assert not view.isVisible() and view.result() == view.DialogCode.Rejected


@pytest.mark.parametrize('trigger', ['cancel', 'escape', 'close'])
def test_unsaved_drawing_asks_before_discarding(app, scene, trigger):
    project, reference, _, _ = scene
    i18n.set_language('zh_CN')
    view = RegionDialog(reference, project)
    view.show(); app.processEvents()
    view.canvas.commit(_drawn(project))
    assert view.has_unsaved_changes()
    def attempt():
        if trigger == 'cancel':
            view.cancel_button.click()
        elif trigger == 'escape':
            QTest.keyClick(view, Qt.Key.Key_Escape)
        else:
            view.close()
        app.processEvents()
    seen = []
    _answer_modal(app, '继续编辑', seen)
    attempt()
    assert view.isVisible(), 'keep editing must leave the drawing open'
    assert seen[0]['buttons'] and set(seen[0]['buttons']) == {'使用这些分区', '放弃更改', '继续编辑'}
    assert seen[0]['default'] == '继续编辑'
    _answer_modal(app, '放弃更改', seen)
    attempt()
    assert not view.isVisible() and view.result() == view.DialogCode.Rejected


def test_discard_prompt_can_use_the_drawing(app, scene):
    project, reference, _, _ = scene
    i18n.set_language('zh_CN')
    view = RegionDialog(reference, project)
    view.show(); app.processEvents()
    view.canvas.pending = [[0., .6], [1., .62]]
    seen = []
    _answer_modal(app, '使用这些分区', seen)
    view.reject(); app.processEvents()
    assert view.result() == view.DialogCode.Accepted
    assert any(op['kind'] == 'horizon' for op in view.project['operations'])


def test_hidden_programmatic_reject_never_blocks(app, scene):
    project, reference, _, _ = scene
    view = RegionDialog(reference, project)
    view.canvas.commit(_drawn(project))
    view.reject()  # no visible editor, no prompt, no hang
    assert view.result() == view.DialogCode.Rejected


# ---- TrailReviewDialog -----------------------------------------------------------------------

@pytest.fixture
def trails(app, session, monkeypatch):
    monkeypatch.setattr(trail_review_dialog.service, 'candidate_context', lambda *a: {'panels': [], 'notice': 'context'})
    created = []
    def make(value=None):
        view = trail_review_dialog.TrailReviewDialog(value or session)
        created.append(view)
        return view
    yield make
    for view in created:
        view._dirty = False
        view._finish_workers()
        view.reject()
    app.processEvents()


def test_enter_in_review_never_batch_applies(app, trails):
    view = trails()
    view.show(); app.processEvents()
    for item in view._items.values():
        item.setSelected(True)
    view.label_choice.setCurrentIndex(view.label_choice.findData('meteor'))
    view.action_choice.setCurrentIndex(view.action_choice.findData('keep'))
    buttons = view.findChildren(QPushButton)
    assert not any(b.isDefault() or b.autoDefault() for b in buttons)
    for target in (view.list, view, view.label_choice):
        QTest.keyClick(target, Qt.Key.Key_Return)
        QTest.keyClick(target, Qt.Key.Key_Enter)
    app.processEvents()
    assert view.decisions == {} and not view._dirty and view.isVisible()
    assert themes.role(view.apply_button) == 'primary'
    assert not any(w.styleSheet() for w in view.findChildren(QWidget))


@pytest.mark.parametrize('language,expected', [('zh_CN', {'保存草稿', '不保存', '继续复核'}),
                                               ('en', {'Save draft', "Don't save", 'Keep reviewing'})])
def test_save_prompt_buttons_follow_interface_language(app, trails, language, expected):
    i18n.set_language(language)
    view = trails()
    view.show(); app.processEvents()
    view.select_candidate('trail-1')
    view.label_choice.setCurrentIndex(view.label_choice.findData('meteor'))
    view.set_selected_decisions()
    seen = []
    keep = [t for t in expected if t in ('继续复核', 'Keep reviewing')][0]
    _answer_modal(app, keep, seen)
    view.reject(); app.processEvents()
    assert set(seen[0]['buttons']) == expected and view.isVisible() and view._dirty
    drop = [t for t in expected if t in ('不保存', "Don't save")][0]
    _answer_modal(app, drop, seen)
    view.close(); app.processEvents()
    assert not view.isVisible() and view.result() == view.DialogCode.Rejected


def test_revision_shows_struck_old_value_new_value_and_provenance(app, trails, session):
    view = trails()
    view.select_candidate('trail-1')
    assert view.revision_empty.isVisibleTo(view) and not view.revision_dot.isVisibleTo(view)
    view.label_choice.setCurrentIndex(view.label_choice.findData('meteor'))
    view.action_choice.setCurrentIndex(view.action_choice.findData('keep'))
    view.set_selected_decisions()
    caption, old, arrow, new = view.revision_rows['label']
    assert isinstance(old, trail_review_dialog.StruckLabel)
    assert old.text() == '未知' and new.text() == '流星' and old.accessibleName() == '原值：未知'
    caption, old, arrow, new = view.revision_rows['override']
    assert old.text() == '省略' and new.text() == '保留'  # fixture decision 'omit', no disposition text
    assert view.revision_dot.kind() == 'human' and view.revision_dot.isVisibleTo(view)
    # Tree columns and machine decisions are unchanged by the display.
    assert view.decisions == {'trail-1': {'label': 'meteor', 'override': 'keep'}}
    assert view.list.columnCount() == 4


def test_agent_draft_choices_keep_agent_provenance_until_changed(app, trails, session):
    draft = copy.deepcopy(session)
    draft.update(reviewer='agent', decisions={'trail-2': {'override': 'keep'}})
    view = trails(draft)
    view.select_candidate('trail-2')
    assert view.revision_kind('trail-2') == 'agent' and view.revision_dot.kind() == 'agent'
    view.action_choice.setCurrentIndex(view.action_choice.findData('omit'))
    view.set_selected_decisions()
    assert view.revision_kind('trail-2') == 'human' and view.revision_dot.kind() == 'human'


def test_failed_save_marks_error_with_icon_without_prefixing_text(app, trails, monkeypatch):
    view = trails()
    def stale(*args, **kwargs):
        raise ValueError('草稿已在另一窗口更改')
    monkeypatch.setattr(trail_review_dialog.service, 'save_review', stale)
    view.show(); app.processEvents()
    assert not view.save()
    assert view.status.text() == '草稿已在另一窗口更改'
    assert themes.role(view.status) == 'error' and view.status_icon.isVisibleTo(view)


def test_trail_location_uses_trail_identity_colours(app, session):
    # 2026-09-27: each identity has its own colour (initially the suggestion's); the selected
    # trail is ringed in the HANDLE yellow instead of being drawn over.
    view = trail_review_dialog.TrailLocationView(session)
    view.resize(600, 400)
    assert view.styles == {'trail-1': ('unknown', False), 'trail-2': ('unknown', False)}
    view.set_styles({'trail-1': ('satellite', True), 'trail-2': ('meteor', True)})
    view.set_candidate('trail-1')
    shown = _rgb(view.grab().toImage())
    assert _has(shown, oc.TRAIL_IDENTITY['meteor']['rgb']) and _has(shown, oc.HANDLE_RGB)
    assert _has(shown, oc.TRAIL_IDENTITY['satellite']['rgb'])   # the label's identity sample
    view.set_candidate(None)
    shown = _rgb(view.grab().toImage())
    assert _has(shown, oc.TRAIL_IDENTITY['satellite']['rgb']) and not _has(shown, oc.HANDLE_RGB)
    assert not _has(shown, (0xe7, 0xff, 0x82))


# ---- advanced tools (styling only) -----------------------------------------------------------

def test_advanced_forms_primary_run_and_zhu_error_with_icon(app, tmp_path):
    from aester_desktop.advanced_tools import AdvancedTools
    from test_i18n_independent import Owner
    owner = Owner(tmp_path)
    tools = AdvancedTools(owner)
    try:
        for key in ('background', 'milkyway', 'layers', 'trails'):
            run = next(b for b in owner.findChildren(QPushButton) if b.objectName() == key + 'Run')
            assert themes.role(run) == 'primary' and not run.autoDefault()
        from PySide6.QtGui import QPalette
        form = owner.forms[2]  # layers
        themes.apply_theme(form, 'dark')
        status = tools.statuses['layers']
        status.setObjectName('muted')  # the real GUI labels carry this colour rule
        form.show(); app.processEvents()
        colour = lambda: status.palette().color(QPalette.ColorRole.WindowText).name()
        assert colour() == themes.COLORS['dark']['muted']
        tools.failed('layers', 'disk full')
        app.processEvents()
        assert status.text().endswith('disk full') and themes.role(status) == 'error'
        assert colour() == themes.COLORS['dark']['zhu_text']
        assert tools.status_icons['layers'].isVisible()
        tools._mark_error('layers', False)
        app.processEvents()
        assert themes.role(status) is None and status.objectName() == 'muted'
        assert colour() == themes.COLORS['dark']['muted'] and tools.status_icons['layers'].isHidden()
        form.hide()
    finally:
        owner.close()


# ---- English mode ----------------------------------------------------------------------------

def _chinese_texts(root):
    found = []
    for widget in [root, *root.findChildren(QWidget)]:
        values = [widget.toolTip(), widget.accessibleName(), widget.windowTitle() if widget is root else '']
        if isinstance(widget, (QLabel, QAbstractButton)):
            values.append(widget.text())
        if isinstance(widget, QLineEdit):
            values += [widget.text(), widget.placeholderText()]
        if isinstance(widget, QComboBox):
            values += [widget.itemText(i) for i in range(widget.count())]
        found += [(type(widget).__name__, v) for v in values if v and CJK.search(v)]
    return found


def test_english_region_editor_and_prompt_have_no_chinese(app, scene):
    project, reference, _, _ = scene
    i18n.set_language('en')
    view = RegionDialog(reference, project)
    try:
        view.show(); app.processEvents()
        assert _chinese_texts(view) == []
        assert 'Orange' in view.findChildren(QLabel)[0].text() and 'pink double line' in view.findChildren(QLabel)[0].text()
        assert view.accept_edge.text() == 'Accept light-blue edge'
        view.canvas.commit(_drawn(project))
        seen = []
        _answer_modal(app, 'Keep editing', seen)
        view.reject(); app.processEvents()
        assert not any(CJK.search(t) for t in [seen[0]['title'], seen[0]['text'], *seen[0]['buttons']])
    finally:
        _answer_modal(app, 'Discard changes', [])
        view.reject(); app.processEvents()


def test_english_trail_review_with_revision_has_no_chinese(app, trails):
    i18n.set_language('en')
    view = trails()
    view.show(); app.processEvents()
    view.select_candidate('trail-1')
    view.label_choice.setCurrentIndex(view.label_choice.findData('meteor'))
    view.action_choice.setCurrentIndex(view.action_choice.findData('keep'))
    view.set_selected_decisions()
    app.processEvents()
    assert _chinese_texts(view) == []
    assert view.revision_rows['label'][3].text() == 'Meteor'
    assert view.revision_source.text() == 'Revised by you'


# ---- 2026-09-27 review fixes -----------------------------------------------------------------

def test_leaving_the_horizon_tool_commits_clicked_points(app, scene):
    project, reference, _, _ = scene
    canvas = RegionCanvas(QImage(reference['preview_path']), project)
    canvas.resize(500, 320)
    canvas.pending = [[.1, .6], [.5, .62], [.9, .6]]
    before = len(canvas.project['operations'])
    canvas.set_mode('ground')
    assert canvas.pending == [] and len(canvas.project['operations']) == before + 1
    assert canvas.project['operations'][-1]['kind'] == 'horizon'
    canvas.undo()                                  # still undoable
    assert len(canvas.project['operations']) == before
    canvas.set_mode('horizon')
    canvas.pending = [[.3, .5]]
    canvas.set_mode('milkyway')                    # a single point is kept for later
    canvas.set_mode('horizon')
    assert canvas.pending == [[.3, .5]]
    canvas.close()


def test_brush_stroke_in_progress_is_drawn(app, scene):
    project, reference, _, _ = scene
    canvas = RegionCanvas(QImage(reference['preview_path']), project)
    canvas.resize(500, 320)
    canvas.set_mode('ground')
    idle = _rgb(canvas.grab().toImage())
    canvas._drag = True
    canvas.pending = [[.2, .3], [.4, .35], [.6, .3]]
    drawing = _rgb(canvas.grab().toImage())
    assert not np.array_equal(idle, drawing) and _has(drawing, oc.GROUND_RGB)
    canvas._drag = False
    canvas.pending = []
    canvas.close()


def test_night_canvas_matches_the_display_filter_and_caches_the_photo(app, scene):
    project, reference, _, _ = scene
    host = QWidget()
    canvas = RegionCanvas(QImage(reference['preview_path']), _drawn(project), host)
    canvas.resize(500, 320)
    canvas.set_mode('milkyway')
    canvas.surround = themes.NIGHT['canvas']         # same ground as night, so only the filter differs
    themes.apply_theme(host, 'dark')
    expected = _rgb(night_display(canvas.grab().toImage())).astype(int)
    themes.apply_theme(host, 'dark', night=True)
    night = _rgb(canvas.grab().toImage()).astype(int)
    assert np.all(night[..., 1:] == 0)
    assert np.abs(night[..., 0] - expected[..., 0]).max() <= 3      # rounding of blended strokes only
    key = canvas._night_frame._key
    canvas._hover = QPointF(100, 100)
    canvas.grab()
    assert canvas._night_frame._key == key                          # hover repaint reuses the photo layer
    themes.apply_theme(host, 'dark')
    host.deleteLater()


def test_editor_instruction_matches_behaviour_and_milkyway_tool_preselected(app, scene):
    project, reference, _, _ = scene
    only_mw = regions.revise(project, milkyway={**project['milkyway'], 'ellipse': [.5, .3, .25, .2]})
    view = RegionDialog(reference, only_mw)
    try:
        assert view.tools.currentData() == 'milkyway' and view.canvas.mode == 'milkyway'
        label = next(w for w in view.findChildren(QLabel) if '使用这些分区' in w.text())
        assert '绘制即生效' not in label.text()
    finally:
        view.canvas.pending = []
        view.done(0)
    assert RegionDialog(reference, project).tools.currentData() == 'horizon'


def test_night_message_box_has_no_system_glyph(app):
    host = QWidget()
    themes.apply_theme(host, 'dark', night=True)
    assert themes.message_icon(host) == QMessageBox.Icon.NoIcon
    themes.apply_theme(host, 'dark')
    assert themes.message_icon(host) == QMessageBox.Icon.Question
    host.deleteLater()


def test_waiting_suggestion_highlights_its_accept_button(app, scene):
    # 2026-09-27 user request: a generated light-blue edge must draw attention to
    # "Accept light-blue edge"; the highlight goes away once nothing is waiting.
    project, reference, _, _ = scene
    view = RegionDialog(reference, project)
    try:
        view.show(); app.processEvents()
        assert themes.role(view.accept_edge) is None and not view.accept_edge.isEnabled()
        view.canvas.set_mode('horizon')
        view.canvas.pending = [[0., .55], [.5, .5], [1., .55]]
        assert view.canvas.finish_horizon()
        view.canvas.suggestion = [[0., .6], [1., .6]]
        view.sync()
        assert view.accept_edge.isEnabled() and themes.role(view.accept_edge) == 'suggestion'
        for name, night in (('dark', False), ('light', False), ('dark', True)):
            assert 'QPushButton[role="suggestion"]' in themes.stylesheet(name, night=night)
        view.accept_suggestion()
        assert not view.canvas.suggestion and themes.role(view.accept_edge) is None
    finally:
        view.hide()  # a hidden editor rejects without the discard prompt
        view.reject()
