# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Photographer-facing batch review. Apply requests orchestration, never publishes."""
from __future__ import annotations

import copy
import math

from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRect, QRectF, QSize, QEvent, QTimer, QItemSelectionModel
from PySide6.QtGui import QPainter, QColor, QIcon, QImage, QImageReader, QPixmap, QPen
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QComboBox, QTreeWidget, QTreeWidgetItem, QSplitter, QCheckBox,
                              QWidget, QAbstractItemView, QMessageBox, QGridLayout, QButtonGroup,
                              QStyledItemDelegate, QStyleOptionViewItem, QStyle, QApplication)

from starstack import overlay_colors as oc
from . import themes
from starstack import trail_review_session as service
from starstack.i18n import tr, translate_message
from .region_canvas import NightFrameCache, cased, is_night, night_display, red_only, shown_color
from .ui_widgets import ErrorIcon, PROVENANCE_KINDS, ProvenanceDot

LABELS = {"meteor": "流星", "satellite": "卫星", "aircraft": "飞机", "artifact": "伪影", "unknown": "未知"}
ACTIONS = {"keep": "保留", "omit": "省略"}
# Contract D (2026-09-27): trustworthy meteors are kept by default by the algorithm.
METEOR_DEFAULT_POLICY = "meteor-default-keep-1"
# Measurement quality flags that make a meteor's automatic keep untrustworthy.
UNTRUSTED_QUALITY = ("width_limit_reached", "profile_boundary_hit", "sideband_insufficient")
DOT_ROLE = Qt.ItemDataRole.UserRole + 1


def _button(text, role=None):
    """Enter never clicks a review button implicitly (the dialog has no default button)."""
    button = QPushButton(text)
    button.setAutoDefault(False)
    button.setDefault(False)
    if role:
        themes.set_role(button, role)
    return button


def ask_save(parent):
    """Save / discard / keep reviewing, with button texts in the interface language."""
    buttons = QMessageBox.StandardButton
    box = QMessageBox(themes.message_icon(parent), tr('保存复核草稿'), tr('关闭前保存这些选择？'),
                      buttons.Save | buttons.Discard | buttons.Cancel, parent)
    box.button(buttons.Save).setText(tr('保存草稿'))
    box.button(buttons.Discard).setText(tr('不保存'))
    box.button(buttons.Cancel).setText(tr('继续复核'))
    box.setDefaultButton(buttons.Save)
    box.setEscapeButton(buttons.Cancel)
    box.exec()
    clicked = box.clickedButton()
    return box.standardButton(clicked) if clicked is not None else buttons.Cancel


# ---- effective identity and disposition (display only; the backend applies the policy) ------

def meteor_default_policy(session) -> bool:
    """True for sessions under the meteor-default-keep policy; old sessions keep explicit-keep-only."""
    if session.get("application_policy_version") == METEOR_DEFAULT_POLICY:
        return True
    return (session.get("application_policy") or {}).get("version") == METEOR_DEFAULT_POLICY


def suggested_identity(candidate) -> str:
    value = (candidate.get("suggestion") or {}).get("class")
    return value if value in LABELS else "unknown"


def recorded_identity(candidate, layered=False):
    """(identity, source) recorded before any draft: the saved label, else the suggestion.

    ``layered``: the view's identity fields already include a saved draft decision
    (contract D), so only the record's own fields (label, suggestion) are used.
    """
    label = candidate.get("label")
    if label in LABELS:
        identity = label
    elif not layered and "label" not in candidate and candidate.get("identity") in LABELS:
        identity = candidate["identity"]
    else:
        effective = candidate.get("effective_label")
        identity = effective if effective in LABELS else suggested_identity(candidate)
    # Records before label_source existed: a saved label was a reviewer's choice.
    derived = "human" if label in LABELS or identity != suggested_identity(candidate) else "algorithm"
    source = None if layered else candidate.get("identity_source")
    if source not in PROVENANCE_KINDS or (source == "algorithm") != (derived == "algorithm"):
        source = derived
    return identity, source


def recorded_override(candidate, layered=False):
    """(override, source) recorded before any draft; None when the default decides."""
    override = candidate.get("override")
    if override not in ACTIONS:
        return None, None
    source = candidate.get("override_source")
    if source not in PROVENANCE_KINDS and not layered and override == "keep":
        source = candidate.get("keep_source")
    return override, source if source in PROVENANCE_KINDS else "human"


def measurement_trustworthy(candidate) -> bool:
    if isinstance(candidate.get("measurement_trustworthy"), bool):
        return candidate["measurement_trustworthy"]
    if candidate.get("status") != "measured":
        return False
    quality = candidate.get("measurement_quality") or {}
    return not any(quality.get(key) for key in UNTRUSTED_QUALITY)


def candidate_state(candidate, decision=None, meteor_default=False, kinds=None, layered=False) -> dict:
    """Effective identity and disposition with a draft ``decision`` layered on the record.

    Mirrors the backend's keep rule for display: an explicit keep/omit wins; otherwise a
    trustworthy meteor is kept by default (meteor-default sessions only) and the keep's
    source is whoever set the identity. Only measured candidates can be added back.
    ``kinds`` names who made each draft field ('human' or 'agent'); ``layered`` says the
    view's own identity/keep fields already include a saved draft for this candidate.
    """
    decision = decision or {}
    kinds = kinds or {}
    identity, identity_source = recorded_identity(candidate, layered)
    if "label" in decision:
        if decision["label"] in LABELS:
            identity, identity_source = decision["label"], kinds.get("label", "human")
        else:
            identity, identity_source = suggested_identity(candidate), "algorithm"
    override, override_source = recorded_override(candidate, layered)
    if "override" in decision:
        override = decision["override"] if decision["override"] in ACTIONS else None
        override_source = kinds.get("override", "human") if override else None
    measured = candidate.get("status") == "measured"
    default = bool(meteor_default and identity == "meteor" and measurement_trustworthy(candidate))
    explicit = override is not None
    if explicit:
        requested, source = override == "keep", override_source
    else:
        requested, source = default, identity_source
        if not requested and not layered and not decision and candidate.get("effective_keep") is True:
            # A recorded keep this display cannot derive (such as a protection mark).
            explicit, requested = True, True
            source = candidate.get("keep_source") if candidate.get("keep_source") in PROVENANCE_KINDS else "human"
    return {"identity": identity, "identity_source": identity_source, "keep": bool(requested and measured),
            "requested": requested, "explicit": explicit, "measured": measured, "keep_source": source,
            "suggestion": suggested_identity(candidate),
            "revised": identity_source != "algorithm" or source != "algorithm"}


def disposition_text(state, meteor_default) -> str:
    """Untranslated disposition for a state (callers pass it through tr)."""
    if state["explicit"]:
        if state["requested"] and not state["measured"]:
            return '明确保留（未测量，不会加回）'
        return '明确保留' if state["requested"] else '明确省略'
    if state["keep"]:
        return '默认保留（算法）' if state["keep_source"] == "algorithm" else '默认保留（按改判身份）'
    if meteor_default and state["identity"] == "meteor":
        return '默认省略（测量不可靠）'
    return '默认省略'


def _source_text(kind):
    return {"human": tr('人工确定'), "agent": tr('Agent 确定')}.get(kind, tr('算法'))


def _candidate_title(candidate):
    return tr('第') + str(candidate.get("frame_index", "")) + tr('张 · 候选') + candidate["id"].rsplit("-", 1)[-1]


# ---- painting helpers ----------------------------------------------------------------------

def paint_provenance_dot(painter, rect, kind, tokens):
    """The 三家星色 mark: human filled 朱, Agent filled 缃, algorithm a hollow ring."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if kind == "algorithm":
        pen = QPen(QColor(tokens["text"]), 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect.adjusted(.75, .75, -.75, -.75))
    else:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens["zhu"] if kind == "human" else tokens["xiang"]))
        painter.drawEllipse(rect)
    painter.restore()


def stroke_trail(painter, color, draw, *, width=oc.LINE_WIDTH, style=oc.SOLID, alpha=255):
    """Stroke ``draw(painter)`` in an identity colour over the dark casing (night aware)."""
    stroke_color = QColor(color)
    stroke_color.setAlpha(alpha)
    outline = QColor(oc.OUTLINE)
    outline.setAlpha(alpha)
    casing = QPen(shown_color(outline), width + oc.OUTLINE_EXTRA)
    casing.setCapStyle(Qt.PenCapStyle.RoundCap)
    casing.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    stroke = QPen(shown_color(stroke_color), width)
    stroke.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    stroke.setCapStyle(Qt.PenCapStyle.RoundCap)
    if style == oc.DOT:
        stroke.setDashPattern(list(oc.DOT_PATTERN))
    elif style == oc.DASH:
        stroke.setDashPattern(list(oc.DASH_PATTERN))
        stroke.setCapStyle(Qt.PenCapStyle.FlatCap)
    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for pen in (casing, stroke):
        painter.setPen(pen)
        draw(painter)
    painter.restore()


def ring_padding(length):
    """Space between a selected trail and its yellow ring, in screen pixels."""
    return max(8., .04 * length)


def draw_ring(painter, a, b):
    """A cased HANDLE-yellow capsule enclosing the whole segment a–b."""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length = math.hypot(dx, dy)
    pad = ring_padding(length)
    shape = QRectF(-length / 2 - pad, -pad, length + 2 * pad, 2 * pad)
    painter.save()
    painter.translate((a + b) / 2)
    painter.rotate(math.degrees(math.atan2(dy, dx)))
    cased(painter, "handle", lambda p: p.drawRoundedRect(shape, pad, pad), width=2.)
    painter.restore()


class ProvenanceDelegate(QStyledItemDelegate):
    """Paints the provenance dot in front of a cell's text (tokens read at paint time)."""

    SIZE = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        blank = QPixmap(self.SIZE, self.SIZE)
        blank.fill(Qt.GlobalColor.transparent)
        self._blank = QIcon(blank)

    def initStyleOption(self, option, index):  # noqa: N802 - Qt override
        super().initStyleOption(option, index)
        if index.data(DOT_ROLE) in PROVENANCE_KINDS:
            # Reserve the decoration slot so the style lays out the text after the dot.
            option.features |= QStyleOptionViewItem.ViewItemFeature.HasDecoration
            option.icon = self._blank
            option.decorationSize = QSize(self.SIZE, self.SIZE)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        kind = index.data(DOT_ROLE)
        if kind not in PROVENANCE_KINDS:
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        slot = style.subElementRect(QStyle.SubElement.SE_ItemViewItemDecoration, opt, widget)
        centre = QRectF(slot).center()
        rect = QRectF(centre.x() - self.SIZE / 2, centre.y() - self.SIZE / 2, self.SIZE, self.SIZE)
        paint_provenance_dot(painter, rect, kind, themes.colors_for(widget))


class StruckLabel(QLabel):
    """A superseded value (底片改判): muted text with a strike line in the text colour.

    A font strike-out sits on the horizontal strokes of CJK glyphs in the same grey and
    disappears, so the line is painted here, brighter than the muted text beneath it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        themes.set_role(self, 'muted')

    def setText(self, text):
        super().setText(text)
        self.setAccessibleName(tr('原值：{0}', text) if text else "")

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.text():
            return
        painter = QPainter(self)
        rect = self.contentsRect()
        metrics = self.fontMetrics()
        width = min(metrics.horizontalAdvance(self.text()), rect.width())
        y = rect.top() + (rect.height() - metrics.height()) / 2 + metrics.ascent() * .62
        pen = painter.pen()
        pen.setColor(QColor(themes.colors_for(self)["text"]))
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.left() + width, y))


def _pixmap(path, width=1600, height=1000):
    if not path:
        return QPixmap()
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid():
        reader.setScaledSize(size.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio))
    return QPixmap.fromImage(reader.read())


def _source_size(path):
    if not path:
        return None
    size = QImageReader(str(path)).size()
    return size if size.isValid() and not size.isEmpty() else None


class TrailSwatch(QWidget):
    """Legend sample: an identity colour, the kept/omitted line style, or the selection ring."""

    STYLE_GREY = "#A3A3A3"   # line-style samples are neutral: the colour belongs to identity

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(30, 16)

    def sizeHint(self):
        return QSize(30, 16)

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()
        super().changeEvent(event)

    def render_image(self) -> QImage:
        """The swatch as drawn on the photo surround (before any night display filter)."""
        ratio = self.devicePixelRatioF()
        image = QImage(self.size() * ratio, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(QColor(themes.SURROUNDS[themes.DEFAULT_SURROUND]))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y, x0, x1 = self.height() / 2, 5., self.width() - 5.
        line = lambda p: p.drawLine(QPointF(x0, y), QPointF(x1, y))
        if self.kind in oc.TRAIL_IDENTITY:
            stroke_trail(painter, oc.TRAIL_IDENTITY[self.kind]["hex"], line)
        elif self.kind == "kept":
            stroke_trail(painter, self.STYLE_GREY, line, style=oc.TRAIL_KEPT_LINE)
        elif self.kind == "omitted":
            stroke_trail(painter, self.STYLE_GREY, line, style=oc.TRAIL_OMITTED_LINE)
        else:
            stroke_trail(painter, self.STYLE_GREY, line, width=1.5)
            cased(painter, "handle", lambda p: p.drawRoundedRect(QRectF(2, 2, self.width() - 4, self.height() - 4), 6, 6),
                  width=1.5)
        painter.end()
        return image

    def paintEvent(self, event):
        image = self.render_image()
        QPainter(self).drawImage(0, 0, night_display(image) if is_night(self) else image)


class TrailLocationView(QWidget):
    """Whole-frame preview of the candidates, zoomable, in native reference coordinates.

    Selected trails get a yellow ring; hovering only thickens a line. Colours follow the
    effective identity, the line style the disposition (kept solid, omitted dotted).
    """
    candidateClicked = Signal(str)
    candidateToggled = Signal(str)
    candidateHovered = Signal(str)
    viewChanged = Signal()

    MIN_ZOOM, MAX_ZOOM, FOCUS_ZOOM = .5, 16., 8.
    STEP = 1.25
    HIT_DISTANCE = 12.

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(tr('亮迹位置预览'))
        self._path = session.get("preview_path")
        self.pixmap = _pixmap(self._path)
        self._source = _source_size(self._path) if not self.pixmap.isNull() else None
        self._detail = None         # (image, normalized source rect, scale, clip) when zoomed in
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(120)
        self._detail_timer.timeout.connect(self._load_detail)
        self._night_frame = NightFrameCache()
        self.geometry_info = session["geometry"]
        self.candidates = session["candidates"]
        self._by_id = {c["id"]: c for c in self.candidates}
        meteor_default = meteor_default_policy(session)
        saved = session.get("decisions") or {}
        kind = "agent" if session.get("reviewer") == "agent" else "human"
        self.styles = {}
        for candidate in self.candidates:
            decision = saved.get(candidate["id"])
            state = candidate_state(candidate, decision, meteor_default, {"label": kind, "override": kind},
                                    layered=bool(decision))
            self.styles[candidate["id"]] = (state["identity"], state["keep"])
        self.visible_ids = None     # None: every candidate is shown
        self.selected_ids = set()
        self.current_id = None
        self.hover_id = None
        self.zoom = 1.
        self.pan = QPointF()
        self._press = None
        self._panning = False

    # -- state set by the dialog --

    def set_candidate(self, candidate_id):
        """Make ``candidate_id`` the only selected (and current) candidate."""
        self.current_id = candidate_id
        self.selected_ids = {candidate_id} if candidate_id else set()
        self.update()

    def set_selection(self, ids, current=None):
        self.selected_ids = set(ids)
        self.current_id = current
        self.update()

    def set_current(self, candidate_id):
        self.current_id = candidate_id
        self.update()

    def set_styles(self, styles):
        self.styles = dict(styles)
        self.update()

    def set_visible(self, ids):
        self.visible_ids = None if ids is None else set(ids)
        if self.hover_id is not None and not self._shown(self.hover_id):
            self.set_hover(None)
        self.update()

    def set_hover(self, candidate_id):
        if candidate_id == self.hover_id:
            return
        self.hover_id = candidate_id
        self.setToolTip(self._describe(candidate_id) if candidate_id else "")
        self._update_cursor()
        self.update()

    def _shown(self, candidate_id):
        return self.visible_ids is None or candidate_id in self.visible_ids

    def _describe(self, candidate_id):
        candidate = self._by_id.get(candidate_id)
        if candidate is None:
            return ""
        identity, keep = self.styles.get(candidate_id, ("unknown", False))
        return " · ".join((_candidate_title(candidate), tr(LABELS.get(identity, '未知')),
                           tr('会加回') if keep else tr('不加回')))

    # -- view transform --

    def _bounds(self):
        return QRectF(self.rect().adjusted(10, 10, -10, -10))

    def _rect(self):
        """The image rectangle on screen, following zoom and pan."""
        bounds = self._bounds()
        width, height = self.geometry_info["width"], self.geometry_info["height"]
        if bounds.width() <= 0 or bounds.height() <= 0 or not width or not height:
            return QRectF(bounds.center().x(), bounds.center().y(), 0., 0.)
        scale = min(bounds.width() / width, bounds.height() / height) * self.zoom
        w, h = width * scale, height * scale
        return QRectF(bounds.center().x() - w / 2 + self.pan.x(), bounds.center().y() - h / 2 + self.pan.y(), w, h)

    def _point(self, point):
        rect = self._rect()
        # Pixel centre x=0 maps to the first pixel centre, not the left image edge.
        return QPointF(rect.x() + (float(point[0])+.5) / self.geometry_info["width"]*rect.width(),
                       rect.y() + (float(point[1])+.5) / self.geometry_info["height"]*rect.height())

    def zoom_to(self, zoom, anchor=None):
        """Zoom to ``zoom`` keeping the image point under ``anchor`` (default: view centre) still."""
        zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, zoom))
        anchor = QPointF(anchor) if anchor is not None else self._bounds().center()
        rect = self._rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        u, v = (anchor.x() - rect.x()) / rect.width(), (anchor.y() - rect.y()) / rect.height()
        self.zoom = zoom
        moved = self._rect()
        self.pan += QPointF(anchor.x() - (moved.x() + u * moved.width()), anchor.y() - (moved.y() + v * moved.height()))
        if abs(self.zoom - 1.) < 1e-9:
            self.pan = QPointF()
        self._clamp_pan()
        self._view_changed()

    def fit_view(self):
        self.zoom = 1.
        self.pan = QPointF()
        self._view_changed()

    def focus_ids(self, ids):
        """Zoom and centre on the candidates ``ids`` (their endpoints' bounding box)."""
        points = [p for cid in ids if cid in self._by_id and self._shown(cid)
                  for p in (self._by_id[cid].get("endpoints_native_reference") or [])[:2]]
        if not points:
            return False
        width, height = self.geometry_info["width"], self.geometry_info["height"]
        xs = [(float(p[0]) + .5) / width for p in points]
        ys = [(float(p[1]) + .5) / height for p in points]
        self.zoom, self.pan = 1., QPointF()
        fit, bounds = self._rect(), self._bounds()
        box_w = max((max(xs) - min(xs)) * fit.width(), 24.)
        box_h = max((max(ys) - min(ys)) * fit.height(), 24.)
        zoom = min(bounds.width() * .6 / box_w, bounds.height() * .6 / box_h, self.FOCUS_ZOOM)
        self.zoom = min(self.MAX_ZOOM, max(1., zoom))
        rect = self._rect()
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.pan = bounds.center() - QPointF(rect.x() + cx * rect.width(), rect.y() + cy * rect.height())
        self._clamp_pan()
        self._view_changed()
        return True

    def ensure_visible(self, candidate_id):
        """When zoomed in, pan so the candidate's midpoint is inside the view."""
        points = (self._by_id.get(candidate_id) or {}).get("endpoints_native_reference")
        if self.zoom <= 1. or not points or len(points) != 2:
            return
        middle = (self._point(points[0]) + self._point(points[1])) / 2
        if self._bounds().adjusted(20, 20, -20, -20).contains(middle):
            return
        self.pan += self._bounds().center() - middle
        self._clamp_pan()
        self._view_changed()

    def _clamp_pan(self):
        """Keep at least part of the photograph inside the view."""
        bounds, rect = self._bounds(), self._rect()
        mx, my = min(40., rect.width() / 2), min(40., rect.height() / 2)
        dx = dy = 0.
        if rect.right() < bounds.left() + mx:
            dx = bounds.left() + mx - rect.right()
        elif rect.left() > bounds.right() - mx:
            dx = bounds.right() - mx - rect.left()
        if rect.bottom() < bounds.top() + my:
            dy = bounds.top() + my - rect.bottom()
        elif rect.top() > bounds.bottom() - my:
            dy = bounds.bottom() - my - rect.top()
        self.pan += QPointF(dx, dy)

    def _view_changed(self):
        self._update_cursor()
        self.update()
        self.viewChanged.emit()
        if self._source is not None:
            self._detail_timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._source is not None and self.zoom > 1.:
            self._detail_timer.start()

    # -- sharper pixels when zoomed past the preview's resolution --

    def _load_detail(self):
        """Decode only the visible part of the preview file at the resolution the view needs."""
        if self._source is None:
            return
        rect = self._rect()
        visible = rect.intersected(QRectF(self.rect()))
        if visible.isEmpty() or rect.width() <= 0:
            return
        sw, sh = self._source.width(), self._source.height()
        needed = rect.width() * self.devicePixelRatioF() / sw       # device px per source px
        if needed <= self.pixmap.width() / sw * 1.2:
            if self._detail is not None:
                self._detail = None
                self.update()
            return
        scale = min(1., needed)
        x0 = max(0, math.floor((visible.left() - rect.left()) / rect.width() * sw))
        y0 = max(0, math.floor((visible.top() - rect.top()) / rect.height() * sh))
        x1 = min(sw, math.ceil((visible.right() - rect.left()) / rect.width() * sw))
        y1 = min(sh, math.ceil((visible.bottom() - rect.top()) / rect.height() * sh))
        if x1 <= x0 or y1 <= y0:
            return
        clip = QRect(x0, y0, x1 - x0, y1 - y0)
        if self._detail is not None and self._detail[3] == clip and self._detail[2] >= scale * .999:
            return
        reader = QImageReader(str(self._path))
        reader.setClipRect(clip)
        reader.setScaledSize(QSize(max(1, round(clip.width() * scale)), max(1, round(clip.height() * scale))))
        image = reader.read()
        if image.isNull():
            return
        self._detail = (image, QRectF(x0 / sw, y0 / sh, (x1 - x0) / sw, (y1 - y0) / sh), scale, clip)
        self.update()

    # -- painting --

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not is_night(self):
            self._paint(painter)
            return
        # Night vision, display only: the photograph goes through the red-only filter once
        # (cached); trails and labels are painted directly in night colours.
        rect = self._rect()
        detail = self._detail[0].cacheKey() if self._detail is not None else None
        key = (self.pixmap.cacheKey(), detail, rect.x(), rect.y(), rect.width(), rect.height())
        painter.drawImage(0, 0, self._night_frame.get(self, key, lambda p: self._paint_base(p, night=True)))
        with red_only():
            self._paint_marks(painter)

    def _paint(self, painter, night=False):
        self._paint_base(painter, night=night)
        self._paint_marks(painter)

    def _paint_base(self, painter, night=False):
        # The photo surround and trail overlays never follow the light/dark theme.
        surround = themes.NIGHT["canvas"] if night else themes.SURROUNDS[themes.DEFAULT_SURROUND]
        painter.fillRect(self.rect(), QColor(surround))
        if self.pixmap.isNull():
            painter.setPen(QColor(themes.COLORS["dark"]["muted"]))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr('本任务没有可用的整图预览'))
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._rect()
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        if self._detail is not None:
            image, part = self._detail[0], self._detail[1]
            target = QRectF(rect.x() + part.x() * rect.width(), rect.y() + part.y() * rect.height(),
                            part.width() * rect.width(), part.height() * rect.height())
            painter.drawImage(target, image, QRectF(image.rect()))
        painter.restore()

    def _visible_segments(self):
        for candidate in self.candidates:
            if not self._shown(candidate["id"]):
                continue
            points = candidate.get("endpoints_native_reference")
            if not points or len(points) != 2:
                continue
            yield candidate, self._point(points[0]), self._point(points[1])

    def _paint_line(self, painter, candidate, a, b, width, alpha):
        identity, keep = self.styles.get(candidate["id"], ("unknown", False))
        colour = oc.TRAIL_IDENTITY.get(identity, oc.TRAIL_IDENTITY["unknown"])["hex"]
        stroke_trail(painter, colour, lambda p: p.drawLine(a, b), width=width, alpha=alpha,
                     style=oc.TRAIL_KEPT_LINE if keep else oc.TRAIL_OMITTED_LINE)

    def _paint_marks(self, painter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        segments = list(self._visible_segments())
        rest = [s for s in segments if s[0]["id"] not in self.selected_ids and s[0]["id"] != self.hover_id]
        hovered = [s for s in segments if s[0]["id"] == self.hover_id and s[0]["id"] not in self.selected_ids]
        selected = [s for s in segments if s[0]["id"] in self.selected_ids]
        for candidate, a, b in rest:
            self._paint_line(painter, candidate, a, b, oc.LINE_WIDTH, 255)
        # Hover only thickens a line; the ring belongs to the selection.
        for candidate, a, b in hovered:
            self._paint_line(painter, candidate, a, b, 3.5, 255)
        # A selected trail's line becomes a yellow ring around it, leaving the trail's own
        # pixels visible for checking; its identity and disposition go into the label.
        for candidate, a, b in selected:
            draw_ring(painter, a, b)
        current = next((s for s in selected if s[0]["id"] == self.current_id), None)
        if current:
            self._paint_title(painter, *current)

    def _paint_title(self, painter, candidate, a, b):
        title = self._describe(candidate["id"])
        # The ring is the segment widened by ``pad`` in every direction, so its bounding box
        # is the segment's box grown by ``pad``: the label sits just above it (else below).
        pad = ring_padding(math.hypot(b.x() - a.x(), b.y() - a.y()))
        view = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        metrics = painter.fontMetrics()
        sample = 18.
        text_width = metrics.horizontalAdvance(title) + sample + 6
        x = max(view.left()+4, min((a.x()+b.x())/2 - text_width/2, view.right()-text_width-4))
        y = min(a.y(), b.y()) - pad - metrics.descent() - 8
        if y - metrics.ascent() - 3 < view.top():
            y = max(a.y(), b.y()) + pad + metrics.ascent() + 8
        y = max(view.top()+metrics.ascent()+3, min(y, view.bottom()-metrics.descent()-3))
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shown_color(oc.OUTLINE))
        painter.drawRoundedRect(QRectF(x-4, y-metrics.ascent()-3, text_width+8, metrics.height()+6), 3, 3)
        middle = y - metrics.ascent() + metrics.height() / 2 - 1
        self._paint_line(painter, candidate, QPointF(x+2, middle), QPointF(x+sample-2, middle), oc.LINE_WIDTH, 255)
        painter.setPen(shown_color(themes.COLORS["dark"]["text"]))
        painter.drawText(QPointF(x + sample + 6, y), title)
        painter.restore()

    # -- interaction --

    def hit_test(self, point):
        """The visible candidate nearest to the screen ``point`` (within 12 px), or None."""
        nearest, distance = None, self.HIT_DISTANCE
        for candidate, a, b in self._visible_segments():
            vx, vy = b.x()-a.x(), b.y()-a.y()
            length2 = vx*vx+vy*vy
            t = min(1., max(0., ((point.x()-a.x())*vx+(point.y()-a.y())*vy)/length2)) if length2 else 0.
            d = math.hypot(point.x()-a.x()-t*vx, point.y()-a.y()-t*vy)
            if d < distance:
                nearest, distance = candidate["id"], d
        return nearest

    def _update_cursor(self):
        if self._panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self.hover_id:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif self.zoom != 1.:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if not steps:
            return
        self.zoom_to(self.zoom * self.STEP ** steps, event.position())
        event.accept()

    def mousePressEvent(self, event):
        position = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._press, self._panning = (position, QPointF(self.pan)), True
            self._update_cursor()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cid = self.hit_test(position)
        if cid:
            if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                self.candidateToggled.emit(cid)
            else:
                self.candidateClicked.emit(cid)
            return
        # Empty space: a drag pans the photograph.
        self._press, self._panning = (position, QPointF(self.pan)), False

    def mouseMoveEvent(self, event):
        position = event.position()
        if self._press is not None:
            start, pan = self._press
            if self._panning or (position - start).manhattanLength() > 3:
                self._panning = True
                self.pan = pan + (position - start)
                self._clamp_pan()
                self._view_changed()
            return
        cid = self.hit_test(position)
        self.set_hover(cid)
        if cid:
            self.candidateHovered.emit(cid)

    def mouseReleaseEvent(self, event):
        if self._press is not None and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._press, self._panning = None, False
            self._update_cursor()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        cid = self.hit_test(event.position())
        if cid:
            self.focus_ids([cid])
        else:
            self.fit_view()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_0:
            self.fit_view()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_to(self.zoom * self.STEP)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom_to(self.zoom / self.STEP)
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        self.set_hover(None)
        super().leaveEvent(event)


class _ContextWorker(QThread):
    resultReady = Signal(int, object)

    def __init__(self, token, session_path, candidate_id, parent=None):
        super().__init__(parent)
        self.token, self.path, self.candidate_id = token, session_path, candidate_id

    def run(self):
        try:
            result = service.candidate_context(self.path, self.candidate_id)
        except Exception as exc:
            result = {"error": str(exc), "panels": []}
        self.resultReady.emit(self.token, result)


class TrailReviewDialog(QDialog):
    """Use open_review asynchronously first. Accepted => apply_session_path.

    The caller should run workflow_edit in its normal background worker after
    acceptance; the dialog does not call core.recompose_transients itself.
    """
    applyRequested = Signal(str)

    def __init__(self, session: dict, parent=None):
        super().__init__(parent)
        self.session = copy.deepcopy(session)
        self.decisions = copy.deepcopy(session.get("decisions") or {})
        self._undo = []
        self._workers = []
        self._context_token = 0
        self._context_pending = None
        self._dirty = False
        self._loaded_decisions = copy.deepcopy(self.decisions)
        self._loaded_kind = "agent" if session.get("reviewer") == "agent" else "human"
        self._states = {}
        self._index, self._index_source = {}, None
        self._filter_identity = None
        self._filter_revised = False
        self._pinned = set()   # just-revised rows stay visible until the filter changes
        self.apply_session_path = None
        self.apply_draft_revision = None
        self.setWindowTitle(tr('亮迹复核 · Aester'))
        self.resize(1220, 880)
        layout = QVBoxLayout(self)
        if self._meteor_default():
            text = tr('选中一条或多条亮迹，分别决定身份与是否保留。测量可靠的流星默认保留（空心圈＝算法）；其他身份默认省略，明确选择“保留”才会加回。改判后列表与右侧预览立即更新，朱点表示你的改判。\n未知不等于卫星；自动省略不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。')
        else:
            text = tr('选中一条或多条亮迹，分别决定身份与是否保留。只有明确选择“保留”才会加回，判断为流星也不会自动加回。\n未知不等于卫星；自动省略不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。')
        intro = QLabel(text)
        intro.setWordWrap(True)
        themes.set_role(intro, 'muted')
        layout.addWidget(intro)
        layout.addLayout(self._build_filters())
        split = QSplitter(Qt.Orientation.Horizontal)
        self.list = QTreeWidget()
        self.list.setHeaderLabels([tr('候选 / 曝光'), tr('身份'), tr('处置状态'), tr('本次选择')])
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setRootIsDecorated(False)
        self.list.setMouseTracking(True)
        self._dots = ProvenanceDelegate(self.list)
        self.list.setItemDelegateForColumn(1, self._dots)
        self.list.setItemDelegateForColumn(2, self._dots)
        self._items = {}
        for candidate in session["candidates"]:
            short_id = candidate["id"].rsplit("-", 1)[-1]
            item = QTreeWidgetItem([tr('第') + str(candidate.get("frame_index", "")) + tr('张 · 候选') + short_id, "", "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, candidate["id"])
            item.setToolTip(0, candidate["id"] + "\n" + str(candidate.get("input") or "") +
                            (tr('\n未能测量定位，不能把省略视为已排除流星') if candidate.get("status") != "measured" else ""))
            self.list.addTopLevelItem(item)
            self._items[candidate["id"]] = item
        self.list.setMinimumWidth(410)
        split.addWidget(self.list)
        split.addWidget(self._build_preview(session))
        split.setSizes([480, 700])
        layout.addWidget(split, 3)
        self.summary = QLabel(tr('选择候选查看来源与邻帧。'))
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(self._build_revision())
        panels = QHBoxLayout()
        self.context_panels = []
        for title in (tr('前一曝光'), tr('本次曝光'), tr('后一曝光')):
            column = QVBoxLayout()
            label = QLabel(title)
            image = QLabel(tr('未选择'))
            image.setMinimumHeight(110)
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setWordWrap(True)
            column.addWidget(label)
            column.addWidget(image, 1)
            panels.addLayout(column, 1)
            self.context_panels.append((label, image))
        layout.addLayout(panels, 1)
        self.context_notice = QLabel(tr('邻帧图使用实际曝光，不将相邻候选误当成相邻曝光。'))
        self.context_notice.setWordWrap(True)
        themes.set_role(self.context_notice, 'hint')
        layout.addWidget(self.context_notice)
        controls = QHBoxLayout()
        self.label_choice = QComboBox()
        self.label_choice.addItem(tr('身份：保持原样'), "unchanged")
        for value, text in LABELS.items():
            self.label_choice.addItem(tr(text), value)
        self.label_choice.addItem(tr('清除已有人工身份'), None)
        self.action_choice = QComboBox()
        reset = tr('恢复默认（按身份）') if self._meteor_default() else tr('恢复默认省略')
        for text, value in ((tr('处置：保持原样'), "unchanged"), (tr('保留'), "keep"), (tr('省略'), "omit"), (reset, None)):
            self.action_choice.addItem(text, value)
        controls.addWidget(self.label_choice)
        controls.addWidget(self.action_choice)
        self.label_choice.setAccessibleName(tr('身份'))
        self.action_choice.setAccessibleName(tr('处置'))
        self.batch_button = _button(tr('设置选中项'))
        self.batch_button.clicked.connect(self.set_selected_decisions)
        controls.addWidget(self.batch_button)
        clear = _button(tr('撤销选中项的本次选择'), 'ghost')
        clear.clicked.connect(self.clear_selected_decisions)
        controls.addWidget(clear)
        self.undo_button = _button(tr('撤销上一步'), 'ghost')
        self.undo_button.clicked.connect(self.undo)
        controls.addWidget(self.undo_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        footer = QHBoxLayout()
        self.status_icon = ErrorIcon(16, accessible_name=tr('错误'))
        self.status_icon.hide()
        footer.addWidget(self.status_icon, 0, Qt.AlignmentFlag.AlignTop)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.export_layer = QCheckBox(tr('加回后导出亮迹层'))
        self.export_layer.setChecked(bool((session.get("options") or {}).get("export_trail_layer")))
        self.export_layer.setToolTip(tr('另存一张只含已保留亮迹的透明 16 位 TIFF 图层（trails_layer_rgba16.tif），叠在基础成片上即得到加回亮迹的成片。至少保留一条亮迹时才导出。'))
        self.export_layer.toggled.connect(self._option_changed)
        footer.addWidget(self.export_layer)
        self.save_button = save = _button(tr('保存草稿'))
        save.clicked.connect(self.save)
        footer.addWidget(save)
        self.apply_button = apply = _button(tr('应用全部选择并出图'), 'primary')
        apply.clicked.connect(self.request_apply)
        footer.addWidget(apply)
        self.close_button = close = _button(tr('关闭'))
        close.clicked.connect(self.reject)
        footer.addWidget(close)
        layout.addLayout(footer)
        self.list.currentItemChanged.connect(self._current_changed)
        self.list.itemSelectionChanged.connect(self._selection_changed)
        # Hovering a row only highlights its line; it never moves the selection or the ring.
        self.list.itemEntered.connect(lambda item, column: self.location.set_hover(item.data(0, Qt.ItemDataRole.UserRole)))
        self._list_viewport = self.list.viewport()
        self._list_viewport.installEventFilter(self)
        self.location.candidateClicked.connect(self.select_candidate)
        self.location.candidateToggled.connect(self.toggle_candidate)
        self._refresh_decisions()
        # Columns fit their (translated) content once; the choice column keeps a fixed width.
        for column, (minimum, maximum) in enumerate(((100, 170), (64, 110), (110, 175))):
            self.list.resizeColumnToContents(column)
            self.list.setColumnWidth(column, min(maximum, max(minimum, self.list.columnWidth(column) + 6)))
        self.list.setColumnWidth(3, 90)
        if self.list.topLevelItemCount():
            self.list.setCurrentItem(self.list.topLevelItem(0))
        else:
            self.status.setText(tr('本次扫描没有候选；这不等于原片不存在未被检测到的流星。'))

    # -- layout pieces --

    def _build_filters(self):
        row = QHBoxLayout()
        caption = QLabel(tr('筛选'))
        themes.set_role(caption, 'muted')
        row.addWidget(caption)
        bar = QWidget()
        themes.set_role(bar, 'segmented')
        inner = QHBoxLayout(bar)
        inner.setContentsMargins(2, 2, 2, 2)
        inner.setSpacing(2)
        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self.filter_buttons = {}
        for key in (None, *LABELS):
            button = _button("", 'segment')
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=key: self.set_filter(value))
            self._filter_group.addButton(button)
            inner.addWidget(button)
            self.filter_buttons[key] = button
        self.filter_buttons[None].setChecked(True)
        row.addWidget(bar)
        revised = QWidget()
        themes.set_role(revised, 'segmented')
        revised_layout = QHBoxLayout(revised)
        revised_layout.setContentsMargins(2, 2, 2, 2)
        self.revised_button = _button("", 'segment')
        self.revised_button.setCheckable(True)
        self.revised_button.setToolTip(tr('只显示身份或处置由人工或 Agent 改判过的候选。'))
        self.revised_button.toggled.connect(self.set_revised_filter)
        revised_layout.addWidget(self.revised_button)
        row.addWidget(revised)
        self.filter_count = QLabel("")
        themes.set_role(self.filter_count, 'hint')
        row.addWidget(self.filter_count)
        row.addStretch(1)
        return row

    def _build_preview(self, session):
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        tools = QHBoxLayout()
        self.fit_button = _button(tr('适合窗口'), 'ghost')
        self.focus_button = _button(tr('定位到所选'), 'ghost')
        self.zoom_out_button = _button("−", 'ghost')
        self.zoom_in_button = _button("+", 'ghost')
        for button, name in ((self.zoom_out_button, tr('缩小')), (self.zoom_in_button, tr('放大'))):
            button.setAccessibleName(name)
            button.setToolTip(name)
        self.zoom_label = QLabel("100%")
        themes.set_role(self.zoom_label, 'caption')
        hint = QLabel(tr('滚轮缩放，拖动空白处或按住中键平移，双击空白处复原；Ctrl 点选可多选。'))
        themes.set_role(hint, 'hint')
        hint.setWordWrap(True)
        for widget in (self.fit_button, self.focus_button, self.zoom_out_button, self.zoom_label, self.zoom_in_button):
            tools.addWidget(widget)
        tools.addWidget(hint, 1)
        column.addLayout(tools)
        self.location = TrailLocationView(session)
        column.addWidget(self.location, 1)
        legend = QHBoxLayout()
        legend.setSpacing(4)
        self.legend_swatches = {}
        entries = [(key, tr(text)) for key, text in LABELS.items()]
        entries += [("kept", tr('会加回')), ("omitted", tr('不加回')), ("selected", tr('选中'))]
        for index, (kind, text) in enumerate(entries):
            if index in (len(LABELS), len(LABELS) + 2):
                legend.addSpacing(10)
            swatch = TrailSwatch(kind)
            swatch.setAccessibleName(text)
            label = QLabel(text)
            themes.set_role(label, 'hint')
            legend.addWidget(swatch)
            legend.addWidget(label)
            legend.addSpacing(6)
            self.legend_swatches[kind] = swatch
        legend.addStretch(1)
        column.addLayout(legend)
        self.fit_button.clicked.connect(self.location.fit_view)
        self.focus_button.clicked.connect(self.focus_selection)
        self.zoom_out_button.clicked.connect(lambda: self.location.zoom_to(self.location.zoom / self.location.STEP))
        self.zoom_in_button.clicked.connect(lambda: self.location.zoom_to(self.location.zoom * self.location.STEP))
        self.location.viewChanged.connect(self._update_zoom_label)
        return box

    def _build_revision(self):
        """底片改判: the recorded value struck through, the new choice, and who chose it."""
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self.revision_rows = {}
        for row, (key, title) in enumerate((("label", tr('身份')), ("override", tr('处置')))):
            caption = QLabel(title)
            themes.set_role(caption, 'muted')
            old = StruckLabel()
            arrow = QLabel("→")
            themes.set_role(arrow, 'muted')
            new = QLabel()
            new.setTextFormat(Qt.TextFormat.PlainText)
            widgets = (caption, old, arrow, new)
            for column, widget in enumerate(widgets):
                grid.addWidget(widget, row, column)
            self.revision_rows[key] = widgets
        self.revision_dot = ProvenanceDot("human")
        self.revision_source = QLabel()
        themes.set_role(self.revision_source, 'hint')
        grid.addWidget(self.revision_dot, 0, 4, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.revision_source, 0, 5)
        self.revision_empty = QLabel(tr('本条尚无本次改判。'))
        themes.set_role(self.revision_empty, 'hint')
        grid.addWidget(self.revision_empty, 0, 0, 1, 4)
        grid.setColumnStretch(6, 1)
        self.revision_box = box
        return box

    # -- effective state --

    def _meteor_default(self):
        return meteor_default_policy(self.session)

    def _candidate(self, candidate_id):
        candidates = self.session["candidates"]
        if self._index_source is not candidates:   # rebuilt when save() replaces the session view
            self._index = {c["id"]: c for c in candidates}
            self._index_source = candidates
        return self._index.get(candidate_id)

    def revision_kind(self, candidate_id):
        """'human' for choices made in this window; choices loaded from a draft keep its source."""
        decision = self.decisions.get(candidate_id)
        if decision is not None and decision == self._loaded_decisions.get(candidate_id):
            return self._loaded_kind
        return "human"

    def _field_kinds(self, candidate_id):
        """Who made each draft field: the loaded draft's reviewer until changed here."""
        decision = self.decisions.get(candidate_id) or {}
        loaded = self._loaded_decisions.get(candidate_id) or {}
        return {field: (self._loaded_kind if field in loaded and loaded[field] == decision[field] else "human")
                for field in ("label", "override") if field in decision}

    def _layered(self, candidate_id):
        """Whether the session view's identity/keep fields already include a saved draft choice."""
        return bool((self.session.get("decisions") or {}).get(candidate_id))

    def candidate_state(self, candidate_id):
        candidate = self._candidate(candidate_id) or {"id": candidate_id}
        return candidate_state(candidate, self.decisions.get(candidate_id), self._meteor_default(),
                               self._field_kinds(candidate_id), self._layered(candidate_id))

    def _recorded_identity(self, candidate):
        return recorded_identity(candidate, self._layered(candidate["id"]))[0]

    def _recorded_disposition(self, candidate):
        """The disposition before any draft choice, as text in the interface language."""
        if not self._meteor_default():
            return translate_message(candidate.get("review_disposition", ACTIONS.get(candidate.get("decision"), '待定')))
        state = candidate_state(candidate, meteor_default=True, layered=self._layered(candidate["id"]))
        return tr(disposition_text(state, True))

    def _disposition(self, candidate, state):
        decision = self.decisions.get(candidate["id"]) or {}
        if not self._meteor_default() and "override" not in decision and candidate.get("review_disposition"):
            return translate_message(candidate["review_disposition"])
        return tr(disposition_text(state, self._meteor_default()))

    def _show_revision(self, candidate_id):
        candidate = self._candidate(candidate_id)
        decision = (self.decisions.get(candidate_id) or {}) if candidate else {}
        values = {}
        if "label" in decision:
            values["label"] = (tr(LABELS[self._recorded_identity(candidate)]),
                               tr(LABELS.get(decision["label"], '清除身份')))
        if "override" in decision:
            if decision["override"] in ACTIONS:
                new = tr(ACTIONS[decision["override"]])
            elif self._meteor_default():
                new = tr(disposition_text(self.candidate_state(candidate_id), True))
            else:
                new = tr('默认省略')
            values["override"] = (self._recorded_disposition(candidate), new)
        for key, (caption, old, arrow, new) in self.revision_rows.items():
            shown = key in values
            for widget in (caption, old, arrow, new):
                widget.setVisible(shown)
            if shown:
                old.setText(values[key][0])
                new.setText(values[key][1])
        self.revision_empty.setVisible(not values)
        kind = self.revision_kind(candidate_id) if values else "human"
        text = tr('Agent 草稿改判') if kind == "agent" else tr('人工改判')
        self.revision_dot.set_kind(kind)
        self.revision_dot.setAccessibleName(text)
        self.revision_source.setText(text)
        self.revision_dot.setVisible(bool(values))
        self.revision_source.setVisible(bool(values))

    def _set_status_error(self, error):
        if themes.role(self.status) != ('error' if error else None):
            themes.set_role(self.status, 'error' if error else None)
        self.status_icon.setVisible(error)

    # -- list, filter and preview --

    def select_candidate(self, candidate_id):
        item = self._items.get(candidate_id)
        if item and not item.isHidden():
            self.list.clearSelection()
            self.list.setCurrentItem(item)
            item.setSelected(True)
            self.list.scrollToItem(item)

    def toggle_candidate(self, candidate_id):
        """Ctrl/Shift-click in the preview adds or removes one candidate from the selection."""
        item = self._items.get(candidate_id)
        if item is None or item.isHidden():
            return
        self.list.setCurrentItem(item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        item.setSelected(not item.isSelected())
        self.list.scrollToItem(item)

    def selected_ids(self):
        return [item.data(0, Qt.ItemDataRole.UserRole) for item in self.list.selectedItems() if not item.isHidden()]

    def focus_selection(self):
        ids = self.selected_ids()
        current = self.list.currentItem()
        if not ids and current is not None and not current.isHidden():
            ids = [current.data(0, Qt.ItemDataRole.UserRole)]
        self.location.focus_ids(ids)

    def _update_zoom_label(self):
        self.zoom_label.setText(f"{round(self.location.zoom * 100)}%")

    def _selection_changed(self):
        current = self.list.currentItem()
        current_id = current.data(0, Qt.ItemDataRole.UserRole) if current is not None and not current.isHidden() else None
        self.location.set_selection(self.selected_ids(), current_id)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.Leave and watched is self._list_viewport:
            try:
                self.location.set_hover(None)
            except RuntimeError:   # the preview is already gone while the dialog closes
                pass
        return super().eventFilter(watched, event)

    def set_filter(self, identity):
        """Show only candidates whose effective identity is ``identity`` (None: all)."""
        self._filter_identity = identity if identity in LABELS else None
        self._pinned.clear()
        button = self.filter_buttons[self._filter_identity]
        if not button.isChecked():
            button.setChecked(True)
        self._apply_filter()

    def set_revised_filter(self, enabled):
        self._filter_revised = bool(enabled)
        self._pinned.clear()
        if self.revised_button.isChecked() != self._filter_revised:
            self.revised_button.setChecked(self._filter_revised)
        self._apply_filter()

    def _matches(self, candidate_id):
        state = self._states[candidate_id]
        return ((self._filter_identity is None or state["identity"] == self._filter_identity)
                and (not self._filter_revised or state["revised"]))

    def visible_ids(self):
        return [cid for cid, item in self._items.items() if not item.isHidden()]

    def _apply_filter(self):
        shown = set()
        for cid, item in self._items.items():
            visible = cid in self._pinned or self._matches(cid)
            if visible:
                shown.add(cid)
            if item.isHidden() == visible:
                item.setHidden(not visible)
            if not visible and item.isSelected():
                item.setSelected(False)
        self.location.set_visible(None if len(shown) == len(self._items) else shown)
        self.filter_count.setText(tr('显示 {0} / {1} 条', len(shown), len(self._items)) if self._items else "")
        current = self.list.currentItem()
        if current is not None and current.isHidden():
            first = next((self._items[cid] for cid in self._items if cid in shown), None)
            if first is not None:
                self.list.setCurrentItem(first)
            else:
                self.list.setCurrentItem(None)
                self.location.set_current(None)
                self.summary.setText(tr('当前筛选下没有候选。'))
        self._selection_changed()

    def _update_filter_buttons(self):
        counts = {key: 0 for key in LABELS}
        revised = 0
        for state in self._states.values():
            counts[state["identity"]] += 1
            revised += bool(state["revised"])
        for key, button in self.filter_buttons.items():
            name = tr('全部') if key is None else tr(LABELS[key])
            button.setText(f"{name} {len(self._states) if key is None else counts[key]}")
            button.setAccessibleName(button.text())
        self.revised_button.setText(f"{tr('已改判')} {revised}")
        self.revised_button.setAccessibleName(self.revised_button.text())

    def _refresh_decisions(self):
        meteor_default = self._meteor_default()
        if meteor_default:
            policy_note = tr('\n测量可靠的流星默认加回；其他身份只有明确选择保留才会加回。')
        else:
            policy_note = tr('\n只有明确选择保留才会加入新成片；自动建议不代表已采用。')
        styles = {}
        for cid, item in self._items.items():
            candidate = self._candidate(cid) or {"id": cid}
            state = self.candidate_state(cid)
            self._states[cid] = state
            styles[cid] = (state["identity"], state["keep"])
            identity = tr(LABELS[state["identity"]])
            item.setText(1, identity)
            item.setData(1, DOT_ROLE, state["identity_source"])
            reasons = [str(r) for r in ((candidate.get("suggestion") or {}).get("reasons") or [])[:3]]
            item.setToolTip(1, tr('身份：{0} · 来源：{1}', identity, _source_text(state["identity_source"])) + "\n" +
                            tr('算法建议：{0}', tr(LABELS[state["suggestion"]])) +
                            "".join("\n· " + reason for reason in reasons))
            disposition = self._disposition(candidate, state)
            item.setText(2, disposition)
            item.setData(2, DOT_ROLE, state["keep_source"])
            item.setToolTip(2, tr('处置：{0} · 来源：{1}', disposition, _source_text(state["keep_source"])) + policy_note)
            decision = self.decisions.get(cid, {})
            parts = []
            if "label" in decision:
                parts.append(tr(LABELS.get(decision["label"], '清除身份')))
            if "override" in decision:
                reset = tr('恢复默认') if meteor_default else tr('默认省略')
                parts.append(tr(ACTIONS[decision["override"]]) if decision["override"] in ACTIONS else reset)
            item.setText(3, " / ".join(parts))
        self.location.set_styles(styles)
        self._update_filter_buttons()
        self._apply_filter()
        self.undo_button.setEnabled(bool(self._undo))
        current = self.list.currentItem()
        if current is not None:
            self._show_revision(current.data(0, Qt.ItemDataRole.UserRole))
        self._set_status_error(False)
        self._refresh_status()

    def _refresh_status(self):
        self.status.setText(tr("{0} 条有本次选择", len(self.decisions)) + (tr(' · 尚未保存') if self._dirty else tr(' · 草稿已保存')))

    def _option_changed(self, checked):
        self._dirty = True
        self._set_status_error(False)
        self._refresh_status()

    def _push_undo(self):
        self._undo.append(copy.deepcopy(self.decisions))
        self._undo = self._undo[-50:]
        self._dirty = True

    def set_selected_decisions(self):
        items = [item for item in self.list.selectedItems() if not item.isHidden()]
        label, action = self.label_choice.currentData(), self.action_choice.currentData()
        if not items or (label == action == "unchanged"):
            self.status.setText(tr('先选择候选和要更改的身份或处置。'))
            return
        self._push_undo()
        for item in items:
            cid = item.data(0, Qt.ItemDataRole.UserRole)
            entry = self.decisions.setdefault(cid, {})
            if label != "unchanged":
                entry["label"] = label
            if action != "unchanged":
                entry["override"] = action
            if self._filter_identity is not None or self._filter_revised:
                self._pinned.add(cid)
        self._refresh_decisions()

    def clear_selected_decisions(self):
        items = [item for item in self.list.selectedItems() if not item.isHidden()]
        if not items:
            return
        self._push_undo()
        for item in items:
            self.decisions.pop(item.data(0, Qt.ItemDataRole.UserRole), None)
        self._refresh_decisions()

    def undo(self):
        if self._undo:
            self.decisions = self._undo.pop()
            self._dirty = True
            self._refresh_decisions()

    def save(self):
        kwargs = {"reviewer": "human", "expected_draft_revision": self.session["draft_revision"]}
        if self.export_layer.isChecked() or isinstance(self.session.get("options"), dict):
            kwargs["options"] = {"export_trail_layer": self.export_layer.isChecked()}
        try:
            self.session = service.save_review(self.session["session_path"], self.decisions, **kwargs)
        except Exception as exc:
            self._set_status_error(True)
            self.status.setText(translate_message(str(exc)))
            return False
        self._dirty = False
        self._loaded_decisions = copy.deepcopy(self.session.get("decisions") or self.decisions)
        self._loaded_kind = "agent" if self.session.get("reviewer") == "agent" else "human"
        self._refresh_decisions()
        return True

    def request_apply(self):
        if self.save():
            self.apply_session_path = self.session["session_path"]
            self.apply_draft_revision = self.session["draft_revision"]
            self.applyRequested.emit(self.apply_session_path)
            self._finish_workers()
            super().accept()

    def _current_changed(self, item, previous=None):
        if item is None:
            return
        cid = item.data(0, Qt.ItemDataRole.UserRole)
        self.location.set_current(cid)
        self.location.ensure_visible(cid)
        self._selection_changed()
        self._show_revision(cid)
        candidate = self._candidate(cid)
        review = candidate.get("recall_review") or {}
        self.summary.setText(tr("{0} · {1}\n已记录身份：{2}；复核处置：{3}。 ",
                             cid, candidate.get('input') or '',
                             tr(LABELS[self._recorded_identity(candidate)]),
                             self._recorded_disposition(candidate)) + translate_message(review.get("message") or ""))
        self._context_token += 1
        self._context_pending = (self._context_token, cid)
        for _, label in self.context_panels:
            label.clear()
            label.setText(tr('读取实际曝光…'))
        self._start_pending_context()

    def _start_pending_context(self):
        if any(worker.isRunning() for worker in self._workers) or self._context_pending is None:
            return
        token, cid = self._context_pending
        self._context_pending = None
        worker = _ContextWorker(token, self.session["session_path"], cid, self)
        worker.resultReady.connect(self._context_ready)
        worker.finished.connect(self._start_pending_context)
        self._workers.append(worker)
        worker.start()

    def _context_ready(self, token, result):
        if token != self._context_token:
            return
        self.context_notice.setText(translate_message(result.get("error") or result.get("notice", "")))
        by_role = {panel["role"]: panel for panel in result.get("panels", [])}
        for role, (heading, label), title in zip(("previous", "source", "next"), self.context_panels, (tr('前一曝光'), tr('本次曝光'), tr('后一曝光'))):
            panel = by_role.get(role, {})
            heading.setText(title + (" · " + str(panel["frame_index"]) if panel.get("frame_index") else ""))
            label.clear()
            if panel.get("available"):
                label.setPixmap(_pixmap(panel["path"], 360, 170))
                label.setToolTip(panel.get("input", ""))
            else:
                label.setText(translate_message(panel.get("reason") or '此条没有可靠的测量位置'))

    def _finish_workers(self):
        self._context_pending = None
        self._context_token += 1
        for worker in self._workers:
            worker.wait()

    def reject(self):
        if self._dirty:
            answer = ask_save(self)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Save and not self.save():
                return
        self._finish_workers()
        super().reject()

    def closeEvent(self, event):
        self.reject()
        if self.isVisible():
            event.ignore()
        else:
            event.accept()
