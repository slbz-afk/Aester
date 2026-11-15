# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Photographer-facing batch review. Apply requests orchestration, never publishes."""
from __future__ import annotations

import copy
import functools
import math
import re
from pathlib import Path

from PySide6.QtCore import (Qt, QThread, Signal, QPointF, QRect, QRectF, QSize, QEvent, QTimer, QItemSelectionModel,
                            QStandardPaths)
from PySide6.QtGui import QPainter, QColor, QIcon, QImage, QImageReader, QPixmap, QPen
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QComboBox, QTreeWidget, QTreeWidgetItem, QSplitter, QCheckBox,
                              QWidget, QAbstractItemView, QMessageBox, QGridLayout, QButtonGroup,
                              QStyledItemDelegate, QStyleOptionViewItem, QStyle, QApplication,
                              QScrollArea, QFrame, QFileDialog, QMenu, QLayout)

from starstack import overlay_colors as oc
from . import themes, trail_manifest
from starstack import trail_review_session as service
from starstack.i18n import get_language, tr, translate_message
from .region_canvas import NightFrameCache, cased, is_night, night_display, red_only, shown_color
from .ui_widgets import ErrorIcon, PROVENANCE_KINDS, ProvenanceDot

LABELS = {"meteor": "流星", "satellite": "卫星", "aircraft": "飞机", "artifact": "伪影", "unknown": "未知"}
ACTIONS = {"keep": "保留", "omit": "不加回"}
# Contract D (2026-09-27): trustworthy meteors are kept by default by the algorithm.
METEOR_DEFAULT_POLICY = "meteor-default-keep-1"
# Measurement quality flags that make a meteor's automatic keep untrustworthy.
UNTRUSTED_QUALITY = ("width_limit_reached", "profile_boundary_hit", "sideband_insufficient")
DOT_ROLE = Qt.ItemDataRole.UserRole + 1
# Recall review priority (transient_review): a hint for what to look at first, never a confidence. It is
# given to automatic omissions (and unmeasured keep requests); kept or already decided candidates have none.
PRIORITIES = {"high": "高", "normal": "一般", "low": "低"}
_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}
ATTENTION = ("omitted_unverified", "requested_keep_unmeasured")
KEEP_FILTERS = {"keep": "会加回", "omit": "不加回"}
(COL_ID, COL_IDENTITY, COL_DISPOSITION, COL_CHOICE,
 COL_PRIORITY, COL_TIME, COL_SNR, COL_LENGTH) = range(8)
SORTABLE = (COL_ID, COL_PRIORITY, COL_TIME, COL_SNR, COL_LENGTH)
_FIRST_DESCENDING = (COL_SNR, COL_LENGTH)     # first click: brightest / longest first


def _button(text, role=None):
    """Enter never clicks a review button implicitly (the dialog has no default button)."""
    button = QPushButton(text)
    button.setAutoDefault(False)
    button.setDefault(False)
    if role:
        themes.set_role(button, role)
    return button


class _FlowRow(QLayout):
    """Widgets in a row that wraps onto a second line when the dialog is narrower than the row, so the
    row never sets the dialog's minimum width (longer English labels included)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 - Qt override
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802 - Qt override
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802 - Qt override
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt override
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802 - Qt override
        return True

    def heightForWidth(self, width):  # noqa: N802 - Qt override
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):  # noqa: N802 - Qt override
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self):  # noqa: N802 - Qt override
        shown = [item.sizeHint() for item in self._items if not item.isEmpty()]
        width = sum(hint.width() for hint in shown) + self._gap() * max(0, len(shown) - 1)
        return QSize(width, max((hint.height() for hint in shown), default=0))

    def minimumSize(self):  # noqa: N802 - Qt override
        size = QSize(0, 0)
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        return size

    def _gap(self):
        return self.spacing() if self.spacing() >= 0 else 6

    def _place(self, rect, apply):
        """Lay the items out left to right, wrapping at ``rect``'s width; returns the height used."""
        gap = self._gap()
        lines, line, used = [], [], 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if line and used + gap + hint.width() > rect.width():
                lines.append(line)
                line, used = [], 0
            used += (gap if line else 0) + hint.width()
            line.append((item, hint))
        if line:
            lines.append(line)
        y = rect.y()
        for number, line in enumerate(lines):
            height = max(hint.height() for _, hint in line)
            x = rect.x()
            for item, hint in line:
                if apply:     # each item centred in its line's height
                    item.setGeometry(QRect(x, y + (height - hint.height()) // 2, hint.width(), hint.height()))
                x += hint.width() + gap
            y += height + (gap if number < len(lines) - 1 else 0)
        return y - rect.y()


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
        return '明确保留' if state["requested"] else '明确不加回'
    if state["keep"]:
        return '默认保留（算法）' if state["keep_source"] == "algorithm" else '默认保留（按改判身份）'
    if meteor_default and state["identity"] == "meteor":
        return '默认不加回（测量不可靠）'
    return '默认不加回'


def _source_text(kind):
    return {"human": tr('人工确定'), "agent": tr('Agent 确定')}.get(kind, tr('算法'))


def action_text(action) -> str:
    """A keep / omit choice in the interface language: 保留 / 不加回; English Keep / Omit (the 不加回
    filter and legend read 'Not added back' there)."""
    if action == "omit":
        return tr('不加回（选择）') if get_language() == "en" else ACTIONS["omit"]
    return tr(ACTIONS.get(action, str(action)))


def record_text(text) -> str:
    """Wording from the engine's record, for display. English goes through translate_message; the Chinese
    interface says 不加回 where older records say 省略. The record keeps its own words."""
    text = str(text or "")
    if get_language() == "en":
        return translate_message(text)
    from .translations_dialogs import ZH_PLAIN_RECORD_WORDS
    return ZH_PLAIN_RECORD_WORDS.get(text, text)


def _file_name(path) -> str:
    """The last part of a Windows or POSIX path."""
    return re.split(r"[\\/]", str(path))[-1] if path else ""


def _exposure_text(candidate) -> str:
    """' · 曝光 10 秒' from the view's exptime_s (FITS EXPTIME); empty when it is not recorded."""
    seconds = _finite(candidate.get("exptime_s"))
    if seconds is None or seconds <= 0:
        return ""
    return " · " + tr('曝光 {0} 秒', f"{seconds:g}")


def _candidate_title(candidate):
    return tr('第') + str(candidate.get("frame_index", "")) + tr('张 · 候选') + candidate["id"].rsplit("-", 1)[-1]


def review_priority(candidate):
    """'high' / 'normal' / 'low' for candidates the recall review lists for attention, else None."""
    review = candidate.get("recall_review") or {}
    if review.get("status") in ATTENTION and review.get("priority") in PRIORITIES:
        return review["priority"]
    return None


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# The scan's date_obs is the frame's FITS DATE-OBS in UT (for camera RAW, Siril converts the camera's clock
# with the processing computer's time zone). The list shows it in this computer's zone, which gives back
# the camera clock when both use the same zone. None: this computer's zone (tests set a fixed one).
LOCAL_ZONE = None


def capture_clock(date_obs) -> str:
    """'04:22:06' (local time) from the UT '2026-08-12T20:22:06'; the text as written when unreadable."""
    local = trail_manifest.capture_time(date_obs, LOCAL_ZONE)
    return local.strftime("%H:%M:%S") if local is not None else str(date_obs or "")


def capture_tooltip(date_obs) -> str:
    local = trail_manifest.capture_time(date_obs, LOCAL_ZONE)
    if local is None:
        return str(date_obs or "")
    return tr('{0}（这台电脑的时区 {1}）\n记录的世界时：{2}', local.strftime("%Y-%m-%d %H:%M:%S"),
              trail_manifest.utc_offset(local), str(date_obs))


def _time_key(date_obs):
    local = trail_manifest.capture_time(date_obs)
    return local.timestamp() if local is not None else None


@functools.lru_cache(maxsize=1)
def _zh_patterns():
    from .translations_dialogs import ZH_SUGGESTION_REASONS
    return tuple((re.compile(pattern), template) for pattern, template in ZH_SUGGESTION_REASONS)


def _zh_value(name, value):
    from . import translations_dialogs as zh
    if value is None:
        return ""
    if name.endswith("_level"):
        return zh.ZH_REASON_LEVEL.replace("{value}", value)
    if name in ("frame", "frames") and value not in zh.ZH_REASON_VALUES:
        frame = lambda match: zh.ZH_REASON_FRAME.replace("{number}", str(int(match.group(1))))
        return zh.ZH_REASON_LIST_SEPARATOR.join(re.sub(r"light_(\d+)", frame, part.strip()) for part in value.split(","))
    return zh.ZH_REASON_VALUES.get(value, value)


def suggestion_text(reason) -> str:
    """One of the scan's English suggestion reasons, in Chinese for the Chinese interface (display only).

    The record keeps the English text; an unknown or older wording is shown as it is.
    """
    text = str(reason)
    if get_language() == "en":
        return text
    for pattern, template in _zh_patterns():
        match = pattern.fullmatch(text)
        if match:
            shown = template
            for name, value in match.groupdict().items():
                shown = shown.replace("{" + name + "}", _zh_value(name, value))
            return shown
    return text


def _sort_keys(candidate):
    number = candidate["id"].rsplit("-", 1)[-1]
    priority = review_priority(candidate)
    return {COL_ID: (candidate.get("frame_index") or 0, int(number) if number.isdigit() else 0),
            COL_PRIORITY: None if priority is None else _PRIORITY_RANK[priority],
            COL_TIME: _time_key(candidate.get("date_obs")),
            COL_SNR: _finite(candidate.get("total_snr")), COL_LENGTH: _finite(candidate.get("length_px"))}


class _CandidateTree(QTreeWidget):
    """The candidate list; ``sort_state`` is (column, descending) of the last sort the dialog asked for."""

    sort_state = (COL_ID, False)


class _Row(QTreeWidgetItem):
    """One candidate. Columns sort by fixed keys, ties by record order, so rows never trade ids."""

    def __init__(self, texts, order, keys):
        super().__init__(texts)
        self.order, self.keys = order, keys

    def __lt__(self, other):
        tree = self.treeWidget()
        column, descending = tree.sort_state if isinstance(tree, _CandidateTree) else (COL_ID, False)
        a, b = self.keys.get(column), getattr(other, "keys", {}).get(column)
        if (a is None) != (b is None):
            return (a is not None) != descending      # rows without a value stay last in both orders
        if a is not None and a != b:
            return a < b
        return (self.order < getattr(other, "order", 0)) != descending


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

    def __init__(self, session: dict, parent=None, *, workflow: dict | None = None, originals: dict | None = None,
                 cache_root: str | None = None):
        super().__init__(parent)
        self.session = copy.deepcopy(session)
        # The workflow record (read only): which candidates its published trails version added back.
        self.workflow = copy.deepcopy(workflow) if isinstance(workflow, dict) else None
        # Original photographs behind DNG working copies made in this session (trail_manifest.import_sources).
        self.originals = dict(originals or {})
        # The main window's cache folder: the composite list looks up the import records there for stacked
        # inputs this session did not convert (trail_manifest.with_cache_records). None: no lookup.
        self.cache_root = cache_root
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
        self._filter_priority = None
        self._filter_keep = None
        self._pinned = set()   # just-revised rows stay visible until the filter changes
        self.apply_session_path = None
        self.apply_draft_revision = None
        self.setWindowTitle(tr('亮迹复核 · Aester'))
        self._fit_to_screen(parent)
        layout = QVBoxLayout(self)
        if self._meteor_default():
            text = tr('选中一条或多条亮迹，分别决定身份与是否保留。测量可靠的流星默认保留（空心圈＝算法）；其他身份默认不加回，明确选择“保留”才会加回。改判后列表与右侧预览立即更新，朱点表示你的改判。\n未知不等于卫星；自动不加回不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。')
        else:
            text = tr('选中一条或多条亮迹，分别决定身份与是否保留。只有明确选择“保留”才会加回，判断为流星也不会自动加回。\n未知不等于卫星；自动不加回不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。')
        intro = QLabel(text)
        intro.setWordWrap(True)
        themes.set_role(intro, 'muted')
        layout.addWidget(intro)
        self._priorities = {c["id"]: review_priority(c) for c in session["candidates"]}
        layout.addLayout(self._build_filters())
        self.split = split = QSplitter(Qt.Orientation.Horizontal)
        self.list = _CandidateTree()
        self.list.setHeaderLabels([tr('候选 / 曝光'), tr('身份'), tr('是否加回'), tr('本次选择'),
                                   tr('优先级'), tr('时间'), tr('信噪比'), tr('长度')])
        header_tips = {COL_ID: tr('点击按帧序号（加入照片的顺序）排列。'),
                       COL_PRIORITY: tr('先看哪些的提示，不是可信度。自动不加回的候选里，端点渐隐、碰到画框边或测量到边界的为“高”；已请求保留但未能测量的也为“高”。已加回或此前已明确选择的显示“—”；本窗口里的新选择应用后才更新。'),
                       COL_TIME: tr('拍摄时刻，按这台电脑的时区显示。相机和电脑设在同一时区时就是相机上的时间；不同时会差整小时。'),
                       COL_SNR: tr('沿轨迹的总信噪比（S/N），可当亮度参考：越大越明显；轨迹越长也越大，不等于亮度。'),
                       COL_LENGTH: tr('测量出的长度（像素）。')}
        for column, tip in header_tips.items():
            self.list.headerItem().setToolTip(column, tip)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setRootIsDecorated(False)
        self.list.setUniformRowHeights(True)
        self.list.setMouseTracking(True)
        self._dots = ProvenanceDelegate(self.list)
        self.list.setItemDelegateForColumn(COL_IDENTITY, self._dots)
        self.list.setItemDelegateForColumn(COL_DISPOSITION, self._dots)
        self._items = {}
        for order, candidate in enumerate(session["candidates"]):
            short_id = candidate["id"].rsplit("-", 1)[-1]
            priority = self._priorities[candidate["id"]]
            snr, length = _finite(candidate.get("total_snr")), _finite(candidate.get("length_px"))
            item = _Row([tr('第') + str(candidate.get("frame_index", "")) + tr('张 · 候选') + short_id, "", "", "",
                         tr(PRIORITIES[priority]) if priority else "—", capture_clock(candidate.get("date_obs")),
                         f"{snr:.0f}" if snr is not None else "", f"{length:.0f}" if length is not None else ""],
                        order, _sort_keys(candidate))
            item.setData(0, Qt.ItemDataRole.UserRole, candidate["id"])
            item.setToolTip(0, candidate["id"] + "\n" + str(candidate.get("input") or "") +
                            (tr('\n未能测量定位，不能把“不加回”视为已排除流星') if candidate.get("status") != "measured" else ""))
            if candidate.get("date_obs"):
                item.setToolTip(COL_TIME, capture_tooltip(candidate["date_obs"]))
            for column in (COL_SNR, COL_LENGTH):
                item.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._items[candidate["id"]] = item
        self.list.addTopLevelItems(list(self._items.values()))
        header = self.list.header()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._header_clicked)
        self.list.setMinimumWidth(360)
        split.addWidget(self.list)
        split.addWidget(self._build_preview(session))
        split.addWidget(self._build_details())
        split.setCollapsible(0, False)
        split.splitterMoved.connect(self._splitter_moved)
        layout.addWidget(split, 1)
        controls = _FlowRow()       # wraps on narrow screens instead of widening the dialog
        self.next_button = _button(tr('下一条待复核'))
        self.next_button.setToolTip(tr('跳到列表中下一条还没有选择的候选（回车）。'))
        self.next_button.clicked.connect(self.next_to_review)
        controls.addWidget(self.next_button)
        self.label_choice = QComboBox()
        self.label_choice.addItem(tr('身份：保持原样'), "unchanged")
        for value, text in LABELS.items():
            self.label_choice.addItem(tr(text), value)
        self.label_choice.addItem(tr('清除已有人工身份'), None)
        self.action_choice = QComboBox()
        reset = tr('恢复默认（按身份）') if self._meteor_default() else tr('恢复默认（不加回）')
        for text, value in ((tr('是否加回：保持原样'), "unchanged"), (action_text("keep"), "keep"),
                            (action_text("omit"), "omit"), (reset, None)):
            self.action_choice.addItem(text, value)
        controls.addWidget(self.label_choice)
        controls.addWidget(self.action_choice)
        self.label_choice.setAccessibleName(tr('身份'))
        self.action_choice.setAccessibleName(tr('是否加回'))
        self.batch_button = _button(tr('设置选中项'))
        self.batch_button.clicked.connect(self.set_selected_decisions)
        controls.addWidget(self.batch_button)
        clear = _button(tr('撤销选中项的本次选择'), 'ghost')
        clear.clicked.connect(self.clear_selected_decisions)
        controls.addWidget(clear)
        self.undo_button = _button(tr('撤销上一步'), 'ghost')
        self.undo_button.clicked.connect(self.undo)
        controls.addWidget(self.undo_button)
        # Every row the current filter shows at once (asks first; one undo step restores it).
        self.bulk_button = _button(tr('当前筛选全部…'), 'ghost')
        self.bulk_button.setToolTip(tr('对当前筛选显示的全部候选一次设为保留或不加回；先选一个筛选。'))
        bulk = QMenu(self.bulk_button)
        self.bulk_keep_action = bulk.addAction(tr('全部保留…'))
        self.bulk_keep_action.triggered.connect(lambda: self.bulk_choice("keep"))
        self.bulk_omit_action = bulk.addAction(tr('全部不加回…'))
        self.bulk_omit_action.triggered.connect(lambda: self.bulk_choice("omit"))
        self.bulk_button.setMenu(bulk)
        controls.addWidget(self.bulk_button)
        layout.addLayout(controls)
        footer = QHBoxLayout()
        self.status_icon = ErrorIcon(16, accessible_name=tr('错误'))
        self.status_icon.hide()
        footer.addWidget(self.status_icon, 0, Qt.AlignmentFlag.AlignTop)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(self.status, 1)
        self.export_button = _button(tr('导出合成清单…'), 'ghost')
        self.export_button.clicked.connect(lambda: self.export_manifest())
        footer.addWidget(self.export_button)
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
        self.list.installEventFilter(self)
        self.location.candidateClicked.connect(self.select_candidate)
        self.location.candidateToggled.connect(self.toggle_candidate)
        self._refresh_decisions()
        self._update_export_button()
        self._fit_columns()
        # Shown order: candidate, priority, identity, decision, time, brightness, length, this choice
        # (logical column numbers stay as they were).
        header.moveSection(header.visualIndex(COL_PRIORITY), 1)
        header.moveSection(header.visualIndex(COL_CHOICE), header.count() - 1)
        self._sized = False
        self._initial_sizes()
        # Open sorted by review priority: high first; kept or decided candidates last (they have a filter).
        self.sort_by(COL_PRIORITY, descending=False)
        if self.list.topLevelItemCount():
            self.list.setCurrentItem(self.list.topLevelItem(0))
        else:
            self.status.setText(tr('本次扫描没有候选；这不等于原片不存在未被检测到的流星。'))

    # -- layout pieces --

    def _screen_area(self, parent=None):
        screen = parent.screen() if parent is not None else None
        screen = screen or self.screen() or QApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 720)

    def _fit_to_screen(self, parent):
        """Size the dialog to the screen's free area (1280 x 720 laptops included) and centre it there.

        Called before the layout exists and again when first shown, when the style and fonts are final:
        never below the layout's minimum (that would only push the dialog off the screen's right edge).
        """
        area = self._screen_area(parent)
        minimum = QSize(0, 0)
        if self.layout() is not None:
            self.layout().activate()      # the minimum set before the style was applied may be stale
            minimum = self.minimumSizeHint()
        # Room for the window frame and title bar inside the free area.
        width = max(minimum.width(), 640, min(1500, area.width() - 24))
        height = max(minimum.height(), 480, min(940, area.height() - 48))
        self.resize(width, height)
        self.move(area.x() + max(0, (area.width() - width) // 2), area.y() + max(0, (area.height() - height - 32) // 2))

    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._sized:      # sizes set before the first layout are only proportions; fonts are final now
            self._sized = True
            self._fit_to_screen(self.parentWidget())
            self._fit_columns()
            self._initial_sizes()

    def _fit_columns(self):
        """Columns fit their (translated) content; the identity and decision columns leave room for the dot."""
        for column, (minimum, maximum, extra) in enumerate(((92, 170, 6), (64, 110, 18), (100, 175, 18))):
            self.list.resizeColumnToContents(column)
            self.list.setColumnWidth(column, min(maximum, max(minimum, self.list.columnWidth(column) + extra)))
        self.list.setColumnWidth(COL_CHOICE, 76)
        for column in (COL_PRIORITY, COL_TIME, COL_SNR, COL_LENGTH):
            self.list.resizeColumnToContents(column)
            self.list.setColumnWidth(column, min(90, max(44, self.list.columnWidth(column) + 16)))

    def _initial_sizes(self):
        width = self.width()
        details = max(300, min(520, int(width * .28))) if width >= 1100 else 0
        listed = max(380, min(600, int(width * .42)))
        self.split.setSizes([listed, max(300, width - listed - details - 40), details])
        self.details_button.blockSignals(True)
        self.details_button.setChecked(bool(details))
        self.details_button.blockSignals(False)

    def _build_filters(self):
        rows = QVBoxLayout()
        rows.setSpacing(4)
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
        self.revised_button.setToolTip(tr('只显示身份或是否加回由人工或 Agent 改判过的候选。'))
        self.revised_button.toggled.connect(self.set_revised_filter)
        revised_layout.addWidget(self.revised_button)
        row.addWidget(revised)
        row.addStretch(1)
        rows.addLayout(row)
        second = QHBoxLayout()
        self.priority_buttons = self._segments(second, tr('优先级'), PRIORITIES, self.set_priority_filter)
        for key, button in self.priority_buttons.items():
            if key:
                button.setToolTip(tr('复核优先级只是先看哪些的提示，不是可信度。'))
        second.addSpacing(10)
        self.keep_buttons = self._segments(second, tr('加回'), KEEP_FILTERS, self.set_keep_filter)
        self.filter_count = QLabel("")
        themes.set_role(self.filter_count, 'hint')
        second.addWidget(self.filter_count)
        second.addStretch(1)
        # Two lines on laptop screens, broken only between whole keys (never inside 恢复默认).
        pairs = (tr('K 保留'), tr('O 不加回'), tr('U 恢复默认'), tr('回车 下一条'))
        joined = [pair.replace(" ", "\u00a0") for pair in pairs]
        text = " · ".join(joined) if self.width() >= 1400 else " · ".join(joined[:2]) + "\n" + " · ".join(joined[2:])
        keys = QLabel(text)
        keys.setToolTip(tr('快捷键作用于选中的候选；焦点在列表或预览上时可用。'))
        keys.setWordWrap(False)
        self.keys_hint = keys
        keys.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        themes.set_role(keys, 'hint')
        second.addWidget(keys)
        rows.addLayout(second)
        return rows

    def _segments(self, row, title, choices, handler):
        """A captioned exclusive segment bar: 全部 plus one button per choice; returns {key: button}."""
        caption = QLabel(title)
        themes.set_role(caption, 'muted')
        row.addWidget(caption)
        bar = QWidget()
        themes.set_role(bar, 'segmented')
        inner = QHBoxLayout(bar)
        inner.setContentsMargins(2, 2, 2, 2)
        inner.setSpacing(2)
        group = QButtonGroup(self)
        group.setExclusive(True)
        buttons = {}
        for key in (None, *choices):
            button = _button("", 'segment')
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=key: handler(value))
            group.addButton(button)
            inner.addWidget(button)
            buttons[key] = button
        buttons[None].setChecked(True)
        row.addWidget(bar)
        return buttons

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
        self.details_button = _button(tr('详情'), 'ghost')
        self.details_button.setCheckable(True)
        self.details_button.setChecked(True)
        self.details_button.setToolTip(tr('显示或收起右侧的缩略图、邻帧对照与改判记录。'))
        self.details_button.toggled.connect(self.show_details)
        for widget in (self.fit_button, self.focus_button, self.zoom_out_button, self.zoom_label, self.zoom_in_button):
            tools.addWidget(widget)
        tools.addStretch(1)
        tools.addWidget(self.details_button)
        column.addLayout(tools)
        self.location = TrailLocationView(session)
        column.addWidget(self.location, 1)
        # Two legend rows (identities; line styles) so the preview can be narrow on laptops.
        legend, styles = QHBoxLayout(), QHBoxLayout()
        legend.setSpacing(4)
        styles.setSpacing(4)
        self.legend_swatches = {}
        entries = [(key, tr(text)) for key, text in LABELS.items()]
        entries += [("kept", tr('会加回')), ("omitted", tr('不加回')), ("selected", tr('选中'))]
        for index, (kind, text) in enumerate(entries):
            row = legend if index < len(LABELS) else styles
            if index == len(LABELS) + 2:
                row.addSpacing(4)
            swatch = TrailSwatch(kind)
            swatch.setAccessibleName(text)
            label = QLabel(text)
            themes.set_role(label, 'hint')
            row.addWidget(swatch)
            row.addWidget(label)
            row.addSpacing(6)
            self.legend_swatches[kind] = swatch
        legend.addStretch(1)
        styles.addStretch(1)
        column.addLayout(legend)
        column.addLayout(styles)
        hint = QLabel(tr('滚轮缩放，拖动空白处或按住中键平移，双击空白处复原；Ctrl 点选可多选。'))
        themes.set_role(hint, 'hint')
        hint.setWordWrap(True)
        column.addWidget(hint)
        self.fit_button.clicked.connect(self.location.fit_view)
        self.focus_button.clicked.connect(self.focus_selection)
        self.zoom_out_button.clicked.connect(lambda: self.location.zoom_to(self.location.zoom / self.location.STEP))
        self.zoom_in_button.clicked.connect(lambda: self.location.zoom_to(self.location.zoom * self.location.STEP))
        self.location.viewChanged.connect(self._update_zoom_label)
        return box

    def _build_details(self):
        """The collapsible side panel: scan thumbnail, the selected candidate, revision, real neighbours."""
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(4, 0, 6, 0)
        column.setSpacing(4)
        heading = QLabel(tr('扫描缩略图'))
        themes.set_role(heading, 'muted')
        column.addWidget(heading)
        self.thumbnail = QLabel(tr('未选择'))
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setMinimumHeight(60)
        self.thumbnail.setWordWrap(True)
        self.thumbnail.setAccessibleName(tr('扫描缩略图'))
        column.addWidget(self.thumbnail)
        self.thumbnail_caption = QLabel(tr('左：本张曝光；右：减去堆栈结果后剩下的部分。'))
        self.thumbnail_caption.setWordWrap(True)
        themes.set_role(self.thumbnail_caption, 'hint')
        column.addWidget(self.thumbnail_caption)
        self.thumbnail_note = QLabel("")
        self.thumbnail_note.setWordWrap(True)
        themes.set_role(self.thumbnail_note, 'hint')
        column.addWidget(self.thumbnail_note)
        column.addSpacing(6)
        self.summary = QLabel(tr('选择候选查看来源与邻帧。'))
        self.summary.setWordWrap(True)
        column.addWidget(self.summary)
        column.addWidget(self._build_revision())
        column.addSpacing(6)
        # The real neighbouring exposures side by side, so a trail that continues is seen at a glance.
        panels = QHBoxLayout()
        panels.setSpacing(6)
        self.context_panels = []
        for title in (tr('前一曝光'), tr('本次曝光'), tr('后一曝光')):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            label = QLabel(title)
            label.setWordWrap(True)
            image = QLabel(tr('未选择'))
            image.setMinimumHeight(90)
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setWordWrap(True)
            cell.addWidget(label)
            cell.addWidget(image)
            cell.addStretch(1)
            panels.addLayout(cell, 1)
            self.context_panels.append((label, image))
        column.addLayout(panels)
        self.context_notice = QLabel(tr('邻帧图使用实际曝光，不将相邻候选误当成相邻曝光。'))
        self.context_notice.setWordWrap(True)
        themes.set_role(self.context_notice, 'hint')
        column.addWidget(self.context_notice)
        column.addSpacing(6)
        # Why the scan suggested this identity (its own words, in Chinese for the Chinese interface);
        # below the neighbours, which are the evidence to look at first.
        self.reasons = QLabel("")
        self.reasons.setWordWrap(True)
        themes.set_role(self.reasons, 'hint')
        column.addWidget(self.reasons)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(box)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(220)
        self.details = scroll
        return scroll

    def show_details(self, shown):
        """Show or collapse the side panel; the preview takes its room."""
        sizes = self.split.sizes()
        if shown and sizes[2] == 0:
            width = max(300, min(520, int(self.width() * .28)))
            sizes = [sizes[0], max(200, sizes[1] - width), width]
        elif not shown and sizes[2]:
            sizes = [sizes[0], sizes[1] + sizes[2], 0]
        self.split.setSizes(sizes)
        self._splitter_moved()

    def _splitter_moved(self, *args):
        shown = self.split.sizes()[2] > 0
        if self.details_button.isChecked() != shown:
            self.details_button.blockSignals(True)
            self.details_button.setChecked(shown)
            self.details_button.blockSignals(False)

    def _image_width(self):
        return max(160, self.details.viewport().width() - 14)

    def _image_height(self):
        """Smaller images on short screens, so the neighbours stay in view under the thumbnail."""
        return 160 if self.height() >= 800 else 110

    def _panel_width(self):
        """One of the three neighbour panels side by side."""
        return max(70, (self.details.viewport().width() - 14 - 12) // 3)

    def _build_revision(self):
        """底片改判: the recorded value struck through, the new choice, and who chose it."""
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self.revision_rows = {}
        for row, (key, title) in enumerate((("label", tr('身份')), ("override", tr('是否加回')))):
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
            if candidate.get("review_disposition"):
                return record_text(candidate["review_disposition"])
            decision = candidate.get("decision")
            return action_text(decision) if decision in ACTIONS else tr('待定')
        state = candidate_state(candidate, meteor_default=True, layered=self._layered(candidate["id"]))
        return tr(disposition_text(state, True))

    def _disposition(self, candidate, state):
        decision = self.decisions.get(candidate["id"]) or {}
        if not self._meteor_default() and "override" not in decision and candidate.get("review_disposition"):
            return record_text(candidate["review_disposition"])
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
                new = action_text(decision["override"])
            elif self._meteor_default():
                new = tr(disposition_text(self.candidate_state(candidate_id), True))
            else:
                new = tr('默认不加回')
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
        elif event.type() == QEvent.Type.KeyPress and watched is getattr(self, "list", None) and self._review_key(event):
            return True     # before the list's own type-to-search takes the letter
        return super().eventFilter(watched, event)

    # -- keyboard: K keep, O omit (不加回), U reset, Enter next ------------------------------------

    _KEY_ACTIONS = {Qt.Key.Key_K: "keep", Qt.Key.Key_O: "omit", Qt.Key.Key_U: None}

    def _review_key(self, event):
        """Handle one review key press; True when it was a review key."""
        modifiers = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                         | Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.MetaModifier)
        if modifiers != Qt.KeyboardModifier.NoModifier or (event.isAutoRepeat() and event.key() in self._KEY_ACTIONS):
            return False
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.next_to_review()
            return True
        if key in self._KEY_ACTIONS:
            self.set_override_for_selection(self._KEY_ACTIONS[key])
            return True
        return False

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        """Keys that no focused control used (the preview, a button) reach the review keys here."""
        if self._review_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def set_override_for_selection(self, action):
        """Keep, omit or reset ("恢复默认") the selected shown rows, as one undoable step."""
        ids = self.selected_ids()
        current = self.list.currentItem()
        if not ids and current is not None and not current.isHidden():
            ids = [current.data(0, Qt.ItemDataRole.UserRole)]
        if not ids:
            self.status.setText(tr('先选择候选。'))
            return False
        self._choose(ids, "unchanged", action)
        return True

    def next_to_review(self):
        """Select the next shown row below the current one (wrapping) that has no choice yet."""
        count = self.list.topLevelItemCount()
        current = self.list.currentItem()
        start = self.list.indexOfTopLevelItem(current) + 1 if current is not None else 0
        for step in range(count):
            item = self.list.topLevelItem((start + step) % count)
            if item.isHidden():
                continue
            cid = item.data(0, Qt.ItemDataRole.UserRole)
            if not self.decisions.get(cid) and not self._states[cid]["revised"]:
                self.select_candidate(cid)
                return True
        self.status.setText(tr('当前筛选下的候选都已有选择。') if self.shown_order() else tr('当前筛选下没有候选。'))
        return False

    # -- sorting ---------------------------------------------------------------------------------

    def sort_by(self, column, descending=None):
        """Sort the rows by ``column`` (candidate, priority, time, brightness, length); ids never change.

        ``descending`` None: a second click on the same column reverses it; a new column starts with
        brightest / longest first and otherwise ascending.
        """
        if column not in SORTABLE:
            return
        last, last_descending = self.list.sort_state
        if descending is None:
            descending = (not last_descending) if column == last else column in _FIRST_DESCENDING
        self.list.sort_state = (column, bool(descending))
        self.list.sortItems(column, Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder)
        current = self.list.currentItem()
        if current is not None and not current.isHidden():
            self.list.scrollToItem(current)

    def _header_clicked(self, column):
        if column in SORTABLE:
            self.sort_by(column)
        else:   # identity / decision / choice change while reviewing: filter them instead
            last, descending = self.list.sort_state
            self.list.header().setSortIndicator(last, Qt.SortOrder.DescendingOrder if descending
                                                else Qt.SortOrder.AscendingOrder)

    def shown_order(self):
        """Candidate ids of the shown rows, top to bottom."""
        rows = (self.list.topLevelItem(i) for i in range(self.list.topLevelItemCount()))
        return [item.data(0, Qt.ItemDataRole.UserRole) for item in rows if not item.isHidden()]

    # -- filters ---------------------------------------------------------------------------------

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

    def set_priority_filter(self, priority):
        """Show only candidates with this review priority (None: all)."""
        self._filter_priority = priority if priority in PRIORITIES else None
        self._pinned.clear()
        button = self.priority_buttons[self._filter_priority]
        if not button.isChecked():
            button.setChecked(True)
        self._apply_filter()

    def set_keep_filter(self, value):
        """'keep': only what will be added back; 'omit': only what will not (None: all)."""
        self._filter_keep = value if value in KEEP_FILTERS else None
        self._pinned.clear()
        button = self.keep_buttons[self._filter_keep]
        if not button.isChecked():
            button.setChecked(True)
        self._apply_filter()

    def _filtering(self):
        return (self._filter_identity is not None or self._filter_revised or self._filter_priority is not None
                or self._filter_keep is not None)

    def _matches(self, candidate_id):
        state = self._states[candidate_id]
        return ((self._filter_identity is None or state["identity"] == self._filter_identity)
                and (not self._filter_revised or state["revised"])
                and (self._filter_priority is None or self._priorities.get(candidate_id) == self._filter_priority)
                and (self._filter_keep is None or state["keep"] == (self._filter_keep == "keep")))

    def visible_ids(self):
        return [cid for cid, item in self._items.items() if not item.isHidden()]

    def _apply_filter(self):
        shown, matching = set(), 0
        for cid, item in self._items.items():
            matches = self._matches(cid)
            matching += matches
            visible = matches or cid in self._pinned
            if visible:
                shown.add(cid)
            if item.isHidden() == visible:
                item.setHidden(not visible)
            if not visible and item.isSelected():
                item.setSelected(False)
        self.location.set_visible(None if len(shown) == len(self._items) else shown)
        self.filter_count.setText(tr('显示 {0} / {1} 条', len(shown), len(self._items)) if self._items else "")
        # Pinned rows (just changed, no longer in the filter) are shown but never set in one step.
        self.bulk_button.setEnabled(self._filtering() and matching > 0)
        current = self.list.currentItem()
        if current is not None and current.isHidden():
            order = self.shown_order()
            if order:
                self.list.setCurrentItem(self._items[order[0]])
            else:
                self.list.setCurrentItem(None)
                self.location.set_current(None)
                self.summary.setText(tr('当前筛选下没有候选。'))
        self._selection_changed()

    def _update_filter_buttons(self):
        counts = {key: 0 for key in LABELS}
        revised = kept = 0
        for state in self._states.values():
            counts[state["identity"]] += 1
            revised += bool(state["revised"])
            kept += bool(state["keep"])
        for key, button in self.filter_buttons.items():
            name = tr('全部') if key is None else tr(LABELS[key])
            button.setText(f"{name} {len(self._states) if key is None else counts[key]}")
            button.setAccessibleName(button.text())
        self.revised_button.setText(f"{tr('已改判')} {revised}")
        self.revised_button.setAccessibleName(self.revised_button.text())
        priorities = {key: 0 for key in PRIORITIES}
        for value in self._priorities.values():
            if value in priorities:
                priorities[value] += 1
        for key, button in self.priority_buttons.items():
            button.setText(tr('全部') if key is None else f"{tr(PRIORITIES[key])} {priorities[key]}")
            button.setAccessibleName(tr('优先级') + " " + button.text())
        values = {None: None, "keep": kept, "omit": len(self._states) - kept}
        for key, button in self.keep_buttons.items():
            button.setText(tr('全部') if key is None else f"{tr(KEEP_FILTERS[key])} {values[key]}")
            button.setAccessibleName(tr('加回') + " " + button.text())

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
                            "".join("\n· " + suggestion_text(reason) for reason in reasons))
            disposition = self._disposition(candidate, state)
            item.setText(2, disposition)
            item.setData(2, DOT_ROLE, state["keep_source"])
            item.setToolTip(2, tr('是否加回：{0} · 来源：{1}', disposition, _source_text(state["keep_source"])) + policy_note)
            decision = self.decisions.get(cid, {})
            parts = []
            if "label" in decision:
                parts.append(tr(LABELS.get(decision["label"], '清除身份')))
            if "override" in decision:
                reset = tr('恢复默认') if meteor_default else tr('默认不加回')
                parts.append(action_text(decision["override"]) if decision["override"] in ACTIONS else reset)
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
            self.status.setText(tr('先选择候选和要更改的身份或是否加回。'))
            return
        self._choose([item.data(0, Qt.ItemDataRole.UserRole) for item in items], label, action)

    def _choose(self, ids, label, action):
        """Record one undoable choice for ``ids``; rows that leave the current filter stay shown until it changes."""
        self._push_undo()
        for cid in ids:
            entry = self.decisions.setdefault(cid, {})
            if label != "unchanged":
                entry["label"] = label
            if action != "unchanged":
                entry["override"] = action
            if self._filtering():
                self._pinned.add(cid)
        self._refresh_decisions()

    def bulk_choice(self, action, confirm=None):
        """Keep or omit every row that matches the current filter, after asking; one undo step restores it.

        Rows changed a moment ago that no longer match stay shown (pinned) but are left as they are.
        """
        shown = self.shown_order()
        ids = [cid for cid in shown if self._matches(cid)]
        if not self._filtering() or not ids:
            self.status.setText(tr('先选一个筛选，再对筛选结果整批设置。'))
            return False
        if not (confirm or self._confirm_bulk)(action, len(ids), len(shown) - len(ids)):
            return False
        self._choose(ids, "unchanged", action)
        self.status.setText(tr('已把当前筛选的 {0} 条设为“{1}”；“撤销上一步”可恢复。', len(ids), action_text(action)))
        return True

    def _confirm_bulk(self, action, count, left_out=0):
        buttons = QMessageBox.StandardButton
        if action == "keep":
            text = tr('把当前筛选的 {0} 条全部设为保留？\n没看过的候选里可能有飞机或卫星，保留后会加进含亮迹版本。“撤销上一步”可恢复。', count)
        else:
            text = tr('把当前筛选的 {0} 条全部设为不加回？\n“撤销上一步”可恢复。', count)
        if left_out:
            text += tr('\n另有 {0} 条刚改过、已不在此筛选，不会改动。', left_out)
        box = QMessageBox(themes.message_icon(self), tr('整批设置'), text, buttons.Ok | buttons.Cancel, self)
        box.button(buttons.Ok).setText(tr('全部保留') if action == "keep" else tr('全部不加回'))
        box.button(buttons.Cancel).setText(tr('取消'))
        box.setDefaultButton(buttons.Cancel)
        box.setEscapeButton(buttons.Cancel)
        box.exec()
        return box.clickedButton() is box.button(buttons.Ok)

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

    # -- composite list ------------------------------------------------------------------------

    def _update_export_button(self):
        available = trail_manifest.published_keeps(self.session, self.workflow) is not None
        self.export_button.setEnabled(available)
        self.export_button.setToolTip(
            tr('把含亮迹版本加回的每条流星（来自哪张原片、拍摄时刻、位置）存成 CSV 和 JSON，保存在你选的位置。')
            if available else tr('这次复核还没有对应的含亮迹版本，暂时不能导出合成清单。'))

    def export_manifest(self, path=None, confirm_overwrite=None):
        """Write the composite list of the published trails version as CSV + JSON; returns their paths or None.

        Read only: the list describes what was added back to the delivered photograph, not unapplied
        draft choices. ``path`` None asks where to save; the works folder is refused. The file dialog asks
        before replacing the CSV; an existing JSON beside it is replaced only after asking too.
        """
        from starstack.core import load_job
        try:
            job = load_job(self.session["job_dir"])
            originals = trail_manifest.with_cache_records(job, self.originals, self.cache_root)
            manifest = trail_manifest.build(self.session, job, self.workflow, originals=originals, zone=LOCAL_ZONE)
        except Exception as exc:
            self._set_status_error(True)
            self.status.setText(translate_message(str(exc)))
            return None
        if path is None:
            folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            name = tr('合成清单') + "-" + (manifest["header"]["trails_folder"] or "aester") + ".csv"
            path, _ = QFileDialog.getSaveFileName(self, tr('导出合成清单'), str(Path(folder) / name),
                                                  tr('CSV 表格 (*.csv)'))
            if not path:
                return None
        if trail_manifest.inside_works(path, self.workflow):
            self._set_status_error(True)
            self.status.setText(tr('作品文件夹只放成片图像，请把清单存到别处。'))
            return None
        json_path = trail_manifest.paths(path)[1]
        if json_path.exists() and not (confirm_overwrite or self._confirm_overwrite)(json_path.name):
            return None
        try:
            written = trail_manifest.write(manifest, path)
        except OSError as exc:
            self._set_status_error(True)
            self.status.setText(translate_message(str(exc)))
            return None
        self._set_status_error(False)
        pending = len(self.decisions)
        self.status.setText(tr('已导出合成清单：加回 {0} 条。\n{1}\n{2}', len(manifest["rows"]), *written) +
                            (tr('\n清单按已出图的版本；本窗口里还没应用的选择不在其中。') if pending else ""))
        return written

    def _confirm_overwrite(self, name):
        buttons = QMessageBox.StandardButton
        box = QMessageBox(themes.message_icon(self), tr('导出合成清单'),
                          tr('旁边已有同名的 {0}，导出时会一起替换。替换吗？', name), buttons.Ok | buttons.Cancel, self)
        box.button(buttons.Ok).setText(tr('替换'))
        box.button(buttons.Cancel).setText(tr('取消'))
        box.setDefaultButton(buttons.Cancel)
        box.setEscapeButton(buttons.Cancel)
        box.exec()
        return box.clickedButton() is box.button(buttons.Ok)

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
        # The file name only: a full path does not wrap in the side panel (it stays in the tooltip).
        self.summary.setText(tr("{0} · {1}\n已记录身份：{2}；是否加回：{3}。 ",
                             cid, _file_name(candidate.get('input')) + _exposure_text(candidate),
                             tr(LABELS[self._recorded_identity(candidate)]),
                             self._recorded_disposition(candidate)) + record_text(review.get("message") or ""))
        self.summary.setToolTip(str(candidate.get('input') or ''))
        reasons = [suggestion_text(r) for r in ((candidate.get("suggestion") or {}).get("reasons") or [])]
        self.reasons.setText(tr('算法建议：{0}', tr(LABELS[suggested_identity(candidate)])) +
                             "".join("\n· " + reason for reason in reasons))
        self._show_thumbnail(candidate)
        self._context_token += 1
        self._context_pending = (self._context_token, cid)
        for _, label in self.context_panels:
            label.clear()
            label.setText(tr('读取实际曝光…'))
        self._start_pending_context()

    def _show_thumbnail(self, candidate):
        """The scan thumbnail (source | residual), with the view's note on how far it can be trusted."""
        path = candidate.get("review_png")
        pixmap = _pixmap(path, self._image_width(), self._image_height()) if path else QPixmap()
        self.thumbnail.clear()
        if pixmap.isNull():
            self.thumbnail.setText(tr('此候选没有扫描缩略图。') if not path else tr('扫描缩略图无法读取。'))
            self.thumbnail.setToolTip("")
        else:
            self.thumbnail.setPixmap(pixmap)
            self.thumbnail.setToolTip(str(path))
        # The side panel scrolls: let the image take its own height instead of being squeezed.
        self.thumbnail.setMinimumHeight(max(60, pixmap.height()))
        self.thumbnail_caption.setVisible(not pixmap.isNull())
        note = candidate.get("preview_note")
        self.thumbnail_note.setText(translate_message(note) if note else "")
        self.thumbnail_note.setVisible(bool(note))

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
                pixmap = _pixmap(panel["path"], self._panel_width(), self._image_height())
                label.setPixmap(pixmap)
                label.setMinimumHeight(max(90, pixmap.height()))
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
