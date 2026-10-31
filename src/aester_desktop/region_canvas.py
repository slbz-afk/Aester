# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Interactive annotation in normalized image coordinates, never window coordinates."""
from __future__ import annotations
import contextlib
import copy
import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from starstack import overlay_colors as oc
from starstack import regions
from . import themes
from starstack.region_preview import overlay_rgba


def is_night(widget) -> bool:
    """Whether the owning window shows the red-only night palette (read at paint time)."""
    return themes.colors_for(widget) is themes.NIGHT


def night_display(image: QImage) -> QImage:
    """Display-only night filter: Rec. 709 luma in the red channel, green = blue = 0.

    Used only for on-screen painting in night vision; never for exported pixels.
    """
    source = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = source.width(), source.height()
    if not width or not height:
        return source
    stride = source.bytesPerLine()
    data = np.frombuffer(source.constBits(), np.uint8, count=stride * height).reshape(height, stride)
    rgba = data[:, :width * 4].reshape(height, width, 4)
    luma = rgba[..., :3].astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32)
    result = np.zeros_like(rgba)
    result[..., 0] = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    result[..., 3] = rgba[..., 3]
    output = QImage(result.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
    output.setDevicePixelRatio(image.devicePixelRatio())
    return output


_RED_ONLY = [False]  # true only while vector strokes are painted onto a night frame


@contextlib.contextmanager
def red_only():
    """Paint vector strokes directly in night colours (luma in red, green = blue = 0).

    Equivalent to ``night_display`` of the same strokes (luma is linear, so blending the
    luma equals the luma of the blend), without filtering the whole frame on every mouse move.
    """
    previous = _RED_ONLY[0]
    _RED_ONLY[0] = True
    try:
        yield
    finally:
        _RED_ONLY[0] = previous


def shown_color(color) -> QColor:
    """``color`` as painted now: unchanged by day, its luma in the red channel in night vision."""
    color = QColor(color)
    if not _RED_ONLY[0]:
        return color
    luma = int(round(.2126 * color.red() + .7152 * color.green() + .0722 * color.blue()))
    return QColor(min(255, luma), 0, 0, color.alpha())


class NightFrameCache:
    """The night-filtered static layer of a view, rebuilt only when its key changes."""

    def __init__(self):
        self._key = None
        self._image = QImage()

    def get(self, widget: QWidget, key, paint) -> QImage:
        ratio = widget.devicePixelRatioF()
        key = (key, widget.width(), widget.height(), ratio)
        if key != self._key:
            buffer = QImage(widget.size() * ratio, QImage.Format.Format_ARGB32_Premultiplied)
            buffer.setDevicePixelRatio(ratio)
            painter = QPainter(buffer)
            painter.setFont(widget.font())
            paint(painter)
            painter.end()
            self._key, self._image = key, night_display(buffer)
        return self._image


def overlay_pens(name: str, *, width: float = oc.LINE_WIDTH, style: str | None = None,
                 alpha: int = 255) -> tuple[QPen, QPen]:
    """(casing, stroke) pens for an overlay colour: draw the dark casing first."""
    style = style or oc.OVERLAYS[name]['line']
    color = QColor(oc.OVERLAYS[name]['hex'])
    color.setAlpha(alpha)
    color = shown_color(color)
    outline = QColor(oc.OUTLINE)
    outline.setAlpha(alpha)
    outline = shown_color(outline)
    casing = QPen(outline, width + oc.OUTLINE_EXTRA)
    casing.setCapStyle(Qt.PenCapStyle.RoundCap)
    casing.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    stroke = QPen(color, width)
    stroke.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if style == oc.DASH:
        stroke.setDashPattern(list(oc.DASH_PATTERN))
        stroke.setCapStyle(Qt.PenCapStyle.FlatCap)
    elif style == oc.DOT:
        stroke.setDashPattern(list(oc.DOT_PATTERN))
        stroke.setCapStyle(Qt.PenCapStyle.RoundCap)
    return casing, stroke


def cased(painter: QPainter, name: str, draw, **options) -> None:
    """Stroke ``draw(painter)`` twice: the OUTLINE casing, then the overlay colour."""
    casing, stroke = overlay_pens(name, **options)
    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for pen in (casing, stroke):
        painter.setPen(pen)
        draw(painter)
    painter.restore()


def draw_handle(painter: QPainter, point: QPointF, *, square: bool = False,
                radius: float = oc.HANDLE_RADIUS) -> None:
    """A HANDLE-yellow control point with a dark ring; a square marks the feather handle."""
    painter.save()
    painter.setPen(QPen(shown_color(oc.OUTLINE), 2))
    painter.setBrush(shown_color(oc.HANDLE))
    if square:
        painter.drawRect(QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2))
    else:
        painter.drawEllipse(point, radius, radius)
    painter.restore()


def draw_double_ellipse(painter: QPainter, rx: float, ry: float, *, alpha: int = 255) -> None:
    """Suzhou-style Milky Way: two parallel MILKYWAY strokes around (0, 0), each cased."""
    half = oc.DOUBLE_GAP / 2 + oc.LINE_WIDTH / 2
    rings = [QRectF(-(rx + d), -(ry + d), 2 * (rx + d), 2 * (ry + d))
             for d in (-half, half) if rx + d > 0 and ry + d > 0]
    casing, stroke = overlay_pens('milkyway', style=oc.SOLID, alpha=alpha)
    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for pen in (casing, stroke):
        painter.setPen(pen)
        for ring in rings:
            painter.drawEllipse(ring)
    painter.restore()


class OverlaySwatch(QWidget):
    """A short line sample of one overlay style for the editor legend."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(34, 16)

    def sizeHint(self):
        return QSize(34, 16)

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()
        super().changeEvent(event)

    def render_image(self) -> QImage:
        """The swatch as drawn on the photo surround (before any night display filter)."""
        ratio = self.devicePixelRatioF()
        image = QImage(self.size() * ratio, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(QColor(themes.SURROUNDS['xuan']))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y, x0, x1 = self.height() / 2, 3., self.width() - 3.
        if self.kind == 'milkyway':
            half = oc.DOUBLE_GAP / 2 + oc.LINE_WIDTH / 2
            cased(painter, 'milkyway', lambda p: (p.drawLine(QPointF(x0, y - half), QPointF(x1, y - half)),
                                                  p.drawLine(QPointF(x0, y + half), QPointF(x1, y + half))),
                  style=oc.SOLID)
        elif self.kind == 'handle':
            cased(painter, 'horizon', lambda p: p.drawLine(QPointF(x0, y), QPointF(x1, y)))
            draw_handle(painter, QPointF(self.width() / 2, y), radius=4.)
        else:
            cased(painter, self.kind, lambda p: p.drawLine(QPointF(x0, y), QPointF(x1, y)))
        painter.end()
        return image

    def paintEvent(self, event):
        image = self.render_image()
        QPainter(self).drawImage(0, 0, night_display(image) if is_night(self) else image)


class OverlayLegend(QWidget):
    """Legend row: each overlay named in the interface language beside its line style."""

    def __init__(self, entries, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.swatches = {}
        self.labels = {}
        for kind, text in entries:
            swatch = OverlaySwatch(kind, self)
            swatch.setAccessibleName(text)
            label = QLabel(text, self)
            themes.set_role(label, 'muted')
            layout.addWidget(swatch)
            layout.addWidget(label)
            layout.addSpacing(10)
            self.swatches[kind] = swatch
            self.labels[kind] = label
        layout.addStretch(1)


class RegionCanvas(QWidget):
    changed = Signal()

    def __init__(self, image: QImage, project: dict, parent=None):
        super().__init__(parent)
        self.image = image
        self.project = copy.deepcopy(project)
        self.mode = 'horizon'
        self.radius = .018
        self.pending = []
        self._drag = False
        self._start = None
        self._drag_kind = None
        self._drag_project = None
        self._hover = None
        self._undo, self._redo = [], []
        self._overlay = QImage()
        self.suggestion = None
        self.zoom = 1.
        self.pan = QPointF()
        self._pan_start = None
        self.surround = None  # a SURROUNDS hex overrides the theme's photo surround
        self._night_frame = NightFrameCache()
        self._kept_point = None  # a single horizon click kept while another tool is used
        self.setMinimumSize(400, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._refresh_overlay()

    def image_rect(self):
        bounds = self.rect().adjusted(12, 12, -12, -12)
        if self.image.isNull():
            return QRectF()
        scale = min(bounds.width() / self.image.width(), bounds.height() / self.image.height()) * self.zoom
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF(bounds.x() + (bounds.width() - width) / 2 + self.pan.x(),
                      bounds.y() + (bounds.height() - height) / 2 + self.pan.y(), width, height)

    def fit_view(self):
        self.zoom = 1.
        self.pan = QPointF()
        self.update()

    def wheelEvent(self, event):
        point = self.normalized_point(event.position())
        if point is None or self._drag:
            return
        self.zoom = min(8., max(.2, self.zoom * (1.25 if event.angleDelta().y() > 0 else .8)))
        moved = self.screen_point(point)
        self.pan += event.position() - moved
        if self.zoom == 1.:
            self.pan = QPointF()
        self.update()
        event.accept()

    def normalized_point(self, position, *, outside=False):
        rect = self.image_rect()
        if rect.isEmpty() or (not outside and not rect.contains(position)):
            return None
        return [(position.x() - rect.x()) / rect.width(),
                (position.y() - rect.y()) / rect.height()]

    def brush_screen_radius(self):
        return self.radius * min(self.image_rect().width(), self.image_rect().height())

    def ellipse_handles(self):
        mw = self.project['milkyway']
        if not mw.get('ellipse'):
            return {}
        cx, cy, rx, ry = mw['ellipse']
        rect = self.image_rect()
        center = self.screen_point([cx, cy])
        angle = math.radians(mw.get('angle_degrees', 0.))
        u, v = QPointF(math.cos(angle), math.sin(angle)), QPointF(-math.sin(angle), math.cos(angle))
        ax, ay = rx * rect.width(), ry * rect.height()
        return {'center': center, 'east': center + u * ax, 'west': center - u * ax,
                'south': center + v * ay, 'north': center - v * ay,
                'rotate': center - v * (ay + 28),
                'feather': center + u * (ax * (1 - mw['feather']))}

    def _hit_ellipse(self, position):
        handles = self.ellipse_handles()
        for name, point in handles.items():
            if (point - position).manhattanLength() <= 13:
                return name
        if handles:
            mw = self.project['milkyway']
            delta = position - handles['center']
            angle = math.radians(mw.get('angle_degrees', 0.))
            x = math.cos(angle) * delta.x() + math.sin(angle) * delta.y()
            y = -math.sin(angle) * delta.x() + math.cos(angle) * delta.y()
            rect = self.image_rect()
            if (x / (mw['ellipse'][2] * rect.width())) ** 2 + (y / (mw['ellipse'][3] * rect.height())) ** 2 < 1:
                return 'center'
        return 'new'

    def clear_milkyway(self):
        self.commit(regions.clear_milkyway(self.project))

    def clear_landscape(self):
        self.pending = []
        self.commit(regions.clear_landscape(self.project))

    def screen_point(self, point):
        rect = self.image_rect()
        return QPointF(rect.x() + point[0] * rect.width(), rect.y() + point[1] * rect.height())

    def set_mode(self, mode):
        # Leaving the horizon tool never drops clicked points silently: two or more points
        # become a finished (undoable) horizon; a single point is kept for when it returns.
        if self.mode == 'horizon' and mode != 'horizon':
            if len(self.pending) >= 2:
                self.finish_horizon()
            elif len(self.pending) == 1:
                self._kept_point = list(self.pending)
        self.mode = mode
        if mode == 'horizon' and self._kept_point:
            self.pending, self._kept_point = self._kept_point, None
            self.suggestion = None
            self.update()
            return
        self.pending = []
        self.suggestion = None
        self.update()

    def commit(self, project):
        if project['revision'] == self.project['revision']:
            return
        self._undo.append(copy.deepcopy(self.project))
        self._redo.clear()
        self.project = copy.deepcopy(project)
        self._refresh_overlay()
        self.changed.emit()

    def undo(self):
        if self.pending:
            self.pending.pop()
            self.update()
        elif self._undo:
            self._redo.append(self.project)
            self.project = self._undo.pop()
            self.suggestion = None
            self._refresh_overlay()
            self.changed.emit()

    def redo(self):
        if self._redo:
            self._undo.append(self.project)
            self.project = self._redo.pop()
            self._refresh_overlay()
            self.changed.emit()

    def finish_horizon(self):
        if len(self.pending) < 2:
            return False
        unique = {p[0]: p for p in self.pending}
        points = [unique[x] for x in sorted(unique)]
        if len(points) < 2:
            return False
        self.commit(regions.revise(self.project, scene_mode='landscape',
                                  operations=self.project['operations'] + [{'kind': 'horizon', 'points': points}]))
        self.pending = []
        self.update()
        return True

    def accept_suggestion(self):
        if self.suggestion:
            operations = copy.deepcopy(self.project['operations'])
            index = next(i for i in range(len(operations) - 1, -1, -1) if operations[i]['kind'] == 'horizon')
            operations[index] = {'kind': 'horizon', 'points': self.suggestion}
            self.commit(regions.revise(self.project, scene_mode='landscape',
                                      operations=operations))
            self.suggestion = None
            self.update()

    def _refresh_overlay(self):
        if self.image.isNull():
            return
        sky, mw = regions.rasterize(self.project, self.image.width(), self.image.height())
        rgba = overlay_rgba(sky, mw)
        self._overlay = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        self.update()

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not is_night(self):
            self._paint(painter)
            return
        # Night vision, display only: the photo and region fill go through the red-only
        # filter once (cached); strokes and handles are painted directly in night colours.
        rect = self.image_rect()
        key = (self.image.cacheKey(), self._overlay.cacheKey(), rect.x(), rect.y(), rect.width(),
               rect.height(), self.surround)
        painter.drawImage(0, 0, self._night_frame.get(self, key, lambda p: self._paint_base(p, night=True)))
        with red_only():
            self._paint_marks(painter)

    def _paint(self, painter):
        self._paint_base(painter, night=is_night(self))
        self._paint_marks(painter)

    def _paint_base(self, painter, *, night=False):
        # The photo surround is neutral and independent of the light/dark theme.
        surround = self.surround or (themes.NIGHT['canvas'] if night
                                     else themes.SURROUNDS[themes.DEFAULT_SURROUND])
        painter.fillRect(self.rect(), QColor(surround))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        painter.drawImage(rect, self._overlay)

    def _paint_marks(self, painter):
        rect = self.image_rect()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mw = self.project['milkyway']
        editing_mw = self.mode == 'milkyway'
        if mw.get('ellipse'):
            center = self.screen_point(mw['ellipse'][:2])
            rx, ry = mw['ellipse'][2] * rect.width(), mw['ellipse'][3] * rect.height()
            painter.save()
            painter.translate(center)
            painter.rotate(mw.get('angle_degrees', 0.))
            draw_double_ellipse(painter, rx, ry, alpha=255 if editing_mw else 190)
            if editing_mw:
                f = 1 - mw['feather']
                cased(painter, 'milkyway', lambda p: p.drawEllipse(QRectF(-rx*f, -ry*f, rx*2*f, ry*2*f)),
                      width=1.5, style=oc.DASH)
            painter.restore()
        if self.suggestion:
            suggestion = [self.screen_point(point) for point in self.suggestion]
            cased(painter, 'suggestion', lambda p: p.drawPolyline(suggestion))
        if self.pending and self.mode == 'horizon':
            pending = [self.screen_point(point) for point in self.pending]
            if len(pending) > 1:
                cased(painter, 'horizon', lambda p: p.drawPolyline(pending))
            for point in pending:
                draw_handle(painter, point, radius=4.)
        if len(self.pending) > 1 and self.mode in ('ground', 'sky'):
            # The brush stroke in progress, as wide as the brush footprint.
            stroke = [self.screen_point(point) for point in self.pending]
            pen = QPen(shown_color(QColor(*oc.rgba(self.mode, 110))), max(1., 2 * self.brush_screen_radius()))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.save()
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(stroke)
            painter.restore()
            cased(painter, self.mode, lambda p: p.drawPolyline(stroke), width=1.5)
        if self._drag and editing_mw and len(self.pending) > 1:
            box = QRectF(self.screen_point(self.pending[0]), self.screen_point(self.pending[-1])).normalized()
            cased(painter, 'milkyway', lambda p: p.drawEllipse(box), style=oc.DASH)
        if editing_mw and mw.get('ellipse'):
            handles = self.ellipse_handles()
            cased(painter, 'handle', lambda p: p.drawLine(handles['north'], handles['rotate']), width=1.5)
            for name, point in handles.items():
                draw_handle(painter, point, square=name == 'feather')
        if self.mode in ('ground', 'sky') and self._hover is not None and rect.contains(self._hover):
            radius = self.brush_screen_radius()
            painter.setPen(QPen(shown_color(oc.OUTLINE), 1.5 + oc.OUTLINE_EXTRA))
            painter.setBrush(shown_color(QColor(*oc.rgba('ground' if self.mode == 'ground' else 'sky', 45))))
            painter.drawEllipse(self._hover, radius, radius)
            painter.setPen(QPen(shown_color('#ffffff'), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self._hover, radius, radius)

    def leaveEvent(self, event):
        self._hover = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.normalized_point(event.position(), outside=self.mode == 'milkyway')
        if point is None:
            return
        if self.mode == 'horizon':
            self.pending.append(point)
        else:
            self._drag = True
            self.pending = [point]
            if self.mode == 'milkyway':
                self._start = point
                self._drag_project = copy.deepcopy(self.project)
                self._drag_kind = ('new' if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                                   else self._hit_ellipse(event.position()))
        self.update()

    def mouseMoveEvent(self, event):
        self._hover = event.position()
        if self._pan_start is not None:
            self.pan += event.position() - self._pan_start
            self._pan_start = event.position()
            self.update()
            return
        point = self.normalized_point(event.position(), outside=self.mode == 'milkyway')
        if self._drag and point is not None:
            if self.mode == 'milkyway':
                self._update_ellipse_drag(point)
            else:
                self.pending.append(point)
        self.update()

    def _update_ellipse_drag(self, point):
        base = self._drag_project
        mw = base['milkyway']
        angle = mw.get('angle_degrees', 0.)
        feather = mw['feather']
        kind = self._drag_kind
        if kind == 'new':
            a = self._start
            ellipse = [(a[0]+point[0])/2, (a[1]+point[1])/2,
                       abs(point[0]-a[0])/2, abs(point[1]-a[1])/2]
            angle = 0.
        else:
            cx, cy, rx, ry = mw['ellipse']
            ellipse = [cx, cy, rx, ry]
            if kind == 'center':
                ellipse[:2] = [cx+point[0]-self._start[0], cy+point[1]-self._start[1]]
            else:
                rect = self.image_rect()
                dx, dy = (point[0]-cx)*rect.width(), (point[1]-cy)*rect.height()
                theta = math.radians(angle)
                x, y = math.cos(theta)*dx+math.sin(theta)*dy, -math.sin(theta)*dx+math.cos(theta)*dy
                if kind in ('east', 'west'):
                    ellipse[2] = abs(x)/rect.width()
                elif kind in ('north', 'south'):
                    ellipse[3] = abs(y)/rect.height()
                elif kind == 'rotate':
                    angle = ((math.degrees(math.atan2(dy, dx))+90+180) % 360)-180
                elif kind == 'feather':
                    feather = min(.95, max(.1, 1-abs(x)/(rx*rect.width())))
        if min(ellipse[2:]) < .003:
            return
        try:
            proposed = regions.draw_milkyway(base, ellipse, angle_degrees=angle, feather=feather)
        except ValueError:
            return  # Keep the last valid extent, never clip a center at the image edge.
        self.project = proposed
        self._refresh_overlay()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            return
        if not self._drag or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = False
        point = self.normalized_point(event.position(), outside=self.mode == 'milkyway')
        if self.mode == 'milkyway':
            if point is not None:
                self._update_ellipse_drag(point)
            proposed = self.project
            self.project = self._drag_project
            self.commit(proposed)
            self._drag_project = None
            self._drag_kind = None
            self.pending = []
            self.update()
            return
        if point is not None:
            self.pending.append(point)
        if self.mode in ('ground', 'sky'):
            self.commit(regions.revise(self.project, scene_mode='landscape', operations=self.project['operations'] +
                                      [{'kind': 'paint', 'target': self.mode, 'points': self.pending, 'radius': self.radius}]))
        self.pending = []
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.mode == 'horizon':
            self.finish_horizon()
