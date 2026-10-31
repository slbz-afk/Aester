# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""「拓星」 custom widgets painted with QPainter.

Colours are read from the owning window's tokens at paint time (``themes.colors_for``), so
theme and night switches repaint without rebuilding.  Hairlines use cosmetic pens; dither
textures are cached per device pixel ratio.  No user-visible text is defined here: every
label, tooltip and accessible name is passed in by the caller (already translated).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Iterable, Mapping, Sequence

from PySide6.QtCore import (QAbstractAnimation, QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, QVariantAnimation,
                            Signal)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy,
                               QSlider, QStyle, QStyleOptionSlider, QWidget)

from . import fonts, themes


STAGE_STATES = ("done", "current", "todo", "disabled")
PROVENANCE_KINDS = ("human", "agent", "algorithm")
FRAME_STATES = ("done", "skipped", "todo")


def _pen(color: str | QColor, width: float = 1.0) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    return pen


# --------------------------------------------------------------------------- motion

_SPI_GETCLIENTAREAANIMATION = 0x1042


def reduced_motion() -> bool:
    """True when interface animation should stay still (0 ms, static drawing).

    ``AESTER_REDUCED_MOTION`` (1/0, true/false) overrides; otherwise Windows' "Show
    animations in Windows" setting (SPI_GETCLIENTAREAANIMATION) is followed.  Read on every
    start, so a changed system setting applies to the next activity.
    """
    value = os.environ.get("AESTER_REDUCED_MOTION", "").strip().lower()
    if value:
        return value in ("1", "true", "yes", "on")
    if sys.platform == "win32":
        try:
            import ctypes
            enabled = ctypes.c_int(1)
            if ctypes.windll.user32.SystemParametersInfoW(_SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(enabled), 0):
                return not bool(enabled.value)
        except (AttributeError, OSError, ValueError):
            pass
    return False


def _mix(low: str, high: str, t: float) -> QColor:
    """Linear blend of two opaque colours (keeps night's G == B <= R)."""
    a, b = QColor(low), QColor(high)
    t = max(0.0, min(1.0, float(t)))
    return QColor(round(a.red() + (b.red() - a.red()) * t), round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


def sweep_level(index: int, head: float, steps: int = 28, tail: int = 7) -> float:
    """Brightness 0..1 of lodge ``index`` when the sweep head is at ``head`` (fades over ``tail``)."""
    behind = (float(head) - index) % steps
    return max(0.0, 1.0 - behind / tail)


def activity_colors(tokens: Mapping[str, str]) -> tuple[str, str]:
    """(floor, head) greys for sweeping lodge lines; night caps the head at the border red."""
    if tokens is themes.NIGHT:
        return tokens["hairline"], tokens["border"]
    return tokens["connector"], tokens["text"]


STATIC_LEVEL = 0.55      # reduced motion: every lodge at one steady level


class ActivityClock(QObject):
    """One shared sweep phase for every activity mark in a window (the 二十八宿 sweep).

    A single QVariantAnimation walks the head around the 28 lodge lines once per period;
    ``stepChanged`` fires only when the whole-number step changes (about 12 per second), so
    marks repaint at that rate and never slide.  The clock runs only while activity is on,
    motion is not reduced and the owning window is shown and not minimized.
    """

    stepChanged = Signal(int)
    STEPS = 28
    PERIOD_MS = 2400

    def __init__(self, window: QWidget | None = None, period_ms: int = PERIOD_MS):
        super().__init__(window)
        self._window = window
        self._active = False
        self._step = 0
        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(float(self.STEPS))
        self._animation.setDuration(max(1, int(period_ms)))
        self._animation.setLoopCount(-1)
        self._animation.valueChanged.connect(self._advance)
        if window is not None:
            window.installEventFilter(self)

    def step(self) -> int:
        return self._step

    def is_active(self) -> bool:
        return self._active

    def is_running(self) -> bool:
        return self._animation.state() == QAbstractAnimation.State.Running

    def is_static(self) -> bool:
        """Active but not animating (reduced motion, or paused while hidden): draw still."""
        return self._active and not self.is_running()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self._sync()

    def _set_step(self, step: int) -> None:
        step = int(step) % self.STEPS
        if step != self._step:
            self._step = step
            self.stepChanged.emit(step)

    def _advance(self, value) -> None:
        self._set_step(math.floor(float(value)))

    def _window_shown(self) -> bool:
        window = self._window
        return window is None or (window.isVisible() and not window.isMinimized())

    def _sync(self) -> None:
        run = self._active and self._window_shown() and not reduced_motion()
        state = self._animation.state()
        if run and state != QAbstractAnimation.State.Running:
            if state == QAbstractAnimation.State.Paused:
                self._animation.resume()
            else:
                self._animation.start()
        elif not run and state == QAbstractAnimation.State.Running:
            if self._active:
                self._animation.pause()           # hidden / minimized: resume where it was
            else:
                self._animation.stop()
        elif not self._active and state == QAbstractAnimation.State.Paused:
            self._animation.stop()
        self.stepChanged.emit(self._step)         # marks repaint for the new (static) state

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.WindowStateChange):
            self._sync()
        return False


def _lodge_colours(tokens: Mapping[str, str], clock: "ActivityClock | None") -> list[QColor]:
    """Colour of each of the 28 lodge lines for the clock's current state."""
    floor, head = activity_colors(tokens)
    if clock is None or not clock.is_running():
        colour = _mix(floor, head, STATIC_LEVEL)
        return [colour] * len(LODGE_LINES)
    step = clock.step()
    return [_mix(floor, head, sweep_level(index, step)) for index in range(len(LODGE_LINES))]


def paint_lodges(painter: QPainter, centre: QPointF, inner: float, outer: float,
                 colours: Sequence[QColor], width_scale: float = 1.0) -> None:
    """The 28 uneven radial lodge lines between two radii, each in its own colour."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for (angle, weight), colour in zip(LODGE_LINES, colours):
        theta = math.radians(angle)
        direction = QPointF(math.cos(theta), math.sin(theta))
        painter.setPen(_pen(colour, max(0.6, weight * width_scale)))
        painter.drawLine(centre + direction * inner, centre + direction * outer)
    painter.restore()


class LodgeSpinner(QWidget):
    """A 16 px 二十八宿 activity mark: the lodge lines with a brightness head going round.

    Fades only (each line changes brightness in place); drawn still under reduced motion.
    The caller sets the accessible name (translated) and shows or hides it.
    """

    SIZE = 16

    def __init__(self, clock: ActivityClock | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._clock: ActivityClock | None = None
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        policy = self.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)
        self.set_clock(clock)

    def clock(self) -> ActivityClock | None:
        return self._clock

    def set_clock(self, clock: ActivityClock | None) -> None:
        if self._clock is not None:
            self._clock.stepChanged.disconnect(self._tick)
        self._clock = clock
        if clock is not None:
            clock.stepChanged.connect(self._tick)
        self.update()

    def _tick(self, _step: int = 0) -> None:
        if self.isVisible():
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self.SIZE, self.SIZE)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        side = min(self.width(), self.height())
        centre = QPointF(self.width() / 2, self.height() / 2)
        paint_lodges(painter, centre, side * 0.2, side / 2 - 0.5,
                     _lodge_colours(themes.colors_for(self), self._clock), 0.8)


class ValueField(QDoubleSpinBox):
    """A typed read-out beside a slider: mono, right-aligned, no arrows, never takes the wheel.

    The value is committed on Enter or focus-out only (keyboard tracking off), so each
    commit reaches the slider once.  Positive values carry an explicit ``+``.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setProperty("role", "value")

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        text = super().textFromValue(value)
        return "+" + text if value > 0 and not text.startswith("+") else text

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()         # the wheel scrolls the panel; values change by drag, keys or typing

    def fit_width(self) -> int:
        """Room for the widest value with its suffix, in the field's own (mono) font."""
        metrics = QFontMetrics(fonts.mono_font(13))
        texts = [self.textFromValue(value) + self.suffix() for value in (self.minimum(), self.maximum())]
        return max(metrics.horizontalAdvance(text) for text in texts) + 16


class UnsignedValueField(ValueField):
    """ValueField for an absolute setting (e.g. JPEG quality 60–100): the plain number, no ``+`` sign."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        return QDoubleSpinBox.textFromValue(self, value)


class StepSpinBox(QDoubleSpinBox):
    """A number box in a scrolling column: the wheel scrolls the column, never the value.

    Sized from its own widest value (minimum or maximum with the suffix) so the label beside
    it keeps its natural single-line width.
    """

    EXTRA = 54             # frame, left padding, stepper column, the edit's own margins

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        event.ignore()

    def fit_width(self) -> int:
        metrics = QFontMetrics(fonts.mono_font(13))
        texts = [self.textFromValue(value) + self.suffix() for value in (self.minimum(), self.maximum())]
        return max(metrics.horizontalAdvance(text) for text in texts) + self.EXTRA

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self.fit_width(), super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self.fit_width(), super().minimumSizeHint().height())


class _ThemedWidget(QWidget):
    """Repaint (and drop cached textures) whenever palette or stylesheet changes."""

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange,
                            QEvent.Type.EnabledChange, QEvent.Type.FontChange):
            self._invalidate()
            self.update()
        super().changeEvent(event)

    def _invalidate(self) -> None:
        pass

    def tokens(self) -> Mapping[str, str]:
        return themes.colors_for(self)


# --------------------------------------------------------------------------- stage bar

class StageNode(QAbstractButton):
    """One star of the 星官 stage bar: dot (done / ring = current / hollow = todo), number, label."""

    PAD = 6
    DOT_BOX = 18

    def __init__(self, key: str, number: str, label: str, tooltip: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._key = key
        self._number = number
        self._state = "todo"
        self._clock: ActivityClock | None = None
        self._busy = False
        self.setText(label)
        self.setAccessibleName(label)
        self.set_tooltip(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    @property
    def key(self) -> str:
        return self._key

    def number(self) -> str:
        return self._number

    def state(self) -> str:
        return self._state

    def set_label(self, label: str) -> None:
        self.setText(label)
        self.setAccessibleName(label)
        self.updateGeometry()
        self.update()

    def set_tooltip(self, tooltip: str) -> None:
        self.setToolTip(tooltip or "")
        self.setAccessibleDescription(tooltip or "")

    def set_state(self, state: str) -> None:
        if state not in STAGE_STATES:
            raise ValueError(f"unknown stage state: {state!r}")
        self._state = state
        self.setEnabled(state != "disabled")
        self.update()

    def is_busy(self) -> bool:
        return self._busy

    def set_busy(self, busy: bool, clock: ActivityClock | None = None) -> None:
        """While work runs in this stage its ring becomes the 28 lodge ticks (sweeping or still)."""
        busy = bool(busy)
        clock = clock if busy else None
        if busy == self._busy and clock is self._clock:
            return
        if self._clock is not None:
            self._clock.stepChanged.disconnect(self._tick)
        self._busy, self._clock = busy, clock
        if clock is not None:
            clock.stepChanged.connect(self._tick)
        self.update()

    def _tick(self, _step: int = 0) -> None:
        if self.isVisible():
            # Only the ring's box repaints; the label never flickers.
            self.update(QRect(0, 0, self.PAD + self.DOT_BOX + 2, self.height()))

    def _label_font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(13)
        font.setWeight(QFont.Weight.Medium if self._state == "current" else QFont.Weight.Normal)
        return font

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        label = QFontMetrics(self._label_font()).horizontalAdvance(self.text())
        number = QFontMetrics(fonts.mono_font(11)).horizontalAdvance(self._number) if self._number else 0
        width = self.PAD + self.DOT_BOX + 6 + (number + 6 if number else 0) + label + self.PAD
        height = max(28, QFontMetrics(self._label_font()).height() + 8)
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.sizeHint()

    def dot_center(self) -> QPointF:
        return QPointF(self.PAD + self.DOT_BOX / 2, self.height() / 2)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.isEnabled():
            self.click()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange, QEvent.Type.FontChange):
            self.updateGeometry()
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        c = themes.colors_for(self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        state = self._state
        center = self.dot_center()
        if self._busy:
            # The ring becomes the 28 lodge ticks; a brightness head goes round (fades only).
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(c["text"]))
            painter.drawEllipse(center, 4.0, 4.0)
            paint_lodges(painter, center, 6.0, 9.0, _lodge_colours(c, self._clock))
        elif state in ("done", "current"):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(c["text"]))
            painter.drawEllipse(center, 4.0, 4.0)
            if state == "current":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(_pen(c["text"], 1.0))
                painter.drawEllipse(center, 8.5, 8.5)
        else:
            painter.setBrush(QColor(c["window"]))
            painter.setPen(_pen(c["disabled"] if state == "disabled" else c["border"], 1.2))
            painter.drawEllipse(center, 3.75, 3.75)
        x = self.PAD + self.DOT_BOX + 6
        hovered = self.underMouse() and self.isEnabled()
        if self._number:
            mono = fonts.mono_font(11)
            painter.setFont(mono)
            # Todo numbers are still clickable text: muted (>= 4.5:1), the hollow dot marks todo.
            number_color = {"current": c["text"], "done": c["muted"], "todo": c["muted"]}.get(state, c["disabled"])
            painter.setPen(QColor(number_color))
            width = QFontMetrics(mono).horizontalAdvance(self._number)
            painter.drawText(QRectF(x, 0, width + 1, self.height()), Qt.AlignmentFlag.AlignVCenter, self._number)
            x += width + 6
        label_font = self._label_font()
        painter.setFont(label_font)
        if state == "disabled":
            label_color = c["disabled"]
        elif state == "current" or hovered:
            label_color = c["text"]
        else:
            label_color = c["muted"]
        painter.setPen(QColor(label_color))
        painter.drawText(QRectF(x, 0, self.width() - x, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(_pen(c["focus"], 1.0))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)


class _Connector(QWidget):
    """Straight line joining two stage stars; lit when the next stage is reached."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.lit = False
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(28)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(52, 28)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(16, 28)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        c = themes.colors_for(self)
        painter = QPainter(self)
        y = math.floor(self.height() / 2) + 0.5
        painter.setPen(_pen(c["text"] if self.lit else c["connector"], 1.0))
        inset = 4 if self.width() > 24 else 2
        painter.drawLine(QPointF(inset, y), QPointF(self.width() - inset, y))


class _TrailingDotFilter(QObject):
    def __init__(self, dot: "ProvenanceDot", parent: QObject | None = None):
        super().__init__(parent)
        self.dot = dot

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self.dot.move(12, max(0, (watched.height() - self.dot.height()) // 2))
            self.dot.raise_()
        return False


class StageBar(QWidget):
    """Horizontal 星官 progress: stage nodes joined by connectors, lit up to the current node.

    Indicates progress and offers shortcuts only; it never switches pages.
    """

    stageClicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._nodes: dict[str, StageNode] = {}
        self._connectors: list[_Connector] = []
        self._trailing: QWidget | None = None
        self._trailing_dot: ProvenanceDot | None = None
        self._trailing_connector: _Connector | None = None
        self._trailing_number: QLabel | None = None
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(16, 8, 16, 8)
        self._layout.setSpacing(0)
        self.setProperty("role", "stage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # stages -------------------------------------------------------------
    def set_stages(self, stages: Iterable[Sequence[str]]) -> None:
        """``stages``: (key, number_text, label, tooltip) in order; replaces existing nodes."""
        for node in self._nodes.values():
            node.hide()
            node.deleteLater()
        for connector in self._connectors:
            connector.hide()
            connector.deleteLater()
        self._nodes, self._connectors = {}, []
        for stage in stages:
            key, number, label, *rest = stage
            if key in self._nodes:
                raise ValueError(f"duplicate stage key: {key!r}")
            if self._nodes:
                self._connectors.append(_Connector(self))
            node = StageNode(key, number, label, rest[0] if rest else "", self)
            node.clicked.connect(lambda checked=False, value=key: self.stageClicked.emit(value))
            self._nodes[key] = node
        self._rebuild()
        self._refresh_connectors()

    def keys(self) -> list[str]:
        return list(self._nodes)

    def node(self, key: str) -> StageNode:
        return self._nodes[key]

    def nodes(self) -> list[StageNode]:
        return list(self._nodes.values())

    def state(self, key: str) -> str:
        return self._nodes[key].state()

    def states(self) -> dict[str, str]:
        return {key: node.state() for key, node in self._nodes.items()}

    def set_state(self, key: str, state: str) -> None:
        self._nodes[key].set_state(state)
        self._refresh_connectors()

    def set_states(self, states: Mapping[str, str]) -> None:
        for key, state in states.items():
            self._nodes[key].set_state(state)
        self._refresh_connectors()

    def set_tooltip(self, key: str, tooltip: str) -> None:
        self._nodes[key].set_tooltip(tooltip)

    def connector_lit(self) -> list[bool]:
        return [connector.lit for connector in self._connectors]

    # trailing item -------------------------------------------------------
    def add_trailing_widget(self, widget: QWidget, provenance: str | None = None,
                            number: str | None = None) -> QWidget:
        """Place an external widget (e.g. the existing retouch QPushButton) after the stages.

        With ``provenance`` ("agent", "human", "algorithm") the widget is styled as the
        secondary stage action and carries a ProvenanceDot on its left.  With ``number``
        (e.g. "06") it becomes an optional last node: joined by a connector that never
        lights (the step is optional) and preceded by its mono number.  Without a number
        it sits at the right end.
        """
        if self._trailing is not None and self._trailing is not widget:
            self._layout.removeWidget(self._trailing)
        self._trailing = widget
        widget.setParent(self)
        if provenance is not None:
            themes.set_role(widget, "stage-action")
            if self._trailing_dot is None:
                self._trailing_dot = ProvenanceDot(provenance, widget)
                self._trailing_dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                widget.installEventFilter(_TrailingDotFilter(self._trailing_dot, widget))
            else:
                self._trailing_dot.set_kind(provenance)
            self._trailing_dot.move(12, max(0, (widget.height() - 8) // 2))
            self._trailing_dot.show()
        if number:
            if self._trailing_connector is None:
                self._trailing_connector = _Connector(self)
                self._trailing_number = QLabel(self)
                self._trailing_number.setObjectName("stageTrailingNumber")
                themes.set_role(self._trailing_number, "caption")
            self._trailing_number.setText(number)
        elif self._trailing_connector is not None:
            self._trailing_connector.hide()
            self._trailing_number.hide()
            self._trailing_connector = self._trailing_number = None
        self._rebuild()
        widget.show()
        return widget

    def trailing_widget(self) -> QWidget | None:
        return self._trailing

    def trailing_dot(self) -> "ProvenanceDot | None":
        return self._trailing_dot

    def trailing_number(self) -> QLabel | None:
        return self._trailing_number

    def trailing_connector_lit(self) -> bool | None:
        return None if self._trailing_connector is None else self._trailing_connector.lit

    # activity ------------------------------------------------------------
    def set_busy(self, key: str | None, clock: "ActivityClock | None" = None) -> None:
        """Mark the stage doing the work (its ring sweeps); ``None`` clears every mark."""
        for name, node in self._nodes.items():
            node.set_busy(name == key, clock)

    def busy_key(self) -> str | None:
        return next((name for name, node in self._nodes.items() if node.is_busy()), None)

    # internals -----------------------------------------------------------
    def _rebuild(self) -> None:
        while self._layout.count():
            self._layout.takeAt(0)
        nodes = list(self._nodes.values())
        for index, node in enumerate(nodes):
            if index:
                self._layout.addWidget(self._connectors[index - 1])
            self._layout.addWidget(node)
        if self._trailing is not None and self._trailing_connector is not None:
            self._layout.addWidget(self._trailing_connector)
            self._layout.addWidget(self._trailing_number, 0, Qt.AlignmentFlag.AlignVCenter)
            self._layout.addSpacing(6)
            self._layout.addWidget(self._trailing, 0, Qt.AlignmentFlag.AlignVCenter)
            self._trailing_connector.show()
            self._trailing_number.show()
            self._layout.addStretch(1)
            return
        self._layout.addStretch(1)
        if self._trailing is not None:
            self._layout.addSpacing(12)
            self._layout.addWidget(self._trailing, 0, Qt.AlignmentFlag.AlignVCenter)

    def _refresh_connectors(self) -> None:
        nodes = list(self._nodes.values())
        for index, connector in enumerate(self._connectors):
            lit = nodes[index].state() in ("done", "current") and nodes[index + 1].state() in ("done", "current")
            if lit != connector.lit:
                connector.lit = lit
                connector.update()


# --------------------------------------------------------------------------- frame dots

class FrameDots(_ThemedWidget):
    """One equal dot per frame; the colour reference is ringed; overflow shows ``+N``.

    Optional per-frame states: done (filled), skipped (hollow), todo (dim) — the same dots
    can double as stack progress.
    """

    DOT = 8
    PITCH = 14
    RING = 12

    def __init__(self, parent: QWidget | None = None, overflow_label: QLabel | None = None):
        super().__init__(parent)
        self._count = 0
        self._reference = 0
        self._max_dots = 60
        self._states: list[str] | None = None
        self._overflow_label = overflow_label
        self._shown = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(16)

    def set_overflow_label(self, label: QLabel | None) -> None:
        self._overflow_label = label
        self._relayout()

    def set_count(self, n: int, reference_index: int = 0, max_dots: int = 60) -> None:
        self._count = max(0, int(n))
        self._reference = int(reference_index) if reference_index is not None else -1
        self._max_dots = max(1, int(max_dots))
        if self._states is not None and len(self._states) != self._count:
            self._states = None
        self.updateGeometry()
        self._relayout()

    def set_states(self, states: Sequence[str] | None) -> None:
        if states is not None:
            states = list(states)
            bad = [s for s in states if s not in FRAME_STATES]
            if bad:
                raise ValueError(f"unknown frame state: {bad[0]!r}")
            if len(states) != self._count:
                raise ValueError("states must match the frame count")
        self._states = states
        self.update()

    def count(self) -> int:
        return self._count

    def reference_index(self) -> int:
        return self._reference

    def visible_count(self) -> int:
        return self._shown

    def overflow(self) -> int:
        return self._count - self._shown

    def overflow_text(self) -> str:
        return f"+{self.overflow()}" if self.overflow() > 0 else ""

    def _overflow_width(self) -> int:
        if self._overflow_label is not None or self._count <= self._max_dots:
            return 0
        return QFontMetrics(fonts.mono_font(11)).horizontalAdvance(f"+{self._count}") + 6

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        dots = min(self._count, self._max_dots)
        return QSize(max(self.PITCH, dots * self.PITCH + self._overflow_width()), 16)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self.PITCH, 16)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        cap = min(self._count, self._max_dots)
        if cap < self._count:
            room = self.width() - self._overflow_width()
        else:
            room = self.width()
        fits = max(1, (room + (self.PITCH - self.DOT)) // self.PITCH) if self.width() > 0 else cap
        self._shown = min(cap, fits)
        if self._shown < self._count and self._overflow_label is None:
            # Re-fit with the overflow text reserved.
            room = self.width() - QFontMetrics(fonts.mono_font(11)).horizontalAdvance(f"+{self._count}") - 6
            self._shown = min(cap, max(1, (room + (self.PITCH - self.DOT)) // self.PITCH)) if self.width() > 0 else cap
        if self._overflow_label is not None:
            self._overflow_label.setText(self.overflow_text())
            self._overflow_label.setVisible(self.overflow() > 0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        c = self.tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2
        for index in range(self._shown):
            cx = index * self.PITCH + self.RING / 2
            center = QPointF(cx, cy)
            state = self._states[index] if self._states is not None else "done"
            if index == self._reference:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(_pen(c["text"], 1.0))
                painter.drawEllipse(center, self.RING / 2 - 0.5, self.RING / 2 - 0.5)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(c["text"]))
                painter.drawEllipse(center, 2.0, 2.0)
                continue
            if state == "skipped":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(_pen(c["border"], 1.0))
                painter.drawEllipse(center, self.DOT / 2 - 0.5, self.DOT / 2 - 0.5)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(c["text"] if state == "done" else c["connector"]))
                painter.drawEllipse(center, self.DOT / 2, self.DOT / 2)
        if self.overflow() > 0 and self._overflow_label is None:
            font = fonts.mono_font(11)
            painter.setFont(font)
            painter.setPen(QColor(c["muted"]))
            x = self._shown * self.PITCH + 2
            painter.drawText(QRectF(x, 0, self.width() - x, self.height()), Qt.AlignmentFlag.AlignVCenter,
                             self.overflow_text())


# --------------------------------------------------------------------------- tick slider

class TickSlider(QSlider):
    """A QSlider with graduated ticks and a longer centre tick for bipolar (-a..+a) ranges.

    Still a plain QSlider: signals, ranges, tracking and any debounce timers are unchanged.
    Double-click resets to ``default_value`` when one is set (emits valueChanged as usual).
    """

    TICK_SPACE = 10

    def __init__(self, orientation: Qt.Orientation | QWidget | None = Qt.Orientation.Horizontal,
                 parent: QWidget | None = None, *, intervals: int = 8, default_value: int | None = None):
        if isinstance(orientation, QWidget):
            parent, orientation = orientation, Qt.Orientation.Horizontal
        super().__init__(orientation or Qt.Orientation.Horizontal, parent)
        self._intervals = max(1, int(intervals))
        self._default = default_value
        # No wheel focus: turning the wheel over a slider scrolls the column instead.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_intervals(self, intervals: int) -> None:
        self._intervals = max(1, int(intervals))
        self.update()

    def intervals(self) -> int:
        return self._intervals

    def set_default_value(self, value: int | None) -> None:
        self._default = value

    def default_value(self) -> int | None:
        return self._default

    def is_bipolar(self) -> bool:
        return self.minimum() < 0 < self.maximum()

    def _extra(self) -> QSize:
        if self.orientation() == Qt.Orientation.Horizontal:
            return QSize(0, self.TICK_SPACE)
        return QSize(self.TICK_SPACE, 0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return super().sizeHint() + self._extra()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return super().minimumSizeHint() + self._extra()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        # A value the person set must never change while they scroll past it.
        event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._default is not None and self.isEnabled():
            self.setValue(int(self._default))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt override
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self.update()
        super().changeEvent(event)

    def tick_positions(self) -> list[tuple[float, int, bool]]:
        """(x, value, is_centre) for each tick, in widget coordinates (horizontal only)."""
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
        span = max(1, groove.width() - handle.width())
        lo, hi = self.minimum(), self.maximum()

        def x_of(value: int) -> float:
            offset = QStyle.sliderPositionFromValue(lo, hi, value, span, option.upsideDown)
            return groove.x() + handle.width() / 2 + offset

        ticks: list[tuple[float, int, bool]] = []
        if hi <= lo:
            return ticks
        for index in range(self._intervals + 1):
            value = round(lo + (hi - lo) * index / self._intervals)
            if self.is_bipolar() and value == 0:
                continue
            ticks.append((x_of(value), value, False))
        if self.is_bipolar():
            ticks.append((x_of(0), 0, True))
        return ticks

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self.orientation() != Qt.Orientation.Horizontal:
            return
        c = themes.colors_for(self)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                             QStyle.SubControl.SC_SliderHandle, self)
        top = handle.bottom() + 3
        bottom = self.height() - 1
        if bottom - top < 2:
            top = max(0, bottom - 4)
        painter = QPainter(self)
        edge = c["disabled"] if not self.isEnabled() else c["border"]
        for index, (x, value, centre) in enumerate(self.tick_positions()):
            px = math.floor(x) + 0.5
            if centre:
                painter.setPen(_pen(c["disabled"] if not self.isEnabled() else c["text"], 1.0))
                length = bottom - top
            else:
                painter.setPen(_pen(edge, 1.0))
                length = 4 if index % 2 == 0 else 3
            painter.drawLine(QPointF(px, top), QPointF(px, min(bottom, top + length)))


# --------------------------------------------------------------------------- seal & provenance

class Seal(_ThemedWidget):
    """Cinnabar seal (e.g. 「人定」) marking a human confirmation; never for Agent actions."""

    SIZES = (24, 40)

    def __init__(self, text: str, accessible_name: str = "", size: int = 24, parent: QWidget | None = None):
        super().__init__(parent)
        self._text = text
        self._size = 24
        self.set_size(size)
        self.setAccessibleName(accessible_name or text)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self.update()

    def seal_size(self) -> int:
        return self._size

    def set_size(self, size: int) -> None:
        self._size = 40 if int(size) >= 32 else 24
        self.setFixedSize(self._size, self._size)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self._size, self._size)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        c = self.tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(c["zhu"]))
        radius = 3 if self._size >= 40 else 2
        painter.drawRoundedRect(QRectF(self.rect()), radius, radius)
        chars = [ch for ch in self._text if not ch.isspace()]
        if not chars:
            return
        painter.setPen(QColor(c["seal_text"]))
        inner = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if all(ord(ch) < 0x2E80 for ch in chars):
            # Latin captions do not stack vertically; fit them on one line.
            font = fonts.human_font(max(6, int(self._size * 0.3)))
            metrics = QFontMetrics(font)
            while font.pixelSize() > 6 and metrics.horizontalAdvance(self._text) > inner.width():
                font.setPixelSize(font.pixelSize() - 1)
                metrics = QFontMetrics(font)
            painter.setFont(font)
            painter.drawText(inner, Qt.AlignmentFlag.AlignCenter, self._text.strip())
            return
        columns = 1 if len(chars) <= 2 else math.ceil(len(chars) / 2)
        per_column = math.ceil(len(chars) / columns)
        pixel = max(6, int(min(self._size * 0.75 / per_column, self._size * 0.75 / columns)))
        painter.setFont(fonts.human_font(pixel))
        cell_h = pixel * 1.05
        cell_w = pixel * 1.05
        block_w = cell_w * columns
        block_h = cell_h * per_column
        left = (self.width() - block_w) / 2
        top = (self.height() - block_h) / 2
        for index, ch in enumerate(chars):
            column = index // per_column          # vertical-rl: first column on the right
            row = index % per_column
            x = left + (columns - 1 - column) * cell_w
            painter.drawText(QRectF(x, top + row * cell_h, cell_w, cell_h), Qt.AlignmentFlag.AlignCenter, ch)


class ProvenanceDot(_ThemedWidget):
    """Who decided: human (filled zhu), agent (filled xiang), algorithm (hollow ring)."""

    SIZE = 8

    def __init__(self, kind: str = "algorithm", parent: QWidget | None = None, accessible_name: str = ""):
        super().__init__(parent)
        self._kind = "algorithm"
        self.set_kind(kind)
        self.setFixedSize(self.SIZE, self.SIZE)
        if accessible_name:
            self.setAccessibleName(accessible_name)

    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str) -> None:
        if kind not in PROVENANCE_KINDS:
            raise ValueError(f"unknown provenance: {kind!r}")
        self._kind = kind
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self.SIZE, self.SIZE)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        c = self.tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        if self._kind == "algorithm":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(_pen(c["text"], 1.5))
            painter.drawEllipse(rect.adjusted(0.75, 0.75, -0.75, -0.75))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(c["zhu"] if self._kind == "human" else c["xiang"]))
            painter.drawEllipse(rect)


class ErrorIcon(_ThemedWidget):
    """Small zhu ring with '!' that accompanies (never replaces) error text."""

    def __init__(self, size: int = 16, parent: QWidget | None = None, accessible_name: str = ""):
        super().__init__(parent)
        self.setFixedSize(int(size), int(size))
        if accessible_name:
            self.setAccessibleName(accessible_name)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.size()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        c = self.tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        stroke = max(1.5, side / 10)
        color = QColor(c["zhu"])
        pen = QPen(color, stroke)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect.adjusted(stroke / 2, stroke / 2, -stroke / 2, -stroke / 2))
        cx = rect.center().x()
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx, rect.top() + side * 0.26), QPointF(cx, rect.top() + side * 0.56))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, rect.top() + side * 0.73), stroke * 0.6, stroke * 0.6)


# --------------------------------------------------------------------------- empty state

# Radial lines (degrees, weight) after the Suzhou star-map rubbing: 28 uneven lodges.
LODGE_LINES = (
    (270.0, 1.3), (281.9, .6), (290.7, .6), (305.5, .6), (310.5, .6), (315.4, .6), (333.1, .6),
    (344.0, 1.3), (9.6, .6), (17.5, .6), (29.4, .6), (39.2, .6), (56.0, .6), (71.7, .6),
    (80.7, 1.3), (96.4, .6), (108.3, .6), (122.1, .6), (132.9, .6), (148.7, .6), (150.6, .6),
    (159.5, 1.3), (192.1, .6), (196.0, .6), (210.8, .6), (217.8, .6), (235.5, .6), (253.3, .6),
)
# Milky Way double line as cubic Béziers in units of the outer radius, centre at origin.
MILKY_WAY_PATHS = (
    ((-90, 70), (-52, 40), (-22, 38), (6, 16), (34, -6), (58, -40), (98, -58)),
    ((-98, 56), (-58, 28), (-28, 24), (-2, 4), (24, -16), (48, -50), (94, -72)),
)
BAYER4 = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))
STAR_CELLS = 21
_star_cache: dict[tuple, QPixmap] = {}


def empty_state_colors(surround: str | None = None, *, night: bool = False) -> dict[str, str]:
    """Line/star greys that read on a photo surround (xuan #1f1f1f, mid grey, black)."""
    if night:
        # While processing, the sweep head is capped at the border red over a hairline floor.
        return {"empty_line": themes.NIGHT["empty_line"], "empty_star": themes.NIGHT["empty_star"],
                "sweep_floor": themes.NIGHT["hairline"], "sweep_head": themes.NIGHT["border"]}
    value = themes.SURROUNDS.get(surround or "", surround or themes.SURROUNDS[themes.DEFAULT_SURROUND])
    grey = int(str(value)[1:3], 16) if str(value).startswith("#") and len(str(value)) == 7 else 0x1f
    if grey >= 0x60:
        return {"empty_line": "#4a4a4a", "empty_star": "#f2f2f0", "sweep_floor": "#4a4a4a", "sweep_head": "#101010"}
    return {"empty_line": "#6b6b6b", "empty_star": "#e6e6e3", "sweep_floor": "#6b6b6b", "sweep_head": "#e6e6e3"}


def _star_mask() -> list[list[bool]]:
    centre = STAR_CELLS // 2
    rows = []
    for j in range(STAR_CELLS):
        row = []
        for i in range(STAR_CELLS):
            dx, dy = i - centre, j - centre
            r2 = dx * dx + dy * dy
            core = math.exp(-r2 / 4.5)
            spikes = math.exp(-abs(dy) / 0.55) * math.exp(-abs(dx) / 4.5) + \
                math.exp(-abs(dx) / 0.55) * math.exp(-abs(dy) / 4.5)
            halo = 0.28 * math.exp(-r2 / 30.0)
            value = min(1.0, core + 0.85 * spikes + halo)
            row.append(value > (BAYER4[j % 4][i % 4] + 0.5) / 16.0)
        rows.append(row)
    return rows


_STAR_MASK = _star_mask()


def dither_star_pixmap(size: float, color: str, dpr: float) -> QPixmap:
    """Ordered-dither (Bayer 4×4) star, blocks snapped to whole device pixels and cached."""
    cell = max(1, round(size / STAR_CELLS * dpr))
    key = (cell, QColor(color).rgba(), round(dpr, 3))
    cached = _star_cache.get(key)
    if cached is not None:
        return cached
    side = cell * STAR_CELLS
    image = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    fill = QColor(color)
    for j, row in enumerate(_STAR_MASK):
        for i, on in enumerate(row):
            if on:
                painter.fillRect(QRect(i * cell, j * cell, cell, cell), fill)
    painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(dpr)
    if len(_star_cache) > 32:
        _star_cache.clear()
    _star_cache[key] = pixmap
    return pixmap


def empty_state_rect(rect: QRect | QRectF) -> QRectF:
    """Where paint_empty_state draws its motif inside ``rect`` (for partial repaints)."""
    area = QRectF(rect)
    side = max(24.0, min(area.width() * 0.5, area.height() * 0.7, 220.0))
    motif = QRectF(0, 0, side, side)
    motif.moveCenter(area.center())
    return motif


def paint_empty_state(painter: QPainter, rect: QRect | QRectF, colors: Mapping[str, str] | None = None,
                      *, sweep: int | None = None) -> QRectF:
    """Rubbing linework (concentric circles, 28 uneven radial lines, double Milky Way) and a
    dithered star, grey only; for the photo area when there is no photo.  Returns the motif
    rectangle so the caller can place its (translated) message and action beside it.

    ``sweep`` (a head step 0..27) marks processing: the radial lodge lines brighten in turn
    from ``sweep_floor`` to ``sweep_head`` (fades only); ``None`` draws the still motif.
    """
    colors = colors or themes.COLORS[themes.DEFAULT_THEME]
    line = colors.get("empty_line") or colors.get("border", "#6b6b6b")
    star = colors.get("empty_star") or colors.get("text", "#e6e6e3")
    motif = empty_state_rect(rect)
    side = motif.width()
    radius = side / 2
    centre = motif.center()
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for fraction, width in ((24 / 74, 0.8), (50 / 74, 1.0), (1.0, 0.8)):
        painter.setPen(_pen(line, width))
        painter.drawEllipse(centre, radius * fraction, radius * fraction)
    inner = radius * 24 / 74
    if sweep is not None:
        floor = colors.get("sweep_floor") or line
        head = colors.get("sweep_head") or star
        colours = [_mix(floor, head, sweep_level(index, sweep)) for index in range(len(LODGE_LINES))]
    else:
        colours = [QColor(line)] * len(LODGE_LINES)
    for (angle, width), colour in zip(LODGE_LINES, colours):
        theta = math.radians(angle)
        direction = QPointF(math.cos(theta), math.sin(theta))
        painter.setPen(_pen(colour, width))
        painter.drawLine(centre + direction * inner, centre + direction * radius)
    scale = radius / 74
    painter.setPen(_pen(line, 0.8))
    painter.setClipRect(motif.adjusted(-2, -2, 2, 2))
    for points in MILKY_WAY_PATHS:
        pts = [centre + QPointF(x * scale, y * scale) for x, y in points]
        path = QPainterPath(pts[0])
        path.cubicTo(pts[1], pts[2], pts[3])
        path.cubicTo(pts[4], pts[5], pts[6])
        painter.drawPath(path)
    painter.setClipping(False)
    device = painter.device()
    dpr = float(device.devicePixelRatioF()) if device is not None else 1.0
    pixmap = dither_star_pixmap(side * 0.34, star, dpr)
    logical = pixmap.width() / dpr
    target = QPointF(round((centre.x() - logical / 2) * dpr) / dpr, round((centre.y() - logical / 2) * dpr) / dpr)
    painter.drawPixmap(target, pixmap)
    painter.restore()
    return motif
