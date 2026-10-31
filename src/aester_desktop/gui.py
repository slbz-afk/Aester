# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Small desktop client for the shared StarStack background-job API."""

from __future__ import annotations

import os
import math
import sys
import threading
from pathlib import Path
from starstack.i18n import tr, set_language, get_language, normalize_language, startup_language, translate_message
from typing import Any, Callable

# Initialize numerical DLLs on the main thread before QRunnable callbacks import them.
from astropy.io import fits as _fits

from PySide6.QtCore import (QObject, QRect, QRectF, QRunnable, QSettings, QSignalBlocker, QSize, QStandardPaths, Qt,
                            QThreadPool, QTimer, QUrl, Signal, Slot)
from PySide6.QtGui import (QAction, QActionGroup, QColor, QDesktopServices, QFontDatabase, QIcon, QImage, QImageReader,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMenu, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QAbstractItemView, QListWidget, QListWidgetItem, QTabBar, QTextBrowser, QVBoxLayout, QWidget,
)

from starstack.core import RAW, cancel_job, create_demo, doctor, load_job, resume_job, submit_job
from starstack import jpeg_estimate
from . import fonts, themes, ui_widgets


WORKSPACE = Path(__file__).resolve().parents[2]
ACTIVE_STATES = {"queued", "running", "cancel_requested"}
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
STATE_LABELS = {
    "queued": "等待处理", "running": "正在处理", "succeeded": "处理完成",
    "failed": "处理未完成", "cancel_requested": "正在取消", "cancelled": "已取消",
}
STAGE_LABELS = {
    "queued": "准备任务", "inspect": "检查素材", "scan": "检查素材",
    "prepare": "准备素材", "convert": "读取图像", "decode": "读取图像",
    "register": "对齐星空", "registration": "对齐星空", "align": "对齐星空",
    "stack": "堆栈天空", "stacking": "堆栈天空", "foreground": "处理地景",
    "composite": "合成天空与地景", "blend": "合成天空与地景",
    "render": "生成成片", "styles": "生成三种风格", "export": "保存结果",
    "done": "已完成", "complete": "已完成",
}

# Kept for integrations which used the original dark stylesheet.
STYLESHEET = themes.stylesheet(themes.DEFAULT_THEME)


class _WorkerSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)


class _Worker(QRunnable):
    def __init__(self, name: str, operation: Callable[[], Any]):
        super().__init__()
        self.name = name
        self.operation = operation
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self.name, self.operation())
        except Exception as exc:
            self.signals.failed.emit(self.name, str(exc) or type(exc).__name__)


# Preference values of ``appearance/surround`` (stored) and the theme module's surround keys.
SURROUND_PREFERENCES = {"xuan": "xuan", "grey": "mid_grey", "black": "black"}
_qt_translator_language: str | None = None


def surround_key(value: object) -> str:
    """themes.SURROUNDS key for a stored preference ('grey' and the module's 'mid_grey' both accepted)."""
    text = str(value or "")
    return themes.normalize_surround(SURROUND_PREFERENCES.get(text, text))


def surround_preference(key: str) -> str:
    return next((stored for stored, name in SURROUND_PREFERENCES.items() if name == key), "xuan")


def _preference_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() in ("1", "true", "yes", "on")


def _install_qt_translations(language: str) -> None:
    """Qt's own zh_CN strings (context menus, standard buttons) once per process and language."""
    global _qt_translator_language
    app = QApplication.instance()
    if app is None or _qt_translator_language == language:
        return
    fonts.install_qt_translator(app, language)
    _qt_translator_language = language


class _ReadoutMessage(QLabel):
    """The status line: full text always in text(); an error only changes colour and the icon beside it."""

    errorChanged = Signal(bool)

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._error = False
        self._pending_error = False

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override (Python callers)
        super().setText(text)
        error, self._pending_error = self._pending_error, False
        if error != self._error:
            self._error = error
            themes.set_role(self, "error" if error else None)
            self.errorChanged.emit(error)

    def set_error_text(self, text: str) -> None:
        self._pending_error = True
        self.setText(text)

    def is_error(self) -> bool:
        return self._error


class _Hairline(QWidget):
    """A 1 px divider painted in the current theme's hairline token (vertical or horizontal)."""

    def __init__(self, vertical: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self._vertical = vertical
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if vertical:
            self.setFixedWidth(1)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else:
            self.setFixedHeight(1)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(themes.colors_for(self)["hairline"]))


class _ElidedChip(QLabel):
    """The recipe tile: shrinks before any view button is squeezed.

    ``text()`` keeps the full text (also its accessible name).  When space runs out the
    leading title ("相机观感 / Camera look") is dropped first, so the body, the borrowed table
    and the strength stay readable; only then is the rest elided on the right.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override (Python callers)
        self._full = text
        self.setAccessibleName(text)
        self._elide()
        self.updateGeometry()

    def text(self) -> str:  # noqa: D401 - Qt override (Python callers)
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 72), hint.height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        margins = self.contentsMargins()
        width = self.fontMetrics().horizontalAdvance(self._full) + margins.left() + margins.right() + 4
        return QSize(width, super().sizeHint().height())

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()

    def shown_text(self, width: int | None = None) -> str:
        """What fits in ``width`` (default: the current contents width)."""
        width = max(0, self.contentsRect().width()) if width is None else max(0, int(width))
        if not width:
            return self._full
        metrics = self.fontMetrics()
        if metrics.horizontalAdvance(self._full) <= width:
            return self._full
        head, separator, rest = self._full.partition(" · ")
        candidate = rest if separator and rest else self._full
        if metrics.horizontalAdvance(candidate) <= width:
            return candidate
        return metrics.elidedText(candidate, Qt.TextElideMode.ElideRight, width)

    def _elide(self) -> None:
        shown = self.shown_text()
        if QLabel.text(self) != shown:
            QLabel.setText(self, shown)


class ImagePreview(QWidget):
    """Fit an actual decoded image without stretching or cropping it.

    The surround is a viewing condition (玄 / mid grey / black), independent of the interface
    theme.  Night vision only changes what is drawn here (red single channel, a permanent
    "not for colour judgement" tag); the stored pixmap and every export stay untouched.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("imagePreview")
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap()
        self._message = tr('从一组星空开始\n\n添加或拖入照片，首帧将自动预览。')
        self._art = True
        self._surround = themes.DEFAULT_SURROUND
        self._night = False
        self._night_cache: tuple[int | None, QPixmap] = (None, QPixmap())
        self._activity: ui_widgets.ActivityClock | None = None
        self._motif = QRectF()
        self.night_tag = QLabel(tr('不可判色'), self)
        self.night_tag.setObjectName("nightTag")
        self.night_tag.setToolTip(tr('夜视只改变显示：画面以红色单通道显示，不能用于判断颜色；导出不受影响。'))
        self.night_tag.setAccessibleName(tr('夜视模式：不可判色'))
        themes.set_role(self.night_tag, "badge")
        self.night_tag.hide()

    def show_message(self, message: str, *, art: bool = False) -> None:
        self._pixmap = QPixmap()
        self._message = message
        self._art = art
        self.update()

    def set_image(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_activity(self, clock: ui_widgets.ActivityClock | None) -> None:
        """While processing without a photograph, the empty-state lodge lines sweep."""
        if clock is self._activity:
            return
        if self._activity is not None:
            self._activity.stepChanged.disconnect(self._activity_step)
        self._activity = clock
        if clock is not None:
            clock.stepChanged.connect(self._activity_step)
        self.update()

    def is_sweeping(self) -> bool:
        """The empty-state motif is drawn with the moving sweep right now."""
        return (self._pixmap.isNull() and self._art and self._activity is not None
                and self._activity.is_running())

    def _activity_step(self, _step: int = 0) -> None:
        if self._pixmap.isNull() and self._art and self.isVisible():
            # Only the motif repaints (about 12 times a second); never the whole canvas.
            self.update(self._motif.toAlignedRect().adjusted(-2, -2, 2, 2) if not self._motif.isEmpty() else self.rect())

    def set_surround(self, name: str) -> None:
        self._surround = themes.normalize_surround(name)
        self.update()

    def surround(self) -> str:
        return self._surround

    def set_night(self, night: bool) -> None:
        self._night = bool(night)
        self.night_tag.setVisible(self._night)
        self._place_tag()
        self.update()

    def is_night(self) -> bool:
        return self._night

    def background_color(self) -> QColor:
        return QColor(themes.NIGHT["canvas"] if self._night else themes.SURROUNDS[self._surround])

    def display_pixmap(self) -> QPixmap:
        """The pixmap drawn now: the stored one, or its red single-channel copy in night vision."""
        if not self._night or self._pixmap.isNull():
            return self._pixmap
        key = self._pixmap.cacheKey()
        if self._night_cache[0] != key:
            grey = self._pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
            red = grey.convertToFormat(QImage.Format.Format_RGB32)
            painter = QPainter(red)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
            painter.fillRect(red.rect(), QColor(255, 0, 0))
            painter.end()
            self._night_cache = (key, QPixmap.fromImage(red))
        return self._night_cache[1]

    def _place_tag(self) -> None:
        self.night_tag.adjustSize()
        self.night_tag.move(12, 12)
        self.night_tag.raise_()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._place_tag()

    def _message_color(self) -> QColor:
        if self._night:
            return QColor(themes.NIGHT["text"])
        # Only near-black reaches 4.5:1 on the #777777 evaluation grey (4.69:1).
        return QColor("#000000" if self._surround == "mid_grey" else "#a3a3a3")

    def content_rect(self) -> QRect:
        """Where messages and the photograph go; in night vision it starts below the tag."""
        area = self.rect().adjusted(18, 18, -18, -18)
        if self._night:
            area.setTop(max(area.top(), self.night_tag.geometry().bottom() + 9))
        return area

    def photo_rect(self) -> QRect:
        """The rectangle the photograph is drawn into (empty without a photograph)."""
        if self._pixmap.isNull():
            return QRect()
        area = self.content_rect()
        target = QRect(area)
        target.setSize(self._pixmap.size().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio))
        target.moveCenter(area.center())
        return target

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.background_color())
        area = self.content_rect()
        if self._pixmap.isNull():
            text_area = QRectF(area)
            if self._art and area.height() > 120:
                colours = ui_widgets.empty_state_colors(self._surround, night=self._night)
                sweep = self._activity.step() if self.is_sweeping() else None
                motif = ui_widgets.paint_empty_state(
                    painter, QRectF(area.x(), area.y(), area.width(), area.height() * 0.64), colours, sweep=sweep)
                self._motif = motif
                text_area = QRectF(area.x(), motif.bottom() + 12, area.width(), max(0.0, area.bottom() - motif.bottom() - 12))
            painter.setPen(self._message_color())
            painter.drawText(text_area, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                                            if self._art and text_area != QRectF(area) else Qt.AlignmentFlag.AlignCenter)
                             | int(Qt.TextFlag.TextWordWrap), self._message)
            return
        pixmap = self.display_pixmap()
        target = self.photo_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, pixmap)
        if self._surround == "mid_grey" and not self._night:
            # Evaluation surround: a thin white frame separates the photograph from the grey.
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.drawRect(target.adjusted(-2, -2, 1, 1))


class MainWindow(QMainWindow):
    def __init__(self, *, preferences: QSettings | None = None, persist_preferences: bool = True):
        super().__init__()
        self._preferences = (preferences or QSettings("StarStack", "StarStack")) if persist_preferences else None
        set_language(startup_language(self._preferences))
        self._startup_language = get_language()
        self._selected_language = self._startup_language
        # Bundled OFL fonts (「拓星」 type roles) and Qt's own zh_CN strings, before any widget exists.
        fonts.register_bundled_fonts()
        _install_qt_translations(self._startup_language)
        # Qt's offscreen Windows platform can omit all system font discovery.
        # Load the installed CJK font in that case so previews remain readable.
        if os.name == "nt" and not QFontDatabase.families():
            font_file = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"
            if font_file.is_file():
                QFontDatabase.addApplicationFont(str(font_file))
        self.setWindowTitle(tr('Aester · 星野摄影'))
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "aester.png")))
        self.resize(1200, 800)
        self.setMinimumSize(960, 680)
        self._theme_name = themes.DEFAULT_THEME
        # Night vision and the photo surround are viewing conditions, not themes; they are
        # read from preferences in _restore_preferences and written only on change or close.
        self._night = False
        self._surround = themes.DEFAULT_SURROUND
        # The 中灰 toggle returns to the surround chosen before it (玄 or 纯黑).
        self._surround_before_grey = themes.DEFAULT_SURROUND
        themes.apply_theme(self, self._theme_name, night=self._night)
        # True once the person changes the photo selection after a job: the stage bar then
        # follows the new selection instead of the previous job's state.
        self._job_detached = False
        self._reference_failed = False
        self.job_dir: str | None = None
        self.workflow_dir: str | None = None
        self._workflow: dict[str, Any] | None = None
        self._advanced_result_dir: str | None = None
        self.job_state = ""
        self._job: dict[str, Any] | None = None          # the last job record read (product contract for the notes)
        self._busy = False
        self._closing_requested = False
        self._doctor_serial = 0
        self._preview_serial = 0
        self._artifact_signature: tuple[Any, ...] = ()
        self._jpg_artifacts: list[dict[str, Any]] = []
        self._reference_data: dict[str, Any] | None = None
        # None requests the first exposure's recorded orientation. Once decoded,
        # the same clockwise quarter-turn is applied to the entire selection.
        self._image_rotation: int | None = None
        self._selected_reference_source = ""
        self._orientation_notice = False
        self._regions_project: dict[str, Any] | None = None
        self._regions_path: str | None = None
        self._regions_saved_by_user: dict[str, Any] | None = None  # identity of the last editor save
        self.input_files: list[str] = []
        self._import_originals: dict[str, str] = {}
        self._import_cancel = None
        self._finite_backend_info = {}
        self._reference_generation = 0
        self._reference_cache_epoch = 0
        self._reference_running = False
        self._reference_pending = False
        self._reference_scene: tuple[str, ...] = ()
        self._restoring_project = False
        self._rendered_preview: dict[str, Any] | None = None
        self._raw_audit_dir: str | None = None
        self._star_result: dict[str, Any] | None = None
        self._star_suggestion: dict[str, Any] | None = None
        self._star_suggestion_inputs: tuple[str, str, str] | None = None
        self._view_mode = "results"
        self._path_buttons: list[QPushButton] = []
        self._workers: dict[str, tuple[_Worker, Callable[[Any], None]]] = {}
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(3)
        self._color_timer = QTimer(self)
        self._color_timer.setSingleShot(True)
        self._color_timer.setInterval(90)
        self._color_timer.timeout.connect(self._launch_reference_preview)
        self._reference_quality = 'standard'
        self._color_finish_timer = QTimer(self)
        self._color_finish_timer.setSingleShot(True)
        self._color_finish_timer.setInterval(600)
        self._color_finish_timer.timeout.connect(self._finish_color_preview)
        # JPEG quality estimate: the first-frame preview encoded at the chosen quality (debounced; jpeg_estimate).
        self._jpeg_basis: dict[str, Any] | None = None
        self._jpeg_source_key: Any = None
        self._jpeg_restoring = False
        self._jpeg_unsynced = False
        # The person's preference (QSettings output/jpeg_quality) and, while a loaded workflow is shown, the value
        # frozen in it ({"quality": int, or None for a workflow submitted without one}); the slider then shows the
        # frozen value without writing it as the preference.
        self._jpeg_preference = jpeg_estimate.DEFAULT
        self._jpeg_job: dict[str, Any] | None = None
        self._jpeg_estimate_timer = QTimer(self)
        self._jpeg_estimate_timer.setSingleShot(True)
        self._jpeg_estimate_timer.setInterval(150)
        self._jpeg_estimate_timer.timeout.connect(self._update_jpeg_estimate)
        self._build_ui()
        self._restore_preferences()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh_job)
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(50)
        self._close_timer.timeout.connect(self._close_after_workers)
        self.siril_path.editingFinished.connect(self._siril_path_edited)
        self.input_dir.textChanged.connect(self._update_controls)
        self.output_dir.textChanged.connect(self._update_controls)
        self.sky_mask.textChanged.connect(self._update_mask_notice)
        for slider in (self.warmth_slider, self.tint_slider, self.exposure_slider):
            slider.valueChanged.connect(self._color_changed)
        self.setAcceptDrops(True)
        for field in (self.star_separation, self.star_milkyway_mask, self.star_milkyway_evidence):
            field.textChanged.connect(self._star_inputs_changed)
        self.star_background_strength.valueChanged.connect(self._update_star_strength_note)
        self.star_milkyway_offset.valueChanged.connect(self._update_star_strength_note)
        self._update_star_strength_note()
        self._update_controls()
        QTimer.singleShot(0, self.check_backend)
        from .experience import Experience
        self.experience = Experience(self)
        self._update_controls()

    def _siril_path_edited(self) -> None:
        """Only a real edit of the Siril path re-checks the backend and invalidates the reference.

        Pressing Enter in the field (or leaving it) without changing it keeps the regions.
        """
        if not self.siril_path.isModified():
            return
        self.siril_path.setModified(False)
        self.check_backend()
        self._invalidate_reference_backend()

    def _processing_root(self) -> str:
        return self.experience.work_root()

    @staticmethod
    def _label(text: str, name: str = "muted") -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        return label

    def _slider_control(self, layout: QVBoxLayout, title: str, name: str, minimum: int,
                        maximum: int, value: int, low: str, high: str, scale: int = 1) -> QSlider:
        """A colour slider with a typed read-out: drag, arrow keys or type a value, never the wheel.

        The slider holds the value (its valueChanged drives the preview); the field mirrors it
        and writes back only on Enter or focus-out, so each commit changes the colour once.
        """
        row = QHBoxLayout()
        row.setSpacing(4)
        label = self._label(f"{title} · {low} / {high}", "field")
        row.addWidget(label, 1)
        delta = QLabel("Δ")
        delta.setObjectName(name + "Delta")
        themes.set_role(delta, "mono")
        delta.setAccessibleName(tr('已调整'))
        retain = delta.sizePolicy()
        retain.setRetainSizeWhenHidden(True)
        delta.setSizePolicy(retain)
        row.addWidget(delta, 0, Qt.AlignmentFlag.AlignVCenter)
        field = ui_widgets.ValueField()
        field.setObjectName(name + "Value")
        field.setAccessibleName(title)
        field.setDecimals(1 if scale == 10 else 0)
        field.setSingleStep(1 / scale)
        field.setRange(minimum / scale, maximum / scale)
        field.setSuffix(" EV" if scale == 10 else "")
        field.setFixedWidth(field.fit_width())
        field.setToolTip(tr('输入数值后按 Enter 应用；滚轮不会改变数值。'))
        label.setBuddy(field)
        row.addWidget(field, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(row)
        slider = ui_widgets.TickSlider(Qt.Orientation.Horizontal, default_value=0)
        slider.setObjectName(name)
        slider.setAccessibleName(title)
        slider.setRange(minimum, maximum)
        slider.setValue(value)

        def from_slider(v: int) -> None:
            with QSignalBlocker(field):
                field.setValue(v / scale)
            # A changed value shows Δ and is set in weight 500, so the read-out shows what moved.
            delta.setVisible(bool(v))
            if bool(field.property("changed")) != bool(v):
                field.setProperty("changed", bool(v))
                for widget in (field, field.lineEdit()):
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
                field.update()

        def from_field(v: float) -> None:
            slider.setValue(round(v * scale))    # emits once (drives the preview) only if it changed

        from_slider(value)
        slider.valueChanged.connect(from_slider)
        field.valueChanged.connect(from_field)
        layout.addWidget(slider)
        slider.setToolTip(f"{low} ← {title} → {high}\n" + tr('双击滑块恢复为 0'))
        slider.value_field = field
        slider.delta_label = delta
        return slider

    # ------------------------------------------------------------------ JPEG quality (fixed bottom of the panel)

    def _build_jpeg_quality(self, layout: QVBoxLayout) -> None:
        """「JPEG 质量」: the unsigned slider variant (60–100, default 95) with a typed field and a size estimate.

        Drag, arrow keys (1) or Page Up/Down (5), or type a value and press Enter; never the wheel. A double-click
        on the slider returns to 95. The value is a global output preference (QSettings ``output/jpeg_quality``),
        sent with every job and with the GUI's other JPEG writers.
        """
        title = tr('JPEG 质量')
        row = QHBoxLayout()
        row.setSpacing(6)
        label = self._label(title, "field")
        label.setObjectName("jpegQualityLabel")
        label.setWordWrap(False)
        label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        slider = ui_widgets.TickSlider(Qt.Orientation.Horizontal, default_value=jpeg_estimate.DEFAULT)
        slider.setObjectName("jpegQuality")
        slider.setAccessibleName(title)
        slider.setRange(jpeg_estimate.MINIMUM, jpeg_estimate.MAXIMUM)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setValue(jpeg_estimate.DEFAULT)
        slider.setToolTip(tr('60–100：数值越高，JPEG 越清晰、文件越大；16 位 TIFF 和母版不受影响。'
                             '开始合成时随任务保存，之后的亮迹版本沿用。双击滑块恢复为 95。'))
        row.addWidget(slider, 1, Qt.AlignmentFlag.AlignVCenter)
        field = ui_widgets.UnsignedValueField()
        field.setObjectName("jpegQualityValue")
        field.setAccessibleName(title)
        field.setDecimals(0)
        field.setSingleStep(1)
        field.setRange(jpeg_estimate.MINIMUM, jpeg_estimate.MAXIMUM)
        field.setValue(jpeg_estimate.DEFAULT)
        field.setFixedWidth(field.fit_width())
        field.setToolTip(tr('输入数值后按 Enter 应用；滚轮不会改变数值。'))
        label.setBuddy(field)
        row.addWidget(field, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(row)
        estimate = self._label("", "muted")
        estimate.setObjectName("jpegEstimate")
        themes.set_role(estimate, "hint")
        estimate.setTextFormat(Qt.TextFormat.PlainText)
        estimate.hide()
        layout.addWidget(estimate)

        def from_slider(value: int) -> None:
            with QSignalBlocker(field):
                field.setValue(value)
            self._jpeg_quality_changed(value)

        slider.valueChanged.connect(from_slider)
        field.valueChanged.connect(lambda value: slider.setValue(round(value)))
        slider.value_field = field
        self.jpeg_quality_slider = slider
        self.jpeg_quality_value = field
        self.jpeg_estimate_label = estimate

    def jpeg_quality(self) -> int:
        """The JPEG quality every JPEG the GUI delivers is encoded at (60–100)."""
        return int(self.jpeg_quality_slider.value())

    def _set_jpeg_quality_quietly(self, value: int) -> None:
        """Show ``value`` without writing the preference (restore, or a loaded workflow's frozen value)."""
        self._jpeg_restoring = True
        try:
            self.jpeg_quality_slider.setValue(value)
        finally:
            self._jpeg_restoring = False
        self._schedule_jpeg_estimate()          # the shown job's note may change even when the value does not

    def _jpeg_quality_changed(self, value: int) -> None:
        if not self._jpeg_restoring:
            self._jpeg_preference = int(value)
            settings = getattr(self, "_preferences", None)
            if settings is not None:
                settings.setValue("output/jpeg_quality", int(value))
                self._jpeg_unsynced = True      # written to disk once the drag settles (the debounced estimate)
        self._schedule_jpeg_estimate()

    def _show_workflow_jpeg_quality(self, config: dict[str, Any]) -> None:
        """A loaded or submitted workflow: the slider shows the quality frozen in it (its later trails versions use
        it); a workflow without one shows the preference and a note. The preference itself is not written."""
        frozen = config.get("jpeg_quality")
        valid = type(frozen) is int and jpeg_estimate.MINIMUM <= frozen <= jpeg_estimate.MAXIMUM
        self._jpeg_job = {"quality": frozen if valid else None}
        self._set_jpeg_quality_quietly(frozen if valid else self._jpeg_preference)

    def _forget_workflow_jpeg_quality(self) -> None:
        """The view left the loaded workflow (a new selection): the slider returns to the preference."""
        if self._jpeg_job is not None:
            self._jpeg_job = None
            self._set_jpeg_quality_quietly(self._jpeg_preference)

    def _jpeg_job_note(self) -> str:
        """What the shown workflow keeps when the slider no longer says it (empty: nothing to add)."""
        job = self._jpeg_job
        if job is None:
            return ""
        if job["quality"] is None:
            return tr('当前任务未设定 JPEG 质量，之后的亮迹版本按 97 保存')
        if job["quality"] != self.jpeg_quality():
            return tr('当前任务仍按 {0} 保存', job["quality"])
        return ""

    def _schedule_jpeg_estimate(self) -> None:
        timer = getattr(self, "_jpeg_estimate_timer", None)
        if timer is not None and not self._closing_requested:
            timer.start()

    def _jpeg_estimate_source(self) -> tuple[str | None, int, int] | None:
        """(usable standard preview path or None, full width, full height) of the current reference, if any.

        The 640 px interactive preview is never used (it overstates the size); the last standard preview stays the
        basis while the colours are dragged.
        """
        reference = self._reference_data
        if not reference:
            return None
        try:
            width, height = int(reference["width"]), int(reference["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        path = str(reference.get("preview_path") or "")
        usable = bool(path) and reference.get("preview_quality") != "interactive" and Path(path).is_file()
        return (path if usable else None, width, height)

    def _track_jpeg_estimate_source(self) -> None:
        """Recompute the estimate once the reference preview or the selection it belongs to changes."""
        key = (self._jpeg_estimate_source(), tuple(self.input_files))
        if key != self._jpeg_source_key:
            self._jpeg_source_key = key
            self._schedule_jpeg_estimate()

    def _jpeg_estimate_bytes(self, quality: int) -> tuple[int, bool] | None:
        """(rough bytes per JPEG, from a preview) at ``quality``, or None when no frame size is known."""
        source = self._jpeg_estimate_source()
        files = tuple(self.input_files)
        basis = self._jpeg_basis
        if source is not None and source[0]:
            path, width, height = source
            if basis is None or basis["path"] != path:
                try:
                    # A FITS reference shows its full decoded frame: reduced to a standard preview's size first
                    # (the estimate scales by pixel count), so a slider step never encodes a full frame here.
                    image = jpeg_estimate.load_preview(path, max_edge=jpeg_estimate.PREVIEW_EDGE)
                except (OSError, ValueError):
                    image = None
                basis = {"path": path, "image": image, "sizes": {}}
            basis.update(width=width, height=height, files=files)
            self._jpeg_basis = basis
        elif basis is not None and basis["files"] == files and (
                source is None and files or source is not None and source[1] * source[2] == basis["width"] * basis["height"]):
            pass                    # the last standard preview of this selection (an interactive preview, or a run)
        elif source is not None:
            return jpeg_estimate.estimate_without_preview(source[1], source[2], quality), False
        else:
            return None
        if basis.get("image") is None:
            return jpeg_estimate.estimate_without_preview(basis["width"], basis["height"], quality), False
        sizes = basis["sizes"]
        if quality not in sizes:
            sizes[quality] = jpeg_estimate.estimate_from_preview(basis["image"], basis["width"], basis["height"], quality)
        return sizes[quality], True

    def _update_jpeg_estimate(self) -> None:
        label = getattr(self, "jpeg_estimate_label", None)
        if label is None:
            return
        settings = getattr(self, "_preferences", None)
        if settings is not None and self._jpeg_unsynced:
            self._jpeg_unsynced = False
            settings.sync()
        try:
            estimate = self._jpeg_estimate_bytes(self.jpeg_quality())
        except (OSError, ValueError):
            estimate = None
        note = self._jpeg_job_note()
        if estimate is None and not note:
            label.clear()
            label.setToolTip("")
            label.hide()
            return
        tooltip = tr('滑块用于新任务；已开始的任务沿用开始时保存的质量。') if note else ""
        if estimate is None:
            text = note
        else:
            size, from_preview = estimate
            text = tr('约 {0} MB／张（粗略估计）', jpeg_estimate.megabytes(size))
            if self.trails_enabled.isChecked():
                text += tr('；含亮迹版本另存一张，大小相近')
            if note:
                text = tr('{0}；{1}', text, note)
            hint = (tr('按首帧预览和所选质量估算；实际大小随画面细节与噪点变化，可能相差两成以上。') if from_preview
                    else tr('尚无首帧预览，按典型星野照片估算；实际大小随画面细节与噪点变化。'))
            tooltip = hint + ("\n" + tooltip if tooltip else "")
        label.setText(text)
        label.setToolTip(tooltip)
        label.show()

    def _step_heading(self, layout: QVBoxLayout, number: str, title: str) -> QHBoxLayout:
        """A step header in the right column: circled number (mono) and title; returns its row."""
        holder = QWidget()
        holder.setObjectName("stepHeading")
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 6, 0, 0)
        row.setSpacing(8)
        mark = QLabel(number)
        themes.set_role(mark, "mono")
        row.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
        heading = self._label(title, "section")
        heading.setWordWrap(False)
        row.addWidget(heading, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        layout.addWidget(holder)
        return row

    def color_adjustments(self) -> dict[str, float]:
        return {"warmth": float(self.warmth_slider.value()), "tint": float(self.tint_slider.value()),
                "exposure_ev": self.exposure_slider.value() / 10.0}

    def reset_color_adjustments(self) -> None:
        for slider in (self.warmth_slider, self.tint_slider, self.exposure_slider):
            slider.setValue(0)

    def _color_changed(self) -> None:
        if self._restoring_project:
            return
        self._reference_generation += 1
        self._reference_pending = True
        self._reference_quality = 'interactive'
        self._color_timer.start()
        self._color_finish_timer.start()
        if self.input_files:
            self.preview_note.setText(tr('正在更新首帧预览；相同调整将用于下次合成。'))

    def _finish_color_preview(self) -> None:
        if self._closing_requested or not self.input_files:
            return
        self._reference_quality = 'standard'
        self._reference_generation += 1
        self._reference_pending = True
        self._color_timer.start(1)

    def _invalidate_reference_backend(self) -> None:
        if self.input_files:
            self._regions_project = None
            self._regions_path = None
            self.milkyway_enabled.setChecked(False)
            self.sky_mask.clear()
            self._reference_cache_epoch += 1
            self._reference_data = None
            self._reference_scene = ()
            self._reference_generation += 1
            self._reference_pending = True
            self._color_timer.start()

    def _build_ui(self) -> None:
        self._advanced_dialogs: dict[str, QDialog] = {}
        self._extra_tools = None
        self._action_buttons: list[tuple[QAction, QPushButton]] = []
        root = QWidget()
        root.setObjectName("aesterRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Brand row: word mark, subtitle, and the local-processing note on the right.
        header_bar = QWidget()
        header_bar.setObjectName("brandRow")
        themes.set_role(header_bar, "toolbar")
        header = QHBoxLayout(header_bar)
        header.setContentsMargins(20, 10, 20, 4)
        header.setSpacing(12)
        brand = QLabel("AESTER")
        brand.setObjectName("brandMark")
        themes.set_role(brand, "brand")
        header.addWidget(brand)
        header.addWidget(self._label(tr('让星空自然呈现'), "subtitle"))
        header.addStretch(1)
        privacy = self._label(tr('本地处理 · 原片不改动'), "muted")
        privacy.setWordWrap(False)
        header.addWidget(privacy)
        layout.addWidget(header_bar)
        brand.ensurePolished()
        fonts.apply_tracking(brand, 0.2)

        # 星官 stage bar: progress and shortcuts only; it never switches pages.
        self.stage_bar = ui_widgets.StageBar()
        self.stage_bar.setObjectName("stageBar")
        self.stage_bar.set_stages([
            ("materials", "01", tr('素材'), tr('添加照片；也可以把照片拖进窗口。')),
            ("reference", "02", tr('首帧'), tr('查看首帧校色预览。')),
            ("regions", "03", tr('分区'), tr('标记天空、地景与银河。')),
            ("compose", "04", tr('合成'), tr('选择效果与保存位置，然后开始合成。')),
            ("results", "05", tr('成片'), tr('查看成片。')),
        ])
        self.stage_bar.stageClicked.connect(self._stage_clicked)
        self.retouch_button = QPushButton(tr('让 Agent 帮你修'))
        self.retouch_button.setObjectName('retouchAgentButton')
        self.retouch_button.setToolTip(tr('06 精修：复制修图提示词给 Agent；可使用已有成片，也可稍后选择照片。'))
        self.retouch_button.setAccessibleDescription(tr('06 精修'))
        self.retouch_button.clicked.connect(lambda: self.experience.show_retouch())
        self.stage_bar.add_trailing_widget(self.retouch_button, provenance="agent", number="06")
        self.stage_bar.trailing_dot().setAccessibleName(tr('Agent 参与'))
        layout.addWidget(self.stage_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        layout.addLayout(body, 1)

        # Left: the viewing area.  The preview, its tool rows and the file strip share one
        # parent so their geometries are directly comparable.
        viewing = QWidget()
        viewing.setObjectName("viewingArea")
        work = QVBoxLayout(viewing)
        work.setContentsMargins(20, 12, 16, 10)
        work.setSpacing(8)
        body.addWidget(viewing, 1)
        heading = QHBoxLayout()
        heading.setSpacing(8)
        self.selection_label = self._label(tr('添加一组连拍照片'), "section")
        self.selection_label.setWordWrap(False)
        heading.addWidget(self.selection_label)
        self.frame_overflow = QLabel("")
        self.frame_overflow.setObjectName("frameOverflow")
        themes.set_role(self.frame_overflow, "caption")
        self.frame_dots = ui_widgets.FrameDots(overflow_label=self.frame_overflow)
        self.frame_dots.setObjectName("frameDots")
        self.frame_dots.setToolTip(tr('每个点是一张照片；带环的是颜色参考（首帧）。'))
        self.frame_dots.setAccessibleName(tr('尚未添加照片'))
        self.frame_dots.setMinimumWidth(14)
        heading.addWidget(self.frame_dots, 1, Qt.AlignmentFlag.AlignVCenter)
        heading.addWidget(self.frame_overflow)
        self.add_files_button = QPushButton(tr('添加照片'))
        self.add_files_button.setObjectName("addPhotosButton")
        self.add_files_button.clicked.connect(self._pick_input)
        self.remove_files_button = QPushButton(tr('移除所选'))
        self.remove_files_button.setObjectName("removePhotosButton")
        self.remove_files_button.clicked.connect(self.remove_selected_files)
        self.clear_files_button = QPushButton(tr('清空素材'))
        self.clear_files_button.setObjectName("clearPhotosButton")
        self.clear_files_button.setToolTip(tr('从当前列表移除全部照片；磁盘上的原片不删除。'))
        self.clear_files_button.clicked.connect(self.clear_input_files)
        heading.addWidget(self.add_files_button)
        heading.addWidget(self.remove_files_button)
        heading.addWidget(self.clear_files_button)
        work.addLayout(heading)

        view_buttons = QHBoxLayout()
        view_buttons.setSpacing(8)
        self.view_switch = QWidget()
        self.view_switch.setObjectName("viewSwitch")
        themes.set_role(self.view_switch, "segmented")
        switch = QHBoxLayout(self.view_switch)
        switch.setContentsMargins(2, 2, 2, 2)
        switch.setSpacing(2)
        self.results_view_button = QPushButton(tr('成片'))
        self.results_view_button.setObjectName("resultsViewButton")
        self.results_view_button.setCheckable(True)
        self.results_view_button.clicked.connect(self.show_results)
        self.reference_view_button = QPushButton(tr('首帧预览'))
        self.reference_view_button.setObjectName("referenceViewButton")
        self.reference_view_button.setCheckable(True)
        self.reference_view_button.clicked.connect(self.show_reference)
        self.stars_view_button = QPushButton(tr('星点整理'))
        self.stars_view_button.setObjectName("starsViewButton")
        self.stars_view_button.setCheckable(True)
        self.stars_view_button.clicked.connect(self.show_star_result)
        for button in (self.reference_view_button, self.results_view_button, self.stars_view_button):
            themes.set_role(button, "segment")
            button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            switch.addWidget(button)
        view_buttons.addWidget(self.view_switch)
        # Regenerates the colour-reference preview when a run has cleared it (same semantics
        # as the former hidden button; also reachable from stage 02).
        self.reference_button = QPushButton(tr('刷新首帧预览'))
        self.reference_button.setObjectName("prepareReferenceButton")
        self.reference_button.setToolTip(tr('用当前照片与颜色调整重新生成首帧校色预览；原片不改动。'))
        themes.set_role(self.reference_button, "ghost")
        self.reference_button.clicked.connect(self.prepare_reference)
        self.reference_button.hide()
        view_buttons.addWidget(self.reference_button)
        self.style_tabs = QTabBar()
        self.style_tabs.setObjectName("styleTabs")
        self.style_tabs.setExpanding(False)
        self.style_tabs.setDrawBase(False)
        self.style_tabs.currentChanged.connect(self.select_style)
        view_buttons.addWidget(self.style_tabs)
        view_buttons.addStretch(1)
        self.recipe_label = _ElidedChip()
        self.recipe_label.setObjectName("cameraLookTile")
        themes.set_role(self.recipe_label, "chip")
        self.recipe_label.hide()
        view_buttons.addWidget(self.recipe_label)
        self.open_button = QPushButton(tr('打开作品'))
        self.open_button.setObjectName("openResultsButton")
        self.open_button.clicked.connect(self.open_results)
        view_buttons.addWidget(self.open_button)
        work.addLayout(view_buttons)

        self.preview = ImagePreview()
        work.addWidget(self.preview, 1)
        self.preview_note = self._label(tr('拖入 RAW 或 FITS；首张照片会自动生成校色预览。'))
        self.preview_note.setObjectName("previewNote")
        themes.set_role(self.preview_note, "hint")
        self.preview_note.setMaximumHeight(36)
        work.addWidget(self.preview_note)
        orientation_row = QHBoxLayout()
        orientation_row.setSpacing(6)
        self.rotate_left_button = QPushButton(tr('左转 90°'))
        self.rotate_left_button.setObjectName('rotateAllLeft')
        self.rotate_left_button.clicked.connect(lambda: self.rotate_input_images(-90))
        self.rotate_right_button = QPushButton(tr('右转 90°'))
        self.rotate_right_button.setObjectName('rotateAllRight')
        self.rotate_right_button.clicked.connect(lambda: self.rotate_input_images(90))
        self.restore_orientation_button = QPushButton(tr('恢复拍摄方向'))
        self.restore_orientation_button.setObjectName('restoreCaptureOrientation')
        self.restore_orientation_button.clicked.connect(self.restore_capture_orientation)
        for button in (self.rotate_left_button, self.rotate_right_button, self.restore_orientation_button):
            button.setToolTip(tr('同时调整本组素材、标记画面与最终成片的方向；原片不改动。已有分区需要重新绘制。'))
            orientation_row.addWidget(button)
        orientation_row.addStretch(1)
        # Viewing conditions (display only; never written into photographs or exports).
        self.surround_grey_button = QPushButton(tr('中灰'))
        self.surround_grey_button.setObjectName("surroundGreyToggle")
        self.surround_grey_button.setCheckable(True)
        self.surround_grey_button.setAccessibleName(tr('中灰评估'))
        self.surround_grey_button.setToolTip(tr('画面四周换成中灰，导出前评估亮度与对比；只改显示，不改照片。'))
        self.surround_grey_button.toggled.connect(self._surround_toggled)
        self.night_button = QPushButton(tr('夜视'))
        self.night_button.setObjectName("nightToggle")
        self.night_button.setCheckable(True)
        self.night_button.setAccessibleName(tr('夜视模式'))
        self.night_button.setToolTip(tr('夜视：界面改为纯红，画面以红色单通道显示，不可判色；不写入导出。快捷键 Ctrl+Alt+N。'))
        self.night_button.toggled.connect(lambda on: self.set_night(on) if on != self._night else None)
        orientation_row.addWidget(self.surround_grey_button)
        orientation_row.addWidget(self.night_button)
        work.addLayout(orientation_row)
        self.file_strip = QListWidget()
        self.file_strip.setObjectName("selectedPhotos")
        self.file_strip.setFlow(QListWidget.Flow.LeftToRight)
        self.file_strip.setWrapping(False)
        self.file_strip.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_strip.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.file_strip.setMaximumHeight(64)
        self.file_strip.setMinimumHeight(52)
        self.file_strip.itemSelectionChanged.connect(lambda: self.remove_files_button.setEnabled(bool(self.file_strip.selectedItems())))
        work.addWidget(self.file_strip)

        # Right: the steps in working order, ① regions → ② colour → ③ effects, then a fixed
        # bottom with the save location and the one primary action.
        body.addWidget(_Hairline(vertical=True))
        panel = QFrame()
        panel.setObjectName("controlPanel")
        themes.set_role(panel, "toolbar")
        panel.setFixedWidth(352 if self._startup_language == 'en' else 302)
        sidebar = QVBoxLayout(panel)
        sidebar.setContentsMargins(16, 12, 14, 12)
        sidebar.setSpacing(10)
        body.addWidget(panel)
        scroll = QScrollArea()
        scroll.setObjectName("stepsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._steps_scroll = scroll
        content = QWidget()
        content.setObjectName("stepsContent")
        left = QVBoxLayout(content)
        left.setContentsMargins(0, 0, 6, 0)
        left.setSpacing(8)
        scroll.setWidget(content)
        sidebar.addWidget(scroll, 1)

        regions_head = self._step_heading(left, "①", tr('分区'))
        self.regions_provenance = ui_widgets.ProvenanceDot("human", accessible_name=tr('由你确认'))
        self.regions_provenance.setObjectName("regionsProvenance")
        self.regions_seal = ui_widgets.Seal(tr('人定'), tr('人工确认的分区'), 24)
        self.regions_seal.setObjectName("regionsSeal")
        self.regions_seal.setToolTip(tr('分区由你在标记画面中保存并确认。'))
        regions_head.addWidget(self.regions_provenance, 0, Qt.AlignmentFlag.AlignVCenter)
        regions_head.addWidget(self.regions_seal, 0, Qt.AlignmentFlag.AlignVCenter)
        self.regions_provenance.hide()
        self.regions_seal.hide()
        self.mask_notice = self._label(tr('有地景就标记边界，有银河就画椭圆；未标记的部分不启用相应处理。'), "notice")
        self.mask_notice.setObjectName("maskNotice")
        left.addWidget(self.mask_notice)
        self.scene_mode = QComboBox()
        self.scene_mode.setObjectName("sceneMode")
        self.scene_mode.addItem(tr('请选择：纯天空 / 含地景'), "unknown")
        self.scene_mode.addItem(tr('纯天空（全画面参与）'), "sky")
        self.scene_mode.addItem(tr('含地景（在照片上标记）'), "landscape")
        self.scene_mode.setCurrentIndex(1)
        self.scene_mode.currentIndexChanged.connect(self._update_mask_notice)
        self.scene_mode.hide()  # Kept only for old editing-project compatibility.
        self.sky_mask = QLineEdit()
        self.sky_mask.setObjectName("skyMaskPath")
        self.sky_mask.setPlaceholderText(tr('已有天空 / 地景分区 PNG'))
        self.sky_mask.setClearButtonEnabled(True)
        self.sky_mask.textChanged.connect(self.sky_mask.setToolTip)
        self.sky_mask.hide()
        self.mask_browse_button = QPushButton(tr('载入分区'))
        self.mask_browse_button.clicked.connect(self._pick_mask)
        self._path_buttons.append(self.mask_browse_button)
        self.mask_browse_button.hide()
        self.regions_button = QPushButton(tr('标记天空 / 地景 / 银河'))
        self.regions_button.setObjectName("editRegions")
        self.regions_button.clicked.connect(self.edit_regions)
        left.addWidget(self.regions_button)

        colour_head = self._step_heading(left, "②", tr('调色'))
        self.reset_color_button = QPushButton(tr('重置调整'))
        self.reset_color_button.setObjectName("resetColorButton")
        themes.set_role(self.reset_color_button, "ghost")
        self.reset_color_button.clicked.connect(self.reset_color_adjustments)
        colour_head.addWidget(self.reset_color_button)
        self.warmth_slider = self._slider_control(left, tr('冷暖'), "warmth", -100, 100, 0, tr('冷'), tr('暖'))
        self.tint_slider = self._slider_control(left, tr('色调'), "tint", -100, 100, 0, tr('绿'), tr('洋红'))
        self.exposure_slider = self._slider_control(left, tr('曝光'), "displayExposure", -40, 40, 0, "−4 EV", "+4 EV", scale=10)
        self.warmth_value = self.warmth_slider.value_field
        self.tint_value = self.tint_slider.value_field
        self.exposure_value = self.exposure_slider.value_field

        self._effects_heading = self._step_heading(left, "③", tr('可选效果')).parentWidget()
        self.trails_enabled = QCheckBox(tr('保留流星等亮迹'))
        self.trails_enabled.setObjectName("trailsEnabled")
        self.trails_enabled.setToolTip(tr('仅在开启时扫描。可信的流星自动加回；其他亮迹等你复核后再加回。'))
        left.addWidget(self.trails_enabled)
        self.trails_note = self._label(tr('流星自动加回，另存含亮迹版本；其他亮迹待复核。基础成片同时保留。'))
        themes.set_role(self.trails_note, "hint")
        self.trails_note.hide()
        self.trails_enabled.toggled.connect(self.trails_note.setVisible)
        self.trails_enabled.toggled.connect(lambda: self._schedule_jpeg_estimate())
        left.addWidget(self.trails_note)
        self.stars_enabled = QCheckBox(tr('整理星点'))
        self.stars_enabled.setObjectName("starsEnabled")
        left.addWidget(self.stars_enabled)
        self.workflow_star_panel = QWidget()
        self.workflow_star_panel.setObjectName("workflowStarPanel")
        star_options = QVBoxLayout(self.workflow_star_panel)
        star_options.setContentsMargins(18, 0, 0, 3)
        self.workflow_auto_strength = QCheckBox(tr('使用星点密度建议'))
        self.workflow_auto_strength.setObjectName("workflowAutoStrength")
        self.workflow_auto_strength.setChecked(True)
        star_options.addWidget(self.workflow_auto_strength)
        self.workflow_background_strength = self._color_control(star_options, tr('背景移星力度'), "workflowBackgroundStrength", 0, 100, 55, 5, " %")
        self.workflow_milkyway_offset = self._color_control(star_options, tr('银河相对偏移'), "workflowMilkywayOffset", -100, 100, 0, 5, " %")
        star_note = self._label(tr('银河力度＝背景＋偏移；负值少移星。需圈定并确认银河。100%移除后端分离的全部星层。'))
        themes.set_role(star_note, "hint")
        star_options.addWidget(star_note)
        # Optional extra layers, copied into the works folder only when asked for.
        layer_tip = tr('只在天空中去星；地景和过渡羽化带不变。去星模型为试验性。'
                       '无星图＋星点层＝整理前的成片。')
        self.workflow_save_starless = QCheckBox(tr('另存无星图'))
        self.workflow_save_starless.setObjectName("workflowSaveStarless")
        self.workflow_save_starless.setToolTip(tr('在作品文件夹另存 16 位无星图（starless_srgb16.tif）。') + "\n" + layer_tip)
        star_options.addWidget(self.workflow_save_starless)
        self.workflow_save_star_layer = QCheckBox(tr('另存星点层'))
        self.workflow_save_star_layer.setObjectName("workflowSaveStarLayer")
        self.workflow_save_star_layer.setToolTip(
            tr('在作品文件夹另存分离出的全部星点（加、减两层 16 位 TIFF），与整理力度无关。') + "\n" + layer_tip)
        star_options.addWidget(self.workflow_save_star_layer)
        self.star_component_note = self._label(tr('需要去星模型：在“设置 → 处理后端”选择；未安装时在“组件与提示词管理”准备。'))
        themes.set_role(self.star_component_note, "hint")
        star_options.addWidget(self.star_component_note)
        self.star_component_button = QPushButton(tr('选择星点整理后端'))
        self.star_component_button.setObjectName("chooseStarBackend")
        self.star_component_button.clicked.connect(lambda: self._show_advanced("settings"))
        star_options.addWidget(self.star_component_button)
        self.workflow_star_panel.hide()
        self.stars_enabled.toggled.connect(self.workflow_star_panel.setVisible)
        self.stars_enabled.toggled.connect(lambda: self._update_controls())
        self.workflow_auto_strength.toggled.connect(lambda: self._update_controls())
        left.addWidget(self.workflow_star_panel)
        self.milkyway_enabled = QCheckBox(tr('突出银河 · 需先圈定范围'))
        self.milkyway_enabled.setObjectName("milkywayEnabled")
        self.milkyway_enabled.setEnabled(False)
        self.milkyway_enabled.setToolTip(tr('先在标记画面中圈定银河，并明确确认可见银河。'))
        left.addWidget(self.milkyway_enabled)
        self.milkyway_strength = self._color_control(left, tr('银河增强'), "milkywayStrength", 0, 100, 60, 5, " %")
        self.milkyway_enabled.toggled.connect(lambda: self._update_controls())
        left.addStretch(1)

        # Fixed bottom of the step column.
        sidebar.addWidget(_Hairline(vertical=False))
        self.output_dir = self._path_field(sidebar, tr('作品保存位置'), tr('选择结果文件夹'), "outputDirectory", self._pick_output)
        self._output_browse_button = self._path_buttons[-1]
        themes.set_role(self.output_dir, "mono")
        pictures = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        self.output_dir.setText(str(Path(pictures or str(Path.home() / "Pictures")) / "Aester"))
        self._build_jpeg_quality(sidebar)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start_button = QPushButton(tr('开始合成'))
        self.start_button.setObjectName("primary")
        themes.set_role(self.start_button, "primary")
        self.start_button.setMinimumHeight(44)
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton(tr('停止'))
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setMinimumHeight(44)
        self.cancel_button.clicked.connect(self.cancel_processing)
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.cancel_button)
        sidebar.addLayout(buttons)
        self.trail_review_button = QPushButton(tr('复核亮迹候选'))
        self.trail_review_button.setObjectName("trailReviewButton")
        self.trail_review_button.clicked.connect(self.open_trail_review)
        sidebar.addWidget(self.trail_review_button)
        # Directory mode remains a compatibility bridge for restored older jobs.
        self.input_dir = QLineEdit(self)
        self.input_dir.setObjectName("inputDirectory")
        self.input_dir.textChanged.connect(self.input_dir.setToolTip)
        self.input_dir.hide()

        # Bottom readout: only real data (stage, progress, percentage, the full message).
        readout = QWidget()
        readout.setObjectName("readoutStrip")
        themes.set_role(readout, "readout")
        progress_heading = QHBoxLayout(readout)
        progress_heading.setContentsMargins(20, 8, 16, 8)
        progress_heading.setSpacing(10)
        # One shared 二十八宿 sweep for every activity mark (spinner, stage ring, empty canvas).
        self._activity_clock = ui_widgets.ActivityClock(self)
        self._activity_on = False
        self._activity_timer = QTimer(self)
        self._activity_timer.setSingleShot(True)
        self._activity_timer.timeout.connect(self._settle_activity)
        self.activity_spinner = ui_widgets.LodgeSpinner(self._activity_clock)
        self.activity_spinner.setObjectName("activitySpinner")
        self.activity_spinner.setAccessibleName(tr('正在处理'))
        self.activity_spinner.hide()
        progress_heading.addWidget(self.activity_spinner, 0, Qt.AlignmentFlag.AlignVCenter)
        self.stage_label = QLabel(tr('准备就绪'))
        self.stage_label.setObjectName("stageLabel")
        self.stage_label.setWordWrap(False)
        progress_heading.addWidget(self.stage_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("jobProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setFixedWidth(150)
        progress_heading.addWidget(self.progress_bar, 0, Qt.AlignmentFlag.AlignVCenter)
        self.percent_label = QLabel("—")
        self.percent_label.setObjectName("percentLabel")
        themes.set_role(self.percent_label, "mono")
        self.percent_label.setMinimumWidth(36)
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_heading.addWidget(self.percent_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.error_icon = ui_widgets.ErrorIcon(16, accessible_name=tr('错误'))
        self.error_icon.setObjectName("errorIcon")
        self.error_icon.hide()
        progress_heading.addWidget(self.error_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        self.message_label = _ReadoutMessage(tr('添加照片即可开始；有地景或银河时再标记。'))
        self.message_label.setObjectName("jobMessage")
        self.message_label.setWordWrap(True)
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message_label.setMaximumHeight(40)
        self.message_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.message_label.errorChanged.connect(self.error_icon.setVisible)
        progress_heading.addWidget(self.message_label, 1, Qt.AlignmentFlag.AlignVCenter)
        # A failed or cancelled job offers its next step right here (same handler as the menu).
        self.resume_button = QPushButton(tr('继续任务'))
        self.resume_button.setObjectName("resumeJobButton")
        self.resume_button.setToolTip(tr('从检查点继续这个失败或已停止的任务；已完成的步骤不会重做。'))
        self.resume_button.clicked.connect(self.resume_processing)
        self.resume_button.hide()
        progress_heading.addWidget(self.resume_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.details_button = QPushButton(tr('详情'))
        self.details_button.setObjectName("jobDetailsButton")
        themes.set_role(self.details_button, "ghost")
        self.details_button.clicked.connect(self.show_job_details)
        progress_heading.addWidget(self.details_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(readout)
        # Advanced controls retain their existing callbacks and state contracts.
        # They live in dialogs so the basic path never needs a scrolling checklist.
        settings = self._dialog_form("settings", tr('处理后端'), 640, 540)
        self.siril_path = self._path_field(settings, tr('Siril · 官方处理引擎'), tr('留空自动寻找 Siril 1.4.4'), "sirilPath", self._pick_siril)
        self.backend_status = self._label(tr('正在检查处理引擎…'), "status")
        self.backend_status.setObjectName("backendStatus")
        settings.addWidget(self.backend_status)
        self.dng_converter_path = self._path_field(settings, tr('RAW 兼容转换 · Adobe DNG Converter'),
            tr('自动寻找已安装的转换器；也可选择程序'), 'dngConverterPath', self._pick_dng_converter)
        converter_download = QPushButton(tr('获取 Adobe 官方免费转换器'))
        converter_download.clicked.connect(self._open_dng_download)
        settings.addWidget(converter_download)
        self.stack_range_checkbox = QCheckBox(tr('保护堆栈高光'))
        self.stack_range_checkbox.setObjectName("protectStackHighlights")
        self.stack_range_checkbox.setChecked(False)
        self.stack_range_checkbox.setToolTip(tr('保留额外高光范围；单通道零值和负值照常堆栈，暗端与普通堆栈相同在0处截断；遇非有限值会停止并提示。'))
        settings.addWidget(self.stack_range_checkbox)
        self.stack_engine_path = self._path_field(
            settings, tr('Siril · Aester 改进构建（可选）'), tr('保留有效零值、负值与高光余量'), "stackEnginePath",
            lambda: self._pick_optional_file(self.stack_engine_path, tr('选择数值保真堆栈引擎'), tr('程序 (*.exe);;所有文件 (*)')))
        self.stack_backend_choice = self._choice_control(settings, tr('使用的堆栈后端'), 'stackBackendChoice', [
            (tr('Siril 官方构建'), 'official'), (tr('Siril · Aester 改进构建'), 'finite'),
            (tr('自选兼容构建'), 'custom')])
        self.finite_backend_note = self._label(tr('正在检查 Aester 改进构建…'))
        settings.addWidget(self.finite_backend_note)
        self.install_finite_button = QPushButton(tr('安装已有 Aester 改进构建…'))
        self.install_finite_button.clicked.connect(self._install_finite_backend)
        settings.addWidget(self.install_finite_button)
        self.finite_sources_button = QPushButton(tr('打开改进构建的源码与许可'))
        self.finite_sources_button.clicked.connect(self._open_finite_sources)
        self.finite_sources_button.setVisible(False)
        settings.addWidget(self.finite_sources_button)
        self.stack_backend_choice.currentIndexChanged.connect(self._select_stack_backend)
        self.star_model_path = self._path_field(
            settings, tr('去星模型 · 可选'), tr('选择自备 ONNX 模型'), "starModelPath",
            lambda: self._pick_optional_file(self.star_model_path, tr('选择去星模型'), tr('ONNX 模型 (*.onnx)')))
        self.model_channel_order = self._choice_control(settings, tr('颜色通道'), "modelChannelOrder", [("RGB", "rgb"), ("BGR", "bgr")])
        self.model_range = self._choice_control(settings, tr('模型数值范围'), "modelRange", [(tr('0 至 1'), "0..1"), (tr('−1 至 1'), "-1..1")])
        self.model_layout = self._choice_control(settings, tr('张量布局'), "modelLayout", [("NHWC", "nhwc"), ("NCHW", "nchw")])
        self.model_stitch = self._choice_control(settings, tr('分块拼接'), "modelStitch", [(tr('渐变融合'), "blend"), (tr('裁取中心'), "crop")])
        self.star_model_path.textChanged.connect(lambda: self._update_controls())
        self.stack_engine_path.textChanged.connect(lambda: self._update_controls())
        self.stack_engine_path.editingFinished.connect(self._sync_stack_backend_choice)
        self.workflow_settings_layout = settings
        settings.addStretch(1)

        colors = self._dialog_form("colors", tr('旧任务颜色工具'), 560, 360)
        self.red_gain = self._color_control(colors, tr('红通道'), "redGain", .25, 4, 1, .05, " ×")
        self.blue_gain = self._color_control(colors, tr('蓝通道'), "blueGain", .25, 4, 1, .05, " ×")
        self.exposure_ev = self._color_control(colors, tr('曝光'), "exposureEV", -3, 3, 0, .25, " EV")
        color_buttons = QHBoxLayout()
        self.reset_preview_button = QPushButton(tr('重置参数'))
        self.reset_preview_button.setObjectName("resetPreviewButton")
        self.reset_preview_button.clicked.connect(self.reset_preview_colors)
        self.render_preview_button = QPushButton(tr('另存新预览'))
        self.render_preview_button.setObjectName("renderPreviewButton")
        self.render_preview_button.clicked.connect(self.render_preview)
        color_buttons.addWidget(self.reset_preview_button)
        color_buttons.addWidget(self.render_preview_button, 1)
        colors.addLayout(color_buttons)
        colors.addStretch(1)

        stars = self._dialog_form("stars", tr('星点整理'), 650, 640)
        self._build_star_controls(stars)
        stars.addStretch(1)
        audit = self._dialog_form("raw", tr('RAW 检查'), 650, 400)
        self.raw_audit_button = QPushButton(tr('选择并检查一张 RAW'))
        self.raw_audit_button.setObjectName("inspectRawButton")
        self.raw_audit_button.clicked.connect(self.inspect_single_raw)
        audit.addWidget(self.raw_audit_button)
        self.open_raw_audit_button = QPushButton(tr('打开检查结果'))
        self.open_raw_audit_button.setObjectName("openRawAuditButton")
        self.open_raw_audit_button.clicked.connect(self.open_raw_audit)
        audit.addWidget(self.open_raw_audit_button)
        self.raw_audit_status = self._label(tr('等待选择文件'))
        self.raw_audit_status.setObjectName("rawAuditStatus")
        self.raw_audit_status.setTextFormat(Qt.TextFormat.PlainText)
        self.raw_audit_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        audit.addWidget(self.raw_audit_status)
        audit.addStretch(1)
        details = self._dialog_form("details", tr('任务详情'), 760, 480)
        self.job_path_label = self._label("")
        self.job_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.addWidget(self.job_path_label)
        self.job_details = QTextBrowser()
        self.job_details.setObjectName("jobDetails")
        details.addWidget(self.job_details, 1)

        # Invisible callback anchors also keep existing integrations compatible.
        self.star_controls_button = self._hidden_button(tr('调整星点密度'), self.show_star_controls)
        self.load_button = self._hidden_button(tr('打开已有任务'), self._pick_job_file)
        self.demo_button = self._hidden_button(tr('生成合成演示数据'), self.generate_demo)
        self.open_reference_button = self._hidden_button(tr('打开参考目录'), self.open_reference)
        self._build_menus()
        self._apply_tab_order()

    def _hidden_button(self, title: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(title, self)
        button.clicked.connect(callback)
        button.hide()
        return button

    def _restore_preferences(self) -> None:
        settings = self._preferences
        if settings is None:
            return
        self.set_theme(settings.value("appearance/theme", themes.DEFAULT_THEME), persist=False)
        # Viewing conditions are restored without writing anything back.
        self.set_surround(surround_key(settings.value("appearance/surround", "xuan")), persist=False)
        self.set_night(_preference_bool(settings.value("appearance/night", False)), persist=False)
        for field, key in ((self.output_dir, "output_directory"), (self.siril_path, "siril_path"),
                           (self.stack_engine_path, "stack_engine_path"), (self.star_model_path, "star_model_path"),
                           (self.dng_converter_path, 'dng_converter_path')):
            value = settings.value(key, "")
            if isinstance(value, str) and value:
                field.setText(value)
                field.setCursorPosition(len(field.text()))
                field.deselect()
        for control, key in ((self.model_channel_order, "channel_order"), (self.model_range, "value_range"),
                             (self.model_layout, "layout"), (self.model_stitch, "stitch")):
            value = settings.value("model/" + key, "")
            index = control.findData(value)
            if index >= 0:
                control.setCurrentIndex(index)
        stored_quality = settings.value("output/jpeg_quality", None)
        try:
            quality = jpeg_estimate.validate_quality(int(str(stored_quality).strip()))
        except (TypeError, ValueError):
            quality = jpeg_estimate.DEFAULT       # absent or unusable: the default, nothing written back
        self._jpeg_preference = quality
        self._set_jpeg_quality_quietly(quality)
        mode = settings.value('stack_backend_mode', 'custom' if self.stack_engine_path.text().strip() else 'official')
        self.stack_backend_choice.blockSignals(True)
        self.stack_backend_choice.setCurrentIndex(max(0, self.stack_backend_choice.findData(mode)))
        self.stack_backend_choice.blockSignals(False)
        from starstack.raw_import import find_converter
        found = find_converter(self.dng_converter_path.text().strip() or None)
        if found['available']:
            self.dng_converter_path.setText(found['path'])

    def _save_preferences(self) -> None:
        settings = self._preferences
        if settings is None:
            return
        settings.setValue("appearance/theme", self._theme_name)
        settings.setValue("appearance/surround", surround_preference(self._surround))
        settings.setValue("appearance/night", bool(self._night))
        settings.setValue('stack_backend_mode', self.stack_backend_choice.currentData())
        # The preference, not a loaded workflow's frozen value the slider may be showing.
        settings.setValue("output/jpeg_quality", self._jpeg_preference)
        for field, key in ((self.output_dir, "output_directory"), (self.siril_path, "siril_path"),
                           (self.stack_engine_path, "stack_engine_path"), (self.star_model_path, "star_model_path"),
                           (self.dng_converter_path, 'dng_converter_path')):
            settings.setValue(key, field.text().strip())
        for control, key in ((self.model_channel_order, "channel_order"), (self.model_range, "value_range"),
                             (self.model_layout, "layout"), (self.model_stitch, "stitch")):
            settings.setValue("model/" + key, control.currentData())
        settings.sync()

    def set_theme(self, name: str, *, persist: bool = True) -> None:
        """Change interface chrome immediately; never rerender a photographic preview."""
        self._theme_name = themes.apply_theme(self, name, night=self._night)
        for key in themes.THEME_LABELS:
            item = getattr(self, "menu_actions", {}).get("theme_" + key)
            if item is not None:
                item.setChecked(key == self._theme_name)
        settings = getattr(self, "_preferences", None)
        if persist and settings is not None:
            settings.setValue("appearance/theme", self._theme_name)
            settings.sync()

    def set_night(self, on: bool, *, persist: bool = True) -> None:
        """Red-only interface and a red single-channel preview; display only, never exported."""
        self._night = bool(on)
        themes.apply_theme(self, self._theme_name, night=self._night)
        self.preview.set_night(self._night)
        for control in (getattr(self, "menu_actions", {}).get("night"), getattr(self, "night_button", None)):
            if control is not None and control.isChecked() != self._night:
                control.blockSignals(True)
                control.setChecked(self._night)
                control.blockSignals(False)
        settings = getattr(self, "_preferences", None)
        if persist and settings is not None:
            settings.setValue("appearance/night", self._night)
            settings.sync()

    def set_surround(self, name: str, *, persist: bool = True) -> None:
        """The photo surround (玄 / mid grey / black) is a viewing condition, independent of theme."""
        self._surround = surround_key(name)
        if self._surround != "mid_grey":
            self._surround_before_grey = self._surround
        self.preview.set_surround(self._surround)
        actions = getattr(self, "menu_actions", {})
        for stored, key in SURROUND_PREFERENCES.items():
            item = actions.get("surround_" + stored)
            if item is not None:
                item.setChecked(key == self._surround)
        grey = getattr(self, "surround_grey_button", None)
        if grey is not None and grey.isChecked() != (self._surround == "mid_grey"):
            grey.blockSignals(True)
            grey.setChecked(self._surround == "mid_grey")
            grey.blockSignals(False)
        settings = getattr(self, "_preferences", None)
        if persist and settings is not None:
            settings.setValue("appearance/surround", surround_preference(self._surround))
            settings.sync()

    def _surround_toggled(self, on: bool) -> None:
        if on != (self._surround == "mid_grey"):
            self.set_surround("mid_grey" if on else self._surround_before_grey)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        # The native caption follows the dark theme and night vision (Windows DWM only).
        themes.set_title_bar_dark(self, self._night or self._theme_name == "dark", night=self._night)

    def select_language(self, name: str) -> None:
        """Persist the next-start language without interrupting edits or workers."""
        self._selected_language = normalize_language(name)
        if self._preferences is not None:
            self._preferences.setValue("appearance/language", self._selected_language)
            self._preferences.sync()
        for value in ("zh_CN", "en"):
            self.menu_actions["language_" + value].setChecked(value == self._selected_language)
        if self._selected_language == self._startup_language:
            message = tr("已保持当前界面语言。")
        else:
            message = tr("语言已保存，下次启动生效。请先保存编辑项目；当前任务继续运行。")
        self.message_label.setText(message)
        # A non-modal notice also remains visible if a running task updates status.
        if "language_notice" not in self._advanced_dialogs:
            form = self._dialog_form("language_notice", tr("界面语言"), 640, 310)
            self._language_notice = self._label("")
            form.addWidget(self._language_notice)
            form.addStretch(1)
        self._language_notice.setText(message)
        self._show_advanced("language_notice")

    def _choice_control(self, layout: QVBoxLayout, title: str, name: str,
                        options: list[tuple[str, str]]) -> QComboBox:
        row = QHBoxLayout()
        row.addWidget(self._label(title, "field"), 1)
        control = QComboBox()
        control.setObjectName(name)
        control.setAccessibleName(title)
        for label, value in options:
            control.addItem(label, value)
        control.setMinimumWidth(180)
        row.addWidget(control)
        layout.addLayout(row)
        return control

    def _pick_optional_file(self, field: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, field.text() or str(WORKSPACE), file_filter)
        if path:
            field.setText(path)
            if field is self.stack_engine_path:
                self._sync_stack_backend_choice()
            field.setCursorPosition(len(field.text()))
            field.deselect()

    def _dialog_form(self, key: str, title: str, width: int, height: int) -> QVBoxLayout:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setObjectName(key + "Dialog")
        dialog.resize(width, height)
        dialog.setMinimumSize(400, 300)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(18, 18, 18, 14)
        outer.setSpacing(12)
        outer.addWidget(self._label(title, "section"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        form = QVBoxLayout(content)
        form.setContentsMargins(2, 4, 12, 6)
        form.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        help_button = QPushButton(tr('使用帮助'))
        help_button.setObjectName("dialogHelpButton")
        themes.set_role(help_button, "ghost")
        help_button.clicked.connect(lambda: self.show_help(key))
        buttons.addWidget(help_button)
        buttons.addStretch(1)
        close_button = QPushButton(tr('关闭'))
        close_button.setObjectName("dialogCloseButton")
        close_button.clicked.connect(dialog.close)
        buttons.addWidget(close_button)
        outer.addLayout(buttons)
        # Enter in a path field must never press "使用帮助" (or any other button).
        for button in (help_button, close_button):
            button.setAutoDefault(False)
            button.setDefault(False)
        self._advanced_dialogs[key] = dialog
        return form

    @staticmethod
    def _no_default_buttons(dialog: QDialog) -> None:
        """QDialog makes its first auto-default button the Enter target; these forms have none."""
        for button in dialog.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _show_advanced(self, key: str) -> None:
        dialog = self._advanced_dialogs[key]
        self._no_default_buttons(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _build_menus(self) -> None:
        self.menu_actions: dict[str, QAction] = {}
        bar = self.menuBar()
        bar.setNativeMenuBar(False)
        files, settings, help_menu = [bar.addMenu(name) for name in (tr('文件'), tr('设置'), tr('帮助'))]

        def action(menu: QMenu | None, key: str, text: str, callback: Callable[[], None],
                   *, button: QPushButton | None = None, shortcut: str = "") -> QAction:
            item = menu.addAction(text) if menu is not None else QAction(text, self)
            item.setObjectName(key + "Action")
            item.triggered.connect(callback)
            if shortcut:
                item.setShortcut(shortcut)
            self.menu_actions[key] = item
            if button is not None:
                self._action_buttons.append((item, button))
            return item

        action(files, "input", tr('添加照片…'), self._pick_input, shortcut="Ctrl+I")
        action(files, "save_project", tr('保存编辑项目…'), self.save_editing_project, shortcut="Ctrl+S")
        action(files, "open_project", tr('打开编辑项目…'), self.open_editing_project, shortcut="Ctrl+Shift+O")
        action(files, "output", tr('更改保存位置…'), self._pick_output)
        files.addSeparator()
        action(files, "load", tr('打开已有任务…'), self.load_button.click, button=self.load_button, shortcut="Ctrl+O")
        action(files, "resume", tr('继续未完成的任务…'), self.resume_button.click, button=self.resume_button)
        action(files, "results", tr('打开作品目录'), self.open_button.click, button=self.open_button)
        files.addSeparator()
        action(files, "quit", tr('退出'), self.close, shortcut="Ctrl+Q")
        action(settings, "settings", tr('处理后端…'), lambda: self._show_advanced("settings"))
        action(settings, "storage", tr('作品与缓存…'), lambda: self.experience.show_storage())
        connection = action(settings, "assistant", tr('连接 AI 助手…'), lambda: self.experience.show_connection())
        self.menu_actions["connect_ai"] = connection
        action(settings, "import_mask", tr('导入已有天空分区…'), self._pick_mask)
        appearance = settings.addMenu(tr('外观'))
        appearance.setObjectName("appearanceMenu")
        self._theme_actions = QActionGroup(self)
        self._theme_actions.setExclusive(True)
        for theme_name, label in themes.THEME_LABELS.items():
            item = action(appearance, "theme_" + theme_name, tr(label),
                          lambda checked=False, value=theme_name: self.set_theme(value))
            item.setCheckable(True)
            item.setChecked(theme_name == self._theme_name)
            self._theme_actions.addAction(item)
        appearance.addSeparator()
        night = action(appearance, "night", tr('夜视模式'), lambda checked=False: self.set_night(checked),
                       shortcut="Ctrl+Alt+N")
        night.setCheckable(True)
        night.setChecked(self._night)
        night.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        surround = appearance.addMenu(tr('画面四周'))
        surround.setObjectName("surroundMenu")
        self._surround_actions = QActionGroup(self)
        self._surround_actions.setExclusive(True)
        for stored, label in (("xuan", tr('玄 · 默认')), ("grey", tr('中灰评估')), ("black", tr('纯黑 · 仅观赏'))):
            item = action(surround, "surround_" + stored, label,
                          lambda checked=False, value=SURROUND_PREFERENCES[stored]: self.set_surround(value))
            item.setCheckable(True)
            item.setChecked(SURROUND_PREFERENCES[stored] == self._surround)
            self._surround_actions.addAction(item)
        language_menu = settings.addMenu(tr("语言 / Language"))
        language_menu.setObjectName("languageMenu")
        self._language_actions = QActionGroup(self)
        self._language_actions.setExclusive(True)
        for value, label in (("zh_CN", "简体中文"), ("en", "English")):
            item = action(language_menu, "language_" + value, label,
                          lambda checked=False, language=value: self.select_language(language))
            item.setCheckable(True)
            item.setChecked(value == self._selected_language)
            self._language_actions.addAction(item)
        action(help_menu, "help", tr('第一次使用'), self.show_help, shortcut="F1")
        action(help_menu, "help_features", tr('让 AI 帮我做'), lambda: self.show_help("mcp"))
        action(help_menu, "help_retouch", tr('自然语言修图'), lambda: self.show_help("retouch"))
        action(help_menu, "help_limits", tr('支持范围与遇到问题'), lambda: self.show_help("limits"))
        action(help_menu, "about", tr('关于 Aester'), self.show_about)
        # Compatibility anchors are deliberately absent from the user menu.
        for key, text, callback in (
            ("colors", tr('旧任务颜色工具'), lambda: self._show_advanced("colors")),
            ("stars", tr('调整已有星点图层'), self.show_star_controls),
            ("raw", tr('RAW 检查'), lambda: self._show_advanced("raw")),
            ("background", tr('背景协调'), lambda: self._show_tool("background")),
            ("milkyway", tr('银河局部增强'), lambda: self._show_tool("milkyway")),
            ("trails", tr('亮迹复核'), lambda: self._show_tool("trails")),
            ("layers", tr('导出编辑图层'), lambda: self._show_tool("layers")),
            ("details", tr('任务详情'), self.show_job_details),
            ("reference_files", tr('打开参考目录'), self.open_reference)):
            action(None, key, text, callback)

    def _show_tool(self, key: str) -> None:
        if self._extra_tools is None:
            from aester_desktop.advanced_tools import AdvancedTools
            self._extra_tools = AdvancedTools(self)
        self._extra_tools.show(key)
        self._update_controls()

    def show_help(self, section: str = "") -> None:
        from starstack.user_guide import guide_text
        if not hasattr(self, "help_browser"):
            form = self._dialog_form("help", tr('使用指南 · GUI 与 MCP'), 900, 700)
            self.help_browser = QTextBrowser()
            self.help_browser.setObjectName("userGuide")
            self.help_browser.setOpenExternalLinks(True)
            form.addWidget(self.help_browser, 1)
        self.help_browser.setMarkdown(guide_text())
        themes.style_rich_text(self.help_browser, self._theme_name)
        self._show_advanced("help")
        if section:
            self.help_browser.scrollToAnchor(section)
            if section == 'retouch':
                # Qt's Markdown reader can omit raw HTML anchor positions.
                # Locate the visible heading after the browser has its layout.
                QTimer.singleShot(0, self._scroll_retouch_help)
        else:
            self.help_browser.verticalScrollBar().setValue(0)

    def show_about(self) -> None:
        from aester_desktop.about import show_about
        show_about(self, self._theme_name)

    def _scroll_retouch_help(self) -> None:
        cursor = self.help_browser.document().find(tr('自然语言修图'))
        if not cursor.isNull():
            top = self.help_browser.document().documentLayout().blockBoundingRect(cursor.block()).top()
            self.help_browser.verticalScrollBar().setValue(int(top))

    def retouch_photo_path(self) -> str | None:
        """Use a recorded display TIFF, never guess a sibling of a preview."""
        if self._view_mode == 'advanced':
            return None
        if self._view_mode == 'stars' and self._star_result:
            record = self._star_result
        elif self._rendered_preview and self._rendered_preview.get('job_dir') == self.job_dir:
            record = self._rendered_preview
        else:
            record = self._workflow or self._job or {}
        artifacts = record.get('artifacts') or []
        if isinstance(artifacts, dict):
            artifacts = list(artifacts.values())
        artifacts = list(artifacts)
        if record is self._workflow and self._has_trails_version(record):
            # The base delivery is recorded too; its 16-bit photograph backs the 基础成片 tab.
            artifacts.append({'path': str(Path(str(record['base_delivery_dir'])) / 'final_srgb16.tif'), 'kind': 'tiff'})
        candidates = [item for item in artifacts if isinstance(item, dict)
                      and Path(str(item.get('path') or '')).suffix.lower() in ('.tif', '.tiff')
                      and 'master' not in Path(str(item.get('path') or '')).name.lower()
                      and 'linear' not in Path(str(item.get('path') or '')).name.lower()]
        index = self.style_tabs.currentIndex()
        jpeg = self._jpg_artifacts[index] if 0 <= index < len(self._jpg_artifacts) else {}
        stem = Path(str(jpeg.get('path') or '')).stem
        folder = str(Path(str(jpeg.get('path'))).parent) if jpeg.get('path') else ''
        display_stems = (stem, stem + '_srgb16', stem + '_display_16bit') if stem else ()
        candidates = [item for item in candidates if Path(str(item['path'])).name in
                      ('final_srgb16.tif', 'star_reduced_srgb16.tif', 'natural_display_16bit.tif')
                      or Path(str(item['path'])).stem in display_stems]
        # The TIFF beside the photograph on screen wins (with-trails first, then the base).
        candidates.sort(key=lambda item: (
            bool(folder) and not self._same_dir(Path(str(item['path'])).parent, folder),
            Path(str(item['path'])).stem not in display_stems,
            Path(str(item['path'])).name != 'final_srgb16.tif'))
        for item in candidates:
            path = Path(str(item['path']))
            if path.is_absolute() and path.is_file():
                return str(path)
        return None

    def show_job_details(self) -> None:
        import json
        text = self.message_label.text()
        record = self._workflow or self._job
        if record:
            text += "\n\n" + json.dumps(record, ensure_ascii=False, indent=2)
        self.job_details.setPlainText(text)
        self._show_advanced("details")

    @staticmethod
    def _status_message(record: dict[str, Any], fallback: str) -> str:
        """Localize presentation while retaining backend diagnostics in the record."""
        english = get_language() == "en"
        error = record.get("error")
        if error:
            # A stop the user asked for is not a failure: no "Operation incomplete:" prefix.
            prefix = tr("操作未完成：") if english and record.get("state") != "cancelled" else ""
            return prefix + translate_message(str(error))
        message = str(record.get("message") or fallback)
        localized = translate_message(message)
        if english and any("\u4e00" <= char <= "\u9fff" for char in localized):
            # Detailed dynamic worker text remains available in Job details.
            if record.get('state') in ACTIVE_STATES | {'succeeded'}:
                return fallback
            # Unknown/error-only responses must retain their diagnosis even if
            # there is no persisted job record to inspect afterward.
            return fallback + "\n" + message
        return localized

    def _color_control(self, layout: QVBoxLayout, title: str, name: str,
                       minimum: float, maximum: float, value: float, step: float,
                       suffix: str) -> QDoubleSpinBox:
        # The label stays on one line beside a box sized to its widest value; the wheel scrolls
        # the column rather than changing a value the person set.
        row = QHBoxLayout()
        row.setSpacing(8)
        label = self._label(title, "field")
        label.setWordWrap(False)
        label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        control = ui_widgets.StepSpinBox()
        control.setObjectName(name)
        control.setAccessibleName(title)
        control.setRange(minimum, maximum)
        control.setDecimals(2)
        control.setSingleStep(step)
        control.setValue(value)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        themes.set_role(control, "mono")
        control.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        label.setBuddy(control)
        row.addWidget(control)
        layout.addLayout(row)
        return control

    def _build_star_controls(self, layout: QVBoxLayout) -> None:
        card = QFrame()
        self._star_controls_card = card
        card.setObjectName("card")
        form = QVBoxLayout(card)
        form.setContentsMargins(13, 12, 13, 12)
        form.setSpacing(9)
        self._star_path_buttons: list[QPushButton] = []
        self.star_separation = self._path_field(
            form, tr('分层结果'), tr('选择分层的 result.json'), "starSeparationResult",
            lambda: self._pick_star_file(self.star_separation, tr('选择分层结果'), tr('分层记录 (*.json)')))
        self._star_path_buttons.append(self._path_buttons[-1])
        self.star_milkyway_mask = self._path_field(
            form, tr('银河渐变蒙版 · 可选'), tr('与成片同尺寸的蒙版'), "starMilkywayMask",
            lambda: self._pick_star_file(self.star_milkyway_mask, tr('选择银河蒙版'), tr('蒙版 (*.tif *.tiff *.png)')))
        self._star_path_buttons.append(self._path_buttons[-1])
        self.star_milkyway_evidence = self._path_field(
            form, tr('银河区域确认 · 可选'), tr('与当前图像及蒙版对应的确认文件'), "starMilkywayEvidence",
            lambda: self._pick_star_file(self.star_milkyway_evidence, tr('选择银河区域确认'), tr('确认记录 (*.json)')))
        self._star_path_buttons.append(self._path_buttons[-1])
        self.star_background_strength = self._color_control(
            form, tr('背景去星强度'), "starBackgroundStrength", 0, 100, 55, 5, " %")
        self.star_milkyway_offset = self._color_control(
            form, tr('银河相对偏移'), "starMilkywayOffset", -100, 100, -20, 5, tr(' 个百分点'))
        self.star_strength_note = self._label("")
        self.star_strength_note.setObjectName("starStrengthNote")
        self.star_strength_note.setTextFormat(Qt.TextFormat.PlainText)
        form.addWidget(self.star_strength_note)
        buttons = QHBoxLayout()
        self.star_analyze_button = QPushButton(tr('分析星点密度'))
        self.star_analyze_button.setObjectName("analyzeStarDensityButton")
        self.star_analyze_button.clicked.connect(self.analyze_star_density)
        self.star_apply_button = QPushButton(tr('采用建议'))
        self.star_apply_button.setObjectName("applyStarSuggestionButton")
        self.star_apply_button.clicked.connect(self.apply_star_suggestion)
        buttons.addWidget(self.star_analyze_button, 1)
        buttons.addWidget(self.star_apply_button)
        form.addLayout(buttons)
        self.star_status = self._label(tr('等待载入分层结果'))
        self.star_status.setObjectName("starReductionStatus")
        self.star_status.setTextFormat(Qt.TextFormat.PlainText)
        self.star_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addWidget(self.star_status)
        self.star_render_button = QPushButton(tr('生成星点整理成片'))
        self.star_render_button.setObjectName("renderStarReductionButton")
        self.star_render_button.clicked.connect(self.render_star_reduction)
        form.addWidget(self.star_render_button)
        self.star_open_button = QPushButton(tr('打开星点整理目录'))
        self.star_open_button.setObjectName("openStarReductionButton")
        self.star_open_button.clicked.connect(self.open_star_result)
        form.addWidget(self.star_open_button)
        layout.addWidget(card)

    def show_star_controls(self) -> None:
        self._show_advanced("stars")
        self.star_separation.setFocus()

    def _pick_star_file(self, field: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, field.text() or self.output_dir.text() or str(WORKSPACE), file_filter)
        if path:
            field.setText(path)
            field.setCursorPosition(len(field.text()))
            field.deselect()

    def _star_input_values(self) -> tuple[str, str, str]:
        return (self.star_separation.text().strip(), self.star_milkyway_mask.text().strip(),
                self.star_milkyway_evidence.text().strip())

    def _star_inputs_changed(self) -> None:
        self._star_suggestion = None
        self._star_suggestion_inputs = None
        self.star_status.setText(tr('输入已改变；可重新分析星点密度。现有成片仍可查看。'))
        self._update_star_strength_note()
        self._update_controls()

    def _update_star_strength_note(self) -> None:
        background = self.star_background_strength.value()
        offset = self.star_milkyway_offset.value()
        milkyway = max(0., min(100., background + offset))
        text = tr('背景 {0:.1f}% · 银河中心 {1:.1f}%（偏移 {2:+.1f} 个百分点）。', background, milkyway, offset)
        text += tr('\n未确认银河时，整片天空使用背景强度。')
        if max(background, milkyway) > 75:
            text += tr('\n超过 75% 后亮星也逐步减弱；100% 使用已分离的无星背景，其残晕仍可能存在。')
        else:
            text += tr('\n0% 保留星点；中低强度优先整理弱星，保留亮星。')
        self.star_strength_note.setText(text)

    def _star_operation_inputs(self) -> tuple[str, str, str, str] | None:
        if self._busy or self._closing_requested or self.job_state in ACTIVE_STATES:
            return None
        separation, mask, evidence = self._star_input_values()
        output = self.output_dir.text().strip()
        if not separation or not Path(separation).is_file():
            self.star_status.setText(tr('请选择存在的分层 result.json。'))
            return None
        if not output:
            self.star_status.setText(tr('请先选择结果文件夹。'))
            return None
        if evidence and not mask:
            self.star_status.setText(tr('银河区域确认需要同时选择它对应的蒙版。'))
            return None
        if any(path and not Path(path).is_file() for path in (mask, evidence)):
            self.star_status.setText(tr('找不到银河蒙版或确认文件，请重新选择。'))
            return None
        return separation, mask, evidence, output

    @staticmethod
    def _checked_star_suggestion(result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict) or result.get("state") != "succeeded":
            raise ValueError(str(result.get("error") or tr('星点密度分析未完成。')) if isinstance(result, dict)
                             else tr('星点密度分析未返回有效记录。'))
        recommendation = result.get("recommendation")
        if not isinstance(recommendation, dict) or type(recommendation.get("available")) is not bool:
            raise ValueError(tr('密度分析未返回有效建议。'))
        if recommendation["available"]:
            for key, minimum in (("background_strength", 0), ("milkyway_offset", -1)):
                value = recommendation.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= 1:
                    raise ValueError(tr('密度建议参数超出支持范围。'))
        return dict(recommendation)

    def analyze_star_density(self) -> None:
        selected = self._star_operation_inputs()
        if selected is None:
            return
        separation, mask, evidence, output = selected
        self._busy = True
        self._star_suggestion = None
        self._star_suggestion_inputs = None
        self.star_status.setText(tr('正在分析星点密度…完成后可选择采用建议，当前参数保持。'))
        self._update_controls()

        def analyze() -> dict[str, Any]:
            from starstack.star_reduction import suggest_star_reduction
            return suggest_star_reduction(separation, output, milkyway_mask=mask or None,
                                          milkyway_evidence=evidence or None)

        def analyzed(result: dict[str, Any]) -> None:
            recommendation = self._checked_star_suggestion(result)
            self._busy = False
            if self._star_input_values() != selected[:3]:
                self.star_status.setText(tr('分析期间输入已改变，旧建议未采用；请重新分析。'))
                self._update_controls()
                return
            self._star_suggestion = recommendation
            self._star_suggestion_inputs = selected[:3]
            reason = translate_message(recommendation.get("reason") or "")
            if recommendation["available"]:
                bg = recommendation["background_strength"] * 100
                shift = recommendation["milkyway_offset"] * 100
                text = tr('建议背景 {0:.1f}%、银河偏移 {1:+.1f} 个百分点。\n{2}\n尚未采用。', bg, shift, reason)
            else:
                text = tr('本次未给出可靠强度建议，请自行调整。\n') + reason
            self.star_status.setText(text)
            self._update_controls()

        self._run_async("star-analysis", analyze, analyzed)

    def apply_star_suggestion(self) -> None:
        if self._busy or self._closing_requested or self.job_state in ACTIVE_STATES:
            return
        recommendation = self._star_suggestion
        if not recommendation or not recommendation.get("available") or self._star_suggestion_inputs != self._star_input_values():
            return
        self.star_background_strength.setValue(recommendation["background_strength"] * 100)
        self.star_milkyway_offset.setValue(recommendation["milkyway_offset"] * 100)
        self.star_status.setText(tr('建议已采用；可继续修改，再生成成片。\n') + translate_message(recommendation.get("reason") or ""))

    def render_star_reduction(self) -> None:
        selected = self._star_operation_inputs()
        if selected is None:
            return
        separation, mask, evidence, output = selected
        background = self.star_background_strength.value() / 100
        offset = self.star_milkyway_offset.value() / 100
        quality = self.jpeg_quality()
        self._busy = True
        self.star_status.setText(tr('正在生成星点整理成片…原图与原分层保持。'))
        self._update_controls()

        def render() -> dict[str, Any]:
            from starstack.star_reduction import reduce_stars
            return reduce_stars(separation, output, background_strength=background, milkyway_offset=offset,
                                milkyway_mask=mask or None, milkyway_evidence=evidence or None,
                                jpeg_quality=quality)

        def rendered(result: dict[str, Any]) -> None:
            if result.get("state") != "succeeded":
                raise ValueError(str(result.get("error") or tr('未完成成片生成。')))
            directory = Path(str(result.get("processing_directory") or ""))
            preview = (result.get("artifacts") or {}).get("star_reduced.jpg") or {}
            preview_path = Path(str(preview.get("path") or ""))
            if not result.get("processing_directory") or not directory.is_dir() or not preview_path.is_file():
                raise ValueError(tr('结果未返回可读取的成片与保存目录。'))
            self._star_result = dict(result, gui_background_strength=background, gui_milkyway_offset=offset)
            self._busy = False
            notice = tr('成片已另存。可在预览中查看，或打开目录取得 JPEG、16 位 TIFF 与处理记录。')
            limited = int(result.get("full_removal_limited_pixels") or 0)
            if limited > 0:
                total = int(result.get("full_removal_requested_interior_pixels") or 0)
                notice += tr('\n完全去星区域中 {0:,} / {1:,} 像素为保留亮迹而限制去星量，未完全达到无星背景。', limited, total)
            self.star_status.setText(notice)
            self.show_star_result()

        self._run_async("star-render", render, rendered)

    def show_star_result(self) -> None:
        if not self._star_result or self._busy:
            return
        result = self._star_result
        preview = result["artifacts"]["star_reduced.jpg"]["path"]
        self._view_mode = "stars"
        self.stage_label.setText(tr('星点整理成片'))
        background = float(result["gui_background_strength"]) * 100
        offset = float(result["gui_milkyway_offset"]) * 100
        gate = result.get("milkyway_gate") or {}
        summary = (tr('银河中心 {0:.1f}%', max(0., min(100., background + offset)))
                   if gate.get("enabled") else tr('银河未确认：使用背景强度'))
        self.preview_note.setText(tr('星点整理 · 背景 {0:.1f}% · {1}\n成片另存；原图与缓存中的分层结果保持不变。', background, summary))
        self.preview_note.setToolTip(str(result.get("separation_result") or ""))
        self._load_preview_image(Path(preview), auto_transform=True)
        self._update_controls()

    def open_star_result(self) -> None:
        directory = str((self._star_result or {}).get("processing_directory") or "")
        if directory and Path(directory).is_dir():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(directory)):
                self.star_status.setText(tr('无法自动打开目录，可复制此路径手动打开：\n') + directory)

    def _path_field(self, layout: QVBoxLayout, title: str, placeholder: str, name: str, callback: Callable[[], None]) -> QLineEdit:
        layout.addWidget(self._label(title, "field"))
        line = QLineEdit()
        line.setObjectName(name)
        line.setPlaceholderText(placeholder)
        line.setClearButtonEnabled(True)
        line.textChanged.connect(line.setToolTip)
        browse = QPushButton(tr('选择'))
        self._path_buttons.append(browse)
        browse.setAccessibleName(tr('选择') + title)
        browse.clicked.connect(callback)
        row = QHBoxLayout()
        row.setSpacing(7)
        row.addWidget(line, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        return line

    def _pick_input(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr('选择星空照片（可多选）'), self.input_dir.text() or str(Path.home()),
            tr('星空素材 (*.nef *.cr2 *.cr3 *.arw *.dng *.raf *.rw2 *.orf *.pef *.srw *.raw *.fit *.fits *.fts);;所有文件 (*)'))
        if paths:
            self.add_input_files(paths)

    def add_input_files(self, paths: list[str]) -> None:
        if self._busy or self.job_state in ACTIVE_STATES or self._closing_requested:
            return
        existing = {os.path.normcase(str(Path(self._import_originals.get(p, p)).resolve())) for p in self.input_files}
        added = []
        for item in paths:
            path = Path(item).expanduser().resolve()
            key = os.path.normcase(str(path))
            if path.is_file() and path.suffix.lower() in RAW | {".fit", ".fits", ".fts"} and key not in existing:
                added.append(str(path))
                existing.add(key)
        if not added:
            self.message_label.setText(tr('未添加新照片：请选择支持的 RAW / FITS 文件；重复文件只保留一次。'))
            return
        if len(self.input_files) + len(added) > 300:
            self.message_label.setText(tr('每个任务最多选择 300 张照片，请分组处理。'))
            return
        selected = [self._import_originals.get(p, p) for p in self.input_files] + added
        if any(Path(p).suffix.lower() in RAW - {'.dng'} for p in selected):
            self._import_compatible_photos(selected)
        else:
            self.input_files.extend(added)
            self._selection_changed()

    def _pick_dng_converter(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr('选择 Adobe DNG Converter'),
            self.dng_converter_path.text(), tr('程序 (*.exe);;所有文件 (*)'))
        if path:
            self.dng_converter_path.setText(path)

    def _select_stack_backend(self) -> None:
        mode = self.stack_backend_choice.currentData()
        if mode == 'official':
            self.stack_engine_path.clear()
        elif mode == 'finite':
            path = self._finite_backend_info.get('path')
            if path:
                self.stack_engine_path.setText(path)
            else:
                self.stack_engine_path.clear()
                self.finite_backend_note.setText(tr('尚未安装改进构建。请重新运行安装程序并勾选该组件，或选择 Siril 官方构建。'))
        self._update_controls() if hasattr(self, 'menu_actions') else None

    def _sync_stack_backend_choice(self) -> None:
        path = self.stack_engine_path.text().strip()
        mode = ('finite' if path == self._finite_backend_info.get('path') else 'custom') if path else 'official'
        self.stack_backend_choice.blockSignals(True)
        self.stack_backend_choice.setCurrentIndex(self.stack_backend_choice.findData(mode))
        self.stack_backend_choice.blockSignals(False)

    def _set_finite_backend(self, result) -> None:
        self._finite_backend_info = dict(result)
        self.finite_sources_button.setVisible(bool(result.get('distribution_path')))
        if result.get('available'):
            self.finite_backend_note.setText(tr('Aester 改进构建已就绪；选择后用于新作业。'))
            self.finite_backend_note.setToolTip(result['path'])
            if self.stack_backend_choice.currentData() == 'finite':
                self.stack_engine_path.setText(result['path'])
        else:
            self.finite_backend_note.setText(tr('改进构建未安装。可重新运行安装程序勾选“Aester 改进构建”；官方 Siril 可继续使用。'))
            self.finite_backend_note.setToolTip(translate_message(result.get('reason') or ''))

    def _open_finite_sources(self) -> None:
        path = self._finite_backend_info.get('distribution_path')
        if path and Path(path).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path))))

    def _install_finite_backend(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES or self._closing_requested:
            return
        source = self._finite_backend_info.get('install_source')
        if not source:
            source = QFileDialog.getExistingDirectory(self, tr('选择 Aester 改进构建目录'))
        if not source:
            return
        from starstack.components import install_finite_backend
        self._busy = True
        self._update_controls()
        self.finite_backend_note.setText(tr('正在核验并安装已有构建…'))
        def installed(result):
            self._busy = False
            self._set_finite_backend(result)
            self._update_controls()
        self._run_async('install-finite', lambda: install_finite_backend(source=source), installed)

    def _open_dng_download(self) -> None:
        from starstack.raw_import import ADOBE_DOWNLOAD_URL
        QDesktopServices.openUrl(QUrl(ADOBE_DOWNLOAD_URL))

    def _import_compatible_photos(self, selected: list[str]) -> None:
        from starstack.raw_import import prepare_import
        try:
            cache = self._processing_root()
        except ValueError as error:
            self.message_label.setText(translate_message(str(error)))
            return
        self._busy = True
        self._import_cancel = threading.Event()
        cancelled = self._import_cancel
        converter = self.dng_converter_path.text().strip() or None
        self.message_label.setText(tr('正在检查 RAW 兼容性；需要时转换为 DNG 工作副本，原片保留。'))
        self._update_controls()
        def imported(result):
            self._busy = False
            self._import_cancel = None
            self.input_files = list(result['input_files'])
            self._import_originals = dict(zip(result['input_files'], result['originals']))
            self._selection_changed()
            if result['converted']:
                self.message_label.setText(tr('已导入 DNG 兼容副本；原 RAW 保持原位，来源记录保存在缓存。'))
        self._run_async('import-raw', lambda: prepare_import(selected, cache,
            converter_path=converter, cancelled=cancelled.is_set), imported)

    def remove_selected_files(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES or self._closing_requested:
            return
        indices = {self.file_strip.row(item) for item in self.file_strip.selectedItems()}
        self.input_files = [p for i, p in enumerate(self.input_files) if i not in indices]
        self._selection_changed()

    def clear_input_files(self) -> None:
        """Clear this selection, never the files or an active job."""
        if self._busy or self.job_state in ACTIVE_STATES or self._closing_requested:
            return
        if not self.input_files:
            return
        self.input_files.clear()
        self._import_originals.clear()
        self._color_timer.stop()
        self._color_finish_timer.stop()
        self._selection_changed()
        self.preview_note.setText(tr('素材列表已清空，原片仍保留。可添加另一组照片。'))
        self.message_label.setText(tr('素材列表已清空，原片仍保留。可添加另一组照片。'))

    def _selection_changed(self, *, preview: bool = True) -> None:
        first = (os.path.normcase(str(Path(self._import_originals.get(self.input_files[0], self.input_files[0])).resolve()))
                 if self.input_files else '')
        if preview and first != self._selected_reference_source:
            self._image_rotation = None
            self._orientation_notice = False
        self._selected_reference_source = first
        if preview:
            self._job_detached = True
            self._reference_failed = False
            self._forget_workflow_jpeg_quality()
        self._color_finish_timer.stop()
        self._reference_quality = 'standard'
        self.file_strip.clear()
        for index, path in enumerate(self.input_files):
            item = QListWidgetItem((tr('首帧 · ') if index == 0 else f"{index + 1:02d} · ") + Path(path).name)
            original = self._import_originals.get(path, path)
            item.setToolTip(original if original == path else tr('原片：{0}\n兼容副本：{1}', original, path))
            self.file_strip.addItem(item)
        self.selection_label.setText(tr('已选 {0} 张照片', len(self.input_files)) if self.input_files else tr('添加一组连拍照片'))
        self.frame_dots.set_count(len(self.input_files), 0)
        self.frame_dots.setAccessibleName(tr('{0} 张照片；第一张是颜色参考', len(self.input_files))
                                          if self.input_files else tr('尚未添加照片'))
        self.input_dir.setText(str(Path(self.input_files[0]).parent) if self.input_files else "")
        if preview:
            self._regions_project = None
            self._regions_path = None
            self.milkyway_enabled.setChecked(False)
            self._reference_cache_epoch += 1
            self._reference_generation += 1
            self._reference_data = None
            self._reference_scene = ()
            self._preview_serial += 1
            self.sky_mask.clear()
            self.scene_mode.setCurrentIndex(1)
            if self.input_files and Path(self.input_files[0]).suffix.lower() not in RAW:
                self._restoring_project = True
                self.reset_color_adjustments()
                self._restoring_project = False
            self._reference_pending = bool(self.input_files)
            if self.input_files:
                self.preview.show_message(tr('正在准备首帧校色预览…'))
                self._color_timer.start(1)
            else:
                self.preview.show_message(tr('从一组星空开始\n\n添加或拖入照片，首帧将自动预览。'), art=True)
        self._update_controls()

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls() and not self._busy and self.job_state not in ACTIVE_STATES:
            if any(url.isLocalFile() for url in event.mimeData().urls()):
                event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:
        self.add_input_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, tr('选择结果文件夹'), self.output_dir.text() or str(WORKSPACE))
        if path:
            self.output_dir.setText(path)

    def _pick_siril(self) -> None:
        file_filter = tr('Siril 程序 (*.exe);;所有文件 (*)') if os.name == "nt" else tr('所有文件 (*)')
        path, _ = QFileDialog.getOpenFileName(self, tr('选择 Siril 命令行程序'), self.siril_path.text() or str(WORKSPACE), file_filter)
        if path:
            self.siril_path.setText(path)
            self.check_backend()
            self._invalidate_reference_backend()

    def _pick_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr('选择天空蒙版：白天空、黑地景'), self.input_dir.text() or str(WORKSPACE), tr('PNG 图像 (*.png)'))
        if path:
            self._regions_project = None
            self._regions_path = None
            self.milkyway_enabled.setChecked(False)
            self.sky_mask.setText(path)

    def inspect_single_raw(self) -> None:
        if self._busy or self._closing_requested or self.job_state in ACTIVE_STATES:
            return
        output_path = self.output_dir.text().strip()
        if not output_path:
            self.raw_audit_status.setText(tr('请先选择检查报告的结果文件夹。'))
            return
        extensions = " ".join(f"*{suffix}" for suffix in sorted(RAW))
        source, _ = QFileDialog.getOpenFileName(
            self, tr('选择一张 RAW 进行检查'), self.input_dir.text() or str(WORKSPACE),
            tr('RAW / CFA DNG ({0});;所有文件 (*)', extensions),
        )
        if not source:
            return
        self._busy = True
        self._raw_audit_dir = None
        self.raw_audit_status.setToolTip("")
        self.raw_audit_status.setText(tr('正在检查单张 RAW 并保存解码证据…'))
        self.message_label.setText(tr('正在检查所选单张 RAW，当前任务与预览保留。'))
        self._update_controls()

        def inspect() -> dict[str, Any]:
            from starstack.raw_audit import inspect_raw
            return inspect_raw(source, output_path)

        def inspected(result: dict[str, Any]) -> None:
            directory = Path(str(result.get("audit_dir") or ""))
            report = directory / "audit.json"
            if not result.get("audit_dir") or not report.is_file():
                raise ValueError(tr('检查未返回可读取的报告文件。'))
            self._raw_audit_dir = str(directory)
            self._busy = False
            if result.get("state") == "succeeded":
                has_derivative = bool((result.get("artifacts") or {}).get("black_subtracted_cfa"))
                summary = (tr('检查完成，已保存解码证据；黑位衍生可用。') if has_derivative
                           else tr('检查完成，仅保存解码采样；黑位衍生暂不可用。'))
                detail = tr('{0}\n保存目录：{1}', summary, directory)
            elif result.get("state") == "failed":
                summary = tr('单张 RAW 检查未完成：') + translate_message(result.get("error") or tr('请查看失败报告。'))
                detail = tr('{0}\n失败报告：{1}', summary, report)
            else:
                raise ValueError(tr('检查返回的状态不完整，请查看报告：') + str(report))
            self.raw_audit_status.setText(detail)
            self.raw_audit_status.setToolTip(str(report))
            self.message_label.setText(summary)
            self._update_controls()

        self._run_async("raw-audit", inspect, inspected)

    def open_raw_audit(self) -> None:
        if self._raw_audit_dir and Path(self._raw_audit_dir).is_dir():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(self._raw_audit_dir)):
                self.raw_audit_status.setText(tr('无法自动打开检查目录，可复制此路径手动打开：\n') + self._raw_audit_dir)

    def _pick_job_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr('打开已有任务'), self.output_dir.text() or str(WORKSPACE / "outputs"),
            tr('任务记录 (workflow.json job.json);;JSON 文件 (*.json)'),
        )
        if path:
            self.load_existing_job(path)

    def _pick_job_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, tr('选择包含 job.json 的任务文件夹'), self.output_dir.text() or str(WORKSPACE / "outputs"),
        )
        if path:
            self.load_existing_job(path)

    def load_existing_job(self, path: str) -> None:
        """Attach the GUI to a persisted task without resubmitting it."""
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        selected = Path(path).expanduser().resolve()
        if selected.is_file():
            if selected.name.lower() not in {"job.json", "workflow.json"}:
                self.message_label.setText(tr('请选择任务目录中的 workflow.json 或 job.json。'))
                return
            selected = selected.parent
        if (selected / "workflow.json").is_file():
            from starstack.workflow import workflow_status
            self._busy = True
            self.message_label.setText(tr('正在读取完整出图任务…'))
            self._update_controls()
            self._run_async("load-workflow", lambda: workflow_status(str(selected)), self._workflow_submitted)
            return
        if not (selected / "job.json").is_file():
            self.message_label.setText(tr('该文件夹中没有 job.json，请选择实际任务目录。'))
            return
        self._busy = True
        self.message_label.setText(tr('正在读取已有任务…'))
        self._update_controls()

        def loaded(job: dict[str, Any]) -> None:
            job = dict(job)
            job["job_dir"] = str(selected)
            self._restoring_project = True
            self._image_rotation = int(job.get('image_rotation', 0))
            self._orientation_notice = False
            self.input_files = []
            self._selection_changed(preview=False)
            self._reference_cache_epoch += 1
            self._reference_generation += 1
            self._reference_pending = False
            self._reference_data = None
            self._reference_scene = ()
            self._color_timer.stop()
            self._regions_project = None
            self._regions_path = None
            self.milkyway_enabled.setChecked(False)
            self.scene_mode.setCurrentIndex(0)
            self.reset_color_adjustments()
            self.trails_enabled.setChecked(False)
            self.stars_enabled.setChecked(False)
            self._restoring_project = False
            inputs = job.get("inputs") or []
            if inputs and isinstance(inputs[0], dict) and inputs[0].get("path"):
                self.input_dir.setText(str(Path(inputs[0]["path"]).parent))
                self.input_dir.setCursorPosition(len(self.input_dir.text()))
                self.input_dir.deselect()
            self.output_dir.setText(str(selected.parent))
            self.output_dir.setCursorPosition(len(self.output_dir.text()))
            self.output_dir.deselect()
            mask = job.get("mask") or {}
            self.sky_mask.setText(str(mask.get("path") or "") if isinstance(mask, dict) else "")
            self.sky_mask.setCursorPosition(len(self.sky_mask.text()))
            self.sky_mask.deselect()
            engine = job.get("engine") or {}
            if isinstance(engine, dict):
                self.siril_path.setText(str(engine.get("executable") or ""))
                self.siril_path.setCursorPosition(len(self.siril_path.text()))
                self.siril_path.deselect()
            self._submitted(job)
            self.check_backend()

        self._run_async("load", lambda: load_job(str(selected)), loaded)

    def _update_mask_notice(self) -> None:
        if not hasattr(self, "mask_notice"):
            return
        mode = self._scene_from_current_marks()
        if mode == 'landscape':
            self.mask_notice.setText(tr('已载入分区 · 分别对齐天空与地景。'))
        else:
            self.mask_notice.setText(tr('尚未画地景，当前按全天空处理；照片含地景时请先标记边界。'))
        self._update_controls() if hasattr(self, "menu_actions") else None

    def _scene_from_current_marks(self) -> str:
        if self._regions_project:
            from starstack.regions import scene_from_marks
            if (not self._regions_project.get('operations') and self.sky_mask.text().strip()
                    and self.scene_mode.currentData() == 'landscape'
                    and self._regions_project.get('scene_mode') == 'landscape'):
                return 'landscape'  # Saved legacy external PNG, without a drawn project.
            return scene_from_marks(self._regions_project)
        # An explicitly loaded old PNG remains a supported landscape scope.
        return 'landscape' if self.sky_mask.text().strip() and not self._regions_project else 'sky'

    def open_trail_review(self) -> None:
        if self._busy or self.job_state != "succeeded" or not self.workflow_dir or not self.job_dir:
            return
        if (self._workflow or {}).get("cache_cleaned"):
            self.message_label.setText(tr('此任务的中间文件已清理，不能再复核亮迹候选；作品不受影响。'))
            return
        from starstack.trail_review_session import open_review
        directory, job_dir = self.workflow_dir, self.job_dir
        self._busy = True
        self.message_label.setText(tr('正在核对亮迹候选…'))
        self._update_controls()

        def opened(session):
            from aester_desktop.trail_review_dialog import TrailReviewDialog
            self._busy = False
            self._update_controls()
            if self.workflow_dir != directory:
                return
            dialog = TrailReviewDialog(session, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.message_label.setText(tr('亮迹复核已关闭；已保存草稿可继续编辑。'))
                return
            from starstack.workflow_edit import apply_review
            self._busy = True
            self.message_label.setText(tr('正在按整批选择生成亮迹版本；原成片保留…'))
            self._update_controls()
            self._run_async("apply-trail-review", lambda: apply_review(directory,
                dialog.apply_session_path, expected_draft_revision=dialog.apply_draft_revision), self._review_published)

        self._run_async("open-trail-review", lambda: open_review(job_dir), opened)

    def _review_published(self, result: dict[str, Any]) -> None:
        """After a review batch: refresh the results only (no resubmission, sliders untouched)."""
        self._busy = False
        workflow = result.get("workflow")
        if isinstance(workflow, dict):
            self._view_mode = "results"
            self._rendered_preview = None
            self._artifact_signature = ()        # repopulate the tabs: the photograph may be new
            self._apply_workflow_record(workflow)
        if "published" in result:                # contract C
            if result.get("published"):
                folder = str(result.get("delivery_dir") or (self._workflow or {}).get("delivery_dir") or "")
                message = tr('含亮迹版本已另存：保留 {0} 条亮迹。\n保存在：{1}', int(result.get("kept_count") or 0), folder)
            else:
                message = tr('没有保留任何亮迹；成片与基础版一致，身份改判已记录')
        else:                                   # older workflow_edit without the summary fields
            message = (tr('亮迹复核版本已另存。请检查保留候选及恢复范围。') if result.get("delivery")
                       else tr('没有新的处置选择；基础成片保持，候选仍可继续复核。'))
        self.message_label.setText(message)
        self._update_controls()

    def edit_regions(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        reference = self._reference_data
        if not reference or not self._orientation_ready():
            self.prepare_reference()
            self.message_label.setText(tr('正在准备首帧预览；完成后再次点击标记画面。'))
            return
        from starstack import regions
        from aester_desktop.region_dialog import RegionDialog
        files, current = list(self.input_files), self._regions_project
        try:
            work_root = self._processing_root()
        except ValueError as error:
            self.message_label.setText(tr('操作未完成：') + translate_message(str(error))); return
        self._busy = True
        self._update_controls()
        def prepared(project):
            self._busy = False
            dialog = RegionDialog(reference, project, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self._update_controls()
                return
            selected = dialog.project
            self._busy = True
            self._update_controls()
            def saved(result):
                self._busy = False
                self._regions_project = selected
                self._regions_saved_by_user = selected
                self._regions_path = result['project_path']
                self.sky_mask.setText(result['sky_mask_path'])
                self.scene_mode.setCurrentIndex(self.scene_mode.findData(selected['scene_mode']))
                if selected['milkyway']['status'] != 'present':
                    self.milkyway_enabled.setChecked(False)
                self.message_label.setText(tr('分区已保存。原片未改动；颜色调整可继续沿用这些范围。'))
                self._update_controls()
            self._run_async('save-regions', lambda: regions.build_masks(selected, work_root, input_files=files), saved)
        self._run_async('prepare-regions', lambda: regions.validate_project(current, input_files=files) if current else
                        regions.create_project(reference, files), prepared)

    def save_editing_project(self, path: str | None = None) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        if not self._reference_data or not self._orientation_ready():
            self.message_label.setText(tr('请等待首帧预览完成后保存项目。'))
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, tr('保存编辑项目'), str(Path(self.output_dir.text()) / tr('我的星空.aester.json')), tr('Aester 项目 (*.aester.json)'))
        if not path:
            return
        from starstack import regions
        from starstack.editing_project import save_project
        reference, files = dict(self._reference_data), list(self.input_files)
        state = self._editing_state()
        current = self._regions_project
        def save():
            project = current or regions.create_project(reference, files)
            if state['scene_mode'] == 'sky':
                project = regions.revise(project, scene_mode='sky')
            return save_project(path, state, reference, project)
        self._busy = True
        self._update_controls()
        def saved(result):
            self._busy = False
            self.message_label.setText(tr('编辑项目已保存：') + result['project_path'])
            self._update_controls()
        self._run_async('save-editing-project', save, saved)

    def _editing_state(self) -> dict:
        return {'input_files': list(self.input_files), 'color_adjustments': self.color_adjustments(),
                'image_rotation': self._effective_image_rotation(),
                'output_root': self.output_dir.text(), 'cache_root': self._processing_root(), 'siril_path': self.siril_path.text(),
                'stack_engine_path': self.stack_engine_path.text(), 'star_model_path': self.star_model_path.text(),
                'scene_mode': self._scene_from_current_marks(), 'sky_mask_path': self.sky_mask.text(),
                'trails_enabled': self.trails_enabled.isChecked(), 'stars_enabled': self.stars_enabled.isChecked(),
                'milkyway_enabled': self.milkyway_enabled.isChecked(), 'milkyway_strength': self.milkyway_strength.value() / 100,
                'auto_strength': self.workflow_auto_strength.isChecked(), 'background_strength': self.workflow_background_strength.value() / 100,
                'milkyway_offset': self.workflow_milkyway_offset.value() / 100,
                'save_starless': self.workflow_save_starless.isChecked(),
                'save_star_layer': self.workflow_save_star_layer.isChecked(),
                'stack_range': self.stack_range_checkbox.isChecked(),
                'backend_options': {key: widget.currentData() for key, widget in [('channel_order', self.model_channel_order),
                    ('value_range', self.model_range), ('layout', self.model_layout), ('stitch', self.model_stitch)]}}

    def open_editing_project(self, path: str | None = None) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, tr('打开编辑项目'), self.output_dir.text(), tr('Aester 项目 (*.aester.json)'))
        if not path:
            return
        from starstack.editing_project import load_project
        self._busy = True
        self._update_controls()
        self._run_async('open-editing-project', lambda: load_project(path), self._restore_editing_project)

    def _restore_editing_project(self, saved: dict) -> None:
        self._busy = False
        state = saved['state']
        self.timer.stop()
        self.workflow_dir = self.job_dir = None
        self.job_state = ''
        self._workflow = self._job = None
        self._artifact_signature = ()
        self._jpg_artifacts = []
        self._star_result = None
        self._restoring_project = True
        self.input_files = state['input_files']
        self._selection_changed(preview=False)
        self._reference_cache_epoch += 1
        self._reference_generation += 1
        self._regions_project = saved['regions']
        self._regions_path = saved['regions_path']
        self._reference_data = saved['reference']
        self._reference_scene = tuple(self.input_files)
        self._image_rotation = int(state.get('image_rotation', 0))
        self._orientation_notice = False
        if state.get('cache_root'):
            self.cache_dir.setText(state['cache_root'])
        for widget, key in [(self.output_dir, 'output_root'), (self.siril_path, 'siril_path'), (self.sky_mask, 'sky_mask_path'),
                            (self.stack_engine_path, 'stack_engine_path'), (self.star_model_path, 'star_model_path')]:
            widget.setText(state.get(key, ''))
        for widget, key in [(self.trails_enabled, 'trails_enabled'), (self.stars_enabled, 'stars_enabled'),
                            (self.milkyway_enabled, 'milkyway_enabled'), (self.workflow_auto_strength, 'auto_strength'),
                            (self.stack_range_checkbox, 'stack_range'),
                            # Projects saved before these options existed load with both off.
                            (self.workflow_save_starless, 'save_starless'),
                            (self.workflow_save_star_layer, 'save_star_layer')]:
            widget.setChecked(state.get(key, False) is True)
        colors = state['color_adjustments']
        self.warmth_slider.setValue(round(colors['warmth']))
        self.tint_slider.setValue(round(colors['tint']))
        self.exposure_slider.setValue(round(colors['exposure_ev'] * 10))
        self.milkyway_strength.setValue(state['milkyway_strength'] * 100)
        self.workflow_background_strength.setValue(state['background_strength'] * 100)
        self.workflow_milkyway_offset.setValue(state.get('milkyway_offset', 0) * 100)
        for widget, key in [(self.model_channel_order, 'channel_order'), (self.model_range, 'value_range'),
                            (self.model_layout, 'layout'), (self.model_stitch, 'stitch')]:
            widget.setCurrentIndex(widget.findData(state['backend_options'][key]))
        self.scene_mode.setCurrentIndex(self.scene_mode.findData(state['scene_mode']))
        self._sync_stack_backend_choice()
        self._restoring_project = False
        self._reference_pending = True
        self._color_timer.start(1)
        self.message_label.setText(tr('项目已打开；分区、颜色和效果已恢复。'))
        self._update_controls()

    def prepare_reference(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        self._reference_generation += 1
        self._reference_pending = True
        self._launch_reference_preview()

    def _effective_image_rotation(self) -> int:
        return int(self._image_rotation if self._image_rotation is not None else
                   (self._reference_data or {}).get('image_rotation', 0))

    def _orientation_ready(self) -> bool:
        return bool(self._reference_data and self._reference_scene == tuple(self.input_files)
                    and int(self._reference_data.get('image_rotation', 0)) == self._effective_image_rotation())

    def rotate_input_images(self, clockwise_degrees: int) -> None:
        if clockwise_degrees not in (-90, 90):
            raise ValueError('Rotation must be a clockwise or anticlockwise quarter turn')
        self._set_input_orientation((self._effective_image_rotation() + clockwise_degrees) % 360)

    def restore_capture_orientation(self) -> None:
        reference = self._reference_data or {}
        capture_rotation = reference.get('capture_rotation')
        # Old references predate recorded orientation; reread only in that case.
        self._set_input_orientation(int(capture_rotation) if capture_rotation is not None else None)

    def _set_input_orientation(self, rotation: int | None) -> None:
        if (self._busy or self._closing_requested or self.job_state in ACTIVE_STATES
                or self._reference_running or self._reference_pending or not self.input_files
                or not self._orientation_ready()):
            return
        if rotation is not None and rotation == self._effective_image_rotation():
            return
        self._image_rotation = rotation
        self._regions_project = None
        self._regions_path = None
        self.milkyway_enabled.setChecked(False)
        self.sky_mask.clear()
        self.scene_mode.setCurrentIndex(1)
        self._reference_cache_epoch += 1
        self._reference_generation += 1
        self._preview_serial += 1
        self._orientation_notice = True
        if rotation is None:
            self._reference_data = None
            self._reference_scene = ()
        self._reference_pending = True
        self._reference_quality = 'standard'
        self._color_finish_timer.stop()
        self._color_timer.stop()
        self.preview.show_message(tr('正在调整整组方向…'))
        self.preview_note.setText(tr('方向调整将用于整组照片和最终成片；请在更新后重新标记地景与银河。'))
        self.message_label.setText(tr('正在调整整组方向；旧分区已清除，原片未改动。'))
        self._update_controls()
        self._launch_reference_preview()

    def _launch_reference_preview(self) -> None:
        if self._reference_running or not self._reference_pending or self._closing_requested:
            return
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        files = tuple(self.input_files)
        input_path = self.input_dir.text().strip()
        if not files and not Path(input_path).is_dir():
            return
        try:
            output_path = self._processing_root()
        except ValueError as error:
            self.message_label.setText(tr('操作未完成：') + translate_message(str(error)))
            return
        generation = self._reference_generation
        cache_epoch = self._reference_cache_epoch
        adjustments = self.color_adjustments()
        quality = self._reference_quality
        image_rotation = self._image_rotation
        siril = self.siril_path.text().strip() or None
        current = self._reference_data if self._reference_scene == files else None
        reference_dir = str((current or {}).get("reference_dir") or "")
        # Legacy directory-only requests retain their full-frame bridge.
        legacy = not files
        self._reference_pending = False
        self._reference_running = True
        self._reference_failed = False
        self._update_controls()
        name = f"reference-build:{generation}"
        self.message_label.setText(tr('正在更新首帧颜色…') if current else tr('正在读取首帧并校准颜色…'))
        def prepare() -> dict[str, Any]:
            from starstack import references
            if current and not legacy:
                if image_rotation is not None and int(current.get('image_rotation', 0)) != image_rotation:
                    rotated = references.rotate_reference(reference_dir, image_rotation=image_rotation)
                    return {**rotated, **references.render_reference(rotated['reference_dir'],
                        color_adjustments=adjustments, quality=quality)}
                return references.render_reference(reference_dir, color_adjustments=adjustments, quality=quality)
            if legacy:
                return references.prepare_reference(input_path, output_path, siril_path=siril,
                    **({'image_rotation': image_rotation} if image_rotation is not None else {}))
            return references.prepare_reference(input_path, output_path, siril_path=siril,
                                                 input_files=list(files), color_adjustments=adjustments,
                                                 **({'image_rotation': image_rotation} if image_rotation is not None else {}))
        def prepared(result: dict[str, Any]) -> None:
            self._reference_running = False
            reference = {**(current or {}), **dict(result)}
            if not Path(str(reference.get("preview_path", ""))).is_file():
                raise ValueError(tr('首帧预览文件不存在，请重试。'))
            accepted = (tuple(self.input_files) == files and cache_epoch == self._reference_cache_epoch
                        and image_rotation == self._image_rotation)
            if accepted:
                self._reference_data = reference
                self._reference_scene = files
                self._image_rotation = int(reference.get('image_rotation', 0))
            if accepted and generation == self._reference_generation:
                self.show_reference()
                self.message_label.setText(tr('整组方向已更新；旧分区已清除，请重新标记地景与银河。') if self._orientation_notice
                    else tr('首帧预览已更新；有地景或银河时先标记，再开始合成。'))
                self._orientation_notice = False
            self._update_controls()
            if self._reference_pending:
                self._color_timer.start(1)
        self._run_async(name, prepare, prepared)

    def show_reference(self) -> None:
        if not self._reference_data:
            return
        reference = self._reference_data
        self._view_mode = "reference"
        if self.job_dir:
            self.stage_label.setText(tr('原任务 · ') + tr(STATE_LABELS.get(self.job_state, '读取状态')))
        width, height = int(reference["width"]), int(reference["height"])
        source = Path(str(reference.get("source_path") or tr('首帧'))).name
        rendering = (reference.get("metadata") or {}).get("rendering") or {}
        label = tr('已校色首帧') if rendering.get("color_calibrated") else tr('首帧预览 · 校色未确认')
        self.preview_note.setText(f"{label} · {width} × {height} · {source}")
        self.preview_note.setToolTip(str(reference.get("source_path") or ""))
        self._load_preview_image(Path(str(reference["preview_path"])),
                                 auto_transform=False, expected_size=QSize(int(reference.get("preview_width", width)), int(reference.get("preview_height", height))))
        self._update_controls()

    def show_results(self) -> None:
        self._view_mode = "results"
        if self.job_dir:
            self.stage_label.setText(tr(STATE_LABELS.get(self.job_state, '读取状态')))
        self.preview_note.setToolTip("")
        self._set_result_note()
        if self._jpg_artifacts:
            index = max(0, min(self.style_tabs.currentIndex(), len(self._jpg_artifacts) - 1))
            self.select_style(index)
        else:
            self._preview_serial += 1
            self.preview.show_message(tr('本次尚无可预览成片\n\n完成处理后可在这里查看成片预览。'))
        self._update_controls()

    def open_reference(self) -> None:
        if not self._reference_data:
            return
        directory = Path(str(self._reference_data.get("reference_dir") or ""))
        if directory.is_dir() and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            self.message_label.setText(tr('无法自动打开参考目录：') + str(directory))

    def reset_preview_colors(self) -> None:
        self.red_gain.setValue(1.0)
        self.blue_gain.setValue(1.0)
        self.exposure_ev.setValue(0.0)
        self.message_label.setText(tr('预览参数已重置；点击“重新生成预览”后应用。'))

    def render_preview(self) -> None:
        if self._busy or self.job_state != "succeeded" or not self.job_dir:
            return
        job_dir = self.job_dir
        red, blue, exposure = self.red_gain.value(), self.blue_gain.value(), self.exposure_ev.value()
        self._busy = True
        self.message_label.setText(tr('正在从现有母版生成新预览…'))
        self._update_controls()

        def render() -> dict[str, Any]:
            from starstack.preview import render_job_preview
            return render_job_preview(job_dir, red_gain=red, blue_gain=blue, exposure_ev=exposure)

        def rendered(result: dict[str, Any]) -> None:
            self._busy = False
            if self.job_dir != job_dir:
                self._update_controls()
                return
            preview = dict(result)
            jpgs = [dict(item) for item in preview.get("artifacts", []) if item.get("kind") == "jpeg"]
            if not 1 <= len(jpgs) <= 3:
                raise ValueError(tr('新预览未返回可显示的 JPG。原任务结果仍保留。'))
            directory = Path(str(preview.get("directory") or ""))
            if not preview.get("directory") or not directory.is_dir():
                raise ValueError(tr('新预览目录不存在。'))
            for item in jpgs:
                path = Path(str(item.get("path") or ""))
                if not path.is_absolute():
                    path = directory / path
                if not path.is_file():
                    raise ValueError(tr('新预览中的 JPG 文件缺失。'))
                item["path"] = str(path)
            preview.update(job_dir=job_dir, jpgs=jpgs, gains=(red, blue, exposure))
            self._rendered_preview = preview
            self._jpg_artifacts = jpgs
            self._artifact_signature = tuple((str(a["path"]), str(a.get("label"))) for a in jpgs)
            self._view_mode = "results"
            self._populate_styles()
            self._set_result_note()
            self.message_label.setText(tr('新预览已另存。原任务母版保持不变，可打开预览目录查看 JPG、TIFF 和配方。'))
            self._update_controls()

        self._run_async("render-preview", render, rendered)

    def _set_result_note(self) -> None:
        if self._rendered_preview and self._rendered_preview.get("job_dir") == self.job_dir:
            red, blue, exposure = self._rendered_preview["gains"]
            renderer = str(self._rendered_preview.get("renderer") or "")
            self.preview_note.setText(
                tr('新预览 · 红 {0:.2f} × · 蓝 {1:.2f} × · 曝光 {2:+.2f} EV', red, blue, exposure))
            self.preview_note.setToolTip(str(self._rendered_preview.get("recipe_path") or ""))
        else:
            self.preview_note.setText(self._products_note())
            self.preview_note.setToolTip("")

    def _trails_note(self) -> str:
        """'含亮迹版本 · 已加回 N 条（算法 a / 人工 h / Agent g）· 另有 M 条待核实' (contract B), or ''."""
        record = self._workflow or {}
        if not self._has_trails_version(record):
            return ''
        review = record.get("trail_review") or {}
        parts = [tr('含亮迹版本')]
        kept = review.get("kept_count")
        if isinstance(kept, int) and not isinstance(kept, bool):
            by = review.get("kept_by") or {}
            parts.append(tr('已加回 {0} 条（算法 {1} / 人工 {2} / Agent {3}）', kept, int(by.get("algorithm") or 0),
                            int(by.get("human") or 0), int(by.get("agent") or 0)))
        pending = review.get("pending_count")
        if isinstance(pending, int) and not isinstance(pending, bool) and pending > 0:
            parts.append(tr('另有 {0} 条待核实', pending))
        elif pending is None and record.get("needs_review"):
            parts.append(tr('仍有候选待核实'))
        return " · ".join(parts)

    def _extras_note(self) -> str:
        """Extra layers saved beside the photograph on request (contract B ``extras``)."""
        roles = {str(item.get("role")) for item in (self._workflow or {}).get("extras") or [] if isinstance(item, dict)}
        names = [name for role, name in (("starless", tr('无星图')), ("stars_add", tr('星点层')),
                                         ("trails_layer", tr('亮迹层')))
                 if role in roles or (role == "stars_add" and "stars_subtract" in roles)]
        return tr(' · 另存：{0}', ' / '.join(names)) if names else ''

    def _products_note(self) -> str:
        """What the result tabs show, by product contract (M/D jobs vs the legacy three-tone display)."""
        if self._workflow and self._workflow.get("state") == "succeeded":
            extras = self._extras_note()
            if self._workflow.get("cache_cleaned") and (self._workflow.get("request") or {}).get("trails_enabled"):
                extras += tr(' · 中间文件已清理，亮迹候选不能再复核')
            trails = self._trails_note()
            if trails:
                return trails + extras
            if self._workflow.get("needs_review"):
                return tr('基础成片与母版已保存 · 亮迹候选待复核，尚未加回') + extras
            if (self._workflow.get("trail_review") or {}).get("state") == "unavailable":
                return tr('基础成片与母版已保存 · 亮迹扫描未完成，尚未加回') + extras
            finishing = self._workflow.get("finishing") or {}
            if finishing.get("state") == "succeeded":
                return tr('成片与后期母版已保存 · 星点整理已运行') + extras
            if finishing.get("state") == "reduction_skipped":
                return tr('成片与后期母版已保存 · 密度建议不足，已保留缩星前成片') + extras
            return tr('成片与后期母版已保存 · 未开启星点整理，已跳过缩星') + extras
        colour = ((self._job or {}).get("diagnostics") or {}).get("color_export") or {}
        if colour.get("state") == "succeeded":
            return tr('成片已保存 · JPEG / 16 位 TIFF / 浮点后期母版')
        if colour.get("state") == "fallback_legacy":
            return tr('母版 / 成片写出失败，当前显示旧色调预览；请查看任务详情。')
        return tr('旧版任务预览 · 母版和 TIFF 见结果目录')

    def _update_controls(self) -> None:
        active = self.job_state in ACTIVE_STATES
        idle = not self._busy and not active and not self._closing_requested
        if hasattr(self, 'cache_dir'):
            self.cache_dir.setEnabled(idle)
        for control in (self.add_files_button, self.file_strip, self.scene_mode, self.trails_enabled,
                        self.stars_enabled, self.warmth_slider, self.tint_slider, self.exposure_slider,
                        self.reset_color_button):
            control.setEnabled(idle)
        color_available = not self.input_files or Path(self.input_files[0]).suffix.lower() in RAW
        for control in (self.warmth_slider, self.tint_slider, self.exposure_slider, self.reset_color_button):
            control.setEnabled(idle and color_available)
        for slider in (self.warmth_slider, self.tint_slider, self.exposure_slider):
            slider.value_field.setEnabled(slider.isEnabled())
        # Frozen into a job at its start: no change while one runs (later trails versions keep the job's value).
        self.jpeg_quality_slider.setEnabled(idle)
        self.jpeg_quality_value.setEnabled(idle)
        self._track_jpeg_estimate_source()
        self.reset_color_button.setToolTip(tr('FITS 当前使用基础显示，白平衡与曝光调整尚未接入。') if not color_available else tr('恢复拍摄时白平衡和零曝光补偿'))
        self.remove_files_button.setEnabled(idle and bool(self.file_strip.selectedItems()))
        self.clear_files_button.setEnabled(idle and bool(self.input_files))
        orientation_ready = self._orientation_ready()
        for control in (self.rotate_left_button, self.rotate_right_button, self.restore_orientation_button):
            control.setEnabled(idle and bool(self.input_files) and orientation_ready
                               and not self._reference_running and not self._reference_pending)
        self.regions_button.setEnabled(idle and bool(self.input_files) and orientation_ready)
        from starstack.regions import milkyway_active
        mw_ready = bool(self._regions_project and milkyway_active(self._regions_project))
        self.milkyway_enabled.setEnabled(idle and mw_ready)
        self.milkyway_strength.setEnabled(idle and mw_ready and self.milkyway_enabled.isChecked())
        self.workflow_milkyway_offset.setEnabled(idle and mw_ready and self.stars_enabled.isChecked())
        can_review = bool(self.workflow_dir and self.job_dir and self.job_state == 'succeeded'
                          and (self._workflow or {}).get('request', {}).get('trails_enabled'))
        # After 设置 → 作品与缓存 removed the frames, review is refused by the backend: say so on the button.
        cache_cleaned = bool((self._workflow or {}).get('cache_cleaned'))
        self.trail_review_button.setEnabled(idle and can_review and not cache_cleaned)
        self.trail_review_button.setVisible(can_review)
        self.trail_review_button.setToolTip(tr('此任务的中间文件已清理，不能再复核亮迹候选；作品不受影响。')
                                            if can_review and cache_cleaned else '')
        if 'save_project' in self.menu_actions:
            self.menu_actions['save_project'].setEnabled(idle and orientation_ready)
            self.menu_actions['open_project'].setEnabled(idle)
            self.menu_actions['import_mask'].setEnabled(idle)
        explicit_stack = bool(self.stack_engine_path.text().strip())
        self.stack_backend_choice.setEnabled(idle)
        self.install_finite_button.setEnabled(idle)
        self.stack_range_checkbox.setEnabled(not self._busy and not active and not self._closing_requested and not explicit_stack)
        self.stack_range_checkbox.setToolTip(tr('由指定数值保真堆栈引擎保留范围。') if explicit_stack else
                                            tr('保留额外高光范围；单通道零值和负值照常堆栈，暗端与普通堆栈相同在0处截断；遇非有限值会停止并提示。'))
        for field in (self.input_dir, self.output_dir, self.siril_path, self.sky_mask,
                      self.stack_engine_path, self.star_model_path, self.dng_converter_path):
            field.setEnabled(not self._busy and not active and not self._closing_requested)
        for button in self._path_buttons:
            button.setEnabled(not self._busy and not active and not self._closing_requested)
        restart_required = getattr(getattr(self, 'experience', None), '_restart_required', False)
        self.start_button.setEnabled(idle and not restart_required and bool(self.input_dir.text().strip()) and bool(self.output_dir.text().strip())
                                     and (not self.input_files or (len(self.input_files) >= 3 and orientation_ready)))
        self.start_button.setToolTip(self._start_blocker(restart_required) if not self.start_button.isEnabled() else '')
        self.cancel_button.setEnabled(self._import_cancel is not None or
                                      (not self._busy and self.job_state in {"queued", "running"}))
        self.resume_button.setEnabled(not self._busy and not active)
        self.resume_button.setVisible(self._can_resume_current_job())
        self.load_button.setEnabled(not self._busy and not active)
        self.demo_button.setEnabled(not self._busy and not active)
        self.raw_audit_button.setEnabled(not self._busy and not active and not self._closing_requested
                                         and bool(self.output_dir.text().strip()))
        self.open_raw_audit_button.setEnabled(not self._busy and not self._closing_requested
                                              and bool(self._raw_audit_dir))
        directory = self._advanced_result_dir if self._view_mode == "advanced" else self.workflow_dir or self.job_dir
        self.open_button.setEnabled(bool(directory) and Path(directory).is_dir())
        ready = not self._busy and not active and bool(self.input_dir.text().strip()) and bool(self.output_dir.text().strip())
        self.reference_button.setEnabled(ready)
        self.open_reference_button.setEnabled(not self._busy and bool(self._reference_data))
        self.reference_view_button.setEnabled(not self._busy and orientation_ready)
        self.reference_view_button.setVisible(bool(self._reference_data))
        self.reference_view_button.setChecked(self._view_mode == "reference")
        self.results_view_button.setEnabled(not self._busy and bool(self._jpg_artifacts))
        self.results_view_button.setChecked(self._view_mode == "results")
        self.style_tabs.setEnabled(not self._busy and bool(self._jpg_artifacts) and self._view_mode == "results")
        self.style_tabs.setVisible(len(self._jpg_artifacts) > 1 and self._view_mode == "results")
        star_idle = not self._busy and not active and not self._closing_requested
        for widget in (self.star_separation, self.star_milkyway_mask, self.star_milkyway_evidence,
                       self.star_background_strength, self.star_milkyway_offset, *self._star_path_buttons):
            widget.setEnabled(star_idle)
        star_ready = star_idle and bool(self.star_separation.text().strip()) and bool(self.output_dir.text().strip())
        self.star_analyze_button.setEnabled(star_ready)
        self.star_render_button.setEnabled(star_ready)
        self.star_apply_button.setEnabled(star_idle and bool(self._star_suggestion)
                                         and bool(self._star_suggestion.get("available")))
        self.star_open_button.setEnabled(star_idle and bool(self._star_result))
        self.stars_view_button.setEnabled(not self._busy and bool(self._star_result) and not self._closing_requested)
        # A workflow's final.jpg already includes the requested star reduction.
        # The separate view is only for an explicit advanced layer edit.
        self.stars_view_button.setVisible(bool(self._star_result) and not self.workflow_dir)
        self.stars_view_button.setChecked(self._view_mode == "stars")
        can_render = not self._busy and bool(self.job_dir) and self.job_state == "succeeded"
        for widget in (self.red_gain, self.blue_gain, self.exposure_ev,
                       self.reset_preview_button, self.render_preview_button):
            widget.setEnabled(can_render)
        self.open_button.setText(tr('打开派生') if self._view_mode == "advanced" else tr('打开预览') if self._rendered_preview
                                 else tr('打开作品目录'))
        for action, button in self._action_buttons:
            action.setEnabled(button.isEnabled() and not self._closing_requested)
        for key in ("input", "output"):
            self.menu_actions[key].setEnabled(not self._busy and not active and not self._closing_requested)
        workflow_idle = not self._busy and not active and not self._closing_requested
        has_model = bool(self.star_model_path.text().strip())
        for control in (self.model_channel_order, self.model_range, self.model_layout, self.model_stitch):
            control.setEnabled(workflow_idle and has_model)
        self.workflow_auto_strength.setEnabled(workflow_idle and self.stars_enabled.isChecked())
        self.workflow_background_strength.setEnabled(workflow_idle and self.stars_enabled.isChecked() and not self.workflow_auto_strength.isChecked())
        for control in (self.workflow_save_starless, self.workflow_save_star_layer):
            control.setEnabled(workflow_idle and self.stars_enabled.isChecked())
        self.star_component_note.setText(tr('已选择星点整理后端。') if has_model else tr('需要去星模型：在“设置 → 处理后端”选择；未安装时在“组件与提示词管理”准备。'))
        if self._extra_tools is not None:
            self._extra_tools.set_busy(not workflow_idle)
        self.reference_button.setVisible(bool(self.input_files) and not self._reference_data and not active
                                         and not self._reference_running and not self._reference_pending)
        origin = self._regions_origin()
        if origin is not None:
            self.regions_provenance.set_kind(origin)
            self.regions_provenance.setAccessibleName(tr('由你确认') if origin == 'human' else tr('Agent 参与'))
        self.regions_provenance.setVisible(origin is not None)
        # The 「人定」 seal is only for a human confirmation; an Agent never seals.
        self.regions_seal.setVisible(origin == 'human')
        self._update_recipe_tile()
        self._update_stages()
        self._update_activity()
        experience = getattr(self, 'experience', None)
        if experience is not None:
            experience.update_storage_controls()     # cache cleanup is offered only while idle

    # ------------------------------------------------------------------ 「拓星」 main-window chrome

    STAGE_KEYS = ("materials", "reference", "regions", "compose", "results")

    def _regions_drawn(self) -> bool:
        """Saved regions carry explicit ground marks or a confirmed Milky Way ellipse."""
        project = self._regions_project
        if not project:
            return False
        from starstack.regions import milkyway_active
        return bool(project.get('operations')) or milkyway_active(project)

    def _regions_origin(self) -> str | None:
        """Who decided the current regions: 'human' (saved in the region editor), 'agent', or unknown."""
        if not self._regions_drawn():
            return None
        project = self._regions_project
        from starstack.regions import milkyway_active
        milkyway = project.get('milkyway') or {}
        # The recorded authority wins: accepting an Agent's ellipse unchanged in the editor
        # still saves authority 'agent', so it must never show the human seal.
        if milkyway_active(project) and milkyway.get('authority') == 'agent':
            return 'agent'
        if project is getattr(self, '_regions_saved_by_user', None):
            return 'human'
        if milkyway_active(project) and milkyway.get('authority') == 'human' and milkyway.get('reviewer') == 'GUI user':
            return 'human'
        return None

    def _can_resume_current_job(self) -> bool:
        """A failed or stopped job of the current selection can be continued from its checkpoint."""
        return (not self._job_detached and self.job_state in ("failed", "cancelled")
                and bool(self.workflow_dir or self.job_dir))

    def _start_blocker(self, restart_required: bool = False) -> str:
        """Why 「开始合成」 is unavailable now (the first failing condition), or ''."""
        if self._busy or self.job_state in ACTIVE_STATES or self._closing_requested:
            return tr('正在处理，完成后可用。')
        if restart_required:
            return tr('组件已更新，请重启 Aester 后再开始。')
        if not self.input_files and not self.input_dir.text().strip():
            return tr('先添加至少 3 张照片。')
        if not self.output_dir.text().strip():
            return tr('先选择作品保存位置。')
        if self.input_files and len(self.input_files) < 3:
            return tr('至少需要 3 张照片（当前 {0} 张）。', len(self.input_files))
        if self.input_files and not self._orientation_ready():
            if self._reference_running or self._reference_pending:
                return tr('首帧预览生成后才能开始。')
            if self._reference_failed:
                return tr('首帧预览未完成：点 02 首帧重试后再开始。')
            return tr('合成后首帧预览已清空：点 02 首帧或“刷新首帧预览”后再开始。')
        return ''

    def _current_stage(self) -> str:
        active = self.job_state in ACTIVE_STATES
        if active:
            return "compose"
        # A finished job only drives the bar while its selection is still the current one.
        if not self._job_detached:
            if self.job_state in ("failed", "cancelled"):
                return "compose"
            if self.job_state == "succeeded" and (self._jpg_artifacts or self.workflow_dir or self.job_dir):
                return "results"
        if not self.input_files:
            return "materials"
        if not self._orientation_ready():
            return "reference"
        if self._scene_from_current_marks() == 'landscape' or self._regions_drawn():
            return "compose"
        return "regions"

    def _stage_availability(self) -> dict[str, tuple[bool, str]]:
        """(clickable, tooltip) per stage, derived from the existing controls' enabled state."""
        busy_reason = tr('正在处理，完成后可用。')
        idle = not self._busy and self.job_state not in ACTIVE_STATES and not self._closing_requested
        can_show = bool(self._reference_data) and self.reference_view_button.isEnabled()
        can_rebuild = (not self._reference_data and bool(self.input_files) and self.reference_button.isEnabled()
                       and not self._reference_running and not self._reference_pending)
        if can_show:
            reference = (True, tr('查看首帧校色预览。'))
        elif can_rebuild and self._reference_failed:
            reference = (True, tr('首帧预览未完成，点此重试。'))
        elif can_rebuild and self._can_resume_current_job():
            reference = (True, tr('重新生成首帧预览：合成后首帧预览已清空，点此按当前照片与颜色重新生成。'
                                  '要继续这个任务，请点下方“继续任务”。'))
        elif can_rebuild and (self.workflow_dir or self.job_dir):
            reference = (True, tr('重新生成首帧预览：合成后首帧预览已清空，点此按当前照片与颜色重新生成。'))
        elif can_rebuild:
            reference = (True, tr('生成首帧预览：点此按当前照片与颜色生成。'))
        elif not self.input_files:
            reference = (False, tr('先添加照片，首帧会自动生成校色预览。'))
        elif not idle:
            reference = (False, busy_reason)
        else:
            reference = (False, tr('首帧预览正在生成。'))
        if self.regions_button.isEnabled():
            regions = (True, tr('可选：有地景或银河时标记；纯天空可直接开始合成。'))
        elif not idle:
            regions = (False, busy_reason)
        else:
            regions = (False, tr('首帧预览完成后才能标记分区。'))
        materials = ((True, tr('添加照片；也可以把照片拖进窗口。')) if self.add_files_button.isEnabled()
                     else (False, busy_reason))
        results = ((True, tr('查看成片。')) if self.results_view_button.isEnabled()
                   else (False, tr('合成完成后可查看成片。')))
        if not idle:
            compose = (False, busy_reason)
        elif self.start_button.isEnabled():
            compose = (True, tr('选择效果与保存位置，然后开始合成。'))
        else:
            compose = (True, tr('选择效果与保存位置。') + ' ' + self._start_blocker())
        return {"materials": materials, "reference": reference, "regions": regions,
                "compose": compose, "results": results}

    def _update_stages(self) -> None:
        bar = getattr(self, "stage_bar", None)
        if bar is None:
            return
        current = self.STAGE_KEYS.index(self._current_stage())
        availability = self._stage_availability()
        for index, key in enumerate(self.STAGE_KEYS):
            state = "done" if index < current else "current" if index == current else "todo"
            clickable, tip = availability[key]
            if state == "todo" and not clickable:
                state = "disabled"
            bar.set_state(key, state)
            # A reached stage keeps its done/current mark even when it cannot be clicked now.
            bar.node(key).setEnabled(clickable)
            bar.set_tooltip(key, tip)
            # Screen readers also hear where the person is (the dots only show it visually).
            spoken = {"done": tr('已完成'), "current": tr('当前步骤')}.get(state)
            bar.node(key).setAccessibleDescription(spoken + ' · ' + tip if spoken else tip)

    # ------------------------------------------------------------------ activity (二十八宿 sweep)

    ACTIVITY_SHOW_DELAY_MS = 300     # short first-frame updates never flash the marks
    ACTIVITY_LINGER_MS = 200         # chained updates keep one steady mark

    def _activity_wanted(self) -> bool:
        return bool(self._busy or self.job_state in ACTIVE_STATES or self._reference_running) \
            and not self._closing_requested

    def _activity_stage(self) -> str:
        """The stage doing the work: 04 for a job, 02 for a first-frame build, else the current one."""
        if self.job_state in ACTIVE_STATES:
            return "compose"
        if self._reference_running and not self._busy:
            return "reference"
        return self._current_stage()

    def _update_activity(self) -> None:
        """Show the activity marks at once for a job or explicit work; first-frame updates after a pause."""
        if not hasattr(self, "_activity_clock"):
            return
        wanted = self._activity_wanted()
        immediate = self._busy or self.job_state in ACTIVE_STATES
        if wanted == self._activity_on:
            self._activity_timer.stop()
        elif wanted and immediate:
            self._activity_timer.stop()
            self._activity_on = True
        elif not self._activity_timer.isActive():
            self._activity_timer.start(self.ACTIVITY_SHOW_DELAY_MS if wanted else self.ACTIVITY_LINGER_MS)
        self._apply_activity()

    def _settle_activity(self) -> None:
        self._activity_on = self._activity_wanted()
        self._apply_activity()

    def _apply_activity(self) -> None:
        on = self._activity_on
        clock = self._activity_clock
        clock.set_active(on)
        self.activity_spinner.setAccessibleName(tr('正在停止') if self.job_state == "cancel_requested" else tr('正在处理'))
        self.activity_spinner.setVisible(on)
        self.stage_bar.set_busy(self._activity_stage() if on else None, clock)
        self.preview.set_activity(clock if on else None)

    def _stage_clicked(self, key: str) -> None:
        if key == "materials" and self.add_files_button.isEnabled():
            self._pick_input()
        elif key == "reference":
            if self._reference_data and self.reference_view_button.isEnabled():
                self.show_reference()
            elif (not self._reference_data and self.input_files and self.reference_button.isEnabled()
                  and not self._reference_running and not self._reference_pending):
                self.prepare_reference()
        elif key == "regions" and self.regions_button.isEnabled():
            self.edit_regions()
        elif key == "compose":
            self._steps_scroll.ensureWidgetVisible(self._effects_heading)
            (self.start_button if self.start_button.isEnabled() else self.trails_enabled).setFocus()
        elif key == "results" and self.results_view_button.isEnabled():
            self.show_results()

    def _camera_look_text(self, reference: dict[str, Any]) -> tuple[str, str]:
        """The recipe tile: the real camera look in the reference's colour cache, else '无'."""
        title = tr('相机观感')
        cache = reference.get('color_cache') if isinstance(reference, dict) else None
        params = (cache or {}).get('base_params') or {}
        look_id = params.get('look_id') if params.get('look') == 'camera_hsm' else None
        from starstack import camera_looks
        if not look_id or look_id not in camera_looks.LOOKS:
            return f"{title} · {tr('无')}", tr('首帧预览未使用相机观感表。')
        from starstack import devices
        info = camera_looks.LOOKS[look_id]
        body = camera_looks.camera_of((cache or {}).get('master_metadata') or {}) or {}
        entry = devices.lookup(body.get('make'), body.get('model')) if body else None
        donor_make, _, donor_model = str(info.get('donor') or '').partition(' ')
        donor = devices.lookup(donor_make, donor_model)

        def short(item: dict | None, fallback: str) -> str:
            name = str((item or {}).get('name') or fallback)
            first, _, rest = name.partition(' ')
            make = str((item or {}).get('make') or '')
            return rest if rest and make and first.lower() == make.split()[0].lower() else name

        body_name = short(entry, str(body.get('model') or ''))
        donor_name = short(donor, str(info.get('donor') or look_id))
        strength = float(params.get('look_strength', 1.0))
        parts = [title]
        if body_name:
            parts.append(body_name)
        if not entry or not donor or entry.get('name') != donor.get('name'):
            parts.append(tr('借 {0} 表', donor_name))
        parts.append(f"{strength:.2f}")
        tip = tr('观感表来源：{0}（{1}）；强度 {2:.2f}。只影响预览与成片的颜色风格，原片不改动。',
                 info.get('source', look_id), info.get('licence', ''), strength)
        return " · ".join(parts), tip

    def _update_recipe_tile(self) -> None:
        reference = self._reference_data
        key = str((reference or {}).get('reference_dir') or id(reference)) if reference else None
        if key == getattr(self, '_recipe_key', ''):
            return
        self._recipe_key = key
        if not reference:
            self.recipe_label.hide()
            return
        try:
            text, tip = self._camera_look_text(reference)
        except (ValueError, TypeError, KeyError, AttributeError):
            text, tip = f"{tr('相机观感')} · {tr('无')}", ""
        self.recipe_label.setText(text)
        self.recipe_label.setToolTip(text + "\n" + tip if tip else text)
        self.recipe_label.show()

    def _apply_tab_order(self) -> None:
        """Keyboard order follows the work: photos → view tools → steps ① ② ③ → start."""
        chain = [
            self.add_files_button, self.remove_files_button, self.clear_files_button, self.file_strip,
            self.reference_view_button, self.results_view_button, self.stars_view_button, self.reference_button,
            self.style_tabs, self.open_button,
            self.rotate_left_button, self.rotate_right_button, self.restore_orientation_button,
            self.surround_grey_button, self.night_button,
            self.regions_button,
            self.reset_color_button, self.warmth_slider, self.warmth_value, self.tint_slider, self.tint_value,
            self.exposure_slider, self.exposure_value,
            self.trails_enabled, self.stars_enabled, self.workflow_auto_strength, self.workflow_background_strength,
            self.workflow_milkyway_offset, self.workflow_save_starless, self.workflow_save_star_layer,
            self.star_component_button, self.milkyway_enabled, self.milkyway_strength,
            self.output_dir, self._output_browse_button, self.jpeg_quality_slider, self.jpeg_quality_value,
            self.start_button, self.cancel_button,
            self.trail_review_button, self.details_button,
            *self.stage_bar.nodes(), self.retouch_button,
        ]
        for first, second in zip(chain, chain[1:]):
            QWidget.setTabOrder(first, second)
        self._tab_chain = chain

    def _run_async(self, name: str, operation: Callable[[], Any], callback: Callable[[Any], None]) -> None:
        if self._closing_requested:
            return
        worker = _Worker(name, operation)
        worker.signals.finished.connect(self._worker_finished)
        worker.signals.failed.connect(self._worker_failed)
        self._workers[name] = (worker, callback)
        self._pool.start(worker)

    @Slot(str, object)
    def _worker_finished(self, name: str, result: Any) -> None:
        record = self._workers.pop(name, None)
        if record and not self._closing_requested:
            try:
                record[1](result)
            except Exception as exc:
                self._handle_error(name, str(exc))

    @Slot(str, str)
    def _worker_failed(self, name: str, message: str) -> None:
        self._workers.pop(name, None)
        if not self._closing_requested:
            self._handle_error(name, message)

    def _handle_error(self, name: str, message: str) -> None:
        # Backend text stays Chinese in records; English UI shows it translated.
        message = translate_message(message)
        if name.startswith("doctor:"):
            if name == f"doctor:{self._doctor_serial}":
                self.backend_status.setText(tr('Siril 检查未完成：') + message)
        elif name.startswith("preview:"):
            if name == f"preview:{self._preview_serial}":
                self.preview.show_message(tr('无法显示这张预览\n') + message)
        elif name.startswith("reference-build:"):
            self._reference_running = False
            if name == f"reference-build:{self._reference_generation}":
                rotation_failed = bool(self._orientation_notice and self._reference_data
                                       and self._reference_scene == tuple(self.input_files))
                if rotation_failed:
                    # Failed derived-cache creation never becomes the selected
                    # direction. Keep a visible, retryable, unambiguous frame.
                    self._image_rotation = int(self._reference_data.get('image_rotation', 0))
                    self._orientation_notice = False
                    self.show_reference()
                self._reference_failed = not rotation_failed
                self.message_label.set_error_text((tr('方向调整未完成，已保留之前的方向；请重试：') if rotation_failed
                                                  else tr('首帧预览未完成：')) + message)
                self.preview_note.setText(tr('预览未更新；检查文件或恢复默认颜色后重试。'))
            if self._reference_pending:
                self._color_timer.start(1)
            self._update_controls()
        else:
            self._busy = False
            if name == 'import-raw':
                self._import_cancel = None
            if name == "raw-audit":
                self.raw_audit_status.setText(tr('单张 RAW 检查未完成：') + message)
            elif name in {"star-analysis", "star-render"}:
                self.star_status.setText(tr('星点整理未完成：') + message)
            elif name.startswith("advanced:") and self._extra_tools is not None:
                self._extra_tools.failed(name.partition(":")[2], message)
            self.message_label.set_error_text(tr('操作未完成：') + message)
            self._update_controls()

    def check_backend(self) -> None:
        self._doctor_serial += 1
        serial = self._doctor_serial
        selected = self.siril_path.text().strip() or None
        self.backend_status.setText(tr('正在检查 Siril…'))

        def checked(result: dict[str, Any]) -> None:
            if serial != self._doctor_serial:
                return
            if result.get("available"):
                version = str(result.get("version") or "").strip().splitlines()
                self.backend_status.setText(tr('Siril 已就绪') + (" · " + version[0][:70] if version else ""))
                self.backend_status.setToolTip(str(result.get("executable") or ""))
            else:
                detail = str(result.get("message") or "")
                self.backend_status.setText(tr('未找到 Siril，请选择已安装的程序。') + ("\n" + translate_message(detail) if detail else ""))

            if 'aester_finite' in result:
                self._set_finite_backend(result['aester_finite'])

        def inspect():
            from .backend_preferences import discover_finite_backend
            return {**doctor(selected), 'aester_finite': discover_finite_backend()}
        self._run_async(f"doctor:{serial}", inspect, checked)

    def start_processing(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        if self.experience._restart_required:
            self.message_label.setText(tr('组件已更新，请先保存编辑项目并从“组件与提示词管理”重新启动 Aester。'))
            return
        input_path = self.input_dir.text().strip()
        output_path = self.output_dir.text().strip()
        try:
            work_path = self._processing_root()
        except ValueError as error:
            self.message_label.setText(tr('操作未完成：') + translate_message(str(error)))
            return
        mask = self.sky_mask.text().strip() or None
        siril = self.siril_path.text().strip() or None
        engine = self.stack_engine_path.text().strip() or None
        if self.stack_backend_choice.currentData() == 'finite' and not engine:
            self.message_label.setText(tr('尚未安装改进构建。请重新运行安装程序并勾选该组件，或选择 Siril 官方构建。'))
            return
        if self.input_files and not self._orientation_ready():
            self.message_label.setText(tr('请等待整组方向与首帧预览更新完成，再开始合成。'))
            return
        model = self.star_model_path.text().strip() or None
        files = list(self.input_files) if self.input_files else None
        image_rotation = self._effective_image_rotation()
        trails_enabled = self.trails_enabled.isChecked()
        stars_enabled = self.stars_enabled.isChecked()
        # Contract A: only non-default options are sent, so older requests keep their digest. The JPEG quality is
        # always sent: the backend's unset value means each JPEG's historical encoding, not the slider's 95.
        options: dict[str, Any] = {"jpeg_quality": self.jpeg_quality()}
        if trails_enabled:
            options['trail_default'] = 'meteor'
        if stars_enabled and self.workflow_save_starless.isChecked():
            options['save_starless'] = True
        if stars_enabled and self.workflow_save_star_layer.isChecked():
            options['save_star_layer'] = True
        adjustments = self.color_adjustments()
        region_project = self._regions_project
        regions_path = self._regions_path
        scene_mode = self._scene_from_current_marks()
        milkyway_enabled = self.milkyway_enabled.isChecked()
        milkyway_strength = self.milkyway_strength.value() / 100
        from starstack.regions import milkyway_active
        milkyway_offset = (self.workflow_milkyway_offset.value() / 100
                           if stars_enabled and region_project and milkyway_active(region_project) else 0.)
        range_mode = "positive_power2_v1" if self.stack_range_checkbox.isChecked() else "legacy"
        strength = None if self.workflow_auto_strength.isChecked() else self.workflow_background_strength.value() / 100
        backend_options = {
            "channel_order": self.model_channel_order.currentData(),
            "value_range": self.model_range.currentData(),
            "layout": self.model_layout.currentData(),
            "stitch": self.model_stitch.currentData(),
        }
        if not Path(input_path).is_dir():
            self.message_label.setText(tr('请选择存在的素材文件夹。'))
            return
        if not output_path:
            self.message_label.setText(tr('请选择结果文件夹。'))
            return
        if files:
            if len(files) < 3:
                self.message_label.setText(tr('至少选择 3 张连拍照片；单张仍可预览。'))
                return
            if scene_mode == "sky" or (not mask and not region_project):
                reference = self._reference_data or {}
                if self._reference_scene != tuple(files) or not reference.get("mask_template_path"):
                    self.message_label.setText(tr('纯天空范围需要首帧尺寸，请等待预览完成后开始。'))
                    return
                # No ground marks means the full image is sky. The main view
                # explains this default; drawing the boundary changes it.
                mask = str(reference["mask_template_path"])
        for selected, label in ((mask, tr('天空蒙版')), (engine, tr('堆栈引擎')), (model if stars_enabled else None, tr('去星模型'))):
            if selected and not Path(selected).is_file():
                self.message_label.setText(tr('找不到{0}，请重新选择。', label))
                return
        if stars_enabled and not model:
            self.message_label.setText(tr('整理星点需要去星模型；请在设置 → 处理后端中选择，或关闭整理星点。'))
            return
        if stars_enabled and not mask:
            self.message_label.setText(tr('去星与缩星需要明确的天空分区，请先载入蒙版。'))
            return
        self._busy = True
        self.message_label.setText(tr('正在提交完整出图任务…'))
        self._update_controls()

        def submit() -> dict[str, Any]:
            from starstack.workflow import submit_workflow
            extra = {}
            # A saved legacy external mask carries an empty region placeholder
            # for source identity only; it must not displace the user's PNG.
            if region_project and (region_project.get('operations') or scene_mode == 'sky'):
                from starstack import regions
                effective_regions = regions.revise(region_project, scene_mode=scene_mode)
                saved = regions.save_project(effective_regions, work_path)
                extra['regions_path'] = saved['project_path']
            elif regions_path and region_project is None:
                extra['regions_path'] = regions_path
            if milkyway_enabled:
                extra.update(milkyway_enabled=True, milkyway_strength=milkyway_strength)
            extra.update(options)
            return submit_workflow(input_path, work_path, delivery_root=output_path, siril_path=siril,
                                   sky_mask_path=None if extra.get('regions_path') else mask,
                                   stack_engine_path=engine, star_model_path=model,
                                   stack_range_mode=range_mode, background_strength=strength,
                                   milkyway_offset=milkyway_offset, backend_options=backend_options if model else None,
                                   input_files=files, trails_enabled=trails_enabled, stars_enabled=stars_enabled,
                                   color_adjustments=adjustments, image_rotation=image_rotation, **extra)

        self._run_async("submit", submit, self._workflow_submitted)

    def _workflow_submitted(self, result: dict[str, Any]) -> None:
        self._busy = False
        self._job_detached = False
        directory = result.get("workflow_dir")
        if not directory:
            self.message_label.setText(self._status_message(result, tr('后台未返回完整任务目录。')))
            self._update_controls()
            return
        self.workflow_dir = str(Path(directory).resolve())
        self._reference_cache_epoch += 1
        self._reference_generation += 1
        self._reference_pending = False
        self._reference_data = None
        self._reference_scene = ()
        self._color_timer.stop()
        self._workflow = dict(result)
        config = result.get("config") or {}
        self._image_rotation = int(config.get('image_rotation', 0))
        self._orientation_notice = False
        if config.get('delivery_root'):
            self.cache_dir.setText(config['output_root'])
        self._restoring_project = True
        self._regions_path = config.get('regions_path') or None
        self._regions_project = None
        if self._regions_path:
            from starstack import regions
            self._regions_project = regions.load_project(self._regions_path, verify_sources=False)
        self.milkyway_enabled.setChecked(config.get('milkyway_enabled') is True)
        self.milkyway_strength.setValue(float(config.get('milkyway_strength', .6)) * 100)
        self.workflow_milkyway_offset.setValue(float(config.get('milkyway_offset', 0)) * 100)
        restored_mode = ((self._regions_project or {}).get('scene_mode')
                         or ('landscape' if config.get('sky_mask_path') else 'unknown'))
        self.scene_mode.setCurrentIndex(max(0, self.scene_mode.findData(restored_mode)))
        selected_files = config.get("input_files")
        self.input_files = [str(path) for path in selected_files] if isinstance(selected_files, list) else []
        self._selection_changed(preview=False)
        for widget, key in ((self.trails_enabled, "trails_enabled"), (self.stars_enabled, "stars_enabled")):
            effective = config.get(key) is True
            if key == "stars_enabled" and key not in config:
                # Pre-P0 records intentionally resume their original implicit model choice.
                effective = bool(config.get("star_model_path"))
            widget.setChecked(effective)
        request = result.get("request") or {}
        for widget, key in ((self.workflow_save_starless, "save_starless"),
                            (self.workflow_save_star_layer, "save_star_layer")):
            widget.setChecked(config.get(key) is True or request.get(key) is True)
        colors = config.get("color_adjustments") or {}
        for widget, key, scale in ((self.warmth_slider, "warmth", 1), (self.tint_slider, "tint", 1),
                                   (self.exposure_slider, "exposure_ev", 10)):
            widget.setValue(round(float(colors.get(key, 0)) * scale))
        self._restoring_project = False
        self._show_workflow_jpeg_quality(config)
        for widget, key in ((self.input_dir, "input_dir"), (self.output_dir, "delivery_root" if config.get('delivery_root') else "output_root"),
                            (self.sky_mask, "sky_mask_path"), (self.siril_path, "siril_path"),
                            (self.stack_engine_path, "stack_engine_path"), (self.star_model_path, "star_model_path")):
            if key in config:
                widget.setText(str(config[key] or ""))
                widget.setCursorPosition(len(widget.text()))
                widget.deselect()
        self._sync_stack_backend_choice()
        options = config.get("backend_options") or {}
        for widget, key in ((self.model_channel_order, "channel_order"), (self.model_range, "value_range"),
                            (self.model_layout, "layout"), (self.model_stitch, "stitch")):
            if key in options:
                index = widget.findData(options[key])
                if index >= 0:
                    widget.setCurrentIndex(index)
        self.stack_range_checkbox.setChecked(config.get("stack_range_mode") == "positive_power2_v1")
        if "background_strength" in config:
            self.workflow_auto_strength.setChecked(config["background_strength"] is None)
            if config["background_strength"] is not None:
                self.workflow_background_strength.setValue(float(config["background_strength"]) * 100)
        self.job_dir = str(result.get("stack_job_dir") or "") or None
        self.job_state = str(result.get("state") or "queued")
        self._job = None
        self._view_mode = "results"
        self._rendered_preview = None
        self._star_result = None
        self._star_suggestion = None
        self._star_suggestion_inputs = None
        self.star_separation.clear()
        self.star_milkyway_mask.clear()
        self.star_milkyway_evidence.clear()
        self.red_gain.setValue(1.0)
        self.blue_gain.setValue(1.0)
        self.exposure_ev.setValue(0.0)
        self._artifact_signature = ()
        self._jpg_artifacts = []
        self._preview_serial += 1
        self.preview.show_message(tr('正在处理素材\n\n成片保存后会显示在这里。'), art=True)
        self.preview_note.setText(tr('完整出图任务已提交'))
        self.job_path_label.setText(tr('任务目录：') + self.workflow_dir)
        self.job_path_label.setToolTip(self.workflow_dir)
        self.timer.start()
        self.refresh_job()
        self._update_controls()

    def refresh_workflow(self) -> None:
        from starstack.workflow import workflow_status
        try:
            record = workflow_status(self.workflow_dir)
        except Exception as exc:
            self.message_label.setText(tr('暂时无法读取完整任务状态：') + translate_message(str(exc)))
            return
        self._apply_workflow_record(record)

    @staticmethod
    def _same_dir(first: object, second: object) -> bool:
        if not first or not second:
            return False
        return os.path.normcase(os.path.abspath(str(first))) == os.path.normcase(os.path.abspath(str(second)))

    @classmethod
    def _has_trails_version(cls, record: dict[str, Any] | None) -> bool:
        """The current delivery adds trails to a separate base photograph (contract B)."""
        record = record or {}
        base, current = record.get("base_delivery_dir"), record.get("delivery_dir")
        return bool(base and current and not cls._same_dir(base, current))

    @classmethod
    def _workflow_jpgs(cls, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Result tabs: the delivery's final.jpg, and with added trails also the base photograph."""
        artifacts = record.get("artifacts") or []
        if isinstance(artifacts, dict):
            artifacts = list(artifacts.values())
        jpgs = [dict(a) for a in artifacts if isinstance(a, dict) and
                (a.get("kind") == "jpeg" or str(a.get("path", "")).lower().endswith((".jpg", ".jpeg")))]
        jpgs.sort(key=lambda a: Path(str(a.get("path", ""))).name != "final.jpg")
        final_jpgs = [a for a in jpgs if Path(str(a.get('path', ''))).name == 'final.jpg']
        if final_jpgs:
            jpgs = final_jpgs
        if len(jpgs) == 1 and cls._has_trails_version(record):
            base = Path(str(record["base_delivery_dir"])) / "final.jpg"
            # The with-trails photograph comes first; the base stays one click away.
            jpgs = [dict(jpgs[0], label='含亮迹')]
            if base.is_file():
                jpgs.append({"path": str(base), "name": "final.jpg", "kind": "jpeg", "label": '基础成片'})
        return jpgs

    def _apply_workflow_record(self, record: dict[str, Any]) -> None:
        self._workflow = dict(record)
        self.job_state = str(record.get("state") or "")
        stack_dir = record.get("stack_job_dir")
        if stack_dir:
            self.job_dir = str(stack_dir)
            try:
                self._job = load_job(self.job_dir)
            except (OSError, ValueError):
                pass
        title = tr(STATE_LABELS.get(self.job_state, '读取任务状态'))
        stage = str(record.get("stage") or "")
        workflow_stages = {"stack": tr('对齐并堆栈'), "separation": tr('分离星点图层'), "starless": tr('分离星点图层'),
                           "density_advice": tr('评估星点密度'), "star_reduction": tr('整理星点'),
                           "layer_export": tr('保存后期图层'), "publish": tr('保存结果'), "export": tr('保存结果')}
        self.stage_label.setText(workflow_stages.get(stage, tr(STAGE_LABELS.get(stage, title)))
                                 if self.job_state == "running" else title)
        try:
            progress = max(0, min(100, int(record.get("progress") or 0)))
        except (TypeError, ValueError):
            progress = 0
        self.progress_bar.setValue(progress)
        self.percent_label.setText(f"{progress}%")
        self._show_status(self._status_message(record, self.stage_label.text()))
        jpgs = self._workflow_jpgs(record)
        if self._rendered_preview and self._rendered_preview.get("job_dir") == self.job_dir:
            jpgs = self._rendered_preview["jpgs"]
        signature = tuple((str(a.get("path")), str(a.get("label"))) for a in jpgs)
        if signature != self._artifact_signature:
            self._artifact_signature = signature
            self._jpg_artifacts = jpgs
            self._populate_styles()
        separation = record.get("separation_result")
        if separation and Path(str(separation)).is_file() and self.star_separation.text() != str(separation):
            self.star_separation.setText(str(separation))
        if self.job_state in TERMINAL_STATES:
            self._job_finished_alert()
            self.timer.stop()
            if self.job_state == "succeeded":
                finishing = record.get("finishing") or {}
                reduction = finishing.get("reduction")
                if isinstance(reduction, dict) and reduction.get("state") == "succeeded":
                    self._star_result = dict(reduction)
                    self._star_result.update(gui_background_strength=float(finishing["background_strength_applied"]),
                                             gui_milkyway_offset=float((record.get("request") or {}).get("milkyway_offset", 0.0)))
                self._set_result_note()
            elif not jpgs:
                self.preview.show_message(tr('本次尚无可预览成片\n\n请查看下方原因或任务详情。'))
        self._update_controls()

    def _show_status(self, text: str) -> None:
        """Job status in the readout; a failed job shows the same full text in the error colour."""
        if self.job_state == "failed":
            self.message_label.set_error_text(text)
        else:
            self.message_label.setText(text)

    def _job_finished_alert(self) -> None:
        """Flash the taskbar entry once when a running job ends while the window is in the background."""
        if self.timer.isActive() and not self.isActiveWindow():
            QApplication.alert(self)

    def _submitted(self, result: dict[str, Any]) -> None:
        self._image_rotation = int(result.get('image_rotation', 0))
        self._orientation_notice = False
        self._busy = False
        job_dir = result.get("job_dir")
        if not job_dir:
            self.message_label.setText(self._status_message(result, tr('后台未返回任务目录。')))
            self._update_controls()
            return
        self.workflow_dir = None
        self._workflow = None
        self._forget_workflow_jpeg_quality()
        self._job_detached = False
        self.job_dir = str(Path(job_dir).resolve())
        self.job_state = str(result.get("state") or "queued")
        self._job = dict(result)
        # Missing fields in older jobs never imply that range protection ran.
        self.stack_range_checkbox.setChecked(result.get("stack_range_mode") == "positive_power2_v1")
        self._view_mode = "results"
        self._rendered_preview = None
        self.red_gain.setValue(1.0)
        self.blue_gain.setValue(1.0)
        self.exposure_ev.setValue(0.0)
        self._artifact_signature = ()
        self._jpg_artifacts = []
        self._preview_serial += 1
        self.style_tabs.setEnabled(False)
        self.preview.show_message(tr('正在处理素材\n\n成片保存后，真实预览会显示在这里。'), art=True)
        self.preview_note.setText(tr('任务在后台运行，可取消后从检查点恢复。'))
        self.message_label.setText(self._status_message(result, tr('任务已提交。')))
        self.job_path_label.setText(tr('任务目录：') + self.job_dir)
        self.job_path_label.setToolTip(self.job_dir)
        self.timer.start()
        self.refresh_job()
        self._update_controls()

    def cancel_processing(self) -> None:
        if self._import_cancel is not None:
            self._import_cancel.set()
            self.message_label.setText(tr('正在停止导入，原片保持不变…'))
            return
        if not (self.workflow_dir or self.job_dir) or self._busy or self.job_state not in {"queued", "running"}:
            return
        job_dir = self.job_dir
        self._busy = True
        self.message_label.setText(tr('正在请求取消，等待当前步骤停止…'))
        self._update_controls()

        def cancelled(_: Any) -> None:
            self._busy = False
            self.refresh_job()

        if self.workflow_dir:
            from starstack.workflow import cancel_workflow
            directory = self.workflow_dir
            self._run_async("cancel", lambda: cancel_workflow(directory), cancelled)
        else:
            self._run_async("cancel", lambda: cancel_job(job_dir), cancelled)

    def resume_processing(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        if self.workflow_dir and self.job_state in {"failed", "cancelled"}:
            from starstack.workflow import resume_workflow
            directory = self.workflow_dir
            self._busy = True
            self.message_label.setText(tr('正在恢复完整出图任务…'))
            self._update_controls()
            self._run_async("resume-workflow", lambda: resume_workflow(directory), self._workflow_submitted)
            return
        job_dir = self.job_dir if self.job_state in {"failed", "cancelled"} else None
        if not job_dir:
            job_dir = QFileDialog.getExistingDirectory(self, tr('选择要恢复的任务目录'), self.output_dir.text() or str(WORKSPACE / "outputs"))
        if not job_dir:
            return
        selected = str(job_dir)
        self._busy = True
        self.message_label.setText(tr('正在恢复任务…'))
        self._update_controls()
        if (Path(selected) / "workflow.json").is_file():
            from starstack.workflow import resume_workflow
            self._run_async("resume-workflow", lambda: resume_workflow(selected), self._workflow_submitted)
            return
        self._run_async("resume", lambda: resume_job(selected), self._submitted)

    def generate_demo(self) -> None:
        if self._busy or self.job_state in ACTIVE_STATES:
            return
        output_path = self.output_dir.text().strip() or str(WORKSPACE / "outputs")
        self._busy = True
        self.message_label.setText(tr('正在生成合成演示素材…'))
        self._update_controls()

        def generated(input_path: str) -> None:
            self._busy = False
            self.input_dir.setText(str(input_path))
            self.input_dir.setCursorPosition(len(self.input_dir.text()))
            self.input_dir.deselect()
            mask = Path(input_path) / "sky_mask.png"
            self.sky_mask.setText(str(mask) if mask.is_file() else "")
            self.sky_mask.setCursorPosition(len(self.sky_mask.text()))
            self.sky_mask.deselect()
            self.message_label.setText(tr('合成数值测试素材已生成。点击“开始处理”运行真实处理流程；此演示不代表真实 RAW 画质。'))
            self._update_controls()

        self._run_async("demo", lambda: create_demo(output_path), generated)

    def refresh_job(self) -> None:
        """The timer only reads the existing job; it never starts processing."""
        if self.workflow_dir:
            self.refresh_workflow()
            return
        if not self.job_dir:
            return
        try:
            job = load_job(self.job_dir)
        except Exception as exc:
            self.message_label.setText(tr('暂时无法读取任务状态：') + translate_message(str(exc)))
            return
        self._job = job
        self.job_state = str(job.get("state") or "")
        state_title = tr(STATE_LABELS.get(self.job_state, '读取任务状态'))
        stage = str(job.get("stage") or "")
        stage_title = tr(STAGE_LABELS.get(stage, state_title)) if self.job_state == "running" else state_title
        self.stage_label.setText((tr('原任务 · ') if self._view_mode == "reference" else "") + stage_title)
        try:
            progress = max(0, min(100, int(job.get("progress") or 0)))
        except (TypeError, ValueError):
            progress = 0
        self.progress_bar.setValue(progress)
        self.percent_label.setText(f"{progress}%")
        self._show_status(self._status_message(job, stage_title))
        artifacts = job.get("artifacts") or []
        jpgs = [a for a in artifacts if isinstance(a, dict) and (a.get("kind") == "jpeg" or str(a.get("path", "")).lower().endswith((".jpg", ".jpeg")))]
        if self._rendered_preview and self._rendered_preview.get("job_dir") == self.job_dir:
            jpgs = self._rendered_preview["jpgs"]
        signature = tuple((str(a.get("path")), str(a.get("label"))) for a in jpgs)
        if signature != self._artifact_signature:
            self._artifact_signature = signature
            self._jpg_artifacts = jpgs
            self._populate_styles()
        if self.job_state in TERMINAL_STATES:
            self._job_finished_alert()
            self.timer.stop()
            if self.job_state == "succeeded" and self._view_mode == "results":
                self._set_result_note()
            elif not jpgs and self._view_mode == "results":
                self.preview.show_message(tr('本次尚无可预览成片\n\n可查看下方原因，或恢复任务。'))
        self._update_controls()

    def _populate_styles(self) -> None:
        old_label = self.style_tabs.tabText(self.style_tabs.currentIndex())
        self.style_tabs.blockSignals(True)
        while self.style_tabs.count():
            self.style_tabs.removeTab(0)
        for artifact in self._jpg_artifacts:
            self.style_tabs.addTab(translate_message(str(artifact.get("label") or Path(str(artifact.get("path", ""))).stem)))
        self.style_tabs.setEnabled(bool(self._jpg_artifacts) and self._view_mode == "results")
        selected = next((i for i, a in enumerate(self._jpg_artifacts) if translate_message(str(a.get("label") or Path(str(a.get("path", ""))).stem)) == old_label), 0)
        self.style_tabs.setCurrentIndex(selected)
        self.style_tabs.blockSignals(False)
        if self._jpg_artifacts and self._view_mode == "results":
            self.select_style(selected)

    def select_style(self, index: int) -> None:
        if self._view_mode != "results" or not 0 <= index < len(self._jpg_artifacts):
            return
        path = Path(str(self._jpg_artifacts[index].get("path") or ""))
        if not path.is_absolute() and self.job_dir:
            path = Path(self.job_dir) / path
        self._load_preview_image(path, auto_transform=True)

    def _load_preview_image(self, path: Path, *, auto_transform: bool,
                            expected_size: QSize | None = None) -> None:
        self._preview_serial += 1
        serial = self._preview_serial
        self.preview.show_message(tr('正在读取首帧参考…') if self._view_mode == "reference" else tr('正在读取成片预览…'))

        def read_image() -> QImage:
            reader = QImageReader(str(path))
            reader.setAutoTransform(auto_transform)
            size = reader.size()
            if expected_size is not None and size != expected_size:
                raise ValueError(tr('参考 PNG 尺寸与实际解码记录不一致，请重新准备参考。'))
            if size.isValid() and (size.width() > 1800 or size.height() > 1400):
                reader.setScaledSize(size.scaled(QSize(1800, 1400), Qt.AspectRatioMode.KeepAspectRatio))
            image = reader.read()
            if image.isNull():
                raise ValueError(reader.errorString() or tr('图像无法读取'))
            return image

        def display(image: QImage) -> None:
            if serial == self._preview_serial:
                self.preview.set_image(image)

        self._run_async(f"preview:{serial}", read_image, display)

    def open_results(self) -> None:
        directory = str((self._workflow or {}).get("delivery_dir") or self.workflow_dir or self.job_dir or "")
        if self._view_mode == "advanced":
            directory = self._advanced_result_dir
        elif self._rendered_preview and self._rendered_preview.get("job_dir") == self.job_dir:
            directory = str(self._rendered_preview["directory"])
        if directory and Path(directory).is_dir():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(directory)):
                self.message_label.setText(tr('无法自动打开文件夹，可复制任务目录手动打开。'))

    def closeEvent(self, event: Any) -> None:
        """Keep Qt alive until local image/reference callbacks have drained.

        Persistent processing jobs own separate worker processes and do not
        require this window to remain open.
        """
        self.timer.stop()
        self._color_timer.stop()
        self._color_finish_timer.stop()
        if self._import_cancel is not None:
            self._import_cancel.set()
        experience = getattr(self, 'experience', None)
        if experience is not None:
            experience.cancel_cache_listing()       # the 作品与缓存 job list stops between two jobs
        if self._workers or self._pool.activeThreadCount():
            self._closing_requested = True
            self.centralWidget().setEnabled(False)
            self.message_label.setText(tr('正在等待当前本地操作完成，随后关闭窗口…'))
            self._close_timer.start()
            event.ignore()
            return
        self._close_timer.stop()
        self._save_preferences()
        super().closeEvent(event)

    def _close_after_workers(self) -> None:
        if not self._workers and self._pool.activeThreadCount() == 0:
            self.close()


def main() -> int:
    existing = QApplication.instance()
    app = existing or QApplication(sys.argv)
    app.setApplicationName("Aester")
    app.setOrganizationName("StarStack")
    # Fonts and Qt's own translations exist before the first widget (MainWindow repeats both idempotently).
    fonts.register_bundled_fonts()
    _install_qt_translations(startup_language(QSettings("StarStack", "StarStack")))
    window = MainWindow()
    window.show()
    window.experience.maybe_welcome()
    # Keep a reference when an embedding host already owns the event loop.
    app._starstack_window = window
    return 0 if existing else app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
