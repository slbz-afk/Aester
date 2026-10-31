# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""User drawings directly define the editable sky and Milky Way regions."""
from __future__ import annotations
from starstack.i18n import tr
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)
from starstack import regions
from . import themes
from .region_canvas import OverlayLegend, RegionCanvas
from .ui_widgets import TickSlider

# Discard prompt answers (ask_discard's return values).
USE, DISCARD, KEEP_EDITING = 'use', 'discard', 'keep-editing'


def ask_discard(parent) -> str:
    """Ask before unsaved drawing is dropped. Continuing to edit is the default."""
    box = QMessageBox(themes.message_icon(parent), tr('分区尚未使用'),
                      tr('本次绘制还没有使用。要使用这些分区，还是放弃更改？'), QMessageBox.StandardButton.NoButton, parent)
    use = box.addButton(tr('使用这些分区'), QMessageBox.ButtonRole.AcceptRole)
    discard = box.addButton(tr('放弃更改'), QMessageBox.ButtonRole.DestructiveRole)
    keep = box.addButton(tr('继续编辑'), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    box.exec()
    clicked = box.clickedButton()
    return USE if clicked is use else DISCARD if clicked is discard else KEEP_EDITING


def _button(text, role=None):
    """Dialog buttons never become Enter's implicit default (Enter must not click them)."""
    button = QPushButton(text)
    button.setAutoDefault(False)
    button.setDefault(False)
    if role:
        themes.set_role(button, role)
    return button


class RegionDialog(QDialog):
    def __init__(self, reference, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('天空、地景与银河范围'))
        self._fit_to_screen(1120, 810)
        self.reference = reference
        self.project = project
        self._syncing = False
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        instructions = QLabel(tr('橙色是地景，橙色实线是它的边界；粉紫双线是银河。点“使用这些分区”后生效。选择“拖出银河椭圆”后，可拖动中心和黄色控制点移动、缩放、旋转；方形黄点调整羽化。Shift 拖动重画，滚轮缩放，中键平移。'))
        instructions.setWordWrap(True)
        themes.set_role(instructions, 'muted')
        layout.addWidget(instructions)
        toolbar_widget = QWidget()
        themes.set_role(toolbar_widget, 'toolbar')
        toolbar = QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(0, 0, 0, 0)
        self.tools = QComboBox()
        self.tools.setAccessibleName(tr('绘制工具'))
        for label, data in [(tr('沿地平线点选'), 'horizon'), (tr('补涂地景'), 'ground'), (tr('擦回天空'), 'sky'), (tr('拖出银河椭圆'), 'milkyway')]:
            self.tools.addItem(label, data)
        toolbar.addWidget(self.tools)
        self.finish_button = finish = _button(tr('完成地平线'))
        toolbar.addWidget(finish)
        undo, redo = _button(tr('撤销')), _button(tr('重做'))
        toolbar.addWidget(undo)
        toolbar.addWidget(redo)
        self.assist = _button(tr('寻找附近边缘'))
        self.accept_edge = _button(tr('接受浅蓝色边缘'))
        self.accept_edge.setEnabled(False)
        toolbar.addWidget(self.assist)
        toolbar.addWidget(self.accept_edge)
        toolbar.addStretch(1)
        layout.addWidget(toolbar_widget)
        self.canvas = RegionCanvas(QImage(reference['preview_path']), project, self)
        self._initial_revision = self.canvas.project.get('revision')
        layout.addWidget(self.canvas, 1)
        self.legend = OverlayLegend([('ground', tr('地景边界')), ('milkyway', tr('银河范围')),
                                     ('suggestion', tr('建议边缘')), ('handle', tr('地平线点与手柄'))])
        layout.addWidget(self.legend)
        QShortcut(QKeySequence('Ctrl+0'), self).activated.connect(self.canvas.fit_view)
        self.tools.currentIndexChanged.connect(lambda: self.canvas.set_mode(self.tools.currentData()))
        finish.clicked.connect(self.canvas.finish_horizon)
        undo.clicked.connect(self.canvas.undo)
        redo.clicked.connect(self.canvas.redo)
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(self.canvas.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self).activated.connect(self.canvas.redo)
        self.assist.clicked.connect(self.suggest_edge)
        self.accept_edge.clicked.connect(self.accept_suggestion)
        self.canvas.changed.connect(self.sync)
        # Reopening a drawing that only has a Milky Way ellipse starts with its handles live.
        if self.canvas.project['milkyway'].get('ellipse') and not self.canvas.project.get('operations'):
            self.tools.setCurrentIndex(self.tools.findData('milkyway'))
        controls = QHBoxLayout()
        self.clear_ground = _button(tr('清除地景'))
        self.clear_ground.clicked.connect(self.canvas.clear_landscape)
        controls.addWidget(self.clear_ground)
        self.clear_mw = _button(tr('清除银河'))
        self.clear_mw.clicked.connect(self.canvas.clear_milkyway)
        controls.addWidget(self.clear_mw)
        controls.addSpacing(12)
        self.brush = self._slider(controls, tr('画笔大小'), 2, 100, 150, 18)
        self.brush.setValue(18)
        self.brush.valueChanged.connect(self.set_brush)
        self.feather = self._slider(controls, tr('银河羽化'), 10, 95, 150, 50)
        self.feather.valueChanged.connect(self.set_feather)
        self.angle = self._slider(controls, tr('方向'), -180, 180, 120, 0)
        self.angle.valueChanged.connect(self.set_angle)
        controls.addStretch()
        layout.addLayout(controls)
        self.notice = QLabel(tr('边缘辅助只在所画地平线附近寻找明暗变化，不能自动消除树叶运动或反光。'))
        self.notice.setWordWrap(True)
        themes.set_role(self.notice, 'hint')
        layout.addWidget(self.notice)
        bottom = QHBoxLayout()
        fit = _button(tr('适合窗口'), 'ghost')
        fit.clicked.connect(self.canvas.fit_view)
        bottom.addWidget(fit)
        bottom.addStretch()
        self.cancel_button = cancel = _button(tr('取消'))
        self.accept_button = accept = _button(tr('使用这些分区'), 'primary')
        bottom.addWidget(cancel)
        bottom.addWidget(accept)
        layout.addLayout(bottom)
        cancel.clicked.connect(self.reject)
        accept.clicked.connect(self.finish)
        self.sync()

    def _fit_to_screen(self, width, height):
        """Keep the bottom buttons on screen on small or highly scaled displays."""
        screen = (self.parentWidget().screen() if self.parentWidget() else None) or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width, height = min(width, available.width() - 40), min(height, available.height() - 60)
        self.resize(max(width, 640), max(height, 480))

    @staticmethod
    def _slider(row, title, low, high, width, default):
        label = QLabel(title)
        themes.set_role(label, 'muted')
        slider = TickSlider(Qt.Orientation.Horizontal, default_value=default)
        slider.setRange(low, high)
        slider.setMaximumWidth(width)
        slider.setAccessibleName(title)
        label.setBuddy(slider)
        row.addWidget(label)
        row.addWidget(slider)
        return slider

    def sync(self):
        self._syncing = True
        project = self.canvas.project
        mw = project['milkyway']
        self.feather.setValue(round(mw['feather'] * 100))
        self.angle.setValue(round(mw.get('angle_degrees', 0.)))
        self.feather.setEnabled(bool(mw['ellipse']))
        self.angle.setEnabled(bool(mw['ellipse']))
        self.accept_edge.setEnabled(bool(self.canvas.suggestion))
        # While a light-blue suggestion waits, its accept button wears the same colour.
        themes.set_role(self.accept_edge, 'suggestion' if self.canvas.suggestion else None)
        self._syncing = False

    def set_brush(self, value):
        self.canvas.radius = value / 1000
        self.canvas.update()

    def set_feather(self, value):
        if not self._syncing and self.canvas.project['milkyway'].get('ellipse'):
            self.canvas.commit(regions.draw_milkyway(self.canvas.project,
                self.canvas.project['milkyway']['ellipse'], feather=value / 100))

    def set_angle(self, value):
        if not self._syncing and self.canvas.project['milkyway'].get('ellipse'):
            self.canvas.commit(regions.draw_milkyway(self.canvas.project,
                self.canvas.project['milkyway']['ellipse'], angle_degrees=float(value)))

    def suggest_edge(self):
        horizon = next((op for op in reversed(self.canvas.project['operations']) if op['kind'] == 'horizon'), None)
        if not horizon:
            self.notice.setText(tr('请先沿地平线点选，并点击“完成地平线”。'))
            return
        image = np.asarray(Image.open(self.reference['preview_path']).convert('RGB'))
        self.canvas.suggestion = regions.suggest_horizon(image, horizon['points'])
        self.canvas.update()
        self.accept_edge.setEnabled(True)
        themes.set_role(self.accept_edge, 'suggestion')
        self.notice.setText(tr('浅蓝虚线是待接受的建议。确认它沿真实地平线后再接受；也可以继续使用原线。'))

    def accept_suggestion(self):
        self.canvas.accept_suggestion()
        self.sync()

    def has_unsaved_changes(self) -> bool:
        """Drawing differs from what the dialog opened with, or horizon points are pending."""
        return bool(self.canvas.pending) or self.canvas.project.get('revision') != self._initial_revision

    def finish(self):
        if self.canvas.pending and self.canvas.mode == 'horizon':
            self.canvas.finish_horizon()
        project = regions.normalize_marks(self.canvas.project)
        self.project = project
        self.accept()

    def reject(self):
        # Cancel, Esc and the window's close button all arrive here. Only a visible
        # editor can have a person's unsaved drawing to protect.
        if self.isVisible() and self.has_unsaved_changes():
            answer = ask_discard(self)
            if answer == USE:
                self.finish()
                return
            if answer != DISCARD:
                return
        super().reject()
