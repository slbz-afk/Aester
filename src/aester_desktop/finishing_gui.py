# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Optional standalone Qt host for the shared finishing service.

No model, image processing, installer or legacy GUI integration lives here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

from PySide6.QtCore import QEvent, QObject, QRunnable, QThreadPool, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImageReader, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from starstack.finishing_components import INSTALL_URL, discover_gimp


@dataclass(frozen=True)
class HostConfig:
    input_root: str
    output_root: str
    cache_root: str | None = None
    gimp: str | None = None


def mcp_configuration(config: HostConfig) -> dict:
    """A copyable configuration, never executed by this window."""
    args = ["-X", "utf8", "-m", "starstack.finishing_mcp", "--input-root", config.input_root,
            "--output-root", config.output_root]
    if config.cache_root:
        args += ["--cache-root", config.cache_root]
    if config.gimp:
        args += ["--gimp", config.gimp]
    # Development entrypoint: a separate client need not inherit this cwd.
    return {"mcpServers": {"aester-retouch": {"command": str(Path(sys.executable).resolve()),
        "args": args, "env": {"PYTHONPATH": str(Path(__file__).resolve().parent.parent)}}}}


class _Signals(QObject):
    done = Signal(object, object)


class _Work(QRunnable):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.signals = _Signals()

    def run(self):
        try:
            result = self.action()
        except Exception as exc:
            self.signals.done.emit(None, exc)
        else:
            self.signals.done.emit(result, None)


class ImageView(QScrollArea):
    """Display an already prepared service view; no TIFF processing here."""
    source_point = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.label = QLabel("暂无图像")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWidget(self.label)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 140)
        self._image = None
        self._record = None
        self.native = False
        self.label.installEventFilter(self)

    def show_image(self, image, native=False, record=None):
        self._image, self.native = image, native
        self._record = record
        self._display()

    def clear(self):
        self._image = None
        self._record = None
        self.label.clear()
        self.label.setText("暂无图像")
        self.label.resize(180, 140)

    def _display(self):
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image)
        dpr = self.devicePixelRatioF()
        if not self.native:
            size = self.viewport().size() * dpr
            pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        pixmap.setDevicePixelRatio(dpr)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.deviceIndependentSize().toSize())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._display()

    def eventFilter(self, watched, event):
        if (watched is self.label and event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton and self._record is not None):
            pixmap = self.label.pixmap()
            if pixmap is not None and not pixmap.isNull():
                size = pixmap.deviceIndependentSize()
                contents = self.label.contentsRect()
                # QLabel centres its pixmap; account for fractional high-DPI
                # rounding without treating the scroll-area margins as image.
                left = contents.x() + (contents.width() - size.width()) / 2
                top = contents.y() + (contents.height() - size.height()) / 2
                x, y = event.position().x() - left, event.position().y() - top
                if 0 <= x <= size.width() and 0 <= y <= size.height():
                    point = preview_point_to_source(self._record, x, y, size.width(), size.height())
                    self.source_point.emit(*point)
                    return True
        return super().eventFilter(watched, event)


def preview_point_to_source(record, x, y, display_width, display_height):
    """Map Qt logical image edges through the service's two preview scales."""
    if display_width <= 0 or display_height <= 0:
        raise ValueError("Displayed preview dimensions must be positive")
    crop = record["source_crop"]
    return (crop[0] + x / display_width * record["width"] / record["scale_x"],
            crop[1] + y / display_height * record["height"] / record["scale_y"])


def reference_color_text(note):
    """Translate known view notices without changing the service record."""
    return (note.replace("embedded ICC converted to sRGB for display", "参考已按内嵌色彩配置转换为 sRGB 预览")
        .replace("untagged reference assumed sRGB for display; original color space unverified",
                 "参考未带色彩配置，预览暂按 sRGB 显示")
        .replace("; transparency displayed over dark gray", "；透明区域显示为深灰背景"))


def _read_view(record):
    # The service owns decoding, cropping, color handling and file validation.
    # Qt only decodes its viewing artifact, off the event thread.
    reader = QImageReader(record["path"])
    reader.setAutoTransform(False)
    image = reader.read()
    if image.isNull():
        raise ValueError("无法显示服务返回的预览：" + reader.errorString())
    return record, image


class FinishingWindow(QMainWindow):
    def __init__(self, service, config: HostConfig, *, owns_service=False,
                 component_checker=discover_gimp):
        super().__init__()
        self.service, self.config = service, config
        self.owns_service, self.component_checker = owns_service, component_checker
        self.session = None
        self.active_job = None
        self._task = None
        self._closing = False
        self._closed_ready = False
        self._cancelling = False
        self._viewed_candidate = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._session_buttons = []
        self._idle_buttons = []
        self.setWindowTitle("Aester · 照片后期")
        self.resize(1180, 840)
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._poll)
        self._sync_controls()
        self._agent_text()

    def _button(self, text, callback, *, session=False, idle=True):
        button = QPushButton(text)
        button.clicked.connect(callback)
        if session:
            self._session_buttons.append(button)
        elif idle:
            self._idle_buttons.append(button)
        return button

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        page = QWidget()
        body = QVBoxLayout(page)
        source_row = QHBoxLayout()
        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("选择允许输入目录中的 sRGB 16 位 TIFF")
        source_row.addWidget(self.source_path, 1)
        source_row.addWidget(self._button("选择照片", self._choose_source))
        self.create_button = self._button("新建后期会话", self.create_session)
        source_row.addWidget(self.create_button)
        self.open_button = self._button("打开会话", self._choose_session)
        source_row.addWidget(self.open_button)
        body.addLayout(source_row)
        self.session_label = QLabel("尚未打开会话")
        self.session_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.session_label.setWordWrap(True)
        body.addWidget(self.session_label)
        intent_row = QHBoxLayout()
        self.intent = QTextEdit()
        self.intent.setPlaceholderText("描述希望改变和保留的效果；外部 Agent 负责理解并规划")
        self.intent.setMaximumHeight(70)
        intent_row.addWidget(self.intent, 1)
        self.intent_button = self._button("保存意图", self.save_intent, session=True)
        intent_row.addWidget(self.intent_button)
        self.refresh_button = self._button("刷新 Agent 结果", self.refresh, session=True)
        intent_row.addWidget(self.refresh_button)
        body.addLayout(intent_row)
        self.intent_conflict = QLabel("")
        self.intent_conflict.setWordWrap(True)
        body.addWidget(self.intent_conflict)
        views = QSplitter()
        self.views = {}
        for key, title in (("source", "原图"), ("reference", "参考 · 不进入真实照片"),
                           ("candidate", "候选 · 待你挑选")):
            group = QGroupBox(title)
            column = QVBoxLayout(group)
            if key == "reference":
                self.references = QComboBox()
                column.addWidget(self.references)
                self.references.activated.connect(lambda _index: self.load_views())
            elif key == "candidate":
                self.candidates = QComboBox()
                column.addWidget(self.candidates)
                self.candidates.activated.connect(lambda _index: self.load_views())
            else:
                self.source_identity = QLabel("原始成片 · 保留不改")
                column.addWidget(self.source_identity)
            view = ImageView()
            self.views[key] = view
            if key != "reference":
                view.source_point.connect(self.select_detail)
            column.addWidget(view, 1)
            views.addWidget(group)
        self.source_identity.setFixedHeight(self.references.sizeHint().height())
        body.addWidget(views, 1)
        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel("可选精确定位：X / Y / 宽 / 高"))
        self.crop = []
        for index in range(4):
            spin = QSpinBox()
            spin.setRange(0 if index < 2 else 1, 1600)
            spin.setValue(0 if index < 2 else 512)
            spin.setMaximumWidth(85)
            self.crop.append(spin)
            crop_row.addWidget(spin)
        self.crop[0].valueChanged.connect(self._crop_limits)
        self.crop[1].valueChanged.connect(self._crop_limits)
        self.crop_button = self._button("查看 100% 局部", self.load_crop, session=True)
        crop_row.addWidget(self.crop_button)
        crop_row.addWidget(self._button("查看整图", self.load_views, session=True))
        body.addLayout(crop_row)
        self.view_note = QLabel("双击原图或候选查看该处的 100% 局部。")
        self.view_note.setWordWrap(True)
        body.addWidget(self.view_note)
        actions = QHBoxLayout()
        self.render_button = self._button("生成当前方案预览", self.render, session=True)
        self.cancel_button = self._button("取消本窗口作业", self.cancel, idle=False)
        self.export_button = self._button("导出 JPEG / TIFF / XCF", self.export, session=True)
        actions.addWidget(self.render_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        actions.addWidget(self.export_button)
        body.addLayout(actions)
        feedback_row = QHBoxLayout()
        self.feedback_text = QLineEdit()
        self.feedback_text.setPlaceholderText("看图后的意见，例如：地景仍太暗，保留天空颜色")
        self.decision = QComboBox()
        for text, value in (("记录意见", "observe"), ("需要修改", "revise"),
                            ("拒绝候选", "reject"), ("接受候选", "accept")):
            self.decision.addItem(text, value)
        self.feedback_button = self._button("提交我的反馈", self.feedback, session=True)
        feedback_row.addWidget(self.feedback_text, 1)
        feedback_row.addWidget(self.decision)
        feedback_row.addWidget(self.feedback_button)
        body.addLayout(feedback_row)
        self.tabs.addTab(page, "看图与反馈")
        setup = QWidget()
        form = QFormLayout(setup)
        self.reference_path = QLineEdit()
        reference_row = QHBoxLayout()
        reference_row.addWidget(self.reference_path, 1)
        reference_row.addWidget(self._button("选择参考", self._choose_reference))
        form.addRow("参考文件", reference_row)
        self.reference_kind = QComboBox()
        for text, value in (("外部作品", "external"), ("生成的审美参考", "generated"),
                            ("原始照片参考", "original")):
            self.reference_kind.addItem(text, value)
        form.addRow("参考来源", self.reference_kind)
        self.reference_note = QLineEdit()
        form.addRow("参考说明", self.reference_note)
        form.addRow(self._button("登记参考", self.add_reference, session=True))
        roots = QLabel("输入：" + self.config.input_root + "\n作品：" + self.config.output_root
                       + "\n会话缓存：" + (self.config.cache_root or self.config.output_root))
        roots.setWordWrap(True)
        roots.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("启动时设置的目录", roots)
        self.gimp_path = QLineEdit(self.config.gimp or "")
        gimp_row = QHBoxLayout()
        gimp_row.addWidget(self.gimp_path, 1)
        gimp_row.addWidget(self._button("选择程序", self._choose_gimp))
        form.addRow("GIMP console", gimp_row)
        checks = QHBoxLayout()
        checks.addWidget(self._button("应用并检查路径", self.apply_gimp))
        checks.addWidget(self._button("查找已有安装", lambda: self.check_gimp(False)))
        checks.addWidget(self._button("检查版本", lambda: self.check_gimp(True)))
        checks.addWidget(self._button("官方安装说明", lambda: QDesktopServices.openUrl(QUrl(INSTALL_URL)), idle=False))
        form.addRow(checks)
        self.component_note = QLabel("GIMP 可选；此窗口不会安装、升级或认领已有安装。")
        self.component_note.setWordWrap(True)
        form.addRow(self.component_note)
        self.tabs.addTab(setup, "参考与 GIMP")
        agent = QWidget()
        agent_layout = QVBoxLayout(agent)
        self.agent_instructions = QTextEdit()
        self.agent_instructions.setReadOnly(True)
        self.agent_config = QTextEdit()
        self.agent_config.setReadOnly(True)
        agent_layout.addWidget(self.agent_instructions)
        agent_layout.addWidget(QLabel("独立 MCP 配置（复制到支持本地 MCP 的客户端；此处不会启动 Agent）"))
        agent_layout.addWidget(self.agent_config)
        agent_layout.addWidget(self._button("复制 MCP 配置", self.copy_config, idle=False))
        self.tabs.addTab(agent, "Agent 接续")
        layout.addWidget(self.tabs)
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        for label in root.findChildren(QLabel):
            label.setTextFormat(Qt.TextFormat.PlainText)

    def _sync_controls(self):
        idle = self._task is None and self.active_job is None and not self._closing
        for button in self._idle_buttons:
            button.setEnabled(idle)
        for button in self._session_buttons:
            button.setEnabled(idle and self.session is not None)
        candidate = self.candidates.currentData() is not None
        self.export_button.setEnabled(idle and candidate)
        self.feedback_button.setEnabled(idle and candidate and self._viewed_candidate == self.candidates.currentData()
                                        and callable(getattr(self.service, "feedback_user", None)))
        self.intent_button.setEnabled(idle and self.session is not None and callable(getattr(self.service, "set_intent", None)))
        self.cancel_button.setEnabled(self.active_job is not None and not self._cancelling and not self._closing)
        self.references.setEnabled(idle)
        self.candidates.setEnabled(idle)
        self.gimp_path.setEnabled(idle)
        for widget in (self.source_path, self.intent, self.reference_path, self.reference_kind,
                       self.reference_note, self.feedback_text, self.decision, *self.crop):
            widget.setEnabled(idle)

    def _run(self, action, success, label, failure=None):
        if self._task is not None:
            return False
        self.status.setText(label)
        work = _Work(action)
        self._task = work
        def finished(result, error):
            self._task = None
            if error is not None:
                self.status.setText("未完成：" + str(error))
                if failure:
                    failure(error)
            else:
                try:
                    success(result)
                except Exception as exc:
                    self.status.setText("未完成：" + str(exc))
            self._sync_controls()
            if self._closing:
                self._finish_close()
        work.signals.done.connect(finished)
        self._sync_controls()
        self._pool.start(work)
        return True

    def _choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择后期源图", self.config.input_root,
                                             "16 位 TIFF (*.tif *.tiff)")
        if path:
            self.source_path.setText(path)

    def _choose_session(self):
        path = QFileDialog.getExistingDirectory(self, "选择后期会话目录",
                         str(Path(self.config.cache_root or self.config.output_root) / "finishing"))
        if path:
            self.open_session(path)

    def _choose_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考图", self.config.input_root,
                                    "参考图 (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.reference_path.setText(path)

    def _choose_gimp(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 GIMP console 程序", "", "程序 (*.exe)")
        if path:
            self.gimp_path.setText(path)

    def create_session(self):
        source, intent = self.source_path.text(), self.intent.toPlainText()
        self._run(lambda: self.service.create(source=source, intent=intent), self._loaded, "正在登记源图…")

    def open_session(self, directory):
        self._run(lambda: self.service.get(session_dir=directory), self._loaded, "正在打开会话…")

    def refresh(self):
        if self.session:
            self.open_session(self.session["directory"])

    def _loaded(self, value):
        self._apply_session(value)
        if not self._closing:
            self.load_views()

    def _apply_session(self, value):
        previous_directory = self.session["directory"] if self.session else None
        old_candidate = self.candidates.currentData()
        old_reference = self.references.currentData()
        self.session = value
        self.session_label.setText("会话：" + value["directory"] + "  ·  方案版本 " + str(value["revision"]))
        self.source_path.setText(value["source"]["path"])
        self.intent.setPlainText(value["intent"])
        self.intent_conflict.clear()
        self.candidates.clear()
        self.candidates.addItem("尚未选择候选", None)
        for item in value["candidates"]:
            if item["status"] == "validated":
                self.candidates.addItem("候选 " + str(self.candidates.count()) + " · 方案版本 " + str(item["revision"]), item["id"])
                self.candidates.setItemData(self.candidates.count() - 1, item["id"], Qt.ItemDataRole.ToolTipRole)
        index = self.candidates.findData(old_candidate)
        self.candidates.setCurrentIndex(index if index > 0 else self.candidates.count() - 1)
        self.references.clear()
        self.references.addItem("未选择参考", None)
        labels = {"generated": "生成参考", "external": "外部作品", "original": "原始参考"}
        for item in value["references"]:
            self.references.addItem(labels.get(item["kind"], item["kind"]) + " · " + item.get("note", ""), item["id"])
        index = self.references.findData(old_reference)
        self.references.setCurrentIndex(index if index > 0 else self.references.count() - 1)
        self.crop[0].setMaximum(value["source"]["width"] - 1)
        self.crop[1].setMaximum(value["source"]["height"] - 1)
        if previous_directory != value["directory"]:
            self.crop[0].setValue(0)
            self.crop[1].setValue(0)
            for view in self.views.values():
                view.clear()
        self._crop_limits()
        self._agent_text()
        self.status.setText("会话已读取；请查看图像后决定。")

    def _crop_limits(self):
        if self.session:
            for axis in (0, 1):
                size = self.session["source"]["width" if axis == 0 else "height"]
                self.crop[axis + 2].setMaximum(min(1600, size - self.crop[axis].value()))

    def load_crop(self):
        self.load_views(crop=[spin.value() for spin in self.crop])

    def select_detail(self, source_x, source_y):
        if not self.session or self._task is not None or self.active_job is not None or self._closing:
            return
        width, height = self.crop[2].value(), self.crop[3].value()
        source = self.session["source"]
        x = max(0, min(source["width"] - width, math.floor(source_x - width / 2)))
        y = max(0, min(source["height"] - height, math.floor(source_y - height / 2)))
        self.crop[0].setValue(x)
        self.crop[1].setValue(y)
        self.crop[2].setValue(width)
        self.crop[3].setValue(height)
        self.load_crop()

    def load_views(self, _checked=False, *, crop=None):
        if not self.session or self._closing:
            return
        directory = self.session["directory"]
        candidate, reference = self.candidates.currentData(), self.references.currentData()
        self._viewed_candidate = None
        for view in self.views.values():
            view.clear()
        def read():
            values = {"source": _read_view(self.service.preview(session_dir=directory, crop=crop))}
            if candidate:
                values["candidate"] = _read_view(self.service.preview(session_dir=directory, candidate_id=candidate, crop=crop))
            if reference:
                values["reference"] = _read_view(self.service.reference_preview(session_dir=directory, reference_id=reference))
            if crop is not None:
                for key in ("source", "candidate"):
                    if key in values and not values[key][0].get("native_scale"):
                        raise ValueError("服务返回了缩放局部，不能显示为 100%；请选择不超过 1600 像素的局部")
            return values
        def show(values):
            for key, view in self.views.items():
                if key in values:
                    record, image = values[key]
                    view.show_image(image, native=crop is not None and key != "reference",
                                    record=record if key != "reference" else None)
                else:
                    view.clear()
            self.view_note.setText("100% 局部；点击“查看整图”返回。" if crop is not None
                                   else "双击原图或候选查看该处的 100% 局部。")
            if "reference" in values:
                color_note = values["reference"][0].get("color_note", "")
                self.view_note.setText(self.view_note.text() + "\n" + reference_color_text(color_note))
                self.view_note.setToolTip(color_note)
            else:
                self.view_note.setToolTip("")
            self._viewed_candidate = candidate
            self.status.setText("预览已读取；参考不会进入渲染。")
        self._run(read, show, "正在读取预览…")

    def save_intent(self):
        if not self.session or not callable(getattr(self.service, "set_intent", None)):
            return
        directory, text = self.session["directory"], self.intent.toPlainText()
        revision = self.session.get("intent_revision", 0)
        def failed(error):
            from starstack.finishing_session import RevisionConflict
            if isinstance(error, RevisionConflict):
                def conflict(value):
                    self._apply_session(value)
                    self.intent.setPlainText(text)
                    self.intent_conflict.setText("其他窗口的最新意图：" + value["intent"]
                        + "\n你的未保存草稿仍在编辑框中。请比较并编辑，再点击保存；下一次保存会基于最新意图版本。")
                    self.status.setText("意图版本冲突；保留草稿，请比较后再次明确保存。")
                self._run(lambda: self.service.get(session_dir=directory),
                    conflict,
                    "意图冲突，正在刷新…")
        self._run(lambda: self.service.set_intent(session_dir=directory, intent=text,
                      expected_intent_revision=revision), self._apply_session, "正在保存意图…", failed)

    def add_reference(self):
        if self.session:
            directory, path = self.session["directory"], self.reference_path.text()
            kind, note = self.reference_kind.currentData(), self.reference_note.text()
            self._run(lambda: self.service.add_reference(session_dir=directory, path=path, kind=kind, note=note),
                      self._loaded, "正在登记参考…")

    def render(self):
        if not self.session or self.active_job:
            return
        directory, revision = self.session["directory"], self.session["revision"]
        def submitted(value):
            self.active_job = value["job_id"]
            self._cancelling = False
            self.status.setText("当前配方已提交渲染；可以取消。")
            self.timer.start()
        self._run(lambda: self.service.submit_render(session_dir=directory, expected_revision=revision),
                  submitted, "正在提交当前配方…")

    def _poll(self):
        if self.active_job and self._task is None:
            job_id = self.active_job
            def done(value):
                state = value["status"]
                self.status.setText({"running": "正在渲染…", "cancelling": "正在取消本窗口作业…"}.get(state, state))
                if state in {"validated", "cancelled", "failed"}:
                    self.timer.stop()
                    self.active_job = None
                    self._cancelling = False
                    if state == "validated" and not self._closing:
                        self.refresh()
                    elif state != "validated":
                        self.status.setText("作业" + ("已取消" if state == "cancelled" else "失败") + "：" + value.get("error", ""))
            def failed(error):
                self.timer.stop()
                self.status.setText("无法读取作业状态；仍保留本窗口作业标识，可取消后关闭：" + str(error))
            self._run(lambda: self.service.render_status(job_id=job_id), done, "读取渲染状态…", failed)

    def cancel(self):
        if self.active_job and not self._cancelling:
            try:
                # The public cancellation call only sets an Event and snapshots
                # its Future; it never waits for the GIMP process to exit.
                self.service.cancel_render(job_id=self.active_job)
                self._cancelling = True
                self.status.setText("正在取消本窗口作业…")
                self.timer.start()
            except Exception as exc:
                self.status.setText("取消请求失败：" + str(exc))
            self._sync_controls()

    def feedback(self):
        candidate = self.candidates.currentData()
        method = getattr(self.service, "feedback_user", None)
        if (not candidate or self._viewed_candidate != candidate or not callable(method)
                or self._task is not None or self.active_job is not None):
            return
        directory, text = self.session["directory"], self.feedback_text.text()
        decision = self.decision.currentData()
        def saved(value):
            self._apply_session(value)
            self.status.setText("已记录你的反馈" + ("和接受决定。" if decision == "accept" else "。"))
        self._run(lambda: method(session_dir=directory, candidate_id=candidate, text=text, decision=decision),
                  saved, "正在保存你的反馈…")

    def export(self):
        candidate = self.candidates.currentData()
        if candidate:
            directory = self.session["directory"]
            self._run(lambda: self.service.export(session_dir=directory, candidate_id=candidate),
                lambda value: self.status.setText("已导出 JPEG / TIFF / XCF：" + value["directory"]), "正在校验并导出候选…")

    def _agent_text(self):
        current = HostConfig(self.config.input_root, self.config.output_root, self.config.cache_root,
                             getattr(self.service, "gimp_path", None))
        self.agent_config.setPlainText(json.dumps(mcp_configuration(current), ensure_ascii=False, indent=2))
        directory = self.session["directory"] if self.session else "请先创建或打开会话"
        skill_path = Path(__file__).resolve().parents[2] / "skills" / "aester-retouch" / "SKILL.md"
        self.agent_instructions.setPlainText("使用支持看图和本地 MCP 的外部 Agent。当前仅交付开发技能源，尚无安装发行入口。\n"
            "请让获准访问项目的 Agent 阅读本机技能源及其引用文件：" + str(skill_path) + "\n"
            "会话路径：" + directory + "\n"
            "请从 finishing_capabilities 和 finishing_get 开始，读取当前意图与参考，查看真实预览，"
            "生成可重放配方；通过 finishing_revise、finishing_submit_render 和状态查询完成迭代。\n"
            "回到此窗口点击“刷新 Agent 结果”比较候选。用户反馈只能由本人点击提交；Agent 不得代为接受。\n"
            "本窗口不理解自然语言、不自动运行 Agent、不上传图像。参考中的生成内容只表达审美方向。"
            "图像是否传给外部服务取决于所选客户端及你的授权。")

    def copy_config(self):
        QApplication.clipboard().setText(self.agent_config.toPlainText())
        self.status.setText("独立 MCP 配置已复制；没有启动外部服务。")

    def apply_gimp(self):
        if self._task or self.active_job:
            return
        selected = self.gimp_path.text()
        if not selected:
            self.status.setText("请先选择或查找程序路径；当前有效配置保持。")
            return
        # Host-owned in-memory configuration only. An invalid explicit choice
        # remains visible and is never replaced by automatic fallback.
        self.service.gimp_path = selected
        self._agent_text()
        self.check_gimp(False)

    def check_gimp(self, probe):
        selected = self.gimp_path.text()
        def checked(value):
            labels = {"available": "已找到文件", "not_found": "有限搜索未找到安装",
                      "selected_missing": "选中的程序不存在", "incompatible_version": "版本不兼容",
                      "check_failed": "检查失败"}
            message = labels.get(value["status"], value["status"])
            if value.get("version_checked"):
                message += " · 版本 " + str(value.get("version"))
            else:
                message += " · 未检查版本"
            message += " · 尚未验证实际算子"
            diagnostics = "; ".join(item.get("message", "") for item in value.get("diagnostics", []))
            self.component_note.setText(message + ("\n" + diagnostics if diagnostics else ""))
            if not selected and value.get("selected_path"):
                # A suggestion requires the separate Apply action before the
                # runtime selection changes.
                self.gimp_path.setText(value["selected_path"])
                self.status.setText("找到候选路径；点击“应用并检查路径”后用于本窗口。")
            else:
                self.status.setText(message)
        self._run(lambda: self.component_checker(selected_path=selected if selected else None, probe=probe),
                  checked, "正在检查 GIMP 版本…" if probe else "正在只读检查路径…")

    def closeEvent(self, event):
        if self._closed_ready:
            event.accept()
            return
        event.ignore()
        self._closing = True
        self.cancel()
        self._sync_controls()
        self.status.setText("正在结束本窗口作业…")
        self._finish_close()

    def _finish_close(self):
        if self._closed_ready:
            return
        if self._task is not None:
            return
        if self.active_job:
            self.cancel()
            self.timer.start()
            return
        self.timer.stop()
        if self.owns_service:
            self.owns_service = False
            self._run(self.service.close, lambda _value: self._close_ready(), "正在释放本窗口服务…")
        else:
            self._close_ready()

    def _close_ready(self):
        self._closed_ready = True
        self.close()


def main(argv=None):
    from starstack.finishing_cli import add_host_arguments, configured_service
    parser = argparse.ArgumentParser(description="Aester 可选独立照片后期窗口")
    add_host_arguments(parser)
    parser.add_argument("--session", help="打开允许会话根内的既有后期会话")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    try:
        service = configured_service(args)
        workspace = service.workspace
        config = HostConfig(str(workspace.input_root),
            str(workspace.delivery_root or workspace.output_root),
            str(workspace.output_root) if workspace.delivery_root else None, service.gimp_path)
        window = FinishingWindow(service, config, owns_service=True)
    except Exception as exc:
        print("Finishing GUI startup failed: " + str(exc), file=sys.stderr)
        return 1
    window.show()
    if args.session:
        window.open_session(args.session)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
