# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Small desktop forms for the same explicit advanced operations as MCP."""
from __future__ import annotations
from starstack.i18n import tr, translate_message

from pathlib import Path
import uuid

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton

from . import themes
from .ui_widgets import ErrorIcon


def _button(text, role=None):
    """Form buttons never become Enter's implicit default."""
    button = QPushButton(text)
    button.setAutoDefault(False)
    if role:
        themes.set_role(button, role)
    return button


class AdvancedTools:
    def __init__(self, owner):
        self.owner = owner
        self.fields = {}
        self.controls = []
        self.statuses = {}
        self.status_icons = {}
        self.outputs = {}
        self.candidates = []
        self._trails_directory = None
        self._milkyway_confirmation = None
        self._build_background()
        self._build_milkyway()
        self._build_layers()
        self._build_trails()

    def _form(self, key, title):
        form = self.owner._dialog_form(key, title, 720, 650)
        self.fields[key] = {}
        status = self.owner._label(tr('等待设置'))
        status.setTextFormat(Qt.TextFormat.PlainText)
        status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statuses[key] = status
        return form

    def _path(self, form, key, name, label, *, directory=False, optional=False):
        def pick():
            field = self.fields[key][name]
            initial = field.text() or self.owner.output_dir.text()
            if directory:
                selected = QFileDialog.getExistingDirectory(self.owner, label, initial)
            else:
                selected, _ = QFileDialog.getOpenFileName(self.owner, label, initial, tr('图像或记录 (*.tif *.tiff *.png *.json);;所有文件 (*)'))
            if selected:
                field.setText(selected)
                field.setCursorPosition(len(field.text()))
                field.deselect()
        field = self.owner._path_field(form, label + (tr(' · 可选') if optional else ""), tr('选择文件夹') if directory else tr('选择文件'), key + name, pick)
        self.fields[key][name] = field
        self.controls.extend([field, self.owner._path_buttons[-1]])
        return field

    def _choice(self, form, key, name, title, options):
        control = self.owner._choice_control(form, title, key + name, options)
        self.fields[key][name] = control
        self.controls.append(control)
        return control

    def _number(self, form, key, name, title, lo, hi, value, step=5, suffix=" %"):
        control = self.owner._color_control(form, title, key + name, lo, hi, value, step, suffix)
        self.fields[key][name] = control
        self.controls.append(control)
        return control

    def _buttons(self, form, key, title, callback):
        run = _button(title, "primary")
        run.setObjectName(key + "Run")
        run.clicked.connect(callback)
        form.addWidget(run)
        self.controls.append(run)
        status_row = QHBoxLayout()
        icon = ErrorIcon(16, accessible_name=tr('错误'))
        icon.hide()
        self.status_icons[key] = icon
        status_row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        status_row.addWidget(self.statuses[key], 1)
        form.addLayout(status_row)
        open_button = _button(tr('打开本次结果'), "ghost")
        open_button.clicked.connect(lambda: self.open_result(key))
        form.addWidget(open_button)
        form.addStretch(1)

    def _values(self, key):
        result = {}
        for name, control in self.fields[key].items():
            if isinstance(control, QComboBox):
                result[name] = control.currentData()
            elif hasattr(control, "value"):
                result[name] = control.value()
            elif isinstance(control, QCheckBox):
                result[name] = control.isChecked()
            else:
                result[name] = control.text().strip()
        return result

    def _run(self, key, operation, callback=None):
        if self.owner._busy or self.owner.job_state in {"queued", "running", "cancel_requested"}:
            return
        self.owner._busy = True
        self._mark_error(key, False)
        self.statuses[key].setText(tr('正在处理…'))
        self.owner.message_label.setText(tr('正在处理高级操作…'))
        self.owner._update_controls()

        def done(result):
            self.owner._busy = False
            if result.get("state") != "succeeded":
                raise ValueError(result.get("error") or tr('操作未成功完成：') + str(result.get("state") or tr('无完成状态')))
            directory = result.get("processing_directory") or result.get("directory") or result.get("job_dir")
            if directory:
                self.outputs[key] = str(directory)
            self.statuses[key].setText(tr('处理完成') + ("\n" + str(directory) if directory else ""))
            if callback:
                callback(result)
            artifacts = result.get("artifacts") or []
            if isinstance(artifacts, dict):
                artifacts = list(artifacts.values())
            jpgs = [item for item in artifacts if isinstance(item, dict)
                    and str(item.get("path", "")).lower().endswith((".jpg", ".jpeg"))]
            if key == "background":
                jpgs.sort(key=lambda item: Path(str(item["path"])).name != "after.jpg")
            if jpgs:
                picture = Path(jpgs[0]["path"])
                self.owner._view_mode = "advanced"
                self.owner._advanced_result_dir = str(picture.parent)
                self.owner.preview_note.setText(tr('高级操作派生 · 基础成片保持'))
                self.owner._load_preview_image(picture, auto_transform=True)
            self.owner.message_label.setText(tr('高级操作已完成；可在相应窗口打开结果。'))
            self.owner._update_controls()
        self.owner._run_async("advanced:" + key, operation, done)

    def set_busy(self, busy):
        for control in self.controls:
            control.setEnabled(not busy)

    def _mark_error(self, key, error):
        """Errors read in zhu with an icon beside the message (the text itself is unchanged)."""
        status, icon = self.statuses.get(key), self.status_icons.get(key)
        if status is not None and themes.role(status) != ("error" if error else None):
            # An object-name colour rule (e.g. QLabel#muted) would outrank the role selector.
            if error:
                status.setProperty("aesterObjectName", status.objectName())
                status.setObjectName("")
            else:
                status.setObjectName(status.property("aesterObjectName") or status.objectName())
            themes.set_role(status, "error" if error else None)
        if icon is not None:
            icon.setVisible(error)

    def failed(self, key, message):
        if key in self.statuses:
            self._mark_error(key, True)
            self.statuses[key].setText(tr('未完成：') + translate_message(message))

    def open_result(self, key):
        directory = self.outputs.get(key)
        if directory and Path(directory).is_dir():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(directory)):
                self.statuses[key].setText(tr('请手动打开：') + directory)
        else:
            self.statuses[key].setText(tr('尚无本次结果。'))

    def show(self, key):
        fields = self.fields[key]
        if "job" in fields and self.owner.job_dir and not fields["job"].text():
            fields["job"].setText(self.owner.job_dir)
        if key == "layers" and self.owner.star_separation.text():
            fields["separation"].setText(self.owner.star_separation.text())
        self.owner._show_advanced(key)

    def _build_background(self):
        key = "background"
        form = self._form(key, tr('背景协调'))
        self._path(form, key, "job", tr('已完成的堆栈任务'), directory=True)
        self._path(form, key, "samples", tr('背景采样蒙版 PNG'))
        self._choice(form, key, "mode", tr('处理方式'), [(tr('渐变与偏色'), "both"), (tr('仅渐变'), "gradient"), (tr('仅偏色'), "neutralize")])
        self._number(form, key, "strength", tr('力度'), 0, 100, 100)
        form.addWidget(themes.set_role(self.owner._label(tr('未校色输入背景派生 · 另存，不覆盖默认成片')), "hint"))
        self._buttons(form, key, tr('另存背景协调结果'), self.run_background)

    def run_background(self):
        from starstack.background_job import render_background
        values = self._values("background")
        self._run("background", lambda: render_background(values["job"], values["samples"],
                  mode=values["mode"], strength=values["strength"] / 100))

    def _build_milkyway(self):
        key = "milkyway"
        form = self._form(key, tr('银河局部增强'))
        for name, label, optional in (("source", tr('成片 TIFF'), False), ("sky", tr('天空权重蒙版'), False),
                                     ("protect", tr('亮迹保护蒙版'), False), ("milkyway", tr('银河渐变蒙版'), False),
                                     ("evidence", tr('已有银河确认记录'), True),
                                     ("transient_background", tr('配对的无亮迹背景'), True),
                                     ("starless_background", tr('已绑定的无星背景'), True)):
            self._path(form, key, name, label, optional=optional)
        confirm = QCheckBox(tr('我已确认选区覆盖可见银河'))
        confirm.setObjectName("milkywayHumanConfirmation")
        form.addWidget(confirm)
        self.fields[key]["confirm"] = confirm
        self.controls.append(confirm)
        confirm.toggled.connect(self._confirm_milkyway)
        for name in ("source", "milkyway"):
            self.fields[key][name].textChanged.connect(lambda: confirm.setChecked(False))
        self._number(form, key, "strength", tr('总力度'), 0, 100, 60)
        self._number(form, key, "brightness", tr('背景提亮'), 0, 10, 2, .5)
        self._number(form, key, "contrast", tr('对比'), 0, 100, 60)
        self._number(form, key, "structure", tr('结构'), 0, 100, 90)
        self._number(form, key, "color", tr('色彩'), 0, 100, 35)
        self._buttons(form, key, tr('另存银河增强结果'), self.run_milkyway)

    def _confirm_milkyway(self, checked):
        self._milkyway_confirmation = None
        if not checked:
            return
        from starstack.photographic_job import milkyway_confirmation
        fields = self.fields["milkyway"]
        source, mask = fields["source"].text().strip(), fields["milkyway"].text().strip()
        try:
            if not source or not mask or not Path(source).is_file() or not Path(mask).is_file():
                raise ValueError(tr('请先选择实际源图和银河蒙版。'))
            self._milkyway_confirmation = milkyway_confirmation(source, mask)
        except (OSError, ValueError) as exc:
            fields["confirm"].setChecked(False)
            self.statuses["milkyway"].setText(tr('未确认：') + translate_message(str(exc)))

    def run_milkyway(self):
        from starstack.photographic_job import finish_milkyway
        values = self._values("milkyway")
        output = self.owner.output_dir.text().strip()
        confirmation = dict(self._milkyway_confirmation or {})
        # The main window's JPEG quality, read now (a window without the setting keeps the historical 97).
        quality = self.owner.jpeg_quality() if callable(getattr(self.owner, "jpeg_quality", None)) else None
        encoding = {} if quality is None else {"jpeg_quality": quality}
        if not values["confirm"] and not values["evidence"]:
            self.statuses["milkyway"].setText(tr('需要已有确认记录，或由你明确确认可见银河；尚未运行。'))
            return

        def run():
            evidence = values["evidence"] or None
            if values["confirm"]:
                from starstack.photographic_job import write_milkyway_review, MilkywayConfirmationChanged
                directory = Path(output) / ("gui-milkyway-review-" + uuid.uuid4().hex[:12])
                try:
                    evidence = write_milkyway_review(confirmation, values["source"], values["milkyway"], directory,
                        reviewer="GUI user", rationale="用户在银河增强窗口显式确认所选源图的银河选区。")
                except MilkywayConfirmationChanged:
                    raise ValueError(tr('源图或银河蒙版在确认后已改变，请重新查看并确认。')) from None
            return finish_milkyway(values["source"], output, values["sky"], values["protect"], values["milkyway"],
                milkyway_evidence=evidence, transient_background=values["transient_background"] or None,
                starless_background=values["starless_background"] or None,
                **{key: values[key] / 100 for key in ("brightness", "contrast", "structure", "color", "strength")},
                **encoding)
        def done(result):
            gate = result.get("subject_gate") or {}
            if not gate.get("enabled"):
                self.statuses["milkyway"].setText(tr('已跳过银河增强：') + translate_message(gate.get("reason") or tr('确认依据不足')))
        self._run("milkyway", run, done)

    def _build_layers(self):
        key = "layers"
        form = self._form(key, tr('导出编辑图层'))
        self._path(form, key, "separation", tr('分层结果记录'))
        self._buttons(form, key, tr('导出 TIFF 图层包'), self.run_layers)

    def run_layers(self):
        from starstack.star_layer_export import export_star_layers
        values = self._values("layers")
        output = self.owner.output_dir.text().strip()
        self._run("layers", lambda: export_star_layers(values["separation"], output))

    def _build_trails(self):
        key = "trails"
        form = self._form(key, tr('亮迹复核'))
        source = self._path(form, key, "job", tr('已完成的堆栈任务'), directory=True)
        load = _button(tr('读取候选亮迹'))
        load.clicked.connect(self.load_trails)
        form.addWidget(load)
        self.controls.append(load)
        self.trail_candidates = QComboBox()
        self.trail_candidates.setObjectName("trailCandidates")
        self.trail_candidates.currentIndexChanged.connect(self.show_trail_candidate)
        source.textChanged.connect(self._trails_source_changed)
        form.addWidget(self.trail_candidates)
        self.controls.append(self.trail_candidates)
        self.trail_preview = QLabel(tr('选择候选后显示原位置参考'))
        self.trail_preview.setMinimumHeight(160)
        self.trail_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form.addWidget(self.trail_preview)
        self.trail_summary = self.owner._label("")
        form.addWidget(self.trail_summary)
        self._choice(form, key, "label", tr('人工身份'), [(tr('不更改'), "unchanged"), (tr('流星'), "meteor"), (tr('卫星'), "satellite"),
                          (tr('飞机'), "aircraft"), (tr('伪影'), "artifact"), (tr('未知'), "unknown"), (tr('撤销人工身份'), None)])
        self._choice(form, key, "decision", tr('成片处置'), [(tr('不更改'), "unchanged"), (tr('保留'), "keep"), (tr('省略'), "omit"), (tr('恢复自动'), None)])
        self._buttons(form, key, tr('保存此条复核并重做亮迹层'), self.run_trail_review)

    @staticmethod
    def _read_trails(job_dir):
        from starstack.trail_review_session import list_trail_candidates
        return list_trail_candidates(job_dir)

    def _trails_source_changed(self):
        self._trails_directory = None
        self.candidates = []
        self.trail_candidates.clear()

    @staticmethod
    def _check_review_scope(directory):
        from starstack.workflow_edit import trail_rederive_scope
        reason = trail_rederive_scope(directory)
        if reason == "workflow_not_succeeded":
            raise ValueError(tr('完整出图任务尚未成功结束，不能重做其中的亮迹层。'))
        if reason == "frozen_job_index":
            raise ValueError(tr('此旧完整任务冻结了原作业索引，仅能读取复核；需新快照任务才能追加亮迹派生。'))
        if reason == "snapshot_missing":
            raise ValueError(tr('完整任务的不可变堆栈快照缺失。'))

    def _trails_loaded(self, result):
        self._trails_directory = str(Path(result["job_dir"]).resolve())
        self.candidates = result.get("candidates") or []
        self.trail_candidates.clear()
        for candidate in self.candidates:
            self.trail_candidates.addItem(str(candidate["id"]) + " · " + str(candidate.get("frame") or ""), candidate["id"])
        self.statuses["trails"].setText(tr("已读取 {0} 条候选；分类与保留意图分开记录。", len(self.candidates)))
        self.show_trail_candidate()

    def load_trails(self):
        values = self._values("trails")
        self._run("trails", lambda: self._read_trails(values["job"]), self._trails_loaded)

    def show_trail_candidate(self):
        index = self.trail_candidates.currentIndex()
        if not 0 <= index < len(self.candidates):
            self.trail_preview.clear()
            self.trail_summary.clear()
            return
        candidate = self.candidates[index]
        labels = {"meteor": tr('流星'), "satellite": tr('卫星'), "aircraft": tr('飞机'), "artifact": tr('伪影'), "unknown": tr('未知')}
        self.trail_summary.setText(tr('身份：') + labels.get(candidate.get("effective_label"), tr('未知')) + tr(' · 处置：') +
                                  {"keep": tr('保留'), "omit": tr('省略')}.get(candidate.get("decision"), tr('待定')) +
                                  "\n" + str(candidate.get("protection_reason") or candidate.get("decision_reason") or ""))
        path = candidate.get("review_png")
        pixmap = QPixmap(str(path)) if path else QPixmap()
        if pixmap.isNull():
            self.trail_preview.setText(tr('此条暂无位置预览'))
        else:
            self.trail_preview.setPixmap(pixmap.scaled(620, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.fields["trails"]["label"].setCurrentIndex(0)
        self.fields["trails"]["decision"].setCurrentIndex(0)

    def run_trail_review(self):
        from starstack.core import recompose_transients
        index = self.trail_candidates.currentIndex()
        values = self._values("trails")
        if not 0 <= index < len(self.candidates):
            self.statuses["trails"].setText(tr('请先读取并选择一条候选。'))
            return
        if values["label"] == values["decision"] == "unchanged":
            self.statuses["trails"].setText(tr('尚未选择要保存的变更。'))
            return
        directory = Path(values["job"]).resolve()
        if str(directory) != self._trails_directory:
            self.statuses["trails"].setText(tr('任务路径已改变，请重新读取候选再复核。'))
            return
        candidate_id = self.candidates[index]["id"]
        labels = None if values["label"] == "unchanged" else {candidate_id: values["label"]}
        overrides = None if values["decision"] == "unchanged" else {candidate_id: values["decision"]}
        def run():
            self._check_review_scope(directory)
            recompose_transients(str(directory), labels=labels, overrides=overrides)
            return self._read_trails(str(directory))
        self._run("trails", run, self._trails_loaded)
