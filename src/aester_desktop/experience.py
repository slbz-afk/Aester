# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Aester setup dialogs. Heavy work stays in the shared window worker pool."""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from starstack.i18n import tr, get_language, translate_message

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTextBrowser, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout)

from . import themes, ui_widgets

# The job list on 作品与缓存 (column titles and plan states, as tr() sources).
CACHE_COLUMNS = ('任务', '日期', '作品文件夹', '占用', '可释放', '状态')
CACHE_STATES = {'cleanable': '可清理', 'partial': '部分已清理，可再次清理', 'cleaned': '已清理',
                'nothing': '没有可清理的文件', 'running': '正在运行或使用中'}


def _literal(text):
    """Menu and button text shows '&' as written (Qt reads a single '&' as a mnemonic)."""
    return text.replace('&', '&&')


def ask_cleanup(parent, text) -> bool:
    """Confirm deleting intermediate files; not cleaning is the default (Enter and Esc keep everything)."""
    box = QMessageBox(themes.message_icon(parent), tr('清理中间文件'), text, QMessageBox.StandardButton.NoButton, parent)
    clean = box.addButton(tr('清理中间文件'), QMessageBox.ButtonRole.DestructiveRole)
    keep = box.addButton(tr('暂不清理'), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep)
    box.setEscapeButton(keep)
    box.exec()
    return box.clickedButton() is clean


class Experience(QObject):
    progress = Signal(str)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.progress.connect(self._progress)
        self._installing = False
        self._installing_retouch = False
        self._testing = False
        self._checking_components = False
        self._listing_cache = False
        self._cleaning_cache = False
        self._cache_serial = 0
        self._cache_cancel = None              # threading.Event of the job listing in flight
        self._cache_refresh_pending = False    # a refresh asked for while a listing ran
        self._model_source = None
        self._component_cancel = None
        self._pending_nox_install = False
        self._config = None
        self._component_result = None
        self._next_python = None
        self._restart_required = False
        self._gimp_path = str(window._preferences.value('retouch/gimp_path', '')
                              if window._preferences else '')
        from .backend_preferences import apply_backend_defaults
        self.backend_migrations = apply_backend_defaults(window)
        restored = [change for change in self.backend_migrations if change['previous']]
        if restored:
            names = {'siril_path': 'Siril', 'star_model_path': tr('去星模型')}
            window.message_label.setText(tr('已自动更新失效的组件位置：') +
                '、'.join(names[change['key']] for change in restored) + tr('。可在设置 → 处理后端中查看。'))
        from starstack.storage import default_cache_root
        self.cache_dir = QLineEdit(os.environ.get('AESTER_CACHE_ROOT') or str(default_cache_root()))
        self.cache_dir.setObjectName('cacheDirectory')
        if window._preferences:
            self.cache_dir.setText(str(window._preferences.value('storage/cache_root', self.cache_dir.text())))
        window.cache_dir = self.cache_dir
        settings = window.menu_actions['settings'].parent()
        # '&' marks a mnemonic in menu and button text; the English name contains a literal one.
        for key, title, callback in [('components', _literal(tr('组件与提示词管理…')), self.show_components)]:
            action = settings.addAction(title)
            action.triggered.connect(callback)
            window.menu_actions[key] = action
        help_menu = window.menu_actions['help'].parent()
        action = help_menu.addAction(tr('欢迎与开始…')); action.triggered.connect(self.show_welcome)
        window.menu_actions['welcome'] = action

    def work_root(self) -> str:
        if not self.cache_dir.text().strip():
            raise ValueError(tr('请在设置 → 作品与缓存中选择缓存目录'))
        return str(Path(self.cache_dir.text().strip()).expanduser().absolute())

    def _dialog(self, key, title, width=700, height=500):
        if key in self.window._advanced_dialogs:
            return None
        return self.window._dialog_form(key, title, width, height)

    def _show(self, key):
        self.window._show_advanced(key)

    @staticmethod
    def _label(text, role=None):
        label = QLabel(text); label.setWordWrap(True)
        if role:
            themes.set_role(label, role)
        return label

    def _pick_cache(self):
        selected = QFileDialog.getExistingDirectory(self.window, tr('选择工作缓存文件夹'), self.cache_dir.text())
        if selected:
            self.cache_dir.setText(selected)
            self._save_cache()

    def _save_cache(self):
        from starstack.onboarding import checked_directory as _directory
        try:
            cache = _directory(self.cache_dir.text().strip())
            out = _directory(self.window.output_dir.text().strip())
            if cache.is_relative_to(out) or out.is_relative_to(cache):
                raise ValueError(tr('缓存和作品目录不能相互包含'))
            if self.window._preferences:
                self.window._preferences.setValue('storage/cache_root', str(cache))
                self.window._preferences.sync()
            self.storage_note.setText(tr('新任务使用此缓存位置。已有任务保持原位置；不会自动搬移或删除。'))
            self.refresh_cache_jobs()      # the list covers the default root and this preference
        except (ValueError, OSError) as error:
            self.storage_note.setText(tr('操作未完成：') + translate_message(str(error)))

    def show_storage(self):
        form = self._dialog('storage', tr('作品与工作缓存'), 900, 640)
        if form is not None:
            form.addWidget(self._label(tr('作品：仅保存 JPEG、16位成片 TIFF 和浮点线性母版 TIFF；处理记录保存在工作缓存中。')))
            self.product_location = self._label('', 'mono'); form.addWidget(self.product_location)
            form.addWidget(self._label(tr('缓存：解码、配准、天空／地景和复核所需文件，可能占用几十GB。建议使用空间充足的本地SSD。')))
            row = QHBoxLayout(); row.addWidget(self.cache_dir, 1)
            choose = QPushButton(tr('选择缓存位置')); choose.clicked.connect(self._pick_cache); row.addWidget(choose)
            form.addLayout(row)
            self.cache_dir.editingFinished.connect(self._save_cache)
            self.storage_note = self._label(tr('完成的作品可以独立查看和编辑。清理工作缓存后，原任务可能无法恢复、重算分区或复核亮迹。'), 'hint')
            form.addWidget(self.storage_note)
            form.addWidget(self._label(tr('已完成任务的中间文件'), 'section'))
            form.addWidget(self._label(tr('只删除可由原片重建的逐帧中间文件（转换帧、配准帧、原片副本等）；作品、原片、母版、处理记录和日志都保留。清理后可以查看作品、另存新预览和重做星点整理，但不能再复核亮迹候选。'), 'muted'))
            self.cache_jobs = QTreeWidget(); self.cache_jobs.setObjectName('cacheJobs')
            self.cache_jobs.setColumnCount(len(CACHE_COLUMNS))
            self.cache_jobs.setHeaderLabels([tr(title) for title in CACHE_COLUMNS])
            self.cache_jobs.setRootIsDecorated(False)
            self.cache_jobs.setUniformRowHeights(True)
            self.cache_jobs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.cache_jobs.setMinimumHeight(200)
            self.cache_jobs.itemSelectionChanged.connect(self.update_storage_controls)
            form.addWidget(self.cache_jobs, 1)
            buttons = QHBoxLayout()
            self.refresh_cache_button = QPushButton(tr('刷新列表')); self.refresh_cache_button.setObjectName('refreshCacheJobs')
            self.refresh_cache_button.clicked.connect(self.refresh_cache_jobs); buttons.addWidget(self.refresh_cache_button)
            buttons.addStretch(1)
            self.clean_cache_button = QPushButton(tr('清理所选任务的中间文件…')); self.clean_cache_button.setObjectName('cleanCacheButton')
            self.clean_cache_button.clicked.connect(self.clean_selected_caches); buttons.addWidget(self.clean_cache_button)
            form.addLayout(buttons)
            self.cache_status = self._label('', 'hint'); self.cache_status.setObjectName('cacheStatus')
            form.addWidget(self.cache_status)
            self.cache_result = self._label('', 'hint'); self.cache_result.setObjectName('cacheResult')
            form.addWidget(self.cache_result)
            # Closing the page stops a listing still in flight (between two jobs).
            self.window._advanced_dialogs['storage'].finished.connect(self.cancel_cache_listing)
        self.product_location.setText(self.window.output_dir.text())
        self.cache_dir.setEnabled(self.window.job_state not in ('queued','running','cancel_requested') and not self.window._busy)
        self.update_storage_controls()
        self._show('storage')
        self.refresh_cache_jobs()

    # ---- per-job cache cleanup (cache_cleanup; the MCP side only inspects) ----

    def _cache_roots(self):
        """The default cache root and the current preference; jobs keep their own recorded root."""
        from starstack.storage import default_cache_root
        values = [str(default_cache_root()), os.environ.get('AESTER_CACHE_ROOT') or '', self.cache_dir.text().strip()]
        roots = {}
        for value in values:
            if value:
                path = str(Path(value).expanduser().absolute())
                roots.setdefault(os.path.normcase(path), path)
        return list(roots.values())

    def _storage_idle(self):
        w = self.window
        return (not w._busy and w.job_state not in ('queued', 'running', 'cancel_requested')
                and not w._closing_requested and not self._cleaning_cache)

    def _selected_plans(self):
        jobs = getattr(self, 'cache_jobs', None)
        if jobs is None:
            return []
        plans = [item.data(0, Qt.ItemDataRole.UserRole) for item in jobs.selectedItems()]
        return [plan for plan in plans if isinstance(plan, dict) and plan.get('plan_id')]

    def update_storage_controls(self):
        """Cleaning is offered only while the window is idle and cleanable jobs are selected."""
        button = getattr(self, 'clean_cache_button', None)
        if button is None:
            return
        idle = self._storage_idle()
        selected = bool(self._selected_plans())
        button.setEnabled(idle and selected and not self._listing_cache)
        button.setToolTip('' if button.isEnabled() else
                          tr('等待当前处理或清理结束后再清理。') if not idle else tr('请先在列表中选择可清理的任务。'))
        self.refresh_cache_button.setEnabled(not self._listing_cache and not self._cleaning_cache)

    def _storage_visible(self):
        dialog = self.window._advanced_dialogs.get('storage')
        return dialog is not None and dialog.isVisible()

    def cancel_cache_listing(self):
        """Stop the job listing in flight (the page was closed or the window is closing); nothing is lost."""
        self._cache_refresh_pending = False
        if self._cache_cancel is not None:
            self._cache_cancel.set()

    def refresh_cache_jobs(self):
        """Rebuild the job list off the UI thread from records and file sizes (no image is read).

        Asked for while a listing runs (e.g. the cache folder changed meanwhile), a new listing of the
        current folders starts as soon as that one ends.
        """
        if getattr(self, 'cache_jobs', None) is None or self._cleaning_cache:
            return
        if self._listing_cache:
            self._cache_refresh_pending = True
            return
        roots = self._cache_roots()
        cancel = threading.Event()
        self._cache_cancel = cancel
        self._cache_refresh_pending = False
        self._listing_cache = True
        self._cache_serial += 1
        serial = self._cache_serial
        self.cache_status.setText(tr('正在检查缓存中的任务…'))
        self.update_storage_controls()

        def inspect():
            try:
                from starstack.cache_cleanup import list_workflow_caches
                return {'ok': True, 'listing': list_workflow_caches(roots, cancelled=cancel.is_set)}
            except Exception as error:
                return {'ok': False, 'message': str(error)}

        def listed(result):
            self._listing_cache = False
            if self._cache_cancel is cancel:
                self._cache_cancel = None
            if serial == self._cache_serial:
                if not result['ok']:
                    self.cache_status.setText(tr('无法读取缓存中的任务：') + translate_message(result['message']))
                elif result['listing'].get('cancelled'):
                    self.cache_status.setText(tr('列表检查已停止；点“刷新列表”重新检查。'))
                else:
                    self._fill_cache_jobs(result['listing'])
            self.update_storage_controls()
            if self._cache_refresh_pending:
                self._cache_refresh_pending = False
                if self._storage_visible():
                    self.refresh_cache_jobs()
        self.window._run_async('cache-list', inspect, listed)

    @staticmethod
    def _gigabytes(value):
        return f'{(value or 0) / 1024**3:.1f} GB'

    @staticmethod
    def _local_time(value):
        from datetime import datetime
        try:
            return datetime.fromisoformat(str(value)).astimezone().strftime('%Y-%m-%d %H:%M')
        except (TypeError, ValueError):
            return ''

    @staticmethod
    def cache_status_text(plan):
        """(status column, tooltip) of one job: cleanable / already cleaned / running / the refusal reason."""
        state = plan.get('state')
        reason = translate_message(plan.get('refusal') or '')
        if state == 'refused' or state not in CACHE_STATES:
            text = tr('不能清理：') + reason
            return text, text
        return tr(CACHE_STATES[state]), reason or tr(CACHE_STATES[state])

    def _fill_cache_jobs(self, listing):
        self.cache_jobs.clear()
        for plan in listing.get('jobs', []):
            job = plan.get('job') or {}
            name = job.get('name') or Path(plan['workflow_dir']).name
            frames = job.get('frames')
            status, reason = self.cache_status_text(plan)
            item = QTreeWidgetItem([tr('{0} · {1} 张', name, frames) if frames else name,
                                    self._local_time(job.get('created_at')), job.get('delivery_dir') or '',
                                    self._gigabytes(plan.get('bytes_total')),
                                    self._gigabytes(plan.get('bytes_deletable')) if plan.get('plan_id') else '—',
                                    status])
            item.setData(0, Qt.ItemDataRole.UserRole, plan)
            item.setToolTip(0, plan['workflow_dir'])
            item.setToolTip(2, job.get('delivery_dir') or '')
            item.setToolTip(5, reason)
            if not plan.get('plan_id'):
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)          # shown, never selectable
            self.cache_jobs.addTopLevelItem(item)
        for column in range(len(CACHE_COLUMNS) - 1):
            if column != 2:                       # the works path is long: elided, full path in its tooltip
                self.cache_jobs.resizeColumnToContents(column)
        self.cache_jobs.setColumnWidth(2, 240)
        count = sum(1 for plan in listing.get('jobs', []) if plan.get('plan_id'))
        free = sum(plan.get('bytes_deletable') or 0 for plan in listing.get('jobs', []) if plan.get('plan_id'))
        text = (tr('{0} 个任务可清理，共可释放约 {1:.1f} GB。选择任务后点“清理所选任务的中间文件…”。', count, free / 1024**3)
                if count else tr('没有可清理的任务。'))
        problems = listing.get('problems') or []
        if problems:
            text += '\n' + '\n'.join(problem['cache_root'] + ' · ' + translate_message(problem['error']) for problem in problems)
        self.cache_status.setText(text)

    def cleanup_confirmation(self, plans):
        """The confirmation text: the space freed, what is kept and what is lost."""
        freed = sum(plan.get('bytes_deletable') or 0 for plan in plans) / 1024**3
        text = tr('将清理所选 {0} 个任务的中间文件，约释放 {1:.1f} GB。\n\n作品文件夹中的照片和原片都会保留，母版、处理记录和日志也保留。\n清理后这些任务不能再复核亮迹候选，也不能恢复或重做亮迹合成；如需重做，请从原片新建任务。',
                  len(plans), freed)
        kept = sum(plan.get('stage_copies_kept') or 0 for plan in plans)
        if kept:          # their originals were moved or deleted: the cache copy may be the last one
            kept_bytes = sum(plan.get('stage_copies_kept_bytes') or 0 for plan in plans) / 1024**3
            text += '\n\n' + tr('有 {0} 张原片已不在原来的位置或已改变，它们在缓存中的副本（约 {1:.1f} GB）会保留，不会删除。',
                                  kept, kept_bytes)
        return text

    def clean_selected_caches(self):
        if not self._storage_idle():
            return
        plans = self._selected_plans()
        if not plans or not ask_cleanup(self.window, self.cleanup_confirmation(plans)) or not self._storage_idle():
            return
        selected = [(plan['workflow_dir'], plan['plan_id']) for plan in plans]
        self.window._busy = True
        self._cleaning_cache = True
        self.cache_result.setText(tr('正在清理所选任务的中间文件…'))
        self.window._update_controls()

        def clean():
            from starstack.cache_cleanup import clean_workflow_cache
            results = []
            for directory, plan_id in selected:
                try:
                    results.append({'ok': True, 'workflow_dir': directory, 'result': clean_workflow_cache(directory, plan_id)})
                except Exception as error:
                    results.append({'ok': False, 'workflow_dir': directory, 'message': str(error)})
            return results
        self.window._run_async('cache-clean', clean, self._cleaned)

    def _cleaned(self, results):
        self.window._busy = False
        self._cleaning_cache = False
        done = [entry['result'] for entry in results if entry['ok']]
        freed = sum(result['removed_bytes'] for result in done)
        lines = [tr('已释放 {0:.2f} GB。作品与原片保持不变。', freed / 1024**3)]
        partial = [result for result in done if result['state'] == 'partial']
        if partial:
            lines.append(tr('{0} 个任务有文件暂时无法删除（可能被其他程序占用）；关闭相关程序后可再次清理。', len(partial)))
        for entry in results:
            if not entry['ok']:
                lines.append(tr('未清理：') + Path(entry['workflow_dir']).name + ' · ' + translate_message(entry['message']))
        self.cache_result.setText('\n'.join(lines))
        self.window._update_controls()
        if self.window.workflow_dir:
            self.window.refresh_workflow()      # a cleaned current job disables trail review
        self.refresh_cache_jobs()

    def maybe_welcome(self):
        prefs = self.window._preferences
        if prefs is not None and str(prefs.value('experience/welcome_seen', '0')) != '1':
            QTimer.singleShot(0, self.show_welcome)

    def _welcome_seen(self, *_):
        if self.window._preferences:
            self.window._preferences.setValue('experience/welcome_seen', '1'); self.window._preferences.sync()

    def _welcome_done(self):
        self._welcome_seen()
        self.window._advanced_dialogs['welcome'].close()

    def _nox_description(self):
        from starstack.components import nox_install_info
        text = tr('可选 nox 组件用于星点整理、缩星，以及导出无星图和星点层。第三方模型按 MIT 发布，由 Aester 转换为 ONNX；训练数据权利尚未核清。安装后仍由你选择是否开启效果。')
        try:
            info = nox_install_info()
            size = (info['model_bytes'] + info['runtime_bytes']) / (1024 * 1024)
            text += '\n' + tr('需要下载模型与运行环境，约 {0} MiB。', f'{size:.0f}')
            text += '\n' + translate_message(info['message'])
        except (OSError, ValueError, KeyError, TypeError):
            text += '\n' + tr('下载配置尚不可用；可在组件管理中选择已有的已验转换模型。')
        return text

    def _welcome_nox(self):
        """An explicit consent button; queue behind the existing read-only component check."""
        w = self.window
        if self._installing or self._installing_retouch or w._busy or w.job_state in ('queued', 'running', 'cancel_requested'):
            w.message_label.setText(tr('请等当前处理完成后准备组件。'))
            return
        self._pending_nox_install = True
        self._welcome_done()
        self.show_components()

    def show_welcome(self):
        form = self._dialog('welcome', tr('欢迎使用 Aester'), 690, 600)
        if form is not None:
            form.addWidget(self._label(tr('从一组星空照片开始'), 'title'))
            form.addWidget(self._label(tr('添加同一场景的连拍照片，在预览上标出天空和地景，选择效果，然后开始合成。原片保持不变。')))
            form.addWidget(self._label(tr('你可以直接操作，也可以让已经配置的大模型助手通过 MCP 操作。连接助手是可选的，不需要在 Aester 填入模型密钥。'), 'muted'))
            begin = QPushButton(tr('直接开始')); begin.setObjectName('primary'); themes.set_role(begin, 'primary')
            begin.clicked.connect(self._welcome_done); form.addWidget(begin)
            form.addWidget(self._label(self._nox_description(), 'muted'))
            nox = QPushButton(tr('同意并准备可选 nox 组件'))
            nox.setObjectName('welcomeNoxConsent'); nox.setAutoDefault(False)
            nox.clicked.connect(self._welcome_nox); form.addWidget(nox)
            later = QPushButton(tr('稍后再说'))
            later.setObjectName('welcomeNoxLater'); later.clicked.connect(self._welcome_done); form.addWidget(later)
            connect = QPushButton(tr('连接 AI 助手')); connect.clicked.connect(lambda:(self._welcome_done(),self.show_connection())); form.addWidget(connect)
            component = QPushButton(tr('查看已安装组件')); component.clicked.connect(self.show_components); form.addWidget(component)
            form.addWidget(self._label(tr('深色、浅色与夜视可在“设置 → 外观”切换。作品与缓存位置可在设置里查看。'), 'hint'))
            form.addStretch(1)
            # Closing the welcome in any way counts as seen; it reopens from Help → 欢迎与开始.
            self.window._advanced_dialogs['welcome'].finished.connect(self._welcome_seen)
        self._show('welcome')

    def show_connection(self):
        form = self._dialog('connect_ai', tr('连接 AI 助手 · MCP'), 840, 650)
        if form is not None:
            form.addWidget(self._label(tr('选择助手可以读取的素材文件夹。配置只包含本机路径；Aester不会替你上传照片。MCP客户端是否发送图像给模型，由你的客户端和请求决定。'), 'muted'))
            row = QHBoxLayout(); self.mcp_input = QLineEdit(); self.mcp_input.setObjectName('mcpInputRoot')
            saved = self.window._preferences.value('mcp/input_root','') if self.window._preferences else ''
            self.mcp_input.setText(str(saved or self.window.input_dir.text()))
            row.addWidget(self.mcp_input, 1); pick = QPushButton(tr('选择素材范围')); pick.clicked.connect(self._pick_mcp_input);row.addWidget(pick);form.addLayout(row)
            self.connection_text = QTextBrowser(); self.connection_text.setObjectName('mcpConfiguration'); themes.set_role(self.connection_text, 'mono');form.addWidget(self.connection_text,1)
            buttons = QHBoxLayout(); copy = QPushButton(tr('复制配置'));copy.clicked.connect(self._copy_config);buttons.addWidget(copy)
            prompt = QPushButton(tr('复制给助手的提示词'));prompt.clicked.connect(self._copy_prompt);buttons.addWidget(prompt)
            self.test_connection = QPushButton(tr('检测本机服务'));self.test_connection.clicked.connect(self._test_connection);buttons.addWidget(self.test_connection);form.addLayout(buttons)
            self.connection_status = self._label(tr('复制配置不会自动连接客户端。检测仅验证本机服务，完成后仍需在Agent客户端添加配置。'), 'hint');form.addWidget(self.connection_status)
            self.mcp_input.editingFinished.connect(self._refresh_connection)
        self._refresh_connection();self._show('connect_ai')

    def _pick_mcp_input(self):
        value = QFileDialog.getExistingDirectory(self.window,tr('选择允许读取的素材范围'),self.mcp_input.text())
        if value:self.mcp_input.setText(value);self._refresh_connection()

    def _refresh_connection(self):
        from starstack.onboarding import connection_config
        w = self.window
        previous = self._config
        try:
            self._config = connection_config(self.mcp_input.text(),w.output_dir.text(),self.work_root(),
                python_path=self._next_python,
                siril_path=w.siril_path.text() or None,stack_engine_path=w.stack_engine_path.text() or None,
                star_model_path=w.star_model_path.text() or None,
                gimp_path=self._gimp_path or None,
                dng_converter_path=w.dng_converter_path.text() or None,
                backend_options={'channel_order':w.model_channel_order.currentData(),'value_range':w.model_range.currentData(),
                                 'layout':w.model_layout.currentData(),'stitch':w.model_stitch.currentData()})
            self.connection_text.setPlainText(json.dumps(self._config,ensure_ascii=False,indent=2))
            if self._config != previous:
                self.connection_status.setText(tr('配置已更新，请重新检测。复制配置不会自动连接客户端。'))
            if w._preferences:w._preferences.setValue('mcp/input_root',self.mcp_input.text())
        except (ValueError,OSError) as error:
            self._config=None;self.connection_text.setPlainText(tr('请先选择有效路径：')+translate_message(str(error)))
            self.connection_status.setText(tr('配置无效，请修正路径后重新检测。尚未确认客户端连接。'))

    def _copy_config(self):
        self._refresh_connection()
        if self._config:
            QApplication.clipboard().setText(json.dumps(self._config,ensure_ascii=False,indent=2))
            self.connection_status.setText(tr('配置已复制。请粘贴到支持 stdio MCP 的客户端设置；尚未确认客户端连接。'))

    def _copy_prompt(self):
        from starstack.onboarding import setup_prompt
        self._refresh_connection()
        if self._config:
            QApplication.clipboard().setText(setup_prompt(self._config))
            self.connection_status.setText(tr('提示词已复制，包含启动配置和目录范围；不含模型密钥。'))

    def _test_connection(self):
        if self._testing:return
        self._refresh_connection()
        if not self._config:return
        from starstack.onboarding import probe_connection
        config=self._config;self._testing=True;self.test_connection.setEnabled(False)
        self.connection_status.setText(tr('正在启动本机服务并检查工具…'))
        def check():
            try:return probe_connection(config)
            except Exception as error:return {'state':'failed','message':tr('本机服务未通过检测：')+translate_message(str(error))}
        def checked(result):
            self._testing=False;self.test_connection.setEnabled(True)
            self._refresh_connection()
            self.connection_status.setText(translate_message(result['message']) if self._config == config
                else tr('检测期间配置已改变，请按新配置重新检测。'))
        self.window._run_async('mcp-handshake',check,checked)

    def show_retouch(self):
        form = self._dialog('retouch', tr('自然语言修图'), 850, 760)
        if form is not None:
            # The Agent loop, stated honestly: Aester writes the prompt; the Agent does the work.
            header = QHBoxLayout(); header.setSpacing(10)
            self.retouch_provenance = ui_widgets.ProvenanceDot('agent', accessible_name=tr('Agent 参与'))
            self.retouch_provenance.setObjectName('retouchProvenance')
            header.addWidget(self.retouch_provenance, 0, Qt.AlignmentFlag.AlignTop)
            self.retouch_loop = self._label(tr('Aester 只准备提示词，不启动 Agent。你的 Agent 通过 Aester 的主 MCP 读取成片并修图；修好的版本另存到作品目录，原成片保留。'), 'field')
            self.retouch_loop.setObjectName('retouchLoop')
            header.addWidget(self.retouch_loop, 1)
            form.addLayout(header)
            form.addWidget(self._label(tr('描述想改变和保留的内容，复制提示词给支持看图与本地 MCP 的 Agent。没有成片时也可先了解步骤。'), 'muted'))
            form.addWidget(self._label(tr('成片 TIFF（可选）'), 'field'))
            row = QHBoxLayout()
            self.retouch_photo = QLineEdit()
            self.retouch_photo.setObjectName('retouchPhotoPath')
            self.retouch_photo.setReadOnly(True)
            themes.set_role(self.retouch_photo, 'mono')
            self.retouch_photo.setPlaceholderText(tr('尚未选择；可稍后交给 Agent'))
            row.addWidget(self.retouch_photo, 1)
            choose = QPushButton(tr('选择成片…'))
            choose.clicked.connect(self._pick_retouch_photo)
            row.addWidget(choose)
            clear = QPushButton(tr('清除选择'))
            clear.clicked.connect(self.retouch_photo.clear)
            row.addWidget(clear)
            form.addLayout(row)
            # Which photograph this is (with added trails or the base), said in words.
            self.retouch_photo_hint = self._label('', 'hint')
            self.retouch_photo_hint.setObjectName('retouchPhotoHint')
            self.retouch_photo_hint.hide()
            form.addWidget(self.retouch_photo_hint)
            # An optional aesthetic reference for the Agent (registered by kind, never pasted in).
            form.addWidget(self._label(tr('参考图（可选）'), 'field'))
            row = QHBoxLayout()
            self.retouch_reference = QLineEdit()
            self.retouch_reference.setObjectName('retouchReferencePath')
            self.retouch_reference.setReadOnly(True)
            themes.set_role(self.retouch_reference, 'mono')
            self.retouch_reference.setPlaceholderText(tr('没有参考图；Agent 会先问你是否有'))
            row.addWidget(self.retouch_reference, 1)
            self.retouch_reference_kind = QComboBox()
            self.retouch_reference_kind.setObjectName('retouchReferenceKind')
            self.retouch_reference_kind.setAccessibleName(tr('参考图来源'))
            for text, value in ((tr('他人作品'), 'external'), (tr('我自己的作品'), 'original'),
                                (tr('AI 生成图'), 'generated')):
                self.retouch_reference_kind.addItem(text, value)
            row.addWidget(self.retouch_reference_kind)
            pick_reference = QPushButton(tr('选择参考图…'))
            pick_reference.setObjectName('pickRetouchReference')
            pick_reference.clicked.connect(self._pick_retouch_reference)
            row.addWidget(pick_reference)
            clear_reference = QPushButton(tr('清除'))
            clear_reference.setObjectName('clearRetouchReference')
            clear_reference.setAccessibleName(tr('清除参考图'))
            clear_reference.clicked.connect(self.retouch_reference.clear)
            row.addWidget(clear_reference)
            form.addLayout(row)
            self.retouch_reference_tip = self._label(
                tr('小提示：给 Agent 一张你喜欢的效果图作参考（他人作品、你以前的作品或 AI 生成图）。'
                   'Agent 只借鉴它的色调、明暗、反差和气氛，不会把参考图里的星点、云、山脊线或纹理搬进你的照片。'), 'hint')
            self.retouch_reference_tip.setObjectName('retouchReferenceTip')
            form.addWidget(self.retouch_reference_tip)
            self.retouch_reference_warning = self._label('', 'error')
            self.retouch_reference_warning.setObjectName('retouchReferenceWarning')
            self.retouch_reference_warning.hide()
            form.addWidget(self.retouch_reference_warning)
            self.retouch_intent = QPlainTextEdit()
            self.retouch_intent.setObjectName('retouchIntent')
            themes.set_role(self.retouch_intent, 'human')  # words a person wrote
            self.retouch_intent.setPlaceholderText(tr('例如：银河层次更清楚，保留气辉和地景的自然颜色；先给我看整图与局部，再决定接受哪个版本。'))
            self.retouch_intent.setMaximumHeight(80)
            form.addWidget(self._label(tr('你的需求'), 'field'))
            form.addWidget(self.retouch_intent)
            # The action row sits right under the request so it is always in view;
            # the machine-written prompt preview takes the remaining height.
            buttons = QHBoxLayout()
            self.retouch_copy = QPushButton(tr('复制修图提示词'))
            self.retouch_copy.setObjectName('retouchCopy')
            themes.set_role(self.retouch_copy, 'primary')
            self.retouch_copy.clicked.connect(self._copy_retouch_prompt)
            buttons.addWidget(self.retouch_copy)
            components = QPushButton(_literal(tr('组件与提示词管理…')))
            components.setObjectName('openComponents')
            components.clicked.connect(self.show_components)
            buttons.addWidget(components)
            connect = QPushButton(tr('连接 AI 助手'))
            connect.clicked.connect(self.show_connection)
            buttons.addWidget(connect)
            form.addLayout(buttons)
            self.retouch_message = self._label(tr('复制只生成文字，不启动 Agent。Aester 本地处理照片；外部客户端是否发送图像取决于你的请求和授权。'), 'hint')
            form.addWidget(self.retouch_message)
            self.retouch_status = self._label('', 'hint')
            self.retouch_status.setObjectName('retouchStatus')
            form.addWidget(self.retouch_status)
            self.retouch_prompt = QTextBrowser()
            self.retouch_prompt.setObjectName('retouchPrompt')
            themes.set_role(self.retouch_prompt, 'mono')  # machine-written text
            self.retouch_prompt.setMinimumHeight(96)
            form.addWidget(self._label(tr('将复制给 Agent 的提示词'), 'field'))
            form.addWidget(self.retouch_prompt, 1)
            self.retouch_photo.textChanged.connect(self._refresh_retouch_prompt)
            self.retouch_photo.textChanged.connect(self._update_retouch_photo_hint)
            self.retouch_intent.textChanged.connect(self._refresh_retouch_prompt)
            self.retouch_reference.textChanged.connect(self._refresh_retouch_prompt)
            self.retouch_reference.textChanged.connect(self._update_reference_warning)
            self.retouch_reference_kind.currentIndexChanged.connect(self._refresh_retouch_prompt)
        w = self.window
        if w.workflow_dir and w.job_state == 'succeeded' and not w._busy:
            # A with-trails photograph may have been published since (another window or the MCP).
            try:
                from starstack.workflow import workflow_status
                record = workflow_status(w.workflow_dir)
            except (OSError, ValueError, KeyError):
                record = None
            if (record and record.get('state') == 'succeeded'
                    and not w._same_dir(record.get('delivery_dir'), (w._workflow or {}).get('delivery_dir'))):
                w.refresh_workflow()
        # Fill in the current photograph only while the field is empty or still holds the
        # previous automatic choice; a TIFF the person chose is never replaced or cleared.
        automatic = w.retouch_photo_path() or ''
        previous = getattr(self, '_retouch_auto_photo', '')
        current = self.retouch_photo.text()
        if (not current or current == previous) and current != automatic:
            self.retouch_photo.setText(automatic)
        self._retouch_auto_photo = automatic
        self._update_retouch_photo_hint()
        self._update_reference_warning()
        self._refresh_retouch_prompt()
        self._show('retouch')

    def _update_retouch_photo_hint(self):
        """'当前成片：已加回亮迹的版本' / '当前成片：基础成片（未加回亮迹）' for a delivery photograph."""
        if not hasattr(self, 'retouch_photo_hint'):
            return
        w = self.window
        record = w._workflow or {}
        text = ''
        path = self.retouch_photo.text().strip()
        if path and (record.get('request') or {}).get('trails_enabled'):
            folder = Path(path).parent
            trails = [item.get('delivery_dir') for item in record.get('review_deliveries') or [] if isinstance(item, dict)]
            if w._has_trails_version(record):
                trails.append(record.get('delivery_dir'))
            base = record.get('base_delivery_dir') or record.get('delivery_dir')
            if any(w._same_dir(folder, item) for item in trails if item):
                text = tr('当前成片：已加回亮迹的版本')
            elif w._same_dir(folder, base):
                text = tr('当前成片：基础成片（未加回亮迹）')
        self.retouch_photo_hint.setText(text)
        self.retouch_photo_hint.setVisible(bool(text))

    def _reference_roots(self):
        w = self.window
        roots = []
        saved = w._preferences.value('mcp/input_root', '') if w._preferences else ''
        for value in (saved, w.output_dir.text()):
            if isinstance(value, str) and value.strip():
                roots.append(Path(value.strip()).expanduser().absolute())
        return roots

    def _update_reference_warning(self):
        """The Agent can only open files inside its material scope or the photographs folder."""
        if not hasattr(self, 'retouch_reference_warning'):
            return
        path = self.retouch_reference.text().strip()
        outside = False
        if path:
            target = Path(path).expanduser().absolute()
            outside = not any(target.is_relative_to(root) for root in self._reference_roots())
        self.retouch_reference_warning.setText(
            tr('这张参考图不在助手可读取的素材范围或作品文件夹内，Agent 可能打不开它。'
               '请把它复制到作品文件夹，或在“连接 AI 助手”中调整素材范围。') if outside else '')
        self.retouch_reference_warning.setVisible(outside)

    def _pick_retouch_reference(self):
        start = self.retouch_reference.text() or self.window.output_dir.text()
        path, _ = QFileDialog.getOpenFileName(self.window, tr('选择修图参考图'), start,
                                              tr('参考图 (*.jpg *.jpeg *.png *.tif *.tiff)'))
        if path:
            self.retouch_reference.setText(path)

    def retouch_reference_images(self):
        """[{'path', 'kind'}] for the prompt, or None without a reference."""
        path = self.retouch_reference.text().strip() if hasattr(self, 'retouch_reference') else ''
        if not path:
            return None
        return [{'path': path, 'kind': self.retouch_reference_kind.currentData()}]

    def _pick_retouch_photo(self):
        # Start in the photographs folder: that is where finished TIFFs are saved.
        start = self.retouch_photo.text() or self.window.output_dir.text()
        path, _ = QFileDialog.getOpenFileName(self.window, tr('选择带 sRGB 色彩描述的成片 TIFF'),
                                             start, tr('TIFF 成片 (*.tif *.tiff)'))
        if path:
            self.retouch_photo.setText(path)

    def _refresh_retouch_prompt(self):
        from starstack.retouch_integration import build_retouch_prompt, get_retouch_info
        try:
            info = get_retouch_info(gimp_path=self._gimp_path or None)
            self.retouch_status.setText(self._retouch_status_text(info))
            prompt = build_retouch_prompt(photo_path=self.retouch_photo.text() or None,
                                          intent=self.retouch_intent.toPlainText().strip() or None,
                                          gimp_path=self._gimp_path or None, language=get_language(),
                                          reference_images=self.retouch_reference_images())
            self.retouch_prompt.setPlainText(prompt)
            self.retouch_copy.setEnabled(True)
        except (OSError, ValueError) as error:
            self.retouch_prompt.setPlainText('')
            self.retouch_status.setText(tr('操作未完成：') + translate_message(str(error)))
            self.retouch_copy.setEnabled(False)

    def _copy_retouch_prompt(self):
        self._refresh_retouch_prompt()
        if self.retouch_copy.isEnabled():
            QApplication.clipboard().setText(self.retouch_prompt.toPlainText())
            self.retouch_message.setText(tr('修图提示词已复制。请粘贴到 Agent，并按需要补充照片和审美要求。'))

    @staticmethod
    def _retouch_status_text(info):
        gimp, skill = info.get('gimp', {}), info.get('skill', {})
        gimp_state = tr('已找到程序；实际修图前需验证执行能力') if gimp.get('available') else tr('未找到可用 GIMP；可在“组件与提示词管理”中准备')
        skill_state = tr('技能包可读取；建议交给 Agent 加载') if skill.get('available') else tr('技能包尚不可用；可先复制提示词，稍后准备')
        if gimp.get('status') == 'check_failed':
            gimp_state = tr('GIMP 检查未完成；请查看“组件与提示词管理”')
        if skill.get('status') == 'check_failed':
            skill_state = tr('技能检查未完成；可先复制提示词')
        lines = ['GIMP: ' + gimp_state]
        if gimp.get('selected_path'):
            lines.append(str(gimp['selected_path']))
        lines.append(tr('Aester 修图技能') + ': ' + skill_state)
        if skill.get('read_path'):
            lines.append(str(skill['read_path']))
        return '\n'.join(lines)

    def show_components(self):
        form = self._dialog('components',tr('组件与提示词管理'),800,760)
        if form is not None:
            form.addWidget(self._label(tr('基础运行环境和 Siril 用于合成。nox 去星组件为可选项，用于分层与星点整理；安装组件不会自动开启效果。'), 'muted'))
            self.component_text=QTextBrowser();themes.set_role(self.component_text, 'mono')
            # A few status lines; kept short so the prompt button below stays in view.
            self.component_text.setMaximumHeight(132);form.addWidget(self.component_text)
            form.addWidget(self._label(self._nox_description(), 'hint'))
            self.star_support=QCheckBox(tr('同意准备 nox 模型与运行环境（可选）'));form.addWidget(self.star_support)
            self.model_select=QPushButton(tr('选择已有 nox ONNX 模型…'));self.model_select.clicked.connect(self._pick_model);form.addWidget(self.model_select)
            row=QHBoxLayout();self.component_install=QPushButton(tr('准备／修复所选组件'));self.component_install.clicked.connect(self._install_components);row.addWidget(self.component_install)
            refresh=QPushButton(tr('重新检查'));refresh.clicked.connect(self._load_components);row.addWidget(refresh)
            self.component_cancel=QPushButton(tr('取消准备'));self.component_cancel.setObjectName('cancelComponentInstall')
            self.component_cancel.setEnabled(False);self.component_cancel.clicked.connect(self._cancel_component_install);row.addWidget(self.component_cancel)
            form.addLayout(row)
            form.addWidget(self._label(tr('自然语言修图：执行需要 GIMP；建议让 Agent 加载或安装修图技能。未安装技能也能复制提示词和使用 MCP。')))
            self.retouch_component_text = self._label('', 'hint')
            self.retouch_component_text.setObjectName('retouchComponentStatus')
            form.addWidget(self.retouch_component_text)
            self.gimp_support = QCheckBox(tr('准备 GIMP（自然语言修图执行所需）'))
            self.gimp_support.setObjectName('gimpSupport')
            self.skill_support = QCheckBox(tr('准备修图技能包（建议加载或安装）'))
            self.skill_support.setObjectName('retouchSkillSupport')
            form.addWidget(self.gimp_support)
            form.addWidget(self.skill_support)
            row = QHBoxLayout()
            select_gimp = QPushButton(tr('选择已有 GIMP…'))
            select_gimp.clicked.connect(self._pick_gimp)
            row.addWidget(select_gimp)
            auto_gimp = QPushButton(tr('自动寻找 GIMP'))
            auto_gimp.clicked.connect(self._use_auto_gimp)
            row.addWidget(auto_gimp)
            row.addStretch(1)
            form.addLayout(row)
            # One highlighted action: prepare what is ticked, then write and copy the prompt.
            self.retouch_install = QPushButton(tr('按所选组件，生成给助手的提示词'))
            self.retouch_install.setObjectName('generateAssistantPrompt')
            themes.set_role(self.retouch_install, 'primary')
            self.retouch_install.setMinimumHeight(44)
            self.retouch_install.clicked.connect(self._install_retouch_components)
            form.addWidget(self.retouch_install)
            self.retouch_plan = self._label('', 'hint')
            self.retouch_plan.setObjectName('generatePromptPlan')
            form.addWidget(self.retouch_plan)
            self.gimp_support.toggled.connect(self._update_retouch_plan)
            self.skill_support.toggled.connect(self._update_retouch_plan)
            self._update_retouch_plan()
            form.addWidget(self._label(tr('GIMP 由官方安装程序独立管理，卸载 Aester 时保留。技能包可直接交给 Agent 读取，不会自动改写客户端的技能目录。')))
            self.restart_button=QPushButton(tr('保存编辑项目后，重新启动 Aester'))
            self.restart_button.clicked.connect(self._restart);self.restart_button.setVisible(False);form.addWidget(self.restart_button)
            self.component_message=self._label(tr('组件有独立版本和校验记录。nox 在主动同意后准备；可稍后重试或选择已有的本项目转换模型，不会更改原片或作品。'));form.addWidget(self.component_message)
            form.addStretch(1)
        self._show('components');self._load_components()

    def _load_components(self):
        if self._checking_components or self._installing or self._installing_retouch:return
        self._checking_components=True
        self.component_install.setEnabled(False)
        from starstack.components import component_status
        def load():
            try:
                from starstack.retouch_integration import get_retouch_info
                result = component_status()
                result['retouch_info'] = get_retouch_info(gimp_path=self._gimp_path or None)
                return result
            except Exception as error:return {'ready':False,'error':str(error),'components':[]}
        def loaded(result):
            self._checking_components=False;self.component_install.setEnabled(True);self._show_components(result)
            if self._pending_nox_install:
                self._pending_nox_install=False
                self.star_support.setChecked(True)
                self._install_components()
        self.window._run_async('component-status',load,loaded)

    def _pick_model(self):
        if self._installing or self._installing_retouch:
            return
        selected,_=QFileDialog.getOpenFileName(self.window,tr('选择本项目转换的 nox ONNX 模型'),'',tr('ONNX 模型 (*.onnx)'))
        if selected:
            self._model_source=selected;self.star_support.setChecked(True)
            self.component_message.setText(tr('已选择模型，准备时将核验身份和运行能力：')+selected)

    def _show_components(self,result):
        self._component_result=result
        names={'base':tr('基础运行环境与 Siril'),'nox':tr('nox 去星与星点整理'),
               'gimp':'GIMP', 'retouch_skill':tr('Aester 修图技能')}
        states={'ready':tr('已就绪'),'installed':tr('已安装'),'missing':tr('未安装'),'unavailable':tr('尚不可用'),'invalid':tr('需要修复'),'broken':tr('需要修复'),'failed':tr('准备失败')}
        labels=[f"{names.get(item.get('id'),item.get('id'))}: {states.get(item.get('status'),item.get('status'))} · {translate_message(item.get('detail',''))}"
                for item in result.get('components',[]) if item.get('id') not in ('gimp', 'retouch_skill')]
        self.component_text.setPlainText('\n'.join(labels)+ tr('\n\n组件位置：')+str(result.get('component_root',''))
                                         + ('\n'+translate_message(str(result['error'])) if result.get('error') else ''))
        if result.get('retouch_info'):
            self._retouch_info = result['retouch_info']
            self.retouch_component_text.setText(self._retouch_status_text(result['retouch_info']))
            self._update_retouch_plan()

    def _update_retouch_plan(self, *_):
        """What the highlighted button will prepare before it writes the prompt."""
        if not hasattr(self, 'retouch_plan'):
            return
        items = []
        if self.gimp_support.isChecked():
            found = ((getattr(self, '_retouch_info', None) or {}).get('gimp') or {}).get('available')
            items.append(tr('GIMP（使用本机已有的程序）') if found else tr('GIMP（下载官方安装程序，来自官方网站或所列镜像）'))
        if self.skill_support.isChecked():
            items.append(tr('修图技能包'))
        if items:
            separator = ', ' if get_language() == 'en' else '、'
            text = tr('将先准备：{0}；完成后生成提示词并复制。', separator.join(items))
        else:
            text = tr('未勾选组件：直接生成提示词并复制。')
        self.retouch_plan.setText(text)

    def _pick_gimp(self):
        if self._installing or self._installing_retouch or self._checking_components:
            return
        selected, _ = QFileDialog.getOpenFileName(self.window, tr('选择 GIMP console 程序'),
                                                  self._gimp_path, tr('程序 (*.exe);;所有文件 (*)'))
        if not selected:
            return
        from starstack.retouch_integration import get_retouch_info
        try:
            info = get_retouch_info(gimp_path=selected)
            gimp = info.get('gimp', {})
            if not (gimp.get('status') == 'available' and gimp.get('path_exists')):
                self.component_message.setText(tr('所选 GIMP 尚不可用，原有选择保持。'))
                self.retouch_component_text.setText(self._retouch_status_text(info))
                return
            self._gimp_path = selected
            if self.window._preferences:
                self.window._preferences.setValue('retouch/gimp_path', selected)
                self.window._preferences.sync()
            self.retouch_component_text.setText(self._retouch_status_text(info))
            self.component_message.setText(tr('已选择 GIMP。实际修图前仍需检查版本和执行能力。'))
            if hasattr(self, 'retouch_prompt'):
                self._refresh_retouch_prompt()
        except (OSError, ValueError) as error:
            self.component_message.setText(tr('操作未完成：') + translate_message(str(error)))

    def _install_retouch_components(self):
        w = self.window
        include_gimp, include_skill = self.gimp_support.isChecked(), self.skill_support.isChecked()
        if not include_gimp and not include_skill and not self._installing_retouch:
            # Nothing to prepare: write and copy the prompt right away (also while a job runs).
            self._generate_assistant_prompt()
            return
        if self._checking_components or self._installing or self._installing_retouch or w._busy or w.job_state in ('queued', 'running', 'cancel_requested'):
            self.component_message.setText(tr('请等当前处理完成后准备组件。'))
            return
        from starstack.components import install_retouch_components
        selected = self._gimp_path or None
        self._installing_retouch = True
        self.retouch_install.setEnabled(False)
        self.component_install.setEnabled(False)
        w._busy = True
        w._update_controls()
        def install():
            try:
                return install_retouch_components(include_gimp=include_gimp, include_skill=include_skill,
                    selected_gimp_path=selected,
                    progress=lambda item: self.progress.emit(str(item.get('message') or item.get('phase') or tr('正在准备组件…'))))
            except Exception as error:
                return {'ready': False, 'error': str(error)}
        def complete(result):
            self._installing_retouch = False
            self.retouch_install.setEnabled(True)
            self.component_install.setEnabled(True)
            w._busy = False
            w._update_controls()
            ready = bool(result.get('ready'))
            notes = []
            if ready:
                from starstack.retouch_distribution import describe_notices
                # An existing GIMP is used as it is; say so, with the tested version.
                notes = describe_notices(result.get('notices'), get_language())
            else:
                # A failed preparation never produces a prompt.
                self.component_message.setText(tr('准备未完成：') + translate_message(str(result.get('error', tr('请检查详情并重试')))))
            self._load_components()
            if ready:
                self._generate_assistant_prompt(notes)
            elif hasattr(self, 'retouch_prompt'):
                self._refresh_retouch_prompt()
        w._run_async('retouch-component-install', install, complete)

    def _generate_assistant_prompt(self, notes=()):
        """Open 自然语言修图, write the prompt for the current photograph and copy it."""
        self.show_retouch()
        self._copy_retouch_prompt()
        if self.retouch_copy.isEnabled():
            message = tr('已按所选组件生成提示词并复制。粘贴给 Agent 即可开始修图。')
        else:
            message = tr('组件已准备，但提示词未能生成：') + self.retouch_status.text()
        if hasattr(self, 'component_message'):
            self.component_message.setText('\n'.join([message, *notes]))

    def _use_auto_gimp(self):
        if self._checking_components or self._installing or self._installing_retouch:
            return
        self._gimp_path = ''
        if self.window._preferences:
            self.window._preferences.remove('retouch/gimp_path')
            self.window._preferences.sync()
        self._load_components()
        if hasattr(self, 'retouch_prompt'):
            self._refresh_retouch_prompt()

    def _progress(self,message):
        if hasattr(self,'component_message'):self.component_message.setText(translate_message(message))

    def _cancel_component_install(self):
        if self._component_cancel is not None:
            self._component_cancel.set()
            self.component_cancel.setEnabled(False)
            self.component_message.setText(tr('正在取消；当前下载或校验停止后可重试，原有配置保持。'))

    def _install_components(self):
        w=self.window
        if self._checking_components:
            self.component_message.setText(tr('正在检查现有组件，请检查完成后再准备。'));return
        if self._installing or self._installing_retouch or w._busy or w.job_state in ('queued','running','cancel_requested'):
            self.component_message.setText(tr('请等当前处理完成后准备组件。'));return
        from starstack.components import install_components, install_nox_support
        include=self.star_support.isChecked();model_source=self._model_source
        cancel=threading.Event();self._component_cancel=cancel
        self._installing=True;self.component_install.setEnabled(False)
        self.component_cancel.setEnabled(True);self.star_support.setEnabled(False);self.model_select.setEnabled(False)
        self.retouch_install.setEnabled(False)
        w._busy=True;w._update_controls()
        def install():
            try:
                progress=lambda item:self.progress.emit(str(item.get('message') or item.get('phase') or tr('正在准备组件…')))
                if include:
                    return install_nox_support(consent=True,model_source=model_source,progress=progress,cancelled=cancel.is_set)
                return install_components(progress=progress,cancelled=cancel.is_set)
            except Exception as error:return {'ready':False,'error':str(error),'cancelled':cancel.is_set(),'components':[]}
        def complete(result):
            self._component_cancel=None
            self.component_cancel.setEnabled(False);self.star_support.setEnabled(True);self.model_select.setEnabled(True)
            self.retouch_install.setEnabled(True)
            self._installing=False;self.component_install.setEnabled(True);self._show_components(result)
            w._busy=False;w._update_controls()
            if result.get('ready'):
                if result.get('python'):
                    self._next_python=result['python']
                    # python/pythonw in the same runtime have identical modules.
                    self._restart_required=Path(result['python']).resolve().parent != Path(sys.executable).resolve().parent
                if result.get('siril'):w.siril_path.setText(result['siril'])
                if result.get('model_path'):
                    w.star_model_path.setText(result['model_path'])
                    for control,key in [(w.model_channel_order,'channel_order'),(w.model_range,'value_range'),(w.model_layout,'layout'),(w.model_stitch,'stitch')]:
                        value=result.get('star_backend_options',{}).get(key)
                        if value is not None:control.setCurrentIndex(control.findData(value))
                w._save_preferences();w.check_backend()
                self.restart_button.setVisible(self._restart_required)
                message=(tr('组件已安装；新运行环境需要重新启动。请先保存编辑项目，再点击“保存编辑项目后，重新启动 Aester”。')
                         if self._restart_required else tr('组件已就绪；效果仍按你的选择开启。'))
                self.component_message.setText(message)
                if self._restart_required:w.message_label.setText(message)
                w._update_controls()
            else:self.component_message.setText((tr('已取消，可重新准备：') if result.get('cancelled') else tr('准备未完成，可重试：'))+translate_message(str(result.get('error',tr('请检查详情并重试')))))
        w._run_async('component-install',install,complete)

    def _restart(self):
        """Explicit user action; never discard an editing session automatically."""
        import subprocess
        w=self.window
        if not self._restart_required or w._busy or w._workers or w._pool.activeThreadCount():
            self.component_message.setText(tr('请等待当前本地操作完成，再重新启动。'));return
        if w.job_state in ('queued','running','cancel_requested'):return
        w._save_preferences()
        root=Path(__file__).resolve().parents[2]
        launcher=Path(os.environ.get('AESTER_INSTALL_ROOT',''))/'Aester.exe'
        env=dict(os.environ)
        if os.environ.get('AESTER_INSTALL_ROOT') and launcher.is_file():
            command=[str(launcher)]
        else:
            python=Path(self._next_python)
            if python.with_name('pythonw.exe').is_file():python=python.with_name('pythonw.exe')
            command=[str(python),'-X','utf8','-m','aester_desktop.gui']
            env['PYTHONPATH']=str(root/'src')
        try:
            subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except OSError as error:
            self.component_message.setText(tr('未能重启，请关闭后从快捷方式打开：')+translate_message(str(error)));return
        w.close()
