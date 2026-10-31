# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""About Aester: the GPLv3 "Appropriate Legal Notices" (section 5(d)) of the desktop app.

Shows the copyright line, both licences, the absence of warranty, that the source ships with the program, the
name/icon term, and links to the licence texts, the third-party notices and the trademark note in the installed tree.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

import aester_desktop
from starstack.i18n import tr

from . import themes

_PACKAGE = Path(aester_desktop.__file__).resolve().parent
# The engine's licence and notices sit at the top of the source tree, two levels above this package, in development
# and in an installation (versions\<version>\ holds LICENSE, NOTICE.md, THIRD_PARTY.md, TRADEMARKS.md beside src\).
_TREE = _PACKAGE.parents[1]


def _link(path: Path, label: str) -> str:
    if not path.is_file():
        return f"{label}（{tr('未找到')}）"
    return f"[{label}]({QUrl.fromLocalFile(str(path)).toString()})"


def about_markdown() -> str:
    links = [
        _link(_PACKAGE / "LICENSE.txt", tr('桌面程序许可证（GPL-3.0-or-later）')),
        _link(_TREE / "LICENSE", tr('处理引擎许可证（Apache-2.0）')),
        _link(_TREE / "THIRD_PARTY.md", tr('第三方组件')),
        _link(_TREE / "TRADEMARKS.md", tr('名称与图标')),
    ]
    return "\n\n".join([
        "## Aester",
        aester_desktop.__copyright__,
        tr('Aester 是自由软件，完整免费；支持开发完全自愿，不解锁任何功能。'),
        tr('桌面程序按 GNU 通用公共许可证第 3 版或（由您选择）任何更新版本授权：您可以按该许可证再分发和修改它。'
           '处理引擎、命令行与 MCP 服务按 Apache License 2.0 授权。'),
        tr('本程序不提供任何担保，包括适销性或特定用途适用性的默示担保；详见许可证。'),
        tr('完整源码随程序一起安装。'),
        tr('Aester 名称与图标不在许可范围内；修改后再发布的版本须改用其他名称和图标。'),
        "### " + tr('许可与源码'),
        "\n".join("- " + link for link in links),
    ])


def show_about(parent, theme_name: str) -> None:
    dialog = QDialog(parent)
    dialog.setObjectName("aboutDialog")
    dialog.setWindowTitle(tr('关于 Aester'))
    dialog.resize(640, 520)
    layout = QVBoxLayout(dialog)
    browser = QTextBrowser(dialog)
    browser.setObjectName("aboutText")
    browser.setOpenExternalLinks(True)
    browser.setMarkdown(about_markdown())
    themes.style_rich_text(browser, theme_name)
    layout.addWidget(browser, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
