# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Bundled OFL fonts (「拓星」 type roles) and Qt's own translations for standard widgets.

Register once after the QApplication exists.  The window stylesheet sets the UI family
for every widget, so role fonts are chosen with ``themes.set_role(widget, "mono")`` etc.;
the helpers below are for QPainter text, which the stylesheet does not reach.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication


FONT_DIR = Path(__file__).parent / "assets" / "fonts"
FAMILY_UI = "Noto Sans SC"
FAMILY_MONO = "IBM Plex Mono"
FAMILY_HUMAN = "LXGW WenKai Lite"
FAMILY_BRAND = "IBM Plex Sans"
FALLBACK_UI = ("Microsoft YaHei UI", "sans-serif")
FALLBACK_MONO = ("Consolas", "monospace")
ROLE_FAMILIES = {"ui": FAMILY_UI, "mono": FAMILY_MONO, "human": FAMILY_HUMAN, "brand": FAMILY_BRAND}
FONT_FILES = (
    "NotoSansSC-Regular.otf", "NotoSansSC-Medium.otf",
    "IBMPlexSans-Regular.ttf", "IBMPlexSans-Medium.ttf", "IBMPlexSans-SemiBold.ttf",
    "IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf",
    "LXGWWenKaiLite-Regular.ttf",
)

_registered: dict[str, str] | None = None
_translators: list[QTranslator] = []


def register_bundled_fonts() -> dict[str, str]:
    """Add the packaged fonts to Qt's database once; returns {role: family} that resolved.

    Returns an empty mapping (and retries next call) when no QGuiApplication exists yet.
    Missing files are skipped so a damaged install still starts with system fallbacks.
    """
    global _registered
    if _registered is not None:
        return dict(_registered)
    if QGuiApplication.instance() is None:
        return {}
    for name in FONT_FILES:
        path = FONT_DIR / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    available = set(QFontDatabase.families())
    _registered = {role: family for role, family in ROLE_FAMILIES.items() if family in available}
    return dict(_registered)


def _font(families: list[str], size: int, weight: int) -> QFont:
    font = QFont()
    font.setFamilies(families)
    font.setPixelSize(int(size))
    font.setWeight(QFont.Weight(int(weight)))
    return font


def ui_font(size: int = 13, weight: int = 400) -> QFont:
    """Latin and digits in IBM Plex Sans, Han in Noto Sans SC."""
    return _font([FAMILY_BRAND, FAMILY_UI, *FALLBACK_UI], size, weight)


def mono_font(size: int = 11, weight: int = 400) -> QFont:
    """Readout font with tabular figures (``tnum``) when the Qt build supports features."""
    font = _font([FAMILY_MONO, *FALLBACK_MONO, FAMILY_UI], size, weight)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    try:
        font.setFeature(QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError):
        pass
    return font


def human_font(size: int = 16, weight: int = 400) -> QFont:
    """Only for words a person wrote: instructions, annotations, confirmations."""
    return _font([FAMILY_HUMAN, FAMILY_UI, *FALLBACK_UI], size, weight)


def brand_font(size: int = 13, weight: int = 500, tracking: float = 0.2) -> QFont:
    """IBM Plex Sans word mark; letter spacing in em (QSS cannot express tracking)."""
    font = _font([FAMILY_BRAND, FAMILY_UI, *FALLBACK_UI], size, weight)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, float(size) * float(tracking))
    return font


def apply_tracking(widget, tracking: float = 0.2) -> None:
    """Letter-space a widget's current (stylesheet-resolved) font, e.g. the brand label."""
    font = widget.font()
    size = font.pixelSize() if font.pixelSize() > 0 else font.pointSizeF() * 96 / 72
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, float(size) * float(tracking))
    widget.setFont(font)


def install_qt_translator(app: QCoreApplication | None, language: str) -> QTranslator | None:
    """Load Qt's own zh_CN strings (context menus, standard buttons); no-op when missing.

    Calling again replaces the previously installed Qt translator; ``en`` removes it.
    """
    app = app or QCoreApplication.instance()
    if app is None:
        return None
    for old in _translators:
        app.removeTranslator(old)
    _translators.clear()
    if not str(language or "").lower().startswith("zh"):
        return None
    directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    translator = QTranslator(app)
    loaded = False
    for base in ("qtbase", "qt"):
        if translator.load(QLocale(QLocale.Language.Chinese, QLocale.Country.China), base, "_", directory):
            loaded = True
            break
    if not loaded:
        return None
    app.installTranslator(translator)
    _translators.append(translator)
    return translator
