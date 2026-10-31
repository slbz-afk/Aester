# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Application chrome only: photo pixels and annotation colours never depend on theme.

The 「拓星」 tokens (docs/UI_DESIGN_DIRECTION_20260927.md): every surface, border and
text colour is neutral (CIELAB C* <= 2); colour is reserved for the photograph and for
provenance (zhu = human, xiang = Agent).  Night vision is a separate red-only palette
applied on top of a theme, never a third theme, and never written into exports.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QDialog, QMessageBox, QTextBrowser, QToolTip, QWidget


DEFAULT_THEME = "dark"
THEME_LABELS = MappingProxyType({"dark": "深色", "light": "浅色"})
ASSETS = Path(__file__).parent / "assets"

# Semantic colours: interface only, never inside the photograph.
ZHU = "#d64532"          # human decision graphic / seal / error icon (>= 3:1 on window)
XIANG = "#e3cc8f"        # Agent proposal (dark)
SEAL_INK = "#161616"     # characters cut into the seal

COLORS = MappingProxyType({
    "dark": MappingProxyType({
        # Existing keys (kept for every caller).
        "window": "#161616", "surface": "#1b1b1b", "base": "#121212",
        "text": "#e6e6e3", "title": "#f2f2f0", "muted": "#a3a3a3",
        "disabled": "#7a7a7a", "disabled_bg": "#1b1b1b", "border": "#6b6b6b",
        "button": "#1b1b1b", "hover": "#262626", "pressed": "#101010",
        "selection": "#3a3a3a", "selected_text": "#f2f2f0",
        "accent": "#f2f2f0", "accent_hover": "#ffffff", "accent_text": "#161616",
        "focus": "#f2f2f0", "track": "#6b6b6b", "handle": "#e6e6e3",
        "link": "#f2f2f0", "tooltip": "#262626", "tooltip_text": "#e6e6e3",
        # 拓星 neutral scale.
        "canvas": "#1f1f1f", "raised": "#262626", "hairline": "#333333",
        "control_edge": "#6b6b6b", "inset": "#101010", "connector": "#3a3a3a",
        # Semantic (provenance / error).
        "zhu": ZHU, "zhu_text": "#ec6453", "xiang": XIANG, "xiang_text": XIANG,
        "error": "#ec6453", "seal_text": SEAL_INK,
        "empty_line": "#6b6b6b", "empty_star": "#e6e6e3",
    }),
    "light": MappingProxyType({
        "window": "#f2f2f1", "surface": "#e7e7e6", "base": "#ffffff",
        "text": "#1c1c1c", "title": "#1c1c1c", "muted": "#555555",
        "disabled": "#737373", "disabled_bg": "#e4e4e3", "border": "#808080",
        "button": "#ffffff", "hover": "#e7e7e6", "pressed": "#dadad9",
        "selection": "#d4d4d3", "selected_text": "#1c1c1c",
        "accent": "#1c1c1c", "accent_hover": "#333333", "accent_text": "#f2f2f1",
        "focus": "#1c1c1c", "track": "#808080", "handle": "#1c1c1c",
        "link": "#1c1c1c", "tooltip": "#ffffff", "tooltip_text": "#1c1c1c",
        # The photo area stays dark (玄) in the light theme.
        "canvas": "#1f1f1f", "raised": "#ffffff", "hairline": "#d4d4d3",
        "control_edge": "#858585", "inset": "#e7e7e6", "connector": "#c4c4c3",
        "zhu": ZHU, "zhu_text": "#b8321f", "xiang": "#8a6a14", "xiang_text": "#8a6a14",
        "error": "#b8321f", "seal_text": SEAL_INK,
        # Empty-state linework is drawn on the (dark) canvas, not on paper.
        "empty_line": "#6b6b6b", "empty_star": "#e6e6e3",
    }),
})

# Night vision: black ground, red only (every colour has G == B <= R).
NIGHT = MappingProxyType({
    "window": "#000000", "surface": "#000000", "base": "#000000",
    "text": "#ff1a1a", "title": "#ff1a1a", "muted": "#ff1a1a",
    "disabled": "#c20000", "disabled_bg": "#000000", "border": "#c20000",
    "button": "#000000", "hover": "#1f0000", "pressed": "#2a0000",
    "selection": "#2a0000", "selected_text": "#ff1a1a",
    "accent": "#000000", "accent_hover": "#1f0000", "accent_text": "#ff1a1a",
    "focus": "#ff1a1a", "track": "#c20000", "handle": "#ff1a1a",
    "link": "#ff1a1a", "tooltip": "#000000", "tooltip_text": "#ff1a1a",
    "canvas": "#000000", "raised": "#000000", "hairline": "#5a0000",
    "control_edge": "#c20000", "inset": "#000000", "connector": "#5a0000",
    "zhu": "#ff1a1a", "zhu_text": "#ff1a1a", "xiang": "#c20000", "xiang_text": "#ff1a1a",
    "error": "#ff1a1a", "seal_text": "#000000",
    "empty_line": "#c20000", "empty_star": "#ff1a1a",
})

# Photo surround (independent of theme; preference ``appearance/surround``).
SURROUNDS = MappingProxyType({"xuan": "#1f1f1f", "mid_grey": "#777777", "black": "#000000"})
DEFAULT_SURROUND = "xuan"

ROLES = ("mono", "human", "brand", "title", "display", "section", "field", "muted", "caption",
         "hint", "error", "primary", "secondary", "ghost", "segment", "segmented", "stage-action",
         "card", "panel", "toolbar", "stage", "readout", "chip", "badge", "value")
CONTAINER_ROLES = frozenset({"card", "panel", "toolbar", "stage", "readout", "chip", "segmented"})


def normalize_theme(name: object) -> str:
    """Unrecognised or old preference values safely retain the approved dark default."""
    return name if isinstance(name, str) and name in COLORS else DEFAULT_THEME


def normalize_surround(name: object) -> str:
    return name if isinstance(name, str) and name in SURROUNDS else DEFAULT_SURROUND


def colors(name: object = DEFAULT_THEME, *, night: bool = False) -> Mapping[str, str]:
    """The token mapping in force for a theme, or the night palette when night is on."""
    return NIGHT if night else COLORS[normalize_theme(name)]


def colors_for(widget: QWidget | None) -> Mapping[str, str]:
    """Tokens applied to the window that owns ``widget`` (read at paint time)."""
    current = widget
    while current is not None:
        theme = current.property("aesterTheme")
        if theme is not None:
            return colors(theme, night=bool(current.property("aesterNight")))
        current = current.parentWidget()
    return COLORS[DEFAULT_THEME]


def _palette_from(c: Mapping[str, str]) -> QPalette:
    result = QPalette()
    roles = {
        QPalette.ColorRole.Window: "window", QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Base: "base", QPalette.ColorRole.AlternateBase: "surface",
        QPalette.ColorRole.Text: "text", QPalette.ColorRole.Button: "button",
        QPalette.ColorRole.ButtonText: "text", QPalette.ColorRole.Highlight: "selection",
        QPalette.ColorRole.HighlightedText: "selected_text", QPalette.ColorRole.Link: "link",
        QPalette.ColorRole.LinkVisited: "link", QPalette.ColorRole.ToolTipBase: "tooltip",
        QPalette.ColorRole.ToolTipText: "tooltip_text", QPalette.ColorRole.PlaceholderText: "muted",
        QPalette.ColorRole.Light: "hover", QPalette.ColorRole.Mid: "border",
        QPalette.ColorRole.Dark: "pressed", QPalette.ColorRole.Shadow: "border",
        QPalette.ColorRole.BrightText: "title", QPalette.ColorRole.Accent: "focus",
        QPalette.ColorRole.Midlight: "hairline",
    }
    for role, key in roles.items():
        result.setColor(role, QColor(c[key]))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        result.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["disabled"]))
    result.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(c["disabled_bg"]))
    return result


def palette(name: str, *, night: bool = False) -> QPalette:
    return _palette_from(colors(name, night=night))


def _mode(name: object, night: bool) -> str:
    return "night" if night else normalize_theme(name)


def stylesheet(name: str, *, night: bool = False) -> str:
    c = dict(colors(name, night=night))
    mode = _mode(name, night)
    c["checkmark"] = (ASSETS / ("theme-check-" + mode + ".svg")).as_posix()
    c["spin_up"] = (ASSETS / ("spin-up-" + mode + ".svg")).as_posix()
    c["spin_down"] = (ASSETS / ("spin-down-" + mode + ".svg")).as_posix()
    c["primary_border"] = c["focus"] if night else c["accent"]
    # The checked view segment: the hairline fill, or night's dim red (4.93:1 with #ff1a1a).
    c["segment_on"] = c["pressed"] if night else c["hairline"]
    c["checked_weight"] = 500 if night else 400
    # "Accept the light-blue edge" wears the suggested edge's colour while a suggestion
    # waits (the overlay sky #66CCEE; a deeper tone keeps >= 3:1 on the light chrome).
    c["suggestion_edge"] = c["focus"] if night else ("#2b8cb0" if normalize_theme(name) == "light" else "#66CCEE")
    return """
QWidget { background: %(window)s; color: %(text)s; font-size: 13px; font-weight: 400;
          font-family: "IBM Plex Sans", "Noto Sans SC", "Microsoft YaHei UI", sans-serif; }
QMainWindow, QDialog, QScrollArea, QScrollArea > QWidget > QWidget { background: %(window)s; }
QLabel { background: transparent; }
QLabel#title, *[role="title"] { font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
          font-size: 16px; font-weight: 500; color: %(title)s; }
*[role="display"] { font-size: 28px; font-weight: 500; color: %(title)s; }
*[role="brand"] { font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif; font-size: 13px;
          font-weight: 500; color: %(title)s; }
QLabel#subtitle, QLabel#muted, QLabel#notice, QLabel#status, *[role="muted"] { color: %(muted)s; }
QLabel#section, *[role="section"] { font-size: 16px; font-weight: 500; color: %(title)s; }
QLabel#field, *[role="field"] { font-weight: 400; color: %(text)s; }
*[role="hint"] { font-size: 12px; color: %(muted)s; }
*[role="caption"] { font-family: "IBM Plex Mono", "Consolas", monospace; font-size: 11px; color: %(muted)s; }
*[role="mono"] { font-family: "IBM Plex Mono", "Consolas", monospace; }
*[role="human"] { font-family: "LXGW WenKai Lite", "Noto Sans SC", serif; font-size: 16px; }
*[role="error"] { color: %(zhu_text)s; }
QFrame#card, *[role="card"] { border: 1px solid %(hairline)s; border-radius: 8px; background: %(surface)s; }
*[role="panel"] { background: %(surface)s; border: none; border-left: 1px solid %(hairline)s; }
*[role="toolbar"] { background: %(window)s; border: none; }
*[role="stage"] { background: %(window)s; border: none; border-bottom: 1px solid %(hairline)s; }
*[role="readout"] { background: %(inset)s; border: none; border-top: 1px solid %(hairline)s; }
*[role="chip"] { background: %(surface)s; border: 1px solid %(hairline)s; border-radius: 4px;
                padding: 4px 8px; }
*[role="badge"] { background: transparent; color: %(text)s; border: 1px solid %(border)s;
                 border-radius: 3px; padding: 1px 6px; font-size: 11px; }
*[role="segmented"] { background: %(window)s; border: 1px solid %(hairline)s; border-radius: 6px; }
QLineEdit, QDoubleSpinBox, QSpinBox { padding: 6px 10px; background: %(base)s; border: 1px solid %(border)s;
                          border-radius: 4px; selection-background-color: %(selection)s;
                          selection-color: %(selected_text)s; }
QLineEdit[role="human"] { font-family: "LXGW WenKai Lite", "Noto Sans SC", serif; font-size: 16px; }
QDoubleSpinBox, QSpinBox { min-width: 88px; padding-right: 32px; }
QDoubleSpinBox QLineEdit, QSpinBox QLineEdit { padding: 0px; border: none; background: transparent; }
QDoubleSpinBox::up-button, QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
    width: 26px; border-left: 1px solid %(border)s; border-bottom: 1px solid %(border)s;
    background: %(button)s; }
QDoubleSpinBox::down-button, QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
    width: 26px; border-left: 1px solid %(border)s; background: %(button)s; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: %(hover)s; }
QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed,
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed { background: %(pressed)s; }
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow { image: url("%(spin_up)s"); width: 10px; height: 6px; }
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow { image: url("%(spin_down)s"); width: 10px; height: 6px; }
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {
    color: %(disabled)s; background: %(disabled_bg)s; border-color: %(hairline)s; }
/* Number boxes sized to their values: the stepper column is already outside the edit field. */
QDoubleSpinBox[role="mono"] { min-width: 0px; padding-right: 6px; }
QDoubleSpinBox[role="value"] { min-width: 0px; padding: 2px 4px; border: 1px solid transparent;
    border-radius: 4px; background: transparent; font-family: "IBM Plex Mono", "Consolas", monospace; }
QDoubleSpinBox[role="value"][changed="true"] { font-weight: 500; }
QDoubleSpinBox[role="value"]:hover { border-color: %(border)s; }
QDoubleSpinBox[role="value"]:focus { border-color: %(focus)s; background: %(base)s; }
QDoubleSpinBox[role="value"]:disabled { color: %(disabled)s; background: transparent; border-color: transparent; }
QComboBox { padding: 6px 10px; background: %(button)s; border: 1px solid %(border)s; border-radius: 4px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url("%(spin_down)s"); width: 10px; height: 6px; }
QComboBox QAbstractItemView { background: %(base)s; color: %(text)s; border: 1px solid %(border)s;
                            selection-background-color: %(selection)s; selection-color: %(selected_text)s; }
QPushButton { background: transparent; border: 1px solid %(border)s; padding: 6px 14px;
              border-radius: 6px; color: %(text)s; }
QPushButton:hover { background: %(hover)s; }
QPushButton:pressed { background: %(pressed)s; }
QPushButton:checked { background: %(selection)s; color: %(selected_text)s; border-color: %(focus)s;
                      font-weight: %(checked_weight)s; }
QPushButton:disabled { color: %(disabled)s; background: transparent; border-color: %(hairline)s; }
QPushButton#primary, QPushButton[role="primary"] { background: %(accent)s; color: %(accent_text)s;
              border: 1px solid %(primary_border)s; font-weight: 500; padding: 10px 20px; }
QPushButton#primary:hover, QPushButton[role="primary"]:hover { background: %(accent_hover)s; }
QPushButton#primary:disabled, QPushButton[role="primary"]:disabled {
              background: %(disabled_bg)s; color: %(disabled)s; border-color: %(hairline)s; }
QPushButton[role="ghost"] { border-color: transparent; color: %(muted)s; padding: 6px 10px; }
QPushButton[role="ghost"]:hover { color: %(text)s; }
QPushButton[role="ghost"]:disabled { color: %(disabled)s; }
QPushButton[role="stage-action"] { padding-left: 26px; }
QPushButton[role="segment"] { border: 1px solid transparent; border-radius: 4px; color: %(muted)s;
              padding: 5px 14px; font-size: 12px; }
QPushButton[role="segment"]:checked { background: %(segment_on)s; color: %(text)s; border-color: transparent; }
QPushButton[role="segment"]:checked:disabled { background: transparent; color: %(disabled)s; }
QPushButton[role="segment"]:focus { border: 1px solid %(focus)s; }
QPushButton[role="error"] { color: %(zhu_text)s; border-color: %(zhu)s; }
QPushButton[role="suggestion"] { border: 2px solid %(suggestion_edge)s; color: %(title)s; font-weight: 500;
              padding: 5px 13px; }
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QPushButton:focus,
QTextBrowser:focus, QListWidget:focus, QTreeWidget:focus { border: 1px solid %(focus)s; }
QProgressBar { border: none; border-radius: 1px; background: %(hairline)s; height: 4px; max-height: 4px; }
QProgressBar::chunk { background: %(focus)s; border-radius: 1px; }
QTabBar::tab { background: transparent; border-bottom: 1px solid %(hairline)s; padding: 7px 19px;
              color: %(muted)s; min-width: 42px; }
QTabBar::tab:selected { background: %(selection)s; color: %(selected_text)s; border-color: %(focus)s; }
QTabBar::tab:disabled { color: %(disabled)s; }
QScrollArea { border: none; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 0px; }
QScrollBar::handle:vertical { background: %(border)s; min-height: 24px; border-radius: 3px; margin: 2px; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 0px; }
QScrollBar::handle:horizontal { background: %(border)s; min-width: 24px; border-radius: 3px; margin: 2px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; border: none; background: transparent; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QMenuBar { background: %(window)s; padding: 4px 14px; border-bottom: 1px solid %(hairline)s; }
QMenuBar::item { padding: 6px 12px; background: transparent; }
QMenuBar::item:selected, QMenu::item:selected { background: %(selection)s; color: %(selected_text)s; }
QMenu { background: %(surface)s; border: 1px solid %(border)s; padding: 6px; }
QMenu::item { padding: 8px 28px 8px 24px; }
QMenu::item:disabled { color: %(disabled)s; }
QMenu::indicator { width: 8px; height: 8px; }
QMenu::indicator:checked { background: %(focus)s; border-radius: 4px; }
QMenu::separator { height: 1px; background: %(hairline)s; margin: 5px 8px; }
QTextBrowser { border: 1px solid transparent; background: %(base)s; padding: 16px;
               selection-background-color: %(selection)s; selection-color: %(selected_text)s; }
QToolTip { background: %(tooltip)s; color: %(tooltip_text)s; border: 1px solid %(border)s; padding: 4px 6px; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 2px; background: %(track)s; border-radius: 1px; }
QSlider::sub-page:horizontal, QSlider::add-page:horizontal { background: %(track)s; border-radius: 1px; }
QSlider::handle:horizontal { background: %(handle)s; border: 2px solid %(window)s; width: 10px; height: 10px;
                             margin: -6px 0; border-radius: 7px; }
QSlider::handle:horizontal:focus { background: %(window)s; border: 2px solid %(focus)s; }
QSlider::handle:horizontal:disabled { background: %(disabled)s; }
QCheckBox { spacing: 9px; padding: 5px 0; background: transparent; }
QCheckBox:disabled { color: %(disabled)s; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid %(border)s;
                       background: %(base)s; border-radius: 3px; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(primary_border)s; image: url("%(checkmark)s"); }
QCheckBox::indicator:focus { border: 1px solid %(focus)s; }
QCheckBox::indicator:disabled { background: %(disabled_bg)s; border-color: %(hairline)s; }
QListWidget, QTreeWidget { background: %(base)s; border: 1px solid %(hairline)s; padding: 4px;
                          selection-background-color: %(selection)s; selection-color: %(selected_text)s; }
QListWidget::item { padding: 6px 8px; border: 1px solid transparent; border-radius: 4px; margin: 2px; }
QTreeWidget::item { padding: 3px 0px; }
QListWidget::item:selected, QTreeWidget::item:selected { border-color: %(focus)s;
                          background: %(selection)s; color: %(selected_text)s; }
QHeaderView::section { background: %(surface)s; color: %(muted)s; border: 0px; padding: 5px; }
QSplitter::handle { background: %(hairline)s; }
""" % c


def set_role(widget: QWidget, role: str | None) -> QWidget:
    """Tag a widget for a ``[role="…"]`` rule in the window stylesheet and re-polish it."""
    widget.setProperty("role", role or None)
    if role in CONTAINER_ROLES:
        # Plain QWidget containers only paint QSS backgrounds/borders with this attribute.
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
    return widget


def role(widget: QWidget) -> str | None:
    value = widget.property("role")
    return value if isinstance(value, str) and value else None


def style_rich_text(browser: QTextBrowser, name: str, *, night: bool | None = None) -> None:
    """Qt Markdown stores blue in anchor runs; retint only links, keeping text/scroll/selection."""
    name = normalize_theme(name)
    if night is None:
        night = colors_for(browser) is NIGHT
    tokens = colors(name, night=night)
    browser.setPalette(palette(name, night=night))
    document = browser.document()
    document.setDefaultStyleSheet("a { color: " + tokens["link"] + "; }")
    modified = document.isModified()
    ranges = []
    block = document.begin()
    while block.isValid():
        fragment = block.begin()
        while not fragment.atEnd():
            run = fragment.fragment()
            if run.isValid() and run.charFormat().isAnchor():
                ranges.append((run.position(), run.length()))
            fragment += 1
        block = block.next()
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(tokens["link"]))
    for start, length in ranges:
        cursor = QTextCursor(document)
        cursor.setPosition(start)
        cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(fmt)
    document.setModified(modified)


# DWM window attributes (Windows 10 20H1+ / Windows 11).
_DWMWA_DARK = (20, 19)          # DWMWA_USE_IMMERSIVE_DARK_MODE (20H1+), older builds
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36
_DWM_COLOR_DEFAULT = 0xFFFFFFFF


def _colorref(hex_colour: str) -> int:
    """COLORREF is 0x00BBGGRR."""
    colour = QColor(hex_colour)
    return colour.red() | (colour.green() << 8) | (colour.blue() << 16)


def set_title_bar_dark(widget: QWidget, dark: bool, *, night: bool = False) -> bool:
    """Ask DWM for a dark (or light) caption on Windows; a no-op elsewhere and offscreen.

    In night vision the caption is also painted black with red text (Windows 11); the
    default caption colours come back when night is switched off.
    """
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows" or not widget.isWindow():
        return False
    try:
        import ctypes
        hwnd = ctypes.c_void_p(int(widget.winId()))
        dwm = ctypes.windll.dwmapi

        def put(attribute: int, value: int, kind=ctypes.c_int) -> bool:
            data = kind(value)
            return dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data)) == 0

        applied = any(put(attribute, 1 if (dark or night) else 0) for attribute in _DWMWA_DARK)
        # Caption / text colours are Windows 11 only; failures on older builds are ignored.
        put(_DWMWA_CAPTION_COLOR, _colorref(NIGHT["window"]) if night else _DWM_COLOR_DEFAULT, ctypes.c_uint)
        put(_DWMWA_TEXT_COLOR, _colorref(NIGHT["text"]) if night else _DWM_COLOR_DEFAULT, ctypes.c_uint)
        return applied
    except (AttributeError, OSError, ValueError):
        return False


def theme_state(widget: QWidget | None) -> tuple[str, bool]:
    """(theme, night) applied to the window that owns ``widget``."""
    current = widget
    while current is not None:
        theme = current.property("aesterTheme")
        if theme is not None:
            return normalize_theme(theme), bool(current.property("aesterNight"))
        current = current.parentWidget()
    return DEFAULT_THEME, False


def is_night(widget: QWidget | None) -> bool:
    return theme_state(widget)[1]


def message_icon(parent: QWidget | None) -> QMessageBox.Icon:
    """Question glyph by day; no (coloured) system glyph in night vision."""
    return QMessageBox.Icon.NoIcon if is_night(parent) else QMessageBox.Icon.Question


def sync_title_bar(widget: QWidget) -> bool:
    theme, night = theme_state(widget)
    return set_title_bar_dark(widget, night or theme == "dark", night=night)


class _CaptionFollower(QObject):
    """Keeps native captions of a themed window's dialogs in step with theme and night."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        kind = event.type()
        if kind in (QEvent.Type.ChildAdded, QEvent.Type.ChildPolished):
            # ChildPolished arrives once a child dialog is fully constructed.
            child = event.child()
            if isinstance(child, QDialog):
                child.removeEventFilter(self)
                child.installEventFilter(self)
        elif kind == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            sync_title_bar(watched)
        return False


def _follow_captions(window: QWidget) -> None:
    follower = window.findChild(_CaptionFollower, "aesterCaptionFollower")
    if follower is None:
        follower = _CaptionFollower(window)
        follower.setObjectName("aesterCaptionFollower")
        window.installEventFilter(follower)
    for dialog in window.findChildren(QDialog):
        dialog.removeEventFilter(follower)
        dialog.installEventFilter(follower)


def _native_dialogs_for(night: bool) -> None:
    """Night vision uses Qt's own file dialogs so they inherit the red window stylesheet."""
    app = QCoreApplication.instance()
    if app is None:
        return
    if app.property("aesterNativeDialogs") is None:
        app.setProperty("aesterNativeDialogs",
                        not QCoreApplication.testAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs))
    native = bool(app.property("aesterNativeDialogs"))
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, night or not native)


def apply_theme(widget: QWidget, name: str, *, night: bool = False) -> str:
    """Retint a window and its open child dialogs without changing application-global chrome.

    ``night`` overlays the red-only palette; dialogs parented to ``widget`` inherit the
    window stylesheet (no per-widget stylesheets are set).
    """
    name = normalize_theme(name)
    night = bool(night)
    colors_ = palette(name, night=night)
    widget.setProperty("aesterTheme", name)
    widget.setProperty("aesterNight", night)
    widget.setPalette(colors_)
    widget.setStyleSheet(stylesheet(name, night=night))
    # Child windows do not always inherit a parent's palette on every Qt platform.
    for dialog in widget.findChildren(QDialog):
        dialog.setPalette(colors_)
        if dialog.isVisible():
            sync_title_bar(dialog)
    for browser in widget.findChildren(QTextBrowser):
        style_rich_text(browser, name, night=night)
    # Tooltips are top-level Qt windows; their inactive palette must be explicit.
    QToolTip.setPalette(colors_)
    if widget.isWindow():
        _follow_captions(widget)
        _native_dialogs_for(night)
        if widget.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
            set_title_bar_dark(widget, night or name == "dark", night=night)
    widget.update()
    return name
