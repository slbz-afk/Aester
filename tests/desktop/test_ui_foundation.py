"""「拓星」 foundation: token constraints (U2), bundled fonts (U5) and custom widgets."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFontInfo, QImage, QPainter, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QMainWindow, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from starstack import overlay_colors
from aester_desktop import fonts, themes, ui_widgets


# --------------------------------------------------------------------------- colour science

def _lin(value):
    value /= 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _hex_lin(color):
    return [_lin(int(color[i:i + 2], 16)) for i in (1, 3, 5)]


def _lab_from_lin(rgb):
    r, g, b = rgb
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _lab(color):
    return _lab_from_lin(_hex_lin(color))


def _chroma(color):
    _, a, b = _lab(color)
    return math.hypot(a, b)


def _luminance(color):
    r, g, b = _hex_lin(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    x, y = sorted((_luminance(a), _luminance(b)))
    return (y + 0.05) / (x + 0.05)


def _de2000(l1, l2):
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1 = math.degrees(math.atan2(b1, a1p)) % 360
    h2 = math.degrees(math.atan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dh = 0
    else:
        dh = h2 - h1
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
    dH = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2))
    Lm, Cmp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hm = h1 + h2
    elif abs(h1 - h2) <= 180:
        hm = (h1 + h2) / 2
    elif h1 + h2 < 360:
        hm = (h1 + h2 + 360) / 2
    else:
        hm = (h1 + h2 - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hm - 30)) + 0.24 * math.cos(math.radians(2 * hm))
         + 0.32 * math.cos(math.radians(3 * hm + 6)) - 0.20 * math.cos(math.radians(4 * hm - 63)))
    SL = 1 + 0.015 * (Lm - 50) ** 2 / math.sqrt(20 + (Lm - 50) ** 2)
    SC, SH = 1 + 0.045 * Cmp, 1 + 0.015 * Cmp * T
    RT = -2 * math.sqrt(Cmp ** 7 / (Cmp ** 7 + 25 ** 7)) * math.sin(math.radians(60 * math.exp(-((hm - 275) / 25) ** 2)))
    return math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH))


# Machado, Oliveira & Fernandes 2009, severity 1.0 (linear RGB).
MACHADO = {
    "normal": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}


def _simulated_lab(color, vision):
    rgb = _hex_lin(color)
    m = MACHADO[vision]
    return _lab_from_lin([min(max(sum(m[i][j] * rgb[j] for j in range(3)), 0), 1) for i in range(3)])


SEMANTIC_KEYS = {"zhu", "zhu_text", "xiang", "xiang_text", "error"}
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def app():
    value = QApplication.instance() or QApplication([])
    fonts.register_bundled_fonts()
    return value


def _render(widget, dpr):
    image = QImage(widget.size() * dpr, QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    widget.render(image)
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.width(), 4).copy()


def _count_colour(pixels, color, tolerance=2):
    target = np.asarray([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    return int(np.count_nonzero(np.all(np.abs(pixels[..., :3].astype(int) - target) <= tolerance, axis=2)))


@pytest.fixture
def themed(app):
    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    window.setCentralWidget(central)
    themes.apply_theme(window, "dark")
    window.resize(900, 420)
    yield window, layout
    window.close()
    app.processEvents()


# --------------------------------------------------------------------------- U2 tokens

@pytest.mark.parametrize("theme", ["dark", "light"])
def test_surfaces_text_and_borders_are_neutral(theme):
    c = themes.COLORS[theme]
    neutral = [key for key in c if key not in SEMANTIC_KEYS]
    assert {"window", "surface", "base", "text", "muted", "border", "hairline", "canvas"} <= set(neutral)
    for key in neutral:
        assert _chroma(c[key]) <= 2.0, (theme, key, c[key], round(_chroma(c[key]), 2))


def test_photo_surrounds_are_neutral_and_theme_independent():
    for name, value in themes.SURROUNDS.items():
        assert _chroma(value) <= 2.0, name
    assert themes.SURROUNDS[themes.DEFAULT_SURROUND] == "#1f1f1f"
    assert themes.COLORS["dark"]["canvas"] == themes.COLORS["light"]["canvas"] == "#1f1f1f"
    assert themes.normalize_surround("bogus") == themes.DEFAULT_SURROUND


def test_all_existing_theme_keys_retained_and_specified_values():
    assert set(themes.COLORS["dark"]) == set(themes.COLORS["light"]) == set(themes.NIGHT)
    d = themes.COLORS["dark"]
    assert (d["window"], d["surface"], d["raised"], d["hairline"], d["border"], d["muted"], d["text"], d["focus"]) == \
        ("#161616", "#1b1b1b", "#262626", "#333333", "#6b6b6b", "#a3a3a3", "#e6e6e3", "#f2f2f0")
    assert (d["accent"], d["accent_text"]) == ("#f2f2f0", "#161616")
    assert (d["zhu"], d["zhu_text"], d["xiang"]) == ("#d64532", "#ec6453", "#e3cc8f")
    lt = themes.COLORS["light"]
    assert (lt["window"], lt["text"], lt["muted"], lt["zhu_text"], lt["xiang_text"]) == \
        ("#f2f2f1", "#1c1c1c", "#555555", "#b8321f", "#8a6a14")


@pytest.mark.parametrize("mode", ["dark", "light", "night"])
def test_contrast_pairs(mode):
    c = themes.NIGHT if mode == "night" else themes.COLORS[mode]
    text_pairs = [("text", "window"), ("text", "base"), ("text", "surface"), ("text", "button"),
                  ("muted", "window"), ("muted", "surface"), ("title", "window"), ("selected_text", "selection"),
                  ("accent_text", "accent"), ("tooltip_text", "tooltip"), ("link", "base"),
                  ("zhu_text", "window"), ("zhu_text", "surface"), ("xiang_text", "window"), ("error", "window")]
    for fg, bg in text_pairs:
        assert _contrast(c[fg], c[bg]) >= 4.5, (mode, fg, bg, round(_contrast(c[fg], c[bg]), 2))
    graphic_pairs = [("border", "window"), ("border", "base"), ("border", "surface"), ("focus", "window"),
                     ("focus", "base"), ("disabled", "disabled_bg"), ("zhu", "window"), ("xiang", "window"),
                     ("handle", "window"), ("track", "window")]
    for fg, bg in graphic_pairs:
        assert _contrast(c[fg], c[bg]) >= 3.0, (mode, fg, bg, round(_contrast(c[fg], c[bg]), 2))


def test_night_palette_values_from_spec_and_red_only():
    n = themes.NIGHT
    assert (n["window"], n["text"], n["border"]) == ("#000000", "#ff1a1a", "#c20000")
    assert _contrast(n["text"], "#000000") >= 4.5 and _contrast(n["border"], "#000000") >= 3.0
    for key, value in n.items():
        r, g, b = (int(value[i:i + 2], 16) for i in (1, 3, 5))
        assert g == b and r >= g, (key, value)


def test_night_stylesheet_and_its_icons_contain_no_non_red_colour():
    sheet = themes.stylesheet("dark", night=True)
    colours = set(HEX.findall(sheet))
    for path in re.findall(r'url\("([^"]+)"\)', sheet):
        assert "-night" in Path(path).name and Path(path).is_file(), path
        colours |= set(HEX.findall(Path(path).read_text(encoding="utf-8")))
    assert colours
    for value in colours:
        r, g, b = (int(value[i:i + 2], 16) for i in (1, 3, 5))
        assert g == b and r >= g, value
    assert themes.stylesheet("light", night=True) == sheet  # night does not depend on the theme


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_semantic_colours_are_far_from_overlay_colours(mode):
    c = themes.COLORS[mode]
    for key in ("zhu", "zhu_text", "xiang", "xiang_text"):
        for name in overlay_colors.DISTINCT:
            distance = _de2000(_lab(c[key]), _lab(overlay_colors.OVERLAYS[name]["hex"]))
            assert distance >= 10, (mode, key, name, round(distance, 1))
    assert _de2000(_lab(c["zhu"]), _lab(c["xiang"])) >= 10


@pytest.mark.parametrize("vision", ["normal", "protan", "deutan", "tritan"])
def test_overlay_colours_distinct_under_colour_vision_deficiency(vision):
    names = overlay_colors.DISTINCT
    worst = min(_de2000(_simulated_lab(overlay_colors.OVERLAYS[a]["hex"], vision),
                        _simulated_lab(overlay_colors.OVERLAYS[b]["hex"], vision))
                for a, b in itertools.combinations(names, 2))
    assert worst >= 10, (vision, round(worst, 1))


def test_overlay_constants_are_consistent_and_qt_free():
    assert overlay_colors.SKY == "#66CCEE" and overlay_colors.GROUND == "#D55E00"
    assert overlay_colors.MILKYWAY == "#CC79A7" and overlay_colors.TRAIL == "#009E73"
    assert overlay_colors.HORIZON == overlay_colors.HANDLE == "#F0E442"
    assert overlay_colors.SUGGESTION == overlay_colors.SKY and overlay_colors.OUTLINE == "#101010"
    for spec in overlay_colors.OVERLAYS.values():
        assert spec["rgb"] == tuple(int(spec["hex"][i:i + 2], 16) for i in (1, 3, 5))
    assert overlay_colors.OVERLAYS["sky"]["line"] == overlay_colors.DASH
    assert overlay_colors.OVERLAYS["milkyway"]["line"] == overlay_colors.DOUBLE
    assert overlay_colors.OVERLAYS["trail"]["line"] == overlay_colors.DOT
    assert overlay_colors.rgba("ground", 128) == (0xD5, 0x5E, 0x00, 128)
    assert "PySide6" not in Path(overlay_colors.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", [("dark", False), ("light", False), ("dark", True)])
def test_stylesheet_weights_fonts_and_roles(mode):
    sheet = themes.stylesheet(mode[0], night=mode[1])
    weights = set(re.findall(r"font-weight:\s*(\w+)", sheet))
    assert weights <= {"400", "500"}, weights
    assert '"Noto Sans SC", "Microsoft YaHei UI", sans-serif' in sheet
    for role in ("mono", "human", "brand", "section", "muted", "primary", "error", "card", "toolbar", "stage"):
        assert f'[role="{role}"]' in sheet, role
    for name in ("QLabel#title", "QLabel#section", "QFrame#card", "QPushButton#primary", "QLabel#field"):
        assert name in sheet
    assert "padding-right: 32px" in sheet and "width: 26px" in sheet


# --------------------------------------------------------------------------- themes on real widgets

def test_night_applies_to_window_and_parented_dialogs_via_window_stylesheet_only(themed, app):
    window, layout = themed
    dialog = QDialog(window)
    label = QLabel("Reading", dialog)
    QVBoxLayout(dialog).addWidget(label)
    dialog.show()
    window.show()
    app.processEvents()
    assert themes.apply_theme(window, "light", night=True) == "light"
    app.processEvents()
    assert window.property("aesterNight") is True and window.property("aesterTheme") == "light"
    assert window.palette().color(QPalette.ColorRole.Window).name() == "#000000"
    assert dialog.palette().color(QPalette.ColorRole.WindowText).name() == "#ff1a1a"
    assert not dialog.styleSheet() and not label.styleSheet()
    assert themes.colors_for(label) is themes.NIGHT
    pixels = _render(dialog, 1)
    assert _count_colour(pixels, "#000000", 0) > pixels.shape[0] * pixels.shape[1] // 2
    themes.apply_theme(window, "light")
    app.processEvents()
    assert window.property("aesterNight") is False
    assert themes.colors_for(label) is themes.COLORS["light"]
    assert dialog.palette().color(QPalette.ColorRole.Window).name() == "#f2f2f1"
    dialog.close()


def test_apply_theme_keeps_backward_compatible_signature(themed):
    window, _ = themed
    assert themes.apply_theme(window, "unknown") == "dark"
    assert themes.palette("dark").color(QPalette.ColorRole.Window).name() == "#161616"
    assert themes.palette("dark", night=True).color(QPalette.ColorRole.Window).name() == "#000000"


def test_set_role_repolishes_and_selects_role_fonts(themed, app):
    window, layout = themed
    reading = QLabel("12.50 EV")
    words = QLabel("sky a little brighter")
    brand = QLabel("AESTER")
    button = QPushButton("Start")
    for widget in (reading, words, brand, button):
        layout.addWidget(widget)
    window.show()
    app.processEvents()
    themes.set_role(reading, "mono")
    themes.set_role(words, "human")
    themes.set_role(brand, "brand")
    themes.set_role(button, "primary")
    app.processEvents()
    assert themes.role(reading) == "mono"
    assert QFontInfo(reading.font()).family() == fonts.FAMILY_MONO
    assert QFontInfo(words.font()).family() == fonts.FAMILY_HUMAN
    assert QFontInfo(brand.font()).family() == fonts.FAMILY_BRAND
    # Latin and digits resolve to IBM Plex Sans; Han falls back to Noto Sans SC (spec §3.2).
    assert QFontInfo(button.font()).family() == fonts.FAMILY_BRAND
    assert button.font().families()[:2] == [fonts.FAMILY_BRAND, fonts.FAMILY_UI]
    pixels = _render(button, 1)
    assert _count_colour(pixels, themes.COLORS["dark"]["accent"], 1) > 100, "primary is the inverse fill"
    themes.set_role(reading, None)
    app.processEvents()
    assert themes.role(reading) is None and QFontInfo(reading.font()).family() == fonts.FAMILY_BRAND
    fonts.apply_tracking(brand, 0.2)
    assert brand.font().letterSpacing() > 0
    container = QWidget()
    themes.set_role(container, "card")
    assert container.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)


# --------------------------------------------------------------------------- U5 fonts

def test_bundled_fonts_register_idempotently_and_resolve(app):
    first = fonts.register_bundled_fonts()
    assert first == {"ui": "Noto Sans SC", "mono": "IBM Plex Mono", "human": "LXGW WenKai Lite", "brand": "IBM Plex Sans"}
    assert fonts.register_bundled_fonts() == first
    for name in fonts.FONT_FILES:
        assert (fonts.FONT_DIR / name).is_file(), name
    for factory, family in ((fonts.ui_font, fonts.FAMILY_BRAND), (fonts.mono_font, fonts.FAMILY_MONO),
                            (fonts.human_font, fonts.FAMILY_HUMAN), (fonts.brand_font, fonts.FAMILY_BRAND)):
        assert QFontInfo(factory()).family() == family
    medium = fonts.mono_font(11, 500)
    assert QFontInfo(medium).weight() == 500 and medium.pixelSize() == 11
    assert any(tag.toString() == "tnum" for tag in medium.featureTags())
    assert fonts.brand_font(13).letterSpacing() == pytest.approx(2.6, abs=0.05)


def test_qt_translator_loads_zh_cn_and_can_be_removed(app):
    try:
        translator = fonts.install_qt_translator(app, "zh_CN")
        assert translator is not None
        assert QCoreApplication.translate("QPlatformTheme", "Cancel") == "取消"
        assert fonts.install_qt_translator(app, "zh_CN") is not None  # replaces, never stacks
        assert len(fonts._translators) == 1
    finally:
        assert fonts.install_qt_translator(app, "en") is None
    assert QCoreApplication.translate("QPlatformTheme", "Cancel") == "Cancel"


# --------------------------------------------------------------------------- widgets

STAGES = [("materials", "01", "Photos", "Add photos"), ("reference", "02", "First frame", "Preview"),
          ("regions", "03", "Regions", "Mark regions"), ("stack", "04", "Stack", "Start"),
          ("result", "05", "Result", "View"), ("retouch", "06", "Retouch", "Agent")]


def test_stage_bar_states_clicks_connectors_and_trailing_button(themed, app):
    window, layout = themed
    bar = ui_widgets.StageBar()
    layout.addWidget(bar)
    bar.set_stages(STAGES)
    retouch = QPushButton("Let an Agent retouch")
    retouch.setObjectName("retouchAgentButton")
    bar.add_trailing_widget(retouch, provenance="agent")
    window.show()
    app.processEvents()
    assert bar.keys() == [s[0] for s in STAGES]
    node = bar.node("regions")
    assert isinstance(node, ui_widgets.StageNode) and node.accessibleName() == "Regions"
    assert node.toolTip() == "Mark regions" and node.focusPolicy() == Qt.FocusPolicy.StrongFocus
    bar.set_states({"materials": "done", "reference": "done", "regions": "current", "stack": "todo",
                    "result": "disabled", "retouch": "todo"})
    assert bar.connector_lit() == [True, True, False, False, False]
    assert bar.state("regions") == "current"
    clicked = []
    bar.stageClicked.connect(clicked.append)
    QTest.mouseClick(node, Qt.MouseButton.LeftButton)
    QTest.mouseClick(bar.node("result"), Qt.MouseButton.LeftButton)  # disabled: explains, never acts
    bar.node("stack").setFocus()
    QTest.keyClick(bar.node("stack"), Qt.Key.Key_Return)
    assert clicked == ["regions", "stack"]
    assert not bar.node("result").isEnabled()
    with pytest.raises(ValueError):
        bar.set_state("stack", "running")
    assert bar.trailing_widget() is retouch and retouch.parent() is bar
    assert themes.role(retouch) == "stage-action" and bar.trailing_dot().kind() == "agent"
    assert retouch.geometry().left() > bar.node("retouch").geometry().right()
    for dpr in (1, 2):
        pixels = _render(bar, dpr)
        assert _count_colour(pixels, themes.COLORS["dark"]["text"], 2) > 10
    dot_pixels = _render(retouch, 2)
    assert _count_colour(dot_pixels, themes.COLORS["dark"]["xiang"], 3) > 4
    themes.apply_theme(window, "light")
    app.processEvents()
    assert _count_colour(_render(bar, 1), themes.COLORS["light"]["text"], 2) > 10
    bar.set_stages(STAGES[:3])
    assert bar.keys() == ["materials", "reference", "regions"] and len(bar.connector_lit()) == 2


def test_frame_dots_reference_ring_overflow_and_states(themed, app):
    window, layout = themed
    overflow = QLabel()
    dots = ui_widgets.FrameDots(overflow_label=overflow)
    layout.addWidget(dots)
    layout.addWidget(overflow)
    window.show()
    dots.set_count(75, reference_index=0, max_dots=60)
    app.processEvents()
    assert dots.visible_count() + dots.overflow() == 75 and dots.overflow() >= 15
    assert overflow.text() == f"+{dots.overflow()}" and overflow.isVisible()
    dots.set_count(5)
    assert dots.overflow() == 0 and not overflow.isVisible()
    dots.set_states(["done", "skipped", "todo", "done", "done"])
    with pytest.raises(ValueError):
        dots.set_states(["done"])
    inline = ui_widgets.FrameDots()
    inline.resize(200, 16)
    inline.set_count(40, max_dots=60)
    assert inline.overflow() > 0 and inline.overflow_text().startswith("+")
    for dpr in (1, 2):
        assert _count_colour(_render(dots, dpr), themes.COLORS["dark"]["text"], 2) > 5
        assert _count_colour(_render(inline, dpr), themes.COLORS["dark"]["text"], 2) > 5


def test_tick_slider_is_a_qslider_with_centre_tick(themed, app):
    window, layout = themed
    slider = ui_widgets.TickSlider(Qt.Orientation.Horizontal)
    assert isinstance(slider, QSlider)
    slider.setRange(-100, 100)
    slider.set_default_value(0)
    layout.addWidget(slider)
    window.show()
    app.processEvents()
    values = []
    slider.valueChanged.connect(values.append)
    slider.setValue(40)
    assert values == [40]
    centre = [t for t in slider.tick_positions() if t[2]]
    assert len(centre) == 1 and centre[0][1] == 0
    groove_mid = slider.width() / 2
    assert abs(centre[0][0] - groove_mid) <= 2
    assert len(slider.tick_positions()) == 9
    assert slider.sizeHint().height() == QSlider.sizeHint(slider).height() + slider.TICK_SPACE
    QTest.mouseDClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(5, slider.height() // 2))
    assert slider.value() == 0
    unipolar = ui_widgets.TickSlider(parent=window)
    unipolar.setRange(0, 100)
    assert not unipolar.is_bipolar() and not any(t[2] for t in unipolar.tick_positions())
    for dpr in (1, 2):
        pixels = _render(slider, dpr)
        assert _count_colour(pixels, themes.COLORS["dark"]["border"], 2) > 5


def test_seal_provenance_error_icon_paint_at_dpr_1_and_2(themed, app):
    window, layout = themed
    seal = ui_widgets.Seal("人定", "Human confirmed", size=40)
    small = ui_widgets.Seal("人定", "Human confirmed", size=24)
    latin = ui_widgets.Seal("OK", "Human confirmed", size=24)
    human = ui_widgets.ProvenanceDot("human", accessible_name="You")
    agent = ui_widgets.ProvenanceDot("agent")
    algorithm = ui_widgets.ProvenanceDot("algorithm")
    error = ui_widgets.ErrorIcon(16, accessible_name="Error")
    for widget in (seal, small, latin, human, agent, algorithm, error):
        layout.addWidget(widget)
    window.show()
    app.processEvents()
    assert seal.size() == QSize(40, 40) and small.size() == QSize(24, 24)
    assert seal.accessibleName() == "Human confirmed" and error.accessibleName() == "Error"
    assert human.size() == QSize(8, 8)
    with pytest.raises(ValueError):
        ui_widgets.ProvenanceDot("robot")
    c = themes.COLORS["dark"]
    for dpr in (1, 2):
        seal_pixels = _render(seal, dpr)
        assert _count_colour(seal_pixels, c["zhu"], 1) > seal_pixels.shape[0] * seal_pixels.shape[1] // 2
        assert _count_colour(seal_pixels, c["seal_text"], 30) > 10, "characters are cut in"
        assert _count_colour(_render(latin, dpr), c["zhu"], 1) > 100
        assert _count_colour(_render(human, dpr), c["zhu"], 2) > 4
        assert _count_colour(_render(agent, dpr), c["xiang"], 2) > 4
        ring = _render(algorithm, dpr)
        centre = ring[ring.shape[0] // 2, ring.shape[1] // 2, :3].astype(int)
        assert _count_colour(ring, c["text"], 40) > 4 and centre.max() < 0x80, "hollow ring"
        assert _count_colour(_render(error, dpr), c["zhu"], 30) > 10


@pytest.mark.parametrize("dpr", [1, 2])
def test_empty_state_is_grey_only_and_dither_cached_per_dpr(app, dpr):
    image = QImage(QSize(480, 320) * dpr, QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(dpr)
    image.fill(0xFF1F1F1F)
    painter = QPainter(image)
    motif = ui_widgets.paint_empty_state(painter, QRect(0, 0, 480, 320), themes.COLORS["dark"])
    painter.end()
    assert motif.width() > 100 and QRect(0, 0, 480, 320).contains(motif.toAlignedRect())
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.width(), 4)
    rgb = pixels[..., :3].astype(int)
    assert int((rgb.max(axis=2) - rgb.min(axis=2)).max()) <= 3, "grey only"
    assert _count_colour(pixels, "#e6e6e3", 1) >= 20 * dpr * dpr, "dithered star"
    assert np.count_nonzero(np.abs(rgb - 0x1F).max(axis=2) > 8) > 500 * dpr, "linework"
    first = ui_widgets.dither_star_pixmap(60, "#e6e6e3", float(dpr))
    assert ui_widgets.dither_star_pixmap(60, "#e6e6e3", float(dpr)) is first
    assert first.devicePixelRatio() == dpr and first.width() % ui_widgets.STAR_CELLS == 0
    assert ui_widgets.empty_state_colors("mid_grey")["empty_line"] != ui_widgets.empty_state_colors("xuan")["empty_line"]
    assert ui_widgets.empty_state_colors(night=True)["empty_star"] == "#ff1a1a"


def test_widget_modules_define_no_user_visible_chinese_strings():
    import ast
    for module in (ui_widgets, fonts, overlay_colors):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body
                      and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert not re.search(r"[一-鿿]", node.value), (module.__name__, node.value)


@pytest.mark.parametrize("mode", ["dark", "light", "night"])
def test_checked_view_segment_and_selection_text_contrast(mode):
    night = mode == "night"
    c = themes.NIGHT if night else themes.COLORS[mode]
    sheet = themes.stylesheet("dark" if night else mode, night=night)
    rule = re.search(r'QPushButton\[role="segment"\]:checked \{ background: (#[0-9a-f]{6}); color: (#[0-9a-f]{6})', sheet)
    assert rule, "checked segment rule"
    assert _contrast(rule.group(2), rule.group(1)) >= 4.5, (mode, rule.groups())
    assert 'QPushButton[role="segment"]:checked:disabled' in sheet
    if night:
        # Checked / selected is a dim fill with red text, never a full-intensity red block.
        assert c["selection"] != c["text"] and c["selected_text"] == c["text"]


# --------------------------------------------------------------------------- activity (二十八宿 sweep)

def _red_only(pixels):
    rgb = pixels[..., :3].astype(int)
    return bool(np.all(rgb[..., 1] == rgb[..., 2]) and np.all(rgb[..., 1] <= rgb[..., 0]))


def test_reduced_motion_follows_the_override(monkeypatch):
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "1")
    assert ui_widgets.reduced_motion()
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "0")
    assert not ui_widgets.reduced_motion()
    assert ui_widgets.sweep_level(3, 3) == 1.0 and ui_widgets.sweep_level(0, 7) == 0.0
    assert ui_widgets.sweep_level(27, 0) == pytest.approx(1 - 1 / 7)       # the tail wraps round


@pytest.mark.parametrize("mode", ["dark", "light", "night"])
def test_activity_marks_sweep_and_paint_at_dpr_1_and_2(themed, app, mode, monkeypatch):
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "0")
    window, layout = themed
    themes.apply_theme(window, "light" if mode == "light" else "dark", night=mode == "night")
    clock = ui_widgets.ActivityClock(window)
    spinner = ui_widgets.LodgeSpinner(clock)
    bar = ui_widgets.StageBar()
    bar.set_stages(STAGES[:4])
    layout.addWidget(spinner)
    layout.addWidget(bar)
    window.show()
    app.processEvents()
    assert spinner.size() == QSize(16, 16) and spinner.sizePolicy().retainSizeWhenHidden()
    bar.set_states({"materials": "done", "reference": "done", "regions": "done", "stack": "current"})
    bar.set_busy("stack", clock)
    assert bar.busy_key() == "stack" and bar.node("stack").is_busy() and not bar.node("regions").is_busy()
    clock.set_active(True)
    assert clock.is_running()
    tokens = themes.colors_for(spinner)
    floor, head = ui_widgets.activity_colors(tokens)
    if mode == "night":
        assert (floor, head) == (themes.NIGHT["hairline"], themes.NIGHT["border"])   # head capped at border
    node = bar.node("stack")
    ground = tokens["window"]
    for dpr in (1, 2):
        clock._set_step(0)
        first = _render_on(spinner, dpr, ground)
        ring_first = _render_on(node, dpr, ground)
        clock._set_step(14)
        second = _render_on(spinner, dpr, ground)
        ring_second = _render_on(node, dpr, ground)
        assert not np.array_equal(first, second), "the head moved"
        assert not np.array_equal(ring_first, ring_second)
        assert np.count_nonzero(_contrast_to(first, ground) > 8) > 12 * dpr, "lodge lines drawn"
        if mode == "night":
            assert _red_only(first) and _red_only(second) and _red_only(ring_first)
    # At step 0 the head (lodge 0, straight up) and its tail (upper left) stand out most.
    clock._set_step(0)
    contrast = _contrast_to(_render_on(spinner, 2, ground), ground)
    half = contrast.shape[0] // 2
    assert contrast[:half].sum() > contrast[half:].sum()
    # Idle and reduced motion: never animating; reduced draws one still frame.
    clock.set_active(False)
    assert not clock.is_running() and not clock.is_static()
    monkeypatch.setenv("AESTER_REDUCED_MOTION", "1")
    clock.set_active(True)
    assert not clock.is_running() and clock.is_static()
    still = _render(spinner, 1)
    clock._set_step(9)
    assert np.array_equal(still, _render(spinner, 1))
    bar.set_busy(None)
    assert bar.busy_key() is None
    window.hide()
    app.processEvents()


def _render_on(widget, dpr, colour):
    """Render over an opaque ground in ``colour`` (the marks paint no background of their own)."""
    image = QImage(widget.size() * dpr, QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(dpr)
    image.fill(QColor(colour))
    widget.render(image)
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.width(), 4).copy()


def _contrast_to(pixels, colour):
    target = np.asarray([int(colour[i:i + 2], 16) for i in (1, 3, 5)])
    return np.abs(pixels[..., :3].astype(int) - target).max(axis=2)


@pytest.mark.parametrize("dpr", [1, 2])
def test_empty_state_sweep_changes_only_the_lodge_lines(app, dpr):
    def paint(sweep, colours):
        image = QImage(QSize(480, 320) * dpr, QImage.Format.Format_ARGB32)
        image.setDevicePixelRatio(dpr)
        image.fill(0xFF1F1F1F if colours is not ui_widgets.empty_state_colors(night=True) else 0xFF000000)
        painter = QPainter(image)
        ui_widgets.paint_empty_state(painter, QRect(0, 0, 480, 320), colours, sweep=sweep)
        painter.end()
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        return np.frombuffer(converted.constBits(), np.uint8).reshape(converted.height(), converted.width(), 4).copy()
    day = ui_widgets.empty_state_colors("xuan")
    still, a, b = paint(None, day), paint(0, day), paint(14, day)
    assert not np.array_equal(a, b) and not np.array_equal(still, a)
    for pixels in (a, b):
        rgb = pixels[..., :3].astype(int)
        assert int((rgb.max(axis=2) - rgb.min(axis=2)).max()) <= 3, "grey only"
    night = ui_widgets.empty_state_colors(night=True)
    assert night["sweep_head"] == themes.NIGHT["border"] and night["sweep_floor"] == themes.NIGHT["hairline"]
    assert _red_only(paint(5, night))
