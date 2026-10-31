"""Real Qt checks for theme persistence, readable states and photograph invariance."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import copy

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFontDatabase, QImage, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QMenu, QStyle, QStyleOptionButton

from starstack import regions
from aester_desktop import gui, themes, trail_review_dialog
from aester_desktop.region_dialog import RegionDialog


@pytest.fixture(scope="module")
def app():
    value = QApplication.instance() or QApplication([])
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
    return value


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    result = gui.MainWindow(persist_preferences=False)
    result.show()
    app.processEvents()
    yield result
    result._color_timer.stop()
    result.timer.stop()
    result.close()
    app.processEvents()


def _pixels(image):
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return converted.size(), bytes(converted.constBits())


def _contrast(a, b):
    def luminance(color):
        values = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
        values = [v / 12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in values]
        return sum(v*w for v, w in zip(values, (.2126, .7152, .0722)))
    aa, bb = sorted((luminance(a), luminance(b)))
    return (bb+.05)/(aa+.05)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_readable_text_selection_tooltip_and_keyboard_focus(theme):
    c = themes.COLORS[theme]
    for foreground, background in (("text", "window"), ("text", "base"), ("text", "button"),
                                   ("muted", "surface"), ("muted", "window"),
                                   ("selected_text", "selection"), ("link", "base"),
                                   ("tooltip_text", "tooltip"), ("accent_text", "accent")):
        assert _contrast(c[foreground], c[background]) >= 4.5, (theme, foreground, background)
    assert _contrast(c["disabled"], c["disabled_bg"]) >= 3
    assert _contrast(c["focus"], c["window"]) >= 3
    assert _contrast(c["focus"], c["base"]) >= 3
    p = themes.palette(theme)
    assert p.color(QPalette.ColorRole.Highlight).name() == c["selection"]
    assert p.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name() == c["disabled"]


def test_menu_keyboard_switch_is_exclusive_and_persists_across_windows(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, "check_backend", lambda self: None)
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    first = gui.MainWindow(preferences=settings)
    first.show()
    app.processEvents()
    try:
        menu = first.findChild(QMenu, "appearanceMenu")
        menu.popup(first.mapToGlobal(first.rect().center()))
        menu.setActiveAction(first.menu_actions["theme_light"])
        QTest.keyClick(menu, Qt.Key.Key_Return)
        app.processEvents()
        assert first._theme_name == "light"
        assert first.menu_actions["theme_light"].isChecked()
        assert not first.menu_actions["theme_dark"].isChecked()
        assert settings.value("appearance/theme") == "light"
    finally:
        first.close()
    restored = gui.MainWindow(preferences=settings)
    try:
        assert restored._theme_name == "light"
        assert restored.palette().color(QPalette.ColorRole.Window).name() == themes.COLORS["light"]["window"]
    finally:
        restored.close()


@pytest.mark.parametrize("invalid", [None, "system", "unknown", [], 3])
def test_unknown_preference_retains_dark_default(invalid):
    assert themes.normalize_theme(invalid) == "dark"


def test_help_open_before_and_after_theme_switch_retints_anchors_without_changing_contents(window, app):
    window.show_help()
    browser = window.help_browser
    browser.setMarkdown("# 说明\n\n[可访问的链接](https://example.com)\n\n" + "后期调整说明。\n\n" * 40)
    themes.style_rich_text(browser, "dark")
    browser.verticalScrollBar().setValue(45)
    text = browser.toPlainText()
    scrollbar = browser.verticalScrollBar().value()
    window.set_theme("light")
    app.processEvents()
    assert browser.toPlainText() == text
    assert browser.verticalScrollBar().value() == scrollbar
    links = []
    block = browser.document().begin()
    while block.isValid():
        fragments = block.begin()
        while not fragments.atEnd():
            fmt = fragments.fragment().charFormat()
            if fmt.isAnchor():
                links.append(fmt.foreground().color().name())
            fragments += 1
        block = block.next()
    assert links and set(links) == {themes.COLORS["light"]["link"]}
    window.show_help()
    assert browser.palette().color(QPalette.ColorRole.Link).name() == themes.COLORS["light"]["link"]


def test_image_preview_changing_theme_never_reprocesses_or_changes_display_pixels(window, app, monkeypatch):
    pixels = np.zeros((60, 90, 3), np.uint8)
    pixels[..., 0] = np.arange(90, dtype=np.uint8)[None, :] * 2
    pixels[..., 1] = 80
    pixels[25:40, 25:40, 2] = 240
    image = QImage(pixels.data, 90, 60, 90*3, QImage.Format.Format_RGB888).copy()
    window.preview.set_image(image)
    calls = []
    monkeypatch.setattr(window, "_launch_reference_preview", lambda: calls.append(True))
    app.processEvents()
    before = _pixels(window.preview.grab().toImage())
    source = _pixels(window.preview._pixmap.toImage())
    window.set_theme("light")
    app.processEvents()
    assert _pixels(window.preview._pixmap.toImage()) == source
    assert _pixels(window.preview.grab().toImage()) == before
    assert not calls and not window._color_timer.isActive()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_checked_option_has_visible_tick_not_only_coloured_box(window, app, theme):
    window.set_theme(theme)
    control = QCheckBox("已开启", window)
    control.resize(120, 40)
    control.setChecked(True)
    control.show()
    app.processEvents()
    rendered = control.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(rendered.constBits(), np.uint8).reshape(rendered.height(), rendered.width(), 4)
    # 2026-09-27 「拓星」: the tick is the neutral inverse of the checked fill (accent_text on
    # accent), no longer the green-tinted #232920.  Look inside the indicator box only, because
    # the dark tick now equals the window colour around the box.
    option = QStyleOptionButton()
    control.initStyleOption(option)
    box = control.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, control).adjusted(2, 2, -2, -2)
    indicator = pixels[box.top():box.bottom() + 1, box.left():box.right() + 1, :3].astype(int)
    c = themes.COLORS[theme]
    def count(color):
        target = np.asarray([int(color[i:i+2], 16) for i in (1, 3, 5)])
        return np.count_nonzero(np.all(np.abs(indicator - target) <= 1, axis=2))
    assert count(c["accent_text"]) >= 5
    assert count(c["accent"]) >= 20
    control.hide()


def test_existing_region_and_trail_windows_inherit_theme_without_changing_images_or_overlays(window, app, tmp_path, monkeypatch):
    source = tmp_path / "source.nef"
    source.write_bytes(b"synthetic source")
    linear = tmp_path / "reference.fit"
    fits.writeto(linear, np.zeros((3, 60, 90), np.float32))
    preview = tmp_path / "preview.png"
    Image.new("RGB", (90, 60), (23, 57, 88)).save(preview)
    reference = {"state": "succeeded", "source_path": str(source), "source_sha256": regions._sha(source),
                 "linear_path": str(linear), "linear_sha256": regions._sha(linear), "width": 90, "height": 60,
                 "preview_path": str(preview), "preview_sha256": regions._sha(preview)}
    project = regions.create_project(reference, [str(source)])
    project = regions.revise(project, scene_mode="landscape", operations=[{"kind": "horizon", "points": [[0, .7], [1, .7]]}],
                             milkyway={**project["milkyway"], "ellipse": [.5, .3, .15, .2]})
    editor = RegionDialog(reference, project, window)
    session = {"session_path": str(tmp_path / "draft.json"), "draft_revision": "1", "base_revision": {"id": "1"},
               "geometry": {"width": 90, "height": 60}, "decisions": {}, "preview_path": str(preview),
               "candidates": [{"id": "trail-1", "frame_index": 1, "suggestion": {"class": "unknown"},
                               "decision": "omit", "endpoints_native_reference": [[10, 10], [50, 20]]}]}
    monkeypatch.setattr(trail_review_dialog.service, "candidate_context", lambda *args: {"panels": []})
    trails = trail_review_dialog.TrailReviewDialog(session, window)
    try:
        editor.show()
        trails.show()
        app.processEvents()
        # Let the exposure-context worker deliver its text first: the comparison below is
        # about the theme switch, not about that asynchronous label update.
        for worker in list(trails._workers):
            worker.wait(5000)
        app.processEvents()
        before_project = copy.deepcopy(editor.canvas.project)
        raw = _pixels(editor.canvas.image)
        overlay = _pixels(editor.canvas._overlay)
        region_display = _pixels(editor.canvas.grab().toImage())
        trail_display = _pixels(trails.location.grab().toImage())
        window.set_theme("light")
        app.processEvents()
        assert not trails.list.styleSheet(), "child stylesheet would mask future theme changes"
        assert editor.palette().color(QPalette.ColorRole.Window).name() == themes.COLORS["light"]["window"]
        assert trails.list.palette().color(QPalette.ColorRole.Text).name() == themes.COLORS["light"]["text"]
        assert editor.canvas.project == before_project
        assert _pixels(editor.canvas.image) == raw
        assert _pixels(editor.canvas._overlay) == overlay
        assert _pixels(editor.canvas.grab().toImage()) == region_display
        assert _pixels(trails.location.grab().toImage()) == trail_display
    finally:
        editor.reject()
        trails._dirty = False
        trails._finish_workers()
        trails.reject()
