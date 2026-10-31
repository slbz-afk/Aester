"""Language is a presentation choice; review and region data stay unchanged."""
import ast
import copy
from pathlib import Path
from string import Formatter

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from starstack import i18n, regions
from aester_desktop import trail_review_dialog
from aester_desktop.region_dialog import RegionDialog
from test_region_preview_p1 import scene
from test_trail_review_dialog import session


@pytest.fixture(autouse=True)
def restore_language():
    old = i18n.get_language()
    yield
    i18n.set_language(old)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_language_preference_and_installer_hint(tmp_path, monkeypatch):
    monkeypatch.delenv('AESTER_LANGUAGE', raising=False)
    monkeypatch.setenv('AESTER_INSTALL_ROOT', str(tmp_path))
    (tmp_path/'install-language.txt').write_text('en', encoding='utf8')
    settings = QSettings(str(tmp_path/'settings.ini'), QSettings.IniFormat)
    assert i18n.startup_language(settings) == 'en'
    settings.setValue('appearance/language', 'zh_CN')
    assert i18n.startup_language(settings) == 'zh_CN'
    settings.remove('appearance/language')
    monkeypatch.setenv('AESTER_LANGUAGE', 'zh_CN')
    assert i18n.startup_language(settings) == 'zh_CN'


def test_template_values_are_not_translated_or_evaluated():
    i18n.set_language('en')
    name = '银河{0}原片.NEF'
    assert i18n.tr('{0} 条有本次选择', name) == name + ' candidates have review choices'
    assert i18n.tr('external error {unbound}') == 'external error {unbound}'


def test_english_region_edit_preserves_project_and_provenance(app, scene):
    project, reference, image, out = scene
    baseline = copy.deepcopy(project)
    i18n.set_language('en')
    view = RegionDialog(reference, project)
    assert view.windowTitle() == 'Sky, foreground and Milky Way regions'
    assert view.tools.currentText() == 'Mark horizon'
    assert view.tools.currentData() == 'horizon'
    assert view.canvas.project == baseline
    view.reject()


def test_english_trail_labels_keep_machine_decisions(app, session, monkeypatch):
    monkeypatch.setattr(trail_review_dialog.service, 'candidate_context', lambda *a: {'panels': []})
    i18n.set_language('en')
    before = copy.deepcopy(session)
    view = trail_review_dialog.TrailReviewDialog(session)
    try:
        view.select_candidate('trail-1')
        view.label_choice.setCurrentIndex(view.label_choice.findData('meteor'))
        view.action_choice.setCurrentIndex(view.action_choice.findData('keep'))
        assert view.label_choice.currentText() == 'Meteor'
        assert view.action_choice.currentText() == 'Keep'
        view.set_selected_decisions()
        assert view.decisions == {'trail-1': {'label': 'meteor', 'override': 'keep'}}
        assert session == before
    finally:
        view._dirty = False
        view._finish_workers()
        view.reject()
        app.processEvents()


def test_dialog_catalog_covers_literals_and_format_fields():
    from aester_desktop.translations_dialogs import EN
    catalog = i18n.english_catalog()
    for name in ('region_dialog', 'trail_review_dialog', 'advanced_tools'):
        source = Path(__file__).resolve().parents[2]/'src/aester_desktop'/(name+'.py')
        for node in ast.walk(ast.parse(source.read_text(encoding='utf8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tr' and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    assert arg.value in catalog, (name, node.lineno, arg.value)
    for source, target in EN.items():
        fields = lambda value: [(name, spec, conv) for _, name, spec, conv in Formatter().parse(value) if name is not None]
        assert sorted(fields(source)) == sorted(fields(target)), source
