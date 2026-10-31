"""Independent language boundary checks: no user settings or production edits."""
import copy
import asyncio
import json
import os
from pathlib import Path
import re

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QPushButton, QWidget, QTextBrowser

from starstack import i18n, regions
from aester_desktop import trail_review_dialog as trails
from aester_desktop.region_dialog import RegionDialog
from aester_desktop.advanced_tools import AdvancedTools
from test_region_preview_p1 import scene
from test_trail_review_dialog import session
from test_trail_review_session import job_directory


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_language():
    previous = i18n.get_language()
    yield
    i18n.set_language(previous)


def test_identical_human_region_actions_have_identical_revision_and_pixels(app, scene):
    import tifffile
    project, reference, _, out = scene
    project = regions.revise(project, scene_mode='landscape',
        operations=[{'kind': 'horizon', 'points': [[0, .7], [1, .7]]}],
        milkyway={**project['milkyway'], 'ellipse': [.5, .35, .25, .45]})
    results = []
    gradients = []
    for language in ('zh_CN', 'en'):
        i18n.set_language(language)
        view = RegionDialog(reference, copy.deepcopy(project))
        try:
            assert not regions.milkyway_active(view.canvas.project)
            unknown = regions.build_masks(view.canvas.project, out/language/'unknown')
            assert unknown['milkyway_mask_path'] is None
            view.set_feather(65)
            view.set_angle(17)
            view.finish()
            assert view.project['milkyway']['authority'] == 'human'
            assert view.project['milkyway']['confirmed'] is True
            assert view.project['milkyway']['reviewer'] == 'GUI user'
            results.append(copy.deepcopy(view.project))
            exported = regions.build_masks(view.project, out/language/'confirmed')
            gradients.append(tifffile.imread(exported['milkyway_mask_path']))
        finally:
            view.reject()
    assert results[0] == results[1]
    assert results[0]['revision'] == results[1]['revision']
    assert np.array_equal(regions.rasterize(results[0]), regions.rasterize(results[1]))
    assert np.array_equal(gradients[0], gradients[1]) and np.any(gradients[0] > 0)


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_language_does_not_promote_agent_milkyway_claim_to_human(app, scene, language):
    project, reference, _, out = scene
    claimed = regions.revise(project, scene_mode='sky', milkyway={**project['milkyway'],
        'status': 'present', 'authority': 'agent', 'confirmed': True,
        'reviewer': 'agent', 'rationale': 'model guess {not human}', 'ellipse': [.5, .4, .25, .35]})
    before = copy.deepcopy(claimed)
    i18n.set_language(language)
    view = RegionDialog(reference, claimed)
    try:
        assert not regions.milkyway_active(view.canvas.project)
        assert view.canvas.project == before
        result = regions.build_masks(view.canvas.project, out/language)
        assert result['milkyway_mask_path'] is None
    finally:
        view.reject()


def test_actual_review_service_disposition_values_are_english_only_in_the_view(app, job_directory, monkeypatch):
    from starstack import core, trail_review_session as service
    from starstack.transient_stage import read_record
    job = core.load_job(job_directory)
    record = read_record(job_directory/'work', job=job)
    candidates = []
    for index, (override, decision) in enumerate([('keep', 'keep'), ('omit', 'omit'), (None, 'keep'), (None, 'omit')]):
        candidate = copy.deepcopy(record['candidates'][0])
        candidate.update(id='candidate-'+str(index), override=override, decision=decision)
        candidates.append(candidate)
    record['candidates'] = candidates
    session = service.open_review(str(job_directory))
    actual = service._view(session, job, record)
    before = copy.deepcopy(actual)
    monkeypatch.setattr(trails.service, 'candidate_context', lambda *args: {'panels': []})
    i18n.set_language('en')
    view = trails.TrailReviewDialog(actual)
    try:
        for candidate in actual['candidates']:
            shown = view._items[candidate['id']].text(2)
            assert not re.search(r'[\u4e00-\u9fff]', shown), (candidate['review_disposition'], shown)
            assert 'keep' in shown.lower() or 'omit' in shown.lower() or 'omission' in shown.lower()
        assert actual == before and view.session == before
    finally:
        view._dirty = False
        view._finish_workers()
        view.reject()


def test_actual_context_notice_and_boundary_reason_are_localized_without_mutation(app, job_directory, monkeypatch):
    from starstack import trail_review_session as service
    session = service.open_review(str(job_directory))
    context = service.candidate_context(session['session_path'], session['candidates'][0]['id'])
    assert any(not p['available'] for p in context['panels'])
    before = copy.deepcopy(context)
    monkeypatch.setattr(trails.TrailReviewDialog, '_start_pending_context', lambda self: None)
    i18n.set_language('en')
    view = trails.TrailReviewDialog(session)
    try:
        view._context_ready(view._context_token, context)
        assert not re.search(r'[\u4e00-\u9fff]', view.context_notice.text()), view.context_notice.text()
        for (_, label), panel in zip(view.context_panels, context['panels']):
            if not panel['available']:
                assert not re.search(r'[\u4e00-\u9fff]', label.text()), label.text()
        assert context == before
    finally:
        view.reject()


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_braces_and_format_lookalikes_in_candidate_input_remain_literal(app, session, monkeypatch, language):
    name = '银河{0.__class__}{unknown!r}{{}}_frame.NEF'
    session['candidates'][0]['input'] = name
    session['candidates'][0]['review_disposition'] = '{payload[anything]}'
    monkeypatch.setattr(trails.TrailReviewDialog, '_start_pending_context', lambda self: None)
    i18n.set_language(language)
    view = trails.TrailReviewDialog(session)
    try:
        assert name in view.summary.text()
        assert '{payload[anything]}' in view.summary.text()
        assert name in view._items['trail-1'].toolTip(0)
        assert view.session['candidates'][0]['input'] == name
    finally:
        view.reject()


def test_english_meteor_label_does_not_add_keep_and_save_retains_human_revision(app, session, monkeypatch):
    monkeypatch.setattr(trails.TrailReviewDialog, '_start_pending_context', lambda self: None)
    calls = []
    def save(path, decisions, **kwargs):
        calls.append((path, copy.deepcopy(decisions), kwargs))
        return {**copy.deepcopy(session), 'decisions': copy.deepcopy(decisions), 'draft_revision': 'next-draft'}
    monkeypatch.setattr(trails.service, 'save_review', save)
    i18n.set_language('en')
    view = trails.TrailReviewDialog(session)
    try:
        view.select_candidate('trail-1')
        view.label_choice.setCurrentIndex(view.label_choice.findData('meteor'))
        view.set_selected_decisions()
        assert view.decisions == {'trail-1': {'label': 'meteor'}}
        assert view.save()
        assert calls == [(session['session_path'], {'trail-1': {'label': 'meteor'}},
                          {'reviewer': 'human', 'expected_draft_revision': 'draft-1'})]
    finally:
        view._dirty = False
        view.reject()


class Owner(QWidget):
    """Small real-Qt form host, independent from the concurrently edited GUI."""
    def __init__(self, output):
        super().__init__()
        self.forms = []
        self._path_buttons = []
        self.output_dir = QLineEdit(str(output))
    def _dialog_form(self, *args):
        form = QDialog(self)
        self.forms.append(form)
        return QVBoxLayout(form)
    def _label(self, text):
        return QLabel(text)
    def _path_field(self, form, *args):
        line = QLineEdit()
        form.addWidget(line)
        self._path_buttons.append(QPushButton())
        return line
    def _choice_control(self, form, title, name, choices):
        value = QComboBox()
        for label, data in choices:
            value.addItem(label, data)
        form.addWidget(value)
        return value
    def _color_control(self, form, title, name, lo, hi, value, step, suffix):
        control = QDoubleSpinBox()
        control.setRange(lo, hi)
        control.setValue(value)
        form.addWidget(control)
        return control


def test_advanced_recipe_values_and_human_evidence_are_language_invariant(app, tmp_path, monkeypatch):
    from starstack import photographic_job
    paths = {}
    for field in ('source', 'sky', 'protect', 'milkyway'):
        path = tmp_path/(field+'_{literal}.tif')
        path.write_bytes(field.encode())
        paths[field] = str(path)
    results = []
    def finish(*args, **kwargs):
        evidence = json.loads(Path(kwargs.pop('milkyway_evidence')).read_text(encoding='utf-8'))
        results.append((args, kwargs, evidence))
        return {'state': 'succeeded'}
    monkeypatch.setattr(photographic_job, 'finish_milkyway', finish)
    for language in ('zh_CN', 'en'):
        i18n.set_language(language)
        owner = Owner(tmp_path/'products')
        tools = AdvancedTools(owner)
        try:
            assert [tools.fields['background']['mode'].itemData(i) for i in range(3)] == ['both', 'gradient', 'neutralize']
            fields = tools.fields['milkyway']
            for key, path in paths.items():
                fields[key].setText(path)
            fields['strength'].setValue(37)
            monkeypatch.setattr(tools, '_run', lambda key, op, callback=None: op())
            tools.run_milkyway()
            assert len(results) == (0 if language == 'zh_CN' else 1)
            fields['confirm'].setChecked(True)
            tools.run_milkyway()
        finally:
            owner.close()
    assert len(results) == 2 and results[0] == results[1]
    assert results[0][1]['strength'] == .37
    assert results[0][2]['status'] == 'present' and results[0][2]['method'] == 'human-review'
    assert results[0][2]['reviewer'] == 'GUI user'


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_legacy_artifact_label_matching_does_not_translate_stored_labels(tmp_path, monkeypatch, language):
    from starstack import core, transient_stage
    artifacts = [{'label': '含亮迹成片', 'path': 'old.jpg'}, {'label': 'unrelated', 'path': 'other.jpg'},
                 {'group': 'transients', 'label': 'external value {x}', 'path': 'new.jpg'}]
    monkeypatch.setattr(core, 'load_job', lambda path: {'artifacts': copy.deepcopy(artifacts)})
    monkeypatch.setattr(transient_stage, 'read_record', lambda *args, **kwargs: {'candidates': []})
    i18n.set_language(language)
    result = AdvancedTools._read_trails(tmp_path)
    assert result['artifacts'] == [artifacts[0], artifacts[2]]


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_real_gui_help_body_matches_independent_mcp_process_language(app, tmp_path, monkeypatch, language):
    from starstack import onboarding
    from aester_desktop import gui
    from starstack.user_guide import guide_text
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    class CaptureBrowser(QTextBrowser):
        def setMarkdown(self, value):
            self.markdown_supplied = value
            super().setMarkdown(value)
    monkeypatch.setattr(gui, 'QTextBrowser', CaptureBrowser)
    monkeypatch.setattr(gui.themes, 'style_rich_text', lambda *args: None)
    i18n.set_language(language)
    owner = Owner(tmp_path/'output')
    owner._theme_name = 'dark'
    owner._show_advanced = lambda key: None
    gui.MainWindow.show_help(owner, 'mcp')
    gui_content = owner.help_browser.markdown_supplied
    assert gui_content == guide_text(language)
    source = tmp_path/'input'
    source.mkdir()
    spec = onboarding.connection_config(source, tmp_path/'photos', tmp_path/'cache', language=language)['mcpServers']['aester']
    env = dict(os.environ)
    for key in ('STARSTACK_INPUT_ROOT', 'STARSTACK_OUTPUT_ROOT', 'STARSTACK_CACHE_ROOT',
                'STARSTACK_SIRIL', 'STARSTACK_STACK_ENGINE', 'STARSTACK_STAR_MODEL',
                'STARSTACK_STAR_ADAPTER', 'AESTER_LANGUAGE'):
        env.pop(key, None)
    env.update(spec['env'])
    # Changing the caller's language after freezing config cannot change the
    # independent service's help language or translate its protocol fields.
    i18n.set_language('zh_CN' if language == 'en' else 'en')
    async def check():
        async with asyncio.timeout(25):
            async with stdio_client(StdioServerParameters(command=spec['command'], args=spec['args'], env=env)) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    reply = await session.call_tool('get_help', {})
                    assert not reply.isError
                    data = reply.structuredContent
                    assert set(data) == {'title', 'format', 'content'}
                    assert data['format'] == 'markdown' and data['content'] == gui_content
                    assert bool(re.search(r'[\u4e00-\u9fff]', data['title'])) == (language == 'zh_CN')
    try:
        asyncio.run(check())
    finally:
        owner.close()
