"""GUI integration with read-only status and stubbed component installation."""
import re
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QAbstractButton, QLineEdit

from test_gui_language_routing import app, make_window
from starstack import i18n


def retouch_info(*, available=False, skill=False, path=None):
    return {'schema': 1, 'trial': True,
            'gimp': {'status': 'available' if available else 'selected_missing',
                     'available': available, 'path_exists': available, 'selected_path': path,
                     'version_checked': False, 'compatible': None},
            'skill': {'status': 'available' if skill else 'missing', 'available': skill,
                      'installed': None, 'required': False,
                      'read_path': 'C:/test/skills/aester-retouch/SKILL.md' if skill else None}}


def patch_status(monkeypatch, value=None):
    from starstack import retouch_integration
    monkeypatch.setattr(retouch_integration, 'get_retouch_info', lambda **kwargs: value or retouch_info())
    return retouch_integration


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_prompt_opens_without_photos_gimp_or_skill_and_copies(make_window, app, monkeypatch, language):
    # Other GUI agents can share the Windows clipboard during parallel checks.
    # Verify this window's clipboard call without racing another process.
    class Clipboard:
        value = ''
        def setText(self, text):
            self.value = text
        def text(self):
            return self.value
    clipboard = Clipboard()
    monkeypatch.setattr(QApplication, 'clipboard', lambda: clipboard)
    shared = patch_status(monkeypatch)
    calls = []
    monkeypatch.setattr(shared, 'build_retouch_prompt', lambda **kwargs: calls.append(kwargs) or 'Prompt from shared API')
    window, _ = make_window(language)
    assert window.retouch_button.isEnabled()
    window.retouch_button.click()
    app.processEvents()
    exp = window.experience
    assert window._advanced_dialogs['retouch'].isVisible()
    assert exp.retouch_photo.text() == ''
    assert exp.retouch_copy.isEnabled()
    assert calls[-1]['photo_path'] is None and calls[-1]['language'] == language
    exp.retouch_copy.click()
    assert QApplication.clipboard().text() == 'Prompt from shared API'
    assert window.workflow_dir is None and window.job_dir is None
    assert not window._busy and not window._workers


def test_current_delivery_tiff_and_literal_intent_go_through_shared_prompt(make_window, app, monkeypatch, tmp_path):
    shared = patch_status(monkeypatch)
    calls = []
    monkeypatch.setattr(shared, 'build_retouch_prompt', lambda **kwargs: calls.append(kwargs) or 'Shared prompt')
    window, _ = make_window()
    master, photo = tmp_path / 'master_scene_linear.tif', tmp_path / 'final_srgb16.tif'
    master.write_bytes(b'master identity only')
    photo.write_bytes(b'photo identity only')
    window._workflow = {'state': 'succeeded', 'artifacts': [
        {'path': str(master), 'kind': 'tiff'}, {'path': str(photo), 'kind': 'tiff'}]}
    window.retouch_button.click()
    window.experience.retouch_intent.setPlainText('保留银河 {0} 和地景，不要新增星点')
    app.processEvents()
    window.experience.retouch_copy.click()
    assert calls[-1]['photo_path'] == str(photo)
    assert calls[-1]['intent'] == '保留银河 {0} 和地景，不要新增星点'
    assert master.read_bytes() == b'master identity only'
    assert photo.read_bytes() == b'photo identity only'


def test_never_handoff_master_mask_or_missing_photo_as_finished_tiff(make_window, tmp_path):
    window, _ = make_window()
    master, mask = tmp_path / 'master_scene_linear.tif', tmp_path / 'sky_mask.tif'
    master.touch(); mask.touch()
    window._workflow = {'artifacts': [{'path': str(p), 'kind': 'tiff'} for p in
                                     (master, mask, tmp_path / 'final_srgb16.tif')]}
    assert window.retouch_photo_path() is None


def test_selected_preview_uses_its_recorded_display_tiff(make_window, tmp_path):
    window, _ = make_window()
    old, current = tmp_path / 'final_srgb16.tif', tmp_path / 'natural.tif'
    old.touch(); current.touch()
    window.job_dir = str(tmp_path / 'job')
    window._workflow = {'artifacts': [{'path': str(old), 'kind': 'tiff'}]}
    window._rendered_preview = {'job_dir': window.job_dir,
                                'artifacts': [{'path': str(current), 'kind': 'tiff'}]}
    window._jpg_artifacts = [{'path': str(tmp_path / 'natural.jpg')}]
    window.style_tabs.blockSignals(True)
    window.style_tabs.addTab('natural')
    window.style_tabs.blockSignals(False)
    assert window.retouch_photo_path() == str(current)


def open_components(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, '_run_async', lambda *args: calls.append(args))
    window.experience.show_components()
    calls.pop()[2]({'ready': False, 'components': [], 'retouch_info': retouch_info()})
    return calls


class FakeClipboard:
    """This window's clipboard calls, without racing other processes on the Windows clipboard."""
    value = ''

    def setText(self, text):
        self.value = text

    def text(self):
        return self.value


def fake_clipboard(monkeypatch):
    clipboard = FakeClipboard()
    monkeypatch.setattr(QApplication, 'clipboard', lambda: clipboard)
    return clipboard


def test_skill_only_install_is_separate_from_base_and_gimp(make_window, monkeypatch):
    # 2026-09-27: the highlighted button prepares the ticked components and then writes and
    # copies the assistant prompt, so the prompt API and the clipboard are stubbed too.
    from starstack import components
    shared = patch_status(monkeypatch)
    monkeypatch.setattr(shared, 'build_retouch_prompt', lambda **kwargs: 'Prompt after preparing')
    clipboard = fake_clipboard(monkeypatch)
    window, _ = make_window()
    queue = open_components(window, monkeypatch)
    exp = window.experience
    installs = []
    monkeypatch.setattr(components, 'install_retouch_components', lambda **kwargs: installs.append(kwargs) or {'ready': True})
    monkeypatch.setattr(components, 'install_components', lambda **kwargs: pytest.fail('Base install was not selected'))
    assert not exp.gimp_support.isChecked() and not exp.skill_support.isChecked()
    exp.skill_support.setChecked(True)
    exp.retouch_install.click()
    assert len(queue) == 1 and queue[0][0] == 'retouch-component-install'
    assert window._busy and not exp.retouch_install.isEnabled()
    assert clipboard.value == ''                     # nothing is copied before preparation ends
    result = queue[0][1]()
    assert installs[0]['include_skill'] and not installs[0]['include_gimp']
    assert 'gimp_source' not in installs[0]
    queue.pop()[2](result)
    assert not window._busy and exp.retouch_install.isEnabled()
    assert queue[-1][0] == 'component-status'
    assert clipboard.value == 'Prompt after preparing'
    assert window._advanced_dialogs['retouch'].isVisible()
    assert exp.component_message.text().startswith('已按所选组件生成提示词并复制。')
    queue.pop()[2]({'ready': False, 'components': [], 'retouch_info': retouch_info(skill=True)})


def test_retouch_install_failure_releases_controls_and_reports_reason(make_window, monkeypatch):
    from starstack import components
    clipboard = fake_clipboard(monkeypatch)
    window, _ = make_window()
    queue = open_components(window, monkeypatch)
    exp = window.experience
    def fail(**kwargs):
        raise ValueError('download verification failed')
    monkeypatch.setattr(components, 'install_retouch_components', fail)
    exp.gimp_support.setChecked(True)
    exp.retouch_install.click()
    action = queue.pop()
    action[2](action[1]())
    assert not window._busy and exp.retouch_install.isEnabled()
    assert 'download verification failed' in exp.component_message.text()
    # A failed preparation never produces or copies a prompt.
    assert clipboard.value == '' and 'retouch' not in window._advanced_dialogs
    queue.pop()[2]({'ready': False, 'components': [], 'retouch_info': retouch_info()})


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_components_and_prompts_entry_and_highlighted_generate_button(make_window, app, monkeypatch, language):
    """Item 15: 组件管理 → 组件与提示词管理; one highlighted button that ends with a copied prompt."""
    from aester_desktop import themes
    shared = patch_status(monkeypatch)
    monkeypatch.setattr(shared, 'build_retouch_prompt', lambda **kwargs: 'Prompt for ' + kwargs['language'])
    clipboard = fake_clipboard(monkeypatch)
    window, _ = make_window(language)
    en = language == 'en'
    # Menu and button text escape the literal '&' (a single one would become a mnemonic).
    assert window.menu_actions['components'].text() == ('Components && prompts…' if en else '组件与提示词管理…')
    queue = open_components(window, monkeypatch)
    exp = window.experience
    dialog = window._advanced_dialogs['components']
    assert dialog.windowTitle() == ('Components & prompts' if en else '组件与提示词管理')
    button = dialog.findChild(QAbstractButton, 'generateAssistantPrompt')
    assert button is exp.retouch_install and themes.role(button) == 'primary' and button.minimumHeight() >= 44
    assert button.text() == ('Generate the assistant prompt for the selected components' if en
                             else '按所选组件，生成给助手的提示词')
    plan = dialog.findChild(QLabel, 'generatePromptPlan')
    assert plan.text() == ('Nothing ticked: writes and copies the prompt right away.' if en else '未勾选组件：直接生成提示词并复制。')
    exp.gimp_support.setChecked(True)
    exp.skill_support.setChecked(True)
    assert ('GIMP (download the official installer)' if en else 'GIMP（下载官方安装程序，来自官方网站或所列镜像）') in plan.text()
    assert ('retouch skill package' if en else '修图技能包') in plan.text()
    exp.gimp_support.setChecked(False)
    exp.skill_support.setChecked(False)
    # Nothing ticked: no installation, the prompt is written and copied right away.
    from starstack import components
    monkeypatch.setattr(components, 'install_retouch_components', lambda **kwargs: pytest.fail('nothing was ticked'))
    button.click()
    app.processEvents()
    assert not queue and clipboard.value == 'Prompt for ' + language
    assert window._advanced_dialogs['retouch'].isVisible()
    assert exp.component_message.text() == ('Prompt generated for the selected components and copied. Paste it into your Agent to start retouching.'
                                            if en else '已按所选组件生成提示词并复制。粘贴给 Agent 即可开始修图。')
    opener = window._advanced_dialogs['retouch'].findChild(QAbstractButton, 'openComponents')
    assert opener.text() == ('Components && prompts…' if en else '组件与提示词管理…')


def _delivery(folder, name='final_srgb16.tif'):
    folder.mkdir(parents=True, exist_ok=True)
    for item in ('final_srgb16.tif', 'final.jpg', 'master_scene_linear.tif'):
        (folder / item).write_bytes(b'synthetic')
    return [{'path': str(folder / item), 'name': item, 'kind': kind} for item, kind in
            (('master_scene_linear.tif', 'tiff'), ('final_srgb16.tif', 'tiff'), ('final.jpg', 'jpeg'))]


def _trail_records(tmp_path):
    base, trails = tmp_path / 'photos' / 'delivery-abc', tmp_path / 'photos' / 'delivery-trails-2kept-123'
    base_record = {'state': 'succeeded', 'artifacts': _delivery(base), 'delivery_dir': str(base),
                   'request': {'trails_enabled': True}}
    trails_record = dict(base_record, artifacts=_delivery(trails), delivery_dir=str(trails), base_delivery_dir=str(base),
                         trail_review={'automatic_addback': True, 'kept_count': 2})
    return base, trails, base_record, trails_record


def test_agent_retouch_uses_the_with_trails_tiff_even_when_published_elsewhere(make_window, app, monkeypatch, tmp_path):
    """Item 17: after trails were added back, 让 Agent 帮你修 picks that photograph and says so."""
    patch_status(monkeypatch)
    window, _ = make_window()
    base, trails, base_record, trails_record = _trail_records(tmp_path)
    window.workflow_dir = str(tmp_path / 'workflow')
    window.job_state = 'succeeded'
    window._view_mode = 'results'
    window._apply_workflow_record(base_record)
    window.retouch_button.click()
    app.processEvents()
    exp = window.experience
    assert exp.retouch_photo.text() == str(base / 'final_srgb16.tif')
    assert exp.retouch_photo_hint.text() == '当前成片：基础成片（未加回亮迹）'
    window._advanced_dialogs['retouch'].close()
    # The review was applied elsewhere (the MCP): the GUI still holds the base record.
    monkeypatch.setattr('starstack.workflow.workflow_status', lambda *_: trails_record)
    window.retouch_button.click()
    app.processEvents()
    assert window._workflow['delivery_dir'] == str(trails)
    assert exp.retouch_photo.text() == str(trails / 'final_srgb16.tif')
    assert exp.retouch_photo_hint.isVisible() and exp.retouch_photo_hint.text() == '当前成片：已加回亮迹的版本'
    window.job_state = ''
    window.workflow_dir = None


def test_a_chosen_tiff_survives_a_new_automatic_photograph(make_window, app, monkeypatch, tmp_path):
    patch_status(monkeypatch)
    window, _ = make_window()
    base, trails, base_record, trails_record = _trail_records(tmp_path)
    window._workflow = base_record
    window._jpg_artifacts = [a for a in base_record['artifacts'] if a['kind'] == 'jpeg']
    window.retouch_button.click()
    app.processEvents()
    exp = window.experience
    manual = tmp_path / 'mine.tif'
    manual.write_bytes(b'synthetic')
    exp.retouch_photo.setText(str(manual))
    window._workflow = trails_record
    window._jpg_artifacts = [a for a in trails_record['artifacts'] if a['kind'] == 'jpeg']
    window.retouch_button.click()
    app.processEvents()
    assert exp.retouch_photo.text() == str(manual)
    exp.retouch_photo.clear()                          # an empty field is filled again
    window.retouch_button.click()
    app.processEvents()
    assert exp.retouch_photo.text() == str(trails / 'final_srgb16.tif')
    opened = []
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: opened.append(args) or ('', ''))
    exp.retouch_photo.clear()
    window._workflow = None
    window._jpg_artifacts = []
    exp._pick_retouch_photo()
    assert opened[-1][2] == window.output_dir.text()   # starts in the photographs folder


def test_reference_image_goes_into_the_prompt_with_its_kind(make_window, app, monkeypatch, tmp_path):
    """Item 16: an optional aesthetic reference, its provenance kind, and a scope warning."""
    shared = patch_status(monkeypatch)
    calls = []
    monkeypatch.setattr(shared, 'build_retouch_prompt', lambda **kwargs: calls.append(kwargs) or 'Prompt')
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    settings = QSettings(str(tmp_path / 'scope.ini'), QSettings.Format.IniFormat)
    settings.setValue('appearance/language', 'zh_CN')
    settings.setValue('mcp/input_root', str(inputs))
    window, _ = make_window(settings=settings)
    window.retouch_button.click()
    app.processEvents()
    exp = window.experience
    assert calls[-1]['reference_images'] is None
    assert exp.retouch_reference.isReadOnly() and exp.retouch_reference.objectName() == 'retouchReferencePath'
    assert [exp.retouch_reference_kind.itemData(i) for i in range(exp.retouch_reference_kind.count())] == \
        ['external', 'original', 'generated']
    assert '星点、云、山脊线或纹理' in exp.retouch_reference_tip.text()
    outside = tmp_path / 'elsewhere' / 'look.jpg'
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(outside), ''))
    window._advanced_dialogs['retouch'].findChild(QAbstractButton, 'pickRetouchReference').click()
    assert exp.retouch_reference.text() == str(outside)
    assert calls[-1]['reference_images'] == [{'path': str(outside), 'kind': 'external'}]
    assert exp.retouch_reference_warning.isVisible() and '素材范围' in exp.retouch_reference_warning.text()
    exp.retouch_reference_kind.setCurrentIndex(2)
    assert calls[-1]['reference_images'] == [{'path': str(outside), 'kind': 'generated'}]
    inside = inputs / 'mine.png'
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(inside), ''))
    exp._pick_retouch_reference()
    assert not exp.retouch_reference_warning.isVisible()
    in_works = Path(window.output_dir.text()) / 'old' / 'work.tif'
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(in_works), ''))
    exp._pick_retouch_reference()
    assert not exp.retouch_reference_warning.isVisible()
    window._advanced_dialogs['retouch'].findChild(QAbstractButton, 'clearRetouchReference').click()
    assert exp.retouch_reference.text() == '' and calls[-1]['reference_images'] is None
    assert not exp.retouch_reference_warning.isVisible()


def test_existing_gimp_selection_is_shared_and_invalid_choice_keeps_preference(make_window, monkeypatch):
    shared = patch_status(monkeypatch, retouch_info(available=True, path='C:/chosen/gimp-console-3.2.exe'))
    window, prefs = make_window()
    open_components(window, monkeypatch)
    exp = window.experience
    selected = 'C:/chosen/gimp-console-3.2.exe'
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (selected, ''))
    exp._pick_gimp()
    assert exp._gimp_path == selected and prefs.value('retouch/gimp_path') == selected
    monkeypatch.setattr(shared, 'get_retouch_info', lambda **kwargs: retouch_info(path='C:/bad/gimp.exe'))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: ('C:/bad/gimp.exe', ''))
    exp._pick_gimp()
    assert exp._gimp_path == selected and prefs.value('retouch/gimp_path') == selected


def test_reset_gimp_is_explicit_and_forwarded_to_connection(make_window, monkeypatch, tmp_path):
    from starstack import onboarding
    window, prefs = make_window()
    queue = open_components(window, monkeypatch)
    exp = window.experience
    selected = 'C:/chosen/gimp-console-3.2.exe'
    exp._gimp_path = selected
    prefs.setValue('retouch/gimp_path', selected)
    captured = []
    monkeypatch.setattr(onboarding, 'connection_config', lambda *args, **kwargs: captured.append(kwargs) or {'mcpServers': {}})
    exp.show_connection()
    assert captured[-1]['gimp_path'] == selected
    exp._use_auto_gimp()
    assert exp._gimp_path == '' and prefs.value('retouch/gimp_path') is None
    queue.pop()[2]({'ready': False, 'components': [], 'retouch_info': retouch_info()})
    exp._refresh_connection()
    assert captured[-1]['gimp_path'] is None


@pytest.mark.parametrize('language', ['zh_CN', 'en'])
def test_help_entry_and_component_options_are_localized(make_window, app, monkeypatch, language):
    patch_status(monkeypatch)
    from starstack import retouch_integration
    monkeypatch.setattr(retouch_integration, 'build_retouch_prompt', lambda **kwargs: 'Prompt')
    window, _ = make_window(language)
    window.menu_actions['help_retouch'].trigger()
    app.processEvents()
    assert window._advanced_dialogs['help'].isVisible()
    assert ('Natural-language retouching' if language == 'en' else '自然语言修图') in window.help_browser.toPlainText()
    assert window.help_browser.verticalScrollBar().value() > 0
    window.retouch_button.click()
    open_components(window, monkeypatch)
    app.processEvents()
    if language == 'en':
        for key in ('retouch', 'components'):
            dialog = window._advanced_dialogs[key]
            for control in dialog.findChildren(QLabel) + dialog.findChildren(QAbstractButton) + dialog.findChildren(QLineEdit):
                assert not re.search('[一-龥]', control.text()), (key, control.text())
