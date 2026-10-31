"""Installer integration with synthetic payloads; no real setup is launched."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from starstack import components
from test_finite_installer_distribution import builder_case, worker_case

ROOT = Path(__file__).resolve().parents[1]
REAL_RUN = subprocess.run


def test_selected_retouch_components_prepare_before_activation(worker_case, monkeypatch):
    c = worker_case
    result = {'ready': True, 'gimp': {'selected_path': 'external GIMP'}, 'skill': {'path': 'local skill'}}
    def prepare(root, **kwargs):
        c.calls.append(('retouch', root, kwargs))
        return result
    def verify(root, **kwargs):
        c.calls.append(('verify-retouch', root, kwargs))
        assert kwargs['expected'] == result
        return result
    monkeypatch.setattr(components, 'install_retouch_components', prepare)
    monkeypatch.setattr(components, 'verify_prepared_retouch_components', verify)
    before = c.state.read_bytes(), c.pointer.read_bytes()
    assert c.run('--preflight') == 0
    assert c.run('--gimp', '--retouch-skill') == 0
    assert (c.state.read_bytes(), c.pointer.read_bytes()) == before
    prepared = json.loads(c.prepared.read_text())
    assert prepared['retouch_components'] == result
    args = next(item[2] for item in c.calls if item[0] == 'retouch')
    assert args['include_gimp'] is True and args['include_skill'] is True
    assert args['app_root'] == c.app
    c.calls.clear()
    assert c.run('--activate', str(c.state)) == 0
    assert [item[0] for item in c.calls] == ['verify-retouch', 'record', 'import']


def test_retouch_failure_is_reported_and_never_fails_aester(worker_case, monkeypatch):
    # 2026-09-27 user decision: optional GIMP/skill problems are shown, not fatal.
    c = worker_case
    def failed(*args, **kwargs):
        raise ValueError('upstream cancelled or skill changed')
    monkeypatch.setattr(components, 'install_retouch_components', failed)
    before = c.state.read_bytes(), c.pointer.read_bytes()
    assert c.run('--preflight') == 0
    assert c.run('--gimp') == 0
    prepared = json.loads(c.prepared.read_text())
    assert 'retouch_components' not in prepared
    assert [n['code'] for n in prepared['notices']] == ['retouch_prepare_failed']
    assert 'upstream cancelled' in c.prepared.with_suffix('.notices.txt').read_text(encoding='utf-8')
    assert (c.state.read_bytes(), c.pointer.read_bytes()) == before
    assert c.run('--rollback') == 0


def test_basic_installer_never_calls_retouch(worker_case, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Basic preparation entered optional GIMP/skill code')
    monkeypatch.setattr(components, 'install_retouch_components', unexpected)
    monkeypatch.setattr(components, 'verify_prepared_retouch_components', unexpected)
    assert worker_case.run() == 0
    assert 'retouch_components' not in json.loads(worker_case.prepared.read_text())


def test_skill_checkbox_does_not_select_gimp(worker_case, monkeypatch):
    calls = []
    def prepare(root, **kwargs):
        calls.append(kwargs)
        return {'ready': True, 'gimp': None, 'skill': {'status': 'ready'}}
    monkeypatch.setattr(components, 'install_retouch_components', prepare)
    monkeypatch.setattr(components, 'verify_prepared_retouch_components', lambda *args, **kwargs: kwargs['expected'])
    assert worker_case.run('--retouch-skill') == 0
    assert calls[0]['include_gimp'] is False and calls[0]['include_skill'] is True


def test_retouch_build_contains_exact_skill_zip_and_stdlib_bootstrap(builder_case):
    c = builder_case
    source = c.builder.ROOT
    shutil.copytree(ROOT / 'skills/aester-retouch', source / 'skills/aester-retouch')
    for name in ('retouch_distribution.py', 'components.py', 'install_ownership.py', 'install_lifecycle.py',
                 'finishing_components.py', 'finishing_assets.py', 'process_guard.py'):
        shutil.copyfile(ROOT / 'src/starstack' / name, source / 'src/starstack' / name)
    receipt = c.builder.build(c.output, c.compiler, c.model, include_retouch=True)
    assert receipt['installer_components'] == ['base', 'gimp', 'retouch_skill']
    assert receipt['nox_model_bundled'] is False
    assert '/DIncludeRetouch=1' in c.compiled[0]
    stage = Path(receipt['stage'])
    app = stage / 'application'
    contents = json.loads((app / 'PACKAGE_CONTENTS.json').read_text())
    names = [name for name in contents['sha256'] if name.startswith('skills/')]
    assert len(names) == 8 and all(name.endswith(('.md', '.zip')) for name in names)
    assert (app / 'skills/aester-retouch.zip').exists()
    assert not list(stage.rglob('gimp-*-setup.exe'))  # upstream download, no binary redistribution
    boot = stage / 'bootstrap'
    command = [sys.executable, '-I', '-c',
        'import sys; sys.path.insert(0, sys.argv[1]); import components, retouch_distribution; '
        'import starstack.finishing_components; '
        'print(len(retouch_distribution.skill_files(sys.argv[1]+"/retouch-skill/aester-retouch")))', str(boot)]
    checked = REAL_RUN(command, capture_output=True, text=True, check=True)
    assert checked.stdout.strip() == '7'
    for language_file in ('INSTALL-NOTES.txt', 'INSTALL-NOTES-en.txt'):
        notes = (stage / language_file).read_text(encoding='utf-8-sig')
        assert 'GIMP' in notes
        assert '全局技能目录' in notes if language_file == 'INSTALL-NOTES.txt' else 'global skill directories' in notes


def test_retouch_build_refuses_extra_unlicensed_learning_asset(builder_case):
    c = builder_case
    source = c.builder.ROOT
    shutil.copytree(ROOT / 'skills/aester-retouch', source / 'skills/aester-retouch')
    (source / 'skills/aester-retouch/learning-photo.tif').write_bytes(b'not allowed')
    with pytest.raises(ValueError, match='exactly the reviewed'):
        c.builder.build(c.output, c.compiler, c.model, include_retouch=True)
    assert c.compiled == []


def test_inno_optional_component_labels_and_real_worker_arguments():
    script = (ROOT / 'packaging/aester_p2/Aester.iss').read_text()
    messages = (ROOT / 'packaging/aester_p2/installer-messages.iss').read_text()
    assert "WizardIsComponentSelected('gimp') then Args := Args + ' --gimp'" in script
    assert "WizardIsComponentSelected('retouch_skill') then Args := Args + ' --retouch-skill'" in script
    assert '自然语言修图所需' in messages and '卸载Aester时保留' in messages
