"""Optional component lifecycle checks; never run the real GIMP installer."""
import copy
import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from starstack import components as c, install_ownership as ownership
from starstack import retouch_distribution as d

ROOT = Path(__file__).resolve().parents[1]
REAL_DOWNLOAD = c._download


@pytest.fixture
def case(tmp_path):
    app = tmp_path / '独立 安装' / 'versions' / 'app-v1'
    app.mkdir(parents=True)
    area = tmp_path / 'components'
    area.mkdir()
    ownership.register_install(app.parent.parent, area, area / 'downloads')
    shutil.copytree(ROOT / 'skills/aester-retouch', app / 'skills/aester-retouch')
    (area / 'current.json').write_text('{"version":"base-must-remain"}')
    return app, area


def found(path, *, status='available', version='3.2.6'):
    return {'status': status, 'selected_path': str(path), 'version': version,
            'version_compatible': version == '3.2.6', 'managed_by_aester': False,
            'selection_locked': False, 'ownership': 'external_unmanaged'}


def test_source_manifest_matches_fixed_upstream_identity():
    manifest = json.loads((ROOT / 'packaging/aester_p2/retouch-components.json').read_text())
    for key, value in d.GIMP_INSTALLER.items():
        assert manifest['gimp'][key] == value
    assert manifest['gimp']['main_source']['sha256'] == d.GIMP_SOURCE['sha256']
    assert len(d.skill_files(ROOT / 'skills/aester-retouch')) == 7


def test_skill_only_never_discovers_runs_or_downloads_gimp(case, monkeypatch):
    app, area = case
    def unexpected(*args, **kwargs):
        pytest.fail('Skill preparation must not inspect or start GIMP')
    monkeypatch.setattr(d, '_discover', unexpected)
    monkeypatch.setattr(c, '_download', unexpected)
    before = (area / 'current.json').read_bytes()
    result = c.install_retouch_components(area, app_root=app, include_gimp=False)
    assert result['ready'] is True and result['gimp'] is None
    assert result['skill']['global_install_performed'] is False
    assert result['global_skill_install_performed'] is False
    assert (area / 'current.json').read_bytes() == before
    assert c.verify_prepared_retouch_components(area, app_root=app, expected=result) == result


def test_skill_zip_is_deterministic_exactly_text_and_reusable(case, tmp_path):
    app, area = case
    a = d.create_skill_archive(app / 'skills/aester-retouch', tmp_path / 'a.zip')
    b = d.create_skill_archive(app / 'skills/aester-retouch', tmp_path / 'b.zip')
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as archive:
        assert len(archive.namelist()) == 7
        assert all(name.endswith('.md') for name in archive.namelist())
    first = d.prepare_skill(area, app_root=app)
    again = d.prepare_skill(area, app_root=app)
    assert first == again


@pytest.mark.parametrize('mutation', ['extra-photo', 'changed-text', 'missing-reference'])
def test_skill_source_rejects_nonfrozen_or_extra_content_before_preparation(case, mutation):
    app, area = case
    source = app / 'skills/aester-retouch'
    if mutation == 'extra-photo':
        (source / 'photo.jpg').write_bytes(b'not redistributable')
    elif mutation == 'changed-text':
        (source / 'SKILL.md').write_text('unreviewed instructions')
    else:
        (source / 'references/goal-loop.md').unlink()
    with pytest.raises(ValueError):
        d.prepare_skill(area, app_root=app)
    assert not (area / 'versions').exists()


def test_skill_corruption_repairs_new_version_and_uninstall_keeps_unknown(case):
    app, area = case
    first = d.prepare_skill(area, app_root=app)
    folder = Path(first['receipt_path']).parent
    (folder / 'user-sentinel.txt').write_text('preserve')
    Path(first['path']).write_text('changed')
    second = d.prepare_skill(area, app_root=app)
    assert second['receipt_path'] != first['receipt_path']
    report = ownership.cleanup(app.parent.parent)
    assert report['state'] == 'complete'
    assert (folder / 'user-sentinel.txt').read_text() == 'preserve'
    assert not Path(second['path']).exists()
    assert (app / 'skills/aester-retouch/SKILL.md').exists()  # no app ownership recorded in fixture


def test_copied_skill_receipt_cannot_bind_another_app(case):
    app, area = case
    prepared = d.prepare_skill(area, app_root=app)
    other = app.with_name('app-v2')
    with pytest.raises(ValueError, match='another application'):
        d.verify_skill(Path(prepared['receipt_path']).parent, app_root=other)
    assert d.skill_status(area, app_root=other)['status'] == 'missing'


def test_skill_copy_failure_keeps_base_pointer_and_records_failure(case, monkeypatch):
    app, area = case
    before = (area / 'current.json').read_bytes()
    def failed(*args):
        raise OSError('copy interrupted')
    monkeypatch.setattr(d.shutil, 'copyfile', failed)
    with pytest.raises(OSError, match='interrupted'):
        d.prepare_skill(area, app_root=app)
    assert (area / 'current.json').read_bytes() == before
    assert len(list((area / 'staging').glob('*/failure.json'))) == 1
    assert not (area / 'versions').exists()


def test_existing_gimp_is_reused_without_download_install_or_ownership(case, monkeypatch, tmp_path):
    app, area = case
    executable = tmp_path / 'external GIMP/bin/gimp-console-3.2.exe'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'external untouched')
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: found(executable))
    def unexpected(*args, **kwargs):
        pytest.fail('An existing GIMP must never be downloaded or installed again')
    monkeypatch.setattr(c, '_download', unexpected)
    monkeypatch.setattr(d, '_run_upstream_installer', unexpected)
    result = c.install_retouch_components(area, app_root=app, include_skill=False)
    assert result['gimp']['installation_action'] == 'reused-existing'
    assert d.saved_gimp_path(area, app_root=app) == str(executable)
    assert d.saved_gimp_path(area, app_root=app.with_name('other-version')) is None
    ownership.cleanup(app.parent.parent)
    assert executable.read_bytes() == b'external untouched'
    assert not Path(result['receipt_path']).exists()


# 2026-09-27 user decision: an existing GIMP of any version is used as it is with
# a notice; nothing ever fails Aester, and nothing is downloaded or run over it.
@pytest.mark.parametrize('status,version,exists,action,code', [
    ('incompatible_version', '3.0.8', True, 'reused-existing', 'gimp_reused_other_version'),
    ('incompatible_version', '3.3.0', True, 'reused-existing', 'gimp_reused_other_version'),
    ('check_failed', None, True, 'reused-existing', 'gimp_reused_unverified'),
    ('selected_missing', None, False, 'not-installed', 'gimp_selected_missing'),
    ('check_failed', None, False, 'not-installed', 'gimp_check_incomplete')])
def test_existing_or_selected_bad_gimp_never_overwritten(case, monkeypatch, status, version, exists, action, code):
    app, area = case
    record = {**found('/bad/gimp.exe', status=status, version=version), 'path_exists': exists}
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: record)
    monkeypatch.setattr(d, '_run_upstream_installer', lambda *args, **kwargs: pytest.fail('must not install'))
    monkeypatch.setattr(c, '_download', lambda *args, **kwargs: pytest.fail('must not download'))
    result = d.install_gimp(area, app_root=app)
    assert result['installation_action'] == action
    assert [n['code'] for n in result['notices']] == [code]
    assert result['notices'][0]['tested_version'] == '3.2.6'
    if code == 'gimp_reused_other_version':
        assert result['notices'][0]['found_version'] == version
        text = d.describe_notices(result['notices'], 'zh_CN')[0]
        assert version in text and '3.2.6' in text and '兼容性不能保证' in text
        assert 'compatibility is not guaranteed' in d.describe_notices(result['notices'], 'en')[0]


def test_same_version_existing_gimp_is_reused_silently(case, monkeypatch, tmp_path):
    app, area = case
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: {**found(tmp_path / 'g.exe'), 'path_exists': True})
    monkeypatch.setattr(d, '_run_upstream_installer', lambda *args, **kwargs: pytest.fail('must not install'))
    result = d.install_gimp(area, app_root=app)
    assert result['installation_action'] == 'reused-existing' and result['notices'] == []


@pytest.fixture
def fresh_gimp(case, tmp_path, monkeypatch):
    app, area = case
    target = tmp_path / 'separate Programs/GIMP 3'
    installer = tmp_path / 'official-source.exe'
    installer.write_bytes(b'pinned official fixture, never execute')
    monkeypatch.setattr(d, 'GIMP_INSTALLER', {'filename': 'gimp-3.2.6-setup.exe', 'size': installer.stat().st_size,
        'sha256': c._digest(installer), 'url': d.GIMP_INSTALLER['url']})
    monkeypatch.setattr(d, '_registered_locations', lambda: [])
    monkeypatch.setattr(d, '_install_directory', lambda: target)
    def discover(path=None, *, probe=False):
        if target.exists():
            return found(target / 'bin/gimp-console-3.2.exe')
        return {'status': 'not_found', 'selection_locked': False}
    monkeypatch.setattr(d, '_discover', discover)
    monkeypatch.setattr(c, '_download', lambda item, cache, progress=None, sources=None: cache / item['filename'])
    calls = []
    def install(executable, destination, log, progress):
        calls.append((executable, destination, log))
        assert c._valid_file(executable, d.GIMP_INSTALLER)
        (destination / 'bin').mkdir(parents=True)
        (destination / 'bin/gimp-console-3.2.exe').write_bytes(b'official external fixture')
    monkeypatch.setattr(d, '_run_upstream_installer', install)
    return app, area, installer, target, calls


def test_fresh_install_download_owned_external_program_unowned(fresh_gimp):
    app, area, installer, target, calls = fresh_gimp
    result = c.install_retouch_components(area, app_root=app, include_skill=False, gimp_source=installer)
    assert len(calls) == 1
    assert result['gimp']['installation_action'] == 'installed-upstream'
    assert result['gimp']['managed_by_aester'] is False
    assert result['gimp']['retained_on_aester_uninstall'] is True
    ownership.cleanup(app.parent.parent)
    assert (target / 'bin/gimp-console-3.2.exe').exists()
    assert not (area / 'downloads/gimp-3.2.6-setup.exe').exists()
    assert installer.exists()


def test_untrusted_installer_rejected_before_execution(fresh_gimp):
    app, area, installer, target, calls = fresh_gimp
    installer.write_bytes(b'changed')
    with pytest.raises(ValueError, match='official pinned'):
        d.install_gimp(area, app_root=app, source=installer)
    assert calls == [] and not target.exists()


def test_broken_registry_or_occupied_target_never_overwritten(fresh_gimp, monkeypatch):
    app, area, installer, target, calls = fresh_gimp
    monkeypatch.setattr(d, '_registered_locations', lambda: [('some stale registry', 'registry', None)])
    result = d.install_gimp(area, app_root=app, source=installer)
    assert result['installation_action'] == 'not-installed'
    assert [n['code'] for n in result['notices']] == ['gimp_registered_missing']
    monkeypatch.setattr(d, '_registered_locations', lambda: [])
    target.mkdir(parents=True)
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: {'status': 'not_found', 'selection_locked': False})
    result = d.install_gimp(area, app_root=app, source=installer)
    assert result['installation_action'] == 'not-installed'
    assert [n['code'] for n in result['notices']] == ['gimp_target_occupied']
    assert result['notices'][0]['target'] == str(target)
    assert calls == []


def test_new_gimp_during_download_stops_before_install(fresh_gimp, monkeypatch):
    app, area, installer, target, calls = fresh_gimp
    checks = iter([[], [('registered while downloading', '', '3.2.6')]])
    monkeypatch.setattr(d, '_registered_locations', lambda: next(checks))
    result = d.install_gimp(area, app_root=app, source=installer)
    assert result['installation_action'] == 'not-installed'
    assert [n['code'] for n in result['notices']] == ['gimp_appeared_during_preparation']
    assert calls == []


def test_official_install_failure_preserves_base_and_does_not_delete_external_files(fresh_gimp, monkeypatch):
    app, area, installer, target, calls = fresh_gimp
    before = (area / 'current.json').read_bytes()
    def fail(*args):
        target.mkdir(parents=True)
        (target / 'partial-from-upstream').write_text('official repair required')
        raise RuntimeError('official installation interrupted')
    monkeypatch.setattr(d, '_run_upstream_installer', fail)
    # A failed optional GIMP step is reported, never raised into the Aester installer.
    result = c.install_retouch_components(area, app_root=app, include_skill=False, gimp_source=installer)
    assert result['ready'] is True and result['gimp']['installation_action'] == 'failed'
    assert [n['code'] for n in result['notices']] == ['gimp_install_failed']
    assert 'interrupted' in d.describe_notices(result['notices'], 'zh_CN')[0]
    assert (area / 'current.json').read_bytes() == before
    assert (target / 'partial-from-upstream').exists()
    assert d.saved_gimp_path(area, app_root=app) is None


def test_upstream_install_without_confirmed_version_keeps_program_and_warns(fresh_gimp, monkeypatch):
    app, area, installer, target, calls = fresh_gimp
    real = d._discover
    def discover(path=None, *, probe=False):
        record = real(path, probe=probe)
        return {**record, 'status': 'check_failed', 'version': None, 'version_compatible': None} if target.exists() else record
    monkeypatch.setattr(d, '_discover', discover)
    result = d.install_gimp(area, app_root=app, source=installer)
    assert len(calls) == 1 and result['installation_action'] == 'installed-upstream'
    assert [n['code'] for n in result['notices']] == ['gimp_installed_unverified']
    assert (target / 'bin/gimp-console-3.2.exe').exists()


def test_activation_does_not_require_gimp_to_still_be_the_tested_version(fresh_gimp, monkeypatch):
    app, area, installer, target, calls = fresh_gimp
    result = c.install_retouch_components(area, app_root=app, include_skill=False, gimp_source=installer)
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: pytest.fail('activation must not re-probe GIMP'))
    assert c.verify_prepared_retouch_components(area, app_root=app, expected=result) == result


def test_every_notice_code_has_both_languages_and_no_placeholders_left():
    codes = set(d._NOTICE_TEXT['zh_CN'])
    assert codes == set(d._NOTICE_TEXT['en'])
    for code in codes:
        notice = {'code': code, 'tested_version': '3.2.6', 'found_version': '3.0.4', 'path': 'C:/g.exe',
                  'target': 'C:/GIMP 3', 'error': 'network down'}
        for language in ('zh_CN', 'en'):
            [text] = d.describe_notices([notice], language)
            assert '{' not in text and '}' not in text
    assert d.describe_notices([{'code': 'unknown'}], 'en') == []


def test_readonly_status_precedence_and_missing_saved_selection(case, monkeypatch, tmp_path):
    app, area = case
    old = tmp_path / 'old/bin/gimp-console-3.2.exe'
    d._save_result(area, app, {'ready': True, 'gimp': found(old), 'skill': None})
    observed = []
    def discover(path=None, *, probe=False):
        assert probe is False
        observed.append(path)
        return found(path or 'registry', status='selected_missing' if path else 'not_found')
    monkeypatch.setattr(d, '_discover', discover)
    monkeypatch.delenv('AESTER_GIMP', raising=False)
    status = c.retouch_component_status(area, app_root=app)
    assert observed[-1] == str(old)
    assert status['gimp']['selection_source'] == 'application-retouch-receipt'
    monkeypatch.setenv('AESTER_GIMP', 'environment-choice')
    c.retouch_component_status(area, app_root=app)
    assert observed[-1] is None  # discover_gimp then reads AESTER_GIMP itself
    c.retouch_component_status(area, app_root=app, selected_gimp_path='explicit-choice')
    assert observed[-1] == 'explicit-choice'


@pytest.mark.parametrize('target', ['top-receipt', 'skill-receipt', 'skill-text', 'skill-zip'])
def test_activation_rejects_changed_prepared_skill(case, target):
    app, area = case
    result = c.install_retouch_components(area, app_root=app, include_gimp=False)
    path = Path(result['receipt_path'] if target == 'top-receipt' else
                result['skill'][{'skill-receipt': 'receipt_path', 'skill-text': 'path', 'skill-zip': 'archive_path'}[target]])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        c.verify_prepared_retouch_components(area, app_root=app, expected=result)


def test_invalid_boolean_options_have_no_side_effects(case):
    app, area = case
    with pytest.raises(ValueError):
        c.install_retouch_components(area, app_root=app, include_gimp='yes')
    assert not (area / 'versions').exists()


def test_inaccessible_component_cache_keeps_bundled_skill_and_registry_discovery(case, monkeypatch):
    app, area = case
    original = Path.glob
    def restricted(path, pattern):
        if path == area / 'versions':
            raise PermissionError('cache not readable')
        return original(path, pattern)
    monkeypatch.setattr(Path, 'glob', restricted)
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: found('registered GIMP'))
    status = c.retouch_component_status(area, app_root=app)
    assert status['gimp']['status'] == 'available'
    assert status['skill']['status'] == 'bundled'
    assert status['gimp']['diagnostics'][0]['code'] == 'saved_selection_unavailable'
    assert status['skill']['diagnostics'][0]['code'] == 'skill_cache_unavailable'
    with pytest.raises(ValueError, match='previous GIMP selection'):
        d.install_gimp(area, app_root=app)


class _Body(io.BytesIO):
    def __init__(self, body, url):
        super().__init__(body)
        self.status, self.headers, self.url = 200, {}, url


@pytest.mark.parametrize('mode', ['official', 'auto'])
def test_regional_redirect_refusal_moves_on_at_once_to_the_next_gimp_source(fresh_gimp, monkeypatch, mode):
    """2026-10-09: download.gimp.org redirects by location (seen: ziply.mm.fcix.net, mirror.umd.edu). Such a
    host is refused, and the next listed source is tried at once instead of three times on the same URL."""
    app, area, installer, target, calls = fresh_gimp
    monkeypatch.setattr(c, '_download', REAL_DOWNLOAD)
    monkeypatch.setattr(c.time, 'sleep', lambda *args: None)
    official, opencolo = c.candidate_urls(d.GIMP_INSTALLER, 'official')
    assert official == d.GIMP_INSTALLER['url'] and opencolo.startswith('https://opencolo.mm.fcix.net/')
    if mode == 'auto':  # measured order; the official host failed its measurement for the same reason
        ranking = [{'source': base, 'host': base.split('/')[2], 'ok': True}
                   for base in ('https://download.gimp.org/gimp/', 'https://opencolo.mm.fcix.net/gimp/gimp/')]
        monkeypatch.setattr(c, '_probe', lambda family, item, progress=None: ranking)
    requests = []

    class Opener:
        def __init__(self, redirect):
            self.redirect = redirect

        def open(self, request, timeout=None):
            requests.append(request.full_url)
            if request.full_url == official:
                self.redirect.redirect_request(request, None, 302, 'Found', {},
                                               'https://ziply.mm.fcix.net/gimp/v3.2/windows/gimp-3.2.6-setup.exe')
            return _Body(installer.read_bytes(), request.full_url)
    monkeypatch.setattr(c.urllib.request, 'build_opener', lambda redirect: Opener(redirect))
    result = d.install_gimp(area, app_root=app, download_source=mode)
    assert requests == [official, opencolo]
    assert result['installation_action'] == 'installed-upstream' and len(calls) == 1
    assert c._valid_file(area / 'downloads' / d.GIMP_INSTALLER['filename'], d.GIMP_INSTALLER)


@pytest.mark.parametrize('mode', ['official', 'auto'])
def test_the_source_choice_reaches_the_gimp_download_through_the_component_entry(fresh_gimp, monkeypatch, mode):
    """Review 2026-10-09: install_retouch_components -> install -> install_gimp -> _download carries the choice;
    dropping it at any hop would silently turn /DOWNLOADSOURCE=official into auto for GIMP."""
    app, area, installer, target, calls = fresh_gimp
    seen = []

    def download(item, cache, progress=None, sources=None):
        seen.append(sources)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(installer, cache / item['filename'])
        return cache / item['filename']
    monkeypatch.setattr(c, '_download', download)
    result = c.install_retouch_components(area, app_root=app, include_skill=False, download_source=mode)
    assert result['gimp']['installation_action'] == 'installed-upstream' and len(calls) == 1
    assert len(seen) == 1 and isinstance(seen[0], c.DownloadSources) and seen[0].mode == mode


def test_gimp_download_source_choice_is_validated_before_any_change(case, monkeypatch):
    app, area = case
    monkeypatch.setattr(d, '_discover', lambda *args, **kwargs: pytest.fail('must not inspect GIMP'))
    with pytest.raises(ValueError, match='download_source'):
        c.install_retouch_components(area, app_root=app, download_source='fastest')
    with pytest.raises(ValueError, match='download_source'):
        d.install_gimp(area, app_root=app, download_source='mirror')
    assert not (area / 'versions').exists()
