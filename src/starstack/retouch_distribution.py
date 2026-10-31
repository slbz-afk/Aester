"""Optional upstream GIMP installation and application-local retouch skill.

GIMP remains a separately registered upstream application. Aester owns only its
download cache and skill copies; it never adopts or removes GIMP program files.
This module also runs in the installer's standard-library-only bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import zipfile

try:
    from . import components
except ImportError:
    import components

GIMP_VERSION = "3.2.6"
GIMP_INSTALLER = {
    "filename": "gimp-3.2.6-setup.exe", "size": 190358080,
    "sha256": "9337cccbc01d4098ee7a3dab215b3afbe6ece99c5287c92441d6f12cf541ebca",
    "url": "https://download.gimp.org/gimp/v3.2/windows/gimp-3.2.6-setup.exe",
}
# The installer is fetched from the official host or a listed mirror, whichever
# measures faster (components.DOWNLOAD_SOURCES, family "gimp"), and accepted only
# with the size and SHA256 above. download.gimp.org itself redirects by location
# to mirrors not on that list; such a redirect is refused and the next listed
# source is tried at once.
GIMP_SOURCE = {
    "url": "https://download.gimp.org/gimp/v3.2/gimp-3.2.6.tar.xz",
    "sha256": "40b15e90ad0c0c631b76da3c467ea9847fa5c24f37413ac5b492804860a28cd8",
    "size": 35004888,
    "scope": "GIMP main source, including upstream Windows installer scripts; not every bundled dependency source",
}
SKILL_FILES = {
    "SKILL.md": "565b00900df96b1d8102006f415e25e5a2dee97d0bfa9a233f8a700c9d27fb9f",
    "references/backend-interface.md": "9a91184defd25ae606e95241ef28f7f8555d75af5776bd95e30badc4d6b7ec10",
    "references/diagnosis-and-operations.md": "1c3bf963e641cf3f6d95af53dd0a96490fc2ddf0f7284d30dad4763274e9f791",
    "references/evidence-and-sources.md": "5868674322bc4e8c137d16f68eee9a371378c59046bfabf2a4a72773fa9a58fb",
    "references/generalization-evaluation.md": "7ace0941005d95454fc4b0d986f06906c4a73254b1858082b9d1fe0093caf69b",
    "references/goal-loop.md": "d29d4a8e3d3b0e27716d3a795a3cc5d63a79ba9b933b6f115d701bac618e79f4",
    "references/spatial-and-layers.md": "8f9051f1556b335950859284e5751e34dc643fc0350327aaf3d2fdc78633f5ce",
}
SKILL_VERSION = "nl2-v5.1"
_SKILL_ID = hashlib.sha256(json.dumps(SKILL_FILES, sort_keys=True).encode()).hexdigest()


def skill_files(source):
    """Return the exact reviewed seven-file text package; reject extra assets."""
    root = components._safe_root(source)
    actual = set()
    for path in root.rglob('*'):
        components._inside(root, path)
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(SKILL_FILES):
        raise ValueError("The retouch skill must contain exactly the reviewed text files")
    paths = []
    for relative, expected in sorted(SKILL_FILES.items()):
        path = components._inside(root, root / relative)
        if components._digest(path) != expected:
            raise ValueError("The frozen retouch skill changed: " + relative)
        paths.append((path, Path(relative)))
    return paths


def create_skill_archive(source, destination):
    """Write a deterministic, portable skill ZIP without global installation."""
    files = skill_files(source)
    destination = components._safe_root(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files:
            entry = zipfile.ZipInfo('aester-retouch/' + relative.as_posix(), (2026, 9, 27, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    return destination


def _prefix(app):
    return 'retouch-skill-' + _SKILL_ID[:12] + '-' + hashlib.sha256(str(app).encode()).hexdigest()[:10]


def verify_skill(directory, *, app_root):
    folder = components._safe_root(directory)
    app = components._safe_root(app_root)
    receipt_path = folder / 'retouch-skill-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('app_root') != str(app) or receipt.get('skill_id') != _SKILL_ID
            or receipt.get('version') != SKILL_VERSION):
        raise ValueError('Retouch skill receipt belongs to another application or version')
    # Unknown files added by users are retained and never adopted for uninstall.
    for relative, expected in SKILL_FILES.items():
        path = components._inside(folder, folder / 'aester-retouch' / relative)
        if components._digest(path) != expected:
            raise ValueError('Prepared retouch skill changed: ' + relative)
    archive = components._inside(folder, folder / 'aester-retouch.zip')
    if components._digest(archive) != receipt.get('archive_sha256'):
        raise ValueError('Prepared retouch skill ZIP changed')
    with zipfile.ZipFile(archive) as zipped:
        if sorted(zipped.namelist()) != sorted('aester-retouch/' + name for name in SKILL_FILES):
            raise ValueError('Unexpected retouch skill ZIP contents')
        for name, expected in SKILL_FILES.items():
            if hashlib.sha256(zipped.read('aester-retouch/' + name)).hexdigest() != expected:
                raise ValueError('Retouch skill ZIP text changed')
    return {'status': 'ready', 'path': str(folder / 'aester-retouch/SKILL.md'),
            'archive_path': str(archive), 'version': SKILL_VERSION,
            'receipt_path': str(receipt_path), 'receipt_sha256': components._digest(receipt_path),
            'global_install_performed': False, 'managed_by_aester': True}


def prepare_skill(component_root, *, app_root, source=None):
    root = components._safe_root(component_root)
    app = components._safe_root(app_root)
    source = components._safe_root(source or app / 'skills/aester-retouch')
    files = skill_files(source)
    owner = components.ownership.identity_for_app(app)
    if owner is not None:
        components.ownership.register_install(owner['install_root'], root, root / 'downloads')
    with components._lock(root):
        for candidate in sorted((root / 'versions').glob(_prefix(app) + '-*'), reverse=True)[:100]:
            components._inside(root, candidate)
            try:
                result = verify_skill(candidate, app_root=app)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                continue
            components.ownership.claim_version(root, candidate.name, app)
            return result
        version = _prefix(app) + '-' + uuid.uuid4().hex[:8]
        stage = components._inside(root, root / 'staging' / uuid.uuid4().hex)
        final = components._inside(root, root / 'versions' / version)
        stage.mkdir(parents=True)
        try:
            for path, relative in files:
                target = stage / 'aester-retouch' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            archive = create_skill_archive(stage / 'aester-retouch', stage / 'aester-retouch.zip')
            components._atomic(stage / 'retouch-skill-receipt.json', {
                'schema_version': 1, 'app_root': str(app), 'skill_id': _SKILL_ID,
                'version': SKILL_VERSION, 'archive_sha256': components._digest(archive),
                'global_install_performed': False})
            verify_skill(stage, app_root=app)
            components.ownership.seal_version(stage, version, app)
            final.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(final)
            components.ownership.claim_version(root, version, app)
            return verify_skill(final, app_root=app)
        except Exception as error:
            if stage.exists():
                components._atomic(stage / 'failure.json', {'error': str(error), 'type': type(error).__name__})
            raise


def skill_status(component_root, *, app_root):
    root, app = components._safe_root(component_root), components._safe_root(app_root)
    diagnostics = []
    try:
        candidates = sorted((root / 'versions').glob(_prefix(app) + '-*'), reverse=True)[:100]
    except OSError as error:
        candidates = []  # App-local skill remains usable without component-cache access.
        diagnostics.append({'code': 'skill_cache_unavailable', 'message': str(error)})
    for candidate in candidates:
        try:
            components._inside(root, candidate)
            return {**verify_skill(candidate, app_root=app), 'diagnostics': diagnostics}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    source = app / 'skills/aester-retouch'
    try:
        skill_files(source)
    except (OSError, ValueError):
        return {'status': 'missing', 'path': None, 'archive_path': None,
                'global_install_performed': False, 'diagnostics': diagnostics}
    archive = app / 'skills/aester-retouch.zip'
    return {'status': 'bundled', 'path': str(source / 'SKILL.md'),
            'archive_path': str(archive) if archive.is_file() else None,
            'version': SKILL_VERSION, 'global_install_performed': False,
            'managed_by_aester': True, 'diagnostics': diagnostics}


def _discover(selected_path=None, *, probe=False):
    # The bootstrap includes this small stdlib-only package as well.
    from starstack.finishing_components import discover_gimp
    return discover_gimp(selected_path, probe=probe)


def _onboarding_prefix(app):
    return 'retouch-onboarding-' + hashlib.sha256(str(app).encode()).hexdigest()[:10]


def saved_gimp_path(component_root, *, app_root, diagnostics=None):
    """Read the most recent path saved for this exact application version.

    A missing saved executable remains a pinned missing selection. AESTER_GIMP
    and explicit UI selections take priority in the caller. No external files
    become owned merely because their path occurs in this receipt.
    """
    root, app = components._safe_root(component_root), components._safe_root(app_root)
    records = []
    try:
        paths = list((root / 'versions').glob(_onboarding_prefix(app) + '-*/retouch-receipt.json'))
        paths = sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True)[:100]
    except OSError as error:
        paths = []  # Unavailable cache cannot prevent registry or explicit discovery.
        if diagnostics is not None:
            diagnostics.append({'code': 'saved_selection_unavailable', 'message': str(error)})
    for path in paths:
        try:
            components._inside(root, path)
            value = json.loads(path.read_text(encoding='utf-8'))
            gimp = value.get('result', {}).get('gimp')
            if (value.get('app_root') != str(app) or not isinstance(value.get('created_ns'), int)
                    or not isinstance(gimp, dict) or gimp.get('managed_by_aester') is not False
                    or gimp.get('version') != GIMP_VERSION):
                continue
            selected = Path(gimp['selected_path'])
            if selected.is_absolute() and selected.name.lower() in {'gimp-console-3.2.exe', 'gimp-console-3.0.exe'}:
                records.append((value['created_ns'], str(selected)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return max(records)[1] if records else None


def discover(component_root, *, app_root, selected_gimp_path=None, probe=False):
    selected = selected_gimp_path
    saved = False
    diagnostics = []
    if selected is None and not os.environ.get('AESTER_GIMP'):
        selected = saved_gimp_path(component_root, app_root=app_root, diagnostics=diagnostics)
        saved = selected is not None
    result = _discover(selected, probe=probe)
    if saved:
        result['selection_source'] = 'application-retouch-receipt'
    result.setdefault('diagnostics', []).extend(diagnostics)
    return result


def _save_result(root, app, result):
    owner = components.ownership.identity_for_app(app)
    if owner is not None:
        components.ownership.register_install(owner['install_root'], root, root / 'downloads')
    with components._lock(root):
        version = _onboarding_prefix(app) + '-' + uuid.uuid4().hex[:8]
        stage = components._inside(root, root / 'staging' / uuid.uuid4().hex)
        final = components._inside(root, root / 'versions' / version)
        stage.mkdir(parents=True)
        components._atomic(stage / 'retouch-receipt.json', {
            'schema_version': 1, 'app_root': str(app), 'created_ns': time.time_ns(), 'result': result})
        components.ownership.seal_version(stage, version, app)
        final.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(final)
        components.ownership.claim_version(root, version, app)
        receipt = final / 'retouch-receipt.json'
        result.update(receipt_path=str(receipt), receipt_sha256=components._digest(receipt))


def _registered_locations():
    from starstack.finishing_components import _registry_locations
    locations, diagnostics = _registry_locations()
    if diagnostics:
        raise ValueError('Cannot safely establish existing GIMP registration; choose the existing installation')
    return locations


def _install_directory():
    local = os.environ.get('LOCALAPPDATA')
    if os.name != 'nt' or not local:
        raise ValueError('Official optional GIMP installation is supported on Windows only')
    return components._safe_root(Path(local) / 'Programs/GIMP 3')


def _run_upstream_installer(executable, destination, log, progress=None):
    from starstack.process_guard import guarded_process
    command = [str(executable), '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
               '/NOCLOSEAPPLICATIONS', '/NORESTARTAPPLICATIONS', '/TYPE=full', '/TASKS=',
               '/DIR=' + str(destination), '/LOG=' + str(log)]
    with guarded_process(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
        started = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - started > 1200:
                raise TimeoutError('Official GIMP installation timed out; inspect its separate installation log')
            components._notify(progress, phase='install-gimp', message='Official GIMP installer is running')
            time.sleep(0.25)
        if process.returncode != 0:
            raise RuntimeError(f'Official GIMP installation did not finish (exit {process.returncode}); see {log}')


_NOTICE_TEXT = {
    'zh_CN': {
        'gimp_reused_other_version': 'GIMP：检测到本机已安装 GIMP {found}，已直接使用，没有重新安装。Aester 的自然语言修图基于 GIMP {tested} 开发和验证，版本不同时兼容性不能保证（目前修图渲染要求 {tested}）。如需修图，可从 GIMP 官网安装 {tested}，再在「设置 → 组件与提示词管理」中选择。',
        'gimp_reused_unverified': 'GIMP：检测到本机已安装 GIMP（{path}），已直接使用，没有重新安装，但无法确认它的版本。Aester 修图使用 GIMP {tested}，版本不同时兼容性不能保证。',
        'gimp_registered_missing': 'GIMP：系统里有 GIMP 的安装登记，但找不到它的程序文件。为避免与原有安装冲突，没有替你安装。可先用 GIMP 自带的修复或卸载入口处理，再在「设置 → 组件与提示词管理」中准备。Aester 修图使用 GIMP {tested}。',
        'gimp_selected_missing': 'GIMP：之前选择的 GIMP 程序已不存在，没有替你安装。可在「设置 → 组件与提示词管理」中重新选择。',
        'gimp_check_incomplete': 'GIMP：无法完整检查本机的 GIMP 安装，为避免冲突没有安装。可在「设置 → 组件与提示词管理」中选择或重试。',
        'gimp_target_occupied': 'GIMP：安装目录 {target} 已存在且不属于 Aester，没有替你安装。可在「设置 → 组件与提示词管理」中选择已有的 GIMP。',
        'gimp_appeared_during_preparation': 'GIMP：准备期间出现了新的 GIMP 安装，没有再安装。可在「设置 → 组件与提示词管理」中确认。',
        'gimp_installed_unverified': 'GIMP：官方安装程序已安装 GIMP {tested}，但 Aester 未能确认它的版本。修图前可在「设置 → 组件与提示词管理」中检查。',
        'gimp_install_failed': 'GIMP：没有安装成功（{error}）。Aester 其余部分不受影响，可稍后在「设置 → 组件与提示词管理」中重试。',
        'retouch_prepare_failed': '修图组件：准备未完成（{error}）。Aester 其余部分不受影响，可稍后在「设置 → 组件与提示词管理」中重试。',
    },
    'en': {
        'gimp_reused_other_version': 'GIMP: GIMP {found} is already installed and is used as it is; it was not reinstalled. Aester natural-language retouching is developed and verified with GIMP {tested}; with a different version, compatibility is not guaranteed (retouch rendering currently requires {tested}). To retouch, install GIMP {tested} from gimp.org and select it under Settings > Components & prompts.',
        'gimp_reused_unverified': 'GIMP: an existing GIMP ({path}) is used as it is and was not reinstalled, but its version could not be confirmed. Aester retouching uses GIMP {tested}; with a different version, compatibility is not guaranteed.',
        'gimp_registered_missing': 'GIMP: Windows lists a GIMP installation, but its program files are missing. To avoid conflicting with it, GIMP was not installed. Repair or uninstall it with GIMP\'s own entry, then prepare it under Settings > Components & prompts. Aester retouching uses GIMP {tested}.',
        'gimp_selected_missing': 'GIMP: the previously selected GIMP no longer exists, so nothing was installed. Select it again under Settings > Components & prompts.',
        'gimp_check_incomplete': 'GIMP: the existing GIMP installation could not be fully checked, so nothing was installed to avoid a conflict. Select or retry under Settings > Components & prompts.',
        'gimp_target_occupied': 'GIMP: the folder {target} already exists and does not belong to Aester, so GIMP was not installed. Select an existing GIMP under Settings > Components & prompts.',
        'gimp_appeared_during_preparation': 'GIMP: a new GIMP installation appeared during preparation, so nothing more was installed. Check it under Settings > Components & prompts.',
        'gimp_installed_unverified': 'GIMP: the official installer installed GIMP {tested}, but Aester could not confirm its version. Check it under Settings > Components & prompts before retouching.',
        'gimp_install_failed': 'GIMP: installation did not succeed ({error}). The rest of Aester is unaffected; retry later under Settings > Components & prompts.',
        'retouch_prepare_failed': 'Retouch components: preparation did not finish ({error}). The rest of Aester is unaffected; retry later under Settings > Components & prompts.',
    },
}


def describe_notices(notices, language='zh_CN'):
    """One readable paragraph per notice, in Chinese or English."""
    table = _NOTICE_TEXT['en' if str(language or '').lower().startswith('en') else 'zh_CN']
    lines = []
    for notice in notices or []:
        template = table.get(notice.get('code'))
        if template is None:
            continue
        values = {'found': notice.get('found_version') or '?', 'tested': notice.get('tested_version') or GIMP_VERSION,
                  'path': notice.get('path') or '?', 'target': notice.get('target') or '?',
                  'error': str(notice.get('error') or '?')[:300]}
        lines.append(template.format(**values))
    return lines


def _notice(code, gimp=None, **extra):
    """A language-neutral note for the installer and GUI to word for the user."""
    gimp = gimp or {}
    return {'code': code, 'tested_version': GIMP_VERSION, 'found_version': gimp.get('version'),
            'path': gimp.get('selected_path'), **extra}


def _not_installed(code, gimp, **extra):
    """GIMP was left exactly as found; nothing was downloaded or started."""
    return {'status': (gimp or {}).get('status', 'not_found'), 'installation_action': 'not-installed',
            'selected_path': None, 'version': None, 'version_compatible': None,
            'managed_by_aester': False, 'retained_on_aester_uninstall': True,
            'notices': [_notice(code, gimp, **extra)]}


def _sources(download_source):
    if isinstance(download_source, components.DownloadSources):
        return download_source
    return components.DownloadSources(download_source, [GIMP_INSTALLER])


def install_gimp(component_root, *, app_root, cache_dir=None, source=None, selected_path=None, progress=None,
                 download_source='auto'):
    """Reuse any GIMP already present; install upstream only when none is found.

    An existing program is used as it is, whatever its version, with a notice
    when it is not the tested version. A registration without its program, a
    missing pinned selection or an occupied target is never changed or
    reinstalled over. The independent GIMP installer owns its files, shortcuts
    and registration; GIMP is retained if Aester later fails or is removed.
    download_source is 'auto', 'official' or a components.DownloadSources.
    """
    sources = _sources(download_source)
    root, app = components._safe_root(component_root), components._safe_root(app_root)
    components._notify(progress, phase='check-gimp', message='Looking for an existing GIMP')
    selected = discover(root, app_root=app, selected_gimp_path=selected_path, probe=True)
    if any(item.get('code') == 'saved_selection_unavailable' for item in selected.get('diagnostics', [])):
        raise ValueError('Cannot read the previous GIMP selection; select an existing GIMP explicitly before preparing components')
    status = selected['status']
    if status in ('available', 'incompatible_version') or (status == 'check_failed' and selected.get('path_exists')):
        notices = []
        if status == 'incompatible_version':
            notices.append(_notice('gimp_reused_other_version', selected))
        elif status != 'available' or selected.get('version_compatible') is not True:
            notices.append(_notice('gimp_reused_unverified', selected))
        return {**selected, 'installation_action': 'reused-existing', 'retained_on_aester_uninstall': True,
                'notices': notices}
    if status == 'selected_missing' or selected.get('selection_locked'):
        return _not_installed('gimp_selected_missing', selected)
    if status != 'not_found':
        return _not_installed('gimp_check_incomplete', selected)
    if _registered_locations():
        return _not_installed('gimp_registered_missing', selected)
    destination = _install_directory()
    if destination.exists():
        return _not_installed('gimp_target_occupied', selected, target=str(destination))
    cache = components._safe_root(cache_dir or root / 'downloads')
    owner = components.ownership.identity_for_app(app)
    if owner is not None:
        components.ownership.register_install(owner['install_root'], root, cache)
    with components._lock(root), components.ownership.download_lock(cache):
        existed = (cache / GIMP_INSTALLER['filename']).exists()
        if source is not None:
            supplied = components._safe_root(source)
            if not components._valid_file(supplied, GIMP_INSTALLER):
                raise ValueError('GIMP installer differs from the official pinned release')
            target = components._inside(cache, cache / GIMP_INSTALLER['filename'])
            if supplied != target:
                shutil.copyfile(supplied, target)
        executable = components._download(GIMP_INSTALLER, cache, progress, sources)
        if owner is not None:
            components.ownership.claim_download(cache, GIMP_INSTALLER, app, existed_before=existed)
        # A download can take time: repeat the external-state check immediately
        # before starting upstream so a new registration is never installed over.
        if _registered_locations() or destination.exists() or _discover(selected_path)['status'] != 'not_found':
            return _not_installed('gimp_appeared_during_preparation', selected)
        log = components._inside(root, root / ('gimp-install-' + uuid.uuid4().hex + '.log'))
        _run_upstream_installer(executable, destination, log, progress)
        result = _discover(str(destination / 'bin/gimp-console-3.2.exe'), probe=True)
        notices = []
        if result['status'] != 'available' or result.get('version_compatible') is not True:
            # The official installer reported success; keep its program and say
            # that our own check could not confirm it, instead of failing Aester.
            notices.append(_notice('gimp_installed_unverified', result, log=str(log)))
        return {**result, 'installation_action': 'installed-upstream', 'installation_log': str(log),
                'installer_sha256': GIMP_INSTALLER['sha256'], 'retained_on_aester_uninstall': True,
                'notices': notices}


def install(component_root, *, include_gimp=True, include_skill=True, progress=None,
            app_root=None, cache_dir=None, gimp_source=None, selected_gimp_path=None, skill_source=None,
            download_source='auto'):
    if type(include_gimp) is not bool or type(include_skill) is not bool:
        raise ValueError('Retouch component selections must be bool')
    sources = _sources(download_source)
    root = components._safe_root(component_root or components.default_component_root())
    app = components._safe_root(app_root or components._app_root())
    result = {'schema_version': 1, 'ready': False, 'gimp': None, 'skill': None,
              'global_skill_install_performed': False}
    if include_skill:
        components._notify(progress, phase='prepare-retouch-skill', message='Preparing application-local skill source and ZIP')
        result['skill'] = prepare_skill(root, app_root=app, source=skill_source)
    if include_gimp:
        try:
            result['gimp'] = install_gimp(root, app_root=app, cache_dir=cache_dir,
                source=gimp_source, selected_path=selected_gimp_path, progress=progress,
                download_source=sources)
        except Exception as error:
            # GIMP is an optional external program: a failed download or upstream
            # installation is reported, never allowed to fail Aester itself.
            # (Installer cancellation derives from BaseException and still stops.)
            result['gimp'] = {'status': 'install_failed', 'installation_action': 'failed',
                              'selected_path': None, 'version': None, 'version_compatible': None,
                              'managed_by_aester': False, 'retained_on_aester_uninstall': True,
                              'error': str(error), 'notices': [_notice('gimp_install_failed', error=str(error))]}
    result['notices'] = list((result['gimp'] or {}).get('notices', []))
    result['ready'] = True
    if include_gimp or include_skill:
        _save_result(root, app, result)
    return result


def verify_prepared(component_root, *, app_root, expected):
    root, app = components._safe_root(component_root), components._safe_root(app_root)
    if not isinstance(expected, dict) or expected.get('ready') is not True:
        raise ValueError('Retouch preparation is incomplete')
    if expected.get('receipt_path'):
        receipt = components._inside(root, Path(expected['receipt_path']))
        if (receipt.parent.parent != root / 'versions'
                or not receipt.parent.name.startswith(_onboarding_prefix(app) + '-')
                or receipt.name != 'retouch-receipt.json'
                or components._digest(receipt) != expected.get('receipt_sha256')):
            raise ValueError('Prepared retouch component receipt changed')
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if (saved.get('app_root') != str(app)
                or saved.get('result') != {k: v for k, v in expected.items() if k not in ('receipt_path', 'receipt_sha256')}):
            raise ValueError('Prepared retouch component identity changed')
    if expected.get('skill') is not None:
        skill = expected['skill']
        receipt = components._inside(root, Path(skill['receipt_path']))
        if (receipt.parent.parent != root / 'versions' or not receipt.parent.name.startswith(_prefix(app) + '-')
                or receipt.name != 'retouch-skill-receipt.json' or components._digest(receipt) != skill.get('receipt_sha256')):
            raise ValueError('Prepared retouch skill receipt changed')
        verified = verify_skill(receipt.parent, app_root=app)
        if verified != skill:
            raise ValueError('Prepared retouch skill identity changed')
    # GIMP is an external program recorded with its notices at preparation; its
    # current state is shown by the component manager, never re-required here.
    return expected
