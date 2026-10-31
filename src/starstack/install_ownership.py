"""Explicit installation ownership and conservative, retryable uninstallation.

Standard-library only: the uninstall copy runs outside every tree it deletes.
No directory from user input or from a receipt is recursively deleted.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import re
import stat
import time
import uuid

IDENTITY = 'aester-installation.json'
VERSION_FILES = '.aester-owned-files.json'
DOWNLOAD_OWNERS = '.aester-download-owners.json'


def safe(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    if path == Path(path.anchor):
        raise ValueError('不能使用文件系统根目录')
    for part in (path, *path.parents):
        try: info = part.lstat()
        except FileNotFoundError: continue
        if stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('卸载路径不能经过重解析点：' + str(part))
    return path


def child(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('无效的相对清单路径')
    bits = PurePosixPath(relative)
    if bits.is_absolute() or not bits.parts or any(x in ('', '.', '..') or ':' in x or x[-1:] in (' ', '.') for x in bits.parts):
        raise ValueError('清单路径越界')
    root = safe(root)
    path = safe(root.joinpath(*bits.parts))
    if not path.is_relative_to(root) or path == root:
        raise ValueError('清单路径越界')
    return path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, data):
    path = safe(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _read(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def _token(value):
    if not isinstance(value, str) or not re.fullmatch('[a-zA-Z0-9-]+', value):
        raise ValueError('无效版本身份')
    return value


def _identity(root):
    root = safe(root)
    data = _read(root / IDENTITY)
    if (data.get('schema_version') != 1 or data.get('install_root') != str(root)
            or not re.fullmatch('[a-f0-9]{32}', data.get('installation_id', ''))):
        raise ValueError('安装归属记录与路径不一致')
    return data


def register_install(install_root, component_root, cache_root):
    """Called before prepare. Existing legacy directories are never adopted."""
    root, components, cache = map(safe, (install_root, component_root, cache_root))
    if components == root or root.is_relative_to(components):
        raise ValueError('组件根不能包含安装根')
    if cache == root or root.is_relative_to(cache) or cache.is_relative_to(root / 'versions') or cache.is_relative_to(components / 'versions'):
        raise ValueError('下载缓存不能包含安装文件或运行库')
    root.mkdir(parents=True, exist_ok=True)
    if (root / IDENTITY).exists():
        data = _identity(root)
    else:
        data = {'schema_version': 1, 'installation_id': uuid.uuid4().hex,
                'install_root': str(root), 'component_roots': [], 'download_roots': [], 'application_versions': {}}
    for field, item in [('component_roots', components), ('download_roots', cache)]:
        if str(item) not in data[field]: data[field].append(str(item))
    atomic(root / IDENTITY, data)
    return data


def identity_for_app(app_root):
    app = safe(app_root)
    root = app.parent.parent
    if app.parent.name != 'versions' or not (root / IDENTITY).is_file():
        return None
    data = _identity(root)
    if app.parent != root / 'versions': raise ValueError('应用版本不属于安装根')
    _token(app.name)
    return data


def record_application(app_root):
    app = safe(app_root)
    data = identity_for_app(app)
    if data is None: return
    manifest = _read(app / 'PACKAGE_CONTENTS.json')
    files = manifest['sha256']
    if manifest['version_id'] != app.name: raise ValueError('应用清单身份不一致')
    for relative, expected in files.items():
        if digest(child(app, relative)) != expected: raise ValueError('应用文件改变：' + relative)
    data['application_versions'][app.name] = dict(files)
    data['application_versions'][app.name]['PACKAGE_CONTENTS.json'] = digest(app / 'PACKAGE_CONTENTS.json')
    root = Path(data['install_root'])
    data['launchers'] = {name: digest(root / name) for name in ('Aester.exe', 'Aester-MCP.exe', 'Aester.ico') if (root / name).is_file()}
    atomic(Path(data['install_root']) / IDENTITY, data)


def seal_version(stage, version, app_root):
    """Seal only a brand-new stage after validation, before atomic publication."""
    stage = safe(stage)
    data = identity_for_app(app_root)
    if data is None: return
    files = {}
    for path in sorted(stage.rglob('*')):
        safe(path)
        if path.is_file():
            relative = path.relative_to(stage).as_posix()
            files[relative] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    atomic(stage / VERSION_FILES, {'schema_version': 1, 'version': _token(version),
           'app_root': str(safe(app_root)), 'created_for': data['installation_id'], 'files': files})


def claim_version(component_root, version, app_root):
    data = identity_for_app(app_root)
    if data is None: return False
    root = safe(component_root)
    if str(root) not in data['component_roots']: raise ValueError('组件根没有登记给本安装')
    seal = root / 'versions' / _token(version) / VERSION_FILES
    if not seal.is_file(): return False  # An old unsealed version remains unowned.
    manifest = _read(seal)
    if manifest.get('version') != version or manifest.get('app_root') != str(safe(app_root)):
        raise ValueError('组件清单不属于本应用')
    owners = root / 'ownership'; owners.mkdir(exist_ok=True)
    record = owners / (data['installation_id'] + '.json')
    owned = _read(record) if record.exists() else {'schema_version': 1, 'installation_id': data['installation_id'],
                   'install_root': data['install_root'], 'versions': {}}
    if owned.get('installation_id') != data['installation_id'] or owned.get('install_root') != data['install_root']:
        raise ValueError('组件所有者记录不一致')
    owned['versions'][version] = {'manifest_sha256': digest(seal)}
    # Keep a second, installation-bound copy until Inno's final commit. A
    # partial metadata cleanup must not destroy the information needed to retry.
    fresh = _identity(data['install_root'])
    versions = fresh.setdefault('component_versions', {}).setdefault(str(root), {})
    versions[version] = {'manifest_sha256': digest(seal), 'manifest_text': seal.read_bytes().decode('utf-8')}
    atomic(Path(data['install_root']) / IDENTITY, fresh)
    atomic(record, owned)
    return True


@contextmanager
def download_lock(cache):
    cache = safe(cache); cache.mkdir(parents=True, exist_ok=True)
    handle = (cache / '.aester-download.lock').open('a+b')
    try:
        if not handle.tell(): handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def claim_download(cache, item, app_root, *, existed_before):
    data = identity_for_app(app_root)
    if data is None: return
    cache = safe(cache)
    if str(cache) not in data['download_roots']: raise ValueError('下载根没有登记给本安装')
    name = item['filename']; path = child(cache, name)
    if path.parent != cache: raise ValueError('下载文件名无效')
    ledger = cache / DOWNLOAD_OWNERS
    owned = _read(ledger) if ledger.exists() else {'schema_version': 1, 'files': {}}
    entry = owned['files'].get(name)
    # Pre-existing unregistered files are not silently adopted, even if pinned.
    if entry is None and existed_before: return
    if path.stat().st_size != item['size'] or digest(path) != item['sha256']: raise ValueError('下载文件改变')
    if entry is None:
        entry = {'sha256': item['sha256'], 'bytes': item['size'], 'owners': {}}
        owned['files'][name] = entry
    if entry['sha256'] != item['sha256'] or entry['bytes'] != item['size']: raise ValueError('下载归属冲突')
    entry['owners'][data['installation_id']] = data['install_root']
    atomic(ledger, owned)


def _generated_bytecode(root, source_files):
    result = set()
    for relative in source_files:
        if not relative.endswith('.py'): continue
        source = child(root, relative)
        directory = source.parent / '__pycache__'
        if not directory.is_dir(): continue
        safe(directory)
        for candidate in directory.glob(source.stem + '.cpython-*.pyc'):
            safe(candidate)
            if not re.fullmatch(re.escape(source.stem) + r'\.cpython-\d+(?:\.opt-[12])?\.pyc', candidate.name): continue
            with candidate.open('rb') as stream: magic = stream.read(4)
            if magic == importlib.util.MAGIC_NUMBER:
                result.add(candidate.relative_to(root).as_posix())
    return result


def _all_owner_records(root):
    result = {}
    directory = safe(root) / 'ownership'
    if not directory.exists(): return result
    safe(directory)
    for file in directory.glob('*.json'):
        record = _read(file)
        ident = record.get('installation_id', '')
        if not re.fullmatch('[a-f0-9]{32}', ident) or file.name != ident + '.json' or record.get('schema_version') != 1:
            raise ValueError('共享组件所有者记录无效')
        for version in record['versions']: _token(version)
        result[ident] = record
    return result


def describe_installation(root):
    """Small read-only UI summary from owned records, without scanning user data."""
    owner = _identity(root)
    count = total = 0
    for area, versions in owner.get('component_versions', {}).items():
        safe(area)
        records = _all_owner_records(area)
        for name, saved in versions.items():
            _token(name)
            if any(name in rec['versions'] for ident, rec in records.items() if ident != owner['installation_id']):
                continue
            manifest = json.loads(saved['manifest_text'])
            count += len(manifest['files'])
            total += sum(item['bytes'] for item in manifest['files'].values())
    for area in owner.get('download_roots', []):
        path = safe(area) / DOWNLOAD_OWNERS
        if not path.exists(): continue
        for entry in _read(path)['files'].values():
            if set(entry['owners']) == {owner['installation_id']}:
                count += 1; total += entry['bytes']
    return {'owned_component_and_download_files': count, 'owned_component_and_download_bytes': total,
            'summary_zh': f'本安装独占组件和下载包：约 {total / (1024 ** 3):.2f} GB，{count:,} 个文件。'}


def cleanup(install_root, *, progress=None):
    """Delete explicitly owned files. Caller must hold lifecycle exclusive gate.

    Unknown content and changed bytes survive. Permission/locking errors abort
    with a persistent journal so Inno must leave its uninstall entry intact.
    """
    root = safe(install_root); owner = _identity(root); ident = owner['installation_id']
    journal = root / 'uninstall-result.json'
    report = {'schema_version': 1, 'installation_id': ident, 'state': 'planning',
              'deleted_files': 0, 'deleted_bytes': 0, 'preserved': [], 'errors': [], 'user_data_deleted': False}
    component_roots = [safe(x) for x in owner['component_roots']]
    caches = [safe(x) for x in owner['download_roots']]
    plans = []
    updates = []
    prune = set()
    try:
        try: from . import components
        except ImportError: import components
        with ExitStack() as locks:
            for area in sorted(component_roots): locks.enter_context(components._lock(area))
            for area in sorted(caches): locks.enter_context(download_lock(area))
            # Complete all structural checks before deleting the first byte.
            for area in component_roots:
                records = _all_owner_records(area)
                record = records.get(ident)
                saved_versions = owner.get('component_versions', {}).get(str(area), {})
                owned_names = set(saved_versions) | (set(record['versions']) if record else set())
                for existing in (area / 'versions').glob('*'):
                    safe(existing)
                    if existing.is_dir() and existing.name not in owned_names:
                        report['preserved'].append({'path': str(existing), 'reason': '其他安装或旧版无归属组件'})
                for existing in area.iterdir():
                    safe(existing)
                    if existing.name not in {'versions', 'ownership', 'current.json', 'components.lock', 'downloads'}:
                        if existing.is_file() or (existing.is_dir() and any(existing.iterdir())):
                            report['preserved'].append({'path': str(existing), 'reason': '诊断日志、未完成准备或未归属文件'})
                for name in sorted(owned_names):
                    target = child(area, 'versions/' + _token(name))
                    if any(name in x['versions'] for key, x in records.items() if key != ident):
                        report['preserved'].append({'path': str(target), 'reason': '另一安装仍登记使用'})
                        continue
                    seal_path = target / VERSION_FILES
                    saved = saved_versions.get(name)
                    if saved:
                        encoded = saved['manifest_text'].encode('utf-8')
                        if hashlib.sha256(encoded).hexdigest() != saved['manifest_sha256']:
                            raise ValueError('安装内组件归属备份已改变')
                        seal = json.loads(saved['manifest_text'])
                        expected_manifest_hash = saved['manifest_sha256']
                    elif record:
                        expected_manifest_hash = record['versions'][name]['manifest_sha256']
                        seal = _read(seal_path)
                    else: raise ValueError('组件清单没有归属')
                    if seal_path.exists() and digest(seal_path) != expected_manifest_hash:
                        raise ValueError('组件所有权清单已改变')
                    if seal.get('version') != name or Path(seal['app_root']).parent != root / 'versions':
                        raise ValueError('组件清单不属于安装路径')
                    files = seal['files']
                    expected = set(files) | _generated_bytecode(target, files)
                    for relative in sorted(expected):
                        path = child(target, relative)
                        if not path.exists(): continue
                        if not path.is_file(): raise ValueError('文件清单目标不是普通文件')
                        check = files.get(relative)
                        if check and (path.stat().st_size != check['bytes'] or digest(path) != check['sha256']):
                            report['preserved'].append({'path': str(path), 'reason': '原登记文件已修改'})
                            continue
                        plans.append((path, check or {'sha256': digest(path), 'bytes': path.stat().st_size}))
                    for path in target.rglob('*'):
                        safe(path)
                        if path.is_file() and path.relative_to(target).as_posix() not in expected | {VERSION_FILES}:
                            report['preserved'].append({'path': str(path), 'reason': '未归属文件'})
                    # Keep every manifest until all payload deletions succeed;
                    # otherwise a later locked file would make retry impossible.
                    if seal_path.exists():
                        updates.append(('file', seal_path, {'sha256': digest(seal_path), 'bytes': seal_path.stat().st_size}))
                    prune.add(target)
                if record: updates.append(('owner', area / 'ownership' / (ident + '.json'), None))
                pointer = area / 'current.json'
                if pointer.exists():
                    current = _read(pointer)
                    if current.get('version') in owned_names and not any(current.get('version') in x['versions'] for key, x in records.items() if key != ident):
                        updates.append(('file', pointer, {'sha256': digest(pointer), 'bytes': pointer.stat().st_size}))
            for cache in caches:
                ledger = cache / DOWNLOAD_OWNERS
                if not ledger.exists():
                    report['preserved'].append({'path': str(cache), 'reason': '旧版或未归属下载缓存'})
                    continue
                record = _read(ledger)
                if record.get('schema_version') != 1: raise ValueError('下载归属版本无效')
                for existing in cache.iterdir():
                    safe(existing)
                    if existing.name not in set(record['files']) | {DOWNLOAD_OWNERS, '.aester-download.lock'}:
                        report['preserved'].append({'path': str(existing), 'reason': '未归属下载目录内容'})
                for name, entry in list(record['files'].items()):
                    path = child(cache, name)
                    if path.parent != cache: raise ValueError('下载文件路径越界')
                    if ident not in entry['owners']: continue
                    del entry['owners'][ident]
                    if entry['owners']:
                        report['preserved'].append({'path': str(path), 'reason': '共享下载缓存'})
                        continue
                    if path.exists():
                        if path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']:
                            plans.append((path, {'sha256': entry['sha256'], 'bytes': entry['bytes']}))
                        else: report['preserved'].append({'path': str(path), 'reason': '下载文件已修改'})
                    del record['files'][name]
                updates.append(('json', ledger, record))
            for version, files in owner['application_versions'].items():
                app = child(root, 'versions/' + _token(version))
                for relative, expected in files.items():
                    path = child(app, relative)
                    if not path.exists(): continue
                    if digest(path) != expected:
                        raise ValueError('程序文件已修改，无法安全自动卸载：' + str(path))
                    plans.append((path, {'sha256': expected, 'bytes': path.stat().st_size}))
                for relative in _generated_bytecode(app, files):
                    path = child(app, relative)
                    plans.append((path, {'sha256': digest(path), 'bytes': path.stat().st_size}))
                prune.add(app)
            for relative, expected in owner.get('launchers', {}).items():
                if relative not in ('Aester.exe', 'Aester-MCP.exe', 'Aester.ico'): raise ValueError('启动器清单无效')
                path = child(root, relative)
                if path.exists():
                    if digest(path) != expected: raise ValueError('启动器文件改变')
                    plans.append((path, {'sha256': expected, 'bytes': path.stat().st_size}))
            report['state'] = 'deleting'; atomic(journal, report)
            for path, check in plans:
                safe(path)
                if not path.exists(): continue
                if path.stat().st_size != check['bytes'] or digest(path) != check['sha256']:
                    raise ValueError('执行前文件身份改变：' + str(path))
                path.unlink()
                report['deleted_files'] += 1; report['deleted_bytes'] += check['bytes']
                if progress and report['deleted_files'] % 100 == 0: progress(dict(report))
            for kind, path, data in updates:
                safe(path)
                if kind == 'json':
                    if data['files']: atomic(path, data)
                    else: path.unlink(missing_ok=True)
                else: path.unlink(missing_ok=True)
            for target in prune:
                if not target.exists(): continue
                for directory in sorted((x for x in target.rglob('*') if x.is_dir()), key=lambda p: len(p.parts), reverse=True):
                    safe(directory)
                    if not any(directory.iterdir()): directory.rmdir()
                if not any(target.iterdir()): target.rmdir()
            report['state'] = 'complete'; atomic(journal, report)
            return report
    except Exception as error:
        report['state'] = 'failed'; report['errors'].append(str(error)); atomic(journal, report)
        raise
