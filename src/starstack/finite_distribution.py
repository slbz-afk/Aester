"""Read-only verification of the optional, self-contained Siril distribution.

This module is also used as a loose stdlib-only installer bootstrap module.
The reviewed manifest binds binaries, corresponding source and legal notices;
an arbitrary locally written manifest is never treated as a release approval.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

try:
    from . import finite_backend_assets as assets
except ImportError:
    import finite_backend_assets as assets


# Reviewed 2026-10-02 (P1b W7, review FIN-1): package-v3 = package-v2 plus the two
# GLib spawn helpers, runtime/runtime_aux_manifest.json (finite_backend_assets
# RUNTIME_AUX_MANIFEST_SHA256) and legal/RUNTIME_AUX_NOTICE.md, which names the
# helpers, their package (glib2 2.90.0-1, LGPL-2.1-or-later) and where their source
# and licence are. Engine, runtime manifest, corresponding source and package-v2's
# four legal files are package-v2's bytes; the helpers' glib2 source package and
# licence were already in it. The 2026-10-01 assembly without the notice (6b554232...)
# was never released. The only release accepted for new installs and builds.
DISTRIBUTION_MANIFEST_SHA256 = "df71e8ca4d827c664b721f6aba910a0e34da93b6f1a6565dfb47ef3d737e40d5"
# Earlier reviewed releases, readable only with validate_distribution(earlier=True).
# package-v2, reviewed 2026-09-25: fixed runtime, 52 exact dependency source packages,
# 1,914 frozen Siril source files, complete patch, build recipes and notices. Its
# installed add-ons keep working: discovery and every F job check the runtime itself
# (same engine and runtime manifest), never the distribution manifest.
EARLIER_DISTRIBUTION_MANIFEST_SHA256 = ("0a1c1ffd663fec069adf106f8a52b777edceea530c216b1700d89d7aadfd25c2",)
MANIFEST = "distribution-manifest.json"


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular_path(path):
    path = Path(path).expanduser().absolute()
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Distribution paths must not contain links or junctions')
    return path


def member(root, name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Invalid distribution member')
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or relative.as_posix() != name
            or relative.parts[0] not in ('runtime', 'legal', 'sources')
            or any(p in ('', '.', '..') or ':' in p or p[-1:] in (' ', '.') for p in relative.parts)):
        raise ValueError('Invalid distribution member')
    path = regular_path(Path(root).joinpath(*relative.parts))
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing distribution member: ' + name)
    return path


def validate_distribution(directory, *, installed=False, earlier=False):
    """Verify pinned bytes, including source archives, without executing code.

    Installed directories can contain ownership records and unknown user files;
    these are never imported from a package or adopted for later deletion.
    ``earlier`` also accepts an earlier reviewed release (read-only checks); the
    result names the manifest actually found.
    """
    root = regular_path(directory)
    manifest = regular_path(root / MANIFEST)
    accepted = (DISTRIBUTION_MANIFEST_SHA256, *(EARLIER_DISTRIBUTION_MANIFEST_SHA256 if earlier else ()))
    found = digest(manifest) if manifest.is_file() else None
    if not all(re.fullmatch(r'[a-f0-9]{64}', value) for value in accepted) or found not in accepted:
        raise ValueError('The Aester distribution manifest is not the reviewed release')
    value = json.loads(manifest.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('schema_version') != 1
            or value.get('engine_sha256') != assets.ENGINE_SHA256
            or value.get('runtime_manifest_sha256') != assets.RUNTIME_MANIFEST_SHA256
            or value.get('runtime_aux_manifest_sha256', assets.RUNTIME_AUX_MANIFEST_SHA256)
            != assets.RUNTIME_AUX_MANIFEST_SHA256):
        raise ValueError('The distribution does not match the verified stack engine')
    files = value.get('files')
    if (not isinstance(files, dict) or not files
            or not any(n.startswith('sources/') for n in files)
            or not any(n.startswith('legal/') for n in files)):
        raise ValueError('The distribution must include corresponding source and notices')
    # The runtime's GLib helpers are not covered by the 74-DLL notes of package-v2, so a
    # distribution that ships them names its own listed notice (FIN-1).
    aux_listed = {f'runtime/{name}' for name in (*assets.RUNTIME_AUX_FILES, assets.RUNTIME_AUX_MANIFEST)} & files.keys()
    notice = value.get('runtime_aux_notice')
    if (('runtime_aux_manifest_sha256' in value or aux_listed)
            and (not isinstance(notice, str) or not notice.startswith('legal/') or notice not in files)):
        raise ValueError('The distribution must include the runtime aux notice')
    size = 0
    for name, expected in files.items():
        path = member(root, name)
        if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected) or digest(path) != expected:
            raise ValueError('A distribution file is missing or changed: ' + name)
        size += path.stat().st_size
    runtime = assets.validate_runtime(root / 'runtime/siril-cli.exe')
    members = {f'runtime/{name}' for name in assets.runtime_members(root / 'runtime')}
    if 'runtime_aux_manifest_sha256' in value:
        members.add('runtime/' + assets.RUNTIME_AUX_MANIFEST)
    if not members <= files.keys():
        raise ValueError('The distribution inventory omits runtime files')
    if not installed:
        actual = set()
        for path in root.rglob('*'):
            regular_path(path)
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        if actual != set(files) | {MANIFEST}:
            raise ValueError('The distribution includes unlisted files')
    return {'manifest_sha256': found,
            'distribution_sha256': found,
            'files': dict(files), 'bytes': size + manifest.stat().st_size,
            'runtime': runtime}
