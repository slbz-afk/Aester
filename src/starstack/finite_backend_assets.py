"""Pinned optional stack backend discovery; availability never enables a mode."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

ENGINE_SHA256 = '67681ac2eb7fb31074072c9fb059d301b2ae0a0b880d682936a8fd35553635cf'
RUNTIME_MANIFEST_SHA256 = '2c732c1e45a7c4490fca116808bdfee1aa3821bf7801a0d931ef42b35f2cb2de'
ENVIRONMENT_KEY = 'AESTER_FINITE_SIRIL'
RUNTIME_VERSION = 'finite-validity-v1'

# P1b W7 (docs/EFFICIENCY_P1B_PREREG_20261001.md section 8): GLib starts its spawn helpers by path from its own
# folder at run time, so the PE import closure in runtime_manifest.json cannot list them. They are listed in
# runtime_aux_manifest.json, pinned here and in finite_stack (same value), and are part of neither the runtime
# identity above nor the F-job contract. A runtime without the aux manifest stays valid; a program GLib would
# start by name (GLIB_STARTED) must be listed in a verified manifest, or the runtime is rejected.
RUNTIME_AUX_MANIFEST = 'runtime_aux_manifest.json'
RUNTIME_AUX_MANIFEST_SHA256 = '7116a60af17aea36bc3b11ce81caf040b552d71d4ee5905bbabbdab17f4ed941'
RUNTIME_AUX_PACKAGE = 'mingw-w64-ucrt-x86_64-glib2-2.90.0-1'   # the package of the runtime's libglib-2.0-0.dll
RUNTIME_AUX_FILES = {   # sha256 as in that package's pacman mtree
    'gspawn-win64-helper-console.exe': '983e9549d1e1cd22913043e69e9cba60de38db0bf8153c5552d523a552f8ac9d',
    'gspawn-win64-helper.exe': '73a497a865b76a4d65014f8a1c34bf6e9c066d9881e5778814011cbf39671086',
}
GLIB_STARTED = re.compile(r'(gspawn-win(32|64)-helper(-console)?|gdbus)\.exe', re.IGNORECASE)


def runtime_aux_manifest_bytes():
    """The exact bytes of runtime_aux_manifest.json (sha256 = RUNTIME_AUX_MANIFEST_SHA256), for the build record."""
    value = {'version': 1, 'package': RUNTIME_AUX_PACKAGE, 'files': dict(RUNTIME_AUX_FILES), 'purpose': (
        "GLib spawn helpers from the package of this runtime's libglib-2.0-0.dll. GLib starts them by path from "
        "its own folder at run time (g_spawn_*), so the PE import closure in runtime_manifest.json cannot list "
        "them. They are pinned here, outside the F-job contract; gdbus.exe is deliberately not included.")}
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _regular_file(path):
    # Reject directory junctions as well as final-file links. No supplied
    # executable is ever launched during discovery or installation.
    for member in (path, *path.parents):
        try:
            info = member.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            return False
    return path.is_file()


def validate_runtime(executable):
    """Read only the pinned flat runtime. Do not execute a user-supplied file."""
    path = Path(executable).expanduser().absolute()
    if path.name != 'siril-cli.exe' or not _regular_file(path):
        raise ValueError('The verified Aester stack executable is missing')
    manifest = path.parent / 'runtime_manifest.json'
    if not _regular_file(manifest) or _digest(path) != ENGINE_SHA256 or _digest(manifest) != RUNTIME_MANIFEST_SHA256:
        raise ValueError('The Aester stack engine or its runtime manifest differs from the verified build')
    value = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not isinstance(value.get('files'), dict):
        raise ValueError('Invalid finite backend manifest')
    files = value['files']
    if value.get('executable') != path.name or files.get(path.name) != ENGINE_SHA256:
        raise ValueError('Invalid finite backend manifest')
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid finite backend member')
        member = path.parent / name
        if not _regular_file(member) or _digest(member) != digest:
            raise ValueError('A verified finite backend file is missing or changed: ' + name)
    _validate_aux(path.parent, files)
    # The result does not report the aux files: it reaches the GUI and installer results, and a runtime that passes
    # holds no unverified GLib-started program either way.
    return {'path': str(path), 'version': RUNTIME_VERSION, 'engine_sha256': ENGINE_SHA256,
            'runtime_manifest_sha256': RUNTIME_MANIFEST_SHA256, 'files': len(files)}


def _validate_aux(runtime, listed):
    """runtime_aux_manifest.json is absent, or it is the pinned file and every file it lists is present and
    unchanged. Either way, each program GLib would start by name from this folder must be listed in a verified
    manifest: an unverified executable is never left where GLib would run it."""
    aux_path = runtime / RUNTIME_AUX_MANIFEST
    aux = {}
    if os.path.lexists(aux_path):
        if not _regular_file(aux_path) or _digest(aux_path) != RUNTIME_AUX_MANIFEST_SHA256:
            raise ValueError('The finite backend aux manifest differs from the verified build')
        value = json.loads(aux_path.read_text(encoding='utf-8'))
        aux = value.get('files') if isinstance(value, dict) else None
        if not isinstance(aux, dict) or not aux:
            raise ValueError('Invalid finite backend aux manifest')
        for name, digest in aux.items():
            if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..') or name in listed
                    or name in ('runtime_manifest.json', RUNTIME_AUX_MANIFEST)):
                raise ValueError('Invalid finite backend member')
            member = runtime / name
            if not _regular_file(member) or _digest(member) != digest:
                raise ValueError('A verified finite backend file is missing or changed: ' + name)
    covered = {name.lower() for name in (*listed, *aux)}
    for name in os.listdir(runtime):
        if GLIB_STARTED.fullmatch(name) and name.lower() not in covered:
            raise ValueError('An unverified program that GLib would start is in the finite backend: ' + name)


def runtime_members(directory):
    """The files of a verified runtime folder, in copy order: runtime_manifest.json's files and that manifest, then,
    when present, runtime_aux_manifest.json's files and that manifest. Call validate_runtime first."""
    folder = Path(directory)
    names = [*json.loads((folder / 'runtime_manifest.json').read_text(encoding='utf-8'))['files'],
             'runtime_manifest.json']
    if os.path.lexists(folder / RUNTIME_AUX_MANIFEST):
        names += [*json.loads((folder / RUNTIME_AUX_MANIFEST).read_text(encoding='utf-8'))['files'],
                  RUNTIME_AUX_MANIFEST]
    return names


def addon_prefix(app_root):
    return 'finite-' + ENGINE_SHA256[:12] + '-' + hashlib.sha256(str(Path(app_root).absolute()).encode()).hexdigest()[:10]


def discover_finite_backend(app_root=None, environ=None, component_root=None):
    """Report verified choices without changing settings or processing defaults.

    Managed add-ons must name this exact application root. A shared component
    current.json pointer is deliberately not authoritative for this lookup.
    """
    env = os.environ if environ is None else environ
    app = Path(app_root or env.get('AESTER_APP_ROOT') or Path(__file__).resolve().parents[2]).absolute()
    area = component_root or env.get('AESTER_COMPONENT_ROOT')
    candidates = []
    if env.get(ENVIRONMENT_KEY):
        candidates.append((Path(env[ENVIRONMENT_KEY]), 'configured'))
    if area:
        area = Path(area).absolute()
        # These bounded names are dedicated immutable add-ons, never the base
        # runtime pointer or a version belonging to another application.
        receipts = sorted((area / 'versions').glob(addon_prefix(app) + '*/finite-receipt.json'), reverse=True)
        if len(receipts) > 100:
            receipts = []
        for receipt_path in receipts:
            try:
                value = json.loads(receipt_path.read_text(encoding='utf-8'))
                if not isinstance(value, dict) or value.get('app_root') != str(app) or value.get('version') != RUNTIME_VERSION:
                    continue
                candidate = receipt_path.parent / 'runtime/siril-cli.exe'
                if value.get('engine_sha256') != ENGINE_SHA256:
                    continue
                candidates.append((candidate, 'installed-addon'))
            except (OSError, ValueError, TypeError):
                continue
    # The checkout has a local toolchain; a release does not search arbitrary
    # drives or remember the developer's absolute project path.
    local = app / 'tools/finite-siril' / RUNTIME_VERSION / 'siril-cli.exe'
    if (app / 'src/starstack/finite_stack.py').is_file() and local.is_file():
        candidates.append((local, 'local-development'))
    failures = []
    for executable, source in candidates:
        try:
            result = validate_runtime(executable)
            distribution_path = None
            if source == 'installed-addon':
                receipt = executable.parent.parent / 'finite-receipt.json'
                value = json.loads(receipt.read_text(encoding='utf-8'))
                if value.get('source_kind') == 'reviewed-distribution':
                    distribution_path = str(receipt.parent)
            return {'available': True, **result, 'source': source, 'reason': '',
                    'distribution_path': distribution_path,
                    'installable': source == 'local-development',
                    'install_source': str(executable.parent) if source == 'local-development' else None}
        except (OSError, ValueError, TypeError) as error:
            failures.append(str(error))
    return {'available': False, 'path': None, 'version': RUNTIME_VERSION,
            'source': None, 'reason': '; '.join(failures) or 'The optional Aester stack backend is not installed',
            'installable': False, 'install_source': None}
