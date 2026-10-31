"""P1b W7: the finite runtime's GLib spawn helpers, listed in runtime_aux_manifest.json.

GLib starts gspawn-win64-helper(-console).exe by path from its own folder, so the PE import closure in
runtime_manifest.json never listed them, and a finite Siril started the old way (inherited environment, a desktop
data folder) could not run Python and deleted a healthy venv as "unhealthy". The helpers come from the package of
the runtime's libglib-2.0-0.dll and are listed in a separate, pinned aux manifest that both validators check
(finite_backend_assets.validate_runtime for discovery/installation/distributions, finite_stack.describe_engine for
every F job). Neither contract() nor describe_engine()'s result includes it, so F-job signatures do not change.
A runtime without the aux manifest stays valid; a program GLib would start by name must be listed in a verified
manifest. The synthetic runtimes here are never executed; the real tests need the local runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from starstack import components, finite_backend_assets as assets, finite_distribution as distribution
from starstack import finite_stack as fs, process_guard, siril, stack_range as sr

REPO = Path(__file__).resolve().parents[1]
# A worktree's tools/ may be a junction; the validators refuse links, so use the real folder.
REAL_RUNTIME = (REPO / 'tools/finite-siril/finite-validity-v1').resolve()
SIGNED_BIN = (REPO / 'tools/signed-build/msys64/ucrt64/bin').resolve()
PACKAGE_V2 = REPO / 'outputs/aester-finite-distribution-20260925/package-v2'
PACKAGE_V3 = REPO / 'outputs/aester-finite-distribution-20260925/package-v3'
UNRELEASED_V3 = '6b554232668f69011130015ff633112dd24dd5192b530e267b16971843c742c4'   # 2026-10-01, no aux notice
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
HELPERS = ('gspawn-win64-helper-console.exe', 'gspawn-win64-helper.exe')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_bytes(files):
    return (json.dumps({'version': 1, 'purpose': 'synthetic fixture', 'files': files},
                       sort_keys=True, indent=2) + '\n').encode()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """A synthetic runtime (not executable) pinned for both validators, plus its aux files kept aside."""
    folder = tmp_path / 'runtime'
    folder.mkdir()
    for name, body in {'siril-cli.exe': b'NON-EXECUTABLE engine fixture', 'libglib-2.0-0.dll': b'glib'}.items():
        (folder / name).write_bytes(body)
    files = {p.name: digest(p) for p in folder.iterdir()}
    (folder / 'runtime_manifest.json').write_text(json.dumps({'executable': 'siril-cli.exe', 'files': files}))
    aux = {name: b'NON-EXECUTABLE helper fixture ' + name.encode() for name in HELPERS}
    raw = _manifest_bytes({name: hashlib.sha256(body).hexdigest() for name, body in aux.items()})
    for module in (assets, fs):
        monkeypatch.setattr(module, 'ENGINE_SHA256', files['siril-cli.exe'])
        monkeypatch.setattr(module, 'RUNTIME_MANIFEST_SHA256', digest(folder / 'runtime_manifest.json'))
        monkeypatch.setattr(module, 'RUNTIME_AUX_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
    return folder, aux, raw


def add_aux(folder, aux, raw):
    for name, body in aux.items():
        (folder / name).write_bytes(body)
    (folder / assets.RUNTIME_AUX_MANIFEST).write_bytes(raw)


def check_assets(folder):
    return assets.validate_runtime(folder / 'siril-cli.exe')


def check_engine(folder):
    return fs.describe_engine(folder / 'siril-cli.exe')


VALIDATORS = pytest.mark.parametrize('check', [check_assets, check_engine], ids=['validate_runtime', 'describe_engine'])


# --------------------------------------------------------------------------- pins

def test_aux_pin_is_the_rendered_manifest_of_the_two_glib_helpers():
    raw = assets.runtime_aux_manifest_bytes()
    assert hashlib.sha256(raw).hexdigest() == assets.RUNTIME_AUX_MANIFEST_SHA256 == fs.RUNTIME_AUX_MANIFEST_SHA256
    assert assets.RUNTIME_AUX_MANIFEST == fs.RUNTIME_AUX_MANIFEST == 'runtime_aux_manifest.json'
    assert assets.GLIB_STARTED.pattern == fs.GLIB_STARTED.pattern
    value = json.loads(raw)
    assert value['version'] == 1 and value['package'] == 'mingw-w64-ucrt-x86_64-glib2-2.90.0-1'
    # The prereg's sizes 25,607 / 24,468 B; digests as in that package's pacman mtree.
    assert value['files'] == {
        'gspawn-win64-helper-console.exe': '983e9549d1e1cd22913043e69e9cba60de38db0bf8153c5552d523a552f8ac9d',
        'gspawn-win64-helper.exe': '73a497a865b76a4d65014f8a1c34bf6e9c066d9881e5778814011cbf39671086'}
    assert 'gdbus.exe' not in value['files']


def test_f_job_identity_constants_are_unchanged():
    # The values of the base commit 8c60867: the helpers change neither the runtime identity nor the contract.
    assert fs.ENGINE_SHA256 == assets.ENGINE_SHA256 == '67681ac2eb7fb31074072c9fb059d301b2ae0a0b880d682936a8fd35553635cf'
    assert (fs.RUNTIME_MANIFEST_SHA256 == assets.RUNTIME_MANIFEST_SHA256
            == '2c732c1e45a7c4490fca116808bdfee1aa3821bf7801a0d931ef42b35f2cb2de')
    assert sr.fingerprint(fs.contract()) == '4cc4ae2d3a1d7cdabd795a342c70e6f9512bff3c2fd8a8d27272631b9ca91290'
    assert 'runtime_aux_manifest_sha256' not in fs.contract()


# --------------------------------------------------------------------------- rules

@VALIDATORS
def test_runtime_without_aux_manifest_stays_valid(runtime, check):
    folder, _, _ = runtime
    assert check(folder)


@VALIDATORS
def test_valid_aux_manifest_is_accepted_and_changes_no_result(runtime, check):
    folder, aux, raw = runtime
    before = check(folder)
    add_aux(folder, aux, raw)
    assert check(folder) == before


@VALIDATORS
@pytest.mark.parametrize('name', ['gspawn-win64-helper.exe', 'gspawn-win64-helper-console.exe',
                                  'GSPAWN-WIN64-HELPER.EXE', 'gspawn-win32-helper.exe', 'gdbus.exe'])
def test_glib_started_program_without_a_verified_manifest_is_rejected(runtime, check, name):
    folder, _, _ = runtime
    (folder / name).write_bytes(b'unverified program GLib would start')
    with pytest.raises(ValueError):
        check(folder)


@VALIDATORS
def test_unlisted_glib_program_next_to_a_valid_aux_manifest_is_rejected(runtime, check):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    (folder / 'gdbus.exe').write_bytes(b'not shipped: session-bus hazard SF-2')
    with pytest.raises(ValueError):
        check(folder)


@VALIDATORS
@pytest.mark.parametrize('change', ['alter', 'remove'])
@pytest.mark.parametrize('name', HELPERS)
def test_tampered_or_missing_helper_is_rejected(runtime, check, change, name):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    if change == 'alter':
        (folder / name).write_bytes(aux[name] + b'changed')
    else:
        (folder / name).unlink()
    with pytest.raises(ValueError):
        check(folder)


@VALIDATORS
def test_tampered_aux_manifest_is_rejected(runtime, check):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    (folder / assets.RUNTIME_AUX_MANIFEST).write_bytes(raw + b' ')
    with pytest.raises(ValueError):
        check(folder)


@VALIDATORS
def test_a_self_consistent_unpinned_aux_manifest_cannot_certify_a_helper(runtime, check):
    folder, aux, _ = runtime
    forged = dict(aux, **{'gspawn-win64-helper.exe': b'another helper'})
    add_aux(folder, forged, _manifest_bytes({n: hashlib.sha256(b).hexdigest() for n, b in forged.items()}))
    with pytest.raises(ValueError):
        check(folder)


@VALIDATORS
def test_aux_manifest_that_is_a_folder_is_rejected(runtime, check):
    folder, _, _ = runtime
    (folder / assets.RUNTIME_AUX_MANIFEST).mkdir()
    with pytest.raises(ValueError):
        check(folder)


@VALIDATORS
def test_unrelated_user_files_stay_tolerated(runtime, check):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    (folder / 'my-photo.tif').write_bytes(b'private image')
    (folder / 'notes.exe').write_bytes(b'not a name GLib starts')
    assert check(folder)


def test_old_f_job_descriptor_still_verifies_after_the_helpers_are_added(runtime):
    folder, aux, raw = runtime
    descriptor = fs.describe_engine(folder / 'siril-cli.exe')
    add_aux(folder, aux, raw)
    fs.verify_engine(descriptor)                      # an F job admitted before W7 resumes unchanged
    assert fs.describe_engine(folder / 'siril-cli.exe') == descriptor
    (folder / HELPERS[0]).write_bytes(b'tampered after admission')
    with pytest.raises(ValueError):
        fs.verify_engine(descriptor)


def test_runtime_members_lists_aux_files_only_when_present(runtime):
    folder, aux, raw = runtime
    base = ['siril-cli.exe', 'libglib-2.0-0.dll', 'runtime_manifest.json']
    assert sorted(assets.runtime_members(folder)) == sorted(base)
    add_aux(folder, aux, raw)
    members = assets.runtime_members(folder)
    assert members[2] == 'runtime_manifest.json'
    assert members[-3:] == [*HELPERS, assets.RUNTIME_AUX_MANIFEST]
    assert sorted(members) == sorted([*base, *HELPERS, assets.RUNTIME_AUX_MANIFEST])


# --------------------------------------------------------------------------- installation and distribution

def test_local_install_copies_the_helpers_and_their_manifest(tmp_path, runtime):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    result = components.install_finite_backend(tmp_path / 'components', source=folder, app_root=tmp_path / 'app')
    installed = Path(result['path']).parent
    assert result['available'] and result['source'] == 'installed-addon'
    assert {p.name for p in installed.iterdir()} == set(assets.runtime_members(folder))
    assert all((installed / name).read_bytes() == body for name, body in aux.items())
    fs.verify_engine(fs.describe_engine(result['path']))


def test_local_install_without_aux_copies_no_extra_file(tmp_path, runtime):
    folder, _, _ = runtime
    result = components.install_finite_backend(tmp_path / 'components', source=folder, app_root=tmp_path / 'app')
    assert {p.name for p in Path(result['path']).parent.iterdir()} == {
        'siril-cli.exe', 'libglib-2.0-0.dll', 'runtime_manifest.json'}


NOTICE = 'legal/RUNTIME_AUX_NOTICE.md'


def _bundle(root, folder, *, aux_key=None, omit=(), notice=NOTICE):
    """A synthetic distribution; one that ships the aux files also gets a listed runtime aux notice (FIN-1)."""
    shutil.copytree(folder, root / 'runtime')
    (root / 'sources').mkdir()
    (root / 'legal').mkdir()
    (root / 'sources/complete-source.zip').write_bytes(b'SYNTHETIC source archive')
    (root / 'legal/NOTICES.txt').write_bytes(b'SYNTHETIC notices')
    aux = aux_key is not None or (folder / assets.RUNTIME_AUX_MANIFEST).exists()
    if aux:
        (root / NOTICE).write_bytes(b'SYNTHETIC runtime aux notice')
    files = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    for name in omit:
        files.pop(name)
    value = {'schema_version': 1, 'engine_sha256': assets.ENGINE_SHA256,
             'runtime_manifest_sha256': assets.RUNTIME_MANIFEST_SHA256, 'files': files}
    if aux_key is not None:
        value['runtime_aux_manifest_sha256'] = aux_key
    if aux and notice is not None:
        value['runtime_aux_notice'] = notice
    (root / distribution.MANIFEST).write_text(json.dumps(value), encoding='utf-8')
    return digest(root / distribution.MANIFEST)


def test_distribution_with_helpers_validates_and_installs_them(tmp_path, runtime, monkeypatch):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    bundle = tmp_path / 'package-v3'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256',
                        _bundle(bundle, folder, aux_key=assets.RUNTIME_AUX_MANIFEST_SHA256))
    checked = distribution.validate_distribution(bundle)
    assert checked['manifest_sha256'] == checked['distribution_sha256'] == distribution.DISTRIBUTION_MANIFEST_SHA256
    result = components.install_finite_distribution(tmp_path / 'components', source=bundle, app_root=tmp_path / 'app')
    runtime_dir = Path(result['path']).parent
    assert all((runtime_dir / name).read_bytes() == body for name, body in aux.items())
    assert (runtime_dir / assets.RUNTIME_AUX_MANIFEST).read_bytes() == raw
    assert components.verify_installed_finite_distribution(tmp_path / 'components', app_root=tmp_path / 'app',
                                                           expected=result) == result


@pytest.mark.parametrize('omit', ['runtime/gspawn-win64-helper.exe', 'runtime/runtime_aux_manifest.json'])
def test_distribution_inventory_must_list_the_aux_files(tmp_path, runtime, monkeypatch, omit):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', _bundle(bundle, folder, omit=[omit]))
    with pytest.raises(ValueError, match='omits runtime files'):
        distribution.validate_distribution(bundle, installed=True)
    with pytest.raises(ValueError):
        distribution.validate_distribution(bundle)


def test_distribution_declaring_another_aux_pin_is_rejected(tmp_path, runtime, monkeypatch):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', _bundle(bundle, folder, aux_key='0' * 64))
    with pytest.raises(ValueError, match='verified stack engine'):
        distribution.validate_distribution(bundle)


@pytest.mark.parametrize('aux_key', [True, False], ids=['declared', 'listed-only'])
@pytest.mark.parametrize('notice', [None, 'legal/MISSING.md', 'runtime/gspawn-win64-helper.exe', 'NOTICE.md'],
                         ids=['absent', 'unlisted', 'not-legal', 'top-level'])
def test_distribution_shipping_helpers_must_name_its_listed_notice(tmp_path, runtime, monkeypatch, notice, aux_key):
    folder, aux, raw = runtime
    add_aux(folder, aux, raw)
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', _bundle(
        bundle, folder, aux_key=assets.RUNTIME_AUX_MANIFEST_SHA256 if aux_key else None, notice=notice))
    for installed in (False, True):
        with pytest.raises(ValueError, match='runtime aux notice'):
            distribution.validate_distribution(bundle, installed=installed)


def test_distribution_without_helpers_needs_no_aux_notice(tmp_path, runtime, monkeypatch):
    folder, _, _ = runtime
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', _bundle(bundle, folder))
    value = json.loads((bundle / distribution.MANIFEST).read_text(encoding='utf-8'))
    assert 'runtime_aux_notice' not in value and NOTICE not in value['files']
    assert distribution.validate_distribution(bundle)


def test_distribution_declaring_aux_without_shipping_it_is_rejected(tmp_path, runtime, monkeypatch):
    folder, _, _ = runtime
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256',
                        _bundle(bundle, folder, aux_key=assets.RUNTIME_AUX_MANIFEST_SHA256))
    with pytest.raises(ValueError, match='omits runtime files'):
        distribution.validate_distribution(bundle)


def test_earlier_release_is_readable_but_never_installed(tmp_path, runtime, monkeypatch):
    folder, _, _ = runtime
    bundle = tmp_path / 'package-v2'
    earlier = _bundle(bundle, folder)
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', 'f' * 64)
    monkeypatch.setattr(distribution, 'EARLIER_DISTRIBUTION_MANIFEST_SHA256', (earlier,))
    with pytest.raises(ValueError, match='reviewed release'):
        distribution.validate_distribution(bundle)
    checked = distribution.validate_distribution(bundle, earlier=True)
    assert checked['manifest_sha256'] == checked['distribution_sha256'] == earlier
    with pytest.raises(ValueError, match='reviewed release'):
        components.install_finite_distribution(tmp_path / 'components', source=bundle, app_root=tmp_path / 'app')
    assert not (tmp_path / 'components' / 'versions').exists()


def test_add_on_installed_from_an_earlier_release_keeps_working(tmp_path, runtime, monkeypatch):
    folder, _, _ = runtime
    bundle = tmp_path / 'package-v2'
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', _bundle(bundle, folder))
    area, app = tmp_path / 'components', tmp_path / 'app'
    old = components.install_finite_distribution(area, source=bundle, app_root=app)
    monkeypatch.setattr(distribution, 'EARLIER_DISTRIBUTION_MANIFEST_SHA256',
                        (distribution.DISTRIBUTION_MANIFEST_SHA256,))
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', 'e' * 64)   # a newer release is current
    found = assets.discover_finite_backend(app, {}, area)
    assert found['available'] and found['path'] == old['path'] and found['source'] == 'installed-addon'
    fs.verify_engine(fs.describe_engine(found['path']))
    assert distribution.validate_distribution(Path(old['receipt_path']).parent, installed=True, earlier=True)


def test_shipped_pins_name_package_v3_and_keep_package_v2_as_earlier():
    assert distribution.DISTRIBUTION_MANIFEST_SHA256 == 'df71e8ca4d827c664b721f6aba910a0e34da93b6f1a6565dfb47ef3d737e40d5'
    assert distribution.EARLIER_DISTRIBUTION_MANIFEST_SHA256 == (
        '0a1c1ffd663fec069adf106f8a52b777edceea530c216b1700d89d7aadfd25c2',)
    # The first package-v3 assembly (no runtime aux notice, FIN-1) was never released: not even readable.
    assert UNRELEASED_V3 not in (distribution.DISTRIBUTION_MANIFEST_SHA256,
                                 *distribution.EARLIER_DISTRIBUTION_MANIFEST_SHA256)


@pytest.mark.skipif(not (PACKAGE_V2 / distribution.MANIFEST).is_file(), reason='needs the local package-v2 record')
def test_real_package_v2_is_an_earlier_release():
    with pytest.raises(ValueError, match='reviewed release'):
        distribution.validate_distribution(PACKAGE_V2)
    checked = distribution.validate_distribution(PACKAGE_V2, earlier=True)
    assert checked['manifest_sha256'] == distribution.EARLIER_DISTRIBUTION_MANIFEST_SHA256[0]
    assert len(checked['files']) == 81


@pytest.mark.skipif(not (PACKAGE_V3 / distribution.MANIFEST).is_file(), reason='needs the local package-v3 record')
def test_real_package_v3_is_the_release_and_its_notice_covers_the_helpers():
    """FIN-1: package-v2's LICENSE_NOTES.md (unchanged) describes one executable plus 74 DLLs, so the helpers
    have their own notice naming them, their package and licence, and where their source and COPYING are."""
    checked = distribution.validate_distribution(PACKAGE_V3)
    assert checked['manifest_sha256'] == distribution.DISTRIBUTION_MANIFEST_SHA256
    value = json.loads((PACKAGE_V3 / distribution.MANIFEST).read_text(encoding='utf-8'))
    assert value['runtime_aux_notice'] == NOTICE and value['license_notes'] == 'legal/LICENSE_NOTES.md'
    added = sorted(['runtime/' + n for n in (*HELPERS, assets.RUNTIME_AUX_MANIFEST)] + [NOTICE])
    if (PACKAGE_V2 / distribution.MANIFEST).is_file():
        v2 = json.loads((PACKAGE_V2 / distribution.MANIFEST).read_text(encoding='utf-8'))
        assert {n: d for n, d in value['files'].items() if n in v2['files']} == v2['files']
        assert sorted(set(value['files']) - set(v2['files'])) == added
    raw = (PACKAGE_V3 / NOTICE).read_bytes()
    assert b'\r' not in raw
    notice = raw.decode('utf-8')
    for name, size in (('gspawn-win64-helper.exe', 25607), ('gspawn-win64-helper-console.exe', 24468)):
        path = PACKAGE_V3 / 'runtime' / name
        assert path.stat().st_size == size and digest(path) == assets.RUNTIME_AUX_FILES[name]
        assert f'| `runtime/{name}` | {size:,} | `{assets.RUNTIME_AUX_FILES[name]}` |' in notice
    package = 'dependencies/mingw-w64-glib2-2.90.0-1.src.tar.zst'
    copying = 'licenses/mingw-w64-ucrt-x86_64-glib2/glib2/COPYING'
    with zipfile.ZipFile(PACKAGE_V3 / value['source_archive']) as bundle:
        source, source_copying = bundle.read(package), bundle.read(copying)
    with zipfile.ZipFile(PACKAGE_V3 / 'legal/DEPENDENCY_LICENSES.zip') as legal:
        assert legal.read(copying) == source_copying
    for text in ('mingw-w64-ucrt-x86_64-glib2', '`2.90.0-1`', 'LGPL-2.1-or-later', value['source_archive'], package,
                 f'SHA-256 `{hashlib.sha256(source).hexdigest()}`', 'legal/DEPENDENCY_LICENSES.zip', copying,
                 f'SHA-256 `{hashlib.sha256(source_copying).hexdigest()}`', 'runtime/runtime_aux_manifest.json',
                 assets.RUNTIME_AUX_MANIFEST_SHA256, 'gdbus.exe', '## English', '## 中文'):
        assert text in notice, text


# --------------------------------------------------------------------------- real runtime (never the tools tree)

def _helper_source():
    """Where the pinned helper bytes are on this machine: the dev runtime once W7 is installed, else msys2."""
    for folder in (REAL_RUNTIME, SIGNED_BIN):
        if all((folder / n).is_file() and digest(folder / n) == assets.RUNTIME_AUX_FILES[n] for n in HELPERS):
            return folder
    return None


real = pytest.mark.skipif(os.name != 'nt' or not (REAL_RUNTIME / 'siril-cli.exe').is_file() or _helper_source() is None,
                          reason='needs the local finite runtime and the pinned glib2 helpers')


def _copy_runtime(target, *, helpers):
    """A copy of the real runtime's 76 pinned files, optionally with the helpers and the aux manifest."""
    target.mkdir(parents=True)
    names = [*json.loads((REAL_RUNTIME / 'runtime_manifest.json').read_text(encoding='utf-8'))['files'],
             'runtime_manifest.json']
    for name in names:
        shutil.copyfile(REAL_RUNTIME / name, target / name)
    if helpers:
        add_helpers(target)
    return target


def add_helpers(target):
    source = _helper_source()
    for name in HELPERS:
        shutil.copyfile(source / name, target / name)
    (target / assets.RUNTIME_AUX_MANIFEST).write_bytes(assets.runtime_aux_manifest_bytes())


@real
def test_real_runtime_identity_is_byte_identical_with_and_without_helpers(tmp_path):
    folder = _copy_runtime(tmp_path / 'runtime', helpers=False)
    before_engine, before_assets = fs.describe_engine(folder / 'siril-cli.exe'), assets.validate_runtime(folder / 'siril-cli.exe')
    add_helpers(folder)
    after_engine, after_assets = fs.describe_engine(folder / 'siril-cli.exe'), assets.validate_runtime(folder / 'siril-cli.exe')

    def fragment(engine):   # what core.submit_job hashes for an F job, besides inputs and the stock engine
        return json.dumps({'stack_range_contract': fs.contract(), 'stack_engine': engine}, sort_keys=True).encode()

    assert json.dumps(after_engine, sort_keys=True) == json.dumps(before_engine, sort_keys=True)
    assert fragment(after_engine) == fragment(before_engine)
    assert after_assets == before_assets and after_assets['files'] == 75
    fs.verify_engine(before_engine)
    assert sorted(p.name for p in folder.iterdir()) == sorted(assets.runtime_members(folder))


def _frames(folder, shape=(3, 64, 96)):
    import numpy as np
    from astropy.io import fits
    rng = np.random.default_rng(7)
    for number in (1, 2, 3):
        data = (.2 + rng.normal(0, .01, shape)).astype(np.float32)
        data[:, 0, :2] = np.nan
        header = fits.Header()
        header['ROWORDER'] = 'BOTTOM-UP'
        header['DATAMIN'], header['DATAMAX'] = float(np.nanmin(data)), float(np.nanmax(data))
        fits.PrimaryHDU(data, header).writeto(folder / f'q_{number:05d}.fit')


COMMANDS = ['offline', 'setext fit', 'set32bits', 'setcompress 0', 'set gui.icc_autoassignment=0',
            'set gui.icc_autoconversion=0', 'setref q 1', fs.STACK]


@pytest.fixture(scope='module')
def venv_template(tmp_path_factory):
    """A real, healthy Python venv (stdlib venv + bundled ensurepip wheel; nothing is downloaded)."""
    template = tmp_path_factory.mktemp('w7') / 'venv-template'
    base = Path(getattr(sys, '_base_executable', sys.executable))
    subprocess.run([str(base), '-m', 'venv', str(template)], check=True, timeout=300,
                   capture_output=True, creationflags=NO_WINDOW)
    assert healthy(template)
    return template


def healthy(venv):
    """Siril's own health check (siril_pythonmodule.c:1981-2083): the modules, then a working pip."""
    python = venv / 'Scripts' / 'python.exe'
    if not python.is_file():
        return False
    for args in (['-c', 'import pip, venv, ensurepip'], ['-m', 'pip', '--version']):
        if subprocess.run([str(python), *args], capture_output=True, timeout=120,
                          creationflags=NO_WINDOW).returncode != 0:
            return False
    return True


def desktop(root, template):
    """A sentinel 'desktop Siril' data folder holding a real, healthy venv."""
    shutil.copytree(template, root / 'siril' / 'venv')
    (root / 'siril' / 'sentinel.txt').write_text('desktop Siril data\n', encoding='utf-8')
    return root


def tree(root):
    return {p.relative_to(root).as_posix(): (p.is_dir(), digest(p) if p.is_file() else None)
            for p in sorted(root.rglob('*'))}


def old_launch(exe, run, xdg):
    """As builds before S-4b launched Siril: the inherited environment, so GLib's user data folder is the desktop
    folder (here a sentinel) and MSYSTEM is unset. DBUS_SESSION_BUS_ADDRESS=disabled: only keeps this test off
    the shared session bus (SF-2); Siril has no D-Bus processing path.

    Siril's venv thread runs beside the script and dies with the process, so the stack must outlast it: three
    1200x1800 frames on one thread take about 4.4 s here, the helpers' health check ends about 1.2 s after launch
    (measured 2026-10-01; a 3-frame 64x96 stack ends at 0.2 s, before the check)."""
    _frames(run, (3, 1200, 1800))
    (run / 'siril.ini').touch()
    (run / 'stack.ssf').write_text(siril.script_text([COMMANDS[0], 'setcpu 1', *COMMANDS[1:]]), encoding='utf-8')
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ('MSYSTEM', 'XDG_DATA_HOME', 'AESTER_SIRIL_PIN', 'DBUS_SESSION_BUS_ADDRESS')}
    env.update(XDG_DATA_HOME=str(xdg), DBUS_SESSION_BUS_ADDRESS='disabled:')
    assert xdg.is_relative_to(run.parent.parent)       # never a real user folder
    with (run / 'out.log').open('ab') as log, process_guard.guarded_process(
            [str(exe), '-i', str(run / 'siril.ini'), '-d', str(run), '-s', str(run / 'stack.ssf')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
            env=env) as process:
        returncode = process.wait(timeout=180)
    return returncode, process.accounting or {}, (run / 'out.log').read_bytes().decode('utf-8', errors='replace')


@real
def test_real_old_launch_deletes_a_healthy_venv_only_without_the_helpers_and_aester_touches_nothing(
        tmp_path, monkeypatch, venv_template):
    """W7-1. Process counts are the guard's job accounting (Siril plus its descendants)."""
    report = {}
    finite = _copy_runtime(tmp_path / 'runtime', helpers=False)

    # 1. Without the helpers GLib cannot start Python: the health check fails and Siril deletes the venv.
    xdg = desktop(tmp_path / 'a' / 'desktop', venv_template)
    (tmp_path / 'a' / 'run').mkdir()
    returncode, accounting, log = old_launch(finite / 'siril-cli.exe', tmp_path / 'a' / 'run', xdg)
    report['without_helpers'] = {'rc': returncode, 'processes': accounting.get('descendant_processes'),
                                 'entries': sorted(p.name for p in (xdg / 'siril').iterdir())}
    assert returncode == 0, log
    assert 'Venv exists but is unhealthy' in log
    assert not (xdg / 'siril' / 'venv').exists()

    # 2. With the verified helpers the same launch finds the venv healthy and keeps it.
    add_helpers(finite)
    assert fs.describe_engine(finite / 'siril-cli.exe') and assets.validate_runtime(finite / 'siril-cli.exe')
    xdg = desktop(tmp_path / 'b' / 'desktop', venv_template)
    (tmp_path / 'b' / 'run').mkdir()
    returncode, accounting, log = old_launch(finite / 'siril-cli.exe', tmp_path / 'b' / 'run', xdg)
    report['with_helpers'] = {'rc': returncode, 'processes': accounting.get('descendant_processes'),
                              'entries': sorted(p.name for p in (xdg / 'siril').iterdir())}
    assert returncode == 0, log
    assert 'unhealthy' not in log and 'Failed to execute helper program' not in log
    assert 'Preparing python virtual environment' in log          # the health check passed
    assert healthy(xdg / 'siril' / 'venv')
    assert accounting.get('descendant_processes', 0) > 3           # Python ran through the helpers

    # 3. The Aester launch (siril.run_script, its own environment block) leaves the desktop folder untouched.
    xdg = desktop(tmp_path / 'c' / 'desktop', venv_template)
    run = tmp_path / 'c' / 'run'
    run.mkdir()
    _frames(run)
    monkeypatch.setenv('XDG_DATA_HOME', str(xdg))
    monkeypatch.delenv('MSYSTEM', raising=False)
    monkeypatch.delenv(siril.SIRIL_PIN_ENV, raising=False)
    monkeypatch.setenv('DBUS_SESSION_BUS_ADDRESS', 'disabled:')     # S-4d's value, before and after it lands
    before = tree(xdg)
    processes = []
    original = siril.guarded_process

    @contextmanager
    def observing(args, **kwargs):
        with original(args, **kwargs) as process:
            processes.append(process)
            yield process

    monkeypatch.setattr(siril, 'guarded_process', observing)
    siril.run_script(str(finite / 'siril-cli.exe'), run, 'stack', COMMANDS, lambda: None)
    accounting = processes[0].accounting or {}
    report['aester_launch'] = {'processes': accounting.get('descendant_processes')}
    assert tree(xdg) == before
    assert not (run / 'siril').exists()
    assert accounting.get('descendant_processes', 99) <= 3, accounting
    assert all((run / name).is_file() for name in ('finite_master.fit', 'finite_master_nused.fit',
                                                    'finite_master_nvalid.fit'))
    print('W7-1', json.dumps(report, sort_keys=True))
