"""Installation boundary tests independent of the numerical stack implementation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

import starstack
from starstack import components, install_lifecycle, install_ownership

ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'components', components)
    monkeypatch.setitem(sys.modules, 'install_lifecycle', install_lifecycle)
    monkeypatch.setitem(sys.modules, 'install_ownership', install_ownership)
    worker = _module('finite_distribution_worker_test', ROOT / 'packaging/aester_p2/installer_worker.py')
    installed = tmp_path / '自定义目录 with spaces' / 'Aester'
    app = installed / 'versions' / 'new-version'
    app.mkdir(parents=True)
    area = tmp_path / 'private components'
    area.mkdir()
    state = installed / 'install-state.json'
    state.write_text('{"old":"active"}', encoding='utf-8')
    pointer = area / 'current.json'
    pointer.write_text('{"version":"old-runtime"}', encoding='utf-8')
    prepared = tmp_path / 'prepared.json'
    receipt = area / 'versions/base/receipt.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'app_root': str(app), 'python': str(area / 'fake-python.exe'), 'version': 'base'}))
    bundle = tmp_path / 'source binary package'
    bundle.mkdir()
    calls = []
    finite = {'available': True, 'path': str(area / 'versions/finite/runtime/siril-cli.exe'),
              'engine_sha256': 'e' * 64, 'distribution_sha256': 'd' * 64,
              'receipt_path': str(area / 'versions/finite/finite-receipt.json'), 'receipt_sha256': 'f' * 64}

    def prepare(root, **kwargs):
        calls.append(('base', root, kwargs))
        assert kwargs['activate'] is False
        return {'receipt_path': str(receipt), 'receipt_sha256': components._digest(receipt)}

    def install(root, **kwargs):
        calls.append(('finite', root, kwargs))
        return finite.copy()

    def verify(root, **kwargs):
        calls.append(('verify', root, kwargs))
        assert kwargs['expected'] == finite
        return finite.copy()

    monkeypatch.setattr(components, 'install_components', prepare)
    monkeypatch.setattr(components, 'install_finite_distribution', install, raising=False)
    monkeypatch.setattr(components, 'verify_installed_finite_distribution', verify, raising=False)
    monkeypatch.setattr(worker.ownership, 'record_application', lambda path: calls.append(('record', path)))
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs: calls.append(('import', args)) or subprocess.CompletedProcess(args, 0))
    base = ['worker', '--app-root', str(app), '--component-root', str(area), '--result', str(prepared)]

    def run(*options):
        monkeypatch.setattr(sys, 'argv', base + list(options))
        return worker.main()

    return types.SimpleNamespace(run=run, worker=worker, app=app, area=area, installed=installed,
        bundle=bundle, state=state, pointer=pointer, prepared=prepared, calls=calls, finite=finite)


def test_selected_bundle_is_prepared_and_rechecked_before_activation(worker_case):
    c = worker_case
    old_state, old_pointer = c.state.read_bytes(), c.pointer.read_bytes()
    assert c.run('--preflight') == 0
    assert c.run('--finite-bundle', str(c.bundle)) == 0
    assert c.state.read_bytes() == old_state
    assert c.pointer.read_bytes() == old_pointer
    assert json.loads(c.prepared.read_text())['finite_backend'] == c.finite
    selected = next(item for item in c.calls if item[0] == 'finite')
    assert selected[2] == {'source': str(c.bundle), 'app_root': c.app}
    c.calls.clear()
    assert c.run('--activate', str(c.state)) == 0
    assert [item[0] for item in c.calls] == ['verify', 'record', 'import']
    assert json.loads(c.state.read_text())['app_root'] == str(c.app)


def test_basic_install_never_enters_optional_distribution_code(worker_case, monkeypatch):
    c = worker_case
    def unexpected(*args, **kwargs):
        raise AssertionError('Basic installation must not require optional files or modules')
    monkeypatch.setattr(components, 'install_finite_distribution', unexpected)
    monkeypatch.setattr(components, 'verify_installed_finite_distribution', unexpected)
    assert c.run('--preflight') == 0
    assert c.run() == 0
    assert 'finite_backend' not in json.loads(c.prepared.read_text())
    assert c.run('--activate', str(c.state)) == 0


@pytest.mark.parametrize('failure_stage', ['copy', 'verify'])
def test_failed_finite_preparation_preserves_old_install_and_cancel_reopens_it(worker_case, monkeypatch, failure_stage):
    c = worker_case
    before = c.state.read_bytes(), c.pointer.read_bytes()
    def failed(*args, **kwargs):
        raise ValueError('Changed runtime/source package')
    target = 'install_finite_distribution' if failure_stage == 'copy' else 'verify_installed_finite_distribution'
    monkeypatch.setattr(components, target, failed)
    assert c.run('--preflight') == 0
    assert c.run('--finite-bundle', str(c.bundle)) == 1
    assert not c.prepared.exists()
    assert (c.state.read_bytes(), c.pointer.read_bytes()) == before
    assert c.run('--rollback') == 0
    with install_lifecycle.runtime_lease(c.installed):
        pass


def test_changed_addon_between_prepare_and_activation_stops_before_opening_launches(worker_case, monkeypatch):
    c = worker_case
    before = c.state.read_bytes(), c.pointer.read_bytes()
    assert c.run('--preflight') == 0
    assert c.run('--finite-bundle', str(c.bundle)) == 0
    c.calls.clear()
    def fail(*args, **kwargs):
        raise ValueError('Source archive changed after preparation')
    monkeypatch.setattr(components, 'verify_installed_finite_distribution', fail)
    assert c.run('--activate', str(c.state)) == 1
    assert c.calls == []
    assert (c.state.read_bytes(), c.pointer.read_bytes()) == before
    assert c.run('--rollback') == 0
    with install_lifecycle.runtime_lease(c.installed):
        pass


@pytest.fixture
def builder_case(tmp_path, monkeypatch):
    builder = _module('finite_distribution_builder_test', ROOT / 'scripts/build_aester_installer.py')
    source = tmp_path / 'snapshot'
    package = source / 'packaging/aester_p2'
    package.parent.mkdir(parents=True)
    shutil.copytree(ROOT / 'packaging/aester_p2', package, ignore=shutil.ignore_patterns('__pycache__'))
    modules = source / 'src/starstack'
    modules.mkdir(parents=True)
    for name in ('components.py', 'install_ownership.py', 'install_lifecycle.py', 'uninstall_options.py',
                 'finite_backend_assets.py', 'finite_distribution.py'):
        (modules / name).write_text('# isolated builder fixture\n')
    desktop_assets = source / 'src/aester_desktop/assets'   # the brand icon ships with the desktop package
    desktop_assets.mkdir(parents=True)
    shutil.copyfile(ROOT / 'src/aester_desktop/assets/aester.png', desktop_assets / 'aester.png')
    for name in ('LICENSE', 'NOTICE.md', 'THIRD_PARTY.md'):
        (source / name).write_text('Test snapshot\n')
    monkeypatch.setattr(builder, 'ROOT', source)
    seed = tmp_path / 'seed.zip'
    seed.write_bytes(b'fake-python-archive')
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'fake-model')
    compiler = tmp_path / 'ISCC.exe'
    compiler.write_bytes(b'fake-build-tool')
    data = {'python': {'filename': seed.name, 'sha256': components._digest(seed), 'size': seed.stat().st_size},
            'nox': {'sha256': components._digest(model), 'size': model.stat().st_size}}
    monkeypatch.setattr(components, '_manifest', lambda path: data)
    monkeypatch.setattr(components, '_download', lambda *args, **kwargs: seed)
    def extract(archive, destination, **kwargs):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'python.exe').write_bytes(b'fake-runtime')
    monkeypatch.setattr(components, '_extract', extract)
    compiled = []
    def run(command, **kwargs):
        if str(command[0]) == str(compiler):
            compiled.append(command)
            options = dict(arg[2:].split('=', 1) for arg in command if arg.startswith('/D'))
            target = Path(options['BuildOutput']) / (options['InstallerName'] + '.exe')
            target.write_bytes(('fake-setup-' + options['VersionId']).encode())
        else:
            for arg in command:
                if arg.startswith('/out:'):
                    Path(arg[5:]).write_bytes(b'fake-launcher')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(builder.subprocess, 'run', run)
    bundle = tmp_path / 'reviewed-bundle'
    for relative in ('runtime/siril-cli.exe', 'legal/LICENSE.txt', 'sources/siril.tar.xz', 'distribution-manifest.json'):
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative.encode())
    def validate(path):
        return {'manifest_sha256': components._digest(Path(path) / 'distribution-manifest.json')}
    monkeypatch.setattr(starstack, 'finite_distribution', types.SimpleNamespace(validate_distribution=validate), raising=False)
    return types.SimpleNamespace(builder=builder, output=tmp_path / 'builds', bundle=bundle,
        compiler=compiler, model=model, compiled=compiled)


def test_builder_embeds_complete_bundle_and_conditional_localized_notes(builder_case):
    c = builder_case
    receipt = c.builder.build(c.output, c.compiler, c.model, finite_bundle=c.bundle)
    assert receipt['finite_distribution']['included'] is True
    assert receipt['finite_distribution']['files'] == 4
    assert receipt['finite_distribution']['changes_backend_preference'] is False
    assert receipt['finite_distribution']['source_files_sha256'] == {
        'sources/siril.tar.xz': components._digest(c.bundle / 'sources/siril.tar.xz')}
    assert receipt['installer_components'] == ['base', 'finite']
    assert receipt['nox_model_bundled'] is False
    assert not (Path(receipt['stage']) / 'model').exists()
    assert receipt['default_installation_type'] == 'basic'
    assert '/DIncludeFinite=1' in c.compiled[0]
    stage = Path(receipt['stage'])
    assert (stage / 'finite-bundle/sources/siril.tar.xz').read_bytes() == b'sources/siril.tar.xz'
    assert (stage / 'bootstrap/finite_distribution.py').is_file()
    contents = json.loads((stage / 'application/PACKAGE_CONTENTS.json').read_text())
    assert len([key for key in contents['build_inputs'] if key.startswith('finite-bundle/')]) == 4
    for filename in ('INSTALL-NOTES.txt', 'INSTALL-NOTES-en.txt'):
        notes = (stage / filename).read_text(encoding='utf-8-sig')
        assert '{{FINITE_COMPONENT_NOTE}}' not in notes
        assert '改进构建' in notes if filename == 'INSTALL-NOTES.txt' else 'improved build' in notes


def test_builder_basic_has_no_finite_files_bootstrap_dependency_or_component_define(builder_case):
    c = builder_case
    receipt = c.builder.build(c.output, c.compiler, c.model)
    assert receipt['finite_distribution'] == {'included': False}
    assert receipt['installer_components'] == ['base']
    assert '/DIncludeFinite=1' not in c.compiled[0]
    stage = Path(receipt['stage'])
    assert not (stage / 'finite-bundle').exists()
    assert not (stage / 'bootstrap/finite_distribution.py').exists()
    assert 'improved build' not in (stage / 'INSTALL-NOTES-en.txt').read_text(encoding='utf-8-sig')


def test_bundle_source_changes_get_distinct_frozen_installer_identity(builder_case):
    c = builder_case
    first = c.builder.build(c.output, c.compiler, c.model, finite_bundle=c.bundle)
    repeated = c.builder.build(c.output, c.compiler, c.model, finite_bundle=c.bundle)
    assert repeated['version_id'] == first['version_id']
    assert len(c.compiled) == 1
    archive = c.bundle / 'sources/siril.tar.xz'
    archive.write_bytes(b'new corresponding source')
    second = c.builder.build(c.output, c.compiler, c.model, finite_bundle=c.bundle)
    assert first['version_id'] != second['version_id']
    assert Path(first['installer']).is_file()


def test_unreviewed_bundle_rejected_before_staging_or_compiling(builder_case, monkeypatch):
    c = builder_case
    def fail(path):
        raise ValueError('Unreviewed distribution manifest')
    monkeypatch.setattr(starstack.finite_distribution, 'validate_distribution', fail)
    with pytest.raises(ValueError, match='Unreviewed'):
        c.builder.build(c.output, c.compiler, c.model, finite_bundle=c.bundle)
    assert c.compiled == []
    assert list(c.output.glob('build-*')) == []


@pytest.fixture
def real_bundle_case(tmp_path, monkeypatch):
    from starstack import finite_backend_assets as assets, finite_distribution as distribution
    bundle = tmp_path / 'reviewed-distribution'
    runtime = bundle / 'runtime'
    runtime.mkdir(parents=True)
    (runtime / 'siril-cli.exe').write_bytes(b'synthetic PE fixture; never executed')
    (runtime / 'libsynthetic.dll').write_bytes(b'synthetic dependency fixture')
    runtime_files = {p.name: components._digest(p) for p in runtime.iterdir()}
    manifest = runtime / 'runtime_manifest.json'
    manifest.write_text(json.dumps({'executable': 'siril-cli.exe', 'files': runtime_files}))
    (bundle / 'sources').mkdir()
    (bundle / 'sources/corresponding-source.tar.xz').write_bytes(b'synthetic compressed source fixture')
    (bundle / 'legal').mkdir()
    (bundle / 'legal/LICENSE.txt').write_text('Synthetic license fixture; not a release grant')
    value = {'schema_version': 1, 'engine_sha256': runtime_files['siril-cli.exe'],
             'runtime_manifest_sha256': components._digest(manifest),
             'files': {p.relative_to(bundle).as_posix(): components._digest(p) for p in bundle.rglob('*') if p.is_file()}}
    package_manifest = bundle / 'distribution-manifest.json'
    package_manifest.write_text(json.dumps(value))
    monkeypatch.setattr(assets, 'ENGINE_SHA256', value['engine_sha256'])
    monkeypatch.setattr(assets, 'RUNTIME_MANIFEST_SHA256', value['runtime_manifest_sha256'])
    monkeypatch.setattr(distribution, 'DISTRIBUTION_MANIFEST_SHA256', components._digest(package_manifest))
    install = tmp_path / '自定义 安装'
    app = install / 'versions' / 'new-version'
    app.mkdir(parents=True)
    area = tmp_path / 'isolated-components'
    area.mkdir()
    install_ownership.register_install(install, area, area / 'downloads')
    pointer = area / 'current.json'
    pointer.write_text('{"version":"base-untouched"}')
    return types.SimpleNamespace(bundle=bundle, install=install, app=app, area=area, pointer=pointer,
        distribution=distribution, assets=assets, value=value)


def test_real_component_preserves_unknown_files_on_reuse_and_removal(real_bundle_case):
    c = real_bundle_case
    initial_pointer = c.pointer.read_bytes()
    first = components.install_finite_distribution(c.area, source=c.bundle, app_root=c.app)
    component_dir = Path(first['receipt_path']).parent
    unknown = component_dir / 'user-notes.txt'
    unknown.write_bytes(b'owned by the user')
    repeated = components.install_finite_distribution(c.area, source=c.bundle, app_root=c.app)
    assert repeated['receipt_path'] == first['receipt_path']
    assert c.pointer.read_bytes() == initial_pointer
    seal = json.loads((component_dir / install_ownership.VERSION_FILES).read_text())
    assert 'user-notes.txt' not in seal['files']
    report = install_ownership.cleanup(c.install)
    assert report['state'] == 'complete'
    assert unknown.read_bytes() == b'owned by the user'
    assert not (component_dir / 'runtime/siril-cli.exe').exists()
    assert not (component_dir / 'sources/corresponding-source.tar.xz').exists()
    assert (c.bundle / 'runtime/siril-cli.exe').is_file()
    assert c.pointer.read_bytes() == initial_pointer


def test_real_component_damaged_source_repairs_into_new_owned_version(real_bundle_case):
    c = real_bundle_case
    original = components.install_finite_distribution(c.area, source=c.bundle, app_root=c.app)
    old_dir = Path(original['receipt_path']).parent
    changed = old_dir / 'sources/corresponding-source.tar.xz'
    changed.write_bytes(b'changed after installation')
    with pytest.raises(ValueError, match='missing or changed'):
        components.verify_installed_finite_distribution(c.area, app_root=c.app, expected=original)
    repaired = components.install_finite_distribution(c.area, source=c.bundle, app_root=c.app)
    assert repaired['receipt_path'] != original['receipt_path']
    assert changed.read_bytes() == b'changed after installation'
    assert components.verify_installed_finite_distribution(c.area, app_root=c.app, expected=repaired)['available']
    report = install_ownership.cleanup(c.install)
    assert report['state'] == 'complete'
    assert changed.read_bytes() == b'changed after installation'
    assert not Path(repaired['path']).exists()


def test_real_component_cannot_be_reused_for_a_different_app_version(real_bundle_case):
    c = real_bundle_case
    first = components.install_finite_distribution(c.area, source=c.bundle, app_root=c.app)
    other = c.install / 'versions' / 'another-version'
    other.mkdir()
    with pytest.raises(ValueError, match='another application'):
        components.verify_installed_finite_distribution(c.area, app_root=other, expected=first)
    second = components.install_finite_distribution(c.area, source=c.bundle, app_root=other)
    assert second['receipt_path'] != first['receipt_path']
    assert components.verify_installed_finite_distribution(c.area, app_root=c.app, expected=first)['available']


def test_real_distribution_will_not_adopt_extra_package_files(real_bundle_case):
    c = real_bundle_case
    (c.bundle / 'extra-user-file.txt').write_bytes(b'not approved for packaging')
    with pytest.raises(ValueError, match='unlisted files'):
        components.install_finite_distribution(c.area, source=c.bundle, app_root=c.app)
    assert list((c.area / 'versions').glob('*')) == []


def test_bootstrap_loose_modules_install_and_verify_without_starstack_package(real_bundle_case, tmp_path):
    c = real_bundle_case
    bootstrap = tmp_path / 'bootstrap-only'
    bootstrap.mkdir()
    for name in ('components.py', 'finite_backend_assets.py', 'finite_distribution.py', 'install_ownership.py'):
        shutil.copyfile(ROOT / 'src/starstack' / name, bootstrap / name)
    # The synthetic hash pins are explicit test data. The child has only the
    # copied loose modules and stdlib; no product package or real engine runs.
    script = (
        'import sys,json; sys.path.insert(0,sys.argv[1]); '
        'import components,finite_backend_assets as a,finite_distribution as d; '
        'a.ENGINE_SHA256=sys.argv[5]; a.RUNTIME_MANIFEST_SHA256=sys.argv[6]; '
        'd.DISTRIBUTION_MANIFEST_SHA256=sys.argv[7]; '
        'r=components.install_finite_distribution(sys.argv[2],source=sys.argv[3],app_root=sys.argv[4]); '
        'v=components.verify_installed_finite_distribution(sys.argv[2],app_root=sys.argv[4],expected=r); '
        'print(json.dumps({"available":v["available"],"module":components.__file__}))'
    )
    completed = subprocess.run([sys.executable, '-I', '-B', '-X', 'utf8', '-c', script,
        str(bootstrap), str(c.area), str(c.bundle), str(c.app), c.assets.ENGINE_SHA256,
        c.assets.RUNTIME_MANIFEST_SHA256, c.distribution.DISTRIBUTION_MANIFEST_SHA256],
        check=True, capture_output=True, text=True, timeout=20)
    result = json.loads(completed.stdout)
    assert result['available'] is True
    assert Path(result['module']) == bootstrap / 'components.py'
