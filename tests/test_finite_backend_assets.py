"""Optional backend availability must not silently select a processing mode."""
from pathlib import Path
import hashlib
import json
import shutil

import pytest

from starstack import components, finite_backend_assets as assets
from starstack import install_ownership as own


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    folder = tmp_path / 'verified-source'
    folder.mkdir()
    files = {'siril-cli.exe': b'not executable', 'dependency.dll': b'dependency'}
    for name, body in files.items():
        (folder / name).write_bytes(body)
    entries = {name: hashlib.sha256(body).hexdigest() for name, body in files.items()}
    manifest = folder / 'runtime_manifest.json'
    manifest.write_text(json.dumps({'executable': 'siril-cli.exe', 'files': entries}))
    monkeypatch.setattr(assets, 'ENGINE_SHA256', entries['siril-cli.exe'])
    monkeypatch.setattr(assets, 'RUNTIME_MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    return folder


def test_constants_match_processor_contract():
    from starstack import finite_stack
    assert assets.ENGINE_SHA256 == finite_stack.ENGINE_SHA256
    assert assets.RUNTIME_MANIFEST_SHA256 == finite_stack.RUNTIME_MANIFEST_SHA256
    assert assets.RUNTIME_AUX_MANIFEST_SHA256 == finite_stack.RUNTIME_AUX_MANIFEST_SHA256


@pytest.mark.parametrize('name', ['siril-cli.exe', 'dependency.dll', 'runtime_manifest.json'])
def test_tampered_member_is_not_available(tmp_path, runtime, name):
    (runtime / name).write_bytes((runtime / name).read_bytes() + b'changed')
    result = assets.discover_finite_backend(tmp_path / 'app', {assets.ENVIRONMENT_KEY: str(runtime / 'siril-cli.exe')})
    assert result['available'] is False and result['path'] is None
    assert result['reason']


def test_configured_backend_only_reports_availability(tmp_path, runtime):
    settings = {'siril_path': 'chosen', 'stack_engine_path': ''}
    before = dict(settings)
    result = assets.discover_finite_backend(tmp_path / 'app', {assets.ENVIRONMENT_KEY: str(runtime / 'siril-cli.exe')})
    assert result['available'] and result['source'] == 'configured'
    assert settings == before and not result['installable']


def test_local_checkout_is_auto_discoverable_without_developer_absolute_path(tmp_path, runtime):
    app = tmp_path / 'checkout'
    (app / 'src/starstack').mkdir(parents=True)
    (app / 'src/starstack/finite_stack.py').write_text('# contract')
    folder = app / 'tools/finite-siril' / assets.RUNTIME_VERSION
    shutil.copytree(runtime, folder)
    result = assets.discover_finite_backend(app, {})
    assert result['available'] and result['installable']
    assert result['install_source'] == str(folder)


def test_copy_is_immutable_idempotent_and_does_not_change_base_pointer(tmp_path, runtime):
    root = tmp_path / 'components'
    root.mkdir()
    pointer = root / 'current.json'
    pointer.write_bytes(b'{"version":"existing-official"}')
    original = {p.name: p.read_bytes() for p in runtime.iterdir()}
    app = tmp_path / 'app'
    result = components.install_finite_backend(root, source=runtime / 'siril-cli.exe', app_root=app)
    assert result['available'] and result['source'] == 'installed-addon'
    assert Path(result['path']).read_bytes() == original['siril-cli.exe']
    assert pointer.read_bytes() == b'{"version":"existing-official"}'
    assert components.install_finite_backend(root, source=runtime, app_root=app)['path'] == result['path']
    assert {p.name: p.read_bytes() for p in runtime.iterdir()} == original
    assert not assets.discover_finite_backend(tmp_path / 'other-app', {}, root)['available']
    assert len(list((root / 'versions').iterdir())) == 1


def test_corrupt_addon_repairs_to_new_directory_and_keeps_old(tmp_path, runtime):
    root, app = tmp_path / 'components', tmp_path / 'app'
    old = components.install_finite_backend(root, source=runtime, app_root=app)
    path = Path(old['path'])
    path.write_bytes(b'damaged')
    assert not assets.discover_finite_backend(app, {}, root)['available']
    new = components.install_finite_backend(root, source=runtime, app_root=app)
    assert new['available'] and new['path'] != old['path']
    assert path.read_bytes() == b'damaged'


def test_partial_copy_is_not_published_and_preserves_base(tmp_path, runtime, monkeypatch):
    root, app = tmp_path / 'components', tmp_path / 'app'
    real_copy = shutil.copyfile
    def fail(source, target):
        if Path(source).name == 'dependency.dll':
            raise OSError('simulated disk full')
        return real_copy(source, target)
    monkeypatch.setattr(components.shutil, 'copyfile', fail)
    with pytest.raises(OSError, match='disk full'):
        components.install_finite_backend(root, source=runtime, app_root=app)
    assert list((root / 'staging').glob('*/failure.json'))
    assert not assets.discover_finite_backend(app, {}, root)['available']
    assert not (root / 'current.json').exists()


def test_owned_addon_uninstalls_but_user_file_survives(tmp_path, runtime):
    install = tmp_path / 'install'
    app = install / 'versions/v1'
    app.mkdir(parents=True)
    root = tmp_path / 'components'
    own.register_install(install, root, tmp_path / 'downloads')
    result = components.install_finite_backend(root, source=runtime, app_root=app)
    engine = Path(result['path'])
    custom = engine.parent / 'my-photo.tif'
    custom.write_bytes(b'private photo')
    assert (engine.parent.parent / own.VERSION_FILES).is_file()
    outcome = own.cleanup(install)
    assert outcome['state'] == 'complete'
    assert not engine.exists()
    assert custom.read_bytes() == b'private photo'
    assert (runtime / 'siril-cli.exe').is_file()


def test_malformed_addon_receipt_is_skipped(tmp_path, runtime):
    app, root = tmp_path / 'app', tmp_path / 'components'
    folder = root / 'versions' / (assets.addon_prefix(app) + '-test')
    folder.mkdir(parents=True)
    (folder / 'finite-receipt.json').write_text('[]')
    assert not assets.discover_finite_backend(app, {}, root)['available']


def test_real_pinned_local_runtime_without_launching():
    checkout = Path(__file__).resolve().parents[1]
    # resolve(): in a git worktree tools/ is a junction to the main checkout's, which validate_runtime refuses.
    executable = (checkout / 'tools/finite-siril/finite-validity-v1').resolve() / 'siril-cli.exe'
    if not executable.exists():
        pytest.skip('Local verified runtime is not part of distributed source tests')
    result = assets.validate_runtime(executable)
    assert result['files'] == 75     # runtime_manifest.json only; P1b W7's aux files are not counted
