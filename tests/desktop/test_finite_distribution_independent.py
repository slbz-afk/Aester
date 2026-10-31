"""Independent distribution boundaries; no real install or photograph is mutated.

The tiny runtime is not executable: these tests cover routing, provenance and
failure boundaries, separately from real Siril/installer acceptance.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from starstack import components, finite_backend_assets as assets, finite_stack
from starstack import install_ownership as ownership, regions, workflow
from aester_desktop import gui
from starstack.mcp_server import WorkspaceTools
from starstack.onboarding import connection_config


@pytest.fixture(autouse=True)
def _engine_path_limit_off(monkeypatch):
    # These tests admit F/R jobs under pytest's temporary folder but never run a stack engine.  The engine path limit
    # (efficiency P1a S-8) is tested in test_engine_path_admission.py; a deep --basetemp must not refuse these.
    from starstack import siril
    monkeypatch.setattr(siril, "ENGINE_PATH_LIMIT", 1 << 15)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / 'source runtime'
    root.mkdir()
    for name, content in {'siril-cli.exe': b'NON-EXECUTABLE fixture',
                          'dependent.dll': b'fixture library'}.items():
        (root / name).write_bytes(content)
    files = {p.name: digest(p) for p in root.iterdir()}
    (root / 'runtime_manifest.json').write_text(json.dumps({'executable': 'siril-cli.exe', 'files': files}))
    for module in (assets, finite_stack):
        monkeypatch.setattr(module, 'ENGINE_SHA256', files['siril-cli.exe'])
        monkeypatch.setattr(module, 'RUNTIME_MANIFEST_SHA256', digest(root / 'runtime_manifest.json'))
    return root


@pytest.fixture
def bundle(tmp_path, runtime, monkeypatch):
    from starstack import finite_distribution
    root = tmp_path / 'reviewed bundle'
    shutil.copytree(runtime, root / 'runtime')
    (root / 'sources').mkdir()
    (root / 'legal').mkdir()
    (root / 'sources/complete-source.zip').write_bytes(b'SYNTHETIC source archive, no real claims')
    (root / 'legal/NOTICES.txt').write_bytes(b'SYNTHETIC licence notices')
    manifest = {'schema_version': 1, 'engine_sha256': assets.ENGINE_SHA256,
                'runtime_manifest_sha256': assets.RUNTIME_MANIFEST_SHA256,
                'files': {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*') if p.is_file()}}
    path = root / finite_distribution.MANIFEST
    path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(finite_distribution, 'DISTRIBUTION_MANIFEST_SHA256', digest(path))
    return root


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def windows(app, monkeypatch, tmp_path):
    monkeypatch.setattr(gui.MainWindow, 'check_backend', lambda self: None)
    for key in ('AESTER_INSTALL_ROOT', 'AESTER_APP_ROOT', 'AESTER_COMPONENT_ROOT',
                'STARSTACK_STACK_ENGINE', 'STARSTACK_SIRIL', 'STARSTACK_STAR_MODEL'):
        monkeypatch.delenv(key, raising=False)
    pool = []

    def create(settings=None):
        prefs = settings if settings is not None else QSettings(str(tmp_path / f'prefs-{len(pool)}.ini'), QSettings.Format.IniFormat)
        window = gui.MainWindow(preferences=prefs)
        window.cache_dir.setText(str(tmp_path / 'working cache'))
        window.output_dir.setText(str(tmp_path / 'photographs'))
        pool.append(window)
        return window, prefs

    yield create
    for window in pool:
        window._busy = False
        window.job_state = ''
        window._color_timer.stop()
        window.timer.stop()
        window.close()
    app.processEvents()


@pytest.fixture
def scene(tmp_path, monkeypatch):
    inputs = tmp_path / 'originals'
    inputs.mkdir()
    files = []
    for i in range(3):
        p = inputs / f'image-{i}.NEF'
        p.write_bytes(bytes([i + 11]) * 45)
        files.append(str(p))
    work = tmp_path / 'working cache'
    work.mkdir()
    refdir = work / 'reference'
    refdir.mkdir()
    linear = refdir / 'linear.fit'
    fits.writeto(linear, np.full((3, 64, 80), .15, np.float32))
    mask = refdir / 'sky.png'
    Image.new('L', (80, 64), 255).save(mask)
    reference = {'state': 'succeeded', 'source_path': files[0], 'source_sha256': digest(files[0]),
                 'linear_path': str(linear), 'linear_sha256': digest(linear),
                 'width': 80, 'height': 64, 'image_rotation': 0,
                 'reference_dir': str(refdir), 'mask_template_path': str(mask)}
    official = tmp_path / 'official.exe'
    official.write_bytes(b'non-executable official fixture')
    monkeypatch.setattr(workflow.core, 'doctor', lambda p=None: {'available': True, 'executable': str(official), 'version': '1.4.4'})
    monkeypatch.setattr(workflow, '_spawn', lambda directory: None)
    return {'files': files, 'inputs': inputs, 'work': work, 'reference': reference, 'official': official}


def test_upgrade_requires_current_application_receipt_and_keeps_both_owners(tmp_path, runtime):
    install, area = tmp_path / '自定义 Aester', tmp_path / '组件 area'
    app_v1, app_v2 = install / 'versions/v1', install / 'versions/v2'
    app_v1.mkdir(parents=True)
    app_v2.mkdir()
    identity = ownership.register_install(install, area, area / 'downloads')
    first = components.install_finite_backend(area, source=runtime, app_root=app_v1)
    assert assets.discover_finite_backend(app_v2, {}, area)['available'] is False
    second = components.install_finite_backend(area, source=runtime, app_root=app_v2)
    assert second['path'] != first['path']
    assert finite_stack.describe_engine(second['path'])['sha256'] == digest(runtime / 'siril-cli.exe')
    owner = json.loads((install / ownership.IDENTITY).read_text())
    assert owner['installation_id'] == identity['installation_id']
    assert len(owner['component_versions'][str(area)]) == 2
    foreign = tmp_path / 'foreign app'
    assert assets.discover_finite_backend(foreign, {}, area)['available'] is False
    sentinel = Path(second['path']).parent / 'my-photo.tif'
    sentinel.write_bytes(b'unknown user image')
    assert ownership.cleanup(install)['state'] == 'complete'
    assert not Path(first['path']).exists() and not Path(second['path']).exists()
    assert sentinel.read_bytes() == b'unknown user image'


def test_failed_upgrade_keeps_old_runtime_and_discovery_unavailable_for_new_version(tmp_path, runtime, monkeypatch):
    area, a, b = tmp_path / 'components', tmp_path / 'app a', tmp_path / 'app b'
    first = components.install_finite_backend(area, source=runtime, app_root=a)
    real_copy = components.shutil.copyfile

    def fail(source, destination):
        if Path(source).name == 'dependent.dll':
            raise OSError('independent ENOSPC fixture')
        return real_copy(source, destination)

    monkeypatch.setattr(components.shutil, 'copyfile', fail)
    with pytest.raises(OSError, match='ENOSPC'):
        components.install_finite_backend(area, source=runtime, app_root=b)
    assert assets.discover_finite_backend(a, {}, area)['path'] == first['path']
    assert not assets.discover_finite_backend(b, {}, area)['available']


@pytest.mark.parametrize('mode', ['official', 'custom'])
def test_new_component_discovery_never_overwrites_existing_explicit_choice(windows, tmp_path, mode):
    window, settings = windows()
    old = tmp_path / 'my custom engine.exe'
    old.write_bytes(b'existing custom')
    window.stack_engine_path.setText(str(old) if mode == 'custom' else '')
    window._sync_stack_backend_choice()
    window._save_preferences()
    discovered = str(tmp_path / 'new bundled siril-cli.exe')
    window._set_finite_backend({'available': True, 'path': discovered})
    assert window.stack_backend_choice.currentData() == mode
    assert window.stack_engine_path.text() == (str(old) if mode == 'custom' else '')
    window._save_preferences()
    reopened, _ = windows(settings)
    assert reopened.stack_backend_choice.currentData() == mode
    assert reopened.stack_engine_path.text() == (str(old) if mode == 'custom' else '')


def test_saved_finite_choice_adopts_new_verified_same_contract_path_and_persists(windows, tmp_path):
    window, settings = windows()
    old, new = tmp_path / 'old.exe', tmp_path / 'new.exe'
    old.write_bytes(b'old fixture')
    new.write_bytes(b'new fixture')
    window._set_finite_backend({'available': True, 'path': str(old)})
    window.stack_backend_choice.setCurrentIndex(window.stack_backend_choice.findData('finite'))
    window._save_preferences()
    reopened, _ = windows(settings)
    reopened._set_finite_backend({'available': True, 'path': str(new)})
    reopened._save_preferences()
    assert reopened.stack_backend_choice.currentData() == 'finite'
    assert settings.value('stack_engine_path') == str(new)


def test_disappeared_selected_finite_does_not_start_official(windows, scene, tmp_path, monkeypatch):
    window, _ = windows()
    window.input_dir.setText(str(scene['inputs']))
    missing = str(tmp_path / 'missing finite.exe')
    window._set_finite_backend({'available': True, 'path': missing})
    window.stack_backend_choice.setCurrentIndex(window.stack_backend_choice.findData('finite'))
    window._set_finite_backend({'available': False, 'reason': 'lost installed file'})
    calls = []
    monkeypatch.setattr(window, '_run_async', lambda *args: calls.append(args))
    window.start_processing()
    assert not calls
    assert window.stack_backend_choice.currentData() == 'finite'
    assert window.stack_engine_path.text() == missing


@pytest.mark.parametrize('landscape', [False, True])
def test_gui_finite_reaches_real_workflow_admission_with_effective_sky_geometry(windows, scene, runtime, monkeypatch, landscape):
    window, _ = windows()
    window.input_files = scene['files']
    window._selection_changed(preview=False)
    window._color_timer.stop()
    window._reference_pending = False
    window._reference_data = scene['reference']
    window._reference_scene = tuple(scene['files'])
    window._image_rotation = 0
    engine = str(runtime / 'siril-cli.exe')
    window._set_finite_backend({'available': True, 'path': engine})
    window.stack_backend_choice.setCurrentIndex(window.stack_backend_choice.findData('finite'))
    if landscape:
        project = regions.create_project(scene['reference'], scene['files'])
        project = regions.revise(project, scene_mode='landscape', operations=[
            {'kind': 'horizon', 'points': [[0., .6], [1., .7]]}])
        saved = regions.save_project(project, scene['work'])
        rendered = regions.build_masks(project, scene['work'] / 'initial-masks')
        window._regions_project = project
        window._regions_path = saved['project_path']
        window.sky_mask.setText(rendered['sky_mask_path'])
    pending = []
    monkeypatch.setattr(window, '_run_async', lambda *args: pending.append(args))
    window.start_processing()
    assert len(pending) == 1, window.message_label.text()
    result = pending[0][1]()
    record = workflow._read(Path(result['workflow_dir']))
    assert record['config']['stack_range_mode'] == 'finite_validity_v1'
    assert record['request']['support_mode'] == 'strict_siril_1_4_4'
    assert record['config']['stack_engine_path'] == engine
    mask = np.asarray(Image.open(record['config']['sky_mask_path']))
    assert (mask == 255).any()
    assert bool((mask == 0).any()) == landscape
    assert not record['request']['trails_enabled'] and not record['request']['stars_enabled']
    assert all(Path(p).read_bytes() == bytes([i + 11]) * 45 for i, p in enumerate(scene['files']))


@pytest.mark.parametrize('landscape', [False, True])
def test_mcp_host_configuration_routes_finite_with_shared_region_assets(scene, runtime, landscape):
    project = regions.create_project(scene['reference'], scene['files'])
    project = regions.revise(project, scene_mode='landscape' if landscape else 'sky',
        operations=[{'kind': 'horizon', 'points': [[0., .7], [1., .7]]}] if landscape else [])
    saved = regions.save_project(project, scene['work'])
    config = connection_config(scene['inputs'], scene['work'].parent / 'products', scene['work'],
        siril_path=scene['official'], stack_engine_path=runtime / 'siril-cli.exe')
    env = config['mcpServers']['aester']['env']
    service = WorkspaceTools(env['STARSTACK_INPUT_ROOT'], env['STARSTACK_OUTPUT_ROOT'],
        siril_path=env['STARSTACK_SIRIL'], cache_root=env['STARSTACK_CACHE_ROOT'],
        stack_engine_path=env['STARSTACK_STACK_ENGINE'])
    result = service.submit_workflow(input_files=scene['files'], regions_path=saved['project_path'])
    record = workflow._read(Path(result['workflow_dir']))
    assert record['request']['stack_range_mode'] == 'finite_validity_v1'
    assert record['request']['support_mode'] == 'strict_siril_1_4_4'
    assert record['config']['stack_engine_path'] == str(runtime / 'siril-cli.exe')


def test_mcp_configured_finite_requires_geometry_instead_of_silent_official_fallback(scene, runtime):
    service = WorkspaceTools(scene['inputs'], scene['work'], siril_path=scene['official'],
        stack_engine_path=runtime / 'siril-cli.exe')
    with pytest.raises(ValueError, match='天空蒙版'):
        service.submit_workflow(input_files=scene['files'], image_rotation=0)
    assert not list(scene['work'].glob('workflow-*'))


@pytest.mark.parametrize('member', ['siril-cli.exe', 'dependent.dll', 'runtime_manifest.json'])
def test_installed_runtime_damage_is_rejected_at_processing_boundary(tmp_path, runtime, member):
    installed = components.install_finite_backend(tmp_path / 'components', source=runtime, app_root=tmp_path / 'app')
    descriptor = finite_stack.describe_engine(installed['path'])
    changed = Path(installed['path']).parent / member
    changed.write_bytes(changed.read_bytes() + b'tampered')
    with pytest.raises(ValueError):
        finite_stack.verify_engine(descriptor)


@pytest.mark.parametrize('member', ['sources/complete-source.zip', 'legal/NOTICES.txt',
                                  'runtime/dependent.dll', 'distribution-manifest.json'])
@pytest.mark.parametrize('change', ['remove', 'alter'])
def test_bundle_incomplete_or_altered_is_rejected_before_any_installation(tmp_path, bundle, member, change):
    member_path = bundle / member
    if change == 'remove':
        member_path.unlink()
    else:
        member_path.write_bytes(member_path.read_bytes() + b'changed')
    area = tmp_path / 'destination components'
    with pytest.raises((ValueError, FileNotFoundError)):
        components.install_finite_distribution(area, source=bundle, app_root=tmp_path / 'app')
    assert not (area / 'versions').exists()


def test_bundle_extra_file_never_copied_or_adopted(tmp_path, bundle):
    (bundle / 'runtime/my-photo.tif').write_bytes(b'user photograph')
    with pytest.raises(ValueError, match='unlisted'):
        components.install_finite_distribution(tmp_path / 'components', source=bundle, app_root=tmp_path / 'app')


def test_nonreviewed_inventory_cannot_self_certify(tmp_path, bundle):
    path = bundle / 'distribution-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files'].pop('sources/complete-source.zip')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='reviewed'):
        components.install_finite_distribution(tmp_path / 'components', source=bundle, app_root=tmp_path / 'app')


def test_distribution_install_is_owned_idempotent_and_keeps_base_pointer(tmp_path, bundle):
    install = tmp_path / '自定义 App'
    app_root = install / 'versions/new-version'
    app_root.mkdir(parents=True)
    area = tmp_path / 'private components'
    area.mkdir()
    pointer = area / 'current.json'
    pointer.write_bytes(b'{"version":"unchanged-base-pointer"}')
    ownership.register_install(install, area, area / 'downloads')
    first = components.install_finite_distribution(area, source=bundle, app_root=app_root)
    second = components.install_finite_distribution(area, source=bundle, app_root=app_root)
    assert second == first
    assert pointer.read_bytes() == b'{"version":"unchanged-base-pointer"}'
    folder = Path(first['receipt_path']).parent
    assert (folder / 'sources/complete-source.zip').read_bytes() == (bundle / 'sources/complete-source.zip').read_bytes()
    assert (folder / 'legal/NOTICES.txt').is_file()
    assert (folder / ownership.VERSION_FILES).is_file()
    assert components.verify_installed_finite_distribution(area, app_root=app_root, expected=first) == first


@pytest.mark.parametrize('member', ['sources/complete-source.zip', 'legal/NOTICES.txt',
                                  'runtime/dependent.dll', 'finite-receipt.json'])
def test_prepared_distribution_change_rejected_before_activation(tmp_path, bundle, member):
    area, app_root = tmp_path / 'components', tmp_path / 'app'
    first = components.install_finite_distribution(area, source=bundle, app_root=app_root)
    path = Path(first['receipt_path']).parent / member
    path.write_bytes(path.read_bytes() + b'changed after prepare')
    with pytest.raises(ValueError):
        components.verify_installed_finite_distribution(area, app_root=app_root, expected=first)


def test_receipt_cannot_cross_application_or_component_root(tmp_path, bundle):
    area, app_root = tmp_path / 'components', tmp_path / 'app'
    first = components.install_finite_distribution(area, source=bundle, app_root=app_root)
    with pytest.raises(ValueError):
        components.verify_installed_finite_distribution(area, app_root=tmp_path / 'other app', expected=first)
    with pytest.raises(ValueError):
        components.verify_installed_finite_distribution(tmp_path / 'other components', app_root=app_root, expected=first)


def test_damaged_sources_repair_immutably_and_uninstall_preserves_unknown_image(tmp_path, bundle):
    install = tmp_path / 'App'
    app_root = install / 'versions/v1'
    app_root.mkdir(parents=True)
    area = tmp_path / 'components'
    ownership.register_install(install, area, area / 'downloads')
    first = components.install_finite_distribution(area, source=bundle, app_root=app_root)
    folder = Path(first['receipt_path']).parent
    changed = folder / 'sources/complete-source.zip'
    changed.write_bytes(b'damaged source archive')
    unknown = folder / 'sources/my-photo.tif'
    unknown.write_bytes(b'private image')
    second = components.install_finite_distribution(area, source=bundle, app_root=app_root)
    assert first['path'] != second['path']
    assert changed.read_bytes() == b'damaged source archive'
    assert components.verify_installed_finite_distribution(area, app_root=app_root, expected=second)['available']
    assert ownership.cleanup(install)['state'] == 'complete'
    assert unknown.read_bytes() == b'private image'
    # An altered formerly-owned file is conservatively retained, not deleted.
    assert changed.read_bytes() == b'damaged source archive'
    assert not Path(first['path']).exists() and not Path(second['path']).exists()


def test_partial_source_copy_keeps_old_base_and_never_publishes(tmp_path, bundle, monkeypatch):
    area, app_root = tmp_path / 'components', tmp_path / 'app'
    area.mkdir()
    pointer = area / 'current.json'
    pointer.write_bytes(b'preserved base')
    real_copy = components.shutil.copyfile

    def fail(source, destination):
        if Path(source).name == 'complete-source.zip':
            raise OSError('independent full disk during source copy')
        return real_copy(source, destination)

    monkeypatch.setattr(components.shutil, 'copyfile', fail)
    with pytest.raises(OSError, match='full disk'):
        components.install_finite_distribution(area, source=bundle, app_root=app_root)
    assert pointer.read_bytes() == b'preserved base'
    assert not list((area / 'versions').glob('*/finite-receipt.json'))
    assert list((area / 'staging').glob('*/failure.json'))
