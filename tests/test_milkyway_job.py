"""Public local-Milky-Way boundary, publication and CLI/MCP routing tests."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from starstack import cli, core, milkyway, photographic, photographic_job as job
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


@pytest.fixture
def scene(tmp_path):
    h, w = 128, 192
    yy, xx = np.mgrid[:h, :w]
    light = .18 + .03*np.sin(xx/24) + .02*np.cos(yy/17)
    rgb = np.rint(light[..., None]*np.array([.95, 1., .87])*65535).astype(np.uint16)
    sky = np.full((h, w), 255, np.uint8)
    sky[100:] = 0
    sky[97:100] = 128
    protected = np.zeros((h, w), np.uint8)
    protected[20:80, 85:88] = 255
    mask = np.zeros((h, w), np.uint8)
    mask[:, 40:150] = 255
    mask[:, 35:40] = np.arange(1, 6, dtype=np.uint8)*40
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    metadata = dict(scene_linear=False, display_encoding_applied=True,
                    roworder='TOP-DOWN', color_space='encoded_sRGB')
    source = tmp_path/'source.tif'
    tifffile.imwrite(source, rgb, photometric='rgb', metadata=metadata,
                     extratags=[(34675, 'B', len(icc), icc, False)])
    paths = {}
    for name, values in [('sky', sky), ('protected', protected), ('milkyway', mask)]:
        paths[name] = tmp_path/(name+'.png')
        Image.fromarray(values).save(paths[name])
    return SimpleNamespace(source=source, output=tmp_path/'out', rgb=rgb, sky_data=sky,
                           protect_data=protected, mask_data=mask, icc=icc, metadata=metadata, **paths)


def finish(scene, **options):
    # Synthetic declaration exercises the enabled branch; it is not a real
    # human review or astronomical assertion. Missing-evidence tests are separate.
    if 'milkyway_evidence' not in options and scene.source is not None and scene.milkyway is not None:
        scene.evidence = scene.source.parent/'fixture-evidence.json'
        scene.evidence.write_text(json.dumps(dict(version='milkyway-subject-evidence-1',
            status='present', method='human-review', reviewer='synthetic test fixture',
            rationale='Synthetic enabled-path fixture, not an actual human or sky review.',
            source_sha256=core._sha(scene.source), milkyway_mask_sha256=core._sha(scene.milkyway))), encoding='utf-8')
        options['milkyway_evidence'] = scene.evidence
    return job.finish_milkyway(scene.source, scene.output, scene.sky,
                              scene.protected, scene.milkyway, **options)


def test_real_public_output_has_bound_masks_display_metadata_and_jpegs(scene):
    hashes = {str(p): core._sha(p) for p in (scene.source, scene.sky, scene.protected, scene.milkyway)}
    record = finish(scene)
    assert record['state'] == 'succeeded', record
    hashes[str(scene.evidence)] = core._sha(scene.evidence)
    assert record['source_hashes'] == hashes
    assert record['mask_mode'] == 'explicit-mask'
    assert record['operation'] == 'milkyway_local'
    assert record['presentation_only'] and not record['physical_recovery']
    assert not record['generative_model_used']
    assert record['recipe_version'] == milkyway.VERSION
    assert set(record['masks']) == {'sky', 'protected', 'milkyway'}
    assert record['masks']['milkyway'] == {'path': str(scene.milkyway), 'sha256': hashes[str(scene.milkyway)]}
    assert 'ground_strength' not in record and 'detail_strength' not in record['recipe']
    directory = Path(record['processing_directory'])
    assert directory.name.startswith('milkyway-') and directory.is_dir()
    assert json.loads((directory/'result.json').read_text(encoding='utf-8')) == record
    output = directory/'milkyway_srgb16.tif'
    with tifffile.TiffFile(output) as tf:
        page = tf.pages[0]
        assert page.dtype == np.uint16 and page.shape == scene.rgb.shape
        assert page.tags[34675].value == scene.icc
        metadata = json.loads(page.description)
        assert metadata['roworder'] == 'TOP-DOWN'
        assert metadata['color_space'] == 'encoded_sRGB'
        assert metadata['display_encoding_applied'] and not metadata['scene_linear']
        assert metadata['milkyway_mask_sha256'] == hashes[str(scene.milkyway)]
    actual = tifffile.imread(output)
    held = (scene.mask_data == 0) | (scene.sky_data != 255) | (scene.protect_data > 0)
    np.testing.assert_array_equal(actual[held], scene.rgb[held])
    assert np.any(actual[~held] != scene.rgb[~held]), 'the public path must execute the new renderer'
    for name in ('milkyway_srgb.jpg', 'preview.jpg'):
        with Image.open(directory/name) as image:
            assert image.size == scene.rgb.shape[1::-1]
            assert image.info['icc_profile'] == scene.icc
    for artifact in record['artifacts'].values():
        assert Path(artifact['path']).parent == directory
        assert artifact['sha256'] == core._sha(Path(artifact['path']))
    assert hashes == {p: core._sha(Path(p)) for p in hashes}


def test_zero_strength_and_explicit_empty_mask_are_valid_identity_outputs(scene):
    first = finish(scene, strength=0)
    Image.new('L', scene.rgb.shape[1::-1], 0).save(scene.milkyway)
    second = finish(scene)
    for record in (first, second):
        assert record['state'] == 'succeeded', record
        np.testing.assert_array_equal(tifffile.imread(record['artifacts']['milkyway_srgb16.tif']['path']), scene.rgb)
    assert first['processing_directory'] != second['processing_directory']


@pytest.mark.parametrize('kind', ['linear', 'missing_encoding', 'other_space', 'missing_icc', 'lab_icc', 'float'])
def test_wrong_source_domain_is_rejected_before_output(scene, kind):
    metadata = dict(scene.metadata)
    icc = scene.icc
    rgb = scene.rgb
    if kind == 'linear': metadata['scene_linear'] = True
    elif kind == 'missing_encoding': metadata.pop('display_encoding_applied')
    elif kind == 'other_space': metadata['color_space'] = 'AdobeRGB'
    elif kind == 'missing_icc': icc = None
    elif kind == 'lab_icc': icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB')).tobytes()
    elif kind == 'float': rgb = rgb.astype(np.float32)/65535
    tifffile.imwrite(scene.source, rgb, photometric='rgb', metadata=metadata,
                     extratags=[] if icc is None else [(34675, 'B', len(icc), icc, False)])
    with pytest.raises(ValueError): finish(scene)
    assert not scene.output.exists()


@pytest.mark.parametrize('field', ['sky', 'protected', 'milkyway'])
@pytest.mark.parametrize('kind', ['size', 'rgb', 'orientation', 'nan'])
def test_all_three_masks_are_checked_before_output(scene, field, kind):
    path = getattr(scene, field)
    if kind == 'size': Image.new('L', (191, 128)).save(path)
    elif kind == 'rgb': Image.new('RGB', (192, 128)).save(path)
    else:
        path = path.with_suffix('.tif')
        values = np.zeros((128, 192), np.float64)
        tags = []
        if kind == 'nan': values[30, 50] = np.nan
        else: tags = [(274, 'H', 1, 4, False)]
        tifffile.imwrite(path, values, extratags=tags)
        setattr(scene, field, path)
    with pytest.raises(ValueError): finish(scene)
    assert not scene.output.exists()


@pytest.mark.parametrize('field', ['sky', 'protected', 'milkyway'])
def test_no_implicit_mask_fallback(scene, field):
    setattr(scene, field, None)
    with pytest.raises(ValueError, match='显式'): finish(scene)
    assert not scene.output.exists()


@pytest.mark.parametrize('field,bad', [('brightness', True), ('brightness', .061), ('contrast', -1),
                                    ('structure', 2.1), ('color', float('nan')), ('strength', '1')])
def test_invalid_recipe_does_not_create_output(scene, field, bad):
    with pytest.raises(ValueError): finish(scene, **{field: bad})
    assert not scene.output.exists()


@pytest.mark.parametrize('field', ['source', 'sky', 'protected', 'milkyway'])
@pytest.mark.parametrize('phase', ['render', 'staged_manifest'])
def test_each_input_change_is_refused_before_publication(scene, monkeypatch, field, phase):
    changed = False

    def tamper():
        nonlocal changed
        if not changed:
            with getattr(scene, field).open('ab') as stream:
                stream.write(b'changed during processing')
            changed = True

    if phase == 'render':
        original = milkyway.render

        def render(*args, **kwargs):
            result = original(*args, **kwargs)
            tamper()
            return result

        monkeypatch.setattr(milkyway, 'render', render)
    else:
        original = core.atomic_json

        def save(path, value, **kwargs):
            result = original(path, value, **kwargs)
            if Path(path).name == 'result.json' and value['state'] == 'succeeded': tamper()
            return result

        monkeypatch.setattr(core, 'atomic_json', save)
    record = finish(scene)
    assert changed and record['state'] == 'failed', record
    assert not record['source_unchanged'] and record['artifacts'] == {}
    assert '输入在处理期间改变' in record['error']
    assert Path(record['processing_directory']).name.startswith('.partial-milkyway-')
    assert not list(scene.output.glob('milkyway-*'))
    assert json.loads((Path(record['processing_directory'])/'result.json').read_text(encoding='utf-8'))['state'] == 'failed'


def test_publish_failure_retains_stage_and_never_reports_public_artifacts(scene, monkeypatch):
    real = Path.rename

    def rename(path, target):
        if path.name.startswith('.partial-milkyway-'): raise PermissionError('injected publish failure')
        return real(path, target)

    monkeypatch.setattr(Path, 'rename', rename)
    record = finish(scene)
    assert record['state'] == 'failed' and not record['artifacts']
    assert Path(record['processing_directory']).is_dir()
    assert not list(scene.output.glob('milkyway-*'))


def test_new_entry_never_calls_old_detail_star_or_ground(scene, monkeypatch):
    from starstack import ground_finish
    for module, name in [(photographic, 'prepare'), (photographic, 'render'), (ground_finish, 'finish_ground')]:
        monkeypatch.setattr(module, name, lambda *a, **kw: pytest.fail('old finishing branch reached'))
    record = finish(scene, strength=0)
    assert record['state'] == 'succeeded', record


def test_legacy_photo_keeps_old_artifact_names_and_exposes_quality_notice(scene):
    record = job.finish_photo(scene.source, scene.output, scene.sky, scene.protected,
                              detail_strength=0, star_strength=0)
    assert record['state'] == 'succeeded', record
    assert 'photographic_srgb16.tif' in record['artifacts']
    assert '审美未通过' in record['message']
    assert record['quality_status'] == 'aesthetics_not_accepted'
    assert record['operation'] == 'experimental_all_sky'


def test_cli_routes_only_new_controls_and_requires_explicit_milkyway_mask(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(job, 'finish_milkyway', lambda *args, **kw: calls.append((args, kw)) or {'state': 'succeeded'})
    base = ['starstack', 'finish-milkyway', '--source', 'source.tif', '--output', 'out',
            '--sky-mask', 'sky.png', '--protected-mask', 'guard.png']
    monkeypatch.setattr(sys, 'argv', base)
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code == 2 and not calls
    monkeypatch.setattr(sys, 'argv', base + ['--milkyway-mask', 'galaxy.png', '--brightness', '.03',
                       '--contrast', '.4', '--structure', '.5', '--color', '.2', '--strength', '.7'])
    assert cli.main() == 0
    assert calls == [(('source.tif', 'out', 'sky.png', 'guard.png', 'galaxy.png'),
                      dict(brightness=.03, contrast=.4, structure=.5, color=.2, strength=.7))]
    capsys.readouterr()
    monkeypatch.setattr(sys, 'argv', ['starstack', 'finish-photo', '--help'])
    with pytest.raises(SystemExit): cli.main()
    assert '审美未通过' in capsys.readouterr().out


@pytest.mark.parametrize('outside_index', [0, 1, 2, 3])
def test_mcp_bounds_every_input_before_backend(tmp_path, monkeypatch, outside_index):
    root = tmp_path/'out'; root.mkdir()
    paths = [root/f'{i}.tif' for i in range(4)]
    for path in paths: path.touch()
    outside = tmp_path/'outside.tif'; outside.touch()
    paths[outside_index] = outside
    monkeypatch.setattr(job, 'finish_milkyway', lambda *a, **kw: pytest.fail('backend reached'))
    service = WorkspaceTools(tmp_path/'raw', root)
    with pytest.raises(AccessDenied): service.finish_milkyway(*map(str, paths))


def test_mcp_service_routes_new_controls_and_checks_result_paths(tmp_path, monkeypatch):
    root = tmp_path/'out'; root.mkdir()
    paths = [root/f'{i}.tif' for i in range(4)]
    for path in paths: path.touch()
    target = root/'milkyway-result'; target.mkdir()
    artifact = target/'image.tif'; artifact.touch()
    service = WorkspaceTools(tmp_path/'raw', root)
    calls = []
    result = {'state': 'succeeded', 'processing_directory': str(target),
              'artifacts': {'image.tif': {'path': str(artifact)}}}

    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(job, 'finish_milkyway', backend)
    assert service.finish_milkyway(*map(str, paths), brightness=.04) == result
    assert calls[0][0] == (str(paths[0]), str(root), *map(str, paths[1:]))
    assert calls[0][1] == dict(brightness=.04, contrast=.60, structure=.90, color=.35, strength=.60)
    result['artifacts']['image.tif']['path'] = str(paths[0])
    with pytest.raises(AccessDenied): service.finish_milkyway(*map(str, paths))


def test_registered_async_mcp_tool_has_required_masks_and_strict_parameters(tmp_path):
    pytest.importorskip('mcp')
    calls = []

    class Service(WorkspaceTools):
        def finish_milkyway(self, *args):
            calls.append(args)
            return {'state': 'succeeded'}

    server = create_server(Service(tmp_path/'raw', tmp_path/'out'))

    async def exercise():
        tools = await server.list_tools()
        tool = next(item for item in tools if item.name == 'finish_milkyway')
        schema = tool.inputSchema
        assert set(schema['required']) == {'source', 'sky_mask', 'protected_mask', 'milkyway_mask'}
        assert 'output_root' not in schema['properties']
        assert schema['properties']['brightness']['maximum'] == .06
        params = dict(source='s', sky_mask='k', protected_mask='p', milkyway_mask='m')
        await server.call_tool('finish_milkyway', params)
        assert calls == [('s', 'k', 'p', 'm', .020, .60, .90, .35, .60)]
        for change in [dict(brightness=True), dict(color='0.2'), dict(strength=float('nan')),
                       dict(contrast=1.6), dict(structure=float('inf'))]:
            with pytest.raises(Exception): await server.call_tool('finish_milkyway', params | change)
        with pytest.raises(Exception):
            await server.call_tool('finish_milkyway', {key: value for key, value in params.items() if key != 'milkyway_mask'})
        assert len(calls) == 1

    asyncio.run(exercise())
