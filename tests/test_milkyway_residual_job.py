"""Paired display residual entry: boundary, routing and publication contracts."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from starstack import cli, core, milkyway, photographic_job as job
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


def write_rgb(path, rgb, metadata, icc):
    tifffile.imwrite(path, rgb, photometric='rgb', metadata=metadata,
                     extratags=[] if icc is None else [(34675, 'B', len(icc), icc, False)])


@pytest.fixture
def paired_scene(tmp_path):
    h, w = 160, 192
    yy, xx = np.mgrid[:h, :w]
    base = np.rint((.18+.02*np.sin(xx/24)+.01*np.cos(yy/17))[..., None]
                  *np.array([.95, 1., .87])*65535).astype(np.uint16)
    rgb = base.copy()
    rgb[24:145, 86:88] += 1200
    rgb[24:145, 88:89] -= 400
    protect = np.zeros((h, w), np.uint8); protect[20:150, 84:91] = 255
    sky = np.full((h, w), 255, np.uint8); sky[150:] = 0
    mask = np.zeros((h, w), np.uint8); mask[:, 35:155] = 255
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    metadata = dict(scene_linear=False, display_encoding_applied=True,
                    roworder='TOP-DOWN', color_space='encoded_sRGB')
    source = tmp_path/'source.tif'
    write_rgb(source, rgb, metadata, icc)
    background = tmp_path/'background.tif'
    background_metadata = dict(metadata, transient_background_for_sha256=core._sha(source))
    write_rgb(background, base, background_metadata, icc)
    paths = {}
    for name, values in [('sky', sky), ('protected', protect), ('milkyway', mask)]:
        paths[name] = tmp_path/(name+'.png'); Image.fromarray(values).save(paths[name])
    return SimpleNamespace(source=source, background=background, output=tmp_path/'out',
        rgb=rgb, base=base, protect_data=protect, sky_data=sky, mask_data=mask,
        icc=icc, metadata=metadata, background_metadata=background_metadata, **paths)


def finish(scene, **options):
    if 'milkyway_evidence' not in options:
        path = scene.source.parent/'fixture-evidence.json'
        path.write_text(json.dumps(dict(version='milkyway-subject-evidence-1',
            status='present', method='human-review', reviewer='synthetic test fixture',
            rationale='Synthetic enabled-path fixture, not an actual human or sky review.',
            source_sha256=core._sha(scene.source), milkyway_mask_sha256=core._sha(scene.milkyway))), encoding='utf-8')
        options['milkyway_evidence'] = path
    return job.finish_milkyway(scene.source, scene.output, scene.sky, scene.protected,
        scene.milkyway, transient_background=scene.background, **options)


def test_actual_pair_uses_background_without_corridor_holes_and_publishes_adjusted_pair(paired_scene, monkeypatch):
    scene = paired_scene
    real_prepare, real_render, real_preserve = milkyway.prepare, milkyway.render, milkyway.preserve_transient_residual
    seen = []

    def check_input(args):
        np.testing.assert_array_equal(args[0], scene.base)
        assert not np.any(args[2]), 'the declared layer is not a hole in background estimation'

    def prepare(*args, **kwargs):
        check_input(args); seen.append('prepare')
        return real_prepare(*args, **kwargs)

    def render(*args, **kwargs):
        check_input(args); seen.append('render')
        return real_render(*args, **kwargs)

    def preserve(s, b, q, f):
        seen.append('preserve')
        result = real_preserve(s, b, q, f)
        # Test the wrapper's persistence of the adjusted Q, not just the draft.
        q[60, 86] += 1; f[60, 86] += 1
        return result

    monkeypatch.setattr(milkyway, 'prepare', prepare)
    monkeypatch.setattr(milkyway, 'render', render)
    monkeypatch.setattr(milkyway, 'preserve_transient_residual', preserve)
    record = finish(scene)
    assert record['state'] == 'succeeded', record
    assert seen == ['prepare', 'render', 'preserve']
    assert record['version'] == 'milkyway-job-3'
    assert record['protected_object'] == 'encoded_signed_residual'
    assert record['protection_mode'] == 'exact_signed_display_residual'
    assert record['physical_flux_preserved'] is False
    assert record['quality_status'] == 'aesthetics_not_accepted'
    assert record['statistics'] == record['paired_statistics']
    assert record['background_statistics_scope'] == 'B to proposed Q before shared headroom adjustment'
    assert record['pair_validation']['residual_outside_protected_pixels'] == 0
    assert record['pair_validation']['residual_pixels'] == 121*3
    assert record['pair_validation']['residual_min_code'] == -400
    assert record['pair_validation']['residual_max_code'] == 1200
    assert record['source_hashes'][str(scene.background)] == core._sha(scene.background)
    directory = Path(record['processing_directory'])
    assert json.loads((directory/'result.json').read_text(encoding='utf-8')) == record
    f = tifffile.imread(directory/'milkyway_srgb16.tif')
    q = tifffile.imread(directory/'milkyway_background_srgb16.tif')
    np.testing.assert_array_equal(f.astype(np.int32)-q.astype(np.int32),
                                  scene.rgb.astype(np.int32)-scene.base.astype(np.int32))
    assert np.any(q[24:145, 86:89] != scene.base[24:145, 86:89])
    held = (scene.mask_data == 0) | (scene.sky_data != 255)
    np.testing.assert_array_equal(f[held], scene.rgb[held])
    np.testing.assert_array_equal(q[held], scene.base[held])
    for name in ('milkyway_srgb16.tif', 'milkyway_background_srgb16.tif'):
        with tifffile.TiffFile(directory/name) as tf:
            p = tf.pages[0]; metadata = json.loads(p.description)
            assert p.shape == scene.rgb.shape and p.dtype == np.uint16
            assert p.tags[34675].value == scene.icc
            assert metadata['roworder'] == 'TOP-DOWN' and metadata['color_space'] == 'encoded_sRGB'
            assert metadata['protected_object'] == 'encoded_signed_residual'
            assert 'transient_background_for_sha256' not in metadata, 'no false reuse binding to old S'
    for name in ('milkyway_srgb.jpg', 'preview.jpg'):
        with Image.open(directory/name) as im:
            assert im.size == scene.rgb.shape[1::-1] and im.info['icc_profile'] == scene.icc
    for item in record['artifacts'].values():
        assert Path(item['path']).parent == directory
        assert item['sha256'] == core._sha(Path(item['path']))


def test_paired_zero_strength_is_identity_and_legacy_warns_about_corridor(paired_scene):
    scene = paired_scene
    record = finish(scene, strength=0)
    assert record['state'] == 'succeeded', record
    for name, expected in [('milkyway_srgb16.tif', scene.rgb), ('milkyway_background_srgb16.tif', scene.base)]:
        np.testing.assert_array_equal(tifffile.imread(record['artifacts'][name]['path']), expected)
    legacy = job.finish_milkyway(scene.source, scene.output, scene.sky, scene.protected, scene.milkyway, strength=0)
    assert legacy['state'] == 'succeeded'
    assert legacy['protection_mode'] == 'whole_pixel_hold' and '暗带' in legacy['message']
    assert 'milkyway_background_srgb16.tif' not in legacy['artifacts']


@pytest.mark.parametrize('kind', ['missing_binding', 'wrong_binding', 'list_binding', 'size', 'linear',
                                  'missing_encoding', 'wrong_space', 'float', 'no_icc', 'lab_icc', 'bottom_up',
                                  'orientation', 'planar', 'multipage'])
def test_bad_background_rejected_before_prepare_or_output(paired_scene, monkeypatch, kind):
    scene = paired_scene
    md = scene.background_metadata.copy(); rgb = scene.base; icc = scene.icc
    if kind == 'missing_binding': md.pop('transient_background_for_sha256')
    elif kind == 'wrong_binding': md['transient_background_for_sha256'] = 'f'*64
    elif kind == 'list_binding': md['transient_background_for_sha256'] = [core._sha(scene.source)]
    elif kind == 'size': rgb = rgb[:, :-1]
    elif kind == 'linear': md['scene_linear'] = True
    elif kind == 'missing_encoding': md.pop('display_encoding_applied')
    elif kind == 'wrong_space': md['color_space'] = 'AdobeRGB'
    elif kind == 'float': rgb = rgb.astype(np.float32)/65535
    elif kind == 'no_icc': icc = None
    elif kind == 'lab_icc': icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB')).tobytes()
    elif kind == 'bottom_up': md['roworder'] = 'BOTTOM-UP'
    write_rgb(scene.background, rgb, md, icc)
    if kind == 'orientation':
        tifffile.imwrite(scene.background, rgb, photometric='rgb', metadata=md,
            extratags=[(34675, 'B', len(icc), icc, False), (274, 'H', 1, 4, False)])
    elif kind == 'planar':
        tifffile.imwrite(scene.background, np.moveaxis(rgb, -1, 0), photometric='rgb',
            planarconfig='separate', metadata=md, extratags=[(34675, 'B', len(icc), icc, False)])
    elif kind == 'multipage':
        tifffile.imwrite(scene.background, rgb, photometric='rgb', append=True)
    monkeypatch.setattr(milkyway, 'prepare', lambda *a, **kw: pytest.fail('prepare reached'))
    with pytest.raises(ValueError): finish(scene)
    assert not scene.output.exists()


def test_compressed_background_read_fallback_preserves_identity(paired_scene):
    scene = paired_scene
    tifffile.imwrite(scene.background, scene.base, photometric='rgb', compression='zlib',
        metadata=scene.background_metadata, extratags=[(34675, 'B', len(scene.icc), scene.icc, False)])
    with pytest.raises(ValueError): tifffile.memmap(scene.background, mode='r')
    result = finish(scene, strength=0)
    assert result['state'] == 'succeeded', result
    np.testing.assert_array_equal(tifffile.imread(result['artifacts']['milkyway_srgb16.tif']['path']), scene.rgb)
    np.testing.assert_array_equal(tifffile.imread(result['artifacts']['milkyway_background_srgb16.tif']['path']), scene.base)


def test_source_alias_cannot_be_supplied_as_background(paired_scene):
    paired_scene.background = paired_scene.source
    with pytest.raises(ValueError, match='未绑定'): finish(paired_scene)
    assert not paired_scene.output.exists()


@pytest.mark.parametrize('row,delta', [(0, 1), (159, -1)])
def test_one_signed_code_outside_declared_layer_rejected_and_maps_closed(paired_scene, monkeypatch, row, delta):
    scene = paired_scene
    bad = scene.base.copy(); bad[row, 1, 0] = int(bad[row, 1, 0])+delta
    write_rgb(scene.background, bad, scene.background_metadata, scene.icc)
    maps = []; real = job._read_display

    def read(path):
        value = real(path); maps.append(value); return value

    monkeypatch.setattr(job, '_read_display', read)
    monkeypatch.setattr(milkyway, 'prepare', lambda *a, **kw: pytest.fail('prepare reached'))
    with pytest.raises(ValueError, match='蒙版外'): finish(scene)
    assert not scene.output.exists()
    assert maps and all(value._mmap.closed for value in maps)


@pytest.mark.parametrize('field', ['source', 'background', 'sky', 'protected', 'milkyway'])
@pytest.mark.parametrize('phase', ['render', 'preserve', 'staged_manifest'])
def test_all_paired_inputs_are_late_bound_even_after_encoding(paired_scene, monkeypatch, field, phase):
    scene = paired_scene
    changed = False

    def tamper():
        nonlocal changed
        if not changed:
            with getattr(scene, field).open('ab') as stream: stream.write(b'changed during processing')
            changed = True

    if phase == 'staged_manifest':
        original = core.atomic_json

        def save(path, value, **kwargs):
            result = original(path, value, **kwargs)
            if Path(path).name == 'result.json' and value['state'] == 'succeeded': tamper()
            return result

        monkeypatch.setattr(core, 'atomic_json', save)
    else:
        name = 'render' if phase == 'render' else 'preserve_transient_residual'
        original = getattr(milkyway, name)

        def execute(*args, **kwargs):
            result = original(*args, **kwargs); tamper(); return result

        monkeypatch.setattr(milkyway, name, execute)
    record = finish(scene)
    assert changed and record['state'] == 'failed', record
    assert not record['source_unchanged'] and record['artifacts'] == {}
    assert '输入在处理期间改变' in record['error']
    assert Path(record['processing_directory']).name.startswith('.partial-milkyway-')
    assert not list(scene.output.glob('milkyway-*'))


def test_change_during_pair_validation_rejected_before_processing(paired_scene, monkeypatch):
    real = job._pair_scope

    def validate(*args):
        result = real(*args)
        with paired_scene.background.open('ab') as stream: stream.write(b'changed during validation')
        return result

    monkeypatch.setattr(job, '_pair_scope', validate)
    monkeypatch.setattr(milkyway, 'prepare', lambda *a, **kw: pytest.fail('prepare reached'))
    with pytest.raises(ValueError, match='配对验证期间改变'): finish(paired_scene)
    assert not paired_scene.output.exists()


def test_partial_residual_failure_closes_all_maps_and_publishes_nothing(paired_scene, monkeypatch):
    maps = []; real = tifffile.memmap

    def mapped(*args, **kwargs):
        result = real(*args, **kwargs); maps.append(result); return result

    def fail(s, b, q, f):
        f[:128] = s[:128]; q[:128] = b[:128]
        raise ValueError('injected residual failure after first tile')

    monkeypatch.setattr(tifffile, 'memmap', mapped)
    monkeypatch.setattr(milkyway, 'preserve_transient_residual', fail)
    record = finish(paired_scene)
    assert record['state'] == 'failed' and record['artifacts'] == {}
    assert len(maps) >= 6 and all(value._mmap.closed for value in maps)
    assert not list(paired_scene.output.glob('milkyway-*'))


@pytest.mark.parametrize('phase', ['background_create', 'render', 'jpeg', 'rename'])
def test_paired_failures_close_open_maps_and_leave_failed_stage(paired_scene, monkeypatch, phase):
    maps = []; real_memmap = tifffile.memmap

    def mapped(*args, **kwargs):
        if phase == 'background_create' and Path(args[0]).name == 'milkyway_background_srgb16.tif':
            raise OSError('injected background create failure')
        result = real_memmap(*args, **kwargs); maps.append(result); return result

    monkeypatch.setattr(tifffile, 'memmap', mapped)

    def fail(*args, **kwargs): raise OSError('injected '+phase+' failure')

    if phase == 'render': monkeypatch.setattr(milkyway, 'render', fail)
    elif phase == 'jpeg': monkeypatch.setattr(Image.Image, 'save', fail)
    elif phase == 'rename': monkeypatch.setattr(Path, 'rename', fail)
    record = finish(paired_scene)
    assert record['state'] == 'failed' and record['artifacts'] == {}, record
    assert all(value._mmap.closed for value in maps)
    assert not list(paired_scene.output.glob('milkyway-*'))
    assert json.loads((Path(record['processing_directory'])/'result.json').read_text(encoding='utf-8'))['state'] == 'failed'


@pytest.mark.parametrize('field', ['source', 'background'])
def test_input_change_during_jpeg_save_is_refused(paired_scene, monkeypatch, field):
    real = Image.Image.save
    changed = False

    def save(*args, **kwargs):
        nonlocal changed
        result = real(*args, **kwargs)
        if not changed:
            with getattr(paired_scene, field).open('ab') as stream: stream.write(b'changed during jpeg')
            changed = True
        return result

    monkeypatch.setattr(Image.Image, 'save', save)
    record = finish(paired_scene)
    assert changed and record['state'] == 'failed' and record['artifacts'] == {}
    assert not list(paired_scene.output.glob('milkyway-*'))


def test_cli_routes_explicit_background_without_changing_other_options(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(job, 'finish_milkyway', lambda *a, **kw: calls.append((a, kw)) or {'state': 'succeeded'})
    monkeypatch.setattr(sys, 'argv', ['starstack', 'finish-milkyway', '--source', 's', '--output', 'o',
        '--sky-mask', 'k', '--protected-mask', 'p', '--milkyway-mask', 'm', '--transient-background', 'b'])
    assert cli.main() == 0
    assert calls == [(('s', 'o', 'k', 'p', 'm'), dict(brightness=.020, contrast=.60,
                     structure=.90, color=.35, strength=.60, transient_background='b'))]
    capsys.readouterr()


def test_mcp_bounds_optional_background_before_backend_and_routes_inside_path(tmp_path, monkeypatch):
    root = tmp_path/'out'; root.mkdir()
    paths = [root/f'{i}.tif' for i in range(5)]
    for path in paths: path.touch()
    outside = tmp_path/'outside.tif'; outside.touch()
    service = WorkspaceTools(tmp_path/'raw', root)
    calls = []
    (root/'job').mkdir()
    result = dict(state='succeeded', processing_directory=str(root/'job'), artifacts={})
    monkeypatch.setattr(job, 'finish_milkyway', lambda *a, **kw: calls.append((a, kw)) or result)
    with pytest.raises(AccessDenied):
        service.finish_milkyway(*map(str, paths[:4]), transient_background=str(outside))
    assert not calls
    assert service.finish_milkyway(*map(str, paths[:4]), transient_background=str(paths[4])) == result
    assert calls[0][1]['transient_background'] == str(paths[4])


def test_async_mcp_schema_routes_optional_background_and_rejects_nonpath(tmp_path):
    pytest.importorskip('mcp')
    calls = []

    class Service(WorkspaceTools):
        def finish_milkyway(self, *args, **kwargs):
            calls.append((args, kwargs)); return {'state': 'succeeded'}

    server = create_server(Service(tmp_path/'raw', tmp_path/'out'))

    async def run():
        tool = next(tool for tool in await server.list_tools() if tool.name == 'finish_milkyway')
        assert 'transient_background' in tool.inputSchema['properties']
        assert 'transient_background' not in tool.inputSchema['required']
        params = dict(source='s', sky_mask='k', protected_mask='p', milkyway_mask='m', transient_background='b')
        await server.call_tool('finish_milkyway', params)
        assert calls == [(('s', 'k', 'p', 'm', .020, .60, .90, .35, .60), {'transient_background': 'b'})]
        for bad in (True, 1, ['b'], {}):
            with pytest.raises(Exception): await server.call_tool('finish_milkyway', params | {'transient_background': bad})
        assert len(calls) == 1

    asyncio.run(run())
