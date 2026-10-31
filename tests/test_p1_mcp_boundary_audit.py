"""Independent P1 adapter boundary audit: guarded read order and SDK dispatch.

Only synthetic local sources; these tests never start an imaging worker.
"""
import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from starstack import regions
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


@pytest.fixture
def scene(tmp_path):
    inp, out = tmp_path / 'input', tmp_path / 'output'
    inp.mkdir(); out.mkdir()
    files = []
    for index in range(3):
        path = inp / f'{index}.NEF'
        path.write_bytes(f'original {index}'.encode())
        files.append(str(path))
    directory = out / 'reference-audit'
    directory.mkdir()
    linear = directory / 'reference.fit'
    fits.writeto(linear, np.ones((3, 16, 20), dtype=np.float32))
    reference = {'state': 'succeeded', 'source_path': files[0], 'source_sha256': regions._sha(files[0]),
                 'linear_path': str(linear), 'linear_sha256': regions._sha(linear),
                 'width': 20, 'height': 16, 'reference_dir': str(directory)}
    (directory / 'reference.json').write_text(json.dumps(reference), encoding='utf8')
    engine = tmp_path / 'host-siril.exe'
    engine.write_bytes(b'configured host executable, never run')
    service = WorkspaceTools(inp, out, engine)
    project = service.create_regions(str(directory), files, scene_mode='sky')
    return service, directory, files, reference, project


@pytest.mark.parametrize('kind', ['frame', 'source', 'linear'])
def test_nested_region_external_paths_rejected_before_any_hash(scene, monkeypatch, kind):
    service, _, _, _, project = scene
    record = copy.deepcopy(project)
    external = str(service.output_root.parent / 'private.NEF')
    if kind == 'frame':
        record['source_binding']['input_files'][1]['path'] = external
    else:
        record['source_binding']['reference'][kind + '_path'] = external
    Path(project['project_path']).write_text(json.dumps(regions._seal(record)), encoding='utf8')
    monkeypatch.setattr(regions, '_sha', lambda path: pytest.fail(f'No hash should run before path admission: {path}'))
    # The source checks hash in batches (P1 hashing); they are the same guarded reads.
    for batch in ('_sha_many', '_first_changed'):
        monkeypatch.setattr(regions, batch, lambda paths, *rest: pytest.fail(f'No hash should run before path admission: {paths}'))
    with pytest.raises(AccessDenied):
        service.get_regions(project['project_path'])


def test_region_get_uses_admitted_snapshot_without_second_read(scene, monkeypatch):
    """A replacement between pre-check and shared validation must not read private bytes."""
    service, _, _, _, project = scene
    private = service.output_root.parent / 'private.NEF'
    private.write_bytes(b'outside host input root')
    forged = copy.deepcopy(project)
    forged['source_binding']['input_files'][1] = {'path': str(private), 'sha256': regions._sha(private)}
    forged = regions._seal(forged)
    original_load, original_sha = regions.load_project, regions._sha
    original_many, original_changed = regions._sha_many, regions._first_changed
    external_reads = []
    def changed_between_reads(path, **kwargs):
        Path(path).write_text(json.dumps(forged), encoding='utf8')
        return original_load(path, **kwargs)
    def watched_hash(path):
        if Path(path) == private:
            external_reads.append(str(path))
        return original_sha(path)
    def watched_batch(original):
        def batch(paths, *rest):
            external_reads.extend(str(path) for path in paths if Path(path) == private)
            return original(paths, *rest)
        return batch
    monkeypatch.setattr(regions, 'load_project', changed_between_reads)
    monkeypatch.setattr(regions, '_sha', watched_hash)
    monkeypatch.setattr(regions, '_sha_many', watched_batch(original_many))
    monkeypatch.setattr(regions, '_first_changed', watched_batch(original_changed))
    result = service.get_regions(project['project_path'])
    assert not external_reads, 'checked JSON was reread and an outside source was hashed'
    assert result['revision'] == project['revision']


def _workflow(service, *, nested=True):
    root = service.output_root / 'saved-session' if nested else service.output_root
    directory = root / 'workflow-audit'
    directory.mkdir(parents=True)
    config = {'input_dir': str(service.input_root), 'output_root': str(root),
              'siril_path': service.siril_path, 'stack_engine_path': None, 'star_model_path': None, 'sky_mask_path': None}
    source = next(service.input_root.glob('*.NEF'))
    record = {'workflow_dir': str(directory), 'config': config,
              'request': {'config': copy.deepcopy(config), 'input_dir': str(service.input_root),
                          'input_files': [str(source)], 'files': [{'path': str(source)}], 'assets': {}},
              'artifacts': [], 'checkpoints': {}}
    (directory / 'workflow.json').write_text(json.dumps(record), encoding='utf8')
    return directory, record


def test_nested_workflow_allowed_but_external_child_refused_before_load(scene, monkeypatch):
    service = scene[0]
    directory, record = _workflow(service)
    assert service._workflow(str(directory))[0] == directory
    record['stack_job_dir'] = str(service.output_root.parent / 'private' / 'job-secret')
    (directory / 'workflow.json').write_text(json.dumps(record), encoding='utf8')
    from starstack import core
    monkeypatch.setattr(core, 'load_job', lambda *_: pytest.fail('external job must never be loaded'))
    with pytest.raises(AccessDenied):
        service._workflow(str(directory))


@pytest.mark.parametrize('foreign', [True, False])
def test_review_binding_refused_before_service_dispatch(scene, monkeypatch, foreign):
    service = scene[0]
    parent = service.output_root / 'workflow-synthetic'
    child = parent / 'stack' / 'job-synthetic'
    (child / 'review').mkdir(parents=True)
    path = child / 'review' / 'draft-a.json'
    path.write_text(json.dumps({'job_dir': str(service.output_root.parent / 'private') if foreign else str(child),
                               'session_path': str(path) if foreign else str(path.with_name('other.json'))}), encoding='utf8')
    # Isolate new adapter's session admission; parent and child validation have
    # separate tests. No shared service read/write may begin after mismatch.
    monkeypatch.setattr(service, '_workflow', lambda value: (parent, {'stack_job_dir': str(child)}))
    monkeypatch.setattr(service, '_job', lambda value: (child, {}))
    from starstack import trail_review_session
    monkeypatch.setattr(trail_review_session, 'save_review', lambda *a, **k: pytest.fail('must not save'))
    with pytest.raises(AccessDenied):
        service.save_trail_review(str(parent), str(path), {}, 'draft-before')


def test_official_sdk_new_tools_schemas_and_safe_calls(scene):
    service, directory, files, _, project = scene
    server = create_server(service)
    async def exercise():
        listing = {tool.name: tool for tool in await server.list_tools()}
        for name in ['create_regions', 'update_regions', 'get_regions', 'preview_regions', 'open_trail_review',
                     'save_trail_review', 'trail_candidate_context', 'apply_trail_review']:
            assert name in listing
            assert not {'output_root', 'siril_path', 'stack_engine_path', 'star_model_path'} & listing[name].inputSchema['properties'].keys()
        assert 'expected_draft_revision' in listing['save_trail_review'].inputSchema['required']
        assert 'expected_draft_revision' in listing['apply_trail_review'].inputSchema['required']
        response = await server.call_tool('get_regions', {'regions_path': project['project_path']})
        assert str(project['revision']) in str(response)
        before = sorted(p.name for p in service.output_root.iterdir())
        for arguments in [
            {'reference_dir': str(directory), 'input_files': files, 'operations': ['not a dict']},
            {'reference_dir': str(directory), 'input_files': files, 'milkyway': {'status': 'present', 'ellipse': [.5,.5,.2,.2]}, 'user_confirmation': ''},
        ]:
            with pytest.raises(Exception):
                await server.call_tool('create_regions', arguments)
        assert sorted(p.name for p in service.output_root.iterdir()) == before
    asyncio.run(exercise())


def test_region_preview_sdk_image_is_explicit_and_bound(scene):
    from PIL import Image
    service, directory, _, reference, project = scene
    preview = directory / 'preview.png'
    Image.new('RGB', (20, 16), (24, 42, 60)).save(preview)
    reference.update(preview_path=str(preview), preview_sha256=regions._sha(preview))
    (directory / 'reference.json').write_text(json.dumps(reference), encoding='utf8')
    server = create_server(service)
    async def exercise():
        args = {'regions_path': project['project_path'], 'reference_dir': str(directory)}
        default = await server.call_tool('preview_regions', args)
        def blocks(result):
            # (content, structured) for a plain reply; the image reply is a CallToolResult carrying both.
            from mcp.types import CallToolResult
            if isinstance(result, CallToolResult):
                return result.content
            return result[0] if isinstance(result, tuple) else result
        assert not any(getattr(block, 'type', '') == 'image' for block in blocks(default))
        visible = await server.call_tool('preview_regions', {**args, 'include_image': True})
        images = [block for block in blocks(visible) if getattr(block, 'type', '') == 'image']
        assert len(images) == 1 and images[0].mimeType == 'image/png'
        import base64, io
        with Image.open(io.BytesIO(base64.b64decode(images[0].data))) as decoded:
            assert decoded.size == (20, 16)
            assert decoded.getpixel((0, 0)) == (24, 42, 60)
    asyncio.run(exercise())


def test_preview_external_path_refused_before_image_open(scene, monkeypatch):
    service, directory, _, reference, project = scene
    reference['preview_path'] = str(directory.parent.parent / 'private.png')
    (directory / 'reference.json').write_text(json.dumps(reference), encoding='utf8')
    from starstack import region_preview
    monkeypatch.setattr(region_preview, 'render_preview', lambda *a: pytest.fail('no image read before path admission'))
    with pytest.raises(AccessDenied):
        service.preview_regions(project['project_path'], str(directory))
