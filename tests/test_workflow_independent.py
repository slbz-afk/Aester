"""Independent workflow lifetime/ownership probes; never execute an engine.

All inputs and artifacts are synthetic bytes. These checks exercise orchestration
and do not claim valid image processing, RAW decoding or photographic quality.
"""
from copy import deepcopy
from pathlib import Path
import json
import hashlib

import pytest

from starstack import workflow as wf
from starstack.mcp_server import WorkspaceTools


@pytest.fixture
def scene(tmp_path, monkeypatch):
    inp, out = tmp_path / 'raw', tmp_path / 'out'; inp.mkdir(); out.mkdir()
    for i in range(3): (inp / f'exposure{i}.NEF').write_bytes(bytes([i]) * 19)
    exe = tmp_path / 'host-siril.exe'; exe.write_bytes(b'host engine fixture')
    other = tmp_path / 'other-siril.exe'; other.write_bytes(b'different executable fixture')
    unrelated = tmp_path / 'unrelated.NEF'; unrelated.write_bytes(b'outside permitted raw directory')
    def doctor(p=None):
        return {'available': True, 'version': '1.4.4', 'executable': str(Path(p) if p else exe)}
    monkeypatch.setattr(wf.core, 'doctor', doctor)
    monkeypatch.setattr(wf, '_spawn', lambda d: None)
    record = wf.submit_workflow(inp, out, siril_path=exe)
    d = Path(record['workflow_dir'])
    ident = hashlib.sha256(('workflow-stack-' + record['signature']).encode()).hexdigest()[:24]
    nested = d / 'stack' / ('job-' + ident); nested.mkdir(parents=True)
    job = {'schema_version': 1, 'job_dir': str(nested), 'state': 'failed', 'error': 'fixture interruption',
           'updated_at': wf.core.now(), 'worker_pid': None, 'input_kind': 'raw',
           'inputs': [{'path': str(p), 'sha256': wf.core._sha(p), 'size': p.stat().st_size}
                      for p in sorted(inp.glob('*.NEF'))], 'engine': doctor(exe), 'mask': None,
           'support_mode': 'legacy', 'stack_range_mode': 'legacy', 'stack_range_contract': None,
           'frame_masks': None, 'transient_marks': None, 'checkpoints': [], 'checkpoint_files': {}, 'artifacts': []}
    job.update(transient_mode='off', color_adjustments=record['request']['color_adjustments'])
    wf.atomic_json(nested / 'job.json', job)
    wf._update(d, stack_job_dir=str(nested), state='failed', attempt=2)
    return {'inp': inp, 'out': out, 'exe': exe, 'other': other, 'unrelated': unrelated,
            'directory': d, 'nested': nested, 'job': job, 'service': WorkspaceTools(inp, out, exe)}


def change_nested(scene, field):
    job = deepcopy(scene['job'])
    if field == 'engine': job['engine']['executable'] = str(scene['other'])
    elif field == 'inputs':
        p = scene['unrelated']
        job['inputs'][0] = {'path': str(p), 'sha256': wf.core._sha(p), 'size': p.stat().st_size}
    elif field == 'order': job['inputs'].reverse()
    elif field == 'mask': job['mask'] = {'path': str(scene['unrelated']), 'sha256': wf.core._sha(scene['unrelated'])}
    elif field == 'support': job['support_mode'] = 'strict_siril_1_4_4'
    elif field == 'range': job['stack_range_mode'] = 'positive_power2_v1'
    wf.atomic_json(scene['nested'] / 'job.json', job)


@pytest.mark.parametrize('field', ['engine', 'inputs', 'order', 'mask', 'support', 'range'])
def test_worker_refuses_changed_nested_contract_before_resume(scene, monkeypatch, field):
    change_nested(scene, field)
    calls = []
    def resume(path):
        calls.append(str(path)); return wf.core.load_job(path)
    monkeypatch.setattr(wf.core, 'resume_job', resume)
    result = wf.run_workflow(scene['directory'])
    assert result['state'] == 'failed'
    assert calls == [], 'Unbound nested job reached core.resume_job'
    assert not result['artifacts'] and not result['checkpoints']


@pytest.mark.parametrize('field', ['engine', 'inputs', 'mask'])
def test_mcp_resume_blocks_unbound_child_before_spawn(scene, monkeypatch, field):
    change_nested(scene, field)
    spawned = []
    monkeypatch.setattr(wf, '_spawn', lambda d: spawned.append(str(d)))
    with pytest.raises(ValueError): scene['service'].resume_workflow(str(scene['directory']))
    assert spawned == []


@pytest.mark.parametrize('field', ['engine', 'inputs', 'mask'])
def test_cancel_does_not_mutate_unbound_child(scene, monkeypatch, field):
    change_nested(scene, field)
    wf._update(scene['directory'], state='running')
    cancelled = []
    monkeypatch.setattr(wf.core, 'cancel_job', lambda p: cancelled.append(str(p)))
    with pytest.raises(ValueError): wf.cancel_workflow(scene['directory'])
    assert cancelled == [], 'Unbound child received cancellation'


@pytest.mark.parametrize('operation', ['cancel', 'resume-worker'])
def test_core_workflow_cannot_target_sibling_job(scene, monkeypatch, operation):
    sibling = scene['out'] / 'job-unrelated'; sibling.mkdir()
    other = deepcopy(scene['job']); other['job_dir'] = str(sibling)
    wf.atomic_json(sibling / 'job.json', other)
    wf._update(scene['directory'], stack_job_dir=str(sibling), state='running' if operation == 'cancel' else 'failed')
    called = []
    monkeypatch.setattr(wf.core, 'cancel_job', lambda p: called.append(('cancel', str(p))))
    monkeypatch.setattr(wf.core, 'resume_job', lambda p: called.append(('resume', str(p))) or other)
    if operation == 'cancel':
        with pytest.raises(ValueError): wf.cancel_workflow(scene['directory'])
    else:
        try:
            result = wf.run_workflow(scene['directory'])
        except ValueError:
            pass  # Either early admission rejection or a failed record is safe.
        else:
            assert result['state'] == 'failed'
    assert called == []


def test_valid_child_resume_remains_available(scene, monkeypatch):
    called = []
    monkeypatch.setattr(wf.core, 'resume_job', lambda p: called.append(str(p)) or wf.core.load_job(p))
    result = wf.run_workflow(scene['directory'])
    # Synthetic resumed child deliberately stays failed. The concern here is
    # that a correctly bound child remains eligible for actual core recovery.
    assert called == [str(scene['nested'])]
    assert result['state'] == 'failed' and not result['artifacts']


def freeze_completed_stack(scene, colour, paths):
    """Independently build the immutable completed-stack evidence contract."""
    colour = deepcopy(colour); colour['state'] = 'succeeded'
    child = deepcopy(scene['job'])
    child.update(state='succeeded', error=None, diagnostics={'color_export': colour})
    snapshot = scene['directory'] / 'stack_snapshot-independent.json'
    snapshot.write_text(json.dumps(child), encoding='utf8')
    wf.atomic_json(scene['nested'] / 'job.json', child)
    data = {'job': child, 'colour': colour}
    wf._checkpoint(scene['directory'], 'stack', data, [*paths, snapshot])
    return data


def test_publish_failure_keeps_staging_unpublished_and_existing_delivery(scene, monkeypatch):
    d = scene['directory']; products = scene['nested'] / 'products'; products.mkdir()
    files = {}
    for name in ('master_scene_linear.tif', 'photo.tif', 'photo.jpg'):
        p = products / name; p.write_bytes(('independent ' + name).encode()); files[name] = p
    colour = {'state': 'succeeded', 'master': {'path': str(files['master_scene_linear.tif'])},
              'photo': {'tiff': str(files['photo.tif']), 'jpeg': str(files['photo.jpg'])}}
    stack = freeze_completed_stack(scene, colour, list(files.values()))
    old = d / 'delivery-older'; old.mkdir(); keep = old / 'final.jpg'; keep.write_bytes(b'old immutable delivery')
    monkeypatch.setattr(wf, '_stack', lambda *_: stack)
    monkeypatch.setattr(wf, '_finish', lambda *_: {'state': 'skipped'})
    real_copy = wf.shutil.copyfile; n = []
    def fail_second(src, target):
        n.append(str(target))
        if len(n) == 2: raise OSError('independent copy failure')
        return real_copy(src, target)
    monkeypatch.setattr(wf.shutil, 'copyfile', fail_second)
    result = wf.run_workflow(d)
    assert result['state'] == 'failed' and result['artifacts'] == []
    assert 'independent copy failure' in result['error']
    assert list(d.glob('delivery-*')) == [old]
    assert keep.read_bytes() == b'old immutable delivery'
    assert len(list(d.glob('.partial-delivery-*'))) == 1


def test_cancel_between_delivery_copies_never_publishes(scene, monkeypatch):
    d = scene['directory']; products = scene['nested'] / 'products'; products.mkdir()
    paths = []
    for name in ('M.tif', 'D.tif', 'D.jpg'):
        p = products / name; p.write_bytes(name.encode()); paths.append(p)
    colour = {'master': {'path': str(paths[0])}, 'photo': {'tiff': str(paths[1]), 'jpeg': str(paths[2])}}
    stack = freeze_completed_stack(scene, colour, paths)
    monkeypatch.setattr(wf, '_stack', lambda *_: stack)
    monkeypatch.setattr(wf, '_finish', lambda *_: {'state': 'skipped'})
    original = wf.shutil.copyfile
    def request_stop(src, target):
        result = original(src, target); (d / 'cancel.request').touch(); return result
    monkeypatch.setattr(wf.shutil, 'copyfile', request_stop)
    result = wf.run_workflow(d)
    assert result['state'] == 'cancelled' and result['artifacts'] == []
    assert not list(d.glob('delivery-*'))


def test_cached_colour_path_cannot_reach_decoder_outside_workflow(scene, monkeypatch):
    """A checkpoint digest is integrity, not authorization for embedded paths."""
    from starstack import photographic_job
    outside = scene['out'].parent / 'outside-display.tif'; outside.write_bytes(b'private display fixture')
    folder = scene['nested'] / 'products'; folder.mkdir()
    paths = []
    for name in ('M.tif', 'D.tif', 'D.jpg'):
        path = folder / name; path.write_bytes(name.encode()); paths.append(path)
    colour = {'contract': 'M+D photo-default-1', 'master': {'path': str(paths[0])},
              'photo': {'tiff': str(paths[1]), 'jpeg': str(paths[2])}}
    freeze_completed_stack(scene, colour, paths)
    # Leave all original frozen files and the immutable job snapshot intact.
    # Changing the embedded source and its non-signature digest must not grant
    # access to a new image that was never authorized or bound.
    record = wf._read(scene['directory']); checkpoint = record['checkpoints']['stack']
    checkpoint['data']['colour']['photo']['tiff'] = str(outside)
    checkpoint['data_sha256'] = wf._digest(checkpoint['data'])
    wf.atomic_json(scene['directory'] / 'workflow.json', record)
    decoded = []
    def decoder(path):
        decoded.append(str(path)); raise RuntimeError('untrusted path reached decoder')
    monkeypatch.setattr(photographic_job, '_display_source', decoder)
    with pytest.raises(ValueError): wf._masks(scene['directory'], wf._cached(scene['directory'], 'stack'))
    assert decoded == []
