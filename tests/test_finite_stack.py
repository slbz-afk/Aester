"""Independent exact-mean fixtures for the validity/receipt adapter.

The mock engine deliberately does not model normalization or Winsorized rejection;
those numerical algorithms are exercised separately against the fixed executable.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from starstack import finite_stack as fs, registered_support as rs, stack_range as sr, core
from starstack.siril import Cancelled


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, a, count=3):
    fits.writeto(path, a, fits.Header({'ROWORDER': 'BOTTOM-UP', 'STACKCNT': count}), overwrite=True)


def sequence():
    return "S 'r_light_' 1 3 3 5 1 6 0 0 0\nL 3\nI 1 1\nI 2 1\nI 3 1\n" + 'R1 2 2 1 1 .03 40 H 1 0 0 0 1 0 0 0 1\n' * 3


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    folder = tmp_path / 'runtime'; folder.mkdir()
    executable = folder / 'siril-cli.exe'; executable.write_bytes(b'independent adapter mean oracle')
    dll = folder / 'numerics.dll'; dll.write_bytes(b'independent dependency fixture')
    manifest = folder / 'runtime_manifest.json'
    manifest.write_text(json.dumps({'executable': executable.name,
                                   'files': {executable.name: sha(executable), dll.name: sha(dll)}}), encoding='utf8')
    monkeypatch.setattr(fs, 'ENGINE_SHA256', sha(executable))
    monkeypatch.setattr(fs, 'RUNTIME_MANIFEST_SHA256', sha(manifest))
    return fs.describe_engine(str(executable))


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / 'job'; work = directory / 'work'; work.mkdir(parents=True)
    (work / 'r_light_.seq').write_text(sequence(), encoding='utf8')
    (work / rs.EXPORT_SEQUENCE).write_text('immutable geometry fixture', encoding='utf8')
    arrays, masks, frames = [], [], []
    counts = np.zeros((12, 16), dtype=np.uint16)
    for i in range(3):
        yy, xx = np.indices(counts.shape)
        a = np.stack([(xx + 2 * yy + 8 + i) / 128] * 3).astype(np.float32)
        a[0, 3, 4] = [0, .25, .5][i]      # channel zero contributes to mean .25
        a[:, 5, 6] = [-.5, 0, .5][i]       # zero output is present, not missing
        a[:, 2, 7] = [-2, -1, -3][i]       # negative output must survive
        a[:, 4, 8] = [2, 3, 4][i]          # no 0..1 output clipping
        valid = np.ones(counts.shape, bool)
        valid[-1, -1] = False              # top-down geometry, all absent
        if i == 1: valid[2, 3] = False
        a[:, ~valid[::-1]] = 0
        path = work / f'r_light_{i+1:05d}.fit'; write(path, a)
        descriptor = rs.save_array(work / f'frame_support_light_{i+1:05d}.npy', valid)
        frames.append({'stem': f'light_{i+1:05d}', 'supported_pixels': int(valid.sum()),
                       'export_after_sha256': sha(path), 'explicit_validity': descriptor})
        arrays.append(a); masks.append(valid); counts += valid
    support = {'state': 'exports_filtered', 'mode': rs.MODE, 'version': rs.SUPPORT_VERSION,
               'sequence_sha256': sha(work / rs.EXPORT_SEQUENCE), 'frames': frames,
               'shape': list(counts.shape), 'eligible': rs.save_array(work / rs.COUNTS, counts)}
    (work / rs.RECORD).write_text(json.dumps(support), encoding='utf8')
    return directory, work, arrays, masks


class Oracle:
    def __init__(self, prepared, mutation=None):
        self.prepared, self.mutation, self.calls = prepared, mutation, 0

    def __call__(self, exe, directory, name, commands, check):
        self.calls += 1
        assert 'setref q 2' in commands
        assert commands[-1] == fs.STACK
        values = []
        for i, (source, valid) in enumerate(zip(self.prepared[2], self.prepared[3]), 1):
            a = fits.getdata(directory / f'q_{i:05d}.fit')
            np.testing.assert_array_equal(a[:, valid[::-1]], source[:, valid[::-1]])
            assert np.isnan(a[:, ~valid[::-1]]).all()
            values.append(a.astype(np.float64))
        x = np.stack(values); nvalid = np.isfinite(x).sum(axis=0).astype(np.uint16)
        total = np.where(np.isfinite(x), x, 0).sum(axis=0)
        result = np.divide(total, nvalid, where=nvalid > 0, out=np.zeros_like(total)).astype(np.float32)
        write(directory / 'finite_master.fit', result)
        write(directory / 'finite_master_nvalid.fit', nvalid)
        write(directory / 'finite_master_nused.fit', nvalid.copy())
        (directory / 'q_.seq').write_text(sequence().replace('r_light_', 'q_'), encoding='utf8')
        if self.mutation: self.mutation(directory, self.prepared[1])
        check()


def run(prepared, runtime, oracle=None, check=lambda: None):
    return fs.run(runtime, prepared[0], prepared[1], check, oracle or Oracle(prepared))


def test_finite_zero_negative_overone_and_counts_are_preserved(prepared, runtime):
    old = {p.name: sha(p) for p in prepared[1].glob('r_light*')}
    record = run(prepared, runtime)
    a = fits.getdata(prepared[1] / 'sky_master.fit')
    assert a[0, 3, 4] == .25 and np.all(a[:, 5, 6] == 0)
    assert np.all(a[:, 2, 7] == -2) and np.all(a[:, 4, 8] == 3)
    valid = np.load(prepared[1] / fs.VALID)
    assert valid[::-1][5, 6] and not valid[-1, -1]
    assert record['sources'][0]['valid_zero_channels'] > 0
    assert record['reference'] == 'r_light_00002.fit'
    assert record['state'] == 'decoded'
    assert old == {p.name: sha(p) for p in prepared[1].glob('r_light*')}
    support = rs.finish_stack(prepared[1], prepared[1] / 'sky_master.fit', None, explicit_counts=True)
    assert 'nused > 0' in support['zero_semantics']
    assert rs.load_array(prepared[1], support['native_valid'])[::-1][5, 6]


@pytest.mark.parametrize('target', ['siril-cli.exe', 'numerics.dll', 'runtime_manifest.json'])
def test_runtime_tamper_rejected(prepared, runtime, target):
    path = Path(runtime['executable']).parent / target
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='运行时|引擎'):
        run(prepared, runtime)
    assert not (prepared[1] / 'sky_master.fit').exists()


@pytest.mark.parametrize('target', ['r_light_00001.fit', 'frame_support_light_00001.npy', rs.EXPORT_SEQUENCE])
def test_input_identity_change_rejected_before_engine(prepared, runtime, target):
    path = prepared[1] / target; path.write_bytes(path.read_bytes() + b'changed')
    oracle = Oracle(prepared)
    with pytest.raises(ValueError): run(prepared, runtime, oracle)
    assert oracle.calls == 0


@pytest.mark.parametrize('target', ['r_light_00001.fit', 'frame_support_light_00001.npy', rs.EXPORT_SEQUENCE])
def test_input_change_during_engine_blocks_publication(prepared, runtime, target):
    def mutate(_, work):
        path = work / target; path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='已改变'):
        run(prepared, runtime, Oracle(prepared, mutate))
    assert not (prepared[1] / 'sky_master.fit').exists()


@pytest.mark.parametrize('fault', ['nvalid', 'nused', 'missing', 'absent_nonzero', 'nan', 'stackcnt', 'roworder', 'reference'])
def test_bad_output_is_not_published(prepared, runtime, fault):
    def mutate(folder, _):
        if fault == 'missing':
            (folder / 'finite_master_nused.fit').unlink(); return
        if fault == 'reference':
            seq = folder / 'q_.seq'
            seq.write_text(seq.read_text(encoding='utf8').replace('3 3 5 1 6', '3 3 5 0 6'), encoding='utf8'); return
        target = folder / ('finite_master_nvalid.fit' if fault == 'nvalid' else
                           'finite_master_nused.fit' if fault == 'nused' else 'finite_master.fit')
        with fits.open(target, mode='update', memmap=False) as h:
            if fault == 'nvalid': h[0].data[0, 3, 4] -= 1
            elif fault == 'nused': h[0].data[0, 3, 4] += 1
            elif fault == 'absent_nonzero': h[0].data[:, 0, -1] = 1
            elif fault == 'nan': h[0].data[0, 3, 4] = np.nan
            elif fault == 'stackcnt': h[0].header['STACKCNT'] = 2
            elif fault == 'roworder': h[0].header['ROWORDER'] = 'TOP-DOWN'
    with pytest.raises((ValueError, FileNotFoundError)):
        run(prepared, runtime, Oracle(prepared, mutate))
    assert not (prepared[1] / 'sky_master.fit').exists()
    receipts = list(prepared[0].glob('finite-*/receipt.json'))
    assert json.loads(receipts[0].read_text(encoding='utf8'))['state'] == 'failed'


def test_cancel_keeps_prior_master_and_records_cancelled(prepared, runtime):
    prior = prepared[1] / 'sky_master.fit'; prior.write_bytes(b'prior completed master')
    def cancel(*_): raise Cancelled('independent cancellation injection')
    with pytest.raises(Cancelled): run(prepared, runtime, Oracle(prepared, cancel))
    assert prior.read_bytes() == b'prior completed master'
    receipt = json.loads(next(prepared[0].glob('finite-*/receipt.json')).read_text(encoding='utf8'))
    assert receipt['state'] == 'cancelled'


@pytest.mark.parametrize('change', [None, fs.NVALID, fs.NUSED, fs.VALID, 'frame_support_light_00001.npy', 'sky_master.fit'])
def test_checkpoint_resume_identity(prepared, runtime, monkeypatch, change):
    run(prepared, runtime); sr.finalize_receipt(prepared[1])
    monkeypatch.setattr(rs, 'require_engine', lambda _: None)
    job = {'schema_version': 2, 'stack_range_mode': fs.MODE, 'support_mode': rs.MODE, 'mask': {'sha': 'fixture'},
           'engine': {}, 'stack_engine': runtime, 'stack_range_contract': fs.contract(),
           'stack_range_sha256': sr.fingerprint(fs.contract()),
           'checkpoint_files': {'stack': [{'path': 'work/' + fs.RECORD, 'sha256': sha(prepared[1] / fs.RECORD)}]}}
    if change:
        p = prepared[1] / change; p.write_bytes(p.read_bytes() + b'changed')
        with pytest.raises(ValueError): sr.verify_receipt(prepared[1], job)
    else:
        assert sr.verify_job(job)
        assert sr.verify_receipt(prepared[1], job)['state'] == 'complete'


@pytest.mark.parametrize('support,mask', [('legacy', 'white.png'), (rs.MODE, None)])
def test_admission_requires_explicit_geometry_before_any_engine(support, mask, monkeypatch):
    monkeypatch.setattr(core, 'doctor', lambda _: pytest.fail('no engine needed'))
    with pytest.raises(ValueError, match='严格几何支持'):
        core.submit_job('input', 'output', support_mode=support, sky_mask_path=mask,
                        stack_range_mode=fs.MODE, stack_engine_path='runtime/siril-cli.exe')


def test_no_ignored_stack_engine_parameter():
    with pytest.raises(ValueError, match='仅用于'):
        core.submit_job('input', 'output', stack_engine_path='ignored.exe')
