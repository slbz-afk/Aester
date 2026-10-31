"""R-mode contract 1.1.0 (docs/STACK_RANGE_V2_PREREG_20260929.md).

Scalar expectations use math.ldexp and struct's binary32 rounding, not the
production tile function. The fake engine is an independent exact mean over
covered frames with Siril's [0, 1] output clip; it does not model rejection or
normalization. Contract 1.0.0 behaviour stays covered by
test_stack_range_independent.py and test_range_tiles_independent.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from starstack import range_tiles, stack_range as sr
from starstack.range_tiles import rescale_finite_tile, rescale_positive_tile

V1_LITERAL_SEMANTICS = 'common_rgb_zero_is_existing_nodata; true_zero_not_supported'


# --------------------------------------------------------------------------- helpers

def _bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _from_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def _oracle(bits, divisor, restore):
    """Exact expected binary32 bits for one sample, or ValueError per contract 1.1.0."""
    value = _from_bits(bits)
    if not math.isfinite(value):
        raise ValueError('non-finite')
    if restore and value >= 1:
        raise ValueError('encoded sample must be below one')
    exponent = divisor.bit_length() - 1
    if not restore:
        exponent = -exponent
    try:
        result_bits = _bits(math.ldexp(value, exponent))
        result = _from_bits(result_bits)
        inverse_bits = _bits(math.ldexp(result, -exponent))
    except OverflowError as exc:
        raise ValueError('binary32 overflow') from exc
    if not math.isfinite(result) or (not restore and result >= 1) or inverse_bits != bits:
        raise ValueError('not exactly reversible')
    return result_bits


def _tile_oracle(tile, divisor, restore):
    flat = tile.view(np.uint32).ravel()
    return np.array([_oracle(int(b), divisor, restore) for b in flat], dtype=np.uint32).reshape(tile.shape)


def _tile(values):
    """values: list of RGB triples -> (3, 1, n) float32 tile."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32).T[:, None, :])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, array, *, overwrite=False, stackcnt=None):
    header = fits.Header()
    header['ROWORDER'] = 'BOTTOM-UP'
    if stackcnt is not None:
        header['STACKCNT'] = stackcnt
    fits.writeto(path, np.asarray(array, dtype=np.float32), header, overwrite=overwrite)


def _read(path):
    with fits.open(path, memmap=False) as hdus:
        return hdus[0].data.astype(np.float32), hdus[0].header.copy()


def _seq(name='r_light_', *, selected=(True, True, True), ref=1):
    lines = [f"S '{name}' 1 {len(selected)} {sum(selected)} 5 {ref} 6 0 0 0", 'L 3']
    lines += [f'I {i} {int(keep)}' for i, keep in enumerate(selected, 1)]
    for _ in selected:
        lines.append('R1 2 2 1 1 .03 40 H 1 0 0 0 1 0 0 0 1')
    return '\n'.join(lines) + '\n'


def _covered(array):
    return ~np.all(array == 0, axis=0)


def _oracle_mean_clip(arrays):
    """Mean over frames covering each position (not common RGB zero), then [0, 1] clip."""
    values = np.asarray(arrays, dtype=np.float64)
    cover = np.stack([_covered(a) for a in arrays])[:, None]
    count = cover.sum(axis=0)
    total = np.where(cover, values, 0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)
    return np.clip(mean, 0, 1).astype(np.float32)


@pytest.fixture(autouse=True)
def no_runtime_warning_leak():
    with warnings.catch_warnings(record=True) as recorded, np.errstate(all='warn'):
        warnings.simplefilter('always')
        yield
    runtime = [item for item in recorded if issubclass(item.category, RuntimeWarning)]
    assert not runtime, [str(item.message) for item in runtime]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    executable = tmp_path / 'fake-engine.exe'
    executable.write_bytes(b'v2 mean fixture; not executable')
    monkeypatch.setattr(sr, 'ENGINE_SHA256', _sha(executable))
    return {'version': '1.4.4', 'executable': str(executable), 'available': True}


@pytest.fixture
def inputs(tmp_path):
    """Three frames: single-channel zeros, negatives, -0, common-zero gaps and a highlight."""
    work = tmp_path / 'work'
    work.mkdir()
    yy, xx = np.indices((10, 12))
    arrays = []
    for frame in range(3):
        base = np.asarray((xx + yy + 4 + frame) / 64, dtype=np.float32)
        data = np.stack([base, base * np.float32(.75), base * np.float32(.5)])
        data[:, 3, 5] = [1.25, 1.5, 1.125]           # highlight above one: divisor 2
        data[:, -1, -1] = 0                          # no frame covers this position
        data[:, 0, 0] = [-0.25, -0.5, -0.125]        # all channels negative, every frame
        data[1, 2, 2] = 0                            # single-channel zero
        data[2, 6, 1] = -0.0                         # negative zero in one channel
        data[0, 7, 7] = -7.5                         # large negative: must not enter the divisor
        if frame == 0:
            data[:, 4, 4] = 0                        # missing in one frame only
            data[:, 5, 9] = [0.0, -0.0, 0.0]          # common signed zero = no data
        arrays.append(data)
        _write(work / f'r_light_{frame + 1:05d}.fit', data)
    (work / 'r_light_.seq').write_text(_seq(), encoding='utf-8')
    return tmp_path, work, arrays


class ClipMeanEngine:
    """Independent mean over covered frames on the scaled copies, clipped to [0, 1]."""

    def __init__(self, arrays, action=None):
        self.arrays, self.action, self.calls = arrays, action, []

    def __call__(self, executable, directory, name, commands, check):
        attempt = Path(directory)
        assert commands[-2] == 'seqclean r_light -stat' and commands[-1] == sr.STACK
        selected = [int(line.split()[1]) for line in (attempt / 'r_light_.seq').read_text().splitlines()
                    if line.startswith('I ') and line.split()[2] == '1']
        scaled = []
        for number in selected:
            data, header = _read(attempt / f'r_light_{number:05d}.fit')
            assert header['SSRGVER'] == sr.VERSION and header['SSRGUNIT'] == 'scaled'
            assert np.all(data < 1)
            scaled.append(data)
        factor = header['SSRGDIV']
        for number, data in zip(selected, scaled):
            expected = np.ldexp(self.arrays[number - 1], -(int(factor).bit_length() - 1)).astype(np.float32)
            assert np.array_equal(data.view(np.uint32), expected.view(np.uint32))
        out = _oracle_mean_clip(scaled)
        self.calls.append({'attempt': attempt, 'factor': factor, 'out': out.copy()})
        if self.action:
            out = self.action(self, out)
        _write(attempt / 'encoded_master.fit', out, stackcnt=len(selected))


def _v2_job(engine, work=None, contract=None):
    saved = contract if contract is not None else sr.contract()
    job = {'schema_version': 2, 'stack_range_mode': sr.MODE, 'stack_range_contract': saved,
           'stack_range_sha256': hashlib.sha256(json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest(),
           'engine': engine, 'checkpoints': [], 'checkpoint_files': {}}
    if work is not None:
        job['checkpoints'] = ['stack']
        job['checkpoint_files']['stack'] = [{'path': f'work/{n}', 'sha256': _sha(work / n)}
                                            for n in ('sky_master.fit', sr.RECORD, sr.VALID)]
    return job


# --------------------------------------------------------------------------- contracts

def test_new_contract_is_1_1_0_and_1_0_0_is_kept_exactly():
    new = sr.contract()
    assert new['version'] == sr.VERSION == '1.1.0' and new['mode'] == 'positive_power2_v1'
    assert 'single_channel_zero' in new['zero_semantics'] and 'nonfinite_refused' in new['zero_semantics']
    old_literal = {'mode': 'positive_power2_v1', 'version': '1.0.0', 'engine_sha256': sr.ENGINE_SHA256,
                   'backend': 'python_float32_power2_copy', 'input_maximum': 8.0,
                   'target_maximum': 0.75, 'max_attempts': 3,
                   'stack_command': 'stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=encoded_master.fit',
                   'zero_semantics': V1_LITERAL_SEMANTICS}
    assert sr.contract('1.0.0') == old_literal
    assert sr.fingerprint(sr.contract('1.0.0')) == sr.fingerprint(old_literal)
    assert {k: v for k, v in new.items() if k not in ('version', 'zero_semantics')} == \
        {k: v for k, v in old_literal.items() if k not in ('version', 'zero_semantics')}
    first = sr.contract('1.0.0')
    first['version'] = 'mutated'
    assert sr.contract('1.0.0')['version'] == '1.0.0'   # fresh copies, no shared state
    with pytest.raises(ValueError):
        sr.contract('1.2.0')


@pytest.mark.parametrize('version', ['1.0.0', '1.1.0'])
def test_verify_job_accepts_either_saved_contract(engine, version):
    assert sr.verify_job(_v2_job(engine, contract=sr.contract(version))) is True


@pytest.mark.parametrize('tamper', ['semantics_rehashed', 'version_rehashed', 'mixed_rehashed',
                                    'unknown_version', 'fingerprint', 'extra_key_rehashed', 'not_a_dict'])
def test_verify_job_rejects_tampered_contract(engine, tamper):
    saved = sr.contract()
    if tamper == 'semantics_rehashed':
        saved['zero_semantics'] = 'anything'
    elif tamper in ('version_rehashed', 'mixed_rehashed'):
        saved['version'] = '1.0.0'   # 1.0.0 version with 1.1.0 semantics is neither contract
    elif tamper == 'unknown_version':
        saved = {**sr.contract('1.0.0'), 'version': '1.2.0'}
    elif tamper == 'extra_key_rehashed':
        saved['max_attempts'] = 4
    elif tamper == 'not_a_dict':
        saved = ['positive_power2_v1', '1.1.0']
    job = _v2_job(engine, contract=saved)
    if tamper == 'fingerprint':
        job['stack_range_sha256'] = sr.fingerprint(sr.contract('1.0.0'))
    with pytest.raises(ValueError):
        sr.verify_job(job)


# --------------------------------------------------------------------------- tiles

@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('divisor', [1, 2, 8, 65536])
def test_finite_tile_matches_scalar_oracle_bit_for_bit(divisor, restore):
    rng = random.Random(20260929 + divisor + restore)
    values = [0.0, -0.0, -1.0, -3.0e38, -1e-30, 1e-30, 0.5, -0.5, 1e-3, -7.0]
    limit = 1.0 if restore else float(divisor)
    magnitude = 3.0e38 / divisor if restore else 3.4e38   # decode must not overflow binary32

    def admissible(value):
        return value < limit and abs(value) < magnitude and (abs(value) >= 1e-30 or value == 0)
    samples = [value for value in values if admissible(value)]
    while len(samples) < 60:
        value = _from_bits(rng.getrandbits(32))
        if math.isfinite(value) and admissible(value):
            samples.append(value)
    triples = [samples[i:i + 3] for i in range(0, 60, 3)]
    tile = _tile(triples)
    expected = _tile_oracle(tile, divisor, restore)
    result = rescale_finite_tile(tile, divisor, restore=restore)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert np.array_equal(result.view(np.uint32), expected)
    if not restore:   # encoded samples are below one, so decoding is admissible and exact
        back = rescale_finite_tile(result, divisor, restore=True)
        assert np.array_equal(back.view(np.uint32), tile.view(np.uint32))


def test_round_trip_negatives_single_channel_zeros_signed_zero_and_mixed():
    tile = _tile([[-0.25, 0.0, 0.5], [0.0, 0.0, 0.0], [-0.0, 0.25, -0.0], [0.0, -0.0, 0.0],
                  [1.5, -2.0, 0.0], [-100.0, -1e-20, 3.0]])
    encoded = rescale_finite_tile(tile, 4)
    assert np.all(encoded < 1)
    decoded = rescale_finite_tile(encoded, 4, restore=True)
    assert np.array_equal(decoded.view(np.uint32), tile.view(np.uint32))
    # signed zero keeps its sign bit through both directions
    assert encoded.view(np.uint32)[0, 0, 2] == encoded.view(np.uint32)[2, 0, 2] == 0x80000000
    assert decoded.view(np.uint32)[1, 0, 3] == 0x80000000 and decoded.view(np.uint32)[0, 0, 3] == 0


def _subnormal(bits):
    return np.array([bits], np.uint32).view(np.float32)[0]


@pytest.mark.parametrize('value', [_subnormal(1), _subnormal(0x80000001),     # underflow to +/-0
                                   _subnormal(3), _subnormal(0x8000000B)])    # odd ulp count: bit loss
def test_subnormal_underflow_or_bit_loss_is_refused(value):
    tile = _tile([[0.5, value, 0.25], [0.25, 0.25, 0.25]])
    with pytest.raises(ValueError):
        rescale_finite_tile(tile, 2)
    # divisor 1 is the identity and therefore exact
    assert np.array_equal(rescale_finite_tile(tile, 1).view(np.uint32), tile.view(np.uint32))


@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_finite_tile_refuses_non_finite(bad, restore):
    tile = _tile([[0.25, bad, 0.25], [0.1, 0.1, 0.1]])
    with pytest.raises(ValueError):
        rescale_finite_tile(tile, 2, restore=restore)


def test_upper_end_below_one_is_required_on_both_sides():
    with pytest.raises(ValueError):
        rescale_finite_tile(_tile([[0.0, -5.0, 2.0]]), 2)          # 2/2 == 1 after encode
    rescale_finite_tile(_tile([[0.0, -5.0, np.nextafter(np.float32(2), np.float32(0))]]), 2)
    with pytest.raises(ValueError):
        rescale_finite_tile(_tile([[1.0, 0.0, 0.0]]), 2, restore=True)
    assert rescale_finite_tile(_tile([[-7.0, -0.0, 0.0]]), 2, restore=True)[0, 0, 0] == -14.0


@pytest.mark.parametrize('bad', [np.zeros((3, 2, 2), np.float64), np.zeros((2, 2, 2), np.float32),
                                 np.zeros((3, 0, 2), np.float32), [[[0.1]]] * 3])
def test_finite_tile_keeps_structural_refusals(bad):
    with pytest.raises(ValueError):
        rescale_finite_tile(bad, 2)
    for divisor in (0, 3, True, 2.0, 131072):
        with pytest.raises(ValueError):
            rescale_finite_tile(np.full((3, 1, 1), .1, np.float32), divisor)


@pytest.mark.parametrize('values', [[[0.5, 0.0, 0.5]], [[0.5, -0.1, 0.5]], [[-0.0, 0.5, 0.5]]])
def test_v1_tile_still_refuses_single_channel_zero_and_negative(values):
    with pytest.raises(ValueError):
        rescale_positive_tile(_tile(values), 2)
    rescale_finite_tile(_tile(values), 2)


# --------------------------------------------------------------------------- source scan

def test_v2_source_scan_coverage_divisor_and_counts(inputs):
    _, work, arrays = inputs
    seq, records, expected, shape, order, divisor = sr._sources(work, lambda: None, sr.VERSION)
    assert divisor == range_tiles.choose_divisor(1.5) == 2   # -20 does not enter
    np.testing.assert_array_equal(expected, np.logical_or.reduce([_covered(a) for a in arrays]))
    assert not expected[-1, -1] and expected[4, 4] and expected[5, 9] and expected[0, 0]
    assert shape == (3, 10, 12) and order == 'BOTTOM-UP'
    first = records[0]
    assert first['single_channel_zero_samples'] == 2   # (1,2,2)=0 and (2,6,1)=-0
    assert first['negative_samples'] == 3 + 1
    assert first['minimum'] == -7.5 and first['maximum'] == 1.5
    assert first['covered_pixels'] == 10 * 12 - 3
    assert records[1]['covered_pixels'] == 10 * 12 - 1


def test_v2_coverage_ignores_unselected_frames(inputs):
    _, work, arrays = inputs
    extra = np.zeros_like(arrays[0])
    extra[:, -1, -1] = [0.0, 0.3, 0.0]                  # only an unselected frame covers it
    _write(work / 'r_light_00004.fit', extra)
    (work / 'r_light_.seq').write_text(_seq(selected=(True, True, True, False)), encoding='utf-8')
    _, records, expected, *_ = sr._sources(work, lambda: None, sr.VERSION)
    assert not expected[-1, -1] and records[3]['covered_pixels'] == 1
    (work / 'r_light_.seq').write_text(_seq(selected=(True, False, True, True), ref=0), encoding='utf-8')
    _, _, expected, *_ = sr._sources(work, lambda: None, sr.VERSION)
    assert expected[-1, -1]


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_v2_source_scan_refuses_non_finite_naming_only_non_finite(inputs, bad):
    _, work, arrays = inputs
    data = arrays[1].copy()
    data[2, 1, 1] = bad
    _write(work / 'r_light_00002.fit', data, overwrite=True)
    with pytest.raises(ValueError) as caught:
        sr._sources(work, lambda: None, sr.VERSION)
    assert '非有限' in str(caught.value) and '负值' not in str(caught.value) and '零' not in str(caught.value)


@pytest.mark.parametrize('bad', [-8.5, 8.5, -3.0e19, 3.0e19])
def test_v2_source_scan_bounds_sample_magnitude_by_input_maximum(inputs, bad):
    """|sample| <= 8 keeps the engine's float32 squared deviations far from overflow (事后, see prereg)."""
    _, work, arrays = inputs
    data = arrays[1].copy()
    data[0, 8, 8] = bad
    _write(work / 'r_light_00002.fit', data, overwrite=True)
    with pytest.raises(ValueError, match='-8 到 8'):
        sr._sources(work, lambda: None, sr.VERSION)


def test_v2_source_scan_accepts_both_ends_of_the_input_bound(inputs):
    _, work, arrays = inputs
    data = arrays[1].copy()
    data[:, 8, 8] = [-8.0, 8.0, 0.0]
    _write(work / 'r_light_00002.fit', data, overwrite=True)
    _, records, _, _, _, divisor = sr._sources(work, lambda: None, sr.VERSION)
    assert records[1]['minimum'] == -8.0 and records[1]['maximum'] == 8.0
    assert divisor == range_tiles.choose_divisor(8.0) == 16


def test_v2_source_scan_keeps_structural_checks(inputs):
    _, work, _ = inputs
    (work / 'r_light_.seq').write_text(_seq(selected=(True, False, True), ref=1), encoding='utf-8')
    with pytest.raises(ValueError):
        sr._sources(work, lambda: None, sr.VERSION)


def test_v2_source_scan_refuses_all_zero_input(inputs):
    _, work, arrays = inputs
    for number in range(1, 4):
        _write(work / f'r_light_{number:05d}.fit', np.zeros_like(arrays[0]), overwrite=True)
    with pytest.raises(ValueError, match='输入覆盖'):
        sr._sources(work, lambda: None, sr.VERSION)


def test_v1_source_scan_still_refuses_the_same_data(inputs):
    _, work, _ = inputs
    with pytest.raises(ValueError, match='通道独立'):
        sr._sources(work, lambda: None)
    with pytest.raises(ValueError, match='通道独立'):
        sr._sources(work, lambda: None, sr.VERSION_V1)


# --------------------------------------------------------------------------- output check

def _output(tmp_path, values, expected):
    path = tmp_path / 'encoded_master.fit'
    _write(path, values, overwrite=True)
    return sr._output_check(path, expected, values.shape, 'BOTTOM-UP', lambda: None, sr.VERSION)


def test_v2_output_check_counts_lower_zeros_and_upper(tmp_path):
    expected = np.ones((4, 5), bool)
    expected[3, 4] = False
    out = np.full((3, 4, 5), .5, np.float32)
    out[:, 3, 4] = 0
    out[0, 0, 0] = 0
    out[:, 1, 1] = [0.0, -0.0, 0.0]
    out[2, 2, 2] = 1.0
    summary = _output(tmp_path, out, expected)
    assert summary == {'minimum': 0.0, 'maximum': 1.0, 'upper_endpoint_channels': 1, 'lower_endpoint_channels': 4}
    with pytest.raises(ValueError):   # the same output is refused under 1.0.0
        sr._output_check(tmp_path / 'encoded_master.fit', expected, out.shape, 'BOTTOM-UP', lambda: None)


@pytest.mark.parametrize('kind', ['negative', 'nan', 'inf', 'uncovered_nonzero', 'uncovered_negative', 'geometry'])
def test_v2_output_check_failures(tmp_path, kind):
    expected = np.ones((4, 5), bool)
    expected[3, 4] = False
    out = np.full((3, 4, 5), .5, np.float32)
    out[:, 3, 4] = 0
    if kind == 'negative':
        out[1, 0, 0] = -1e-7
    elif kind == 'nan':
        out[1, 0, 0] = np.nan
    elif kind == 'inf':
        out[1, 0, 0] = np.inf
    elif kind == 'uncovered_nonzero':
        out[0, 3, 4] = 1e-6
    elif kind == 'uncovered_negative':
        out[0, 3, 4] = -1e-6
    path = tmp_path / 'encoded_master.fit'
    _write(path, out)
    shape = (3, 4, 6) if kind == 'geometry' else out.shape
    with pytest.raises(ValueError):
        sr._output_check(path, expected, shape, 'BOTTOM-UP', lambda: None, sr.VERSION)


# --------------------------------------------------------------------------- full adapter

def test_v2_run_publishes_decoded_clip_mean_and_counts_lower_end(inputs, engine):
    directory, work, arrays = inputs
    originals = {p.name: _sha(p) for p in work.glob('r_light_*.fit')}
    runner = ClipMeanEngine(arrays)
    record = sr.run_protected_stack(engine, directory, work, lambda: None, runner)
    assert record['contract'] == sr.contract() and record['divisor'] == 2 and len(runner.calls) == 1
    actual, header = _read(work / 'sky_master.fit')
    expected_encoded = runner.calls[0]['out']
    expected = np.ldexp(expected_encoded, 1).astype(np.float32)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert header['SSRGVER'] == '1.1.0' and header['SSRGUNIT'] == 'original'
    assert np.all(actual[:, 0, 0] == 0) and actual[:, 3, 5].max() > 1
    cover = np.load(work / sr.VALID, allow_pickle=False)
    lower = int(np.count_nonzero(expected_encoded[:, cover] == 0))
    assert lower >= 3 and record['lower_endpoint_channels'] == lower
    assert record['trials'][0]['lower_endpoint_channels'] == lower
    assert {p.name: _sha(p) for p in work.glob('r_light_*.fit')} == originals
    sr.finalize_receipt(work, contract=sr.contract())
    job = _v2_job(engine, work)
    assert sr.verify_receipt(work, job)['state'] == 'complete'
    v1_job = _v2_job(engine, work, contract=sr.contract('1.0.0'))
    with pytest.raises(ValueError):
        sr.verify_receipt(work, v1_job)


def test_v2_upper_endpoint_retry_unchanged(inputs, engine):
    directory, work, arrays = inputs

    def once(runner, out):
        if len(runner.calls) == 1:
            out[0, 1, 1] = 1
        return out
    runner = ClipMeanEngine(arrays, once)
    record = sr.run_protected_stack(engine, directory, work, lambda: None, runner)
    assert [c['factor'] for c in runner.calls] == [2, 4] and record['divisor'] == 4
    assert record['trials'][0]['state'] == 'upper_limit_retry'


@pytest.mark.parametrize('kind', ['negative', 'uncovered_nonzero'])
def test_v2_bad_engine_output_is_not_published(inputs, engine, kind):
    directory, work, arrays = inputs

    def bad(runner, out):
        if kind == 'negative':
            out[0, 1, 1] = -0.01
        else:
            out[:, -1, -1] = 0.01
        return out
    with pytest.raises(ValueError):
        sr.run_protected_stack(engine, directory, work, lambda: None, ClipMeanEngine(arrays, bad))
    assert not (work / 'sky_master.fit').exists()


def test_v2_unrestorable_subnormal_fails_in_chinese_before_engine(inputs, engine):
    directory, work, arrays = inputs
    data = arrays[1].copy()
    data[1, 8, 8] = _subnormal(3)                    # divisor 2 would drop its lowest bit
    _write(work / 'r_light_00002.fit', data, overwrite=True)
    runner = ClipMeanEngine(arrays)
    with pytest.raises(ValueError, match='次正规'):
        sr.run_protected_stack(engine, directory, work, lambda: None, runner)
    assert not runner.calls and not (work / 'sky_master.fit').exists()


def test_v1_contract_run_refuses_this_data_before_engine(inputs, engine):
    directory, work, arrays = inputs
    runner = ClipMeanEngine(arrays)
    with pytest.raises(ValueError):
        sr.run_protected_stack(engine, directory, work, lambda: None, runner, contract=sr.contract('1.0.0'))
    assert not runner.calls and not (work / 'sky_master.fit').exists()


def test_run_rejects_unknown_contract_before_attempt(inputs, engine):
    directory, work, arrays = inputs
    with pytest.raises(ValueError):
        sr.run_protected_stack(engine, directory, work, lambda: None, ClipMeanEngine(arrays),
                               contract={**sr.contract(), 'max_attempts': 9})
    assert not list(directory.glob('range-*'))


def test_finalize_refuses_contract_other_than_receipt(inputs, engine):
    directory, work, arrays = inputs
    sr.run_protected_stack(engine, directory, work, lambda: None, ClipMeanEngine(arrays))
    with pytest.raises(ValueError):
        sr.finalize_receipt(work, contract=sr.contract('1.0.0'))
    assert sr.finalize_receipt(work)['state'] == 'complete'


def test_v1_job_resumes_and_verifies_under_its_own_contract(tmp_path, engine):
    """An existing 1.0.0 job: positive data, receipt written under 1.0.0, verified after 1.1.0 exists."""
    work = tmp_path / 'work'
    work.mkdir()
    arrays = []
    for frame in range(3):
        data = np.full((3, 6, 7), (frame + 2) / 16, np.float32)
        data[:, 0, 0] = 0
        arrays.append(data)
        _write(work / f'r_light_{frame + 1:05d}.fit', data)
    (work / 'r_light_.seq').write_text(_seq(), encoding='utf-8')

    def v1_engine(executable, directory, name, commands, check):
        attempt = Path(directory)
        scaled = [_read(attempt / f'r_light_{n:05d}.fit')[0] for n in (1, 2, 3)]
        assert all(_read(attempt / f'r_light_{n:05d}.fit')[1]['SSRGVER'] == '1.0.0' for n in (1, 2, 3))
        _write(attempt / 'encoded_master.fit', _oracle_mean_clip(scaled), stackcnt=3)
    v1 = sr.contract('1.0.0')
    record = sr.run_protected_stack(engine, tmp_path, work, lambda: None, v1_engine, contract=v1)
    assert record['contract'] == v1 and 'lower_endpoint_channels' not in record
    assert set(record['trials'][0]) == {'divisor', 'directory', 'state', 'minimum_positive', 'maximum',
                                        'upper_endpoint_channels'}
    sr.finalize_receipt(work, contract=v1)
    job = _v2_job(engine, work, contract=v1)
    assert sr.verify_job(job) is True
    assert sr.verify_receipt(work, job)['contract']['version'] == '1.0.0'
    with pytest.raises(ValueError):
        sr.verify_receipt(work, _v2_job(engine, work))   # a 1.1.0 job cannot adopt a 1.0.0 receipt


# --------------------------------------------------------------------------- core threading

def test_core_supported_stack_passes_the_jobs_saved_contract(tmp_path, engine, monkeypatch):
    from starstack import core, registered_support as rs
    seen = []
    monkeypatch.setattr(sr, 'run_protected_stack', lambda *args, **kwargs: seen.append(kwargs.get('contract')))
    monkeypatch.setattr(core, '_validate_frame', lambda *args: (3, 4, 5))
    monkeypatch.setattr(core, '_validity_source', lambda *args: object())
    monkeypatch.setattr(core, '_grid_model', lambda *args: None)
    monkeypatch.setattr(core, '_update', lambda *args, **kwargs: None)
    monkeypatch.setattr(rs, 'prepare_exports', lambda *args, **kwargs: None)
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'light_.seq').write_text('x', encoding='utf-8')
    (work / 'light_00001.fit').write_bytes(b'')
    for saved in (sr.contract('1.0.0'), sr.contract()):
        job = {'stack_range_mode': sr.MODE, 'engine': engine, 'stack_range_contract': saved}
        core._supported_stack(engine['executable'], tmp_path, work, job, {}, True, {'dilation_px': 3}, lambda: None,
                              runner=lambda *args: None)
    assert seen == [sr.contract('1.0.0'), sr.contract()]


@pytest.mark.parametrize('own_registration', [False, True], ids=['siril_fallback', 'own_registration'])
def test_run_locked_runs_and_finalizes_under_the_jobs_saved_contract(tmp_path, engine, monkeypatch, own_registration):
    """The real ``core._run_locked`` flow: both protected-stack call sites and the finalize receive
    ``job['stack_range_contract']``; 1.1.0 diagnostics carry ``lower_endpoint_channels``, 1.0.0 do not."""
    import shutil
    from types import SimpleNamespace
    from starstack import core, imaging, registration_calibration as rc
    from starstack.records import atomic_json
    from test_core_two_pass_reference import JobDiagnosticsTest, _seq_text

    source = tmp_path / 'source'
    source.mkdir()
    for number in range(3):
        data = np.full((18, 24, 3), .03 + number * .001, dtype=np.float32)
        data[3:6, 5:8] += .2
        imaging.write_linear(source / f'frame {number}.fit', data, 'Test fixture')
    info = {**engine, 'message': 'Test engine'}
    monkeypatch.setattr(core, 'doctor', lambda *_: dict(info))
    monkeypatch.setattr(core, '_spawn_worker', lambda directory: None)
    monkeypatch.setattr(core.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10**12))
    if own_registration:
        stems = {f'light_{i:05d}' for i in (1, 2, 3)}
        monkeypatch.setattr(rc, 'calibrate', lambda work, **kwargs: JobDiagnosticsTest._fake_calibration(work, stems))
    else:
        # Force the Siril fallback explicitly: the real fit on these tiny frames is not a stable oracle
        # for "unavailable" (it made this case order-dependent in the full suite).
        def unavailable(work, **kwargs):
            summary = {'status': 'unavailable', 'reason': 'test: no distortion fit'}
            (Path(work) / rc.DISTORTION_JSON).write_text(json.dumps(summary), encoding='utf-8')
            return summary
        monkeypatch.setattr(rc, 'calibrate', unavailable)
    scripts = []

    def run_script(executable, directory, name, commands, check):
        work = Path(directory) / 'work'
        scripts.append(name)
        if name == '01_convert':
            for i, path in enumerate(sorted((Path(directory) / 'stage').glob('*')), 1):
                shutil.copyfile(path, work / f'light_{i:05d}.fit')
        elif name == '02_register_fallback':
            (work / core.SEQUENCE_FILE).write_text(_seq_text(3, 1), encoding='utf-8')
        elif name not in ('02_probe', '02_export'):
            raise AssertionError(name)
    monkeypatch.setattr(core, 'run_script', run_script)
    seen = {'run': [], 'finalize': []}
    real_finalize = sr.finalize_receipt

    def fake_protected_stack(engine_, directory, work, check, runner, contract=None):
        seen['run'].append(contract)
        saved = sr.known_contract(contract)
        master = np.mean([imaging.read_rgb(p) for p in sorted(work.glob('light_*.fit'))], axis=0).astype(np.float32)
        imaging.write_linear(work / 'sky_master.fit', master, 'Controlled engine, not Siril', extra={'STACKCNT': 3})
        (work / 'r_light_.seq').write_text('controlled sequence stand-in', encoding='utf-8')
        np.save(work / sr.VALID, np.ones(master.shape[:2], bool), allow_pickle=False)
        record = {'state': 'decoded', 'contract': saved, 'selected': ['r_light_00001', 'r_light_00002', 'r_light_00003'],
                  'sources': [], 'divisor': 1, 'seconds': 0.0, 'sequence_sha256': sr.sha(work / 'r_light_.seq'),
                  'valid_sha256': sr.sha(work / sr.VALID)}
        if saved['version'] != sr.VERSION_V1:
            record['lower_endpoint_channels'] = 7
        atomic_json(work / sr.RECORD, record)
        return record

    def spy_finalize(work, contract=None):
        seen['finalize'].append(contract)
        return real_finalize(work, contract=contract)
    monkeypatch.setattr(sr, 'run_protected_stack', fake_protected_stack)
    monkeypatch.setattr(sr, 'finalize_receipt', spy_finalize)
    for version in (sr.VERSION_V1, sr.VERSION):
        seen['run'].clear()
        seen['finalize'].clear()
        scripts.clear()
        job = core.submit_job(str(source), str(tmp_path / f'out-{version}'), stack_range_mode=sr.MODE,
                              transient_mode='off')
        directory = Path(job['job_dir'])
        assert job['stack_range_contract'] == sr.contract()          # admission always writes the new contract
        saved = sr.contract(version)
        if version == sr.VERSION_V1:                                 # an existing job admitted before 1.1.0
            core._update(directory, stack_range_contract=saved, stack_range_sha256=sr.fingerprint(saved))
        core.run_job(str(directory))
        result = core.load_job(str(directory))
        assert result['state'] == 'succeeded', result.get('error')
        assert ('02_register_fallback' in scripts) is not own_registration and '02_export' in scripts
        assert seen['run'] == [saved] and seen['finalize'] == [saved]
        diagnostics = result['diagnostics']['stack_range']
        assert diagnostics['version'] == version
        if version == sr.VERSION_V1:
            assert 'lower_endpoint_channels' not in diagnostics
        else:
            assert diagnostics['lower_endpoint_channels'] == 7
        assert sr.verify_receipt(directory / 'work', result)['contract'] == saved
