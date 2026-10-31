"""Independent exposure admission/adapter counterexamples (synthetic only).

The live Siril 4-pattern/2-row experiment is kept in outputs, not run by pytest.
These tests intentionally include unknown legacy RGB and hostile runner output.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from astropy.io import fits

from starstack import fits_input as fi


def rgb_header(**changes):
    return dict(NAXIS=3, NAXIS1=80, NAXIS2=64, NAXIS3=3, BITPIX=-32, **changes)


def cfa_header(**changes):
    h = dict(NAXIS=2, NAXIS1=80, NAXIS2=64, BITPIX=16, BZERO=32768,
             BAYERPAT='GRBG', ROWORDER='TOP-DOWN', EXPTIME=20)
    h.update(changes)
    return h


def write_input(path, *, cfa=False, data=None, **cards):
    if data is None:
        data = np.zeros((64, 80) if cfa else (3, 64, 80),
                        dtype=np.uint16 if cfa else np.float32)
    h = fits.Header()
    if cfa:
        h['BAYERPAT'], h['ROWORDER'], h['EXPTIME'] = 'GRBG', 'TOP-DOWN', 20
    for k, v in cards.items():
        if v is not None:
            h[k] = v
        else:
            h.pop(k, None)
    fits.writeto(path, data, h)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('row', [None, 'BOTTOM-UP'])
def test_legacy_unknown_rgb_is_still_usable_and_not_certified(row):
    h = rgb_header(BAYERPAT='GRBG')  # A stale CFA card is not a 2-D mosaic.
    if row:
        h['ROWORDER'] = row
    r = fi.describe_header(h)
    assert r['layout'] == 'rgb' and r['bayer'] is None
    assert r['interpreted_roworder'] == 'BOTTOM-UP'
    assert r['exposure_seconds'] is None
    assert all(v is None for v in r['capture'].values())


@pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
@pytest.mark.parametrize('row', ['TOP-DOWN', 'BOTTOM-UP'])
def test_all_explicit_bayer_rows_have_separate_unmodified_contract(pattern, row):
    h = cfa_header(BAYERPAT=pattern, ROWORDER=row)
    before = deepcopy(h)
    r = fi.describe_header(h)
    assert h == before
    assert (r['layout'], r['bayer'], r['source_roworder'], r['interpreted_roworder']) == (
        'cfa', pattern, row, row)
    assert r['row_policy'] == 'explicit_cfa_header'


def known_missing_row_header():
    h = cfa_header(NAXIS1=2160, NAXIS2=3840, CREATOR='ZWO Seestar S30 Pro',
                   PROGRAM='8.46', WIDECAM=0)
    del h['ROWORDER']
    return h


def test_missing_cfa_row_is_one_narrow_profile_and_explicitly_an_assumption():
    r = fi.describe_header(known_missing_row_header())
    assert r['source_roworder'] is None
    assert r['interpreted_roworder'] == 'TOP-DOWN'
    assert 'assumption' in r['row_policy']


@pytest.mark.parametrize('key,value', [
    ('CREATOR', 'ZWO Seestar S50'), ('CREATOR', None), ('PROGRAM', '8.47'),
    ('PROGRAM', None), ('WIDECAM', 1), ('WIDECAM', False), ('WIDECAM', None),
    ('NAXIS1', 3840), ('NAXIS2', 2160), ('BAYERPAT', 'RGGB'),
    ('ROWORDER', 'unknown'), ('ROWORDER', ''),
])
def test_missing_row_does_not_generalize_by_brand(key, value):
    h = known_missing_row_header()
    h[key] = value
    with pytest.raises(ValueError):
        fi.describe_header(h)


@pytest.mark.parametrize('key', ['STACKCNT', 'NSUBS', 'NCOMBINE'])
@pytest.mark.parametrize('value', [0, 2, 91, -1, 1.5, True, '1', float('inf'), float('nan')])
def test_master_or_ambiguous_count_cannot_masquerade_as_exposure(key, value):
    with pytest.raises(ValueError):
        fi.describe_header(rgb_header(**{key: value}))


@pytest.mark.parametrize('image_type', ['Master Light', 'Stacked', 'Dark Frame', 'FLAT', 'Bias', 1])
def test_calibration_and_master_frames_are_not_lights(image_type):
    with pytest.raises(ValueError):
        fi.describe_header(rgb_header(IMAGETYP=image_type))


def test_conflicting_master_claim_wins_over_one_single_frame_count():
    with pytest.raises(ValueError):
        fi.describe_header(rgb_header(STACKCNT=1, NSUBS=91, EXPTIME=20))


@pytest.mark.parametrize('extra', [
    {'EXPTIME': 20, 'LIVETIME': 300}, {'EXPTIME': 20, 'TOTALEXP': 300},
    {'LIVETIME': 20}, {'EXPTIME': 20, 'EXPOSURE': 30}, {'EXPTIME': 0},
    {'EXPTIME': -20}, {'EXPTIME': True}, {'EXPTIME': '20'},
    {'EXPTIME': float('nan')}, {'EXPTIME': float('inf')},
])
def test_ambiguous_exposure_semantics_are_not_silently_summed(extra):
    with pytest.raises(ValueError):
        fi.describe_header(rgb_header(**extra))


def test_single_frame_consistent_time_claims_are_allowed():
    r = fi.describe_header(rgb_header(EXPTIME=20, EXPOSURE=20.0, LIVETIME=20,
                                     TOTALEXP=20.0, STACKCNT=1, NSUBS=1.0))
    assert r['exposure_seconds'] == 20


@pytest.mark.parametrize('key,value', [
    ('BITPIX', 8), ('BITPIX', -64), ('BITPIX', -32), ('BITPIX', True),
    ('BSCALE', 2), ('BZERO', 0), ('BZERO', 65535), ('BSCALE', True),
    ('XBAYROFF', 1), ('YBAYROFF', -1), ('XBAYROFF', True),
    ('NAXIS1', 79), ('NAXIS2', 63), ('NAXIS1', True),
    ('BAYERPAT', 'GRGB'), ('BAYERPAT', None), ('EXPTIME', None),
])
def test_cfa_rejects_unverified_numeric_and_spatial_encodings(key, value):
    h = cfa_header(**{key: value})
    with pytest.raises(ValueError):
        fi.describe_header(h)


def test_huge_numbers_and_nonfinite_capture_metadata_fail_cleanly():
    for key in ('EXPTIME', 'BSCALE', 'GAIN'):
        with pytest.raises(ValueError):
            fi.describe_header(cfa_header(**{key: 10**1000}))
    with pytest.raises(ValueError):
        fi.describe_header(cfa_header(GAIN=float('nan')))


def test_explicit_top_down_rgb_is_not_accepted_as_implicit_legacy():
    with pytest.raises(ValueError):
        fi.describe_header(rgb_header(ROWORDER='TOP-DOWN'))


def test_uint16_rgb_and_stale_pattern_are_compatible(tmp_path):
    p = write_input(tmp_path/'rgb.fit', data=np.full((3, 64, 80), 65535, np.uint16),
                    BAYERPAT='GRBG', ROWORDER='BOTTOM-UP')
    r = fi.inspect_files([p])
    assert r['layout'] == 'rgb' and r['history_certified'] is False
    assert r['frames'][0]['bitpix'] == 16


@pytest.mark.parametrize('cards', [
    {'EXPTIME': 10}, {'FILTER': 'LP'}, {'GAIN': 80}, {'PROGRAM': '8.46'},
    {'ROWORDER': 'BOTTOM-UP'}, {'CREATOR': 'Different camera'},
])
def test_partially_missing_capture_metadata_is_not_neutral(tmp_path, cards):
    a = write_input(tmp_path/'a.fit')
    b = write_input(tmp_path/'b.fit', **cards)
    with pytest.raises(ValueError):
        fi.inspect_files([a, b])


@pytest.mark.parametrize('delta', [
    {'FILTER': 'IRCUT'}, {'EXPTIME': 10}, {'ROWORDER': 'BOTTOM-UP'},
    {'BAYERPAT': 'RGGB'}, {'GAIN': 90},
])
def test_mixed_cfa_groups_are_rejected(tmp_path, delta):
    base = dict(FILTER='LP', GAIN=80)
    a = write_input(tmp_path/'a.fit', cfa=True, **base)
    b = write_input(tmp_path/'b.fit', cfa=True, **{**base, **delta})
    with pytest.raises(ValueError):
        fi.inspect_files([a, b])


def test_layout_dimensions_bit_depth_mixtures_are_rejected(tmp_path):
    rgb = write_input(tmp_path/'rgb.fit')
    cfa = write_input(tmp_path/'cfa.fit', cfa=True)
    small = write_input(tmp_path/'small.fit', data=np.zeros((3, 62, 80), np.float32))
    integer = write_input(tmp_path/'int.fit', data=np.zeros((3, 64, 80), np.uint16))
    for other in (cfa, small, integer):
        with pytest.raises(ValueError):
            fi.inspect_files([rgb, other])


def test_existing_identical_unknown_rgb_fixtures_and_late_master(tmp_path):
    paths = [write_input(tmp_path/f'{i}.fit') for i in range(3)]
    r = fi.inspect_files(paths)
    assert len(r['frames']) == 3
    assert len({digest(p) for p in paths}) == 1
    master = write_input(tmp_path/'last_master.fit', STACKCNT=33)
    with pytest.raises(ValueError, match='last_master.fit'):
        fi.inspect_files(paths+[master])


def test_contract_freezes_header_claims_and_recipe_not_legacy_rgb_pixels(tmp_path):
    p = write_input(tmp_path/'rgb.fit', EXPTIME=20)
    contract = fi.inspect_files([p])
    job = dict(fits_input=contract, fits_input_sha256=fi.fingerprint(contract))
    assert fi.verify_contract(job, [p]) == contract
    assert fi.verify_contract({}, [p]) == contract
    changed = deepcopy(job)
    changed['fits_input']['recipe'] = 'different_recipe'
    changed['fits_input_sha256'] = fi.fingerprint(changed['fits_input'])
    with pytest.raises(ValueError):
        fi.verify_contract(changed, [p])
    wrong_hash = {**job, 'fits_input_sha256': '0'*64}
    with pytest.raises(ValueError):
        fi.verify_contract(wrong_hash, [p])
    with fits.open(p, mode='update') as f:
        f[0].header['EXPTIME'] = 21
    with pytest.raises(ValueError):
        fi.verify_contract(job, [p])


def test_legacy_cfa_cannot_resume_without_explicit_contract(tmp_path):
    p = write_input(tmp_path/'cfa.fit', cfa=True)
    with pytest.raises(ValueError, match='旧作业'):
        fi.verify_contract({}, [p])


def test_unverified_executable_and_version_are_rejected(tmp_path):
    p = tmp_path/'fake.exe'
    p.write_bytes(b'not the pinned executable')
    for version in ('1.4.4', '1.4.5', None):
        with pytest.raises(ValueError):
            fi.require_engine(dict(version=version, executable=str(p)))


@pytest.fixture
def adapter_case(tmp_path, monkeypatch):
    monkeypatch.setattr(fi, 'require_engine', lambda engine: None)
    stage, work = tmp_path/'sources', tmp_path/'work'
    stage.mkdir(); work.mkdir()
    # Include physical black claim, digital zero, one LSB and saturation.
    raw = np.resize(np.array([0, 1, 1107, 32767, 32768, 65534, 65535], np.uint16), (64, 80))
    paths = [write_input(stage/f'{i}.fit', cfa=True, data=raw+i,
                         BIAS=1107, XBAYROFF=0, YBAYROFF=0) for i in range(2)]
    return dict(root=tmp_path, work=work, paths=paths, contract=fi.inspect_files(paths),
                engine={'executable': 'fake'}, originals={p: p.read_bytes() for p in paths})


def run_adapter(case, runner, check=lambda: None, prefix='light'):
    return fi.convert_cfa(case['engine'], case['root'], case['paths'], case['work'],
                          prefix, case['contract'], check, runner)


def fake_decode(case, *, damage=None, ordinal_shift=0):
    def runner(executable, directory, name, commands, check):
        assert commands.count('set32bits') == 1
        assert any('-debayer' in c for c in commands)
        prefix = next(c.split()[1] for c in commands if c.startswith('convert '))
        for i, source in enumerate(case['paths'], 1):
            physical = fits.getdata(source)
            floated, header = fits.getdata(directory/'float'/f'frame{i:05d}.fit', header=True)
            expected = physical.astype(np.float32)/np.float32(65535)
            assert np.array_equal(floated, expected)
            assert header['BITPIX'] == -32 and header['ROWORDER'] == 'TOP-DOWN'
            assert 'BIAS' not in header and 'BZERO' not in header
            # This controlled output is an adversarial RGB container, not an
            # interpolation oracle. Distinct planes and signed HDR values.
            out = np.stack((expected, expected+.1, expected-.2)).astype(np.float32)
            out[0, 0, 0], out[2, -1, -1] = -0.0, 1.25
            header['FILENAME'] = f'frame{i:05d}.fit'
            if damage and i == len(case['paths']):
                if damage == 'nan': out[1, 3, 5] = np.nan
                elif damage == 'shape': out = out[:, :-1]
                elif damage == 'uint16': out = np.zeros(out.shape, np.uint16)
                elif damage == 'missing': continue
                elif damage == 'raise': raise RuntimeError('injected engine failure')
            fits.writeto(directory/'decoded'/f'{prefix}_{i+ordinal_shift:05d}.fit', out, header)
    return runner


def audit_for(case):
    files = list(case['root'].glob('cfa-*/conversion.json'))
    assert len(files) == 1
    return files[0], json.loads(files[0].read_text(encoding='utf-8'))


def test_adapter_does_not_subtract_bias_or_mutate_sources_and_preserves_decoded_values(adapter_case):
    case = adapter_case
    result = run_adapter(case, fake_decode(case), prefix='ref')
    audit = json.loads(result.read_text(encoding='utf-8'))
    assert audit['state'] == 'succeeded'
    assert audit['absolute_phase_color_verified'] is False
    assert [f['source_bias_claim'] for f in audit['frames']] == ['1107', '1107']
    for source, original in case['originals'].items():
        assert source.read_bytes() == original
    for name in ('ref_00001.fit', 'ref_00002.fit'):
        out, h = fits.getdata(case['work']/name, header=True)
        decoded = fits.getdata(result.parent/'decoded'/name)
        assert np.array_equal(out.view('>u4'), decoded.view('>u4'))
        assert h['ROWORDER'] == 'BOTTOM-UP' and h['BITPIX'] == -32
        assert np.signbit(out[0, 0, 0]) and out[2, -1, -1] == 1.25
        assert 'BAYERPAT' not in h and 'BIAS' not in h


@pytest.mark.parametrize('damage', ['nan', 'shape', 'uint16', 'missing', 'raise'])
def test_bad_late_decoded_frame_is_not_partially_published(adapter_case, damage):
    case = adapter_case
    with pytest.raises((ValueError, RuntimeError)):
        run_adapter(case, fake_decode(case, damage=damage))
    _, audit = audit_for(case)
    assert audit['state'] == 'failed'
    assert not list(case['work'].iterdir())
    assert all(p.read_bytes() == b for p, b in case['originals'].items())


def test_right_count_wrong_ordinals_cannot_be_bound_to_wrong_sources(adapter_case):
    case = adapter_case
    with pytest.raises(ValueError):
        run_adapter(case, fake_decode(case, ordinal_shift=1))
    assert audit_for(case)[1]['state'] == 'failed'
    assert not list(case['work'].iterdir())


def test_changed_staging_contract_is_rejected_before_engine_start(adapter_case):
    case = adapter_case
    with fits.open(case['paths'][1], mode='update') as f:
        f[0].header['EXPTIME'] = 19
    def forbidden(*args):
        pytest.fail('engine must not run after staging metadata changed')
    with pytest.raises(ValueError):
        run_adapter(case, forbidden)
    assert not list(case['root'].glob('cfa-*'))


def test_cancellation_before_decode_leaves_failed_audit_and_no_publication(adapter_case):
    case = adapter_case
    def cancel():
        raise RuntimeError('cancel requested')
    with pytest.raises(RuntimeError, match='cancel requested'):
        run_adapter(case, fake_decode(case), cancel)
    assert audit_for(case)[1]['state'] == 'failed'
    assert not list(case['work'].iterdir())


def test_cancellation_during_publication_never_claims_success(adapter_case):
    case = adapter_case
    def cancel_after_first_copy():
        if (case['work']/'light_00001.fit').exists():
            raise RuntimeError('cancel during publish')
    with pytest.raises(RuntimeError, match='cancel during publish'):
        run_adapter(case, fake_decode(case), cancel_after_first_copy)
    assert audit_for(case)[1]['state'] == 'failed'
    # A partial attempt belongs to the caller's cleanup boundary, not a cache.
    assert not (case['work']/'light_00002.fit').exists()


def test_existing_output_is_never_overwritten(adapter_case):
    case = adapter_case
    target = case['work']/'light_00001.fit'
    original = b'existing artifact owned by someone else'
    target.write_bytes(original)
    with pytest.raises(ValueError):
        run_adapter(case, fake_decode(case))
    assert target.read_bytes() == original
    assert audit_for(case)[1]['state'] == 'failed'
