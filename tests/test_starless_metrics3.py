"""Metrics-3 counterexamples and truth/CLI identity checks; no model or real image."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from starstack import starless_metrics3 as m


def u16(a):
    return np.rint(np.clip(a, 0, 1) * 65535).astype(np.uint16)


@pytest.fixture(scope='module')
def field():
    rng = np.random.default_rng(409)
    base = u16(.25 + rng.normal(0, .002, (384, 384, 3)))
    region = np.ones(base.shape[:2], bool)
    source, truth = m.inject_stars3(base, rng, count=35, saturated=2, region=region,
                                   base_kind='known-background', amplitude_range=(.15, .3))
    return source, base, region, truth


def test_fixed_source_cohorts_and_negative_hole_gate(field):
    source, base, region, truth = field
    yy, xx = np.mgrid[:384, :384]
    mild, pit = base.astype(float) / 65535, base.astype(float) / 65535
    for star in truth['stars']:
        if star['saturated']:
            continue
        r = np.hypot(xx - round(star['x']), yy - round(star['y']))
        mild[r <= 3, 2] += .004
        pit[r <= 5] -= .04
    reference = m.evaluate3(source, u16(mild), region)
    candidate = m.evaluate3(source, u16(pit), region)
    assert reference['guards']['isolated']['count'] >= 20
    assert reference['guards']['isolated']['sample_sha256'] == candidate['guards']['isolated']['sample_sha256']
    assert candidate['guards']['isolated']['colour_shift']['p90'] < reference['guards']['isolated']['colour_shift']['p90']
    verdict = m.e1_guard(reference, candidate)
    assert verdict['status'] == 'reject'
    assert any(f['metric'] in ('residual_negative', 'luminance_negative') for f in verdict['failures'])
    for name in m.m2.STRATA:
        assert reference['guards']['dense_strata'][name]['stars']['sample_sha256'] == candidate['guards']['dense_strata'][name]['stars']['sample_sha256']


def test_equiluminant_colour_change_is_visible():
    rng = np.random.default_rng(410)
    source = u16(.25 + rng.normal(0, .002, (128, 128, 3)))
    candidate = source.astype(float) / 65535 + [.02, -.02 * .2126 / .7152, 0]
    r = m.evaluate3(source, u16(candidate), np.ones((128, 128), bool))
    assert r['guards']['nonstar']['luminance']['p95'] < .05
    assert r['guards']['nonstar']['chroma']['p95'] > 5
    assert r['guards']['nonstar']['rgb']['p95'] > 5


def test_e1_31_negative_residual_recovery_is_not_positive_residual_growth():
    base = np.full((384, 384, 3), round(.25 * 65535), np.uint16)
    region = np.ones(base.shape[:2], bool)
    source, truth = m.inject_stars3(base, np.random.default_rng(230925), count=36, region=region,
                                   base_kind='known-background', amplitude_range=(.15, .3))
    yy, xx = np.mgrid[:384, :384]
    bad = base.astype(float) / 65535
    for star in truth['stars']:
        rr = np.hypot(xx - round(star['x']), yy - round(star['y']))
        bad[rr <= 5] -= .02
        bad[rr <= 3, 2] += .004
    old = m.evaluate3(source, u16(bad), region)
    correct = m.evaluate3(source, base, region)
    verdict = m.e1_guard(old, correct)
    assert verdict['status'] == 'candidate'
    assert any(not x['passes'] for x in verdict['legacy_signed_p90_gate'])
    assert correct['guards']['isolated']['residual_positive']['p90'] == 0
    unremoved = m.evaluate3(source, source, region)
    rejected = m.e1_guard(correct, unremoved)
    assert rejected['status'] == 'reject'
    assert any(x['metric'] == 'residual_positive' for x in rejected['failures'])


def test_curved_background_perfect_removal_and_structure_loss():
    yy, xx = np.mgrid[:128, :128]
    base = u16(np.repeat((.2 + .05 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 50))[..., None], 3, -1))
    region = np.ones((128, 128), bool)
    source, truth = m.inject_stars3(base, np.random.default_rng(411), count=2, region=region,
                                   base_kind='known-background', amplitude_range=(.05, .05))
    correct = m.injection_metrics(source, base, region, base, truth)
    assert correct['target_error']['max'] == 0
    assert correct['unsaturated']['residual_absolute']['max'] == 0
    wrong = m.injection_metrics(source, np.full_like(base, round(.2 * 65535)), region, base, truth)
    assert wrong['target_error']['rmse'] > 0


@pytest.mark.parametrize('changed', ['base', 'source', 'region', 'shape'])
def test_truth_rejects_wrong_identity(field, changed):
    source, base, region, truth = field
    source, base, region = source.copy(), base.copy(), region.copy()
    if changed == 'base': base[0, 0, 0] += 1
    if changed == 'source': source[0, 0, 0] += 1
    if changed == 'region': region[:] = False
    if changed == 'shape': source, base, region = source[:-1], base[:-1], region[:-1]
    with pytest.raises(ValueError, match='identity|shape'):
        m.injection_metrics(source, base, region, base, truth)


def test_clipped_ordinary_star_is_saturated():
    base = np.full((128, 128, 3), round(.9 * 65535), np.uint16)
    region = np.ones((128, 128), bool)
    source, truth = m.inject_stars3(base, np.random.default_rng(412), count=1, region=region,
                                   base_kind='known-background', amplitude_range=(.5, .5), colour_jitter=0)
    star = truth['stars'][0]
    assert not star['requested_saturated'] and star['saturated'] and star['clipped_pixels'] > 0
    assert star['actual_flux_luminance'] < .7 * star['intrinsic_flux_luminance']
    r = m.injection_metrics(source, base, region, base, truth)
    assert r['unsaturated']['count'] == 0 and r['saturated']['count'] == 1
    assert r['saturated']['residual_absolute']['max'] == 0
    bad = copy.deepcopy(truth); bad['stars'][0]['saturated'] = False
    with pytest.raises(ValueError, match='saturation'):
        m.injection_metrics(source, base, region, base, bad)


def test_empty_region_and_valid_injection_support():
    base = np.full((128, 128, 3), 12000, np.uint16)
    region = np.zeros((128, 128), bool)
    with pytest.raises(ValueError, match='no valid'):
        m.inject_stars3(base, np.random.default_rng(1), count=1, region=region, base_kind='known-background')
    source, truth = m.inject_stars3(base, np.random.default_rng(1), count=0, region=region, base_kind='known-background')
    r = m.evaluate3(source, base, region, truth=truth, base=base)
    assert r['injected']['region_pixels'] == 0 and r['injected']['target_error']['rmse'] is None
    region[:, 64:] = True
    source, truth = m.inject_stars3(base, np.random.default_rng(1), count=1, region=region, base_kind='known-background')
    assert np.array_equal(source[:, :64], base[:, :64])
    for star in truth['stars']:
        x0, y0, x1, y1 = star['support']
        assert region[y0:y1, x0:x1].all()


def test_paired_response_requires_original_backend_output(field):
    source, base, region, truth = field
    truth = copy.deepcopy(truth); truth['base_kind'] = 'paired-response'
    with pytest.raises(ValueError, match='base_estimate'):
        m.injection_metrics(source, base, region, base, truth)
    baseline_output = u16(base.astype(float) / 65535 * .9)
    with pytest.raises(ValueError, match='provenance'):
        m.injection_metrics(source, source, region, base, truth, base_estimate=source)
    pairing = dict(version='starless-pairing-3', backend={'backend': 'controlled-test-fixture'},
                   source_array_sha256=m.p.array_identity(source), estimate_array_sha256=m.p.array_identity(baseline_output),
                   base_array_sha256=m.p.array_identity(base), base_estimate_array_sha256=m.p.array_identity(baseline_output),
                   region_array_sha256=m.p.array_identity(region))
    r = m.injection_metrics(source, baseline_output, region, base, truth, base_estimate=baseline_output, pairing=pairing)
    assert r['target_error']['max'] == 0 and r['kind'] == 'paired-response'
    with pytest.raises(ValueError, match='pairing identity'):
        m.injection_metrics(source, baseline_output, region, base, truth, base_estimate=source, pairing=pairing)


def test_opposite_saturation_colours_do_not_cancel_and_worst_is_guarded():
    rng = np.random.default_rng(413)
    yy, xx = np.mgrid[:128, :128]
    base = u16(.25 + rng.normal(0, .002, (128, 128, 3)))
    source = base.copy()
    tinted = base.astype(float) / 65535
    for x, sign in ((36, 1), (90, -1)):
        rr = np.hypot(xx - x, yy - 64)
        source[rr <= 2] = 65535
        tinted[rr <= 5] += sign * np.array([.02, -.02 * .2126 / .7152, 0])
    region = np.ones((128, 128), bool)
    b = m.evaluate3(source, base, region)
    c = m.evaluate3(source, u16(tinted), region)
    assert c['guards']['saturated_regions']['measured'] == 2
    assert min(r['colour_shift'] for r in c['guards']['saturated_regions']['objects']) > 3
    guard = m.e1_guard(b, c)
    assert guard['status'] == 'reject'
    assert guard['saturation_worst']['colour_shift']['delta'] > .5


def test_truth_scalar_tampering_rejected(field):
    source, base, region, truth = field
    bad = copy.deepcopy(truth); bad['stars'][0]['actual_flux_luminance'] *= 2
    with pytest.raises(ValueError, match='increment'):
        m.injection_metrics(source, base, region, base, bad)


def _display(path, array):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tifffile.imwrite(path, array, photometric='rgb', metadata=dict(scene_linear=False, display_encoding_applied=True,
                     color_space='encoded_sRGB', roworder='TOP-DOWN'), extratags=[(34675, 'B', len(icc), icc, False)])


def test_cli_v3_inject_measure_guard_and_file_hash(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import starless_eval as ev
    base = tmp_path / 'base.tif'
    _display(base, np.full((192, 192, 3), 13000, np.uint16))
    inj = tmp_path / 'inject'
    assert ev.main(['inject', '--source', str(base), '--output', str(inj), '--count', '4', '--metrics', '3',
                    '--base-kind', 'known-background']) == 0
    result = tmp_path / 'result.json'
    command = ['measure', '--source', str(inj / 'injected_srgb16.tif'), '--estimate', str(base), '--metrics', '3',
               '--base', str(base), '--truth', str(inj / 'truth.json'), '--output', str(result)]
    assert ev.main(command) == 0
    report = json.loads(result.read_text('utf8'))
    assert report['metrics_version'] == m.VERSION and report['metrics']['injected']['target_error']['max'] == 0
    assert ev.main(['guard', '--reference', str(result), '--candidate', str(result), '--output', str(tmp_path / 'guard.json')]) == 0
    truth = json.loads((inj / 'truth.json').read_text('utf8')); truth['source_sha256'] = '0' * 64
    (inj / 'truth.json').write_text(json.dumps(truth), encoding='utf8')
    with pytest.raises(ValueError, match='file identity'):
        ev.main(command)
    capsys.readouterr()


def test_cli_v3_cases_and_markdown(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import starless_eval as ev
    base = tmp_path / 'base.tif'; _display(base, np.full((128, 128, 3), 13000, np.uint16))
    sky = tmp_path / 'sky.tif'; tifffile.imwrite(sky, np.ones((128, 128), np.float32))
    protect = tmp_path / 'protect.tif'; tifffile.imwrite(protect, np.zeros((128, 128), np.float32))
    ident = lambda path: dict(path=path.name, sha256=ev.core._sha(path))
    case = dict(track='test', shape=[128, 128, 3], source=ident(base), sky_mask=ident(sky), protected_mask=ident(protect),
                backends={'nox': dict(estimate=ident(base))})
    cases = tmp_path / 'cases.json'; cases.write_text(json.dumps(dict(root=str(tmp_path), cases={'a': case})), encoding='utf8')
    output = tmp_path / 'cases-output'
    assert ev.main(['cases', '--cases', str(cases), '--output', str(output), '--metrics', '3']) == 0
    result = json.loads((output / 'BASELINE.json').read_text('utf8'))
    assert result['results']['a-nox']['metrics']['guards']['nonstar']['rgb']['max'] == 0
    assert 'metrics-3' in (output / 'BASELINE.md').read_text('utf8')
    capsys.readouterr()


def test_cli_paired_receipts_bind_input_output_backend_and_scope(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import starless_eval as ev
    base = tmp_path / 'base.tif'; _display(base, np.full((128, 128, 3), 13000, np.uint16))
    sky = tmp_path / 'sky.tif'; tifffile.imwrite(sky, np.ones((128, 128), np.float32))
    protect = tmp_path / 'protect.tif'; tifffile.imwrite(protect, np.zeros((128, 128), np.float32))
    inj = tmp_path / 'injected'
    assert ev.main(['inject', '--source', str(base), '--output', str(inj), '--count', '1', '--metrics', '3',
                    '--base-kind', 'paired-response', '--sky-mask', str(sky), '--protected-mask', str(protect)]) == 0
    source = inj / 'injected_srgb16.tif'

    def receipt(src, seconds):
        return dict(state='succeeded', version='controlled-separation-fixture-1', domain='display-sRGB', shape=[128, 128, 3],
                    separation_scope='pure-sky-outside-protection', source=str(src), processing_base=str(src),
                    source_hashes={str(p): ev.core._sha(p) for p in (src, sky, protect)},
                    masks=dict(sky=str(sky), protected=str(protect)),
                    backend=dict(backend='identity-test', model_sha256='a' * 64, channel_order='rgb', session_load_seconds=seconds),
                    artifacts={'diffuse_srgb16.tif': dict(path=str(src), sha256=ev.core._sha(src))})

    rd, rb = tmp_path / 'inference-injected.json', tmp_path / 'inference-base.json'
    rd.write_text(json.dumps(receipt(source, 1.2)), encoding='utf8')
    original = receipt(base, 8.4)
    rb.write_text(json.dumps(original), encoding='utf8')
    command = ['measure', '--source', str(source), '--estimate', str(source), '--base', str(base), '--base-estimate', str(base),
               '--truth', str(inj / 'truth.json'), '--metrics', '3', '--sky-mask', str(sky), '--protected-mask', str(protect),
               '--estimate-record', str(rd), '--base-estimate-record', str(rb), '--output', str(tmp_path / 'pair.json')]
    assert ev.main(command) == 0
    r = json.loads((tmp_path / 'pair.json').read_text('utf8'))['metrics']['injected']
    assert abs(r['unsaturated']['residual_signed']['median'] - 1) < 1e-12
    wrong_output = command.copy(); wrong_output[wrong_output.index('--base-estimate') + 1] = str(source)
    with pytest.raises(ValueError, match='file hash mismatch'):
        ev.main(wrong_output)
    for field in ('channel_order', 'separation_scope', 'version'):
        wrong = copy.deepcopy(original)
        if field == 'channel_order': wrong['backend'][field] = 'bgr'
        else: wrong[field] = 'different'
        rb.write_text(json.dumps(wrong), encoding='utf8')
        with pytest.raises(ValueError, match='identity differs'):
            ev.main(command)
    capsys.readouterr()
