"""Starless package step 1: the evaluation driver (cases / measure / inject / compare)."""
import json
from pathlib import Path
import sys
import numpy as np
import tifffile
from PIL import Image, ImageCms
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import starless_eval as ev  # noqa: E402
from starstack import starless_metrics as sm  # noqa: E402


def display_tiff(path, array):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    tifffile.imwrite(path, array, photometric='rgb', metadata={'scene_linear': False, 'display_encoding_applied': True,
                     'color_space': 'encoded_sRGB', 'roworder': 'TOP-DOWN'}, extratags=[(34675, 'B', len(icc), icc, False)])


def fixture(tmp_path, seed=3, shape=(320, 400)):
    rng = np.random.default_rng(seed)
    base = 0.15 + rng.normal(0, 0.004, shape + (3,)).astype(np.float32)
    clean = np.rint(np.clip(base, 0, 1) * 65535).astype(np.uint16)
    starry, truth = sm.inject_stars(clean, rng, count=25, saturated=2, min_separation=24, amplitude_range=(0.05, 0.5))
    src, est = tmp_path / 'source.tif', tmp_path / 'estimate.tif'
    display_tiff(src, starry)
    display_tiff(est, clean)
    sky = np.ones(shape, np.float32)
    sky[-40:] = 0
    protect = np.zeros(shape, np.uint8)
    protect[:, :6] = 255
    tifffile.imwrite(tmp_path / 'sky.tif', sky)
    Image.fromarray(protect).save(tmp_path / 'protect.png')
    return src, est, truth


def test_measure_inject_and_compare(tmp_path, capsys):
    src, est, truth = fixture(tmp_path)
    out = tmp_path / 'measure.json'
    assert ev.main(['measure', '--source', str(src), '--estimate', str(est), '--sky-mask', str(tmp_path / 'sky.tif'),
                    '--protected-mask', str(tmp_path / 'protect.png'), '--output', str(out)]) == 0
    report = json.loads(out.read_text('utf8'))
    assert report['region_rule'] == 'sky == 1 and not protected' and report['metrics']['region_pixels'] == 280 * 394
    assert abs(report['metrics']['stars']['residual_ratio']['median']) <= 0.1 and report['metrics']['nonstar']['p95'] == 0.0
    # inject on top of the (already starry) source: truth positions are known, the estimate lacks them
    inj = tmp_path / 'inj'
    assert ev.main(['inject', '--source', str(src), '--output', str(inj), '--count', '20', '--seed', '5', '--min-separation', '24']) == 0
    truth_doc = json.loads((inj / 'truth.json').read_text('utf8'))
    assert len(truth_doc['stars']) == 20 and truth_doc['source_sha256'] == report['inputs']['source']['sha256']
    with tifffile.TiffFile(inj / 'injected_srgb16.tif') as t:
        assert json.loads(t.pages[0].description)['injected_stars_for_sha256'] == truth_doc['source_sha256']
    out2 = tmp_path / 'measure2.json'
    assert ev.main(['measure', '--source', str(inj / 'injected_srgb16.tif'), '--estimate', str(est), '--truth', str(inj / 'truth.json'),
                    '--output', str(out2)]) == 0
    injected = json.loads(out2.read_text('utf8'))['metrics']['injected']
    assert injected['stars'] >= 15 and abs(injected['residual_ratio_analytic']['median']) <= 0.1
    # compare: identical → all zero; a seam-only difference shows up in the seam split
    cmp = tmp_path / 'cmp.json'
    assert ev.main(['compare', '--a', str(src), '--b', str(src), '--tile', '128', '--stride', '64', '--output', str(cmp)]) == 0
    d = json.loads(cmp.read_text('utf8'))['difference_codes']
    assert d['max'] == 0 and d['p99'] == 0 and d['fraction_nonzero'] == 0.0
    seamed = tifffile.imread(src).copy()
    seamed[64:66, :, :] += 50
    display_tiff(tmp_path / 'seamed.tif', seamed)
    assert ev.main(['compare', '--a', str(src), '--b', str(tmp_path / 'seamed.tif'), '--tile', '128', '--stride', '64', '--output', str(cmp)]) == 0
    d = json.loads(cmp.read_text('utf8'))
    assert d['difference_codes']['max'] == 50 and d['difference_codes']['seam']['seam_mean'] > d['difference_codes']['seam']['other_mean']
    assert d['criterion_r8']['consistent'] is True     # p99 stays 0: two rows out of 320 are < 1%
    capsys.readouterr()


def test_cases_verify_hashes_and_freeze_baseline(tmp_path, capsys):
    src, est, truth = fixture(tmp_path)
    root = tmp_path
    sha = ev.core._sha
    doc = {"version": "starless-eval-cases-1", "root": str(root), "tracks": {"t": ["good", "bad"]},
           "cases": {"good": {"track": "t", "shape": [320, 400, 3], "source": {"path": "source.tif", "sha256": sha(src)},
                              "sky_mask": {"path": "sky.tif", "sha256": sha(tmp_path / 'sky.tif')},
                              "protected_mask": {"path": "protect.png", "sha256": sha(tmp_path / 'protect.png')},
                              "backends": {"nox": {"estimate": {"path": "estimate.tif", "sha256": sha(est)}, "seconds_model": 2.0},
                                           "compact": {"estimate": {"path": "source.tif", "sha256": sha(src)}, "seconds_total": 1.0}}},
                     "bad": {"track": "t", "shape": [320, 400, 3], "source": {"path": "source.tif", "sha256": "0" * 64},
                             "sky_mask": {"path": "sky.tif", "sha256": sha(tmp_path / 'sky.tif')},
                             "protected_mask": {"path": "missing.png", "sha256": "1" * 64},
                             "backends": {"nox": {"estimate": {"path": "estimate.tif", "sha256": sha(est)}}}}}}
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps(doc), encoding='utf-8')
    out = tmp_path / 'baseline'
    assert ev.main(['cases', '--cases', str(cases), '--output', str(out)]) == 0
    baseline = json.loads((out / 'BASELINE.json').read_text('utf8'))
    results = baseline['results']
    assert set(results) == {'good-nox', 'good-compact', 'bad-nox'}
    assert results['bad-nox']['status'] == 'hash_mismatch' and results['bad-nox']['inputs']['protected_mask']['error'] == 'missing'
    assert results['good-nox']['status'] == 'measured' and results['good-nox']['runtime']['megapixels_per_second'] == 0.128 / 2.0
    assert 0.99 <= results['good-compact']['metrics']['stars']['residual_ratio']['median'] <= 1.01
    md = (out / 'BASELINE.md').read_text('utf8')
    assert '| good | nox | measured |' in md and '| bad | nox | hash_mismatch |' in md
    assert (out / 'good-nox.json').is_file() and baseline['cases_sha256'] == sha(cases)
    with pytest.raises(FileExistsError):
        ev.main(['cases', '--cases', str(cases), '--output', str(out)])
    capsys.readouterr()


def test_metrics2_cases_and_measure(tmp_path, capsys):
    src, est, truth = fixture(tmp_path)
    sha = ev.core._sha
    doc = {"version": "starless-eval-cases-1", "root": str(tmp_path), "tracks": {"t": ["good"]},
           "cases": {"good": {"track": "t", "shape": [320, 400, 3], "source": {"path": "source.tif", "sha256": sha(src)},
                              "sky_mask": {"path": "sky.tif", "sha256": sha(tmp_path / 'sky.tif')},
                              "protected_mask": {"path": "protect.png", "sha256": sha(tmp_path / 'protect.png')},
                              "backends": {"nox": {"estimate": {"path": "estimate.tif", "sha256": sha(est)}, "seconds_model": 2.0}}}}}
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps(doc), encoding='utf-8')
    out = tmp_path / 'baseline2'
    assert ev.main(['cases', '--cases', str(cases), '--output', str(out), '--metrics', '2']) == 0
    baseline = json.loads((out / 'BASELINE.json').read_text('utf8'))
    assert baseline['metrics_version'] == 'starless-metrics-2'
    strata = baseline['results']['good-nox']['metrics']['strata']
    assert set(strata) == {'sparse', 'medium', 'dense'} and sum(s['blocks'] for s in strata.values()) >= 1
    md = (out / 'BASELINE.md').read_text('utf8')
    assert 'metrics-2' in md and '| good | nox |' in md
    out2 = tmp_path / 'measure2.json'
    assert ev.main(['measure', '--source', str(src), '--estimate', str(est), '--metrics', '2', '--output', str(out2)]) == 0
    assert json.loads(out2.read_text('utf8'))['metrics_version'] == 'starless-metrics-2'
    capsys.readouterr()

