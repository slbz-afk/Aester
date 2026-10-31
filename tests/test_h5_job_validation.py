"""H5 evidence integration, small FITS and real product with a controlled engine."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from starstack import core, imaging
from starstack.compose_evidence import write_weight, read_weight, sha256
from starstack.h5_job_validation import evaluate_job
import validate_h5
import test_s4_accept_transients as fixture_module

# Controlled engines here model Siril before the EFF-D4 stack-settings contract (efficiency P1c §4.2): jobs are
# admitted without it (tests/conftest.py pre_contract_jobs); contract jobs: tests/test_stack_settings.py.
pytestmark = pytest.mark.usefixtures("pre_contract_jobs")


@pytest.fixture
def small_job(tmp_path):
    root = tmp_path / 'job'
    work, export = root / 'work', root / 'exports' / 'revision'
    work.mkdir(parents=True); export.mkdir(parents=True)
    h, w = 64, 80
    pixels = np.ones((h, w), np.uint8) * 255
    pixels[29:35] = np.array([220, 180, 140, 100, 60, 20], np.uint8)[:, None]
    pixels[35:] = 0
    Image.fromarray(pixels).save(root / 'sky_mask.png')
    weight = pixels.astype(np.float32) / 255
    weight[24:28, :8] = 0  # An unobserved patch inside the requested sky.
    sky = np.full((h, w, 3), [.5, .4, .3], np.float32)
    ground = np.full_like(sky, [-.01, .02, 1.2])  # Signed residual and HDR.
    truth = np.empty_like(sky)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                a = float(weight[y, x])
                truth[y, x, c] = a*float(sky[y, x, c])+(1-a)*float(ground[y, x, c])
    composite = export / 'master_linear.fit'
    for path, a in ((work/'sky_master.fit', sky), (work/'foreground_master.fit', ground), (composite, truth)):
        imaging.write_linear(path, a[::-1], 'Independent scalar truth')
    descriptor = write_weight(export, weight, composite)
    job = {'state': 'succeeded', 'mask': True,
           'artifacts': [{'label': '合成线性母版', 'path': str(composite)}],
           'diagnostics': {'sky_validity': {'compose_weight': descriptor}}}
    (root/'job.json').write_text(json.dumps(job), encoding='utf-8')
    return root, job, weight


def test_roundtrip_scalar_truth_and_all_regions(small_job):
    root, job, weight = small_job
    before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
    report = evaluate_job(root)
    assert report['judgement']['status'] == 'passed'
    assert all(c['status'] == 'passed' for c in report['judgement']['checks'])
    assert report['metrics']['regions']['all_pixels']['samples'] == weight.size
    d = job['diagnostics']['sky_validity']['compose_weight']
    np.testing.assert_array_equal(read_weight(root, d, Path(d['composite'])), weight)
    assert before == {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('case, expected', [
    ('missing_descriptor', 'insufficient_data'), ('missing_file', 'insufficient_data'),
    ('wrong_hash', 'failed'), ('wrong_composite', 'failed'),
    ('running', 'insufficient_data'), ('no_mask', 'not_applicable'),
    ('bad_artifact', 'failed'), ('bad_diagnostics', 'failed'), ('bad_validity', 'failed'),
    ('empty_diagnostics_list', 'failed'), ('empty_validity_list', 'failed'), ('inferred_weight', 'failed'),
])
def test_job_evidence_states(small_job, case, expected):
    root, original, _ = small_job
    job = copy.deepcopy(original)
    d = job['diagnostics']['sky_validity']['compose_weight']
    if case == 'missing_descriptor': del job['diagnostics']['sky_validity']['compose_weight']
    elif case == 'missing_file': d['path'] = str(root/'absent.tiff')
    elif case == 'wrong_hash': d['sha256'] = '0'*64
    elif case == 'wrong_composite': d['composite'] = str(root/'work'/'sky_master.fit')
    elif case == 'running': job['state'] = 'running'
    elif case == 'no_mask': job['mask'] = False
    elif case == 'bad_artifact': job['artifacts'] = [None]
    elif case == 'bad_diagnostics': job['diagnostics'] = [1]
    elif case == 'bad_validity': job['diagnostics']['sky_validity'] = [1]
    elif case == 'empty_diagnostics_list': job['diagnostics'] = []
    elif case == 'empty_validity_list': job['diagnostics']['sky_validity'] = []
    elif case == 'inferred_weight': d['source'] = 'inferred_from_coverage'
    assert evaluate_job(root, job=job)['judgement']['status'] == expected


def test_sampling_and_wrong_mask_are_explicit(small_job, tmp_path):
    root, _, weight = small_job
    assert evaluate_job(root, sample_step=4)['judgement']['status'] == 'insufficient_data'
    path = tmp_path/'different.png'
    Image.fromarray(np.zeros(weight.shape, np.uint8)).save(path)
    assert evaluate_job(root, mask_path=path)['judgement']['status'] == 'failed'


def test_cli_new_report_preserves_job_and_previous_report(small_job, tmp_path):
    root, _, _ = small_job
    before = sha256(root/'job.json')
    output = tmp_path/'reports'
    assert validate_h5.main([str(root), '--output-root', str(output)]) == 0
    first = next(output.glob('h5-*'))
    first_hash = sha256(first/'h5.json')
    assert validate_h5.main([str(root), '--output-root', str(output), '--sample-step', '4']) == 3
    assert len(list(output.glob('h5-*'))) == 2
    assert sha256(first/'h5.json') == first_hash
    assert sha256(root/'job.json') == before
    assert '通过' in (first/'H5.md').read_text(encoding='utf-8')


@pytest.fixture
def controlled_product():
    fixture = fixture_module.AcceptanceScriptTest('test_script_judges_a_synthetic_job')
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()
        fixture.doCleanups()


def test_real_product_publication_records_actual_weight(controlled_product):
    f = controlled_product
    before = {p.name: sha256(p) for p in f.source.iterdir()}
    job = core.submit_job(str(f.source), str(f.output), sky_mask_path=str(f.mask_png))
    core.run_job(job['job_dir'])
    result = core.load_job(job['job_dir'])
    assert result['state'] == 'succeeded', result.get('error')
    report = evaluate_job(Path(job['job_dir']))
    assert report['judgement']['status'] == 'passed', report
    assert report['metrics']['regions']['all_pixels']['samples'] == f.HEIGHT*f.WIDTH
    d = result['diagnostics']['sky_validity']['compose_weight']
    assert Path(d['path']).parent == Path(d['composite']).parent
    assert d['source'] == 'recorded_at_composition'
    assert before == {p.name: sha256(p) for p in f.source.iterdir()}
    # A transient revision must not rebind the base's H5 evidence to a different image.
    core.recompose_transients(job['job_dir'])
    assert core.load_job(job['job_dir'])['diagnostics']['sky_validity']['compose_weight'] == d
    assert evaluate_job(Path(job['job_dir']))['judgement']['status'] == 'passed'


def test_evidence_write_failure_never_publishes_success(controlled_product):
    f = controlled_product
    job = core.submit_job(str(f.source), str(f.output), sky_mask_path=str(f.mask_png))
    with patch('starstack.compose_evidence.write_weight', side_effect=OSError('injected storage failure')):
        core.run_job(job['job_dir'])
    result = core.load_job(job['job_dir'])
    assert result['state'] == 'failed'
    assert not any(a.get('label') == '合成线性母版' for a in result.get('artifacts', []))
    assert evaluate_job(Path(job['job_dir']))['judgement']['status'] == 'insufficient_data'
