"""Independent contract acceptance; failures are retained, not marked xfail.

Run with PYTHONDONTWRITEBYTECODE=1 and --basetemp inside this directory.
No production/delivery files are changed. All geometry is analytic fixture data.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest


from starstack import registration_diagnostics as rd


def document(width=100, height=80):
    return {
        'schema_version': 1,
        'coordinate_convention': dict(rd.COORDINATE_CONVENTION),
        'reference': {'frame_id': 'ref', 'width': width, 'height': height, 'coordinate_space_id': 'reference-pixels'},
        'frames': [{
            'frame_id': 'source', 'width': width, 'height': height,
            'coordinate_space_id': 'source-pixels',
            'source_to_reference': np.eye(3).tolist(), 'matches': [],
        }],
    }


def analyze(doc):
    return rd.analyze_frame(doc['reference'], doc['frames'][0])


@pytest.mark.parametrize('scale', [1., -1., 1.e-6, 1.e6, 1.e307, 1.e-323])
def test_homography_scale_identity_analytic(scale):
    doc = document()
    doc['frames'][0]['source_to_reference'] = (scale * np.eye(3)).tolist()
    frame = analyze(doc)
    assert frame['state'] == 'succeeded'
    expected = [[-.5, -.5], [99.5, -.5], [99.5, 79.5], [-.5, 79.5]]
    np.testing.assert_allclose(frame['geometric_overlap']['projected_source_polygon_xy'], expected, rtol=0, atol=1e-11)
    assert frame['geometric_overlap']['intersection_area_px2'] == pytest.approx(8000.)
    assert frame['geometric_overlap']['reference_area_fraction'] == pytest.approx(1.)


def test_projective_trapezoid_continuous_area_exact():
    # In boundary-relative X=x+1/2,Y=y+1/2, X'=X/(1+X/4),Y'=Y/(1+X/4).
    # The 4x2 rectangle becomes a width-2 trapezoid, parallel sides 2 and 1.
    doc = document(4, 2)
    doc['frames'][0]['source_to_reference'] = [[.875, 0, -.0625], [-.125, 1, -.0625], [.25, 0, 1.125]]
    overlap = analyze(doc)['geometric_overlap']
    np.testing.assert_allclose(overlap['projected_source_polygon_xy'], [[-.5, -.5], [1.5, -.5], [1.5, .5], [-.5, 1.5]], rtol=0, atol=1e-14)
    assert overlap['intersection_area_px2'] == 3
    assert overlap['reference_area_fraction'] == .375


@pytest.mark.parametrize('translation,expected_area', [(4., 0.), (3.75, .5), (2., 4.)])
def test_continuous_edge_touch_and_fractional_overlap(translation, expected_area):
    doc = document(4, 2)
    doc['frames'][0]['source_to_reference'][0][2] = translation
    assert analyze(doc)['geometric_overlap']['intersection_area_px2'] == expected_area


@pytest.mark.parametrize('denominator', [[1, 0, -2], [1, 0, .5]])
def test_internal_or_corner_horizon_invalid_even_with_empty_matches(denominator):
    doc = document(4, 2)
    doc['frames'][0]['source_to_reference'][2] = denominator
    with pytest.raises(ValueError, match='frame_crosses_projective_infinity'):
        analyze(doc)
    report = rd.analyze_sequence(doc)
    assert report['state'] == 'failed'
    assert report['frames'][0]['error']['code'] == 'frame_crosses_projective_infinity'
    json.dumps(report, allow_nan=False)


def test_fit_holdout_unknown_kept_separate_and_no_outlier_removal():
    doc = document()
    frame = doc['frames'][0]
    frame['matches'] = [
        {'star_id': 'fit', 'role': 'fit', 'source_xy': [40, 40], 'reference_xy': [40, 40]},
        {'star_id': 'holdout_5', 'role': 'holdout', 'source_xy': [43, 44], 'reference_xy': [40, 40]},
        {'star_id': 'holdout_13', 'role': 'holdout', 'source_xy': [45, 52], 'reference_xy': [40, 40]},
        {'star_id': 'unknown', 'role': 'unknown', 'source_xy': [50, 40], 'reference_xy': [40, 40]},
    ]
    result = analyze(doc)
    assert result['match_counts'] == {'fit': 1, 'holdout': 2, 'unknown': 1}
    assert result['holdout_evidence'] == 'caller_declared'
    assert result['statistics']['fit']['overall']['rms_px'] == 0
    holdout = result['statistics']['holdout']['overall']
    assert holdout == {'count': 2, 'rms_px': pytest.approx(math.sqrt(97), rel=1e-14), 'median_px': 9, 'p95_px': pytest.approx(12.6, rel=1e-14), 'max_px': 13}
    assert result['statistics']['unknown']['overall']['rms_px'] == 10
    assert result['statistics']['holdout']['regions']['center']['count'] == 2
    assert result['residuals'][1]['dx_px'] == 3
    assert result['residuals'][1]['dy_px'] == 4


@pytest.mark.parametrize('bad_xy', [[True, 0], ['1', 0], [float('nan'), 0], [-.50001, 0]])
def test_match_rejection_preserves_other_frame_and_strict_output(bad_xy):
    doc = document()
    bad = copy.deepcopy(doc['frames'][0]); bad['frame_id'] = 'invalid'
    bad['matches'] = [{'star_id': 'p', 'role': 'holdout', 'source_xy': bad_xy, 'reference_xy': [0, 0]}]
    doc['frames'].append(bad)
    report = rd.analyze_sequence(doc)
    assert report['state'] == 'partial'
    assert [f['frame_id'] for f in report['frames']] == ['source', 'invalid']
    assert [f['state'] for f in report['frames']] == ['succeeded', 'invalid']
    assert report['frames'][1]['residuals'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity', '1e400'])
def test_json_parser_rejects_nonfinite_even_in_unknown_metadata(token):
    # A syntactically finite decimal can overflow binary64; parse_constant alone misses this.
    payload = json.dumps(document())[:-1] + ', "optional_metadata": ' + token + '}'
    with pytest.raises(ValueError):
        rd.parse_document(payload.encode('utf-8'))


def test_programmatic_supported_numpy_scalars_leave_json_serializable_report():
    doc = document()
    doc['frames'][0]['elapsed_seconds'] = np.float32(2.5)
    doc['frames'][0]['source_to_reference'] = [[np.float32(v) for v in row] for row in np.eye(3)]
    report = rd.analyze_sequence(doc)
    assert report['state'] == 'succeeded'
    json.dumps(report, allow_nan=False)


def test_large_finite_residual_rms_does_not_emit_infinity():
    # Deliberately extreme but contract-valid integers and finite doubles.
    # A stable scaled RMS remains 1e155 without squaring overflow.
    doc = document(10**156, 1)
    doc['frames'][0]['matches'] = [{'star_id': 'p', 'role': 'holdout', 'source_xy': [1.e155, 0], 'reference_xy': [0, 0]}]
    report = rd.analyze_sequence(doc)
    json.dumps(report, allow_nan=False)
    assert report['frames'][0]['statistics']['holdout']['overall']['rms_px'] == pytest.approx(1.e155)


def test_unrepresentable_frame_extent_is_invalid_without_aborting_other_frames():
    doc = document()
    bad = copy.deepcopy(doc['frames'][0]); bad['frame_id'] = 'huge'; bad['width'] = 10**400
    doc['frames'].append(bad)
    report = rd.analyze_sequence(doc)
    assert report['state'] == 'partial'
    assert [f['state'] for f in report['frames']] == ['succeeded', 'invalid']
    json.dumps(report, allow_nan=False)


def test_unrepresentable_reference_extent_raises_document_valueerror():
    with pytest.raises(ValueError):
        rd.analyze_sequence(document(10**400, 1))


def test_atomic_publish_unsupported_filesystem_fails_closed(tmp_path, monkeypatch):
    source = tmp_path/'input.json'; source.write_text(json.dumps(document()), encoding='utf-8')
    output = tmp_path/'report.json'
    def unsupported(*args, **kwargs):
        raise OSError('controlled filesystem does not support atomic hard-link publication')
    monkeypatch.setattr(rd.os, 'link', unsupported)
    assert rd.main(['--input', str(source), '--output', str(output)]) == 1
    assert not output.exists()
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('state,expected_exit', [('succeeded', 0), ('partial', 2), ('failed', 3)])
def test_cli_states_and_source_unchanged(tmp_path, state, expected_exit):
    doc = document()
    if state in ('partial', 'failed'):
        bad = copy.deepcopy(doc['frames'][0]); bad['frame_id'] = 'bad'
        bad['source_to_reference'] = np.zeros((3, 3)).tolist()
        doc['frames'] = ([doc['frames'][0], bad] if state == 'partial' else [bad])
    source = tmp_path/'input.json'; output = tmp_path/'report.json'
    raw = json.dumps(doc).encode('utf-8'); source.write_bytes(raw)
    assert rd.main(['--input', str(source), '--output', str(output)]) == expected_exit
    assert source.read_bytes() == raw
    report = json.loads(output.read_text(encoding='utf-8'), parse_constant=lambda x: pytest.fail(x))
    assert report['state'] == state
    assert rd.main(['--input', str(source), '--output', str(output)]) == 1
    assert rd.main(['--input', str(source), '--output', str(source), '--overwrite']) == 1
    assert source.read_bytes() == raw


def test_cli_does_not_read_or_execute_payload_paths(tmp_path, monkeypatch):
    doc = document()
    secret = tmp_path/'do-not-read.txt'; secret.write_text('private test sentinel', encoding='utf-8')
    doc['frames'][0]['file_path'] = str(secret)
    doc['frames'][0]['command'] = 'this is inert fixture text'
    source = tmp_path/'input.json'; source.write_text(json.dumps(doc), encoding='utf-8')
    output = tmp_path/'report.json'
    original_read = Path.read_bytes
    read_paths = []
    def guarded_read(path):
        read_paths.append(path)
        assert path == source, 'CLI dereferenced a payload path'
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded_read)
    assert rd.main(['--input', str(source), '--output', str(output)]) == 0
    assert read_paths == [source]


def test_default_no_overwrite_is_atomic_at_publish_boundary(tmp_path, monkeypatch):
    source = tmp_path/'input.json'; source.write_text(json.dumps(document()), encoding='utf-8')
    output = tmp_path/'report.json'
    original_replace = rd.os.replace
    original_link = rd.os.link
    def concurrent_writer_before_publish(temp, dest):
        # A deterministic scheduling point between final exists() and publication.
        Path(dest).write_text('other writers report', encoding='utf-8')
        return original_replace(temp, dest)
    monkeypatch.setattr(rd.os, 'replace', concurrent_writer_before_publish)
    def concurrent_writer_before_link(temp, dest):
        Path(dest).write_text('other writers report', encoding='utf-8')
        return original_link(temp, dest)
    monkeypatch.setattr(rd.os, 'link', concurrent_writer_before_link)
    code = rd.main(['--input', str(source), '--output', str(output)])
    assert output.read_text(encoding='utf-8') == 'other writers report'
    assert code == 1
    assert not list(tmp_path.glob('*.tmp'))
