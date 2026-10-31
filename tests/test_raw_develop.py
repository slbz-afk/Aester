import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from starstack import core, raw_audit, raw_develop
from starstack.color_preview import preview_recipe


@pytest.fixture
def fixture_audit(tmp_path, monkeypatch):
    source = tmp_path/'source'/'test.dng'
    source.parent.mkdir()
    source.write_bytes(b'independent source identity')
    directory = tmp_path/'audit'
    directory.mkdir()
    codes = np.arange(32*48, dtype=np.uint16).reshape(32, 48)
    np.save(directory/'raw_codes.npy', codes)
    tags = {'status': 'ok', 'raw_shape': [32, 48], 'tags': {
        'BlackLevel': {'value': [100.25]}, 'BlackLevelRepeatDim': {'value': [1, 1]},
        'WhiteLevel': {'value': [1000]}, 'ActiveArea': {'value': [0, 0, 32, 48]}}}
    audit = {'schema_version': 1, 'state': 'succeeded', 'audit_dir': str(directory),
             'source': {'path': str(source), 'sha256': core._sha(source), 'size_bytes': source.stat().st_size},
             'geometry': {'width': 48, 'height': 32, 'row_order': 'TOP-DOWN',
                          'coordinate_basis': 'decoder_full_raw_mosaic',
                          'orientation_applied': False, 'full_pattern': [[0, 1], [3, 2]]},
             'libraw_metadata': {'fields': {'color_desc': {'value': 'RGBG'}, 'num_colors': {'value': 3},
                                           'raw_pattern': {'value': [[0, 1], [3, 2]]}}},
             'file_metadata': tags, 'black_model': raw_audit.black_model(tags, codes.shape),
             'artifacts': {'raw_codes': {'path': str(directory/'raw_codes.npy'),
                 'sha256': core._sha(directory/'raw_codes.npy'), 'shape': [32, 48], 'dtype': 'uint16'}}}
    core.atomic_json(directory/'audit.json', audit)
    monkeypatch.setattr(raw_audit, 'read_dng_tags', lambda _: tags)
    # Integration boundaries use a known invertible transform; real LibRaw
    # matrix/WB tests live in test_raw_color and the real validation script.
    def build(metadata, **kw):
        return {'reference_sha256': kw['reference_sha256'],
                'scale': kw['reference_white']-kw['reference_black_green']}
    def apply(data, recipe, *, out, input_state, row_order):
        assert input_state == 'black_subtracted_camera_rgb_DN' and row_order == 'TOP-DOWN'
        out[:] = data / recipe['scale'] * np.array([2., 1., 1.5])
        return out
    fake = SimpleNamespace(build_recipe=build, validate_recipe=lambda *_: None,
                           apply_color=apply, __file__=__file__)
    monkeypatch.setitem(sys.modules, 'starstack.raw_color', fake)
    import starstack
    monkeypatch.setattr(starstack, 'raw_color', fake, raising=False)
    return directory, audit, source


def test_float_products_reuse_recipe_and_leave_sources_unchanged(fixture_audit, tmp_path):
    directory, audit, source = fixture_audit
    original = (source.read_bytes(), (directory/'raw_codes.npy').read_bytes())
    first = raw_develop.develop_raw(directory, tmp_path/'out')
    assert first['state'] == 'succeeded', first
    recipe = first['artifacts']['color_recipe']
    second = raw_develop.develop_raw(directory, tmp_path/'out', recipe_path=recipe['path'], exposure_ev=0)
    assert second['state'] == 'succeeded', second
    assert first['development_dir'] != second['development_dir']
    assert recipe['sha256'] == second['artifacts']['color_recipe']['sha256']
    for name in ('camera_rgb', 'linear_srgb'):
        p = first['artifacts'][name]
        array = tifffile.imread(p['path'])
        assert array.dtype == np.float32 and array.shape == (32, 48, 3)
        assert array.min() < 0 and array.max() > 1
        np.testing.assert_array_equal(array, tifffile.imread(second['artifacts'][name]['path']))
    # A measured red at the top left stays in the same location with one subtraction.
    camera = tifffile.imread(first['artifacts']['camera_rgb']['path'])
    assert camera[0, 0, 0] == -100.25
    assert (source.read_bytes(), (directory/'raw_codes.npy').read_bytes()) == original
    assert first['artifacts']['preview']['recipe'] == preview_recipe(2)


@pytest.mark.parametrize('change', ['source', 'array', 'black', 'row_order', 'pattern', 'foreign_array'])
def test_invalid_audit_fails_before_writing_derivatives(fixture_audit, tmp_path, change):
    directory, audit, source = fixture_audit
    if change == 'source':
        source.write_bytes(b'changed')
    elif change == 'array':
        np.save(directory/'raw_codes.npy', np.ones((32, 48), np.uint16))
    elif change == 'black':
        audit['black_model']['black_values'] = [0]
    elif change == 'row_order':
        audit['geometry']['row_order'] = 'BOTTOM-UP'
    elif change == 'pattern':
        audit['geometry']['full_pattern'] = [[1, 0], [2, 3]]
    else:
        other = tmp_path/'foreign.npy'
        other.write_bytes((directory/'raw_codes.npy').read_bytes())
        audit['artifacts']['raw_codes']['path'] = str(other)
    core.atomic_json(directory/'audit.json', audit)
    with pytest.raises(ValueError):
        raw_develop.develop_raw(directory, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_unsupported_black_model_is_explicit(fixture_audit, tmp_path, monkeypatch):
    directory, audit, _ = fixture_audit
    audit['file_metadata'] = {'status': 'not_dng', 'tags': {}}
    core.atomic_json(directory/'audit.json', audit)
    monkeypatch.setattr(raw_audit, 'read_dng_tags', lambda _: audit['file_metadata'])
    with pytest.raises(ValueError, match='扣黑模型尚未验证'):
        raw_develop.develop_raw(directory, tmp_path/'out')


def test_incompatible_recipe_is_rejected(fixture_audit, tmp_path):
    directory, audit, _ = fixture_audit
    first = raw_develop.develop_raw(directory, tmp_path/'out')
    path = Path(first['artifacts']['color_recipe']['path'])
    recipe = json.loads(path.read_text(encoding='utf-8'))
    recipe['frame_compatibility']['geometry']['width'] = 44
    p = tmp_path/'bad_recipe.json'
    core.atomic_json(p, recipe)
    with pytest.raises(ValueError, match='不兼容'):
        raw_develop.develop_raw(directory, tmp_path/'out', recipe_path=p)


def test_failures_leave_a_failed_manifest(fixture_audit, tmp_path, monkeypatch):
    directory, _, _ = fixture_audit
    def fail(*args, **kwargs):
        raise RuntimeError('test injected output failure')
    monkeypatch.setattr(raw_develop, 'save_preview', fail)
    result = raw_develop.develop_raw(directory, tmp_path/'out')
    assert result['state'] == 'failed'
    saved = json.loads((Path(result['development_dir'])/'development.json').read_text(encoding='utf-8'))
    assert saved['state'] == 'failed' and 'injected' in saved['error']


def test_output_cannot_be_inside_source_or_audit(fixture_audit):
    directory, _, source = fixture_audit
    for out in (source.parent, source.parent/'outputs', source.parent/'nested'/'outputs', directory/'child'):
        with pytest.raises(ValueError):
            raw_develop.develop_raw(directory, out)


def test_audit_edit_between_parse_and_hash_is_rejected(fixture_audit, tmp_path, monkeypatch):
    directory, audit, _ = fixture_audit
    original = raw_audit.read_dng_tags
    def mutate_after_parse(path):
        audit['note'] = 'changed after manifest parse, before CFA verification'
        core.atomic_json(directory/'audit.json', audit)
        return original(path)
    monkeypatch.setattr(raw_audit, 'read_dng_tags', mutate_after_parse)
    with pytest.raises(ValueError, match='哈希'):
        raw_develop.develop_raw(directory, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_metadata_mismatch_reports_versions_and_reaudit(fixture_audit, tmp_path, monkeypatch):
    directory, audit, _ = fixture_audit
    audit['versions'] = {'tifffile': 'older-reader'}
    core.atomic_json(directory/'audit.json', audit)
    monkeypatch.setattr(raw_audit, 'read_dng_tags', lambda _: {'status': 'partial'})
    with pytest.raises(ValueError, match='older-reader') as error:
        raw_develop.develop_raw(directory, tmp_path/'out')
    assert tifffile.__version__ in str(error.value) and '另存重新检查' in str(error.value)
    assert not (tmp_path/'out').exists()


def test_manifest_write_failure_does_not_return_success(fixture_audit, tmp_path, monkeypatch):
    directory, _, _ = fixture_audit
    original = core.atomic_json
    def reject_final(path, record):
        if Path(path).name == 'development.json' and record['state'] == 'succeeded':
            raise OSError('injected full disk when publishing result')
        return original(path, record)
    monkeypatch.setattr(core, 'atomic_json', reject_final)
    with pytest.raises(OSError, match='injected full disk'):
        raw_develop.develop_raw(directory, tmp_path/'out')
    manifests = list((tmp_path/'out').glob('raw-develop-*/development.json'))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding='utf-8'))['state'] == 'running'


def test_provenance_names_the_reused_cfa_evidence(fixture_audit, tmp_path):
    directory, _, _ = fixture_audit
    result = raw_develop.develop_raw(directory, tmp_path/'out')
    assert result['state'] == 'succeeded'
    assert result['processing']['cfa_source'] == 'hash_verified_audit_raw_codes_not_redecoded'
    assert result['processing']['integrity_scope'] == 'local_consistency_hashes_not_signatures'
