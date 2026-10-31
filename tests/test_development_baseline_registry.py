"""The retained development set must not silently shrink when full DNGs retire."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_assets():
    spec = importlib.util.spec_from_file_location('development_assets_test', ROOT/'scripts/development_assets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_preserves_exact_48_identity_and_sampling():
    assets = load_assets()
    rows = assets.BASELINE['frames']
    assert len(rows) == assets.BASELINE['frame_count'] == 48
    assert [r['original_name'] for r in rows] == [f'DSC_{i}.NEF' for i in range(2206, 2348, 3)]
    assert len({r['sha256'] for r in rows}) == 48
    assert all(len(r['sha256']) == 64 and r['bytes'] > 0 for r in rows)
    assert all('z8-meteor-20260916' not in r['dng_path'] for r in rows)
    assert rows[21]['original_name'] == 'DSC_2269.NEF'
    assert rows[21]['dng_path'].endswith('/frame00022.dng')


def test_lookup_uses_original_identity_and_fails_if_retired_or_missing(tmp_path, monkeypatch):
    assets = load_assets()
    monkeypatch.setattr(assets, 'ROOT', tmp_path)
    row = assets.BASELINE['frames'][21]
    with pytest.raises(FileNotFoundError):
        assets.dng_for_original('DSC_2269.dng')
    with pytest.raises(ValueError, match='not in'):
        assets.dng_for_original('DSC_2204.NEF')
    target = tmp_path/row['dng_path']
    target.parent.mkdir(parents=True)
    target.write_bytes(b'changed')
    with pytest.raises(FileNotFoundError):
        assets.dng_for_original('DSC_2269.NEF')


def test_historical_fixture_keeps_observed_misses_without_pixel_dependency():
    r = json.loads((ROOT/'docs/testdata/HISTORICAL_TRANSIENT_CLASSIFICATION_20260925.json').read_text(encoding='utf-8'))
    assert len(r['candidates']) == r['source_candidates'] == 1206
    assert {'light_00009-8', 'light_00074-2', 'light_00022-1'} <= {c['id'] for c in r['candidates']}
    assert all(set(c) == {'id', 'measurement', 'evidence', 'suggestion'} for c in r['candidates'])
    assert 'no pixel data' in r['scope']


def test_retired_starless_case_cannot_be_staged_as_active():
    spec = importlib.util.spec_from_file_location('stage_for_retirement_test', ROOT/'scripts/stage_starless_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    doc = {'cases': {'old': {'lifecycle': {'status': 'retired'}}}}
    with pytest.raises(ValueError, match='retired'):
        module.plan_items(doc, {'old'}, ['estimate'])
    assert module.plan_items(doc, {'another'}, ['estimate']) == []


def test_known_trail_uses_source_hash_after_48_input_reordering():
    spec = importlib.util.spec_from_file_location('trail_accept_retirement_test', ROOT/'scripts/s4_accept_transients.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asset = load_assets().BASELINE['frames'][21]
    job = {'inputs': [{'path': 'renamed/frame00022.dng', 'sha256': 'different'},
                      {'path': 'renamed/frame00002.dng', 'sha256': asset['sha256']}]}
    assert module._stem_of(job, 'DSC_2269.dng') == 'light_00002'
