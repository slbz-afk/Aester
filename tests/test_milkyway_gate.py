"""Declaration-boundary tests; no fixture claims to identify a real galaxy."""
import copy
import json
from pathlib import Path
import sys
import asyncio

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from starstack.milkyway_gate import VERSION, assess_evidence
from starstack import cli, core, milkyway, photographic_job as job
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


SOURCE = 'ab' * 32
MASK = '19' * 32


def evidence(**updates):
    return {
        'version': VERSION,
        'status': 'present',
        'method': 'human-review',
        'reviewer': 'scene reviewer',
        'rationale': '在未增强源图及指定区域看到连续银河带；不是仅依据蒙版。',
        'source_sha256': SOURCE,
        'milkyway_mask_sha256': MASK,
    } | updates


def assess(item=None, **kwargs):
    return assess_evidence(item, **({'source_sha256': SOURCE, 'mask_sha256': MASK} | kwargs))


def test_no_declaration_cannot_enable_or_certify_absence():
    result = assess()
    assert result['enabled'] is False
    assert (result['status'], result['decision']) == ('unknown', 'needs_review')
    assert result['evidence'] is None
    assert isinstance(result['reason'], str) and result['reason']


@pytest.mark.parametrize('method,status,expected', [
    ('human-review', 'present', (True, 'present', 'apply')),
    ('human-review', 'absent', (False, 'absent', 'skipped')),
    ('human-review', 'unknown', (False, 'unknown', 'needs_review')),
    ('unvalidated-auto', 'present', (False, 'unknown', 'needs_review')),
    ('unvalidated-auto', 'absent', (False, 'unknown', 'needs_review')),
    ('unvalidated-auto', 'unknown', (False, 'unknown', 'needs_review')),
])
def test_trust_matrix_preserves_claim_and_does_not_promote_unvalidated_method(method, status, expected):
    original = evidence(method=method, status=status)
    result = assess(original)
    assert (result['enabled'], result['status'], result['decision']) == expected
    assert result['evidence']['status'] == status
    assert result['evidence']['method'] == method
    assert set(result) == {'enabled', 'status', 'decision', 'reason', 'evidence'}
    assert json.loads(json.dumps(result, allow_nan=False, ensure_ascii=False)) == result


@pytest.mark.parametrize('field', ['source_sha256', 'milkyway_mask_sha256'])
@pytest.mark.parametrize('status', ['present', 'absent', 'unknown'])
def test_wrong_input_identity_is_rejected_for_all_claims(field, status):
    with pytest.raises(ValueError, match='different source or mask'):
        assess(evidence(status=status, **{field: 'cc' * 32}))


def test_hash_case_is_normalized_without_mutating_caller():
    original = evidence(source_sha256=SOURCE.upper(), reviewer='  reviewer  ', rationale='  依据  ')
    snapshot = copy.deepcopy(original)
    result = assess(original, source_sha256=SOURCE.upper())
    assert result['enabled'] is True
    assert result['evidence']['source_sha256'] == SOURCE
    assert result['evidence']['reviewer'] == 'reviewer'
    assert result['evidence']['rationale'] == '依据'
    assert original == snapshot


def test_input_and_returned_evidence_are_independently_owned():
    original = evidence()
    result = assess(original)
    original['status'] = 'absent'
    original['rationale'] = 'later mutation'
    assert result['enabled'] is True
    assert result['evidence']['status'] == 'present'
    assert result['evidence']['rationale'] != original['rationale']
    result['evidence']['reviewer'] = 'mutated output'
    assert original['reviewer'] == 'scene reviewer'
    again = assess(original)
    assert again['enabled'] is False and again['status'] == 'absent'


@pytest.mark.parametrize('bad', [None, True, 12, 1.2, b'ab' * 32, '', 'a' * 63,
                                 'a' * 65, 'g' * 64, ' ' + 'a' * 63, 'a' * 63 + '\n'])
@pytest.mark.parametrize('field', ['source_sha256', 'mask_sha256'])
def test_current_input_hashes_are_required_even_without_evidence(field, bad):
    with pytest.raises(ValueError):
        assess(**{field: bad})


@pytest.mark.parametrize('field', ['source_sha256', 'milkyway_mask_sha256'])
@pytest.mark.parametrize('bad', [None, True, 123, [], 'x' * 64, 'a' * 63, 'a' * 65])
def test_evidence_hashes_are_strict(field, bad):
    with pytest.raises(ValueError):
        assess(evidence(**{field: bad}))


@pytest.mark.parametrize('field', ['reviewer', 'rationale'])
@pytest.mark.parametrize('bad', ['', ' \n\t', None, True, 3, float('nan'), {}, [], '\ud800'])
def test_confirmation_text_cannot_be_missing_empty_nonstr_or_unencodable(field, bad):
    with pytest.raises(ValueError):
        assess(evidence(**{field: bad}))


@pytest.mark.parametrize('field,bad', [
    ('status', True), ('status', 'confirmed'), ('status', 'Present'), ('status', ['present']),
    ('method', 'auto'), ('method', None), ('method', []), ('method', 'human-review '),
    ('version', None), ('version', 1), ('version', 'milkyway-subject-evidence-0'),
])
def test_enums_and_schema_are_not_coerced(field, bad):
    with pytest.raises(ValueError):
        assess(evidence(**{field: bad}))


@pytest.mark.parametrize('field', list(evidence()))
def test_every_schema_field_is_required(field):
    item = evidence()
    del item[field]
    with pytest.raises(ValueError):
        assess(item)


@pytest.mark.parametrize('extra', [{'note': 'extra'}, {'score': float('nan')},
                                  {'nested': {'mutable': []}}, {1: 'nonstring key'}])
def test_opaque_extra_data_cannot_escape_into_result(extra):
    with pytest.raises(ValueError):
        assess(evidence() | extra)


@pytest.mark.parametrize('item', [[], True, 1, 'present', tuple(evidence().items())])
def test_evidence_requires_plain_dictionary(item):
    with pytest.raises(ValueError):
        assess(item)


def test_custom_string_and_dict_objects_are_not_accepted_as_trusted_serializable_values():
    class Text(str):
        pass
    class Record(dict):
        pass
    for field in ['status', 'method', 'version', 'reviewer', 'rationale', 'source_sha256']:
        item = evidence()
        item[field] = Text(item[field])
        with pytest.raises(ValueError):
            assess(item)
    with pytest.raises(ValueError):
        assess(Record(evidence()))
    item = evidence()
    value = item.pop('version')
    item[Text('version')] = value
    with pytest.raises(ValueError):
        assess(item)


def entry_inputs(tmp_path, *, paired=False, separated=False, claim=None, method='human-review'):
    """Real TIFF boundary, synthetic stars/trail and nonzero editing permission."""
    h, w = 160, 192
    y, x = np.mgrid[:h, :w]
    rgb = np.rint((.18+.02*np.sin(x/24)+.01*np.cos(y/17))[..., None]
                  *np.array([.95, 1., .87])*65535).astype(np.uint16)
    rgb[45:48, 112:115] += 2000
    base = rgb.copy()
    rgb[24:125, 86:88] += 1100
    rgb[24:125, 88:89] -= 300
    sky = np.full((h, w), 255, np.uint8); sky[145:] = 0; sky[144] = 128
    protect = np.zeros((h, w), np.uint8); protect[20:130, 84:91] = 255
    mask = np.zeros((h, w), np.uint8); mask[:, 35:155] = 255
    files = {}
    for name, data in [('sky', sky), ('protected', protect), ('milkyway', mask)]:
        files[name] = tmp_path/(name+'.png')
        Image.fromarray(data).save(files[name])
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    md = dict(scene_linear=False, display_encoding_applied=True,
              roworder='TOP-DOWN', color_space='encoded_sRGB')
    def write(name, data, extra=None):
        path = tmp_path/(name+'.tif')
        tifffile.imwrite(path, data, photometric='rgb', metadata=md | (extra or {}),
            extratags=[(34675, 'B', len(icc), icc, False)])
        return path
    files['source'] = write('source', rgb)
    options = {}
    if paired:
        files['base'] = write('base', base, {'transient_background_for_sha256': core._sha(files['source'])})
        options['transient_background'] = files['base']
    diffuse = (base if paired else rgb).copy()
    diffuse[45:48, 112:115] -= 2000
    if separated:
        files['diffuse'] = write('diffuse', diffuse, {
            'star_separation_for_sha256': core._sha(files['base'] if paired else files['source']),
            'sky_mask_sha256': core._sha(files['sky']),
            'protected_mask_sha256': core._sha(files['protected'])})
        options['starless_background'] = files['diffuse']
    if claim is not None:
        files['evidence'] = tmp_path/'subject-evidence.json'
        item = evidence(status=claim, method=method, reviewer='synthetic integration fixture',
            rationale='Test declaration only; not actual human review or classification.',
            source_sha256=core._sha(files['source']), milkyway_mask_sha256=core._sha(files['milkyway']))
        files['evidence'].write_text(json.dumps(item), encoding='utf-8')
        options['milkyway_evidence'] = files['evidence']
    return files, options, {'source': rgb, 'base': base, 'diffuse': diffuse}


def run_entry(tmp_path, files, options):
    return job.finish_milkyway(files['source'], tmp_path/'out', files['sky'], files['protected'],
        files['milkyway'], brightness=.06, strength=.6, **options)


@pytest.mark.parametrize('paired,separated', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('claim,method,decision', [
    (None, 'human-review', 'needs_review'),
    ('absent', 'human-review', 'skipped'),
    ('unknown', 'human-review', 'needs_review'),
    ('present', 'unvalidated-auto', 'needs_review'),
    ('absent', 'unvalidated-auto', 'needs_review'),
])
def test_real_entry_unconfirmed_nonzero_mask_and_lift_are_exact_bypass(tmp_path, paired, separated, claim, method, decision):
    files, options, arrays = entry_inputs(tmp_path, paired=paired, separated=separated, claim=claim, method=method)
    hashes = {name: core._sha(path) for name, path in files.items()}
    result = run_entry(tmp_path, files, options)
    assert result['state'] == 'succeeded', result
    assert result['subject_gate']['enabled'] is False
    assert result['subject_gate']['decision'] == decision
    assert result['operation_status'] == decision
    assert result['quality_status'] == 'not_applied'
    assert result['requested_recipe']['brightness'] == .06
    assert result['requested_recipe']['strength'] == .6 and result['recipe']['strength'] == 0
    expected = [('milkyway_srgb16.tif', arrays['source'])]
    if paired: expected.append(('milkyway_background_srgb16.tif', arrays['base']))
    if separated: expected.append(('milkyway_diffuse_srgb16.tif', arrays['diffuse']))
    for name, original in expected:
        actual = tifffile.imread(result['artifacts'][name]['path'])
        np.testing.assert_array_equal(actual, original)
    assert result['statistics']['changed_pixels'] == 0
    assert hashes == {name: core._sha(path) for name, path in files.items()}
    saved = json.loads((Path(result['processing_directory'])/'result.json').read_text('utf-8'))
    assert saved == result


@pytest.mark.parametrize('paired,separated', [(False, False), (True, False), (False, True), (True, True)])
def test_explicit_synthetic_confirmation_retains_the_enabled_path(tmp_path, paired, separated):
    files, options, arrays = entry_inputs(tmp_path, paired=paired, separated=separated, claim='present')
    result = run_entry(tmp_path, files, options)
    assert result['state'] == 'succeeded', result
    assert result['subject_gate']['enabled'] is True and result['operation_status'] == 'apply'
    assert result['recipe']['strength'] == .6
    actual = tifffile.imread(result['artifacts']['milkyway_srgb16.tif']['path'])
    assert np.any(actual != arrays['source'])


@pytest.mark.parametrize('field', ['source_sha256', 'milkyway_mask_sha256'])
def test_real_entry_rejects_wrong_confirmation_binding_before_publication(tmp_path, field):
    files, options, _ = entry_inputs(tmp_path, claim='present')
    item = json.loads(files['evidence'].read_text('utf-8')); item[field] = 'e1'*32
    files['evidence'].write_text(json.dumps(item), encoding='utf-8')
    with pytest.raises(ValueError, match='different source or mask'):
        run_entry(tmp_path, files, options)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('phase', ['read', 'render', 'publication'])
def test_confirmation_file_cannot_change_between_binding_and_publication(tmp_path, monkeypatch, phase):
    files, options, _ = entry_inputs(tmp_path, claim='present')
    def tamper():
        with files['evidence'].open('ab') as stream: stream.write(b' changed')
    if phase == 'read':
        original = Path.read_bytes
        def read(path):
            if path == files['evidence']: tamper()
            return original(path)
        monkeypatch.setattr(Path, 'read_bytes', read)
        with pytest.raises(ValueError, match='依据在读取期间改变'):
            run_entry(tmp_path, files, options)
        assert not (tmp_path/'out').exists()
    else:
        if phase == 'render':
            original = milkyway.render
            def render(*args, **kwargs):
                result = original(*args, **kwargs); tamper(); return result
            monkeypatch.setattr(milkyway, 'render', render)
        else:
            original = core.atomic_json
            def save(path, value, **kwargs):
                result = original(path, value, **kwargs)
                if value['state'] == 'succeeded': tamper()
                return result
            monkeypatch.setattr(core, 'atomic_json', save)
        result = run_entry(tmp_path, files, options)
        assert result['state'] == 'failed' and result['artifacts'] == {}
        assert '输入在处理期间改变' in result['error']
        assert not list((tmp_path/'out').glob('milkyway-*'))


def test_cli_routes_confirmation_only_when_explicit(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(job, 'finish_milkyway', lambda *a, **kw: calls.append((a, kw)) or {'state': 'succeeded'})
    base = ['starstack', 'finish-milkyway', '--source', 's', '--output', 'o', '--sky-mask', 'k',
            '--protected-mask', 'p', '--milkyway-mask', 'm']
    monkeypatch.setattr(sys, 'argv', base)
    assert cli.main() == 0
    assert 'milkyway_evidence' not in calls[-1][1]
    monkeypatch.setattr(sys, 'argv', base+['--milkyway-evidence', 'e.json'])
    assert cli.main() == 0
    assert calls[-1][1]['milkyway_evidence'] == 'e.json'
    capsys.readouterr()


def test_mcp_service_bounds_confirmation_path_and_routes_valid_one(tmp_path, monkeypatch):
    root = tmp_path/'out'; root.mkdir(); target = root/'job'; target.mkdir()
    paths = [root/f'{i}.tif' for i in range(4)]
    for path in paths: path.touch()
    inside = root/'evidence.json'; inside.touch()
    outside = tmp_path/'outside.json'; outside.touch()
    result = dict(state='succeeded', processing_directory=str(target), artifacts={})
    calls = []
    monkeypatch.setattr(job, 'finish_milkyway', lambda *a, **kw: calls.append((a, kw)) or result)
    service = WorkspaceTools(tmp_path/'raw', root)
    with pytest.raises(AccessDenied):
        service.finish_milkyway(*map(str, paths), milkyway_evidence=str(outside))
    assert calls == []
    assert service.finish_milkyway(*map(str, paths), milkyway_evidence=str(inside)) == result
    assert calls[0][1]['milkyway_evidence'] == str(inside)


def test_registered_async_mcp_confirmation_schema_and_routing(tmp_path):
    pytest.importorskip('mcp')
    calls = []
    class Service(WorkspaceTools):
        def finish_milkyway(self, *args, **kwargs):
            calls.append((args, kwargs)); return {'state': 'succeeded'}
    server = create_server(Service(tmp_path/'raw', tmp_path/'out'))
    async def run():
        tool = next(item for item in await server.list_tools() if item.name == 'finish_milkyway')
        assert 'milkyway_evidence' in tool.inputSchema['properties']
        assert 'milkyway_evidence' not in tool.inputSchema['required']
        params = dict(source='s', sky_mask='k', protected_mask='p', milkyway_mask='m', milkyway_evidence='e.json')
        await server.call_tool('finish_milkyway', params)
        assert calls[0][1] == {'milkyway_evidence': 'e.json'}
        for bad in (True, 1, ['e.json'], {}):
            with pytest.raises(Exception):
                await server.call_tool('finish_milkyway', params | {'milkyway_evidence': bad})
        assert len(calls) == 1
    asyncio.run(run())
