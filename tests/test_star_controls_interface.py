"""Public full-range controls and suggestion routing, independent of pixel math."""
import asyncio
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from starstack import cli
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


LEGACY = dict(strength=.55, bright_start=.10, bright_end=.45,
              milkyway_mask=None, milkyway_evidence=None,
              milkyway_relief=.5, bright_radius=None)


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(calls=[], result={'state': 'succeeded'}, error=None)
    module = ModuleType('starstack.star_reduction')

    def record(name):
        def call(*args, **kwargs):
            state.calls.append((name, args, kwargs))
            if state.error:
                raise state.error
            return state.result
        return call

    module.reduce_stars = record('reduce')
    module.suggest_star_reduction = record('suggest')
    monkeypatch.setitem(sys.modules, 'starstack.star_reduction', module)
    return state


@pytest.mark.parametrize('strength,offset', [(0, -1), (.81, -.23), (1, 1)])
def test_cli_full_range_is_explicit_and_routes_endpoints(backend, monkeypatch, capsys, strength, offset):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'layers.json',
        '--output', 'out', '--background-strength', str(strength), '--milkyway-offset', str(offset)])
    assert cli.main() == 0
    assert backend.calls == [('reduce', ('layers.json', 'out'), LEGACY | {
        'background_strength': strength, 'milkyway_offset': offset})]
    assert json.loads(capsys.readouterr().out)['state'] == 'succeeded'


def test_cli_background_uses_new_offset_default(backend, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'layers.json',
        '--background-strength', '.9'])
    assert cli.main() == 0
    assert backend.calls[-1][2] == LEGACY | {'background_strength': .9, 'milkyway_offset': -.2}
    capsys.readouterr()


@pytest.mark.parametrize('option,bad', [
    ('--background-strength', '-.01'), ('--background-strength', '1.01'),
    ('--background-strength', 'nan'), ('--background-strength', 'inf'),
    ('--background-strength', 'bad'), ('--milkyway-offset', '-1.01'),
    ('--milkyway-offset', '1.01'), ('--milkyway-offset', 'nan'),
    ('--milkyway-offset', 'inf'), ('--milkyway-offset', 'bad'),
])
def test_cli_invalid_new_control_never_reaches_core(backend, monkeypatch, option, bad):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'layers.json',
        f'{option}={bad}'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not backend.calls


def test_cli_suggestion_routes_without_reduction(backend, monkeypatch, capsys):
    backend.result = {'state': 'succeeded', 'recommendation': {'available': True, 'background_strength': .83,
        'milkyway_offset': -.16, 'milkyway_strength': .67, 'reason': 'synthetic dense stars'},
        'metrics': {'density': .22}, 'processing_directory': 'out/suggestion'}
    monkeypatch.setattr(sys, 'argv', ['starstack', 'suggest-stars', '--separation-result', 'layers.json',
        '--output', 'out', '--milkyway-mask', 'weight.tif', '--milkyway-evidence', 'present.json'])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == backend.result
    assert backend.calls == [('suggest', ('layers.json', 'out'),
        {'milkyway_mask': 'weight.tif', 'milkyway_evidence': 'present.json'})]


def test_cli_suggestion_needs_record_and_has_no_render_options(backend, monkeypatch):
    for arguments in [[], ['--separation-result', 'layers.json', '--background-strength', '.8']]:
        monkeypatch.setattr(sys, 'argv', ['starstack', 'suggest-stars'] + arguments)
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
    assert not backend.calls


@pytest.mark.parametrize('error', [False, True])
def test_cli_suggestion_failure_is_not_success(backend, monkeypatch, capsys, error):
    backend.result = {'state': 'failed', 'error': 'invalid layers'}
    backend.error = ValueError('invalid layers') if error else None
    monkeypatch.setattr(sys, 'argv', ['starstack', 'suggest-stars', '--separation-result', 'layers.json'])
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)['error'] == 'invalid layers'


@pytest.fixture
def workspace(tmp_path, backend):
    root = tmp_path/'outputs'; root.mkdir()
    for name in ['layers.json', 'mask.tif', 'evidence.json']:
        (root/name).touch()
    target = root/'suggestion'; target.mkdir()
    report = target/'result.json'; report.touch()
    backend.result = {'state': 'succeeded', 'processing_directory': str(target),
        'recommendation': {'available': True, 'background_strength': .8, 'milkyway_offset': -.2,
                           'milkyway_strength': .6, 'reason': 'test'},
        'metrics': {}, 'artifacts': {'result.json': {'path': str(report)}}}
    outside = tmp_path/'outside.json'; outside.touch()
    return SimpleNamespace(root=root, target=target, report=report, outside=outside,
                           service=WorkspaceTools(tmp_path/'raw', root))


def test_mcp_service_suggestion_routes_bounded_paths_and_owned_root(workspace, backend):
    assert workspace.service.suggest_star_reduction('layers.json', 'mask.tif', 'evidence.json') == backend.result
    assert backend.calls == [('suggest', (str(workspace.root/'layers.json'), str(workspace.root)), {
        'milkyway_mask': str(workspace.root/'mask.tif'),
        'milkyway_evidence': str(workspace.root/'evidence.json'), 'allowed_root': workspace.root})]


@pytest.mark.parametrize('strength,offset', [(0, -1), (.8, -.2), (1, 1)])
def test_mcp_service_new_controls_reach_core(workspace, backend, strength, offset):
    workspace.service.reduce_stars('layers.json', background_strength=strength, milkyway_offset=offset)
    assert backend.calls == [('reduce', (str(workspace.root/'layers.json'), str(workspace.root)), LEGACY | {
        'background_strength': strength, 'milkyway_offset': offset, 'allowed_root': workspace.root})]


@pytest.mark.parametrize('field,bad', [(field, bad) for field in ['background_strength', 'milkyway_offset']
    for bad in [True, '.5', float('nan'), float('inf'), -1.01, 1.01]] + [('background_strength', -.01), ('milkyway_offset', None)])
def test_mcp_service_rejects_invalid_controls_before_backend(workspace, backend, field, bad):
    with pytest.raises(ValueError):
        workspace.service.reduce_stars('layers.json', **{field: bad})
    assert not backend.calls


@pytest.mark.parametrize('field', ['separation_result', 'milkyway_mask', 'milkyway_evidence'])
@pytest.mark.parametrize('style', ['absolute', 'traversal', 'missing'])
def test_mcp_suggestion_input_paths_are_bounded(workspace, backend, field, style):
    options = dict(separation_result='layers.json', milkyway_mask='mask.tif', milkyway_evidence='evidence.json')
    options[field] = {'absolute': str(workspace.outside), 'traversal': '../outside.json', 'missing': 'gone.json'}[style]
    with pytest.raises((AccessDenied, FileNotFoundError)):
        workspace.service.suggest_star_reduction(**options)
    assert not backend.calls


@pytest.mark.parametrize('field', ['processing_directory', 'artifact_outside', 'artifact_traversal'])
def test_mcp_suggestion_result_paths_remain_owned(workspace, backend, field):
    if field == 'processing_directory':
        backend.result[field] = str(workspace.outside.parent)
    else:
        backend.result['artifacts']['result.json']['path'] = (str(workspace.outside) if field == 'artifact_outside' else '../layers.json')
    with pytest.raises(AccessDenied):
        workspace.service.suggest_star_reduction('layers.json')


def test_registered_tools_keep_old_calls_and_expose_strict_full_range(tmp_path):
    pytest.importorskip('mcp')
    calls = []

    class Service(WorkspaceTools):
        def reduce_stars(self, *args, **kwargs):
            calls.append(('reduce', args, kwargs))
            return {'state': 'succeeded'}

        def suggest_star_reduction(self, *args, **kwargs):
            calls.append(('suggest', args, kwargs))
            return {'state': 'succeeded', 'recommendation': {'available': True}}

    server = create_server(Service(tmp_path/'raw', tmp_path/'outputs'))

    async def exercise():
        tools = {tool.name: tool for tool in await server.list_tools()}
        reduction = tools['reduce_stars']
        props = reduction.inputSchema['properties']
        numeric = next(option for option in props['background_strength']['anyOf'] if option.get('type') == 'number')
        assert numeric['minimum'] == 0 and numeric['maximum'] == 1
        assert props['background_strength']['default'] is None
        assert props['milkyway_offset']['minimum'] == -1 and props['milkyway_offset']['maximum'] == 1
        assert props['milkyway_offset']['default'] == -.2
        assert not {'output_root', 'allowed_root', 'provider'} & props.keys()
        await server.call_tool('reduce_stars', {'separation_result': 'layers.json'})
        assert calls[-1] == ('reduce', ('layers.json',), LEGACY)
        await server.call_tool('reduce_stars', {'separation_result': 'layers.json', 'background_strength': None})
        assert calls[-1] == calls[-2]
        for strength, offset in [(0, -1), (.88, -.17), (1, 1)]:
            await server.call_tool('reduce_stars', {'separation_result': 'layers.json',
                'background_strength': strength, 'milkyway_offset': offset})
            assert calls[-1] == ('reduce', ('layers.json',), LEGACY | {'background_strength': strength, 'milkyway_offset': offset})
        for field, bad in [(field, bad) for field in ['background_strength', 'milkyway_offset']
                for bad in [True, '.5', float('nan'), float('inf'), -1.01, 1.01]] + [('background_strength', -.01), ('milkyway_offset', None)]:
            with pytest.raises(Exception):
                await server.call_tool('reduce_stars', {'separation_result': 'layers.json', field: bad})
        assert len(calls) == 5
        suggestion = tools['suggest_star_reduction']
        assert suggestion.inputSchema['required'] == ['separation_result']
        assert set(suggestion.inputSchema['properties']) == {'separation_result', 'milkyway_mask', 'milkyway_evidence'}
        assert '不自动渲染' in suggestion.description and '不得伪造' in suggestion.description
        await server.call_tool('suggest_star_reduction', {'separation_result': 'layers.json'})
        assert calls[-1] == ('suggest', ('layers.json',), {'milkyway_mask': None, 'milkyway_evidence': None})
        await server.call_tool('suggest_star_reduction', {'separation_result': 'layers.json',
            'milkyway_mask': 'mask.tif', 'milkyway_evidence': 'present.json'})
        assert calls[-1] == ('suggest', ('layers.json',), {'milkyway_mask': 'mask.tif', 'milkyway_evidence': 'present.json'})
        for field in ['separation_result', 'milkyway_mask', 'milkyway_evidence']:
            for bad in [True, 123, ['file']]:
                with pytest.raises(Exception):
                    await server.call_tool('suggest_star_reduction', {'separation_result': 'layers.json', field: bad})
        with pytest.raises(Exception):
            await server.call_tool('suggest_star_reduction', {})
        assert len(calls) == 7

    asyncio.run(exercise())
