"""CLI/MCP routing and ownership checks, separate from the numerical kernel."""
import asyncio
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from starstack import cli
from starstack.mcp_server import AccessDenied, WorkspaceTools, create_server


DEFAULTS = dict(strength=.55, bright_start=.10, bright_end=.45,
                milkyway_mask=None, milkyway_evidence=None,
                milkyway_relief=.5, bright_radius=None)


@pytest.fixture
def backend(monkeypatch):
    """Only routing is mocked: the core has its own file/numerical tests."""
    calls = []
    module = ModuleType('starstack.star_reduction')
    state = SimpleNamespace(result={'state': 'succeeded'}, calls=calls, error=None)

    def reduce_stars(*args, **kwargs):
        calls.append((args, kwargs))
        if state.error is not None:
            raise state.error
        return state.result

    module.reduce_stars = reduce_stars
    monkeypatch.setitem(sys.modules, 'starstack.star_reduction', module)
    return state


def test_cli_requires_layer_record_and_routes_defaults(backend, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not backend.calls
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'layers/result.json'])
    assert cli.main() == 0
    assert backend.calls == [(('layers/result.json', 'outputs'), DEFAULTS)]
    assert json.loads(capsys.readouterr().out) == {'state': 'succeeded'}


def test_cli_routes_every_explicit_control(backend, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'result.json',
        '--output', 'new', '--strength', '.7', '--bright-start', '.12', '--bright-end', '.5',
        '--milkyway-mask', 'mask.png', '--milkyway-evidence', 'evidence.json',
        '--milkyway-relief', '.6', '--bright-radius', '0'])
    assert cli.main() == 0
    assert backend.calls == [(('result.json', 'new'), dict(strength=.7, bright_start=.12,
        bright_end=.5, milkyway_mask='mask.png', milkyway_evidence='evidence.json',
        milkyway_relief=.6, bright_radius=0))]
    capsys.readouterr()


@pytest.mark.parametrize('option,value', [('--strength', 'bad'), ('--bright-start', 'bad'),
    ('--bright-end', 'bad'), ('--milkyway-relief', 'bad'), ('--bright-radius', '1.5')])
def test_cli_rejects_non_numeric_controls_before_backend(backend, monkeypatch, option, value):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'result.json', option, value])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not backend.calls


def test_cli_help_states_domain_and_function(backend, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--help'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    text = capsys.readouterr().out
    for phrase in ['显示域', '测光星等', '几何去卷积', '--milkyway-evidence']:
        assert phrase in text
    assert not backend.calls


@pytest.mark.parametrize('error', [False, True])
def test_cli_reports_failed_work_without_success_exit(backend, monkeypatch, capsys, error):
    monkeypatch.setattr(sys, 'argv', ['starstack', 'reduce-stars', '--separation-result', 'result.json'])
    if error:
        backend.error = ValueError('bad layer record')
    else:
        backend.result = {'state': 'failed', 'error': 'bad layer record'}
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)['error'] == 'bad layer record'


@pytest.fixture
def workspace(tmp_path, backend):
    root = tmp_path/'outputs'; root.mkdir()
    files = {}
    for name in ['record.json', 'mask.png', 'evidence.json']:
        files[name] = root/name
        files[name].touch()
    target = root/'star-reduction-example'; target.mkdir()
    artifact = target/'star_reduced_srgb16.tif'; artifact.touch()
    backend.result = {'state': 'succeeded', 'processing_directory': str(target),
        'artifacts': {'star_reduced_srgb16.tif': {'path': str(artifact)}}}
    outside = tmp_path/'outside.json'; outside.touch()
    return SimpleNamespace(root=root, files=files, target=target, artifact=artifact, outside=outside,
        service=WorkspaceTools(tmp_path/'raw', root))


def test_mcp_service_binds_layer_references_and_routes_defaults(workspace, backend):
    assert workspace.service.reduce_stars('record.json') == backend.result
    assert backend.calls == [((str(workspace.files['record.json']), str(workspace.root)),
                             DEFAULTS | {'allowed_root': workspace.root})]


def test_mcp_service_routes_all_optional_paths_and_controls(workspace, backend):
    options = dict(strength=.3, bright_start=.15, bright_end=.7, milkyway_relief=.8,
                   bright_radius=0, milkyway_mask='mask.png', milkyway_evidence='evidence.json')
    assert workspace.service.reduce_stars('record.json', **options) == backend.result
    assert backend.calls == [((str(workspace.files['record.json']), str(workspace.root)),
        options | {'milkyway_mask': str(workspace.files['mask.png']),
                   'milkyway_evidence': str(workspace.files['evidence.json']),
                   'allowed_root': workspace.root})]


@pytest.mark.parametrize('field', ['separation_result', 'milkyway_mask', 'milkyway_evidence'])
@pytest.mark.parametrize('style', ['absolute', 'traversal'])
def test_mcp_checks_all_supplied_input_paths_before_core(workspace, backend, field, style):
    params = dict(separation_result='record.json', milkyway_mask='mask.png', milkyway_evidence='evidence.json')
    params[field] = str(workspace.outside) if style == 'absolute' else '../outside.json'
    with pytest.raises(AccessDenied):
        workspace.service.reduce_stars(**params)
    assert not backend.calls


@pytest.mark.parametrize('field', ['separation_result', 'milkyway_mask', 'milkyway_evidence'])
def test_mcp_requires_real_supplied_paths(workspace, backend, field):
    params = dict(separation_result='record.json', milkyway_mask='mask.png', milkyway_evidence='evidence.json')
    params[field] = 'missing.file'
    with pytest.raises(FileNotFoundError):
        workspace.service.reduce_stars(**params)
    assert not backend.calls


@pytest.mark.parametrize('field', ['processing_directory', 'artifact'])
def test_mcp_cannot_publish_results_outside_owned_processing_directory(workspace, backend, field):
    if field == 'processing_directory':
        backend.result[field] = str(workspace.outside.parent)
    else:
        backend.result['artifacts']['star_reduced_srgb16.tif']['path'] = str(workspace.files['record.json'])
    with pytest.raises(AccessDenied):
        workspace.service.reduce_stars('record.json')


def test_mcp_rejects_parent_traversal_in_returned_artifact(workspace, backend):
    backend.result['artifacts']['star_reduced_srgb16.tif']['path'] = '../record.json'
    with pytest.raises(AccessDenied):
        workspace.service.reduce_stars('record.json')


def test_registered_mcp_tool_exposes_only_host_bounded_interface_and_strict_controls(tmp_path):
    pytest.importorskip('mcp')
    calls = []

    class Service(WorkspaceTools):
        def reduce_stars(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {'state': 'succeeded'}

    server = create_server(Service(tmp_path/'raw', tmp_path/'outputs'))

    async def exercise():
        tool = next(item for item in await server.list_tools() if item.name == 'reduce_stars')
        schema = tool.inputSchema
        assert schema['required'] == ['separation_result']
        assert not {'output_root', 'allowed_root', 'provider'} & schema['properties'].keys()
        assert schema['properties']['strength']['minimum'] == 0
        assert schema['properties']['strength']['maximum'] == 1
        assert all(text in tool.description for text in ['显示域', '测光星等', '几何去卷积', '人工确认'])
        await server.call_tool('reduce_stars', {'separation_result': 'result.json'})
        assert calls == [(('result.json',), DEFAULTS)]
        explicit = DEFAULTS | dict(strength=0., bright_start=.12, bright_end=.8,
            milkyway_mask='mask.png', milkyway_evidence='evidence.json', milkyway_relief=1., bright_radius=0)
        await server.call_tool('reduce_stars', {'separation_result': 'result.json'} | explicit)
        assert calls[-1] == (('result.json',), explicit)
        for field in ['strength', 'bright_start', 'bright_end', 'milkyway_relief']:
            for bad in [True, '.5', float('nan'), float('inf'), -.01, 1.01]:
                with pytest.raises(Exception):
                    await server.call_tool('reduce_stars', {'separation_result': 'result.json', field: bad})
        for bad in [True, '1', 1.5, -1, 33, float('nan')]:
            with pytest.raises(Exception):
                await server.call_tool('reduce_stars', {'separation_result': 'result.json', 'bright_radius': bad})
        for field in ['milkyway_mask', 'milkyway_evidence']:
            for bad in [True, 12, ['file']]:
                with pytest.raises(Exception):
                    await server.call_tool('reduce_stars', {'separation_result': 'result.json', field: bad})
        with pytest.raises(Exception):
            await server.call_tool('reduce_stars', {'strength': .5})
        assert len(calls) == 2

    asyncio.run(exercise())
