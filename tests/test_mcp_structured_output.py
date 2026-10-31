"""Every MCP tool answers with structuredContent, not only a JSON string in a text block.

FastMCP (mcp 1.30.0) builds an outputSchema only from a parametrised return annotation: ``-> dict[str, Any]`` gives a
RootModel schema, while a bare ``-> dict`` (a plain class without type hints) or no annotation silently gives none,
and the reply then carries the dict only as text. Thirteen tools of the main server had that before 2026-10-09.

The structured path also serialises with ``model_dump(mode="json")``, which refuses NumPy integers and bools that the
text path quietly wrote as strings, so such a value used to turn a success into a tool error. ``mcp_replies`` registers
every tool with ``structured_output=True`` and converts its result to JSON-native types once, so the text block and
structuredContent carry the same JSON. The conversion runs after the tool's side effects, so a small NumPy array
becomes a list and a large one gives an error that still carries the result's identifiers. No job runs here:
services are stubbed where a real call would need one.
"""
import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydantic_core
import pytest
from astropy.io import fits
from PIL import Image

from starstack import core, finishing_mcp, i18n, regions
from starstack.mcp_replies import ARRAY_LIST_LIMIT, json_native, structured_tools
from starstack.mcp_server import WorkspaceTools, create_server

pytest.importorskip("mcp")
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

# Tools allowed to answer without structured content. Keep empty unless a tool truly returns no JSON object.
UNSTRUCTURED_TOOLS: set[str] = set()


@pytest.fixture
def scene(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith('STARSTACK_')]:
        monkeypatch.delenv(key)
    inp, out = tmp_path / 'input', tmp_path / 'output'
    inp.mkdir(); out.mkdir()
    files = []
    for i in range(3):
        raw = inp / f'{i}.NEF'; raw.write_bytes(bytes([i + 1]) * 43); files.append(str(raw))
    refdir = out / 'reference-small'; refdir.mkdir()
    linear = refdir / 'linear.fit'; fits.writeto(linear, np.ones((3, 24, 32), np.float32))
    preview = refdir / 'preview.png'; Image.new('RGB', (32, 24), (40, 50, 60)).save(preview)
    ref = {'state': 'succeeded', 'source_path': files[0], 'source_sha256': core._sha(Path(files[0])),
           'linear_path': str(linear), 'linear_sha256': core._sha(linear), 'width': 32, 'height': 24,
           'reference_dir': str(refdir), 'preview_path': str(preview), 'preview_sha256': regions._sha(preview),
           'preview_width': 32, 'preview_height': 24}
    (refdir / 'reference.json').write_text(json.dumps(ref), encoding='utf8')
    return WorkspaceTools(inp, out), refdir, files


def _text(result):
    return next(block.text for block in result.content if block.type == 'text')


def _structured(result):
    """The reply must carry the object as structuredContent, and the text block must say the same thing."""
    assert not result.isError, result.content
    assert isinstance(result.structuredContent, dict), result
    assert json.loads(_text(result)) == result.structuredContent
    return result.structuredContent


def _session(server, exercise):
    async def run():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            await exercise(client)
    asyncio.run(asyncio.wait_for(run(), timeout=60))


def _finishing_server(service):
    return finishing_mcp.create_server(service=service)


def test_every_registered_tool_declares_an_object_output_schema(scene):
    server = create_server(scene[0])
    tools = asyncio.run(server.list_tools())
    assert len(tools) == 54  # 53 + inspect_workflow_cache (read-only cache inspection)
    missing = sorted(tool.name for tool in tools
                     if tool.name not in UNSTRUCTURED_TOOLS
                     and (tool.outputSchema is None or tool.outputSchema.get('type') != 'object'))
    assert missing == []
    standalone = asyncio.run(_finishing_server(SimpleNamespace()).list_tools())
    assert len(standalone) == 16
    assert all(tool.outputSchema and tool.outputSchema.get('type') == 'object' for tool in standalone)


def test_every_registered_tool_replies_through_the_json_native_wrapper(scene):
    """A tool registered with a plain ``@server.tool()`` would skip the conversion; this catches it."""
    for server in (create_server(scene[0]), _finishing_server(SimpleNamespace())):
        registered = server._tool_manager.list_tools()
        assert registered
        bypassing = sorted(tool.name for tool in registered if not getattr(tool.fn, 'json_native_reply', False))
        assert bypassing == []
        # functools.wraps keeps the tool's own name, docstring and signature.
        assert all(tool.fn.__name__ == tool.name and tool.fn.__wrapped__.__doc__ == tool.description
                   for tool in registered)


def test_an_undescribable_return_annotation_fails_when_the_server_is_built():
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import InvalidSignature
    tool = structured_tools(FastMCP('probe'))

    async def bare() -> dict:
        return {}

    async def unannotated():
        return {}

    for function in (bare, unannotated):
        with pytest.raises(InvalidSignature):
            tool()(function)

    async def described() -> dict[str, object]:
        return {}

    assert tool()(described) is described          # the decorator hands back the function itself


def test_region_tools_return_structured_content_in_a_real_sdk_session(scene):
    service, refdir, files = scene

    async def exercise(client):
        created = _structured(await client.call_tool('create_regions', {
            'reference_dir': str(refdir), 'input_files': files, 'scene_mode': 'sky'}))
        assert Path(created['project_path']).is_file() and created['revision']
        updated = _structured(await client.call_tool('update_regions', {
            'regions_path': created['project_path'], 'scene_mode': 'landscape',
            'operations': [{'kind': 'horizon', 'points': [[0, .75], [1, .75]]}]}))
        assert updated['revision'] != created['revision']
        got = _structured(await client.call_tool('get_regions', {'regions_path': created['project_path']}))
        assert got['revision'] == created['revision']
        drawn = _structured(await client.call_tool('draw_regions', {
            'reference_dir': str(refdir), 'input_files': files, 'milkyway_ellipse': [1.05, .5, .8, .6]}))
        assert drawn['milkyway_enabled'] and drawn['project']['milkyway']['authority'] == 'agent'
        arguments = {'regions_path': drawn['project_path'], 'reference_dir': str(refdir)}
        default = await client.call_tool('preview_regions', arguments)
        receipt = _structured(default)
        assert [block.type for block in default.content] == ['text']
        visible = await client.call_tool('preview_regions', {**arguments, 'include_image': True})
        shown = _structured(visible)          # every preview call renders into a new folder
        assert set(shown) == set(receipt) and shown['sha256'] == receipt['sha256']
        assert [block.type for block in visible.content] == ['text', 'image']
        # The image reply's text block is still json.dumps(receipt, ensure_ascii=False), as before.
        assert _text(visible) == json.dumps(shown, ensure_ascii=False)
    _session(create_server(service), exercise)


CANNED = {'state': 'running', 'progress': 42.5, 'workflow_dir': 'workflow-x', 'message': '堆栈中',
          'artifacts': [{'role': 'delivery', 'path': 'p'}], 'trail_review': {'kept_ids': [], 'pending_count': 0},
          'error': None, 'published': False}

WORKFLOW_CALLS = {
    'submit_workflow': {},
    'workflow_status': {'workflow_dir': 'workflow-x'},
    'cancel_workflow': {'workflow_dir': 'workflow-x'},
    'resume_workflow': {'workflow_dir': 'workflow-x'},
    'open_trail_review': {'workflow_dir': 'workflow-x'},
    'trail_candidate_context': {'workflow_dir': 'workflow-x', 'session_path': 's', 'candidate_id': 'c'},
    'save_trail_review': {'workflow_dir': 'workflow-x', 'session_path': 's', 'decisions': {},
                          'expected_draft_revision': 'r'},
    'apply_trail_review': {'workflow_dir': 'workflow-x', 'session_path': 's', 'expected_draft_revision': 'r'},
}


def test_workflow_and_trail_tools_return_structured_content(scene, monkeypatch):
    """The service answers with a fixed object, so only the SDK conversion is under test."""
    service = scene[0]
    for name in WORKFLOW_CALLS:
        monkeypatch.setattr(service, name, lambda *args, **kwargs: json.loads(json.dumps(CANNED)))
    # The text block is byte for byte what FastMCP's text path made of the same dict before this change.
    old_text = pydantic_core.to_json(CANNED, fallback=str, indent=2).decode()

    async def exercise(client):
        for name, arguments in WORKFLOW_CALLS.items():
            result = await client.call_tool(name, arguments)
            assert _structured(result) == CANNED, name
            assert _text(result) == old_text, name
    _session(create_server(service), exercise)


NUMPY_RESULT = {'count': np.int64(3), 'ok': np.bool_(True), 'ratio': np.float32(.5), 'wide': np.float64(2.25),
                'small': np.uint8(7), 'none': float('nan'), 'far': -np.inf, 'path': Path('C:/x/y.tif'),
                'pair': (np.int16(1), 2), 'nested': [{'n': np.int32(-4), 'flag': np.bool_(False)}]}
NUMPY_NATIVE = {'count': 3, 'ok': True, 'ratio': .5, 'wide': 2.25, 'small': 7, 'none': None, 'far': None,
                'path': str(Path('C:/x/y.tif')), 'pair': [1, 2], 'nested': [{'n': -4, 'flag': False}]}


def test_numpy_scalars_in_a_result_come_back_as_native_json(scene, monkeypatch):
    service, refdir, _ = scene
    for name in ('workflow_status', 'job_status'):          # newly structured, and structured before
        monkeypatch.setattr(service, name, lambda *args, **kwargs: dict(NUMPY_RESULT))
    preview = Path(json.loads((refdir / 'reference.json').read_text(encoding='utf8'))['preview_path'])
    payload = preview.read_bytes()
    receipt = {'preview_path': str(preview), 'sha256': hashlib.sha256(payload).hexdigest(),
               'width': np.int64(32), 'height': np.int32(24), 'scale': np.float32(.5)}
    monkeypatch.setattr(service, 'preview_regions', lambda *args, **kwargs: dict(receipt))

    async def exercise(client):
        for name, arguments in (('workflow_status', {'workflow_dir': 'w'}), ('job_status', {'job_dir': 'j'})):
            result = await client.call_tool(name, arguments)
            data = _structured(result)
            assert data == NUMPY_NATIVE, name
            assert type(data['count']) is int and type(data['ok']) is bool and type(data['ratio']) is float
        image = await client.call_tool('preview_regions', {'regions_path': 'r', 'reference_dir': str(refdir),
                                                           'include_image': True})
        assert _structured(image) == {**receipt, 'width': 32, 'height': 24, 'scale': .5}
        assert [block.type for block in image.content] == ['text', 'image']
    _session(create_server(service), exercise)


class _FinishingStub:
    def get(self, session_dir):
        if session_dir == 'big':
            return {'session_dir': session_dir, 'revision': np.int64(4), 'pixels': np.zeros(ARRAY_LIST_LIMIT + 1)}
        return {'session_dir': session_dir, **NUMPY_RESULT}

    def capabilities(self):
        return {'schema': 'stub', 'mean': np.float32(.25), 'pixels': np.array([1, 2, 3]),
                'matrix': np.eye(2, dtype=np.float32), 'flags': np.array([True, False])}


def test_standalone_finishing_server_converts_numpy_and_refuses_large_arrays_clearly():
    async def run():
        async with create_connected_server_and_client_session(_finishing_server(_FinishingStub())) as client:
            data = _structured(await client.call_tool('finishing_get', {'session_dir': 's'}))
            assert data == {'session_dir': 's', **NUMPY_NATIVE}
            small = _structured(await client.call_tool('finishing_capabilities', {}))
            assert small == {'schema': 'stub', 'mean': .25, 'pixels': [1, 2, 3], 'matrix': [[1., 0.], [0., 1.]],
                             'flags': [True, False]}
            assert type(small['pixels'][0]) is int and type(small['flags'][0]) is bool
            refused = await client.call_tool('finishing_get', {'session_dir': 'big'})
            text = _text(refused)
            assert refused.isError and "result['pixels']" in text and f'({ARRAY_LIST_LIMIT + 1},)' in text
            assert text.endswith('结果的顶层标量字段：{"session_dir": "big", "revision": 4}')
    asyncio.run(asyncio.wait_for(run(), timeout=30))


def test_a_large_numpy_array_in_a_result_is_a_clear_tool_error(scene, monkeypatch):
    service = scene[0]
    monkeypatch.setattr(service, 'workflow_status', lambda *args, **kwargs: {
        'state': 'done', 'stages': [1, {'mask': np.zeros((5, 13)), 'box': np.arange(4)}]})

    async def exercise(client):
        result = await client.call_tool('workflow_status', {'workflow_dir': 'w'})
        assert result.isError and result.structuredContent is None
        message = _text(result)
        assert ("工具已执行完毕，但结果的 result['stages'][1]['mask'] 是 NumPy 数组（形状 (5, 13)，超过 64 个元素），"
                '不能作为 JSON 返回；结果的顶层标量字段：{"state": "done"}') in message
        english = i18n.translate_message(message.split(': ', 1)[1], 'en')
        assert english == ("The tool has finished, but the result result['stages'][1]['mask'] is a NumPy array "
                           "(shape (5, 13), more than 64 elements) and cannot be returned as JSON; "
                           'the result\'s top-level scalar fields: {"state": "done"}')
    _session(create_server(service), exercise)


# Tools that change something before they reply: the conversion must not cost the agent the identifier it continues
# with (a refused reply after submit_workflow without an idempotency_key would invite a duplicate workflow).
WRITE_CALLS = {
    'submit_workflow': ({}, 'workflow_dir'),
    'cancel_workflow': ({'workflow_dir': 'workflow-x'}, 'workflow_dir'),
    'resume_workflow': ({'workflow_dir': 'workflow-x'}, 'workflow_dir'),
    'save_trail_review': ({'workflow_dir': 'workflow-x', 'session_path': 's', 'decisions': {},
                           'expected_draft_revision': 'r'}, 'session_path'),
    'apply_trail_review': ({'workflow_dir': 'workflow-x', 'session_path': 's', 'expected_draft_revision': 'r'},
                           'workflow_dir'),
    'create_regions': ({'reference_dir': 'ref', 'input_files': ['a.NEF']}, 'project_path'),
    'update_regions': ({'regions_path': 'regions.json'}, 'project_path'),
    'draw_regions': ({'reference_dir': 'ref', 'input_files': ['a.NEF']}, 'project_path'),
}


@pytest.mark.parametrize('size', [4, ARRAY_LIST_LIMIT + 1], ids=['small-array', 'large-array'])
def test_a_write_tool_with_an_array_in_its_result_still_gives_the_agent_its_identifier(scene, monkeypatch, size):
    service = scene[0]
    calls = []

    def stub(name, key):
        def answer(*args, **kwargs):
            calls.append(name)
            return {key: f'{name}-id', 'revision': np.int64(2), 'thumb_box': np.arange(size)}
        return answer

    for name, (_, key) in WRITE_CALLS.items():
        monkeypatch.setattr(service, name, stub(name, key))

    async def exercise(client):
        for name, (arguments, key) in WRITE_CALLS.items():
            result = await client.call_tool(name, arguments)
            if size <= ARRAY_LIST_LIMIT:
                # A small array comes back as a list; before 2026-10-09 the text block said '[0 1 2 3]'.
                assert _structured(result) == {key: f'{name}-id', 'revision': 2, 'thumb_box': list(range(size))}
            else:
                text = _text(result)
                assert result.isError and '工具已执行完毕' in text, name
                fields = json.loads(text.rsplit('结果的顶层标量字段：', 1)[1])
                assert fields == {key: f'{name}-id', 'revision': 2}, name
    _session(create_server(service), exercise)
    assert calls == list(WRITE_CALLS)          # every tool ran exactly once


class _Shown:
    def __str__(self):
        return 'shown-as-text'


def test_json_native_keeps_ordinary_values_and_their_text_unchanged():
    ordinary = {'名称': '银河', 'n': 3, 'x': .1 + .2, 'big': 2 ** 70, 'yes': True, 'nothing': None,
                'items': [1, 'a', [2.5, {'k': False}]], 'path': Path('C:/照片/a.NEF'), 'pair': (1, 'b'),
                'when': datetime.datetime(2026, 10, 9, 12, 30, 5), 'tags': ['a'], 'unknown': _Shown()}
    native = json_native(ordinary)
    old = pydantic_core.to_json(ordinary, fallback=str, indent=2)
    assert pydantic_core.to_json(native, fallback=str, indent=2) == old
    assert native == json.loads(old)
    assert list(native) == list(ordinary)
    plain = {'a': [1, 2.5, 'x', None, True], 'b': {'c': '中'}}
    assert json_native(plain) == plain and json_native(plain) is not plain
    assert json_native(NUMPY_RESULT) == NUMPY_NATIVE
    # Arrays up to the limit become nested lists of native values (NaN as null, like a scalar).
    limit = ARRAY_LIST_LIMIT
    assert json_native({'a': np.arange(limit), 'b': [np.array([[.5, np.nan]])], 'c': np.array(np.int8(3))}) == {
        'a': list(range(limit)), 'b': [[[.5, None]]], 'c': 3}
    assert type(json_native(np.arange(2))[0]) is int
    with pytest.raises(TypeError, match=r"result\[0\]\['a'\] 是 NumPy 数组（形状 \(65,\)，超过 64 个元素）.*：\{\}$"):
        json_native([{'a': np.ones(limit + 1)}])
    # The location names the large array, not a small one before it; only top-level scalars are listed.
    with pytest.raises(TypeError, match=r"result\['b'\]\[1\] .*：\{\"n\": 1, \"p\": \"x\", \"f\": null\}$"):
        json_native({'n': np.uint16(1), 'a': np.ones(2), 'b': [np.ones(2), np.ones((9, 9))], 'p': Path('x'),
                     'f': np.float64('nan'), 'd': {'deep': 'not listed'}})
