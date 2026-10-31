"""Offline language help and local stdio configuration; no user settings writes."""
import asyncio
from contextlib import asynccontextmanager
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from starstack import i18n, onboarding
from starstack.user_guide import guide_text


@pytest.fixture(autouse=True)
def restore_language():
    previous = i18n.get_language()
    yield
    i18n.set_language(previous)


def folders(tmp_path):
    source = tmp_path/'素材 astronomy'
    source.mkdir()
    return source, tmp_path/'photographs', tmp_path/'work cache'


def test_complete_help_keeps_all_anchors_and_both_guides_are_offline():
    chinese, english = guide_text('zh-CN'), guide_text('en-US')
    anchors = lambda text: set(re.findall(r'<a name="([^\"]+)"></a>', text))
    assert anchors(chinese) == anchors(english)
    assert {'start', 'masks', 'stars', 'trails', 'milkyway', 'mcp', 'storage', 'limits'} <= anchors(english)
    for text in (chinese, english):
        assert 'submit_workflow' in text and 'prepare_reference' in text
        assert 'STARSTACK_INPUT_ROOT' in text and 'STARSTACK_CACHE_ROOT' in text
        assert 'include_image=true' in text
        assert 'http://' not in text and 'https://' not in text
    assert 'exactly three images' in english and 'inside the TIFF' in english
    assert 'negative values and highlight headroom' in english
    assert 'copied into the cache are preserved' in english
    assert 'Settings → Language' in english and 'restart Aester' in english
    assert '连接 AI' not in english


def test_guide_defaults_to_runtime_language_but_explicit_language_is_local():
    i18n.set_language('en')
    assert guide_text() == guide_text('en')
    assert guide_text('zh_CN').startswith('# Aester · 开始创作')
    assert i18n.get_language() == 'en'
    i18n.set_language('zh_CN')
    assert guide_text() == guide_text('zh_CN')


@pytest.mark.parametrize('language, expected', [('en-GB', 'en'), ('zh-CN', 'zh_CN')])
def test_configuration_carries_language_without_changing_protocol_or_paths(tmp_path, language, expected):
    source, output, cache = folders(tmp_path)
    before = list(tmp_path.rglob('*'))
    config = onboarding.connection_config(source, output, cache, language=language)
    spec = config['mcpServers']['aester']
    assert spec['args'] == ['-X', 'utf8', '-m', 'starstack.mcp_server']
    assert spec['env']['AESTER_LANGUAGE'] == expected
    assert spec['env']['STARSTACK_INPUT_ROOT'] == str(source)
    assert spec['env']['STARSTACK_OUTPUT_ROOT'] == str(output)
    assert spec['env']['STARSTACK_CACHE_ROOT'] == str(cache)
    assert list(tmp_path.rglob('*')) == before


def test_english_prompt_embeds_the_exact_unchanged_configuration(tmp_path):
    i18n.set_language('en')
    config = onboarding.connection_config(*folders(tmp_path))
    before = copy.deepcopy(config)
    encoded = json.dumps(config, ensure_ascii=False, indent=2)
    prompt = onboarding.setup_prompt(config)
    assert prompt.count(encoded) == 1
    assert config == before
    prose = prompt.replace(encoded, '')
    assert not re.search(r'[\u4e00-\u9fff]', prose)
    assert 'get_help' in prose and 'Do not broaden access' in prose
    assert 'No model API key' in prose and 'do not upload my photographs' in prose
    assert 'exactly three images' in prose and 'float32 linear master TIFF' in prose
    assert 'Do not modify the originals' in prose
    assert json.loads(prompt.split('\n\n')[1]) == config


def test_prompt_uses_frozen_config_language_and_explicit_override_only_changes_prose(tmp_path):
    i18n.set_language('zh_CN')
    config = onboarding.connection_config(*folders(tmp_path), language='en')
    assert onboarding.setup_prompt(config).startswith('Help me')
    assert onboarding.setup_prompt(config, language='zh_CN').startswith('请帮我')
    assert config['mcpServers']['aester']['env']['AESTER_LANGUAGE'] == 'en'
    del config['mcpServers']['aester']['env']['AESTER_LANGUAGE']
    assert onboarding.setup_prompt(config).startswith('请帮我')
    i18n.set_language('en')
    assert onboarding.setup_prompt(config).startswith('Help me')


@pytest.mark.parametrize('legacy_config', [False, True])
def test_probe_clears_ambient_engine_and_language_then_passes_explicit_config(tmp_path, monkeypatch, legacy_config):
    import mcp
    import mcp.client.stdio
    i18n.set_language('en')
    config = onboarding.connection_config(*folders(tmp_path), language='en')
    if legacy_config:
        del config['mcpServers']['aester']['env']['AESTER_LANGUAGE']
    monkeypatch.setenv('STARSTACK_STAR_MODEL', 'stale-model.onnx')
    monkeypatch.setenv('STARSTACK_STAR_ADAPTER', 'stale-adapter')
    monkeypatch.setenv('AESTER_LANGUAGE', 'zh_CN')
    seen = {}
    @asynccontextmanager
    async def transport(parameters):
        seen['parameters'] = parameters
        yield None, None
    class Session:
        def __init__(self, *args):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def initialize(self):
            return SimpleNamespace(serverInfo=SimpleNamespace(name='Aester test transport'))
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name=n) for n in
                ('get_help', 'inspect_sources', 'submit_workflow', 'workflow_status')])
        async def call_tool(self, name, args):
            assert (name, args) == ('get_help', {})
            return SimpleNamespace(isError=False)
    monkeypatch.setattr(mcp.client.stdio, 'stdio_client', transport)
    monkeypatch.setattr(mcp, 'ClientSession', Session)
    result = asyncio.run(onboarding.probe_connection_async(config))
    env = seen['parameters'].env
    assert 'STARSTACK_STAR_MODEL' not in env and 'STARSTACK_STAR_ADAPTER' not in env
    assert env.get('AESTER_LANGUAGE') == (None if legacy_config else 'en')
    assert seen['parameters'].command == config['mcpServers']['aester']['command']
    assert seen['parameters'].args == config['mcpServers']['aester']['args']
    assert result['external_client_connected'] is False
    assert result['message'].startswith('The local service')


def test_english_path_errors_keep_the_same_restrictions(tmp_path):
    i18n.set_language('en')
    source, output, cache = folders(tmp_path)
    for value in ('', ' ', None, {}):
        with pytest.raises(ValueError, match='folder path is required'):
            onboarding.connection_config(value, output, cache)
    with pytest.raises(ValueError, match='different folders'):
        onboarding.connection_config(source, source, cache)
    with pytest.raises(ValueError, match='must not contain each other'):
        onboarding.connection_config(source, output, output/'cache')
    file = tmp_path/'file.txt'
    file.write_text('not a directory')
    with pytest.raises(ValueError, match='Select a folder'):
        onboarding.connection_config(file, output, cache)
    with pytest.raises(ValueError, match='existing file'):
        onboarding.connection_config(source, output, cache, siril_path=source)


def test_derived_offline_documents_match_both_languages():
    root = Path(__file__).resolve().parents[1]
    assert (root/'docs/USER_GUIDE.md').read_text(encoding='utf-8') == guide_text('zh_CN')
    assert (root/'docs/USER_GUIDE_EN.md').read_text(encoding='utf-8') == guide_text('en')


def test_actual_separate_stdio_process_uses_configured_english_help(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import os
    config = onboarding.connection_config(*folders(tmp_path), language='en')
    spec = config['mcpServers']['aester']
    env = dict(os.environ)
    for key in ('STARSTACK_INPUT_ROOT', 'STARSTACK_OUTPUT_ROOT', 'STARSTACK_CACHE_ROOT',
                'STARSTACK_SIRIL', 'STARSTACK_STACK_ENGINE', 'STARSTACK_STAR_MODEL',
                'STARSTACK_STAR_ADAPTER', 'AESTER_LANGUAGE'):
        env.pop(key, None)
    env.update(spec['env'])
    async def check():
        async with asyncio.timeout(25):
            async with stdio_client(StdioServerParameters(command=spec['command'], args=spec['args'], env=env)) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    result = await session.call_tool('get_help', {})
                    assert not result.isError
                    data = result.structuredContent
                    if data is None:
                        data = json.loads(next(item.text for item in result.content if item.type == 'text'))
                    if 'result' in data:
                        data = data['result']
                    assert data['format'] == 'markdown'
                    assert data['content'] == guide_text('en')
    asyncio.run(check())
