"""Local MCP setup: explicit scope, copyable configuration, real handshake."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

from .i18n import get_language, normalize_language, tr


def _directory(value, *, exists=False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(tr('目录不能为空'))
    path = Path(value).expanduser().absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError(tr('请使用真实目录，不使用符号链接或目录联接'))
    path = path.resolve(strict=exists)
    if path.exists() and not path.is_dir():
        raise ValueError(tr('请选择文件夹'))
    return path



# Public name for front ends (the desktop app); the same function object.
checked_directory = _directory


def connection_config(input_root, output_root, cache_root, *, python_path=None,
                      source_root=None, siril_path=None, stack_engine_path=None,
                      star_model_path=None, backend_options=None, language=None, dng_converter_path=None,
                      gimp_path=None) -> dict:
    """No client settings are changed. The returned command has no shell layer."""
    inp = _directory(input_root, exists=True)
    out = _directory(output_root)
    cache = _directory(cache_root)
    if out == inp or cache == inp:
        raise ValueError(tr('素材、作品与缓存需要使用不同目录'))
    if out.is_relative_to(cache) or cache.is_relative_to(out):
        raise ValueError(tr('作品与缓存目录不能相互包含'))
    python = Path(python_path or sys.executable).resolve(strict=True)
    if python.name.lower() == 'pythonw.exe' and python.with_name('python.exe').is_file():
        python = python.with_name('python.exe')
    if not python.is_file():
        raise ValueError(tr('未找到当前安装的运行程序'))
    source = _directory(source_root or Path(__file__).resolve().parent.parent, exists=True)
    env = {'STARSTACK_INPUT_ROOT': str(inp), 'STARSTACK_OUTPUT_ROOT': str(out),
           'STARSTACK_CACHE_ROOT': str(cache), 'PYTHONPATH': str(source),
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
           'AESTER_LANGUAGE': normalize_language(get_language() if language is None else language)}
    env['AESTER_APP_ROOT'] = str(source.parent)
    if os.environ.get('AESTER_COMPONENT_ROOT'):
        env['AESTER_COMPONENT_ROOT'] = str(_directory(os.environ['AESTER_COMPONENT_ROOT']))
    for name, value in [('STARSTACK_SIRIL', siril_path), ('STARSTACK_STACK_ENGINE', stack_engine_path),
                        ('STARSTACK_STAR_MODEL', star_model_path), ('AESTER_DNG_CONVERTER', dng_converter_path)]:
        if value:
            path = Path(value).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(tr('处理组件需要是已有文件'))
            env[name] = str(path)
    if gimp_path:
        # A moved/uninstalled optional GIMP must not block the base MCP
        # connection. Preserve the explicit choice for retouch diagnostics.
        env['AESTER_GIMP'] = str(Path(gimp_path).expanduser().resolve(strict=False))
    if star_model_path and backend_options:
        env['STARSTACK_STAR_ADAPTER'] = json.dumps(backend_options, ensure_ascii=False)
    return {'mcpServers': {'aester': {'command': str(python),
            'args': ['-X', 'utf8', '-m', 'starstack.mcp_server'], 'env': env}}}


def setup_prompt(config: dict, language=None) -> str:
    selected = (config.get('mcpServers', {}).get('aester', {}).get('env', {}).get('AESTER_LANGUAGE', get_language())
                if language is None else language)
    if normalize_language(selected) == 'en':
        return ('Help me connect the local Aester installation to this Agent client. '
                'The standard stdio configuration is below. If this client requires another format, '
                'convert it according to the client\'s official format while preserving the command, arguments and folder-access limits. '
                'Copying a configuration is not proof of a successful connection: after setup, list the Aester tools and call get_help to verify it. '
                'Do not broaden access to input folders. No model API key is required, and do not upload my photographs.\n\n'
                + json.dumps(config, ensure_ascii=False, indent=2)
                + '\n\nOnce connected, inspect only the same-scene photos I list. Show a colour-calibrated preview of the first photo '
                'and the sky/foreground regions. Leave trail replay, star reduction and Milky Way enhancement off initially. '
                'After I confirm the regions, stack the photos and deliver exactly three images: a float32 linear master TIFF, '
                'a 16-bit finished TIFF and a JPEG. Keep processing logs and recipe records in the work cache. '
                'Do not modify the originals, and do not record a model\'s guess as human confirmation without my approval.')
    return ('请帮我把本机 Aester 连接到你所在的 Agent 客户端。下面是标准 stdio 配置；'
            '若该客户端使用其他配置格式，请按其官方格式转换，保留命令、参数和目录权限范围。'
            '不要把复制配置当成连接成功：完成后列出 Aester 工具并调用 get_help 验证。'
            '不要扩大素材目录权限，不需要填写模型 API 密钥，也不要上传照片。\n\n'
            + json.dumps(config, ensure_ascii=False, indent=2)
            + '\n\n连接后，请先检查我列出的同一场景照片，给出首帧校色预览和天空／地景分区。'
            '亮迹、星点整理、银河增强先关闭；确认范围后合成，仅交付浮点线性母版 TIFF、16位成片 TIFF 和 JPEG 三份图像，处理记录留工作缓存。'
            '不要改动原片，未经我确认不要把模型猜测记成人工确认。')


async def probe_connection_async(config: dict, *, timeout=35.) -> dict:
    """Initialize a local stdio MCP subprocess; never contacts an LLM provider."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    spec = config['mcpServers']['aester']
    env = dict(os.environ)
    # Do not let unrelated host configuration silently choose extra engines.
    for key in ('STARSTACK_INPUT_ROOT','STARSTACK_OUTPUT_ROOT','STARSTACK_CACHE_ROOT',
                'STARSTACK_SIRIL','STARSTACK_STACK_ENGINE','STARSTACK_STAR_MODEL','STARSTACK_STAR_ADAPTER',
                'AESTER_LANGUAGE', 'AESTER_DNG_CONVERTER', 'AESTER_GIMP', 'AESTER_APP_ROOT',
                'AESTER_COMPONENT_ROOT'):
        env.pop(key, None)
    env.update(spec['env'])
    parameters = StdioServerParameters(command=spec['command'], args=spec['args'], env=env)
    async with asyncio.timeout(timeout):
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                initialized = await session.initialize()
                listing = await session.list_tools()
                names = {tool.name for tool in listing.tools}
                required = {'get_help', 'inspect_sources', 'submit_workflow', 'workflow_status'}
                if not required <= names:
                    raise ValueError(tr('服务缺少完整出图工具，请修复安装'))
                help_result = await session.call_tool('get_help', {})
                if help_result.isError:
                    raise ValueError(tr('服务已响应，但帮助调用失败'))
                return {'state': 'ready', 'server_name': initialized.serverInfo.name,
                        'tool_count': len(names), 'checked': ['initialize', 'list_tools', 'get_help'],
                        'external_client_connected': False,
                        'message': tr('本机服务已通过握手与工具调用；请在你的 Agent 客户端添加配置后再确认连接。')}


def probe_connection(config: dict, *, timeout=35.) -> dict:
    return asyncio.run(probe_connection_async(config, timeout=timeout))
