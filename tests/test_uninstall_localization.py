"""Presentation is bilingual while the permission-bearing plan stays untouched."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from starstack import install_lifecycle, install_ownership, uninstall_options


@pytest.fixture
def worker(monkeypatch):
    for name, module in [('install_lifecycle', install_lifecycle),
                         ('install_ownership', install_ownership), ('uninstall_options', uninstall_options)]:
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        'uninstall_worker_localization', Path(__file__).parents[1] / 'packaging/aester_p2/uninstall_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('language', ['en', 'zh_CN'])
def test_summary_preserves_plans_and_defaults(worker, language):
    installation = {'owned_component_and_download_files': 14019, 'owned_component_and_download_bytes': 2**30}
    selection = {'cache_plans': [{'bytes': 2**31, 'files': ['owned.tif']}],
                 'preserved': [{'path': 'unknown.raw'}], 'default_clear_safe_cache': False,
                 'default_clear_settings': False}
    before = deepcopy((installation, selection))
    result = worker.format_inspection(installation, selection, language)
    assert (installation, selection) == before
    assert '14,019' in result and '1.00 GB' in result and '2.00 GB' in result
    if language == 'en':
        assert '1 completed jobs' in result and '1 unverified items' in result
        assert 'Original photographs' in result and 'job records' in result
        assert 'Shared components' in result
        assert not any('\u4e00' <= char <= '\u9fff' for char in result)
    else:
        assert '1 个已完成作业' in result and '1 项无法安全确认' in result
        assert '作业记录' in result and '原片' in result and '共享组件' in result


@pytest.mark.parametrize('language', ['en', 'zh_CN'])
def test_empty_summary_has_no_stale_human_summary(worker, language):
    result = worker.format_inspection(
        {'owned_component_and_download_files': 0, 'owned_component_and_download_bytes': 0},
        {'summary_zh': 'STALE_SUMMARY', 'cache_plans': [], 'preserved': []}, language)
    assert 'STALE_SUMMARY' not in result
    assert '0.00 GB' in result


def test_translation_catalog_has_same_keys():
    lines = (Path(__file__).parents[1] / 'packaging/aester_p2/installer-messages.iss').read_text(encoding='utf-8').splitlines()
    mappings = {lang: {} for lang in ('en', 'zh_CN')}
    for line in lines:
        if '=' in line:
            name, text = line.split('=', 1)
            lang, key = name.split('.', 1)
            assert key not in mappings[lang]
            mappings[lang][key] = text
    assert mappings['en'].keys() == mappings['zh_CN'].keys()
    assert all(mappings['en'].values()) and all(mappings['zh_CN'].values())
    for key in mappings['en']:
        for placeholder in ('%1', '%2'):
            assert mappings['en'][key].count(placeholder) == mappings['zh_CN'][key].count(placeholder)
