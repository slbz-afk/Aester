"""The installer's activation probe imports the modules the application itself names (open/closed split).

Official builds name the desktop app's entry module in PACKAGE_CONTENTS.json; the open installer recipe never
names it. A missing list falls back to the engine's modules; a malformed one is refused, never executed.
"""
from pathlib import Path
import importlib.util
import json
import sys

import pytest

from starstack import components, install_lifecycle, install_ownership

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setitem(sys.modules, 'components', components)
    monkeypatch.setitem(sys.modules, 'install_lifecycle', install_lifecycle)
    monkeypatch.setitem(sys.modules, 'install_ownership', install_ownership)
    spec = importlib.util.spec_from_file_location('activation_probe_worker_test', ROOT / 'packaging/aester_p2/installer_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_a_list_the_engine_modules_are_probed(worker, tmp_path):
    assert worker.activation_probe(tmp_path) == ['starstack.mcp_server', 'starstack.workflow']
    (tmp_path / 'PACKAGE_CONTENTS.json').write_text(json.dumps({'schema_version': 1}), encoding='utf-8')
    assert worker.activation_probe(tmp_path) == ['starstack.mcp_server', 'starstack.workflow']


def test_the_application_list_is_used(worker, tmp_path):
    probe = ['desktop_app.gui', 'starstack.mcp_server', 'starstack.workflow']
    (tmp_path / 'PACKAGE_CONTENTS.json').write_text(json.dumps({'activation_probe': probe}), encoding='utf-8')
    assert worker.activation_probe(tmp_path) == probe


@pytest.mark.parametrize('probe', [['os; import evil'], ['a b'], ['1x'], [3], 'starstack.gui'])
def test_a_malformed_list_is_refused(worker, tmp_path, probe):
    (tmp_path / 'PACKAGE_CONTENTS.json').write_text(json.dumps({'activation_probe': probe}), encoding='utf-8')
    with pytest.raises(ValueError):
        worker.activation_probe(tmp_path)
