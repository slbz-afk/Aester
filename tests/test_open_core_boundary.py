"""The Apache-2.0 engine never depends on the GPL desktop app (engine/app boundary since 2026-10-14).

* O1: no open Python file imports ``aester_desktop``, PySide6 or shiboken6, and the open product code (the
  engine package and the installer recipe) never names the desktop package at all;
* O2: every engine module imports, and the engine's i18n and help work, with the desktop package and Qt
  refused by an import hook (in a fresh process);
* O3: every engine name the desktop app uses (``src/starstack/public_api.json``) exists with its recorded
  signature, so an engine change that would break the app fails here, on the open side;
* O4: the open distribution (pyproject) carries neither Qt nor the app;
* O5: ``i18n.register_catalogs`` keeps the pre-split catalogue order and clears both caches.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib
import inspect
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP = tomllib.loads((ROOT / "open-core.toml").read_text(encoding="utf-8"))
CLOSED_GLOBS = MAP["gpl"]["paths"]
INTERNAL_GLOBS = MAP["internal"]["paths"]
FORBIDDEN_IMPORTS = ("aester_desktop", "PySide6", "shiboken6")
_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")


def _matches(relative: str, globs) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) or relative == pattern.removesuffix("/**")
               or relative.startswith(pattern.removesuffix("**")) and pattern.endswith("/**") for pattern in globs)


def _open_python_files():
    for folder in ("src/starstack", "packaging", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if "__pycache__" in path.parts or _matches(relative, CLOSED_GLOBS) or _matches(relative, INTERNAL_GLOBS):
                continue
            yield relative, path


def test_o1_open_python_never_imports_the_desktop_app_or_qt():
    offenders = []
    for relative, path in _open_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None))
                  in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant)):
                names = [str(node.args[0].value)]
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    offenders.append(f"{relative}:{node.lineno} {name}")
    assert offenders == []


def test_o1_open_product_code_never_names_the_desktop_package():
    """The engine and the installer recipe find the app through data (PACKAGE_CONTENTS entry modules).
    Documentation may describe where the app's files are installed; code and configuration may not name it."""
    offenders = []
    for folder in ("src/starstack", "packaging"):
        for path in sorted((ROOT / folder).rglob("*")):
            if (not path.is_file() or "__pycache__" in path.parts
                    or path.suffix.lower() not in (".py", ".cs", ".iss", ".isl", ".json", ".toml", ".cmd", ".ps1")):
                continue
            text = path.read_bytes().decode("utf-8", errors="ignore")
            if "aester_desktop" in text:
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


_GUARD = r'''
import importlib, importlib.abc, pkgutil, sys
attempts = []
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("aester_desktop", "PySide6", "shiboken6"):
            attempts.append(name)
            raise ImportError("refused by the open-core boundary test: " + name)
        return None
sys.meta_path.insert(0, Refuse())
import starstack
failed = []
for module in pkgutil.iter_modules(starstack.__path__):
    try:
        importlib.import_module("starstack." + module.name)
    except Exception as error:
        failed.append(f"{module.name}: {type(error).__name__}: {error}")
from starstack import i18n, user_guide
assert i18n.english_catalog(), "the engine's own catalogue is empty"
i18n.message_patterns()
i18n.translate_message("目录不能为空", "en")
assert user_guide.guide_text("en")
print("FAILED", failed)
print("ATTEMPTS", attempts)
'''


def test_o2_every_engine_module_imports_without_the_desktop_app_or_qt():
    import os
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "AESTER_", "STARSTACK_"))}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", _GUARD], capture_output=True, text=True,
                            encoding="utf-8", env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    lines = dict(line.split(" ", 1) for line in result.stdout.splitlines() if line.startswith(("FAILED", "ATTEMPTS")))
    assert lines["FAILED"] == "[]" and lines["ATTEMPTS"] == "[]", result.stdout


def test_o3_the_api_the_desktop_app_uses_exists_with_its_recorded_signatures():
    surface = json.loads((ROOT / "src/starstack/public_api.json").read_text(encoding="utf-8"))
    problems = []
    for module, names in surface["modules"].items():
        imported = importlib.import_module("starstack" if module == "__init__" else f"starstack.{module}")
        for name, recorded in names.items():
            if name.startswith("_"):
                problems.append(f"{module}.{name}: private name in the public surface")
                continue
            if not hasattr(imported, name):
                problems.append(f"{module}.{name}: missing")
                continue
            obj = getattr(imported, name)
            if inspect.isclass(obj):
                live = "class"
            elif callable(obj):
                try:
                    live = _ADDRESS.sub("", str(inspect.signature(obj)))
                except (TypeError, ValueError):
                    live = "callable"
            else:
                live = "constant:" + type(obj).__name__
            if live != recorded:
                problems.append(f"{module}.{name}: {recorded!r} -> {live!r}")
    assert problems == [], "engine API used by the desktop app changed (update the app, then " \
                           "scripts/split/api_surface.py --write): " + "; ".join(problems)


def test_o4_the_open_distribution_carries_neither_qt_nor_the_app():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any("pyside6" in dependency.lower() for dependency in project["project"]["dependencies"])
    assert all(not target.split(":")[0].startswith(("aester_desktop", "starstack.gui", "starstack.finishing_gui"))
               for target in project["project"]["scripts"].values())
    found = project["tool"]["setuptools"]["packages"]["find"]
    assert found["include"] == ["starstack", "starstack.*"]
    assert not (ROOT / "src/starstack/assets").exists()


def test_o5_registered_catalogues_keep_the_pre_split_order_and_clear_the_caches(monkeypatch):
    from starstack import i18n
    monkeypatch.setattr(i18n, "_REGISTERED", [])
    i18n.english_catalog.cache_clear()
    base = i18n.english_catalog()
    assert set(base) == set(importlib.import_module("starstack.translations_onboarding").EN)
    import types
    first, second = types.ModuleType("first_catalog_for_test"), types.ModuleType("second_catalog_for_test")
    first.EN, second.EN = {"甲": "first", "乙": "only-first"}, {"甲": "second"}
    monkeypatch.setitem(sys.modules, first.__name__, first)
    monkeypatch.setitem(sys.modules, second.__name__, second)
    i18n.register_catalogs(first.__name__, second.__name__, "missing_catalog_module_for_test")
    i18n.register_catalogs(first.__name__)                       # again: harmless, order unchanged
    catalog = i18n.english_catalog()
    assert catalog["甲"] == "second" and catalog["乙"] == "only-first"
    assert i18n._REGISTERED == [first.__name__, second.__name__, "missing_catalog_module_for_test"]
    for bad in ("", "a..b", "1x", "a b", None):
        with pytest.raises(ValueError):
            i18n.register_catalogs(bad)
    i18n.english_catalog.cache_clear()
    i18n._translate_english.cache_clear()
