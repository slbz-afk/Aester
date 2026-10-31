# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See src/aester_desktop/LICENSE.txt.
"""The desktop app reaches the engine only through its declared public surface (engine/app boundary).

* C1: ``src/starstack/public_api.json`` equals a fresh scan of what the app uses (new engine names are added on
  purpose: ``scripts/split/api_surface.py --write``);
* C2: no private (``_``) engine name and no ``starstack.cli`` import;
* C3: relative imports stay inside the desktop package;
* C4: engine record files are parsed by the engine only (no evidence/snapshot/checkpoint/package literals);
* C8: every desktop module carries the GPL-3.0-or-later SPDX header and copyright line; the package defines its
  notice; LICENSE.txt carries the notice, the trademark term and the full GPLv3 text; no EULA remains.
"""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLOSED = ROOT / "src" / "aester_desktop"


def _api_surface():
    spec = importlib.util.spec_from_file_location("api_surface_for_test", ROOT / "scripts/split/api_surface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules():
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in sorted(CLOSED.rglob("*.py"))]


def test_c1_public_surface_file_matches_what_the_app_uses():
    tool = _api_surface()
    assert (ROOT / "src/starstack/public_api.json").read_bytes().decode("utf-8") == tool.render(tool.surface()), \
        "run scripts/split/api_surface.py --write after reviewing the new engine names the app uses"


def test_c2_no_private_engine_names_and_no_cli():
    offenders = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith("starstack"):
                if node.module == "starstack.cli" or (node.module == "starstack" and any(a.name == "cli" for a in node.names)):
                    offenders.append(f"{path.name}:{node.lineno} cli")
                offenders += [f"{path.name}:{node.lineno} {node.module}.{a.name}" for a in node.names if a.name.startswith("_")]
    used = _api_surface().used_names()
    offenders += [f"{module}.{name}" for module, names in used.items() for name in names if name.startswith("_")]
    assert offenders == []


def test_c3_relative_imports_resolve_inside_the_closed_package():
    for path, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                assert base.is_relative_to(CLOSED), (path.name, node.lineno, "relative import escaped desktop")
                targets = [node.module.replace('.', '/')] if node.module else [a.name for a in node.names]
                for target in targets:
                    assert (base / f"{target}.py").is_file() or (base / target).is_dir(), (path.name, node.lineno, target)


def test_c4_engine_records_are_parsed_by_the_engine_only():
    forbidden = ("evidence.json", "stack_snapshot-", "PACKAGE_CONTENTS.json", "'checkpoints'", '"checkpoints"')
    offenders = []
    for path in sorted(CLOSED.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {literal}" for literal in forbidden if literal in text]
    assert offenders == []


HEADER = "# SPDX-License-Identifier: GPL-3.0-or-later\n# Copyright (C) 2026 Wang Jian (王俭)."


def test_c8_gpl_headers_notice_and_licence_text():
    import aester_desktop
    assert aester_desktop.__copyright__ == "Copyright (C) 2026 Wang Jian (王俭)"
    assert aester_desktop.__license__ == "GPL-3.0-or-later"
    problems = [f"{path.name}: header" for path, _ in _modules() if not path.read_text(encoding="utf-8").startswith(HEADER)]
    licence = (CLOSED / "LICENSE.txt").read_text(encoding="utf-8")
    assert "Copyright (C) 2026 Wang Jian (王俭)" in licence and "either version 3 of the License, or" in licence
    assert "GPLv3 section 7(e)" in licence and "TRADEMARKS.md" in licence
    gpl = (ROOT / "packaging/aester_p2/GPL-3.0.txt").read_text(encoding="utf-8")
    assert licence.endswith(gpl), "LICENSE.txt must end with the unmodified GPLv3 text"
    assert not (CLOSED / "legal").exists()
    assert problems == []
