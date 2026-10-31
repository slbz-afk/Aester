# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See src/aester_desktop/LICENSE.txt.
"""The build ships the desktop app as Python source under GPL-3.0-or-later (open-source migration, 2026-10-31).

Uses the isolated builder fixture (fake Inno and fake bundled runtime) plus a tiny desktop package, smoke-imported by
the real interpreter: the .py sources and the licence ship, no bytecode is left in the staged tree, PACKAGE_CONTENTS
names the entry modules and records the source form, the licence pages carry the GPL notice, and release builds
refuse while a release gate is open.
"""
from pathlib import Path
import json
import subprocess
import shutil
import sys

import pytest

from test_finite_installer_distribution import builder_case  # noqa: F401  (fixture)

PYTHON = Path(sys.executable)
GPL = (Path(__file__).resolve().parents[2] / "packaging/aester_p2/GPL-3.0.txt").read_text(encoding="utf-8")


@pytest.fixture
def real_python(builder_case, monkeypatch):
    """The fixture stubs subprocess.run (fake Inno, fake runtime); commands run by the real interpreter run."""
    stub = subprocess.run

    def run(command, **kwargs):
        if str(command[0]) != str(PYTHON):
            return stub(command, **kwargs)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8") as process:
            out, err = process.communicate(timeout=600)
        if kwargs.get("check") and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, out, err)
        return subprocess.CompletedProcess(command, process.returncode, out, err)
    monkeypatch.setattr(subprocess, "run", run)
    return builder_case


def _desktop(root, licence):
    """A tiny desktop package with the real catalogues' shape and the engine i18n it needs."""
    desktop = root / "src/aester_desktop"
    desktop.mkdir(parents=True, exist_ok=True)   # the fixture snapshot already carries the app's assets
    (desktop / "__init__.py").write_text(
        "# SPDX-License-Identifier: GPL-3.0-or-later\n"
        "from starstack import i18n as _i18n\n"
        "_i18n.register_catalogs('aester_desktop.translations_gui', 'aester_desktop.translations_dialogs')\n", encoding="utf-8")
    (desktop / "translations_gui.py").write_text("EN = {'测试': 'Test'}\n", encoding="utf-8")
    (desktop / "translations_dialogs.py").write_text("EN = {}\n", encoding="utf-8")
    (desktop / "gui.py").write_text("VALUE = 'desktop gui'\n", encoding="utf-8")
    (desktop / "LICENSE.txt").write_text(licence, encoding="utf-8")
    engine = root / "src/starstack"
    for name in ("__init__.py", "i18n.py", "translations_onboarding.py", "install_lifecycle.py"):
        shutil.copyfile(Path(__file__).resolve().parents[2] / "src/starstack" / name, engine / name)
    return desktop


def test_the_desktop_app_ships_as_gpl_source(real_python):
    c = real_python
    _desktop(c.builder.ROOT, "Aester desktop application\nCopyright (C) 2026 Test Holder\n\n" + GPL)
    receipt = c.builder.build(c.output, c.compiler, c.model, smoke_python=str(PYTHON))
    stage = Path(receipt["stage"])
    app = stage / "application"
    shipped = app / "src/aester_desktop"
    assert sorted(p.name for p in shipped.glob("*.py")) == ["__init__.py", "gui.py", "translations_dialogs.py",
                                                           "translations_gui.py"]
    assert not list(shipped.rglob("*.pyc")) and not list(shipped.rglob("__pycache__"))
    assert (shipped / "LICENSE.txt").read_text(encoding="utf-8").startswith("Aester desktop application")
    assert not (shipped / "legal").exists()
    contents = json.loads((app / "PACKAGE_CONTENTS.json").read_text(encoding="utf-8"))
    assert contents["entry_modules"] == {"mcp": "starstack.mcp_server", "gui": "aester_desktop.gui"}
    assert contents["activation_probe"][0] == "aester_desktop.gui"
    assert "src/aester_desktop/gui.py" in contents["sha256"] and "src/aester_desktop/LICENSE.txt" in contents["sha256"]
    desktop = contents["parts"]["desktop"]
    assert desktop["licence"] == "GPL-3.0-or-later" and desktop["form"] == "source" and desktop["modules"] == 4
    # the shipped source imports in a fresh interpreter with only the staged tree on the path
    probe = subprocess.run([str(PYTHON), "-I", "-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                            "import aester_desktop.gui as g; from starstack import i18n; i18n.set_language('en'); "
                            "print(g.VALUE, i18n.tr('测试'), g.__file__.endswith('.py'))", str(app / "src")],
                           capture_output=True, text=True, encoding="utf-8", check=True)
    assert probe.stdout.split() == ["desktop", "gui", "Test", "True"]
    pages = {language: (stage / f"LICENSE-PAGE.{language}.txt").read_text(encoding="utf-8-sig") for language in ("zh_CN", "en")}
    assert "GPL-3.0-or-later" in pages["en"] and "Copyright (C) 2026 Test Holder" in pages["en"]
    assert "Apache License 2.0" in pages["en"] and "TRADEMARKS.md" in pages["en"] and "Preview build" in pages["en"]
    assert "GPL-3.0-or-later" in pages["zh_CN"] and "预览版" in pages["zh_CN"] and "EULA" not in pages["zh_CN"] + pages["en"]
    assert receipt["channel"] == "preview" and receipt["licences"] == {"engine": "Apache-2.0", "desktop": "GPL-3.0-or-later"}
    assert "eula_status" not in receipt
    assert not {"G1_gui_licence", "G2_copyright"} & set(receipt["release_gates"])


def test_placeholder_notices_open_the_licence_gates(real_python):
    c = real_python
    _desktop(c.builder.ROOT, "LICENSOR-NAME-PENDING\n")
    receipt = c.builder.build(c.output, c.compiler, c.model, smoke_python=str(PYTHON))
    assert {"G1_gui_licence", "G2_copyright"} <= set(receipt["release_gates"])


def test_release_builds_refuse_while_a_gate_is_open(real_python):
    c = real_python
    _desktop(c.builder.ROOT, "LICENSOR-NAME-PENDING\n")
    with pytest.raises(ValueError, match="release gates"):
        c.builder.build(c.output, c.compiler, c.model, smoke_python=str(PYTHON), channel="release")
    assert not list(c.output.glob("*.exe"))


def test_an_official_build_requires_the_desktop_app(builder_case):
    c = builder_case
    with pytest.raises(ValueError, match="desktop package"):
        c.builder.build(c.output, c.compiler, c.model, desktop="required")
