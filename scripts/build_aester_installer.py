"""Build the per-user Aester installer from an explicit source/model allowlist.

The compiler is an official Inno Setup distribution extracted locally. This
script never installs a build tool globally or copies developer environments.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from starstack import components


def _finite_bundle_files(bundle):
    """Accept only the separately reviewed complete binary/source distribution."""
    from starstack import finite_distribution
    # Validate the caller's path before any resolution that could hide a link.
    bundle = Path(bundle).expanduser().absolute()
    checked = finite_distribution.validate_distribution(bundle)
    files = [(p, p.relative_to(bundle)) for p in sorted(bundle.rglob("*")) if p.is_file()]
    return bundle, checked, files


# The desktop app (GPL-3.0-or-later since 2026-10-31; LICENSING.md). This constant is the one place in the
# Apache-2.0 code that names it. The build ships its Python source, like the engine's.
DESKTOP_PACKAGE = "aester_desktop"
DESKTOP = Path("src") / DESKTOP_PACKAGE
CHANNELS = ("preview", "release")
LICENCE_TEXTS = ("LGPL-3.0.txt", "GPL-3.0.txt", "flatbuffers-Apache-2.0.txt")
# Bare bundled runtime (no third-party wheels): the app package imports from its staged source and its
# catalogues apply. -B keeps the staged tree free of __pycache__.
_SMOKE = r'''
import sys
sys.path.insert(0, sys.argv[1])
import aester_desktop, aester_desktop.translations_gui as catalogue, aester_desktop.translations_dialogs
from starstack import i18n
i18n.set_language("en")
key = next(iter(catalogue.EN))
if catalogue.EN[key] == key or i18n.tr(key) != catalogue.EN[key]:
    raise SystemExit("desktop catalogue not applied: " + key)
print("desktop smoke ok")
'''


def _desktop_sources(root):
    folder = Path(root) / DESKTOP
    return [(p, p.relative_to(root)) for p in sorted(folder.rglob("*.py")) if "__pycache__" not in p.parts]


def _smoke_desktop(python, application_src):
    subprocess.run([str(python), "-I", "-B", "-X", "utf8", "-c", _SMOKE, str(application_src)],
                   capture_output=True, text=True, check=True, timeout=300)


def _source_commit(root):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True, timeout=60).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def release_gates(root, desktop_present):
    """Open release gates {name: reason}. A ``--channel release`` build refuses while any is open; a preview
    build records them in its receipt and says so on the licence page."""
    root = Path(root)
    gates = {}
    licence = root / DESKTOP / "LICENSE.txt"
    text = licence.read_text(encoding="utf-8") if licence.is_file() else ""
    if "GNU GENERAL PUBLIC LICENSE" not in text or "Version 3, 29 June 2007" not in text:
        gates["G1_gui_licence"] = "src/aester_desktop/LICENSE.txt lacks the GPLv3 notice and full text"
    notices = [licence, root / DESKTOP / "__init__.py"]
    if any(p.is_file() and "LICENSOR-NAME-PENDING" in p.read_text(encoding="utf-8") for p in notices):
        gates["G2_copyright"] = "the copyright holder is still LICENSOR-NAME-PENDING"
    missing = [name for name in LICENCE_TEXTS if not (root / "packaging/aester_p2" / name).is_file()]
    if missing:
        gates["G3_licence_texts"] = "missing: " + ", ".join(missing)
    manual = root / "packaging/release-gates.json"
    if manual.is_file():
        for name, gate in sorted((json.loads(manual.read_text(encoding="utf-8")).get("manual") or {}).items()):
            if not isinstance(gate, dict) or gate.get("state") != "closed":
                gates["G4_" + name] = (gate or {}).get("note") or "open"
    else:
        gates["G4_manual"] = "packaging/release-gates.json is missing"
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "packaging", "scripts/build_aester_installer.py"],
                               cwd=root, capture_output=True, text=True, check=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        dirty = "git status unavailable"
    if dirty:
        gates["G5_traceability"] = "uncommitted changes in src/, packaging/ or the build script (or no git)"
    if not desktop_present:
        gates["G6_product"] = "the desktop package is absent"
    return gates


def _licence_pages(root, stage, desktop_present, channel):
    """Inno licence pages: the desktop app's GPL notice, the engine's Apache-2.0 notice, third parties, the name."""
    root = Path(root)
    if not desktop_present:
        apache = (root / "LICENSE").read_text(encoding="utf-8")
        for language in ("zh_CN", "en"):
            (Path(stage) / f"LICENSE-PAGE.{language}.txt").write_text(apache, encoding="utf-8-sig")
        return
    texts = {
        "zh_CN": ("【预览版】\n\n" if channel != "release" else "") +
                 "Aester 是自由软件，完整免费；支持开发完全自愿，不解锁任何功能。\n\n"
                 "一、Aester 桌面程序：GNU 通用公共许可证第 3 版或更新版本（GPL-3.0-or-later）\n\n{notice}\n\n"
                 "本程序是自由软件：您可以依自由软件基金会发布的 GNU 通用公共许可证第 3 版或（由您选择）任何更新版本再分发"
                 "和/或修改它。本程序不提供任何担保。安装目录 versions\\<版本>\\src\\aester_desktop 中附带桌面程序的完整"
                 "源码，许可证全文见同目录 LICENSE.txt。\n\n"
                 "二、处理引擎：Apache License 2.0\n\n本安装包同时包含 StarStack 处理引擎、命令行与 MCP 服务的源码，按 Apache "
                 "License 2.0 提供，全文见安装目录 versions\\<版本>\\LICENSE。\n\n"
                 "三、第三方组件\n\nPython、Qt/PySide6（LGPL-3.0）、LibRaw（LGPL-2.1）、可选的 Siril 改进构建（GPL-3.0，附对应源码）"
                 "等各自保留原许可证，见安装目录中的 THIRD_PARTY.md、NOTICE.md 与许可证正文。\n\n"
                 "四、名称与图标\n\nAester 名称与图标不在上述许可证范围内；修改后再发布的版本须改用其他名称和图标，见 TRADEMARKS.md。\n",
        "en": ("[Preview build]\n\n" if channel != "release" else "") +
              "Aester is free software, free of charge; supporting development is voluntary and unlocks nothing.\n\n"
              "1. Aester desktop application: GNU General Public License v3.0 or later (GPL-3.0-or-later)\n\n{notice}\n\n"
              "This program is free software: you can redistribute it and/or modify it under the terms of the GNU General "
              "Public License as published by the Free Software Foundation, either version 3 of the License, or (at your "
              "option) any later version. This program comes with ABSOLUTELY NO WARRANTY. Its complete source is installed "
              "in versions\\<version>\\src\\aester_desktop, with the full licence text in LICENSE.txt there.\n\n"
              "2. Processing engine: Apache License 2.0\n\nThis installer also contains the source code of the StarStack "
              "processing engine, command line and MCP server under the Apache License 2.0 (full text: "
              "versions\\<version>\\LICENSE in the installation folder).\n\n"
              "3. Third-party components\n\nPython, Qt/PySide6 (LGPL-3.0), LibRaw (LGPL-2.1), the optional Siril improved "
              "build (GPL-3.0, with its corresponding source) and others keep their own licences; see THIRD_PARTY.md, "
              "NOTICE.md and the licence texts in the installation folder.\n\n"
              "4. Name and icon\n\nThe name Aester and its icon are not licensed under the licences above; modified "
              "versions must use a different name and icon (see TRADEMARKS.md).\n",
    }
    first = (root / DESKTOP / "LICENSE.txt").read_text(encoding="utf-8").splitlines()[:2]
    for language, text in texts.items():
        (Path(stage) / f"LICENSE-PAGE.{language}.txt").write_text(text.format(notice="\n".join(first)),
                                                                    encoding="utf-8-sig")


FONT_DIR = Path("src/aester_desktop/assets/fonts")
FONT_SUFFIXES = {".otf", ".ttf"}


def _font_files(root):
    """Bundled OFL fonts, each checked against the reviewed FONTS.json record.

    A snapshot without the font module and font directory (an isolated test
    fixture) has nothing to ship. Once either exists, every recorded file must be
    present and unmodified, nothing unrecorded may be shipped, and every font
    needs its SIL OFL licence text beside it; otherwise the build is refused.
    """
    root = Path(root)
    folder = root / FONT_DIR
    if not folder.exists() and not (root / "src/aester_desktop/fonts.py").is_file():
        return []
    manifest = folder / "FONTS.json"
    if not manifest.is_file():
        raise ValueError("Bundled fonts have no FONTS.json record; refusing to build")
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        expected = {entry["file"]: entry for entry in entries}
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("FONTS.json is not a valid font record; refusing to build") from exc
    if len(expected) != len(entries) or any(Path(name).name != name for name in expected):
        raise ValueError("FONTS.json lists duplicate or nested files; refusing to build")
    present = {p.name: p for p in folder.iterdir()}
    unexpected = sorted(set(present) - set(expected) - {"FONTS.json"})
    if unexpected:
        raise ValueError("Unreviewed files in the bundled font directory: " + ", ".join(unexpected))
    missing = sorted(name for name in expected if name not in present or not present[name].is_file()
                     or present[name].is_symlink())
    if missing:
        raise ValueError("Bundled font files are missing: " + ", ".join(missing))
    for name, entry in sorted(expected.items()):
        path = present[name]
        if entry.get("licence") != "SIL OFL 1.1":
            raise ValueError(f"Bundled font file {name} is not recorded as SIL OFL 1.1")
        if path.stat().st_size != entry.get("bytes") or components._digest(path) != entry.get("sha256"):
            raise ValueError(f"Bundled font file {name} does not match FONTS.json; refusing to build")
    fonts = [name for name in expected if Path(name).suffix.lower() in FONT_SUFFIXES]
    licences = [name for name in expected if name.startswith("OFL-") and name.endswith(".txt")]
    if fonts and not licences:
        raise ValueError("Bundled fonts are missing their SIL OFL licence texts")
    return [(present[name], FONT_DIR / name) for name in sorted(expected)] + [(manifest, FONT_DIR / "FONTS.json")]


def build(output, compiler, model=None, seed_cache=None, *, app_identity=None, no_icons=False, finite_bundle=None,
          include_retouch=False, download_source="auto", channel="preview", desktop="optional", smoke_python=None):
    """``desktop``: "required" (the command line: an official build must carry the app) or "optional" (engine-only
    test snapshots). ``smoke_python`` replaces the bundled runtime for the desktop import smoke (tests only)."""
    components.DownloadSources(download_source)  # refuse an unknown source choice before building anything
    if channel not in CHANNELS:
        raise ValueError("channel must be preview or release")
    if desktop not in ("optional", "required"):
        raise ValueError("desktop must be optional or required")
    if app_identity is not None and not re.fullmatch(r'[A-Fa-f0-9]{8}(?:-[A-Fa-f0-9]{4}){3}-[A-Fa-f0-9]{12}', app_identity):
        raise ValueError('app_identity must be a GUID without braces')
    desktop_present = (ROOT / DESKTOP / "__init__.py").is_file()
    if desktop == "required" and not desktop_present:
        raise ValueError("The desktop package (src/aester_desktop) is absent; refusing an official build")
    gates = release_gates(ROOT, desktop_present)
    if channel == "release" and gates:
        raise ValueError("Release build refused; open release gates: " + "; ".join(f"{k}: {v}" for k, v in gates.items()))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = components._manifest(ROOT)
    allow = [(p, p.relative_to(ROOT)) for p in sorted((ROOT / "src/starstack").glob("*.py"))]
    allow += [(p, p.relative_to(ROOT)) for p in sorted((ROOT / "src/aester_desktop/assets").glob("*")) if p.suffix in {".png", ".svg"}]
    allow += _font_files(ROOT)
    allow += [(ROOT / name, Path(name)) for name in ("LICENSE", "NOTICE.md", "THIRD_PARTY.md")]
    allow += [(ROOT / name, Path(name)) for name in ("LICENSING.md", "TRADEMARKS.md") if (ROOT / name).is_file()]
    desktop_sources = _desktop_sources(ROOT) if desktop_present else []
    if desktop_present:   # GPL-3.0-or-later: the app ships as source, with its licence
        allow += desktop_sources + [(ROOT / DESKTOP / "LICENSE.txt", DESKTOP / "LICENSE.txt")]
    allow += [(p, p.relative_to(ROOT)) for p in sorted((ROOT / "packaging/aester_p2").glob("*")) if p.suffix in {".json", ".md", ".txt"}]
    if include_retouch:
        from starstack import retouch_distribution
        allow += [(p, Path('skills/aester-retouch') / relative)
                  for p, relative in retouch_distribution.skill_files(ROOT / 'skills/aester-retouch')]
    hashes = {str(rel).replace("\\", "/"): components._digest(p) for p, rel in allow}
    build_inputs = {"installer/" + p.name: components._digest(p) for p in (ROOT / "packaging/aester_p2").glob("*") if p.is_file()}
    build_inputs.update({"scripts/build_aester_installer.py": components._digest(Path(__file__)),
                         "compiler/ISCC.exe": components._digest(Path(compiler))})
    finite_files, finite_info = [], None
    if finite_bundle is not None:
        finite_bundle, finite_info, finite_files = _finite_bundle_files(finite_bundle)
        build_inputs.update({"finite-bundle/" + rel.as_posix(): components._digest(p) for p, rel in finite_files})
    build_inputs["build/channel"] = channel
    version = hashlib.sha256(json.dumps({"files": hashes, "build_inputs": build_inputs}, sort_keys=True).encode()).hexdigest()[:12]
    variant = ('-' + app_identity[:8] if app_identity else '') + ('-noshell' if no_icons else '')
    exe = output / f"Aester-P2-{version}{variant}-Setup.exe"
    published = output / (exe.stem + ".json")
    if exe.exists():
        if not published.is_file():
            raise ValueError("An existing installer has no trusted build receipt; it will not be overwritten")
        receipt = json.loads(published.read_text(encoding="utf-8"))
        if receipt.get("sha256") != components._digest(exe) or receipt.get("bytes") != exe.stat().st_size:
            raise ValueError("The existing frozen installer changed; it will not be overwritten")
        print(json.dumps({**receipt, "frozen_reuse": True}, indent=2))
        return receipt
    stage = output / ("build-" + version + "-" + uuid.uuid4().hex[:6])
    stage.mkdir()
    app = stage / "application"
    for source, relative in allow:
        dest = app / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if components._digest(dest) != hashes[str(relative).replace("\\", "/")]:
            raise ValueError("Source changed during the build; rebuild a consistent snapshot")
    if include_retouch:
        archive = retouch_distribution.create_skill_archive(app / 'skills/aester-retouch', app / 'skills/aester-retouch.zip')
        hashes['skills/aester-retouch.zip'] = components._digest(archive)
    cache = output / "download-cache"
    cache.mkdir(exist_ok=True)
    if seed_cache:
        source = Path(seed_cache) / data["python"]["filename"]
        if components._valid_file(source, data["python"]):
            shutil.copyfile(source, cache / source.name)
    # The Python seed comes from python.org or a listed mirror (auto: the fastest); its pinned SHA-256 decides.
    def report(event):
        if event.get("phase") == "source":
            print(f"Python seed download source: {event.get('host')}", flush=True)
    python_zip = components._download(data["python"], cache, report,
                                      components.DownloadSources(download_source, [data["python"]]))
    boot = stage / "bootstrap"
    components._extract(python_zip, boot)
    desktop_record = {"included": False}
    if desktop_present:
        python = Path(smoke_python) if smoke_python else boot / "python.exe"
        _smoke_desktop(python, app / "src")
        cached = sorted(p.relative_to(app).as_posix() for p in (app / DESKTOP).rglob("__pycache__"))
        if cached:
            raise ValueError("The desktop smoke left bytecode in the staged tree: " + ", ".join(cached))
        desktop_record = {"included": True, "form": "source", "modules": len(desktop_sources),
                          "source_commit": _source_commit(ROOT)}
    entry_modules = {"mcp": "starstack.mcp_server", **({"gui": DESKTOP_PACKAGE + ".gui"} if desktop_present else {})}
    activation_probe = ([DESKTOP_PACKAGE + ".gui"] if desktop_present else []) + ["starstack.mcp_server", "starstack.workflow"]
    (app / "PACKAGE_CONTENTS.json").write_text(json.dumps({"schema_version": 1, "version_id": version,
        "sha256": hashes, "build_inputs": build_inputs, "entry_modules": entry_modules,
        "activation_probe": activation_probe, "channel": channel,
        "parts": {"engine": {"licence": "Apache-2.0"},
                  **({"desktop": {"licence": "GPL-3.0-or-later", **desktop_record}} if desktop_present else {})}},
        indent=2), encoding="utf-8")
    if finite_files:
        for source, relative in finite_files:
            destination = stage / "finite-bundle" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if components._digest(destination) != build_inputs["finite-bundle/" + relative.as_posix()]:
                raise ValueError("Finite distribution changed during the build")
        # Recheck the complete staged distribution, not only the executable.
        _finite_bundle_files(stage / "finite-bundle")
    for name, source in {"components.py": ROOT / "src/starstack/components.py", "install_ownership.py": ROOT / "src/starstack/install_ownership.py", "install_lifecycle.py": ROOT / "src/starstack/install_lifecycle.py", "installer_worker.py": ROOT / "packaging/aester_p2/installer_worker.py", "installer_progress.py": ROOT / "packaging/aester_p2/installer_progress.py", "download-manifest.json": ROOT / "packaging/aester_p2/download-manifest.json", "nox-LICENSE.txt": ROOT / "packaging/aester_p2/nox-LICENSE.txt"}.items():
        shutil.copyfile(source, boot / name)
    for name in ("nox-author-license-readme.md", "nox-conversion-evidence.json"):
        shutil.copyfile(ROOT / "packaging/aester_p2" / name, boot / name)
    if finite_files:
        for name in ("finite_backend_assets.py", "finite_distribution.py"):
            shutil.copyfile(ROOT / "src/starstack" / name, boot / name)
    if include_retouch:
        shutil.copyfile(ROOT / 'src/starstack/retouch_distribution.py', boot / 'retouch_distribution.py')
        # These discovery/owned-process helpers have no third-party imports.
        helpers = boot / 'starstack'
        helpers.mkdir()
        (helpers / '__init__.py').write_text('', encoding='utf-8')
        for name in ('finishing_components.py', 'finishing_assets.py', 'process_guard.py'):
            shutil.copyfile(ROOT / 'src/starstack' / name, helpers / name)
        shutil.copytree(app / 'skills/aester-retouch', boot / 'retouch-skill/aester-retouch')
    # Embed the compressed official Python seed too, so preparation requires no second Python download.
    shutil.copyfile(python_zip, boot / python_zip.name)
    support = stage / 'uninstall-support'
    components._extract(python_zip, support)
    for name in ('components.py', 'install_ownership.py', 'install_lifecycle.py', 'uninstall_options.py'):
        shutil.copyfile(ROOT / 'src/starstack' / name, support / name)
    shutil.copyfile(ROOT / 'packaging/aester_p2/uninstall_worker.py', support / 'uninstall_worker.py')
    # Frozen hashes are compiled into Inno, rather than read from a mutable
    # manifest beside the helper. Only fixed flat bootstrap files are copied.
    copy_lines = ['procedure CopyUninstallSupport(const Dest: String);', 'begin', '  ForceDirectories(Dest);']
    for file in sorted(support.iterdir()):
        if not file.is_file() or "'" in file.name: raise ValueError('Unexpected uninstall support layout')
        digest = components._digest(file)
        source = "ExpandConstant('{app}\\uninstall-support\\" + file.name + "')"
        copy_lines += [f"  if GetSHA256OfFile({source}) <> '{digest}' then RaiseException(CustomMessage('HelperChanged'));",
                       f"  if not FileCopy({source}, Dest + '\\{file.name}', False) then RaiseException(CustomMessage('HelperCopyFailed'));"]
    copy_lines += ['end;']
    (stage / 'uninstall-files.iss').write_text('\n'.join(copy_lines), encoding='utf-8')
    # ``model`` remains an ignored compatibility argument for old build callers.
    # Model installation now requires an explicit choice in the running application.
    from PIL import Image
    Image.open(ROOT / "src/aester_desktop/assets/aester.png").save(stage / "Aester.ico", sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
    compiler_cs = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    for name, target, extra in [("Aester.exe", "winexe", []), ("Aester-MCP.exe", "exe", ["/define:MCP"])]:
        subprocess.run([str(compiler_cs), "/nologo", "/optimize+", f"/target:{target}", "/platform:anycpu",
             "/reference:System.Web.Extensions.dll", "/reference:System.Windows.Forms.dll", f"/win32icon:{stage / 'Aester.ico'}",
             f"/out:{stage / name}", *extra, str(ROOT / "packaging/aester_p2/Launcher.cs")], check=True)
    (stage / "INSTALL-NOTES.txt").write_text(
        "Aester P2 Preview\n\n安装时下载固定版本基础组件约290 MB。\n"
        "安装器先测速，从官方服务器或所列镜像中较快的一处下载；每个文件都按固定的大小与 SHA-256 校验。\n"
        "安装包不包含 nox 模型。首次启动可自愿安装第三方 nox：约218 MB模型和15 MB运行库，下载地址须已配置。\n"
        "也可稍后在组件管理安装或选择已有兼容模型。模型结果是算法估计，不保证深空无星层质量。\n\n"
        + ("可选 Siril · Aester 改进构建已内附运行库、许可证与对应源码，选中后无需另找文件或额外下载。\n"
           "它保留有效零值、负值与高光余量；安装后可在设置的处理后端中选择，不覆盖已有后端偏好。\n\n" if finite_files else "") +
        ("可选 GIMP 3.2.6（自然语言修图所需）由官方独立安装器下载、校验并安装，约190 MB。已有GIMP只检测，绝不覆盖升级。\n"
         "GIMP有自己的开始菜单、文件关联和卸载入口；Aester失败或卸载时保留它，可在Windows设置中单独卸载。\n"
         "可选修图技能仅在应用内提供文本与ZIP；建议Agent加载，或用户自行安装。不会写入全局技能目录。\n\n" if include_retouch else "") +
        "请保持网络连接直至安装完成。完成后GUI和MCP均可离线启动，无需全局Python或命令行环境搭建。\n"
        "不修改系统PATH。卸载默认移除程序及本安装独占的运行库、模型和已登记下载包；作品、原片与未知文件保留。\n"
        "运行中的GUI、Agent或后台作业会阻止卸载；文件删除失败时保留卸载入口，排除问题后可重试。\n"
        "共享或旧版未登记归属的组件保留并说明；作业缓存与设置默认保留，可在卸载时选择安全清理。\n"
        "首次启动可选择连接Agent（MCP），也可跳过。安装不自动开启去星或亮迹保护。\n"
        "Aester 是自由软件：处理引擎按 Apache-2.0、桌面程序按 GPL-3.0-or-later 授权，均附源码；许可证文本在安装目录 versions\\<版本>\\ 下。\n",
        encoding="utf-8-sig")
    english_notes = (ROOT / 'packaging/aester_p2/INSTALL-NOTES-en.txt').read_text(encoding='utf-8')
    english_notes = english_notes.replace('{{FINITE_COMPONENT_NOTE}}',
        ('The optional Siril - Aester improved build includes its runtime, licenses and corresponding source. '
         'Selecting it requires no separate file selection or additional download. It preserves valid zero, negative and '
         'above-range values. Choose it under Settings > Processing backend after installation; existing backend preferences '
         'are not changed.') if finite_files else '')
    if include_retouch:
        english_notes += ('\n\nOptional GIMP 3.2.6 is required for natural-language retouching. '
            'Its official independent installer is downloaded (about 190 MB), verified and run. '
            'Existing GIMP installations are detected and never upgraded or overwritten. '
            'GIMP creates its own Start menu entry, file associations and Windows uninstall entry. '
            'It remains installed if Aester fails or is uninstalled; remove it separately in Windows Settings. '
            'The optional retouch skill is provided as application-local text and ZIP. '
            'Ask an Agent to load it or install it yourself; Aester never writes to global skill directories.\n')
    (stage / 'INSTALL-NOTES-en.txt').write_text(english_notes, encoding='utf-8-sig')
    _licence_pages(ROOT, stage, desktop_present, channel)
    # Installed application size for the components page estimate (Inno adds each component's extra).
    app_mb = sum(path.stat().st_size for folder in (app, support) for path in folder.rglob('*') if path.is_file()) / 1048576
    app_mb += sum((stage / name).stat().st_size for name in ('Aester.exe', 'Aester-MCP.exe', 'Aester.ico')) / 1048576
    command = [str(compiler), "/Qp", f"/DStagePath={stage}", f"/DVersionId={version}", f"/DBuildOutput={output}",
               f"/DInstallerName={exe.stem}", f"/DAppMB={max(1, round(app_mb))}"]
    if app_identity: command.append('/DAppIdentity=' + app_identity)
    if no_icons: command.append('/DNoIcons=1')
    if finite_files: command.append('/DIncludeFinite=1')
    if include_retouch: command.append('/DIncludeRetouch=1')
    command.append(str(ROOT / "packaging/aester_p2/Aester.iss"))
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    (stage / "compiler.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    receipt = {"installer": str(exe), "bytes": exe.stat().st_size, "sha256": components._digest(exe),
               "version_id": version, "stage": str(stage), "source_files": len(hashes), "signature": "unsigned",
               "compiler": str(Path(compiler).resolve()), "compiler_sha256": components._digest(Path(compiler)),
               "model_sha256": None, "nox_model_bundled": False,
               "network_required_during_install": True, "network_required_after_install": False,
               "nox_installation": "first-launch-opt-in"}
    receipt.update(app_identity=app_identity or 'BD0B9C56-1F68-4D33-9C51-208037FC67B7', no_icons=bool(no_icons))
    receipt["finite_distribution"] = ({"included": True,
        "manifest_sha256": finite_info["manifest_sha256"], "files": len(finite_files),
        "bytes": sum(source.stat().st_size for source, _ in finite_files),
        "source_files_sha256": {relative.as_posix(): build_inputs['finite-bundle/' + relative.as_posix()]
                                for _, relative in finite_files if relative.parts[0] == 'sources'},
        "installation": "optional-offline", "changes_backend_preference": False} if finite_files else {"included": False})
    receipt["installer_components"] = ['base'] + (['finite'] if finite_files else []) + (['gimp', 'retouch_skill'] if include_retouch else [])
    receipt['retouch'] = {'included': bool(include_retouch),
        'gimp_distribution': 'official-upstream-download-independent-install' if include_retouch else None,
        'gimp_retained_on_aester_uninstall': True, 'global_skill_install_performed': False}
    receipt["default_installation_type"] = 'basic'
    receipt.update(channel=channel, release_gates=gates, desktop=desktop_record,
                   licences={"engine": "Apache-2.0", **({"desktop": "GPL-3.0-or-later"} if desktop_present else {})})
    published.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs/aester-p2-20260924/installer"))
    parser.add_argument("--compiler", default=str(ROOT / "tools/aester-installer/compiler/{app}/ISCC.exe"))
    parser.add_argument("--model", help="Deprecated compatibility option; models are never bundled")
    parser.add_argument("--seed-cache")
    parser.add_argument('--app-identity', help='Separate per-user registration for isolated test installations')
    parser.add_argument('--no-icons', action='store_true', help='Do not create shared start-menu or desktop entries')
    parser.add_argument('--finite-bundle', help='Reviewed complete Aester Siril binary/source distribution directory')
    parser.add_argument('--include-retouch', action='store_true', help='Include optional official GIMP installation and application-local skill source/ZIP')
    parser.add_argument('--download-source', choices=components.DOWNLOAD_SOURCE_MODES, default='auto',
                        help='Python seed download: auto measures python.org and the listed mirrors; official uses '
                             'python.org only. Either way the pinned SHA-256 decides')
    parser.add_argument('--channel', choices=CHANNELS, default='preview',
                        help='release refuses while any release gate is open (GUI licence, copyright holder, licence '
                             'texts, manual gates in packaging/release-gates.json, uncommitted sources)')
    args = parser.parse_args()
    build(args.output, args.compiler, args.model, args.seed_cache, app_identity=args.app_identity, no_icons=args.no_icons,
          finite_bundle=args.finite_bundle, include_retouch=args.include_retouch, download_source=args.download_source,
          channel=args.channel, desktop='required')
