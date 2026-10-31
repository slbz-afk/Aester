# Licensing

Aester is free and open-source software. It has two parts with different licences.

- **Processing engine: Apache License 2.0** (`LICENSE`).
  - Covers the StarStack engine `src/starstack`: the engine, command line, MCP server and evaluation tools.
  - Also covers the installer recipe and workers in `packaging/`, the retouch skill in `skills/`, and the scripts and tests not listed under `[gpl]` in `open-core.toml`.
- **Desktop application: GNU General Public License v3.0 or later** (`src/aester_desktop/LICENSE.txt`).
  - Covers the package `src/aester_desktop` with its icon and artwork, its tests in `tests/desktop`, and the GUI scripts listed under `[gpl]` in `open-core.toml`.
  - Copyright (C) 2026 Wang Jian (王俭).
  - Installers ship its complete Python source.
- **Why two licences.**
  - The engine stays permissive so that other tools and AI assistants can build on it.
  - The application is copyleft so that modified versions of the app stay free.
  - Apache-2.0 code may be used in a GPLv3 program, but not the other way round. Code therefore does not move from the application into the engine without the copyright holder's consent, and the engine never imports the application (`tests/test_open_core_boundary.py`).
- **Third-party components** keep their own licences (`THIRD_PARTY.md`, `NOTICE.md`). Examples: LGPL-3.0 for Qt/PySide6, GPL-3.0-or-later for Siril and GIMP, SIL OFL 1.1 for the bundled fonts.
- **Name and icon.**
  - Neither licence grants trademark rights.
  - "Aester" and its icon identify the official project, so modified versions must use a different name and icon (GPLv3 section 7(e); see `TRADEMARKS.md`).
- **Contributions** are accepted under the licence of the part they change, with a Developer Certificate of Origin sign-off (`CONTRIBUTING.md`).

## History

- **Before 2026-10-14**: all source code was under Apache-2.0, including the desktop interface of that time. That licence is not withdrawn for those copies.
- **2026-10-14 to 2026-10-31**: the desktop application was proprietary.
  - Internal preview builds shipped it as compiled bytecode under a placeholder EULA.
  - The EULA was never finalised, and there was no paid release.
- **From 2026-10-31**: the desktop application is under GPL-3.0-or-later, the software is free of charge, and support is voluntary (user decision of 2026-10-26).

> Aester 是自由开源软件：
>
> - **处理引擎**：引擎、命令行与 MCP 服务按 Apache-2.0 授权；
> - **桌面程序**：按 GPL-3.0-or-later 授权，安装包附带完整源码；
> - **使用与支持**：软件完整免费，支持开发完全自愿，不解锁任何功能；
> - **名称与图标**：不在许可范围内，修改后再发布的版本须改用其他名称和图标（见 `TRADEMARKS.md`）。
