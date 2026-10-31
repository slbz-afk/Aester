# Optional natural-language retouching components

GIMP 3.2.6 is required by the current optional retouch executor. When selected,
Aester downloads the unmodified Windows installer from the official GIMP
distribution service or a listed mirror (opencolo.mm.fcix.net from GIMP's own
mirror network, Nanjing University, USTC, Aliyun), whichever measures faster,
verifies its exact size and SHA256, and runs a per-user installation. The
pinned SHA256 makes every source yield the same official installer. Aester does
not include or redistribute a private GIMP runtime.
The checked identities are recorded in `retouch-components.json` and enforced
in `retouch_distribution.py`.

Existing GIMP installations are only detected. Compatible GIMP 3.2.6 is reused;
an older, newer, missing or damaged selection requires the user to select or
manage GIMP separately. Aester never upgrades, adopts or removes an existing
GIMP. New installations are registered by the official installer, with its
Start menu entry, supported-file associations and own Windows uninstall entry.
Uninstall GIMP separately from Windows Settings when desired.

GIMP is free software under GPL-3.0-or-later; the official installer retains its
license notices. [Official downloads](https://www.gimp.org/downloads/) provide
the fixed binary and main-source hashes. The
[main source](https://download.gimp.org/gimp/v3.2/gimp-3.2.6.tar.xz) includes
the installer script. Its fixed `GIMP-3` identity is why another invocation is
not a private portable installation. The main tarball alone is not a complete
source redistribution for all bundled dependencies; Aester does not claim it
is, or bundle the binary.

GIMP installation is an independent transaction. Failed or cancelled Aester
preparation keeps the prior active Aester version. If GIMP completed first, it
remains independently installed and can be removed with its own uninstaller.
Interrupted upstream installation may require upstream repair. Aester never
tries to repair this by deleting an arbitrary GIMP directory or registry key.

The optional retouch skill is provided as seven reviewed text files and a ZIP.
Preparing it creates an application-bound copy in Aester's private component
area. Ask your Agent to read the complete directory, or choose to install the
ZIP yourself. Aester never writes to global Agent skill folders or starts an
Agent. No tutorial photographs are included, and the package does not grant
redistribution rights to linked third-party learning material.

自然语言修图选装 GIMP 使用官方固定版本安装器。已有安装只检测和复用，不覆盖或升级。
GIMP 独立登记，卸载 Aester 时保留；需要时请在 Windows 设置中单独卸载 GIMP。
修图技能仅在应用内提供源码和 ZIP，建议 Agent 加载或用户自行安装，不写全局技能目录。
