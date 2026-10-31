# 第三方组件与分发闭包

加固工作包 H1（`docs/HARDENING_PACKAGE_20260922.md`）建立本表，2026-10-05 按当前安装器的实际内容修订（nox 一行按用户决定 NOX-RIGHTS 写，见 `docs/OPEN_DECISIONS_20261005.md`）。PyPI 表的版本取自 `requirements-lock.txt`（SHA-256 `871cdddaa6f72eb5aad53d2c5f5e2ab2a641ebb2d747e1fd6860acbccbb56a80`）；`scripts/check_third_party.py` 把锁文件与下表逐项对账（缺项、多项须为 0），并在已安装的环境里把声明的许可证与 wheel 元数据对照。该脚本不核对安装器的下载清单，与安装器有关的各栏按 `packaging/aester_p2/download-manifest.json` 手工填写。许可证为 SPDX 标识；"使用方式"只描述本产品怎样用它，不是法律意见。

本产品分两部分，均为自由软件（见 `LICENSING.md`）：处理引擎 `src/starstack`（引擎、CLI、MCP、评测）与安装器配方等为 Apache-2.0（用户决定 L1，2026-09-22；正文 `LICENSE`，声明 `NOTICE.md`）；桌面程序 `aester_desktop` 自 2026-10-31 起为 GPL-3.0-or-later，随安装包附带源码（正文 `src/aester_desktop/LICENSE.txt`）。第三方组件各自保留原许可证；独立进程、动态链接与本产品许可证不能代替各组件的分发义务核查。本文记录工程分发方案，不宣称已完成商业法律审查。

## 分发方式：当前安装器与旧便携 ZIP

本节按 2026-10-15 第二包安装配方更新（`packaging/aester_p2/Aester.iss`、`scripts/build_aester_installer.py`、`packaging/aester_p2/download-manifest.json`）。此前 10-14 候选仍含 nox，见 `docs/AESTER_AFDIAN_LAUNCH_AUDIT_20261014.md`。当前 Setup 移除模型及安装勾选，改为首次启动自愿准备；生产下载地址尚未配置，权利核查仍未完成。完整本机验证边界见 `docs/launch/package-b-20261015/README.md`。

- **Setup.exe 内附**：
  - 本产品的应用文件，安装到 `{app}\versions\<版本>\`：源码、图标、四款 OFL 字体、`LICENSE`、`NOTICE.md`、本文，以及 `packaging/aester_p2/` 下的说明与许可证正文；
  - `Aester.exe`、`Aester-MCP.exe` 启动器（本项目自有，由 `Launcher.cs` 编译）；
  - Inno Setup 的安装与卸载运行部分，以及简体中文界面文件；
  - Python 3.13.15 官方嵌入式包；
  - nox 许可、作者声明与转换证据；不含 ONNX 模型本体；
  - 构建时传入了已核验分发包的安装包，还内附 Siril · Aester 改进构建的完整分发包：运行库、对应源码与许可证。
  - nox 模型和改进构建在这样的 Setup.exe 里始终都有，只有勾选对应组件时才解包写入磁盘。
- **安装时下载**：每个文件都从官方主机或所列镜像下载，所有文件都按固定的大小与 SHA-256 校验（2026-10-09 起；来源表是 `src/starstack/components.py` 的 `DOWNLOAD_SOURCES`）。默认先对每类文件的全部来源各取前 256 KiB 测速，再按快慢依次尝试，所以同一个安装包在中国大陆和海外都能用；Setup 命令行加 `/DOWNLOADSOURCE=official` 时只用官方地址（GIMP 另加其自身镜像网络中的 opencolo.mm.fcix.net）、不测速，重定向也只能落在这些主机上。镜像只是换一处取同样的字节：2026-10-09 核对时，每个所列镜像上的全部固定文件大小都与官方一致，抽样完整下载的文件 SHA-256 也全部一致；安装时的校验不依赖这次核对。首个地址和每次重定向都只能落在本类来源的主机上（HTTPS、443 端口、不带用户信息）。wheel 与 Siril 原样解包，GIMP 安装器原样运行。wheel 不经 pip 安装，dist-info 中的许可证文件一并保留。
  - PyPI wheel：files.pythonhosted.org（官方），或腾讯云 mirrors.cloud.tencent.com、华为云 repo.huaweicloud.com、清华 pypi.tuna.tsinghua.edu.cn、中科大 mirrors.ustc.edu.cn、北外 mirrors.bfsu.edu.cn、阿里云 mirrors.aliyun.com、南大 mirror.nju.edu.cn 的 PyPI 镜像。
  - Siril 1.4.4：只从官方 free-astro.org 下载，没有镜像。
  - GIMP 安装器：download.gimp.org（官方；它按访问地点转到 GIMP 的某个镜像，不在本表的主机会被拒绝，随即改用下一处），GIMP 镜像网络中的 opencolo.mm.fcix.net，或南大、中科大、阿里云镜像。
  - Python 嵌入式包随 Setup.exe 附带，安装时不下载；只有构建安装包和应用内修复时才从 www.python.org 或 npmmirror、华为云、清华、中科大、南大的镜像下载。
  - 基础组件（必装）：41 个 PyPI wheel 与 Siril 1.4.4 官方便携包，约 290 MB。
  - 勾选 nox：另加 3 个 wheel，约 15 MB。
  - 勾选 GIMP：GIMP 官方安装器，约 190 MB。
- **默认选择**：首次安装默认“基础安装”，只装基础组件。改进构建、nox、GIMP 和修图技能只在“完整安装”中预先勾选，或由用户在“自定义”里勾选；GIMP 和修图技能只出现在含修图支持的构建里。升级时沿用上次的安装类型和组件。
- **修图技能包**（nl2-v5.1）是本项目自有的七个文本文件及其 ZIP，不含教程图片，也不授予所链接的第三方学习资料的再分发权。
- **旧便携 ZIP**：H1 阶段的便携试用 ZIP（`docs/TRIAL_RELEASE.md`）只含本产品源码、脚本、说明与引导程序，不含第三方二进制，首次启动时从官方来源获取固定组件。它已被安装器取代，其“不含二进制”的结论不适用于安装器。
- **Siril · Aester 改进构建**是与官方 Siril 并存的可选组件，以独立进程运行。只有完整分发包通过固定清单核验的构建才显示安装选项；该组件携带固定运行库、对应源码归档、许可证、修改说明和构建资料，安装到私有组件目录。具体内容以该组件的 `distribution-manifest.json` 与 `legal/` 为准。安装后可在“设置 → 处理后端”选择，不覆盖已有后端偏好。它不是 Siril 官方发布，也没有官方背书；许可证与源码材料在工程上已闭合，不等于商业发行的所有法律事项都已全面验收。

## PyPI 组件（与 `requirements-lock.txt` 逐项对应）

“随产品分发”一栏：“安装时下载（基础）”指 Setup.exe 不含，由安装器在安装时从 PyPI 下载原版 wheel 装入私有运行环境；“安装器不下载”指它在锁文件（开发环境）里，但不在安装器清单里，安装版中没有它。

| 组件 | 版本 | 许可证 | 使用方式 | 随产品分发 | 约束 / 说明 |
|---|---|---|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause | 导入，核心运行时 | 安装时下载（基础） | wheel 元数据另含随附组件 0BSD / MIT / Zlib / CC0-1.0；`numpy.libs` 另带 scipy-openblas（OpenBLAS 为 BSD-3-Clause、LAPACK 为 BSD-3-Clause-Open-MPI，内含 GPL-3.0-or-later WITH GCC-exception-3.1 的 libgfortran 运行库）和一份 msvcp140 DLL，说明见 wheel 内 `LICENSE.txt`；保留版权声明 |
| astropy | 8.0.1 | BSD-3-Clause | 导入，FITS 读写 | 安装时下载（基础） | 保留版权声明；wheel 的 `astropy/wcs/_wcs.pyd` 编入 WCSLIB（LGPL-3.0，正文 `licenses/WCSLIB_LICENSE.rst`），另随附 expat、PLY、jQuery 等许可证文件，以 wheel 内 licenses/ 为准 |
| astropy-iers-data | 0.2026.9.14.0.56.43 | BSD-3-Clause | astropy 依赖 | 安装时下载（基础） | 本产品不用其地球定向数据 |
| pyerfa | 2.0.1.5 | BSD-3-Clause | astropy 依赖 | 安装时下载（基础） | 内含 ERFA（源自 SOFA 的宽松许可证） |
| pyyaml | 6.0.3 | MIT | astropy 依赖 | 安装时下载（基础） | 保留版权声明 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | astropy / pytest 依赖 | 安装时下载（基础） | 二选一 |
| pillow | 12.3.0 | MIT-CMU | 导入，PNG/JPEG 读写 | 安装时下载（基础） | 保留版权声明；随附的编解码库以其 wheel 内 LICENSE 为准。其中 FreeType 2.14.3 为 FTL 或 GPL-2.0-or-later 二选一；按 FTL 使用时须在产品文档中注明使用了 FreeType（本产品使用 FreeType） |
| tifffile | 2026.9.9 | BSD-3-Clause | 导入，TIFF 读写、DNG 标签 | 安装时下载（基础） | 保留版权声明 |
| imagecodecs | 2026.8.16 | BSD-3-Clause | 可选依赖（星层导出压缩），动态导入 | 安装器不下载（仅开发环境） | wheel 捆绑多种编解码库（各自 BSD/MIT/zlib 等），以其 wheel 内许可证文件为准 |
| pyside6 | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | 开发环境的元包（依赖 essentials 与 addons），桌面程序 `aester_desktop`（GPL-3.0-or-later） | 安装器不下载 | 按 LGPL-3.0 使用：不静态链接、保留 Qt 许可证声明、允许用户替换 Qt 库；桌面程序本身为 GPL-3.0-or-later，与 LGPL-3.0 兼容。Qt 许可证正文的缺口见 pyside6-essentials 一行 |
| pyside6-essentials | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | pyside6 组成部分：GUI 实际使用的 Qt 6 库与 PySide6 模块（wheel 还带 MSVC 运行库与 `opengl32sw.dll`） | 安装时下载（基础） | 同上。该 wheel 只带 `LicenseRef-Qt-Commercial.txt`；LGPL-3.0 及其引用的 GPL-3.0 正文已于 2026-10-14 补入安装包（`packaging/aester_p2/LGPL-3.0.txt`、`GPL-3.0.txt`，见“许可证正文”一节）。Qt 源码获取说明仍待补（已知缺口 1） |
| pyside6-addons | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | pyside6 组成部分 | 安装器不下载 | 同上；本产品未使用其附加模块 |
| shiboken6 | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | pyside6 绑定层 | 安装时下载（基础） | 同上；该 wheel 同样只带 `LicenseRef-Qt-Commercial.txt` |
| rawpy | 0.27.1 | MIT | 可选依赖（RAW 元数据、颜色底子），动态导入 | 安装时下载（基础） | wheel 内含 LibRaw 0.22.1（LGPL-2.1-only OR CDDL-1.0）；wheel 只带 `LICENSE`（MIT）与 `LICENSE.LibRaw`（LGPL-2.1 正文），未见 GPL demosaic pack 的文件；动态链接、保留声明 |
| pywin32 | 312 | PSF-2.0 | mcp 的 Windows 依赖 | 安装时下载（基础） | PSF 只是 PyPI 元数据；wheel 内许可证正文实为 Mark Hammond 的 BSD-3-Clause 式许可（win32/、com/、pythonwin/），adodbapi 子包为 LGPL-2.1，MAPIStubLibrary 为 MIT，Scintilla 另有许可；保留全部文件；安装器不执行其安装后脚本 |
| mcp | 1.30.0 | MIT | 可选依赖（`starstack-mcp` 服务） | 安装时下载（基础） | 保留版权声明 |
| anyio | 4.15.1 | MIT | mcp 依赖 | 安装时下载（基础） | |
| attrs | 26.1.0 | MIT | jsonschema 依赖 | 安装时下载（基础） | |
| httpx | 0.28.1 | BSD-3-Clause | mcp 依赖 | 安装时下载（基础） | 本产品不经 httpx 联网；安装与组件修复只用标准库，从 `components.py` 的 `DOWNLOAD_SOURCES` 所列官方主机或镜像下载，按 SHA-256 校验 |
| httpx-sse | 0.4.3 | MIT | mcp 依赖 | 安装时下载（基础） | |
| httpcore | 1.0.9 | BSD-3-Clause | httpx 依赖 | 安装时下载（基础） | |
| h11 | 0.16.0 | MIT | httpcore / uvicorn 依赖 | 安装时下载（基础） | |
| idna | 3.19 | BSD-3-Clause | httpx 依赖 | 安装时下载（基础） | |
| certifi | 2026.7.22 | MPL-2.0 | httpx 依赖（CA 证书包） | 安装时下载（基础） | MPL-2.0 文件级 copyleft，仅当修改其源文件时须公开该文件；本产品未修改 |
| jsonschema | 4.26.0 | MIT | mcp 依赖 | 安装时下载（基础） | |
| jsonschema-specifications | 2025.9.1 | MIT | jsonschema 依赖 | 安装时下载（基础） | |
| referencing | 0.37.0 | MIT | jsonschema 依赖 | 安装时下载（基础） | |
| rpds-py | 2026.6.3 | MIT | referencing 依赖 | 安装时下载（基础） | |
| pydantic | 2.13.5 | MIT | mcp 依赖（参数校验） | 安装时下载（基础） | |
| pydantic-core | 2.46.5 | MIT | pydantic 依赖 | 安装时下载（基础） | |
| pydantic-settings | 2.15.0 | MIT | mcp 依赖 | 安装时下载（基础） | |
| annotated-types | 0.8.0 | MIT | pydantic 依赖 | 安装时下载（基础） | |
| typing-extensions | 4.16.0 | PSF-2.0 | pydantic 依赖 | 安装时下载（基础） | |
| typing-inspection | 0.4.4 | MIT | pydantic 依赖 | 安装时下载（基础） | |
| python-dotenv | 1.2.3 | BSD-3-Clause | pydantic-settings 依赖 | 安装时下载（基础） | |
| python-multipart | 0.0.32 | Apache-2.0 | mcp 依赖 | 安装时下载（基础） | |
| pyjwt | 2.14.0 | MIT | mcp 依赖 | 安装时下载（基础） | |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause | pyjwt 依赖 | 安装时下载（基础） | 二选一；含 OpenSSL（Apache-2.0） |
| cffi | 2.1.1 | MIT-0 | cryptography 依赖 | 安装时下载（基础） | MIT No Attribution（wheel 元数据 License-Expression: MIT-0） |
| pycparser | 3.0 | BSD-3-Clause | cffi 依赖 | 安装时下载（基础） | |
| sse-starlette | 3.4.11 | BSD-3-Clause | mcp 依赖 | 安装时下载（基础） | |
| starlette | 1.6.0 | BSD-3-Clause | mcp 依赖 | 安装时下载（基础） | |
| uvicorn | 0.53.0 | BSD-3-Clause | mcp 依赖 | 安装时下载（基础） | 本产品以 stdio 运行 MCP，不监听端口 |
| click | 8.5.0 | BSD-3-Clause | uvicorn 依赖 | 安装时下载（基础） | |
| colorama | 0.4.6 | BSD-3-Clause | pytest 的 Windows 依赖（click 8.5.0 不依赖它） | 安装器不下载（仅开发环境） | |
| pytest | 9.1.1 | MIT | 开发 / 测试 | 不随产品分发 | 仅开发环境 |
| pluggy | 1.6.0 | MIT | pytest 依赖 | 不随产品分发 | 仅开发环境 |
| iniconfig | 2.3.0 | MIT | pytest 依赖 | 不随产品分发 | 仅开发环境 |
| pygments | 2.21.0 | BSD-2-Clause | pytest 依赖 | 不随产品分发 | 仅开发环境 |

锁文件的 49 项中，安装器下载 41 项。pyside6（元包）、pyside6-addons、imagecodecs 不下载；pytest、pluggy、iniconfig、pygments、colorama 只用于开发环境。

## 非 PyPI 组件

| 组件 | 版本 | 许可证 | 使用方式 | 随产品分发 | 约束 / 说明 |
|---|---|---|---|---|---|
| Siril 官方构建 | 1.4.4（`siril-1.4.4-ucrt64_win.zip`，165,994,089 B，SHA-256 `ab276bed…a6ee`） | GPL-3.0-or-later；包内随附库各保留原许可证，其中有 GPL 编解码器 x264、x265、xvidcore 和 Siril 自带的 Python 3.12 | 独立进程：本产品生成命令脚本并以命令行调用 | Setup.exe 不含。属于基础组件（必装）：安装时从 free-astro.org 下载原版并校验，原样解包到私有组件目录，压缩包留在已登记的下载缓存里 | 上游许可证文件（如 `share/doc/siril/LICENSE.md`）随解包目录保留；独立进程调用不代替具体分发义务核查。若将来把该压缩包内置进安装包，须先随附（或按 GPL 提供）Siril 及其 GPL 编解码器的对应源码；现行“安装时从上游下载”方式的分发义务边界未做法律审查 |
| Siril · Aester 改进构建 | finite-validity-v1（基于 Siril 1.4.4）。下一次构建用 package-v3：分发清单 SHA-256 `df71e8ca…40d5`，86 个文件，754,747,951 B；引擎 SHA-256 `67681ac2…35cf`。09-28 测试版按推定内附 package-v2（清单 `0a1c1ffd…5c2`） | Siril 源码与本项目 2026-09-20 的修改：GPL-3.0-or-later；随附运行库各保留原许可证（见下一行） | 可选独立进程，保留有效零值、负值与高光余量 | **是**（仅限构建时传入已核验分发包的 Setup.exe）：可选组件，默认不勾，只有勾选时才连同 `legal/` 与 `sources/` 一起安装 | 同一组件附完整对应源码 `sources/Aester-Siril-finite-validity-v1-Corresponding-Source.zip`（623,339,097 B），内含上游 `siril-1.4.4.tar.bz2`、1,914 个冻结源文件、完整补丁、构建脚本和 52 个 MSYS2 源码包。`legal/` 中有 GPL 正文 `SIRIL-GPL-3.md`、`LICENSE_NOTES.md`、带修改说明的 `SOURCE_ACCESS_AND_BUILD.md`（改了 6 个代码文件）、`RUNTIME_AUX_NOTICE.md` 和 `DEPENDENCY_LICENSES.zip`。不冒称官方版本或全面法律验收。52 个源码包未做 PGP 验签，也未从源码全量重建依赖。重建的引擎须在 Aester 中登记新哈希才会被它的核验路径采用；随附文件说明这不附加其他限制 |
| 改进构建随附的运行库（MSYS2 UCRT64 二进制包） | 53 个固定二进制包，共 74 个 DLL；package-v3 另含 glib2 2.90.0-1 的 `gspawn-win64-helper.exe`、`gspawn-win64-helper-console.exe`（`gdbus.exe` 有意不带） | 按包而异：GTK3、GLib、ATK、gdk-pixbuf、FriBidi、libiconv 等为 LGPL-2.1-or-later；Cairo 为 LGPL-2.1-or-later OR MPL-1.1（两份正文都随附）；Pango 包元数据为 LGPL-2.1，随附正文为 GNU LGPL 2.0；GSL 为 GPL-3.0-or-later；FFTW、JBIG-KIT 为 GPL-2.0-or-later（据源码头）；gettext-runtime 为 GPL-3.0-or-later AND LGPL-2.1-or-later；xz 为 0BSD AND LGPL-2.1-or-later AND GPL-2.0-or-later；GCC 运行库为 GPL-3.0-or-later WITH GCC-exception-3.1（libquadmath 为 LGPL-2.1-or-later）；OpenCV、OpenSSL、oneTBB、LERC 为 Apache-2.0；libidn2、libunistring、zstd、freetype 等可在多种许可证中选择；lcms2 包元数据为 MIT AND GPL-3.0-or-later，其中 GPL 部分对应未随附的 fast_float 插件，随附的 liblcms2-2.dll 为 MIT；bzip2、fontconfig、libpng、CFITSIO 为自定义宽松许可；其余多为 MIT／BSD／Zlib 类 | 由改进构建的 `siril-cli.exe` 动态加载；两个 GLib 辅助程序由 `libglib-2.0-0.dll` 按路径启动 | **是**：随改进构建分发（可选，默认不勾） | 各包许可证正文在 `legal/DEPENDENCY_LICENSES.zip`，由其中的 `LICENSE_COVERAGE.json` 逐包映射；对应源码与配方在源码 ZIP 的 `dependencies/` 下；DLL 与包的对应见源码 ZIP 内 `provenance/ORIGINAL_DEPENDENCY_INVENTORY.json`。LGPL 库可由用户替换；Aester 只对自己的核验路径固定哈希 |
| Siril 1.4.4 源码内的第三方子项目（healpix_cxx、htmesh、kplot、librtprocess、wcslib、yyjson） | 随 Siril 1.4.4 源码 | healpix_cxx：GPL-2.0-or-later（源码头写明 “version 2 … or (at your option) any later version”；正文 `COPYING` 为 GPL v2）；htmesh：修改版 BSD，另有 6 个接口文件为修改版 BSD 与 GPL-2.0-or-later 双许可；kplot：ISC；librtprocess：GPL-3.0-or-later；wcslib：LGPL-3.0-or-later；yyjson：MIT | 按 Siril 的构建配置编入 `siril-cli.exe`；各子项目的实际链接情况未逐个核对 | 以源码形式随改进构建的源码 ZIP 分发（可选，默认不勾） | healpix_cxx、htmesh、librtprocess、wcslib、yyjson 的许可证文件在源码 ZIP 的 `siril-modified/subprojects/` 内；kplot 没有单独的许可证文件，ISC 许可写在各源文件头；`legal/` 下没有单列说明 |
| Python 嵌入式运行时（`python-3.13.15-embed-amd64.zip`） | 3.13.15（11,009,825 B，SHA-256 `d1f04d99…a2cf`） | PSF-2.0；归档内 `LICENSE.txt` 另含 Microsoft Distributable Code 附加条件、bzip2 与 libffi（MIT）的许可，以及一份未标名的 Apache-2.0 正文（对应随附的 OpenSSL 3 `libcrypto-3.dll`／`libssl-3.dll`）；随附的 SQLite（`sqlite3.dll`）以及编入扩展模块的 xz、expat 在 `LICENSE.txt` 中未单列 | 安装器引导、私有运行环境和卸载助手所用的 Python | **是**：原样内附于 Setup.exe，共三处：引导临时目录、下载缓存种子、解包到 `{app}\uninstall-support` 的副本。安装时从种子解包到私有运行环境，不再联网下载 | `LICENSE.txt` 随归档和各解包副本原样保留；私有运行环境只改写配置文件 `python313._pth`，不改程序文件；不修改系统 Python 或 PATH |
| Microsoft Visual C++ 运行库（`vcruntime140*.dll` 等） | 随 Python 3.13.15 嵌入包（`vcruntime140.dll`、`vcruntime140_1.dll`）；另随安装时下载的 wheel：pyside6-essentials 6.11.2 与 shiboken6 6.11.2（各带 `concrt140`、`msvcp140*`、`vcomp140`、`vcruntime140*`）、numpy 2.5.3（`numpy.libs/msvcp140-*.dll`）、rawpy 0.27.1（`rawpy/vcomp140.dll`） | Microsoft Distributable Code 条款（专有，允许再分发） | Python、Qt、numpy、rawpy 的 C/C++ 运行库 | Python 中的副本：**是**（Setup.exe 内附）；各 wheel 中的副本：Setup.exe 不含，随这些 wheel 在安装时下载 | 条款摘要见 Python 的 `LICENSE.txt`：不得改动 Microsoft 的声明，不得用其商标暗示来源或背书，只在 Microsoft 平台上运行，不得用于恶意软件，并要求下游接收方给予同等保护。本产品条款是否需要另行转述这项要求，正式发行前核对 |
| Qt（经 PySide6） | 6.11.2 | 各模块分别适用 LGPL / GPL / 商业许可，须按模块核对 | 动态加载的 GUI 库 | Setup.exe 不含；随 pyside6-essentials、shiboken6 wheel 在安装时下载（基础） | LGPL-3.0/GPL-3.0 正文已补；对应源码提供安排与库替换验收仍待完成。实际 wheel 还含 Lottie、Quick Timeline 等 GPL/商业许可模块，须核对使用与传递加载范围，不能将整个 wheel 笼统视为 LGPL。详见 10-15 准备包 |
| LibRaw（经 rawpy） | 0.22.1 | LGPL-2.1-only OR CDDL-1.0 | 动态链接的 RAW 解码库 | Setup.exe 不含；随 rawpy wheel 在安装时下载（基础） | 见上表 rawpy 一行；wheel 内带 `LICENSE.LibRaw` |
| RawTherapee `NIKON Z 8.dcp` | RawTherapee 5.11 提交 `db575c66` | 文件自述 EmbedPolicy=3（允许嵌入）、"public domain"（`docs/DCP_TECHNICAL_CONTRACT_20260920.md`） | 可选数据文件：仅 Z8 的 `--dcp-profile` 观感 / 矩阵对照 | **否**：不在仓库与安装包内，用户自备并以路径传入 | 保留来源与版权标签；不宣称 Adobe / Nikon 认证 |
| RawTherapee / ART DCP 的日光色相/饱和度表数值（16 个机型：索尼 α7 III／α7 IV／α7S III／α6400，尼康 Z6／Z6II／Z7／D850／D750，佳能 6D／5D Mark IV／R／RP／R5／R6／R8） | RawTherapee 提交 `6c4cb596`、ART 提交 `fae1b3bd`；每个文件的 SHA-256 固定在 `camera_look_data.py` | 各文件 ProfileCopyright 标签为“RawTherapee CC0”或“public domain”（个人署名的文件一律不用） | 数据：各机型默认观感，α7R V、α7C II、Z7 II、R6 Mark II 借用其他机型的表（用户 2026-09-26 决定；`docs/SONY_A7RV_LOOK_PREREG_20260926.md`、`docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md`） | **是**：每张 32400 字节，以 base64 写在 `src/starstack/camera_look_data.py`（由 `scripts/build_camera_look_data.py` 从固定文件生成），随应用源码进入安装包；不含 DCP 文件其余部分 | 保留来源与版权标签；这些是色卡准确度校正表，不宣称是厂商观感或 Adobe／厂商认证。来源登记见 `data/external/SOURCES.md` §6、§7 |
| nox 去星模型（本项目的 ONNX 转换 `nox_color.onnx`） | 上游 nox v1.1.0 `noxGeneratorColor.pb`（Christopher Harvey，2023；SHA-256 `d54bdca7…ba546`）。本项目用 tf2onnx 1.16.1、opset 17 转换，ONNX 217,729,909 B，SHA-256 `72451a04…6752e` | 代码与训练权重：MIT（据作者 README，快照见 `packaging/aester_p2/nox-author-license-readme.md`）。作者列出的训练背景数据中，SIDD 官方声明为 MIT，RENOIR 的条款**未核实** | 可选去星／缩星后端：`starless_runtime` 经 onnxruntime 在 CPU 上分块推理。输出是算法估计，可能留下光晕或误删紧凑星云；安装后不会自动启用 | **不在 Setup 内**。首次启动或组件管理中主动同意后，使用清单固定的 ONNX 大小与散列校验，再安装运行库并配置路径；也可选择已有一致模型。当前生产 URL 为空。许可、作者声明和转换证据随安装配方保留 | 2026-10-15 第二包用户决定取代旧测试包“Setup 内附、默认不勾”的交付方式。作者对代码及权重声明 MIT，训练数据权利链仍未核清；按需下载不使此事项自动关闭。七组有限输入对照不构成全输入等价证明。历史发行证据保留 |
| Noto Sans SC 字体（Regular／Medium，`NotoSansSC-*.otf`） | notofonts/noto-cjk 提交 `165c01b4`（Sans/SubsetOTF/SC） | SIL OFL 1.1（无保留字体名；许可证正文 `OFL-NotoSansCJK.txt`） | 界面字体：`aester_desktop.fonts` 启动时向 Qt 注册，仅本进程使用，不安装到系统 | **是**：随附，未修改（bundled, unmodified），与许可证正文同在 `src/aester_desktop/assets/fonts/` | 来源 https://github.com/notofonts/noto-cjk ；逐文件 URL、字节数与 SHA-256 见 `src/aester_desktop/assets/fonts/FONTS.json`，安装器构建逐项核对，缺失或改动即拒绝构建 |
| IBM Plex Sans 字体（Regular／Medium／SemiBold，`IBMPlexSans-*.ttf`） | @ibm/plex-sans 1.1.0 | SIL OFL 1.1，保留字体名 “Plex”（IBM；许可证正文 `OFL-IBMPlex.txt`） | 品牌字与标题字体，同上仅进程内注册 | **是**：随附，未修改（bundled, unmodified），同目录 | 来源 https://github.com/IBM/plex/releases/download/@ibm/plex-sans@1.1.0/ibm-plex-sans.zip ；未修改、未改名，若日后子集化或修改须按 OFL 另起不含 “Plex” 的字体名 |
| IBM Plex Mono 字体（Regular／Medium，`IBMPlexMono-*.ttf`） | google/fonts 提交 `0b58fb37`（ofl/ibmplexmono） | SIL OFL 1.1，保留字体名 “Plex”（IBM；许可证正文 `OFL-IBMPlex.txt`） | 数值读数等宽字体，同上仅进程内注册 | **是**：随附，未修改（bundled, unmodified），同目录 | 来源 https://github.com/google/fonts/tree/main/ofl/ibmplexmono ；保留字体名约束同上 |
| 霞鹜文楷 Lite（LXGW WenKai Lite Regular，`LXGWWenKaiLite-Regular.ttf`） | v1.522 | SIL OFL 1.1（无保留字体名；许可证正文 `OFL-LXGWWenKai.txt`，含 Klee 项目版权声明） | 人工输入／确认文字字体，同上仅进程内注册 | **是**：随附，未修改（bundled, unmodified），同目录 | 来源 https://github.com/lxgw/LxgwWenKai-Lite/releases/tag/v1.522 ；OFL 不允许单独出售字体文件本身，随软件捆绑分发允许 |
| Inno Setup | 6.7.3（来源与 SHA-256 见 `tools/aester-installer/tool-provenance.json`） | Inno Setup License（zlib 类宽松许可；版权 Jordan Russell、Martijn Laan） | 生成安装包。安装程序与卸载程序的运行部分编入 Setup.exe，卸载程序装到 `{app}\uninstall` | **是**：安装与卸载运行部分。编译器 ISCC 只在构建时使用 | 须保留版权声明与网址，不得冒称原作，修改版须标明。许可证正文 `Inno-LICENSE.txt`（与上游原文逐字节相同）随安装附带。发行方请求、但不强制商业用户购买商业许可。安装包与启动器未做代码签名 |
| Inno Setup 简体中文界面文件（`ChineseSimplified.isl`，kira-96 社区翻译） | 提交 `1ff90acc`（2026-09-24 下载，未修改） | MIT | 安装界面的简体中文文本 | **是**：编入 Setup.exe | MIT 正文 `inno-chinese-LICENSE.txt` 随安装附带；来源见 `inno-chinese-SOURCE.md`；不是 Inno 官方翻译。Aester 自己的安装文案在 `installer-messages.iss` |
| GIMP | 3.2.6（`gimp-3.2.6-setup.exe`，190,358,080 B，SHA-256 `9337cccb…ebca`） | GPL-3.0-or-later；随附的上游依赖各保留原许可证 | 可选：自然语言修图的执行器，以独立程序运行 | Setup.exe 不含。只在含修图支持的构建中出现，可选、默认不勾。勾选后，安装时从 download.gimp.org 或所列镜像（opencolo.mm.fcix.net、南大、中科大、阿里云）中测速较快的一处下载原版安装器，按固定大小与 SHA-256 校验后以 `/CURRENTUSER /VERYSILENT` 安装成独立程序。已有 GIMP 只检测、复用；卸载 Aester 时保留 GIMP | 本项目不内附、也不重新打包 GIMP。静默安装不会显示 GIMP 自己的许可页；GIMP 官方安装器仍保留其自带的许可声明，本项目的 GPL 说明见 `RETOUCH-COMPONENTS.md`、`retouch-components.json`、本文与 `NOTICE.md`。主源码 `gimp-3.2.6.tar.xz`（SHA-256 `40b15e90…`）已固定并记录，但不随附；不宣称覆盖 GIMP 全部依赖的源码 |
| 构建期工具：Inno Setup 编译器 ISCC、innounp 2、.NET Framework `csc.exe`、生成图标用的 Pillow；转换 nox 用的 tf2onnx 1.16.1、TensorFlow 2.15、onnx 1.16.2 | — | 各自许可证 | 只在构建安装包或转换模型时使用 | 否 | 不在分发闭包内。启动器依赖 Windows 自带的 .NET Framework 4.x，本产品不分发它 |
| AI 编程助手与开发记录工具 | — | 开发工具 | 仅开发过程 | 否 | 不在分发闭包内 |

## 可选组件（未进锁文件；勾选 nox 时在安装时下载）

这三个包的版本与 SHA-256 已固定在安装器清单的 `optional_wheels` 中，但没有进 `requirements-lock.txt`，因此不在 `check_third_party.py` 的对账范围内，下表手工维护。

| 组件 | 版本 | 许可证 | 使用方式 | 随产品分发 | 约束 / 说明 |
|---|---|---|---|---|---|
| onnxruntime | 1.30.0（安装器清单）；开发环境另用 `pip install .[starless]`（≥ 1.17） | MIT | 由 `starstack starless` / `starless_runtime.OnnxModel` 懒加载，在 CPU 上分块推理 nox 模型 | Setup.exe 不含；只有勾选 nox 时才在安装时从 PyPI 下载原版 wheel | wheel 中的 `onnxruntime/LICENSE`（MIT）与 `onnxruntime/ThirdPartyNotices.txt` 在包目录内（dist-info 中没有许可证文件），随解包原样保留。无星图工作包步 2 见 `docs/STARLESS_PACKAGE_20260923.md` |
| flatbuffers | 25.12.19 | Apache-2.0 | onnxruntime 依赖 | 同上 | Apache-2.0 正文由本产品补在 `packaging/aester_p2/flatbuffers-Apache-2.0.txt`（2026-10-14，见“许可证正文”一节）；该 wheel 本身不含许可证文件 |
| protobuf | 7.36.2 | BSD-3-Clause | onnxruntime 依赖 | 同上 | 许可证随 dist-info 原样保留 |

## 已知缺口（正式发行前处理）

1. **Qt 源码获取说明**：pyside6-essentials 与 shiboken6 的 wheel 只带 `LicenseRef-Qt-Commercial.txt`。LGPL-3.0／GPL-3.0 正文已于 2026-10-14 补入安装包；Qt 源码的获取说明（上游 https://download.qt.io/ 与 PySide6 源码发布）尚未写入安装包。
2. **nox**：RENOIR 训练数据的条款未核实。2026-10-15 第二包起 Setup 不再内置 nox，改为首次启动由用户主动同意后下载；生产下载地址尚未配置，权利核查未完成，G4（`nox_rights`、`nox_download`）仍未关闭。
3. **对账范围**：`check_third_party.py` 只核对锁文件；安装器清单与本文之间没有自动对账，nox 的三个 wheel 不在锁文件里。
4. **Microsoft Distributable Code**：已核实际随包 Python 许可证确有下游同等保护要求；须以随附声明转述仅适用于 Microsoft 组件的对应条款，并核清各载荷版本与许可映射。原计划写入 EULA；2026-10-31 改为开源后不再有 EULA，改由 `NOTICE.md` 或安装说明转述，尚未落实。
5. **GIMP 许可页**：静默安装不显示 GIMP 自己的许可页；GIMP 官方安装器保留其自带的许可声明，本项目的 GPL 说明见修图组件文档、本文与 `NOTICE.md`。
6. **改进构建包内的旧措辞**：`runtime/runtime_manifest.json` 的 scope 仍写着 “redistribution/license closure not certified”，`LICENSE_NOTES.md` 写的是 “one Siril executable and 74 DLLs”。两者为与 package-v2 保持逐字节一致而保留；`RUNTIME_AUX_NOTICE.md` 只补充了“一个可执行文件和 74 个 DLL”的描述，没有说明 `runtime_manifest.json` 中 scope 字样的范围。
7. **源码验证范围**：52 个 MSYS2 源码包未做 PGP 验签，也未从源码全量重建依赖。
8. **签名**：安装包与启动器都未做代码签名。
9. **Python “unmodified” 的含义**：`packaging/aester_p2/THIRD_PARTY_INSTALLER.md` 说 Python 原样内附，指的是归档本身；私有运行环境改写了配置文件 `python313._pth`。
10. **官方 Siril 压缩包**：目前安装时从上游下载；如改为内置进安装包，须先随附（或按 GPL 提供）Siril 及其 GPL 编解码器的对应源码。现行下载方式的分发义务边界未做法律审查。
11. **flatbuffers 许可证正文**：已于 2026-10-14 补入 `packaging/aester_p2/flatbuffers-Apache-2.0.txt`（本条保留作记录）。

## 许可证正文（2026-10-14 补入安装包）

2026-10-14 补入，随安装器的 `packaging/aester_p2/` 文本一起装到 `versions\<版本>\packaging\aester_p2\`；构建的发行门槛 G3 检查它们存在。

| 文件 | 来源 | 字节 | SHA-256 |
|---|---|---|---|
| `LGPL-3.0.txt` | https://www.gnu.org/licenses/lgpl-3.0.txt | 7,652 | `e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118` |
| `GPL-3.0.txt` | https://www.gnu.org/licenses/gpl-3.0.txt | 35,149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
| `flatbuffers-Apache-2.0.txt` | https://www.apache.org/licenses/LICENSE-2.0.txt | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
