# NOTICE

Aester（暂定产品名；项目原名 StarStack，源码包与内部接口仍叫 `starstack`）：星野／银河＋地景自动后期软件，当前为测试预览版。

Copyright (C) 2026 Wang Jian (王俭).

The open-source processing engine (StarStack: engine, command line, MCP server and evaluation tools) is licensed under the Apache License, Version 2.0 (the "License"); you may not use it except in compliance with the License. You may obtain a copy of the License at `LICENSE` in this repository or http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

本产品分两部分，均为自由软件（见 `LICENSING.md`）：处理引擎（`starstack`：引擎、命令行、MCP 与评测工具）按 Apache-2.0 授权（用户决定 L1，2026-09-22）；Aester 桌面程序（`aester_desktop`）自 2026-10-31 起按 GPL-3.0-or-later 授权，安装包附带完整源码（2026-10-14 至 10-31 期间的内部预览版曾为专有字节码，没有正式发行）。拆分之前分发的预览版与试用安装包中的全部源代码均以 Apache-2.0 授权，不因后来的调整撤回。第三方组件各自保留原许可证。两种许可证都不授予商标权：Aester 名称与图标不在授权范围内，修改后的版本须改用其他名称和图标（`TRADEMARKS.md`）。

第三方组件与各自许可证见 `THIRD_PARTY.md` 和 `packaging/aester_p2/THIRD_PARTY_INSTALLER.md`。本说明按 2026-10-15 第二包安装配方更新：Setup 不内置 nox；首次启动由用户主动同意后准备可选模型与运行库。当前生产模型下载地址为空，使用已有已验证模型及本机夹具分别验收，尚不提供公网一键下载。10-14 旧包仍含 nox，不受此次配方变更影响。

- 安装时下载的文件来自官方主机或 `THIRD_PARTY.md` 所列镜像（默认测速后用较快的一处），每个文件都按固定的大小与 SHA-256 校验，所以无论从哪里下载，得到的都是与官方相同的字节。
- 官方 Siril 1.4.4 不在安装包内：安装时从官方 free-astro.org 下载原版便携包（没有镜像）并校验，以独立进程调用，不链接。
- **Siril · Aester 改进构建**（finite-validity-v1，本项目修改的 Siril 1.4.4，GPL-3.0-or-later）在构建时传入已核验分发包的安装包中**随附**（09-28 测试版即如此；下次构建只接受 package-v3）。它是可选组件，默认不勾选；安装包内附它的运行库、完整对应源码、许可证与修改说明，勾选后一起安装。它不是 Siril 官方发布，也没有 Siril 官方背书。
- **Python 3.13.15**（python.org 官方嵌入式包，PSF-2.0，含其随附组件与 Microsoft Visual C++ 运行库）**随安装包分发**，原样内附，带其 `LICENSE.txt`。
- 其余 PyPI 依赖不在安装包内，安装时从 PyPI 或所列 PyPI 镜像下载原版 wheel，保留其元数据与许可证文件。GUI 使用 Qt（PySide6）动态库，rawpy 使用 LibRaw。PySide6 与 shiboken6 wheel 未带的 LGPL-3.0 及 GPL-3.0 正文、flatbuffers 的 Apache-2.0 正文已于 2026-10-14 补入 `packaging/aester_p2/`。Qt 对应源码提供安排及完整 wheel 的模块许可范围仍待核验，不能由已附正文推定全部发行义务已完成（见 `THIRD_PARTY.md`）。
- **nox** 是第三方可选去星／星点整理后端，使用本项目从上游 TensorFlow 文件转换的 ONNX，不是作者发布的 ONNX。Setup 不含模型；首次启动明确同意后按发布清单固定大小与 SHA-256 取得模型及运行库，也可选择一致的已有模型。许可、作者声明和转换证据随配方保留。作者对权重声明 MIT；训练数据权利链仍未核清，按需下载不改变此未决事项。安装不自动开启图像效果。
- 本产品**不含** RawTherapee 的 Z8 DCP 文件；`--dcp-profile` 依赖用户自备的数据文件。
- 本产品**包含** RawTherapee／ART 中 16 个机型 DCP 的日光色相/饱和度表数值（CC0 或 public domain，见 `src/starstack/camera_look_data.py`），用作各机型的默认观感。
- 本产品**随附** Noto Sans SC、IBM Plex Sans／Plex Mono（保留字体名 “Plex”）与霞鹜文楷 Lite 四款 SIL OFL 1.1 字体，文件未经修改，各自许可证正文（`OFL-*.txt`）与来源清单 `FONTS.json` 同在 `src/aester_desktop/assets/fonts/`（见 `THIRD_PARTY.md` 非 PyPI 组件表）。
- 可选的 GIMP 3.2.6（GPL-3.0-or-later）不在安装包内：勾选后从 download.gimp.org 或所列镜像下载原版安装器，按 SHA-256 校验后安装为独立程序，卸载 Aester 时保留。
- 安装程序由 Inno Setup 6.7.3 生成（Inno Setup License），简体中文界面文件为 MIT；两份许可证正文都随安装附带。安装包与启动器未做代码签名。

安装后，`LICENSE`、`NOTICE.md`、`THIRD_PARTY.md` 和 `packaging/aester_p2/` 下的许可证正文位于程序目录的 `versions\<版本>\` 中；各组件自带的许可证随其组件目录保留。

`scripts/check_third_party.py` 把 `requirements-lock.txt` 与 `THIRD_PARTY.md` 对账（`python scripts/check_third_party.py`，`--files` 另列已安装 wheel 内的许可证文件）。它不核对安装器的下载清单。

以上是工程记录，不是法律意见，也不表示商业发行的全部法律事项已审查完毕。
