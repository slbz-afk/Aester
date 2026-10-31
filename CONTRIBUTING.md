# Contributing to Aester

Thank you for helping. This page covers bug reports, sample reports and code. [中文说明在下方](#参与-aester)。

## Reporting problems

- Use [Issues](https://github.com/slbz-afk/Aester/issues) and pick the matching template. Include:
  - the Aester version (Help → About Aester);
  - your Windows version and camera model;
  - what you did and what happened.
- Attach the job's log or the message from "Details" when the window offers one.
- **Do not post private RAW files publicly.** Describe the scene first. If a file is needed, say so, and we will agree on a private way to share it.

## Contributing code

1. **Fork the repository** and create a short-lived branch, for example `fix/chinese-path`.
2. **Set up a development environment.** With Python 3.12 or 3.13 on Windows:

   ```powershell
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements-lock.txt
   $env:PYTHONPATH = "src"
   .venv\Scripts\python -X utf8 -m pytest -q
   ```

   - Real processing also needs Siril 1.4.4.
   - Tests that need private data or tools skip themselves when those are missing.
3. **Keep the engine/app boundary.**
   - The engine (`src/starstack`) must never import the desktop app (`src/aester_desktop`) or Qt.
   - The app uses only the engine names listed in `src/starstack/public_api.json`. If you add one on purpose, regenerate the file with `python scripts/split/api_surface.py --write`.
   - `tests/test_open_core_boundary.py` and `tests/desktop/test_desktop_boundary.py` check this.
4. **Add or update tests** for what you change, and run the relevant tests before opening a pull request.
5. **Open a pull request.** Describe:
   - what changes for users;
   - how you verified it;
   - what is still unverified, especially image quality on real photos.

## Licence of contributions and sign-off

- Contributions are accepted under the licence of the part they change:
  - Apache-2.0 for the engine, packaging, scripts and the engine's tests;
  - GPL-3.0-or-later for `src/aester_desktop` and `tests/desktop`.
- Every commit must carry a [Developer Certificate of Origin](https://developercertificate.org/) sign-off. It certifies that you wrote the change or otherwise have the right to submit it under that licence. Add it with `git commit -s`, which appends:

  ```
  Signed-off-by: Your Name <your.email@example.com>
  ```

- Please do not submit code, images or data that you are not allowed to share. That includes material from tutorials, other applications or other people's photos.

## Name and icon

Forks are welcome under the licences above. Please give modified versions their own name and icon ([TRADEMARKS.md](TRADEMARKS.md)).

---

# 参与 Aester

谢谢你的帮助。

## 反馈问题

- 在 [Issues](https://github.com/slbz-afk/Aester/issues) 选择对应模板，写明：
  - Aester 版本（帮助 → 关于 Aester）；
  - Windows 版本和相机型号；
  - 操作步骤和实际结果。
- 界面有“详情”时，请附上其中的消息或任务日志。
- **不要把私人 RAW 原片公开贴出。** 先描述场景；确实需要文件时说明一声，我们再约定私下传递的方式。

## 提交代码

1. **Fork 仓库**，建一个短期分支，例如 `fix/chinese-path`。
2. **搭建开发环境**：在 Windows 上用 Python 3.12 或 3.13，按上方英文部分的命令建立环境并运行测试。
   - 真实处理还需要 Siril 1.4.4。
   - 缺少私有数据或工具的测试会自动跳过。
3. **守住引擎与桌面程序的边界**：
   - 引擎（`src/starstack`）不能导入桌面程序（`src/aester_desktop`）或 Qt；
   - 桌面程序只使用 `src/starstack/public_api.json` 列出的引擎接口。有意新增接口时，用 `python scripts/split/api_surface.py --write` 重新生成这个文件。
4. **为改动补充或更新测试**，提交 PR 前运行相关测试。
5. **提交 PR**，说明：
   - 用户能看到什么变化；
   - 你怎样验证的；
   - 还有什么没验证，尤其是真实照片上的画质。

## 贡献的许可与签署

- 贡献按所改部分的许可证接受：
  - 引擎、打包、脚本及引擎测试为 Apache-2.0；
  - `src/aester_desktop` 和 `tests/desktop` 为 GPL-3.0-or-later。
- 每个提交都要带 [开发者原创声明（DCO）](https://developercertificate.org/) 签署，表示改动是你写的，或你有权按该许可证提交。用 `git commit -s` 提交即可自动加上 `Signed-off-by` 一行。
- 请不要提交你无权分享的代码、图片或数据，包括教程素材、其他软件的内容或别人的照片。

## 名称与图标

欢迎按上述许可证分叉。修改后的版本请使用自己的名称和图标（见 [TRADEMARKS.md](TRADEMARKS.md)）。
