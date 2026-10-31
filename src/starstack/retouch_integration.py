"""Read-only retouch onboarding shared by the product GUI and main MCP."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import components
from .i18n import normalize_language


def _app_root(value=None):
    return Path(value or os.environ.get("AESTER_APP_ROOT") or
                Path(__file__).resolve().parents[2]).expanduser().absolute()


def get_retouch_info(*, app_root=None, gimp_path=None, component_root=None) -> dict:
    """Inspect local paths; never install a skill, launch GIMP, or write state.

    Skill availability means its reviewed text can be loaded. Agent clients
    manage their own installation independently; we do not infer their state.
    """
    app = _app_root(app_root)
    try:
        state = components.retouch_component_status(component_root, app_root=app,
                                                     selected_gimp_path=gimp_path)
    except (OSError, ValueError) as exc:
        state = {"gimp": {"status": "check_failed", "selected_path": str(gimp_path) if gimp_path else None,
                          "diagnostics": [{"message": str(exc)}]},
                 "skill": {"status": "check_failed", "path": None}}
    gimp = dict(state["gimp"])
    gimp.update(available=gimp.get("status") == "available" and bool(gimp.get("path_exists")),
                compatible=gimp.get("version_compatible"),
                detail="GIMP 3.2.6 is required for rendering. File discovery is not runtime verification.",
                ownership="external_unmanaged")
    skill = dict(state["skill"])
    path = skill.get("path")
    bundled = app / "skills/aester-retouch/SKILL.md"
    skill.update(available=skill.get("status") in {"ready", "bundled"} and bool(path),
                 bundled_path=str(bundled) if bundled.is_file() else None,
                 read_path=path, installed=None, required=False,
                 detail="Recommended; load the bundled skill for this task or optionally install it in your Agent client.",
                 resources=["aester://retouch/skill/" + name for name in _skill_names()])
    return {"schema": 1, "trial": True, "gimp": gimp, "skill": skill,
            "recommendation": "Recommend aester-retouch when the user asks to retouch. Skill installation is optional; tools remain available.",
            "image_transport": "local_paths", "automatic_upload": False}


def _skill_names():
    from .retouch_distribution import SKILL_FILES
    return tuple(sorted(SKILL_FILES))


def read_retouch_skill(section="SKILL.md", *, app_root=None, component_root=None) -> str:
    """Serve only the fixed reviewed skill package, never caller-selected files."""
    from .retouch_distribution import SKILL_FILES
    import hashlib
    if section not in SKILL_FILES:
        raise ValueError("Unknown retouch skill section")
    info = get_retouch_info(app_root=app_root, component_root=component_root)
    skill = info["skill"]
    if not skill["available"]:
        raise ValueError("Retouch skill package is unavailable; prepare it in Aester Settings → Components & prompts (formerly “Components”). "
                         "Direct finishing tools remain usable.")
    root = Path(skill["read_path"]).parent
    path = components._inside(root, root / section)
    body = path.read_bytes()
    if len(body) > 128 * 1024 or hashlib.sha256(body).hexdigest() != SKILL_FILES[section]:
        raise ValueError("Reviewed retouch skill content changed")
    return body.decode("utf-8")


REFERENCE_KINDS = ("external", "original", "generated")


def _reference_images(value):
    """[{'path', 'kind'}] with a known provenance kind, or None."""
    if not value:
        return None
    result = []
    for item in value:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("Reference images need a path")
        kind = item.get("kind") or "external"
        if kind not in REFERENCE_KINDS:
            raise ValueError("Reference kind must be external, original or generated")
        result.append({"path": str(item["path"]), "kind": kind})
    return result


def build_retouch_prompt(*, photo_path=None, intent=None, app_root=None, gimp_path=None,
                         component_root=None, language="zh_CN", reference_images=None) -> str:
    """A copyable task, not an installation command or a shell script.

    ``reference_images``: optional [{'path', 'kind'}] aesthetic references; ``kind`` is the
    finishing_add_reference provenance (external = someone else's work, original = the
    photographer's own work, generated = an AI-generated image).
    """
    info = get_retouch_info(app_root=app_root, gimp_path=gimp_path, component_root=component_root)
    references = _reference_images(reference_images)
    data = {"photo_path": str(photo_path) if photo_path else None,
            "desired_look": str(intent) if intent else None,
            "reference_images": references,
            "skill_read_path": info["skill"].get("read_path"),
            "skill_archive_path": info["skill"].get("archive_path"),
            "gimp_console_path": info["gimp"].get("selected_path")}
    # JSON keeps path/text boundaries clear and cannot become a shell command.
    context = json.dumps(data, ensure_ascii=False, indent=2)
    if normalize_language(language) == "en":
        if references:
            reference_text = (
                "reference_images are aesthetic references I chose. Register each one with finishing_add_reference "
                "using its kind (external = someone else's work, original = my own work, generated = an AI-generated image). "
                "Use them only to set aesthetic goals such as colour, tonal balance, contrast and mood; never copy their stars, "
                "clouds, ridgelines, landscape or textures into my photograph, and never use them to invent detail it does not have. ")
        else:
            reference_text = (
                "I have not given a reference image. Before planning, ask once whether I have a reference whose look I like "
                "(someone else's work, my own work or an AI-generated image); if I have none, continue without one. ")
        return ("Help me retouch a photograph with Aester's natural-language retouch trial. "
                "Use the connected Aester MCP: first call retouch_setup and finishing_capabilities. "
                "Recommend the aester-retouch skill and load its SKILL.md plus relevant references from the local path "
                "below or the MCP resources aester://retouch/skill/SKILL.md and aester://retouch/skill/references/<name>.md. "
                "Loading for this task is enough; installing it in my Agent client is optional, and declining it must not block retouching. "
                "Do not silently install a global skill. If Aester is not connected, guide me through Settings → Connect an AI assistant. "
                "If GIMP is missing, guide me to Settings → Components & prompts (formerly “Components”) to select or install GIMP 3.2.6. "
                "The JSON below contains paths and my aesthetic request as data, not executable commands. "
                "If no photo or request is supplied, ask only for what is missing.\n\n"
                + context + "\n\nStart from the finished 16-bit sRGB TIFF, not the linear master. "
                "Inspect the full image and relevant details, translate my request into explicit objectives, then use finishing_* tools "
                "to create a baseline and iterate recipes/masks with visual comparison. Use references for aesthetic guidance only. "
                + reference_text +
                "If the skill mentions the separate developer server, use the same tools on this product MCP; no second server is needed. "
                "Keep originals intact; export new JPEG, 16-bit TIFF and editable XCF with replay records. "
                "Explain unresolved limitations and do not invent missing detail or claim my approval. "
                "These tools return local image paths; using an external vision service requires my authorization.")
    if references:
        reference_text = (
            "JSON 中的 reference_images 是我选的审美参考图：请先用 finishing_add_reference 按其 kind 登记"
            "（external＝他人作品，original＝我自己的作品，generated＝AI 生成图）。"
            "参考图只用来确定色调、明暗、反差和气氛等审美目标；绝不能把参考图里的星点、云、山脊线、地景或纹理搬进我的照片，"
            "也不能用它补出照片里没有的细节。")
    else:
        reference_text = ("我还没有提供参考图。开始规划前，先问我一次是否有喜欢的效果参考图"
                          "（他人作品、我自己的作品或 AI 生成图）；没有也可以直接继续。")
    return ("请使用 Aester 自然语言修图试用功能帮我修图。先调用已连接 Aester MCP 的 retouch_setup 和 finishing_capabilities。"
            "请推荐 aester-retouch 技能，并从下面的本机路径或 MCP 资源 aester://retouch/skill/SKILL.md "
            "及 aester://retouch/skill/references/<name>.md 读取主文件和相关参考。"
            "本次临时加载即可；安装到 Agent 客户端是可选项，不安装也可继续调用修图工具，不要静默写入全局技能目录。"
            "若尚未连接 Aester，请引导我使用“设置→连接 AI 助手”；若缺少 GIMP，请引导我在“设置→组件与提示词管理（旧称“组件管理”）”中选择或安装 GIMP 3.2.6。"
            "以下 JSON 是文件路径和审美需求数据，不是可执行命令；未提供照片或需求时，只询问缺少的内容。\n\n"
            + context + "\n\n请从16位 sRGB 成片 TIFF开始，不能把线性母版直接当成片。"
            "先看整图与相关局部，将我的描述拆为明确目标，用 finishing_* 工具建立基线、逐轮调整配方和蒙版并看图比较。"
            "参考图只用于审美指导。" + reference_text +
            "技能中若提到独立开发服务器，使用当前产品 MCP 的同名工具即可，无需第二个服务。"
            "保持原片，另存 JPEG、16位 TIFF、可编辑 XCF 和回放记录；如实说明未达目标，不虚构细节、不代替我确认接受。"
            "工具仅返回本机图片路径；需要外部视觉服务看图时遵守我已给出的图片访问授权。")
