"""Verified region overlays for human/agent inspection, without approving them."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image

from . import overlay_colors, regions

MAX_PREVIEW_BYTES = 128 * 1024 * 1024
MAX_LONG_EDGE = 1600
GROUND_ALPHA = 95           # translucent protected-foreground fill
MILKYWAY_ALPHA = 110        # peak alpha of the feathered Milky Way fill
EDGE_WIDTH = 2              # foreground boundary line, image pixels (ground side)


def _grow(mask: np.ndarray, steps: int) -> np.ndarray:
    """8-neighbour (square) binary dilation, separable; outside the image counts as empty."""
    grown = mask
    for _ in range(steps):
        wide = grown.copy()
        wide[:, 1:] |= grown[:, :-1]
        wide[:, :-1] |= grown[:, 1:]
        grown = wide.copy()
        grown[1:] |= wide[:-1]
        grown[:-1] |= wide[1:]
    return grown


def overlay_rgba(sky: np.ndarray, milkyway: np.ndarray) -> np.ndarray:
    """The shared GUI/MCP region overlay: fills plus a cased foreground boundary.

    Colours come from ``overlay_colors`` and never depend on the interface theme:
    the protected foreground is a translucent GROUND fill bounded by a solid GROUND
    line with an OUTLINE casing; the Milky Way is a feathered MILKYWAY fill (its
    double outline is drawn separately as vectors). The image frame is not a boundary.
    """
    height, width = sky.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    ground = sky == 0
    rgba[ground] = [*overlay_colors.GROUND_RGB, GROUND_ALPHA]
    selected = milkyway > 0
    rgba[selected, :3] = overlay_colors.MILKYWAY_RGB
    rgba[selected, 3] = (milkyway[selected] * MILKYWAY_ALPHA).astype(np.uint8)
    if ground.any() and not ground.all():
        line = ground & _grow(~ground, EDGE_WIDTH)
        casing = _grow(line, 1) & ~line
        rgba[casing] = [*overlay_colors.OUTLINE_RGB, 255]
        rgba[line] = [*overlay_colors.GROUND_RGB, 255]
    return rgba


def ellipse_outline(project: dict, width: int, height: int, *, offset: float = 0., samples: int = 256) -> list:
    """Pixel-space points of the Milky Way ellipse, grown by ``offset`` pixels.

    Matches RegionCanvas: normalized radii scale by the image width/height, then the
    ellipse rotates by ``angle_degrees`` in pixel space. Empty without an ellipse.
    """
    import math
    mw = project.get('milkyway') or {}
    if not mw.get('ellipse'):
        return []
    cx, cy, rx, ry = mw['ellipse']
    angle = math.radians(mw.get('angle_degrees', 0.))
    ax, ay = max(rx * width + offset, .5), max(ry * height + offset, .5)
    ca, sa = math.cos(angle), math.sin(angle)
    points = []
    for index in range(samples + 1):
        t = 2 * math.pi * index / samples
        x, y = ax * math.cos(t), ay * math.sin(t)
        points.append((cx * width + ca * x - sa * y, cy * height + sa * x + ca * y))
    return points


def _draw_milkyway_outline(image: Image.Image, project: dict) -> Image.Image:
    """Suzhou-style double outline in MILKYWAY, each stroke cased with OUTLINE."""
    from PIL import ImageDraw
    width, height = image.size
    half = overlay_colors.DOUBLE_GAP / 2 + overlay_colors.LINE_WIDTH / 2
    rings = [ellipse_outline(project, width, height, offset=sign * half) for sign in (-1, 1)]
    if not rings[0]:
        return image
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(overlay_colors.LINE_WIDTH))
    casing = max(stroke + 1, round(overlay_colors.LINE_WIDTH + overlay_colors.OUTLINE_EXTRA))
    for ring in rings:
        draw.line(ring, fill=overlay_colors.OUTLINE_RGB, width=casing, joint='curve')
    for ring in rings:
        draw.line(ring, fill=overlay_colors.MILKYWAY_RGB, width=stroke, joint='curve')
    return image


def render_preview(project: dict, reference: dict, output_root) -> dict:
    """Render the admitted full-frame reference with exactly the GUI's overlays.

    Host adapters must bound all referenced paths before calling. This helper
    checks the region's schema/identity, hashes the same encoded preview bytes
    it subsequently decodes, and never reads image data a second time. It does
    not infer sky, change an annotation, or confirm Milky Way presence.
    """
    regions.validate_project(project, verify_sources=False)
    if not isinstance(reference, dict):
        raise ValueError('分区预览缺少参考图记录')
    bound = project['source_binding']['reference']
    from .orientation import normalize_rotation
    if normalize_rotation(reference.get('image_rotation', 0)) != normalize_rotation(bound.get('image_rotation', 0)):
        raise ValueError('分区预览的方向与项目不一致')
    for key in ('source_path', 'source_sha256', 'linear_path', 'linear_sha256', 'width', 'height'):
        if reference.get(key) != bound[key]:
            raise ValueError('分区预览的原片、参考图或尺寸与项目不一致')
    preview_path = reference.get('preview_path')
    if not isinstance(preview_path, str) or not Path(preview_path).is_absolute():
        raise ValueError('参考预览须为明确的绝对路径')
    with Path(preview_path).open('rb') as stream:
        payload = stream.read(MAX_PREVIEW_BYTES + 1)
    if len(payload) > MAX_PREVIEW_BYTES:
        raise ValueError('参考预览文件过大，请重新准备缩略预览')
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference.get('preview_sha256'):
        raise ValueError('参考预览内容已改变，不能绘制可靠分区')
    with Image.open(BytesIO(payload)) as decoded:
        width, height = decoded.size
        if min(width, height) < 1 or width * height > 100_000_000:
            raise ValueError('参考预览尺寸无效或过大')
        declared_width, declared_height = reference.get('preview_width'), reference.get('preview_height')
        if ((declared_width is not None and declared_width != width)
                or (declared_height is not None and declared_height != height)):
            raise ValueError('参考预览尺寸与记录不一致')
        # One pixel allows ordinary thumbnail rounding; cropping or stretching
        # would make normalized annotations point at a different scene region.
        expected_height = bound['height'] * width / bound['width']
        if abs(height - expected_height) > 1.01:
            raise ValueError('参考预览不保持原图完整比例，不能叠加分区')
        profile = decoded.info.get('icc_profile')
        base = decoded.convert('RGB')
        if max(base.size) > MAX_LONG_EDGE:
            base.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)
    width, height = base.size
    sky, milkyway = regions.rasterize(project, width, height)
    rgba = overlay_rgba(sky, milkyway)
    output = Image.alpha_composite(base.convert('RGBA'), Image.fromarray(rgba)).convert('RGB')
    if milkyway.any():
        output = _draw_milkyway_outline(output, project)
    root = regions._safe_root(output_root, project)
    root.mkdir(parents=True, exist_ok=True)
    ident = uuid.uuid4().hex
    destination, stage = root / ('region-preview-' + ident), root / ('.partial-region-preview-' + ident)
    stage.mkdir()
    try:
        image_path = stage / 'regions.png'
        output.save(image_path, **({'icc_profile': profile} if profile else {}))
        receipt = {'preview_path': str(destination / 'regions.png'),
                   'sha256': regions._sha(image_path), 'width': width, 'height': height,
                   'legend': {'ground': '橙色：受保护地景（橙色实线为边界，深色描边）；画笔分区，不是自动识别结果',
                              'milkyway': '粉紫色：候选银河渐变（粉紫双线为椭圆范围）；叠色不代表已确认存在',
                              'unchanged': '未叠色：保持首帧预览',
                              'notice': '仅用于检查范围。此操作不确认银河、不更改素材或分区。'},
                   'region_revision': project['revision'],
                   'reference_preview_sha256': digest, 'scene_mode': project['scene_mode'],
                   'milkyway_status': project['milkyway']['status'],
                   'row_order': 'TOP-DOWN', 'kind': 'annotation-inspection-preview'}
        (stage / 'preview.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
        os.rename(stage, destination)
        return receipt
    finally:
        if stage.exists():
            shutil.rmtree(stage)
