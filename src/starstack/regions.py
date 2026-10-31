"""Source-bound, editable top-down regions. Geometry is independent of colour.

These masks are explicit annotations, not an automatic semantic segmentation.
The local horizon assistant only searches near the supplied boundary.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from . import hashing
from .orientation import normalize_rotation

SCHEMA = "aester-regions-1"
DRAWING = 'explicit-drawing-1'


def _sha(path):
    return hashing.sha256_file(Path(path), source='regions._sha')


def _sha_many(paths):
    """``[_sha(p) for p in paths]``: same digests and first failure, hashed in parallel."""
    return hashing.sha256_many([Path(path) for path in paths], source='regions._sha')


def _first_changed(paths, expected):
    """Index of the first path whose digest is not ``expected[i]`` (None if all match), as a serial loop of _sha."""
    return hashing.first_mismatch([Path(path) for path in paths], expected, source='regions._sha')


def _scoped(label):
    """Run the decorated call inside a hashing run scope, so its repeated source checks share one ledger."""
    def decorate(function):
        @functools.wraps(function)
        def scoped(*args, **kwargs):
            with hashing.run_scope(label):
                return function(*args, **kwargs)
        return scoped
    return decorate


def _body(project):
    return {k: v for k, v in project.items() if k not in ('revision', 'project_path')}


def _revision(project):
    return hashlib.sha256(json.dumps(_body(project), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _seal(project):
    result = copy.deepcopy(_body(project))
    result['revision'] = _revision(result)
    return result


def _number(value, low=0., high=1.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('分区坐标或参数超出范围')
    return float(value)


def _points(points, minimum=1):
    if not isinstance(points, list) or not minimum <= len(points) <= 100000:
        raise ValueError('分区点数无效')
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('分区坐标须为 [x,y]')
        for value in point:
            _number(value)


def _validate_project(project, *, input_files=None, verify_sources=True):
    if not isinstance(project, dict) or project.get('schema') != SCHEMA:
        raise ValueError('不是 Aester 分区项目')
    if project.get('revision') != _revision(project):
        raise ValueError('分区记录内容已改变')
    if project.get('coordinate_system') != 'normalized-top-down':
        raise ValueError('分区坐标方向无效')
    binding = project.get('source_binding', {})
    frames, ref = binding.get('input_files', []), binding.get('reference', {})
    normalize_rotation(ref.get('image_rotation', 0))
    if not frames or not isinstance(frames, list) or len(frames) > 300:
        raise ValueError('分区缺少素材绑定')
    paths = [str(Path(frame['path']).resolve()) for frame in frames]
    for frame in [*frames, {'path': ref['linear_path'], 'sha256': ref['linear_sha256']}]:
        if not Path(frame['path']).is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', frame['sha256']):
            raise ValueError('分区源路径或哈希无效')
    if len(set(os.path.normcase(p) for p in paths)) != len(paths):
        raise ValueError('分区素材重复')
    if input_files is not None and paths != [str(Path(p).resolve()) for p in input_files]:
        raise ValueError('分区与本次素材顺序不一致')
    if not Path(ref.get('source_path', '')).is_absolute() or str(Path(ref.get('source_path', '')).resolve()) != paths[0] or ref.get('source_sha256') != frames[0].get('sha256'):
        raise ValueError('分区首帧身份不一致')
    for key in ('width', 'height'):
        if isinstance(ref.get(key), bool) or not isinstance(ref.get(key), int) or not 1 <= ref[key] <= 100000:
            raise ValueError('分区尺寸无效')
    if ref['width'] * ref['height'] > 500_000_000:
        raise ValueError('分区尺寸过大')
    if verify_sources:
        # Frames in order, then the reference: the first difference decides the message.
        changed = _first_changed([frame['path'] for frame in frames] + [ref['linear_path']],
                                 [frame['sha256'] for frame in frames] + [ref['linear_sha256']])
        if changed is not None:
            raise ValueError('分区绑定的素材已改变' if changed < len(frames) else '分区参考图已改变')
        from astropy.io import fits
        header = fits.getheader(ref['linear_path'])
        if (header.get('NAXIS1'), header.get('NAXIS2')) != (ref['width'], ref['height']):
            raise ValueError('分区尺寸与参考图不一致')
    if project.get('scene_mode') not in ('unknown', 'sky', 'landscape'):
        raise ValueError('分区场景类型无效')
    operations = project.get('operations')
    if not isinstance(operations, list) or len(operations) > 1000:
        raise ValueError('分区操作数无效')
    for op in operations:
        if op.get('kind') == 'horizon':
            _points(op.get('points'), 2)
            xs = [p[0] for p in op['points']]
            if any(a >= b for a, b in zip(xs, xs[1:])):
                raise ValueError('地平线的点须从左到右')
        elif op.get('kind') == 'paint':
            _points(op.get('points'))
            if op.get('target') not in ('sky', 'ground'):
                raise ValueError('画笔目标无效')
            _number(op.get('radius'), .0001, .25)
        else:
            raise ValueError('未知分区操作')
    mw = project.get('milkyway', {})
    if mw.get('status') not in ('unknown', 'absent', 'present') or mw.get('authority') not in ('human', 'agent'):
        raise ValueError('银河确认类型无效')
    if 'selection' in mw and mw['selection'] != DRAWING:
        raise ValueError('银河绘制来源无效')
    _number(mw.get('feather'), .1, .95)
    _number(mw.get('angle_degrees', 0.), -180., 180.)
    if mw.get('ellipse') is not None:
        if not isinstance(mw['ellipse'], list) or len(mw['ellipse']) != 4:
            raise ValueError('银河椭圆参数无效')
        for value in mw['ellipse'][:2]:
            _number(value, -4., 5.)
        for value in mw['ellipse'][2:]:
            _number(value, .001, 8.)
    if mw.get('status') == 'present':
        if mw.get('confirmed') is not True or not mw.get('ellipse') or not str(mw.get('reviewer', '')).strip() or not str(mw.get('rationale', '')).strip():
            raise ValueError('银河区域需要显式存在判断、判断者和说明')
    return project


def validate_project(project, *, input_files=None, verify_sources=True):
    try:
        return _validate_project(project, input_files=input_files, verify_sources=verify_sources)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('分区记录结构无效') from exc


@_scoped('regions.create_project')
def create_project(reference, input_files):
    if reference.get('state') != 'succeeded' or not input_files:
        raise ValueError('请先完成首帧预览')
    resolved, sources, failure = [], [], None
    for p in input_files:
        try:
            resolved.append(str(Path(p).resolve()))
        except Exception as exc:   # raised after every earlier file is hashed, as in the serial loop
            failure = exc
            break
        sources.append(p)
    digests = _sha_many(sources)
    if failure is not None:
        raise failure
    files = [{'path': path, 'sha256': digest} for path, digest in zip(resolved, digests)]
    ref = {k: reference[k] for k in ('source_path', 'source_sha256', 'linear_path', 'linear_sha256', 'width', 'height')}
    if 'image_rotation' in reference:
        ref['image_rotation'] = normalize_rotation(reference['image_rotation'])
    project = _seal({'schema': SCHEMA, 'coordinate_system': 'normalized-top-down',
                     'source_binding': {'input_files': files, 'reference': ref},
                     'scene_mode': 'unknown', 'operations': [],
                     'milkyway': {'status': 'unknown', 'ellipse': None, 'feather': .5, 'angle_degrees': 0.,
                                  'authority': 'human', 'confirmed': False, 'reviewer': '', 'rationale': ''}})
    return validate_project(project, verify_sources=True)


def revise(project, *, operations=None, scene_mode=None, milkyway=None):
    validate_project(project, verify_sources=False)
    result = copy.deepcopy(_body(project))
    for key, value in [('operations', operations), ('scene_mode', scene_mode), ('milkyway', milkyway)]:
        if value is not None:
            result[key] = copy.deepcopy(value)
    return validate_project(_seal(result), verify_sources=False)


def scene_from_marks(project):
    """Scene routing follows explicit ground marks; an unmarked frame is sky."""
    return 'landscape' if any(op.get('kind') == 'horizon' or
        (op.get('kind') == 'paint' and op.get('target') == 'ground')
        for op in project.get('operations', [])) else 'sky'


def milkyway_active(project):
    """Accept an explicit edit, retaining its real origin; no inferred ellipse."""
    mw = project.get('milkyway', {})
    return bool(mw.get('status') == 'present' and mw.get('confirmed') is True
                and mw.get('ellipse') and (mw.get('authority') == 'human' or
                (mw.get('authority') == 'agent' and mw.get('selection') == DRAWING)))


def normalize_marks(project):
    return revise(project, scene_mode=scene_from_marks(project))


def draw_milkyway(project, ellipse, *, angle_degrees=None, feather=None,
                 authority='human', reviewer='GUI user', rationale='Explicit Milky Way drawing'):
    """A caller's deliberate drawing is an edit instruction, not identification."""
    mw = {**project['milkyway'], 'ellipse': list(ellipse), 'status': 'present',
          'confirmed': True, 'selection': DRAWING, 'authority': authority,
          'reviewer': reviewer, 'rationale': rationale}
    if angle_degrees is not None:
        mw['angle_degrees'] = float(angle_degrees)
    if feather is not None:
        mw['feather'] = float(feather)
    return normalize_marks(revise(project, milkyway=mw))


def clear_milkyway(project):
    mw = {**project['milkyway'], 'ellipse': None, 'status': 'absent', 'confirmed': False,
          'reviewer': '', 'rationale': ''}
    mw.pop('selection', None)
    return normalize_marks(revise(project, milkyway=mw))


def clear_landscape(project):
    return revise(project, operations=[], scene_mode='sky')


def _safe_root(root, project):
    root = Path(root).resolve()
    for frame in project['source_binding']['input_files']:
        if root.is_relative_to(Path(frame['path']).parent.resolve()):
            raise ValueError('分区与作品必须保存在原片文件夹之外')
    return root


def save_project(project, output_root):
    validate_project(project)
    root = _safe_root(output_root, project)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ('regions-' + project['revision'] + '.json')
    payload = json.dumps(_body(project) | {'revision': project['revision']}, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists():
        if load_project(path, verify_sources=False)['revision'] != project['revision']:
            raise ValueError('分区修订文件冲突')
    else:
        temp = root / ('.regions-' + uuid.uuid4().hex + '.tmp')
        if not hashing.digest_writes():   # outside a hash run scope or with the ledger off: the former write
            temp.write_text(payload, encoding='utf-8')
            with hashing.writing(path):
                os.replace(temp, path)
        else:
            # Exactly the bytes temp.write_text(payload, encoding='utf-8') writes (newlines become os.linesep);
            # their digest is recorded for the new file, so the submit-time check of it needs no read-back.
            data = payload.replace('\n', os.linesep).encode('utf-8')
            temp.write_bytes(data)
            with hashing.writing(path) as block:
                os.replace(temp, path)
                block.written(hashing.sha256_bytes(data, source='regions.save_project'), 'regions.save_project')
    return copy.deepcopy(project) | {'project_path': str(path)}


def load_project(path, *, input_files=None, verify_sources=True):
    path = Path(path).resolve()
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('分区文件过大')
    result = json.loads(path.read_text(encoding='utf-8'))
    validate_project(result, input_files=input_files, verify_sources=verify_sources)
    return result | {'project_path': str(path)}


def rasterize(project, width=None, height=None):
    validate_project(project, verify_sources=False)
    ref = project['source_binding']['reference']
    width, height = width or ref['width'], height or ref['height']
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    if project['scene_mode'] != 'sky':
        for op in project['operations']:
            points = [(x * (width - 1), y * (height - 1)) for x, y in op['points']]
            if op['kind'] == 'horizon':
                # Every accepted horizon replaces the rough horizon; paint edits after it remain local.
                mask.paste(255, (0, 0, width, height))
                draw.polygon([(0, points[0][1]), *points, (width - 1, points[-1][1]),
                              (width - 1, height - 1), (0, height - 1)], fill=0)
            else:
                radius = max(1., op['radius'] * min(width, height))
                value = 255 if op['target'] == 'sky' else 0
                if len(points) > 1:
                    draw.line(points, fill=value, width=max(1, round(2 * radius)))
                for x, y in points:
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
    sky = np.asarray(mask).copy()
    mw = project['milkyway']
    weight = np.zeros((height, width), dtype=np.float32)
    if mw.get('ellipse'):
        cx, cy, rx, ry = mw['ellipse']
        angle = math.radians(mw.get('angle_degrees', 0.))
        cosine, sine = math.cos(angle), math.sin(angle)
        # Calculate row blocks, avoiding a 3x full-resolution coordinate mesh.
        x = (np.linspace(0, 1, width, dtype=np.float32) - cx) * width
        for y0 in range(0, height, 256):
            y = (np.arange(y0, min(height, y0 + 256), dtype=np.float32) / max(1, height - 1) - cy) * height
            xx = (cosine * x[None, :] + sine * y[:, None]) / (rx * width)
            yy = (-sine * x[None, :] + cosine * y[:, None]) / (ry * height)
            radius = np.sqrt(xx * xx + yy * yy)
            fade = np.clip((1 - radius) / mw['feather'], 0, 1)
            weight[y0:y0 + len(y)] = fade * fade * (3 - 2 * fade)
        weight[sky == 0] = 0
    return sky, weight


@_scoped('regions.build_masks')
def build_masks(project_or_path, output_root, *, input_files=None):
    project = (load_project(project_or_path, input_files=input_files) if isinstance(project_or_path, (str, Path))
               else validate_project(project_or_path, input_files=input_files))
    root = _safe_root(output_root, project)
    saved = save_project(project, root)
    directory = root / ('region-assets-' + project['revision'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    sky, mw = rasterize(project)
    sky_path, mw_path = directory / 'sky.png', directory / 'milkyway.tif'
    present = milkyway_active(project)
    if not np.any(sky):
        raise ValueError('分区没有剩余天空，请检查')
    import tifffile
    if present and not np.any(mw):
        raise ValueError('银河范围未覆盖天空')
    lock = root / ('.regions-' + project['revision'] + '.lock')
    from .locking import file_lock, Busy
    stage = root / ('.partial-regions-' + uuid.uuid4().hex)
    try:
        with file_lock(lock):
            # Never silently reuse modified generated masks. OS lock ownership
            # ends on process crash; the persistent lock filename is harmless.
            if directory.exists():
                if not sky_path.is_file() or not np.array_equal(np.asarray(Image.open(sky_path)), sky):
                    raise ValueError('已保存天空分区被修改')
                if present and (not mw_path.is_file() or not np.array_equal(tifffile.imread(mw_path), mw)):
                    raise ValueError('已保存银河分区被修改')
            else:
                stage.mkdir()
                Image.fromarray(sky).save(stage / 'sky.png')
                if present:
                    tifffile.imwrite(stage / 'milkyway.tif', mw, metadata={'axes': 'YX', 'row_order': 'TOP-DOWN'})
                os.rename(stage, directory)
    except Busy as exc:
        raise ValueError('此分区正在保存，请稍后重试') from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    ref = project['source_binding']['reference']
    return {'project_path': saved['project_path'], 'revision': project['revision'],
            'sky_mask_path': str(sky_path), 'sky_mask_sha256': _sha(sky_path),
            'milkyway_mask_path': str(mw_path) if present else None,
            'milkyway_mask_sha256': _sha(mw_path) if present else None,
            'milkyway_presence': copy.deepcopy(project['milkyway']),
            'width': ref['width'], 'height': ref['height'], 'source_binding': copy.deepcopy(project['source_binding'])}


def suggest_horizon(preview_rgb, points, band=.035):
    """Return a nearby luminance-edge suggestion; caller must explicitly accept.

    A narrow search band and a displacement penalty keep stars and unrelated
    hills from turning this assistance into an unbounded segmentation claim.
    """
    _points(points, 2)
    image = np.asarray(preview_rgb, dtype=np.float32)
    if image.ndim != 3 or min(image.shape[:2]) < 2 or image.shape[2] < 3 or not np.isfinite(image).all():
        raise ValueError('边缘辅助需要可用的预览')
    # This is the 8-bit display preview, not scientific linear data. Pillow is
    # already a product dependency; edge assistance must not require SciPy.
    blurred = Image.fromarray(np.clip(image[..., :3], 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.))
    luminance = np.asarray(blurred, dtype=np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    edge = np.abs(np.gradient(luminance, axis=0))
    height, width = edge.shape
    xs = np.linspace(0, width - 1, min(width, 100))
    ys = np.interp(xs / max(1, width - 1), [p[0] for p in points], [p[1] for p in points]) * (height - 1)
    radius = max(2, round(height * _number(band, .002, .1)))
    result = []
    for x, y in zip(xs, ys):
        a, b = max(0, round(y) - radius), min(height, round(y) + radius + 1)
        column = np.median(edge[a:b, max(0, round(x) - 3):min(width, round(x) + 4)], axis=1)
        score = column * (1 - .5 * np.abs(np.arange(a, b) - y) / radius)
        pick = a + int(np.argmax(score)) if np.max(score) > 0 else round(y)
        result.append([float(x / max(1, width - 1)), float(pick / max(1, height - 1))])
    return result
