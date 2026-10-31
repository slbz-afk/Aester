"""Atomic GUI editing sessions; immutable region revisions carry source identity."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import uuid

from . import color_adjustments, hashing, regions, orientation


def _validate(state, reference, project):
    regions.validate_project(project, input_files=state['input_files'])
    color_adjustments.normalize(state['color_adjustments'])
    rotations = [orientation.normalize_rotation(value.get('image_rotation', 0))
                 for value in (state, reference, project['source_binding']['reference'])]
    if len(set(rotations)) != 1:
        raise ValueError('编辑项目、首帧与绘制分区的方向不一致')
    for key, value in project['source_binding']['reference'].items():
        if (reference.get(key, 0) if key == 'image_rotation' else reference.get(key)) != value:
            raise ValueError('编辑项目的首帧不一致')
    if regions._sha(reference['preview_path']) != reference['preview_sha256']:
        raise ValueError('编辑项目的预览已改变')
    for key in ('trails_enabled', 'stars_enabled', 'milkyway_enabled', 'auto_strength', 'stack_range'):
        if not isinstance(state.get(key), bool):
            raise ValueError('编辑项目开关无效')
    for key in ('milkyway_strength', 'background_strength'):
        regions._number(state.get(key))
    regions._number(state.get('milkyway_offset', 0), -1., 1.)
    if state.get('scene_mode') not in ('sky', 'landscape', 'unknown'):
        raise ValueError('编辑项目场景无效')
    if state['milkyway_enabled'] and (project['milkyway']['status'] != 'present' or project['milkyway']['authority'] != 'human'):
        raise ValueError('银河增强缺少人工存在确认')
    allowed = {'channel_order': ('rgb', 'bgr'), 'value_range': ('0..1', '-1..1'),
               'layout': ('nhwc', 'nchw'), 'stitch': ('blend', 'crop')}
    for key, options in allowed.items():
        if state['backend_options'].get(key) not in options:
            raise ValueError('编辑项目后端选项无效')


@regions._scoped('editing_project.save_project')
def save_project(path, state, reference, project):
    _validate(state, reference, project)
    path = Path(path).resolve()
    regions._safe_root(path.parent, project)
    if path in [Path(f['path']).resolve() for f in project['source_binding']['input_files']]:
        raise ValueError('不能覆盖原片')
    saved_regions = regions.save_project(project, path.parent / (path.stem + '.assets'))
    record = {'schema': 'aester-editing-project-1', 'state': copy.deepcopy(state),
              'reference': copy.deepcopy(reference), 'regions_path': saved_regions['project_path'],
              'regions_revision': saved_regions['revision']}
    # External compatibility masks remain bound, rather than silently loading a changed file.
    mask = state.get('sky_mask_path')
    record['external_mask_sha256'] = regions._sha(mask) if mask else None
    record['signature'] = regions._revision(record)
    temp = path.parent / ('.editing-' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with hashing.writing(path):
        os.replace(temp, path)
    return record | {'project_path': str(path)}


@regions._scoped('editing_project.load_project')
def load_project(path):
    path = Path(path).resolve()
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('编辑项目过大')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('schema') != 'aester-editing-project-1':
            raise ValueError('不是 Aester 编辑项目')
        signed = {k: v for k, v in record.items() if k != 'signature'}
        if record.get('signature') != regions._revision(signed):
            raise ValueError('编辑项目内容已改变')
        project = regions.load_project(record['regions_path'], input_files=record['state']['input_files'])
        if project['revision'] != record['regions_revision']:
            raise ValueError('编辑项目分区已改变')
        _validate(record['state'], record['reference'], project)
        mask = record['state'].get('sky_mask_path')
        if mask and regions._sha(mask) != record['external_mask_sha256']:
            raise ValueError('编辑项目的分区文件已改变')
        return record | {'project_path': str(path), 'regions': project}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('编辑项目格式无效') from exc
