"""Opt-in, audit-bound float RAW/color verification, separate from stacking.

The local bilinear backend is a numerical baseline, not a claim of final
photographic quality. Existing original files, audit arrays and jobs are read
only. All results go to a unique directory with a fixed reference color recipe.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import tifffile

from . import core, raw_audit, __version__
from .float_demosaic import ALGORITHM, demosaic_bilinear, rgb_pattern
from .color_preview import preview_recipe, save_preview
# The same local evidence validation rules are used by manual trace capture.
from .transients import _path, _json_with_hash, _record_path, _sha_checked, _load_codes


def load_development_audit(audit_dir) -> tuple[dict, np.ndarray, dict]:
    audit, codes, model, _ = _load_development_audit(audit_dir)
    return audit, codes, model


def _load_development_audit(audit_dir) -> tuple[dict, np.ndarray, dict, str]:
    directory = _path(audit_dir)
    manifest = directory/'audit.json'
    audit, manifest_hash = _json_with_hash(manifest)
    if (audit.get('state') != 'succeeded' or audit.get('schema_version') != 1
            or _record_path(audit.get('audit_dir')) != directory):
        raise ValueError('需要所属目录一致的成功 RAW 检查记录')
    source = audit.get('source', {})
    source_path = _record_path(source.get('path'))
    if source_path.stat().st_size != source.get('size_bytes'):
        raise ValueError('检查后原片尺寸发生变化')
    _sha_checked(source_path, source.get('sha256'))
    geometry = audit.get('geometry', {})
    h, w = geometry.get('height'), geometry.get('width')
    if (type(h) is not int or type(w) is not int or min(h, w) < 2
            or h*w > 120_000_000 or geometry.get('row_order') != 'TOP-DOWN'
            or geometry.get('coordinate_basis') != 'decoder_full_raw_mosaic'
            or geometry.get('orientation_applied') is not False):
        raise ValueError('需要完整、未旋转的 TOP-DOWN CFA 坐标')
    rgb_pattern(geometry.get('full_pattern'))
    metadata = audit.get('libraw_metadata', {})
    fields = metadata.get('fields', {})
    if (fields.get('color_desc', {}).get('value') != 'RGBG'
            or fields.get('num_colors', {}).get('value') != 3
            or fields.get('raw_pattern', {}).get('value') != geometry['full_pattern']):
        raise ValueError('此实验仅支持已确认的三色 RGBG Bayer，不能猜测第四种颜色')
    # Re-read small public file tags: this also prevents accidentally substituting
    # another file's black model or relying on an edited audit field.
    dng = raw_audit.read_dng_tags(source_path)
    if dng != audit.get('file_metadata'):
        previous = audit.get('versions', {}).get('tifffile', '未记录')
        raise ValueError('当前 DNG 文件标签与检查记录不一致；请另存重新检查，勿修改旧记录 '
                         f'(tifffile: 检查时 {previous}，当前 {tifffile.__version__})')
    model = raw_audit.black_model(dng, (h, w))
    if model.get('status') != 'available' or model != audit.get('black_model'):
        raise ValueError('此文件的浮点扣黑模型尚未验证；保留 CFA 检查结果')
    specification = audit.get('artifacts', {}).get('raw_codes', {})
    codes_path = _record_path(specification.get('path'))
    if codes_path != directory/'raw_codes.npy':
        raise ValueError('只能使用当前检查所属的 raw_codes.npy')
    codes = _load_codes(codes_path, specification)
    if codes.shape != (h, w):
        raise ValueError('CFA 与检查画幅不一致')
    _sha_checked(manifest, manifest_hash)
    return audit, codes, model, manifest_hash


def _array_artifact(path, array, **extra):
    minimum, maximum, negative, above_one = float('inf'), -float('inf'), 0, 0
    for y in range(0, array.shape[0], 128):
        tile = np.asarray(array[y:y+128])
        if not np.isfinite(tile).all():
            raise ValueError('浮点产物包含非有限数值')
        minimum, maximum = min(minimum, float(tile.min())), max(maximum, float(tile.max()))
        negative += int(np.count_nonzero(tile < 0))
        above_one += int(np.count_nonzero(tile > 1))
    return {'path': str(path), 'sha256': core._sha(path), 'shape': list(array.shape),
            'dtype': str(array.dtype), 'size_bytes': path.stat().st_size,
            'minimum': minimum, 'maximum': maximum, 'negative_samples': negative,
            'samples_above_numeric_one': above_one, **extra}


def develop_raw(audit_dir, output_root, *, recipe_path=None, exposure_ev=2.0) -> dict:
    """Create a signed camera-RGB master plus fixed-color derivative and preview.

    Canon/unknown black models fail explicitly. The color recipe can be reused
    within the same verified sensor/scale family; exposure correction, calibrated
    photometry, DNG full profiles and automatic scene WB are outside this API.
    """
    from . import raw_color
    display = preview_recipe(exposure_ev)
    audit, codes, model, manifest_hash = _load_development_audit(audit_dir)
    directory = _path(audit_dir)
    output = _path(output_root, exists=False)
    source_path = _record_path(audit['source']['path'])
    if output.is_relative_to(source_path.parent) or output.is_relative_to(directory):
        raise ValueError('浮点衍生需另存于原片文件夹（含子目录）和检查目录之外')
    recipe_source = None
    if recipe_path is not None:
        recipe_source = _path(recipe_path)
        recipe, recipe_hash = _json_with_hash(recipe_source)
        raw_color.validate_recipe(recipe, audit['libraw_metadata'])
        _sha_checked(recipe_source, recipe_hash)
    else:
        # One common denominator from the reference, not a per-channel exposure
        # normalization; each frame's declared black is subtracted separately.
        repeat = model['repeat_shape']
        black_tile = np.tile(np.asarray(model['black_values']).reshape(repeat),
                             (2//repeat[0], 2//repeat[1]))
        green = rgb_pattern(audit['geometry']['full_pattern']) == 1
        reference_black_green = float(black_tile[green].mean())
        recipe = raw_color.build_recipe(audit['libraw_metadata'],
            reference_sha256=audit['source']['sha256'], reference_white=model['white_level'],
            reference_black_green=reference_black_green)
    compatibility = {'geometry': {k: audit['geometry'][k] for k in
                       ('width', 'height', 'row_order', 'full_pattern')},
                     'black_model': model}
    if recipe_source is None:
        recipe['frame_compatibility'] = compatibility
    elif recipe.get('frame_compatibility') != compatibility:
        raise ValueError('这份配方的 CFA、画幅或黑白电平与当前帧不兼容')
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < codes.size*32 + 64*1024**2:
        raise ValueError('空间不足以保存两个独立浮点 RGB 产物')
    target = output/f'raw-develop-{uuid.uuid4().hex}'
    target.mkdir()
    result = {'schema_version': 1, 'state': 'running', 'development_dir': str(target),
              'created_at': core.now(), 'source': audit['source'], 'artifacts': {},
              'audit': {'path': str(directory/'audit.json'), 'sha256': manifest_hash},
              'recipe_mode': 'reused_reference' if recipe_source is not None else 'new_reference',
              'recipe_source': ({'path': str(recipe_source), 'sha256': recipe_hash}
                                if recipe_source is not None else None),
              'geometry': audit['geometry'], 'error': None,
              'implementation': {'version': __version__, 'algorithm': ALGORITHM,
                  'module_sha256': {p.name: core._sha(p) for p in
                    [Path(__file__), Path(raw_color.__file__),
                     Path(__file__).with_name('float_demosaic.py'),
                     Path(__file__).with_name('color_preview.py')]}},
              'processing': {'black_subtractions': 1, 'demosaic': ALGORITHM,
                  'cfa_source': 'hash_verified_audit_raw_codes_not_redecoded',
                  'integrity_scope': 'local_consistency_hashes_not_signatures',
                  'row_order': 'TOP-DOWN', 'rotation': False, 'crop': False,
                  'clipping_in_masters': False, 'per_frame_white_balance': False,
                  'automatic_brightness': False, 'photographic_quality_accepted': False},
              'limitations': ['浮点双线性为数值验证基准，未验收最终星点重建画质',
                  '颜色为 LibRaw 矩阵近似；未执行完整 DNG profile、OpcodeList3 或镜头校正',
                  '固定参考白平衡不保证混合光源、天文发射线或绝对颜色精度',
                  '跨帧一致性要求显式复用同一 color_recipe.json；未自动建立批次',
                  '此入口输出 TIFF；进入 Siril 前的数值单位和导入行为仍需另验',
                  '独立实验入口，尚未替换默认堆栈／GUI／MCP 前端']}
    core.atomic_json(target/'development.json', result)
    try:
        core.atomic_json(target/'audit_snapshot.json', audit)
        core.atomic_json(target/'color_recipe.json', recipe)
        result['artifacts']['color_recipe'] = {'path': str(target/'color_recipe.json'),
            'sha256': core._sha(target/'color_recipe.json')}
        result['artifacts']['audit_snapshot'] = {'path': str(target/'audit_snapshot.json'),
            'sha256': core._sha(target/'audit_snapshot.json')}
        signal = raw_audit.subtract_declared_black(codes, model)
        shape = (*signal.shape, 3)
        camera_path = target/'camera_rgb_dn_32bit.tif'
        camera = tifffile.memmap(camera_path, shape=shape, dtype=np.float32, photometric='rgb',
            metadata={'color_space': 'camera_RGB', 'linear': True, 'row_order': 'TOP-DOWN',
                      'units': 'decoder_DN_minus_declared_black', 'demosaic': ALGORITHM})
        demosaic_bilinear(signal, audit['geometry']['full_pattern'], out=camera)
        camera.flush()
        del signal
        result['artifacts']['camera_rgb'] = _array_artifact(camera_path, camera,
            color_space='camera_RGB', units='decoder_DN_minus_declared_black',
            white_balance_applied=False, color_matrix_applied=False)
        work_path = target/'linear_srgb_32bit.tif'
        work = tifffile.memmap(work_path, shape=shape, dtype=np.float32, photometric='rgb',
            metadata={'color_space': 'linear_sRGB_D65_approximation', 'transfer': 'linear',
                      'row_order': 'TOP-DOWN', 'negative_and_over_one_preserved': True})
        raw_color.apply_color(camera, recipe, input_state='black_subtracted_camera_rgb_DN',
                              row_order='TOP-DOWN', out=work)
        work.flush()
        result['artifacts']['linear_srgb'] = _array_artifact(work_path, work,
            color_space='linear_sRGB_D65_approximation', icc_profile=None,
            units='fixed_reference_normalized_linear_values',
            icc_note='没有附加使用非线性传递函数的普通 sRGB ICC；线性空间写入 TIFF 描述与配方')
        preview_path = target/'preview.jpg'
        preview = save_preview(preview_path, work, display)
        result['artifacts']['preview'] = {'path': str(preview_path), 'sha256': core._sha(preview_path), **preview}
        del camera, work, codes
        _sha_checked(source_path, audit['source']['sha256'])
        _sha_checked(directory/'audit.json', manifest_hash)
        _sha_checked(_record_path(audit['artifacts']['raw_codes']['path']), audit['artifacts']['raw_codes']['sha256'])
        if recipe_source is not None:
            _sha_checked(recipe_source, recipe_hash)
        result.update(state='succeeded', source_unchanged=True, completed_at=core.now())
    except Exception as exc:
        result.update(state='failed', error=str(exc), error_type=type(exc).__name__, completed_at=core.now())
    core.atomic_json(target/'development.json', result)
    return result
