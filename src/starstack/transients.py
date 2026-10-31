"""Independent, manually selected decoded-CFA evidence; no detection/composite."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np

from . import core, hashing

CLASSIFICATIONS = {'unknown', 'meteor', 'satellite', 'aircraft', 'artifact'}
MAX_RECORD_BYTES = 2 * 1024**2


def _path(value, *, exists=True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value) or '\x00' in str(value):
        raise ValueError('无效的证据路径')
    path = Path(value).expanduser().absolute()
    if '..' in path.parts:
        raise ValueError('证据路径不能包含上级目录跳转')
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError('证据路径不允许符号链接或目录联接')
    return path.resolve(strict=exists)


def _json(path: Path) -> dict:
    return _json_with_hash(path)[0]


def _json_with_hash(path: Path) -> tuple[dict, str]:
    """Parse and fingerprint the same bounded byte buffer (local evidence only)."""
    path = _path(path)
    if not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError('证据记录大小或类型不受支持')
    with path.open('rb') as stream:
        data = stream.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError('证据记录大小或类型不受支持')
    record = json.loads(data.decode('utf-8'),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('非有限数值')))
    if not isinstance(record, dict):
        raise ValueError('证据记录必须是对象')
    return record, hashing.sha256_bytes(data, source="transients.record")


def _record_path(value) -> Path:
    if not isinstance(value,str) or not Path(value).is_absolute():
        raise ValueError('持久证据记录中的路径必须是绝对路径')
    return _path(value)


def _sha_checked(path: Path, expected) -> str:
    digest = core._sha(path)
    if not isinstance(expected, str) or digest != expected:
        raise ValueError('证据哈希不匹配：' + path.name)
    return digest


def _rectangle(rectangle, width, height) -> tuple[int, int, int, int]:
    if (not isinstance(rectangle, list) or len(rectangle) != 4
            or any(type(v) is not int for v in rectangle)):
        raise ValueError('区域须为四个整数 [x,y,width,height]')
    x, y, w, h = rectangle
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
        raise ValueError('区域超出完整 CFA 或宽高不为正数')
    return x, y, w, h


def _array_record(path: Path, array: np.ndarray) -> dict:
    return {'path':str(path), 'sha256':core._sha(path), 'shape':list(array.shape),
            'dtype':str(array.dtype), 'size_bytes':path.stat().st_size}


def _load_codes(path: Path, specification: dict) -> np.ndarray:
    _sha_checked(path, specification.get('sha256'))
    array = np.load(path, allow_pickle=False, mmap_mode='r')
    if (not isinstance(array,np.ndarray) or array.ndim != 2
            or array.dtype.kind != 'u' or array.dtype.itemsize > 2
            or not 0 < array.size <= 120_000_000
            or list(array.shape) != specification.get('shape')
            or str(array.dtype) != specification.get('dtype')):
        raise ValueError('CFA 证据的尺寸或数值类型不匹配')
    return array


def _selection(result: dict) -> dict:
    return {'schema_version':1,'candidate_id':result['candidate_id'],
            'rectangle':result['rectangle'],'crop_cfa_pattern':result['crop_cfa_pattern'],
            'geometry':result['geometry'],'source_sha256':result['source']['sha256'],
            'audit_sha256':result['audit']['sha256'],
            'crop_sha256':result['artifacts']['crop']['sha256']}


def record_candidate(audit_dir, output_root, *, rectangle: list[int],
                     classification='unknown', note='') -> dict:
    """Retain a source-verified crop in its own folder, with an audit snapshot.

    Coordinates refer to full decoded CFA, never a registered/cropped preview.
    The source and audit must still match at creation. Later read-back validates
    only the independent retained evidence, allowing an audit cache to be absent.
    """
    if not isinstance(classification,str) or classification not in CLASSIFICATIONS:
        raise ValueError('未知的人工分类')
    if not isinstance(note,str) or len(note) > 4000 or '\x00' in note:
        raise ValueError('备注须为不超过 4000 字的文本')
    directory = _path(audit_dir)
    manifest = _path(directory/'audit.json')
    audit, manifest_hash = _json_with_hash(manifest)
    if (audit.get('schema_version') != 1 or audit.get('state') != 'succeeded'
            or _record_path(audit.get('audit_dir')) != directory):
        raise ValueError('需要所属目录一致的成功 RAW 检查记录')
    geometry = audit.get('geometry', {})
    width, height = geometry.get('width'), geometry.get('height')
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or width*height > 120_000_000 or geometry.get('row_order') != 'TOP-DOWN'
            or geometry.get('coordinate_basis') != 'decoder_full_raw_mosaic'):
        raise ValueError('检查记录不具有已验证的完整 CFA 坐标')
    x,y,w,h = _rectangle(rectangle,width,height)
    source = audit.get('source',{})
    source_path = _record_path(source.get('path'))
    if not source_path.is_file() or source_path.stat().st_size != source.get('size_bytes'):
        raise ValueError('源文件尺寸或类型发生变化')
    _sha_checked(source_path,source.get('sha256'))
    specification = audit.get('artifacts',{}).get('raw_codes',{})
    codes_path = _record_path(specification.get('path'))
    if codes_path != directory/'raw_codes.npy':
        raise ValueError('CFA 必须为该检查独立保存的 raw_codes.npy')
    codes = _load_codes(codes_path,specification)
    if codes.shape != (height,width):
        raise ValueError('CFA 尺寸与坐标记录不匹配')
    crop = np.array(codes[y:y+h,x:x+w],copy=True)
    del codes
    _sha_checked(codes_path,specification['sha256'])
    _sha_checked(source_path,source['sha256'])
    _sha_checked(manifest,manifest_hash)
    output = _path(output_root,exists=False)
    if output.is_relative_to(source_path.parent) or output.is_relative_to(directory):
        raise ValueError('候选证据须另存于原片文件夹（含子目录）和检查目录之外')
    output.mkdir(parents=True,exist_ok=True)
    target = output/f'candidate-{uuid.uuid4().hex}'
    target.mkdir()
    result = {'schema_version':1,'state':'preparing','candidate_dir':str(target),
        'candidate_id':target.name,'created_at':core.now(), 'source':source,
        'audit':{'path':str(manifest),'sha256':manifest_hash},
        'capture_time':audit.get('capture_time',{'value':None,'timezone':None,'status':'unknown'}),
        'geometry':geometry,'rectangle':list(rectangle),'classification':classification,
        'classification_source':'manual','note':note,'artifacts':{},
        'processing':{'detection_applied':False,'rejection_applied':False,
                      'demosaic_applied':False,'black_subtraction_applied':False,
                      'registration_applied':False,'composite_applied':False},
        'limitations':['人工矩形采样证据，未经物理分类或背景分离；不是流星图层',
                       '源坐标为完整CFA，未变换到天空堆栈坐标'], 'error':None}
    try:
        np.save(target/'crop.npy',crop,allow_pickle=False)
        result['artifacts']['crop']=_array_record(target/'crop.npy',crop)
        core.atomic_json(target/'audit_snapshot.json',audit)
        result['artifacts']['audit_snapshot']={'path':str(target/'audit_snapshot.json'),
            'sha256':core._sha(target/'audit_snapshot.json')}
        pattern=geometry.get('full_pattern')
        result['crop_cfa_pattern']=None
        if isinstance(pattern,list):
            pattern=np.asarray(pattern)
            if pattern.ndim==2 and 0 < pattern.size <= 256:
                result['crop_cfa_pattern']=np.roll(pattern,(-y % pattern.shape[0],-x % pattern.shape[1]),axis=(0,1)).tolist()
        core.atomic_json(target/'selection.json',_selection(result))
        result['artifacts']['selection']={'path':str(target/'selection.json'),
            'sha256':core._sha(target/'selection.json')}
        _sha_checked(source_path,source['sha256'])
        _sha_checked(manifest,manifest_hash)
        result.update(state='succeeded',source_verified_at_creation=True,completed_at=core.now())
    except Exception as exc:
        result.update(state='failed',error=str(exc),completed_at=core.now())
    core.atomic_json(target/'candidate.json',result)
    return result


def read_candidate(candidate_dir) -> dict:
    """Verify retained evidence without following external source/audit paths."""
    directory=_path(candidate_dir)
    result=_json(directory/'candidate.json')
    if (result.get('schema_version')!=1 or result.get('state')!='succeeded'
            or _record_path(result.get('candidate_dir'))!=directory or result.get('candidate_id')!=directory.name):
        raise ValueError('候选不是所属目录一致的成功记录')
    artifacts=result.get('artifacts',{})
    for key,filename in (('crop','crop.npy'),('audit_snapshot','audit_snapshot.json'),('selection','selection.json')):
        item=artifacts.get(key,{})
        path=_record_path(item.get('path'))
        if path!=directory/filename:
            raise ValueError('候选产物必须保存在其独立目录内')
        _sha_checked(path,item.get('sha256'))
    if _json(directory/'selection.json')!=_selection(result):
        raise ValueError('候选坐标或来源与独立选区记录不匹配')
    audit=_json(directory/'audit_snapshot.json')
    geometry=result.get('geometry',{})
    if (audit.get('state')!='succeeded' or audit.get('source')!=result.get('source')
            or audit.get('geometry')!=geometry):
        raise ValueError('候选来源与所保留的检查快照不匹配')
    _,_,w,h=_rectangle(result.get('rectangle'),geometry['width'],geometry['height'])
    crop=_load_codes(directory/'crop.npy',artifacts['crop'])
    if crop.shape!=(h,w):
        raise ValueError('候选裁块与矩形尺寸不一致')
    del crop
    return {**result,'integrity':{'owned_evidence_verified':True,
        'external_source_rechecked':False,'external_audit_rechecked':False,
        'meaning':'校验独立裁块与来源快照；不要求原检查缓存仍存在'}}
