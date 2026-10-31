"""Explicit exposure-sequence admission and a bounded floating CFA adapter.

Header claims describe provenance, not proof of linearity or true sensor colour.
Original files are never edited. RGB packaging follows measured Siril geometry.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import uuid

from . import hashing

VERSION = 'fits-exposure-1'
ENGINE_SHA256 = '132c735ecb623559a945c43a617811da84c19c43988c8a234861f903dd362e0b'
CAPTURE_KEYS = ('CREATOR', 'INSTRUME', 'TELESCOP', 'PROGRAM', 'FILTER', 'GAIN', 'OFFSET', 'WIDECAM', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ')


def _number(value, key):
    if type(value) not in (int, float):
        raise ValueError(f'FITS {key} 必须是有限数值')
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f'FITS {key} 数值过大') from exc
    if not math.isfinite(result):
        raise ValueError(f'FITS {key} 必须是有限数值')
    return result


def describe_header(h: dict) -> dict:
    """Describe one supported exposure; reject recognizable non-exposures."""
    for key in ('NAXIS', 'NAXIS1', 'NAXIS2'):
        if type(h.get(key)) is not int or h[key] <= 0:
            raise ValueError(f'FITS {key} 需要正整数')
    for key in ('STACKCNT', 'NSUBS', 'NCOMBINE'):
        if key in h and _number(h[key], key) != 1:
            raise ValueError('堆栈入口需要单次曝光；检测到母版或非单帧计数：' + key)
    image_type = h.get('IMAGETYP', '')
    if not isinstance(image_type, str) or any(t in image_type.lower() for t in ('master', 'stack', 'dark', 'flat', 'bias')):
        raise ValueError('堆栈入口需要光照曝光，不能混入母版或校准帧')
    exposures = [_number(h[k], k) for k in ('EXPTIME', 'EXPOSURE') if k in h]
    if any(v <= 0 for v in exposures) or (exposures and any(v != exposures[0] for v in exposures)):
        raise ValueError('单次曝光时长无效或 EXPTIME / EXPOSURE 冲突')
    exposure = exposures[0] if exposures else None
    for key in ('LIVETIME', 'TOTALEXP'):
        if key in h:
            total = _number(h[key], key)
            if total <= 0 or exposure is None or total != exposure:
                raise ValueError('累计曝光与单次曝光不一致，不能作为单帧自动堆栈：' + key)
    bitpix = h.get('BITPIX')
    if type(bitpix) is not int or bitpix not in (16, -32):
        raise ValueError('FITS 仅接受 uint16 物理采样或 float32 RGB')
    if _number(h.get('BSCALE', 1), 'BSCALE') != 1 or _number(h.get('BZERO', 0), 'BZERO') != (32768 if bitpix == 16 else 0):
        raise ValueError('FITS 数值缩放不在已验证范围')
    shape = [h['NAXIS2'], h['NAXIS1']]
    row = h.get('ROWORDER')
    if h['NAXIS'] == 3 and type(h.get('NAXIS3')) is int and h['NAXIS3'] == 3:
        if row not in (None, 'BOTTOM-UP'):
            raise ValueError('RGB FITS 仅支持 BOTTOM-UP 的坐标，请先统一行方向')
        layout, pattern, policy = 'rgb', None, 'explicit' if row else 'legacy_rgb_bottom_up'
        interpreted_row = 'BOTTOM-UP'
    elif h['NAXIS'] == 2 and bitpix == 16:
        layout, pattern = 'cfa', h.get('BAYERPAT')
        if pattern not in ('RGGB', 'BGGR', 'GRBG', 'GBRG') or any(n % 2 for n in shape):
            raise ValueError('CFA 需要偶数画幅和明确的标准 2x2 Bayer 图案')
        for key in ('XBAYROFF', 'YBAYROFF'):
            if _number(h.get(key, 0), key) != 0:
                raise ValueError('当前 CFA 适配仅验证了零 Bayer 偏移')
        if row in ('TOP-DOWN', 'BOTTOM-UP'):
            interpreted_row, policy = row, 'explicit_cfa_header'
        elif (row is None and h.get('CREATOR') == 'ZWO Seestar S30 Pro'
              and str(h.get('PROGRAM')) == '8.46' and type(h.get('WIDECAM')) is int and h.get('WIDECAM') == 0
              and pattern == 'GRBG' and shape == [3840, 2160]):
            interpreted_row, policy = 'TOP-DOWN', 'seestar_s30pro_8_46_siril_default_assumption'
        else:
            raise ValueError('CFA 行方向缺失或未知，当前没有该组合的已验证适配')
        if exposure is None:
            raise ValueError('CFA 曝光序列需要明确的单次曝光时长')
    else:
        raise ValueError('FITS 需要三平面 RGB，或支持的二维 uint16 Bayer 曝光')
    capture = {k: h.get(k) for k in CAPTURE_KEYS}
    for key, value in capture.items():
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError(f'FITS {key} 元数据类型无效')
        if type(value) in (float, int):
            _number(value, key)
    return {'layout': layout, 'shape_hw': shape, 'bitpix': bitpix, 'bayer': pattern,
            'source_roworder': row, 'interpreted_roworder': interpreted_row, 'row_policy': policy,
            'exposure_seconds': exposure, 'capture': capture,
            'calibration': 'upstream_unknown_no_added_bias_dark_flat_or_white_balance'}


def inspect_files(paths: list[Path]) -> dict:
    from astropy.io import fits
    records = []
    for p in paths:
        try:
            records.append(describe_header(dict(fits.getheader(p))))
        except ValueError as exc:
            raise ValueError(f'{Path(p).name}: {exc}') from exc
    if not records:
        raise ValueError('空 FITS 序列')
    # Equality also rejects partially missing capture fields/exposure metadata.
    if any(r != records[0] for r in records[1:]):
        raise ValueError('请勿混合 CFA/RGB、画幅、曝光、滤镜、设备设置或行方向解释；请分组后堆栈')
    return {'version': VERSION, 'layout': records[0]['layout'], 'frames': records,
            'recipe': ('uint16_physical_to_float32_div65535_siril_debayer_rgb_no_flip'
                       if records[0]['layout'] == 'cfa' else 'existing_rgb_convert'),
            'engine_sha256': ENGINE_SHA256 if records[0]['layout'] == 'cfa' else None,
            'history_certified': False}


def fingerprint(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_engine(engine: dict) -> None:
    if engine.get('version') != '1.4.4':
        raise ValueError('CFA 浮点适配需要已验证的 Siril 1.4.4 构建')
    digest = hashing.sha256_file(Path(engine['executable']), source='fits_input.require_engine')
    if digest != ENGINE_SHA256:
        raise ValueError('CFA 适配的 Siril 构建尚未验证，请选用项目已测试的引擎')


def verify_contract(job: dict, paths: list[Path]) -> dict:
    current = inspect_files(paths)
    saved = job.get('fits_input')
    if saved is None:
        if current['layout'] == 'cfa':
            raise ValueError('旧作业没有 CFA 输入配方，请重新提交')
        return current
    if current != saved or fingerprint(saved) != job.get('fits_input_sha256'):
        raise ValueError('FITS 输入状态或适配配方已改变，请新建作业')
    if current['layout'] == 'cfa':
        require_engine(job['engine'])
    return current


def convert_cfa(engine: dict, directory: Path, staged: list[Path], work: Path,
                prefix: str, contract: dict, check, runner) -> Path:
    """Isolated new attempt, source bytes kept, commit only validated RGB outputs."""
    import numpy as np
    from astropy.io import fits
    from .records import atomic_json
    from .siril import Cancelled
    require_engine(engine)
    if inspect_files(staged) != contract:
        raise ValueError('CFA 工作副本与冻结输入配方不符')
    attempt = directory / ('cfa-' + uuid.uuid4().hex[:12])
    stage, decoded, canonical = attempt/'float', attempt/'decoded', attempt/'canonical'
    stage.mkdir(parents=True); decoded.mkdir(); canonical.mkdir()
    audit = {'version': VERSION, 'contract_sha256': fingerprint(contract), 'frames': [],
             'state': 'running', 'absolute_phase_color_verified': False}
    atomic_json(attempt/'conversion.json', audit)
    try:
        for i, (p, description) in enumerate(zip(staged, contract['frames']), 1):
            check()
            with fits.open(p, memmap=False) as hdus:
                raw = hdus[0].data
                if raw.dtype != np.uint16 or list(raw.shape) != description['shape_hw']:
                    raise ValueError('CFA 实际采样布局与输入头不符')
                a = raw.astype(np.float32) / np.float32(65535)
                header = hdus[0].header.copy()
            # BIAS is an uncertain source claim in source code units. Preserve
            # it in the audit, not as an actionable offset on normalized pixels.
            bias = header.get('BIAS')
            for key in ('BZERO', 'BSCALE', 'BIAS', 'CHECKSUM', 'DATASUM'):
                header.pop(key, None)
            header['ROWORDER'] = description['interpreted_roworder']
            header.add_history('Starstack CFA: physical uint16 / 65535 float32; no bias/dark/flat/WB.')
            fits.PrimaryHDU(a, header=header).writeto(stage/f'frame{i:05d}.fit')
            audit['frames'].append({'source_staged': str(p), 'source_bias_claim': str(bias) if bias is not None else None})
        commands = ['offline', 'setext fit', 'set32bits', 'setcompress 0', 'cd float',
                    f'convert {prefix} -debayer -out=../decoded']
        runner(engine['executable'], attempt, 'debayer', commands, check)
        outputs = sorted(decoded.glob(f'{prefix}_*.fit'))
        expected = [f'{prefix}_{i:05d}.fit' for i in range(1, len(staged)+1)]
        if [p.name for p in outputs] != expected:
            raise ValueError('CFA 去马赛克没有生成完整且编号一致的序列')
        for item, p in zip(audit['frames'], outputs):
            check()
            with fits.open(p, memmap=False) as hdus:
                a, h = hdus[0].data, hdus[0].header.copy()
                if h['BITPIX'] != -32 or a.shape != (3, *contract['frames'][0]['shape_hw']) or not np.isfinite(a).all():
                    raise ValueError('CFA 去马赛克输出位深、尺寸或有限性不符')
                for key in ('BAYERPAT', 'XBAYROFF', 'YBAYROFF', 'BZERO', 'BSCALE', 'CHECKSUM', 'DATASUM'):
                    h.pop(key, None)
                h['ROWORDER'] = 'BOTTOM-UP'
                h.add_history('Starstack RGB packaging: stored array unchanged, Siril display geometry.')
                fits.PrimaryHDU(a, header=h).writeto(canonical/p.name, overwrite=False)
                item.update(min=float(a.min()), max=float(a.max()), above_one=int(np.count_nonzero(a > 1)))
        # All decoded frames pass before any publication. A cancelled copy can
        # still leave partial work files; the caller owns/clears that directory
        # and may checkpoint only after this function returns successfully.
        for name in expected:
            check()
            if (work/name).exists():
                raise ValueError('CFA 输出位置已有文件，不能覆盖')
            shutil.copyfile(canonical/name, work/name)
        audit['state'] = 'succeeded'
    except Exception as exc:
        audit.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        if isinstance(exc, RuntimeError) and not isinstance(exc, Cancelled):
            raise RuntimeError(f"{exc}。CFA 诊断：{attempt/'conversion.json'}；"
                               f"引擎日志：{attempt/'debayer.log'}") from exc
        raise
    finally:
        atomic_json(attempt/'conversion.json', audit)
    return attempt/'conversion.json'
