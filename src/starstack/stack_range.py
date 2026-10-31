"""Opt-in positive-domain range protection around the pinned Siril stack.

Registered source frames remain in original units for transient processing.
Every attempt owns its scaled copies; only a validated decoded master is
published. Common RGB zero retains the existing pipeline's no-data meaning.
Contract 1.0.0 accepts only all-positive samples; contract 1.1.0 (new jobs)
accepts every other finite value with magnitude at most input_maximum (8),
leaving the low end to Siril's clip at 0 as in legacy mode and counting it.
Each job runs, verifies and resumes under the contract saved with it. This
is not a signed-RAW or a true-zero adapter.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import time
import uuid

import numpy as np
from astropy.io import fits

from .records import atomic_json
from .siril import Cancelled
from .siril_sequence import read_seq
from . import hashing, range_tiles

MODE = 'positive_power2_v1'
VERSION_V1 = '1.0.0'
VERSION = '1.1.0'   # contract for new jobs
RECORD = 'stack_range.json'
VALID = 'stack_range_valid.npy'
ENGINE_SHA256 = '132c735ecb623559a945c43a617811da84c19c43988c8a234861f903dd362e0b'
STACK = 'stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=encoded_master.fit'
INPUT_LIMIT = 8.0   # contract input_maximum; under 1.1.0 it bounds |sample|


SOURCE = 'stack_range.sha'


def sha(path):
    return hashing.sha256_file(Path(path), source=SOURCE)


def _sha_many(paths):
    """[sha(p) for p in paths], hashed in parallel; fails like that loop (lowest failing index)."""
    return hashing.sha256_many([Path(p) for p in paths], source=SOURCE)


class _ReadAhead:
    """``sha(paths[i])`` for i = 0, 1, 2, ... in order, the next files hashed ahead on a small pool.

    ``get(i)`` returns (or raises) exactly what ``sha(paths[i])`` would at that point of a
    serial loop, so the caller's own checks and scans between files keep their order; at
    most as many files as there are hash workers are read ahead of the caller.
    """

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.workers = hashing.resolve_workers(None)
        self.pool = (ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='aester-range-hash')
                     if self.workers > 1 and len(self.paths) > 1 else None)
        self.futures = {}
        self.submitted = 0

    def get(self, index):
        if self.pool is None:
            return sha(self.paths[index])
        work = hashing.carry_scope(sha)
        while self.submitted < min(len(self.paths), index + 1 + self.workers):
            self.futures[self.submitted] = self.pool.submit(work, self.paths[self.submitted])
            self.submitted += 1
        return self.futures.pop(index).result()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


ZERO_SEMANTICS = {
    VERSION_V1: 'common_rgb_zero_is_existing_nodata; true_zero_not_supported',
    VERSION: ('common_rgb_zero_is_existing_nodata; other_finite_values_including_single_channel_zero_'
              'and_negative_are_samples; low_end_clipped_at_0_by_siril_like_legacy_and_counted; nonfinite_refused'),
}


def contract(version=VERSION):
    """A fresh copy of one known contract; the default is the one for new jobs."""
    if not isinstance(version, str) or version not in ZERO_SEMANTICS:
        raise ValueError('堆栈范围配方缺失或已改变，请新建作业')
    return {'mode': MODE, 'version': version, 'engine_sha256': ENGINE_SHA256,
            'backend': 'python_float32_power2_copy', 'input_maximum': INPUT_LIMIT,
            'target_maximum': 0.75, 'max_attempts': 3, 'stack_command': STACK,
            'zero_semantics': ZERO_SEMANTICS[version]}


def known_contract(value):
    """A fresh copy of ``value`` when it is exactly one known contract; otherwise refuse."""
    version = value.get('version') if isinstance(value, dict) else None
    if not isinstance(version, str) or version not in ZERO_SEMANTICS or value != contract(version):
        raise ValueError('堆栈范围配方缺失或已改变，请新建作业')
    return contract(version)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def longest_engine_path(job_dir):
    """The longest path Siril writes in a range-protected stack of a new job in ``job_dir`` (admission check S-8):
    the last trial's encoded master. run_protected_stack names its attempt 'range-' + 12 hex digits inside
    Path(directory).resolve(); the placeholder has that length."""
    return (Path(job_dir).resolve() / ('range-' + '0' * 12) / ('trial' + str(contract()['max_attempts']))
            / 'encoded_master.fit')


def require_engine(engine):
    if engine.get('version') != '1.4.4' or sha(engine['executable']) != ENGINE_SHA256:
        raise ValueError('高光范围保护需要项目已验证的 Siril 1.4.4 构建')


def verify_job(job):
    mode = job.get('stack_range_mode', 'legacy')
    if mode == 'finite_validity_v1':
        from . import finite_stack
        return finite_stack.verify_job(job)
    saved = job.get('stack_range_contract')
    if mode == 'legacy' and saved is None and job.get('schema_version', 1) < 2:
        return False
    if job.get('schema_version') != 2 or mode != MODE:
        raise ValueError('堆栈范围配方缺失或已改变，请新建作业')
    known_contract(saved)   # either the 1.0.0 or the 1.1.0 contract, exactly as saved
    if job.get('stack_range_sha256') != fingerprint(saved):
        raise ValueError('堆栈范围配方缺失或已改变，请新建作业')
    require_engine(job['engine'])
    return True


@contextmanager
def image(path, mode='readonly'):
    with fits.open(path, mode=mode, memmap=True, do_not_scale_image_data=True) as hdus:
        h, a = hdus[0].header, hdus[0].data
        if (len(hdus) != 1 or h.get('BITPIX') != -32 or a is None
                or a.ndim != 3 or a.shape[0] != 3 or min(a.shape) <= 0
                or a.dtype.kind != 'f' or a.dtype.itemsize != 4
                or h.get('BSCALE', 1) != 1 or h.get('BZERO', 0) != 0
                or h.get('ROWORDER', 'BOTTOM-UP') not in ('BOTTOM-UP', 'TOP-DOWN')):
            raise ValueError('范围保护要求未压缩、无隐式比例的三通道 float32 FITS')
        yield hdus, a


def _tiles(a, check):
    for start in range(0, a.shape[1], 64):
        check()
        yield start, np.array(a[:, start:start + 64], dtype=np.float32, order='C')


def _sources(work, check, version=VERSION_V1):
    finite = version != VERSION_V1
    seq_path = work / 'r_light_.seq'
    sequence = read_seq(seq_path)
    h = sequence['header']
    frames = sequence['images']
    selected = [f for f in frames if f['selected']]
    if (h['name'] != 'r_light_' or h['version'] != 6 or h['variable_size']
            or h['fz_flag'] or h['drizzle'] or sequence['layers'] != 3
            or len(selected) < 3 or h['nb_selected'] != len(selected)
            or not frames[h['reference_image']]['selected']
            or not sequence['all_identity_per_layer']
            or not all(sequence['all_identity_per_layer'].values())):
        raise ValueError('范围保护要求已完成配准的普通RGB序列及有效参考帧')
    names = [f['stem'] + '.fit' for f in frames]
    if (len(names) != len(set(names)) or any(not re.fullmatch(r'r_light_\d+\.fit', n) for n in names)
            or {p.name for p in work.glob('r_light_*.fit')} != set(names)):
        raise ValueError('范围保护的注册帧与序列记录不一致')
    expected, shape, order, maximum, records = None, None, None, 0.0, []
    # The frames' digests are read ahead in parallel; each is taken (or its error raised) at the
    # point where the serial loop hashed that frame, after its check() and the earlier scans.
    ahead = _ReadAhead([work / name for name in names])
    try:
        for index, (frame, name) in enumerate(zip(frames, names)):
            check()
            path = work / name
            digest = ahead.get(index)
            with image(path) as (hdus, a):
                current_order = hdus[0].header.get('ROWORDER', 'BOTTOM-UP')
                if shape is None:
                    shape, order = a.shape, current_order
                    expected = np.zeros(shape[1:], dtype=bool)
                elif a.shape != shape or current_order != order:
                    raise ValueError('范围保护的注册帧几何或行序不一致')
                if finite:
                    records.append(_scan_finite(frame, name, digest, a, expected, check))
                    maximum = max(maximum, records[-1]['maximum'])
                    continue
                low, high, count = float('inf'), 0.0, 0
                for y, tile in _tiles(a, check):
                    positive = (tile > 0).all(axis=0)
                    zero = (tile == 0).all(axis=0)
                    if not np.isfinite(tile).all() or not (positive | zero).all():
                        raise ValueError('高光范围保护暂不支持负值、非有限值或RGB通道独立的零值；未裁切这些数据')
                    if positive.any():
                        low = min(low, float(tile[:, positive].min()))
                        high = max(high, float(tile[:, positive].max()))
                    if frame['selected']:
                        expected[y:y + tile.shape[1]] |= positive
                        count += int(positive.sum())
                maximum = max(maximum, high)
                records.append({'file': name, 'sha256': digest, 'selected': frame['selected'],
                                'minimum_positive': low if np.isfinite(low) else None,
                                'maximum': high, 'positive_pixels': count})
    finally:
        ahead.close()
    if not expected.any():
        raise ValueError('范围保护没有任何有输入覆盖的位置' if finite else '范围保护没有有效的正值输入')
    divisor = range_tiles.choose_divisor(maximum)
    return sequence, records, expected, shape, order, divisor


def _scan_finite(frame, name, digest, a, expected, check):
    """Contract 1.1.0 scan of one frame: covered = not common RGB zero.

    Negative values do not enter the maximum, which only sets the divisor."""
    low, high, covered_pixels, channel_zero, negative = float('inf'), 0.0, 0, 0, 0
    for y, tile in _tiles(a, check):
        if not np.isfinite(tile).all():
            raise ValueError('高光范围保护不支持非有限值（NaN 或无穷）；未裁切这些数据')
        covered = ~(tile == 0).all(axis=0)
        if covered.any():
            values = tile[:, covered]
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
            channel_zero += int(np.count_nonzero(values == 0))
            negative += int(np.count_nonzero(values < 0))
        if frame['selected']:
            expected[y:y + tile.shape[1]] |= covered
        covered_pixels += int(covered.sum())
    # The contract's input bound (8) applies to magnitude: squared deviations in
    # the engine's float32 rejection statistics stay far from overflow, so the
    # low end is clipped exactly as in legacy mode whatever the divisor.
    if low < -INPUT_LIMIT or high > INPUT_LIMIT:
        raise ValueError('高光范围保护的输入须在 -8 到 8 之间；未裁切这些数据')
    return {'file': name, 'sha256': digest, 'selected': frame['selected'],
            'minimum': low if np.isfinite(low) else None, 'maximum': high,
            'covered_pixels': covered_pixels, 'single_channel_zero_samples': channel_zero,
            'negative_samples': negative}


def _write_scaled(source, target, divisor, restore, check, version=VERSION_V1):
    """One task-owned copy, chunk arithmetic; never mutate the source file."""
    rescale = range_tiles.rescale_positive_tile if version == VERSION_V1 else range_tiles.rescale_finite_tile
    check()
    with hashing.writing(target):   # a copy, then rewritten in place (update mode)
        _write_scaled_in_place(source, target, divisor, restore, check, version, rescale)
    check()


def _write_scaled_in_place(source, target, divisor, restore, check, version, rescale):
    shutil.copyfile(source, target)
    with image(target, 'update') as (hdus, a):
        for y, tile in _tiles(a, check):
            try:
                a[:, y:y + tile.shape[1]] = rescale(tile, divisor, restore=restore)
            except ValueError:
                if version == VERSION_V1:
                    raise
                # e.g. a subnormal sample that the power-of-two scale would underflow
                raise ValueError('高光范围保护的缩放无法逐位还原全部数据（如极小的次正规值）；未裁切这些数据') from None
        header = hdus[0].header
        for key in ('DATAMIN', 'DATAMAX', 'CHECKSUM', 'DATASUM'):
            header.pop(key, None)
        header['SSRGVER'] = version
        header['SSRGDIV'] = divisor
        header['SSRGUNIT'] = 'original' if restore else 'scaled'
        header.add_history('Starstack range: ' + ('restored original units' if restore else 'common power-of-two scale'))
        hdus.flush()


def _output_check(path, expected, shape, order, check, version=VERSION_V1):
    if version != VERSION_V1:
        return _output_check_finite(path, expected, shape, order, check)
    upper = 0
    minimum, maximum = float('inf'), 0.0
    with image(path) as (hdus, a):
        if a.shape != shape or hdus[0].header.get('ROWORDER', 'BOTTOM-UP') != order:
            raise ValueError('范围保护输出几何或行序不一致')
        for y, tile in _tiles(a, check):
            valid = expected[y:y + tile.shape[1]]
            if not np.isfinite(tile).all() or np.any(tile[:, ~valid] != 0):
                raise ValueError('范围保护输出非有限，或缺数据位置出现数值')
            if np.any(tile[:, valid] <= 0):
                raise ValueError('有输入覆盖的位置输出为零或负值，无法证明下限未裁切；请保留本作业供诊断')
            if valid.any():
                vals = tile[:, valid]
                upper += int(np.count_nonzero(vals >= 1))
                minimum = min(minimum, float(vals.min()))
                maximum = max(maximum, float(vals.max()))
    return {'minimum_positive': minimum, 'maximum': maximum, 'upper_endpoint_channels': upper}


def _output_check_finite(path, expected, shape, order, check):
    """Contract 1.1.0: a covered zero is Siril's low clip (counted); negatives fail."""
    upper, lower = 0, 0
    minimum, maximum = float('inf'), 0.0
    with image(path) as (hdus, a):
        if a.shape != shape or hdus[0].header.get('ROWORDER', 'BOTTOM-UP') != order:
            raise ValueError('范围保护输出几何或行序不一致')
        for y, tile in _tiles(a, check):
            valid = expected[y:y + tile.shape[1]]
            if not np.isfinite(tile).all() or np.any(tile[:, ~valid] != 0):
                raise ValueError('范围保护输出非有限，或缺数据位置出现数值')
            if valid.any():
                vals = tile[:, valid]
                if np.any(vals < 0):
                    raise ValueError('有输入覆盖的位置输出为负值，堆栈引擎结果异常；请保留本作业供诊断')
                upper += int(np.count_nonzero(vals >= 1))
                lower += int(np.count_nonzero(vals == 0))
                minimum = min(minimum, float(vals.min()))
                maximum = max(maximum, float(vals.max()))
    return {'minimum': minimum, 'maximum': maximum, 'upper_endpoint_channels': upper,
            'lower_endpoint_channels': lower}


def run_protected_stack(engine, directory, work, check, runner, contract=None):
    """Run under the job's saved contract; ``None`` means the contract for new jobs."""
    directory, work = Path(directory).resolve(), Path(work).resolve()
    if work != directory / 'work':
        raise ValueError('范围保护仅写入本作业工作目录')
    saved = _job_contract(contract)
    version = saved['version']
    require_engine(engine)
    attempt = directory / ('range-' + uuid.uuid4().hex[:12])
    attempt.mkdir()
    record = {'state': 'running', 'contract': saved, 'attempt': attempt.name, 'trials': []}
    receipt = attempt / 'receipt.json'
    atomic_json(receipt, record)
    start = time.monotonic()
    try:
        seq, sources, expected, shape, order, divisor = _sources(work, check, version)
        seq_path = work / 'r_light_.seq'
        sequence_sha = sha(seq_path)
        record.update(sources=sources, sequence_sha256=sequence_sha, shape_cyx=list(shape), roworder=order,
                      reference=seq['images'][seq['header']['reference_image']]['stem'],
                      selected=[f['stem'] for f in seq['images'] if f['selected']],
                      input_presence_semantics=('at least one selected nonzero RGB sample; not survivor count'
                                                if version == VERSION_V1 else
                                                'at least one selected sample that is not common RGB zero; '
                                                'not survivor count'))
        np.save(attempt / VALID, expected, allow_pickle=False)
        for index in range(saved['max_attempts']):
            check()
            trial = attempt / f'trial{index + 1}'
            trial.mkdir()
            detail = {'divisor': divisor, 'directory': trial.name, 'state': 'encoding'}
            record['trials'].append(detail)
            atomic_json(receipt, record)
            for source in sources:
                _write_scaled(work / source['file'], trial / source['file'], divisor, False, check, version)
            shutil.copyfile(seq_path, trial / 'r_light_.seq')
            commands = ['setext fit', 'set32bits', 'setcompress 0', 'seqclean r_light -stat', STACK]
            detail['state'] = 'stacking'
            atomic_json(receipt, record)
            runner(engine['executable'], trial, 'stack', commands, check)
            summary = _output_check(trial / 'encoded_master.fit', expected, shape, order, check, version)
            after_seq = read_seq(trial / 'r_light_.seq')
            if ([(f['stem'], f['selected']) for f in after_seq['images']]
                    != [(f['stem'], f['selected']) for f in seq['images']]
                    or after_seq['header']['reference_image'] != seq['header']['reference_image']
                    or fits.getheader(trial / 'encoded_master.fit').get('STACKCNT') != len(record['selected'])):
                raise ValueError('范围保护堆栈改变了选片、参考或参与数量')
            detail.update(summary, state='checked')
            if summary['upper_endpoint_channels']:
                detail['state'] = 'upper_limit_retry'
                divisor *= 2
                atomic_json(receipt, record)
                continue
            _write_scaled(trial / 'encoded_master.fit', attempt / 'decoded.fit', divisor, True, check, version)
            # Frozen sources remain in original units for all later consumers.
            if hashing.first_mismatch([work / source['file'] for source in sources],
                                      [source['sha256'] for source in sources], source=SOURCE, check=check) is not None:
                raise ValueError('范围保护期间注册帧发生变化')
            if sha(seq_path) != sequence_sha:
                raise ValueError('范围保护期间原序列发生变化')
            if version != VERSION_V1:
                record['lower_endpoint_channels'] = summary['lower_endpoint_channels']
            record.update(state='decoded', divisor=divisor,
                          decoded_sha256=sha(attempt / 'decoded.fit'), seconds=time.monotonic() - start)
            atomic_json(receipt, record)
            # Copies are committed only after validation, cancellation and identity checks.
            check()
            staging = work / ('range-master-' + uuid.uuid4().hex + '.tmp')
            shutil.copyfile(attempt / 'decoded.fit', staging)
            check()
            with hashing.writing(work / VALID):
                shutil.copyfile(attempt / VALID, work / VALID)
            # A real read-back of the copy just written ("fresh"): sealed as written when it would be re-read
            # as racy later, guarded against a change between read and seal (P1b-5b).
            record['valid_sha256'] = hashing.sha256_file(work / VALID, source=SOURCE, fresh=True)
            atomic_json(work / RECORD, record)
            check()
            with hashing.writing(work / 'sky_master.fit'):
                os.replace(staging, work / 'sky_master.fit')
            return record
        raise ValueError('范围保护增加尺度后仍触及上限，已停止发布')
    except BaseException as exc:
        record.update(state='cancelled' if isinstance(exc, Cancelled) else 'failed',
                      error=str(exc), seconds=time.monotonic() - start)
        atomic_json(receipt, record)
        raise


def _job_contract(value):
    return contract() if value is None else known_contract(value)


def _receipt_matches(record, saved):
    try:
        return known_contract(record.get('contract')) == saved
    except ValueError:
        return False


def finalize_receipt(work, contract=None):
    """Complete a decoded receipt. ``contract`` is the job's saved one; ``None``
    accepts the known contract the receipt itself was written under."""
    work = Path(work)
    record = json.loads((work / RECORD).read_text(encoding='utf-8'))
    if record.get('contract', {}).get('mode') == 'finite_validity_v1':
        from . import finite_stack
        return finite_stack.finalize_receipt(work)
    try:
        saved = known_contract(record.get('contract') if contract is None else contract)
    except ValueError:
        raise ValueError('范围保护未完成解码') from None
    if record.get('state') != 'decoded' or not _receipt_matches(record, saved):
        raise ValueError('范围保护未完成解码')
    names = [name for name in ('sky_master.fit', 'sky_master_undistorted.fit') if (work / name).is_file()]
    record['masters'] = dict(zip(names, _sha_many([work / name for name in names])))
    record['state'] = 'complete'
    atomic_json(work / RECORD, record)
    return record


def verify_receipt(work, job, *, check=lambda: None):
    if job.get('stack_range_mode') == 'finite_validity_v1':
        from . import finite_stack
        return finite_stack.verify_receipt(work, job, check=check)
    work = Path(work)
    if not verify_job(job):
        if (work / RECORD).exists():
            raise ValueError('作业模式与已有范围保护收据不一致')
        return None
    bound = [f for f in job.get('checkpoint_files', {}).get('stack', [])
             if Path(f['path']).as_posix() == 'work/' + RECORD]
    if len(bound) != 1 or not (work / RECORD).is_file() or sha(work / RECORD) != bound[0]['sha256']:
        raise ValueError('范围保护收据与已提交检查点不一致')
    record = json.loads((work / RECORD).read_text(encoding='utf-8'))
    if (record.get('state') != 'complete' or not _receipt_matches(record, known_contract(job['stack_range_contract']))
            or 'sky_master.fit' not in record.get('masters', {})):
        raise ValueError('范围保护收据缺失、未完成或版本不一致')
    records = [(name, digest) for name, digest in record['masters'].items()]
    records += [(f['file'], f['sha256']) for f in record['sources']]
    records += [('r_light_.seq', record['sequence_sha256']), (VALID, record['valid_sha256'])]
    # Serial reference: for each (name, digest): the name check, check(), then a missing file or
    # another digest raises; the files are hashed in parallel, the first failure in order is raised.
    hashing.verify_entries(records, source=SOURCE, check=check, validate=_valid_name,
                           error=lambda entry: ValueError('范围保护的母版、支持或注册源文件已改变，请新建作业'),
                           path=lambda entry: work / entry[0], precheck=lambda entry, target: target.is_file(),
                           expected=lambda entry: entry[1])
    return record


def _valid_name(entry):
    if Path(entry[0]).name != entry[0]:
        raise ValueError('范围保护收据含无效文件名')
