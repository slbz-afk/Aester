"""Explicit-validity stack boundary for the separately pinned finite Siril backend.

Geometry remains owned by the original certified engine. Only task-local copies
sent to this stack backend carry NaN for independently known missing coverage.
Finite zeros/negative values are never relabelled, offset or clipped.

Efficiency P1c F-3 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.3): run() checks its inputs (F-3b,
``finite_inventory``) and writes the q_ copies (F-3a, ``finite_write``) on frame pools, the q_ copies with one
sequential read and one sequential write each (the streamed body) instead of a copy rewritten in place (the
reference body, kept: AESTER_FINITE_WRITE=reference, and every frame the streamed body cannot plan). This module is in
the frozen holdout closure (rule R22): it decides no worker count or memory budget and imports neither performance nor
resources. core hands run() a ``Pools`` that does; without one both loops run serially in the calling thread with the
reference body, the pre-P1c code. Outputs are byte for byte the same either way.
"""
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple
import json
import os
import re
import shutil
import time
import traceback
import uuid

import numpy as np
from astropy.io import fits

from . import frame_pool, hashing
from .records import atomic_json
from .siril import Cancelled
from .siril_sequence import read_seq

MODE = 'finite_validity_v1'
VERSION = '1.0.0'
ENGINE_SHA256 = '67681ac2eb7fb31074072c9fb059d301b2ae0a0b880d682936a8fd35553635cf'
# runtime_manifest.json pins the engine and its PE import closure (75 files). These two hashes are in contract(),
# so in every F job's signature, and they do not change when helper files are added. GLib starts two more programs
# by path from its own folder, which no import closure finds: gspawn-win64-helper.exe and -console.exe (glib2
# 2.90.0-1, the package of the runtime's libglib-2.0-0.dll; P1b W7). runtime_aux_manifest.json lists them,
# RUNTIME_AUX_MANIFEST_SHA256 pins it (finite_backend_assets holds the same value) and describe_engine checks it
# with the runtime, but neither contract() nor describe_engine()'s result includes it: adding the helpers changes
# no F job. A runtime without the aux manifest stays valid; any program GLib would start by name must be listed in
# a verified manifest. gdbus.exe is deliberately absent (session-bus hazard SF-2). Aester starts this runtime only
# through siril.run_script, whose per-run environment keeps Siril out of the user's data folder and session bus
# (S-4b, S-4d), so the helpers matter only when the runtime is started outside Aester.
RUNTIME_MANIFEST_SHA256 = '2c732c1e45a7c4490fca116808bdfee1aa3821bf7801a0d931ef42b35f2cb2de'
RUNTIME_AUX_MANIFEST = 'runtime_aux_manifest.json'
RUNTIME_AUX_MANIFEST_SHA256 = '7116a60af17aea36bc3b11ce81caf040b552d71d4ee5905bbabbdab17f4ed941'
GLIB_STARTED = re.compile(r'(gspawn-win(32|64)-helper(-console)?|gdbus)\.exe', re.IGNORECASE)
RECORD = 'stack_range.json'
VALID = 'stack_range_valid.npy'
NVALID = 'sky_master_nvalid.fit'
NUSED = 'sky_master_nused.fit'
STACK = 'stack q rej w 3 3 -norm=addscale -32b -signed_output -nan_missing_v1 -out=finite_master.fit'


SOURCE = 'finite_stack.sha'
LONGEST_ENGINE_NAME = 'finite_master_nvalid.fit'   # the longest file the engine writes in an attempt (run())
STREAMED, REFERENCE = 'streamed', 'reference'      # the two q_ write bodies (P1c F-3a)
STREAM_CHUNK = 8 << 20                             # bytes per read or write call of the streamed body; a probe each
STREAM_SYNC = 32 << 20                             # the streamed body fsyncs its q_ after every this many bytes
_DATA_DTYPE = np.dtype('>f4')                      # a BITPIX -32 data unit as image() maps it


def sha(path, check=None):
    return hashing.sha256_file(Path(path), source=SOURCE, check=check)


class Plan(NamedTuple):
    """One frame pool's plan (P1c F-3): its worker count, ``unit_bytes`` (rules R10/R17), the q_ write body
    (finite_write only; the reference body always runs on one worker) and the in-flight bound (None: run_ordered's
    default)."""
    workers: int = 1
    unit_bytes: int | None = None
    body: str = REFERENCE
    inflight: int | None = None


class Pools:
    """How run() fans out its two per-frame loops (P1c F-3b inventory, F-3a q_ writes). This base is the pre-P1c code:
    both loops serial, the reference write body, no timer operation, no notes. core._FinitePools decides the worker
    counts and the body (performance.frame_workers), opens the timer operations and notes each pool run under timings
    (performance.note_pool; resources.shadow_pool for the governor), diagnostics only (rule R19)."""

    def inventory(self, *, units, pixels):
        """The inventory pool's Plan: ``units`` frames of ``pixels`` pixels each (None: unknown)."""
        return Plan()

    def write(self, *, units, file_bytes, pixels, width):
        """The q_ write pool's Plan: ``units`` frames; ``file_bytes`` the largest source file, ``pixels`` and
        ``width`` the frame's (each None: unknown; pixels and width are None together)."""
        return Plan()

    def operation(self, name):
        """The timer operation a pool runs in."""
        return nullcontext()

    def note_pool(self, site, stats, **extra):
        """The timings note of one pool run: run_ordered's ``stats`` and the site's own figures (``extra``)."""

    def shadow_pool(self, footprint, *, actual, items, frame_pixels):
        """The governor's shadow decision for one pool run (``actual`` workers)."""


def longest_engine_path(job_dir):
    """The longest path the finite engine writes for a job in ``job_dir`` (admission check S-8). run() names its
    attempt 'finite-' + 12 hex digits inside Path(directory).resolve(); the placeholder has that length."""
    return Path(job_dir).resolve() / ('finite-' + '0' * 12) / LONGEST_ENGINE_NAME


def _sha_many(paths):
    """[sha(p) for p in paths], hashed in parallel; fails like that loop (lowest failing index)."""
    return hashing.sha256_many([Path(p) for p in paths], source=SOURCE)


def contract():
    return {'mode': MODE, 'version': VERSION, 'engine_sha256': ENGINE_SHA256,
            'runtime_manifest_sha256': RUNTIME_MANIFEST_SHA256,
            'backend': 'siril_finite_explicit_validity', 'max_attempts': 1,
            'input_absolute_limit': 8.0, 'stack_command': STACK,
            'zero_semantics': 'all_finite_samples_valid; missing_from_independent_geometry; nused_defines_output_presence'}


def describe_engine(executable):
    if not executable:
        raise ValueError('有限值堆栈需要明确指定已验证的独立堆栈引擎路径')
    path = Path(executable).expanduser().resolve()
    manifest_path = path.parent / 'runtime_manifest.json'
    if (not path.is_file() or path.name != 'siril-cli.exe' or sha(path) != ENGINE_SHA256
            or not manifest_path.is_file() or sha(manifest_path) != RUNTIME_MANIFEST_SHA256):
        raise ValueError('有限值堆栈引擎或运行时清单未验证；请使用已核验的完整运行时目录')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = manifest.get('files', {})
    if manifest.get('executable') != path.name or files.get(path.name) != ENGINE_SHA256:
        raise ValueError('有限值堆栈运行时清单缺失引擎身份')
    runtime = path.parent
    # Serial reference: for each (name, digest), a bad name, a missing file or another digest
    # raises for that name.  The runtime files are hashed in parallel.
    hashing.verify_entries(
        list(files.items()), source=SOURCE,
        error=lambda entry: ValueError('有限值堆栈运行时文件缺失或改变：' + str(entry[0])),
        path=lambda entry: runtime / entry[0] if isinstance(entry[0], str) else None,
        precheck=lambda entry, target: (target is not None and Path(entry[0]).name == entry[0]
                                        and target.is_file()),
        expected=lambda entry: entry[1])
    _verify_runtime_aux(runtime, files)
    return {'executable': str(path), 'sha256': ENGINE_SHA256,
            'runtime_manifest_sha256': RUNTIME_MANIFEST_SHA256, 'version': VERSION}


def _verify_runtime_aux(runtime, listed):
    """runtime_aux_manifest.json is absent, or it is the pinned file and each file it lists is present and
    unchanged; and every program GLib would start by name from the runtime folder is listed in a verified
    manifest. Mirrors finite_backend_assets._validate_aux (this module must not import that one)."""
    aux_path = runtime / RUNTIME_AUX_MANIFEST
    aux = {}
    if os.path.lexists(aux_path):
        if not aux_path.is_file() or sha(aux_path) != RUNTIME_AUX_MANIFEST_SHA256:
            raise ValueError('有限值堆栈引擎或运行时清单未验证；请使用已核验的完整运行时目录')
        value = json.loads(aux_path.read_text(encoding='utf-8'))
        aux = value.get('files') if isinstance(value, dict) else None
        if not isinstance(aux, dict) or not aux:
            raise ValueError('有限值堆栈引擎或运行时清单未验证；请使用已核验的完整运行时目录')
        hashing.verify_entries(
            list(aux.items()), source=SOURCE,
            error=lambda entry: ValueError('有限值堆栈运行时文件缺失或改变：' + str(entry[0])),
            path=lambda entry: runtime / entry[0] if isinstance(entry[0], str) else None,
            precheck=lambda entry, target: (target is not None and Path(entry[0]).name == entry[0]
                                            and entry[0] not in listed
                                            and entry[0] not in ('runtime_manifest.json', RUNTIME_AUX_MANIFEST)
                                            and target.is_file()),
            expected=lambda entry: entry[1])
    covered = {name.lower() for name in (*listed, *aux)}
    for name in os.listdir(runtime):
        if GLIB_STARTED.fullmatch(name) and name.lower() not in covered:
            raise ValueError('有限值堆栈运行时文件缺失或改变：' + str(name))


def verify_engine(engine):
    if not isinstance(engine, dict) or describe_engine(engine.get('executable')) != engine:
        raise ValueError('有限值堆栈运行时身份与作业不一致')


def verify_job(job):
    from . import stack_range as sr, registered_support as rs
    if (job.get('schema_version') != 2 or job.get('stack_range_mode') != MODE
            or job.get('stack_range_contract') != contract()
            or job.get('stack_range_sha256') != sr.fingerprint(contract())
            or job.get('support_mode') != rs.MODE or not job.get('mask')):
        raise ValueError('有限值堆栈配方或独立几何支持缺失，请新建作业')
    rs.require_engine(job['engine'])
    verify_engine(job.get('stack_engine'))
    return True


def _frame_pixels(shape):
    """H x W of a support record's [H, W], read without raising (a bad record fails later, in its own place); None when
    it is not two positive integers."""
    if isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape):
        return shape[0] * shape[1]
    return None


def _source_inventory(work, check, pools=None):
    from . import registered_support as rs
    pools = Pools() if pools is None else pools
    seq = read_seq(work / 'r_light_.seq')
    h = seq['header']; frames = seq['images']
    selected = [f for f in frames if f['selected']]
    if (h['name'] != 'r_light_' or h['version'] != 6 or h['variable_size'] or h['fz_flag']
            or h['drizzle'] or seq['layers'] != 3 or not 3 <= len(selected) <= 65535
            or h['nb_selected'] != len(selected) or not frames[h['reference_image']]['selected']
            or not seq['all_identity_per_layer'] or not all(seq['all_identity_per_layer'].values())):
        raise ValueError('有限值堆栈要求已完成几何变换的RGB序列及有效参考')
    support = json.loads((work / rs.RECORD).read_text(encoding='utf-8'))
    if (support.get('state') != 'exports_filtered' or support.get('version') != rs.SUPPORT_VERSION
            or support.get('mode') != rs.MODE or support.get('sequence_sha256') != sha(work / rs.EXPORT_SEQUENCE)):
        raise ValueError('有限值堆栈独立支持记录缺失或改变')
    mapping = {f['stem']: f for f in support['frames']}
    if len(mapping) != len(support['frames']) or set(mapping) != {f['stem'][2:] for f in selected}:
        raise ValueError('独立支持与实际选片不一致')
    if {p.stem for p in work.glob('r_light_*.fit')} != {f['stem'] for f in frames}:
        raise ValueError('注册帧与序列文件集合不一致')

    def inventory(f, unit):
        # P1c F-3b: the serial loop's body after its check() (run_ordered calls check() before each frame), one frame
        # per unit; the entries come back in frame order. A worker probes the export's hash every 8 MiB (rule R18);
        # the serial path hashes as before, without a probe.
        if not re.fullmatch(r'r_light_\d+', f['stem']):
            raise ValueError('无效注册帧名称')
        item = mapping[f['stem'][2:]]
        name = f['stem'] + '.fit'
        if item['export_after_sha256'] != sha(work / name, check=unit.check if frame_pool.in_pool() else None):
            raise ValueError('注册帧在几何支持检查后改变')
        # The descriptor is independent of RGB values, retained by prepare_exports.
        valid = rs.load_frame_support(work, item.get('explicit_validity'))
        if np.count_nonzero(valid) != item['supported_pixels']:   # int(valid.sum()) before P1c: equal for a bool map
            raise ValueError('逐帧有效性数量与记录不一致')
        return {'file': name, 'sha256': item['export_after_sha256'],
                'validity': item['explicit_validity'], 'supported_pixels': item['supported_pixels']}

    pixels = _frame_pixels(support.get('shape'))
    plan = pools.inventory(units=len(selected), pixels=pixels)
    pool_stats = {}
    with pools.operation('finite_inventory'):
        items = frame_pool.run_ordered(inventory, selected, workers=plan.workers, name='finite-inventory', check=check,
                                       inflight=plan.inflight, stats=pool_stats, unit_bytes=plan.unit_bytes,
                                       blas=False)
    pools.note_pool('finite_inventory', pool_stats)
    pools.shadow_pool('py.finite_inventory', actual=pool_stats.get('workers'), items=len(selected),
                      frame_pixels=pixels)
    eligible = rs.load_array(work, support['eligible'])
    if eligible.dtype != np.uint16 or list(eligible.shape) != support['shape']:
        raise ValueError('支持计数类型或形状不一致')
    reference = frames[h['reference_image']]['stem'] + '.fit'
    return items, reference, support['eligible'], eligible


def _finite_inventory_reference(work, check):
    """The pre-P1c _source_inventory, statement for statement (test oracle of the finite_inventory pool, P1c A5)."""
    from . import registered_support as rs
    seq = read_seq(work / 'r_light_.seq')
    h = seq['header']; frames = seq['images']
    selected = [f for f in frames if f['selected']]
    if (h['name'] != 'r_light_' or h['version'] != 6 or h['variable_size'] or h['fz_flag']
            or h['drizzle'] or seq['layers'] != 3 or not 3 <= len(selected) <= 65535
            or h['nb_selected'] != len(selected) or not frames[h['reference_image']]['selected']
            or not seq['all_identity_per_layer'] or not all(seq['all_identity_per_layer'].values())):
        raise ValueError('有限值堆栈要求已完成几何变换的RGB序列及有效参考')
    support = json.loads((work / rs.RECORD).read_text(encoding='utf-8'))
    if (support.get('state') != 'exports_filtered' or support.get('version') != rs.SUPPORT_VERSION
            or support.get('mode') != rs.MODE or support.get('sequence_sha256') != sha(work / rs.EXPORT_SEQUENCE)):
        raise ValueError('有限值堆栈独立支持记录缺失或改变')
    mapping = {f['stem']: f for f in support['frames']}
    if len(mapping) != len(support['frames']) or set(mapping) != {f['stem'][2:] for f in selected}:
        raise ValueError('独立支持与实际选片不一致')
    if {p.stem for p in work.glob('r_light_*.fit')} != {f['stem'] for f in frames}:
        raise ValueError('注册帧与序列文件集合不一致')
    items = []
    for f in selected:
        check()
        if not re.fullmatch(r'r_light_\d+', f['stem']):
            raise ValueError('无效注册帧名称')
        item = mapping[f['stem'][2:]]
        name = f['stem'] + '.fit'
        if item['export_after_sha256'] != sha(work / name):
            raise ValueError('注册帧在几何支持检查后改变')
        # The descriptor is independent of RGB values, retained by prepare_exports.
        valid = rs.load_frame_support(work, item.get('explicit_validity'))
        if int(valid.sum()) != item['supported_pixels']:
            raise ValueError('逐帧有效性数量与记录不一致')
        items.append({'file': name, 'sha256': item['export_after_sha256'],
                      'validity': item['explicit_validity'], 'supported_pixels': item['supported_pixels']})
    eligible = rs.load_array(work, support['eligible'])
    if eligible.dtype != np.uint16 or list(eligible.shape) != support['shape']:
        raise ValueError('支持计数类型或形状不一致')
    reference = frames[h['reference_image']]['stem'] + '.fit'
    return items, reference, support['eligible'], eligible


def _largest_file(paths):
    """The largest size among ``paths``, read without raising (P1c F-3a: a file that cannot be stat'ed is left out, and
    its frame fails in its own unit with the serial loop's error); None when no size is known."""
    sizes = []
    for path in paths:
        try:
            sizes.append(os.stat(path).st_size)
        except (OSError, ValueError):
            pass
    return max(sizes, default=None)


def _write_inputs(work, attempt, sources, reference, expected, check, pools=None):
    """``attempt``/q_00001.fit, ... for ``sources`` in order, each frame's support map reloaded and summed (P1c F-3a);
    returns the reference frame's 1-based position. Raises when the sum is not the recorded eligible count ``expected``.

    One frame per unit; the support sums, the source details and the reference position are taken in frame order
    (on_result, rule R3). The streamed body runs on the planned workers, each reusing its read buffer (R12); the
    reference body always runs on one. A unit calls no BLAS or LAPACK routine (R8: element-wise numpy, file I/O and
    astropy header calls), hence ``blas=False``.

    The serial loop added each map to its sum before writing that frame. A unit whose map has another shape than
    ``expected`` repeats that addition on a scratch sum first, so a map that cannot be added fails with the serial
    loop's numpy error before its q_ exists; one that broadcasts goes on, as there. An ``expected`` that is not one
    frame (two dimensions) is planned as unknown (the reference body, one worker), never read for its width."""
    from . import registered_support as rs
    pools = Pools() if pools is None else pools
    accumulated = np.zeros_like(expected)
    ref = None
    fallbacks = {}
    frame = expected.ndim == 2
    plan = pools.write(units=len(sources), file_bytes=_largest_file([work / s['file'] for s in sources]),
                       pixels=int(expected.size) if frame else None, width=int(expected.shape[1]) if frame else None)
    streamed = plan.body == STREAMED

    def write(source, unit):
        mask = rs.load_frame_support(work, source['validity'])
        try:
            if mask.shape != expected.shape:               # the serial loop's `accumulated += mask`, before the write
                scratch = np.zeros_like(expected)
                scratch += mask
                scratch = None
            destination = attempt / f'q_{unit.index + 1:05d}.fit'
            if streamed:
                details, fallback = _write_input_streamed(work / source['file'], destination, mask, unit.check,
                                                          unit.state)
            else:
                details, fallback = _write_input(work / source['file'], destination, mask, unit.check), None
            return mask, details, fallback
        finally:
            mask = None                                    # a failure's traceback does not keep the map

    def deliver(index, result):
        nonlocal accumulated, ref
        mask, details, fallback = result
        accumulated += mask
        sources[index].update(details)
        if sources[index]['file'] == reference:
            ref = index + 1
        if fallback is not None:
            fallbacks[fallback] = fallbacks.get(fallback, 0) + 1

    pool_stats = {}
    with pools.operation('finite_write'):
        frame_pool.run_ordered(write, sources, workers=plan.workers if streamed else 1, name='finite-write',
                               check=check, on_result=deliver, init=_ReadBuffer if streamed else None,
                               finalize=_ReadBuffer.release if streamed else None, inflight=plan.inflight,
                               stats=pool_stats, unit_bytes=plan.unit_bytes, blas=False)
    pools.note_pool('finite_write', pool_stats, body=plan.body, fallbacks=fallbacks)
    pools.shadow_pool('py.finite_write', actual=pool_stats.get('workers'), items=len(sources),
                      frame_pixels=int(expected.size))
    if not np.array_equal(accumulated, expected):
        raise ValueError('逐帧独立有效性之和与已记录覆盖数量不一致')
    return ref


def _finite_write_reference(work, attempt, sources, reference, expected, check):
    """The pre-P1c loop of run(), statement for statement (test oracle of the finite_write pool, P1c A5)."""
    from . import registered_support as rs
    accumulated = np.zeros_like(expected)
    for number, source in enumerate(sources, 1):
        check()
        mask = rs.load_frame_support(work, source['validity'])
        accumulated += mask
        details = _write_input(work / source['file'], attempt / f'q_{number:05d}.fit', mask, check)
        source.update(details)
        if source['file'] == reference:
            ref = number
    if not np.array_equal(accumulated, expected):
        raise ValueError('逐帧独立有效性之和与已记录覆盖数量不一致')
    del accumulated, mask
    return ref


def _write_input(source, destination, valid, check):
    with hashing.writing(destination):   # a copy, then rewritten in place (update mode)
        return _write_input_in_place(source, destination, valid, check)


def _write_input_in_place(source, destination, valid, check):
    """The reference body: a copy, rewritten in place (update mode); the pre-P1c statements, in three helpers."""
    from .stack_range import image
    shutil.copyfile(source, destination)
    with image(destination, 'update') as (hdus, array):
        _check_geometry(hdus, array, valid)
        low, high, zeros = _mask_tiles(array, valid, check)
        _finish_header(hdus, low, high)
    return {'minimum': low, 'maximum': high, 'valid_zero_channels': zeros}


def _check_geometry(hdus, array, valid):
    if hdus[0].header.get('ROWORDER') != 'BOTTOM-UP' or array.shape[1:] != valid.shape:
        raise ValueError('有限值堆栈需要明确BOTTOM-UP且与有效性相同画幅的输入')


def _mask_tiles(array, valid, check):
    """Missing samples (``valid`` false) of ``array`` set to NaN in 64-row tiles, ``check()`` before each; returns the
    valid samples' (minimum, maximum, zero count). A ValueError leaves the tiles before the failing one rewritten."""
    from .stack_range import _tiles
    low, high, zeros = float('inf'), -float('inf'), 0
    ordered = valid[::-1]
    for y, tile in _tiles(array, check):
        support = ordered[y:y + tile.shape[1]]
        if not np.isfinite(tile).all():
            raise ValueError('配准源包含非有限值，不能推断其有效性')
        values = tile[:, support]
        if values.size:
            if np.any(np.abs(values) >= 8):
                raise ValueError('有限值堆栈输入超出已验证范围 (-8,8)；未裁切')
            low, high = min(low, float(values.min())), max(high, float(values.max()))
            zeros += int(np.count_nonzero(values == 0))
        tile[:, ~support] = np.nan
        array[:, y:y + tile.shape[1]] = tile
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError('所选帧没有有效样本')
    return low, high, zeros


def _finish_header(hdus, low, high):
    header = hdus[0].header
    for key in ('CHECKSUM', 'DATASUM'):
        header.pop(key, None)
    header['DATAMIN'], header['DATAMAX'] = low, high
    header['SSFINVER'] = VERSION
    header.add_history('Independent geometry mask: NaN missing; all finite values retained')
    hdus.flush()


class _ReadBuffer:
    """A worker's reusable read buffer for the streamed body (frame_pool state). Rule R12: a unit reads its whole
    source into it, and checks that it did, before it uses a byte, so nothing of an earlier frame reaches a q_."""
    __slots__ = ('array',)

    def __init__(self):
        self.array = None

    def get(self, size):
        if self.array is None or self.array.size != size:
            self.array = None                              # the old one goes first: never two at once
            self.array = np.empty(size, dtype=np.uint8)
        return self.array

    def release(self):
        self.array = None


def _identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _stream_plan(source, valid):
    """(file size, data offset, data bytes, data shape, file identity) of ``source``: integers only, read through a
    read-only mapping that charges no commit ('denywrite'; astropy's default 'readonly' maps copy-on-write) and is gone
    before the caller takes a buffer. It applies the checks the reference body applies to its copy, and
    hdus.verify('exception') as that body's update-mode close does; the caller sends any failure to the reference
    body, which raises it with its own bytes. Calls no check() or probe."""
    from .stack_range import image
    before = _identity(os.stat(source))
    with image(source, 'denywrite') as (hdus, array):
        hdus.verify('exception')
        _check_geometry(hdus, array, valid)
        if array.dtype != _DATA_DTYPE:
            raise ValueError('not a big-endian float32 data unit')
        offset, nbytes, shape = int(hdus.fileinfo(0)['datLoc']), int(array.nbytes), tuple(int(n) for n in array.shape)
    after = os.stat(source)
    if _identity(after) != before or offset + nbytes > after.st_size:
        raise ValueError('source changed while it was planned')
    return after.st_size, offset, nbytes, shape, before


def _read_into(reader, view, check):
    """Fill ``view`` from ``reader`` in STREAM_CHUNK pieces, ``check()`` before each; the bytes read (fewer at EOF)."""
    got, size = 0, len(view)
    while got < size:
        check()
        count = reader.readinto(view[got:got + STREAM_CHUNK])
        if not count:
            break
        got += count
    return got


def _write_all(writer, view, check, sync=False):
    """Write all of ``view`` in STREAM_CHUNK pieces, ``check()`` before each unless it is None. ``sync``: os.fsync
    after every STREAM_SYNC bytes and at the end, so that no flush waits on more than STREAM_SYNC of dirty data (a
    cancel is seen within one such flush, rule R18) and the system never holds a whole q_ of modified pages."""
    done, size, synced = 0, len(view), 0
    while done < size:
        if check is not None:
            check()
        count = writer.write(view[done:done + STREAM_CHUNK])
        if not count:
            raise OSError(f'short write to {writer.name}')
        done += count
        if sync and done - synced >= STREAM_SYNC:
            os.fsync(writer.fileno())
            synced = done
    if sync and synced < done:
        os.fsync(writer.fileno())


def _write_input_streamed(source, destination, valid, check, buffer):
    """The reference body's q_ (_write_input: a copy, rewritten in place) with one sequential read of ``source`` and
    one sequential write of ``destination`` (P1c F-3a). Returns (details, fallback), ``fallback`` naming why the
    reference body ran instead (None when streamed).

    The source, opened before the destination as copyfile does, is read whole into ``buffer`` (a _ReadBuffer); it must
    give exactly the planned size, then end, with the planned identity, or both files are closed and the reference body
    runs. The reference body's tile statements mask the data unit in the buffer (the same dtype, shape, strides and
    bytes as its memory map); a ValueError there first writes the buffer out, the bytes the reference body leaves. The
    buffer is written out and fsync'd, every STREAM_SYNC bytes and before the close (so the header update below finds
    no dirty page to flush, no gigabytes of modified pages wait for the finite Siril, which fails below ~4 GB
    available, and a cancel waits for at most one STREAM_SYNC flush), and the header changed in update mode with the
    reference body's astropy calls. A cancel, MemoryError or I/O error may leave the q_ empty or partial (the
    reference body leaves a whole copy; prereg §5 item 2): only in a failed attempt folder."""
    from .stack_range import image
    with hashing.writing(destination):   # written sequentially, then the header in place (update mode)
        try:
            planned, fallback = _stream_plan(source, valid), None
        except Cancelled:
            raise
        except Exception as exc:                           # no check() ran: a cancel cannot be swallowed here
            planned, fallback = None, 'plan:' + type(exc).__name__
        if planned is None:                                # outside the handler: no chained or kept plan error
            return _write_input_in_place(source, destination, valid, check), fallback
        size, offset, nbytes, shape, identity = planned
        array = view = data = None
        try:
            array = buffer.get(size)
            view = memoryview(array)
            with open(source, 'rb', buffering=0) as reader, open(destination, 'wb', buffering=0) as writer:
                got = _read_into(reader, view, check)
                if got != size or reader.read(1) or _identity(os.fstat(reader.fileno())) != identity:
                    fallback = 'read:short' if got != size else 'read:changed'
                else:
                    fallback = None
                    data = array[offset:offset + nbytes].view(_DATA_DTYPE).reshape(shape)
                    try:
                        low, high, zeros = _mask_tiles(data, valid, check)
                    except ValueError:
                        _write_all(writer, view, None)    # no probe: the error raised is the reference body's
                        raise
                    _write_all(writer, view, check, sync=True)
            if fallback is not None:
                return _write_input_in_place(source, destination, valid, check), fallback
            with image(destination, 'update') as (hdus, _):
                _finish_header(hdus, low, high)
            return {'minimum': low, 'maximum': high, 'valid_zero_channels': zeros}, None
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)      # the inner frames' views of the buffer and the map go now
            raise
        finally:
            array = view = data = None


def _counts(path, shape):
    # Count images are integers, never photographic RGB scaled by 65535.
    with fits.open(path, memmap=False) as hdus:
        a = hdus[0].data
        if (len(hdus) != 1 or a is None or a.dtype != np.uint16 or a.shape != (3, *shape)
                or hdus[0].header.get('ROWORDER') != 'BOTTOM-UP'):
            raise ValueError('有限值堆栈计数必须是同画幅BOTTOM-UP uint16三通道FITS')
        return a


def validate_outputs(stack_path, nvalid_path, nused_path, expected, selected_count, check):
    from .stack_range import image, _tiles
    nvalid, nused = _counts(nvalid_path, expected.shape), _counts(nused_path, expected.shape)
    valid = np.ones(expected.shape, dtype=bool)
    with image(stack_path) as (hdus, array):
        if (array.shape != (3, *expected.shape) or hdus[0].header.get('ROWORDER') != 'BOTTOM-UP'
                or hdus[0].header.get('STACKCNT') != selected_count):
            raise ValueError('有限值堆栈输出画幅、方向或实际参与数改变')
        for y, tile in _tiles(array, check):
            nv, nu = nvalid[:, y:y + tile.shape[1]], nused[:, y:y + tile.shape[1]]
            if (not np.all(nv == expected[::-1][None, y:y + tile.shape[1]])
                    or np.any(nu > nv) or not np.isfinite(tile).all() or np.any(tile[nu == 0] != 0)):
                raise ValueError('有限值堆栈图像／有效数／实际保留数不满足独立支持合同')
            valid[y:y + tile.shape[1]] = np.all(nu > 0, axis=0)
    return valid[::-1].copy(), {'valid_samples': int(nvalid.sum(dtype=np.uint64)),
                              'used_samples': int(nused.sum(dtype=np.uint64))}


def counted_support(work, stack_path, expected, check):
    """Use mandatory survivor counts, including a valid zero/negative output."""
    record = json.loads((Path(work) / RECORD).read_text(encoding='utf-8'))
    if record.get('contract') != contract() or record.get('state') not in ('decoded', 'complete'):
        raise ValueError('有限值计数没有匹配的堆栈收据')
    for name in (NVALID, NUSED):
        if sha(Path(work) / name) != record['counts'][name]:
            raise ValueError('有限值堆栈计数已改变')
    return validate_outputs(stack_path, Path(work) / NVALID, Path(work) / NUSED,
                            expected, len(record['selected']), check)[0]


def run(engine, directory, work, check, runner, *, pools=None):
    """``pools`` (a Pools; None: the base class, the pre-P1c serial code) plans the inventory and q_ write pools."""
    from . import registered_support as rs
    directory, work = Path(directory).resolve(), Path(work).resolve()
    if work != directory / 'work':
        raise ValueError('有限值堆栈仅可发布到本作业工作目录')
    verify_engine(engine)
    attempt = directory / ('finite-' + uuid.uuid4().hex[:12]); attempt.mkdir()
    record = {'state': 'running', 'contract': contract(), 'attempt': attempt.name, 'stack_engine': engine}
    receipt = attempt / 'receipt.json'; atomic_json(receipt, record)
    start = time.monotonic()
    try:
        sources, reference, eligible_descriptor, expected = _source_inventory(work, check, pools)
        record.update(sources=sources, sequence_sha256=sha(work / 'r_light_.seq'),
                      export_sequence_sha256=sha(work / rs.EXPORT_SEQUENCE), eligible=eligible_descriptor,
                      selected=[s['file'] for s in sources], reference=reference)
        ref = _write_inputs(work, attempt, sources, reference, expected, check, pools)
        commands = ['offline', 'setext fit', 'set32bits', 'setcompress 0',
                    'set gui.icc_autoassignment=0', 'set gui.icc_autoconversion=0', f'setref q {ref}', STACK]
        record.update(commands=commands, state='stacking'); atomic_json(receipt, record)
        check(); verify_engine(engine)
        runner(engine['executable'], attempt, 'stack', commands, check)
        actual_sequence = read_seq(attempt / 'q_.seq')
        actual_header = actual_sequence['header']
        if (actual_header['name'] != 'q_' or actual_header['reference_image'] != ref - 1
                or actual_header['nb_selected'] != len(sources)
                or [(f['stem'], f['selected']) for f in actual_sequence['images']]
                != [(f'q_{i:05d}', True) for i in range(1, len(sources) + 1)]):
            raise ValueError('有限值堆栈引擎改变了实际参考帧或选片')
        record['actual_reference_position'] = actual_header['reference_image']
        record['actual_sequence_sha256'] = sha(attempt / 'q_.seq')
        valid, stats = validate_outputs(attempt / 'finite_master.fit', attempt / 'finite_master_nvalid.fit',
                                       attempt / 'finite_master_nused.fit', expected, len(sources), check)
        np.save(attempt / VALID, valid, allow_pickle=False)
        # Outputs remain private until source, runtime and cancellation checks pass.
        _verify_sources(work, record, check)
        verify_engine(engine); check()
        copies = [('finite_master_nvalid.fit', NVALID), ('finite_master_nused.fit', NUSED), (VALID, VALID)]
        for source_name, final_name in copies:
            temporary = work / (final_name + '.pending')
            with hashing.writing(temporary):
                shutil.copyfile(attempt / source_name, temporary)
            check()
            with hashing.writing(work / final_name):
                os.replace(temporary, work / final_name)
        seconds = time.monotonic() - start
        # The copies' read-backs and the decoded master, hashed in parallel (same order as before).  The
        # read-backs are real reads of copies this worker just wrote ("fresh"): one that would be re-read as
        # racy on its next lookup is sealed as written, guarded against a change between read and seal (P1b-5b).
        nvalid_sha, nused_sha, valid_sha, decoded_sha = hashing.sha256_many(
            [work / NVALID, work / NUSED, work / VALID, attempt / 'finite_master.fit'], source=SOURCE,
            fresh=(True, True, True, False))
        record.update(state='decoded', seconds=seconds, counts_summary=stats,
                      counts={NVALID: nvalid_sha, NUSED: nused_sha},
                      valid_sha256=valid_sha, decoded_sha256=decoded_sha)
        atomic_json(receipt, record)
        temporary = work / 'finite-master.pending'
        with hashing.writing(temporary):
            shutil.copyfile(attempt / 'finite_master.fit', temporary)
        check(); atomic_json(work / RECORD, record)
        with hashing.writing(work / 'sky_master.fit'):
            os.replace(temporary, work / 'sky_master.fit')
        return record
    except BaseException as exc:
        record.update(state='cancelled' if isinstance(exc, Cancelled) else 'failed',
                      error=str(exc), seconds=time.monotonic() - start)
        atomic_json(receipt, record)
        raise


def _valid_name(entry):
    name = entry[0]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('有限值堆栈收据含无效文件名')


def _verify_named(work, files, check, message):
    """Serial reference: for each (name, digest): the name check, check(), then a missing file or
    another digest raises ``message``.  The files are hashed in parallel; the first failure in input
    order is raised."""
    hashing.verify_entries(files, source=SOURCE, check=check, validate=_valid_name,
                           error=lambda entry: ValueError(message),
                           path=lambda entry: work / entry[0], precheck=lambda entry, target: target.is_file(),
                           expected=lambda entry: entry[1])


def _verify_sources(work, record, check):
    from . import registered_support as rs
    files = [(s['file'], s['sha256']) for s in record['sources']]
    files += [(s['validity']['file'], s['validity']['sha256']) for s in record['sources']]
    files += [('r_light_.seq', record['sequence_sha256']), (rs.EXPORT_SEQUENCE, record['export_sequence_sha256']),
              (record['eligible']['file'], record['eligible']['sha256'])]
    _verify_named(work, files, check, '有限值堆栈源、序列或独立有效性已改变')


def finalize_receipt(work):
    work = Path(work)
    record = json.loads((work / RECORD).read_text(encoding='utf-8'))
    if record.get('state') != 'decoded' or record.get('contract') != contract():
        raise ValueError('有限值堆栈尚未验证完成')
    names = [name for name in ('sky_master.fit', 'sky_master_undistorted.fit') if (work / name).is_file()]
    record['masters'] = dict(zip(names, _sha_many([work / name for name in names])))
    record['state'] = 'complete'; atomic_json(work / RECORD, record)
    return record


def verify_receipt(work, job, *, check=lambda: None):
    verify_job(job); work = Path(work)
    bound = [f for f in job.get('checkpoint_files', {}).get('stack', []) if Path(f['path']).as_posix() == 'work/' + RECORD]
    if len(bound) != 1 or not (work / RECORD).is_file() or sha(work / RECORD) != bound[0]['sha256']:
        raise ValueError('有限值堆栈收据与已提交检查点不一致')
    record = json.loads((work / RECORD).read_text(encoding='utf-8'))
    if (record.get('state') != 'complete' or record.get('contract') != contract()
            or record.get('stack_engine') != job.get('stack_engine') or 'sky_master.fit' not in record.get('masters', {})):
        raise ValueError('有限值堆栈收据缺失、未完成或版本不一致')
    _verify_sources(work, record, check)
    items = [*record['masters'].items(), *record['counts'].items(), (VALID, record['valid_sha256'])]
    _verify_named(work, items, check, '有限值堆栈母版、计数或支持已改变')
    return record
