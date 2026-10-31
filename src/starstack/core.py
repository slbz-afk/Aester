"""Durable on-disk jobs shared by GUI and CLI; original sources are read-only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .records import atomic_json, read_json
from . import frame_pool
from . import hashing
from . import profiling
from . import stack_settings
from .mark_validation import validate_mark_values
from . import transient_revision
from .siril import (doctor, run_script, Cancelled, conversion_failure_message, engine_path_refusal,
                    job_stage_launches, siril_pin)
from .siril_sequence import read_seq, SequenceError
from .locking import file_lock, is_locked, pid_alive, Busy

RAW = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw"}
FITS = {".fit", ".fits", ".fts"}
TERMINAL = {"succeeded", "failed", "cancelled"}
SHA_SOURCE = "core._sha"
# Hash handover (docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md): a launcher that has this
# variable set (a fresh token per launch) while it calls submit_job/resume_job passes it to
# the job worker it spawns (the environment is inherited); that worker exports its ledger
# entries for the job directory to ``hashing.handover_path(token)`` just before it
# publishes "succeeded".  Same name as workflow.HANDOVER_TOKEN_ENV.
HANDOVER_TOKEN_ENV = "AESTER_HASH_HANDOVER_TOKEN"
# A workflow folder holding either marker had its cache cleaned: resume, review and recompose are refused.
# The uninstaller writes ``uninstall_options.CLEANUP_MARKER`` (this name; it removes masters and products), the
# in-app cleanup on 设置 → 作品与缓存 (``cache_cleanup``) writes its own file (it removes per-frame files). Separate
# files let each cleanup still free what the other one keeps.
CACHE_CLEANUP_MARKER = ".aester-cache-cleaned.json"
APP_CACHE_CLEANUP_MARKER = ".aester-app-cache-cleaned.json"
CACHE_CLEANUP_MARKERS = (APP_CACHE_CLEANUP_MARKER, CACHE_CLEANUP_MARKER)


def cache_cleanup_marked(workflow_dir: str | Path) -> bool:
    """Whether the in-app cleanup or the uninstaller started cleaning this workflow folder's cache."""
    return any((Path(workflow_dir) / name).exists() for name in CACHE_CLEANUP_MARKERS)


# Siril names the converted sequence "light_" (basename "light" plus its own
# underscore); the sequence file only appears once a Siril script loads it.
SEQUENCE_NAME = "light_"
SEQUENCE_FILE = f"{SEQUENCE_NAME}.seq"
SEQUENCE_SNAPSHOT = f"{SEQUENCE_NAME}.initial.seq"
# Registration (Siril 1.4.4, verified on the real Z8 16-frame job, RUN_LOG §14):
# one-pass ``setref light 1`` + ``register`` detected the reference at 16-bit
# and the other frames at 32-bit, so long-span frames lost pairs and smeared
# 1-2 px in the corners.  Two-pass detection is consistent across frames and
# lets Siril pick a mid-sequence reference; ``seqapplyreg`` then writes the
# ``r_`` frames (default prefix ``r_``, framing ``current`` = reference frame).
#
# Lens distortion (RUN_LOG §17-§19, §21-§22): over a 31-minute sequence a
# global homography leaves 2-6 px corner errors and Siril drops the farthest
# frames; one radial term shared by all frames brings every frame to <= 0.5 px.
# Siril's own star matching, however, still fails on frames 10+ minutes from
# the reference even when it is given the term as a SIP file (50-160 pairs,
# 1-23 px corner errors, dropped frames — §22).  The product therefore runs a
# transforms-only probe registration first (star lists + homographies, no r_
# frames), fits the term per job and re-associates every frame's full star
# list under it (``registration_calibration``), writes the term as a Siril
# distortion file *and its own homographies into the sequence file*
# (``write_registration_sequence``, convention proven by the synthetic shift
# test of §22), and lets Siril only apply them: ``seqapplyreg`` undistorts and
# resamples, ``stack`` rejects and averages.  The stacked sky is warped back
# to the native geometry afterwards (user decision: the ground and the mask
# stay pixel-exact, the output framing is not changed).  When the term cannot
# be fitted the flow falls back to Siril's plain two-pass registration.
PROBE_SCRIPT = ["cd work", "register light -2pass -transf=homography"]
DISTORTION_WCS = "distortion.wcs"
SKY_VALIDITY_DILATION_PX = 3      # RUN_LOG §25: seqapplyreg mixes one pixel at a zero edge; margin for rotation/undistortion
SKY_COVERAGE_STEP = 4
STACK_COMMAND = "stack r_light rej w 3 3 -norm=addscale -32b -filter-included -out=sky_master.fit"
REFERENCE_SELECTION = "siril_2pass_auto"
REGISTRATION_SOURCE_OWN = "starstack_calibration"      # matrices written by registration_calibration, Siril applies them
REGISTRATION_SOURCE_SIRIL = "siril_2pass"              # Siril's own two-pass matching (fallback)


def registration_commands(distortion_file: str | None = None, own_registration: bool = False) -> list[str]:
    """Siril commands of the final registration.

    ``own_registration``: the sequence file already carries our matrices and
    the ``D`` line for ``distortion_file`` (written by
    ``registration_calibration.write_registration_sequence``), so Siril only
    exports the registered frames.  Otherwise Siril's two-pass matching (with
    ``-disto=file`` when a distortion file is given).
    """
    if own_registration:
        return ["seqapplyreg light -filter-included"]
    register = "register light -2pass -transf=homography"
    if distortion_file:
        register += f" -disto=file {distortion_file}"
    return [register, "seqapplyreg light -filter-included"]


def stack_script(distortion_file: str | None = None, own_registration: bool = False) -> list[str]:
    """The single-process L script (registration, export and stack in one Siril run).

    Used by jobs run before efficiency P1b and by the diagnostic combined layout
    (AESTER_L_SIRIL_LAYOUT=combined); the default L layout is export_script +
    LEGACY_STACK_SCRIPT, and stack_script(d, own) == export_script(d, own) + [STACK_COMMAND].
    """
    return ["cd work", *registration_commands(distortion_file, own_registration), STACK_COMMAND]


def export_script(distortion_file: str | None = None, own_registration: bool = False) -> list[str]:
    """L mode's export process (efficiency P1b-3): the registration commands only, writing the r_light frames."""
    return ["cd work", *registration_commands(distortion_file, own_registration)]


REGISTRATION_COMMANDS = tuple(registration_commands(None))
STACK_SCRIPT = stack_script(None)
OWN_REGISTRATION_STACK_SCRIPT = stack_script(DISTORTION_WCS, own_registration=True)
# Efficiency P1b-3: L mode runs Siril's export and its stack in two processes (like R, F and strict-legacy), so a
# later resource governor can constrain the export without touching the stack.  The stack re-reads everything from
# disk; seqclean first, because a rerun on a sequence with cached statistics would normalize with rounded values.
# A job admitted under the EFF-D4 stack-settings contract (efficiency P1c, stack_settings.py) runs this script with
# its settings lines before seqclean (_contract_stack); a job admitted before it runs the script as it is.
LEGACY_STACK_SCRIPT = ("cd work", "seqclean r_light -stat", STACK_COMMAND)
L_LAYOUT_ENV = "AESTER_L_SIRIL_LAYOUT"   # diagnostics only: 'split' (default) or 'combined' (the pre-P1b script)


def legacy_stack_layout(value: str | None = None) -> str:
    """'split' unless AESTER_L_SIRIL_LAYOUT=combined. Never recorded in a signature or data digest.

    Read once per job attempt at run time (never at submit), and only by an attempt that may run L mode's
    export and stack (_may_run_legacy_stack); a malformed value fails such an attempt before any Siril run.
    Both layouts write the same products and records (efficiency P1b-3).
    """
    text = os.environ.get(L_LAYOUT_ENV) if value is None else value
    if text in (None, "", "split"):
        return "split"
    if text == "combined":
        return "combined"
    raise ValueError(f"{L_LAYOUT_ENV} must be 'split' or 'combined'")


def _may_run_legacy_stack(job: dict) -> bool:
    """Whether this job attempt may reach L mode's export and stack (``export_then_stack`` in _run_locked).

    Not for strict support (strict, F), not for a stack-range mode (R, F), and not once the stack checkpoint is
    passed.  Read from the job's own fields at attempt start: a superset of the attempts that call the helper (an
    invalid range record still fails later), so F, R, strict and resumed-after-stack attempts ignore
    AESTER_L_SIRIL_LAYOUT as they did before P1b (review lsplit-2).
    """
    from . import registered_support as rs
    return (job.get("support_mode", "legacy") != rs.MODE and job.get("stack_range_mode", "legacy") == "legacy"
            and "stack" not in job["checkpoints"])


# Efficiency P1c F-3 (prereg §3.3): the finite stack's two frame pools, decided here because finite_stack is in the
# frozen holdout closure (rule R22).
FINITE_WRITE_ENV = "AESTER_FINITE_WRITE"   # diagnostics only: 'auto' (default) or 'reference' (the pre-P1c q_ writer)
FINITE_POOL_CAP = 4                        # frames at once, inventory and q_ writes (prereg §2.3)
FINITE_INVENTORY_HASH_BYTES = 8 << 20      # the export hash's chunk buffer, beside the support map's bytes and array
FINITE_TILE_ROWS = 64                      # stack_range._tiles: rows per float32 tile of the masking


def finite_write_setting(value: str | None = None) -> str:
    """'auto' unless AESTER_FINITE_WRITE=reference: every q_ through the reference body (a copy rewritten in place) on
    one worker, for A/B runs and as a field fallback. Diagnostics only, never in a signature or a record (like
    AESTER_STRICT_KERNEL). Read once per job attempt, only by one that may write the finite stack's inputs
    (_may_run_finite_write); a malformed value fails that attempt at its start (P1c U-C2)."""
    text = os.environ.get(FINITE_WRITE_ENV) if value is None else value
    setting = (text or "").strip().lower() or "auto"
    if setting in ("auto", "reference"):
        return setting
    raise ValueError(f"{FINITE_WRITE_ENV} must be 'auto' or 'reference'")


def _may_run_finite_write(job: dict) -> bool:
    """Whether this job attempt may write the finite stack's q_ inputs: an F job before its stack checkpoint."""
    return job.get("stack_range_mode") == stack_settings.FINITE_MODE and "stack" not in job["checkpoints"]


class _FinitePools:
    """finite_stack.Pools for one job attempt (efficiency P1c F-3, prereg §3.3; finite_stack is in the frozen holdout
    closure, so the decisions are made here, rule R22).

    - inventory (F-3b): ``2P + 8 MiB`` per frame (the support map's bytes and array, the hash buffer), at most
      FINITE_POOL_CAP frames, through performance.frame_workers (AESTER_FRAME_WORKERS overrides it).
    - q_ writes (F-3a): the streamed body's unit is ``F + 2P + 6T`` (the read buffer, the support map and the one in
      flight, six float32 tiles); it runs when the memory budget holds one such unit, on up to FINITE_POOL_CAP workers
      (in flight = workers, R17). Otherwise, and under AESTER_FINITE_WRITE=reference, the reference body runs on one
      worker, also under AESTER_FRAME_WORKERS=max; its unit is ``3P + 6T`` (the serial loop still holds the previous
      frame's map while the next one's bytes and array are read; then six tiles), measured on F48 frames at 3.12P.
      A frame of unknown size (a count array that is not one frame) gets the reference body on one worker, unsized.
    - Each pool runs in its timer operation; its timings note (frame_pool.<site>, R19) adds the deciding bound, and a
      pool the plan itself made serial gives the pre-registered reason (route: the reference body; budget: memory),
      unless AESTER_FRAME_WORKERS=1 asked for one worker.
    """

    def __init__(self, timer=None, write_setting: str = "auto"):
        self.timer, self.write_setting = timer, write_setting
        self.decisions = {}            # site -> (frame_workers decision, body)

    def inventory(self, *, units, pixels):
        from . import finite_stack as fs, performance
        if pixels is None:             # a malformed support record: it fails right after this pool, serially
            return fs.Plan()
        unit_bytes = 2 * pixels + FINITE_INVENTORY_HASH_BYTES
        decision = {}
        workers = performance.frame_workers(units, unit_bytes, FINITE_POOL_CAP, blas=False, decision=decision)
        self.decisions["finite_inventory"] = (decision, None)
        return fs.Plan(workers, unit_bytes)

    def write(self, *, units, file_bytes, pixels, width):
        from . import finite_stack as fs, performance
        if pixels is None or width is None:   # a malformed count array: the reference body, serially; it fails there
            self.decisions["finite_write"] = ({}, fs.REFERENCE)
            return fs.Plan()
        tile = 3 * FINITE_TILE_ROWS * width * 4
        frame = file_bytes if file_bytes is not None else 12 * pixels    # no source could be stat'ed: its data unit
        streamed = frame + 2 * pixels + 6 * tile
        decision = {}
        workers = performance.frame_workers(units, streamed, FINITE_POOL_CAP, blas=False, decision=decision)
        if self.write_setting == "reference" or decision.get("memory", 0) < 1:   # no "memory": memory unknown
            plan = fs.Plan(1, 3 * pixels + 6 * tile, fs.REFERENCE)
        else:
            plan = fs.Plan(workers, streamed, fs.STREAMED, workers)
        self.decisions["finite_write"] = (decision, plan.body)
        return plan

    def operation(self, name):
        return self.timer.operation(name) if self.timer is not None else contextlib.nullcontext()

    def note_pool(self, site, stats, **extra):
        from . import finite_stack as fs, performance
        entry = dict(stats)
        decision, body = self.decisions.get(site, ({}, None))
        setting = frame_pool.worker_setting()
        if (entry.get("serial") and entry.get("reason") == "workers"
                and not (setting.isdigit() and int(setting) == 1)):
            if body == fs.REFERENCE:
                entry["reason"] = "route"
            elif decision.get("bound") in ("memory", "memory-unknown"):
                entry["reason"] = "budget"
        entry.update(extra, bound=decision.get("bound"))
        performance.note_pool(site, entry)

    def shadow_pool(self, footprint, *, actual, items, frame_pixels):
        from . import resources
        resources.shadow_pool(footprint, actual=actual, items=items, frame_pixels=frame_pixels)


def sequence_reference(sequence_path: Path, inputs: list[dict]) -> dict:
    """Report which input Siril used as the sky reference, read from ``light_.seq``.

    Two-pass registration lets Siril choose the reference by frame quality, and
    the choice is recorded only in the sequence header (``reference_image`` is a
    0-based position; ``I`` lines bind positions to image numbers, which follow
    the staged input order ``frame00001`` -> ``light_00001`` -> ``inputs[0]``).
    A missing or unreadable sequence never fails a job that already produced
    its master: the reference is then reported as unknown rather than guessed.
    """
    unknown = {"reference_selection": "unknown", "reference_input": None, "reference_stem": None,
               "reference_position": None, "reference_sequence": str(sequence_path)}
    try:
        sequence = read_seq(sequence_path)
    except (OSError, SequenceError, UnicodeDecodeError, ValueError) as exc:
        return {**unknown, "reference_error": f"{type(exc).__name__}: {exc}"}
    position = sequence["header"]["reference_image"]
    image = sequence["images"][position]
    if not 1 <= image["index"] <= len(inputs):
        return {**unknown, "reference_stem": image["stem"], "reference_position": position,
                "reference_error": f"sequence image {image['index']} has no staged input"}
    return {"reference_selection": REFERENCE_SELECTION, "reference_input": inputs[image["index"] - 1]["path"],
            "reference_stem": image["stem"], "reference_position": position,
            "reference_sequence": str(sequence_path)}


def registration_transforms(work: Path, calibration: dict | None, own_registration: bool, height: int) -> tuple[dict, object]:
    """Every stacked frame's top-down frame -> reference matrix plus the grid model they act in (RUN_LOG §25).

    Own registration: the calibration's matrices (magnified undistorted grid)
    and its distortion model; Siril's two-pass fallback: the matrices Siril
    wrote into ``light_.seq`` converted to top-down native coordinates, no
    model.  Frames Siril or the calibration left out are absent.
    """
    import numpy as np
    if own_registration and calibration and calibration.get("registration"):
        from .registration_distortion import RadialDistortion
        frames = calibration["registration"]["frames"]
        transforms = {stem: np.asarray(f["H_top_down_magnified"], dtype=np.float64) for stem, f in frames.items()
                      if f.get("status") == "registered" and f.get("H_top_down_magnified") is not None}
        return transforms, RadialDistortion(**calibration["parameters"])
    from . import registration_adapter as adapter
    sequence = read_seq(work / SEQUENCE_FILE)
    hyp = adapter.HYPOTHESES[adapter.DEFAULT_HYPOTHESIS]
    transforms = {}
    regdata = sequence["regdata"].get(1, [])
    for image, reg in zip(sequence["images"], regdata):
        if not image.get("selected", True) or reg is None or reg.get("H") is None:
            continue
        h = np.asarray(reg["H"], dtype=np.float64)
        if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
            continue
        transforms[image["stem"]] = adapter.convert_matrix(hyp, h, height, height)
    return transforms, None


def first_frame_geometry(work: Path, calibration: dict | None, own_registration: bool, shape, inputs):
    """Freeze the geometry used by this stack; later canvas edits do not read mutable sequences."""
    import numpy as np
    record = {"version": "first-frame-geometry-1", "shape": list(shape),
              "input": inputs[0], "coordinate_system": "TOP-DOWN native pixel centres",
              "distortion": calibration.get("parameters") if own_registration and calibration else None}
    try:
        matrices, _ = registration_transforms(work, calibration, own_registration, int(shape[0]))
        matrix = matrices.get("light_00001")
        if matrix is None or np.asarray(matrix).shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-15:
            raise ValueError("首帧未成功配准")
        return {**record, "status": "succeeded", "matrix": np.asarray(matrix).tolist()}
    except (OSError, ValueError, SequenceError) as exc:
        return {**record, "status": "unavailable", "reason": str(exc), "matrix": None}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_job(job_dir: str | Path) -> dict:
    directory = Path(job_dir)
    job = read_json(directory / "job.json")  # polled while the job worker replaces it
    stale = job["state"] in {"running", "cancel_requested"} and not is_locked(directory / "worker.lock")
    if job["state"] == "queued":
        stale = bool(job.get("worker_pid")) and not pid_alive(job["worker_pid"])
        if not job.get("worker_pid"):
            stale = (datetime.now(timezone.utc) - datetime.fromisoformat(job["updated_at"])).total_seconds() > 60
    if stale:
        # Read-only derived state: a crashed worker need not have written "failed".
        job.update(state="failed", message="处理进程异常退出，请新建任务", error="Worker stopped before completing the job",
                   recovery_blocked=True)
    return job


def _update(job_dir: Path, **changes) -> dict:
    with file_lock(job_dir / "metadata.lock", timeout=20):
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        job.update(changes, updated_at=now())
        atomic_json(job_dir / "job.json", job)
        return job


def _sha(path: Path, *, trust: str = hashing.TRUST_LEDGER) -> str:
    """SHA-256 of a file (via starstack.hashing, counted per timed stage as "core._sha").

    Inside a run scope an unchanged file may be answered from the run ledger;
    ``trust="real"`` always reads it.
    """
    path.open  # a value without Path.open fails as before (AttributeError), before any hashing
    return hashing.sha256_file(path, source=SHA_SOURCE, trust=trust)


def _sha_many(paths, *, trust: str = hashing.TRUST_LEDGER, on_error=None) -> list[str]:
    """[_sha(p) for p in paths], hashed in parallel; fails like that loop (the lowest failing index).

    ``on_error``: see :func:`starstack.hashing.sha256_many`.
    """
    return hashing.sha256_many(list(paths), source=SHA_SOURCE, trust=trust, on_error=on_error)


def _checkpoint_records(directory: Path, files) -> list[dict]:
    """[{"path": relative path, "sha256": _sha(p)} for p in files], the files hashed in parallel.

    Fails like that comprehension: a file outside ``directory`` raises only after
    every file before it was hashed, and only if none of those failed.
    """
    files = list(files)
    relative, failure = [], None
    for path in files:
        try:
            relative.append(str(path.relative_to(directory)))
        except BaseException as exc:
            failure = exc
            break
    digests = _sha_many(files[:len(relative)])
    if failure is not None:
        raise failure
    return [{"path": name, "sha256": digest} for name, digest in zip(relative, digests)]


def _job_handover_token():
    """The hash-handover token this job worker was started with, or None."""
    token = os.environ.get(HANDOVER_TOKEN_ENV)
    try:
        hashing.handover_path(token)
    except (TypeError, ValueError):
        return None
    return token


def _export_hash_handover(directory: Path) -> None:
    """Hand this worker's verified digests of files in the job directory to its launcher (never fails the job)."""
    token = _job_handover_token()
    if token is None:
        return
    try:
        profiling.note("hash_handover", {"exported": hashing.export_handover(None, directory, token=token)})
    except Exception as exc:   # without a handover the launcher simply reads the files
        profiling.note("hash_handover", {"error": f"{type(exc).__name__}: {exc}"[:300]})


def _discard_hash_handover() -> None:
    """A job that did not succeed leaves no handover behind."""
    token = _job_handover_token()
    if token is not None:
        try:
            os.remove(hashing.handover_path(token))
        except OSError:
            pass


def _note_siril_pin() -> None:
    """Diagnostics only: put an opt-in AESTER_SIRIL_PIN into the running timing receipts.

    Never raises and never feeds a signature or digest. A malformed pin is
    noted with its error here; the Siril run itself still refuses it.
    """
    try:
        try:
            from . import siril
            pin = siril.pin_diagnostics()
        except Exception as exc:
            raw = os.environ.get("AESTER_SIRIL_PIN", "")
            pin = ({"source": "AESTER_SIRIL_PIN", "value": raw[:80], "error": f"{type(exc).__name__}: {exc}"[:300]}
                   if raw.strip() else None)
        if pin:
            profiling.note("siril_pin", pin)
    except Exception:
        pass


def exposure_diagnostics(work: Path, converted: list[Path], contract: dict | None, stacked_count) -> dict:
    """Sum actual selected exposures; unresolved membership must remain unknown."""
    import math
    from astropy.io import fits
    from .exposure_summary import summarize_exposures
    values = []
    for i, path in enumerate(converted):
        if contract:
            value = contract['frames'][i]['exposure_seconds']
        else:
            header = fits.getheader(path)
            found = [header[k] for k in ('EXPTIME', 'EXPOSURE') if k in header]
            value = found[0] if found and all(v == found[0] for v in found) else None
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                value = None
        values.append(value)
    initial = summarize_exposures(values, [])
    try:
        seq = read_seq(work / SEQUENCE_FILE)
        images = seq['images']
        if len(images) != len(values) or sorted(im['index'] for im in images) != list(range(1, len(values)+1)):
            raise ValueError('sequence membership does not match inputs')
        if any(im['stem'] != converted[im['index']-1].stem for im in images):
            raise ValueError('sequence names do not match converted exposures')
        chosen = [im['index']-1 for im in images if im['selected']]
        if type(stacked_count) is not int or len(chosen) != stacked_count:
            raise ValueError('selection count differs from actual stack count or is unknown')
        return {**summarize_exposures(values, chosen), 'membership_status': 'verified_sequence',
                'included_input_indices_zero_based': chosen, 'semantics': 'sum_of_included_individual_exposures'}
    except (OSError, SequenceError, ValueError, KeyError) as exc:
        return {**initial, 'included_count': stacked_count, 'included_total_seconds': None,
                'unknown_included_count': None, 'membership_status': 'unknown', 'reason': str(exc)}


def inspect_sources(input_dir: str | None, *, input_files: list[str] | None = None, min_count: int = 3) -> dict:
    """Inspect a directory, or exactly the selected files in the supplied order.

    Explicit selections never scan sibling files. ``input_dir`` is only context
    in that mode; each actual source parent is separately recorded.
    """
    if type(min_count) is not int or not 1 <= min_count <= 300:
        raise ValueError("min_count 须为 1 至 300 的整数")
    if input_files is None:
        if not isinstance(input_dir, (str, Path)) or not str(input_dir).strip():
            raise ValueError("请选择存在的素材文件夹")
        directory = Path(input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("请选择存在的素材文件夹")
        files = sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() in RAW | FITS)
        selection_mode = "directory"
    else:
        if type(input_files) is not list or not input_files:
            raise ValueError("input_files 须为非空文件列表")
        files = []
        identities = set()
        for value in input_files:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError("素材列表仅接受文件路径")
            path = Path(value).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() not in RAW | FITS:
                raise ValueError(f"素材不存在或格式不支持：{path}")
            identity = os.path.normcase(str(path))
            if identity in identities:
                raise ValueError("素材列表包含重复文件")
            identities.add(identity)
            files.append(path)
        directory = Path(input_dir).expanduser().resolve() if input_dir is not None else files[0].parent
        if not directory.is_dir():
            raise ValueError("素材上下文文件夹不存在")
        selection_mode = "files"
    if len(files) < min_count:
        raise ValueError(f"需要至少 {min_count} 张 RAW 或 FITS；请勿混入 JPG")
    kinds = {"raw" if p.suffix.lower() in RAW else "fits" for p in files}
    if len(kinds) != 1 or ("raw" in kinds and len({p.suffix.lower() for p in files}) != 1):
        raise ValueError("请将同机型、同曝光的 RAW 放在一个文件夹，不要混合 RAW 与 FITS")
    if len(files) > 300:
        raise ValueError("原型单个任务最多支持 300 帧，请先选择较小的序列")
    fits_contract = None
    if "fits" in kinds:
        from .fits_input import inspect_files
        fits_contract = inspect_files(files)
    return {"input_dir": str(directory), "kind": kinds.pop(), "files": [str(p) for p in files],
            "count": len(files), "bytes": sum(p.stat().st_size for p in files), "fits_input": fits_contract,
            "selection_mode": selection_mode, "input_roots": list(dict.fromkeys(str(p.parent) for p in files))}


def submit_job(input_dir: str | None, output_root: str, siril_path: str | None = None,
               sky_mask_path: str | None = None, idempotency_key: str | None = None, frame_masks_dir: str | None = None,
               transient_marks: list | str | None = None, support_mode: str = "legacy",
               stack_range_mode: str = "legacy", force_restart: bool = False,
               stack_engine_path: str | None = None, input_files: list[str] | None = None,
               transient_mode: str | None = None, color_adjustments: dict | None = None,
               compose_feather_mode: str | None = None, image_rotation: int | None = None) -> dict:
    """Admit a job.  ``force_restart`` (docs/HARDENING_PACKAGE_20260922.md H2): when the idempotent directory already
    exists for the *same* submission and its job is finished (succeeded / failed / cancelled, including a crashed
    worker's ``recovery_blocked``) with no live worker, the old directory is renamed ``job-<ident>.restarted-<UTC>``
    (nothing is deleted) and a fresh job of the same name is created.  Without it the existing job is returned as before."""
    from . import registered_support as rs
    from . import stack_range as sr
    from . import finite_stack as fs
    if image_rotation is not None:
        from .orientation import normalize_rotation
        image_rotation = normalize_rotation(image_rotation)
    if compose_feather_mode is not None and compose_feather_mode != "canvas-inward-feather-1":
        raise ValueError("未知的天空地景过渡方式")
    if compose_feather_mode is not None and not sky_mask_path:
        raise ValueError("天空地景柔和过渡需要显式天空蒙版")
    if transient_mode is not None and (type(transient_mode) is not str or transient_mode not in {"off", "review", "legacy"}):
        raise ValueError("transient_mode 须为 off、review 或 legacy")
    if color_adjustments is not None:
        from .color_adjustments import normalize
        color_adjustments = normalize(color_adjustments)
    if transient_mode == "off" and transient_marks:
        raise ValueError("亮迹关闭时不能同时提交亮迹标记")
    if type(force_restart) is not bool:
        raise ValueError("force_restart 须为 bool")
    if stack_range_mode not in ('legacy', sr.MODE, fs.MODE):
        raise ValueError('未知堆栈范围模式')
    finite_enabled = stack_range_mode == fs.MODE
    if finite_enabled and (support_mode != rs.MODE or not sky_mask_path):
        raise ValueError('有限值堆栈需要严格几何支持与明确天空蒙版；纯天空可提供全白蒙版')
    if stack_engine_path and not finite_enabled:
        raise ValueError('独立堆栈引擎路径仅用于 finite_validity_v1 模式')
    stack_engine = fs.describe_engine(stack_engine_path) if finite_enabled else None
    if support_mode not in rs.MODES:
        raise ValueError("unknown interpolation support mode")
    if support_mode != "legacy" and not sky_mask_path:
        raise ValueError("严格支持候选需要天空蒙版")
    info = doctor(siril_path)
    if not info["available"]:
        raise ValueError(info["message"])
    if support_mode != "legacy":
        rs.require_engine(info)
    if stack_range_mode != 'legacy':
        sr.require_engine(info)
    sources = inspect_sources(input_dir, input_files=input_files) if input_files is not None else inspect_sources(input_dir)
    if sources["kind"] == "fits" and color_adjustments is not None and any(color_adjustments.values()):
        raise ValueError("FITS 暂不支持白平衡或曝光调整；请保持默认值")
    from . import fits_input as fi
    fits_contract = sources.get("fits_input")
    cfa = bool(fits_contract and fits_contract['layout'] == 'cfa')
    if cfa:
        fi.require_engine(info)
    root = Path(output_root).expanduser().resolve()
    source_roots = [Path(p) for p in sources.get("input_roots", [sources["input_dir"]])]
    if any(root == parent or parent in root.parents for parent in source_roots):
        raise ValueError("结果文件夹必须位于所有素材文件夹之外")
    root.mkdir(parents=True, exist_ok=True)
    from . import disk_budget
    disk_estimate = disk_budget.estimate(sources, fits_contract, stack_range_mode=stack_range_mode,
                                         range_attempts=sr.contract()["max_attempts"])
    free_bytes = int(shutil.disk_usage(root).free)
    disk_estimate["free_bytes_at_admission"] = free_bytes
    if free_bytes < disk_estimate["required_bytes"]:
        raise ValueError(disk_budget.shortfall_message(disk_estimate, free_bytes))
    # [{"path", "size", "sha256"} for each file]: sizes in order, then the digests in parallel (a
    # failing size is raised only after the files before it were hashed, as in that comprehension;
    # a file that fails to hash is stat'ed again, so one removed meanwhile fails at its size, as there).
    paths = [Path(f) for f in sources["files"]]
    sizes, failure = [], None
    for path in paths:
        try:
            sizes.append(path.stat().st_size)
        except BaseException as exc:
            failure = exc
            break

    def restat(index, exc):
        if isinstance(exc, OSError):
            paths[index].stat()
        return None
    digests = _sha_many(paths[:len(sizes)], on_error=restat)
    if failure is not None:
        raise failure
    files = [{"path": f, "size": size, "sha256": digest} for f, size, digest in zip(sources["files"], sizes, digests)]
    if fits_contract and fi.inspect_files([Path(f['path']) for f in files]) != fits_contract:
        raise ValueError('FITS 输入在检查期间发生变化，请重新提交')
    mask = None
    if sky_mask_path:
        mask_path = Path(sky_mask_path).expanduser().resolve()
        if not mask_path.is_file() or mask_path.suffix.lower() != ".png":
            raise ValueError("天空蒙版需要 PNG 文件，白色为天空，黑色为地景")
        mask = {"path": str(mask_path), "sha256": _sha(mask_path)}
    frame_masks = None
    if frame_masks_dir:
        # per-frame sky masks (RUN_LOG §26): <input file stem>.png in the frames' native geometry; a frame without its
        # own file uses the job's static mask, so the static mask is required alongside
        if not mask:
            raise ValueError("逐帧天空蒙版需要同时提供整体天空蒙版（没有自己蒙版的帧用它）")
        masks_dir = Path(frame_masks_dir).expanduser().resolve()
        if not masks_dir.is_dir():
            raise ValueError("逐帧蒙版目录不存在")
        entries = {}
        for i, source in enumerate(files, 1):
            candidate = masks_dir / (Path(source["path"]).stem + ".png")
            if candidate.is_file():
                entries[f"light_{i:05d}"] = {"path": str(candidate), "sha256": _sha(candidate), "input": source["path"]}
        if not entries:
            raise ValueError("逐帧蒙版目录里没有与任何输入同名的 PNG")
        frame_masks = {"dir": str(masks_dir), "files": entries, "count": len(entries)}
    marks = None
    if transient_marks:
        # manual bright-trail marks (RUN_LOG §28): a JSON file or a list of {input, x0, y0, x1, y1[, half_width_px]}
        if isinstance(transient_marks, (str, Path)):
            marks_path = Path(transient_marks).expanduser().resolve()
            if not marks_path.is_file():
                raise ValueError("亮迹标记文件不存在")
            marks = json.loads(marks_path.read_text(encoding="utf-8"))
        else:
            marks = list(transient_marks)
        if not isinstance(marks, list):
            raise ValueError("亮迹标记须为列表")
        marks = _normalise_marks(marks, files)
    support_signature = {} if support_mode == "legacy" else {"support_mode": support_mode, "support_version": rs.SUPPORT_VERSION}
    input_signature = {'fits_input': fits_contract} if fits_contract else {}
    range_contract = fs.contract() if finite_enabled else sr.contract() if stack_range_mode != 'legacy' else None
    range_signature = {'stack_range_contract': range_contract} if range_contract else {}
    if finite_enabled:
        range_signature['stack_engine'] = stack_engine
    options = {}
    if input_files is not None:
        options["input_selection"] = {"mode": "files", "files": list(sources["files"]), "input_roots": sources["input_roots"]}
    if transient_mode is not None:
        options["transient_mode"] = transient_mode
    if color_adjustments is not None:
        options["color_adjustments"] = color_adjustments
    if compose_feather_mode is not None:
        options["compose_feather_mode"] = compose_feather_mode
    if image_rotation is not None:
        options["image_rotation"] = image_rotation
    # EFF-D4 (efficiency P1c §4.2): a new L, strict-L or R job on Siril 1.4.4 stacks under the stack-settings contract,
    # bound into the signature through admission_options.  The preimage without it is the same submission admitted
    # before the contract (legacy_signature): such a job keeps its old contract.
    legacy_options = options
    settings_contract = stack_settings.admission_field(info, stack_range_mode)
    if settings_contract is not None:
        options = {**options, stack_settings.FIELD: settings_contract}

    def sign(chosen):
        options_signature = {"admission_options": chosen} if chosen else {}
        return hashlib.sha256(json.dumps({"files": files, "mask": mask, "frame_masks": frame_masks, "engine": info, "transient_marks": marks, **support_signature, **input_signature, **range_signature, **options_signature},
                                         sort_keys=True).encode()).hexdigest()
    signature = sign(options)
    legacy_signature = sign(legacy_options) if settings_contract is not None else None
    ident = hashlib.sha256(idempotency_key.encode()).hexdigest()[:24] if idempotency_key else uuid.uuid4().hex[:16]
    directory = root / f"job-{ident}"
    job = {"schema_version": 2 if range_contract else 1, "job_dir": str(directory), "state": "queued", "stage": "queued",
           "progress": 0, "message": "等待处理", "error": None, "created_at": now(), "updated_at": now(),
           "inputs": files, "input_kind": sources["kind"], "mask": mask, "frame_masks": frame_masks, "transient_marks": marks, "engine": info,
           "fits_input": fits_contract, "fits_input_sha256": fi.fingerprint(fits_contract) if fits_contract else None,
           "signature": signature, "checkpoints": [], "checkpoint_files": {}, "artifacts": [], "transient_revision": None,
           "sky_validity_required": bool(mask),
           "support_mode": support_mode,
           "stack_range_mode": stack_range_mode, "stack_range_contract": range_contract,
           "stack_range_sha256": sr.fingerprint(range_contract) if range_contract else None,
           **({'stack_engine': stack_engine} if finite_enabled else {}),
           **({**options, "admission_options": options, "admission_options_sha256": _options_digest(options)} if options else {}),
           "support_version": rs.SUPPORT_VERSION if support_mode != "legacy" else None,
           "disk_estimate": disk_estimate,
           "limitations": ["未做暗场/平场校准", "RAW 相机色彩未标定",
                           "地景合成需与原片画幅一致的天空蒙版；星空母版取 Siril 两段式自选参考帧时刻的位置（见 diagnostics.reference_input）",
                           "镜头畸变按作业从星表自拟（径向 k1/k2/中心），只在留出残差改善时启用；启用时各帧配准矩阵由本产品在畸变模型下重新关联全部星表算出并写入序列文件，Siril 只做去畸变导出与堆栈，未能关联的帧不参与堆栈（见 diagnostics.registration_detail）；星空母版最后变换回原生几何（见 diagnostics.distortion）",
                           "未实现自动天空分割或逐帧遮挡补偿", "三种色调仅为导出与预览验证",
                           "亮迹候选由配准帧减母版的残差检出并只给建议标签（流星/卫星/飞机/伪迹/未知），人工标签与处置策略决定哪些经独立瞬态层加回含亮迹成片；基础成片不主动加回亮迹层，堆栈中仍可能残留亮迹，自动分类质量未经真实流星集验收（见 diagnostics.transients）"]}
    if finite_enabled:
        job['limitations'].append('有限值堆栈保留几何有效域中的零、负值和超1值，并输出逐通道有效/保留计数；仅堆栈边界已验证，不能证明RAW解码、插值或颜色前端未丢失信息')
    elif range_contract:
        job['limitations'].append('高光范围保护：共同RGB零值沿用缺数据语义，其余有限值（含单通道零值和负值）照常堆栈，暗端与legacy相同由Siril在0处截断并计数；非有限值会停止，不解决原始饱和或严格RAW有符号堆栈')
    if cfa:
        job['limitations'].extend(['FITS 上游校准状态未明；BIAS只记录来源，不再次扣除',
                                  'CFA行方向按输入头或已验证设备组合解释；绝对相位、天球方向和颜色仍未认证',
                                  '浮点适配不保证后续Siril重采样与堆栈保留全部负值和超界值'])
    with file_lock(root / ".admission.lock", timeout=30):
        existing = None
        if directory.exists():
            existing = load_job(directory)
            if existing["signature"] != signature:
                if legacy_signature is None or existing["signature"] != legacy_signature:
                    raise ValueError("相同提交编号对应了不同素材或参数")
                # EFF-D4: the same submission, admitted before the stack-settings contract: it is returned, resumed or
                # restarted under the contract it was admitted with (a workflow that crashed before recording its
                # stack job would otherwise be refused forever, force_restart included)
                job = _without_stack_settings(job, legacy_options, legacy_signature)
            if not force_restart:
                return existing
        # S-8: refuse an F/R job whose stack engine could not write its outputs, before anything is renamed or created.
        refusal = stack_engine_path_refusal(root, directory, stack_range_mode)
        if existing is not None:
            retired = _retire_for_restart(directory, existing, refusal=refusal)
            job["restarted_from"] = {"directory": str(retired), "previous_state": existing["state"],
                                     "previous_error": existing.get("error"), "at": now()}
        elif refusal:
            raise ValueError(refusal)
        directory.mkdir()
        atomic_json(directory / "job.json", job)
        _spawn_worker(directory)
    return load_job(directory)


def stack_engine_path_refusal(root, job_dir, stack_range_mode) -> str | None:
    """S-8: None when this mode's stack engine can write every file it needs under ``job_dir`` (L: always None).

    The one check behind both admission sites (submit_job and workflow.submit_workflow). A pure function of the
    resolved paths: it reads, writes and records nothing."""
    from . import finite_stack as fs, stack_range as sr
    if stack_range_mode == fs.MODE:
        longest = fs.longest_engine_path(job_dir)
    elif stack_range_mode == sr.MODE:
        longest = sr.longest_engine_path(job_dir)
    else:
        return None
    return engine_path_refusal(root, longest)


def _retire_for_restart(directory: Path, existing: dict, *, refusal: str | None = None) -> Path:
    """Rename a finished job directory out of the way of a forced restart (H2); refuse anything that may still be alive.

    Called under the admission lock.  The old directory keeps every file byte for byte (its ``job.json`` still names
    the original path, which is how a retired copy can be told apart).  ``refusal`` (S-8, a stack engine path the
    new job could not write) is raised after the liveness refusals and before anything is renamed."""
    state = existing["state"]
    if state not in TERMINAL:
        raise ValueError(f"作业状态为 {state}，仍在运行或排队，不能用同一提交编号重跑；请先取消并等待它停止")
    if is_locked(directory / "worker.lock"):
        raise ValueError("旧作业的工作进程仍持有锁，不能重跑；请等待它退出")
    pid = existing.get("worker_pid")
    if pid_alive(pid):
        raise ValueError(f"旧作业记录的工作进程 {pid} 仍存活，不能重跑；请先确认它已退出")
    if refusal:
        raise ValueError(refusal)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    retired = directory.with_name(f"{directory.name}.restarted-{stamp}")
    if retired.exists():
        raise ValueError(f"目标目录已存在：{retired}")
    try:
        os.rename(directory, retired)   # same volume, nothing copied or deleted
    except OSError as exc:
        raise ValueError(f"无法改名旧作业目录（{exc}）；请关闭正在使用它的程序后重试") from exc
    return retired


def _spawn_worker(directory: Path) -> None:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with (directory / "worker.log").open("ab") as log:
        try:
            process = subprocess.Popen([sys.executable, "-m", "starstack.worker", str(directory)], env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                             start_new_session=(os.name != "nt"))
            _update(directory, worker_pid=process.pid)
        except OSError as exc:
            _update(directory, state="failed", stage="launch", error=str(exc), message="无法启动处理进程")
            raise


def cancel_job(job_dir: str) -> None:
    directory = Path(job_dir).resolve()
    if load_job(directory)["state"] not in TERMINAL:
        (directory / "cancel.request").touch()


def resume_job(job_dir: str) -> dict:
    directory = Path(job_dir).resolve()
    with file_lock(directory.parent / ".admission.lock", timeout=30):
        return _resume_locked(directory)


def _options_digest(options: dict) -> str:
    return hashlib.sha256(json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _without_stack_settings(job: dict, options: dict, signature: str) -> dict:
    """The record submit_job wrote for ``job``'s submission before EFF-D4: no stack-settings field (same key order)."""
    record = {key: value for key, value in job.items() if key != stack_settings.FIELD}
    record["signature"] = signature
    if options:
        record.update(admission_options=options, admission_options_sha256=_options_digest(options))
    else:
        record.pop("admission_options", None)
        record.pop("admission_options_sha256", None)
    return record


def _verify_admission_options(job: dict) -> None:
    """Bind opt-in settings and selected order through a worker/resume attempt.

    Old jobs have no receipt. This is an integrity check, not a signature against
    an attacker who can rewrite the entire job and its hashes.
    """
    names = ("input_selection", "transient_mode", "color_adjustments", "compose_feather_mode", "image_rotation",
             stack_settings.FIELD)
    if "admission_options" not in job:
        if any(name in job for name in names):
            raise ValueError("作业选项缺少提交记录，请新建作业")
        return
    options = job["admission_options"]
    if (not isinstance(options, dict) or not options or set(options) - set(names)
            or job.get("admission_options_sha256") != _options_digest(options)
            or any((name in job) != (name in options) or job.get(name) != options.get(name) for name in names)):
        raise ValueError("作业选项已改变，请新建作业")
    if "transient_mode" in options and (type(options["transient_mode"]) is not str or options["transient_mode"] not in {"off", "review", "legacy"}):
        raise ValueError("无效的亮迹处理选项")
    if "compose_feather_mode" in options and (options["compose_feather_mode"] != "canvas-inward-feather-1" or not job.get("mask")):
        raise ValueError("无效的天空地景过渡选项")
    if "color_adjustments" in options:
        from .color_adjustments import normalize
        if normalize(options["color_adjustments"]) != options["color_adjustments"]:
            raise ValueError("无效的颜色调整选项")
    if "image_rotation" in options:
        from .orientation import normalize_rotation
        normalize_rotation(options['image_rotation'])
    stack_settings.job_contract(job)   # EFF-D4: exactly the contract, on a non-finite Siril 1.4.4 job
    if "input_selection" in options:
        selection = options["input_selection"]
        paths = [source["path"] for source in job["inputs"]]
        roots = list(dict.fromkeys(str(Path(path).parent) for path in paths))
        if selection != {"mode": "files", "files": paths, "input_roots": roots}:
            raise ValueError("所选素材顺序或来源已改变，请新建作业")


def gc_refusal(job: dict, action: str) -> str | None:
    """Why ``action`` is impossible after the rebuildable intermediates were removed (H3).

    ``starstack gc --work-cache`` and the in-app cleanup on the 作品与缓存 page (``cache_cleanup``, whose record
    says ``source: app``) both write ``job.json.gc`` before their first deletion.
    """
    record = job.get("gc") or {}
    if record.get("work_cache_removed"):
        if record.get("source") == "app":
            return _cache_cleaned_message(action, in_app=True)
        return (f"作业已于 {record.get('at')} 执行 starstack gc --work-cache，转换帧、配准帧与序列文件已删除，"
                f"不能{action}；请新建作业")
    return None


def _cache_cleaned_message(action: str, *, in_app: bool = False) -> str:
    if in_app:
        return (f"此任务的缓存已清理（中间文件已在“设置 → 作品与缓存”中删除），不能{action}；"
                f"作品与原片不受影响，如需重做请从原片新建任务")
    return f"此任务的缓存已清理（中间文件已删除），不能{action}；作品与原片不受影响，如需重做请从原片新建任务"


def cache_cleanup_refusal(job_dir: str | Path, action: str) -> str | None:
    """Why ``action`` is impossible for a workflow's stack job whose workflow cache was cleaned.

    The markers are written by the in-app cleanup (``cache_cleanup``) and by the uninstaller (``uninstall_options``)
    before their first deletion; either may have removed frames or images this action needs. Callers check
    ``gc_refusal`` first: the in-app cleanup also records ``job.json.gc`` (its message names the page).
    """
    directory = Path(job_dir).resolve()
    workflow = directory.parent.parent
    if (directory.parent.name == "stack" and (workflow / "workflow.json").is_file()
            and cache_cleanup_marked(workflow)):
        return _cache_cleaned_message(action)
    return None


def _resume_locked(directory: Path) -> dict:
    job = load_job(directory)
    _verify_admission_options(job)
    refusal = gc_refusal(job, "恢复")
    if refusal:
        raise ValueError(refusal)
    if job.get("recovery_blocked"):
        raise ValueError("工作进程异常退出，尚不能确认所有子进程已停止；请新建任务，避免并发写入旧目录")
    if is_locked(directory / "worker.lock"):
        raise ValueError("工作进程尚未停止，请稍后恢复")
    if job["state"] not in {"failed", "cancelled"}:
        raise ValueError("只有失败或已取消的作业可以恢复")
    # Serial reference: each input must exist and keep its digest (hashed in parallel, first failure raised).
    hashing.verify_entries(job["inputs"], source=SHA_SOURCE,
                           error=lambda source: ValueError("原始素材已改变或缺失，不能恢复"),
                           path=lambda source: Path(source["path"]), precheck=lambda source, path: path.is_file())
    if doctor(job["engine"]["executable"])["version"] != job["engine"]["version"]:
        raise ValueError("Siril 版本已变化，请创建新作业")
    from . import stack_range as sr
    sr.verify_job(job)
    if 'stack' in job['checkpoints']:
        sr.verify_receipt(directory / 'work', job)
    if job['input_kind'] == 'fits':
        from .fits_input import verify_contract
        verify_contract(job, [Path(source['path']) for source in job['inputs']])
    # Serial reference: stage by stage, a stage without records raises, then each record's file must
    # exist and keep its digest.  All files are hashed in parallel; the first failure in order is raised.
    entries = []
    for stage in job["checkpoints"]:
        records = job.get("checkpoint_files", {}).get(stage)
        if not records:
            entries.append(None)          # raised by _stage_has_records at this position
            break
        entries.extend(records)
    hashing.verify_entries(entries, source=SHA_SOURCE, validate=_stage_has_records,
                           error=lambda record: ValueError("检查点文件缺失或已改变，请创建新作业"),
                           path=lambda record: directory / record["path"],
                           precheck=lambda record, artifact: artifact.is_file())
    (directory / "cancel.request").unlink(missing_ok=True)
    _update(directory, state="queued", worker_pid=None, error=None, message="准备从完整阶段恢复")
    _spawn_worker(directory)
    return load_job(directory)


def _stage_has_records(record) -> None:
    if record is None:
        raise ValueError("旧检查点没有完整性记录，请创建新作业")


def run_job(job_dir: str) -> None:
    directory = Path(job_dir).resolve()
    try:
        with file_lock(directory / "worker.lock"):
            # One run ledger for the whole job attempt (AESTER_HASH_LEDGER: on / shadow / off).
            with hashing.run_scope("stack-job"):
                _run_locked(directory)
    except Busy:
        return


def _stage_inputs(job: dict, stage: Path, check) -> None:
    """Copy each original to ``stage`` unless an identical copy is there, and read every new copy back.

    Serial reference: for each input, keep an existing copy with the recorded digest, otherwise
    check(), copy, and read the copy back for real (another digest raises).  With more than one
    hash worker, the read-back of a copy runs on a helper thread while the next input is checked
    and copied; at most one read-back is pending.  It is settled before the next read-back starts,
    as soon as it is done, and before any failure of a later input (a cancel included) is raised,
    so the error reported is always the serial loop's, a changed copy stops the staging after at
    most one more copy, and a cancel waits for at most one read-back.
    """
    pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="aester-stage-readback")
            if hashing.resolve_workers(None) > 1 else None)
    readback = hashing.carry_scope(lambda target: _sha(target, trust=hashing.TRUST_REAL))
    pending = []   # [(future, recorded digest)] of the one copy whose read-back is running

    def settle():
        if pending:
            future, digest = pending.pop()
            if future.result() != digest:
                raise ValueError("素材复制时发生变化，请重新提交")

    try:
        for i, source in enumerate(job["inputs"], 1):
            target = stage / f"frame{i:05d}{Path(source['path']).suffix.lower()}"
            if pending and pending[0][0].done():
                settle()
            if not target.exists() or _sha(target) != source["sha256"]:
                check()
                with hashing.writing(target):
                    shutil.copyfile(source["path"], target)
                if pool is None:
                    if readback(target) != source["sha256"]:
                        raise ValueError("素材复制时发生变化，请重新提交")
                    continue
                settle()
                pending.append((pool.submit(readback, target), source["sha256"]))
        settle()
    except BaseException:
        settle()   # the earlier copy's read-back is what the serial loop reported first
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


def _run_locked(directory: Path) -> None:
    import numpy as np
    from .imaging import read_rgb, write_linear, median_foreground, blend_with_mask
    from .profiling import StageTimer
    from . import resources
    # host: processor groups and SHA-256 rate, once per worker process; peak sampler: P1b-2 local peaks (diagnostics)
    timer = StageTimer(host=True, peak_sampler_hz=10)
    _note_siril_pin()  # prereg 1.3: a pinned Siril run says so in its timings (diagnostics only)
    prior_timing_attempts = []
    stack_settings_notes = []

    def timed_script(executable, folder, name, commands, cancel_check, **options):
        # ``options`` (config, stack_settings) only for a contract stack (EFF-D4); every other launch is unchanged
        resources.shadow_siril(name, folder, commands, executable)   # P1b-2 governor: shadow only, never raises
        with job_stage_launches():   # S-4c: job stages may run Siril below normal priority (siril.siril_priority)
            return timer.call("script:" + name, run_script, executable, folder, name, commands, cancel_check, **options)

    def note_stack_settings(settings):
        # EFF-D4: T', B, p and n of every contract stack of this attempt, under timings only (never recorded elsewhere)
        stack_settings_notes.append(settings)
        timer.attach_note(stack_settings.NOTE_KEY, stack_settings_notes)

    def export_then_stack(export_commands):
        # Efficiency P1b-3: L mode's export (registration + seqapplyreg) and its stack run as two Siril processes,
        # the layout R, F and strict-legacy already use.  Same products and records; no checkpoint in between (an
        # interrupted attempt re-exports).  AESTER_L_SIRIL_LAYOUT=combined (diagnostics only) runs the pre-P1b script;
        # `layout` is read in the job-start block, for every attempt that can get here (_may_run_legacy_stack).
        if layout == "combined":
            timed_script(exe, directory, "02_stack", [*export_commands, STACK_COMMAND], check)
            return
        timed_script(exe, directory, "02_export", list(export_commands), check)
        check()
        if settings_contract is None:   # a job admitted before EFF-D4: the P1b script, byte for byte
            timed_script(exe, directory, "02_stack", list(LEGACY_STACK_SCRIPT), check)
        else:
            _contract_stack(timed_script, exe, directory, list(LEGACY_STACK_SCRIPT), check, note=note_stack_settings)

    def timed_diagnostics(diagnostics, state):
        # What the job's run ledger did (reads, hits, shadow checks and mismatches): diagnostics only.
        hashing.note_scope_report()
        timings = timer.finish()
        timings.update(state=state, recorded_at=now())
        # Publish the receipt with the terminal state, before a workflow can
        # freeze the child job. Each resumed attempt retains a distinct total.
        return {**(diagnostics or {}), "timings": timings, "timing_attempts": prior_timing_attempts}

    def check():
        if (directory / "cancel.request").exists():
            _update(directory, state="cancel_requested", message="正在停止处理")
            raise Cancelled("用户取消")

    def checkpoint(name, files):
        current = load_job(directory)
        records = dict(current.get("checkpoint_files", {}))
        records[name] = _checkpoint_records(directory, files)
        _update(directory, checkpoints=[*[c for c in current["checkpoints"] if c != name], name], checkpoint_files=records)

    # Efficiency P1c O-1 (prereg §3.6): the attempt's overlap window (None: none open) and whether its decision was
    # noted. Only this (main) thread calls checkpoint() and _update() with list fields; B writes job.json only through
    # check() (state and message of a cancel).
    overlap, o1_noted = [None], [False]

    def o1_begin(*, own, stack_done, calibration, validity_record, native_workers=0, native_unit_bytes=0):
        """Decide O-1 once per attempt, note the decision under timings and start B when eligible."""
        o1_noted[0] = True
        reason, plan, frozen = _o1_decide(transient_mode=transient_mode, own_registration=own, stack_done=stack_done,
                                          work=work, job=job, calibration=calibration,
                                          support_enabled=support_enabled, validity_record=validity_record,
                                          native_workers=native_workers, native_unit_bytes=native_unit_bytes)
        note = {"schema": O1_NOTE_SCHEMA, "eligible": reason is None, "reason": reason, **plan}
        if reason is None:
            o1 = _Overlap(directory, work, plan, frozen, timer, note)
            try:
                o1.start(check)
            except Exception as exc:          # O-1's own machinery: never a failure the serial run would not have
                # close() removes what B may have written with the marker; a folder it cannot remove keeps the marker
                # (logged) and the serial code goes on: the marker rule at the trail position clears it before the
                # serial scan, as for any earlier attempt's marker
                o1.close(f"start failed: {type(exc).__name__}")
                note.update(eligible=False, reason=f"overlap_error:{type(exc).__name__}")
                timer.attach_note(O1_NOTE, note)
                return None
            timer.attach_note(O1_NOTE, note)
            overlap[0] = o1
            _o1_started(o1)
            return o1
        timer.attach_note(O1_NOTE, note)
        return None

    def close_overlap(why):
        o1, overlap[0] = overlap[0], None
        if o1 is not None:
            o1.close(why)

    try:
        job = load_job(directory)
        earlier_diagnostics = job.get("diagnostics") or {}
        prior_timing_attempts = list(earlier_diagnostics.get("timing_attempts") or [])
        if earlier_diagnostics.get("timings"):
            prior_timing_attempts.append(earlier_diagnostics["timings"])
        _verify_admission_options(job)
        # --- P1b job-start block (CROSS_CHECK X6), in this order: (1) frame_pool.validate_environment() [P1b-1],
        # (2) the L Siril layout [P1b-3], read only by an attempt that may run L's export and stack, (3)
        # resources.shadow_start(...) [P1b-2]. Malformed knobs fail the attempt here, before any Siril run (the
        # governor falls back to off instead).
        frame_pool.validate_environment()   # (1) P1b-1: a malformed AESTER_FRAME_WORKERS fails the job at its start
        # (2) P1b-3: diagnostics only, never in a signature or data digest.  F, R, strict and an attempt resumed past
        # the stack checkpoint never run L's export and stack: they ignore the variable, as before P1b (None).
        layout = legacy_stack_layout() if _may_run_legacy_stack(job) else None
        # EFF-D4: the combined layout cannot plan the contract stack (n and p exist only after the export)
        settings_contract = stack_settings.job_contract(job)
        if layout == "combined" and settings_contract is not None:
            raise ValueError(f"{L_LAYOUT_ENV}=combined cannot run a job admitted under {stack_settings.VERSION} "
                             "(its stack settings need the export's frame count and size); unset it and resume")
        if layout not in (None, "split"):
            profiling.note("l_siril_layout", layout)   # timing receipt only (an excluded key)
        # (2b) P1c F-3a: AESTER_FINITE_WRITE, diagnostics only, read only by an attempt that may write the finite
        # stack's q_ inputs (others ignore it: None).
        finite_write = finite_write_setting() if _may_run_finite_write(job) else None
        # (3) P1b-2 governor, shadow only: notes under timings, never changes a setting, never fails the job
        # (a malformed knob -> off with a note).
        resources.shadow_start(resources.JOB_SCOPE, timer=timer, job=job, work_dir=directory)
        # --- end of the P1b job-start block
        transient_mode = job.get("transient_mode", "legacy")
        exe = job["engine"]["executable"]
        from . import registered_support as rs
        from . import stack_range as sr
        from . import finite_stack as fs
        range_enabled = sr.verify_job(job)
        finite_enabled = job.get('stack_range_mode') == fs.MODE
        support_enabled = job.get("support_mode", "legacy") == rs.MODE
        if support_enabled:
            if job.get("support_version") != rs.SUPPORT_VERSION:
                raise ValueError("插值支持算法版本已变化，请新建作业")
            rs.require_engine(job["engine"])
        stage, work, products = directory / "stage", directory / "work", directory / "products"
        for p in (stage, work, products):
            p.mkdir(exist_ok=True)
        if 'stack' in job['checkpoints']:
            sr.verify_receipt(work, job)
        _update(directory, state="running", stage="import", progress=5, error=None, message="准备独立工作副本")
        timer.phase("input_stage")
        check()
        _stage_inputs(job, stage, check)
        fits_contract = None
        if job['input_kind'] == 'fits':
            from .fits_input import verify_contract
            fits_contract = verify_contract(job, [stage / f"frame{i:05d}{Path(s['path']).suffix.lower()}"
                                                   for i, s in enumerate(job['inputs'], 1)])
        if job["mask"]:
            mask_target = directory / "sky_mask.png"
            if not mask_target.exists():
                shutil.copyfile(job["mask"]["path"], mask_target)
            if _sha(mask_target) != job["mask"]["sha256"]:
                raise ValueError("天空蒙版已改变，请新建作业")
            for stem, entry in (job.get("frame_masks") or {}).get("files", {}).items():
                target = directory / "frame_masks" / f"{stem}.png"
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copyfile(entry["path"], target)
                if _sha(target) != entry["sha256"]:
                    raise ValueError(f"逐帧天空蒙版 {stem} 已改变，请新建作业")
        checkpoints = job["checkpoints"]
        timer.phase("convert")
        if "convert" not in checkpoints:
            _update(directory, stage="convert", progress=15, message="Siril 正在转换素材")
            conversion = "convertraw light -debayer -out=../work" if job["input_kind"] == "raw" else "convert light -out=../work"
            for existing in list(work.iterdir()):
                if existing.is_file():
                    existing.unlink()  # only files in this generated job's isolated work directory
            conversion_log = directory / "01_convert.log"
            log_start = conversion_log.stat().st_size if conversion_log.exists() else 0
            try:
                cfa_audit = None
                if fits_contract and fits_contract['layout'] == 'cfa':
                    from .fits_input import convert_cfa
                    cfa_audit = convert_cfa(job['engine'], directory, sorted(stage.glob('frame*')), work,
                                            'light', fits_contract, check, timed_script)
                else:
                    timed_script(exe, directory, "01_convert", ["setext fit", "set32bits", "setcompress 0", "cd stage", conversion], check)
            except Cancelled:
                raise
            except RuntimeError as exc:
                if fits_contract and fits_contract['layout'] == 'cfa':
                    raise
                raise RuntimeError(conversion_failure_message(conversion_log, str(exc), start_offset=log_start)) from exc
            converted = sorted(work.glob("light_*.fit"))
            if len(converted) != len(job["inputs"]):
                raise RuntimeError(conversion_failure_message(
                    conversion_log,
                    f"Siril 未生成完整转换序列：预期 {len(job['inputs'])} 帧，实际找到 {len(converted)} 帧",
                    start_offset=log_start))
            if job.get('image_rotation', 0):
                from .orientation import rotate_fits
                for path in converted:
                    rotate_fits(path, path, job['image_rotation'], check=check)
                # Conversion can describe the old dimensions in a .seq file.
                # Let subsequent registration build it from the rotated FITS.
                (work / SEQUENCE_FILE).unlink(missing_ok=True)
            shape = _validate_frame(converted[0])
            for p in converted:
                _validate_frame(p, shape)
            # Sequence metadata is mutable during registration; checkpoint pixels only.
            sequence = work / SEQUENCE_FILE
            immutable_sequence = work / SEQUENCE_SNAPSHOT
            if sequence.is_file():
                shutil.copyfile(sequence, immutable_sequence)
            if job["mask"]:
                timer.call("source_sky_validity", _ensure_sky_validity, directory, work, converted, shape, check, checkpoint, load_job(directory)["checkpoints"],
                           timer=timer)
            checkpoint("convert", converted + ([immutable_sequence] if immutable_sequence.is_file() else [])
                       + ([work / "sky_validity.json"] if job["mask"] else []) + ([cfa_audit] if cfa_audit else []))
        converted = sorted(work.glob("light_*.fit"))
        if len(converted) != len(job["inputs"]):
            raise RuntimeError("转换检查点文件缺失，请创建新作业")
        from . import sky_validity as sv
        validity_record = _read_sky_validity(work, job) if job["mask"] else None
        if job["mask"] and validity_record is None and "stack" not in checkpoints:
            # a job converted before sky validity existed (or interrupted before the record): mask now, re-describe the frames
            shape = _validate_frame(converted[0])
            timer.call("source_sky_validity", _ensure_sky_validity, directory, work, converted, shape, check, checkpoint, checkpoints,
                       timer=timer, rezero=True)
            validity_record = sv.read_record(work)
            immutable_sequence = work / SEQUENCE_SNAPSHOT
            checkpoint("convert", converted + ([immutable_sequence] if immutable_sequence.is_file() else []) + [work / sv.VALIDITY_RECORD])
        from .registration_calibration import (DISTORTION_JSON, calibrate, load_calibration, native_from_undistorted,
                                               write_registration_sequence)
        calibration = None
        sequence_registration = None
        timer.phase("distortion")
        if "stack" not in checkpoints:
            for existing in list(work.glob("r_light*")) + [work / "sky_master.fit", work / "sky_master_undistorted.fit"]:
                if existing.is_file():
                    existing.unlink()
            if range_enabled:
                for stale in [work / sr.RECORD, work / sr.VALID, *work.glob('range-master-*.tmp')]:
                    stale.unlink(missing_ok=True)
            if finite_enabled:
                for stale in [work / fs.NVALID, work / fs.NUSED, *work.glob('frame_support_light_*.npy'),
                              *work.glob('*.pending')]:
                    stale.unlink(missing_ok=True)
            if support_enabled:
                # P1c F-2: a failed or cancelled export pool may leave up to `workers` sidecar temporaries
                for stale in sorted([*work.glob('frame_support_light_*.npy.tmp'), *work.glob('sky_support_*.npy.tmp')]):
                    stale.unlink(missing_ok=True)
            if (work / O1_MARKER).exists():
                # P1c O-1's marker protocol: an earlier attempt's overlapped scan left output; without the marker
                # (every serial history) nothing changes
                _o1_require_clear(work, directory)
            if "distortion" in checkpoints and (work / DISTORTION_JSON).is_file():
                calibration = load_calibration(work)
            if calibration is None:
                # Start the probe registration from a sequence without stale transforms or reference.
                if (work / SEQUENCE_SNAPSHOT).is_file():
                    with hashing.writing(work / SEQUENCE_FILE):
                        shutil.copyfile(work / SEQUENCE_SNAPSHOT, work / SEQUENCE_FILE)
                else:
                    (work / SEQUENCE_FILE).unlink(missing_ok=True)
                _update(directory, stage="calibrate", progress=30, message="Siril 正在探测配准，随后拟合镜头畸变")
                for stale in (work / DISTORTION_JSON, work / DISTORTION_WCS):
                    stale.unlink(missing_ok=True)
                timed_script(exe, directory, "02_probe", list(PROBE_SCRIPT), check)
                check()
                calibration = timer.call("distortion_fit", calibrate, work, check=check)   # cancellable inside the fit (RUN_LOG §24)
                check()
                distortion_files = [work / DISTORTION_JSON]
                if calibration.get("status") == "fitted" and (work / DISTORTION_WCS).is_file():
                    distortion_files.append(work / DISTORTION_WCS)
                checkpoint("distortion", distortion_files)
            # Retrying an interrupted stack must not leave excluded old exports
            # available to Siril's regenerated output sequence or transient scan.
            for stale in work.glob("r_light_*.fit"):
                if stale.stem[8:].isdigit():
                    stale.unlink()
            (work / "r_light_.seq").unlink(missing_ok=True)
            registration = calibration.get("registration") or {}
            own_registration = (calibration.get("status") == "fitted" and (work / DISTORTION_WCS).is_file()
                                and registration.get("registered", 0) >= 3 and bool(calibration.get("sequence_template")))
            timer.phase("registration_stack")
            if own_registration:
                # Our matrices (every frame re-associated under the fitted term) go into the sequence file; Siril only
                # applies them.  Rewritten on every attempt: Siril re-saves the file after seqapplyreg.
                sequence_registration = write_registration_sequence(work, calibration)
                _update(directory, stage="stack", progress=40,
                        message=f"Siril 正在按作业自拟的镜头畸变与配准矩阵导出 {sequence_registration['registered']} 帧并堆栈")
                if support_enabled:
                    _supported_stack(exe, directory, work, job, calibration, True, validity_record, check, runner=timed_script, timer=timer,
                                     note=note_stack_settings, finite_write=finite_write)
                elif range_enabled:
                    timed_script(exe, directory, '02_export', ['cd work', *registration_commands(DISTORTION_WCS, True)], check)
                    sr.run_protected_stack(job['engine'], directory, work, check,
                                           _range_runner(timed_script, job, note=note_stack_settings),
                                           contract=job['stack_range_contract'])
                else:
                    export_then_stack(export_script(DISTORTION_WCS, own_registration=True))   # P1b-3 (3a)
            else:
                _update(directory, stage="stack", progress=40, message="Siril 正在对齐星空并堆栈")
                (work / SEQUENCE_FILE).unlink(missing_ok=True)   # Siril's own two-pass registration recomputes the transforms
                gate = calibration.get("quality_gate") or {}
                excluded = list(gate.get("excluded_frames") or []) if gate.get("applied") else []
                if excluded or support_enabled or range_enabled:
                    from .sequence_selection import apply_exclusions
                    timed_script(exe, directory, "02_register_fallback", list(PROBE_SCRIPT), check)
                    selection = apply_exclusions(work / SEQUENCE_FILE, excluded)
                    atomic_json(work / "quality_gate_application.json", {"source": REGISTRATION_SOURCE_SIRIL, **selection})
                    if support_enabled:
                        _supported_stack(exe, directory, work, job, calibration, False, validity_record, check, runner=timed_script, timer=timer,
                                         note=note_stack_settings, finite_write=finite_write)
                    elif range_enabled:
                        timed_script(exe, directory, '02_export', ['cd work', 'seqapplyreg light -filter-included'], check)
                        sr.run_protected_stack(job['engine'], directory, work, check,
                                               _range_runner(timed_script, job, note=note_stack_settings),
                                               contract=job['stack_range_contract'])
                    else:
                        export_then_stack(["cd work", "seqapplyreg light -filter-included"])   # P1b-3 (3b)
                else:
                    export_then_stack(export_script(None))   # P1b-3 (3c): register + seqapplyreg, then the stack
            use_distortion = own_registration
            timer.phase("native_remap_support")
            stack_files = [work / "sky_master.fit"]
            if use_distortion:
                check()
                _update(directory, stage="stack", progress=60, message="星空母版变换回原生几何")
                undistorted = work / "sky_master_undistorted.fit"
                native_shape = _validate_frame(converted[0])
                _validate_master(work / "sky_master.fit", native_shape)   # Siril's stack, undistorted grid
                with hashing.writing(work / "sky_master.fit"), hashing.writing(undistorted):
                    os.replace(work / "sky_master.fit", undistorted)
                native_workers = native_unit_bytes = 0
                if support_enabled:
                    # Efficiency P1c G-4t: the native mask's row blocks on the frame pool (quick route only; the
                    # reference route named by the support record runs no pool and leaves no note)
                    native_height, native_width = native_shape[1], native_shape[2]
                    native_decision = {}
                    native_workers, native_unit_bytes, native_blocks = _native_mask_workers(
                        native_height, native_width, native_decision)
                # Efficiency P1c O-1: after the rename, B may start beside the rest of this stage and the diagnostics
                o1 = o1_begin(own=True, stack_done=False, calibration=calibration, validity_record=validity_record,
                              native_workers=native_workers, native_unit_bytes=native_unit_bytes)
                if support_enabled:
                    from . import performance
                    if o1 is not None:
                        o1.release("native_mask")     # G-4t's pool reserves its own workers while it runs
                    pool_stats = {"decision": native_decision}
                    support_record = rs.finish_stack(work, undistorted, _grid_model(calibration, True), check=check,
                                                     native_workers=native_workers,
                                                     native_unit_bytes=native_unit_bytes, pool_stats=pool_stats,
                                                     operation=timer.operation,
                                                     **({'explicit_counts': True} if finite_enabled else {}))
                    if support_record.get("implementation") == rs.SUPPORT_IMPLEMENTATION:
                        performance.note_pool("native_mask", pool_stats)
                        resources.shadow_pool("py.native_mask", actual=pool_stats.get("workers"),
                                              items=native_blocks, width=native_width)
                native = timer.call("native_remap", native_from_undistorted, read_rgb(undistorted), calibration["parameters"])
                if support_enabled:
                    native_supported = rs.load_array(work, support_record["native_valid"])
                    native[~native_supported[::-1]] = 0.0
                from astropy.io import fits as _fits
                stack_header = _fits.getheader(undistorted)
                carried = {key: stack_header[key] for key in ("STACKCNT", "EXPTIME", "DATE-OBS") if key in stack_header}
                write_linear(work / "sky_master.fit", native, "Sky stack (undistorted reference grid) warped back to native geometry, cubic",
                             extra=carried)
                stack_files.append(undistorted)
            else:
                # Efficiency P1c O-1: not with Siril's fallback registration (its grid master is sky_master.fit, which
                # the strict support rewrites in place just below): noted only
                o1_begin(own=False, stack_done=False, calibration=calibration, validity_record=validity_record)
                if support_enabled:
                    support_record = rs.finish_stack(work, work / "sky_master.fit", None, check=check,
                                                     **({'explicit_counts': True} if finite_enabled else {}))
                    sv.apply_validity_to_fits(work / "sky_master.fit", rs.load_array(work, support_record["native_valid"]), check=check)
            if support_enabled:
                support_record["native_master_sha256"] = rs.sha256(work / "sky_master.fit")
                support_record["mask_sha256"] = validity_record["mask_sha256"]
                atomic_json(work / rs.RECORD, support_record)
                stack_files.extend(work / name for name in (rs.RECORD, rs.COUNTS, rs.STACK_VALID, rs.NATIVE_VALID, rs.EXPORT_SEQUENCE))
            _validate_master(work / "sky_master.fit", _validate_frame(converted[0]))
            if overlap[0] is not None:
                # O-1: A's planned part (the native remap and the master's check) is allocated or freed now; what A
                # still holds the OS figures show, so its piece goes (a pool deciding from here on would count it twice)
                overlap[0].release("a")
            if validity_record is not None:
                check()
                _update(directory, stage="stack", progress=68, message="按各帧配准计算天空有效帧数图")
                coverage_cap = None
                if overlap[0] is not None:            # O-1: the plan's coverage share is the pool's to reserve now
                    overlap[0].release("coverage")
                    coverage_cap = overlap[0].plan["cov_plan"] or None
                stack_files.append(timer.call("coverage", _write_coverage, work, directory, calibration, own_registration, validity_record, check, job=job,
                                              worker_cap=coverage_cap))
            if range_enabled:
                sr.finalize_receipt(work, contract=job['stack_range_contract'])
                stack_files.extend([work / sr.RECORD, work / sr.VALID])
            if finite_enabled:
                stack_files.extend(_finite_stack_files(work))
            checkpoint("stack", stack_files)
        timer.phase("diagnostics")
        if calibration is None:
            calibration = load_calibration(work) or {"status": "unavailable", "reason": "no calibration record"}
        check()
        from astropy.io import fits
        header = fits.getheader(work / "sky_master.fit")
        stacked_count = header.get("STACKCNT")
        if stacked_count is not None and stacked_count < 3:
            raise ValueError("实际参与堆栈不足 3 帧，请检查配准日志")
        skipped_count = len(converted) - stacked_count if stacked_count is not None else None
        gate = calibration.get("quality_gate") or {}
        gate_excluded = list(gate.get("excluded_frames") or []) if used_gate(calibration, work) else []
        if stacked_count is None:
            participation = f"天空实际参与帧数未知（输入 {len(converted)} 帧），未参与数量未知。"
        else:
            participation = f"天空堆栈实际使用 {stacked_count}／{len(converted)} 帧，未参与 {skipped_count} 帧。"
            if skipped_count > 0 and gate_excluded and skipped_count == len(gate_excluded):
                participation += f"未参与的是像质门槛排除的 {'、'.join(gate_excluded)}（{'；'.join(e['stem'] + '：' + '，'.join(e['reasons']) for e in gate.get('excluded', []))}）。"
            elif skipped_count > 0:
                participation += "请检查配准日志。"
        used = calibration.get("status") == "fitted" and (work / "sky_master_undistorted.fit").is_file()
        distortion_record = {k: calibration.get(k) for k in ("status", "reason", "parameters", "corner_shift_px", "magnification",
                                                             "validation_corner_rms_median_px", "frames_dropped_by_siril", "fit_frames",
                                                             "calibration_version", "seconds") if k in calibration}
        reassociation = calibration.get("reassociation") or {}
        distortion_record["reassociation"] = {k: reassociation.get(k) for k in ("matched_fraction_median", "corner_rms_median_px",
                                                                                 "recovered_dropped_frames") if k in reassociation}
        distortion_record["applied"] = bool(used)
        quality_record = ({k: gate.get(k) for k in ("version", "enabled", "applied", "min_frames", "thresholds", "frames_with_quality", "medians",
                                                     "excluded", "excluded_frames", "flagged_not_applied", "reference_flagged")}
                          if gate else {"applied": False, "reason": "no calibration"})
        quality_record["in_effect"] = bool(gate_excluded) or (bool(gate.get("applied")) and used_gate(calibration, work))
        quality_record["frames"] = {stem: {"ratios": f.get("ratios"), "flags": f.get("flags")} for stem, f in (gate.get("frames") or {}).items()}
        registration = calibration.get("registration") or {}
        registration_record = {"source": REGISTRATION_SOURCE_OWN if used else REGISTRATION_SOURCE_SIRIL,
                               "commands": registration_commands(DISTORTION_WCS if used else None, own_registration=used)}
        if support_enabled:
            registration_record["commands"] = ([rs.EXPORT_COMMAND] if used else [PROBE_SCRIPT[1], rs.EXPORT_COMMAND])
            registration_record["support_mode"] = rs.MODE
        if not used and calibration.get("status") == "fitted":
            registration_record["fallback_reason"] = (f"calibration registered only {registration.get('registered', 0)} frames (need 3)"
                                                      if registration else "calibration has no per-frame registration")
        elif not used:
            registration_record["fallback_reason"] = f"distortion {calibration.get('status')}: {calibration.get('reason')}"
        if used:
            registration_record.update({k: registration.get(k) for k in ("registered", "registered_frames", "unregistered_frames",
                                                                         "quality_excluded_frames",
                                                                         "validation_overall_rms_median_px", "validation_corner_rms_median_px",
                                                                         "max_rms_px", "min_pairs") if k in registration})
            registration_record["frames"] = {stem: {k: f.get(k) for k in ("span", "status", "reason", "pairs", "validation")}
                                             for stem, f in (registration.get("frames") or {}).items()}
            if sequence_registration is None and (work / SEQUENCE_FILE).is_file():
                sequence_registration = {"file": str(work / SEQUENCE_FILE), "registered": registration.get("registered")}
            registration_record["sequence"] = sequence_registration
        exposure_record = exposure_diagnostics(work, converted, fits_contract, stacked_count)
        range_diagnostics = {}
        if range_enabled:
            receipt = json.loads((work / sr.RECORD).read_text(encoding='utf-8'))
            range_diagnostics = {'stack_range': {'mode': receipt['contract']['mode'], 'version': receipt['contract']['version'],
                'state': receipt['state'], **({'divisor': receipt['divisor']} if 'divisor' in receipt else {}),
                **({'counts_summary': receipt['counts_summary'], 'stack_engine': receipt['stack_engine']}
                   if finite_enabled else {}), 'seconds': receipt['seconds'],
                **({'lower_endpoint_channels': receipt['lower_endpoint_channels']}
                   if 'lower_endpoint_channels' in receipt else {}),
                'backend': receipt['contract']['backend'], 'selected_count': len(receipt['selected']),
                'zero_semantics': receipt['contract']['zero_semantics'],
                'receipt': str(work / sr.RECORD)}}
        _update(directory, diagnostics={"input_count": len(converted), "stacked_count": stacked_count,
                "exposure": exposure_record, **range_diagnostics,
                "skipped_count": skipped_count, **sequence_reference(work / SEQUENCE_FILE, job["inputs"]),
                "registration": registration_record["commands"], "registration_source": registration_record["source"],
                "registration_detail": registration_record, "distortion": distortion_record, "quality_gate": quality_record,
                "sky_geometry": "native_via_undistorted_stack" if used else "native_direct", "width": header["NAXIS1"],
                "height": header["NAXIS2"], "roworder": "BOTTOM-UP", "camera_color_calibrated": False,
                "sky_validity": ({"applied": True, **{k: validity_record.get(k) for k in ("version", "source", "dilation_px", "invalid_fraction", "frames_masked", "own_mask_count")}}
                                 if validity_record else {"applied": False, "reason": "no sky mask" if not job["mask"] else "frames converted before sky validity"})})
        # bright-trail candidates (RUN_LOG §28): the exported frames against the master on the stack grid, recorded once
        timer.phase("transient_scan")
        if not o1_noted[0]:                   # efficiency P1c O-1: an attempt resumed past the stack only notes why not
            o1_begin(own=used, stack_done=True, calibration=calibration, validity_record=validity_record)
        transients_record = None
        transients_error = None
        if transient_mode != "off":
            from . import transient_stage as ts
            validated_marks = _normalise_marks(job.get("transient_marks") or [], job["inputs"], work=work)
            checkpointed = "transients" in load_job(directory)["checkpoints"]
            if overlap[0] is None and (work / O1_MARKER).exists():
                # P1c O-1's marker protocol, serial and overlapped alike: an earlier attempt's overlapped scan left
                # output without a transients checkpoint (else only the marker is stale)
                if checkpointed:
                    try:                      # as _Overlap.keep: a marker that cannot go stays, logged
                        _o1_retry(lambda: (work / O1_MARKER).unlink(missing_ok=True))
                    except OSError as exc:
                        _o1_log(directory, f"O-1: could not remove the stale marker {O1_MARKER} "
                                           f"({type(exc).__name__}: {exc}); it stays")
                else:
                    _o1_require_clear(work, directory)
            # Internal resume reads the detection checkpoint; public readers
            # only see the immutable revision published with the finished job.
            transients_record = ts.tt.read_candidates(work / ts.TRANSIENTS_DIR) if checkpointed else None
            if transients_record is None:
                check()
                _update(directory, stage="transients", progress=70, message="在各帧配准帧减母版的残差里寻找亮迹候选")
                height_px = int(header["NAXIS2"])
                o1 = overlap[0]
                if o1 is not None:            # O-1's join guards, outside the inner try (prereg §3.6)
                    mismatch = o1.mismatch(used=used, height_px=height_px, calibration=calibration,
                                           inputs=job["inputs"], marks=validated_marks)
                    if mismatch is not None:
                        overlap[0] = None
                        o1.discard(mismatch)
                        o1 = None
                try:
                    if o1 is not None:
                        with timer.operation("transient_join"):
                            transients_record = o1.result(check)
                    else:
                        transients_record = _detect_transients(work, calibration, used, height_px, job["inputs"],
                                                               validated_marks, check=check, timer=timer)
                    checkpoint("transients", [work / ts.TRANSIENTS_DIR / ts.CANDIDATES_FILE])
                    if o1 is not None:
                        overlap[0] = None
                        o1.keep("joined")
                except (Cancelled, hashing.HashLedgerMismatch):   # shadow mode: a wrong ledger answer fails the job
                    raise
                except Exception as exc:   # retain the base composite on a detection failure, visibly
                    with (directory / "worker.log").open("a", encoding="utf-8") as handle:
                        traceback.print_exc(file=handle)
                    transients_error = f"{type(exc).__name__}: {exc}"
                    transients_record = None
                    if o1 is not None:        # B's files stay as the serial scan's would; the marker goes
                        overlap[0] = None
                        o1.keep("recorded")
        timer.phase("base_composition")
        sky = read_rgb(work / "sky_master.fit")
        _update(directory, stage="compose", progress=75, message="保存母版与准备预览")
        artifacts = [{"path": str(work / "sky_master.fit"), "kind": "fits", "label": "星空线性母版"}]
        composed = _compose_sky(directory, job, work, calibration, used, validity_record, sky, converted, header, check)
        output, sky_validity_record = composed["output"], composed["record"]
        artifacts.extend(composed["artifacts"])
        # Build a new export directory on each attempt, then publish its artifact list atomically.
        export_dir = products / uuid.uuid4().hex[:12]
        export_dir.mkdir()
        composite_path = export_dir / "master_linear.fit"
        write_linear(composite_path, output, "Prototype composite; no calibration or camera characterization")
        if composed["weight"] is not None:
            from .compose_evidence import write_weight
            # Record the actual weight applied to this base composite before
            # the single publication commit; validation must not guess it.
            sky_validity_record = dict(sky_validity_record or {},
                                       compose_weight=write_weight(export_dir, composed["weight"], composite_path))
        artifacts.append({"path": str(composite_path), "kind": "fits", "label": "合成线性母版"})
        _update(directory, stage="compose", progress=80, message="生成后期母版与成片")
        timer.phase("base_color")
        colour_artifacts, colour_record = _colour_products(directory, job, work, output, export_dir, composite_path, check)
        artifacts.extend(colour_artifacts)
        check()
        del output
        revision = None
        timer.phase("transient_export")
        if transient_mode == "off":
            transients_diag = {"status": "skipped", "reason": "not_requested", "mode": "off", "composite": None}
        elif transients_record is not None and transient_mode == "review":
            revision = transient_revision.prepare(export_dir, transients_record)
            transients_diag = {"status": "needs_review", "mode": "review", "composite": None,
                               "frames_scanned": transients_record.get("frames_scanned"),
                               "summary": transients_record.get("summary"), "seconds": transients_record.get("seconds"),
                               "record": revision["record"], "revision_id": revision["id"], "record_sha256": revision["sha256"],
                               "message": "候选与处置均仅为建议；请复核后确认重合成，尚未加回亮迹"}
        elif transients_record is not None:
            _update(directory, stage="compose", progress=90, message="按处置策略合成保留的亮迹")
            transients_diag, transient_artifacts = _export_transients(directory, job, work, calibration, used, validity_record, sky, converted,
                                                                      header, check, transients_record, export_dir, composed["weight"])
            artifacts.extend(transient_artifacts)
            revision = transient_revision.prepare(export_dir, transients_record)
            transients_diag.update(record=revision["record"], revision_id=revision["id"], record_sha256=revision["sha256"])
        else:
            transients_diag = {"status": "failed", "error": transients_error}
        current = load_job(directory).get("diagnostics") or {}
        timer.phase("publication")
        diagnostics_update = {**current, "transients": transients_diag, "color_export": colour_record,
                              "first_frame_geometry": first_frame_geometry(work, calibration, used,
                                  (int(header["NAXIS2"]), int(header["NAXIS1"])), job["inputs"])}
        if sky_validity_record is not None:
            diagnostics_update["sky_validity"] = {"applied": True, **sky_validity_record}
        check()
        saved = "线性母版、后期母版与成片已保存" if colour_record.get("state") == "succeeded" else "线性母版、16位预览渲染与三种色调已保存"
        if colour_record.get("state") == "fallback_legacy":
            saved += "（后期母版/成片写出失败，已回退旧产物，详见诊断 color_export）"
        # After the last write in the job directory and before the launcher can see "succeeded".
        _export_hash_handover(directory)
        _update(directory, diagnostics=timed_diagnostics(diagnostics_update, "succeeded"), transient_revision=revision, state="succeeded", stage="complete", progress=100,
                message=f"处理完成：{participation}{_transients_message(transients_diag)}{saved}", artifacts=artifacts)
    except Cancelled:
        close_overlap("the main thread was cancelled")   # P1c O-1: B is joined first, its output removed
        _discard_hash_handover()
        _update(directory, state="cancelled", stage="cancelled", message="已停止，可从完整阶段恢复",
                diagnostics=timed_diagnostics(load_job(directory).get("diagnostics"), "cancelled"))
    except Exception as exc:
        close_overlap(f"the main thread failed: {type(exc).__name__}")
        _discard_hash_handover()
        with (directory / "worker.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        _update(directory, state="failed", message="处理失败，可查看日志或恢复", error=str(exc),
                diagnostics=timed_diagnostics(load_job(directory).get("diagnostics"), "failed"))
    finally:
        close_overlap("the worker is stopping")          # KeyboardInterrupt / SystemExit; else nothing is open


def _read_sky_validity(work: Path, job: dict) -> dict | None:
    from . import sky_validity as sv
    record = sv.read_record(work)
    previously_applied = ((job.get("diagnostics") or {}).get("sky_validity") or {}).get("applied")
    if record is None and (job.get("sky_validity_required") or previously_applied):
        raise ValueError("天空有效区记录缺失，不能按未屏蔽地景的旧流程合成；请新建作业")
    return record


def _ensure_sky_validity(directory: Path, work: Path, converted: list, shape, check, checkpoint, checkpoints, *,
                         timer=None, rezero=False) -> dict:
    """Ground master from the unmasked frames, then zero every frame's invalid sky in place (RUN_LOG §25).

    Efficiency P1c C-1: the median's integer-network blocks run on the frame pool (its own thread decision), in the
    timer operation ``foreground_median`` of ``timer`` (the job's StageTimer). C-3: the frames are zeroed on the frame
    pool (_sky_zero, timer operation ``sky_zero``), except on the re-zeroing route of a job converted before sky
    validity existed (``rezero``): its frames already have a convert checkpoint whose digests a resume checks, so a
    frame zeroed past a stop could make a resumable job unresumable; that route keeps the serial loop."""
    from . import sky_validity as sv
    from .imaging import median_foreground, write_linear
    planes, height, width = shape
    mask_path = directory / "sky_mask.png"
    job = load_job(directory)
    source = _validity_source(directory, job, width, height, SKY_VALIDITY_DILATION_PX)
    ground_path = work / "foreground_master.fit"
    if "foreground" not in checkpoints or not ground_path.is_file():
        _update(directory, stage="convert", progress=20, message="从未屏蔽的原帧生成地景母版")
        ground_path.unlink(missing_ok=True)
        ground = median_foreground(converted, check, workers=None, timer=timer)
        write_linear(ground_path, ground, "Unregistered temporal median of the converted frames, taken before the sky masking; fixed tripod only")
        checkpoint("foreground", [ground_path])
        del ground   # efficiency P1c U-M14: the 0.55 GB median is not held through the zeroing of every frame
    _update(directory, stage="convert", progress=25, message=f"屏蔽各帧地景（天空蒙版外扩 {SKY_VALIDITY_DILATION_PX} px），天空堆栈只用有效天空")
    if rezero:
        stats = _sky_zero_reference(converted, source, check)
    else:
        stats = _sky_zero(converted, source, check, width, timer=timer)
    record = {"version": sv.SKY_VALIDITY_VERSION, **source.describe(), "mask_file": mask_path.name, "mask_sha256": sv.sha256_file(mask_path),
              "frames_masked": len(stats), "zeroed_pixels_per_frame": {s_["file"]: s_["zeroed_pixels"] for s_ in stats}, "applied_at": now()}
    sv.write_record(work, record)
    return record


# Efficiency P1c C-3 (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.1): sky zeroing, one frame per frame-pool unit.
SKY_ZERO_WORKER_CAP = 4
SKY_ZERO_ROWS = 256              # sky_validity.apply_validity_to_fits's row block (its default)


def _sky_zero_unit_bytes(width: int) -> int:
    """One zeroing unit's memory (P1c R10/R17): per 256-row block of a float32 frame (the upper bound; 16-bit frames
    need about half) the read block, its zeroed copy, the digest thread's queue of written blocks and the one it
    hashes, and the block's boolean masks -- 63.6 MB at W = 8280. Payloads are a path and a reference to the
    source's validity array, results a small dict."""
    from . import sky_validity as sv
    return SKY_ZERO_ROWS * width * (4 * (2 + sv._DIGEST_QUEUE + 1) + 2)


def _sky_zero_reference(converted: list, source, check) -> list:
    """The pre-P1c serial loop, verbatim: the C-3 tests' reference, and the re-zeroing route (one thread)."""
    from . import sky_validity as sv
    stats = []
    for path in converted:
        check()
        stats.append(sv.apply_validity_to_fits(path, source.for_frame(path.stem), check=check))
    return stats


def _sky_zero(converted: list, source, check, width: int, *, workers: int | None = None, inflight=None,
              timer=None) -> list:
    """Zero every frame's invalid sky in place on the frame pool (efficiency P1c C-3); the per-frame statistics in
    frame order, as _sky_zero_reference returns them.

    The validity lookup (``source.for_frame``, which loads and caches a per-frame mask) stays on the calling thread
    in frame order (prepare); a unit is sky_validity.apply_validity_to_fits on one frame, unchanged, with the unit's
    probe as its per-block check (R18); the statistics are collected in frame order (on_result). Each unit reads and
    rewrites only its own file (integer work, one division in the statistics) and records its written digest with
    the ledger by path, so files, digests and the record do not depend on ``workers`` (None: frame_workers with cap
    SKY_ZERO_WORKER_CAP, no BLAS in a unit). A failure stops at the lowest frame; frames after it may already be
    zeroed (prereg §5 item 2): this route has no convert checkpoint, so a resume converts again."""
    from . import performance, resources
    from . import sky_validity as sv
    unit_bytes = _sky_zero_unit_bytes(width)
    if workers is None:
        workers = performance.frame_workers(len(converted), unit_bytes, SKY_ZERO_WORKER_CAP, blas=False)
    stats = []

    def prepare(i, path, _check):
        # The pool polls check() right before each prepare. A frame's own mask (PerFrameMaskSource) is loaded and
        # dilated here, in the frozen sky_validity.py, without a probe: 0.6-1.1 s at 45.7 MP, the one exception
        # segment of C-3 (prereg §3.1, §5 item 5), as long as in the serial loop. Not in a unit: a lookup peaks at
        # about 0.55 GB, four concurrent ones would hold over 2 GB.
        return path, source.for_frame(path.stem)

    def unit(payload, frame):
        path, validity = payload
        return sv.apply_validity_to_fits(path, validity, check=frame.check)

    pool_stats = {}
    operation = timer.operation if timer is not None else contextlib.nullcontext
    with operation("sky_zero"):
        frame_pool.run_ordered(unit, list(converted), workers=workers, name="sky-zero", check=check, prepare=prepare,
                               on_result=lambda i, frame_stats: stats.append(frame_stats), inflight=inflight,
                               stats=pool_stats, unit_bytes=unit_bytes, blas=False)
    performance.note_pool("sky_zero", pool_stats)
    resources.shadow_pool("py.sky_zero", actual=pool_stats.get("workers"), items=len(converted), width=width)
    return stats


def _compose_sky(directory: Path, job: dict, work: Path, calibration, used: bool, validity_record, sky, converted: list, header, check,
                 *, weight=None) -> dict:
    """Sky master (bottom-up) + ground -> composite; returns the output, the sky-validity record, the ground artifacts and
    the sky weight used (so a second composite with the transient layer can reuse it)."""
    import numpy as np
    from . import sky_validity as sv
    from .imaging import read_rgb, write_linear, median_foreground, blend_with_mask, blend_with_weight
    output = sky
    artifacts = []
    sky_validity_record = None
    feather_mode = job.get("compose_feather_mode")
    if feather_mode not in (None, "canvas-inward-feather-1"):
        raise ValueError("未知的天空地景过渡方式")
    if job["mask"]:
        ground_path = work / "foreground_master.fit"
        if validity_record is None:
            if feather_mode is not None:
                raise ValueError("缺少天空有效区记录，不能应用画布过渡")
            # frames were never masked (job predates sky validity): the original compose
            if not ground_path.is_file():
                ground = median_foreground(converted, check)
                write_linear(ground_path, ground, "Unregistered temporal median; fixed tripod only")
            else:
                ground = read_rgb(ground_path)
            output = blend_with_mask(sky, ground, directory / "sky_mask.png")
        else:
            if not ground_path.is_file():
                raise RuntimeError("地景母版缺失而各帧已屏蔽地景，请创建新作业")
            ground = read_rgb(ground_path)
            height_px, width_px = int(header["NAXIS2"]), int(header["NAXIS1"])
            source = _validity_source(directory, job, width_px, height_px, int(validity_record["dilation_px"]))
            mask_td = source.static.mask if hasattr(source, "static") else source.mask
            if not (work / sv.COVERAGE_PNG).is_file():
                _write_coverage(work, directory, calibration, used, validity_record, check, job=job)
            compose_info = None
            reused_weight = weight is not None
            if weight is None:
                from . import registered_support as rs
                if job.get("support_mode") == rs.MODE:
                    supported, support_record = rs.read_native(work, (height_px, width_px))
                    if support_record.get("mask_sha256") != sv.sha256_file(directory / "sky_mask.png"):
                        raise ValueError("天空蒙版与插值支持记录不一致")
                    weight = mask_td.astype(np.float32) * supported
                    compose_info = {"interpolation_support": rs.MODE, "support_version": rs.SUPPORT_VERSION,
                                    "coverage_semantics": support_record["counts_semantics"],
                                    "unsupported_sky_pixels": int(np.count_nonzero((mask_td > 0) & ~supported)),
                                    "ambiguous_zero_or_nonfinite_pixels": support_record["ambiguous_zero_or_nonfinite_pixels"]}
                else:
                    transforms, model = registration_transforms(work, calibration, used, height_px)
                    check()
                    reference_stem = (calibration.get("registration") or {}).get("reference") or sequence_reference(work / SEQUENCE_FILE, job["inputs"]).get("reference_stem")
                    validity_ref = source.for_frame(reference_stem) if reference_stem else source.for_frame("")
                    weight, compose_info = sv.effective_sky_weight(mask_td, validity_ref, transforms, model, check=check, validity=source)
            if feather_mode is not None:
                from .region_blend import inward_weight
                if reused_weight:
                    feather_info = {"recipe": feather_mode, "reused_base_weight": True}
                else:
                    weight, feather_info = inward_weight(weight, check=check)
                compose_info = {**(compose_info or {}), "feather": feather_info}
            output = blend_with_weight(sky, ground, weight)
            coverage_summary = json.loads((work / "sky_coverage.json").read_text(encoding="utf-8")) if (work / "sky_coverage.json").is_file() else None
            sky_validity_record = {**{k: validity_record.get(k) for k in ("version", "source", "dilation_px", "invalid_fraction", "mask_sha256", "frames_masked",
                                                                          "frames_with_own_mask", "own_mask_count")},
                                   "coverage": coverage_summary, "compose": compose_info}
            label = (f"天空完整支持图（每 {SKY_COVERAGE_STEP} px 采样，0/1，不代表曝光数）" if job.get("support_mode") == "strict_siril_1_4_4"
                     else f"天空有效帧数图（每 {SKY_COVERAGE_STEP} px 采样，值 = 帧数）")
            artifacts.append({"path": str(work / sv.COVERAGE_PNG), "kind": "png", "label": label})
        artifacts.append({"path": str(ground_path), "kind": "fits", "label": "地景线性母版"})
    return {"output": output, "record": sky_validity_record, "artifacts": artifacts, "weight": weight}


def _colour_products(directory: Path, job: dict, work: Path, output, export_dir: Path, composite_path: Path, check) -> tuple[list, dict]:
    """Products for a composite (docs/COLOR_CONTRACT_PACKAGE_20260922.md §10): M + D for RAW-input jobs whose recorded
    reference frame yields a colour basis; the legacy display products otherwise.  A failure of the new writer never
    fails the job: the legacy products are written instead and the error is recorded (visible, not silent)."""
    from .imaging import export_products
    from .profiling import StageTimer
    timer = StageTimer()
    if job.get("input_kind") != "raw":
        artifacts = timer.call("legacy_export", export_products, output, export_dir, check)
        return artifacts, {"state": "legacy", "reason": "non-RAW inputs have no camera colour basis", "timings": timer.finish()}
    try:
        from . import color_export as ce
        first_selected = "color_adjustments" in job
        reference = (job["inputs"][0]["path"] if first_selected
                     else sequence_reference(work / SEQUENCE_FILE, job["inputs"]).get("reference_input"))
        identity = next((item for item in job['inputs'] if item['path'] == reference), {})
        if identity.get('sha256') and _sha(Path(reference)) != identity['sha256']:
            raise ValueError('颜色参考原片在处理期间发生变化')
        basis = timer.call("colour_basis", ce.job_colour_basis, reference)
        if identity.get('sha256') and _sha(Path(reference)) != identity['sha256']:
            raise ValueError('颜色参考原片在读取期间发生变化')
        mask_path = directory / "sky_mask.png" if job.get("mask") and (directory / "sky_mask.png").is_file() else None
        adjustments = {"color_adjustments": job["color_adjustments"]} if first_selected else {}
        artifacts, record = timer.call("master_and_photo_export", ce.export_job_products, output, export_dir, check,
                                      basis=basis, source_fits=composite_path, sky_mask_path=mask_path, **adjustments)
        return artifacts, {**record, "reference_input": reference,
                           "reference_selection": "first_selected_input" if first_selected else "geometric_reference",
                           "timings": timer.finish()}
    except (Cancelled, hashing.HashLedgerMismatch):   # shadow mode: a wrong ledger answer fails the job
        raise
    except Exception as exc:   # the composite is already saved; fall back to the legacy products and say so
        with (directory / "worker.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        from . import color_export as ce
        from . import master_export as me, photo_render as pr
        for name in (me.FILENAME, me.MASK_FILENAME, me.RECIPE_FILENAME, ce.D_TIFF_NAME, ce.D_JPEG_NAME, pr.RECORD_NAME, pr.FLAGS_NAME):
            partial = export_dir / name
            if partial.exists():
                partial.unlink()
        artifacts = timer.call("fallback_export", export_products, output, export_dir, check)
        return artifacts, {"state": "fallback_legacy", "error": f"{type(exc).__name__}: {exc}", "timings": timer.finish()}


def _export_transients(directory: Path, job: dict, work: Path, calibration, used: bool, validity_record, sky, converted: list, header, check,
                       record: dict, export_dir: Path, weight) -> tuple[dict, list]:
    """Second composite with the kept candidates added through the transient layer (RUN_LOG §28); ``sky`` is modified in
    place.  Returns the diagnostics entry and the artifacts (tagged ``group: transients`` so a recompose can replace them)."""
    from . import transient_stage as ts
    from . import transient_trails as tt
    from .imaging import write_linear
    height_px, width_px = int(header["NAXIS2"]), int(header["NAXIS1"])
    model = _grid_model(calibration, used)
    boxes, kept = ts.native_boxes(work, record, own_registration=used, model=model, native_shape=(height_px, width_px), check=check)
    record["composition_version"] = ts.STAGE_VERSION
    record["composition_rule"] = "union_same_source_footprints_then_add_distinct_exposures"
    diag = {"version": record.get("version"), "stage_version": record.get("stage_version"), "grid": record.get("grid"),
            "composition_version": record["composition_version"], "composition_rule": record["composition_rule"],
            "source_frames_kept": len({c["stem"] for c in kept}), "union_boxes": len(boxes),
            "frames_scanned": record.get("frames_scanned"), "seconds": record.get("seconds"), "policy": record.get("policy"),
            "summary": record.get("summary"), "kept": kept, "record": str(work / ts.TRANSIENTS_DIR / ts.CANDIDATES_FILE),
            "composite": None, "marks": len(record.get("marks") or [])}
    artifacts = []
    if not boxes:
        ts.finish_application(record, boxes, composite_path=None)
        diag["summary"] = record["summary"]
        return diag, artifacts
    tt.apply_boxes(sky, boxes)
    composed = _compose_sky(directory, job, work, calibration, used, validity_record, sky, converted, header, check, weight=weight)
    out_dir = export_dir / "transients"
    out_dir.mkdir(exist_ok=True)
    composite_path = out_dir / "master_linear_transients.fit"
    write_linear(composite_path, composed["output"], "Prototype composite plus the kept bright-trail candidates (transient layer); no calibration",
                 extra={"TRANSNT": len(kept)})
    artifacts.append({"path": str(composite_path), "kind": "fits", "label": f"含亮迹合成线性母版（保留 {len(kept)} 条）", "group": "transients"})
    colour_artifacts, colour_record = _colour_products(directory, job, work, composed["output"], out_dir, composite_path, check)
    for item in colour_artifacts:
        artifacts.append({**item, "label": "含亮迹：" + item["label"], "group": "transients"})
    diag["color_export"] = colour_record
    ts.finish_application(record, boxes, weight=composed["weight"], weight_known=not job["mask"] or composed["weight"] is not None,
                          native_warped=model is not None, composite_path=str(composite_path))
    diag["summary"] = record["summary"]
    diag["application_by_source"] = record.get("application_by_source")
    diag["composite"] = str(composite_path)
    return diag, artifacts


def _grid_model(calibration, used: bool):
    """The distortion model the exported frames and the undistorted master live in (None with Siril's two-pass fallback)."""
    if not used or not calibration or not calibration.get("parameters"):
        return None
    from .registration_distortion import RadialDistortion
    return RadialDistortion(**calibration["parameters"])


def _transients_message(diag: dict) -> str:
    if (diag or {}).get("status") == "skipped":
        return "亮迹功能未开启。"
    if (diag or {}).get("status") == "needs_review":
        return "亮迹候选待复核，尚未加回亮迹。"
    if (diag or {}).get("status") == "failed":
        return "亮迹候选检测失败（见 diagnostics.transients），成片不含亮迹层。"
    summary = (diag or {}).get("summary") or {}
    if not summary:
        return ""
    names = {"meteor": "流星", "satellite": "卫星", "aircraft": "飞机", "artifact": "伪迹", "unknown": "未知"}
    counts = summary.get("by_effective_label") or {}
    detail = "、".join(f"{names[k]} {counts.get(k, 0)}" for k in names if counts.get(k))
    kept = len(summary.get("kept") or [])
    measured = summary.get("measured", summary.get("candidates", 0))
    intent = summary.get("protection_intent") or {}
    withheld = int(intent.get("withheld_untrustworthy_default_keep", 0))
    review = int(intent.get("needs_review", 0)) - withheld
    applied = summary.get("application") or {}
    uncertain = sum(applied.get(key, 0) for key in ("partial", "unknown", "not_applied"))
    protection_message = f"保护请求中 {review} 条尚不能量测，{uncertain} 条未完全兑现或范围未证实，请查看保护记录。" if review or uncertain else ""
    if withheld:
        protection_message += f"{withheld} 条流星候选量测不可靠，未默认加回。"
    recall = summary.get("recall_review") or {}
    recall_message = recall.get("message") or "旧摘要尚未核实自动省略与未测量候选；请读取亮迹列表重新计算，当前不能保证没有遗漏。"
    if summary.get("candidates", 0) == 0 or measured == 0:
        return (protection_message or "未发现可量测的亮迹候选。") + recall_message
    return f"亮迹候选 {measured} 条（建议标签：{detail}），{kept} 条具备加回条件{'并另出含亮迹成片' if kept else ''}；{protection_message}{recall_message}标签与策略可用 starstack transients 修改。"


def _normalise_marks(marks, inputs: list[dict], *, work: Path | None = None) -> list[dict]:
    """Validate structure at admission and actual source-frame bounds before processing."""
    if marks is not None and not isinstance(marks, list):
        raise ValueError("亮迹标记须为列表")
    by_name = {}
    for i, source in enumerate(inputs, 1):
        for alias in (Path(source["path"]).name, Path(source["path"]).stem, str(source["path"]), f"light_{i:05d}"):
            by_name[alias] = f"light_{i:05d}"
    out, known = [], {}
    for mark in marks or []:
        values = validate_mark_values(mark)
        key = str(mark.get("stem") or mark.get("input") or "")
        stem = by_name.get(key) or by_name.get(Path(key).name)
        if stem is None:
            raise ValueError(f"亮迹标记的帧 {key!r} 不在作业输入里")
        if work is not None:
            from astropy.io import fits
            header = fits.getheader(work / f"{stem}.fit")
            values = validate_mark_values(mark, width=int(header["NAXIS1"]), height=int(header["NAXIS2"]))
        endpoints = sorted(((values["x0"], values["y0"]), (values["x1"], values["y1"])))
        identity = (stem, *endpoints, values["half_width_px"])
        if identity not in known:
            out.append({"stem": stem, "input": key, **values})
            known[identity] = out[-1]
        elif values.get("protect") is True:
            known[identity]["protect"] = True
    return out


def recompose_transients(job_dir: str, *, labels: dict | None = None, overrides: dict | None = None, policy: dict | None = None,
                         marks: list | None = None, notes: dict | None = None, layer_style: str | None = None,
                         expected_revision: dict | None = None, default_keep_rule: str | None = None,
                         reviewer: str | None = None) -> dict:
    """Change labels / overrides / policy or add manual marks on a finished job and rebuild only the transient layer and the
    'with transients' composite (the stack and the base composite are untouched).  Refused while the job is running.
    ``reviewer`` records who changed labels/overrides; ``default_keep_rule`` withholds untrustworthy default keeps."""
    # P1b-1: a missing coverage map is rebuilt on the frame pool (_compose_sky -> _write_coverage), so a malformed
    # AESTER_FRAME_WORKERS is refused here with the job's message, before any lock or export directory.
    frame_pool.validate_environment()
    directory = Path(job_dir).resolve()
    job = load_job(directory)
    if job["state"] != "succeeded":
        raise ValueError("只有已完成的作业可以修改亮迹处置")
    # The converted frames this needs may be gone: in-app/uninstaller cache cleanup (workflow marker) or gc.
    refusal = gc_refusal(job, "重做亮迹合成") or cache_cleanup_refusal(directory, "重做亮迹合成")
    if refusal:
        raise ValueError(refusal)
    if is_locked(directory / "worker.lock"):
        raise ValueError("作业正在运行，请稍后再试")
    with file_lock(directory / "worker.lock"):
        job = load_job(directory)  # pin the current revision only after acquiring the writer lock
        _verify_admission_options(job)
        if expected_revision is not None and job.get("transient_revision") != expected_revision:
            raise ValueError("亮迹记录已有新版本，请重新打开复核，避免覆盖其他决定")
        if job["state"] != "succeeded":
            raise ValueError("只有已完成的作业可以修改亮迹处置")
        refusal = gc_refusal(job, "重做亮迹合成") or cache_cleanup_refusal(directory, "重做亮迹合成")
        if refusal:
            raise ValueError(refusal)
        return _recompose_locked(directory, job, labels, overrides, policy, marks, notes, layer_style, default_keep_rule, reviewer)


def _recompose_locked(directory: Path, job: dict, labels, overrides, policy, marks, notes, layer_style=None,
                      default_keep_rule=None, reviewer=None) -> dict:
    from astropy.io import fits
    from . import sky_validity as sv
    from . import transient_stage as ts
    from .imaging import read_rgb
    from .registration_calibration import load_calibration
    work, products = directory / "work", directory / "products"
    from . import stack_range as sr
    # Timing marks only (no-ops unless a caller activated a StageTimer).
    profiling.switch(".receipt")
    sr.verify_receipt(work, job)
    record = ts.read_record(work, job=job)
    if record is None:
        raise ValueError("这个作业没有亮迹记录（在亮迹保护之前完成），请新建作业")
    if not (work / "sky_master.fit").is_file():
        raise ValueError("星空母版缺失，不能重做合成")
    calibration = load_calibration(work) or {"status": "unavailable"}
    used = calibration.get("status") == "fitted" and (work / "sky_master_undistorted.fit").is_file()
    header = fits.getheader(work / "sky_master.fit")
    height_px = int(header["NAXIS2"])

    def check():
        pass

    validated_marks = _normalise_marks(marks, job["inputs"], work=work) if marks else []
    known_marks = {ts._mark_key(m): m for m in record.get("marks", [])}
    if validated_marks and not any((labels, overrides, policy, notes, layer_style, default_keep_rule)) and all(
            ts._mark_key(m) in known_marks and (m.get("protect") is not True or known_marks[ts._mark_key(m)].get("protect") is True)
            for m in validated_marks):
        return job  # identical request (including reversed endpoints): no new export or revision
    export_dir = products / uuid.uuid4().hex
    export_dir.mkdir()
    profiling.switch(".record")
    if validated_marks:
        transforms, model = registration_transforms(work, calibration, used, height_px)
        record = ts.detect_marks(work, record, validated_marks, own_registration=used, transforms=transforms, model=model,
                                 inputs=job["inputs"], check=check, persist=False, review_dir=export_dir / "transients")
    record = ts.update_record(work, record, labels=labels, overrides=overrides, policy=policy, notes=notes, layer_style=layer_style, persist=False,
                              default_keep_rule=default_keep_rule, reviewer=reviewer)
    profiling.switch(".compose")
    converted = sorted(work.glob("light_*.fit"))
    validity_record = _read_sky_validity(work, job) if job["mask"] else None
    sky = read_rgb(work / "sky_master.fit")
    diag, artifacts = _export_transients(directory, job, work, calibration, used, validity_record, sky, converted, header, check, record,
                                         export_dir, None)
    profiling.switch(".write")
    revision = transient_revision.prepare(export_dir, record)
    diag.update(record=revision["record"], revision_id=revision["id"], record_sha256=revision["sha256"])
    kept_artifacts = [a for a in job.get("artifacts", []) if a.get("group") != "transients"]
    diagnostics = {**(job.get("diagnostics") or {}), "transients": {**diag, "recomposed_at": now()}}
    _update(directory, artifacts=kept_artifacts + artifacts, diagnostics=diagnostics, transient_revision=revision,
            transients_updated_at=now(), message=f"亮迹处置已更新：{_transients_message(diag)}")
    return load_job(directory)


def used_gate(calibration: dict, work: Path) -> bool:
    """Confirm exclusion flags on the sequence actually used, including fallback."""
    gate = calibration.get("quality_gate") or {}
    if not gate.get("applied"):
        return False
    try:
        selected = {f["stem"]: f["selected"] for f in read_seq(work / SEQUENCE_FILE)["images"]}
        return all(stem in selected and not selected[stem] for stem in gate.get("excluded_frames") or [])
    except (OSError, ValueError, SequenceError):
        return False


def _validity_source(directory: Path, job: dict, width: int, height: int, dilation_px: int):
    """The job's validity source: its static sky mask, plus per-frame masks when the job carries them (RUN_LOG §26)."""
    from . import sky_validity as sv
    static = sv.StaticMaskSource(sv.load_mask_top_down(directory / "sky_mask.png", width, height), dilation_px=dilation_px)
    files = (job.get("frame_masks") or {}).get("files") or {}
    if not files:
        return static
    return sv.PerFrameMaskSource(static, {stem: directory / "frame_masks" / f"{stem}.png" for stem in files}, width, height)


def _coverage_unit_bytes(height: int, width: int) -> int:
    """One coverage pool unit's bytes: COVERAGE_UNIT_BYTES (measured at 45.7 MP, the floor) scaled by the frame's
    pixels (P1b review pool-1)."""
    from . import sky_validity as sv
    return max(sv.COVERAGE_UNIT_BYTES, -(-sv.COVERAGE_UNIT_BYTES * int(height) * int(width) // sv.COVERAGE_UNIT_PIXELS))


def _write_coverage(work: Path, directory: Path, calibration, own_registration: bool, validity_record: dict, check, *,
                    job: dict | None = None, worker_cap: int | None = None) -> Path:
    """Sampled coverage map + summary next to the sky master (how many frames observed each reference pixel).

    ``worker_cap``: O-1's coverage plan while the trail scan overlaps (efficiency P1c §3.6), else None."""
    import numpy as np
    from . import sky_validity as sv
    from astropy.io import fits as _fits
    header = _fits.getheader(work / "sky_master.fit")
    height, width = int(header["NAXIS2"]), int(header["NAXIS1"])
    job = job or load_job(directory)
    source = _validity_source(directory, job, width, height, int(validity_record["dilation_px"]))
    mask = source.static.mask if hasattr(source, "static") else source.mask
    from . import registered_support as rs
    if job.get("support_mode") == rs.MODE:
        supported, record = rs.read_native(work, (height, width))
        # This native map is Boolean support, deliberately not an exposure count.
        cov = supported[::SKY_COVERAGE_STEP, ::SKY_COVERAGE_STEP].astype(np.uint16)
        frame_count = len(record["frames"])
    else:
        transforms, model = registration_transforms(work, calibration, own_registration, height)
        from . import performance
        # P1b-1p (G-6 pilot): frames on the frame pool, counts identical for any worker count. Efficiency P1c O-1: the
        # decision is performance.frame_workers (rule R20; AESTER_FRAME_WORKERS overrides it for measurements), capped
        # by ``worker_cap`` while the trail scan overlaps; the pool's figures go to the timings note
        # frame_pool.coverage, one entry per call (R19). A unit's memory grows with the frame's pixels. sky_validity is
        # in the frozen holdout closure and declares no unit_bytes, so the workers' bytes are reserved here while the
        # pool runs (frame_pool.reserved): a decision taken meanwhile on O-1's helper thread sees them.
        unit_bytes = _coverage_unit_bytes(height, width)
        cap = sv.COVERAGE_WORKER_CAP if worker_cap is None else max(1, min(sv.COVERAGE_WORKER_CAP, int(worker_cap)))
        decision = {}
        workers = performance.frame_workers(len(transforms), unit_bytes, cap, blas=True, decision=decision)
        held = workers * (unit_bytes + frame_pool.BLAS_SLOT_BYTES) if workers > 1 else 0
        decision["reserved_bytes"] = held
        if worker_cap is not None:
            decision["o1_cap"] = int(worker_cap)
        pool_stats = {"decision": decision}
        try:
            with frame_pool.reserved(held):
                cov = sv.coverage_map(source, transforms, model, step=SKY_COVERAGE_STEP, check=check,
                                      shape=(height, width), workers=workers, stats=pool_stats)
        finally:
            performance.note_pool("coverage", pool_stats)
        from . import resources
        resources.shadow_pool("py.coverage", actual=pool_stats.get("workers"), items=len(transforms),
                              frame_pixels=height * width)   # P1b-2 X3: the governor's first frame-pool decision
        frame_count = len(transforms)
    sv.write_coverage_png(cov, work / sv.COVERAGE_PNG)
    summary = sv.coverage_summary(cov, mask, step=SKY_COVERAGE_STEP, frames=1 if job.get("support_mode") == rs.MODE else frame_count)
    if job.get("support_mode") == rs.MODE:
        summary.update(semantics="native_full_dependency_support_boolean_not_frame_count", interpolation_support=rs.MODE,
                       input_frames=frame_count, counts_file=rs.COUNTS, counts_grid="registered_before_native_warp")
    summary["registration_source"] = REGISTRATION_SOURCE_OWN if own_registration else REGISTRATION_SOURCE_SIRIL
    atomic_json(work / "sky_coverage.json", summary)
    return work / sv.COVERAGE_PNG


# ---- Efficiency P1c O-1: the trail scan beside the post-stack work (docs/EFFICIENCY_P1C_PREREG_20261005.md §3.6) ----
#
# The trail scan (B: registration transforms, reference stem, transient_stage.detect_job) runs on one frame_pool.Beside
# helper thread while the main thread does the native remap, the strict-support finish, the coverage count and the
# diagnostics (A), and is joined at its serial position: the main thread's code and its _update / checkpoint calls are
# unchanged and stay on the main thread (B writes job.json only through check(): state and message of a cancel).
# Records and products are bit-identical; only timings (the note overlap.o1, the operations transient_detect and
# transient_join) differ. Not eligible (the first failing condition is noted): the serial code runs unchanged.

O1_NOTE = "overlap.o1"                     # timings note, one per attempt that reached the decision
O1_NOTE_SCHEMA = "overlap-o1-1"
O1_MARKER = "transients.o1-pending"        # work/<marker>: B's output may be in work/transients (no hex suffix)
O1_REMOVE_SECONDS = 2.0                    # bounded retry of Windows sharing denials (records.atomic_json's rule)
O1_RESERVED_CORES = 1                      # the scan's ceiling leaves the core of the main thread's native remap
O1_REFERENCE_PIXELS = 5520 * 8280          # 45.7 MP: the frame the figures below were measured on
O1_SCAN_CPU_SECONDS = 3.6                  # scan child CPU per reference frame (K L48/F48/R48: 175-191 CPU s / 48)
# A's duration at the reference size, below the measured native_remap_support phases (P1b/K: L/R 36-38 s, F 44 s
# before G-4t's ~5 s gain): a low estimate only turns O-1 off where it would gain little.
O1_A_SECONDS = {"lr": 34.0, "strict": 38.0}
O1_MARGIN_SECONDS = 5.0                    # prereg §3.6: contention allowance of the time criterion
# A's memory (prereg §3.6 "按模式取实测峰值，再按 P 缩放"). L/R: the native remap's input and output (24 B/px: the
# measured 1.1 GB rise of the native_remap operation at 45.7 MP) or the native master still held (12 B/px) plus the
# coverage pool, whichever is larger; at 8 coverage workers this is 2.75 GB >= the measured 2.65 GB rise of the L48
# job (0.95 -> 3.58 GB private, coverage operation). F / strict: 1.7 GB after U-M14, plus G-4t's workers x unit_bytes.
O1_A_LR_REMAP_BYTES_PER_PIXEL = 24
O1_A_LR_HELD_BYTES_PER_PIXEL = 12
O1_A_STRICT_BYTES = 1_700_000_000
_O1_SHARING_DENIALS = {5, 32, 33}
# rmtree of work/transients also retries ERROR_DIR_NOT_EMPTY (145): a file inside is still delete-pending while the
# handle of a just-ended scan process or of a virus scanner closes (transient_scan's REMOVAL_SETTLE_SECONDS settles the
# same); the same O1_REMOVE_SECONDS bound. Beyond prereg §3.6's 5/32/33 (a known deviation of the O-1 fixes).
_O1_REMOVE_RETRIED = _O1_SHARING_DENIALS | {145}


def _o1_started(overlap) -> None:
    """Test seam: called with the _Overlap once B has started; a no-op in the product."""


def _o1_log(directory, line: str) -> None:
    """One line in the job's worker.log; never raises (diagnostics)."""
    try:
        with (Path(directory) / "worker.log").open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")
    except Exception:
        pass


def _o1_retry(action, retried=_O1_SHARING_DENIALS) -> None:
    """Run ``action`` retrying the Windows errors ``retried`` (default: the access / sharing denials 5, 32, 33) for
    O1_REMOVE_SECONDS."""
    deadline = time.monotonic() + O1_REMOVE_SECONDS
    while True:
        try:
            action()
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in retried or time.monotonic() >= deadline:
                raise
            time.sleep(.02)


def _o1_clear(work: Path, directory=None) -> bool:
    """O-1's marker protocol (prereg §3.6): remove work/transients, then the marker. A sharing denial (for the folder
    also a delete-pending entry, _O1_REMOVE_RETRIED) is retried for O1_REMOVE_SECONDS; when the folder cannot be
    removed the marker stays (the next attempt removes both) and the result is False. Never raises (except
    KeyboardInterrupt / SystemExit)."""
    from .transient_stage import TRANSIENTS_DIR
    target = Path(work) / TRANSIENTS_DIR
    try:
        _o1_retry(lambda: shutil.rmtree(target) if target.exists() else None, _O1_REMOVE_RETRIED)
        _o1_retry(lambda: (Path(work) / O1_MARKER).unlink(missing_ok=True))
        return True
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if directory is not None:
            _o1_log(directory, f"O-1: could not remove {target} ({type(exc).__name__}: {exc}); the marker "
                               f"{O1_MARKER} stays for the next attempt")
        return False


def _o1_require_clear(work: Path, directory) -> None:
    """_o1_clear, or the attempt fails visibly (a resume retries): a scan must never write into an old one's folder."""
    if not _o1_clear(work, directory):
        folder = Path(work) / "transients"
        raise RuntimeError(f"无法删除上次中断的亮迹扫描输出（{folder}），请关闭占用它的程序后恢复作业")


def _detect_transients(work: Path, calibration, used: bool, height_px: int, inputs, marks, *, check, timer,
                       workers=None, crop_cap=None, scan_done=None) -> dict:
    """The trail scan of a finished stack (the serial body of the transient phase; O-1 runs the same calls on its
    helper thread with ``workers`` and ``crop_cap`` from its plan, and ``scan_done`` releasing its reservation of the
    scan processes' bytes once they have ended)."""
    from . import transient_stage as ts
    try:
        transforms, model = registration_transforms(work, calibration, used, height_px)
    except (OSError, SequenceError, ValueError) as exc:
        transforms, model = {}, None
        print(f"transients: registration transforms unavailable ({exc}); native geometry and marks skipped")
    reference_stem = (calibration.get("registration") or {}).get("reference") or sequence_reference(work / SEQUENCE_FILE, inputs).get("reference_stem")
    extra = {}
    if workers is not None:
        extra["workers"] = workers
    if crop_cap is not None:
        extra["crop_cap"] = crop_cap
    if scan_done is not None:
        extra["scan_done"] = scan_done
    return ts.detect_job(work, own_registration=used, transforms=transforms, model=model, inputs=inputs,
                         reference_stem=reference_stem, marks=marks, check=check, log=print, timer=timer, **extra)


def _o1_estimates(mode: str, pixels: int, frames: int) -> tuple[float, float]:
    """(S_est, A_est) of the time criterion: the scan's child CPU seconds and A's duration, scaled by the pixels."""
    scale = pixels / O1_REFERENCE_PIXELS
    return O1_SCAN_CPU_SECONDS * frames * scale, O1_A_SECONDS[mode] * scale


def _o1_budget(available: int) -> int:
    """frame_pool.default_workers' memory budget: min(available / 3, available - 1 GiB)."""
    return max(0, min(available // 3, available - (1 << 30)))


def _o1_decide(*, transient_mode, own_registration: bool, stack_done: bool, work: Path, job: dict, calibration,
               support_enabled: bool, validity_record, native_workers: int = 0, native_unit_bytes: int = 0):
    """O-1's eligibility (prereg §3.6), on the main thread after the rename: (reason, plan, frozen). ``reason`` is the
    first condition that fails (None: eligible); ``plan`` the figures of the timings note; ``frozen`` B's inputs, frozen
    here (deep copies). Any exception while deciding makes O-1 ineligible (``overlap_error:<type>``)."""
    plan = {}
    try:
        if transient_mode == "off":
            return "trails_off", plan, None
        if not own_registration:
            return "fallback_registration", plan, None
        if stack_done:
            return "stack_checkpointed", plan, None
        if not frame_pool.overlap_enabled():
            if (os.environ.get(frame_pool.ENV_OVERLAP) or "").strip().lower() == "off":
                return "overlap_off", plan, None
            return ("serial_workers" if frame_pool.worker_setting() == "1" else "in_pool"), plan, None
        from astropy.io import fits as _fits
        from . import performance
        from . import transient_stage as ts
        undistorted = work / "sky_master_undistorted.fit"
        header = _fits.getheader(undistorted)
        stacked = header.get("STACKCNT")
        if stacked is not None and stacked < 3:
            return "stackcnt", plan, None
        height, width = header["NAXIS2"], header["NAXIS1"]
        if type(height) is not int or type(width) is not int:
            return "naxis2", plan, None
        try:
            marks = _normalise_marks(job.get("transient_marks") or [], job["inputs"], work=work)
        except Exception:
            return "marks_invalid", plan, None
        physical, commit = performance.available_memory(), performance.available_commit()
        if physical is None or commit is None:
            return "memory_unknown", plan, None
        master = ts.grid_master_path(work, True)
        if master != undistorted:
            return "overlap_error:grid_master", plan, None
        pixels = height * width
        frames = len(ts.frame_order(work))
        mode = "strict" if support_enabled else "lr"
        cov_plan = cov_share = native_share = 0
        if mode == "lr":
            if validity_record is not None:
                from . import sky_validity as sv
                transforms, _ = registration_transforms(work, calibration, True, height)
                cov_unit = _coverage_unit_bytes(height, width)
                if transforms:
                    cov_plan = performance.frame_workers(len(transforms), cov_unit, sv.COVERAGE_WORKER_CAP, blas=True)
                cov_share = cov_plan * (cov_unit + frame_pool.BLAS_SLOT_BYTES)
            a_base = max(O1_A_LR_REMAP_BYTES_PER_PIXEL * pixels, O1_A_LR_HELD_BYTES_PER_PIXEL * pixels + cov_share)
            a_base -= cov_share
        else:
            native_share = int(native_workers) * int(native_unit_bytes)
            a_base = int(O1_A_STRICT_BYTES * pixels / O1_REFERENCE_PIXELS)
        a_reserve = a_base + cov_share + native_share
        # The crop pool on B's thread: its cap leaves the coverage pool's slots of the 16 (prereg §3.6); B_inproc =
        # its workers x 32 MiB (T-3m) + B's own BLAS slot. AESTER_FRAME_WORKERS (benches) overrides any cap.
        crop_cap = max(1, min(ts.TRAIL_CROP_CAP, frame_pool.MAX_FRAME_WORKERS - cov_plan))
        setting = frame_pool.worker_setting()
        crop_workers = (crop_cap if setting == "auto" else
                        min(frame_pool.MAX_FRAME_WORKERS, frame_pool.usable_cpus()) if setting == "max" else int(setting))
        b_inproc = crop_workers * ts.TRAIL_CROP_UNIT_BYTES + frame_pool.BLAS_SLOT_BYTES
        unit = ts.scan_unit_bytes(pixels)
        avail_physical = int(physical) - a_reserve - b_inproc
        avail_commit = int(commit) - a_reserve - b_inproc
        override, variable = ts.scan_override()
        if override is not None:
            n_o1 = override
        else:
            n_o1 = ts.plan_scan_workers(pixels, frames=frames, available=min(avail_physical, avail_commit),
                                        floor=False, reserved_cores=O1_RESERVED_CORES)
        n_o1 = max(1, min(int(n_o1), max(1, frames)))           # detect_job's own clamp
        n_serial = max(1, min(int(ts.default_workers(pixels, frames=frames)), max(1, frames)))
        scan_bytes = n_o1 * unit
        scan_cpu, a_seconds = _o1_estimates(mode, pixels, frames)
        extra_seconds = scan_cpu * (1 / n_o1 - 1 / n_serial)
        plan.update(mode=mode, pixels=pixels, frames=frames, n_o1=n_o1, n_serial=n_serial, override=variable,
                    cov_plan=cov_plan, crop_cap=crop_cap, native_workers=int(native_workers),
                    scan_unit_bytes=unit, a_reserve_bytes=a_reserve, b_inproc_bytes=b_inproc,
                    headroom_bytes={"physical": _o1_budget(avail_physical) - scan_bytes,
                                    "commit": _o1_budget(avail_commit) - scan_bytes},
                    predicted_s={"scan_cpu": round(scan_cpu, 2), "a": round(a_seconds, 2),
                                 "scan_serial": round(scan_cpu / n_serial, 2), "scan_o1": round(scan_cpu / n_o1, 2),
                                 "extra": round(extra_seconds, 2), "margin": O1_MARGIN_SECONDS},
                    pieces={"a": a_base, "coverage": cov_share, "native_mask": native_share, "scan": scan_bytes})
        if scan_bytes > _o1_budget(avail_physical):
            return "memory_tight", plan, None
        if scan_bytes > _o1_budget(avail_commit):
            return "commit_tight", plan, None
        if not extra_seconds + O1_MARGIN_SECONDS < a_seconds:
            return "slower_than_serial", plan, None
        import copy
        frozen = {"used": True, "height_px": height, "master": str(master),
                  "calibration": copy.deepcopy(calibration), "inputs": copy.deepcopy(job["inputs"]),
                  "marks": copy.deepcopy(marks)}
        return None, plan, frozen
    except Exception as exc:
        return f"overlap_error:{type(exc).__name__}", plan, None


class _Overlap:
    """One O-1 window of a job attempt (prereg §3.6): B on a frame_pool.Beside helper, the in-process reservations of
    the plan (``pieces``: A's part not yet allocated, released as A reaches each part -- 'a' once its native master is
    written and checked, 'native_mask' and 'coverage' as their pools start, which reserve their own workers -- and the
    scan processes' bytes, released on B's thread once scan_frames has returned or raised: the processes have ended),
    the marker and the timings note. Every method runs on the main thread except ``release`` from B's ``scan_done``."""

    def __init__(self, directory: Path, work: Path, plan: dict, frozen: dict, timer, note: dict):
        self.directory, self.work, self.plan, self.frozen, self.timer, self.note = \
            Path(directory), Path(work), plan, frozen, timer, note
        self.helper = None
        self.held = {}                     # piece -> ExitStack holding its frame_pool.reserved()
        self._held_lock = threading.Lock()  # release() also runs on B's thread (the scan piece)
        self.taken = False                 # B's outcome was handed out at its serial position (value or exception)
        self.kept = False                  # B's output belongs to the job now (checkpointed, or a recorded failure)
        self.window = {}

    def _now(self):
        started = getattr(self.timer, "started", None)
        return round(time.perf_counter() - started, 3) if isinstance(started, (int, float)) else None

    def _attach(self):
        try:
            self.timer.attach_note(O1_NOTE, self.note)
        except Exception:
            pass

    def reserve(self, name: str, nbytes: int) -> None:
        if nbytes > 0:
            stack = contextlib.ExitStack()
            stack.enter_context(frame_pool.reserved(int(nbytes)))
            with self._held_lock:
                self.held[name] = stack

    def release(self, name: str) -> None:
        """Release one piece; repeatable, from either thread."""
        with self._held_lock:
            stack = self.held.pop(name, None)
        if stack is not None:
            stack.close()

    def _scan_ended(self) -> None:
        """detect_job's scan_done, on B's thread: the scan processes have ended, their bytes are no longer promised.
        Never raises (it runs in detect_job's finally)."""
        try:
            self.release("scan")
        except Exception:
            pass

    def release_all(self) -> None:
        with self._held_lock:
            names = list(self.held)
        for name in names:
            self.release(name)

    def start(self, check) -> None:
        """Reserve the plan's pieces, write the marker and start B."""
        pieces = self.plan["pieces"]
        for name in ("a", "coverage", "native_mask", "scan"):
            self.reserve(name, pieces.get(name, 0))
        (self.work / O1_MARKER).write_bytes(b"")
        work, frozen, timer = self.work, self.frozen, self.timer
        workers, crop_cap = self.plan["n_o1"], self.plan["crop_cap"]
        window = self.window

        def scan(probe):
            window["b_start_s"] = self._now()
            try:
                with timer.operation("transient_detect"):
                    return _detect_transients(work, frozen["calibration"], frozen["used"], frozen["height_px"],
                                              frozen["inputs"], frozen["marks"], check=probe, timer=timer,
                                              workers=workers, crop_cap=crop_cap, scan_done=self._scan_ended)
            finally:
                window["b_end_s"] = self._now()

        self.helper = frame_pool.Beside.start(scan, "o1-scan", check=check)

    def mismatch(self, **live) -> str | None:
        """The first join guard that does not hold (plain comparisons; a comparison that fails counts as a mismatch):
        the main thread only reads these objects between the start and the join, so none is expected."""
        try:
            for name in ("used", "height_px", "calibration", "inputs", "marks"):
                if not (self.frozen[name] == live[name]):
                    return name
        except Exception as exc:
            return f"compare:{type(exc).__name__}"
        return None

    def result(self, wait_check):
        """B's record (or its exception) at the serial position, waiting with ``wait_check`` polled every 0.1 s (a
        cancel there abandons B and is raised); A's piece is released first (A is done; normally it went once A's
        native master was checked), the scan's once B has ended (normally on B's thread when scan_frames ended; this
        is the fallback). The helper keeps no reference to the outcome (frame_pool.Beside.result)."""
        self.release("a")
        self.window["join_start_s"] = self._now()
        self.note["b_done_at_join"] = self.helper.done()
        started = time.perf_counter()
        try:
            return self.helper.result(wait_check)
        except BaseException as exc:
            self.note["failure"] = type(exc).__name__
            raise
        finally:
            # taken: B's outcome was handed out -- or the wait was interrupted (KeyboardInterrupt inside the join),
            # which leaves B running: close() abandons a helper still alive whatever ``taken`` says
            self.taken = True
            self.note["join_wait_s"] = round(time.perf_counter() - started, 3)
            self.window["join_end_s"] = self._now()
            if self.helper.done():
                self.release("scan")          # normally gone already (B's scan_done); B still running: close() does
            self._attach()

    def keep(self, outcome: str) -> None:
        """B's output now belongs to the job, as the serial scan's would: checkpoint('transients') is written
        (``joined``), or B's failure became transients_error and the job goes on (``recorded``). The marker goes."""
        self.kept = True
        self.note["outcome"] = outcome
        try:
            _o1_retry(lambda: (self.work / O1_MARKER).unlink(missing_ok=True))
        except OSError as exc:
            _o1_log(self.directory, f"O-1: could not remove the marker {O1_MARKER} ({type(exc).__name__}: {exc})")
        self.close("kept", log=False)

    def discard(self, why: str) -> None:
        """A join guard failed: B is abandoned and its output removed (the serial code runs next in its place); a
        folder that cannot be removed fails the attempt (the serial scan must not write next to it)."""
        self.note["outcome"] = f"guard_mismatch:{why}"
        if not self.close(f"guard mismatch ({why})"):
            _o1_require_clear(self.work, self.directory)

    def close(self, why: str, *, log: bool = True) -> bool:
        """Abandon B unless its outcome was taken and its thread has ended (it ends at its next probe; scan_frames'
        finally ends the scan processes and the guards reap them; a join interrupted by KeyboardInterrupt leaves B
        running although ``taken`` is set), release every reservation and, unless the job kept B's output, remove it
        with the marker -- only once B's thread has ended. Repeatable. True when nothing of B is left over. Never raises
        except KeyboardInterrupt / SystemExit: nothing replaces the main thread's exception (prereg §3.6)."""
        try:
            helper = self.helper
            if helper is not None and (not self.taken or not helper.done()):
                was_done = helper.done()
                helper.abandon()              # repeatable; harmless after the outcome was handed out
                self.taken = True
                if log:
                    _o1_log(self.directory, f"O-1: the trail scan was discarded ({why}); it had "
                                            f"{'ended' if was_done else 'not ended'}; its outcome is dropped"
                            + (f"; abandon: {helper.errors}" if helper.errors else ""))
            if not self.kept:
                self.note.setdefault("outcome", f"discarded:{why}")
            self.release_all()
            clear = True
            if not self.kept and helper is not None and not helper.done():
                # abandon() could not wait B out (its errors are logged above): never remove a folder B may still be
                # writing; the marker stays for the next attempt
                _o1_log(self.directory, f"O-1: the trail scan is still running ({why}); its output and the marker "
                                        f"{O1_MARKER} stay for the next attempt")
                clear = False
            elif not self.kept:
                clear = _o1_clear(self.work, self.directory)
                self.kept = clear              # nothing left to remove (or the marker stays for the next attempt)
            self.note["window_s"] = dict(self.window)
            self._attach()
            return clear
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            _o1_log(self.directory, f"O-1: closing the overlap failed ({type(exc).__name__}: {exc})")
            return False


def _contract_stack(runner, exe, directory: Path, commands, check, *, note=None) -> dict:
    """An L or strict-L ``02_stack`` under the EFF-D4 stack-settings contract (efficiency P1c §4.2).

    Planned from the export's r_light_.seq and first selected frame; the settings lines go before ``seqclean`` and
    the Siril process runs with OMP_NUM_THREADS=1 and its own configuration ``work/siril-stack.ini`` (copied from the
    job's siril.ini, removed afterwards), so the memory setting never reaches a later export or registration of the
    job.  Afterwards this attempt's log section and the master's STACKCNT are checked; a mismatch raises
    RuntimeError before the stack checkpoint (a resume re-exports and stacks again).
    """
    work = Path(directory) / "work"
    settings = stack_settings.plan(work, pin=siril_pin(), script="02_stack")
    config = work / stack_settings.CONFIG_NAME
    if (Path(directory) / "siril.ini").is_file():
        shutil.copyfile(Path(directory) / "siril.ini", config)
    else:
        config.write_bytes(b"")
    log = Path(directory) / "02_stack.log"
    offset = stack_settings.log_offset(log)
    if note is not None:
        note(settings)
    try:
        runner(exe, directory, "02_stack", stack_settings.with_settings(commands, settings), check,
               config=config, stack_settings=True)
    finally:
        config.unlink(missing_ok=True)
    stack_settings.verify_log(log, offset, settings)
    stack_settings.verify_stack_count(work / "sky_master.fit", settings)
    return settings


def _range_runner(runner, job: dict, *, note=None):
    """The Siril runner handed to stack_range.run_protected_stack (R), which stays unchanged (frozen holdout closure).

    For a job under the EFF-D4 contract, its ``stack`` launches get the settings lines before ``seqclean``, planned
    from the trial folder's r_light_.seq and first selected frame, and the contract environment; the trial's
    stack.log is checked afterwards (stack_range checks STACKCNT itself).  Other jobs get ``runner`` itself."""
    if stack_settings.job_contract(job) is None:
        return runner

    def run(executable, folder, name, commands, cancel_check):
        if name != "stack":
            return runner(executable, folder, name, commands, cancel_check)
        folder = Path(folder)
        settings = stack_settings.plan(folder, pin=siril_pin(), script=f"{folder.parent.name}/{folder.name}/{name}")
        log = folder / f"{name}.log"
        offset = stack_settings.log_offset(log)
        if note is not None:
            note(settings)
        result = runner(executable, folder, name, stack_settings.with_settings(commands, settings), cancel_check,
                        stack_settings=True)
        stack_settings.verify_log(log, offset, settings)
        return result
    return run


def _finite_stack_files(work: Path) -> list:
    """F's own stack-checkpoint files: the finite count maps, then the retained per-frame support maps by name (rule
    R16, efficiency P1c: F-2's pool writes the maps in any order, and FAT/exFAT list a folder in creation order)."""
    from . import finite_stack as fs
    return [work / fs.NVALID, work / fs.NUSED, *sorted(work.glob('frame_support_light_*.npy'))]


def _untimed(name):
    """A timer's ``operation`` for callers without a timer (tests): times nothing."""
    from contextlib import nullcontext
    return nullcontext()


def _strict_support_workers(work: Path, job: dict, height: int, width: int, units: int, decision: dict):
    """F-2's worker count and unit bytes (efficiency P1c §3.2; core decides, rule R22).

    unit_bytes: registered_support's measured unit peak scaled by the frame's pixels (never below the measured figure),
    plus one anchor grid in flight when the job has per-frame masks (R17). A per-frame mask source also decodes masks
    in prepare, ahead of the workers, and its cache keeps every own mask (1 B/px): both come off the memory first
    (``extra_bytes``). The cap is STRICT_WORKER_CAP, 2 when the work folder is on a removable or network volume
    (GetDriveType; an HDD cannot be told cheaply: P2). ``decision`` receives the bounds for the timings note."""
    from . import performance, registered_support as rs, resources
    pixels = height * width
    unit_bytes = max(rs.STRICT_UNIT_BYTES, -(-rs.STRICT_UNIT_BYTES * pixels // rs.STRICT_UNIT_PIXELS))
    own_masks = len((job.get("frame_masks") or {}).get("files") or {})
    extra_bytes = 0
    if own_masks:
        unit_bytes += pixels                                  # the anchor grid of a frame in flight
        extra_bytes = rs.STRICT_MASK_DECODE_BYTES_PER_PIXEL * pixels + own_masks * pixels
    cap = rs.STRICT_WORKER_CAP
    volume = resources.drive_type(work)
    if volume in ("removable", "remote"):
        cap = min(cap, rs.STRICT_VOLUME_CAP)
    workers = performance.frame_workers(units, unit_bytes, cap, blas=True, extra_bytes=extra_bytes,
                                        decision=decision)
    decision.update(volume=volume, own_masks=own_masks)
    if cap < rs.STRICT_WORKER_CAP and decision.get("bound") == "cap":
        decision["bound"] = "volume"
    return workers, unit_bytes


def _native_mask_workers(height: int, width: int, decision: dict):
    """G-4t's worker count and unit bytes (efficiency P1c §3.2): the worst-case 64-row block scaled by the width, never
    below the measured figure; no BLAS. ``decision`` receives the bounds for the timings note."""
    from . import performance, registered_support as rs
    unit_bytes = max(rs.NATIVE_UNIT_BYTES, -(-rs.NATIVE_UNIT_BYTES * width // rs.NATIVE_UNIT_WIDTH))
    blocks = -(-height // 64)
    workers = performance.frame_workers(blocks, unit_bytes, rs.NATIVE_WORKER_CAP, blas=False, decision=decision)
    return workers, unit_bytes, blocks


def _supported_stack(exe, directory, work, job, calibration, own, validity_record, check, *, runner=None, timer=None,
                     note=None, finite_write=None):
    """Split export/stack so unsupported samples never enter rejection or normalization.

    ``finite_write``: the attempt's AESTER_FINITE_WRITE (finite_write_setting(); None: read it now)."""
    from . import registered_support as rs
    runner = runner or run_script
    if validity_record is None:
        raise ValueError("strict support requires a source-validity record")
    with hashing.writing(work / rs.EXPORT_SEQUENCE):
        shutil.copyfile(work / SEQUENCE_FILE, work / rs.EXPORT_SEQUENCE)
    runner(exe, directory, "02_export", ["cd work", rs.EXPORT_COMMAND], check)
    shape = _validate_frame(next(work.glob("light_*.fit")))
    source = _validity_source(directory, job, shape[2], shape[1], int(validity_record["dilation_px"]))
    def progress(number, total, stem):
        _update(directory, stage="support", progress=45 + int(10 * number / total),
                message=f"检查插值完整支持 {number}／{total} 帧（{stem}）")
    from . import finite_stack as fs
    from . import performance, resources
    finite_enabled = job.get('stack_range_mode') == fs.MODE
    # Efficiency P1c F-2: the exports on the frame pool; the decision and the pool's figures go to the timings notes
    # only (AESTER_FRAME_WORKERS overrides the count for measurements).
    units = len(list(work.glob("r_light_*.fit")))
    decision = {}
    workers, unit_bytes = _strict_support_workers(work, job, shape[1], shape[2], units, decision)
    pool_stats = {"decision": decision}
    operation = timer.operation if timer is not None else _untimed
    with operation("registered_support"):
        rs.prepare_exports(work, source, _grid_model(calibration, own), check=check, progress=progress,
                           workers=workers, unit_bytes=unit_bytes, pool_stats=pool_stats,
                           **({'retain_frame_support': True} if finite_enabled else {}))
    performance.note_pool("strict_support", pool_stats)
    resources.shadow_pool("py.strict_exports", actual=pool_stats.get("workers"), items=units,
                          frame_pixels=shape[1] * shape[2])
    _update(directory, stage="stack", progress=57, message="按完整支持重新计算归一化并堆栈天空")
    from . import stack_range as sr
    if finite_enabled:
        pools = _FinitePools(timer, finite_write_setting() if finite_write is None else finite_write)
        fs.run(job['stack_engine'], directory, work, check, runner, pools=pools)
    elif job.get('stack_range_mode') == sr.MODE:
        sr.run_protected_stack(job['engine'], directory, work, check, _range_runner(runner, job, note=note),
                               contract=job.get('stack_range_contract'))
    elif stack_settings.job_contract(job) is None:   # a job admitted before EFF-D4: unchanged
        runner(exe, directory, "02_stack", ["cd work", "seqclean r_light -stat", STACK_COMMAND], check)
    else:
        _contract_stack(runner, exe, directory, ["cd work", "seqclean r_light -stat", STACK_COMMAND], check, note=note)


def _validate_master(path: Path, expected_shape=None):
    return _validate_frame(path, expected_shape, require_float=True)


def _validate_frame(path: Path, expected_shape=None, require_float=False):
    import numpy as np
    from astropy.io import fits
    from .imaging import complete_unsigned16_shape
    with fits.open(path, memmap=False) as hdus:
        # Efficiency P1a C-2: when astropy itself would return a complete pseudo-unsigned 16-bit primary image (Siril's
        # converted frames) -- a uint16 array of exactly `known` shape, always finite -- its header decides the checks;
        # the data unit is still read once, unconverted.  Everything else is read and checked pixel by pixel as before.
        known = complete_unsigned16_shape(path, hdus)
        data = hdus[0].data if known is None else None
        is_float = data is not None and data.dtype.kind == "f" and data.dtype.itemsize == 4
        is_uint16 = known is not None or (data is not None and data.dtype.kind == "u" and data.dtype.itemsize == 2)
        if require_float and not is_float:
            raise ValueError("Siril 产物必须是 32 位浮点 FITS")
        if not (is_float or is_uint16):
            raise ValueError("转换帧必须是 16 位无符号整数或 32 位浮点 FITS")
        if hdus[0].header.get("ROWORDER", "BOTTOM-UP") != "BOTTOM-UP":
            raise ValueError("此原型仅支持 BOTTOM-UP 的 FITS 坐标，请先统一行方向")
        shape = known if known is not None else data.shape
        if len(shape) != 3 or shape[0] != 3 or (known is None and not np.isfinite(data).all()):
            raise ValueError("Siril 产物必须是有限值的三通道图像")
        if expected_shape is not None and shape != expected_shape:
            raise ValueError("Siril 输出改变了参考画幅，不能继续合成")
        return shape


def create_demo(output_root: str) -> str:
    from .imaging import create_demo as generate
    return generate(output_root)
