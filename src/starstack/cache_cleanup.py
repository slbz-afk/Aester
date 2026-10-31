"""In-app cleanup of a finished job's rebuildable per-frame intermediates (REL-CACHE option A).

One entry point shared by the desktop page 设置 → 作品与缓存 and the read-only MCP tool
``inspect_workflow_cache``. A 48-frame job leaves 50-80 GB in its cache, nearly all of it
per-frame files of the workflow's stack job that are read only while that job runs:

* converted frames ``work/light_*.fit`` and registered frames ``work/r_light*.fit``;
* the Siril sequence files ``work/*.seq``;
* the staged copies of the originals ``stage/frameNNNNN.*``, only those whose original is still at its
  recorded path with its recorded size (a copy whose original was moved or deleted may be the last copy of
  that photo: it is kept and counted in ``stage_copies_kept``);
* the stack-range trial copies ``range-*/trial*/*.fit``;
* the range-preserving (finite) stack inputs ``finite-*/q_*.fit``;
* the per-frame validity arrays ``work/frame_support_light_*.npy``;
* for FITS CFA input, the conversion's per-frame copies ``cfa-*/{float,decoded,canonical}/*.fit``
  (``fits_input.convert_cfa`` copies the packaged frames to ``work/light_*.fit``; nothing reads them again).

After success these are read only by a stack-job resume (a succeeded workflow is never
resumed, and ``core.gc_refusal`` refuses it), by the finite / range receipt checks of that
resume and of a trail recompose, by the trail recompose itself (``work/light_*.fit``) and by
the trail-review neighbour crops (``work/r_light_*.fit`` / ``work/light_*.fit``). Review and
recompose are refused after cleanup (``core.cache_cleanup_refusal``, ``core.gc_refusal``).

Everything else stays: masters (``work/sky_master*.fit``, ``work/foreground_master.fit``, the
finite and range masters), ``products/``, the workflow's checkpoint images and ``finishing/``
layers, publication receipts, ``workflow.json``/``job.json``, stack snapshots, transient
records and crops, logs, previews and every file not matched above. Works stay viewable,
"另存新预览" (``preview.render_job_preview`` reads the products master and ``sky_mask.png``)
and the star re-runs (``finishing/``) keep working.

``inspect_workflow_cache`` (and the job list) changes nothing, reports a refusal in its plan instead of
raising and reads no image: the works folders are checked by their records and file sizes only.
``clean_workflow_cache`` first re-plans with every works file hashed (``storage.verify_delivery``), outside
any lock; it then takes, without waiting, the cache root's admission lock, the workflow's
worker/edit/metadata locks and the stack job's worker/metadata/trail_review locks, re-plans inside them
by cheap identities (tree walk, record digests, works files' size and mtime) and requires the inspected
``plan_id``. Before the first deletion it records the stack job's ``job.json.gc`` (the existing gc refusal)
and its own workflow marker ``.aester-app-cache-cleaned.json`` (state ``started``), then releases the
admission lock (submit and resume elsewhere wait at most 30 s for it; the marker already refuses both
for this job); afterwards the marker says ``completed`` or ``partial``. A partial run (a file held open
by another program) may be cleaned again by the same function. The original RAWs need not exist; the
works folders must verify.

The uninstaller (``uninstall_options``) writes its own marker ``.aester-cache-cleaned.json`` and removes
only masters and products, never a file matched here: a job it cleaned is still offered here, and a job
cleaned here is still offered to it.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from fnmatch import fnmatchcase
import hashlib
import json
import os
import re
import stat

from . import core, storage
from . import workflow as wf
from .locking import Busy, file_lock, is_locked
from .records import atomic_json, read_json

PLAN_VERSION = "aester-app-cache-plan-1"
RESULT_VERSION = "aester-app-cache-result-1"
MARKER_VERSION = "aester-app-cache-cleaned-1"
GC_VERSION = "app-cache-cleanup-1"
LISTING_VERSION = "aester-app-cache-listing-1"
MARKER = core.APP_CACHE_CLEANUP_MARKER              # this cleanup's own record
UNINSTALL_MARKER = core.CACHE_CLEANUP_MARKER        # == uninstall_options.CLEANUP_MARKER
UNINSTALL_MARKER_VERSION = "aester-cache-cleaned-1"
UNINSTALL_STATES = ("cleanup_started", "partial", "completed")
STAGE_NAME = re.compile(r"frame([0-9]{5,})(.*)", re.S)
WORKFLOW_NAME = re.compile(r"workflow-[0-9a-f]+")
JOB_NAME = re.compile(r"job-[0-9a-f]+")
PLAN_ID = re.compile(r"[0-9a-f]{64}")
MAX_JOBS = 1000
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
NAME_SURROGATE = 0x20000000                  # reparse tags of symlinks and junctions
# category, path patterns relative to the stack job folder (one fnmatch pattern per path part), label
GROUPS = (
    ("converted_frames", (("work", "light_*.fit"),), "转换帧（可由原片重建）"),
    ("registered_frames", (("work", "r_light*.fit"),), "配准帧（可由原片重建）"),
    ("sequences", (("work", "*.seq"),), "Siril 序列文件"),
    ("stage_copies", (("stage", "*"),), "原片副本（可由原片重新复制）"),
    ("range_trials", (("range-*", "trial*", "*.fit"),), "范围保护试验副本"),
    ("finite_frames", (("finite-*", "q_*.fit"),), "数值保真堆栈输入帧"),
    ("frame_support", (("work", "frame_support_light_*.npy"),), "逐帧有效性数组"),
    # FITS CFA input (fits_input.convert_cfa): float copies, Siril's debayered frames and the packaged frames
    # copied to work/light_*.fit; only the conversion audit and logs beside them are kept.
    ("cfa_frames", (("cfa-*", "float", "*.fit"), ("cfa-*", "decoded", "*.fit"), ("cfa-*", "canonical", "*.fit")),
     "CFA 去马赛克中间帧"),
)
STATES = ("cleanable", "partial", "cleaned", "nothing", "running", "refused")
BUSY = "任务正在运行或被其他窗口使用，不能清理；请稍后再试"
ADMISSION_BUSY = "已有任务正在提交或恢复，暂不能清理；请稍后再试"
REVIEW_BUSY = "亮迹复核正在进行，不能清理；请先关闭复核后再试"
CHANGED = "清理计划已改变，未删除任何文件；请重新检查后再清理"
FOREIGN = "此任务的缓存清理记录无法识别，不在应用内清理"
CLEANED = "此任务的中间文件已经清理"
NOTHING = "此任务没有可清理的中间文件"
COPIES_KEPT = "剩下的原片副本对应的原片已不在原来的位置或已改变，为免删除最后一份照片而保留"


class CleanupRefused(ValueError):
    """A job that may not be cleaned now; ``state`` is the plan state it reports (running, cleaned, ...)."""

    def __init__(self, message, state="refused"):
        super().__init__(message)
        self.state = state


def inspect_workflow_cache(workflow_dir) -> dict:
    """Read-only plan for one ``workflow-*`` job folder; nothing is changed or created.

    Returns plan_id (None unless the job can be cleaned now), state (cleanable, partial,
    cleaned, nothing, running, refused), refusal (the reason, or None), the deletable groups
    (category, label, files, bytes), files/bytes deletable, kept and total, the cache root
    (the job's recorded ``config.output_root``), the verified works folders and a job summary.
    """
    return _plan(workflow_dir)[0]


def list_workflow_caches(roots, *, cancelled=None) -> dict:
    """Plans of every ``workflow-*`` folder directly inside each given cache root (each root once).

    Missing roots are skipped; an unreadable root is reported under ``problems``; a damaged job
    folder is reported as that job's refusal. Jobs are newest first. Nothing is changed and no
    image is read. ``cancelled()`` (checked before each job) stops the listing early; the result
    then says ``cancelled: true`` and holds the jobs listed so far.
    """
    seen, jobs, problems, listed, stopped = set(), [], [], [], False
    for value in roots:
        if stopped or not value:
            continue
        root = Path(os.path.abspath(Path(value).expanduser()))
        key = os.path.normcase(str(root))
        if key in seen:
            continue
        seen.add(key)
        try:
            if not root.is_dir():
                continue
            storage._no_links(root)
            folders = sorted(path for path in root.iterdir() if WORKFLOW_NAME.fullmatch(path.name) and path.is_dir())
        except (OSError, ValueError) as error:
            problems.append({"cache_root": str(root), "error": str(error)})
            continue
        if len(folders) > MAX_JOBS:
            problems.append({"cache_root": str(root), "error": f"缓存目录中的任务超过 {MAX_JOBS} 个，只列出前 {MAX_JOBS} 个"})
            folders = folders[:MAX_JOBS]
        listed.append(str(root))
        for folder in folders:
            if cancelled is not None and cancelled():
                stopped = True
                break
            jobs.append(inspect_workflow_cache(folder))
    jobs.sort(key=lambda plan: str((plan.get("job") or {}).get("created_at") or ""), reverse=True)
    return {"version": LISTING_VERSION, "cache_roots": listed, "jobs": jobs, "problems": problems,
            "cancelled": stopped}


def clean_workflow_cache(workflow_dir, plan_id) -> dict:
    """Delete exactly the planned intermediates of one job, under its locks, after re-planning.

    Refuses (nothing deleted) when a lock is held, when the job may not be cleaned, or when
    the plan inside the locks differs from ``plan_id``. A file that cannot be removed (held
    open elsewhere, or changed since planning) is kept and reported; the result is then
    ``partial`` and a later inspect + clean removes what is left.
    """
    if not isinstance(plan_id, str) or not PLAN_ID.fullmatch(plan_id):
        raise ValueError("清理计划编号无效；请重新检查任务")
    directory = _admit_folder(workflow_dir)
    # Every works file is hashed here, before any lock: the cache root's admission lock is never held
    # while gigabytes are read (submit and resume in that root wait at most 30 s for it).
    verified, _, works = _plan(directory, verify=True)
    _require_plan(verified, plan_id)
    record = wf._read(directory)
    root = _cache_root(directory, record)
    child = _child(directory, record)
    admission = ExitStack()
    try:
        with ExitStack() as held:
            for index, (path, message) in enumerate(_locks(root, directory, child)):
                try:
                    (admission if index == 0 else held).enter_context(file_lock(path))
                except Busy:
                    raise CleanupRefused(message, "running") from None
            # Inside the locks, by cheap identities: the same files, records and (unchanged since hashed) works.
            plan, entries, current = _plan(directory, held=True)
            _require_plan(plan, plan_id)
            if current != works:
                raise CleanupRefused(CHANGED, "changed")
            return _execute(directory, child, wf._read(directory), plan, entries, admission)
    finally:
        admission.close()


def _require_plan(plan: dict, plan_id: str) -> None:
    if plan["plan_id"] is None:
        raise CleanupRefused(plan["refusal"] or CHANGED, plan["state"])
    if plan["plan_id"] != plan_id:
        raise CleanupRefused(CHANGED, "changed")


# --------------------------------------------------------------------------- planning

def _skeleton(directory: Path) -> dict:
    return {"version": PLAN_VERSION, "workflow_dir": str(directory), "cache_root": None,
            "state": "refused", "plan_id": None, "refusal": None, "groups": [],
            "files_deletable": 0, "bytes_deletable": 0, "files_kept": 0, "bytes_kept": 0,
            "files_total": 0, "bytes_total": 0, "job": {}, "delivery_dirs": [], "previous_cleanup": None,
            "uninstaller_cleanup": None, "stage_copies_kept": 0, "stage_copies_kept_bytes": 0,
            "keeps_works_and_originals": True, "loses_trail_review_resume_and_recompose": True}


def _plan(workflow_dir, *, held=False, verify=False) -> tuple[dict, list[dict], list]:
    """(plan, the entries to delete, the works files' identities); a refusal is reported in the plan.

    ``verify``: hash every works file (only to clean); otherwise records and sizes only.
    """
    plan = _skeleton(Path(os.path.abspath(Path(workflow_dir).expanduser())))
    works = []
    try:
        directory = _admit_folder(workflow_dir)
        plan["workflow_dir"] = str(directory)
        record = wf._read(directory)
        plan["job"] = _job_summary(record)
        plan["cache_root"] = str(_cache_root(directory, record))
        files = _walk(directory)
        plan.update(files_total=len(files), bytes_total=sum(info.st_size for _, info in files))
        plan.update(files_kept=plan["files_total"], bytes_kept=plan["bytes_total"])
        entries = _select(directory, record, files, plan, works, held=held, verify=verify)
    except CleanupRefused as refused:
        plan.update(state=refused.state, refusal=str(refused))
        return plan, [], works
    except (OSError, ValueError) as error:
        plan.update(state="refused", refusal=str(error))
        return plan, [], works
    except Exception as error:      # a damaged record (e.g. JSON that is not an object) refuses this job only
        plan.update(state="refused", refusal=f"任务记录无效或无法读取，不能清理：{type(error).__name__}: {error}")
        return plan, [], works
    return plan, entries, works


def _admit_folder(workflow_dir) -> Path:
    directory = Path(os.path.abspath(Path(workflow_dir).expanduser()))
    if not WORKFLOW_NAME.fullmatch(directory.name):
        raise CleanupRefused("只能清理缓存目录中的单个任务文件夹（workflow-…）")
    storage._no_links(directory)
    if not directory.is_dir():
        raise CleanupRefused("任务文件夹不存在或不是文件夹")
    return directory.resolve(strict=True)


def _cache_root(directory: Path, record: dict) -> Path:
    """The job's recorded cache root (not the current preference); it must be the folder holding the job."""
    recorded = (record.get("config") or {}).get("output_root")
    if not isinstance(recorded, str) or not recorded or Path(recorded).resolve() != directory.parent:
        raise CleanupRefused("任务记录的缓存位置与实际文件夹不符，不能清理")
    return directory.parent


def _child(directory: Path, record: dict) -> Path:
    value = record.get("stack_job_dir")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise CleanupRefused("任务没有登记堆栈子任务，不能清理")
    child = Path(value)
    storage._no_links(child)
    if not child.is_dir():
        raise CleanupRefused("堆栈子任务缺失或记录不符，不能清理")
    child = child.resolve(strict=True)
    if child.parent != directory / "stack" or not JOB_NAME.fullmatch(child.name):
        raise CleanupRefused("堆栈子任务缺失或记录不符，不能清理")
    return child


def _job_summary(record: dict) -> dict:
    request = record.get("request") or {}
    files = request.get("input_files") if isinstance(request.get("input_files"), list) else []
    source = request.get("input_dir") or (files[0] if files else "")
    return {"name": Path(str(source)).name, "frames": len(files), "created_at": record.get("created_at"),
            "state": record.get("state"), "delivery_dir": record.get("delivery_dir")}


def _link(info) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT
        and getattr(info, "st_reparse_tag", 0) & NAME_SURROGATE)


def _walk(directory: Path) -> list[tuple[str, os.stat_result]]:
    """(path relative to the job folder, lstat) of every regular file; any link or junction refuses."""
    files = []

    def failed(error):
        if not isinstance(error, FileNotFoundError):      # a folder removed meanwhile (a running job) is skipped
            raise error

    for current, folders, names in os.walk(directory, followlinks=False, onerror=failed):
        base = Path(current)
        for name in folders:
            try:
                info = os.lstat(base / name)
            except FileNotFoundError:
                continue
            if _link(info):
                raise CleanupRefused("缓存中有软链接或目录联接，不能清理：" + str(base / name))
        for name in names:
            path = base / name
            try:
                info = os.lstat(path)
            except FileNotFoundError:          # removed meanwhile (a running job): not a candidate
                continue
            if _link(info):
                raise CleanupRefused("缓存中有软链接或目录联接，不能清理：" + str(path))
            if stat.S_ISREG(info.st_mode):
                files.append((path.relative_to(directory).as_posix(), info))
    return sorted(files, key=lambda item: item[0])


def _group(parts) -> str | None:
    """The category of a path relative to the stack job folder, or None (kept)."""
    for category, patterns, _ in GROUPS:
        for pattern in patterns:
            if len(parts) == len(pattern) and all(fnmatchcase(part, glob) for part, glob in zip(parts, pattern)):
                return category
    return None


def _read_marker(path: Path):
    """None when absent, False when unreadable, else the parsed JSON."""
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return False


def _bound(data, directory: Path, record: dict) -> bool:
    """The marker names this job: its folder (as resolved here, or as the record and the uninstaller spell it)
    and its workflow signature."""
    return (isinstance(data, dict) and data.get("workflow_dir") in {str(directory), record.get("workflow_dir")}
            and data.get("workflow_signature") == record.get("signature"))


def _marker(directory: Path, record: dict) -> tuple[str | None, dict | None]:
    """(None | "ours" | "foreign", our marker): this cleanup's own record of an earlier run."""
    data = _read_marker(directory / MARKER)
    if data is None:
        return None, None
    ours = _bound(data, directory, record) and data.get("version") == MARKER_VERSION
    return ("ours", data) if ours else ("foreign", None)


def _uninstaller_marker(directory: Path, record: dict) -> tuple[str | None, dict | None]:
    """(None | "uninstaller" | "foreign", its marker): a well-formed record of the uninstaller's cleanup.

    The uninstaller removes only masters and products (never a file matched by GROUPS), so a job it
    cleaned still holds its per-frame files; resume, review and recompose are already refused.
    """
    data = _read_marker(directory / UNINSTALL_MARKER)
    if data is None:
        return None, None
    valid = (_bound(data, directory, record) and data.get("version") == UNINSTALL_MARKER_VERSION
             and data.get("state") in UNINSTALL_STATES)
    return ("uninstaller", data) if valid else ("foreign", None)


def _key(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _protected(record: dict, job: dict) -> set[str]:
    """Files that are never deleted even if a pattern matched: frozen workflow stages, products, inputs."""
    keep = set()
    for checkpoint in (record.get("checkpoints") or {}).values():
        for item in (checkpoint or {}).get("files") or []:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                keep.add(_key(item["path"]))
    request_files = (record.get("request") or {}).get("files") or []
    for item in [*request_files, *(job.get("artifacts") or [])]:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            keep.add(_key(item["path"]))
    revision = job.get("transient_revision")
    if isinstance(revision, dict) and isinstance(revision.get("record"), str):
        keep.add(_key(revision["record"]))
    return keep


def _works_identity(actual: Path, receipt: dict) -> list:
    """(path, size, mtime) of every recorded works file (and of a folder's own delivery.json)."""
    names = [item["file"] for item in receipt["files"]]
    if (actual / "delivery.json").is_file():
        names.append("delivery.json")
    identity = []
    for name in names:
        info = os.stat(actual / name)
        identity.append([str(actual / name), info.st_size, info.st_mtime_ns])
    return identity


def _deliveries(directory: Path, record: dict, works: list, *, verify: bool) -> list[str]:
    """Every current, base and reviewed works folder verifies (storage.verify_delivery).

    ``verify``: every file hashed; otherwise its records and file sizes only (the listing reads no image).
    The identities of the works files are appended to ``works``.
    """
    preserved = {record.get("delivery_dir"), record.get("base_delivery_dir") or record.get("delivery_dir")}
    reviewed = record.get("review_deliveries") or []
    if not isinstance(reviewed, list):
        raise CleanupRefused("作品记录无效，不能清理")
    preserved.update(item.get("delivery_dir") for item in reviewed if isinstance(item, dict))
    preserved.discard(None)
    if not preserved:
        raise CleanupRefused("任务没有已保存的作品，不能清理")
    verified = []
    for value in sorted(preserved):
        try:
            actual = storage._no_links(value).resolve(strict=True)
        except OSError:
            raise CleanupRefused("作品文件夹缺失或无法读取，不能清理：" + str(value)) from None
        if storage._overlap(actual, directory):
            raise CleanupRefused("作品仍位于缓存内")
        try:
            receipt = storage.verify_delivery(actual, receipt=storage.delivery_receipt(record, actual), hashes=False)
            # Taken before hashing: a file changed while or after it was hashed no longer matches inside the locks.
            works.extend(_works_identity(actual, receipt))
            if verify:
                storage.verify_delivery(actual, receipt=storage.delivery_receipt(record, actual))
        except OSError:
            raise CleanupRefused("作品文件夹缺失或无法读取，不能清理：" + str(value)) from None
        except ValueError as error:
            raise CleanupRefused("作品校验未通过，不能清理：" + str(error)) from None
        verified.append(str(actual))
    return verified


def _original_present(name: str, inputs: list, folders: dict) -> bool:
    """``stage/<name>`` is the copy of an original still at its recorded path with its recorded size.

    ``folders`` memoises whether each original's folder exists (a disconnected drive is asked once).
    """
    match = STAGE_NAME.fullmatch(name)
    if not match or not 1 <= int(match.group(1)) <= len(inputs):
        return False
    index = int(match.group(1))
    item = inputs[index - 1]
    if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or type(item.get("size")) is not int
            or name != f"frame{index:05d}{Path(item['path']).suffix.lower()}"):
        return False
    original = Path(item["path"])
    folder = os.path.normcase(str(original.parent))
    if folder not in folders:
        try:
            folders[folder] = original.parent.is_dir()
        except OSError:
            folders[folder] = False
    if not folders[folder]:
        return False
    try:
        info = os.stat(original)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == item["size"]


def _select(directory: Path, record: dict, files: list, plan: dict, works: list, *, held: bool,
            verify: bool) -> list[dict]:
    marker, previous = _marker(directory, record)
    uninstalled, by_uninstaller = _uninstaller_marker(directory, record)
    if "foreign" in (marker, uninstalled):
        raise CleanupRefused(FOREIGN, "cleaned")
    if previous:
        plan["previous_cleanup"] = {key: previous.get(key) for key in
                                    ("state", "attempt", "removed_bytes", "started_at", "finished_at")}
    if by_uninstaller:
        plan["uninstaller_cleanup"] = {"state": by_uninstaller.get("state"),
                                       "removed_bytes": by_uninstaller.get("removed_bytes")}
    state = record.get("state") if held else wf.workflow_status(directory).get("state")
    if state in ("queued", "running", "cancel_requested") or record.get("trail_addback_pending"):
        raise CleanupRefused("任务仍在处理中，不能清理中间文件", "running")
    if state != "succeeded":
        raise CleanupRefused("任务未成功完成，不能清理中间文件；失败或已停止的任务恢复时还需要它们")
    child = _child(directory, record)
    if not held:
        for path, message in _locks(directory.parent, directory, child)[1:]:
            if is_locked(path):
                raise CleanupRefused(message, "running")
    request = record.get("request") or {}
    if not request.get("delivery_root"):
        raise CleanupRefused("旧版任务的作品保存在缓存内，不能在应用内清理")
    if wf._digest(request) != record.get("signature") or record.get("config") != request.get("config"):
        raise CleanupRefused("任务参数记录已改变，不能清理")
    job = read_json(child / "job.json")
    if (not isinstance(job, dict) or job.get("state") != "succeeded"
            or not isinstance(job.get("job_dir"), str) or Path(job["job_dir"]).resolve() != child):
        raise CleanupRefused("堆栈子任务缺失或记录不符，不能清理")
    originals = [*(request.get("input_files") or []),
                 *(item.get("path") for item in job.get("inputs") or [] if isinstance(item, dict))]
    for value in originals:
        path = Path(os.path.abspath(str(value)))
        if path == directory or path.is_relative_to(directory):
            raise CleanupRefused("缓存内包含原片，不能清理")
    plan["delivery_dirs"] = _deliveries(directory, record, works, verify=verify)
    protected = _protected(record, job)
    inputs = job.get("inputs") if isinstance(job.get("inputs"), list) else []
    prefix = child.relative_to(directory).parts
    entries, copies_kept, folders = [], [], {}
    for relative, info in files:
        parts = PurePosixPath(relative).parts
        if parts[:len(prefix)] != prefix:
            continue
        category = _group(parts[len(prefix):])
        path = directory / relative
        if category is None or _key(path) in protected:
            continue
        if category == "stage_copies" and not _original_present(parts[-1], inputs, folders):
            copies_kept.append(info.st_size)          # possibly the last copy of that photo
            continue
        if info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT:
            raise CleanupRefused("待删除文件不是独立的普通文件（可能是链接或硬链接），不能清理：" + str(path))
        entries.append({"path": relative, "category": category, "bytes": info.st_size,
                        "mtime_ns": info.st_mtime_ns, "file_id": info.st_ino})
    plan.update(stage_copies_kept=len(copies_kept), stage_copies_kept_bytes=sum(copies_kept))
    if not entries:
        state = "cleaned" if previous or by_uninstaller else "nothing"
        if copies_kept:
            raise CleanupRefused(COPIES_KEPT, state)
        raise CleanupRefused(CLEANED if state == "cleaned" else NOTHING, state)
    groups = []
    for category, _, label in GROUPS:
        chosen = [entry for entry in entries if entry["category"] == category]
        if chosen:
            groups.append({"category": category, "label": label, "files": len(chosen),
                           "bytes": sum(entry["bytes"] for entry in chosen)})
    deletable = sum(entry["bytes"] for entry in entries)
    plan.update(groups=groups, files_deletable=len(entries), bytes_deletable=deletable,
                files_kept=plan["files_total"] - len(entries), bytes_kept=plan["bytes_total"] - deletable,
                state="partial" if previous else "cleanable",
                plan_id=_plan_id(directory, record, child, plan, entries))
    return entries


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _plan_id(directory: Path, record: dict, child: Path, plan: dict, entries: list[dict]) -> str:
    """Identity of what would be deleted and of the records it depends on (not of the kept files)."""
    identity = {"version": PLAN_VERSION, "workflow_dir": str(directory), "cache_root": plan["cache_root"],
                "signature": record.get("signature"), "workflow_record": _file_digest(directory / "workflow.json"),
                "stack_record": _file_digest(child / "job.json"), "marker": _file_digest(directory / MARKER),
                "uninstall_marker": _file_digest(directory / UNINSTALL_MARKER),
                "deliveries": plan["delivery_dirs"],
                "files": [[e["path"], e["category"], e["bytes"], e["mtime_ns"], e["file_id"]] for e in entries]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _locks(root: Path, directory: Path, child: Path) -> list[tuple[Path, str]]:
    """The lock order; the first (the cache root's admission lock) is taken only to clean."""
    return [(root / ".workflow-admission.lock", ADMISSION_BUSY),
            *((directory / name, BUSY) for name in ("worker.lock", "edit.lock", "metadata.lock")),
            *((child / name, BUSY) for name in ("worker.lock", "metadata.lock")),
            (child / "trail_review.lock", REVIEW_BUSY)]


# --------------------------------------------------------------------------- deletion

def _record_stack_job(child: Path, record: dict) -> None:
    """``job.json.gc`` written while this cleanup holds the stack job's metadata.lock.

    core._update would wait for that same lock (a second handle in this process) and time
    out, so the record is replaced here directly, as core._update does under the lock.
    """
    path = child / "job.json"
    job = read_json(path)
    previous = job.get("gc")
    if isinstance(previous, dict) and previous.get("source") == "app" and previous.get("plan_id") == record["plan_id"]:
        history = list(previous.get("history") or [])          # the same attempt, finished
    elif isinstance(previous, dict):
        history = list(previous.get("history") or []) + [{k: v for k, v in previous.items() if k != "history"}]
    else:
        history = []
    job["gc"] = {**record, "history": history}
    job["updated_at"] = core.now()
    atomic_json(path, job)


def _admit_deletion(directory: Path, child: Path, entry: dict) -> Path:
    """The planned file, still inside the stack job, in its group, unlinked, unchanged and not a link."""
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("待删除文件不在清理范围内：" + entry["path"])
    path = directory.joinpath(*relative.parts)
    try:
        inside = path.relative_to(child).parts
    except ValueError:
        raise ValueError("待删除文件不在清理范围内：" + str(path)) from None
    if _group(inside) != entry["category"]:
        raise ValueError("待删除文件不在清理范围内：" + str(path))
    storage._no_links(path)
    info = os.lstat(path)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError("待删除文件不是独立的普通文件（可能是链接或硬链接），不能清理：" + str(path))
    if (info.st_size, info.st_mtime_ns, info.st_ino) != (entry["bytes"], entry["mtime_ns"], entry["file_id"]):
        raise ValueError("文件在检查后改变，未删除：" + str(path))
    return path


def _execute(directory: Path, child: Path, record: dict, plan: dict, entries: list[dict],
             admission: ExitStack) -> dict:
    _, previous = _marker(directory, record)
    started = core.now()
    removed_before = list((previous or {}).get("removed_files") or [])
    bytes_before = int((previous or {}).get("removed_bytes") or 0)
    gc = {"version": GC_VERSION, "source": "app", "state": "in_progress", "at": started, "finished_at": None,
          "plan_id": plan["plan_id"], "workflow_dir": str(directory), "work_cache": True, "work_cache_removed": True,
          "removed_categories": [group["category"] for group in plan["groups"]],
          "bytes_planned": plan["bytes_deletable"], "files_planned": plan["files_deletable"], "bytes": 0, "files": 0,
          "consequence": "清理后本作业不能恢复、复核亮迹或重做亮迹合成；母版、产物、记录与日志保留"}
    # Both records go first: a deletion that stops half way must still refuse resume, review and recompose.
    _record_stack_job(child, gc)
    marker = {"version": MARKER_VERSION, "cleaned_by": "aester-app", "workflow_dir": str(directory),
              "workflow_signature": record["signature"], "stack_job_dir": str(child), "state": "started",
              "plan_id": plan["plan_id"], "attempt": int((previous or {}).get("attempt") or 0) + 1,
              "started_at": started, "finished_at": None, "groups": plan["groups"],
              "removed_files": removed_before, "removed_bytes": bytes_before, "errors": [],
              "kept": "masters, products, checkpoint images, finishing layers, publication receipts, records, "
                      "snapshots, transient records, logs and unknown files",
              "loses_workflow_resume_and_review": True,
              "after_uninstaller_cleanup": plan["uninstaller_cleanup"],
              "stage_copies_kept": plan["stage_copies_kept"]}
    atomic_json(directory / MARKER, marker)
    # The marker refuses a resubmission and a resume of this job from now on: other jobs may be admitted.
    admission.close()
    removed, problems, freed = [], [], 0
    for entry in entries:
        try:
            _admit_deletion(directory, child, entry).unlink()
        except (OSError, ValueError) as error:
            problems.append({"path": entry["path"], "error": str(error)})
            continue
        removed.append(entry["path"])
        freed += entry["bytes"]
    state = "partial" if problems else "completed"
    finished = core.now()
    _record_stack_job(child, {**gc, "state": "partial" if problems else "done", "finished_at": finished,
                              "bytes": freed, "files": len(removed)})
    marker.update(state=state, finished_at=finished, removed_files=removed_before + removed,
                  removed_bytes=bytes_before + freed, errors=problems)
    atomic_json(directory / MARKER, marker)
    return {"version": RESULT_VERSION, "workflow_dir": str(directory), "plan_id": plan["plan_id"], "state": state,
            "removed_files": len(removed), "removed_bytes": freed, "errors": problems,
            "remaining_files": len(entries) - len(removed),
            "remaining_bytes": plan["bytes_deletable"] - freed, "kept_bytes": plan["bytes_kept"],
            "stage_copies_kept": plan["stage_copies_kept"], "delivery_dirs": plan["delivery_dirs"]}
