"""``starstack gc``: reclaim a job's superseded exports and rebuildable intermediates (docs/HARDENING_PACKAGE_20260922.md H3).

Dry-run by default — ``plan`` only lists what would go and how many bytes; ``apply`` deletes exactly the listed
entries under the job's writer lock and records ``job.json.gc``.  Two categories:

(a) ``superseded_products``: ``products/<id>`` directories that nothing in ``job.json`` references any more (earlier
    base exports, earlier bright-trail revisions).  Only directories named like the job's own exports
    (``uuid4().hex[:12]`` / ``uuid4().hex``) that contain a job export or are empty; anything else under ``products``
    is listed as unrecognised and kept.
(b) ``--work-cache`` (succeeded jobs only): the files the worker can rebuild from the originals — converted frames
    ``work/light_*.fit``, registered frames ``work/r_light*.fit``, sequence files ``work/*.seq``, the staged copies of
    the originals ``stage/*`` and stack-range trial copies ``range-*/trial*/*.fit``.  Afterwards ``resume`` and the
    bright-trail recompose are impossible and are refused with this reason (``core.gc_refusal``).

Never touched: the originals (outside the job), ``work/sky_master.fit``, ``work/sky_master_undistorted.fit``,
``work/foreground_master.fit``, coverage / validity / distortion / support / range records, ``work/cache``,
``work/transients`` (candidate records and review crops), referenced products, ``job.json``, locks and logs, and
anything this module does not recognise.  MCP does not expose gc; deletion stays a CLI decision.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil

from . import core
from .locking import file_lock, is_locked
from .records import atomic_json

VERSION = "gc-1"
PRODUCT_DIR_NAME = re.compile(r"^(?:[0-9a-f]{12}|[0-9a-f]{32})$")
WORK_CACHE_GLOBS = (("work", "light_*.fit"), ("work", "r_light*.fit"), ("work", "*.seq"))
STAGE_GLOB = ("stage", "*")
RANGE_TRIAL_GLOB = "range-*/trial*/*.fit"
LIVE_STATES = {"running", "queued", "cancel_requested"}
CATEGORY_LABELS = {"superseded_products": "被替代的导出（products 下未被 job.json 引用的目录）",
                   "work_cache": "转换帧、配准帧与序列文件（可由原片重建）",
                   "stage_copies": "原片副本（stage/，可由原片重新复制）",
                   "range_trials": "范围保护试验副本（range-*/trial*/）"}


def _tree_size(path: Path) -> tuple[int, int]:
    """(bytes, files) of a directory tree by ``st_size``; symlinks are counted as their own entry, never followed."""
    total, count = 0, 0
    for child in path.rglob("*"):
        if child.is_symlink() or child.is_file():
            total += child.lstat().st_size
            count += 1
    return total, count


def referenced_products(directory: Path, job: dict) -> set[str]:
    """Names of ``products/<id>`` directories that any absolute path string anywhere in ``job.json`` points into."""
    products = (directory / "products").resolve()
    found: set[str] = set()

    def visit(value):
        if isinstance(value, str):
            if "products" not in value:
                return
            try:
                path = Path(value)
                if not path.is_absolute():
                    return
                relative = path.resolve().relative_to(products)
            except (ValueError, OSError, RuntimeError):
                return
            if relative.parts:
                found.add(relative.parts[0])
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(job)
    return found


def _is_job_export(path: Path) -> bool:
    """A directory this job's worker created: a base export, a bright-trail revision, or one left empty by a failed attempt."""
    if not PRODUCT_DIR_NAME.match(path.name):
        return False
    if (path / "master_linear.fit").is_file() or (path / "transients").is_dir():
        return True
    return not any(path.iterdir())


def _check_location(directory: Path, job: dict) -> None:
    recorded = job.get("job_dir")
    if not recorded or Path(recorded).resolve() != directory:
        raise ValueError("作业记录的目录与实际位置不一致（改名保留的旧作业或复制的目录），gc 拒绝处理，以免误删它引用的产物")


def plan(job_dir, *, work_cache: bool = False) -> dict:
    """List what ``apply`` would remove.  Reads only; the tree is not modified."""
    directory = Path(job_dir).resolve()
    job = core.load_job(directory)
    _check_location(directory, job)
    referenced = referenced_products(directory, job)
    entries, kept, unrecognised = [], [], []
    products = directory / "products"
    if products.is_dir():
        for child in sorted(products.iterdir()):
            if child.name in referenced:
                kept.append(child.name)
            elif child.is_dir() and not child.is_symlink() and _is_job_export(child):
                size, count = _tree_size(child)
                entries.append({"category": "superseded_products", "path": str(child.relative_to(directory)), "bytes": size,
                                "files": count})
            else:
                unrecognised.append(child.name)
    if work_cache:
        for sub, pattern in WORK_CACHE_GLOBS:
            for path in sorted((directory / sub).glob(pattern)):
                if path.is_file() and not path.is_symlink():
                    entries.append({"category": "work_cache", "path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                                    "files": 1})
        for path in sorted((directory / STAGE_GLOB[0]).glob(STAGE_GLOB[1])):
            if path.is_file() and not path.is_symlink():
                entries.append({"category": "stage_copies", "path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                                "files": 1})
        for path in sorted(directory.glob(RANGE_TRIAL_GLOB)):
            if path.is_file() and not path.is_symlink():
                entries.append({"category": "range_trials", "path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                                "files": 1})
    by_category = {}
    for entry in entries:
        item = by_category.setdefault(entry["category"], {"label": CATEGORY_LABELS[entry["category"]], "bytes": 0, "files": 0, "entries": 0})
        item["bytes"] += entry["bytes"]
        item["files"] += entry["files"]
        item["entries"] += 1
    previous = job.get("gc")
    return {"version": VERSION, "job_dir": str(directory), "job_state": job["state"], "dry_run": True, "work_cache": work_cache,
            "work_cache_allowed": job["state"] == "succeeded",
            "referenced_products": sorted(referenced), "kept_products": kept, "unrecognised_kept": unrecognised,
            "entries": entries, "bytes": sum(e["bytes"] for e in entries), "files": sum(e["files"] for e in entries),
            "by_category": by_category, "previous_gc": previous,
            "consequence": ("删除后本作业不能 resume，也不能用 starstack transients 重做亮迹合成；母版、被引用的产物、记录与日志保留"
                            if work_cache else "只删被替代的导出；resume 与亮迹重合成不受影响"),
            "note": "字节数为文件 st_size 之和；默认只列清单，--apply 才删除"}


def apply(job_dir, *, work_cache: bool = False) -> dict:
    """Delete exactly what ``plan`` lists, under the job's writer lock, and record it in ``job.json.gc``."""
    directory = Path(job_dir).resolve()
    job = core.load_job(directory)
    _check_location(directory, job)
    if job["state"] in LIVE_STATES or is_locked(directory / "worker.lock"):
        raise ValueError(f"作业状态为 {job['state']} 或工作进程仍持有锁，不能清理；请等待它结束")
    if work_cache and job["state"] != "succeeded":
        raise ValueError(f"只有已完成（succeeded）的作业才能删除转换帧与配准帧；本作业状态为 {job['state']}，恢复还需要它们")
    with file_lock(directory / "worker.lock"):      # excludes a recompose or a resumed worker for the duration
        job = core.load_job(directory)
        if job["state"] in LIVE_STATES:
            raise ValueError(f"作业状态为 {job['state']}，不能清理")
        if work_cache and job["state"] != "succeeded":
            raise ValueError(f"只有已完成（succeeded）的作业才能删除转换帧与配准帧；本作业状态为 {job['state']}")
        planned = plan(directory, work_cache=work_cache)
        previous = job.get("gc")
        if not planned["entries"] and not work_cache:
            return {**planned, "dry_run": False, "removed_bytes": 0, "removed_files": 0, "gc": previous}
        history =list(previous.get("history") or []) + [{k: v for k, v in previous.items() if k != "history"}] if previous else []
        record = {"version": VERSION, "state": "in_progress", "at": core.now(), "work_cache": work_cache,
                  "work_cache_removed": bool(previous and previous.get("work_cache_removed")) or work_cache,
                  "removed_categories": sorted({e["category"] for e in planned["entries"]}),
                  "bytes_planned": planned["bytes"], "files_planned": planned["files"], "bytes": 0, "files": 0,
                  "products_removed": [e["path"] for e in planned["entries"] if e["category"] == "superseded_products"],
                  "referenced_products": planned["referenced_products"], "history": history,
                  "consequence": planned["consequence"]}
        # The record goes first: a deletion that stops half way must still refuse resume / recompose afterwards.
        core._update(directory, gc=record)
        removed_bytes = removed_files = 0
        for entry in planned["entries"]:
            path = directory / entry["path"]
            if entry["category"] == "superseded_products":
                size, count = _tree_size(path)
                shutil.rmtree(path)
            else:
                size, count = path.stat().st_size, 1
                path.unlink()
            removed_bytes += size
            removed_files += count
        record.update(state="done", bytes=removed_bytes, files=removed_files, finished_at=core.now())
        core._update(directory, gc=record)
    return {**planned, "dry_run": False, "removed_bytes": removed_bytes, "removed_files": removed_files, "gc": record}
