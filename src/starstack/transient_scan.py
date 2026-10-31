"""Bounded, owner-protected processes for independent transient-frame scans.

Every child is created through ``guarded_process`` before running Python code.
Windows therefore has the same atomic Job Object protection as the Siril
adapter, including when the owner dies during child startup. Children read
FITS inputs and publish only new per-run JSON files. Linux retains the guard's
documented, direct-child-only fallback; it is not a production guarantee.

Failures are reported in frame order (efficiency P1c T-2,
docs/EFFICIENCY_P1C_PREREG_20261005.md §3.4), so that which failure a scan
reports, and its text, do not depend on the number of scanner processes:

- a child whose ``_scan_frame`` raises publishes ``error-<index>.json``
  (version, frame index, stem, exception type, the first 4000 characters of
  the message) for that frame and re-raises; it publishes nothing further;
- once a child has exited, every result and error file it published is read
  (with the bounded retry of access/sharing denials) before its exit is
  judged; a read still blocked stays pending, never a failure. A failure
  belongs to the frame of its error file (checked: that child's first
  unpublished frame); an exit without one, to that child's first frame
  without a result file on disk; a non-zero exit after every result was
  published ranks after all frames (reported once every child is judged).
  A result that stays unreadable, is not the frame's own, or does not parse
  is a failure of its frame as well;
- with ``stop`` the lowest failed frame, children whose remaining frames all
  lie after ``stop`` are stopped at once (kill errors are logged, never
  hide the frame failure); the frames before ``stop`` are awaited; then,
  after every child has ended, the result / error / pending files of the
  frames after ``stop`` are removed and the failure at ``stop`` is raised.
  Its message names the frame number, the stem, the exception type and
  message (or the exit status); the run folder, the child's log and its
  tail go to ``log`` (the job's worker.log), except that an error record
  that cannot be read is reported with the log tail.
"""
from __future__ import annotations

from contextlib import ExitStack
import errno
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .process_guard import guarded_process

DEFAULT_TOTAL_TIMEOUT = 24 * 60 * 60.0
POLL_SECONDS = 0.1
RESULT_READ_RETRY_SECONDS = 2.0
# The cleanup after a failure (P1c T-2): how long locked files, or the last file operation of a child still ending,
# are waited for, and how long the folder must stay free of the removed files once such a child was seen.
REMOVAL_SECONDS = 5.0
REMOVAL_SETTLE_SECONDS = 0.25
ERROR_MESSAGE_CHARS = 4000          # the child keeps this much of a frame's exception message
# The scanner children run only BLAS calls whose results do not depend on
# OpenBLAS's thread count (2x2 syrk and eigh, two-column gemv, dot products of
# length <= 5; efficiency P1a S-5). One thread is bit-identical and removes
# OpenBLAS's per-thread buffers from each child's commit charge (measured
# 1.62 -> 0.88 GB peak per child at the 24-thread cap).
# tests/test_transient_scan_blas.py pins the audit: re-audit before adding
# BLAS work to the child.
WORKER_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1"}

_ABSENT, _BLOCKED, _DONE = "absent", "blocked", "done"


class ScanWorkerError(RuntimeError):
    """A protected scanner failed, or failed to publish all assigned results."""


def _worker_command(request: Path) -> list[str]:
    # The project may run directly from src without being installed. Pass the
    # exact source root as an argv value; do not alter shared process-wide
    # environment variables while other jobs may be starting children.
    bootstrap = ("import runpy,sys; sys.path.insert(0,sys.argv[1]); "
                 "sys.argv=['starstack.transient_scan_worker',sys.argv[2]]; "
                 "runpy.run_module('starstack.transient_scan_worker',run_name='__main__')")
    return [sys.executable, "-c", bootstrap, str(Path(__file__).resolve().parents[1]), str(request)]


def worker_environment(base=None) -> dict:
    """The complete environment block of a scanner child: this process's
    environment (or ``base``), plus WORKER_ENVIRONMENT."""
    return {**(os.environ if base is None else base), **WORKER_ENVIRONMENT}


def result_name(index: int) -> str:
    return f"result-{index:06d}.json"


def error_name(index: int) -> str:
    return f"error-{index:06d}.json"


def _tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 4000))
        return stream.read().decode("utf-8", errors="replace")


def _safe_tail(path: Path) -> str:
    try:
        return _tail(path)
    except OSError as exc:
        return f"(log unreadable: {type(exc).__name__})"


def _sharing_denial(exc: PermissionError) -> bool:
    # Windows file opens may expose only errno=EACCES, without winerror.
    # Retry only these access/sharing denials.
    winerror = getattr(exc, "winerror", None)
    return winerror in {5, 32, 33} or (winerror is None and exc.errno == errno.EACCES)


class _Child:
    """One scanner process: its number, assignment (frame order) and how its exit was judged."""
    __slots__ = ("number", "process", "logfile", "jobs", "status", "judged", "stopped")

    def __init__(self, number, process, logfile, jobs):
        self.number, self.process, self.logfile, self.jobs = number, process, logfile, jobs
        self.status = None
        self.judged = self.stopped = False


class _Scan:
    """The owner's view of one scan run: results read, failures by frame index, pending read deadlines."""

    def __init__(self, run: Path, indexed: list, log):
        self.run, self.indexed, self.count, self.log = run, indexed, len(indexed), log
        self.scanned = {}            # frame index -> the child's "scanned" payload
        self.failures = {}           # frame index (or count + child number: after every frame) -> exception
        self.absent = set()          # frame indices without a result file at the latest read
        self.deadlines = {}          # file name -> retry deadline of a denied read

    # ---- reading what children published -------------------------------------------------------------------------

    def _read(self, path: Path):
        """(_DONE, text) / (_ABSENT, None) / (_BLOCKED, None); a denial past its deadline is raised to the caller."""
        try:
            if not path.is_file():
                return _ABSENT, None
            text = path.read_text(encoding="utf-8")
        except PermissionError as exc:
            if not _sharing_denial(exc):
                raise
            now = time.monotonic()
            deadline = self.deadlines.setdefault(path.name, now + RESULT_READ_RETRY_SECONDS)
            if now >= deadline:
                raise
            # Back to the owner loop: cancellation, the global timeout and
            # other children's failures remain active.
            return _BLOCKED, None
        self.deadlines.pop(path.name, None)
        return _DONE, text

    def fail(self, key: int, error: BaseException | None = None, *, message: str | None = None, cause=None) -> None:
        if error is None:
            error = ScanWorkerError(message)
            error.__cause__ = cause
        self.failures.setdefault(key, error)

    def read_result(self, job: dict) -> str:
        index, stem = job["index"], job["stem"]
        path = Path(job["result"])
        try:
            state, text = self._read(path)
        except PermissionError as exc:
            if not _sharing_denial(exc):
                self.fail(index, exc)
                return _DONE
            self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）的结果持续无法读取（{path.name}）："
                                     f"{type(exc).__name__}: {exc.strerror or exc.errno}", cause=exc)
            return _DONE
        except Exception as exc:                 # any other I/O or decoding fault: this frame's failure, unchanged
            self.fail(index, exc)
            return _DONE
        if state == _ABSENT:
            self.absent.add(index)
            return state
        self.absent.discard(index)
        if state == _BLOCKED:
            return state
        try:
            payload = json.loads(text)
            if payload.get("stem") != stem or payload.get("version") != 1:
                self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）的结果身份不符（{path.name}）")
                return _DONE
            scanned = payload["scanned"]
        except Exception as exc:                 # a corrupt result: this frame's failure, unchanged
            self.fail(index, exc)
            return _DONE
        self.scanned[index] = scanned
        self.log(f"transients: {stem}: {len(scanned['result']['segments'])} candidate(s), {scanned['seconds']} s")
        return _DONE

    def poll(self, child: _Child) -> None:
        """Read what ``child`` published; once it has exited and nothing it published is still blocked, judge it."""
        if child.judged:
            return
        status = child.process.poll()            # first: a child seen exited has published all it ever will
        blocked = False
        for job in child.jobs:
            if job["index"] in self.scanned or job["index"] in self.failures:
                continue
            blocked |= self.read_result(job) == _BLOCKED
        if status is None or blocked or child.stopped:
            return
        child.status = status
        child.judged = self.judge(child)

    def judge(self, child: _Child) -> bool:
        """An exited child: record the failure its exit means (if any). False while its error record is blocked."""
        missing = next((job for job in child.jobs if job["index"] in self.absent and job["index"] not in self.scanned
                        and job["index"] not in self.failures), None)
        status = child.status
        if missing is None:
            if status != 0:
                self.fail(self.count + child.number,
                          message=f"亮迹扫描失败：扫描进程在发布全部结果后异常退出（{status}）")
                self._detail(child, "exit after all results were published")
            return True
        index, stem = missing["index"], missing["stem"]
        path = Path(missing["error"]) if missing.get("error") else None
        try:
            state, text = self._read(path) if path is not None else (_ABSENT, None)
        except Exception as exc:                 # unreadable error record: report the frame with the log tail
            self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）：扫描进程退出（{status}），"
                                     f"错误记录无法读取（{type(exc).__name__}）\n{_safe_tail(child.logfile)}")
            self._detail(child, f"error record of frame {index + 1} unreadable")
            return True
        if state == _BLOCKED:
            return False
        if state == _ABSENT:
            if status == 0:
                self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）：扫描进程退出但缺少结果")
            else:
                self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）：扫描进程异常退出（{status}），"
                                         f"没有错误记录")
            self._detail(child, f"no result and no error record for frame {index + 1}")
            return True
        try:
            record = json.loads(text)
            if (record.get("version") != 1 or record.get("index") != index or record.get("stem") != stem
                    or not isinstance(record.get("type"), str) or not isinstance(record.get("message"), str)):
                raise ValueError("not this frame's error record")
        except Exception as exc:
            self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）：扫描进程退出（{status}），"
                                     f"错误记录无效（{type(exc).__name__}）\n{_safe_tail(child.logfile)}")
            self._detail(child, f"invalid error record of frame {index + 1}")
            return True
        kind, detail = record["type"], record["message"]
        self.fail(index, message=f"亮迹扫描失败：第 {index + 1} 帧（{stem}）：{kind}: {detail}")
        self._detail(child, f"frame {index + 1} raised {kind}")
        return True

    def _detail(self, child: _Child, what: str) -> None:
        """The run folder, the child's log and its tail: worker.log only, never the reported message."""
        self.log(f"transients: scanner {child.number} {what} (exit {child.status}); run folder {self.run}; "
                 f"log {child.logfile}\n{_safe_tail(child.logfile)}")

    # ---- deciding ------------------------------------------------------------------------------------------------

    def stop(self) -> int | None:
        return min(self.failures) if self.failures else None

    def resolved(self, stop: int, children) -> bool:
        """The failure at ``stop`` can be reported: every frame before it has a result; for a failure ranked after
        all frames (a late non-zero exit), every child has been judged or stopped too."""
        if stop >= self.count:
            return all(child.judged for child in children)
        return all(index in self.scanned for index in range(stop))

    def stop_after(self, children, stop: int) -> None:
        """Stop the children whose remaining frames all lie after the failed frame ``stop`` (their outcome cannot
        matter). A failure ranked after all frames stops nobody: the others are awaited."""
        if stop >= self.count:
            return
        for child in children:
            if child.judged or child.stopped:
                continue
            if all(job["index"] > stop for job in child.jobs if job["index"] not in self.scanned):
                child.stopped = True
                _kill(child, self.log)

    def remove_after(self, stop: int, children=()) -> None:
        """Remove the result / error / pending files of the frames after ``stop`` (every child's guard has closed),
        so that the tree a failed scan leaves does not depend on the number of children.

        A sharing denial (a virus scanner, a terminated child's handle still closing) is retried. A child's job that
        still held a process when its guard closed (the venv launcher is the direct child; the interpreter it started
        may end a moment later and completes the file operation it was in) makes the folder be watched until it has
        shown none of these files for REMOVAL_SETTLE_SECONDS. Both within REMOVAL_SECONDS; a file left is logged."""
        paths = [self.run / name for index in range(stop + 1, self.count)
                 for name in (result_name(index), error_name(index))]
        paths += [path.with_suffix(".pending") for path in paths]
        settle = any((getattr(child.process, "accounting", None) or {}).get("active_processes")
                     for child in children)
        deadline = time.monotonic() + REMOVAL_SECONDS
        seen = time.monotonic()                  # the last pass that found one of the files
        while True:
            locked = []
            for path in paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except PermissionError as exc:
                    if not _sharing_denial(exc):
                        self.log(f"transients: could not remove {path.name} after the failure ({type(exc).__name__})")
                        continue
                    locked.append(path)
                except OSError as exc:
                    self.log(f"transients: could not remove {path.name} after the failure ({type(exc).__name__})")
                    continue
                seen = time.monotonic()
            now = time.monotonic()
            if not locked and (not settle or now - seen >= REMOVAL_SETTLE_SECONDS):
                return
            if now >= deadline:
                for path in locked:
                    self.log(f"transients: could not remove {path.name} after the failure (still locked)")
                return
            time.sleep(.02)


def _kill(child: _Child, log) -> None:
    # kill() targets the child's whole Job Object on Windows. An error here
    # (OSError, the guard's ProcessGuardError) never hides the scan's outcome.
    try:
        if child.process.poll() is None:
            child.process.kill()
    except OSError as exc:
        log(f"transients: stopping scanner {child.number} failed: {type(exc).__name__}: {exc}")


def scan_frames(jobs: dict[str, dict], *, workers: int, output_dir: Path,
                check=None, log=print, total_timeout: float | None = DEFAULT_TOTAL_TIMEOUT) -> dict[str, dict]:
    """Scan a mapping of frame IDs to ``_scan_frame`` keyword arguments.

    Each job has ``frame_path``, ``master_path``, ``detection``, ``grid_marks``
    and ``stats_step``. Results retain the input mapping order, regardless of
    completion order. Cancellation is checked at least once per polling cycle
    while work runs, including with one worker. Any exit (including a startup
    failure) requests termination of *all* live workers before waiting for any
    one; guard cleanup has a bounded wait per worker, never an infinite join.
    The OS may complete termination asynchronously, so this is not permission
    to reuse an old job's files after an unverified interrupted execution.
    A scan failure is reported in frame order (module docstring).
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if total_timeout is not None and (isinstance(total_timeout, bool)
            or not math.isfinite(total_timeout) or total_timeout <= 0):
        raise ValueError("total_timeout must be finite and positive, or None")
    if check:
        check()
    if not jobs:
        return {}
    workers = min(workers, len(jobs))
    run = Path(output_dir).resolve() / "scan-runs" / uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    indexed = list(jobs.items())
    assignments = [[] for _ in range(workers)]
    for index, (stem, args) in enumerate(indexed):
        assignments[index % workers].append({"index": index, "stem": stem, "args": args,
                                             "result": str(run / result_name(index)),
                                             "error": str(run / error_name(index))})
    scan = _Scan(run, indexed, log)
    children = []
    environment = worker_environment()
    started = time.monotonic()
    failure = stop = None
    try:
        with ExitStack() as stack:
            try:
                for number, assignment in enumerate(assignments):
                    if check:
                        check()
                    request = run / f"worker-{number:03d}.json"
                    request.write_text(json.dumps({"version": 1, "jobs": assignment}, allow_nan=False), encoding="utf-8")
                    logfile = run / f"worker-{number:03d}.log"
                    output = stack.enter_context(logfile.open("ab", buffering=0))
                    process = stack.enter_context(guarded_process(
                        _worker_command(request), stdin=subprocess.DEVNULL, stdout=output,
                        stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        env=environment))
                    children.append(_Child(number, process, logfile, assignment))
                while True:
                    if check:
                        check()
                    if total_timeout is not None and time.monotonic() - started >= total_timeout:
                        raise TimeoutError(f"亮迹扫描超过总时限 {total_timeout:g} 秒；诊断目录：{run}")
                    for child in children:
                        scan.poll(child)
                    stop = scan.stop()
                    if stop is not None:
                        scan.stop_after(children, stop)
                        if scan.resolved(stop, children):
                            failure = scan.failures[stop]
                            break
                    elif all(child.judged for child in children):
                        if check:
                            check()
                        return {stem: scan.scanned[index] for index, (stem, _) in enumerate(indexed)}
                    time.sleep(POLL_SECONDS)
            finally:
                # Do not serially wait while other workers are still consuming
                # CPU/memory. ExitStack always runs every guard's bounded
                # cleanup, even if one kill/poll fails; cleanup must not hide
                # the original failure.
                for child in children:
                    _kill(child, log)
    except OSError as exc:
        if failure is None:
            raise
        log(f"transients: scanner cleanup after the frame failure raised {type(exc).__name__}: {exc}")
    # Every guard has run its bounded cleanup: the children have ended (or are ending: remove_after).
    scan.remove_after(stop, children)
    try:
        raise failure
    finally:
        failure = None
